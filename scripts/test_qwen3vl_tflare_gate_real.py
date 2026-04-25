"""
Real-world ZMQ server for Qwen3-VL MoT VLA trained with
train_qwen3vl_tflare_gate.py (gated fusion + tactile-FLARE).

Differences from test_qwen3vl_flare_real.py:
  - Auto-detects use_tactile_flare, n_tfl_tokens_per_step, n_tfl_steps from
    training_args.json
  - Attaches gate_head, tactile_flare_queries/proj, target encoders before
    loading state_dict
  - Custom denoise loop with:
      v_act = final_layer(h_act)
      v_tac = final_layer_tactile(h_tac)
      g     = sigmoid(gate_head([h_act, h_tac]))
      v_t   = (1 - g) * v_act + g * v_tac
    and tactile_flare_queries inserted in the tactile block (so v_tac's
    attention pattern matches training).

Uses KV-cache on the latent block (slow + visual flare queries) across
denoise steps; the action and tactile blocks are recomputed each step
since their content changes with x_t. Mirrors the structure of the
original Qwen3VLVLAModel.denoise_step().
"""

import os, sys, copy
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import argparse, json, io, pickle, traceback
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import zmq
from transformers import AutoProcessor
from janus.models.action_tokenizer import ActionTokenizer

from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds


def _normalize(values, mask, vmin, vmax):
    return np.where(mask, np.clip(2.0 * (values - vmin) / (vmax - vmin + 1e-8) - 1.0, -1.0, 1.0), values)

def _denormalize(norm_values, mask, vmin, vmax):
    return np.where(mask, 0.5 * (norm_values + 1.0) * (vmax - vmin) + vmin, norm_values)


def _build_qwen3vl_from_config(config_path, args):
    with open(config_path) as f:
        full_cfg = json.load(f)
    image_token_id = full_cfg.get("image_token_id", 151655)
    model_type = full_cfg.get("model_type", "qwen2_vl")
    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        vl_config = Qwen3VLConfig(**{k: v for k, v in full_cfg.items() if k not in ("architectures", "transformers_version")})
        text_config = vl_config.text_config
    except Exception:
        from transformers import AutoConfig
        vl_config = AutoConfig.from_pretrained(os.path.dirname(config_path), trust_remote_code=True)
        text_config = getattr(vl_config, "text_config", vl_config)

    tac_isize = getattr(args, "tactile_intermediate_size", 0)
    tac_isize = tac_isize if tac_isize > 0 else None
    model = Qwen3VLVLAModel(
        config=text_config,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        use_tactile_deform=bool(args.use_tactile_deform),
        use_robot_state=bool(args.use_robot_state),
        image_token_id=image_token_id,
        tactile_intermediate_size=tac_isize,
        n_flare_tokens_per_frame=getattr(args, "n_flare_tokens_per_frame", 0),
        n_flare_steps=getattr(args, "n_flare_steps", 0),
    )
    vis_cfg_dict = full_cfg.get("vision_config", {})
    try:
        if model_type == "qwen3_vl":
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
            vis_cfg = Qwen3VLVisionConfig(**{k: v for k, v in vis_cfg_dict.items() if k != "model_type"})
            model.visual = Qwen3VLVisionModel(vis_cfg)
        else:
            from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionModel
            vis_cfg = Qwen2VLVisionConfig(**{k: v for k, v in vis_cfg_dict.items() if k != "model_type"})
            model.visual = Qwen2VLVisionModel(vis_cfg)
    except Exception as e:
        print(f"  Warning: visual tower creation failed: {e}")
        model.visual = None
    try:
        if model_type == "qwen3_vl":
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel as _VLModel
        else:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel as _VLModel
        class _RopeStub:
            def __init__(self, cfg): self.config = cfg
            def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
                return _VLModel.get_rope_index(self, input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        object.__setattr__(model, '_rope_index_fn', _RopeStub(vl_config).get_rope_index)
    except Exception as e:
        print(f"  Warning: rope index setup failed ({e}).")
    return model


def _has_hf_weights(path):
    import glob as _glob
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        if _glob.glob(os.path.join(path, pattern)):
            return True
    return False


def _attach_gate_and_tflare(model, args):
    """Attach modules that the trained checkpoint expects. Values get
    overwritten by state_dict load."""
    H = model.config.hidden_size
    D = args.action_dim
    model.gate_head = nn.Sequential(
        nn.Linear(2 * H, H), nn.SiLU(), nn.Linear(H, D),
    ).to(torch.bfloat16)
    if args.use_tactile_flare:
        K_tac = args.n_tfl_tokens_per_step * args.n_tfl_steps
        model.tactile_flare_queries = nn.Parameter(
            torch.zeros(1, K_tac, H, dtype=torch.bfloat16))
        model.tactile_flare_proj = nn.Sequential(
            nn.Linear(H, H), nn.GELU(), nn.Linear(H, H),
        ).to(torch.bfloat16)
        model.target_tacf6_embedder = copy.deepcopy(model.tacf6_embedder).to(torch.bfloat16)
        if args.use_tactile_deform:
            model.target_deform_proj = copy.deepcopy(model.deform_proj).to(torch.bfloat16)


def model_load(args):
    ckpt = args.checkpoint_path

    ta_path = os.path.join(ckpt, "training_args.json")
    ta = {}
    if os.path.exists(ta_path):
        with open(ta_path) as f:
            ta = json.load(f)
        for key, default in [
            ("tactile_intermediate_size", 0),
            ("n_flare_tokens_per_frame", 0),
            ("n_flare_steps", 0),
            ("flare_layer_index", -1),
            ("use_tactile_flare", 0),
            ("n_tfl_tokens_per_step", 0),
            ("n_tfl_steps", 0),
            ("tactile_flare_stride", 2),
        ]:
            saved = ta.get(key, default)
            cli_val = getattr(args, key, default)
            if saved != default and cli_val == default:
                setattr(args, key, saved)
                print(f"Auto-detected {key}={saved} from training_args.json")

    tac_isize = args.tactile_intermediate_size if args.tactile_intermediate_size > 0 else None

    proc_dir = os.path.join(ckpt, "processor")
    if not os.path.isdir(proc_dir):
        raise FileNotFoundError(f"processor/ not found in checkpoint: {ckpt}")
    processor = AutoProcessor.from_pretrained(proc_dir, trust_remote_code=True)
    print(f"Processor loaded from: {proc_dir}")

    base_model_path = getattr(args, "base_model_path", "")
    ckpt_config = os.path.join(ckpt, "config.json")

    if base_model_path and os.path.isdir(base_model_path) and _has_hf_weights(base_model_path):
        model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
            pretrained_path=base_model_path,
            action_dim=args.action_dim, action_chunk=args.action_chunk,
            use_tactile_deform=bool(args.use_tactile_deform),
            use_robot_state=bool(args.use_robot_state),
            torch_dtype=torch.bfloat16,
            tactile_intermediate_size=tac_isize,
            n_flare_tokens_per_frame=args.n_flare_tokens_per_frame,
            n_flare_steps=args.n_flare_steps,
        )
    elif os.path.exists(ckpt_config):
        model = _build_qwen3vl_from_config(ckpt_config, args)
    else:
        mp = ta.get("model_path", "")
        if mp and os.path.isdir(mp) and _has_hf_weights(mp):
            model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
                pretrained_path=mp,
                action_dim=args.action_dim, action_chunk=args.action_chunk,
                use_tactile_deform=bool(args.use_tactile_deform),
                use_robot_state=bool(args.use_robot_state),
                torch_dtype=torch.bfloat16,
                tactile_intermediate_size=tac_isize,
                n_flare_tokens_per_frame=args.n_flare_tokens_per_frame,
                n_flare_steps=args.n_flare_steps,
            )
        else:
            raise FileNotFoundError(f"Cannot reconstruct model from {ckpt}")

    _attach_gate_and_tflare(model, args)

    sd = torch.load(os.path.join(ckpt, "model.pt"), map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing: print(f"  missing (first 10): {missing[:10]}")
    model = model.to(torch.bfloat16)

    stats_path = args.stats_path or ""
    if not stats_path:
        candidate = os.path.join(ckpt, "stats_data.json")
        if os.path.exists(candidate):
            stats_path = candidate
    if not stats_path or not os.path.exists(stats_path):
        raise FileNotFoundError("Cannot find stats JSON.")
    with open(stats_path) as f:
        stats_raw = json.load(f)
    ds = args.dataset_name if args.dataset_name and args.dataset_name in stats_raw else next(iter(stats_raw))
    def _arr(k, s): return np.array(stats_raw[ds][k][s])
    statistic = {
        "action_mask": _arr("action", "mask"),
        "action_min":  _arr("action", "q01"),
        "action_max":  _arr("action", "q99"),
        "tacf6_mask":  _arr("tactile_f6", "mask"),
        "tacf6_min":   _arr("tactile_f6", "q01"),
        "tacf6_max":   _arr("tactile_f6", "q99"),
    }
    if args.use_robot_state:
        statistic["state_mask"] = _arr("state", "mask")
        statistic["state_min"]  = _arr("state", "q01")
        statistic["state_max"]  = _arr("state", "q99")

    action_tokenizer = ActionTokenizer(processor.tokenizer)
    return model, processor, statistic, action_tokenizer


@torch.inference_mode()
def denoise_gated(
    model, inputs_embeds, position_ids, attention_mask,
    noise, num_steps,
    state_embeds, tactile_f6, tactile_deform, fast_embeds,
    K_tac=0, return_diagnostics=False,
):
    """KV-cached Euler denoising with gated fusion + tactile-FLARE tokens.

    Step 0 processes the full sequence and caches all KVs. Subsequent
    steps crop the action+tactile tail from the cache and re-feed just
    those tokens (their content depends on the new x_t). The latent
    block (slow + flare_q) stays in cache — it's constant across steps.
    Mirrors Qwen3VLVLAModel.denoise_step in modeling_vla.py.
    """
    device, dtype = noise.device, noise.dtype
    dt   = torch.tensor(-1.0 / num_steps, dtype=dtype, device=device)
    x_t  = noise.to(dtype)
    time = torch.tensor(1.0, dtype=dtype, device=device)
    B, H = inputs_embeds.shape[0], inputs_embeds.shape[2]

    # ── Constant across steps ──────────────────────────────────────────
    tac_parts = []
    if tactile_f6 is not None:
        tac_parts.append(model.tacf6_embedder(tactile_f6.to(dtype)))
    if tactile_deform is not None:
        Bs, nf, C, Hh, Ww = tactile_deform.shape
        feats = model.deform_encoder(tactile_deform.view(-1, C, Hh, Ww))
        feats = feats.view(Bs, nf, -1)
        tac_parts.append(model.deform_proj(feats.to(dtype)))
    tactile_embeds = (torch.cat(tac_parts, dim=1) if tac_parts
                      else torch.empty((B, 0, H), device=device, dtype=dtype))
    has_tactile = tactile_embeds.shape[1] > 0

    tflare_q = None
    if has_tactile and K_tac > 0 and hasattr(model, "tactile_flare_queries"):
        tflare_q = model.tactile_flare_queries.expand(B, -1, -1).to(device=device, dtype=dtype)

    if fast_embeds is None:
        fast_embeds = torch.empty((B, 0, H), device=device, dtype=dtype)
    if state_embeds is None:
        state_embeds = torch.empty((B, 0, H), device=device, dtype=dtype)

    n_fast   = fast_embeds.shape[1]
    n_state  = state_embeds.shape[1]
    chunk    = x_t.shape[1]
    n_action = n_fast + n_state + 1 + chunk
    n_tactile = ((tactile_embeds.shape[1]
                  + (tflare_q.shape[1] if tflare_q is not None else 0)
                  + 1 + chunk)
                 if has_tactile else 0)

    diag = {"gates": [], "v_act_norm": [], "v_tac_norm": []} if return_diagnostics else None
    past_kv = None

    while time >= -dt / 2:
        expanded_time = time.expand(B)
        noisy_actions = model.x_embedder(x_t)
        timesteps = model.t_embedder(expanded_time).unsqueeze(1)
        act_seq = torch.cat([fast_embeds, state_embeds, timesteps, noisy_actions], dim=1)

        tac_seq = None
        if has_tactile:
            noisy_actions_tac = model.x_embedder(x_t)
            timesteps_tac = model.t_embedder(expanded_time).unsqueeze(1)
            t_parts = [tactile_embeds]
            if tflare_q is not None:
                t_parts.append(tflare_q)
            t_parts.extend([timesteps_tac, noisy_actions_tac])
            tac_seq = torch.cat(t_parts, dim=1)

        if past_kv is None:
            # ── Step 0: full sequence, fill cache ──────────────────────
            parts = [inputs_embeds, act_seq]
            if tac_seq is not None:
                parts.append(tac_seq)
            full_embeds = torch.cat(parts, dim=1)
            L_latent = inputs_embeds.shape[1]
            L_total  = full_embeds.shape[1]
            latent_indexes  = torch.arange(0, L_latent, device=device)
            action_indexes  = torch.arange(L_latent, L_latent + n_action, device=device)
            tactile_indexes = (torch.arange(L_latent + n_action, L_total, device=device)
                               if has_tactile else torch.arange(0, 0, device=device))

            outputs = model.model(
                inputs_embeds=full_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                latent_indexes=latent_indexes,
                action_indexes=action_indexes,
                tactile_indexes=tactile_indexes,
            )
        else:
            # ── Step 1+: crop stale action+tactile KVs, re-feed those ──
            parts = [act_seq]
            if tac_seq is not None:
                parts.append(tac_seq)
            full_embeds = torch.cat(parts, dim=1)
            past_kv.crop(-(n_action + n_tactile))

            extended_pos = model.model._extend_position_ids(
                position_ids, n_action, n_tactile)
            act_tac_pos = extended_pos[..., -(n_action + n_tactile):]

            L_fed = full_embeds.shape[1]
            latent_indexes  = torch.arange(0, 0, device=device)
            action_indexes  = torch.arange(0, n_action, device=device)
            tactile_indexes = (torch.arange(n_action, L_fed, device=device)
                               if has_tactile else torch.arange(0, 0, device=device))

            outputs = model.model(
                inputs_embeds=full_embeds,
                position_ids=act_tac_pos,
                past_key_values=past_kv,
                use_cache=True,
                latent_indexes=latent_indexes,
                action_indexes=action_indexes,
                tactile_indexes=tactile_indexes,
            )

        past_kv = outputs.past_key_values
        hidden = outputs.last_hidden_state

        # Same extraction rule in both branches: noisy_actions sits right
        # before the tactile block (if any), and noisy_actions_tac is the
        # last `chunk` tokens of the sequence.
        act_block_end = hidden.shape[1] - n_tactile
        h_act_chunk = hidden[:, act_block_end - chunk:act_block_end, :]
        v_act = model.final_layer(h_act_chunk)

        if has_tactile:
            h_tac_chunk = hidden[:, -chunk:, :]
            v_tac = model.final_layer_tactile(h_tac_chunk)
            g = torch.sigmoid(model.gate_head(torch.cat([h_act_chunk, h_tac_chunk], dim=-1)))
            v_t = (1.0 - g) * v_act + g * v_tac
            if diag is not None:
                diag["gates"].append(g.detach().float().cpu())
                diag["v_act_norm"].append(v_act.detach().float().norm(dim=-1).mean().item())
                diag["v_tac_norm"].append(v_tac.detach().float().norm(dim=-1).mean().item())
        else:
            v_t = v_act

        x_t = x_t + dt * v_t
        time = time + dt

    if return_diagnostics:
        return x_t, diag
    return x_t


def model_predict(args, model, processor, statistic, action_tokenizer,
                  task_description, slow_images, fast_images,
                  tactile_f6_input=None, tactile_deform_input=None, state_fast=None,
                  return_diagnostics=False):
    device = f"cuda:{args.cuda}"
    model = model.to(device).eval()

    with torch.inference_mode():
        if args.image_size:
            _sz = tuple(args.image_size)
            slow_images = [img.resize(_sz, Image.LANCZOS) for img in slow_images]
            fast_images = [img.resize(_sz, Image.LANCZOS) for img in fast_images]

        state_embeds = None
        if args.use_robot_state and state_fast is not None:
            norm_state = _normalize(
                np.array(state_fast, dtype=np.float32),
                statistic["state_mask"], statistic["state_min"], statistic["state_max"])
            sv = torch.tensor(norm_state, dtype=torch.bfloat16).unsqueeze(0).to(device)
            state_embeds = model.state_embedder(sv).unsqueeze(1)

        n_slow = len(slow_images)
        all_pil = slow_images + fast_images
        content = []
        for _ in slow_images: content.append({"type": "image"})
        content.append({"type": "text", "text": task_description})
        for _ in fast_images: content.append({"type": "image"})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = processor(text=text, images=all_pil if all_pil else None, return_tensors="pt", padding=False)

        input_ids = inp.input_ids.to(device)
        attention_mask = inp.attention_mask.to(device)
        pixel_values = (inp.pixel_values.to(device, dtype=torch.bfloat16)
                        if getattr(inp, "pixel_values", None) is not None else None)
        image_grid_thw = (inp.image_grid_thw.to(device)
                          if getattr(inp, "image_grid_thw", None) is not None else None)

        inputs_embeds = model.prepare_inputs_embeds(
            input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

        fast_embeds = None
        if image_grid_thw is not None and fast_images:
            merge = getattr(model.visual, "spatial_merge_size",
                            getattr(processor.image_processor, "merge_size", 2))
            n_slow_img_tokens = sum(
                int(g[0] * (g[1] // merge) * (g[2] // merge))
                for g in image_grid_thw[:n_slow])
            slow_embeds, fast_embeds = split_slow_fast_embeds(
                inputs_embeds, input_ids, model.image_token_id, n_slow_img_tokens)
        else:
            slow_embeds = inputs_embeds

        position_ids, _ = model.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
        position_ids = position_ids[:, :, :slow_embeds.shape[1]]

        if model.n_flare_tokens > 0:
            flare_q = model.flare_queries.to(device=slow_embeds.device, dtype=slow_embeds.dtype)
            slow_embeds = torch.cat([slow_embeds, flare_q.expand(1, -1, -1)], dim=1)
            position_ids = extend_position_ids_for_flare(position_ids, model.n_flare_tokens)

        tac_f6_tensor = None
        if args.use_tactile_vec and tactile_f6_input is not None:
            tacf6 = np.array(tactile_f6_input, dtype=np.float32).reshape(-1)
            norm_tacf6 = _normalize(tacf6, statistic["tacf6_mask"],
                                    statistic["tacf6_min"], statistic["tacf6_max"])
            tac_f6_tensor = (torch.tensor(norm_tacf6.reshape(-1, 6), dtype=torch.bfloat16)
                             .unsqueeze(0).to(device))

        tac_deform_tensor = None
        if args.use_tactile_deform and tactile_deform_input is not None:
            if isinstance(tactile_deform_input, (list, tuple)):
                arr = np.stack([
                    (np.array(t, dtype=np.float32) / 255.0 if t.dtype == np.uint8
                     else np.array(t, dtype=np.float32))
                    for t in tactile_deform_input])
            else:
                arr = np.array(tactile_deform_input, dtype=np.float32)
                if arr.max() > 1.0:
                    arr = arr / 255.0
            if arr.ndim == 3:
                tac_deform_tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(2).to(device, dtype=torch.bfloat16)
            elif arr.ndim == 4:
                tac_deform_tensor = torch.tensor(arr).unsqueeze(0).to(device, dtype=torch.bfloat16)

        K_tac = (args.n_tfl_tokens_per_step * args.n_tfl_steps) if args.use_tactile_flare else 0

        noise = torch.randn(1, args.action_chunk, args.action_dim,
                            dtype=torch.bfloat16, device=device)

        result = denoise_gated(
            model=model,
            inputs_embeds=slow_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            noise=noise,
            num_steps=10,
            state_embeds=state_embeds,
            tactile_f6=tac_f6_tensor,
            tactile_deform=tac_deform_tensor,
            fast_embeds=fast_embeds,
            K_tac=K_tac,
            return_diagnostics=return_diagnostics,
        )
        samples, diag = (result if return_diagnostics else (result, None))

        norm_actions = samples[0].float().cpu().numpy()
        actions = _denormalize(norm_actions, statistic["action_mask"],
                               statistic["action_min"], statistic["action_max"])

    if return_diagnostics:
        return list(actions), diag
    return list(actions)


def main(args):
    print(f"Loading VLA model from checkpoint: {args.checkpoint_path}")
    model, processor, statistic, action_tokenizer = model_load(args)
    print("Model loaded successfully!")

    print("Warming up model...")
    dummy_slow  = [Image.new("RGB", (224, 224), color="black")]
    n_fast_cams = 2 if args.action_dim > 31 else 1
    dummy_fast  = [Image.new("RGB", (224, 224), color="black") for _ in range(n_fast_cams)]
    dummy_state = np.zeros(args.action_dim, dtype=np.float32) if args.use_robot_state else None
    dummy_f6    = np.zeros((5, 6), dtype=np.float32) if args.use_tactile_vec else None
    dummy_deform = np.zeros((5, 240, 240), dtype=np.float32) if args.use_tactile_deform else None

    dummy_out = model_predict(args, model, processor, statistic, action_tokenizer,
                              "dummy task", dummy_slow, dummy_fast, dummy_f6, dummy_deform, dummy_state)
    print(f"Warm-up output shape: {np.array(dummy_out).shape}")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"VLA Server listening on port {args.port}...")

    step_counter = 0
    while True:
        try:
            payload = pickle.loads(socket.recv())
            slow_img = Image.open(io.BytesIO(payload["image_head"])).convert("RGB")
            fast_list = [Image.open(io.BytesIO(payload["image_wrist_right"])).convert("RGB")]
            if "image_wrist_left" in payload:
                fast_list.append(Image.open(io.BytesIO(payload["image_wrist_left"])).convert("RGB"))

            tac_f6 = payload.get("tactile_f6") if args.use_tactile_vec else None
            tac_deform = None
            if args.use_tactile_deform:
                tac_deform = payload.get("tactile_deform", payload.get("tactile_image_deform"))

            actions = model_predict(args, model, processor, statistic, action_tokenizer,
                                    payload["task_description"], [slow_img], fast_list,
                                    tac_f6, tac_deform, payload.get("state_fast"))
            socket.send(pickle.dumps({"status": "success", "actions": actions}))
            step_counter += 1
            if step_counter % 10 == 0:
                print(f"Processed {step_counter} requests. Task: {payload['task_description']}")
        except Exception as e:
            traceback.print_exc()
            socket.send(pickle.dumps({"status": "error", "message": str(e)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-world ZMQ server (gated fusion + tactile-FLARE)")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument("--stats_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--action_dim", type=int, default=31)
    parser.add_argument("--action_chunk", type=int, default=8)
    parser.add_argument("--use_robot_state", type=int, default=0)
    parser.add_argument("--use_tactile_deform", type=int, default=1)
    parser.add_argument("--use_tactile_vec", type=int, default=0)
    parser.add_argument("--tactile_intermediate_size", type=int, default=0)
    parser.add_argument("--n_flare_tokens_per_frame", type=int, default=0,
                        help="0 = auto-detect from training_args.json")
    parser.add_argument("--n_flare_steps", type=int, default=0,
                        help="0 = auto-detect from training_args.json")
    parser.add_argument("--use_tactile_flare", type=int, default=0,
                        help="0 = auto-detect from training_args.json")
    parser.add_argument("--n_tfl_tokens_per_step", type=int, default=0)
    parser.add_argument("--n_tfl_steps", type=int, default=0)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))

    args = parser.parse_args()
    main(args)

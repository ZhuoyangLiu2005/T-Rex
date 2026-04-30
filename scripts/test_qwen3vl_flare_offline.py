"""
Offline test + ZeroMQ inference server for the Qwen3-VL MoT VLA model
with flare visual prediction tokens.

Identical to test_qwen3vl_offline.py except:
  - Builds the model with n_flare_tokens_per_frame * n_flare_steps (auto-detected from training_args.json)
  - Appends flare_queries to slow_embeds before calling forward_flow
  - Extends position_ids by total flare tokens
"""

import os
import sys

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import argparse
import json
import io
import pickle
import traceback

import re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import zmq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoProcessor
from janus.models.action_tokenizer import ActionTokenizer

from qwen_vla import Qwen3VLVLAModel, extend_position_ids_for_flare, split_slow_fast_embeds


def _normalize(values, mask, vmin, vmax):
    return np.where(
        mask,
        np.clip(2.0 * (values - vmin) / (vmax - vmin + 1e-8) - 1.0, -1.0, 1.0),
        values,
    )

def _denormalize(norm_values, mask, vmin, vmax):
    return np.where(
        mask,
        0.5 * (norm_values + 1.0) * (vmax - vmin) + vmin,
        norm_values,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build model from config.json
# ─────────────────────────────────────────────────────────────────────────────

def _build_qwen3vl_from_config(config_path, args):
    with open(config_path) as f:
        full_cfg = json.load(f)

    image_token_id = full_cfg.get("image_token_id", 151655)
    model_type = full_cfg.get("model_type", "qwen2_vl")

    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        vl_config = Qwen3VLConfig(**{k: v for k, v in full_cfg.items()
                                     if k not in ("architectures", "transformers_version")})
        text_config = vl_config.text_config
    except Exception:
        from transformers import AutoConfig
        vl_config = AutoConfig.from_pretrained(
            os.path.dirname(config_path), trust_remote_code=True)
        text_config = getattr(vl_config, "text_config", vl_config)

    tac_isize = getattr(args, "tactile_intermediate_size", 0)
    tac_isize = tac_isize if tac_isize > 0 else None
    n_flare_tpf = getattr(args, "n_flare_tokens_per_frame", 0)
    n_flare_steps = getattr(args, "n_flare_steps", 0)

    model = Qwen3VLVLAModel(
        config             = text_config,
        action_dim         = args.action_dim,
        action_chunk       = args.action_chunk,
        use_tactile_deform = bool(args.use_tactile_deform),
        use_robot_state    = bool(args.use_robot_state),
        image_token_id     = image_token_id,
        tactile_intermediate_size = tac_isize,
        n_flare_tokens_per_frame = n_flare_tpf,
        n_flare_steps            = n_flare_steps,
    )

    vis_cfg_dict = full_cfg.get("vision_config", {})
    try:
        if model_type == "qwen3_vl":
            from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
            vis_cfg = Qwen3VLVisionConfig(**{k: v for k, v in vis_cfg_dict.items()
                                             if k != "model_type"})
            model.visual = Qwen3VLVisionModel(vis_cfg)
        else:
            from transformers.models.qwen2_vl.configuration_qwen2_vl import Qwen2VLVisionConfig
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLVisionModel
            vis_cfg = Qwen2VLVisionConfig(**{k: v for k, v in vis_cfg_dict.items()
                                             if k != "model_type"})
            model.visual = Qwen2VLVisionModel(vis_cfg)
        print(f"  Visual tower created from config")
    except Exception as e:
        print(f"  Warning: visual tower creation failed: {e}")
        model.visual = None

    try:
        if model_type == "qwen3_vl":
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel as _VLModel
        else:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel as _VLModel

        class _RopeStub:
            def __init__(self, cfg):
                self.config = cfg
            def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
                return _VLModel.get_rope_index(
                    self, input_ids=input_ids,
                    image_grid_thw=image_grid_thw, attention_mask=attention_mask)

        object.__setattr__(model, '_rope_index_fn', _RopeStub(vl_config).get_rope_index)
        print("  M-RoPE helper ready.")
    except Exception as e:
        print(f"  Warning: rope index setup failed ({e}).")

    return model


def _has_hf_weights(path):
    import glob as _glob
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        if _glob.glob(os.path.join(path, pattern)):
            return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def model_load(args):
    ckpt = args.checkpoint_path

    # Auto-detect from training_args.json
    ta_path = os.path.join(ckpt, "training_args.json")
    if os.path.exists(ta_path):
        with open(ta_path) as f:
            ta = json.load(f)
        for key, default in [("tactile_intermediate_size", 0),
                             ("n_flare_tokens_per_frame", 0),
                             ("n_flare_steps", 0),
                             ("flare_layer_index", -1)]:
            saved = ta.get(key, default)
            cli_val = getattr(args, key, default)
            if saved != default and cli_val == default:
                setattr(args, key, saved)
                print(f"Auto-detected {key}={saved} from training_args.json")

    tac_isize = args.tactile_intermediate_size if args.tactile_intermediate_size > 0 else None
    n_flare_tpf = getattr(args, "n_flare_tokens_per_frame", 0)
    n_flare_steps = getattr(args, "n_flare_steps", 0)

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
            n_flare_tokens_per_frame=n_flare_tpf,
            n_flare_steps=n_flare_steps,
        )
    elif os.path.exists(ckpt_config):
        model = _build_qwen3vl_from_config(ckpt_config, args)
    else:
        pretrained_path = None
        if os.path.exists(ta_path):
            mp = ta.get("model_path", "")
            if mp and os.path.isdir(mp) and _has_hf_weights(mp):
                pretrained_path = mp
        if pretrained_path is None:
            raise FileNotFoundError(f"Cannot reconstruct model from {ckpt}")
        model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
            pretrained_path=pretrained_path,
            action_dim=args.action_dim, action_chunk=args.action_chunk,
            use_tactile_deform=bool(args.use_tactile_deform),
            use_robot_state=bool(args.use_robot_state),
            torch_dtype=torch.bfloat16,
            tactile_intermediate_size=tac_isize,
            n_flare_tokens_per_frame=n_flare_tpf,
            n_flare_steps=n_flare_steps,
        )

    ckpt_file = os.path.join(ckpt, "model.pt")
    sd = torch.load(ckpt_file, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Checkpoint loaded: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  missing (first 10): {missing[:10]}")
    model = model.to(torch.bfloat16)

    n_flare_total = n_flare_tpf * n_flare_steps
    if n_flare_total > 0:
        print(f"Flare alignment: {n_flare_steps} steps × {n_flare_tpf} tok/frame = {n_flare_total} total tokens")

    stats_path = args.stats_path or ""
    if not stats_path:
        for c in [os.path.join(ckpt, "stats_data.json"),
                  args.test_json_path.replace(".json", "_statistics.json") if args.test_json_path else ""]:
            if c and os.path.exists(c):
                stats_path = c
                break
    if not stats_path or not os.path.exists(stats_path):
        raise FileNotFoundError("Cannot find stats JSON.")

    with open(stats_path) as f:
        stats_raw = json.load(f)
    ds = args.dataset_name if args.dataset_name and args.dataset_name in stats_raw \
         else next(iter(stats_raw))

    def _arr(key, sub):
        return np.array(stats_raw[ds][key][sub])

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


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def model_predict(
    args, model, processor, statistic, action_tokenizer,
    task_description, slow_images, fast_images,
    tactile_f6_input=None, tactile_deform_input=None, state_fast=None,
):
    device = f"cuda:{args.cuda}"
    model = model.to(device).eval()

    with torch.inference_mode():

        if args.image_size:
            _sz = tuple(args.image_size)
            slow_images = [img.resize(_sz, Image.LANCZOS) for img in slow_images]
            fast_images = [img.resize(_sz, Image.LANCZOS) for img in fast_images]

        # ── State ────────────────────────────────────────────────────────
        state_embeds = None
        if args.use_robot_state and state_fast is not None:
            norm_state = _normalize(
                np.array(state_fast, dtype=np.float32),
                statistic["state_mask"], statistic["state_min"], statistic["state_max"])
            state_vec = torch.tensor(norm_state, dtype=torch.bfloat16).unsqueeze(0).to(device)
            state_embeds = model.state_embedder(state_vec).unsqueeze(1)

        # ── Single message: [slow_imgs | text | fast_imgs] ──────────────
        n_slow = len(slow_images)
        all_pil = slow_images + fast_images

        content = []
        for _ in slow_images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": task_description})
        for _ in fast_images:
            content.append({"type": "image"})

        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inp = processor(text=text, images=all_pil if all_pil else None,
                        return_tensors="pt", padding=False)

        input_ids = inp.input_ids.to(device)
        attention_mask = inp.attention_mask.to(device)
        pixel_values = (inp.pixel_values.to(device, dtype=torch.bfloat16)
                        if getattr(inp, "pixel_values", None) is not None else None)
        image_grid_thw = (inp.image_grid_thw.to(device)
                          if getattr(inp, "image_grid_thw", None) is not None else None)

        inputs_embeds = model.prepare_inputs_embeds(
            input_ids=input_ids, pixel_values=pixel_values,
            image_grid_thw=image_grid_thw)

        # ── Split slow / fast ────────────────────────────────────────────
        fast_embeds = None
        if image_grid_thw is not None and fast_images:
            merge = getattr(model.visual, "spatial_merge_size",
                            getattr(processor.image_processor, "merge_size", 2))
            n_slow_img_tokens = sum(
                int(g[0] * (g[1] // merge) * (g[2] // merge))
                for g in image_grid_thw[:n_slow])
            slow_embeds, fast_embeds = split_slow_fast_embeds(
                inputs_embeds, input_ids,
                model.image_token_id, n_slow_img_tokens)
        else:
            slow_embeds = inputs_embeds

        # ── M-RoPE: full sequence, truncate to slow ─────────────────────
        position_ids, _ = model.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw,
            attention_mask=attention_mask)
        position_ids = position_ids[:, :, :slow_embeds.shape[1]]

        # ── Append flare query tokens to slow_embeds ────────────────────
        if model.n_flare_tokens > 0:
            flare_q = model.flare_queries.to(
                device=slow_embeds.device, dtype=slow_embeds.dtype)
            slow_embeds = torch.cat([slow_embeds, flare_q.expand(1, -1, -1)], dim=1)
            position_ids = extend_position_ids_for_flare(
                position_ids, model.n_flare_tokens)

        # ── Tactile ──────────────────────────────────────────────────────
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
                    (np.array(t, dtype=np.float32) / 255.0
                     if t.dtype == np.uint8 else np.array(t, dtype=np.float32))
                    for t in tactile_deform_input])
            else:
                arr = np.array(tactile_deform_input, dtype=np.float32)
                if arr.max() > 1.0:
                    arr = arr / 255.0
            if arr.ndim == 3:
                tac_deform_tensor = (torch.tensor(arr).unsqueeze(0).unsqueeze(2)
                                     .to(device, dtype=torch.bfloat16))
            elif arr.ndim == 4:
                tac_deform_tensor = (torch.tensor(arr).unsqueeze(0)
                                     .to(device, dtype=torch.bfloat16))

        # ── Flow-matching denoising ──────────────────────────────────────
        noise = torch.randn(1, args.action_chunk, args.action_dim,
                            dtype=torch.bfloat16, device=device)

        if getattr(args, "use_tactile_refine_flow", 0):
            # Paradigm C: action-only slow flow → Â, then tactile residual flow → Δa
            a_hat, cached_kv, n_action_in_cache = model.forward_flow_action_only(
                inputs_embeds   = slow_embeds,
                position_ids    = position_ids,
                attention_mask  = attention_mask,
                noise           = noise,
                state_embeds    = state_embeds,
                fast_embeds     = fast_embeds,
                num_steps       = getattr(args, "action_flow_eval_steps", 10),
                refresh_clean_kv= True,
            )
            if (tac_f6_tensor is not None) or (tac_deform_tensor is not None):
                delta_a = model.tactile_residual_flow(
                    cached_kv          = cached_kv,
                    latent_position_ids= position_ids,
                    n_action_in_cache  = n_action_in_cache,
                    base_chunk         = a_hat,
                    tactile_f6         = tac_f6_tensor,
                    tactile_deform     = tac_deform_tensor,
                    num_steps          = getattr(args, "tactile_refine_flow_steps", 4),
                    noise_scale        = getattr(args, "tactile_refine_noise_scale", 0.1),
                )
                samples = a_hat + delta_a
            else:
                samples = a_hat
        else:
            samples = model.forward_flow(
                inputs_embeds  = slow_embeds,
                position_ids   = position_ids,
                attention_mask = attention_mask,
                noise          = noise,
                num_steps      = 10,
                state_embeds   = state_embeds,
                tactile_f6     = tac_f6_tensor,
                tactile_deform = tac_deform_tensor,
                fast_embeds    = fast_embeds,
            )

        norm_actions = samples[0].float().cpu().numpy()
        actions = _denormalize(
            norm_actions, statistic["action_mask"],
            statistic["action_min"], statistic["action_max"])

    return list(actions)


# ─────────────────────────────────────────────────────────────────────────────
# Flare similarity evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _load_flare_frame(slow_path, k, frame_stride, data_dir, image_size=None):
    """Load the k-th future frame relative to the current slow image."""
    match = re.search(r'image(\d+)_', os.path.basename(slow_path))
    if not match:
        return None
    current_idx = int(match.group(1))
    flare_idx = current_idx + (k + 1) * frame_stride
    flare_path = re.sub(r'image\d+_', f'image{flare_idx}_', slow_path)
    full_path = os.path.join(data_dir, flare_path) if not os.path.isabs(flare_path) else flare_path
    if not os.path.exists(full_path):
        return None
    img = Image.open(full_path).convert("RGB")
    if image_size is not None:
        img = img.resize(image_size, Image.LANCZOS)
    return img


def compute_flare_similarity(
    args, model, processor, statistic,
    sample, data_dir,
    slow_images, fast_images,
    tactile_f6_input=None, tactile_deform_input=None, state_fast=None,
):
    """
    Run a single forward pass (t=1.0, pure noise) and compare flare predictions
    against ViT-encoded future frame representations.

    Returns
    -------
    dict with:
      "mean_cos_sim": float – average cosine similarity across all K flare tokens
      "per_step_cos_sim": list[float] – average cosine similarity per future step
                          (averaged over tokens_per_frame within each step)
    or None if flare is disabled or future frames are unavailable.
    """
    if model.n_flare_tokens == 0:
        return None

    n_tpf = model.n_flare_tokens_per_frame
    n_steps = model.n_flare_steps
    K = model.n_flare_tokens
    flare_layer_idx = getattr(args, "flare_layer_index", -1)
    frame_stride = args.flare_frame_stride
    device = f"cuda:{args.cuda}"
    img_size = tuple(args.image_size) if args.image_size else None

    # ── Load future frames ──────────────────────────────────────────────
    slow_path = sample["input_image_slow"][0]
    future_pil = []
    for k in range(n_steps):
        fimg = _load_flare_frame(slow_path, k, frame_stride, data_dir, img_size)
        if fimg is None:
            return None  # can't evaluate if future frames are out of range
        future_pil.append(fimg)

    with torch.inference_mode():
        # ── Encode future frames → ViT targets ─────────────────────────
        flare_inp = processor.image_processor(future_pil, return_tensors="pt")
        f_pv = flare_inp.pixel_values.to(device, dtype=torch.bfloat16)
        f_thw = flare_inp.image_grid_thw.to(device)

        vit_out = model.visual(f_pv, grid_thw=f_thw)
        features = vit_out[0] if isinstance(vit_out, (tuple, list)) else vit_out

        merge = getattr(model.visual, "spatial_merge_size", 2)
        frame_feats = []
        offset = 0
        for g in f_thw:
            n_tok = int(g[0] * (g[1] // merge) * (g[2] // merge))
            frame_tokens = features[offset: offset + n_tok]  # [n_tok, H]
            pooled = F.adaptive_avg_pool1d(
                frame_tokens.unsqueeze(0).permute(0, 2, 1),  # [1, H, n_tok]
                n_tpf,
            ).permute(0, 2, 1).squeeze(0)  # [n_tpf, H]
            frame_feats.append(pooled)
            offset += n_tok
        flare_targets = torch.stack(frame_feats).view(1, K, -1)  # [1, K, H]

        # ── Build inputs for a single forward pass ──────────────────────
        # Reuse the same preprocessing as model_predict
        if img_size:
            slow_images = [img.resize(img_size, Image.LANCZOS) for img in slow_images]
            fast_images = [img.resize(img_size, Image.LANCZOS) for img in fast_images]

        n_slow = len(slow_images)
        all_pil = slow_images + fast_images
        content = []
        for _ in slow_images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": sample.get("input_prompt", "")})
        for _ in fast_images:
            content.append({"type": "image"})
        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inp = processor(text=text, images=all_pil if all_pil else None,
                        return_tensors="pt", padding=False)

        input_ids = inp.input_ids.to(device)
        attention_mask = inp.attention_mask.to(device)
        pixel_values = (inp.pixel_values.to(device, dtype=torch.bfloat16)
                        if getattr(inp, "pixel_values", None) is not None else None)
        image_grid_thw = (inp.image_grid_thw.to(device)
                          if getattr(inp, "image_grid_thw", None) is not None else None)

        inputs_embeds = model.prepare_inputs_embeds(
            input_ids=input_ids, pixel_values=pixel_values,
            image_grid_thw=image_grid_thw)

        # Split slow / fast
        fast_embeds = inputs_embeds[:, :0]
        if image_grid_thw is not None and fast_images:
            mg = getattr(model.visual, "spatial_merge_size",
                         getattr(processor.image_processor, "merge_size", 2))
            n_slow_img_tokens = sum(
                int(g[0] * (g[1] // mg) * (g[2] // mg))
                for g in image_grid_thw[:n_slow])
            slow_embeds, fast_embeds = split_slow_fast_embeds(
                inputs_embeds, input_ids,
                model.image_token_id, n_slow_img_tokens)
        else:
            slow_embeds = inputs_embeds

        L_slow = slow_embeds.shape[1]

        # M-RoPE
        position_ids, _ = model.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw,
            attention_mask=attention_mask)
        position_ids = position_ids[:, :, :L_slow]

        # Append flare queries
        flare_q = model.flare_queries.to(device=slow_embeds.device, dtype=slow_embeds.dtype)
        slow_embeds_ext = torch.cat([slow_embeds, flare_q.expand(1, -1, -1)], dim=1)
        position_ids = extend_position_ids_for_flare(position_ids, K)
        L_latent = slow_embeds_ext.shape[1]  # L_slow + K

        # State
        state_embeds = torch.empty((1, 0, slow_embeds.shape[2]),
                                    device=device, dtype=slow_embeds.dtype)
        if args.use_robot_state and state_fast is not None:
            ns = _normalize(np.array(state_fast, dtype=np.float32),
                            statistic["state_mask"], statistic["state_min"], statistic["state_max"])
            sv = torch.tensor(ns, dtype=torch.bfloat16).unsqueeze(0).to(device)
            state_embeds = model.state_embedder(sv).unsqueeze(1)
        n_state = state_embeds.shape[1]

        # Dummy action tokens at t=1.0 (first denoising step)
        noise = torch.randn(1, args.action_chunk, args.action_dim,
                            dtype=torch.bfloat16, device=device)
        noisy_actions = model.x_embedder(noise)
        t_one = torch.tensor([1.0], dtype=torch.bfloat16, device=device)
        timesteps = model.t_embedder(t_one).unsqueeze(1)

        # Tactile embeddings
        tac_parts = []
        if args.use_tactile_vec and tactile_f6_input is not None:
            tacf6 = np.array(tactile_f6_input, dtype=np.float32).reshape(-1)
            norm_tacf6 = _normalize(tacf6, statistic["tacf6_mask"],
                                    statistic["tacf6_min"], statistic["tacf6_max"])
            tf6 = torch.tensor(norm_tacf6.reshape(-1, 6), dtype=torch.bfloat16).unsqueeze(0).to(device)
            tac_parts.append(model.tacf6_embedder(tf6))
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
            deform_t = torch.tensor(arr).to(device, dtype=torch.bfloat16)
            if deform_t.ndim == 3:
                deform_t = deform_t.unsqueeze(0).unsqueeze(2)
            elif deform_t.ndim == 4:
                deform_t = deform_t.unsqueeze(0)
            Bs, nf, C, H, W = deform_t.shape
            feats = model.deform_encoder(deform_t.view(-1, C, H, W))
            feats = feats.view(Bs, nf, -1)
            tac_parts.append(model.deform_proj(feats))

        tactile_embeds = (torch.cat(tac_parts, dim=1) if tac_parts
                          else torch.empty((1, 0, slow_embeds.shape[2]),
                                           device=device, dtype=slow_embeds.dtype))
        has_tactile = tactile_embeds.shape[1] > 0

        # ── Build full sequence & forward ───────────────────────────────
        n_fast = fast_embeds.shape[1]
        n_action = n_fast + n_state + 1 + args.action_chunk

        if has_tactile:
            noisy_actions_tac = model.x_embedder(noise)
            timesteps_tac = model.t_embedder(t_one).unsqueeze(1)
            full_embeds = torch.cat([
                slow_embeds_ext,
                fast_embeds, state_embeds, timesteps, noisy_actions,
                tactile_embeds, timesteps_tac, noisy_actions_tac,
            ], dim=1)
        else:
            full_embeds = torch.cat([
                slow_embeds_ext,
                fast_embeds, state_embeds, timesteps, noisy_actions,
            ], dim=1)

        L_total = full_embeds.shape[1]
        dev = full_embeds.device
        latent_indexes  = torch.arange(0, L_latent, device=dev)
        action_indexes  = torch.arange(L_latent, L_latent + n_action, device=dev)
        if has_tactile:
            tactile_indexes = torch.arange(L_latent + n_action, L_total, device=dev)
        else:
            tactile_indexes = torch.arange(0, 0, device=dev)

        need_all_hs = (flare_layer_idx != -1)
        outputs = model.model(
            inputs_embeds=full_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=need_all_hs,
            latent_indexes=latent_indexes,
            action_indexes=action_indexes,
            tactile_indexes=tactile_indexes,
        )

        # ── Extract flare hidden states ─────────────────────────────────
        if flare_layer_idx == -1:
            flare_source = outputs.last_hidden_state
        else:
            all_hs = outputs.hidden_states
            n_layers = len(all_hs) - 1
            li = (n_layers + flare_layer_idx) if flare_layer_idx < 0 else flare_layer_idx
            li = max(0, min(li, n_layers - 1))
            flare_source = all_hs[li + 1]

        flare_hidden = flare_source[:, L_slow: L_slow + K, :]  # [1, K, H]
        flare_pred = model.flare_proj(flare_hidden)             # [1, K, H]

        # ── Cosine similarity ───────────────────────────────────────────
        pred_norm = F.normalize(flare_pred, dim=-1)
        tgt_norm  = F.normalize(flare_targets, dim=-1)
        cos_sim = (pred_norm * tgt_norm).sum(dim=-1)  # [1, K]
        cos_sim = cos_sim.squeeze(0)                   # [K]

        # Per-step: average over tokens_per_frame within each step
        per_step = cos_sim.view(n_steps, n_tpf).mean(dim=1)  # [n_steps]

        return {
            "mean_cos_sim": cos_sim.mean().item(),
            "per_step_cos_sim": per_step.cpu().tolist(),
        }


def main(args):
    print(f"Loading VLA model from checkpoint: {args.checkpoint_path}")
    model, processor, statistic, action_tokenizer = model_load(args)
    print("Model loaded successfully!")

    # Warm-up (use 2 fast images for bimanual / dual-arm tasks)
    print("Warming up model...")
    dummy_slow = [Image.new("RGB", (224, 224), color="black")]
    n_fast_cams = 2 if args.action_dim > 31 else 1
    dummy_fast = [Image.new("RGB", (224, 224), color="black") for _ in range(n_fast_cams)]
    dummy_state = np.zeros(args.action_dim, dtype=np.float32) if args.use_robot_state else None
    dummy_f6 = np.zeros((5, 6), dtype=np.float32) if args.use_tactile_vec else None
    dummy_deform = np.zeros((5, 240, 240), dtype=np.float32) if args.use_tactile_deform else None

    dummy_out = model_predict(
        args, model, processor, statistic, action_tokenizer,
        "dummy task", dummy_slow, dummy_fast, dummy_f6, dummy_deform, dummy_state)
    print(f"Warm-up output shape: {np.array(dummy_out).shape}")

    if args.test_json_path:
        with open(args.test_json_path) as f:
            train_data = json.load(f)
        if args.num_test_samples > 0:
            train_data = train_data[:args.num_test_samples]

        data_dir = os.path.dirname(os.path.abspath(args.test_json_path))
        _abs = lambda p: p if os.path.isabs(p) else os.path.join(data_dir, p)

        error_sum, n_valid = 0, 0
        all_pred, all_gt = [], []

        # Flare evaluation accumulators
        use_flare_eval = (model.n_flare_tokens > 0)
        flare_sim_sum, flare_n_valid = 0.0, 0
        all_per_step_sims = []  # list of per-step similarity arrays

        for step, sample in enumerate(tqdm(train_data, desc="Testing")):
            try:
                slow_images = [Image.open(_abs(p)).convert("RGB")
                               for p in sample["input_image_slow"]]
                fast_images = [Image.open(_abs(p)).convert("RGB")
                               for p in sample["input_image_fast"]]
                tac_f6 = np.array(sample["tactile_f6"], dtype=np.float32) if args.use_tactile_vec else None
                tac_deform = None
                if args.use_tactile_deform:
                    tac_deform = [np.array(Image.open(_abs(p)).convert("L"), dtype=np.float32) / 255.0
                                  for p in sample.get("tactile_image_deform", [])]
                state_fast = np.array(sample["state_fast"], dtype=np.float32) if args.use_robot_state else None
                gt_action = np.array(sample["action"], dtype=np.float32)

                pred = np.array(model_predict(
                    args, model, processor, statistic, action_tokenizer,
                    sample["input_prompt"], slow_images, fast_images,
                    tac_f6, tac_deform, state_fast))

                all_pred.append(pred[0] if pred.ndim > 1 else pred)
                all_gt.append(gt_action[0] if gt_action.ndim > 1 else gt_action)
                mse = np.mean((pred[:min(len(pred), len(gt_action))] -
                               gt_action[:min(len(pred), len(gt_action))]) ** 2)
                error_sum += mse
                n_valid += 1

                # Flare similarity evaluation
                if use_flare_eval:
                    # Re-open images (model_predict may have resized in-place)
                    slow_imgs_flare = [Image.open(_abs(p)).convert("RGB")
                                       for p in sample["input_image_slow"]]
                    fast_imgs_flare = [Image.open(_abs(p)).convert("RGB")
                                       for p in sample["input_image_fast"]]
                    flare_result = compute_flare_similarity(
                        args, model, processor, statistic,
                        sample, data_dir,
                        slow_imgs_flare, fast_imgs_flare,
                        tac_f6, tac_deform, state_fast,
                    )
                    if flare_result is not None:
                        flare_sim_sum += flare_result["mean_cos_sim"]
                        all_per_step_sims.append(flare_result["per_step_cos_sim"])
                        flare_n_valid += 1

            except FileNotFoundError as e:
                print(f"\n[Warning] Step {step}: {e}")

        print(f"\n=== Test: {n_valid}/{len(train_data)} valid, MSE={error_sum/max(n_valid,1):.6f} ===")
        if use_flare_eval and flare_n_valid > 0:
            avg_sim = flare_sim_sum / flare_n_valid
            per_step_avg = np.mean(all_per_step_sims, axis=0)
            print(f"=== Flare: {flare_n_valid} valid, mean cos_sim={avg_sim:.4f} ===")
            for s_i, s_val in enumerate(per_step_avg):
                print(f"    step {s_i} (t+{(s_i+1)*args.flare_frame_stride}): cos_sim={s_val:.4f}")

        if all_pred and args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            all_pred = np.stack(all_pred)
            all_gt = np.stack(all_gt)
            for idx in range(all_pred.shape[1]):
                plt.figure(figsize=(10, 4))
                plt.plot(all_pred[:, idx], label="Predicted", linestyle="--", color="blue")
                plt.plot(all_gt[:, idx], label="GT", alpha=0.6, color="orange")
                plt.title(f"Action Dimension {idx}")
                plt.legend(); plt.grid(True, linestyle=":", alpha=0.7); plt.tight_layout()
                plt.savefig(os.path.join(args.save_dir, f"action_dim_{idx}.png")); plt.close()
            np.savez(os.path.join(args.save_dir, "action_trajectory.npz"),
                     pred=all_pred, gt=all_gt)

            # Save flare similarity results
            if use_flare_eval and all_per_step_sims:
                sims_arr = np.array(all_per_step_sims)  # [n_samples, n_steps]
                np.savez(os.path.join(args.save_dir, "flare_similarity.npz"),
                         per_sample_per_step=sims_arr,
                         per_step_mean=np.mean(sims_arr, axis=0),
                         overall_mean=np.mean(sims_arr))

                # Per-step similarity over samples (trajectory)
                plt.figure(figsize=(10, 4))
                for s_i in range(sims_arr.shape[1]):
                    plt.plot(sims_arr[:, s_i], label=f"step {s_i} (t+{(s_i+1)*args.flare_frame_stride})",
                             alpha=0.7)
                plt.xlabel("Sample"); plt.ylabel("Cosine Similarity")
                plt.title("Flare Prediction Similarity (per step over samples)")
                plt.legend(fontsize=7); plt.grid(True, linestyle=":", alpha=0.7); plt.tight_layout()
                plt.savefig(os.path.join(args.save_dir, "flare_sim_per_step.png")); plt.close()

                # Bar chart: average similarity per future step
                plt.figure(figsize=(8, 4))
                step_means = np.mean(sims_arr, axis=0)
                step_labels = [f"t+{(i+1)*args.flare_frame_stride}" for i in range(len(step_means))]
                plt.bar(step_labels, step_means, color="steelblue", alpha=0.8)
                plt.xlabel("Future Step"); plt.ylabel("Mean Cosine Similarity")
                plt.title("Flare: Mean Similarity per Future Step")
                plt.ylim(0, 1); plt.grid(True, axis="y", linestyle=":", alpha=0.7); plt.tight_layout()
                plt.savefig(os.path.join(args.save_dir, "flare_sim_bar.png")); plt.close()

            print(f"Saved to {args.save_dir}")

    input("\nPress Enter to start ZMQ server...")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"Server listening on port {args.port}...")

    step_counter = 0
    while True:
        try:
            payload = pickle.loads(socket.recv())
            slow_img = Image.open(io.BytesIO(payload["image_head"])).convert("RGB")
            fast_list = [Image.open(io.BytesIO(payload["image_wrist_right"])).convert("RGB")]
            if "image_wrist_left" in payload:
                fast_list.append(Image.open(io.BytesIO(payload["image_wrist_left"])).convert("RGB"))

            tac_f6 = payload.get("tactile_f6") if args.use_tactile_vec else None
            tac_deform = payload.get("tactile_deform", payload.get("tactile_image_deform")) if args.use_tactile_deform else None

            actions = model_predict(
                args, model, processor, statistic, action_tokenizer,
                payload["task_description"], [slow_img], fast_list,
                tac_f6, tac_deform, payload.get("state_fast"))

            socket.send(pickle.dumps({"status": "success", "actions": actions}))
            step_counter += 1
            if step_counter % 10 == 0:
                print(f"Processed {step_counter} requests.")
        except Exception as e:
            traceback.print_exc()
            socket.send(pickle.dumps({"status": "error", "message": str(e)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Offline test + ZMQ server (with flare alignment)")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument("--stats_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--test_json_path", type=str, default="")
    parser.add_argument("--action_dim", type=int, default=31)
    parser.add_argument("--action_chunk", type=int, default=8)
    parser.add_argument("--use_robot_state", type=int, default=0)
    parser.add_argument("--use_tactile_deform", type=int, default=1)
    parser.add_argument("--use_tactile_vec", type=int, default=0)
    parser.add_argument("--tactile_intermediate_size", type=int, default=0)
    parser.add_argument("--n_flare_tokens_per_frame", type=int, default=0, help="0 = auto-detect from training_args.json")
    parser.add_argument("--n_flare_steps", type=int, default=0, help="0 = auto-detect from training_args.json")
    parser.add_argument("--flare_frame_stride", type=int, default=8, help="Temporal stride for flare evaluation future frames")
    parser.add_argument("--flare_layer_index", type=int, default=-1, help="Layer to extract flare hidden states from (-1=last)")
    parser.add_argument("--save_dir", type=str, default="./test_output_flare")
    parser.add_argument("--num_test_samples", type=int, default=300)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))

    # Paradigm C
    parser.add_argument("--use_tactile_refine_flow", type=int, default=0,
                        help="1: action-only slow flow + tactile residual flow refinement.")
    parser.add_argument("--action_flow_eval_steps", type=int, default=10,
                        help="Number of Euler steps for the slow action-only flow.")
    parser.add_argument("--tactile_refine_flow_steps", type=int, default=4,
                        help="Number of Euler steps for the tactile residual flow.")
    parser.add_argument("--tactile_refine_noise_scale", type=float, default=0.1,
                        help="Initial noise magnitude for the residual flow at τ=1.")

    args = parser.parse_args()
    main(args)

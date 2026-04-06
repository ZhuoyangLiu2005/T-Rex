"""
Real-world ZeroMQ inference server for the Qwen3-VL-based MoT VLA model.

Adapted from test_sharpa_tactile_qwen3vl_offline.py with offline evaluation
removed.  Directly starts the ZMQ server to accept real-world robot client
requests.
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

import numpy as np
import torch
from PIL import Image
import zmq
from transformers import AutoProcessor
from janus.models.action_tokenizer import ActionTokenizer

from qwen_vla import Qwen3VLVLAModel


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(values, mask, vmin, vmax):
    """Min-max normalise values to [-1, 1] where mask is True, else leave unchanged."""
    return np.where(
        mask,
        np.clip(2.0 * (values - vmin) / (vmax - vmin + 1e-8) - 1.0, -1.0, 1.0),
        values,
    )


def _denormalize(norm_values, mask, vmin, vmax):
    """Invert _normalize."""
    return np.where(
        mask,
        0.5 * (norm_values + 1.0) * (vmax - vmin) + vmin,
        norm_values,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Build model from config.json only (no HF base model weights needed)
# ─────────────────────────────────────────────────────────────────────────────

def _build_qwen3vl_from_config(config_path, args):
    """
    Construct a Qwen3VLVLAModel shell from config.json alone.
    All weights are expected to be loaded from model.pt afterwards.

    This avoids calling Qwen3VLForConditionalGeneration.from_pretrained()
    which requires safetensors / pytorch_model.bin files that finetuned
    checkpoints do not have.
    """
    with open(config_path) as f:
        full_cfg = json.load(f)

    image_token_id = full_cfg.get("image_token_id", 151655)
    model_type = full_cfg.get("model_type", "qwen2_vl")

    # ── Build text config ─────────────────────────────────────────────────
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

    # ── Resolve tactile_intermediate_size ─────────────────────────────────
    tac_isize = getattr(args, "tactile_intermediate_size", 0)
    tac_isize = tac_isize if tac_isize > 0 else None

    # ── Create VLA model shell ────────────────────────────────────────────
    model = Qwen3VLVLAModel(
        config             = text_config,
        action_dim         = args.action_dim,
        action_chunk       = args.action_chunk,
        use_tactile_deform = bool(args.use_tactile_deform),
        use_robot_state    = bool(args.use_robot_state),
        image_token_id     = image_token_id,
        tactile_intermediate_size = tac_isize,
    )

    # ── Create visual tower from vision_config ────────────────────────────
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
        print(f"  Visual tower created from config (weights from model.pt)")
    except Exception as e:
        print(f"  Warning: visual tower creation failed: {e}")
        model.visual = None

    # ── Set up M-RoPE helper ──────────────────────────────────────────────
    try:
        if model_type == "qwen3_vl":
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel as _VLModel
        else:
            from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel as _VLModel

        class _RopeStub:
            """Minimal stub providing get_rope_index without loading full VL model."""
            def __init__(self, cfg):
                self.config = cfg
            def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
                return _VLModel.get_rope_index(
                    self, input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    attention_mask=attention_mask,
                )

        rope_stub = _RopeStub(vl_config)
        object.__setattr__(model, '_rope_index_fn', rope_stub.get_rope_index)
        print("  M-RoPE helper ready.")
    except Exception as e:
        print(f"  Warning: rope index setup failed ({e}). Fallback to sequential pos_ids.")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def _has_hf_weights(path):
    """Check if a directory contains standard HF model weight files."""
    import glob as _glob
    for pattern in ("*.safetensors", "pytorch_model*.bin"):
        if _glob.glob(os.path.join(path, pattern)):
            return True
    return False


def model_load(args):
    """
    Returns (model, processor, statistic, action_tokenizer).

    checkpoint_path : finetuned checkpoint dir produced by save_checkpoint(), which contains:
                        model.pt          – full model state dict
                        processor/        – tokenizer + image processor
                        config.json       – base model architecture config (copied from base model)
                        stats_data.json   – normalisation statistics
    """
    ckpt = args.checkpoint_path

    # ── 0. Auto-detect tactile_intermediate_size from training_args.json ──────
    ta_path = os.path.join(ckpt, "training_args.json")
    if os.path.exists(ta_path):
        with open(ta_path) as f:
            ta = json.load(f)
        saved_tis = ta.get("tactile_intermediate_size", 0)
        cli_tis = getattr(args, "tactile_intermediate_size", 0)
        if saved_tis > 0 and cli_tis == 0:
            args.tactile_intermediate_size = saved_tis
            print(f"Auto-detected tactile_intermediate_size={saved_tis} from training_args.json")

    tac_isize = getattr(args, "tactile_intermediate_size", 0)
    tac_isize = tac_isize if tac_isize > 0 else None

    # ── 1. Processor ──────────────────────────────────────────────────────────
    proc_dir = os.path.join(ckpt, "processor")
    if not os.path.isdir(proc_dir):
        raise FileNotFoundError(f"processor/ not found in checkpoint: {ckpt}")
    processor = AutoProcessor.from_pretrained(proc_dir, trust_remote_code=True)
    print(f"Processor loaded from: {proc_dir}")

    # ── 2. Model architecture ─────────────────────────────────────────────────
    # Strategy:
    #   a) If --base_model_path is a full HF model dir → use from_pretrained_qwen3vl
    #   b) If checkpoint has config.json (but only model.pt) → build from config
    #   c) If training_args.json points to a base model → use from_pretrained_qwen3vl
    base_model_path = getattr(args, "base_model_path", "")
    ckpt_config = os.path.join(ckpt, "config.json")

    if base_model_path and os.path.isdir(base_model_path) and _has_hf_weights(base_model_path):
        # (a) Full base model available
        print(f"Using base model path (HF weights): {base_model_path}")
        model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
            pretrained_path    = base_model_path,
            action_dim         = args.action_dim,
            action_chunk       = args.action_chunk,
            use_tactile_deform = bool(args.use_tactile_deform),
            use_robot_state    = bool(args.use_robot_state),
            torch_dtype        = torch.bfloat16,
            tactile_intermediate_size = tac_isize,
        )
    elif os.path.exists(ckpt_config):
        # (b) Finetuned checkpoint with config.json + model.pt only
        print(f"Building model from config.json (no HF weights needed): {ckpt}")
        model = _build_qwen3vl_from_config(ckpt_config, args)
    else:
        # (c) Try training_args.json for base model path
        pretrained_path = None
        if os.path.exists(ta_path):
            mp = ta.get("model_path", "")
            if mp and os.path.isdir(mp):
                pretrained_path = mp
                print(f"Using base model from training_args.json: {pretrained_path}")

        if pretrained_path is None:
            raise FileNotFoundError(
                f"Cannot reconstruct model architecture. No config.json in {ckpt}, "
                f"no training_args.json, and no --base_model_path supplied."
            )

        if _has_hf_weights(pretrained_path):
            model = Qwen3VLVLAModel.from_pretrained_qwen3vl(
                pretrained_path    = pretrained_path,
                action_dim         = args.action_dim,
                action_chunk       = args.action_chunk,
                use_tactile_deform = bool(args.use_tactile_deform),
                use_robot_state    = bool(args.use_robot_state),
                torch_dtype        = torch.bfloat16,
                tactile_intermediate_size = tac_isize,
            )
        else:
            raise FileNotFoundError(
                f"Base model path {pretrained_path} has no HF weight files, "
                f"and checkpoint has no config.json. Cannot build model."
            )

    # ── 3. Load finetuned weights ─────────────────────────────────────────────
    ckpt_file = os.path.join(ckpt, "model.pt")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"model.pt not found in checkpoint: {ckpt}")
    sd = torch.load(ckpt_file, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Checkpoint loaded: {ckpt_file}")
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}")
    if missing:
        print(f"  missing keys (first 10): {missing[:10]}")

    model = model.to(torch.bfloat16)

    # ── 4. Normalisation statistics ───────────────────────────────────────────
    stats_path = args.stats_path or ""
    if not stats_path:
        candidate = os.path.join(ckpt, "stats_data.json")
        if os.path.exists(candidate):
            stats_path = candidate

    if not stats_path or not os.path.exists(stats_path):
        raise FileNotFoundError(
            "Cannot find stats JSON. Supply --stats_path explicitly or place "
            "stats_data.json in the checkpoint dir."
        )

    with open(stats_path, "r") as f:
        stats_raw = json.load(f)

    ds = args.dataset_name if args.dataset_name and args.dataset_name in stats_raw \
         else next(iter(stats_raw))
    print(f"Using stats dataset key: '{ds}'")

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
# Single inference call
# ─────────────────────────────────────────────────────────────────────────────

def model_predict(
    args,
    model,
    processor,
    statistic,
    action_tokenizer,
    task_description: str,
    slow_images,          # list[PIL.Image]
    fast_images,          # list[PIL.Image]
    tactile_f6_input=None,    # np.ndarray (T,6) or None
    tactile_deform_input=None, # np.ndarray (N,H,W) or (N,1,H,W) or None
    state_fast=None,      # np.ndarray (state_dim,) or None
):
    """
    Run one denoising flow and return a list of action arrays (shape: action_chunk x action_dim).
    """
    device = f"cuda:{args.cuda}"
    model = model.to(device).eval()

    with torch.inference_mode():

        # ── Optional image resize ────────────────────────────────────────────
        if args.image_size:
            _sz = tuple(args.image_size)
            slow_images = [img.resize(_sz, Image.LANCZOS) for img in slow_images]
            fast_images = [img.resize(_sz, Image.LANCZOS) for img in fast_images]

        # ── State embedding (MLP → single token) ─────────────────────────────
        state_embeds = None
        if args.use_robot_state and state_fast is not None:
            norm_state = _normalize(
                np.array(state_fast, dtype=np.float32),
                statistic["state_mask"],
                statistic["state_min"],
                statistic["state_max"],
            )
            state_vec = torch.tensor(norm_state, dtype=torch.bfloat16).unsqueeze(0).to(device)  # [1, action_dim]
            state_embeds = model.state_embedder(state_vec).unsqueeze(1)  # [1, 1, H]

        # ── Single message: [slow_imgs | text | fast_imgs] ────────────────
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
        inp = processor(
            text=text,
            images=all_pil if all_pil else None,
            return_tensors="pt", padding=False,
        )

        input_ids = inp.input_ids.to(device)
        attention_mask = inp.attention_mask.to(device)
        pixel_values = (inp.pixel_values.to(device, dtype=torch.bfloat16)
                        if getattr(inp, "pixel_values", None) is not None else None)
        image_grid_thw = (inp.image_grid_thw.to(device)
                          if getattr(inp, "image_grid_thw", None) is not None else None)

        inputs_embeds = model.prepare_inputs_embeds(
            input_ids=input_ids, pixel_values=pixel_values,
            image_grid_thw=image_grid_thw)

        # Split into slow (latent) and fast (action) portions
        fast_embeds = None
        if image_grid_thw is not None and fast_images:
            merge = getattr(model.visual, "spatial_merge_size",
                            getattr(processor.image_processor, "merge_size", 2))
            fast_grid = image_grid_thw[n_slow:]
            n_fast_tokens = sum(
                int(g[0] * (g[1] // merge) * (g[2] // merge))
                for g in fast_grid
            )
            slow_embeds = inputs_embeds[:, :-n_fast_tokens]
            fast_embeds = inputs_embeds[:, -n_fast_tokens:]
        else:
            slow_embeds = inputs_embeds

        # M-RoPE: compute for full sequence, truncate to slow portion
        position_ids, _ = model.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw,
            attention_mask=attention_mask)
        position_ids = position_ids[:, :, :slow_embeds.shape[1]]

        # ── Tactile tensors ───────────────────────────────────────────────────
        tac_f6_tensor = None
        if args.use_tactile_vec and tactile_f6_input is not None:
            tacf6 = np.array(tactile_f6_input, dtype=np.float32).reshape(-1)
            norm_tacf6 = _normalize(tacf6, statistic["tacf6_mask"],
                                    statistic["tacf6_min"], statistic["tacf6_max"])
            tac_f6_tensor = (torch.tensor(norm_tacf6.reshape(-1, 6), dtype=torch.bfloat16)
                             .unsqueeze(0).to(device))

        tac_deform_tensor = None
        if args.use_tactile_deform and tactile_deform_input is not None:
            arr = np.array(tactile_deform_input, dtype=np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if arr.ndim == 3:
                tac_deform_tensor = (torch.tensor(arr)
                                     .unsqueeze(0).unsqueeze(2)
                                     .to(device, dtype=torch.bfloat16))
            elif arr.ndim == 4:
                tac_deform_tensor = (torch.tensor(arr)
                                     .unsqueeze(0)
                                     .to(device, dtype=torch.bfloat16))
            else:
                raise ValueError(f"Unexpected tactile_deform shape: {arr.shape}")

        # ── Flow-matching denoising ───────────────────────────────────────────
        noise = torch.randn(1, args.action_chunk, args.action_dim,
                            dtype=torch.bfloat16, device=device)

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
        )                                               # [1, chunk, action_dim]

        # ── Denormalize ───────────────────────────────────────────────────────
        norm_actions = samples[0].float().cpu().numpy()  # [chunk, action_dim]
        actions = _denormalize(
            norm_actions,
            statistic["action_mask"],
            statistic["action_min"],
            statistic["action_max"],
        )

    return list(actions)


def main(args):
    print(f"Loading VLA model (Qwen3-VL) from checkpoint: {args.checkpoint_path}")
    model, processor, statistic, action_tokenizer = model_load(args)
    print("Model loaded successfully!")

    # ── Warm-up forward pass ──────────────────────────────────────────────────
    print("Warming up model...")
    dummy_slow  = [Image.new("RGB", (224, 224), color="black")]
    dummy_fast  = [Image.new("RGB", (224, 224), color="black")]
    dummy_state = np.zeros(args.action_dim, dtype=np.float32) if args.use_robot_state else None
    dummy_f6    = np.zeros((5, 6), dtype=np.float32) if args.use_tactile_vec else None
    dummy_deform = np.zeros((5, 240, 240), dtype=np.float32) if args.use_tactile_deform else None

    dummy_out = model_predict(
        args, model, processor, statistic, action_tokenizer,
        "dummy task", dummy_slow, dummy_fast, dummy_f6, dummy_deform, dummy_state,
    )
    print(f"Warm-up output shape: {np.array(dummy_out).shape}")
    print("Warm-up complete.")

    # ── ZeroMQ Server ─────────────────────────────────────────────────────────
    context = zmq.Context()
    socket  = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"VLA Server (Qwen3-VL) is listening on port {args.port}...")

    step_counter = 0

    while True:
        try:
            message = socket.recv()
            payload = pickle.loads(message)

            slow_img = Image.open(io.BytesIO(payload["image_head"])).convert("RGB")
            fast_wrist_r = Image.open(io.BytesIO(payload["image_wrist_right"])).convert("RGB")

            slow_image_list = [slow_img]
            fast_image_list = [fast_wrist_r]

            # Include left wrist image if present in payload
            if "image_wrist_left" in payload:
                fast_wrist_l = Image.open(io.BytesIO(payload["image_wrist_left"])).convert("RGB")
                fast_image_list.append(fast_wrist_l)

            task_description = payload["task_description"]
            state_fast       = payload.get("state_fast", None)

            tac_f6 = None
            if args.use_tactile_vec:
                tac_f6 = payload.get("tactile_f6")

            tac_deform = None
            if args.use_tactile_deform:
                tac_deform = payload.get("tactile_deform", payload.get("tactile_image_deform"))
                if tac_deform is not None:
                    print(f"tactile_deform shape: {np.array(tac_deform).shape}")

            actions = model_predict(
                args=args,
                model=model,
                processor=processor,
                statistic=statistic,
                action_tokenizer=action_tokenizer,
                task_description=task_description,
                slow_images=slow_image_list,
                fast_images=fast_image_list,
                tactile_f6_input=tac_f6,
                tactile_deform_input=tac_deform,
                state_fast=state_fast,
            )

            response = {"status": "success", "actions": actions}
            socket.send(pickle.dumps(response))
            step_counter += 1

            if step_counter % 10 == 0:
                print(f"Processed {step_counter} requests. Current task: {task_description}")

        except Exception as e:
            traceback.print_exc()
            print(f"Error during prediction: {e}")
            socket.send(pickle.dumps({"status": "error", "message": str(e)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-world ZMQ server for Qwen3-VL-based MoT VLA"
    )

    # Model path
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Finetuned checkpoint dir (model.pt, processor/, config.json, stats_data.json)")
    parser.add_argument("--base_model_path", type=str, default="", help="Path to base Qwen3-VL model (auto-detected if empty).")
    parser.add_argument("--stats_path", type=str, default="", help="Explicit path to stats JSON (auto-detected if empty).")
    parser.add_argument("--dataset_name", type=str, default="", help="Key in stats JSON to use (uses first key if empty).")

    # Model config (must match training)
    parser.add_argument("--action_dim", type=int, default=29)
    parser.add_argument("--action_chunk", type=int, default=8)
    parser.add_argument("--use_robot_state", type=int, default=1)
    parser.add_argument("--use_tactile_deform", type=int, default=1, help="1=use deformation images, 0=use f6 force/torque vectors.")
    parser.add_argument("--use_tactile_vec", type=int, default=0, help="Kept for compatibility; effective only when use_tactile_deform=0.")
    parser.add_argument("--tactile_intermediate_size", type=int, default=0,
                        help="Intermediate size for tactile expert MLP. "
                             "0 = auto-detect from training_args.json or use default.")

    # Runtime
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"),
                        help="Resize RGB images to W H before tokenization. "
                             "E.g. --image_size 384 288. Default: no resize.")

    args = parser.parse_args()
    main(args)

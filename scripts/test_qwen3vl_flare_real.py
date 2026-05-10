"""
Real-world ZeroMQ inference server for the Qwen3-VL MoT VLA model
with flare visual prediction tokens (multi-token per frame).

Two inference paradigms:

(A) Paradigm A (legacy, --use_tactile_refine_flow 0)
    Single forward_flow per request — fully synchronous.

(C) Paradigm C (--use_tactile_refine_flow 1)
    Stateful slow/fast protocol:
      slow request → forward_flow_action_only → cache (latent + action) KV,
                     return Â chunk.
      fast request → tactile_residual_flow on cached KV with fresh tactile,
                     return A_refined = Â + Δa.
      slow_and_fast → run both in sequence (typical at chunk start).

    The client orchestrates cadence (e.g. slow every 16 robot steps, fast at
    offsets 0, 4, 8, 12).  ZMQ REP is single-threaded so a fast request
    arriving mid-slow naturally waits until slow finishes — the "if a
    refinement has not finished, wait until it finishes" guarantee.
"""

import os
import sys
import threading

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import argparse
import json
import io
import pickle
import time
import traceback

import numpy as np
import torch
from PIL import Image
import zmq
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
        use_tactile_code         = bool(getattr(args, "use_tactile_code", 0)),
        vqvae_codebook_size      = getattr(args, "vqvae_codebook_size", 64),
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

    ta_path = os.path.join(ckpt, "training_args.json")
    ta = {}
    if os.path.exists(ta_path):
        with open(ta_path) as f:
            ta = json.load(f)
        for key, default in [("tactile_intermediate_size", 0),
                             ("n_flare_tokens_per_frame", 0),
                             ("n_flare_steps", 0),
                             ("use_tactile_code", 0),
                             ("vqvae_codebook_size", 64)]:
            saved = ta.get(key, default)
            cli_val = getattr(args, key, default)
            if saved and cli_val == default:
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
            use_tactile_code=bool(getattr(args, "use_tactile_code", 0)),
            vqvae_codebook_size=getattr(args, "vqvae_codebook_size", 64),
        )
    elif os.path.exists(ckpt_config):
        model = _build_qwen3vl_from_config(ckpt_config, args)
    else:
        pretrained_path = None
        if ta:
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
            use_tactile_code=bool(getattr(args, "use_tactile_code", 0)),
            vqvae_codebook_size=getattr(args, "vqvae_codebook_size", 64),
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
        print(f"Flare prediction: {n_flare_steps} steps × {n_flare_tpf} tok/frame = {n_flare_total} total tokens")

    stats_path = args.stats_path or ""
    if not stats_path:
        candidate = os.path.join(ckpt, "stats_data.json")
        if os.path.exists(candidate):
            stats_path = candidate
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
# Tactile encoding helpers (shared across paradigms)
# ─────────────────────────────────────────────────────────────────────────────

def _encode_tactile_f6(tactile_f6_input, statistic, device):
    """Encode F6 for the single-frame tacf6_embedder.

    Accepts either:
      * `[n_fingers, 6]`           — single current frame (legacy clients).
      * `[T, n_fingers, 6]`        — dense rolling window (new clients send
                                     the full VQ-VAE window).  We take the
                                     last frame here for the per-frame
                                     embedder; the full window is consumed
                                     separately by `_push_f6_and_encode`.
    """
    if tactile_f6_input is None:
        return None
    arr = np.asarray(tactile_f6_input, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[-1]                    # most recent frame
    tacf6 = arr.reshape(-1)
    norm_tacf6 = _normalize(tacf6, statistic["tacf6_mask"],
                            statistic["tacf6_min"], statistic["tacf6_max"])
    return (torch.tensor(norm_tacf6.reshape(-1, 6), dtype=torch.bfloat16)
            .unsqueeze(0).to(device))


def _encode_tactile_deform(tactile_deform_input, device):
    if tactile_deform_input is None:
        return None
    arr = np.array(tactile_deform_input, dtype=np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    if arr.ndim == 3:
        return (torch.tensor(arr).unsqueeze(0).unsqueeze(2)
                .to(device, dtype=torch.bfloat16))
    elif arr.ndim == 4:
        return (torch.tensor(arr).unsqueeze(0)
                .to(device, dtype=torch.bfloat16))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Inference (Paradigm A — legacy)
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

        # ── M-RoPE ──────────────────────────────────────────────────────
        position_ids, _ = model.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw,
            attention_mask=attention_mask)
        position_ids = position_ids[:, :, :slow_embeds.shape[1]]

        # ── Append flare query tokens ───────────────────────────────────
        if model.n_flare_tokens > 0:
            flare_q = model.flare_queries.to(
                device=slow_embeds.device, dtype=slow_embeds.dtype)
            slow_embeds = torch.cat([slow_embeds, flare_q.expand(1, -1, -1)], dim=1)
            position_ids = extend_position_ids_for_flare(
                position_ids, model.n_flare_tokens)

        # ── Tactile ──────────────────────────────────────────────────────
        tac_f6_tensor     = _encode_tactile_f6(
            tactile_f6_input if args.use_tactile_vec else None,
            statistic, device)
        tac_deform_tensor = _encode_tactile_deform(
            tactile_deform_input if args.use_tactile_deform else None, device)

        # ── Flow-matching denoising ──────────────────────────────────────
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
        )

        norm_actions = samples[0].float().cpu().numpy()
        actions = _denormalize(
            norm_actions, statistic["action_mask"],
            statistic["action_min"], statistic["action_max"])

    return list(actions)


# ─────────────────────────────────────────────────────────────────────────────
# Paradigm C — stateful slow/fast inference
# ─────────────────────────────────────────────────────────────────────────────

class ParadigmCServer:
    """
    Stateful server for Paradigm C async inference.

    Holds the slow-path snapshot (cached latent + action KV, base chunk Â,
    encoded fast cameras + state, latent position ids) between requests so a
    subsequent fast request can run tactile_residual_flow without re-encoding
    the visual tower.

    A single `lock` serializes all model calls so:
      • only one inference runs on the GPU at a time, and
      • a fast request that arrives while a slow inference is still in flight
        blocks until the slow one finishes ("wait until refinement finishes"
        semantic).  ZMQ REP is already single-threaded; the lock is belt-and-
        braces for any future move to a multi-threaded transport.
    """

    def __init__(self, args, model, processor, statistic):
        self.args      = args
        self.model     = model
        self.processor = processor
        self.statistic = statistic
        self.device    = f"cuda:{args.cuda}"
        self.lock      = threading.Lock()

        # Slow-path snapshot
        self.cached_kv          = None
        self.A_hat              = None             # [B, n_chunk, action_dim] bf16
        self.A_hat_denorm       = None             # numpy after denorm
        self.position_ids       = None
        self.attention_mask     = None
        self.n_action_in_cache  = 0
        self.chunk_id           = -1               # incremented per slow

        # VQ-VAE tactile-code encoder (optional, server-side).  Maintains a
        # rolling F6 buffer so each fast tick can encode the historical
        # 16-frame window — same alignment as offline JSON encoding.
        self.vqvae_model    = None
        self.vqvae_stats    = None
        self.vqvae_window   = 16
        self.f6_buffer: list = []                  # list of [10, 6] np arrays
        if bool(getattr(args, "use_tactile_code", 0)):
            from tactile_vqvae.models.tactile_vqvae import (
                TactileVQVAE, TactileVQVAEConfig)
            from tactile_vqvae.data.stats import TacF6Stats
            blob = torch.load(args.vqvae_ckpt, map_location="cpu",
                              weights_only=False)
            cfg = TactileVQVAEConfig.from_dict(blob["config"])
            self.vqvae_model = TactileVQVAE(cfg)
            self.vqvae_model.load_state_dict(blob["model_state"])
            self.vqvae_model.eval().to(self.device)
            self.vqvae_stats  = TacF6Stats.from_dict(blob["stats"])
            self.vqvae_window = int(cfg.window)
            print(f">>> VQ-VAE loaded for tactile codes "
                  f"(K={cfg.codebook_size}, W={self.vqvae_window})")

    def _push_f6_and_encode(self, tactile_f6_input):
        """Encode an F6 history window into per-hand VQ-VAE codes.

        Accepted inputs:
          * `[10, 6]`         — single current frame.  We append it to a
                                server-side rolling buffer and encode the
                                last `window` frames.  Used by legacy clients
                                that don't track tactile history themselves.
          * `[T, 10, 6]`      — dense rolling window from a client that
                                already maintains the F6 history at fetch-
                                rate (e.g. eval_dexmot.py with the F6_HISTORY
                                deque).  We use the window directly — this
                                matches VQ-VAE training-time temporal density.

        Returns `[1, K]` int64 codes (K=2 hand or K=10 finger) on `self.device`,
        or None when VQ-VAE is disabled / input is missing.
        """
        if self.vqvae_model is None or tactile_f6_input is None:
            return None
        arr = np.asarray(tactile_f6_input, dtype=np.float32)
        w = self.vqvae_window

        if arr.ndim == 3:
            # Client-supplied dense window — bypass the rolling buffer.
            # Trim or left-edge-pad to exactly `w` frames so we always
            # encode the same window length the VQ-VAE was trained on.
            if arr.shape[0] >= w:
                arr = arr[-w:]
            else:
                head = np.repeat(arr[:1], w - arr.shape[0], axis=0)
                arr = np.concatenate([head, arr], axis=0)
        else:
            # Single frame — fall back to server-side rolling buffer.
            f6 = arr.reshape(10, 6)
            self.f6_buffer.append(f6)
            if len(self.f6_buffer) > w:
                self.f6_buffer = self.f6_buffer[-w:]
            if len(self.f6_buffer) < w:
                head = [self.f6_buffer[0]] * (w - len(self.f6_buffer))
                arr = np.stack(head + self.f6_buffer, axis=0)
            else:
                arr = np.stack(self.f6_buffer, axis=0)

        arr_n = self.vqvae_stats.normalize(arr).astype(np.float32, copy=False)

        is_per_finger = getattr(self.vqvae_model.cfg, "granularity", "hand") == "finger"
        n_fingers = int(getattr(self.vqvae_model.cfg, "n_fingers", 5)) if is_per_finger else 1

        if is_per_finger:
            codes = np.zeros((2, n_fingers), dtype=np.int64)     # [2, 5]
        else:
            codes = np.zeros(2, dtype=np.int64)                  # [2]

        for hand in (0, 1):
            wh = arr_n[:, hand * 5: (hand + 1) * 5, :]           # [W, 5, 6]
            t = torch.from_numpy(wh).unsqueeze(0).to(self.device)
            with torch.no_grad():
                idx = self.vqvae_model.encode(t).cpu().numpy()   # [1] or [1, 5]
            if is_per_finger:
                codes[hand] = idx.reshape(-1)
            else:
                codes[hand] = int(idx.item())

        # Flatten and add batch dim → [1, 2] (hand) or [1, 10] (finger).
        flat = codes.reshape(-1)
        return torch.tensor(flat, dtype=torch.long,
                            device=self.device).unsqueeze(0)

    # -- internal: build slow embeddings, run action-only flow, cache state --
    def _run_slow(
        self, task_description, slow_images, fast_images,
        tactile_f6_input=None, tactile_deform_input=None, state_fast=None,
    ):
        args, model, processor, statistic = (
            self.args, self.model, self.processor, self.statistic)
        device = self.device

        if args.image_size:
            _sz = tuple(args.image_size)
            slow_images = [img.resize(_sz, Image.LANCZOS) for img in slow_images]
            fast_images = [img.resize(_sz, Image.LANCZOS) for img in fast_images]

        state_embeds = None
        if args.use_robot_state and state_fast is not None:
            norm_state = _normalize(
                np.array(state_fast, dtype=np.float32),
                statistic["state_mask"], statistic["state_min"], statistic["state_max"])
            state_vec = torch.tensor(norm_state, dtype=torch.bfloat16).unsqueeze(0).to(device)
            state_embeds = model.state_embedder(state_vec).unsqueeze(1)

        n_slow = len(slow_images)
        all_pil = slow_images + fast_images
        content = [{"type": "image"} for _ in slow_images]
        content.append({"type": "text", "text": task_description})
        content += [{"type": "image"} for _ in fast_images]
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

        position_ids, _ = model.get_rope_index(
            input_ids=input_ids, image_grid_thw=image_grid_thw,
            attention_mask=attention_mask)
        position_ids = position_ids[:, :, :slow_embeds.shape[1]]

        if model.n_flare_tokens > 0:
            flare_q = model.flare_queries.to(
                device=slow_embeds.device, dtype=slow_embeds.dtype)
            slow_embeds = torch.cat([slow_embeds, flare_q.expand(1, -1, -1)], dim=1)
            position_ids = extend_position_ids_for_flare(
                position_ids, model.n_flare_tokens)

        noise = torch.randn(1, args.action_chunk, args.action_dim,
                            dtype=torch.bfloat16, device=device)
        a_hat, cached_kv, n_action_in_cache = model.forward_flow_action_only(
            inputs_embeds=slow_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            noise=noise,
            state_embeds=state_embeds,
            fast_embeds=fast_embeds,
            num_steps=args.action_flow_eval_steps,
            refresh_clean_kv=True,
        )

        self.cached_kv         = cached_kv
        self.A_hat             = a_hat
        self.position_ids      = position_ids
        self.attention_mask    = attention_mask
        self.n_action_in_cache = n_action_in_cache
        self.chunk_id         += 1

        norm_actions = a_hat[0].float().cpu().numpy()
        self.A_hat_denorm = _denormalize(
            norm_actions, statistic["action_mask"],
            statistic["action_min"], statistic["action_max"])
        return list(self.A_hat_denorm), self.chunk_id

    # -- internal: run tactile residual flow on cached state --
    def _run_fast(self, tactile_f6_input=None, tactile_deform_input=None):
        args, model, statistic = self.args, self.model, self.statistic
        device = self.device

        if self.cached_kv is None or self.A_hat is None:
            raise RuntimeError(
                "fast request received before any slow request — server has "
                "no cached state. Send mode='slow' or 'slow_and_fast' first.")

        tac_f6_tensor     = _encode_tactile_f6(
            tactile_f6_input if args.use_tactile_vec else None,
            statistic, device)
        tac_deform_tensor = _encode_tactile_deform(
            tactile_deform_input if args.use_tactile_deform else None, device)
        tac_codes_tensor  = self._push_f6_and_encode(tactile_f6_input)

        # Deterministic τ=1 starting point removes the per-fast-tick random
        # component that was the dominant source of inter-tick disagreement
        # (verified offline: splice jumps cut ~85%, jerk_l2 cut ~43%).
        init_noise = None
        if bool(getattr(args, "tactile_zero_init_noise", 0)):
            init_noise = torch.zeros(
                1, args.action_chunk, args.action_dim,
                dtype=torch.bfloat16, device=device)

        delta_a = model.tactile_residual_flow(
            cached_kv          = self.cached_kv,
            latent_position_ids= self.position_ids,
            n_action_in_cache  = self.n_action_in_cache,
            base_chunk         = self.A_hat,
            tactile_f6         = tac_f6_tensor,
            tactile_deform     = tac_deform_tensor,
            num_steps          = args.tactile_refine_flow_steps,
            noise_scale        = args.tactile_refine_noise_scale,
            tactile_codes      = tac_codes_tensor,
            initial_noise      = init_noise,
        )
        a_refined_norm = (self.A_hat + delta_a)[0].float().cpu().numpy()
        a_refined = _denormalize(
            a_refined_norm, statistic["action_mask"],
            statistic["action_min"], statistic["action_max"])
        return list(a_refined), self.chunk_id

    def predict(self, mode, payload):
        """Top-level dispatch.  Returns dict suitable for pickling back to
        the client.  Any exception inside a mode's body is propagated to the
        caller, which logs and replies with status='error'."""
        slow_img = (Image.open(io.BytesIO(payload["image_head"])).convert("RGB")
                    if "image_head" in payload else None)
        fast_list = []
        if "image_wrist_right" in payload:
            fast_list.append(Image.open(io.BytesIO(payload["image_wrist_right"])).convert("RGB"))
        if "image_wrist_left" in payload:
            fast_list.append(Image.open(io.BytesIO(payload["image_wrist_left"])).convert("RGB"))

        tac_f6     = payload.get("tactile_f6")
        tac_deform = payload.get("tactile_deform", payload.get("tactile_image_deform"))
        state_fast = payload.get("state_fast")
        task_desc  = payload.get("task_description", "")

        with self.lock, torch.inference_mode():
            self.model = self.model.to(self.device).eval()
            t0 = time.time()
            if mode == "slow":
                if slow_img is None:
                    raise ValueError("slow request requires image_head")
                actions, cid = self._run_slow(
                    task_desc, [slow_img], fast_list,
                    tac_f6, tac_deform, state_fast)
                latency_ms = (time.time() - t0) * 1000.0
                return {"status": "success", "mode": "slow",
                        "actions": actions, "chunk_id": cid,
                        "latency_ms": latency_ms}
            elif mode == "fast":
                actions, cid = self._run_fast(tac_f6, tac_deform)
                latency_ms = (time.time() - t0) * 1000.0
                return {"status": "success", "mode": "fast",
                        "actions": actions, "chunk_id": cid,
                        "latency_ms": latency_ms}
            elif mode == "slow_and_fast":
                if slow_img is None:
                    raise ValueError("slow_and_fast request requires image_head")
                self._run_slow(task_desc, [slow_img], fast_list,
                               tac_f6, tac_deform, state_fast)
                actions, cid = self._run_fast(tac_f6, tac_deform)
                latency_ms = (time.time() - t0) * 1000.0
                return {"status": "success", "mode": "slow_and_fast",
                        "actions": actions, "chunk_id": cid,
                        "latency_ms": latency_ms}
            else:
                raise ValueError(f"unknown mode: {mode}")


def main(args):
    print(f"Loading VLA model from checkpoint: {args.checkpoint_path}")
    model, processor, statistic, action_tokenizer = model_load(args)
    print("Model loaded successfully!")

    # Warm-up (use 2 fast images for bimanual / dual-arm tasks)
    print("Warming up model...")
    dummy_slow  = [Image.new("RGB", (224, 224), color="black")]
    n_fast_cams = 2 if args.action_dim > 31 else 1
    dummy_fast  = [Image.new("RGB", (224, 224), color="black") for _ in range(n_fast_cams)]
    dummy_state = np.zeros(args.action_dim, dtype=np.float32) if args.use_robot_state else None
    dummy_f6    = np.zeros((5, 6), dtype=np.float32) if args.use_tactile_vec else None
    dummy_deform = np.zeros((5, 240, 240), dtype=np.float32) if args.use_tactile_deform else None

    refine_mode = bool(args.use_tactile_refine_flow)
    if refine_mode:
        server = ParadigmCServer(args, model, processor, statistic)
        # Warm-up: run one slow_and_fast and discard
        dummy_payload = {
            "image_head":         _pil_to_bytes(dummy_slow[0]),
            "image_wrist_right":  _pil_to_bytes(dummy_fast[0]),
            "task_description":   "dummy task",
            "tactile_f6":         dummy_f6,
            "tactile_deform":     dummy_deform,
            "state_fast":         dummy_state,
        }
        if len(dummy_fast) > 1:
            dummy_payload["image_wrist_left"] = _pil_to_bytes(dummy_fast[1])
        result = server.predict("slow_and_fast", dummy_payload)
        print(f"Warm-up [tactile_refine_flow=on] output shape: "
              f"{np.array(result['actions']).shape}, "
              f"latency {result['latency_ms']:.1f} ms")
    else:
        dummy_out = model_predict(
            args, model, processor, statistic, action_tokenizer,
            "dummy task", dummy_slow, dummy_fast, dummy_f6, dummy_deform, dummy_state)
        print(f"Warm-up output shape: {np.array(dummy_out).shape}")

    # ZMQ Server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"VLA Server listening on port {args.port} "
          f"(tactile_refine_flow={'on' if refine_mode else 'off'}, "
          f"single-threaded REP)")

    step_counter = 0
    n_slow = n_fast = 0
    while True:
        try:
            payload = pickle.loads(socket.recv())

            if refine_mode:
                # Default to slow_and_fast for first request, slow_and_fast or
                # slow if the client didn't specify (matches legacy clients).
                mode = payload.get("mode", "slow_and_fast")
                result = server.predict(mode, payload)
                if mode == "fast":
                    n_fast += 1
                else:
                    n_slow += 1
            else:
                slow_img = Image.open(io.BytesIO(payload["image_head"])).convert("RGB")
                fast_list = [Image.open(io.BytesIO(payload["image_wrist_right"])).convert("RGB")]
                if "image_wrist_left" in payload:
                    fast_list.append(Image.open(io.BytesIO(payload["image_wrist_left"])).convert("RGB"))

                tac_f6 = payload.get("tactile_f6") if args.use_tactile_vec else None
                tac_deform = None
                if args.use_tactile_deform:
                    tac_deform = payload.get("tactile_deform", payload.get("tactile_image_deform"))

                actions = model_predict(
                    args, model, processor, statistic, action_tokenizer,
                    payload["task_description"], [slow_img], fast_list,
                    tac_f6, tac_deform, payload.get("state_fast"))
                result = {"status": "success", "actions": actions}

            socket.send(pickle.dumps(result))
            step_counter += 1
            if step_counter % 10 == 0:
                if refine_mode:
                    print(f"Processed {step_counter} requests "
                          f"(slow={n_slow}, fast={n_fast}, "
                          f"chunk_id={server.chunk_id}). "
                          f"Task: {payload.get('task_description', '')}")
                else:
                    print(f"Processed {step_counter} requests. "
                          f"Task: {payload.get('task_description', '')}")

        except Exception as e:
            traceback.print_exc()
            socket.send(pickle.dumps({"status": "error", "message": str(e)}))


def _pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-world ZMQ server (with flare prediction)")
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
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))

    # Paradigm C
    parser.add_argument("--use_tactile_refine_flow", type=int, default=0,
                        help="1: stateful slow/fast inference. The client "
                             "sends payloads with mode='slow' (full payload) "
                             "every chunk_size robot steps and mode='fast' "
                             "(tactile only, reuses cached KV) every "
                             "chunk_size/refine_async_ratio steps. The first "
                             "request must be 'slow' or 'slow_and_fast'.")
    parser.add_argument("--action_flow_eval_steps", type=int, default=10,
                        help="Number of Euler steps for the action-only slow flow.")
    parser.add_argument("--tactile_refine_flow_steps", type=int, default=4,
                        help="Number of Euler steps for the tactile residual flow at fast tick.")
    parser.add_argument("--tactile_refine_noise_scale", type=float, default=0.1,
                        help="Initial noise magnitude for the residual flow at τ=1.")
    parser.add_argument("--tactile_zero_init_noise", type=int, default=0,
                        help="1: pass zeros as the τ=1 starting point of the "
                             "tactile residual flow.  Makes Δa deterministic "
                             "given (cached_kv, tactile) — kills per-fast-tick "
                             "noise that caused inter-tick disagreement.  "
                             "Strongly recommended; offline confirms ~85%% "
                             "reduction in splice jumps.")

    # VQ-VAE tactile code tokens (fast-path only).  When 0 (default) the model
    # graph and behavior are identical to the pre-feature version — flip the
    # flag to revert.  When 1, --vqvae_ckpt must be a TactileVQVAE latest.pt.
    # Server maintains a rolling 16-frame F6 buffer and encodes per-hand on
    # each fast tick.
    parser.add_argument("--use_tactile_code", type=int, default=0,
                        help="1: server-side VQ-VAE encodes a rolling F6 "
                             "window into 2 codes per fast tick.")
    parser.add_argument("--vqvae_codebook_size", type=int, default=64,
                        help="Codebook size of the VQ-VAE that produces the codes.")
    parser.add_argument("--vqvae_ckpt", type=str, default="",
                        help="Path to TactileVQVAE checkpoint (latest.pt). "
                             "Required when --use_tactile_code 1.")

    args = parser.parse_args()
    if bool(args.use_tactile_code) and not args.vqvae_ckpt:
        parser.error("--vqvae_ckpt must be set when --use_tactile_code 1")
    main(args)


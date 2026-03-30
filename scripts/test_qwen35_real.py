"""
Real-world ZeroMQ inference server for the Qwen3.5-based MoT VLA model.

Adapted from test_sharpa_tactile_qwen.py (offline test + server) and
test_sharpa_tactile_ref.py (real-world deployment pattern).
Removes offline evaluation and directly starts the ZMQ server to accept
real-world robot client requests.
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

from qwen_vla import Qwen35VLAModel


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

    proc_dir = os.path.join(ckpt, "processor")
    if not os.path.isdir(proc_dir):
        raise FileNotFoundError(f"processor/ not found in checkpoint: {ckpt}")
    processor = AutoProcessor.from_pretrained(proc_dir, trust_remote_code=True)
    print(f"Processor loaded from: {proc_dir}")

    pretrained_path = None
    if getattr(args, "base_model_path", "") and os.path.isdir(args.base_model_path):
        pretrained_path = args.base_model_path
        print(f"Using base model path: {pretrained_path}")
    elif os.path.exists(os.path.join(ckpt, "config.json")):
        pretrained_path = ckpt
        print(f"Using config.json from checkpoint: {ckpt}")
    else:
        ta_path = os.path.join(ckpt, "training_args.json")
        if os.path.exists(ta_path):
            with open(ta_path) as f:
                ta = json.load(f)
            mp = ta.get("model_path", "")
            if mp and os.path.isdir(mp):
                pretrained_path = mp
                print(f"Using base model from training_args.json: {pretrained_path}")

    if pretrained_path is None:
        raise FileNotFoundError(
            f"Cannot reconstruct model architecture. No config.json in {ckpt}, "
            f"no training_args.json, and no --base_model_path supplied."
        )

    model = Qwen35VLAModel.from_pretrained_qwen35(
        pretrained_path    = pretrained_path,
        action_dim         = args.action_dim,
        action_chunk       = args.action_chunk,
        use_tactile_deform = bool(args.use_tactile_deform),
        use_robot_state    = bool(args.use_robot_state),
        torch_dtype        = torch.bfloat16,
    )

    # ── 3. Load finetuned weights ─────────────────────────────────────────────
    ckpt_file = os.path.join(ckpt, "model.pt")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f"model.pt not found in checkpoint: {ckpt}")
    sd = torch.load(ckpt_file, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Checkpoint loaded: {ckpt_file}")
    print(f"  missing={len(missing)}  unexpected={len(unexpected)}")

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
    tactile_input,        # np.ndarray (T,6) for f6  OR  list/array (N,[H,W]) for deform
    state_fast=None,      # np.ndarray (state_dim,) or None
):
    """
    Run one denoising flow and return a list of action arrays (shape: action_chunk x action_dim).
    """
    device = f"cuda:{args.cuda}"
    model = model.to(device).eval()

    with torch.inference_mode():

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

        # ── Build processor inputs (state NOT in text) ────────────────────────
        content = []
        for _ in slow_images:
            content.append({"type": "image"})
        content.append({"type": "text", "text": task_description})
        for _ in fast_images:
            content.append({"type": "image"})

        messages = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        all_pil = slow_images + fast_images
        inp = processor(
            text=text,
            images=all_pil if all_pil else None,
            return_tensors="pt",
            padding=False,
        )

        input_ids      = inp.input_ids.to(device)
        attention_mask = inp.attention_mask.to(device)
        pixel_values   = (inp.pixel_values.to(device, dtype=torch.bfloat16)
                          if getattr(inp, "pixel_values", None) is not None else None)
        image_grid_thw = (inp.image_grid_thw.to(device)
                          if getattr(inp, "image_grid_thw", None) is not None else None)

        # ── VLM embeddings ────────────────────────────────────────────────────
        inputs_embeds = model.prepare_inputs_embeds(
            input_ids      = input_ids,
            pixel_values   = pixel_values,
            image_grid_thw = image_grid_thw,
        )                                               # [1, L, H]

        position_ids, _ = model.get_rope_index(
            input_ids      = input_ids,
            image_grid_thw = image_grid_thw,
            attention_mask = attention_mask,
        )                                               # [3, 1, L]

        # ── Tactile tensor ────────────────────────────────────────────────────
        if args.use_tactile_deform:
            # tactile_input: np.ndarray (N, H, W) or (N, 1, H, W) from real-world client
            arr = np.array(tactile_input, dtype=np.float32)
            if arr.max() > 1.0:
                arr = arr / 255.0
            if arr.ndim == 3:
                # (N, H, W) -> [1, N, 1, H, W]
                tactile_tensor = (torch.tensor(arr)
                                  .unsqueeze(0)             # [1, N, H, W]
                                  .unsqueeze(2)             # [1, N, 1, H, W]
                                  .to(device, dtype=torch.bfloat16))
            elif arr.ndim == 4:
                # (N, 1, H, W) -> [1, N, 1, H, W]
                tactile_tensor = (torch.tensor(arr)
                                  .unsqueeze(0)             # [1, N, 1, H, W]
                                  .to(device, dtype=torch.bfloat16))
            else:
                raise ValueError(f"Unexpected tactile_deform shape: {arr.shape}")
        else:
            # tactile_f6: (T, 6) -> normalise -> [1, T, 6]
            tacf6 = np.array(tactile_input, dtype=np.float32).reshape(-1, 6)
            norm_tacf6 = _normalize(tacf6, statistic["tacf6_mask"],
                                    statistic["tacf6_min"], statistic["tacf6_max"])
            tactile_tensor = (torch.tensor(norm_tacf6, dtype=torch.bfloat16)
                              .unsqueeze(0)             # [1, T, 6]
                              .to(device))

        # ── Flow-matching denoising ───────────────────────────────────────────
        noise = torch.randn(1, args.action_chunk, args.action_dim,
                            dtype=torch.bfloat16, device=device)

        samples = model.forward_flow(
            inputs_embeds  = inputs_embeds,
            position_ids   = position_ids,
            attention_mask = attention_mask,
            noise          = noise,
            tactile_inputs = tactile_tensor,
            num_steps      = 10,
            state_embeds   = state_embeds,
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
    print(f"Loading VLA model (Qwen3.5) from checkpoint: {args.checkpoint_path}")
    model, processor, statistic, action_tokenizer = model_load(args)
    print("Model loaded successfully!")

    # ── Warm-up forward pass ──────────────────────────────────────────────────
    print("Warming up model...")
    dummy_slow  = [Image.new("RGB", (224, 224), color="black")]
    dummy_fast  = [Image.new("RGB", (224, 224), color="black")]
    dummy_state = np.zeros(args.action_dim, dtype=np.float32) if args.use_robot_state else None
    if args.use_tactile_deform:
        dummy_tac = np.zeros((5, 240, 240), dtype=np.float32)
    else:
        dummy_tac = np.zeros((10, 6), dtype=np.float32)

    dummy_out = model_predict(
        args, model, processor, statistic, action_tokenizer,
        "dummy task", dummy_slow, dummy_fast, dummy_tac, dummy_state,
    )
    print(f"Warm-up output shape: {np.array(dummy_out).shape}")
    print("Warm-up complete.")

    # ── ZeroMQ Server ─────────────────────────────────────────────────────────
    context = zmq.Context()
    socket  = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"VLA Server (Qwen3.5) is listening on port {args.port}...")

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

            if args.use_tactile_deform:
                # Real-world client sends tactile_deform as numpy array (N, H, W)
                tactile_input = payload["tactile_deform"]
                print(f"tactile_deform shape: {np.array(tactile_input).shape}")
            else:
                tactile_input = payload["tactile_f6"]

            actions = model_predict(
                args=args,
                model=model,
                processor=processor,
                statistic=statistic,
                action_tokenizer=action_tokenizer,
                task_description=task_description,
                slow_images=slow_image_list,
                fast_images=fast_image_list,
                tactile_input=tactile_input,
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
        description="Real-world ZMQ server for Qwen3.5-based MoT VLA"
    )

    # Model path
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Finetuned checkpoint dir (model.pt, processor/, config.json, stats_data.json)")
    parser.add_argument("--base_model_path", type=str, default="", help="Path to base Qwen model (auto-detected if empty).")
    parser.add_argument("--stats_path", type=str, default="", help="Explicit path to stats JSON (auto-detected if empty).")
    parser.add_argument("--dataset_name", type=str, default="", help="Key in stats JSON to use (uses first key if empty).")

    # Model config (must match training)
    parser.add_argument("--action_dim", type=int, default=29)
    parser.add_argument("--action_chunk", type=int, default=8)
    parser.add_argument("--use_robot_state", type=int, default=1)
    parser.add_argument("--use_tactile_deform", type=int, default=1, help="1=use deformation images, 0=use f6 force/torque vectors.")
    parser.add_argument("--use_tactile_vec", type=int, default=0, help="Kept for compatibility; effective only when use_tactile_deform=0.")

    # Runtime
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--port", type=int, default=5555)

    args = parser.parse_args()
    main(args)


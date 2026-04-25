"""
Offline evaluation for Qwen3-VL MoT VLA trained with train_qwen3vl_tflare_gate.py.

Produces, beyond the standard action-MSE plots:
  - gate statistics: mean gate per sample, per-action-dim gate distribution,
    gate vs contact-intensity scatter.
  - v_tac vs v_act divergence: ‖v_tac − v_act‖ per sample, correlated with
    F6 force magnitude (contact intensity).
  - tactile-FLARE cosine similarity per future step (diagnostic of whether
    the tactile expert learned to predict future tactile state).

Reuses the denoise_gated() loop from test_qwen3vl_tflare_gate_real.py,
adds a separate single-pass path for the tactile-FLARE cosine evaluation.
"""

import os, sys, copy
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import argparse, json, io, pickle, traceback, re
import numpy as np
import torch
import torch.nn as nn
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

# Reuse helpers from the real-time script
from test_qwen3vl_tflare_gate_real import (
    _normalize, _denormalize, _build_qwen3vl_from_config, _has_hf_weights,
    _attach_gate_and_tflare, denoise_gated, model_load as _model_load_real,
)


def model_load(args):
    # Same loader as the real-time script
    return _model_load_real(args)


def _open_rgb(path, image_size=None):
    img = Image.open(path).convert("RGB")
    if image_size is not None:
        img = img.resize(image_size, Image.LANCZOS)
    return img


def _open_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


# ─── Prepare shared slow/fast/tactile tensors once per sample ──────────────

def _prepare_inputs(args, model, processor, statistic, sample, data_dir):
    _abs = lambda p: p if os.path.isabs(p) else os.path.join(data_dir, p)
    img_size = tuple(args.image_size) if args.image_size else None
    device = f"cuda:{args.cuda}"
    dtype = torch.bfloat16

    slow_images = [_open_rgb(_abs(p), img_size) for p in sample["input_image_slow"]]
    fast_images = [_open_rgb(_abs(p), img_size) for p in sample["input_image_fast"]]

    # State
    state_embeds = None
    if args.use_robot_state and "state_fast" in sample:
        norm_state = _normalize(
            np.array(sample["state_fast"], dtype=np.float32),
            statistic["state_mask"], statistic["state_min"], statistic["state_max"])
        sv = torch.tensor(norm_state, dtype=dtype).unsqueeze(0).to(device)
        state_embeds = model.state_embedder(sv).unsqueeze(1)

    # Chat template
    n_slow = len(slow_images)
    all_pil = slow_images + fast_images
    content = []
    for _ in slow_images: content.append({"type": "image"})
    content.append({"type": "text", "text": sample.get("input_prompt", "")})
    for _ in fast_images: content.append({"type": "image"})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inp = processor(text=text, images=all_pil if all_pil else None, return_tensors="pt", padding=False)

    input_ids = inp.input_ids.to(device)
    attention_mask = inp.attention_mask.to(device)
    pixel_values = inp.pixel_values.to(device, dtype=dtype) if getattr(inp, "pixel_values", None) is not None else None
    image_grid_thw = inp.image_grid_thw.to(device) if getattr(inp, "image_grid_thw", None) is not None else None

    inputs_embeds = model.prepare_inputs_embeds(
        input_ids=input_ids, pixel_values=pixel_values, image_grid_thw=image_grid_thw)

    fast_embeds = None
    if image_grid_thw is not None and fast_images:
        merge = getattr(model.visual, "spatial_merge_size",
                        getattr(processor.image_processor, "merge_size", 2))
        n_slow_img_tokens = sum(int(g[0] * (g[1] // merge) * (g[2] // merge))
                                for g in image_grid_thw[:n_slow])
        slow_embeds, fast_embeds = split_slow_fast_embeds(
            inputs_embeds, input_ids, model.image_token_id, n_slow_img_tokens)
    else:
        slow_embeds = inputs_embeds

    position_ids, _ = model.get_rope_index(
        input_ids=input_ids, image_grid_thw=image_grid_thw, attention_mask=attention_mask)
    position_ids = position_ids[:, :, :slow_embeds.shape[1]]

    if model.n_flare_tokens > 0:
        flare_q = model.flare_queries.to(device=device, dtype=dtype)
        slow_embeds = torch.cat([slow_embeds, flare_q.expand(1, -1, -1)], dim=1)
        position_ids = extend_position_ids_for_flare(position_ids, model.n_flare_tokens)

    # Tactile tensors
    tac_f6_tensor = None
    if args.use_tactile_vec and "tactile_f6" in sample:
        tacf6 = np.array(sample["tactile_f6"], dtype=np.float32).reshape(-1)
        norm_tacf6 = _normalize(tacf6, statistic["tacf6_mask"],
                                statistic["tacf6_min"], statistic["tacf6_max"])
        tac_f6_tensor = torch.tensor(norm_tacf6.reshape(-1, 6), dtype=dtype).unsqueeze(0).to(device)

    tac_deform_tensor = None
    if args.use_tactile_deform and "tactile_image_deform" in sample:
        arr = np.stack([_open_gray(_abs(p)) for p in sample["tactile_image_deform"]])
        tac_deform_tensor = torch.tensor(arr).unsqueeze(0).unsqueeze(2).to(device, dtype=dtype)

    return {
        "slow_embeds": slow_embeds, "position_ids": position_ids,
        "attention_mask": attention_mask,
        "state_embeds": state_embeds, "fast_embeds": fast_embeds,
        "tac_f6": tac_f6_tensor, "tac_deform": tac_deform_tensor,
    }


# ─── Single-pass forward with v_act, v_tac, gate capture ───────────────────

@torch.inference_mode()
def _single_pass_capture(model, prepared, args, noise_level=1.0):
    """One forward at a fixed noise level (t=1.0 = pure noise), returning
    v_act, v_tac, gate, and the internal hidden states + h_tflare (for
    tactile-FLARE cosine eval)."""
    device = prepared["slow_embeds"].device
    dtype = prepared["slow_embeds"].dtype
    B, H = 1, prepared["slow_embeds"].shape[2]

    tac_parts = []
    if prepared["tac_f6"] is not None:
        tac_parts.append(model.tacf6_embedder(prepared["tac_f6"].to(dtype)))
    if prepared["tac_deform"] is not None:
        Bs, nf, C, Hh, Ww = prepared["tac_deform"].shape
        feats = model.deform_encoder(prepared["tac_deform"].view(-1, C, Hh, Ww))
        feats = feats.view(Bs, nf, -1)
        tac_parts.append(model.deform_proj(feats.to(dtype)))
    tactile_embeds = torch.cat(tac_parts, dim=1) if tac_parts else torch.empty((B, 0, H), device=device, dtype=dtype)
    has_tactile = tactile_embeds.shape[1] > 0
    n_tac_input = tactile_embeds.shape[1]

    K_tac = (args.n_tfl_tokens_per_step * args.n_tfl_steps) if args.use_tactile_flare else 0
    tflare_q = None
    if has_tactile and K_tac > 0 and hasattr(model, "tactile_flare_queries"):
        tflare_q = model.tactile_flare_queries.expand(B, -1, -1).to(device=device, dtype=dtype)

    fast_embeds  = (prepared["fast_embeds"] if prepared["fast_embeds"] is not None
                    else torch.empty((B, 0, H), device=device, dtype=dtype))
    state_embeds = (prepared["state_embeds"] if prepared["state_embeds"] is not None
                    else torch.empty((B, 0, H), device=device, dtype=dtype))

    x_t = torch.randn(1, args.action_chunk, args.action_dim, dtype=dtype, device=device) * noise_level
    t_val = torch.tensor([noise_level], dtype=dtype, device=device)
    noisy_actions = model.x_embedder(x_t)
    timesteps = model.t_embedder(t_val).unsqueeze(1)

    L_latent = prepared["slow_embeds"].shape[1]
    n_fast, n_state = fast_embeds.shape[1], state_embeds.shape[1]
    chunk = args.action_chunk
    n_action = n_fast + n_state + 1 + chunk

    parts = [prepared["slow_embeds"], fast_embeds, state_embeds, timesteps, noisy_actions]
    if has_tactile:
        parts.append(tactile_embeds)
        if tflare_q is not None:
            parts.append(tflare_q)
        noisy_actions_tac = model.x_embedder(x_t)
        timesteps_tac = model.t_embedder(t_val).unsqueeze(1)
        parts.extend([timesteps_tac, noisy_actions_tac])
    full_embeds = torch.cat(parts, dim=1)
    L_total = full_embeds.shape[1]

    latent_indexes  = torch.arange(0, L_latent, device=device)
    action_indexes  = torch.arange(L_latent, L_latent + n_action, device=device)
    tactile_indexes = (torch.arange(L_latent + n_action, L_total, device=device)
                       if has_tactile else torch.arange(0, 0, device=device))

    outputs = model.model(
        inputs_embeds=full_embeds, position_ids=prepared["position_ids"],
        attention_mask=prepared["attention_mask"], use_cache=False,
        latent_indexes=latent_indexes, action_indexes=action_indexes,
        tactile_indexes=tactile_indexes,
    )
    hidden = outputs.last_hidden_state

    act_start = L_latent + n_fast + n_state + 1
    h_act = hidden[:, act_start:act_start + chunk, :]
    v_act = model.final_layer(h_act)

    result = {"v_act": v_act, "h_act": h_act, "has_tactile": has_tactile}
    if has_tactile:
        h_tac = hidden[:, -chunk:, :]
        v_tac = model.final_layer_tactile(h_tac)
        gate_logits = model.gate_head(torch.cat([h_act, h_tac], dim=-1))
        g = torch.sigmoid(gate_logits)
        result.update({"v_tac": v_tac, "h_tac": h_tac, "gate": g})

        if K_tac > 0 and hasattr(model, "tactile_flare_proj"):
            tflare_start = L_latent + n_action + n_tac_input
            h_tflare = hidden[:, tflare_start:tflare_start + K_tac, :]
            result["tflare_pred"] = model.tactile_flare_proj(h_tflare)

    return result


# ─── Tactile-FLARE cosine evaluation ────────────────────────────────────────

def _load_future_tactile(sample, sample_idx, all_samples, n_steps, stride):
    """Fetch future tactile from sample_idx + k*stride, episode-guarded."""
    cur_prefix = os.path.dirname(sample["input_image_slow"][0])
    f6_list, deform_paths_list = [], []
    for k in range(n_steps):
        fut_idx = sample_idx + (k + 1) * stride
        fut_sample = None
        if 0 <= fut_idx < len(all_samples):
            cand = all_samples[fut_idx]
            if os.path.dirname(cand["input_image_slow"][0]) == cur_prefix:
                fut_sample = cand
        src = fut_sample if fut_sample is not None else sample
        f6_list.append(src.get("tactile_f6"))
        deform_paths_list.append(list(src.get("tactile_image_deform", []) or []))
    return f6_list, deform_paths_list


@torch.inference_mode()
def _tflare_target(model, statistic, args, f6_list, deform_paths_list, data_dir):
    """Compute tactile-FLARE targets using the frozen target encoders."""
    device = next(model.parameters()).device
    dtype = torch.bfloat16
    S = args.n_tfl_steps
    T_per = args.n_tfl_tokens_per_step
    tgt_parts = []

    if args.use_tactile_vec and f6_list[0] is not None:
        f6_raw = np.array(f6_list, dtype=np.float32).reshape(1, S, -1, 6)
        flat = f6_raw.reshape(S, -1)
        normed = _normalize(flat, statistic["tacf6_mask"],
                            statistic["tacf6_min"], statistic["tacf6_max"])
        f6_t = torch.tensor(normed.reshape(1, S, -1, 6), dtype=dtype).to(device)
        tgt_parts.append(model.target_tacf6_embedder(f6_t))

    if args.use_tactile_deform and deform_paths_list[0]:
        _abs = lambda p: p if os.path.isabs(p) else os.path.join(data_dir, p)
        arr = np.stack([
            np.stack([_open_gray(_abs(p)) for p in paths])
            for paths in deform_paths_list
        ])  # [S, nf, H, W]
        dt_arr = torch.tensor(arr[np.newaxis, :, :, np.newaxis, :, :]).to(device, dtype=dtype)
        Bs, S_, nf, C, Hh, Ww = dt_arr.shape
        feats = model.deform_encoder(dt_arr.view(-1, C, Hh, Ww)).view(Bs, S_, nf, -1)
        tgt_parts.append(model.target_deform_proj(feats.to(dtype)))

    if not tgt_parts:
        return None
    tgt_all = torch.cat(tgt_parts, dim=2)  # [1, S, nf_total, H]
    B_, S_, nf_t, H_ = tgt_all.shape
    flat = tgt_all.view(B_ * S_, nf_t, H_).permute(0, 2, 1)
    pooled = F.adaptive_avg_pool1d(flat.float(), T_per).permute(0, 2, 1).to(dtype)
    return pooled.view(B_, S_ * T_per, H_)


def main(args):
    print(f"Loading VLA model from checkpoint: {args.checkpoint_path}")
    model, processor, statistic, action_tokenizer = model_load(args)
    print("Model loaded successfully!")

    # Warm-up
    from PIL import Image as _PI
    dummy_slow  = [_PI.new("RGB", (224, 224), color="black")]
    n_fast_cams = 2 if args.action_dim > 31 else 1
    dummy_fast  = [_PI.new("RGB", (224, 224), color="black") for _ in range(n_fast_cams)]

    if args.test_json_path:
        with open(args.test_json_path) as f:
            all_samples = json.load(f)
        test_samples = all_samples[:args.num_test_samples] if args.num_test_samples > 0 else all_samples
        data_dir = os.path.dirname(os.path.abspath(args.test_json_path))

        os.makedirs(args.save_dir, exist_ok=True)
        device = f"cuda:{args.cuda}"
        model = model.to(device).eval()
        dtype = torch.bfloat16

        # Containers
        mse_per_sample = []
        all_pred, all_gt = [], []
        gate_per_sample = []             # [N, chunk, D]
        vtac_vact_diff = []              # [N] Frobenius divergence
        contact_force_per_sample = []    # [N] max ||F_xyz||
        tflare_sim_per_sample = []       # list of [S] per sample
        has_tflare_eval = bool(args.use_tactile_flare and hasattr(model, "tactile_flare_proj"))

        K_tac = args.n_tfl_tokens_per_step * args.n_tfl_steps if args.use_tactile_flare else 0

        for step, sample in enumerate(tqdm(test_samples, desc="Offline eval")):
            try:
                prepared = _prepare_inputs(args, model, processor, statistic, sample, data_dir)
            except FileNotFoundError as e:
                print(f"\n[Warning] Step {step}: {e}")
                continue

            # ── (1) Full denoising to get predicted actions ───────────────
            noise = torch.randn(1, args.action_chunk, args.action_dim, dtype=dtype, device=device)
            samples = denoise_gated(
                model=model,
                inputs_embeds=prepared["slow_embeds"],
                position_ids=prepared["position_ids"],
                attention_mask=prepared["attention_mask"],
                noise=noise, num_steps=10,
                state_embeds=prepared["state_embeds"],
                tactile_f6=prepared["tac_f6"],
                tactile_deform=prepared["tac_deform"],
                fast_embeds=prepared["fast_embeds"],
                K_tac=K_tac,
            )
            norm_actions = samples[0].float().cpu().numpy()
            pred = _denormalize(norm_actions, statistic["action_mask"],
                                statistic["action_min"], statistic["action_max"])
            gt_action = np.array(sample["action"], dtype=np.float32)
            n_cmp = min(len(pred), len(gt_action))
            mse = float(np.mean((pred[:n_cmp] - gt_action[:n_cmp]) ** 2))
            mse_per_sample.append(mse)
            all_pred.append(pred[0] if pred.ndim > 1 else pred)
            all_gt.append(gt_action[0] if gt_action.ndim > 1 else gt_action)

            # ── (2) Single-pass capture for gate / divergence / tflare ────
            cap = _single_pass_capture(model, prepared, args, noise_level=1.0)
            if cap["has_tactile"]:
                gate_per_sample.append(cap["gate"].float().cpu().numpy()[0])
                diff = (cap["v_tac"] - cap["v_act"]).float()
                vtac_vact_diff.append(diff.norm().item() / diff.numel() ** 0.5)  # RMS

                # Contact intensity from raw F6 (not normalized)
                f6_raw = np.array(sample["tactile_f6"], dtype=np.float32).reshape(-1, 6)
                contact_force_per_sample.append(float(np.linalg.norm(f6_raw[:, :3], axis=-1).max()))

                # Tactile-FLARE cosine similarity
                if has_tflare_eval and "tflare_pred" in cap:
                    f6_list, deform_paths_list = _load_future_tactile(
                        sample, step, all_samples,
                        args.n_tfl_steps, args.tactile_flare_stride)
                    tflare_tgt = _tflare_target(model, statistic, args,
                                                f6_list, deform_paths_list, data_dir)
                    if tflare_tgt is not None:
                        pred_n = F.normalize(cap["tflare_pred"].float(), dim=-1)
                        tgt_n  = F.normalize(tflare_tgt.float(), dim=-1)
                        cos = (pred_n * tgt_n).sum(dim=-1).squeeze(0)           # [K_tac]
                        per_step = cos.view(args.n_tfl_steps,
                                            args.n_tfl_tokens_per_step).mean(dim=1)
                        tflare_sim_per_sample.append(per_step.cpu().numpy())

        # ── Summary ───────────────────────────────────────────────────────
        n_valid = len(mse_per_sample)
        mean_mse = float(np.mean(mse_per_sample)) if n_valid else float("nan")
        print(f"\n=== Action MSE: {n_valid}/{len(test_samples)} valid, mean={mean_mse:.6f} ===")

        if gate_per_sample:
            gate_arr = np.stack(gate_per_sample)     # [N, chunk, D]
            diff_arr = np.array(vtac_vact_diff)      # [N]
            contact_arr = np.array(contact_force_per_sample)  # [N]
            print(f"=== Gate: mean={gate_arr.mean():.4f} "
                  f"(chunk-avg range {gate_arr.mean(axis=(1,2)).min():.3f} → {gate_arr.mean(axis=(1,2)).max():.3f}) ===")
            print(f"=== ‖v_tac − v_act‖_RMS: mean={diff_arr.mean():.4f}, std={diff_arr.std():.4f} ===")

            # Gate vs contact intensity
            plt.figure(figsize=(7, 4))
            plt.scatter(contact_arr, gate_arr.mean(axis=(1, 2)),
                        alpha=0.5, s=8, color="steelblue")
            plt.xlabel("max finger |F_xyz| (raw F6 units)")
            plt.ylabel("mean gate over (chunk × action_dim)")
            plt.title("Gate opens where force is high?")
            plt.grid(True, linestyle=":", alpha=0.6); plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, "gate_vs_contact.png")); plt.close()

            # ‖v_tac - v_act‖ vs contact
            plt.figure(figsize=(7, 4))
            plt.scatter(contact_arr, diff_arr, alpha=0.5, s=8, color="crimson")
            plt.xlabel("max finger |F_xyz|")
            plt.ylabel("‖v_tac − v_act‖_RMS")
            plt.title("Experts diverge on contact-heavy frames?")
            plt.grid(True, linestyle=":", alpha=0.6); plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, "vtac_vact_diff_vs_contact.png")); plt.close()

            # Per-action-dim gate mean (averaged over samples and chunk)
            perdim = gate_arr.mean(axis=(0, 1))      # [D]
            plt.figure(figsize=(10, 4))
            plt.bar(range(len(perdim)), perdim, color="steelblue", alpha=0.8)
            plt.xlabel("action dim"); plt.ylabel("mean gate")
            plt.title("Per-action-dim gate (higher → tactile expert contributes more)")
            plt.grid(True, axis="y", linestyle=":", alpha=0.5); plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, "gate_per_action_dim.png")); plt.close()

            np.savez(os.path.join(args.save_dir, "gate_diagnostics.npz"),
                     gate=gate_arr, vdiff=diff_arr, contact=contact_arr)

        if tflare_sim_per_sample:
            sims = np.stack(tflare_sim_per_sample)   # [N, S]
            step_mean = sims.mean(axis=0)
            print(f"=== Tactile-FLARE: mean cos_sim={sims.mean():.4f} "
                  f"(per-step: {[f'{v:.3f}' for v in step_mean]}) ===")
            plt.figure(figsize=(8, 4))
            step_labels = [f"t+{(i+1)*args.tactile_flare_stride}" for i in range(len(step_mean))]
            plt.bar(step_labels, step_mean, color="darkgreen", alpha=0.8)
            plt.ylim(0, 1); plt.ylabel("mean cos_sim")
            plt.title("Tactile-FLARE: mean cosine similarity per future step")
            plt.grid(True, axis="y", linestyle=":", alpha=0.5); plt.tight_layout()
            plt.savefig(os.path.join(args.save_dir, "tflare_sim_per_step.png")); plt.close()
            np.savez(os.path.join(args.save_dir, "tflare_similarity.npz"),
                     per_sample_per_step=sims, per_step_mean=step_mean)

        if all_pred:
            all_pred_arr = np.stack(all_pred); all_gt_arr = np.stack(all_gt)
            for idx in range(all_pred_arr.shape[1]):
                plt.figure(figsize=(10, 4))
                plt.plot(all_pred_arr[:, idx], label="Predicted", linestyle="--", color="blue")
                plt.plot(all_gt_arr[:, idx], label="GT", alpha=0.6, color="orange")
                plt.title(f"Action Dimension {idx}")
                plt.legend(); plt.grid(True, linestyle=":", alpha=0.6); plt.tight_layout()
                plt.savefig(os.path.join(args.save_dir, f"action_dim_{idx}.png")); plt.close()
            np.savez(os.path.join(args.save_dir, "action_trajectory.npz"),
                     pred=all_pred_arr, gt=all_gt_arr)

        print(f"Saved diagnostics to {args.save_dir}")

    input("\nPress Enter to start ZMQ server (or Ctrl-C to exit)...")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"Server listening on port {args.port}...")

    from test_qwen3vl_tflare_gate_real import model_predict
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
            actions = model_predict(args, model, processor, statistic, action_tokenizer,
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
    parser = argparse.ArgumentParser(description="Offline eval + ZMQ server (gated fusion + tactile-FLARE)")
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
    parser.add_argument("--n_flare_tokens_per_frame", type=int, default=0)
    parser.add_argument("--n_flare_steps", type=int, default=0)
    parser.add_argument("--use_tactile_flare", type=int, default=0)
    parser.add_argument("--n_tfl_tokens_per_step", type=int, default=0)
    parser.add_argument("--n_tfl_steps", type=int, default=0)
    parser.add_argument("--tactile_flare_stride", type=int, default=2)
    parser.add_argument("--save_dir", type=str, default="./test_output_tflare_gate")
    parser.add_argument("--num_test_samples", type=int, default=300)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))

    args = parser.parse_args()
    main(args)

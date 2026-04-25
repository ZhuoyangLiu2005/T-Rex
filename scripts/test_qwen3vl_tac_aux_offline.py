"""
Offline evaluation for Qwen3-VL MoT VLA trained with train_qwen3vl_tac_aux.py.

Action MSE + rich tactile-aux diagnostics:
  - Per-finger contact classification accuracy + AUC
  - Per-finger force regression MAE (in normalized units and approx N)
  - Tactile-FLARE cosine similarity per future step
  - Scatter: contact prediction confidence vs raw F6 magnitude

Iterates the test JSON with tactile history (T past frames fetched by
index, same episode).
"""

import os, sys, copy
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)
sys.path.insert(0, _SCRIPT_DIR)

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

from test_qwen3vl_tac_aux_real import (
    _normalize, _denormalize, _build_qwen3vl_from_config, _has_hf_weights,
    _attach_tac_aux_modules, model_load as _model_load_real,
    denoise_action, TactileHistoryBuffer,
)


def model_load(args):
    return _model_load_real(args)


def _open_rgb(path, image_size=None):
    img = Image.open(path).convert("RGB")
    if image_size is not None:
        img = img.resize(image_size, Image.LANCZOS)
    return img


def _open_gray(path):
    return np.array(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def _episode_prefix(sample):
    paths = sample.get("input_image_slow", [])
    return os.path.dirname(paths[0]) if paths else ""


def _fetch_tactile_history(sample, idx, all_samples, T):
    """Return (f6_hist [T, nf*6], deform_paths_hist [T, list]) — same episode, pad with current."""
    cur_prefix = _episode_prefix(sample)
    f6_hist, deform_paths_hist = [], []
    for t in range(T - 1, 0, -1):
        past_idx = idx - t
        past_sample = None
        if past_idx >= 0:
            cand = all_samples[past_idx]
            if _episode_prefix(cand) == cur_prefix:
                past_sample = cand
        src = past_sample if past_sample is not None else sample
        f6_hist.append(src.get("tactile_f6"))
        deform_paths_hist.append(list(src.get("tactile_image_deform", []) or []))
    f6_hist.append(sample.get("tactile_f6"))
    deform_paths_hist.append(list(sample.get("tactile_image_deform", []) or []))
    return f6_hist, deform_paths_hist


def _fetch_tactile_future(sample, idx, all_samples, S, stride):
    cur_prefix = _episode_prefix(sample)
    f6_list, deform_paths_list = [], []
    for k in range(S):
        fut_idx = idx + (k + 1) * stride
        fut_sample = None
        if 0 <= fut_idx < len(all_samples):
            cand = all_samples[fut_idx]
            if _episode_prefix(cand) == cur_prefix:
                fut_sample = cand
        src = fut_sample if fut_sample is not None else sample
        f6_list.append(src.get("tactile_f6"))
        deform_paths_list.append(list(src.get("tactile_image_deform", []) or []))
    return f6_list, deform_paths_list


def _prepare_inputs(args, model, processor, statistic, sample, idx, all_samples, data_dir):
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

    # Tactile history
    T = max(args.tactile_history_len, 1)
    f6_hist_list, deform_paths_hist_list = _fetch_tactile_history(sample, idx, all_samples, T)

    tac_f6_hist = None
    if args.use_tactile_vec and any(h is not None for h in f6_hist_list):
        f6_raw = np.array(f6_hist_list, dtype=np.float32).reshape(T, -1, 6)
        flat = f6_raw.reshape(T, -1)
        normed = _normalize(flat, statistic["tacf6_mask"],
                            statistic["tacf6_min"], statistic["tacf6_max"])
        tac_f6_hist = torch.tensor(
            normed.reshape(1, T, -1, 6), dtype=dtype).to(device)

    tac_deform_hist = None
    if args.use_tactile_deform and deform_paths_hist_list[0]:
        arr = np.stack([
            np.stack([_open_gray(_abs(p)) for p in paths])
            for paths in deform_paths_hist_list
        ])  # [T, nf, H, W]
        tac_deform_hist = torch.tensor(arr[np.newaxis, :, :, np.newaxis, :, :]).to(device, dtype=dtype)

    return {
        "slow_embeds": slow_embeds,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "state_embeds": state_embeds,
        "fast_embeds": fast_embeds,
        "tac_f6_hist": tac_f6_hist,
        "tac_deform_hist": tac_deform_hist,
    }


@torch.inference_mode()
def _single_pass_with_tac_heads(model, prepared, args):
    """One forward at noise_level=1.0, includes the tactile query block so
    we can read contact_logits / force_pred / tflare_pred."""
    device = prepared["slow_embeds"].device
    dtype = prepared["slow_embeds"].dtype
    B, H = 1, prepared["slow_embeds"].shape[2]
    n_fingers = args.n_fingers
    K_tac = (args.n_tfl_tokens_per_step * args.n_tfl_steps) if args.use_tactile_flare else 0

    # Encode tactile once
    tac_f6_tok = None
    tac_deform_tok = None
    if prepared["tac_f6_hist"] is not None:
        f6_emb = model.tacf6_embedder(prepared["tac_f6_hist"].to(dtype))
        if f6_emb.shape[1] > 1 and hasattr(model, "tac_pool_f6"):
            tac_f6_tok = model.tac_pool_f6(f6_emb)
        else:
            tac_f6_tok = f6_emb[:, -1]
    if prepared["tac_deform_hist"] is not None:
        d_hist = prepared["tac_deform_hist"]
        Bs, Ts, nf_d, C, Hh, Ww = d_hist.shape
        dfeats = model.deform_encoder(d_hist.view(-1, C, Hh, Ww)).view(Bs, Ts, nf_d, -1)
        def_emb = model.deform_proj(dfeats.to(dtype))
        if def_emb.shape[1] > 1 and hasattr(model, "tac_pool_deform"):
            tac_deform_tok = model.tac_pool_deform(def_emb)
        else:
            tac_deform_tok = def_emb[:, -1]

    fast_embeds  = (prepared["fast_embeds"] if prepared["fast_embeds"] is not None
                    else torch.empty((B, 0, H), device=device, dtype=dtype))
    state_embeds = (prepared["state_embeds"] if prepared["state_embeds"] is not None
                    else torch.empty((B, 0, H), device=device, dtype=dtype))

    chunk = args.action_chunk
    x_t = torch.randn(1, chunk, args.action_dim, dtype=dtype, device=device)
    t_val = torch.tensor([1.0], dtype=dtype, device=device)
    noisy_actions = model.x_embedder(x_t)
    timesteps = model.t_embedder(t_val).unsqueeze(1)

    act_parts = [fast_embeds, state_embeds]
    if tac_f6_tok is not None:    act_parts.append(tac_f6_tok)
    if tac_deform_tok is not None: act_parts.append(tac_deform_tok)
    act_parts.extend([timesteps, noisy_actions])
    action_block = torch.cat(act_parts, dim=1)
    n_action = action_block.shape[1]

    tac_block_parts = []
    if K_tac > 0 and hasattr(model, "tactile_flare_queries"):
        tac_block_parts.append(model.tactile_flare_queries.expand(B, -1, -1).to(device=device, dtype=dtype))
    if hasattr(model, "contact_queries"):
        tac_block_parts.append(model.contact_queries.expand(B, -1, -1).to(device=device, dtype=dtype))
        tac_block_parts.append(model.force_queries.expand(B, -1, -1).to(device=device, dtype=dtype))
    tactile_block = torch.cat(tac_block_parts, dim=1) if tac_block_parts else None
    n_tactile = tactile_block.shape[1] if tactile_block is not None else 0

    L_latent = prepared["slow_embeds"].shape[1]
    if tactile_block is not None:
        full = torch.cat([prepared["slow_embeds"], action_block, tactile_block], dim=1)
    else:
        full = torch.cat([prepared["slow_embeds"], action_block], dim=1)
    L_total = full.shape[1]

    latent_indexes  = torch.arange(0, L_latent, device=device)
    action_indexes  = torch.arange(L_latent, L_latent + n_action, device=device)
    tactile_indexes = (torch.arange(L_latent + n_action, L_total, device=device)
                       if tactile_block is not None else torch.arange(0, 0, device=device))

    outputs = model.model(
        inputs_embeds=full, position_ids=prepared["position_ids"],
        attention_mask=prepared["attention_mask"], use_cache=False,
        latent_indexes=latent_indexes, action_indexes=action_indexes,
        tactile_indexes=tactile_indexes,
    )
    hidden = outputs.last_hidden_state

    # Extract heads
    result = {}
    if tactile_block is not None:
        cursor = L_latent + n_action
        if K_tac > 0 and hasattr(model, "tactile_flare_proj"):
            h_tflare = hidden[:, cursor:cursor + K_tac]
            result["tflare_pred"] = model.tactile_flare_proj(h_tflare)
            cursor += K_tac
        if hasattr(model, "contact_head"):
            h_contact = hidden[:, cursor:cursor + n_fingers]
            cursor += n_fingers
            result["contact_logits"] = model.contact_head(h_contact).squeeze(-1)  # [1, nf]
        if hasattr(model, "force_head"):
            h_force = hidden[:, cursor:cursor + n_fingers]
            cursor += n_fingers
            result["force_pred"] = model.force_head(h_force).squeeze(-1)  # [1, nf]

    return result


@torch.inference_mode()
def _tflare_target(model, statistic, args, f6_list, deform_paths_list, data_dir):
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
        ])
        dt_arr = torch.tensor(arr[np.newaxis, :, :, np.newaxis, :, :]).to(device, dtype=dtype)
        Bs, S_, nf, C, Hh, Ww = dt_arr.shape
        feats = model.deform_encoder(dt_arr.view(-1, C, Hh, Ww)).view(Bs, S_, nf, -1)
        tgt_parts.append(model.target_deform_proj(feats.to(dtype)))

    if not tgt_parts:
        return None
    tgt_all = torch.cat(tgt_parts, dim=2)
    B_, S_, nf_t, H_ = tgt_all.shape
    flat = tgt_all.view(B_ * S_, nf_t, H_).permute(0, 2, 1)
    pooled = F.adaptive_avg_pool1d(flat.float(), T_per).permute(0, 2, 1).to(dtype)
    return pooled.view(B_, S_ * T_per, H_)


def main(args):
    print(f"Loading VLA model from checkpoint: {args.checkpoint_path}")
    model, processor, statistic, action_tokenizer = model_load(args)
    print("Model loaded successfully!")
    device = f"cuda:{args.cuda}"
    dtype = torch.bfloat16
    model = model.to(device).eval()

    with open(args.test_json_path) as f:
        all_samples = json.load(f)
    test_samples = all_samples[:args.num_test_samples] if args.num_test_samples > 0 else all_samples
    data_dir = os.path.dirname(os.path.abspath(args.test_json_path))
    os.makedirs(args.save_dir, exist_ok=True)

    K_tac = (args.n_tfl_tokens_per_step * args.n_tfl_steps) if args.use_tactile_flare else 0
    has_tflare_eval = bool(args.use_tactile_flare and hasattr(model, "tactile_flare_proj"))

    mse_per_sample = []
    all_pred, all_gt = [], []
    tflare_sim_per_sample = []

    contact_pred_all = []      # [N, n_fingers] probabilities
    contact_gt_all = []        # [N, n_fingers] 0/1
    force_pred_all = []        # [N, n_fingers] (in normalized units)
    force_gt_all = []          # [N, n_fingers]
    contact_force_raw_all = [] # [N, n_fingers] raw ||F_xyz||

    for step, sample in enumerate(tqdm(test_samples, desc="Offline eval")):
        try:
            prepared = _prepare_inputs(args, model, processor, statistic,
                                       sample, step, all_samples, data_dir)
        except FileNotFoundError as e:
            print(f"\n[Warning] Step {step}: {e}")
            continue

        # Action denoising (no tactile block — action prediction doesn't need it)
        noise = torch.randn(1, args.action_chunk, args.action_dim, dtype=dtype, device=device)
        samples = denoise_action(
            model=model,
            inputs_embeds=prepared["slow_embeds"],
            position_ids=prepared["position_ids"],
            attention_mask=prepared["attention_mask"],
            noise=noise, num_steps=10,
            state_embeds=prepared["state_embeds"],
            tac_f6_hist=prepared["tac_f6_hist"],
            tac_deform_hist=prepared["tac_deform_hist"],
            fast_embeds=prepared["fast_embeds"],
            include_tactile_block=False,
            n_fingers=args.n_fingers, K_tac=K_tac,
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

        # Full forward with tactile queries for diagnostics
        diag = _single_pass_with_tac_heads(model, prepared, args)

        if "contact_logits" in diag:
            cp = torch.sigmoid(diag["contact_logits"]).float().cpu().numpy()[0]  # [n_fingers]
            contact_pred_all.append(cp)
            f6_raw = np.array(sample["tactile_f6"], dtype=np.float32).reshape(-1, 6)
            force_mag = np.linalg.norm(f6_raw[:, :3], axis=-1)  # [n_fingers]
            contact_gt_all.append((force_mag > args.contact_force_threshold).astype(np.float32))
            contact_force_raw_all.append(force_mag)

        if "force_pred" in diag:
            fp = diag["force_pred"].float().cpu().numpy()[0]  # [n_fingers]
            force_pred_all.append(fp)
            f6_raw = np.array(sample["tactile_f6"], dtype=np.float32).reshape(-1, 6)
            force_mag = np.linalg.norm(f6_raw[:, :3], axis=-1)
            force_gt_all.append((force_mag / max(args.force_scale, 1e-6)).astype(np.float32))

        if has_tflare_eval and "tflare_pred" in diag:
            f6_list, deform_paths_list = _fetch_tactile_future(
                sample, step, all_samples, args.n_tfl_steps, args.tactile_flare_stride)
            tflare_tgt = _tflare_target(model, statistic, args,
                                        f6_list, deform_paths_list, data_dir)
            if tflare_tgt is not None:
                pred_n = F.normalize(diag["tflare_pred"].float(), dim=-1)
                tgt_n  = F.normalize(tflare_tgt.float(), dim=-1)
                cos = (pred_n * tgt_n).sum(dim=-1).squeeze(0)
                per_step = cos.view(args.n_tfl_steps,
                                    args.n_tfl_tokens_per_step).mean(dim=1)
                tflare_sim_per_sample.append(per_step.cpu().numpy())

    # ── Summary ─────────────────────────────────────────────────────────
    n_valid = len(mse_per_sample)
    print(f"\n=== Action MSE: {n_valid}/{len(test_samples)} valid, "
          f"mean={np.mean(mse_per_sample):.6f} ===")

    if contact_pred_all:
        cp = np.stack(contact_pred_all)     # [N, nf]
        cg = np.stack(contact_gt_all)       # [N, nf]
        raw = np.stack(contact_force_raw_all)  # [N, nf]

        # Per-finger + overall accuracy at threshold 0.5
        pred_bin = (cp > 0.5).astype(np.float32)
        acc_per_finger = (pred_bin == cg).mean(axis=0)
        acc_overall = (pred_bin == cg).mean()

        # Per-finger AUC (rank-based, pooled across samples)
        def _auc(scores, labels):
            pos = labels.sum(); neg = (1 - labels).sum()
            if pos == 0 or neg == 0: return float("nan")
            order = np.argsort(-scores)
            tp = np.cumsum(labels[order]); fp = np.cumsum(1 - labels[order])
            return float(np.trapezoid(tp / pos, fp / neg))
        auc_per_finger = np.array([_auc(cp[:, i], cg[:, i]) for i in range(cp.shape[1])])

        print(f"=== Contact classification: overall acc={acc_overall:.3f}, "
              f"per-finger acc={np.round(acc_per_finger, 3).tolist()} ===")
        print(f"=== Contact AUC per finger: {np.round(auc_per_finger, 3).tolist()} ===")

        plt.figure(figsize=(10, 4))
        xs = np.arange(cp.shape[1])
        plt.bar(xs - 0.2, acc_per_finger, width=0.4, label="accuracy", color="steelblue")
        plt.bar(xs + 0.2, auc_per_finger, width=0.4, label="AUC", color="darkorange")
        plt.xlabel("finger index"); plt.ylabel("score"); plt.ylim(0, 1)
        plt.title("Contact head: per-finger accuracy & AUC")
        plt.legend(); plt.grid(True, axis="y", linestyle=":", alpha=0.6); plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "contact_per_finger.png")); plt.close()

        # Scatter: predicted contact prob vs raw force magnitude (flattened)
        plt.figure(figsize=(7, 4))
        plt.scatter(raw.flatten(), cp.flatten(), alpha=0.3, s=6, color="steelblue")
        plt.axvline(args.contact_force_threshold, color="red", linestyle="--",
                    label=f"train threshold {args.contact_force_threshold}")
        plt.xlabel("raw |F_xyz|"); plt.ylabel("predicted contact prob")
        plt.title("Contact prediction vs. ground-truth force magnitude")
        plt.legend(); plt.grid(True, linestyle=":", alpha=0.6); plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "contact_vs_force.png")); plt.close()

        np.savez(os.path.join(args.save_dir, "contact_diagnostics.npz"),
                 pred=cp, gt=cg, raw_force=raw,
                 acc_per_finger=acc_per_finger, auc_per_finger=auc_per_finger)

    if force_pred_all:
        fp = np.stack(force_pred_all)
        fg = np.stack(force_gt_all)
        mae_per_finger = np.abs(fp - fg).mean(axis=0)
        mae_overall = np.abs(fp - fg).mean()
        # Convert to approx Newtons by multiplying by force_scale
        mae_N_overall = mae_overall * args.force_scale
        print(f"=== Force regression: overall MAE={mae_overall:.4f} "
              f"(≈ {mae_N_overall:.4f} N), per-finger MAE={np.round(mae_per_finger, 3).tolist()} ===")

        plt.figure(figsize=(10, 4))
        plt.bar(range(fp.shape[1]), mae_per_finger, color="crimson", alpha=0.8)
        plt.xlabel("finger index"); plt.ylabel("MAE (normalized)")
        plt.title(f"Force regression MAE per finger (scale={args.force_scale})")
        plt.grid(True, axis="y", linestyle=":", alpha=0.6); plt.tight_layout()
        plt.savefig(os.path.join(args.save_dir, "force_mae_per_finger.png")); plt.close()

        np.savez(os.path.join(args.save_dir, "force_diagnostics.npz"),
                 pred=fp, gt=fg, mae_per_finger=mae_per_finger, mae_overall=mae_overall)

    if tflare_sim_per_sample:
        sims = np.stack(tflare_sim_per_sample)
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

    from test_qwen3vl_tac_aux_real import model_predict, TactileHistoryBuffer as _TH
    tac_history = _TH(T=args.tactile_history_len, n_fingers=args.n_fingers)

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
            actions = model_predict(args, model, processor, statistic, action_tokenizer,
                                    tac_history,
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
    parser = argparse.ArgumentParser(description="Offline eval + ZMQ server (tactile-aux)")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument("--stats_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--test_json_path", type=str, default="")
    parser.add_argument("--action_dim", type=int, default=31)
    parser.add_argument("--action_chunk", type=int, default=8)
    parser.add_argument("--use_robot_state", type=int, default=0)
    parser.add_argument("--use_tactile_deform", type=int, default=1)
    parser.add_argument("--use_tactile_vec", type=int, default=1)
    parser.add_argument("--tactile_intermediate_size", type=int, default=0)
    parser.add_argument("--n_flare_tokens_per_frame", type=int, default=0)
    parser.add_argument("--n_flare_steps", type=int, default=0)
    parser.add_argument("--use_tactile_flare", type=int, default=0)
    parser.add_argument("--n_tfl_tokens_per_step", type=int, default=0)
    parser.add_argument("--n_tfl_steps", type=int, default=0)
    parser.add_argument("--tactile_flare_stride", type=int, default=2)
    parser.add_argument("--tactile_history_len", type=int, default=8)
    parser.add_argument("--n_fingers", type=int, default=10)
    parser.add_argument("--contact_force_threshold", type=float, default=0.5)
    parser.add_argument("--force_scale", type=float, default=2.0)
    parser.add_argument("--include_tactile_queries", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./test_output_tac_aux")
    parser.add_argument("--num_test_samples", type=int, default=300)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))

    args = parser.parse_args()
    main(args)

"""
Representation-separation probing for Qwen3-VL MoT VLA trained with
train_qwen3vl_tflare_gate.py.

Produces three pieces of evidence for "action expert ≠ tactile expert":

  P1. Linear probing. Frozen single-pass hidden states from each expert
      (+ latent as a control) are fed through a linear head trained to
      predict:
        - contact_force: max finger ||F_xyz||   (regression, R²)
        - contact_binary: force > threshold     (classification, AUC)
        - xyz_velocity:  ||action[:,:3]||        (regression, R²)
        - hand_velocity: ||action[:,9:31]||      (regression, R²)
      Outputs a (expert × target) heatmap.

  P3. CKA (linear Centered Kernel Alignment) between expert features.
      Low CKA(h_act, h_tac) → experts encode different things. Also
      compute CKA of each expert with raw tactile F6 and raw vision pool
      (pooled slow-image embeddings) as reference reference spaces.

  Gate / divergence: plots reused from the offline script (gate vs
  contact, ‖v_tac − v_act‖ vs contact, per-action-dim gate bar).

Usage: run on a trained checkpoint + a held-out test JSON.
"""

import os, sys, copy
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from test_qwen3vl_tflare_gate_real import (
    _normalize, _denormalize, _attach_gate_and_tflare, model_load,
)
from test_qwen3vl_tflare_gate_offline import (
    _prepare_inputs, _single_pass_capture,
)


# ─── Linear probing ─────────────────────────────────────────────────────────

def _linear_probe_regression(X, y, seed=0, ridge=1e-3):
    """Closed-form ridge regression. Returns R² on held-out (80/20 split)."""
    rng = np.random.RandomState(seed)
    N = X.shape[0]
    perm = rng.permutation(N)
    n_train = int(0.8 * N)
    Xtr, ytr = X[perm[:n_train]], y[perm[:n_train]]
    Xte, yte = X[perm[n_train:]], y[perm[n_train:]]
    # Center
    mu_x, mu_y = Xtr.mean(0, keepdims=True), ytr.mean()
    Xc, yc = Xtr - mu_x, ytr - mu_y
    XtX = Xc.T @ Xc + ridge * np.eye(Xc.shape[1])
    w = np.linalg.solve(XtX, Xc.T @ yc)
    pred = (Xte - mu_x) @ w + mu_y
    ss_res = ((yte - pred) ** 2).sum()
    ss_tot = ((yte - yte.mean()) ** 2).sum() + 1e-8
    return 1.0 - ss_res / ss_tot


def _linear_probe_auc(X, y_bin, seed=0, ridge=1e-3):
    """Ridge-regression score → ROC-AUC (closed-form, no sklearn)."""
    rng = np.random.RandomState(seed)
    N = X.shape[0]
    perm = rng.permutation(N)
    n_train = int(0.8 * N)
    Xtr, ytr = X[perm[:n_train]], y_bin[perm[:n_train]].astype(np.float32)
    Xte, yte = X[perm[n_train:]], y_bin[perm[n_train:]].astype(np.float32)
    mu_x = Xtr.mean(0, keepdims=True)
    Xc = Xtr - mu_x
    XtX = Xc.T @ Xc + ridge * np.eye(Xc.shape[1])
    w = np.linalg.solve(XtX, Xc.T @ (ytr - ytr.mean()))
    scores = (Xte - mu_x) @ w
    # Compute AUC
    order = np.argsort(-scores)
    yte = yte[order]
    pos = yte.sum()
    neg = len(yte) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    tp = np.cumsum(yte)
    fp = np.cumsum(1 - yte)
    tpr = tp / pos
    fpr = fp / neg
    # Trapezoid AUC
    auc = np.trapz(tpr, fpr)
    return float(auc)


# ─── CKA ────────────────────────────────────────────────────────────────────

def linear_cka(X, Y):
    """Centered linear CKA between two feature matrices [N, D_x], [N, D_y]."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = (X.T @ Y)
    xx = (X.T @ X)
    yy = (Y.T @ Y)
    num = (xy ** 2).sum()
    den = np.sqrt((xx ** 2).sum() * (yy ** 2).sum()) + 1e-12
    return float(num / den)


# ─── Main ───────────────────────────────────────────────────────────────────

def _open_gray_from_path(path, data_dir):
    from PIL import Image
    _abs = path if os.path.isabs(path) else os.path.join(data_dir, path)
    return np.array(Image.open(_abs).convert("L"), dtype=np.float32) / 255.0


def collect_features(args, model, processor, statistic, samples, data_dir):
    """Run single forward pass per sample, collect:
       h_act, h_tac, h_latent (pooled), v_act, v_tac, gate, contact_force,
       xyz_velocity, hand_velocity."""
    device = f"cuda:{args.cuda}"
    model = model.to(device).eval()

    feats = {"h_act": [], "h_tac": [], "h_latent": [],
             "v_act": [], "v_tac": [], "gate": [],
             "contact_force": [], "xyz_vel": [], "hand_vel": [],
             "vision_pool": [], "tactile_pool": []}

    for step, sample in enumerate(tqdm(samples, desc="Collecting features")):
        try:
            prepared = _prepare_inputs(args, model, processor, statistic, sample, data_dir)
        except FileNotFoundError as e:
            print(f"\n[skip] {e}")
            continue

        with torch.inference_mode():
            cap = _single_pass_capture(model, prepared, args, noise_level=1.0)

        # Pool hidden states across chunk dim
        h_act = cap["h_act"].mean(dim=1).squeeze(0).float().cpu().numpy()      # [H]
        feats["h_act"].append(h_act)
        feats["v_act"].append(cap["v_act"].squeeze(0).float().cpu().numpy())   # [chunk, D]

        if cap["has_tactile"]:
            h_tac = cap["h_tac"].mean(dim=1).squeeze(0).float().cpu().numpy()
            feats["h_tac"].append(h_tac)
            feats["v_tac"].append(cap["v_tac"].squeeze(0).float().cpu().numpy())
            feats["gate"].append(cap["gate"].squeeze(0).float().cpu().numpy())
        else:
            feats["h_tac"].append(None)
            feats["v_tac"].append(None)
            feats["gate"].append(None)

        # Latent pooled: mean over the slow embeddings (no flare_q)
        L_slow = prepared["slow_embeds"].shape[1] - (
            model.n_flare_tokens if model.n_flare_tokens > 0 else 0)
        with torch.inference_mode():
            h_lat = prepared["slow_embeds"][0, :L_slow].mean(dim=0).float().cpu().numpy()
        feats["h_latent"].append(h_lat)

        # Reference: tactile pool (raw embedded tactile), vision pool (slow tokens)
        with torch.inference_mode():
            tac_pool_parts = []
            if prepared["tac_f6"] is not None:
                tac_pool_parts.append(prepared["tac_f6"].mean(dim=(0, 1)).float().cpu().numpy())
            if prepared["tac_deform"] is not None:
                Bs, nf, C, Hh, Ww = prepared["tac_deform"].shape
                df = model.deform_encoder(prepared["tac_deform"].view(-1, C, Hh, Ww))
                df = df.view(Bs, nf, -1).mean(dim=(0, 1)).float().cpu().numpy()
                tac_pool_parts.append(df)
            tac_pool = np.concatenate(tac_pool_parts) if tac_pool_parts else None
            vision_pool = prepared["slow_embeds"][0, :L_slow].mean(dim=0).float().cpu().numpy()
        feats["tactile_pool"].append(tac_pool)
        feats["vision_pool"].append(vision_pool)

        # Probe targets
        f6_raw = np.array(sample["tactile_f6"], dtype=np.float32).reshape(-1, 6)
        feats["contact_force"].append(float(np.linalg.norm(f6_raw[:, :3], axis=-1).max()))

        act = np.array(sample["action"], dtype=np.float32)
        act0 = act[0] if act.ndim > 1 else act
        # Per-arm split: arm0 = dims [0,9), arm0_hand = [9,31), arm1 = [31,40), arm1_hand = [40,62)
        if args.action_dim >= 62:
            xyz = np.linalg.norm(np.concatenate([act0[0:3], act0[31:34]]))
            hand = np.linalg.norm(np.concatenate([act0[9:31], act0[40:62]]))
        else:
            xyz = np.linalg.norm(act0[0:3])
            hand = np.linalg.norm(act0[9:31])
        feats["xyz_vel"].append(float(xyz))
        feats["hand_vel"].append(float(hand))

    return feats


def main(args):
    print(f"Loading VLA model from checkpoint: {args.checkpoint_path}")
    model, processor, statistic, action_tokenizer = model_load(args)
    print("Model loaded successfully!")

    with open(args.test_json_path) as f:
        all_samples = json.load(f)
    samples = all_samples[:args.num_samples] if args.num_samples > 0 else all_samples
    data_dir = os.path.dirname(os.path.abspath(args.test_json_path))

    os.makedirs(args.save_dir, exist_ok=True)
    feats = collect_features(args, model, processor, statistic, samples, data_dir)

    # Keep only entries where tactile is present
    keep = [i for i, h in enumerate(feats["h_tac"]) if h is not None]
    N = len(keep)
    print(f"Collected {N} samples with tactile.")
    if N < 20:
        print("Too few samples — exiting.")
        return

    def _stack(key):
        return np.stack([feats[key][i] for i in keep])

    h_act    = _stack("h_act")         # [N, H]
    h_tac    = _stack("h_tac")
    h_latent = _stack("h_latent")
    vpool    = _stack("vision_pool")
    tpool    = _stack("tactile_pool") if feats["tactile_pool"][keep[0]] is not None else None

    contact_force = np.array([feats["contact_force"][i] for i in keep])
    xyz_vel       = np.array([feats["xyz_vel"][i] for i in keep])
    hand_vel      = np.array([feats["hand_vel"][i] for i in keep])

    # Binary contact label (median-split)
    thresh = np.median(contact_force)
    contact_bin = (contact_force > thresh).astype(np.int32)

    # ── P1: linear probe heatmap ─────────────────────────────────────────
    experts = {"action": h_act, "tactile": h_tac, "latent": h_latent}
    targets = [
        ("contact_force (R²)",  contact_force, "regression"),
        ("contact_binary (AUC)", contact_bin,  "classification"),
        ("xyz_vel (R²)",         xyz_vel,       "regression"),
        ("hand_vel (R²)",        hand_vel,      "regression"),
    ]
    probe_mat = np.zeros((len(experts), len(targets)))
    for i, (name_e, X) in enumerate(experts.items()):
        for j, (name_t, y, kind) in enumerate(targets):
            if kind == "regression":
                score = _linear_probe_regression(X, y)
            else:
                score = _linear_probe_auc(X, y)
            probe_mat[i, j] = score
            print(f"  Probe {name_e:8s} → {name_t:25s}: {score:.3f}")

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(probe_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([t[0] for t in targets], rotation=15, ha="right")
    ax.set_yticks(range(len(experts)))
    ax.set_yticklabels(list(experts.keys()))
    for i in range(probe_mat.shape[0]):
        for j in range(probe_mat.shape[1]):
            ax.text(j, i, f"{probe_mat[i,j]:.2f}", ha="center", va="center", fontsize=10)
    plt.colorbar(im, ax=ax, label="score")
    plt.title("Linear probes: expert × target"); plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "probe_heatmap.png")); plt.close()

    # ── P3: CKA matrix ───────────────────────────────────────────────────
    cka_space_names = ["action", "tactile", "latent"]
    cka_spaces = [h_act, h_tac, h_latent]
    ref_names, ref_spaces = [], []
    if tpool is not None:
        ref_names.append("tac_ref"); ref_spaces.append(tpool)
    ref_names.append("vis_ref"); ref_spaces.append(vpool)

    all_names  = cka_space_names + ref_names
    all_spaces = cka_spaces + ref_spaces
    M = len(all_spaces)
    cka_mat = np.zeros((M, M))
    for i in range(M):
        for j in range(M):
            cka_mat[i, j] = linear_cka(all_spaces[i], all_spaces[j])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cka_mat, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(M)); ax.set_xticklabels(all_names, rotation=30, ha="right")
    ax.set_yticks(range(M)); ax.set_yticklabels(all_names)
    for i in range(M):
        for j in range(M):
            ax.text(j, i, f"{cka_mat[i,j]:.2f}", ha="center", va="center",
                    color="white" if cka_mat[i, j] < 0.5 else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="linear CKA")
    plt.title("Representation similarity (CKA)"); plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "cka_matrix.png")); plt.close()

    # ── Gate / divergence scatter ────────────────────────────────────────
    gates = np.stack([feats["gate"][i] for i in keep])        # [N, chunk, D]
    v_act = np.stack([feats["v_act"][i] for i in keep])
    v_tac = np.stack([feats["v_tac"][i] for i in keep])
    diff  = np.sqrt(((v_tac - v_act) ** 2).mean(axis=(1, 2)))

    plt.figure(figsize=(7, 4))
    plt.scatter(contact_force, gates.mean(axis=(1, 2)), alpha=0.5, s=8, color="steelblue")
    plt.xlabel("max finger |F_xyz|"); plt.ylabel("mean gate")
    plt.title("Gate vs contact intensity")
    plt.grid(True, linestyle=":", alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "gate_vs_contact.png")); plt.close()

    plt.figure(figsize=(7, 4))
    plt.scatter(contact_force, diff, alpha=0.5, s=8, color="crimson")
    plt.xlabel("max finger |F_xyz|"); plt.ylabel("‖v_tac − v_act‖_RMS")
    plt.title("Expert-output divergence vs contact intensity")
    plt.grid(True, linestyle=":", alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "vtac_vact_diff_vs_contact.png")); plt.close()

    # Per-action-dim gate bar
    perdim = gates.mean(axis=(0, 1))
    plt.figure(figsize=(10, 4))
    plt.bar(range(len(perdim)), perdim, color="steelblue", alpha=0.8)
    plt.xlabel("action dim"); plt.ylabel("mean gate")
    plt.title("Per-action-dim gate openness")
    plt.grid(True, axis="y", linestyle=":", alpha=0.5); plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "gate_per_action_dim.png")); plt.close()

    # Save raw arrays
    np.savez(os.path.join(args.save_dir, "probing_raw.npz"),
             probe_mat=probe_mat, cka_mat=cka_mat, cka_names=np.array(all_names),
             h_act=h_act, h_tac=h_tac, h_latent=h_latent,
             contact_force=contact_force, xyz_vel=xyz_vel, hand_vel=hand_vel,
             gate=gates, v_act=v_act, v_tac=v_tac, diff=diff)

    print(f"\nSaved probing results to {args.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Representation-separation probing (P1 linear probes + P3 CKA + gate/diff).")
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, default="")
    parser.add_argument("--stats_path", type=str, default="")
    parser.add_argument("--dataset_name", type=str, default="")
    parser.add_argument("--test_json_path", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=500)
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
    parser.add_argument("--save_dir", type=str, default="./probe_output")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--image_size", type=int, nargs=2, default=None, metavar=("W", "H"))

    args = parser.parse_args()
    main(args)

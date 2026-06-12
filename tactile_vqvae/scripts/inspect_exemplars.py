"""Quick visual inspection of VQ-VAE exemplars.npz.

For each of the top-K codes, plots:
  - GT F6 trace (5 fingers × 6 dims)  ← thin lines
  - Reconstructed F6 trace             ← dashed

Usage:
    python inspect_exemplars.py --exemplars /path/to/exemplars.npz --out_dir /tmp/vqvae_plots
    python inspect_exemplars.py --exemplars exemplars.npz           # saves next to .npz
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
FT_NAMES     = ["fx", "fy", "fz", "tx", "ty", "tz"]
COLORS       = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]


def plot_code(code_id: int, f6_gt: np.ndarray, f6_rec: np.ndarray,
              magnitude: float, out_path: str):
    """f6_gt / f6_rec: [T, 5, 6]"""
    T = f6_gt.shape[0]
    fig, axes = plt.subplots(5, 6, figsize=(18, 10), sharex=True)
    fig.suptitle(f"Code {code_id}  |  magnitude={magnitude:.2f}", fontsize=12)

    for fi in range(5):
        for di in range(6):
            ax = axes[fi, di]
            ax.plot(f6_gt[:, fi, di],  color=COLORS[fi], lw=1.2, alpha=0.85, label="GT")
            ax.plot(f6_rec[:, fi, di], color=COLORS[fi], lw=1.0, ls="--", alpha=0.6, label="Recon")
            ax.set_ylim(-1.2, 1.2)
            ax.axhline(0, color="gray", lw=0.4, ls=":")
            if fi == 0:
                ax.set_title(FT_NAMES[di], fontsize=8)
            if di == 0:
                ax.set_ylabel(FINGER_NAMES[fi], fontsize=8)
            ax.tick_params(labelsize=6)

    axes[0, 0].legend(fontsize=6, loc="upper right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exemplars", required=True, help="Path to exemplars.npz")
    p.add_argument("--out_dir",   default=None,  help="Where to save PNGs (default: beside .npz)")
    p.add_argument("--max_codes", type=int, default=20, help="How many codes to plot")
    args = p.parse_args()

    out_dir = args.out_dir or os.path.join(os.path.dirname(args.exemplars), "exemplar_plots")
    os.makedirs(out_dir, exist_ok=True)

    data = np.load(args.exemplars, allow_pickle=False)
    codes     = data["code"]        # [N_ex]
    magnitudes= data["magnitude"]   # [N_ex]
    f6        = data["f6"].astype(np.float32)    # [N_ex, T, 5, 6]
    recon     = data["recon"].astype(np.float32) # [N_ex, T, 5, 6]
    top_codes = data["top_codes"]   # [K]
    counts    = data["counts"]      # [codebook_size]

    total = counts.sum()
    plotted = 0
    for ci in top_codes[:args.max_codes]:
        mask = codes == ci
        if mask.sum() == 0:
            continue
        # Pick highest-magnitude exemplar for this code.
        best = np.argmax(magnitudes[mask])
        indices = np.where(mask)[0]
        i = indices[best]
        freq = counts[ci] / max(1, total)
        out_path = os.path.join(out_dir, f"code{ci:04d}_freq{freq*100:.2f}pct.png")
        plot_code(ci, f6[i], recon[i], float(magnitudes[i]), out_path)
        plotted += 1

    # Also plot code frequency histogram.
    fig, ax = plt.subplots(figsize=(12, 3))
    sorted_counts = np.sort(counts)[::-1]
    ax.bar(range(len(sorted_counts)), sorted_counts, width=1.0, color="#377eb8", alpha=0.8)
    ax.set_xlabel("Code rank (by frequency)")
    ax.set_ylabel("Usage count")
    ax.set_title(f"Codebook usage  |  active={int((counts>0).sum())}/{len(counts)}")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "codebook_histogram.png"), dpi=100)
    plt.close(fig)

    print(f"Saved {plotted} code plots + histogram → {out_dir}/")


if __name__ == "__main__":
    main()

# Tactile VQ-VAE

Discrete codebook over per-hand F6 windows, pretrained on the ~100hr midtrain corpus. Output: a sidecar `tactile_codes.h5` next to every episode, ready to be consumed by a downstream MoT midtrain that does BERT-style next-tactile-event prediction.

## What this is

- **Input per sample**: `[T=16, 5 fingers, 6 force/torque]` — one window from one hand of one episode, normalized via the same q01/q99 stats the existing midtrain dataloader uses.
- **Encoder**: small 1-D CNN, time-strided. Produces a single 256-d continuous code per window.
- **Quantizer**: VQ-EMA, 1024 codes, with dead-code revival (replaces unused codes from the current batch's encoder pool every 200 steps).
- **Decoder**: mirrors the encoder, reconstructs the F6 window.
- **Training loss**: MSE recon (magnitude-weighted: high-contact windows get up to 3× weight, free-air windows get 1×) + commitment loss.

Left and right hands are treated as independent samples — the F6 representation has no left/right structural difference at the per-finger level, so we get 2× effective dataset size.

## Folder layout

```
tactile_vqvae/
├── config/vqvae_f6.yaml        # default hyperparameters
├── data/
│   ├── stats.py                # TacF6Stats — pools q01/q99 across batch manifests
│   └── dataset.py              # F6WindowDataset, build_train_val_datasets
├── models/
│   ├── encoder.py              # F6Encoder (1-D conv stack)
│   ├── decoder.py              # F6Decoder (mirror)
│   ├── quantizer.py            # VQEMAQuantizer (DDP-aware, dead-code revival)
│   └── tactile_vqvae.py        # TactileVQVAE + TactileVQVAEConfig
├── train.py                    # accelerate training loop
├── eval.py                     # codebook utilization + recon by F6-magnitude quartile
├── extract_codes.py            # writes tactile_codes.h5 per episode
└── scripts/
    ├── train_vqvae_f6.sh
    └── extract_codes.sh
```

## Requirements

Same env as the rest of `dex_mot_expert` (`accelerate`, `torch`, `h5py`, `numpy`). No additional dependencies.

## Train

```bash
# Defaults: 2 GPUs, batch 256/GPU, 30 epochs, codebook 1024, window 16.
DATA_ROOT=/path/to/merged \
OUTPUT_DIR=/path/to/output \
bash tactile_vqvae/scripts/train_vqvae_f6.sh
```

Override anything via env vars: `WINDOW`, `STRIDE`, `CODEBOOK`, `EMBED`, `EPOCHS`, `BATCH`, `LR`, `RUN_NAME`, `USE_WANDB`, `CUDA_VISIBLE_DEVICES`.

A successful training run should show:
- `recon` decreasing steadily.
- `perplexity` rising and stabilizing at a few hundred (1024-code regime).
- `active=` near 1024/1024 once dead-code revival has had a few cycles.
- `revived=` non-zero in early epochs, near zero later.

## Smoke test

```bash
python -m tactile_vqvae.train \
    --data_root /path/to/merged \
    --output_dir /tmp/vqvae_smoke \
    --smoke_test 1
```

Runs 5 train steps + 1 save and exits — verifies dataset, model wiring, and checkpoint I/O without committing to a real training run.

## Eval

```bash
python -m tactile_vqvae.eval \
    --checkpoint /path/to/run/latest.pt \
    --data_root  /path/to/merged \
    --exemplars  /path/to/run/exemplars.npz
```

Writes a JSON summary with:
- Overall recon MSE.
- Recon MSE split into 4 quartiles by raw F6 magnitude (verifies the magnitude weighting did its job).
- Codebook perplexity, active-code count/ratio, max per-code frequency.
- (Optional) `exemplars.npz` with the top-K most-used codes and example F6 windows for each — open in any notebook to eyeball semantics.

## Extract codes

After training, materialize discrete codes for every training episode:

```bash
CHECKPOINT=/path/to/run/latest.pt \
DATA_ROOT=/path/to/merged \
NUM_WORKERS=8 \
bash tactile_vqvae/scripts/extract_codes.sh
```

Each `episode_*/` gets a `tactile_codes.h5` containing:

| dataset | shape | dtype | description |
|---|---|---|---|
| `codes_per_chunk` | `[M, 2]` | int32 | One code per (chunk, hand). M = ceil(N / window). |
| `codes_per_frame` | `[N, 2]` | int32 | Per-frame broadcast: frame t inherits chunk `t // window`'s code. |
| attrs `window`, `n_frames`, `n_chunks`, `codebook_size`, `checkpoint_path` | — | — | Metadata for downstream consumers. |

Per-frame codes are convenient for the downstream midtrain: at any frame, look up `codes_per_frame[frame_t]` to get `(left_code, right_code)`.

## Design notes

**Why per-hand tokens?** Two codes per window keeps semantics intact — one hand may be in free contact while the other is grasping; merging them blurs that signal. Cost is just 2× tokens, which is negligible.

**Why magnitude-weighted recon?** The F6 distribution is dominated by free-air (~60–70% of frames). Without weighting, the codebook collapses to a "near-zero" cluster and learns nothing about contact events. The smooth sigmoid-based weight (1 + α·sigmoid(‖F6‖/τ - 1)) keeps free-air gradient at 1× and high-contact gradient at up to (1+α)×, no hard threshold.

**Why VQ-EMA + dead-code revival, not FSQ?** VQ-EMA is the well-trodden default; revival fixes the failure mode where >50% of codes go unused. FSQ would be an easy drop-in alternative if revival turns out to be insufficient — see `quantizer.py` for the substitution point.

## Downstream integration (out of scope here)

A separate change will:
1. Load `episode_dir/tactile_codes.h5` from the existing midtrain dataloader.
2. Add a small embedding table `nn.Embedding(codebook_size, hidden_size)` in the tactile expert.
3. Feed code-embedded tokens into the tactile expert stream alongside (or in place of) the raw F6 embedder.
4. Add a BERT-style next-code prediction auxiliary loss (mask K% of future codes in the chunk, predict from context).

That auxiliary loss is what makes the 100hrs of data actually *train* the tactile expert end-to-end, instead of just conditioning it.

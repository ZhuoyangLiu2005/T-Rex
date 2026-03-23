"""
Compare parameter counts between the baseline and reduced-tactile-MLP
Qwen3-VL MoT models.

Usage:
    python compare_params.py --model_path <Qwen3-VL-2B-Instruct path>
    python compare_params.py --model_path <path> --tactile_intermediate_size 2048

If --tactile_intermediate_size is 0 (default), the script prints the
baseline breakdown and then sweeps several reduction ratios
(1/2, 1/4, 1/8) for comparison.
"""

import os, sys

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import argparse
import torch
from qwen_vla import Qwen3VLVLAModel


def count_params(model):
    """Return per-bucket parameter counts."""
    buckets = {
        "visual": 0, "latent_attn": 0, "latent_mlp": 0, "latent_norm": 0,
        "action_attn": 0, "action_mlp": 0, "action_norm": 0,
        "tactile_attn": 0, "tactile_mlp": 0, "tactile_norm": 0,
        "vla_heads": 0, "other": 0,
    }
    for n, p in model.named_parameters():
        cnt = p.numel()
        if n.startswith("visual") or n.startswith("deform_encoder"):
            buckets["visual"] += cnt
        elif any(k in n for k in ("x_embedder", "t_embedder", "final_layer",
                                   "tacf6_embedder", "deform_proj", "state_embedder")):
            buckets["vla_heads"] += cnt
        elif "_tactile" in n:
            if "mlp" in n:    buckets["tactile_mlp"] += cnt
            elif "norm" in n: buckets["tactile_norm"] += cnt
            else:             buckets["tactile_attn"] += cnt
        elif "_action" in n:
            if "mlp" in n:    buckets["action_mlp"] += cnt
            elif "norm" in n: buckets["action_norm"] += cnt
            else:             buckets["action_attn"] += cnt
        elif "self_attn" in n or "rotary" in n:
            buckets["latent_attn"] += cnt
        elif "mlp" in n and "_action" not in n and "_tactile" not in n:
            buckets["latent_mlp"] += cnt
        elif "norm" in n or "embed_tokens" in n:
            buckets["latent_norm"] += cnt
        else:
            buckets["other"] += cnt
    return buckets


def build_model(model_path, tac_isize, action_dim=31, action_chunk=8):
    """Build a Qwen3VLVLAModel shell (CPU, no pretrained weights needed for counting)."""
    # We need the config but not the actual weights.
    # Use from_pretrained_qwen3vl which loads weights, then discard.
    # For speed, build from config directly.
    import json
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found at {model_path}")

    with open(config_path) as f:
        full_cfg = json.load(f)

    image_token_id = full_cfg.get("image_token_id", 151655)

    try:
        from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
        vl_config = Qwen3VLConfig(**{k: v for k, v in full_cfg.items()
                                     if k not in ("architectures", "transformers_version")})
        text_config = vl_config.text_config
    except Exception:
        from transformers import AutoConfig
        vl_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        text_config = getattr(vl_config, "text_config", vl_config)

    model = Qwen3VLVLAModel(
        config=text_config,
        action_dim=action_dim,
        action_chunk=action_chunk,
        use_tactile_deform=True,
        use_robot_state=False,
        image_token_id=image_token_id,
        tactile_intermediate_size=tac_isize,
    )
    return model, text_config


def print_table(rows, col_widths):
    """Print a formatted table."""
    for row in rows:
        line = ""
        for val, w in zip(row, col_widths):
            if isinstance(val, float):
                line += f"{val:>{w}.2f}"
            elif isinstance(val, int):
                line += f"{val:>{w},}"
            else:
                line += f"{val:<{w}}"
        print(line)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to Qwen3-VL pretrained model dir (needs config.json)")
    parser.add_argument("--tactile_intermediate_size", type=int, default=0,
                        help="Specific size to compare. 0 = sweep multiple ratios.")
    parser.add_argument("--action_dim", type=int, default=31)
    parser.add_argument("--action_chunk", type=int, default=8)
    args = parser.parse_args()

    print("Building baseline model (tactile_intermediate_size = default) ...")
    baseline, text_config = build_model(args.model_path, None, args.action_dim, args.action_chunk)
    base_I = text_config.intermediate_size
    base_buckets = count_params(baseline)
    del baseline

    print(f"\nModel config: hidden_size={text_config.hidden_size}, "
          f"intermediate_size={base_I}, "
          f"num_hidden_layers={text_config.num_hidden_layers}\n")

    # Determine which sizes to compare
    if args.tactile_intermediate_size > 0:
        sizes = [args.tactile_intermediate_size]
    else:
        sizes = [base_I // 2, base_I // 4, base_I // 8]

    all_buckets = {"baseline": base_buckets}
    for s in sizes:
        label = f"tac_I={s}"
        print(f"Building model with tactile_intermediate_size={s} ...")
        m, _ = build_model(args.model_path, s, args.action_dim, args.action_chunk)
        all_buckets[label] = count_params(m)
        del m

    # Print comparison table
    configs = ["baseline"] + [f"tac_I={s}" for s in sizes]
    bucket_keys = [k for k in base_buckets if base_buckets[k] > 0]

    print("\n" + "=" * 80)
    print("  Parameter Comparison (millions)")
    print("=" * 80)

    header = f"{'Component':<22}" + "".join(f"{'|  ' + c:>18}" for c in configs)
    print(header)
    print("-" * len(header))

    for bk in bucket_keys:
        row = f"{bk:<22}"
        for cfg in configs:
            val = all_buckets[cfg][bk] / 1e6
            row += f"|  {val:>13.2f} M"
        print(row)

    print("-" * len(header))
    for label in ["Total", "Total (excl. visual)", "Tactile expert total"]:
        row = f"{label:<22}"
        for cfg in configs:
            b = all_buckets[cfg]
            if label == "Total":
                val = sum(b.values()) / 1e6
            elif label == "Total (excl. visual)":
                val = (sum(b.values()) - b["visual"]) / 1e6
            else:
                val = (b["tactile_attn"] + b["tactile_mlp"] + b["tactile_norm"]) / 1e6
            row += f"|  {val:>13.2f} M"
        print(row)

    # Reduction summary
    print("\n" + "=" * 80)
    print("  Reduction vs. Baseline")
    print("=" * 80)
    base_total = sum(base_buckets.values())
    base_tac = base_buckets["tactile_attn"] + base_buckets["tactile_mlp"] + base_buckets["tactile_norm"]
    base_nonvis = base_total - base_buckets["visual"]

    for s in sizes:
        label = f"tac_I={s}"
        b = all_buckets[label]
        new_total = sum(b.values())
        new_tac = b["tactile_attn"] + b["tactile_mlp"] + b["tactile_norm"]
        new_nonvis = new_total - b["visual"]

        saved = base_total - new_total
        tac_saved = base_tac - new_tac
        print(f"\n  tactile_intermediate_size = {s}  (ratio: 1/{base_I // s})")
        print(f"    Tactile expert:  {base_tac/1e6:.2f}M -> {new_tac/1e6:.2f}M  "
              f"(saved {tac_saved/1e6:.2f}M, {tac_saved/base_tac*100:.1f}%)")
        print(f"    Total model:     {base_total/1e6:.2f}M -> {new_total/1e6:.2f}M  "
              f"(saved {saved/1e6:.2f}M, {saved/base_total*100:.1f}%)")
        print(f"    Excl. visual:    {base_nonvis/1e6:.2f}M -> {new_nonvis/1e6:.2f}M  "
              f"(saved {saved/1e6:.2f}M, {saved/base_nonvis*100:.1f}%)")

    print()


if __name__ == "__main__":
    main()

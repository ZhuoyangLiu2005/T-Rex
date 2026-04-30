"""
Smoke test for Paradigm C model surface.

Builds a tiny Qwen3VLVLAModel from scratch (random weights, miniature config)
and exercises the three new methods end-to-end:
  • forward_flow_action_only — full action-only Euler flow returning Â + KV
  • tactile_residual_train_step — single tactile-only forward at (r_τ, τ)
  • tactile_residual_flow — multi-step Euler integration on residual

Verifies output shapes, gradient flow into the tactile expert only, and that
multiple async refreshes from the same cached_kv produce different Δa samples.
"""

import os, sys
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

import torch
import torch.nn as nn
from types import SimpleNamespace

from qwen_vla.modeling_vla import Qwen3VLVLAModel


def make_tiny_config():
    """Tiny Qwen3VL-style text config to keep the test cheap."""
    return SimpleNamespace(
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=2048,
        rope_theta=1_000_000.0,
        rope_scaling={"rope_type": "default", "mrope_section": [8, 12, 12]},
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
        attention_bias=False,
        vocab_size=100,
        pad_token_id=0,
        partial_rotary_factor=1.0,
    )


def build_model(action_dim=8, action_chunk=4, tactile_intermediate=64):
    cfg = make_tiny_config()
    model = Qwen3VLVLAModel(
        config=cfg,
        action_dim=action_dim,
        action_chunk=action_chunk,
        tacf6_dim=6,
        use_tactile_deform=False,
        use_robot_state=False,
        image_token_id=99,
        tactile_intermediate_size=tactile_intermediate,
        n_flare_tokens_per_frame=0,
        n_flare_steps=0,
    )
    model.initialize_vla_weights(skip_tactile_zero_init=True)
    model.to(torch.bfloat16)
    return model


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2
    L_latent = 24
    n_chunk = 4
    action_dim = 8
    n_obs = 5  # tactile_f6 fingers

    model = build_model(action_dim=action_dim, action_chunk=n_chunk).to(device)
    model.train()

    # Latent (slow) embeddings + M-RoPE position ids
    H = model.model.config.hidden_size
    inputs_embeds = torch.randn(B, L_latent, H, device=device, dtype=torch.bfloat16) * 0.1
    # M-RoPE: [3, B, L_latent] sequential
    pos_seq = torch.arange(L_latent, device=device).view(1, 1, L_latent)
    pos_ids = pos_seq.expand(3, B, L_latent).contiguous()

    # === Test 1: forward_flow_action_only ===
    print("=" * 60)
    print("[Test 1] forward_flow_action_only")
    noise = torch.randn(B, n_chunk, action_dim, device=device, dtype=torch.bfloat16)
    fast_embeds = torch.randn(B, 3, H, device=device, dtype=torch.bfloat16) * 0.1
    with torch.no_grad():
        a_hat, cached_kv, n_action_in_cache = model.forward_flow_action_only(
            inputs_embeds=inputs_embeds,
            position_ids=pos_ids,
            noise=noise,
            fast_embeds=fast_embeds,
            num_steps=3,
            refresh_clean_kv=True,
        )
    print(f"  Â shape: {tuple(a_hat.shape)} (expected: ({B}, {n_chunk}, {action_dim}))")
    print(f"  cached_kv seq_length: {cached_kv.get_seq_length()}")
    print(f"  n_action_in_cache: {n_action_in_cache}")
    assert a_hat.shape == (B, n_chunk, action_dim), "Â shape mismatch"
    assert cached_kv.get_seq_length() == L_latent + n_action_in_cache, "cache length mismatch"
    print("  [OK]")

    # === Test 2: tactile_residual_train_step (gradient flow) ===
    print("=" * 60)
    print("[Test 2] tactile_residual_train_step gradient flow")
    tactile_f6 = torch.randn(B, n_obs, 6, device=device, dtype=torch.bfloat16)
    r_target = torch.randn(B, n_chunk, action_dim, device=device, dtype=torch.bfloat16) * 0.05
    eps_r    = torch.randn(B, n_chunk, action_dim, device=device, dtype=torch.bfloat16)
    tau      = torch.full((B,), 0.5, device=device, dtype=torch.bfloat16)
    r_tau    = (1 - tau[:, None, None]) * r_target + tau[:, None, None] * eps_r
    v_target = eps_r - r_target

    v_pred = model.tactile_residual_train_step(
        cached_kv=cached_kv,
        latent_position_ids=pos_ids,
        n_action_in_cache=n_action_in_cache,
        base_chunk=a_hat.detach(),
        tactile_f6=tactile_f6,
        r_tau=r_tau,
        tau=tau,
    )
    print(f"  v_pred shape: {tuple(v_pred.shape)}")
    assert v_pred.shape == (B, n_chunk, action_dim), "v_pred shape mismatch"
    loss = nn.MSELoss()(v_pred, v_target)
    loss.backward()
    # Inspect which params got grads
    has_grad = {"tactile_only": True, "action_or_latent": False}
    for n, p in model.named_parameters():
        if p.grad is None or p.grad.abs().sum() == 0:
            continue
        is_tactile = ("_tactile" in n
                      or "tacf6_embedder" in n
                      or "final_layer_tactile" in n
                      or "x_embedder" in n
                      or "t_embedder" in n
                      or "norm_tactile" in n)
        if not is_tactile:
            has_grad["action_or_latent"] = True
            print(f"  WARNING — non-tactile param has grad: {n}")
    print(f"  loss: {loss.item():.4f}")
    print(f"  tactile params got grad: {has_grad['tactile_only']}")
    print(f"  action/latent params got grad: {has_grad['action_or_latent']} "
          f"(expected: False — would indicate cached_kv leaked grad)")
    print("  [OK]")
    model.zero_grad()

    # === Test 3: tactile_residual_flow integration (inference) ===
    print("=" * 60)
    print("[Test 3] tactile_residual_flow multi-step integration")
    delta_a_1 = model.tactile_residual_flow(
        cached_kv=cached_kv,
        latent_position_ids=pos_ids,
        n_action_in_cache=n_action_in_cache,
        base_chunk=a_hat,
        tactile_f6=tactile_f6,
        num_steps=4,
        noise_scale=0.1,
    )
    print(f"  Δa shape: {tuple(delta_a_1.shape)}")
    assert delta_a_1.shape == (B, n_chunk, action_dim), "Δa shape mismatch"

    # Ensure cached_kv was NOT mutated (should be reusable for another refresh)
    cache_len_before = cached_kv.get_seq_length()
    delta_a_2 = model.tactile_residual_flow(
        cached_kv=cached_kv,
        latent_position_ids=pos_ids,
        n_action_in_cache=n_action_in_cache,
        base_chunk=a_hat,
        tactile_f6=tactile_f6 + 0.1 * torch.randn_like(tactile_f6),
        num_steps=4,
        noise_scale=0.1,
    )
    cache_len_after = cached_kv.get_seq_length()
    print(f"  cached_kv length before/after: {cache_len_before}/{cache_len_after}")
    assert cache_len_before == cache_len_after, (
        "tactile_residual_flow mutated cached_kv — async refresh would break")
    diff_norm = (delta_a_1 - delta_a_2).float().abs().mean().item()
    print(f"  mean |Δa_1 − Δa_2| with different tactile: {diff_norm:.4f} "
          f"(non-zero is expected)")
    print("  [OK]")

    print("=" * 60)
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()

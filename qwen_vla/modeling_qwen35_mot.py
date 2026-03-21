"""
Qwen3.5-0.8B hybrid text backbone with Mixture-of-Transformers (MoT).

Architecture (24 decoder layers, layer_types list from config):
  - linear_attention layers (18/24): Qwen3_5GatedDeltaNet (SHARED) + per-expert MLP + norms
  - full_attention layers (6/24, every 4th): 3-expert GQA attention + per-expert MLP + norms

Qwen3.5 full-attention specifics honoured here:
  • q_proj outputs 2× the query size; the second half is the sigmoid gate applied to attn output.
  • per-head q_norm / k_norm (RMSNorm).
  • GQA: num_attention_heads=8, num_key_value_heads=2, head_dim=256.
  • Partial RoPE: rotary_dim = head_dim × partial_rotary_factor = 64 (first 64 dims of each head).
  • M-RoPE (mrope_section=[11,11,10], mrope_interleaved=True) – same mechanism as Qwen2-VL.
  • No sliding-window; always full causal attention in MoT.

Linear-attention specifics:
  • Qwen3_5GatedDeltaNet is imported from system transformers (>=4.57).
  • All three expert streams (latent / action / tactile) share a single DeltaNet per layer.
  • Per-expert split happens at the MLP + layer-norm level only.

Weight loading:
  Load Qwen3_5ForConditionalGeneration pretrained, then call
  Qwen35ModelMoT.from_pretrained_base(pretrained_text_model).
  Latent-expert weights match 1-to-1.  _action / _tactile weights are new,
  initialised from the latent expert in the training script.
"""

from __future__ import annotations

import math
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.cache_utils import DynamicCache

try:
    from transformers.models.qwen2_vl.modeling_qwen2_vl import (
        apply_multimodal_rotary_pos_emb as _apply_mrope,
    )
    _MROPE_AVAILABLE = True
except ImportError:
    _apply_mrope = None
    _MROPE_AVAILABLE = False

# qwen3_5 is called qwen3_next in transformers 4.57.x
# Import DeltaNet and RotaryEmb separately so a broken causal_conv1d/fla
# installation doesn't silently kill the rotary embedding import too.
try:
    from transformers.models.qwen3_next.modeling_qwen3_next import (
        Qwen3NextGatedDeltaNet as _DeltaNet,
    )
    _HAS_DELTANET = True
except Exception:
    _DeltaNet = None
    _HAS_DELTANET = False

try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding as _Q35RotaryEmb,
    )
    _HAS_QWEN35 = True
except Exception:
    _Q35RotaryEmb = None
    _HAS_QWEN35 = False

# ─── Prefix Cache for KV-cache-style reuse in flow-matching ──────────────────

class MoTPrefixCache:
    """Stores per-layer states from the latent prefix for reuse across denoise steps."""
    __slots__ = ['deltanet_recurrent', 'deltanet_conv', 'full_attn_kv']

    def __init__(self):
        self.deltanet_recurrent = {}   # layer_idx -> recurrent_state Tensor
        self.deltanet_conv = {}        # layer_idx -> conv_state Tensor [B, conv_dim, K]
        self.full_attn_kv = {}         # layer_idx -> (K_lat, V_lat) with RoPE applied


def _deltanet_forward_save(dn: nn.Module, hidden_states: torch.Tensor):
    """Run DeltaNet on prefix tokens. Returns (output, recurrent_state, conv_state)."""
    B, L, _ = hidden_states.shape
    proj_qkvz = dn.in_proj_qkvz(hidden_states)
    proj_ba = dn.in_proj_ba(hidden_states)
    q, k, v, z, b, a = dn.fix_query_key_value_ordering(proj_qkvz, proj_ba)
    q, k, v = (x.reshape(x.shape[0], x.shape[1], -1) for x in (q, k, v))
    mqkv = torch.cat((q, k, v), dim=-1).transpose(1, 2)       # [B, conv_dim, L]

    # Save conv state (last conv_kernel_size timesteps, pre-conv)
    K = dn.conv_kernel_size
    conv_state = F.pad(mqkv, (K - mqkv.shape[-1], 0))          # [B, conv_dim, K]

    # Conv1d
    if dn.causal_conv1d_fn is not None:
        mqkv_c = dn.causal_conv1d_fn(
            x=mqkv, weight=dn.conv1d.weight.squeeze(1),
            bias=dn.conv1d.bias, activation=dn.activation,
        )
    else:
        mqkv_c = F.silu(dn.conv1d(mqkv)[:, :, :L])
    mqkv_c = mqkv_c.transpose(1, 2)                            # [B, L, conv_dim]

    q2, k2, v2 = torch.split(mqkv_c, [dn.key_dim, dn.key_dim, dn.value_dim], dim=-1)
    q2 = q2.reshape(B, L, -1, dn.head_k_dim)
    k2 = k2.reshape(B, L, -1, dn.head_k_dim)
    v2 = v2.reshape(B, L, -1, dn.head_v_dim)

    beta = b.sigmoid()
    g = -dn.A_log.float().exp() * F.softplus(a.float() + dn.dt_bias)
    if dn.num_v_heads // dn.num_k_heads > 1:
        q2 = q2.repeat_interleave(dn.num_v_heads // dn.num_k_heads, dim=2)
        k2 = k2.repeat_interleave(dn.num_v_heads // dn.num_k_heads, dim=2)

    core, recurrent = dn.chunk_gated_delta_rule(
        q2, k2, v2, g=g, beta=beta,
        initial_state=None, output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    z_shape = z.shape
    core = dn.norm(core.reshape(-1, core.shape[-1]), z.reshape(-1, z.shape[-1]))
    core = core.reshape(z_shape).reshape(B, L, -1)
    return dn.out_proj(core), recurrent, conv_state


def _deltanet_forward_resume(dn: nn.Module, hidden_states: torch.Tensor,
                             recurrent_state: torch.Tensor,
                             conv_state: torch.Tensor):
    """Run DeltaNet on suffix tokens continuing from saved prefix states."""
    B, L, _ = hidden_states.shape
    proj_qkvz = dn.in_proj_qkvz(hidden_states)
    proj_ba = dn.in_proj_ba(hidden_states)
    q, k, v, z, b, a = dn.fix_query_key_value_ordering(proj_qkvz, proj_ba)
    q, k, v = (x.reshape(x.shape[0], x.shape[1], -1) for x in (q, k, v))
    mqkv = torch.cat((q, k, v), dim=-1).transpose(1, 2)       # [B, conv_dim, L]

    # Conv1d using saved conv_state as left context
    padded = torch.cat([conv_state, mqkv], dim=-1)              # [B, conv_dim, K + L]
    conv_out = F.conv1d(padded, dn.conv1d.weight, dn.conv1d.bias,
                        groups=dn.conv_dim, padding=0)          # [B, conv_dim, L + 1]
    mqkv_c = dn.act(conv_out[:, :, -L:]).transpose(1, 2)       # [B, L, conv_dim]

    q2, k2, v2 = torch.split(mqkv_c, [dn.key_dim, dn.key_dim, dn.value_dim], dim=-1)
    q2 = q2.reshape(B, L, -1, dn.head_k_dim)
    k2 = k2.reshape(B, L, -1, dn.head_k_dim)
    v2 = v2.reshape(B, L, -1, dn.head_v_dim)

    beta = b.sigmoid()
    g = -dn.A_log.float().exp() * F.softplus(a.float() + dn.dt_bias)
    if dn.num_v_heads // dn.num_k_heads > 1:
        q2 = q2.repeat_interleave(dn.num_v_heads // dn.num_k_heads, dim=2)
        k2 = k2.repeat_interleave(dn.num_v_heads // dn.num_k_heads, dim=2)

    core, _ = dn.chunk_gated_delta_rule(
        q2, k2, v2, g=g, beta=beta,
        initial_state=recurrent_state, output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

    z_shape = z.shape
    core = dn.norm(core.reshape(-1, core.shape[-1]), z.reshape(-1, z.shape[-1]))
    core = core.reshape(z_shape).reshape(B, L, -1)
    return dn.out_proj(core)


# ─── Building blocks ────────────────────────────────────────────────────────

class Q35RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        v = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(v + self.variance_epsilon)
        return self.weight * x.to(dtype)

class Q35MLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj   = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, S, D).reshape(B, H * n_rep, S, D)

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_partial_mrope(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
    mrope_section: List[int],
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply M-RoPE only to the first `rotary_dim` dims of each head."""
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    if _MROPE_AVAILABLE and _apply_mrope is not None and cos.ndim == 4:
        # cos is [3, B, L, d/2] — pre-interleaved format expected by apply_multimodal_rotary_pos_emb
        q_rot, k_rot = _apply_mrope(q_rot, k_rot, cos, sin, mrope_section)
    else:
        # cos is [B, L, d] — already interleaved (output of Qwen3VLTextRotaryEmbedding)
        # unsqueeze head dim so it broadcasts with [B, n_heads, L, d]
        cos_ = cos.unsqueeze(1)
        sin_ = sin.unsqueeze(1)
        q_rot = q_rot * cos_ + rotate_half(q_rot) * sin_
        k_rot = k_rot * cos_ + rotate_half(k_rot) * sin_

    return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)

def build_causal_mask(
    B: int, L: int,
    attention_mask: Optional[torch.Tensor],
    device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Build additive [B,1,L,L] causal mask honouring left-padding."""
    causal = torch.tril(torch.ones(L, L, device=device, dtype=torch.bool))
    if attention_mask is not None:
        # attention_mask covers latent positions; action/tactile are always 1
        L_lat = attention_mask.shape[1]
        L_extra = L - L_lat
        full_attn = torch.cat(
            [attention_mask,
             torch.ones(B, L_extra, device=device, dtype=attention_mask.dtype)],
            dim=1,
        )  # [B, L]
        key_mask = full_attn.bool().unsqueeze(1).unsqueeze(2)  # [B,1,1,L]
        allowed = causal.unsqueeze(0).unsqueeze(0) & key_mask  # [B,1,L,L]
    else:
        allowed = causal.unsqueeze(0).unsqueeze(0).expand(B, 1, L, L)

    return torch.where(allowed, torch.zeros(B, 1, L, L, device=device, dtype=dtype),
                       torch.full((B, 1, L, L), float("-inf"), device=device, dtype=dtype))

class Qwen35FullAttentionMoT(nn.Module):
    """
    Three-expert GQA attention for full_attention layers.
    Gate is extracted from the doubled q_proj output (Qwen3.5 style).
    """
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim
        H = config.hidden_size
        n_q, n_kv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        self.scaling = hd ** -0.5

        rope_params = getattr(config, "rope_parameters", {})
        if isinstance(rope_params, dict):
            prf = rope_params.get("partial_rotary_factor", 1.0)
            self.mrope_section = rope_params.get("mrope_section", None)
        else:
            prf = getattr(rope_params, "partial_rotary_factor", 1.0)
            self.mrope_section = getattr(rope_params, "mrope_section", None)
        self.rotary_dim = int(hd * prf)

        bias = getattr(config, "attention_bias", False)

        # Latent expert
        self.q_proj   = nn.Linear(H, n_q * hd * 2, bias=bias)   # 2× for gate
        self.k_proj   = nn.Linear(H, n_kv * hd,    bias=bias)
        self.v_proj   = nn.Linear(H, n_kv * hd,    bias=bias)
        self.o_proj   = nn.Linear(n_q * hd, H,      bias=bias)
        self.q_norm   = Q35RMSNorm(hd, eps=config.rms_norm_eps)
        self.k_norm   = Q35RMSNorm(hd, eps=config.rms_norm_eps)

        # Action expert
        self.q_proj_action = nn.Linear(H, n_q * hd * 2, bias=bias)
        self.k_proj_action = nn.Linear(H, n_kv * hd,    bias=bias)
        self.v_proj_action = nn.Linear(H, n_kv * hd,    bias=bias)
        self.o_proj_action = nn.Linear(n_q * hd, H,      bias=bias)
        self.q_norm_action = Q35RMSNorm(hd, eps=config.rms_norm_eps)
        self.k_norm_action = Q35RMSNorm(hd, eps=config.rms_norm_eps)

        # Tactile expert
        self.q_proj_tactile = nn.Linear(H, n_q * hd * 2, bias=bias)
        self.k_proj_tactile = nn.Linear(H, n_kv * hd,    bias=bias)
        self.v_proj_tactile = nn.Linear(H, n_kv * hd,    bias=bias)
        self.o_proj_tactile = nn.Linear(n_q * hd, H,      bias=bias)
        self.q_norm_tactile = Q35RMSNorm(hd, eps=config.rms_norm_eps)
        self.k_norm_tactile = Q35RMSNorm(hd, eps=config.rms_norm_eps)

    def _proj_qkv(self, h, q_proj, k_proj, v_proj, q_norm, k_norm):
        """Project one expert stream → (q, k, v, gate)."""
        B, S, _ = h.shape
        n_q, n_kv, hd = self.num_heads, self.num_kv_heads, self.head_dim

        qg = q_proj(h).view(B, S, n_q, hd * 2)
        q, gate = qg[..., :hd], qg[..., hd:]                     # each [B,S,n_q,hd]
        gate = gate.reshape(B, S, n_q * hd)                       # [B, S, n_q*hd]
        q = q_norm(q).transpose(1, 2)                             # [B, n_q, S, hd]
        k = k_norm(self.k_proj(h).view(B, S, n_kv, hd) if k_proj is self.k_proj
                   else k_proj(h).view(B, S, n_kv, hd)).transpose(1, 2)
        v = v_proj(h).view(B, S, n_kv, hd).transpose(1, 2)
        return q, k, v, gate

    def _proj_qkv_expert(self, h, qp, kp, vp, qn, kn):
        B, S, _ = h.shape
        n_q, n_kv, hd = self.num_heads, self.num_kv_heads, self.head_dim
        qg = qp(h).view(B, S, n_q, hd * 2)
        q, gate = qg[..., :hd], qg[..., hd:]
        gate = gate.reshape(B, S, n_q * hd)
        q = qn(q).transpose(1, 2)
        k = kn(kp(h).view(B, S, n_kv, hd)).transpose(1, 2)
        v = vp(h).view(B, S, n_kv, hd).transpose(1, 2)
        return q, k, v, gate

    def forward(
        self,
        lat: torch.Tensor,
        act: torch.Tensor,
        tac: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        causal_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = lat.shape[0]
        S_lat, S_act, S_tac = lat.shape[1], act.shape[1], tac.shape[1]

        q_l, k_l, v_l, gate_l = self._proj_qkv_expert(lat, self.q_proj,   self.k_proj,   self.v_proj,   self.q_norm,   self.k_norm)
        q_a, k_a, v_a, gate_a = self._proj_qkv_expert(act, self.q_proj_action, self.k_proj_action, self.v_proj_action, self.q_norm_action, self.k_norm_action)
        q_t, k_t, v_t, gate_t = self._proj_qkv_expert(tac, self.q_proj_tactile, self.k_proj_tactile, self.v_proj_tactile, self.q_norm_tactile, self.k_norm_tactile)

        # Joint sequence
        q_all = torch.cat([q_l, q_a, q_t], dim=2)   # [B, n_q, S_total, hd]
        k_all = torch.cat([k_l, k_a, k_t], dim=2)   # [B, n_kv, S_total, hd]
        v_all = torch.cat([v_l, v_a, v_t], dim=2)

        # Partial M-RoPE
        if position_embeddings is not None and self.mrope_section is not None:
            cos, sin = position_embeddings
            q_all, k_all = apply_partial_mrope(q_all, k_all, cos, sin,
                                               self.mrope_section, self.rotary_dim)

        # GQA expand
        k_all = repeat_kv(k_all, self.num_kv_groups)
        v_all = repeat_kv(v_all, self.num_kv_groups)

        # Scaled dot-product attention
        attn_out = F.scaled_dot_product_attention(
            q_all, k_all, v_all,
            attn_mask=causal_mask,
            dropout_p=0.0,
            scale=self.scaling,
        )  # [B, n_q, S_total, hd]

        # Split output
        def _out(expert_slice, gate, o_proj):
            x = expert_slice.transpose(1, 2).reshape(B, expert_slice.shape[2], -1)
            x = x * torch.sigmoid(gate)
            return o_proj(x)

        attn_l = _out(attn_out[:, :, :S_lat],             gate_l, self.o_proj)
        attn_a = _out(attn_out[:, :, S_lat:S_lat+S_act],  gate_a, self.o_proj_action)
        attn_t = _out(attn_out[:, :, S_lat+S_act:],        gate_t, self.o_proj_tactile)
        return attn_l, attn_a, attn_t

class Qwen35LinearLayerMoT(nn.Module):
    """
    linear_attention layer: shared DeltaNet + per-expert MLP + norms.
    All three expert streams are processed jointly through DeltaNet.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        H, I = config.hidden_size, config.intermediate_size
        eps = config.rms_norm_eps

        if not _HAS_DELTANET or _DeltaNet is None:
            raise ImportError(
                "Qwen3NextGatedDeltaNet could not be imported. "
                "Check that transformers>=4.57 is installed and causal_conv1d "
                "is either absent or built for the correct PyTorch/CUDA version."
            )
        self.linear_attn = _DeltaNet(config, layer_idx)
        self.input_layernorm = Q35RMSNorm(H, eps)

        # Latent expert (weights will be loaded from base model)
        self.post_attention_layernorm = Q35RMSNorm(H, eps)
        self.mlp = Q35MLP(H, I)

        # Action expert
        self.post_attention_layernorm_action = Q35RMSNorm(H, eps)
        self.mlp_action = Q35MLP(H, I)

        # Tactile expert
        self.post_attention_layernorm_tactile = Q35RMSNorm(H, eps)
        self.mlp_tactile = Q35MLP(H, I)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, L_total, H]  (latent | action | tactile)
        L_lat: int, L_act: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, L_total, H = hidden_states.shape

        # Shared: input norm + DeltaNet on full sequence
        residual = hidden_states
        x = self.input_layernorm(hidden_states)
        # Extend attention_mask to cover action/tactile (always attended)
        if attention_mask is not None:
            L_extra = L_total - attention_mask.shape[1]
            lin_mask = torch.cat(
                [attention_mask,
                 torch.ones(B, L_extra, device=attention_mask.device,
                            dtype=attention_mask.dtype)], dim=1
            )
        else:
            lin_mask = None
        x = self.linear_attn(x, attention_mask=lin_mask)
        hidden = residual + x

        # Per-expert MLP (split along sequence)
        lat = hidden[:, :L_lat]
        act = hidden[:, L_lat:L_lat + L_act]
        tac = hidden[:, L_lat + L_act:]

        lat = lat + self.mlp(self.post_attention_layernorm(lat))
        act = act + self.mlp_action(self.post_attention_layernorm_action(act))
        tac = tac + self.mlp_tactile(self.post_attention_layernorm_tactile(tac))

        return torch.cat([lat, act, tac], dim=1)


class Qwen35FullLayerMoT(nn.Module):
    """
    full_attention layer: 3-expert GQA attention + per-expert MLP + norms.
    """
    def __init__(self, config, layer_idx: int):
        super().__init__()
        H, I = config.hidden_size, config.intermediate_size
        eps = config.rms_norm_eps

        self.self_attn = Qwen35FullAttentionMoT(config)

        # Latent expert (names match base model for weight loading)
        self.input_layernorm = Q35RMSNorm(H, eps)
        self.post_attention_layernorm = Q35RMSNorm(H, eps)
        self.mlp = Q35MLP(H, I)

        # Action expert
        self.input_layernorm_action = Q35RMSNorm(H, eps)
        self.post_attention_layernorm_action = Q35RMSNorm(H, eps)
        self.mlp_action = Q35MLP(H, I)

        # Tactile expert
        self.input_layernorm_tactile = Q35RMSNorm(H, eps)
        self.post_attention_layernorm_tactile = Q35RMSNorm(H, eps)
        self.mlp_tactile = Q35MLP(H, I)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, L_total, H]
        L_lat: int, L_act: int,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]],
        causal_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        lat_h = hidden_states[:, :L_lat]
        act_h = hidden_states[:, L_lat:L_lat + L_act]
        tac_h = hidden_states[:, L_lat + L_act:]

        # Per-expert input norm
        lat = self.input_layernorm(lat_h)
        act = self.input_layernorm_action(act_h)
        tac = self.input_layernorm_tactile(tac_h)

        # Joint MoT attention
        attn_l, attn_a, attn_t = self.self_attn(lat, act, tac, position_embeddings, causal_mask)

        # Residual + per-expert post-norm + MLP
        lat_out = lat_h + attn_l
        act_out = act_h + attn_a
        tac_out = tac_h + attn_t

        lat_out = lat_out + self.mlp(self.post_attention_layernorm(lat_out))
        act_out = act_out + self.mlp_action(self.post_attention_layernorm_action(act_out))
        tac_out = tac_out + self.mlp_tactile(self.post_attention_layernorm_tactile(tac_out))

        return torch.cat([lat_out, act_out, tac_out], dim=1)

class Qwen35ModelMoT(nn.Module):
    """
    Hybrid Qwen3.5 text backbone with Mixture-of-Transformers.
    Accepts a combined [latent | action | tactile] input_embeds tensor.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        H, eps = config.hidden_size, config.rms_norm_eps

        self.embed_tokens = nn.Embedding(config.vocab_size, H,
                                          getattr(config, "pad_token_id", None))

        if not _HAS_QWEN35 or _Q35RotaryEmb is None:
            raise ImportError("transformers >= 4.57 is required for Qwen3.5 rotary embeddings.")
        # Qwen3VLTextRotaryEmbedding expects config.rope_scaling with mrope_section
        # and config.rope_theta. Build a minimal wrapper from config.rope_parameters.
        rope_params = getattr(config, "rope_parameters", {})
        if not isinstance(rope_params, dict):
            rope_params = {}
        _rope_cfg = type("_RopeCfg", (), {
            "hidden_size":            config.hidden_size,
            "num_attention_heads":    config.num_attention_heads,
            "head_dim":               config.head_dim,
            "max_position_embeddings": config.max_position_embeddings,
            "rope_theta":             rope_params.get("rope_theta", 10_000_000),
            "rope_scaling": {
                "rope_type":    rope_params.get("rope_type", "default"),
                "mrope_section": rope_params.get("mrope_section", [11, 11, 10]),
            },
            "partial_rotary_factor":  rope_params.get("partial_rotary_factor", 0.25),
        })()
        self.rotary_emb = _Q35RotaryEmb(config=_rope_cfg)

        layer_types = getattr(config, "layer_types",
                              ["full_attention" if (i + 1) % 4 == 0 else "linear_attention"
                               for i in range(config.num_hidden_layers)])

        self.layers = nn.ModuleList()
        for idx in range(config.num_hidden_layers):
            lt = layer_types[idx]
            if lt == "full_attention":
                self.layers.append(Qwen35FullLayerMoT(config, idx))
            else:
                self.layers.append(Qwen35LinearLayerMoT(config, idx))

        self.norm          = Q35RMSNorm(H, eps)
        self.norm_action   = Q35RMSNorm(H, eps)
        self.norm_tactile  = Q35RMSNorm(H, eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    @staticmethod
    def _extend_position_ids(
        latent_mrope_ids: torch.Tensor,   # [3, B, L_latent]
        n_action: int,
        n_tactile: int,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extend M-RoPE position_ids to cover action and tactile tokens.
        Action/tactile get sequential positions (all 3 dims equal) after max latent pos.
        """
        device = latent_mrope_ids.device
        _, B, _ = latent_mrope_ids.shape
        n_extra = n_action + n_tactile

        if attention_mask is not None:
            # max position per batch element (ignore padding at front)
            mask = attention_mask.bool()                       # [B, L_latent]
            masked = latent_mrope_ids * mask.unsqueeze(0)      # [3, B, L_latent]
            max_pos = masked.max(dim=-1).values.max(dim=0).values  # [B]
        else:
            max_pos = latent_mrope_ids.max(dim=-1).values.max(dim=0).values  # [B]

        offset = torch.arange(1, n_extra + 1, device=device).unsqueeze(0)  # [1, n_extra]
        extra = (max_pos.unsqueeze(-1) + offset).unsqueeze(0).expand(3, -1, -1)  # [3, B, n_extra]
        return torch.cat([latent_mrope_ids, extra], dim=-1)   # [3, B, L_total]

    def forward(
        self,
        inputs_embeds:   torch.Tensor,                  # [B, L_total, H]
        position_ids:    Optional[torch.Tensor] = None, # [3, B, L_total] M-RoPE
        attention_mask:  Optional[torch.Tensor] = None, # [B, L_latent]  (latent padding mask)
        latent_indexes:  Optional[torch.Tensor] = None, # [L_lat]
        action_indexes:  Optional[torch.Tensor] = None, # [L_act]
        tactile_indexes: Optional[torch.Tensor] = None, # [L_tac]
        use_cache:       bool = False,
        return_dict:     bool = True,
    ) -> BaseModelOutputWithPast:

        B, L_total, H = inputs_embeds.shape
        device, dtype = inputs_embeds.device, inputs_embeds.dtype

        # Determine sequence split sizes
        if latent_indexes is not None:
            L_lat = latent_indexes.shape[0]
            L_act = action_indexes.shape[0] if action_indexes is not None else 0
        else:
            L_lat, L_act = L_total, 0

        # M-RoPE position embeddings (computed once for full sequence)
        if position_ids is not None:
            # position_ids: [3, B, L_total]  (mrope dims: temporal, height, width)
            position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        else:
            # Fallback: 1-D sequential
            pos = torch.arange(L_total, device=device).view(1, 1, -1).expand(3, B, -1)
            position_embeddings = self.rotary_emb(inputs_embeds, pos)

        # Causal mask for full_attention layers
        causal_mask = build_causal_mask(B, L_total, attention_mask, device, dtype)

        hidden_states = inputs_embeds

        for layer in self.layers:
            if isinstance(layer, Qwen35LinearLayerMoT):
                hidden_states = layer(hidden_states, L_lat, L_act, attention_mask)
            else:  # Qwen35FullLayerMoT
                hidden_states = layer(hidden_states, L_lat, L_act,
                                      position_embeddings, causal_mask)

        # Per-expert final norms
        lat_out = self.norm(hidden_states[:, :L_lat])
        act_out = self.norm_action(hidden_states[:, L_lat:L_lat + L_act])
        tac_out = self.norm_tactile(hidden_states[:, L_lat + L_act:])
        last_hidden = torch.cat([lat_out, act_out, tac_out], dim=1)

        if return_dict:
            return BaseModelOutputWithPast(last_hidden_state=last_hidden)
        return (last_hidden,)

    # ── KV-cache-style prefix / suffix forward for flow-matching ─────────

    @torch.no_grad()
    def forward_prefix(
        self,
        inputs_embeds:  torch.Tensor,           # [B, L_lat, H]  (latent only)
        position_ids:   torch.Tensor,            # [3, B, L_lat]
        attention_mask:  Optional[torch.Tensor] = None,
    ) -> MoTPrefixCache:
        """Process latent tokens through all layers and cache per-layer states."""
        B, L, H = inputs_embeds.shape
        device, dtype = inputs_embeds.device, inputs_embeds.dtype
        cache = MoTPrefixCache()

        # Position embeddings for the latent prefix
        prefix_pos_emb = self.rotary_emb(inputs_embeds, position_ids)
        prefix_mask = build_causal_mask(B, L, attention_mask, device, dtype)

        hidden = inputs_embeds
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, Qwen35LinearLayerMoT):
                # Shared input norm
                residual = hidden
                x = layer.input_layernorm(hidden)
                # Extend attention mask for DeltaNet (no extra tokens in prefix)
                if attention_mask is not None:
                    lin_mask = attention_mask
                else:
                    lin_mask = None
                # DeltaNet forward with state saving
                dn_out, rec_state, conv_state = _deltanet_forward_save(layer.linear_attn, x)
                cache.deltanet_recurrent[idx] = rec_state
                cache.deltanet_conv[idx] = conv_state
                hidden = residual + dn_out
                # Per-expert MLP (only latent in prefix)
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
            else:
                # Full attention layer — process latent only, save K,V
                attn = layer.self_attn
                lat = layer.input_layernorm(hidden)
                q_l, k_l, v_l, gate_l = attn._proj_qkv_expert(
                    lat, attn.q_proj, attn.k_proj, attn.v_proj,
                    attn.q_norm, attn.k_norm)

                # Apply partial M-RoPE
                if prefix_pos_emb is not None and attn.mrope_section is not None:
                    cos, sin = prefix_pos_emb
                    q_l, k_l = apply_partial_mrope(
                        q_l, k_l, cos, sin, attn.mrope_section, attn.rotary_dim)

                # Save K,V with RoPE for later suffix cross-attention
                cache.full_attn_kv[idx] = (k_l, v_l)

                # Self-attention among latent tokens
                k_exp = repeat_kv(k_l, attn.num_kv_groups)
                v_exp = repeat_kv(v_l, attn.num_kv_groups)
                attn_out = F.scaled_dot_product_attention(
                    q_l, k_exp, v_exp, attn_mask=prefix_mask,
                    dropout_p=0.0, scale=attn.scaling)
                out = attn_out.transpose(1, 2).reshape(B, L, -1)
                out = out * torch.sigmoid(gate_l)
                out = attn.o_proj(out)

                hidden = hidden + out
                hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))

        return cache

    def forward_suffix(
        self,
        suffix_embeds:   torch.Tensor,           # [B, L_suffix, H]  (action+tactile)
        prefix_cache:    MoTPrefixCache,
        suffix_pos_emb:  Tuple[torch.Tensor, torch.Tensor],  # (cos, sin) for suffix
        attention_mask:  Optional[torch.Tensor] = None,       # [B, L_lat] padding mask
        L_act:           int = 0,
        L_prefix:        int = 0,
    ) -> torch.Tensor:
        """Process action/tactile suffix using cached prefix states.
        Returns hidden states for suffix only: [B, L_suffix, H]."""
        B, L_suf, H = suffix_embeds.shape
        device, dtype = suffix_embeds.device, suffix_embeds.dtype

        # Causal mask: suffix Q can attend to prefix K (all allowed) + suffix K (causal)
        L_total = L_prefix + L_suf
        # Build [B, 1, L_suf, L_total] mask
        # Suffix-to-prefix: fully visible (respect prefix padding)
        # Suffix-to-suffix: causal
        causal_suf = torch.tril(torch.ones(L_suf, L_suf, device=device, dtype=torch.bool))
        if attention_mask is not None:
            prefix_vis = attention_mask.bool()  # [B, L_prefix]
        else:
            prefix_vis = torch.ones(B, L_prefix, device=device, dtype=torch.bool)
        # [B, L_suf, L_prefix]: each suffix token sees all non-padded prefix tokens
        cross_vis = prefix_vis.unsqueeze(1).expand(B, L_suf, L_prefix)
        # [B, L_suf, L_total]
        full_vis = torch.cat([cross_vis, causal_suf.unsqueeze(0).expand(B, -1, -1)], dim=-1)
        suffix_mask = torch.where(
            full_vis, torch.zeros(1, device=device, dtype=dtype),
            torch.full((1,), float("-inf"), device=device, dtype=dtype),
        ).unsqueeze(1)  # [B, 1, L_suf, L_total]

        hidden = suffix_embeds
        for idx, layer in enumerate(self.layers):
            if isinstance(layer, Qwen35LinearLayerMoT):
                residual = hidden
                x = layer.input_layernorm(hidden)
                dn_out = _deltanet_forward_resume(
                    layer.linear_attn, x,
                    prefix_cache.deltanet_recurrent[idx],
                    prefix_cache.deltanet_conv[idx])
                hidden = residual + dn_out
                # Per-expert MLP
                act_h = hidden[:, :L_act]
                tac_h = hidden[:, L_act:]
                act_h = act_h + layer.mlp_action(layer.post_attention_layernorm_action(act_h))
                tac_h = tac_h + layer.mlp_tactile(layer.post_attention_layernorm_tactile(tac_h))
                hidden = torch.cat([act_h, tac_h], dim=1)
            else:
                attn = layer.self_attn
                act_h = hidden[:, :L_act]
                tac_h = hidden[:, L_act:]

                # Per-expert input norms
                act_n = layer.input_layernorm_action(act_h)
                tac_n = layer.input_layernorm_tactile(tac_h)

                # Q, K, V for suffix experts
                q_a, k_a, v_a, gate_a = attn._proj_qkv_expert(
                    act_n, attn.q_proj_action, attn.k_proj_action,
                    attn.v_proj_action, attn.q_norm_action, attn.k_norm_action)
                q_t, k_t, v_t, gate_t = attn._proj_qkv_expert(
                    tac_n, attn.q_proj_tactile, attn.k_proj_tactile,
                    attn.v_proj_tactile, attn.q_norm_tactile, attn.k_norm_tactile)

                # Suffix Q/K
                q_suf = torch.cat([q_a, q_t], dim=2)              # [B, n_q, L_suf, hd]
                k_suf = torch.cat([k_a, k_t], dim=2)              # [B, n_kv, L_suf, hd]
                v_suf = torch.cat([v_a, v_t], dim=2)

                # Apply RoPE to suffix Q, K
                if suffix_pos_emb is not None and attn.mrope_section is not None:
                    cos, sin = suffix_pos_emb
                    q_suf, k_suf = apply_partial_mrope(
                        q_suf, k_suf, cos, sin, attn.mrope_section, attn.rotary_dim)

                # Concat cached prefix K,V (already has RoPE) with suffix K,V
                k_lat, v_lat = prefix_cache.full_attn_kv[idx]
                k_all = torch.cat([k_lat, k_suf], dim=2)          # [B, n_kv, L_total, hd]
                v_all = torch.cat([v_lat, v_suf], dim=2)

                # GQA expand
                k_all = repeat_kv(k_all, attn.num_kv_groups)
                v_all = repeat_kv(v_all, attn.num_kv_groups)

                # Attention: suffix Q attends to all K,V
                attn_out = F.scaled_dot_product_attention(
                    q_suf, k_all, v_all, attn_mask=suffix_mask,
                    dropout_p=0.0, scale=attn.scaling)             # [B, n_q, L_suf, hd]

                # Split output back to action / tactile
                S_act, S_tac = act_n.shape[1], tac_n.shape[1]
                def _out(sl, gate, o_proj):
                    x = sl.transpose(1, 2).reshape(B, sl.shape[2], -1)
                    return o_proj(x * torch.sigmoid(gate))

                out_a = _out(attn_out[:, :, :S_act], gate_a, attn.o_proj_action)
                out_t = _out(attn_out[:, :, S_act:], gate_t, attn.o_proj_tactile)

                # Residual + MLP
                act_h = act_h + out_a
                tac_h = tac_h + out_t
                act_h = act_h + layer.mlp_action(layer.post_attention_layernorm_action(act_h))
                tac_h = tac_h + layer.mlp_tactile(layer.post_attention_layernorm_tactile(tac_h))
                hidden = torch.cat([act_h, tac_h], dim=1)

        # Per-expert final norms (no latent norm — prefix was already normed if needed)
        act_out = self.norm_action(hidden[:, :L_act])
        tac_out = self.norm_tactile(hidden[:, L_act:])
        return torch.cat([act_out, tac_out], dim=1)

    @classmethod
    def from_pretrained_base(cls, base_text_model) -> "Qwen35ModelMoT":
        config = base_text_model.config
        mot = cls(config)
        missing, unexpected = mot.load_state_dict(
            base_text_model.state_dict(), strict=False
        )
        print(f"[Qwen35ModelMoT] loaded - missing {len(missing)} (new expert weights), "
              f"unexpected {len(unexpected)}")
        return mot

    @classmethod
    def from_pretrained_weights(
        cls,
        config,
        state_dict: dict,
        torch_dtype=torch.bfloat16,
    ) -> "Qwen35ModelMoT":
        """Load from a pre-extracted state_dict (keys relative to language_model root)."""
        mot = cls(config)
        missing, unexpected = mot.load_state_dict(state_dict, strict=False)
        print(f"[Qwen35ModelMoT] from_pretrained_weights: "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        return mot.to(torch_dtype)


"""
Qwen3-VL text backbone with Mixture-of-Transformers (MoT).

Three expert streams run in every decoder layer:
  • latent   – processes vision / language tokens (the "reason" expert)
  • action   – processes diffusion noisy-action tokens
  • tactile  – processes tactile force / deformation tokens

Key Qwen3 specifics honoured here:
  • Per-head q_norm / k_norm (RMSNorm on each head dimension).
  • Sliding-window alternating layers **disabled** in MoT – all layers use
    full causal attention so that action/tactile tokens can always attend to
    the full visual-language context.
  • M-RoPE (3-D rotary position embeddings) imported from the installed
    `transformers` package (Qwen3-VL / Qwen2-VL).  Action and tactile tokens
    are assigned sequential 1-D positions (all three M-RoPE dims equal) that
    follow the maximum position of the latent sequence.

Weight loading:
  Load `Qwen3VLForConditionalGeneration` pretrained weights with
  `strict=False`.  The base expert weights map 1-to-1; the new
  `_action` / `_tactile` weights are missing from the checkpoint
  and will be initialised separately in the training script
  (either randomly or copied from the base expert).
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast

# ─── M-RoPE helpers ──────────────────────────────────────────────────────────
# In transformers >=4.57, Qwen3-VL uses:
#   - Qwen3VLTextRotaryEmbedding: handles interleaved M-RoPE internally,
#     accepts position_ids [3, B, L], returns cos/sin [B, L, head_dim].
#   - apply_rotary_pos_emb: standard RoPE application (NOT multimodal),
#     because the 3-D → interleaved merging is done inside the RoPE forward.
_VLRotaryEmbedding = None
_apply_rope_fn = None

# Attempt 1: Qwen3-VL (transformers >= 4.57)
try:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextRotaryEmbedding as _VLRotaryEmbedding,
        apply_rotary_pos_emb as _apply_rope_fn,
    )
except ImportError:
    pass

# Attempt 2: Qwen2-VL fallback (older transformers)
if _VLRotaryEmbedding is None:
    try:
        from transformers.models.qwen2_vl.modeling_qwen2_vl import (
            Qwen2VLRotaryEmbedding as _VLRotaryEmbedding,
            apply_multimodal_rotary_pos_emb as _apply_rope_fn,
        )
    except ImportError:
        pass


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_1d(q, k, cos, sin, unsqueeze_dim: int = 1):
    """Standard 1-D RoPE fallback."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ─── Building blocks ─────────────────────────────────────────────────────────

class Qwen3VLRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class Qwen3VLMLP(nn.Module):
    def __init__(self, config, intermediate_size: int = None):
        super().__init__()
        H = config.hidden_size
        I = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.gate_proj = nn.Linear(H, I, bias=False)
        self.up_proj   = nn.Linear(H, I, bias=False)
        self.down_proj = nn.Linear(I, H, bias=False)
        self.act_fn    = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    key_s   = repeat_kv(key,   module.num_key_value_groups)
    value_s = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_s.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_s.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout > 0.0 and module.training:
        attn_weights = nn.functional.dropout(attn_weights, p=dropout)
    attn_output = torch.matmul(attn_weights, value_s)
    return attn_output.transpose(1, 2).contiguous(), attn_weights


# ─── MoT Attention ───────────────────────────────────────────────────────────

class Qwen3VLAttentionMoT(nn.Module):
    """
    Multi-head attention with three parallel expert sets:
    • base  (latent / reason tokens)
    • action
    • tactile

    All tokens attend each other jointly (shared KV space, causal mask).
    Each modality uses its own Q/K/V/O projections and per-head norms.
    """

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        H = config.hidden_size
        A = config.attention_bias

        # ── latent expert ───────────────────────────────────────────────────
        self.q_proj   = nn.Linear(H, self.num_heads    * self.head_dim, bias=A)
        self.k_proj   = nn.Linear(H, self.num_kv_heads * self.head_dim, bias=A)
        self.v_proj   = nn.Linear(H, self.num_kv_heads * self.head_dim, bias=A)
        self.o_proj   = nn.Linear(self.num_heads * self.head_dim, H, bias=A)
        self.q_norm   = Qwen3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm   = Qwen3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # ── action expert ───────────────────────────────────────────────────
        self.q_proj_action  = nn.Linear(H, self.num_heads    * self.head_dim, bias=A)
        self.k_proj_action  = nn.Linear(H, self.num_kv_heads * self.head_dim, bias=A)
        self.v_proj_action  = nn.Linear(H, self.num_kv_heads * self.head_dim, bias=A)
        self.o_proj_action  = nn.Linear(self.num_heads * self.head_dim, H, bias=A)
        self.q_norm_action  = Qwen3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_action  = Qwen3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # ── tactile expert ──────────────────────────────────────────────────
        self.q_proj_tactile = nn.Linear(H, self.num_heads    * self.head_dim, bias=A)
        self.k_proj_tactile = nn.Linear(H, self.num_kv_heads * self.head_dim, bias=A)
        self.v_proj_tactile = nn.Linear(H, self.num_kv_heads * self.head_dim, bias=A)
        self.o_proj_tactile = nn.Linear(self.num_heads * self.head_dim, H, bias=A)
        self.q_norm_tactile = Qwen3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm_tactile = Qwen3VLRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def _proj_qkv(self, hidden, q_proj, k_proj, v_proj, q_norm, k_norm):
        """Project a slice of hidden states through one expert's Q/K/V heads."""
        B, S, _ = hidden.shape
        q = q_norm(q_proj(hidden).view(B, S, self.num_heads,    self.head_dim)).transpose(1, 2)
        k = k_norm(k_proj(hidden).view(B, S, self.num_kv_heads, self.head_dim)).transpose(1, 2)
        v =        v_proj(hidden).view(B, S, self.num_kv_heads, self.head_dim) .transpose(1, 2)
        return q, k, v

    def forward(
        self,
        hidden_states:      torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask:     Optional[torch.Tensor],
        past_key_value:     Optional[Cache] = None,
        cache_position:     Optional[torch.LongTensor] = None,
        latent_indexes:     Optional[torch.LongTensor] = None,
        action_indexes:     Optional[torch.LongTensor] = None,
        tactile_indexes:    Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        input_shape = hidden_states.shape[:-1]

        lat_h = hidden_states[:, latent_indexes]   if len(latent_indexes)  > 0 else hidden_states[:, :0]
        act_h = hidden_states[:, action_indexes]   if len(action_indexes)  > 0 else hidden_states[:, :0]
        tac_h = hidden_states[:, tactile_indexes]  if len(tactile_indexes) > 0 else hidden_states[:, :0]

        # Per-expert projections
        lat_q, lat_k, lat_v = self._proj_qkv(lat_h, self.q_proj,   self.k_proj,   self.v_proj,   self.q_norm,   self.k_norm)
        act_q, act_k, act_v = self._proj_qkv(act_h, self.q_proj_action,  self.k_proj_action,  self.v_proj_action,  self.q_norm_action,  self.k_norm_action)
        tac_q, tac_k, tac_v = self._proj_qkv(tac_h, self.q_proj_tactile, self.k_proj_tactile, self.v_proj_tactile, self.q_norm_tactile, self.k_norm_tactile)

        # Concatenate across sequence dim for joint attention [B, heads, total_seq, head_dim]
        query_states = torch.cat([lat_q, act_q, tac_q], dim=2)
        key_states   = torch.cat([lat_k, act_k, tac_k], dim=2)
        value_states = torch.cat([lat_v, act_v, tac_v], dim=2)

        # Apply position embeddings.
        # Qwen3VLTextRotaryEmbedding already handles interleaved M-RoPE
        # internally, so cos/sin are [B, L, head_dim].  We use the standard
        # apply_rotary_pos_emb (or 1-D fallback).
        cos, sin = position_embeddings
        if _apply_rope_fn is not None:
            query_states, key_states = _apply_rope_fn(
                query_states, key_states, cos, sin
            )
        else:
            query_states, key_states = apply_rotary_pos_emb_1d(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # Use F.scaled_dot_product_attention (flash / memory-efficient backend)
        # for better numerical stability in bf16 vs manual matmul+softmax.
        key_s   = repeat_kv(key_states,   self.num_key_value_groups)
        value_s = repeat_kv(value_states, self.num_key_value_groups)
        causal_mask = attention_mask[:, :, :, :key_s.shape[-2]] if attention_mask is not None else None
        attn_output = nn.functional.scaled_dot_product_attention(
            query_states, key_s, value_s,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            scale=self.scaling,
        )
        attn_weights = None

        # Reshape back to [B, total_seq, hidden]
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()

        # Per-expert output projections
        lat_out = self.o_proj(attn_output[:, latent_indexes])   if len(latent_indexes)  > 0 else attn_output[:, :0]
        act_out = self.o_proj_action(attn_output[:, action_indexes])  if len(action_indexes)  > 0 else attn_output[:, :0]
        tac_out = self.o_proj_tactile(attn_output[:, tactile_indexes]) if len(tactile_indexes) > 0 else attn_output[:, :0]

        # Re-assemble in original order [latent | action | tactile]
        out = torch.cat([lat_out, act_out, tac_out], dim=1)
        return out, attn_weights


# ─── MoT Decoder Layer ───────────────────────────────────────────────────────

class Qwen3VLDecoderLayerMoT(nn.Module):
    """
    Qwen3-VL decoder layer with three independent MLP experts.
    Attention uses joint computation across all three token streams.
    Sliding-window is disabled (all layers use full causal attention).
    """

    def __init__(self, config, layer_idx: int, tactile_intermediate_size: int = None):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn   = Qwen3VLAttentionMoT(config, layer_idx)

        # ── latent expert ──
        self.mlp                       = Qwen3VLMLP(config)
        self.input_layernorm           = Qwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm  = Qwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ── action expert ──
        self.mlp_action                       = Qwen3VLMLP(config)
        self.input_layernorm_action           = Qwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_action  = Qwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ── tactile expert (optionally smaller MLP) ──
        self.mlp_tactile                       = Qwen3VLMLP(config, intermediate_size=tactile_intermediate_size)
        self.input_layernorm_tactile           = Qwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm_tactile  = Qwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states:      torch.Tensor,
        attention_mask:     Optional[torch.Tensor] = None,
        position_ids:       Optional[torch.LongTensor] = None,
        past_key_value:     Optional[Cache] = None,
        output_attentions:  bool = False,
        use_cache:          bool = False,
        cache_position:     Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        latent_indexes:     Optional[torch.LongTensor] = None,
        action_indexes:     Optional[torch.LongTensor] = None,
        tactile_indexes:    Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Tuple:
        residual = hidden_states

        lat_h = hidden_states[:, latent_indexes]
        act_h = hidden_states[:, action_indexes]
        tac_h = hidden_states[:, tactile_indexes]

        # Per-expert pre-attention layer norm, then re-assemble for joint attention
        hidden_states = torch.cat([
            self.input_layernorm(lat_h),
            self.input_layernorm_action(act_h),
            self.input_layernorm_tactile(tac_h),
        ], dim=1)

        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            latent_indexes=latent_indexes,
            action_indexes=action_indexes,
            tactile_indexes=tactile_indexes,
        )

        hidden_states = residual + hidden_states

        # ── Per-expert FFN ──────────────────────────────────────────────────
        residual  = hidden_states
        lat_h = hidden_states[:, latent_indexes]
        act_h = hidden_states[:, action_indexes]
        tac_h = hidden_states[:, tactile_indexes]

        ffn_out = torch.cat([
            self.mlp(self.post_attention_layernorm(lat_h)),
            self.mlp_action(self.post_attention_layernorm_action(act_h)),
            self.mlp_tactile(self.post_attention_layernorm_tactile(tac_h)),
        ], dim=1)

        hidden_states = residual + ffn_out

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


# ─── Rotary Embedding wrapper ─────────────────────────────────────────────────

class Qwen3VLRotaryEmbeddingWrapper(nn.Module):
    """
    Thin wrapper that uses the system-transformers Qwen3-VL rotary embedding
    when available, otherwise falls back to a simple 1-D RoPE.

    The VL embedding accepts position_ids of shape [3, batch, seq_len] (M-RoPE).
    The 1-D fallback accepts [batch, seq_len].
    """

    def __init__(self, config, device=None):
        super().__init__()
        if _VLRotaryEmbedding is not None:
            # Qwen3VLTextRotaryEmbedding expects config with rope_scaling
            # containing mrope_section.  Build a compatible config object.
            rope_scaling = getattr(config, "rope_scaling", None)
            if rope_scaling is None:
                # Qwen3-VL text config may have rope_scaling already set;
                # if not, construct a default one.
                rope_scaling = {
                    "rope_type": "default",
                    "mrope_section": [16, 24, 24],  # Qwen3-VL default
                }
            _rope_cfg = type("_RopeCfg", (), {
                "hidden_size":             config.hidden_size,
                "num_attention_heads":     config.num_attention_heads,
                "head_dim":                getattr(config, "head_dim",
                                                   config.hidden_size // config.num_attention_heads),
                "max_position_embeddings": getattr(config, "max_position_embeddings", 32768),
                "rope_theta":              getattr(config, "rope_theta", 1000000.0),
                "rope_scaling":            rope_scaling,
                "partial_rotary_factor":   getattr(config, "partial_rotary_factor", 1.0),
            })()
            self._rope = _VLRotaryEmbedding(config=_rope_cfg, device=device)
        else:
            # Build a simple 1-D RoPE from config
            self._rope = None
            rope_theta = getattr(config, "rope_theta", 10000.0)
            head_dim   = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
            inv_freq   = 1.0 / (
                rope_theta
                ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        if self._rope is not None:
            return self._rope(x, position_ids)

        # Fallback 1-D RoPE
        if position_ids.ndim == 3:
            # M-RoPE format is [3, B, L]; take temporal dim [0] → [B, L]
            position_ids = position_ids[0]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos_expanded      = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type if x.device.type != "mps" else "cpu", enabled=False):
            freqs = (inv_freq_expanded.float() @ pos_expanded.float()).transpose(1, 2)
            emb   = torch.cat((freqs, freqs), dim=-1)
            cos   = emb.cos() * self.attention_scaling
            sin   = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ─── Causal mask (simple manual implementation) ───────────────────────────────

def build_causal_mask(
    seq_len:      int,
    past_len:     int,
    device:       torch.device,
    dtype:        torch.dtype,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Returns additive causal mask of shape [1, 1, seq_len, past_len + seq_len].
    Combines the standard lower-triangular causal mask with an optional padding
    mask from `attention_mask` (1 = keep, 0 = mask).
    """
    total = past_len + seq_len
    # [1, 1, seq_len, total]
    mask = torch.full((1, 1, seq_len, total), float("-inf"), device=device, dtype=dtype)
    q_pos  = torch.arange(past_len, past_len + seq_len, device=device).view(-1, 1)
    kv_pos = torch.arange(total, device=device).view(1, -1)
    causal = q_pos >= kv_pos
    mask = mask.masked_fill(causal, 0.0)

    if attention_mask is not None:
        # attention_mask: [batch, L_latent] – 1 means attend, 0 means ignore.
        # It may be shorter than `total` when action/tactile tokens are appended;
        # those extra positions are always attended (pad with 1s).
        B_mask, L_mask = attention_mask.shape
        if L_mask < total:
            attention_mask = torch.cat(
                [attention_mask,
                 torch.ones(B_mask, total - L_mask, device=device,
                            dtype=attention_mask.dtype)],
                dim=1,
            )
        # Expand to [batch, 1, 1, total] and apply.
        # NOTE: avoid  (1 - mask) * -inf  because  0.0 * -inf = NaN in IEEE 754.
        # Use masked_fill instead to safely set padded positions to -inf.
        pad_mask = torch.zeros(B_mask, total, dtype=dtype, device=device)
        pad_mask = pad_mask.masked_fill(attention_mask[:, :total] == 0, float("-inf"))
        pad_mask = pad_mask.view(B_mask, 1, 1, total)
        mask = mask + pad_mask

        # Ensure every query can attend to at least itself (self-attention on
        # the diagonal).  Without this, left-padded positions have ALL KV
        # positions masked → softmax([-inf, ...]) = NaN.
        diag_idx = torch.arange(seq_len, device=device)
        mask[:, :, diag_idx, past_len + diag_idx] = 0.0

    return mask


# ─── MoT Qwen3-VL Model ──────────────────────────────────────────────────────

class Qwen3VLModelMoT(nn.Module):
    """
    The Qwen3-VL text backbone modified with Mixture-of-Transformers.

    Accepts position_ids of shape [batch, 3, seq_len] (M-RoPE, from the
    Qwen3-VL processor) for the latent tokens.  Action and tactile
    position_ids are generated internally by extending the latent positions
    sequentially.
    """

    def __init__(self, config, tactile_intermediate_size: int = None):
        super().__init__()
        self.config    = config
        self.vocab_size = config.vocab_size
        self.tactile_intermediate_size = tactile_intermediate_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size,
                                         padding_idx=getattr(config, "pad_token_id", None))
        self.layers = nn.ModuleList([
            Qwen3VLDecoderLayerMoT(config, layer_idx,
                                   tactile_intermediate_size=tactile_intermediate_size)
            for layer_idx in range(config.num_hidden_layers)
        ])
        # Three final norms – one per expert stream
        self.norm         = Qwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_action  = Qwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm_tactile = Qwen3VLRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb   = Qwen3VLRotaryEmbeddingWrapper(config)
        self.gradient_checkpointing = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.embed_tokens = value

    @staticmethod
    def _extend_position_ids(
        latent_position_ids: torch.Tensor,
        n_action:  int,
        n_tactile: int,
    ) -> torch.Tensor:
        """
        Extend M-RoPE position_ids [3, batch, latent_seq] to cover
        action and tactile tokens appended at the end.

        Action / tactile tokens have no spatial structure so all three
        M-RoPE dimensions receive the same sequential value.
        """
        device = latent_position_ids.device
        is_mrope = (latent_position_ids.ndim == 3)

        if is_mrope:
            # latent_position_ids: [3, B, L]
            max_pos = latent_position_ids.max(dim=-1, keepdim=True)[0]  # [3, B, 1]
            if n_action > 0:
                off_a = torch.arange(1, n_action + 1, device=device).view(1, 1, n_action)
                act_pos = max_pos + off_a  # [3, B, n_action] via broadcast
            else:
                act_pos = latent_position_ids[:, :, :0]

            if n_tactile > 0:
                off_t = torch.arange(n_action + 1, n_action + n_tactile + 1, device=device).view(1, 1, n_tactile)
                tac_pos = max_pos + off_t  # [3, B, n_tactile] via broadcast
            else:
                tac_pos = latent_position_ids[:, :, :0]

            return torch.cat([latent_position_ids, act_pos, tac_pos], dim=-1)
        else:
            # 1-D position_ids: [batch, latent_seq]
            max_pos = latent_position_ids.max(dim=-1, keepdim=True)[0]  # [B, 1]
            parts   = [latent_position_ids]
            if n_action > 0:
                off_a = torch.arange(1, n_action + 1, device=device).view(1, n_action)
                parts.append(max_pos + off_a)
            if n_tactile > 0:
                off_t = torch.arange(n_action + 1, n_action + n_tactile + 1, device=device).view(1, n_tactile)
                parts.append(max_pos + off_t)
            return torch.cat(parts, dim=-1)

    def forward(
        self,
        inputs_embeds:   Optional[torch.FloatTensor]  = None,
        attention_mask:  Optional[torch.Tensor]        = None,
        position_ids:    Optional[torch.LongTensor]    = None,   # latent only [B,3,L] or [B,L]
        past_key_values: Optional[Cache]               = None,
        use_cache:       bool                          = False,
        output_attentions:    bool                     = False,
        output_hidden_states: bool                     = False,
        cache_position:  Optional[torch.LongTensor]    = None,
        # MoT routing
        latent_indexes:  Optional[torch.LongTensor]    = None,
        action_indexes:  Optional[torch.LongTensor]    = None,
        tactile_indexes: Optional[torch.LongTensor]    = None,
    ) -> BaseModelOutputWithPast:

        hidden_states = inputs_embeds
        batch_size, seq_len = hidden_states.shape[:2]

        # ── KV-cache bookkeeping ────────────────────────────────────────────
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()
        past_len = past_key_values.get_seq_length() if past_key_values is not None else 0

        if cache_position is None:
            cache_position = torch.arange(past_len, past_len + seq_len, device=hidden_states.device)

        # ── Default routing (all latent, no action/tactile) ─────────────────
        if latent_indexes is None:
            latent_indexes  = torch.arange(seq_len, device=hidden_states.device)
        if action_indexes is None:
            action_indexes  = torch.arange(0, 0, device=hidden_states.device)
        if tactile_indexes is None:
            tactile_indexes = torch.arange(0, 0, device=hidden_states.device)

        n_action  = len(action_indexes)
        n_tactile = len(tactile_indexes)
        n_latent  = len(latent_indexes)

        # ── Position IDs ────────────────────────────────────────────────────
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)  # 1-D fallback [1, seq]

        # Extend position_ids for action / tactile tokens if needed.
        # Only extend when position_ids covers only the latent portion
        # (shape[-1] < seq_len).  When position_ids already covers the
        # full sequence (e.g. pre-extended or KV-cache subsequent steps),
        # skip extension.
        if (n_action + n_tactile) > 0 and position_ids.shape[-1] < seq_len:
            position_ids_full = self._extend_position_ids(
                position_ids, n_action, n_tactile,
            )
        else:
            position_ids_full = position_ids  # already covers full seq

        # ── Causal attention mask ────────────────────────────────────────────
        causal_mask = build_causal_mask(
            seq_len=seq_len,
            past_len=past_len,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            attention_mask=attention_mask,
        )

        # ── Rotary embeddings ────────────────────────────────────────────────
        position_embeddings = self.rotary_emb(hidden_states, position_ids_full)

        # ── Decoder layers ───────────────────────────────────────────────────
        all_hidden_states = () if output_hidden_states else None
        all_attentions    = () if output_attentions    else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_out = torch.utils.checkpoint.checkpoint(
                    layer,
                    hidden_states,
                    causal_mask,
                    None,                # position_ids (unused in layer; passed via position_embeddings)
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                    latent_indexes,
                    action_indexes,
                    tactile_indexes,
                    use_reentrant=False,
                )
            else:
                layer_out = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    latent_indexes=latent_indexes,
                    action_indexes=action_indexes,
                    tactile_indexes=tactile_indexes,
                )

            hidden_states = layer_out[0]
            if output_attentions:
                all_attentions += (layer_out[1],)

        # ── Three final norms, one per stream ────────────────────────────────
        lat_h = self.norm(hidden_states[:, latent_indexes])         if n_latent  > 0 else hidden_states[:, :0]
        act_h = self.norm_action(hidden_states[:, action_indexes])  if n_action  > 0 else hidden_states[:, :0]
        tac_h = self.norm_tactile(hidden_states[:, tactile_indexes]) if n_tactile > 0 else hidden_states[:, :0]
        hidden_states = torch.cat([lat_h, act_h, tac_h], dim=1)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )

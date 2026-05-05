"""
Qwen3VLVLAModel – the full VLA model.

Architecture
────────────
  visual             : Qwen3VL ViT (frozen during visual feature extraction)
  model              : Qwen3VLModelMoT – three-expert Qwen3-VL text backbone
  x_embedder         : noisy actions  → hidden_size
  t_embedder         : diffusion time → hidden_size
  tacf6_embed        : F6 tactile vec → hidden_size   (optional)
  deform_proj        : tactile deform → hidden_size   (optional)
  state_embed        : robot state    → hidden_size   (optional)
  final_layer        : hidden_size    → action_dim    (action expert output)
  final_layer_tactile: hidden_size    → action_dim    (tactile expert residual)

Two flow-matching paradigms are supported:

(A) Paradigm A — joint flow with delta_v residual
    Tactile is in every Euler step alongside action.  v_total = v_act + delta_v.
    Methods: forward_flow / denoise_step (legacy).

(C) Paradigm C — action-only slow flow + tactile residual flow
    Slow path is tactile-blind, integrates 10 Euler steps of action expert →
    clean chunk Â plus cached (latent + action) KV.  Fast path runs the tactile
    expert as its own short flow on the residual r ≈ A_demo − Â, conditioned on
    cached KV.  Final action: A_refined = Â + Δa.
    Methods: forward_flow_action_only, tactile_residual_flow (inference),
             tactile_residual_train_step (training).
"""

import copy
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from .modeling_qwen3vl_mot import Qwen3VLModelMoT
from diffusion import ActionEmbedder, TimestepEmbedder, FinalLayer
from models import DeformEncoder


# ── Default image-pad token id (matches Qwen3-VL / Qwen2-VL) ────────────────
_DEFAULT_IMAGE_TOKEN_ID = 151655   # <|image_pad|>


class Qwen3VLVLAModel(nn.Module):
    """
    Full VLA model wrapping the Qwen3-VL visual encoder and the
    three-expert (MoT) Qwen3 text backbone.
    """

    def __init__(
        self,
        config,
        action_dim: int = 29,
        action_chunk:        int   = 8,
        tacf6_dim:           int   = 6,
        use_tactile_deform:  bool  = False,
        use_robot_state:     bool  = False,
        image_token_id:      int   = _DEFAULT_IMAGE_TOKEN_ID,
        tactile_intermediate_size: int = None,
        n_flare_tokens_per_frame: int = 0,
        n_flare_steps:           int = 0,
        flare_layer_index:       int = -1,
    ):
        super().__init__()
        self.config             = config
        self.action_dim         = action_dim
        self.action_chunk       = action_chunk
        self.tacf6_dim          = tacf6_dim
        self.use_tactile_deform = use_tactile_deform
        self.use_robot_state    = use_robot_state
        self.image_token_id     = image_token_id
        self.tactile_intermediate_size = tactile_intermediate_size
        self.n_flare_tokens_per_frame = n_flare_tokens_per_frame
        self.n_flare_steps            = n_flare_steps
        self.n_flare_tokens           = n_flare_tokens_per_frame * n_flare_steps
        self.flare_layer_index        = flare_layer_index

        self.visual = None

        self.model = Qwen3VLModelMoT(config, tactile_intermediate_size=tactile_intermediate_size)

        H = config.hidden_size
        self.x_embedder          = ActionEmbedder(action_dim, H)
        self.t_embedder          = TimestepEmbedder(H)
        self.final_layer         = FinalLayer(H, action_dim)   # action expert → velocity
        self.final_layer_tactile = FinalLayer(H, action_dim)   # tactile expert → residual
        self.tacf6_embedder      = ActionEmbedder(tacf6_dim, H)

        if use_tactile_deform:
            self.deform_encoder = DeformEncoder()
            self.deform_proj    = ActionEmbedder(28800, H) # 128 * 15 * 15

        if use_robot_state:
            self.state_embedder = ActionEmbedder(action_dim, H)

        # Flare visual prediction tokens for the latent expert
        if self.n_flare_tokens > 0:
            self.flare_queries = nn.Parameter(
                torch.randn(1, self.n_flare_tokens, H) * 0.02)
            self.flare_proj = nn.Sequential(
                nn.Linear(H, H), nn.GELU(), nn.Linear(H, H))

        # Lightweight callable for get_rope_index (NOT an nn.Module — avoids
        # registering the full base model as a submodule).
        # Use object.__setattr__ to bypass nn.Module registration.
        object.__setattr__(self, '_rope_index_fn', None)

    @classmethod
    def from_pretrained_qwen3vl(
        cls,
        pretrained_path: str,
        action_dim:         int  = 29,
        action_chunk:       int  = 8,
        tacf6_dim:          int  = 6,
        use_tactile_deform: bool = False,
        use_robot_state:    bool = False,
        torch_dtype              = torch.bfloat16,
        tactile_intermediate_size: int = None,
        n_flare_tokens_per_frame: int = 0,
        n_flare_steps:            int = 0,
        flare_layer_index:        int = -1,
    ) -> "Qwen3VLVLAModel":
        """
        Build a Qwen3VLVLAModel by:
          1. Loading `Qwen3VLForConditionalGeneration` from `pretrained_path`.
          2. Copying the visual tower directly.
          3. Loading the text model weights into Qwen3VLModelMoT (strict=False).
          4. VLA-specific modules are randomly initialised.
        """
        from transformers import AutoConfig

        # Load the pretrained Qwen3-VL model for weight extraction
        try:
            from transformers import Qwen3VLForConditionalGeneration
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                pretrained_path, torch_dtype=torch_dtype, trust_remote_code=True
            )
        except Exception:
            from transformers import Qwen2VLForConditionalGeneration
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                pretrained_path, torch_dtype=torch_dtype, trust_remote_code=True
            )

        config = base_model.config
        # Qwen3-VL text config lives in config.model_config or directly in config
        text_config = getattr(config, "text_config", config)

        image_token_id = getattr(config, "image_token_id", _DEFAULT_IMAGE_TOKEN_ID)

        # Construct VLA model
        vla = cls(
            config = text_config,
            action_dim = action_dim,
            action_chunk = action_chunk,
            tacf6_dim = tacf6_dim,
            use_tactile_deform = use_tactile_deform,
            use_robot_state = use_robot_state,
            image_token_id = image_token_id,
            tactile_intermediate_size = tactile_intermediate_size,
            n_flare_tokens_per_frame = n_flare_tokens_per_frame,
            n_flare_steps = n_flare_steps,
            flare_layer_index = flare_layer_index,
        )
        # Capture only the get_rope_index function — do NOT store the full
        # base model as an attribute, because nn.Module.__setattr__ would
        # register it as a submodule (duplicating ~2B parameters and breaking
        # the freeze logic for the visual encoder).
        # In Qwen3-VL, get_rope_index lives on Qwen3VLModel (base_model.model),
        # NOT on Qwen3VLForConditionalGeneration (base_model).
        _inner = base_model.model if hasattr(base_model, "model") else base_model
        if hasattr(_inner, "get_rope_index"):
            object.__setattr__(vla, '_rope_index_fn', _inner.get_rope_index)

        # Copy visual tower (base_model.visual is a @property → base_model.model.visual)
        vla.visual = base_model.visual

        # Load text model weights into Qwen3VLModelMoT (strict=False).
        # In Qwen3-VL (transformers >=4.57), base_model.model is Qwen3VLModel
        # which wraps both visual and language_model.  State dict keys are
        # prefixed: 'visual.*' and 'language_model.*'.
        # In Qwen2-VL (older), base_model.model is the text model directly
        # and keys have no prefix (embed_tokens.*, layers.*, norm.*).
        raw_sd = base_model.model.state_dict()

        # Detect the key structure and extract text-model weights
        lang_prefix = "language_model."
        has_lang_prefix = any(k.startswith(lang_prefix) for k in raw_sd)

        text_sd = {}
        if has_lang_prefix:
            # Qwen3-VL: strip 'language_model.' prefix, skip visual keys
            for k, v in raw_sd.items():
                if k.startswith(lang_prefix):
                    text_sd[k[len(lang_prefix):]] = v
        else:
            # Qwen2-VL / direct text model: use keys as-is
            text_sd = dict(raw_sd)

        missing, unexpected = vla.model.load_state_dict(text_sd, strict=False)
        print(f"[Qwen3VLVLAModel] Text model loaded – missing: {len(missing)}, unexpected: {len(unexpected)}")
        if missing:
            # Filter out expected _action / _tactile keys for cleaner logging
            truly_missing = [k for k in missing if "_action" not in k and "_tactile" not in k]
            expected_missing = len(missing) - len(truly_missing)
            if truly_missing:
                print(f"  WARNING – unexpected missing base keys: {truly_missing[:10]} ...")
            print(f"  New MoT expert weights (expected missing): {expected_missing}")

        # Free the base model's language_model to save memory.
        # get_rope_index only needs config + image_token_id, not the weights,
        # but it's a bound method so we keep the Qwen3VLModel alive.
        # Deleting language_model parameters frees ~1.5B params.
        if hasattr(_inner, "language_model"):
            del _inner.language_model
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Qwen3VLVLAModel] Freed base model language_model to save memory.")

        return vla

    def load_deform_encoder_weights(self, ckpt_path: str):
        if not self.use_tactile_deform:
            return
        if not os.path.exists(ckpt_path):
            print(f"Warning: DeformEncoder checkpoint not found at {ckpt_path}")
            return
        print(f"Loading DeformEncoder weights from {ckpt_path} ...")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
        encoder_sd = {}
        for k, v in state_dict.items():
            if k.startswith("encoder."):
                encoder_sd[k[len("encoder."):]] = v
            elif k in self.deform_encoder.state_dict():
                encoder_sd[k] = v
        missing, _ = self.deform_encoder.load_state_dict(encoder_sd, strict=False)
        if missing:
            print(f"  DeformEncoder missing keys: {missing}")
        print("DeformEncoder weights loaded.")

    def initialize_vla_weights(self, skip_tactile_zero_init: bool = False):
        """Xavier-init all new VLA-specific linear layers.

        Parameters
        ----------
        skip_tactile_zero_init : bool
            If True, do NOT zero-init `final_layer_tactile.mlp.fc2`.  When the
            tactile head is trained as a residual-flow velocity predictor
            (use_tactile_refine_flow=1), it must start non-trivial so the
            refinement signal is alive from step 0.  When the tactile head is
            trained as a delta_v residual on top of v_act (the legacy mode),
            keep the zero-init so delta_v starts as a no-op.
        """
        for m in [self.x_embedder, self.t_embedder, self.final_layer,
                  self.final_layer_tactile, self.tacf6_embedder]:
            for mm in m.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    if mm.bias is not None:
                        nn.init.zeros_(mm.bias)
        if self.use_robot_state:
            for mm in self.state_embedder.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    if mm.bias is not None:
                        nn.init.zeros_(mm.bias)
        if self.use_tactile_deform:
            for mm in self.deform_proj.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    if mm.bias is not None:
                        nn.init.zeros_(mm.bias)
        # Zero-init action expert's output head so v_act starts at zero (will
        # be overwritten by the pretrained checkpoint when resuming).
        nn.init.zeros_(self.final_layer.mlp.fc2.weight)
        if self.final_layer.mlp.fc2.bias is not None:
            nn.init.zeros_(self.final_layer.mlp.fc2.bias)
        # Tactile head: zero-init only when used as a delta_v residual.
        if not skip_tactile_zero_init:
            nn.init.zeros_(self.final_layer_tactile.mlp.fc2.weight)
            if self.final_layer_tactile.mlp.fc2.bias is not None:
                nn.init.zeros_(self.final_layer_tactile.mlp.fc2.bias)
        # Flare prediction projection
        if self.n_flare_tokens > 0:
            for mm in self.flare_proj.modules():
                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    if mm.bias is not None:
                        nn.init.zeros_(mm.bias)

    def prepare_inputs_embeds(
        self,
        input_ids:       torch.LongTensor,
        pixel_values:    Optional[torch.Tensor]  = None,
        image_grid_thw:  Optional[torch.Tensor]  = None,
    ) -> torch.Tensor:
        """
        Build the combined vision-language embedding sequence.

        input_ids      : [B, L]  (contains <|image_pad|> at image positions)
        pixel_values   : [N_patches, 3, patch_h, patch_w]  (all images flattened)
        image_grid_thw : [N_images, 3]  – (temporal, height, width) grid per image
        """
        # Text embeddings from the MoT backbone's embedding table
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if pixel_values is not None and self.visual is not None:
            # Run the visual tower
            pixel_values = pixel_values.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
            out = self.visual(pixel_values, grid_thw=image_grid_thw)
            # Qwen3-VL returns (hidden_states, deepstack_features) tuple;
            # Qwen2-VL returns a plain tensor.
            image_features = out[0] if isinstance(out, (tuple, list)) else out
            # image_features: [total_merged_tokens, hidden_size]

            # Locate image-pad positions and inject features
            image_mask = (input_ids == self.image_token_id)
            if image_mask.any():
                inputs_embeds[image_mask] = image_features.to(inputs_embeds.dtype)

        return inputs_embeds

    def get_rope_index(
        self,
        input_ids:      torch.LongTensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Delegate M-RoPE index computation to the underlying Qwen3VL base model.
        Returns (position_ids, mrope_position_deltas).

        position_ids shape: [3, batch, latent_seq_len]
        """
        if self._rope_index_fn is not None:
            return self._rope_index_fn(
                input_ids       = input_ids,
                image_grid_thw  = image_grid_thw,
                attention_mask  = attention_mask,
            )
        # Fallback: plain sequential 1-D position ids
        batch, seq_len = input_ids.shape
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch, -1)
        return position_ids, None

    def denoise_step(
        self,
        inputs_embeds: torch.Tensor, # slow/latent embeddings [B, L, H]
        position_ids: torch.Tensor, # M-RoPE [3, B, L] or 1-D [B, L]
        past_key_values,
        x_t: torch.Tensor, # noisy actions [B, chunk, action_dim]
        timestep: torch.Tensor, # [B]
        tactile_embeds: torch.Tensor, # [B, T, H]  (T=0 when no tactile)
        attention_mask: Optional[torch.Tensor] = None,
        state_embeds: Optional[torch.Tensor] = None, # [B, S, H] or None
        fast_embeds: Optional[torch.Tensor] = None,  # [B, F, H] or None
    ) -> Tuple[torch.Tensor, object]:

        noisy_actions = self.x_embedder(x_t.to(torch.bfloat16)) # [B, chunk, H]
        timesteps = self.t_embedder(timestep).unsqueeze(1) # [B, 1, H]
        n_chunk = noisy_actions.shape[1]

        B = inputs_embeds.shape[0]

        if fast_embeds is None:
            fast_embeds = torch.empty(
                (B, 0, inputs_embeds.shape[2]),
                device=inputs_embeds.device, dtype=inputs_embeds.dtype)

        # Sequence layout: [slow/latent | fast, state?, timestep, noisy_act | tac_seq?]
        act_parts = [fast_embeds]
        if state_embeds is not None and state_embeds.shape[1] > 0:
            act_parts.append(state_embeds)
        act_parts.extend([timesteps, noisy_actions])
        act_seq = torch.cat(act_parts, dim=1)
        n_act = act_seq.shape[1]

        has_tactile = (tactile_embeds.shape[1] > 0)
        if has_tactile:
            tac_seq = torch.cat([tactile_embeds, timesteps, noisy_actions], dim=1)
            n_tac = tac_seq.shape[1]
        else:
            n_tac = 0

        if past_key_values is None:
            parts = [inputs_embeds, act_seq]
            if has_tactile:
                parts.append(tac_seq)
            full_embeds = torch.cat(parts, dim=1)
            L = inputs_embeds.shape[1]
            total = full_embeds.shape[1]
            latent_indexes  = torch.arange(0, L, device=full_embeds.device)
            action_indexes  = torch.arange(L, L + n_act, device=full_embeds.device)
            tactile_indexes = torch.arange(L + n_act, total, device=full_embeds.device)

            outputs = self.model(
                inputs_embeds   = full_embeds,
                position_ids    = position_ids,
                past_key_values = past_key_values,
                attention_mask  = attention_mask,
                use_cache       = True,
                latent_indexes  = latent_indexes,
                action_indexes  = action_indexes,
                tactile_indexes = tactile_indexes,
            )
        else:
            # Subsequent denoise steps: only pass the action/tactile tokens
            parts = [act_seq]
            if has_tactile:
                parts.append(tac_seq)
            full_embeds = torch.cat(parts, dim=1)
            drop = n_act + n_tac
            past_key_values.crop(-drop)
            total = full_embeds.shape[1]
            latent_indexes  = torch.arange(0, 0, device=full_embeds.device)
            action_indexes  = torch.arange(0, n_act, device=full_embeds.device)
            tactile_indexes = torch.arange(n_act, total, device=full_embeds.device)

            extended_pos = self.model._extend_position_ids(
                position_ids, n_act, n_tac,
            )
            act_tac_pos = extended_pos[..., -(n_act + n_tac):]

            outputs = self.model(
                inputs_embeds   = full_embeds,
                position_ids    = act_tac_pos,
                past_key_values = past_key_values,
                use_cache       = True,
                latent_indexes  = latent_indexes,
                action_indexes  = action_indexes,
                tactile_indexes = tactile_indexes,
            )

        hidden = outputs.last_hidden_state
        act_block_end = hidden.shape[1] - n_tac
        h_act_chunk = hidden[:, act_block_end - n_chunk : act_block_end, :]
        v_act = self.final_layer(h_act_chunk)

        if has_tactile:
            h_tac_chunk = hidden[:, -n_chunk:, :]
            delta_v = self.final_layer_tactile(h_tac_chunk)
            v_t = v_act + delta_v
        else:
            v_t = v_act

        return v_t, outputs.past_key_values

    def forward_flow(
        self,
        inputs_embeds: torch.Tensor, # slow/latent embeddings
        position_ids: torch.Tensor, # M-RoPE for latent
        noise: torch.Tensor, # [B, chunk, action_dim]
        tactile_inputs: Optional[torch.Tensor] = None, # legacy
        num_steps: int = 10,
        attention_mask: Optional[torch.Tensor] = None,
        state_embeds: Optional[torch.Tensor] = None,
        tactile_f6: Optional[torch.Tensor] = None,
        tactile_deform: Optional[torch.Tensor] = None,
        fast_embeds: Optional[torch.Tensor] = None, # [B, F, H] fast view embeds for action expert
    ) -> torch.Tensor:
        """Euler integration: x_1 (noise) → x_0 (action)."""
        device = noise.device
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.bfloat16, device=device)
        x_t = noise.to(torch.bfloat16)
        time = torch.tensor(1.0, dtype=torch.bfloat16, device=device)

        # Legacy: single tactile_inputs → route to the appropriate new arg
        if tactile_inputs is not None and tactile_f6 is None and tactile_deform is None:
            if self.use_tactile_deform and tactile_inputs.ndim == 5:
                tactile_deform = tactile_inputs
            elif tactile_inputs.ndim == 3:
                tactile_f6 = tactile_inputs

        # Build tactile_embeds: [f6_tokens, deform_tokens] (concat when both present)
        tac_parts = []
        if tactile_f6 is not None:
            tac_parts.append(self.tacf6_embedder(tactile_f6.to(torch.bfloat16)))
        if tactile_deform is not None:
            B, n_fingers, C, H, W = tactile_deform.shape
            deforms_flat = tactile_deform.view(-1, C, H, W).to(inputs_embeds.dtype)
            deform_feats = self.deform_encoder(deforms_flat)
            deform_feats = deform_feats.view(B, n_fingers, -1)
            tac_parts.append(self.deform_proj(deform_feats.to(torch.bfloat16)))

        if tac_parts:
            tactile_embeds = torch.cat(tac_parts, dim=1)
        else:
            tactile_embeds = torch.empty(
                (noise.shape[0], 0, inputs_embeds.shape[2]),
                device=device, dtype=torch.bfloat16)

        past_kv = None
        while time >= -dt / 2:
            expanded_time = time.expand(x_t.shape[0])
            v_t, past_kv = self.denoise_step(
                inputs_embeds   = inputs_embeds,
                position_ids    = position_ids,
                past_key_values = past_kv,
                x_t             = x_t,
                timestep        = expanded_time,
                tactile_embeds  = tactile_embeds,
                attention_mask  = attention_mask,
                state_embeds    = state_embeds,
                fast_embeds     = fast_embeds,
            )
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t

    # ──────────────────────────────────────────────────────────────────────
    # Paradigm C — action-only slow flow + tactile residual flow
    # ──────────────────────────────────────────────────────────────────────

    def forward_flow_action_only(
        self,
        inputs_embeds: torch.Tensor,            # [B, L_latent, H]   slow + flare embeds
        position_ids: torch.Tensor,             # [3, B, L_latent]   M-RoPE for latent
        noise: torch.Tensor,                    # [B, n_chunk, action_dim]
        attention_mask: Optional[torch.Tensor] = None,
        state_embeds: Optional[torch.Tensor] = None,    # [B, S, H] or None
        fast_embeds: Optional[torch.Tensor] = None,     # [B, F, H] or None
        num_steps: int = 10,
        refresh_clean_kv: bool = True,
    ) -> Tuple[torch.Tensor, "DynamicCache", int]:
        """Tactile-blind Euler flow for the action expert.

        Returns
        -------
        clean_chunk : [B, n_chunk, action_dim]   the integrated Â at τ=0.
        cached_kv   : DynamicCache with [latent_KV | action_KV] suitable for
                      reuse by tactile_residual_flow / tactile_residual_train_step.
        n_action_in_cache : int   number of action tokens in `cached_kv`.

        When `refresh_clean_kv=True`, after integration we run one extra forward
        at τ=0 with the clean Â so that the cached action KV reflects the clean
        state the tactile expert will attend to at refinement time.
        """
        device = noise.device
        dtype  = torch.bfloat16
        dt     = torch.tensor(-1.0 / num_steps, dtype=dtype, device=device)
        x_t    = noise.to(dtype)
        time   = torch.tensor(1.0, dtype=dtype, device=device)
        n_chunk = noise.shape[1]
        B      = inputs_embeds.shape[0]
        H      = inputs_embeds.shape[2]

        if fast_embeds is None:
            fast_embeds = torch.empty((B, 0, H), device=device, dtype=dtype)
        if state_embeds is None:
            state_embeds = torch.empty((B, 0, H), device=device, dtype=dtype)
        n_state = state_embeds.shape[1]
        L_latent = inputs_embeds.shape[1]

        past_kv = None
        n_act = 0
        while time >= -dt / 2:
            timesteps = self.t_embedder(time.expand(B)).unsqueeze(1)
            noisy_act = self.x_embedder(x_t)
            act_parts = [fast_embeds]
            if n_state > 0:
                act_parts.append(state_embeds)
            act_parts += [timesteps, noisy_act]
            act_seq = torch.cat(act_parts, dim=1)
            n_act = act_seq.shape[1]

            if past_kv is None:
                full_embeds = torch.cat([inputs_embeds, act_seq], dim=1)
                outputs = self.model(
                    inputs_embeds=full_embeds,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_kv,
                    use_cache=True,
                    latent_indexes=torch.arange(0, L_latent, device=device),
                    action_indexes=torch.arange(L_latent, L_latent + n_act, device=device),
                    tactile_indexes=torch.arange(0, 0, device=device),
                )
            else:
                past_kv.crop(-n_act)
                extended_pos = self.model._extend_position_ids(position_ids, n_act, 0)
                act_pos = extended_pos[..., -n_act:]
                outputs = self.model(
                    inputs_embeds=act_seq,
                    position_ids=act_pos,
                    past_key_values=past_kv,
                    use_cache=True,
                    latent_indexes=torch.arange(0, 0, device=device),
                    action_indexes=torch.arange(0, n_act, device=device),
                    tactile_indexes=torch.arange(0, 0, device=device),
                )

            hidden = outputs.last_hidden_state
            v_act  = self.final_layer(hidden[:, -n_chunk:, :])
            x_t    = x_t + dt * v_act
            time   = time + dt
            past_kv = outputs.past_key_values

        if refresh_clean_kv and past_kv is not None:
            past_kv.crop(-n_act)
            clean_timesteps = self.t_embedder(
                torch.zeros(B, device=device, dtype=dtype)
            ).unsqueeze(1)
            clean_actions = self.x_embedder(x_t)
            clean_parts = [fast_embeds]
            if n_state > 0:
                clean_parts.append(state_embeds)
            clean_parts += [clean_timesteps, clean_actions]
            clean_seq = torch.cat(clean_parts, dim=1)
            n_act_final = clean_seq.shape[1]
            extended_pos = self.model._extend_position_ids(position_ids, n_act_final, 0)
            act_pos_final = extended_pos[..., -n_act_final:]
            _ = self.model(
                inputs_embeds=clean_seq,
                position_ids=act_pos_final,
                past_key_values=past_kv,
                use_cache=True,
                latent_indexes=torch.arange(0, 0, device=device),
                action_indexes=torch.arange(0, n_act_final, device=device),
                tactile_indexes=torch.arange(0, 0, device=device),
            )
            n_action_in_cache = n_act_final
        else:
            n_action_in_cache = n_act

        return x_t, past_kv, n_action_in_cache

    def _embed_tactile_observations(
        self,
        tactile_f6: Optional[torch.Tensor],
        tactile_deform: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build [B, n_obs, H] tactile observation embedding (vec + deform)."""
        tac_parts = []
        if tactile_f6 is not None and tactile_f6.shape[1] > 0:
            tac_parts.append(self.tacf6_embedder(tactile_f6.to(dtype)))
        if (tactile_deform is not None
                and tactile_deform.shape[1] > 0
                and self.use_tactile_deform):
            Bd, nf, C, Hd, Wd = tactile_deform.shape
            flat = tactile_deform.view(-1, C, Hd, Wd).to(dtype)
            feats = self.deform_encoder(flat).view(Bd, nf, -1)
            tac_parts.append(self.deform_proj(feats.to(dtype)))
        if not tac_parts:
            raise ValueError(
                "tactile residual flow requires tactile_f6 or tactile_deform")
        return torch.cat(tac_parts, dim=1)

    def tactile_residual_train_step(
        self,
        cached_kv,                              # DynamicCache with latent + action KV
        latent_position_ids: torch.Tensor,      # [3, B, L_latent]
        n_action_in_cache: int,
        base_chunk: torch.Tensor,               # [B, n_chunk, action_dim]  Â (detached)
        tactile_f6: Optional[torch.Tensor] = None,
        tactile_deform: Optional[torch.Tensor] = None,
        r_tau: torch.Tensor = None,             # [B, n_chunk, action_dim]  noisy residual
        tau: torch.Tensor = None,               # [B] flow time scalars
    ) -> torch.Tensor:
        """Single tactile-only forward at (r_τ, τ).

        Returns predicted velocity v_pred [B, n_chunk, action_dim] suitable for
        L_refine = MSE(v_pred, ε_r − r_target).  Does NOT integrate; this is the
        training-time analog of one Euler step of `tactile_residual_flow`.
        """
        device = base_chunk.device
        dtype  = torch.bfloat16
        B      = base_chunk.shape[0]
        n_chunk = base_chunk.shape[1]

        tac_obs = self._embed_tactile_observations(
            tactile_f6, tactile_deform, device, dtype)
        n_obs = tac_obs.shape[1]

        tau_emb = self.t_embedder(tau.to(dtype)).unsqueeze(1)            # [B, 1, H]
        r_emb   = self.x_embedder(r_tau.to(dtype))                       # [B, n_chunk, H]
        full_embeds = torch.cat([tac_obs, tau_emb, r_emb], dim=1)
        n_tac_seq = full_embeds.shape[1]

        extended_pos = self.model._extend_position_ids(
            latent_position_ids, n_action_in_cache, n_tac_seq)
        tac_pos = extended_pos[..., -n_tac_seq:]

        outputs = self.model(
            inputs_embeds=full_embeds,
            position_ids=tac_pos,
            past_key_values=cached_kv,
            use_cache=True,
            latent_indexes=torch.arange(0, 0, device=device),
            action_indexes=torch.arange(0, 0, device=device),
            tactile_indexes=torch.arange(0, n_tac_seq, device=device),
        )
        hidden = outputs.last_hidden_state
        v_pred = self.final_layer_tactile(hidden[:, -n_chunk:, :])
        return v_pred

    @staticmethod
    def _clone_dynamic_cache(cache: DynamicCache) -> DynamicCache:
        """Manual clone of a DynamicCache: torch.deepcopy fails on non-leaf
        tensors, so we copy the per-layer K/V tensors with detach().clone()."""
        new_cache = DynamicCache()
        # Mirror per-layer DynamicLayer instances with cloned K/V tensors.
        if hasattr(cache, "layers") and isinstance(cache.layers, list):
            from transformers.cache_utils import DynamicLayer
            for layer in cache.layers:
                new_layer = DynamicLayer()
                if getattr(layer, "is_initialized", False):
                    new_layer.dtype  = layer.dtype
                    new_layer.device = layer.device
                    new_layer.keys   = layer.keys.detach().clone()
                    new_layer.values = layer.values.detach().clone()
                    new_layer.is_initialized = True
                new_cache.layers.append(new_layer)
        else:
            # Older (pre-4.55) API: key_cache / value_cache lists at top level.
            if hasattr(cache, "key_cache"):
                new_cache.key_cache = [k.detach().clone() for k in cache.key_cache]
                new_cache.value_cache = [v.detach().clone() for v in cache.value_cache]
                new_cache._seen_tokens = getattr(cache, "_seen_tokens", 0)
            else:
                raise NotImplementedError(
                    "Unrecognized DynamicCache layout; cannot clone for "
                    "tactile_residual_flow.")
        return new_cache

    @torch.no_grad()
    def tactile_residual_flow(
        self,
        cached_kv,
        latent_position_ids: torch.Tensor,
        n_action_in_cache: int,
        base_chunk: torch.Tensor,
        tactile_f6: Optional[torch.Tensor] = None,
        tactile_deform: Optional[torch.Tensor] = None,
        num_steps: int = 4,
        noise_scale: float = 0.1,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tactile-only Euler flow on the residual r ≈ A_demo − Â.

        Returns Δa [B, n_chunk, action_dim] to be applied as A_refined = Â + Δa.

        `cached_kv` is cloned (manual tensor.clone) so multiple async refreshes
        during a single action chunk can each start from the slow-path snapshot.

        `initial_noise`, when provided, is used as the τ=1 starting point of the
        flow instead of a fresh `randn`.  This is the recommended path at
        inference: the caller (typically `ParadigmCServer`) generates one noise
        sample per slow tick and reuses it across all fast refinements within
        the chunk window, so successive Δa values differ *only* due to changes
        in tactile, not due to per-call random noise.  Removes the dominant
        source of in-chunk action jerk.  The tensor is expected to already be
        scaled appropriately (caller multiplies by noise_scale themselves).
        """
        cache = self._clone_dynamic_cache(cached_kv)
        device = base_chunk.device
        dtype  = torch.bfloat16
        B      = base_chunk.shape[0]
        n_chunk, action_dim = base_chunk.shape[1], base_chunk.shape[2]

        tac_obs = self._embed_tactile_observations(
            tactile_f6, tactile_deform, device, dtype)
        n_obs = tac_obs.shape[1]
        n_tac_seq = n_obs + 1 + n_chunk

        extended_pos = self.model._extend_position_ids(
            latent_position_ids, n_action_in_cache, n_tac_seq)
        tac_pos = extended_pos[..., -n_tac_seq:]

        if initial_noise is not None:
            r = initial_noise.to(device=device, dtype=dtype)
            assert r.shape == (B, n_chunk, action_dim), (
                f"initial_noise shape {tuple(r.shape)} != "
                f"expected ({B}, {n_chunk}, {action_dim})")
        else:
            r = (torch.randn(B, n_chunk, action_dim, device=device) * noise_scale
                 ).to(dtype)
        dt   = torch.tensor(-1.0 / num_steps, dtype=dtype, device=device)
        time = torch.tensor(1.0, dtype=dtype, device=device)

        step = 0
        while time >= -dt / 2:
            tau_emb = self.t_embedder(time.expand(B)).unsqueeze(1)
            r_emb   = self.x_embedder(r)
            full_embeds = torch.cat([tac_obs, tau_emb, r_emb], dim=1)
            if step > 0:
                cache.crop(-n_tac_seq)
            outputs = self.model(
                inputs_embeds=full_embeds,
                position_ids=tac_pos,
                past_key_values=cache,
                use_cache=True,
                latent_indexes=torch.arange(0, 0, device=device),
                action_indexes=torch.arange(0, 0, device=device),
                tactile_indexes=torch.arange(0, n_tac_seq, device=device),
            )
            hidden = outputs.last_hidden_state
            v_r = self.final_layer_tactile(hidden[:, -n_chunk:, :])
            r = r + dt * v_r
            time = time + dt
            step += 1

        return r

    def named_parameters(self, *args, **kwargs):
        return super().named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return super().parameters(*args, **kwargs)


def split_slow_fast_embeds(
    inputs_embeds: torch.Tensor,
    input_ids: torch.LongTensor,
    image_token_id: int,
    n_slow_img_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split inputs_embeds into slow (latent) and fast (action) portions.

    Splits at the <|vision_start|> of the first fast image: everything
    before it is slow (contiguous prefix), everything from it onward is
    fast (contiguous suffix).  This preserves token order — critical
    because the MoT routes tokens by contiguous index ranges and
    position_ids are computed for the original sequence order.

    Fast embeds include the vision brackets, image patches, AND any
    trailing tokens (``<|im_end|>``, generation prompt) that follow the
    fast images in the original sequence.

    Parameters
    ----------
    inputs_embeds     : [B, L, H]
    input_ids         : [B, L]
    image_token_id    : int — the <|image_pad|> token id (e.g. 151655)
    n_slow_img_tokens : int — total <|image_pad|> count for slow images

    Returns
    -------
    slow_embeds : [B, L_slow, H]  — contiguous prefix (text + slow image)
    fast_embeds : [B, L_fast, H]  — contiguous suffix (fast images + trailing)
    """
    B = inputs_embeds.shape[0]
    img_pad_mask = (input_ids == image_token_id)             # [B, L]
    img_cumcount = img_pad_mask.long().cumsum(dim=1)         # [B, L]
    fast_pad_mask = img_pad_mask & (img_cumcount > n_slow_img_tokens)

    n_fast_pads = int(fast_pad_mask[0].sum().item())
    if n_fast_pads == 0:
        return inputs_embeds, inputs_embeds[:, :0]

    # Split at <|vision_start|> of the first fast image (one position
    # before the first fast <|image_pad|> token).
    # All samples have the same fast content length (same image sizes +
    # same trailing tokens), so split_pos is identical across the batch.
    fast_pad_pos = fast_pad_mask[0].nonzero(as_tuple=True)[0]
    split_pos = int(fast_pad_pos[0].item()) - 1  # <|vision_start|>

    slow_embeds = inputs_embeds[:, :split_pos]
    fast_embeds = inputs_embeds[:, split_pos:]
    return slow_embeds, fast_embeds


def extend_position_ids_for_flare(
    pos_ids: torch.Tensor,
    n_flare: int,
) -> torch.Tensor:
    """
    Extend M-RoPE position_ids by n_flare sequential positions.

    Parameters
    ----------
    pos_ids : [3, B, L_slow]  M-RoPE positions for slow tokens
    n_flare : int  number of flare query tokens

    Returns
    -------
    [3, B, L_slow + n_flare]  extended positions
    """
    if n_flare == 0:
        return pos_ids
    device = pos_ids.device
    max_pos = pos_ids.max(dim=-1, keepdim=True)[0]  # [3, B, 1]
    offsets = torch.arange(1, n_flare + 1, device=device).view(1, 1, n_flare)
    flare_pos = max_pos + offsets  # [3, B, n_flare]
    return torch.cat([pos_ids, flare_pos], dim=-1)

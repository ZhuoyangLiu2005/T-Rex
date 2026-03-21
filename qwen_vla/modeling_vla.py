"""
Qwen3VLVLAModel – the full VLA model.

Architecture
────────────
  visual      : Qwen3VL ViT (frozen during visual feature extraction)
  model       : Qwen3VLModelMoT – three-expert Qwen3-VL text backbone
  x_embedder  : noisy actions  → hidden_size
  t_embedder  : diffusion time → hidden_size
  tacf6_embed : F6 tactile vec → hidden_size   (optional)
  deform_proj : tactile deform → hidden_size   (optional)
  state_embed : robot state    → hidden_size   (optional)
  final_layer : hidden_size    → action_dim

Loading from Qwen3VLForConditionalGeneration pretrained weights
──────────────────────────────────────────────────────────────
  Use `Qwen3VLVLAModel.from_pretrained_qwen3vl(path, ...)`.

  Base (latent) expert weights: load directly (strict=False).
  New action / tactile expert weights: missing in the checkpoint;
    they are initialised from the base weights in the training script
    via the --load_action_from_latent flag (same mechanism as in Janus).

Image token handling
────────────────────
  Qwen3-VL uses <|image_pad|> placeholder tokens inside input_ids.
  The visual tower output (already spatially merged) is injected at
  those positions in prepare_inputs_embeds.

M-RoPE position_ids
────────────────────
  The training script calls `model.get_rope_index(...)` (delegated to
  the underlying Qwen3VL model) to get latent position_ids [B, 3, L].
  Action / tactile positions are then appended inside Qwen3VLModelMoT.
"""

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
    ):
        super().__init__()
        self.config             = config
        self.action_dim         = action_dim
        self.action_chunk       = action_chunk
        self.tacf6_dim          = tacf6_dim
        self.use_tactile_deform = use_tactile_deform
        self.use_robot_state    = use_robot_state
        self.image_token_id     = image_token_id

        self.visual = None 

        self.model = Qwen3VLModelMoT(config)

        H = config.hidden_size
        self.x_embedder   = ActionEmbedder(action_dim, H)
        self.t_embedder   = TimestepEmbedder(H)
        self.final_layer  = FinalLayer(H, action_dim)
        self.tacf6_embedder = ActionEmbedder(tacf6_dim, H)

        if use_tactile_deform:
            self.deform_encoder = DeformEncoder()
            self.deform_proj    = ActionEmbedder(28800, H) # 128 * 15 * 15

        if use_robot_state:
            self.state_embedder = ActionEmbedder(action_dim, H)

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

    def initialize_vla_weights(self):
        """Xavier-init all new VLA-specific linear layers."""
        for m in [self.x_embedder, self.t_embedder, self.final_layer, self.tacf6_embedder]:
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
        nn.init.zeros_(self.final_layer.mlp.fc2.weight)
        if self.final_layer.mlp.fc2.bias is not None:
            nn.init.zeros_(self.final_layer.mlp.fc2.bias)

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
        inputs_embeds: torch.Tensor, # latent embeddings [B, L, H]
        position_ids: torch.Tensor, # M-RoPE [3, B, L] or 1-D [B, L]
        past_key_values,
        x_t: torch.Tensor, # noisy actions [B, chunk, action_dim]
        timestep: torch.Tensor, # [B]
        tactile_embeds: torch.Tensor, # [B, T, H]
        attention_mask: Optional[torch.Tensor] = None,
        state_embeds: Optional[torch.Tensor] = None, # [B, S, H] or None
    ) -> Tuple[torch.Tensor, object]:

        noisy_actions = self.x_embedder(x_t.to(torch.bfloat16)) # [B, chunk, H]
        timesteps = self.t_embedder(timestep).unsqueeze(1) # [B, 1, H]

        B = inputs_embeds.shape[0]

        # Sequence layout: [latent | state? | t | actions | tactile | t | actions]
        # (matches the Janus double-stream layout for action + tactile expert)
        # Build act_seq: optionally prepend state_embeds
        act_parts = []
        if state_embeds is not None and state_embeds.shape[1] > 0:
            act_parts.append(state_embeds)
        act_parts.extend([timesteps, noisy_actions])
        act_seq = torch.cat(act_parts, dim=1) # [B, (S+)1+chunk, H]
        tac_seq = torch.cat([tactile_embeds, timesteps, noisy_actions], dim=1) # [B, T+1+chunk, H]

        n_act = act_seq.shape[1]
        n_tac = tac_seq.shape[1]

        if past_key_values is None:
            full_embeds = torch.cat([inputs_embeds, act_seq, tac_seq], dim=1)
            L = inputs_embeds.shape[1]
            total = full_embeds.shape[1]
            latent_indexes = torch.arange(0, L, device=full_embeds.device)
            action_indexes = torch.arange(L, L + n_act, device=full_embeds.device)
            tactile_indexes = torch.arange(L + n_act, total, device=full_embeds.device)

            outputs = self.model(
                inputs_embeds = full_embeds,
                position_ids = position_ids,
                past_key_values = past_key_values,
                attention_mask = attention_mask,
                use_cache = True,
                latent_indexes = latent_indexes,
                action_indexes = action_indexes,
                tactile_indexes = tactile_indexes,
            )
        else:
            # Subsequent denoise steps: only pass the action/tactile tokens
            full_embeds = torch.cat([act_seq, tac_seq], dim=1)
            # Trim cached KVs to remove previous action/tactile entries
            drop = n_act + n_tac
            past_key_values.crop(-drop)
            total = full_embeds.shape[1]
            latent_indexes = torch.arange(0, 0,       device=full_embeds.device)
            action_indexes = torch.arange(0, n_act,   device=full_embeds.device)
            tactile_indexes = torch.arange(n_act, total, device=full_embeds.device)

            # Compute correct M-RoPE position_ids for action/tactile tokens
            # (must match positions used in the first denoise step)
            extended_pos = self.model._extend_position_ids(
                position_ids, n_act, n_tac,
            )
            act_tac_pos = extended_pos[..., -(n_act + n_tac):]

            outputs = self.model(
                inputs_embeds = full_embeds,
                position_ids = act_tac_pos,
                past_key_values = past_key_values,
                use_cache = True,
                latent_indexes = latent_indexes,
                action_indexes = action_indexes,
                tactile_indexes = tactile_indexes,
            )

        hidden = outputs.last_hidden_state
        # Extract velocity prediction from the action-expert positions
        v_t = self.final_layer(hidden)[:, -noisy_actions.shape[1]:, :]
        return v_t, outputs.past_key_values

    def forward_flow(
        self,
        inputs_embeds: torch.Tensor, # latent embeddings
        position_ids: torch.Tensor, # M-RoPE for latent
        noise: torch.Tensor, # [B, chunk, action_dim]
        tactile_inputs: torch.Tensor, # [B, T, 6] or [B, T, 1, H, W]
        num_steps: int = 10,
        attention_mask: Optional[torch.Tensor] = None,
        state_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Euler integration: x_1 (noise) → x_0 (action)."""
        device = noise.device
        dt = torch.tensor(-1.0 / num_steps, dtype=torch.bfloat16, device=device)
        x_t = noise.to(torch.bfloat16)
        time = torch.tensor(1.0, dtype=torch.bfloat16, device=device)

        if self.use_tactile_deform:
            B, n_fingers, C, H, W = tactile_inputs.shape
            deforms_flat = tactile_inputs.view(-1, C, H, W).to(inputs_embeds.dtype)
            deform_feats = self.deform_encoder(deforms_flat)
            deform_feats = deform_feats.view(B, n_fingers, -1)
            tactile_embeds = self.deform_proj(deform_feats.to(torch.bfloat16))
        else:
            tactile_embeds = self.tacf6_embedder(tactile_inputs.to(torch.bfloat16))

        past_kv = None
        while time >= -dt / 2:
            expanded_time = time.expand(x_t.shape[0])
            v_t, past_kv = self.denoise_step(
                inputs_embeds = inputs_embeds,
                position_ids = position_ids,
                past_key_values = past_kv,
                x_t = x_t,
                timestep = expanded_time,
                tactile_embeds = tactile_embeds,
                attention_mask = attention_mask,
                state_embeds = state_embeds,
            )
            x_t = x_t + dt * v_t
            time = time + dt
        return x_t

    def named_parameters(self, *args, **kwargs):
        return super().named_parameters(*args, **kwargs)

    def parameters(self, *args, **kwargs):
        return super().parameters(*args, **kwargs)

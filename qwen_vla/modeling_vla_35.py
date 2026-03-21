"""
Qwen3.5VLAModel - VLA wrapper for Qwen3.5-0.8B MoT backbone.

Key differences from Qwen3VLVLAModel:
  - Uses Qwen35ModelMoT (hybrid linear+full attention MoT backbone).
  - image_token_id = 248056  (<|image_pad|> in Qwen3.5 vocabulary).
  - get_rope_index requires mm_token_type_ids (computed from input_ids).
  - position_ids format: [3, B, L] M-RoPE part only.
  - from_pretrained_qwen35 factory method.
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache

from .modeling_qwen35_mot import Qwen35ModelMoT
from diffusion import ActionEmbedder, TimestepEmbedder, FinalLayer
from models import DeformEncoder

_DEFAULT_IMAGE_TOKEN_ID = 248056   # <|image_pad|> in Qwen3.5
_VIDEO_TOKEN_ID         = 248057   # <|video_pad|>


class _RopeIndexHelper:
    """Plain Python helper (NOT nn.Module) that wraps Qwen3VLModel.get_rope_index.

    Storing an uninitialised nn.Module as an attribute on another nn.Module
    causes `model.named_parameters()` to crash because `_parameters` is never
    set.  Using a plain object avoids registration in the module hierarchy.
    """

    def __init__(self, vl_config):
        self._vl_config = vl_config

    def get_rope_index(self, input_ids, image_grid_thw=None, attention_mask=None):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLModel
        # Bind the method to a minimal duck-typed object with just .config
        _stub = type("_Stub", (), {"config": self._vl_config})()
        return Qwen3VLModel.get_rope_index(
            _stub,
            input_ids      = input_ids,
            image_grid_thw = image_grid_thw,
            attention_mask = attention_mask,
        )


class Qwen35VLAModel(nn.Module):
    """
    Full VLA model wrapping the Qwen3.5-0.8B visual encoder and the
    three-expert (MoT) hybrid text backbone.
    """

    def __init__(
        self,
        config,
        action_dim: int = 29,
        action_chunk: int = 8,
        tacf6_dim: int = 6,
        use_tactile_deform: bool = False,
        use_robot_state: bool = False,
        image_token_id: int  = _DEFAULT_IMAGE_TOKEN_ID,
        video_token_id: int  = _VIDEO_TOKEN_ID,
    ):
        super().__init__()
        self.config = config
        self.action_dim = action_dim
        self.action_chunk = action_chunk
        self.tacf6_dim = tacf6_dim
        self.use_tactile_deform = use_tactile_deform
        self.use_robot_state = use_robot_state
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id

        self.visual = None          # set by from_pretrained_qwen35

        self.model = Qwen35ModelMoT(config)

        H = config.hidden_size
        self.x_embedder = ActionEmbedder(action_dim, H)
        self.t_embedder = TimestepEmbedder(H)
        self.final_layer = FinalLayer(H, action_dim)
        self.tacf6_embedder = ActionEmbedder(tacf6_dim, H)

        if use_tactile_deform:
            self.deform_encoder = DeformEncoder()
            self.deform_proj = ActionEmbedder(28800, H)   # 128 × 15 × 15

        if use_robot_state:
            self.state_embedder = ActionEmbedder(action_dim, H)

        self._qwen35_base = None    # set by from_pretrained_qwen35

    @classmethod
    def from_pretrained_qwen35(
        cls,
        pretrained_path: str,
        action_dim: int  = 29,
        action_chunk: int  = 8,
        tacf6_dim: int  = 6,
        use_tactile_deform: bool = False,
        use_robot_state: bool = False,
        torch_dtype = torch.bfloat16,
    ) -> "Qwen35VLAModel":
        """
        Build a Qwen35VLAModel from a pretrained Qwen3.5-0.8B checkpoint.

        Uses direct safetensors loading since transformers 4.57.x calls this
        architecture 'qwen3_next' while the checkpoint uses 'qwen3_5'.

        Steps:
          1. Read config.json manually.
          2. Build Qwen3NextConfig for the text backbone.
          3. Load safetensors and split into visual / language_model state dicts.
          4. Initialise Qwen3VLVisionModel for the visual tower.
          5. Initialise Qwen35ModelMoT for the language backbone.
          6. Create a lightweight Qwen3VLModel stub for get_rope_index.
          7. Xavier-init VLA-specific modules.
        """
        import json
        import glob as _glob
        from safetensors.torch import load_file
        from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
        from transformers.models.qwen3_vl.configuration_qwen3_vl import (
            Qwen3VLConfig, Qwen3VLVisionConfig,
        )
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            Qwen3VLVisionModel, Qwen3VLModel,
        )

        print(f"Loading Qwen3.5 base model from {pretrained_path} ...")

        # ── 1. Read config ───────────────────────────────────────────────────
        with open(os.path.join(pretrained_path, "config.json")) as f:
            full_cfg = json.load(f)

        image_token_id      = full_cfg.get("image_token_id",       _DEFAULT_IMAGE_TOKEN_ID)
        video_token_id      = full_cfg.get("video_token_id",       _VIDEO_TOKEN_ID)
        vision_start_tok_id = full_cfg.get("vision_start_token_id", 248053)

        # ── 2. Build Qwen3NextConfig for text backbone ───────────────────────
        tc = dict(full_cfg.get("text_config", full_cfg))
        rope_params = tc.pop("rope_parameters", {})
        # Map rope_parameters → fields Qwen3NextConfig understands
        tc.setdefault("rope_theta",          rope_params.get("rope_theta", 10_000_000))
        tc.setdefault("partial_rotary_factor", rope_params.get("partial_rotary_factor", 0.25))
        # Remove fields that don't belong in Qwen3NextConfig
        for bad_key in ("model_type", "architectures", "auto_map",
                        "transformers_version", "dtype", "mamba_ssm_dtype",
                        "attn_output_gate", "mtp_num_hidden_layers",
                        "mtp_use_dedicated_embeddings"):
            tc.pop(bad_key, None)
        text_config = Qwen3NextConfig(**tc)
        # Store rope_parameters on config so Qwen35FullAttentionMoT can read mrope_section
        text_config.rope_parameters = rope_params
        # Set dtype as a torch.dtype so Qwen3NextGatedDeltaNet doesn't call
        # torch.get_current_dtype() which was only added in PyTorch 2.7
        text_config.dtype = torch_dtype

        # ── 3. Instantiate VLA shell ─────────────────────────────────────────
        vla = cls(
            config           = text_config,
            action_dim       = action_dim,
            action_chunk     = action_chunk,
            tacf6_dim        = tacf6_dim,
            use_tactile_deform = use_tactile_deform,
            use_robot_state  = use_robot_state,
            image_token_id   = image_token_id,
            video_token_id   = video_token_id,
        )

        # ── 4. Load safetensors ──────────────────────────────────────────────
        st_files = sorted(_glob.glob(os.path.join(pretrained_path, "*.safetensors")))
        state_dict = {}
        for sf in st_files:
            state_dict.update(load_file(sf, device="cpu"))

        # ── 5. Visual tower ──────────────────────────────────────────────────
        # Always construct the visual tower from config so that its parameters
        # are registered.  When loading from a finetuned checkpoint (model.pt
        # only, no safetensors), the visual weights will be loaded later by
        # load_state_dict(model.pt) instead of from the base safetensors.
        try:
            vis_cfg_dict = dict(full_cfg.get("vision_config", {}))
            vis_cfg_dict.pop("model_type", None)
            vis_cfg = Qwen3VLVisionConfig(**vis_cfg_dict)
            vla.visual = Qwen3VLVisionModel(vis_cfg)

            vis_sd = {k[len("model.visual."):]: v
                      for k, v in state_dict.items() if k.startswith("model.visual.")}
            if vis_sd:
                miss, unexp = vla.visual.load_state_dict(vis_sd, strict=False)
                print(f"  Visual tower: {len(vis_sd)} tensors loaded "
                      f"(missing={len(miss)}, unexpected={len(unexp)})")
            else:
                print(f"  Visual tower: created from config "
                      f"(weights will be loaded from model.pt)")
        except Exception as e:
            print(f"  Warning: visual tower init failed: {e}")
            vla.visual = None

        # ── 6. Language model ────────────────────────────────────────────────
        lm_sd = {k[len("model.language_model."):]: v
                 for k, v in state_dict.items() if k.startswith("model.language_model.")}
        if lm_sd:
            vla.model = Qwen35ModelMoT.from_pretrained_weights(
                text_config, lm_sd, torch_dtype
            )
            print(f"  Language model: {len(lm_sd)} tensors loaded.")

        # ── 7. M-RoPE helper (plain Python, NOT nn.Module) ──────────────────
        try:
            vis_cfg_dict2 = dict(full_cfg.get("vision_config", {}))
            vis_cfg_dict2.pop("model_type", None)
            vl_rope_cfg = Qwen3VLConfig(
                vision_config         = vis_cfg_dict2,
                image_token_id        = image_token_id,
                video_token_id        = video_token_id,
                vision_start_token_id = vision_start_tok_id,
            )
            # Use _RopeIndexHelper (not nn.Module) to avoid parameter traversal errors
            vla._qwen35_base = _RopeIndexHelper(vl_rope_cfg)
            print("  M-RoPE helper ready.")
        except Exception as e:
            print(f"  Warning: M-RoPE helper failed ({e}). Fallback to sequential pos_ids.")
            vla._qwen35_base = None

        # ── 8. Xavier-init VLA-specific modules ──────────────────────────────
        vla._init_vla_weights()
        return vla

    def _init_vla_weights(self):
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
        # zero-init final output so action predictions start small
        nn.init.zeros_(self.final_layer.mlp.fc2.weight)
        if self.final_layer.mlp.fc2.bias is not None:
            nn.init.zeros_(self.final_layer.mlp.fc2.bias)

    def load_deform_encoder_weights(self, ckpt_path: str):
        if not self.use_tactile_deform or not os.path.exists(ckpt_path):
            return
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict", ckpt)
        enc_sd = {}
        for k, v in sd.items():
            nk = k[len("encoder."):] if k.startswith("encoder.") else k
            if nk in self.deform_encoder.state_dict():
                enc_sd[nk] = v
        self.deform_encoder.load_state_dict(enc_sd, strict=False)
        print(f"DeformEncoder weights loaded from {ckpt_path}")

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build the VLM input embeddings, injecting visual features at image-pad positions."""
        inputs_embeds = self.model.get_input_embeddings()(input_ids)

        if pixel_values is not None and self.visual is not None:
            pixel_values  = pixel_values.to(inputs_embeds.device, dtype=inputs_embeds.dtype)
            out = self.visual(pixel_values, grid_thw=image_grid_thw)
            # Qwen3VLVisionModel returns (hidden_states, deepstack_feature_lists)
            image_features = out[0] if isinstance(out, (tuple, list)) else out
            image_mask = (input_ids == self.image_token_id)
            if image_mask.any():
                inputs_embeds[image_mask] = image_features.to(inputs_embeds.dtype)

        return inputs_embeds

    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        image_grid_thw: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute M-RoPE position_ids for the latent (VLM) tokens.
        Returns (position_ids [3, B, L], mrope_position_deltas).
        """
        if self._qwen35_base is not None and hasattr(self._qwen35_base, "get_rope_index"):
            return self._qwen35_base.get_rope_index(
                input_ids      = input_ids,
                image_grid_thw = image_grid_thw,
                attention_mask = attention_mask,
            )

        # Fallback: plain sequential
        B, L = input_ids.shape
        pos = torch.arange(L, device=input_ids.device).view(1, 1, -1).expand(3, B, -1)
        return pos, None

    def denoise_step(
        self,
        inputs_embeds: torch.Tensor,   # [B, L_lat, H]
        position_ids: torch.Tensor,   # [3, B, L_lat]
        attention_mask: Optional[torch.Tensor],
        past_key_values,
        x_t: torch.Tensor,   # [B, chunk, action_dim]
        timestep: torch.Tensor,   # [B]
        tactile_embeds: torch.Tensor,   # [B, T, H]
        state_embeds: Optional[torch.Tensor] = None,  # [B, action_dim, H] or None
    ) -> Tuple[torch.Tensor, object]:

        noisy_actions = self.x_embedder(x_t.to(torch.bfloat16))
        timesteps     = self.t_embedder(timestep).unsqueeze(1)

        # Action expert: [state | timestep | noisy_actions]
        act_parts = []
        if state_embeds is not None and state_embeds.shape[1] > 0:
            act_parts.append(state_embeds)
        act_parts.extend([timesteps, noisy_actions])
        act_seq = torch.cat(act_parts, dim=1)

        tac_seq = torch.cat([tactile_embeds, timesteps, noisy_actions], dim=1)

        n_act = act_seq.shape[1]
        n_tac = tac_seq.shape[1]

        full_embeds = torch.cat([inputs_embeds, act_seq, tac_seq], dim=1)
        L = inputs_embeds.shape[1]

        full_pos = Qwen35ModelMoT._extend_position_ids(
            position_ids, n_act, n_tac, attention_mask
        )

        outputs = self.model(
            inputs_embeds   = full_embeds,
            position_ids    = full_pos,
            attention_mask  = attention_mask,
            latent_indexes  = torch.arange(L,          device=full_embeds.device),
            action_indexes  = torch.arange(L, L+n_act, device=full_embeds.device),
            tactile_indexes = torch.arange(L+n_act, L+n_act+n_tac, device=full_embeds.device),
            use_cache       = False,
            return_dict     = True,
        )
        hidden = outputs.last_hidden_state
        v_t = self.final_layer(hidden)[:, -noisy_actions.shape[1]:, :]
        return v_t, None

    def forward_flow(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        noise: torch.Tensor,   # [B, chunk, action_dim]
        tactile_inputs: torch.Tensor,
        num_steps: int = 10,
        state_embeds: Optional[torch.Tensor] = None,  # [B, action_dim, H] or None
        use_prefix_cache: bool = True,
    ) -> torch.Tensor:
        device = noise.device
        dt  = torch.tensor(-1.0 / num_steps, dtype=torch.bfloat16, device=device)
        x_t = noise.to(torch.bfloat16)
        t   = torch.tensor(1.0, dtype=torch.bfloat16, device=device)

        if self.use_tactile_deform:
            B, nf, C, H, W = tactile_inputs.shape
            enc_dtype = next(self.deform_encoder.parameters()).dtype
            feats = self.deform_encoder(tactile_inputs.view(-1, C, H, W).to(enc_dtype))
            tactile_embeds = self.deform_proj(feats.view(B, nf, -1).to(torch.bfloat16))
        else:
            tactile_embeds = self.tacf6_embedder(tactile_inputs.to(torch.bfloat16))

        if not use_prefix_cache:
            # Fallback: no caching (original slow path)
            while t >= -dt / 2:
                v_t, _ = self.denoise_step(inputs_embeds, position_ids, attention_mask,
                                            None, x_t, t.expand(x_t.shape[0]),
                                            tactile_embeds, state_embeds)
                x_t = x_t + dt * v_t
                t   = t + dt
            return x_t

        # ── Cached path: process latent prefix once, suffix per step ─────────
        L_prefix = inputs_embeds.shape[1]

        # Cache the latent prefix (frozen across all denoise steps)
        prefix_cache = self.model.forward_prefix(
            inputs_embeds, position_ids, attention_mask)

        while t >= -dt / 2:
            noisy_actions = self.x_embedder(x_t.to(torch.bfloat16))
            timesteps     = self.t_embedder(t.expand(x_t.shape[0])).unsqueeze(1)

            # Build suffix: [state | t | actions | tactile | t | actions]
            act_parts = []
            if state_embeds is not None and state_embeds.shape[1] > 0:
                act_parts.append(state_embeds)
            act_parts.extend([timesteps, noisy_actions])
            act_seq = torch.cat(act_parts, dim=1)
            tac_seq = torch.cat([tactile_embeds, timesteps, noisy_actions], dim=1)
            suffix_embeds = torch.cat([act_seq, tac_seq], dim=1)

            n_act = act_seq.shape[1]
            n_tac = tac_seq.shape[1]

            # Position embeddings for suffix
            suffix_pos_ids = Qwen35ModelMoT._extend_position_ids(
                position_ids, n_act, n_tac, attention_mask)
            # Take only the suffix portion: [3, B, n_act + n_tac]
            suffix_pos_ids = suffix_pos_ids[:, :, L_prefix:]
            suffix_pos_emb = self.model.rotary_emb(suffix_embeds, suffix_pos_ids)

            suffix_out = self.model.forward_suffix(
                suffix_embeds   = suffix_embeds,
                prefix_cache    = prefix_cache,
                suffix_pos_emb  = suffix_pos_emb,
                attention_mask  = attention_mask,
                L_act           = n_act,
                L_prefix        = L_prefix,
            )

            v_t = self.final_layer(suffix_out)[:, -noisy_actions.shape[1]:, :]
            x_t = x_t + dt * v_t
            t   = t + dt

        return x_t

"""
Qwen3VLVLAModel – the full VLA model.

Architecture
────────────
  visual             : Qwen3VL ViT (frozen during visual feature extraction)
  model              : Qwen3VLModelMoT – three-expert Qwen3-VL text backbone
  x_embedder         : noisy actions  → hidden_size
  t_embedder         : flow time      → hidden_size
  tacf6_embed        : F6 tactile vec → hidden_size   (optional)
  deform_proj        : tactile deform → hidden_size   (optional)
  state_embed        : robot state    → hidden_size   (optional)
  final_layer        : hidden_size    → action_dim    (action expert velocity)
  final_layer_tactile: hidden_size    → action_dim    (tactile expert velocity)

Cascaded flow matching
──────────────────────
The action expert handles the upper segment τ ∈ [τ_split, 1] and produces an
intermediate state x_split plus cached [latent | action] KV.  The tactile
expert continues the same flow on the lower segment τ ∈ [0, τ_split] from
x_split using the cached KV plus fresh tactile observations, and its
integrated output IS the executed action chunk (no Â + Δa addition).
Both experts predict the same velocity target ε − A_demo on disjoint
sub-intervals; the action expert is additionally trained on the full [0, 1]
range so it can run standalone if tactile drops out.

Methods:
  forward_flow_action_partial    — slow tick (action expert, upper segment).
  tactile_flow_continue          — fast tick (tactile expert, lower segment).
  tactile_flow_train_step        — training-time single tactile forward.
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
        use_tactile_code:        bool = False,
        vqvae_codebook_size:     int  = 64,
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
        self.use_tactile_code         = use_tactile_code
        self.vqvae_codebook_size      = vqvae_codebook_size

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

        # VQ-VAE tactile code embedder (one code per hand → 2 tokens).
        # Created only when explicitly enabled so toggling --use_tactile_code 0
        # gives a graph identical to the pre-feature checkpoint.
        if use_tactile_code:
            self.tactile_code_embedder = nn.Embedding(vqvae_codebook_size, H)

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
        use_tactile_code:         bool = False,
        vqvae_codebook_size:      int  = 64,
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
            use_tactile_code = use_tactile_code,
            vqvae_codebook_size = vqvae_codebook_size,
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
        """Xavier-init all new VLA-specific linear layers.

        Under cascaded flow matching both `final_layer` (action expert) and
        `final_layer_tactile` (tactile expert) are velocity predictors on
        disjoint sub-intervals of the same flow.  Only `final_layer` is
        zero-inited so v_act starts at zero; the tactile head must start
        non-trivial so the lower-segment refinement signal is alive at step 0.
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
        if self.use_tactile_code:
            nn.init.normal_(self.tactile_code_embedder.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.final_layer.mlp.fc2.weight)
        if self.final_layer.mlp.fc2.bias is not None:
            nn.init.zeros_(self.final_layer.mlp.fc2.bias)
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

    def _embed_tactile_observations(
        self,
        tactile_f6: Optional[torch.Tensor],
        tactile_deform: Optional[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype,
        tactile_codes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build [B, n_obs, H] tactile observation embedding (codes + vec + deform).

        When `tactile_codes` is provided ([B, 2] int64) and the model has the
        embedder, two code tokens are prepended to the f6 / deform tokens.
        """
        tac_parts = []
        if (tactile_codes is not None
                and self.use_tactile_code
                and tactile_codes.shape[1] > 0):
            code_emb = self.tactile_code_embedder(tactile_codes.to(device).long())
            tac_parts.append(code_emb.to(dtype))
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
                    "tactile_flow_continue.")
        return new_cache

    # ──────────────────────────────────────────────────────────────────────
    # Cascaded flow matching
    #
    # The action expert handles the upper segment τ ∈ [τ_split, 1] and the
    # tactile expert continues on the lower segment τ ∈ [0, τ_split].  Both
    # are velocity predictors for the same action-distribution flow target
    # u = ε − A_demo on disjoint sub-intervals, so the tactile expert's
    # integrated output IS the clean action — no Â + Δa addition.  The
    # action expert is still trained on the full τ ∈ [0, 1] range so it
    # can run standalone if tactile drops out.
    # ──────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def forward_flow_action_full(
        self,
        inputs_embeds: torch.Tensor,            # [B, L_latent, H]
        position_ids: torch.Tensor,             # [3, B, L_latent]
        noise: torch.Tensor,                    # [B, n_chunk, action_dim]
        attention_mask: Optional[torch.Tensor] = None,
        state_embeds: Optional[torch.Tensor] = None,
        fast_embeds: Optional[torch.Tensor] = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        """Action-expert-only full flow τ ∈ [0, 1].

        Tactile-blind baseline used for the "without tactile expert" ablation.
        The action expert is trained on the full τ ∈ [0, 1] range so it can
        integrate the whole trajectory standalone — this method runs all
        `num_steps` Euler steps using only the action expert and returns the
        clean action chunk directly.  No KV cache is returned because there
        is no downstream tactile pass.

        Returns
        -------
        clean_chunk : [B, n_chunk, action_dim]  the integrated action at τ=0.
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
        for _ in range(num_steps):
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

        return x_t

    @torch.no_grad()
    def forward_flow_action_partial(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        noise: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        state_embeds: Optional[torch.Tensor] = None,
        fast_embeds: Optional[torch.Tensor] = None,
        num_steps_total: int = 10,
        split_step: int = 6,
        refresh_clean_kv: bool = True,
    ) -> Tuple[torch.Tensor, "DynamicCache", int, float]:
        """Cascaded slow-tick: run the action expert for `split_step` of
        `num_steps_total` Euler steps, stopping at τ = 1 − split_step/num_steps_total.

        Returns
        -------
        x_split           : [B, n_chunk, action_dim]  partially-denoised chunk
                                                       at τ = τ_split.
        cached_kv         : DynamicCache with [latent KV | action KV at τ_split].
        n_action_in_cache : int
        tau_split         : float                      ∈ (0, 1)
        """
        if not (0 < split_step < num_steps_total):
            raise ValueError(
                f"split_step must be in (0, num_steps_total); got "
                f"{split_step}/{num_steps_total}.")

        device = noise.device
        dtype  = torch.bfloat16
        # Same dt as the full flow, so the cascaded trajectory is consistent
        # with what a 10-step monolithic flow would integrate.
        dt     = torch.tensor(-1.0 / num_steps_total, dtype=dtype, device=device)
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
        for i in range(split_step):
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

        # `time` is now τ_split.  Refresh KV with the partially-denoised state
        # at τ_split so the tactile expert attends to a coherent action context.
        if refresh_clean_kv and past_kv is not None:
            past_kv.crop(-n_act)
            split_timesteps = self.t_embedder(
                time.expand(B)
            ).unsqueeze(1)
            split_actions = self.x_embedder(x_t)
            clean_parts = [fast_embeds]
            if n_state > 0:
                clean_parts.append(state_embeds)
            clean_parts += [split_timesteps, split_actions]
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

        tau_split = float(time.item())
        return x_t, past_kv, n_action_in_cache, tau_split

    @torch.no_grad()
    def tactile_flow_continue(
        self,
        cached_kv,
        latent_position_ids: torch.Tensor,
        n_action_in_cache: int,
        x_split: torch.Tensor,                  # [B, n_chunk, action_dim] at τ=τ_split
        tau_split: float,
        tactile_f6: Optional[torch.Tensor] = None,
        tactile_deform: Optional[torch.Tensor] = None,
        tactile_codes: Optional[torch.Tensor] = None,
        num_steps_total: int = 10,
        split_step: int = 6,
    ) -> torch.Tensor:
        """Cascaded fast-tick: continue the flow from x_split at τ=tau_split
        down to τ=0 using the tactile expert.  Returns the final clean action
        chunk (NOT a residual — the tactile expert here is a velocity predictor
        for the action distribution itself, restricted to τ ∈ [0, τ_split]).

        `cached_kv` is cloned so multiple fast ticks within one chunk window
        each start from the same slow-tick snapshot.
        """
        cache = self._clone_dynamic_cache(cached_kv)
        device = x_split.device
        dtype  = torch.bfloat16
        B      = x_split.shape[0]
        n_chunk, action_dim = x_split.shape[1], x_split.shape[2]

        tac_obs = self._embed_tactile_observations(
            tactile_f6, tactile_deform, device, dtype,
            tactile_codes=tactile_codes)
        n_obs = tac_obs.shape[1]
        n_tac_seq = n_obs + 1 + n_chunk

        extended_pos = self.model._extend_position_ids(
            latent_position_ids, n_action_in_cache, n_tac_seq)
        tac_pos = extended_pos[..., -n_tac_seq:]

        # Same dt as the action expert used in the slow tick — the cascaded
        # trajectory's integration step matches what a monolithic 10-step flow
        # would use.  `remaining` steps complete the integration.
        dt   = torch.tensor(-1.0 / num_steps_total, dtype=dtype, device=device)
        time = torch.tensor(tau_split, dtype=dtype, device=device)
        remaining = num_steps_total - split_step
        x = x_split.to(dtype)

        for step in range(remaining):
            tau_emb = self.t_embedder(time.expand(B)).unsqueeze(1)
            x_emb   = self.x_embedder(x)
            full_embeds = torch.cat([tac_obs, tau_emb, x_emb], dim=1)
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
            v = self.final_layer_tactile(hidden[:, -n_chunk:, :])
            x    = x + dt * v
            time = time + dt

        return x

    def tactile_flow_train_step(
        self,
        cached_kv,
        latent_position_ids: torch.Tensor,
        n_action_in_cache: int,
        x_tau: torch.Tensor,                    # [B, n_chunk, action_dim], full action state
        tau: torch.Tensor,                      # [B] flow times in [0, tau_split]
        tactile_f6: Optional[torch.Tensor] = None,
        tactile_deform: Optional[torch.Tensor] = None,
        tactile_codes: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single tactile-only forward at (x_τ, τ) for cascaded training.

        Returns predicted velocity v_pred [B, n_chunk, action_dim] suitable for
        L_flow_tactile = MSE(v_pred, ε − A_demo).  Same target as the action
        expert's L_flow loss — both experts are trained as velocity predictors
        for the same flow trajectory, just on disjoint τ ranges.
        """
        device = x_tau.device
        dtype  = torch.bfloat16
        B      = x_tau.shape[0]
        n_chunk = x_tau.shape[1]

        tac_obs = self._embed_tactile_observations(
            tactile_f6, tactile_deform, device, dtype,
            tactile_codes=tactile_codes)
        n_obs = tac_obs.shape[1]

        tau_emb = self.t_embedder(tau.to(dtype)).unsqueeze(1)            # [B, 1, H]
        x_emb   = self.x_embedder(x_tau.to(dtype))                       # [B, n_chunk, H]
        full_embeds = torch.cat([tac_obs, tau_emb, x_emb], dim=1)
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

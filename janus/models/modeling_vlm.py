# Copyright (c) 2023-2024 DeepSeek.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import torch
from attrdict import AttrDict
from einops import rearrange
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.cache_utils import DynamicCache

from janus.models.clip_encoder import CLIPVisionTower
from janus.models.projector import MlpProjector
from janus.diffusion import ActionEmbedder, TimestepEmbedder, FinalLayer
from janus.uni3d import Uni3D
from janus.models.DeformAE import DeformEncoder
import torch.nn as nn

class vision_head(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.output_mlp_projector = torch.nn.Linear(
            params.n_embed, params.image_token_embed
        )
        self.vision_activation = torch.nn.GELU()
        self.vision_head = torch.nn.Linear(
            params.image_token_embed, params.image_token_size
        )

    def forward(self, x):
        x = self.output_mlp_projector(x)
        x = self.vision_activation(x)
        x = self.vision_head(x)
        return x

def model_name_to_cls(cls_name):
    if "MlpProjector" in cls_name:
        cls = MlpProjector

    elif "CLIPVisionTower" in cls_name:
        cls = CLIPVisionTower

    elif "VQ" in cls_name:
        from janus.models.vq_model import VQ_models

        cls = VQ_models[cls_name]
    elif "vision_head" in cls_name:
        cls = vision_head
    else:
        raise ValueError(f"class_name {cls_name} is invalid.")

    return cls


class VisionConfig(PretrainedConfig):
    model_type = "vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class AlignerConfig(PretrainedConfig):
    model_type = "aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenVisionConfig(PretrainedConfig):
    model_type = "gen_vision"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenAlignerConfig(PretrainedConfig):
    model_type = "gen_aligner"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))


class GenHeadConfig(PretrainedConfig):
    model_type = "gen_head"
    cls: str = ""
    params: AttrDict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cls = kwargs.get("cls", "")
        if not isinstance(self.cls, str):
            self.cls = self.cls.__name__

        self.params = AttrDict(kwargs.get("params", {}))

class MLPProjector(nn.Module):
    def __init__(self, vision_dim: int, llm_dim: int, mlp_type: str = "gelu-mlp") -> None:
        super().__init__()
        if mlp_type == "gelu-mlp":
            self.projector = nn.Sequential(
                nn.Linear(vision_dim, llm_dim, bias=True),
                nn.GELU(),
                nn.Linear(llm_dim, llm_dim, bias=True),
            )
        else:
            raise ValueError(f"Projector with `{mlp_type = }` is not supported!")

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        return self.projector(img_patches)
    
    def initialize_weights(self):
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class MultiModalityConfig(PretrainedConfig):
    model_type = "multi_modality"
    vision_config: VisionConfig
    aligner_config: AlignerConfig

    gen_vision_config: GenVisionConfig
    gen_aligner_config: GenAlignerConfig
    gen_head_config: GenHeadConfig

    language_config: LlamaConfig

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        vision_config = kwargs.get("vision_config", {})
        self.vision_config = VisionConfig(**vision_config)

        aligner_config = kwargs.get("aligner_config", {})
        self.aligner_config = AlignerConfig(**aligner_config)

        gen_vision_config = kwargs.get("gen_vision_config", {})
        self.gen_vision_config = GenVisionConfig(**gen_vision_config)

        gen_aligner_config = kwargs.get("gen_aligner_config", {})
        self.gen_aligner_config = GenAlignerConfig(**gen_aligner_config)

        gen_head_config = kwargs.get("gen_head_config", {})
        self.gen_head_config = GenHeadConfig(**gen_head_config)

        language_config = kwargs.get("language_config", {})
        if isinstance(language_config, LlamaConfig):
            self.language_config = language_config
        else:
            self.language_config = LlamaConfig(**language_config)


class MultiModalityPreTrainedModel(PreTrainedModel):
    config_class = MultiModalityConfig
    base_model_prefix = "multi_modality"
    _no_split_modules = []
    _skip_keys_device_placement = "past_key_values"


class MultiModalityCausalLM(MultiModalityPreTrainedModel):
    def __init__(self, config: MultiModalityConfig,
                flow = False,
                use_latent = True,
                robot_state = False,
                fast_and_slow = False,
                fast_image_num = 1,
                action_dim = 58,
                action_chunk = 32,
                tacf6_dim = 6,
                use_tactile_deform = False,
            ):
        super().__init__(config)
        self.fast_and_slow = fast_and_slow
        self.fast_image_num = fast_image_num
        self.use_latent = use_latent
        self.robot_state = robot_state
        self.action_dim = action_dim
        self.action_chunk = action_chunk
        self.tacf6_dim = tacf6_dim
        self.use_tactile_deform = use_tactile_deform

        vision_config = config.vision_config
        vision_cls = model_name_to_cls(vision_config.cls)
        self.vision_model = vision_cls(**vision_config.params)

        aligner_config = config.aligner_config
        aligner_cls = model_name_to_cls(aligner_config.cls)
        self.aligner = aligner_cls(aligner_config.params)

        gen_vision_config = config.gen_vision_config
        gen_vision_cls = model_name_to_cls(gen_vision_config.cls)
        self.gen_vision_model = gen_vision_cls()
        self.gen_vision_config = gen_vision_config

        gen_aligner_config = config.gen_aligner_config
        gen_aligner_cls = model_name_to_cls(gen_aligner_config.cls)
        self.gen_aligner = gen_aligner_cls(gen_aligner_config.params)

        gen_head_config = config.gen_head_config
        gen_head_cls = model_name_to_cls(gen_head_config.cls)
        self.gen_head = gen_head_cls(gen_head_config.params)

        self.gen_embed = torch.nn.Embedding(
            gen_vision_config.params.image_token_size, gen_vision_config.params.n_embed
        )

        language_config = config.language_config
        language_config.torch_dtype = torch.bfloat16
        language_config.bf16 = True
        self.language_model = LlamaForCausalLM(language_config)

        # make config like a llm config
        for key, value in language_config.__dict__.items():
            if key not in self.config.__dict__:
                setattr(self.config, key, value)

        self.tacf6_embedder = ActionEmbedder(action_size=tacf6_dim, hidden_size=language_config.hidden_size)
        if self.use_tactile_deform:
            self.deform_encoder = DeformEncoder()
            self.deform_proj = ActionEmbedder(action_size=28800, hidden_size=language_config.hidden_size)

        self.x_embedder = ActionEmbedder(action_size=action_dim, hidden_size=language_config.hidden_size)
        if self.robot_state:
            self.state_embedder = ActionEmbedder(action_size=action_dim, hidden_size=language_config.hidden_size)
        self.t_embedder = TimestepEmbedder(language_config.hidden_size)
        self.final_layer = FinalLayer(language_config.hidden_size, action_dim)

    def load_deform_encoder_weights(self, ckpt_path: str):
        if not self.use_tactile_deform:
            return

        if not os.path.exists(ckpt_path):
            print(f"Warning: DeformEncoder checkpoint not found at {ckpt_path}")
            return

        print(f"Loading DeformEncoder weights from {ckpt_path} ...")
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get("state_dict", checkpoint)
        
        encoder_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                new_k = k.replace('encoder.', '', 1)
                encoder_state_dict[new_k] = v
            elif k in self.deform_encoder.state_dict():
                encoder_state_dict[k] = v

        missing_keys, unexpected_keys = self.deform_encoder.load_state_dict(encoder_state_dict, strict=False)
        print(f"DeformEncoder weights loaded successfully.")
        if len(missing_keys) > 0:
            print(f"Missing keys (DeformEncoder): {missing_keys}")

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0) 
                nn.init.constant_(module.bias, 0)     
        self.apply(_basic_init)
    
    def denoise_step(self, inputs_embeds, past_key_values, x_t, timestep, tactile_embeds):
        noisy_actions = self.x_embedder(x_t.to(torch.bfloat16))
        timesteps = self.t_embedder(timestep).unsqueeze(1)

        if past_key_values is None:
            inputs_embeds = torch.cat([
                inputs_embeds,
                timesteps,
                noisy_actions,
                tactile_embeds,
                timesteps,
                noisy_actions,
            ], dim=1)
            action_len=578 * self.fast_image_num + 2 + self.action_chunk + 6 + self.action_chunk + self.action_dim
            tactile_len=self.action_chunk + 11
            latent_indexes=torch.arange(0, inputs_embeds.shape[1] - action_len).to(inputs_embeds.device)
            action_indexes=torch.arange(inputs_embeds.shape[1] - action_len, inputs_embeds.shape[1] - tactile_len).to(inputs_embeds.device)
            tactile_indexes=torch.arange(inputs_embeds.shape[1] - tactile_len, inputs_embeds.shape[1]).to(inputs_embeds.device)
        else:
            inputs_embeds = torch.cat([timesteps, noisy_actions, tactile_embeds, timesteps, noisy_actions], dim=1)
            past_key_values = tuple(
                (k[:, :, :-(timesteps.shape[1]+noisy_actions.shape[1]+tactile_embeds.shape[1]+timesteps.shape[1]+noisy_actions.shape[1]), :], 
                 v[:, :, :-(timesteps.shape[1]+noisy_actions.shape[1]+tactile_embeds.shape[1]+timesteps.shape[1]+noisy_actions.shape[1]), :]) 
                 for k, v in past_key_values
            )
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            latent_indexes=torch.arange(0, 0).to(inputs_embeds.device)
            action_indexes=torch.arange(0, inputs_embeds.shape[1] - self.action_chunk - 6).to(inputs_embeds.device)
            tactile_indexes=torch.arange(inputs_embeds.shape[1] - self.action_chunk - 6, inputs_embeds.shape[1]).to(inputs_embeds.device)

        outputs = self.language_model.model(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            latent_indexes=latent_indexes,
            action_indexes=action_indexes,
            tactile_indexes=tactile_indexes,
            use_latent=self.use_latent,
            return_dict=True,
            use_cache=True
        )
        hidden_states = outputs.last_hidden_state
        v_t = self.final_layer(hidden_states)[:, -noisy_actions.shape[1]:, :]
        return v_t, outputs.past_key_values


    def forward_flow(self, inputs_embeds, noise, tactile_inputs, num_steps=10):
        noisy_actions = self.x_embedder(noise.to(torch.bfloat16))
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.bfloat16, device=noisy_actions.device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.bfloat16, device=noisy_actions.device)
        past_key_values = None

        if self.use_tactile_deform:
            B, num_fingers, C, H, W = tactile_inputs.shape
            deforms_flat = tactile_inputs.view(-1, C, H, W)
            deform_features = self.deform_encoder(deforms_flat.to(dtype=inputs_embeds.dtype)) 
            deform_features = deform_features.view(B, num_fingers, -1)
            tactile_embeds = self.deform_proj(deform_features.to(torch.bfloat16))
        else:    
            tactile_embeds = self.tacf6_embedder(tactile_inputs.to(torch.bfloat16))

        while time >= -dt / 2:
            expanded_time = time.expand(noisy_actions.shape[0])
            v_t, past_key_values = self.denoise_step(
                inputs_embeds,
                past_key_values,
                x_t,
                expanded_time,
                tactile_embeds,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t
    

    def forward_flow_rtc(self, inputs_embeds, noise, tactile_f6, num_steps=10, frozen_prefix=None, delay_steps=0):
        noisy_actions = self.x_embedder(noise.to(torch.bfloat16))
        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.bfloat16, device=noisy_actions.device)
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.bfloat16, device=noisy_actions.device)
        past_key_values = None

        tactile_embeds = self.tacf6_embedder(tactile_f6.to(torch.bfloat16))

        while time >= -dt / 2:
            expanded_time = time.expand(noisy_actions.shape[0])
            v_t, past_key_values = self.denoise_step(
                inputs_embeds,
                past_key_values,
                x_t,
                expanded_time,
                tactile_embeds,
            )
            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt

            if frozen_prefix is not None and delay_steps > 0:
                t_next = time.item()
                if t_next > 0:
                    prefix_noise = torch.randn_like(frozen_prefix)
                    x_known_t = t_next * prefix_noise + (1.0 - t_next) * frozen_prefix
                    x_t[:, :delay_steps, :] = x_known_t.to(x_t.dtype)
                else:
                    x_t[:, :delay_steps, :] = frozen_prefix.to(x_t.dtype)
        return x_t
    
    def initialize_weights(self):
        print("init!!!")
        nn.init.normal_(self.x_embedder.mlp.fc1.weight, std=0.02)
        nn.init.normal_(self.x_embedder.mlp.fc2.weight, std=0.02)
        nn.init.constant_(self.x_embedder.mlp.fc1.bias, 0)
        nn.init.constant_(self.x_embedder.mlp.fc2.bias, 0)

        if self.robot_state:
            nn.init.normal_(self.state_embedder.mlp.fc1.weight, std=0.02)
            nn.init.normal_(self.state_embedder.mlp.fc2.weight, std=0.02)
            nn.init.constant_(self.state_embedder.mlp.fc1.bias, 0)
            nn.init.constant_(self.state_embedder.mlp.fc2.bias, 0)

        nn.init.normal_(self.tacf6_embedder.mlp.fc1.weight, std=0.02)
        nn.init.normal_(self.tacf6_embedder.mlp.fc2.weight, std=0.02)
        nn.init.constant_(self.tacf6_embedder.mlp.fc1.bias, 0)
        nn.init.constant_(self.tacf6_embedder.mlp.fc2.bias, 0)

        if self.use_tactile_deform:
            nn.init.normal_(self.deform_proj.mlp.fc1.weight, std=0.02)
            nn.init.normal_(self.deform_proj.mlp.fc2.weight, std=0.02)
            nn.init.constant_(self.deform_proj.mlp.fc1.bias, 0)
            nn.init.constant_(self.deform_proj.mlp.fc2.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.constant_(self.t_embedder.mlp[0].bias, 0)
        nn.init.constant_(self.t_embedder.mlp[2].bias, 0)

        nn.init.normal_(self.final_layer.mlp.fc1.weight, std=0.02)
        nn.init.constant_(self.final_layer.mlp.fc1.bias, 0)
        nn.init.constant_(self.final_layer.mlp.fc2.weight, 0)
        nn.init.constant_(self.final_layer.mlp.fc2.bias, 0)

    def prepare_inputs_embeds(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """
        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        # [b x n, T2, D]
        images_embeds = self.aligner(self.vision_model(images))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds
    
    def prepare_inputs_embeds_gen(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.LongTensor, # should be image_ids
        images_seq_mask: torch.LongTensor,
        images_emb_mask: torch.LongTensor,
        **kwargs,
    ):
        """

        Args:
            input_ids (torch.LongTensor): [b, T]
            pixel_values (torch.FloatTensor):   [b, n_images, 3, h, w]
            images_seq_mask (torch.BoolTensor): [b, T]
            images_emb_mask (torch.BoolTensor): [b, n_images, n_image_tokens]

            assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        Returns:
            input_embeds (torch.Tensor): [b, T, D]
        """
        assert torch.sum(images_seq_mask) == torch.sum(images_emb_mask)

        bs, n = pixel_values.shape[0:2]
        images = rearrange(pixel_values, "b n c h w -> (b n) c h w")
        
        # use vqgan as image encoder
        _, _, info = self.gen_vision_model.encode(images)
        image_ids = info[2].reshape(bs*n, -1)
        images_embeds = self.gen_aligner(self.gen_embed(image_ids))

        # [b x n, T2, D] -> [b, n x T2, D]
        images_embeds = rearrange(images_embeds, "(b n) t d -> b (n t) d", b=bs, n=n)
        # [b, n, T2] -> [b, n x T2]
        images_emb_mask = rearrange(images_emb_mask, "b n t -> b (n t)")

        # [b, T, D]
        input_ids[input_ids < 0] = 0  # ignore the image embeddings
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace with the image embeddings
        inputs_embeds[images_seq_mask] = images_embeds[images_emb_mask]

        return inputs_embeds, image_ids

    def prepare_gen_img_embeds(self, image_ids: torch.LongTensor):
        return self.gen_aligner(self.gen_embed(image_ids))


AutoConfig.register("vision", VisionConfig)
AutoConfig.register("aligner", AlignerConfig)
AutoConfig.register("gen_vision", GenVisionConfig)
AutoConfig.register("gen_aligner", GenAlignerConfig)
AutoConfig.register("gen_head", GenHeadConfig)
AutoConfig.register("multi_modality", MultiModalityConfig)
AutoModelForCausalLM.register(MultiModalityConfig, MultiModalityCausalLM)



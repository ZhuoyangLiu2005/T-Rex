"""Utils for evaluating robot policies in various environments."""

import os
import random
import time
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
import PIL.Image

# from experiments.robot.openvla_utils import (
#     get_vla,
#     get_vla_action,
# )

# Initialize important constants
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)

# Model image size configuration
MODEL_IMAGE_SIZES = {
    "openvla": 224,
    # Add other models as needed
}

from dataclasses import dataclass

@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)

def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_action(
        cfg: Any,
        statistic, 
        action_tokenizer, 
        vl_chat_processor, 
        task_description, 
        vl_gpt, 
        fast_image, 
        slow_image,  
        state=None, 
        pre_image_dir=None, 
        step=0
    ) -> Union[List[np.ndarray], np.ndarray]:
    device = f'cuda:{cfg.cuda}'
    vl_gpt = vl_gpt.to(device).eval()
    parallel_size = 1
    num_image_tokens = 576
    img_size = 384
    patch_size = 16

    def get_state_tokens(state_arr):
        state_arr = np.array(state_arr, dtype=np.float32)
        norm_state = np.where(
            statistic['state_mask'],
            np.clip(2 * (state_arr - statistic['state_min']) / (statistic['state_max'] - statistic['state_min'] + 1e-8) - 1, -1, 1),
            state_arr
        )
        return action_tokenizer(norm_state)

    state_tokens_slow = get_state_tokens(state) if cfg.use_proprio else ""
    state_tokens_fast = get_state_tokens(state) if cfg.use_proprio else ""

    img_placeholder = vl_chat_processor.image_start_tag + vl_chat_processor.pad_tag * num_image_tokens + vl_chat_processor.image_end_tag

    if cfg.use_pred:
        prompts = img_placeholder + task_description + state_tokens_slow + img_placeholder + img_placeholder + state_tokens_fast
    else:
        prompts = img_placeholder + task_description + state_tokens_slow + img_placeholder + state_tokens_fast
    
    conversation = [{"role": "<|User|>", "content": prompts}]
    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=vl_chat_processor.sft_format,
        system_prompt="",
    )

    with torch.inference_mode():
        input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format)).unsqueeze(0).to(device)
        input_ids[input_ids < 0] = 0 

        inputs_embeds = vl_gpt.language_model.get_input_embeddings()(input_ids)

        def get_vqvae_embeds(img):
            pixel_values = vl_chat_processor.image_processor(img, return_tensors="pt")['pixel_values'].to(torch.bfloat16).to(device)
            _, _, info = vl_gpt.gen_vision_model.encode(pixel_values)
            tokens = info[2].detach().reshape(pixel_values.shape[0], -1)
            return vl_gpt.prepare_gen_img_embeds(tokens)

        embeds_slow = get_vqvae_embeds(slow_image)
        embeds_fast = get_vqvae_embeds(fast_image)

        img_start_indices = (input_ids[0] == vl_chat_processor.image_start_id).nonzero().squeeze(-1)
        
        start_slow = img_start_indices[0] + 1
        start_out = img_start_indices[1] + 1 if cfg.use_pred else None
        start_fast = img_start_indices[2] + 1 if cfg.use_pred else img_start_indices[1] + 1

        inputs_embeds[:, start_slow : start_slow + num_image_tokens, :] = embeds_slow
        inputs_embeds[:, start_fast : start_fast + num_image_tokens, :] = embeds_fast

        if cfg.use_pred:
            kv_cache = None
            next_compute_range = (0, start_out) 
            generated_image_tokens = torch.zeros((parallel_size, num_image_tokens), dtype=torch.long, device=device)

            for i in range(num_image_tokens):
                curr_inputs_embeds = inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :]

                outputs = vl_gpt.language_model.model(
                    inputs_embeds=curr_inputs_embeds,
                    latent_indexes=torch.arange(0, curr_inputs_embeds.shape[1]).to(device), 
                    action_indexes=torch.arange(0, 0).to(device), 
                    use_latent=cfg.use_pred,
                    use_cache=True,
                    past_key_values=kv_cache
                )

                kv_cache = outputs.past_key_values
                hidden_states = outputs[0][:, -1:, :] 

                logits = vl_gpt.gen_head(hidden_states)
                next_token = torch.argmax(logits, dim=-1) 
                generated_image_tokens[:, i] = next_token.squeeze(-1)

                img_embeds = vl_gpt.prepare_gen_img_embeds(next_token)
                inputs_embeds[:, start_out + i : start_out + i + 1, :] = img_embeds
                next_compute_range = (start_out + i, start_out + i + 1)
        
        noise = torch.randn(inputs_embeds.shape[0], cfg.num_open_loop_steps, 7, device=device)
        samples = vl_gpt.forward_flow(inputs_embeds, noise)
        
        normalized_actions = samples[0].cpu().numpy()
        if normalized_actions.ndim == 1:
            dim = len(normalized_actions)
            if dim == 7 or dim == 14:
                normalized_actions[6] = 0 if normalized_actions[6] < 0.5 else 1
            if dim == 14:
                normalized_actions[13] = 0 if normalized_actions[13] < 0.5 else 1
        else:
            dim = normalized_actions.shape[1]
            if dim == 7 or dim == 14:
                normalized_actions[:, 6] = (normalized_actions[:, 6] >= 0.5).astype(int)
            if dim == 14:
                normalized_actions[:, 13] = (normalized_actions[:, 13] >= 0.5).astype(int)
                
        actions = np.where(
            statistic['action_mask'],
            0.5 * (normalized_actions + 1) * (statistic['action_max'] - statistic['action_min']) + statistic['action_min'],
            normalized_actions,
        )

        if cfg.use_pred and pre_image_dir is not None:
            os.makedirs(pre_image_dir, exist_ok=True)
            dec = vl_gpt.gen_vision_model.decode_code(
                generated_image_tokens.to(dtype=torch.int), 
                shape=[parallel_size, 8, img_size//patch_size, img_size//patch_size]
            )
            dec = dec.to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1)
            dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)
            
            save_path = os.path.join(pre_image_dir, f'step_{step}.png')
            PIL.Image.fromarray(dec[0]).save(save_path)

        return list(actions)


def get_action_latent(
    cfg: Any,
    statistic,
    action_tokenizer,
    vl_chat_processor,
    task_description,
    vl_gpt,
    fast_image,
    slow_image,
    state=None,
) -> Union[List[np.ndarray], np.ndarray]:

    device = f'cuda:{cfg.cuda}'
    vl_gpt = vl_gpt.to(device).eval()
    parallel_size= 1
    fast_img_len = 1
    slow_img_len = 1
    num_latent_tokens = cfg.latent_size
    
    state_tokens = ""
    if cfg.use_proprio:
        state = np.array(state, dtype=np.float32)
        normalized_state = np.where(
            statistic['state_mask'],
            np.clip(2 * (state - statistic['state_min']) / (statistic['state_max'] - statistic['state_min'] + 1e-8) - 1, -1, 1),
            state
        )
        normalized_state = torch.tensor(normalized_state, dtype=torch.bfloat16).to(device)

    pre_data = []
    input_img_tokens = vl_chat_processor.image_start_tag + vl_chat_processor.image_tag*vl_chat_processor.num_image_tokens +vl_chat_processor.image_end_tag
    
    input_slow_img_tokens = input_img_tokens * slow_img_len
    input_fast_img_tokens = input_img_tokens * fast_img_len
    
    latent_start_str = "<|latent_start|>"
    latent_pad_str = "<|latent_pad|>" * num_latent_tokens
    latent_end_str = "<|latent_end|>"
    latent_str = latent_start_str + latent_pad_str + latent_end_str
    
    user_content = input_slow_img_tokens + task_description + state_tokens + latent_str + input_fast_img_tokens

    conversation = [
                    {"role": "<|User|>","content": user_content}
                ]

    sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
            conversations=conversation,
            sft_format=vl_chat_processor.sft_format,
            system_prompt="",
        )
    all_image = slow_image + fast_image 
    with torch.inference_mode():
        input_image_pixel_values = vl_chat_processor.image_processor(all_image, return_tensors="pt")['pixel_values'].to(torch.bfloat16)

        input_ids =  torch.LongTensor(vl_chat_processor.tokenizer.encode(sft_format))
        tokens = torch.zeros((parallel_size, len(input_ids)), dtype=torch.long)

        for i in range(parallel_size):
            tokens[i, :] = input_ids
            pre_data.append(VLChatProcessorOutput(sft_format=sft_format, pixel_values=input_image_pixel_values, input_ids=tokens[i], num_image_tokens=[vl_chat_processor.num_image_tokens] * (slow_img_len + fast_img_len)))
        prepare_inputs = vl_chat_processor.batchify(pre_data)

        inputs_embeds = vl_gpt.prepare_inputs_embeds(
            input_ids=tokens.to(device),
            pixel_values=prepare_inputs['pixel_values'].to(torch.bfloat16).to(device),
            images_emb_mask=prepare_inputs['images_emb_mask'].to(device),
            images_seq_mask=prepare_inputs['images_seq_mask'].to(device)
        )

        torch.set_printoptions(profile="full")

        input_ids = input_ids.unsqueeze(0)
        latent_indices = (input_ids == 100847).nonzero()
        latent_lists = [[idx[1].item() for idx in latent_indices if idx[0] == i] for i in range(input_ids.shape[0])]
        kv_cache_cot = None
        next_compute_range = (0, latent_indices[:, 1].min().item())
        
        # print("input_ids:", input_ids)
        # print("shape of input ids and latent_indices:", input_ids.shape, latent_indices.shape)
        # print("latent_indices:", latent_indices)
        # input("Press Enter to continue...")

        # inference for latent cot embeddings
        for latent_i in range(num_latent_tokens):
            curr_inputs_embeds = inputs_embeds[:, next_compute_range[0] : next_compute_range[1], :]
            outputs = vl_gpt.language_model.model(
                inputs_embeds=curr_inputs_embeds,
                latent_indexes=torch.arange(0, curr_inputs_embeds.shape[1]).to(device),
                action_indexes=torch.arange(0, 0).to(device),
                use_latent=cfg.use_latent,
                use_cache=True,
                past_key_values=kv_cache_cot if latent_i!=0 else None # for kv cache
            )
            next_compute_range = (
                next_compute_range[1],
                (
                    input_ids.shape[1]
                    if latent_i + 1 >= num_latent_tokens
                    else next_compute_range[1] + 1
                ),
            )
            hidden_states = outputs[0][:, -1:, :]
            assert hidden_states.shape[1] == 1
            kv_cache_cot = outputs.past_key_values
            filling_indices = [
                (instance_idx, mask_list[latent_i])
                for instance_idx, mask_list in enumerate(latent_lists)
                if len(mask_list) > latent_i
            ]
            tensor_list = [
                [
                    inputs_embeds[batch_idx, pos, :]
                    for pos in range(inputs_embeds.shape[1])
                ]
                for batch_idx in range(inputs_embeds.shape[0])
            ]
            for idx_pair in filling_indices:
                batch_idx, token_idx = idx_pair  
                tensor_list[batch_idx][token_idx] = hidden_states[batch_idx][0]
            inputs_embeds = torch.stack(
                [
                    torch.stack(tensor_list[batch_idx])
                    for batch_idx in range(inputs_embeds.shape[0])
                ]
            )

        # noise = torch.randn(inputs_embeds.shape[0], cfg.num_open_loop_steps, 7, device=device)
        noise = torch.randn(inputs_embeds.shape[0], 8, 7, device=device)
        samples = vl_gpt.forward_flow(inputs_embeds, noise)
        
        normalized_actions = samples[0].cpu().numpy()

        if normalized_actions.ndim == 1:
            dim = len(normalized_actions)
            if dim == 7 or dim == 14:
                normalized_actions[6] = 0 if normalized_actions[6] < 0.5 else 1
            if dim == 14:
                normalized_actions[13] = 0 if normalized_actions[13] < 0.5 else 1
        else:
            dim = normalized_actions.shape[1]
            if dim == 7 or dim == 14:
                normalized_actions[:, 6] = (normalized_actions[:, 6] >= 0.5).astype(int)
            if dim == 14:
                normalized_actions[:, 13] = (normalized_actions[:, 13] >= 0.5).astype(int)
        actions = np.where(
            statistic['action_mask'],
            0.5 * (normalized_actions + 1) * (statistic['action_max'] - statistic['action_min']) + statistic['action_min'],
            normalized_actions,
        )

    return list(actions)


def normalize_gripper_action(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """
    Normalize gripper action from [0,1] to [-1,+1] range.

    This is necessary for some environments because the dataset wrapper
    standardizes gripper actions to [0,1]. Note that unlike the other action
    dimensions, the gripper action is not normalized to [-1,+1] by default.

    Normalization formula: y = 2 * (x - orig_low) / (orig_high - orig_low) - 1

    Args:
        action: Action array with gripper action in the last dimension
        binarize: Whether to binarize gripper action to -1 or +1

    Returns:
        np.ndarray: Action array with normalized gripper action
    """
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Normalize the last action dimension to [-1,+1]
    orig_low, orig_high = 0.0, 1.0
    normalized_action[..., -1] = 2 * (normalized_action[..., -1] - orig_low) / (orig_high - orig_low) - 1

    if binarize:
        # Binarize to -1 or +1
        normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action


def invert_gripper_action(action: np.ndarray) -> np.ndarray:
    """
    Flip the sign of the gripper action (last dimension of action vector).

    This is necessary for environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.

    Args:
        action: Action array with gripper action in the last dimension

    Returns:
        np.ndarray: Action array with inverted gripper action
    """
    # Create a copy to avoid modifying the original
    inverted_action = action.copy()

    # Invert the gripper action
    inverted_action[..., -1] *= -1.0

    return inverted_action

import os
import json
import torch
import logging
import argparse
import random
import shutil
import math
import wandb
import PIL.Image
import numpy as np
import time

from typing import List, Dict, Any
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from accelerate import Accelerator
from einops import rearrange
from transformers import (
    set_seed,
)
from transformers import AutoModelForCausalLM
from janus.models import VLChatProcessor, ActionTokenizer


logger = logging.getLogger(__name__)
logging.basicConfig(level='INFO')

from dataclasses import dataclass
@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor

    def __len__(self):
        return len(self.input_ids)

def get_custom_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    min_lr_ratio=0.0, 
    num_cycles=0.5
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
        scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
        return scaled_factor

    return LambdaLR(optimizer, lr_lambda, last_epoch=-1)

def get_learning_rate(step, initial_lr, num_warmup_steps, num_training_steps, min_lr_ratio, num_cycles=0.5):
    if step < num_warmup_steps:
            return float(step) / float(max(1, num_warmup_steps)) * initial_lr
    progress = float(step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * 2 * num_cycles * progress))
    scaled_factor = (1 - min_lr_ratio) * cosine_factor + min_lr_ratio
    return scaled_factor * initial_lr

def create_component_indexes(seq_len, action_len=7, tactile_len=6):
    image_indexes = torch.arange(0, seq_len - action_len)
    action_indexes = torch.arange(seq_len - action_len, seq_len - tactile_len)
    tactile_indexes = torch.arange(seq_len - tactile_len, seq_len)
    return image_indexes, action_indexes, tactile_indexes

class TrainingMetrics:
    def __init__(self, device):
        self.n_step = 0
        self.action_right = torch.Tensor([0]).to(device=device)
        self.action_total = torch.Tensor([0]).to(device=device)
        self.action_loss = torch.Tensor([0]).to(device=device)
        # image prediction
        self.image_right = torch.Tensor([0]).to(device=device)
        self.image_total = torch.Tensor([0]).to(device=device)
        self.image_loss = torch.Tensor([0]).to(device=device)
        self.world_size = dist.get_world_size()

    def __call__(self, has_img, image_logits, image_labels, image_loss, action_loss):
        if has_img:
            return self.update(image_logits, image_labels, image_loss, action_loss)
        else: # action-expert only
            return self.update_action(action_loss)

    def update(self, image_logits, image_labels, image_loss, action_loss):
        self.n_step += 1
        with torch.no_grad():
            shift_image_preds = image_logits.argmax(dim=-1) # logits[..., :-1, :].argmax(dim=-1)
            shift_image_labels = image_labels # labels[..., 1:]
            self.image_right += (shift_image_preds == shift_image_labels).masked_fill(shift_image_labels.eq(-100), 0).sum().item()
            self.image_total += (shift_image_labels != -100).sum().item()
            self.image_loss += image_loss.item()

            self.action_loss += action_loss.item()
            
    def update_action(self, action_loss):
        self.n_step += 1
        with torch.no_grad():
            self.action_loss += action_loss.item()

    def get_metric(self, reset=True):
        dist.all_reduce(self.image_right, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.image_total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.image_loss, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.action_loss, op=torch.distributed.ReduceOp.SUM)

        image_acc = (self.image_right / self.image_total).item()
        image_loss = self.image_loss.item() / (self.world_size * self.n_step)
        action_loss = self.action_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.image_right.fill_(0)
            self.image_total.fill_(0)
            self.image_loss.fill_(0)
            self.action_loss.fill_(0)
        return image_acc, image_loss, action_loss

    def get_metric_action(self, reset=True):
        dist.all_reduce(self.action_total, op=torch.distributed.ReduceOp.SUM)
        dist.all_reduce(self.action_loss, op=torch.distributed.ReduceOp.SUM)
        action_loss = self.action_loss.item() / (self.world_size * self.n_step)

        if reset:
            self.n_step = 0
            self.action_total.fill_(0)
            self.action_loss.fill_(0)
        return 0, 0, action_loss


class SftDataset(Dataset):
    def __init__(self, config, processor, accelerator, model):
        self.model = model
        self.config = config
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.action_tokenizer = ActionTokenizer(self.tokenizer) 
        self.accelerator = accelerator
        self.image_len = 576
        with open(config.data_path,'r') as f:
            self.data = json.load(f)

        statistics_path = config.data_path.replace(".json", "_statistics.json")
        with open(statistics_path, 'r') as f:
            self.stats_data = json.load(f)

        self.dataset_name = next(iter(self.stats_data))
        self.action_mask = np.array(self.stats_data[self.dataset_name]['action']['mask'])
        self.action_min = np.array(self.stats_data[self.dataset_name]['action']['q01'])
        self.action_max = np.array(self.stats_data[self.dataset_name]['action']['q99'])
        self.state_mask = np.array(self.stats_data[self.dataset_name]['state']['mask'])
        self.state_min = np.array(self.stats_data[self.dataset_name]['state']['q01'])
        self.state_max = np.array(self.stats_data[self.dataset_name]['state']['q99'])
        self.tacf6_mask = np.array(self.stats_data[self.dataset_name]['tactile_f6']['mask'])
        self.tacf6_min = np.array(self.stats_data[self.dataset_name]['tactile_f6']['q01'])
        self.tacf6_max = np.array(self.stats_data[self.dataset_name]['tactile_f6']['q99'])

        self.tracking_err_mean = None
        self.tracking_err_std = None
        tracking_info = self.stats_data[self.dataset_name].get('tracking_error', {})
        if 'mean' in tracking_info and 'std' in tracking_info:
            self.tracking_err_mean = np.array(tracking_info['mean'], dtype=np.float32)
            self.tracking_err_std = np.array(tracking_info['std'], dtype=np.float32)
            accelerator.print(f"Loaded tracking error noise stats. Shape: {self.tracking_err_mean.shape}")
        else:
            accelerator.print("Warning: No tracking error stats found. State noise will not be applied.")

        self.img_dir = os.path.dirname(config.data_path)
        accelerator.print(f'Total data amount: {len(self.data)}')

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return self.data[index]

    def process_image(self,image_paths):
        images = [PIL.Image.open(image_path).convert("RGB") for image_path in image_paths]
        images_outputs = self.processor.image_processor(images, return_tensors="pt")
        return images_outputs['pixel_values']

    def sample_beta(self, alpha, beta, bsize, device):
        alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
        beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
        dist = torch.distributions.Beta(alpha_t, beta_t)
        samples = dist.sample((bsize,))
        return samples.to(dtype=torch.bfloat16)

    def sample_time(self, bsize, device):
        time_beta = self.sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.bfloat16, device=device)

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.bfloat16,
            device=device,
        )

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # load images and process to pixel values
        gen_images = [os.path.join(self.img_dir,x['output_image']) for x in batch]
        # gen_images = [os.path.join(self.img_dir,x['output_image']) for x in batch if 'input_image_slow' in x]
        input_images_slow = sum([x['input_image_slow'] for x in batch if 'input_image_slow' in x],[])
        input_images_slow = [os.path.join(self.img_dir,x) for x in input_images_slow]
        input_images_fast = sum([x['input_image_fast'] for x in batch if 'input_image_fast' in x],[])
        input_images_fast = [os.path.join(self.img_dir,x) for x in input_images_fast]
        
        # Get codebook representations for images
        output_pixel_values = self.process_image(gen_images).to(torch.bfloat16) if len(gen_images) > 0 else None
        input_pixel_values_slow = self.process_image(input_images_slow).to(torch.bfloat16) if len(input_images_slow) > 0 else None
        input_pixel_values_fast = self.process_image(input_images_fast).to(torch.bfloat16) if len(input_images_fast) > 0 else None
        
        input_img_tokens = self.processor.image_start_tag + self.processor.image_tag * self.processor.num_image_tokens + self.processor.image_end_tag
        
        # Generate noisy actions and timesteps for diffusion
        actions = [x['action'] for x in batch]
        actions = np.array(actions, dtype=np.float32).reshape(len(actions), -1, self.config.action_dim)
        normalized_actions = np.where(
            self.action_mask,
            np.clip(2 * (actions - self.action_min) / (self.action_max - self.action_min + 1e-8) - 1, -1, 1),
            actions
        )
        normalized_actions = torch.tensor(normalized_actions)

        time = self.sample_time(normalized_actions.shape[0], normalized_actions.device)
        time_expanded = time[:, None, None]

        noise = self.sample_noise(normalized_actions.shape, normalized_actions.device)
        x_t = (time_expanded * noise + (1 - time_expanded) * normalized_actions)
        u_t = (noise - normalized_actions)

        # Tactile f6 signals, for each sample, 10 tokens, do the same preprocess as actions
        normalized_tactile_f6s = None
        if self.config.use_tactile_vec:
            tactile_f6s_ori = [x['tactile_f6'] for x in batch]
            tactile_f6s = np.array(tactile_f6s_ori, dtype=np.float32).reshape(len(tactile_f6s_ori), -1)
            normalized_tactile_f6s = np.where(
                self.tacf6_mask,
                np.clip(2 * (tactile_f6s - self.tacf6_min) / (self.tacf6_max - self.tacf6_min + 1e-8) - 1, -1, 1),
                tactile_f6s
            )
            normalized_tactile_f6s = normalized_tactile_f6s.reshape(len(tactile_f6s_ori), -1, 6)
            normalized_tactile_f6s = torch.tensor(normalized_tactile_f6s)

        tactile_deforms_tensor = None
        if self.config.use_tactile_deform:
            tactile_deforms = []
            for x in batch:
                imgs = []
                for img_path in x.get('tactile_image_deform', []):
                    full_path = img_path if os.path.isabs(img_path) else os.path.join(self.img_dir, img_path)
                    img = PIL.Image.open(full_path).convert("L")
                    img_arr = np.array(img, dtype=np.float32) / 255.0
                    imgs.append(img_arr)
                tactile_deforms.append(imgs)
            # [B, 5, H, W] -> [B, 5, 1, 240, 240]
            tactile_deforms_tensor = torch.tensor(np.array(tactile_deforms)).unsqueeze(2)

        # Prepare data in batch
        pre_data = []
        for x in batch:
            slow_imgs = x.get('input_image_slow', [])
            fast_imgs = x.get('input_image_fast', [])
            slow_img_len = len(slow_imgs)
            fast_img_len = len(fast_imgs)
            all_input_imgs = slow_imgs + fast_imgs # for siglip encoder

            state_tokens_fast = ""
            if self.config.use_robot_state:
                state_fast = np.array(x['state_fast'], dtype=np.float32)
                if self.tracking_err_mean is not None:
                    state_noise = np.random.normal(
                        loc=self.tracking_err_mean, 
                        scale=self.tracking_err_std
                    )
                    state_fast = state_fast + state_noise
                normalized_state_fast = np.where(
                    self.state_mask,
                    np.clip(2 * (state_fast - self.state_min) / (self.state_max - self.state_min + 1e-8) - 1, -1, 1),
                    state_fast
                )
                state_tokens_fast += self.action_tokenizer(normalized_state_fast)

            input_slow_img_tokens = input_img_tokens * slow_img_len
            input_fast_img_tokens = input_img_tokens * fast_img_len

            prompts = input_slow_img_tokens + x['input_prompt'] + input_fast_img_tokens + state_tokens_fast

            conversation = [
                {"role": "<|User|>","content": prompts},
            ]

            pre_format = self.processor.apply_sft_template_for_multi_turn_prompts(
                conversations=conversation,
                sft_format=self.processor.sft_format,
                system_prompt="",
            )
            sft_format = pre_format
            
            if len(all_input_imgs) > 0:
                encoder_pixel_values = self.process_image([os.path.join(self.img_dir, input_img) for input_img in all_input_imgs])
                num_image_tokens = [self.image_len] * len(all_input_imgs)
            else:
                encoder_pixel_values = None
                num_image_tokens = []
            
            input_ids = torch.LongTensor(self.processor.tokenizer.encode(sft_format))
            pre_data.append(
                VLChatProcessorOutput(
                    sft_format=sft_format, 
                    pixel_values=encoder_pixel_values, 
                    input_ids=input_ids, 
                    num_image_tokens=num_image_tokens
                )
            )

        if len(pre_data) > 0:
            prepare_inputs = self.processor.batchify(pre_data)

        return {
            "input_ids": prepare_inputs.input_ids,
            "encoder_pixel_values": prepare_inputs.pixel_values.to(torch.bfloat16),
            "input_pixel_values_slow": input_pixel_values_slow,
            "input_pixel_values_fast": input_pixel_values_fast,
            "output_pixel_values": output_pixel_values,
            "noisy_actions": x_t,
            "target": u_t,
            "timesteps": time,
            "tactile_f6s": normalized_tactile_f6s,
            "tactile_deforms": tactile_deforms_tensor,
            "fast_img_len": fast_img_len,
            "attention_mask": prepare_inputs.attention_mask,
            "images_seq_mask": prepare_inputs['images_seq_mask'],
            "images_emb_mask": prepare_inputs['images_emb_mask'],
        }


def save_checkpoint(
    model,
    processor,
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch: int,
    step: int,
    global_step: int,
    is_last: bool = False,
    stats_data = None
) -> None:

    save_dir = os.path.join(args.output_dir, f"checkpoint-{epoch}-{global_step}")
    
    if accelerator.is_main_process:
        # Manage checkpoint numbers
        checkpoint_files = [f for f in os.listdir(args.output_dir) if f.startswith("checkpoint-")]
        if args.max_ckpts > 0 and len(checkpoint_files) >= args.max_ckpts:
            oldest_ckpt = min(checkpoint_files, key=lambda x: os.path.getctime(os.path.join(args.output_dir, x)))
            shutil.rmtree(os.path.join(args.output_dir, oldest_ckpt))

        os.makedirs(save_dir, exist_ok=True)
        output_dir = os.path.join(save_dir, 'tfmr')

        model.save_pretrained(output_dir, state_dict=accelerator.get_state_dict(model))
        processor.save_pretrained(output_dir)

        with open(os.path.join(save_dir, 'stats_data.json'), 'w') as f:
            json.dump(stats_data, f, indent=2)
            
        logger.info(f"Statistics have been saved to {os.path.join(save_dir, 'stats_data.json')}")

    accelerator.wait_for_everyone()
    logger.info(f'Checkpoint {epoch}-{global_step} saved successfully')



def train(args: argparse.Namespace) -> None:

    accelerator = Accelerator(
        mixed_precision='bf16',
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    # Set random seed
    set_seed(args.seed)

    if accelerator.is_main_process:
        wandb.init(
            project=args.experiment_name,
            name=args.run_name,
            config=args,
            dir=args.log_dir,
        )
    accelerator.state.deepspeed_plugin.deepspeed_config['train_micro_batch_size_per_gpu'] = args.train_bsz_per_gpu
    accelerator.state.deepspeed_plugin.deepspeed_config['train_batch_size'] = (
        args.train_bsz_per_gpu * 
        dist.get_world_size() * 
        accelerator.gradient_accumulation_steps
    )

    processor = VLChatProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        flow = True,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        use_latent=args.use_pred,
        ignore_mismatched_sizes=True,
        use_tactile_deform=args.use_tactile_deform,
    )
    model_action = AutoModelForCausalLM.from_pretrained(
        args.action_model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        flow = True,
        action_dim=args.action_dim,
        action_chunk=args.action_chunk,
        use_latent=args.use_pred,
        ignore_mismatched_sizes=True,
        use_tactile_deform=args.use_tactile_deform,
    )
    if args.use_tactile_deform:
        model.load_deform_encoder_weights(args.deform_encoder_ckpt)

    model_config = model.config

    for name, param in model.named_parameters():
        if '_action' in name:
            if args.load_action_from_latent and name.endswith('.weight'):
                base_name = name.replace('_action', '')
                if base_name in model.state_dict():
                    param.data.copy_(model.state_dict()[base_name])
                    accelerator.print(f"Initialized {name} from {base_name}")
            elif args.load_action_from_pretrain and name.endswith('.weight'):
                base_name = name.replace('_action', '')
                if base_name in model_action.state_dict():
                    param.data.copy_(model_action.state_dict()[base_name])
                    accelerator.print(f"Initialized {name} from {base_name}")
            param.requires_grad = True
        elif '_tactile' in name:
            if args.load_action_from_latent and name.endswith('.weight'):
                base_name = name.replace('_tactile', '')
                if base_name in model.state_dict():
                    param.data.copy_(model.state_dict()[base_name])
                    accelerator.print(f"Initialized {name} from {base_name}")
            elif args.load_action_from_pretrain and name.endswith('.weight'):
                base_name = name.replace('_tactile', '')
                if base_name in model_action.state_dict():
                    param.data.copy_(model_action.state_dict()[base_name])
                    accelerator.print(f"Initialized {name} from {base_name}")
            param.requires_grad = True
        elif 'x_embedder' in name or 'state_embedder' in name or 't_embedder' in name \
            or 'final_layer' in name or 'tacf6_embedder' in name or 'deform_proj' in name:
            if name in model_action.state_dict():
                param.data.copy_(model_action.state_dict()[name])
                accelerator.print(f"Initialized {name} from action model")
            else:
                accelerator.print(f"Initialized {name} from scratch (random weights)")
            param.requires_grad = True
        else:
            if any(name.startswith(prefix) for prefix in ["vision_model", "aligner", "gen_vision_model", "deform_encoder"]):
                param.requires_grad = False
            else:
                param.requires_grad = True

    accelerator.print("\n==== Parameter Freeze Status ====\n")
    for name, param in model.named_parameters():
        status = "TRAINABLE" if param.requires_grad else "FROZEN"
        accelerator.print(f"{name:60}  {status}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    accelerator.print(f"Load action from latent: {args.load_action_from_latent}")
    accelerator.print(f"Load action from pretrain: {args.load_action_from_pretrain}")
    accelerator.print(f"Total parameters: {total_params/1e9:.2f}B")
    accelerator.print(f"Trainable parameters: {trainable_params/1e9:.2f}B")
    accelerator.print(f"Non-trainable parameters: {non_trainable_params/1e9:.2f}B")
    accelerator.print(f"Trainable ratio: {trainable_params/total_params*100:.2f}%")

    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    train_dataset = SftDataset(args, processor, accelerator, model)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.train_bsz_per_gpu,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=8
    )

    num_training_steps = int(len(train_dataloader) * args.n_epochs) // accelerator.gradient_accumulation_steps // dist.get_world_size()
    lr_scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_rates * num_training_steps),
        num_training_steps=num_training_steps,
        min_lr_ratio=args.min_lr_ratio
    )
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    metric = TrainingMetrics(device=torch.cuda.current_device())
    model.train()
    global_step = 0

    for epoch in range(0, args.n_epochs):
        train_iter = tqdm(train_dataloader, total=len(train_dataloader)) if accelerator.is_main_process else train_dataloader
        for batch in train_iter:
            inputs_embeds = model.prepare_inputs_embeds(
                    input_ids=batch['input_ids'],
                    pixel_values=batch['encoder_pixel_values'],
                    images_emb_mask=batch['images_emb_mask'],
                    images_seq_mask=batch['images_seq_mask']
                )

            # torch.set_printoptions(profile="full")
            # print(batch['input_ids'][0])
            # print(batch['input_ids'].shape)
            # print(batch['noisy_actions'].shape)
            # print(batch['timesteps'].shape)
            # print(batch['tactile_f6s'].shape)
            # print("before:", inputs_embeds.shape)
            # input("Press Enter to continue...")
            
            ## Add diffuison related tokens (time + action)
            # for convienience, we directly append the two tokens at the end (1129)
            noisy_actions = model.x_embedder(batch['noisy_actions'].to(inputs_embeds.dtype))
            timesteps = model.t_embedder(batch['timesteps'].to(inputs_embeds.dtype)).unsqueeze(1)
            noisy_actions_tactile = model.x_embedder(batch['noisy_actions'].to(inputs_embeds.dtype))
            timesteps_tactile = model.t_embedder(batch['timesteps'].to(inputs_embeds.dtype)).unsqueeze(1)
            
            if args.use_tactile_deform:
                deforms = batch['tactile_deforms'].to(inputs_embeds.device).to(inputs_embeds.dtype) 
                B, num_fingers, C, H, W = deforms.shape
                deforms_flat = deforms.view(-1, C, H, W)
                with torch.no_grad():
                    deform_features = model.deform_encoder(deforms_flat) # [B*5, 128, 15, 15]
                deform_features = deform_features.view(B, num_fingers, -1) # [B, 5, 28800]
                tactile_embeds = model.deform_proj(deform_features.to(inputs_embeds.dtype))
            elif args.use_tactile_vec:
                tactile_embeds = model.tacf6_embedder(batch['tactile_f6s'].to(inputs_embeds.dtype))
            else:
                tactile_embeds = torch.empty((inputs_embeds.shape[0], 0, inputs_embeds.shape[2]), device=inputs_embeds.device, dtype=inputs_embeds.dtype)
            
            inputs_embeds = torch.cat([
                inputs_embeds,
                timesteps,
                noisy_actions,
                tactile_embeds,
                timesteps_tactile,
                noisy_actions_tactile
            ], dim=1)
            batch['attention_mask'] = torch.cat([
                batch['attention_mask'],
                torch.ones((batch['attention_mask'].shape[0], timesteps.shape[1]), dtype=torch.bool).to(batch['attention_mask'].device),
                torch.ones((batch['attention_mask'].shape[0], noisy_actions.shape[1]), dtype=torch.bool).to(batch['attention_mask'].device),
                torch.ones((batch['attention_mask'].shape[0], tactile_embeds.shape[1]), dtype=torch.bool).to(batch['attention_mask'].device),
                torch.ones((batch['attention_mask'].shape[0], timesteps_tactile.shape[1]), dtype=torch.bool).to(batch['attention_mask'].device),
                torch.ones((batch['attention_mask'].shape[0], noisy_actions_tactile.shape[1]), dtype=torch.bool).to(batch['attention_mask'].device),
            ], dim=1)

            # print("after: ", inputs_embeds.shape)
            # input("after check shape")

            fast_img_len = batch['fast_img_len']
            tactile_embed_len = tactile_embeds.shape[1]
            action_len = 1 + 578 * fast_img_len + 1 + args.action_chunk + tactile_embed_len + 1 + args.action_chunk
            if args.use_robot_state:
                action_len = action_len + args.action_dim # action dim is the number of state tokens for we use openvla tokenizer
            tactile_len = tactile_embed_len + 1 + args.action_chunk
            latent_indexes, action_indexes, tactile_indexes = create_component_indexes(inputs_embeds.shape[1], action_len, tactile_len) # important: 3 means <latent_end>, <timestep>, <noise>
            # print(batch['input_ids'][0], batch['input_ids'].shape)
            # print(latent_indexes, action_indexes)
            # input("check indexes")
            
            outputs = model.language_model.model(
                inputs_embeds=inputs_embeds,
                # attention_mask=batch['attention_mask'],
                return_dict=True,
                use_cache=False,
                latent_indexes=latent_indexes.to(inputs_embeds.device),
                action_indexes=action_indexes.to(inputs_embeds.device),
                tactile_indexes=tactile_indexes.to(inputs_embeds.device),
                use_latent=args.use_pred,
            )
            hidden_states = outputs.last_hidden_state

            predicted_noise = model.final_layer(hidden_states)[:, -(batch['target'].shape[1]*2 + tactile_embed_len + 1):-(batch['target'].shape[1] + tactile_embed_len + 1), :] # the last token is noise
            action_loss = nn.MSELoss()(predicted_noise, batch['target'].to(predicted_noise.dtype))

            predicted_noise_tactile = model.final_layer(hidden_states)[:, -(batch['target'].shape[1]):, :] # the last token is noise
            action_loss_tactile = nn.MSELoss()(predicted_noise_tactile, batch['target'].to(predicted_noise_tactile.dtype))

            loss = action_loss + action_loss_tactile
            metric(args.use_pred, None, None, None, action_loss + action_loss_tactile)

            accelerator.backward(loss)
            if (global_step + 1) % accelerator.gradient_accumulation_steps == 0:
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                image_acc, image_loss, action_loss= metric.get_metric() if args.use_pred else metric.get_metric_action()
                if accelerator.is_main_process:
                    train_iter.set_postfix(
                        epoch=epoch,
                        step=global_step,
                        total_steps=len(train_dataloader),
                        skip=accelerator.optimizer_step_was_skipped,
                        length=len(batch["input_ids"][0]),
                        image_acc=f"{image_acc:.4f}",
                        image_loss=f"{image_loss:.6f}",
                        action_loss=f"{action_loss:.6f}",
                        lr=f"{lr_scheduler.get_last_lr()[0]:.2e}"
                    )
                    wandb.log({
                        'image_acc': image_acc,
                        'image_loss': image_loss,
                        'action_loss': action_loss,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }, step=global_step)
            global_step += 1

        if ((epoch + 1) % args.save_freq == 0) or (epoch == args.n_epochs-1):
            accelerator.wait_for_everyone()
            save_checkpoint(
                model=model,
                processor=processor, 
                accelerator=accelerator,
                args=args,
                epoch=epoch,
                step=global_step-1,
                global_step=global_step,
                is_last=(epoch == args.n_epochs-1),
                stats_data=train_dataset.stats_data,
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pre-training parameter configuration')
    
    # Experiment settings
    parser.add_argument('--experiment_name', type=str, default='janus_train', help='Experiment name')
    parser.add_argument('--run_name', type=str, default='run_1', help='Run name')
    parser.add_argument('--model_path', type=str, default='', help='Pre-trained model path')
    parser.add_argument('--action_model_path', type=str, default='', help='Resume from action checkpoint')

    # Data related
    parser.add_argument('--data_path', type=str, required=True, help='Training data path, can be multiple paths')
    parser.add_argument('--data_root', type=str, required=True, default='')
    parser.add_argument('--output_dir', type=str, default='./', help='Model save path')
    parser.add_argument('--max_ckpts', type=int, default=10, help='Maximum number of checkpoints to save')
    parser.add_argument('--log_dir', type=str, default='./train_logs', help='Log save path')

    # Training related
    parser.add_argument('--max_seq_len', type=int, default=4096, help='Maximum sequence length')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=16, help='Gradient accumulation steps')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping threshold, set to 0 for no clipping')
    parser.add_argument('--train_bsz_per_gpu', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--learning_rate', type=float, default=5e-6, help='Learning rate')
    parser.add_argument('--min_lr_ratio', type=float, default=0., help='Minimum learning rate ratio to peak learning rate')
    parser.add_argument('--warmup_rates', type=float, default=0., help='Warmup ratio')
    parser.add_argument('--n_epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--save_freq', type=int, default=10, help='Save frequency')

    # Others
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--action_dim', type=int, default=7, help='action dim')
    parser.add_argument('--use_robot_state', type=int, default=0)
    parser.add_argument('--load_action_from_latent', type=int, default=0)
    parser.add_argument('--load_action_from_pretrain', type=int, default=0)
    parser.add_argument('--image_token_num', type=int, default=576)
    parser.add_argument('--fast_view_num', type=int, default=1)
    parser.add_argument('--action_chunk', type=int, default=1)
    parser.add_argument('--use_pred', type=int, default=0)
    parser.add_argument('--use_tactile_vec', type=int, default=0)
    parser.add_argument('--use_tactile_deform', type=int, default=0)
    parser.add_argument(
        '--deform_encoder_ckpt', 
        type=str, 
        default='sharpa_wave_deform_encoder.pth',
        help='Path to the pretrained DeformEncoder weights'
    )

    args = parser.parse_args()
    
    # Set paths
    args.log_dir = os.path.join(args.log_dir, args.experiment_name)
    args.output_dir = os.path.join(args.output_dir, args.experiment_name)
    if args.run_name:
        args.output_dir = os.path.join(args.output_dir, args.run_name)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Start training
    train(args)     


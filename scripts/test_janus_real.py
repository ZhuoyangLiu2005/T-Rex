import os
import argparse
import torch
import json
import numpy as np
from PIL import Image
import PIL.Image
import io
import zmq
import pickle
import random
from dataclasses import dataclass
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor, ActionTokenizer

@dataclass
class VLChatProcessorOutput():
    sft_format: str
    input_ids: torch.Tensor
    pixel_values: torch.Tensor
    num_image_tokens: torch.IntTensor
    def __len__(self):
        return len(self.input_ids)

def model_load(args):
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(args.model_path)
    tokenizer = vl_chat_processor.tokenizer
    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
        args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16,
        flow=True, action_dim=args.action_dim, action_chunk=args.action_chunk,
        fast_and_slow=True, fast_image_num=1,
    )
    action_tokenizer = ActionTokenizer(tokenizer)
    statistics_path = os.path.join(os.path.dirname(args.model_path), "stats_data.json")
    with open(statistics_path, 'r') as f:
        stats_data = json.load(f)
    dataset_name=args.dataset_name
    statistic= {}
    statistic['action_mask'] = np.array(stats_data[dataset_name]['action']['mask'])
    statistic['action_min'] = np.array(stats_data[dataset_name]['action']['q01'])
    statistic['action_max'] = np.array(stats_data[dataset_name]['action']['q99'])
    if args.use_robot_state:
        statistic['state_mask'] = np.array(stats_data[dataset_name]['state']['mask'])
        statistic['state_min'] = np.array(stats_data[dataset_name]['state']['q01'])
        statistic['state_max'] = np.array(stats_data[dataset_name]['state']['q99'])
    statistic['tactile_f6_mask'] = np.array(stats_data[dataset_name]['tactile_f6']['mask'])
    statistic['tactile_f6_min'] = np.array(stats_data[dataset_name]['tactile_f6']['q01'])
    statistic['tactile_f6_max'] = np.array(stats_data[dataset_name]['tactile_f6']['q99'])
    return vl_gpt, vl_chat_processor, action_tokenizer, statistic

def model_predict(
        args, 
        vl_gpt, 
        vl_chat_processor, 
        action_tokenizer, 
        statistic, 
        task_description, 
        slow_image, 
        fast_image, 
        tactile_f6s_ori,
        state_fast=None, 
        step=0,
        old_chunk=None, 
        delay_steps=0, 
        blend_k=0.1
    ):
    device = f'cuda:{args.cuda}'
    vl_gpt = vl_gpt.to(device).eval()
    parallel_size = 1
    num_image_tokens = 576
    num_fast_images = 1

    def get_state_tokens(state_arr):
        state_arr = np.array(state_arr, dtype=np.float32)
        norm_state = np.where(
            statistic['state_mask'],
            np.clip(2 * (state_arr - statistic['state_min']) / (statistic['state_max'] - statistic['state_min'] + 1e-8) - 1, -1, 1),
            state_arr
        )
        return action_tokenizer(norm_state)

    state_tokens_fast = get_state_tokens(state_fast) if args.use_robot_state else ""

    # tactile f6 preprocess
    tactile_f6s = np.array(tactile_f6s_ori, dtype=np.float32).reshape(1, -1)
    normalized_tactile_f6s = np.where(
        statistic['tactile_f6_mask'],
        np.clip(2 * (tactile_f6s - statistic['tactile_f6_min']) / (statistic['tactile_f6_max'] - statistic['tactile_f6_min'] + 1e-8) - 1, -1, 1),
        tactile_f6s
    )
    normalized_tactile_f6s = normalized_tactile_f6s.reshape(1, -1, 6)
    normalized_tactile_f6s = torch.tensor(normalized_tactile_f6s).to(device)

    pre_data = []
    img_placeholder = vl_chat_processor.image_start_tag + vl_chat_processor.image_tag * num_image_tokens + vl_chat_processor.image_end_tag

    prompts = img_placeholder + task_description + img_placeholder * num_fast_images + state_tokens_fast

    conversation = [{"role": "<|User|>", "content": prompts}]
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
        
        noise = torch.randn(inputs_embeds.shape[0], args.action_chunk, args.action_dim, device=device)

        frozen_prefix_norm = None
        if args.use_rtc and old_chunk is not None and delay_steps > 0:
            frozen_prefix_unnorm = old_chunk[:delay_steps]
            f_prefix_norm = np.where(
                statistic['action_mask'],
                np.clip(2 * (frozen_prefix_unnorm - statistic['action_min']) / (statistic['action_max'] - statistic['action_min'] + 1e-8) - 1, -1, 1),
                frozen_prefix_unnorm
            )
            frozen_prefix_norm = torch.tensor(f_prefix_norm, dtype=torch.bfloat16, device=device).unsqueeze(0)
            samples = vl_gpt.forward_flow_rtc(
                inputs_embeds, noise, normalized_tactile_f6s,
                frozen_prefix=frozen_prefix_norm, 
                delay_steps=delay_steps
            )
        else:
            samples = vl_gpt.forward_flow(inputs_embeds, noise, normalized_tactile_f6s)
        
        normalized_actions = samples[0].cpu().numpy() 
        actions = np.where(
            statistic['action_mask'],
            0.5 * (normalized_actions + 1) * (statistic['action_max'] - statistic['action_min']) + statistic['action_min'],
            normalized_actions,
        )

        # ======== ACT smooth ========
        # new_chunk = actions  # shape: [chunk_size, action_dim]
        # if old_chunk is not None and delay_steps > 0:
        #     final_chunk = np.zeros_like(new_chunk)
        #     # 1. maintain the frozen prefix
        #     final_chunk[:delay_steps] = old_chunk[:delay_steps]
            
        #     # 2. do the weighted aggression to the overlapped future part
        #     overlap_len = args.action_chunk - delay_steps
        #     if overlap_len > 0:
        #         steps = np.arange(overlap_len)
        #         w_old = np.exp(-blend_k * steps).reshape(-1, 1)
        #         w_new = 1.0 - w_old
                
        #         old_residual = old_chunk[delay_steps:]
        #         new_future = new_chunk[delay_steps:]
                
        #         final_chunk[delay_steps:] = w_old * old_residual + w_new * new_future
                
        #     actions = final_chunk
        # ============================================

        return list(actions)


def main(args):
    print(f"Loading VLA model from {args.model_path}...")
    vl_gpt, vl_chat_processor, action_tokenizer, statistic = model_load(args)
    print("Model loaded successfully!")

    # Dummy Forward
    print("Warming up model...")
    dummy_img_slow = [Image.new('RGB', (224, 224), color = 'black')]
    dummy_img_fast = [Image.new('RGB', (224, 224), color = 'black'), Image.new('RGB', (224, 224), color = 'black'), Image.new('RGB', (224, 224), color = 'black')]
    dummy_state = np.zeros(args.action_dim) if args.use_robot_state else None
    dummy_tactilef6 = np.zeros((10, 6))
    dummy_action = model_predict(
        args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, 
        "dummy task", dummy_img_slow, dummy_img_fast, dummy_tactilef6, dummy_state, dummy_state
    )
    print(dummy_action)
    print("Warm-up complete.")


    with open(args.test_json_path, 'r') as f:
        train_data = json.load(f)
            
        sample = random.choice(train_data)
        task_description = sample["input_prompt"]
        print(f"\n[Test Sample Selected] Task: '{task_description}'")

        try:
            slow_images = [Image.open(img_path).convert('RGB') for img_path in sample["input_image_slow"]]
            fast_images = [Image.open(img_path).convert('RGB') for img_path in sample["input_image_fast"]]
            
            state_fast = np.array(sample["state_fast"], dtype=np.float32) if args.use_robot_state else None
            tactile_f6 = np.array(sample["tactile_f6"], dtype=np.float32)
            
            gt_action = np.array(sample["action"], dtype=np.float32)

            predicted_action = model_predict(
                args, vl_gpt, vl_chat_processor, action_tokenizer, statistic, 
                task_description, slow_images, fast_images, tactile_f6, state_fast
            )
            
            predicted_action = np.array(predicted_action)

            print("\n=== Prediction vs Ground Truth ===")
            print(f"Predicted Action Shape: {predicted_action.shape}")
            print(f"GT Action Shape:        {gt_action.shape}")
            
            pred_step_0 = predicted_action[0] if len(predicted_action.shape) > 1 else predicted_action
            gt_step_0 = gt_action[0] if len(gt_action.shape) > 1 else gt_action
            
            np.set_printoptions(precision=4, suppress=True)
            print(f"Predicted (Step 0): {pred_step_0}")
            print(f"GT Action (Step 0): {gt_step_0}")
            
            min_len = min(len(predicted_action), len(gt_action))
            if min_len > 0:
                mse = np.mean((predicted_action[:min_len] - gt_action[:min_len])**2)
                print(f"Mean Squared Error: {mse:.6f}")
                
        except FileNotFoundError as e:
            print(f"\n[Warning] Cannot load image, skip test\nerror: {e}")

    input("Training set check done, press enter to continue")

    # ZeroMQ Server
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://0.0.0.0:{args.port}")
    print(f"🚀 VLA Server is listening on port {args.port}...")

    step_counter = 0
    last_predicted_chunk = None

    while True:
        try:
            message = socket.recv()
            payload = pickle.loads(message)

            slow_img = Image.open(io.BytesIO(payload['image_head'])).convert('RGB')
            fast_img_head = Image.open(io.BytesIO(payload['image_head'])).convert('RGB')
            fast_img_wrist_right = Image.open(io.BytesIO(payload['image_wrist_right'])).convert('RGB')
            fast_img_wrist_left = Image.open(io.BytesIO(payload['image_wrist_left'])).convert('RGB')

            slow_image_list = [slow_img]
            fast_image_list = [fast_img_wrist_right, fast_img_wrist_left]

            task_description = payload['task_description']
            state_fast = payload['state_fast']

            tactile_f6 = payload['tactile_f6']

            # estimate, or use a reasonable number
            delay_steps = payload.get('delay_steps', 3)

            actions = model_predict(
                args=args,
                vl_gpt=vl_gpt,
                vl_chat_processor=vl_chat_processor,
                action_tokenizer=action_tokenizer,
                statistic=statistic,
                task_description=task_description,
                slow_image=slow_image_list,
                fast_image=fast_image_list,
                tactile_f6=tactile_f6,
                state_fast=state_fast,
                step=step_counter,
                old_chunk=last_predicted_chunk, # history Chunk
                delay_steps=delay_steps if last_predicted_chunk is not None else 0,
                blend_k=0.1
            )

            last_predicted_chunk = np.array(actions) if args.use_rtc else None

            response = {
                'status': 'success',
                'actions': actions
            }
            socket.send(pickle.dumps(response))
            step_counter += 1

            if step_counter % 10 == 0:
                print(f"Processed {step_counter} requests. Current task: {task_description}")

        except Exception as e:
            print(f"Error during prediction: {e}")
            socket.send(pickle.dumps({'status': 'error', 'message': str(e)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default='rlbench')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--test_json_path', type=str, default='0')
    parser.add_argument('--use_robot_state', type=int, default=1)
    parser.add_argument('--action_chunk', type=int, default=1)
    parser.add_argument('--action_dim', type=int, default=7)
    parser.add_argument('--use_pred', type=int, default=0)
    parser.add_argument('--use_rtc', type=int, default=0)
    parser.add_argument('--port', type=int, default=5555, help="Port for ZeroMQ IPC")
    
    args = parser.parse_args()
    main(args)


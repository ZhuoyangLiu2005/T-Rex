import torch

ckpt_path = "/media/liuzhuoyang/new_vla/Any2Point/Any2Point_CLIP_Lang/ckpts/Language_CLIP_Scan.pth"

try:
    print(f"Loading {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    
    # 你的加载逻辑是取 'model' 里的 'encoder.' 部分
    model_state = checkpoint.get('model', checkpoint) # 兼容有些ckpt直接就是state_dict
    
    print("Checking checkpoint values for NaN/Inf...")
    nan_count = 0
    inf_count = 0
    
    for k, v in model_state.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any():
                print(f"Found NaN in checkpoint key: {k}")
                nan_count += 1
            if torch.isinf(v).any():
                print(f"Found Inf in checkpoint key: {k}")
                inf_count += 1
                
    if nan_count == 0 and inf_count == 0:
        print("SUCCESS: The checkpoint file seems CLEAN.")
    else:
        print(f"FAILURE: The checkpoint contains {nan_count} NaNs and {inf_count} Infs.")
        
except Exception as e:
    print(f"Error loading checkpoint: {e}")
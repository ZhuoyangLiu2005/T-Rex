
import torch

base_ckpt_path = '/media/liuzhuoyang/new_vla/Any2Point/Any2Point_CLIP_Lang/ckpts/ViT-L-14.pt'  # Update with your actual path
base_ckpt = torch.load(base_ckpt_path)
pos_embed_1d = base_ckpt.state_dict()['positional_embedding']
print(pos_embed_1d.shape)  # torch.Size([77, 768])
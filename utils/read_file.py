import gzip
import pickle
import numpy as np
from PIL import Image
import json

#######---------------  pkl.gz
# with gzip.open("/gpfs/0607-cluster/guchenyang/Data/R1LITE/Compressed/0822/0.pkl.gz", 'rb') as f:
#     data = pickle.load(f)

# print(data[0])

# # ######---------------  npy
# episode = np.load("/gpfs/0607-cluster/chenhao/data/rlbench/keyframe_fast_slow_chunk8_addlast_0806/for_rlds/close_box/episode1.npy", allow_pickle=True)

# print(episode[0])
# print(len(episode))

# episode = np.load("/gpfs/0607-cluster/chenhao/RL4VLA/sft_data/PutOnPlateInScene25Main-v3/16400_freq5/data/success_proc_0_numid_6_epsid_6.npy", allow_pickle=True)

# print(episode[-1])
# print(len(episode))


# #######---------------  npz
# # 加载 .npz 文件
# with np.load('/gpfs/0607-cluster/chenhao/RL4VLA/sft_data/PutOnPlateInScene25Main-v3/16400_freq5/data/success_proc_0_numid_6_epsid_6.npz', allow_pickle=True) as data:
#     # 提取字典对象
#     episode_data = data['arr_0'].item()
#     print(len(episode_data['image']),len(episode_data['action']))

#     print(episode_data['action'])
#     # print(np.array(episode_data['image'][0]))
#     print(episode_data['action'].shape)
#     # print(episode_data)
#     # # 现在可以访问字典中的各个字段
#     print("Keys in the dictionary:", episode_data.keys())
#     print(episode_data['instruction'][0])
    
#     # # 访问图像数据
#     # images = episode_data['image']
#     # print(f"Number of images: {len(images)}")
#     # print(f"First image shape : {images[0].size}")
#     # print(np.array(images[0]))
    
#     # # 访问指令
#     # instruction = episode_data['instruction']
#     # print(f"Instruction: {instruction}")
    
#     # # 访问动作数据
#     # actions = episode_data['action']
#     # print(f"Actions shape: {actions.shape}")
#     # print(episode_data['action'])
    
#     # # 访问信息数据
#     # info = episode_data['info'][0]['elapsed_steps']
#     # print(f"Info length: {info}")

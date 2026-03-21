export PATH=/root/miniconda3/envs/openvla-oft/bin:$PATH
export HF_HOME=/mnt/data/HuggingFace
export PYTHONPATH=/mnt/data/chenhao/openvla-oft:/mnt/data/chenhao/LIBERO:$PYTHONPATH

source /root/miniconda3/bin/activate /root/miniconda3/envs/openvla-oft
cd /mnt/data/chenhao/openvla-oft
python experiments/robot/libero/regenerate_libero_dataset.py \
    --libero_task_suite libero_10 \
    --libero_raw_data_dir /mnt/data/chenhao_save/data/libero/libero_10 \
    --libero_target_dir /mnt/cpfs/chenhao/libero/libero_10_no_noops
# pretrain_egodex_ray.py
import ray
import subprocess
import os

@ray.remote(num_gpus=8)
def train_on_node(node_rank, master_addr, master_port, num_nodes):

    cmd = f"""
        source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot && \
        export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH && \
        export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface && \
        export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen:$PYTHONPATH && \
        export WANDB_MODE=online && \
        export WANDB_API_KEY=5bdc90c568050775a6d10650e64857fbbc76742e && \
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 && \
        \
        BKL_SRC="/mnt/amlfs-02/shared/human_egocentric/dniu/datasets/bkl_inlab/raw/playdata_reorganized/grouped_untilMarch31_reorganized" && \
        DATA_ROOT="/mnt/amlfs-07/shared/datasets/dniu/egodex/cotrain_processed_new" && \
        for group_dir in "${{BKL_SRC}}"/*/; do \
            group_name=$(basename "${{group_dir}}") && \
            link_path="${{DATA_ROOT}}/bkl_inlab_${{group_name}}" && \
            if [ ! -e "${{link_path}}" ]; then \
                ln -s "${{group_dir}}" "${{link_path}}" && \
                echo "Symlinked: ${{link_path}} -> ${{group_dir}}"; \
            fi; \
        done && \
        \
        cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts && \
        accelerate launch \
            --config_file ../config/sft_qwen.yaml \
            --num_processes {num_nodes * 8} \
            --num_machines {num_nodes} \
            --machine_rank {node_rank} \
            --main_process_ip {master_addr} \
            --main_process_port {master_port} \
            --deepspeed_multinode_launcher standard \
            train_qwen3vl_pretrain_egodex.py \
            --model_path /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/Qwen3-VL-2B-Instruct \
            --data_root ${{DATA_ROOT}} \
            --n_epochs 1 \
            --save_freq 1 \
            --action_dim 62 \
            --action_chunk 16 \
            --train_bsz_per_gpu 8 \
            --learning_rate 1e-4 \
            --min_lr_ratio 0 \
            --warmup_rates 0.03 \
            --weight_decay 0.01 \
            --gradient_accumulation_steps 2 \
            --output_dir /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp \
            --log_dir /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_qwen/exp \
            --experiment_name qwen3vl_egodex_pretrain \
            --run_name qwen3vl_2b_egodex_pretrain_bimanual_62d_stage1_0401 \
            --use_robot_state 1 \
            --image_size 384 288 \
            --num_workers 16
    """
    result = subprocess.run(["bash", "-c", cmd], capture_output=False)
    return result.returncode


def main():
    ray.init(address="auto")

    NUM_NODES = 8
    MASTER_PORT = 29500
    MASTER_ADDR = os.environ["MASTER_ADDR"]

    futures = [
        train_on_node.remote(node_rank, MASTER_ADDR, MASTER_PORT, NUM_NODES)
        for node_rank in range(NUM_NODES)
    ]

    results = ray.get(futures)
    print(f"All nodes finished with return codes: {results}")


if __name__ == "__main__":
    main()
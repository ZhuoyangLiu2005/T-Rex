cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_qwen/scripts

source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot

python extract_flow_tactile_proxy.py --episode_dir /mnt/amlfs-07/shared/datasets/dniu/egodex/cotrain_processed_new/batch1/part2_assemble_disassemble_furniture_bench_drawer_3181 --output_dir ./flow_tactile_vis --hand both --device cuda:0 --save_video

cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/dex_mot_expert

source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/dex_mot/bin:$PATH

python -m tactile_vqvae.extract_codes \
    --checkpoint /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/dex_mot_expert/tactile_vqvae/vqvae_f6_w16_k64_finger_0507_0939/latest.pt \
    --data_root  /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/data/midtrain/merged_inlab \
    --alignment  historical \
    --num_workers 4 \
    --batch_size 512
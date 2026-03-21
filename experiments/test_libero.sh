
cd /mnt/amlfs-01/home/dniu/Project/dex-mot/mot/last0_img
source /mnt/amlfs-01/home/dniu/anaconda3/bin/activate /mnt/amlfs-01/home/dniu/anaconda3/envs/last0
export PATH=/mnt/amlfs-01/home/dniu/anaconda3/envs/last0/bin:$PATH
export HF_HOME=/mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/huggingface
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/last0_img/LIBERO:$PYTHONPATH
export PYTHONPATH=/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/last0_img:/mnt/amlfs-01/home/dniu/Project/dex-mot/mot/last0_img/transformers:$PYTHONPATH
export WANDB_MODE=offline

# Launch LIBERO-Spatial evals
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint /mnt/amlfs-02/shared/human_egocentric/dniu/Dex-MoT/mot_arch/ckpts/last0_img/exp/janus_img_mot_flow_libero/janus_pro_siglip_1B_1e-4_mot_pretrainvlm_spatial_view2_wo_next_img_f1s1_10shot_0221/stage0/checkpoint-199-38600/tfmr \
  --task_suite_name libero_spatial \
  --cuda "0" \
  --use_pred False \
  --seed 0

# # Launch LIBERO-Object evals
# python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint moojink/openvla-7b-oft-fietuned-libero-object \
#   --task_suite_name libero_object

# # Launch LIBERO-Goal evals
# python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-goal \
#   --task_suite_name libero_goal

# # Launch LIBERO-10 (LIBERO-Long) evals
# python experiments/robot/libero/run_libero_eval.py \
#   --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-10 \
#   --task_suite_name libero_10



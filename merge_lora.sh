#!/bin/bash
#SBATCH --partition=intern
#SBATCH --gpus=a5000:2

python /home/avaidya7/CPT_VLM/scripts/merge_lora_weights.py \
    --model-path /home/avaidya7/checkpoints/llava-v1.5-7b-c_floodnet_nola_lora \
    --model-base liuhaotian/llava-v1.5-7b \
    --save-model-path /home/avaidya7/checkpoints/llava-v1.5-7b-c_floodnet_nola_lora_merged
#!/bin/bash
#SBATCH --partition=intern
#SBATCH --gpus=a5000:2

python /home/avaidya7/CPT_VLM/llava/model_vqa.py \
--model-path /home/avaidya7/checkpoints/llava-v1.5-7b-floodnet_nola_lora_merged \
--question-file /home/avaidya7/EuroSat_Eval.json \
--image-folder /home/avaidya7/EuroSAT/2750 \
--answers-file /home/avaidya7/eurosat_answer_nola_floodnet.jsonl



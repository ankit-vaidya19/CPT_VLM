#!/bin/bash
#SBATCH --partition=intern
#SBATCH --gpus=a5000:2

deepspeed /home/avaidya7/CPT_VLM/llava/train/train_mem.py \
    --lora_enable True --nola_enable True --nola_num_basis 512 --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed /home/avaidya7/CPT_VLM/scripts/zero3.json \
    --model_name_or_path liuhaotian/llava-v1.5-7b \
    --version v1 \
    --data_path /home/avaidya7/FloodNet_train.json \
    --image_folder /home/avaidya7/Sat_data/FloodNet/Images/Train_Image \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /home/avaidya7/checkpoints/llava-v1.5-7b-floodnet_nola \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --run_name llava1.5_nola_floodnet

#!/bin/bash

#$ -N norm_CommonVoice_MSE_EncConn
#$ -cwd
#$ -o /mnt/matylda6/ivendrame/wavlm_connector_lm/experiments/logs/$JOB_NAME_$JOB_ID.out
#$ -e /mnt/matylda6/ivendrame/wavlm_connector_lm/experiments/logs/$JOB_NAME_$JOB_ID.err
#$ -l gpu=1
#$ -q long.q@supergpu1*
#$ -l h=!supergpu19
#$ -l gpu_ram=32G,gpu=1,ram_free=50G,mem_free=50G

source  /mnt/matylda6/ivendrame/utils/miniconda/bin/activate /mnt/matylda6/ivendrame/utils/miniconda/envs/alt
echo "here: $(hostname)"

exp_name=norm_CommonVoice_MSE_EncConn
mkdir -p /mnt/matylda6/ivendrame/wavlm_connector_lm/tensorboard/connector_runs/$exp_name

mkdir -p /mnt/matylda6/ivendrame/wavlm_connector_lm/experiments/joint_text_speech/hf_mse_loss/$exp_name/outputs

TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 HF_HOME=/mnt/scratch/tmp/ivendrame/huggingface python -u /mnt/matylda6/ivendrame/wavlm_connector_lm/experiments/joint_text_speech/hf_mse_loss/train_asr.py --text_input --train_with_asr_text --train_encoder --dataset_dir /mnt/matylda6/ivendrame/wavlm_connector_lm/experiments/joint_text_speech/hf_mse_loss/ --output_dir /mnt/matylda6/ivendrame/wavlm_connector_lm/experiments/joint_text_speech/hf_mse_loss/$exp_name/outputs --log_dir /mnt/matylda6/ivendrame/wavlm_connector_lm/tensorboard/connector_runs/$exp_name --hidden_size 2048 --num_heads 12 --num_layers 12 --ff_size 2048 --peak_lr 1e-4 --min_lr_ratio 0.1 --early_stopping_patience 10 --max_steps 60_000 --per_device_train_asr_batch_size 4 --per_device_train_text_batch_size 4 --per_device_eval_batch_size 4 --dataloader_num_workers 2 --validation_steps 10000 --save_steps 10_000 --accumulation_steps 8 --weight_decay 0 --encoder_eval --lm_eval --mask_rate 0.15
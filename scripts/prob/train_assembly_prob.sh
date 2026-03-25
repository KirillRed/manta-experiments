#!/usr/bin/env bash
OPTS="--model_dir=/data1/kredensk/anticipation/manta/normal_train/assembly/checkpoints \
      --results_dir=/data1/kredensk/anticipation/manta/normal_train/assembly/checkpoints \
      --gt_path=/data1/kredensk/anticipation/manta/normal_train/assembly/gt \
      --features_path=/data1/kredensk/anticipation/manta/normal_train/assembly/features \
      --use_features \
      --use_inp_ch_dropout \
      --layer_type mamba \
      --part_obs \
      --conditioned_x0 \
      --num_diff_timesteps 1000 \
      --diff_loss_type l2 \
      --diff_obj pred_x0 \
      --action=train \
      --ds=assembly \
      --bz=16 \
      --lr=0.0005 \
      --model=bit-diff-pred-tcn \
      --num_epochs=100 \
      --epoch=100 \
      --num_stages=1 \
      --num_layers=15 \
      --channel_dropout_prob=0.5 \
      --sample_rate=6"

python ./src/main.py $OPTS



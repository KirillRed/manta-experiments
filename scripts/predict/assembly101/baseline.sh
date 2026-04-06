#!/usr/bin/env bash
OPTS="--model_dir=/data1/kredensk/anticipation/manta/normal_train/assembly/checkpoints \
      --results_dir=/data1/kredensk/anticipation/manta/normal_train/assembly/checkpoints \
      --gt_path=/data1/kredensk/anticipation/manta/datasets/assembly101/gt \
      --features_path=/data1/kredensk/anticipation/manta/datasets/assembly101/features \
      --use_inp_ch_dropout \
      --use_features \
      --part_obs \
      --layer_type mamba \
      --conditioned_x0 \
      --num_diff_timesteps 1000 \
      --num_infr_diff_timesteps 50 \
      --num_samples 25 \
      --diff_loss_type l2 \
      --diff_obj pred_x0 \
      --action=val \
      --ds=assembly \
      --bz=16 \
      --lr=0.0005 \
      --model=bit-diff-pred-tcn \
      --num_epochs=100 \
      --epoch=55 \
      --num_stages=1 \
      --num_layers=15 \
      --channel_dropout_prob=0.5 \
      --sample_rate=6"

python ./src/main.py $OPTS
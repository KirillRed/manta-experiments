#!/usr/bin/env bash
OPTS="--model_dir=/data1/kredensk/anticipation/manta/baseline/split2/50salads/checkpoints \
      --results_dir=/data1/kredensk/anticipation/manta/baseline/split2/50salads/checkpoints \
      --mapping_file=/data1/kredensk/anticipation/manta/datasets/breakfast/data/50salads/mapping.txt \
      --vid_list_file_test=/data1/kredensk/anticipation/manta/datasets/breakfast/data/50salads/splits/test.split2.bundle \
      --vid_list_file=/data1/kredensk/anticipation/manta/datasets/breakfast/data/50salads/splits/train.split2.bundle \
      --gt_path=/data1/kredensk/anticipation/manta/datasets/breakfast/data/50salads/groundTruth/ \
      --features_path=/data1/kredensk/anticipation/manta/datasets/breakfast/data/50salads/features/ \
      --split=2 \
      --conditioned_x0 \
      --use_features \
      --use_inp_ch_dropout \
      --layer_type mamba \
      --part_obs \
      --num_diff_timesteps 1000 \
      --num_infr_diff_timesteps 10 \
      --diff_loss_type l2 \
      --diff_obj pred_x0 \
      --action=train \
      --ds=bf \
      --bz=8 \
      --lr=0.001 \
      --model=bit-diff-pred-tcn \
      --num_epochs=100 \
      --epoch=100 \
      --num_stages=1 \
      --num_layers=15 \
      --channel_dropout_prob=0.4 \
      --sample_rate=3 \
      --num_workers=0"

python ./src/main.py $OPTS



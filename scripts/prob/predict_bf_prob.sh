#!/usr/bin/env bash
OPTS="--model_dir=/home/user/checkpoints/bf \
      --results_dir=/home/user/checkpoints/bf \
      --mapping_file=./datasets/breakfast/mapping.txt \
      --vid_list_file_test=./datasets/breakfast/splits/test.split1.bundle \
      --vid_list_file=./datasets/breakfast/splits/train.split1.bundle \
      --gt_path=./datasets/breakfast/groundTruth/ \
      --features_path=/home/user/bf/features/ \
      --split=1 \
      --ds=bf \
      --conditioned_x0 \
      --use_features \
      --use_inp_ch_dropout \
      --part_obs \
      --layer_type mamba \
      --num_diff_timesteps 1000 \
      --num_infr_diff_timesteps 50 \
      --diff_obj pred_x0 \
      --num_samples 25 \
      --action=val \
      --bz=1 \
      --lr=0.0005 \
      --model=bit-diff-pred-tcn \
      --num_epochs=100 \
      --epoch=90 \
      --num_stages=1 \
      --num_layers=15 \
      --channel_dropout_prob=0.4 \
      --sample_rate=3"

python ./src/main.py $OPTS




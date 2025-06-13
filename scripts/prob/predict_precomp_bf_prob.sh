#!/usr/bin/env bash
OPTS="--split=1 \
      --conditioned_x0 \
      --num_diff_timesteps 1000 \
      --num_infr_diff_timesteps 50 \
      --diff_loss_type l2 \
      --diff_obj pred_x0 \
      --num_samples 25 \
      --test_num_samples 25 \
      --ds=bf \
      --model=bit-diff-pred-tcn \
      --epoch=90 \
      --mapping_file=./datasets/breakfast/mapping.txt \
      --num_stages=1 \
      --num_layers=15 \
      --layer_type mamba \
      --sample_rate=3"

python ./src/main_diff_evaluate.py $OPTS




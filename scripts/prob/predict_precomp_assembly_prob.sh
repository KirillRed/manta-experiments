#!/usr/bin/env bash
OPTS="--conditioned_x0 \
      --num_diff_timesteps 1000 \
      --num_infr_diff_timesteps 50 \
      --diff_loss_type l2 \
      --diff_obj pred_x0 \
      --num_samples 25 \
      --test_num_samples 25 \
      --ds=assembly \
      --layer_type=mamba \
      --model=bit-diff-pred-tcn \
      --epoch=55 \
      --num_stages=1 \
      --num_layers=15 \
      --sample_rate=6"

python ./src/main_diff_evaluate.py $OPTS






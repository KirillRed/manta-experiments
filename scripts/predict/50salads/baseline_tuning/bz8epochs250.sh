#!/usr/bin/env bash
OPTS="--model_dir=/data1/kredensk/anticipation/manta/baseline/bz8epochs250/50salads/checkpoints \
      --results_dir=/data1/kredensk/anticipation/manta/baseline/bz8epochs250/50salads/checkpoints \
      --mapping_file=/data1/kredensk/anticipation/manta/datasets/breakfast/data/50salads/mapping.txt \
      --vid_list_file_test=/data1/kredensk/anticipation/manta/datasets/breakfast/data/50salads/splits/test.split1.bundle \
      --vid_list_file=/data1/kredensk/anticipation/manta/datasets/breakfast/data/50salads/splits/train.split1.bundle \
      --gt_path=/data1/kredensk/anticipation/manta/datasets/breakfast/data/50salads/groundTruth/ \
      --features_path=/data1/kredensk/anticipation/manta/datasets/breakfast/data/50salads/features/ \
      --split=1 \
      --ds=bf \
      --conditioned_x0 \
      --use_features \
      --use_inp_ch_dropout \
      --part_obs \
      --layer_type mamba \
      --num_diff_timesteps 1000 \
      --num_infr_diff_timesteps 10 \
      --diff_obj pred_x0 \
      --num_samples 25 \
      --action=val \
      --bz=1 \
      --lr=0.001 \
      --model=bit-diff-pred-tcn \
      --num_epochs=250 \
      --epoch=250 \
      --num_stages=1 \
      --num_layers=15 \
      --channel_dropout_prob=0.4 \
      --sample_rate=3"

python ./src/main.py $OPTS




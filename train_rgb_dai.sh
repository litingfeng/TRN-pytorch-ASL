#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 WANDB_NAME="debug" python main.py \
                     --dataset dai --modality RGB \
                     --arch BNInception --num_segments 3 \
                     --consensus_type TRN --batch_size 64





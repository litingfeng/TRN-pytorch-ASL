#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=5 WANDB_NAME="rgb_bs64_dai_hand" python main-hand.py \
                     --dataset dai --modality RGB \
                     --arch BNInception --num_segments 2 \
                     --consensus_type TRN --batch_size 64





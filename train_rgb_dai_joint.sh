#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=4 WANDB_NAME="rgb_bs64_dai_handjoint" python main-handjoint.py \
                     --dataset dai --modality RGB \
                     --arch BNInception --num_segments 3 \
                     --consensus_type TRN --batch_size 64





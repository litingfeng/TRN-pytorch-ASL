# Make prediction from mp4 video file (ffmpeg is required)
#python test_video.py --video_file sample_data/juggling.mp4 --rendered_output sample_data/predicted_video.mp4 --weight pretrain/TRN_moments_RGB_InceptionV3_TRNmultiscale_segment8_best.pth.tar --arch InceptionV3 --dataset moments

# Make prediction with input a a folder name with RGB frames
python test_video_dai.py \
       --frame_folder /dresden/gpu2/tl6012/data/ASL/isolated_signs/ \
       --test_segments 3 \
       --consensus_type TRNmultiscale \
       --weight /dresden/gpu2/tl6012/TRN/model/TRN_dai_RGB_BNInception_TRNmultiscale_segment3_newL3_lossnll_bs64_best.pth.tar \
       --arch BNInception \
       --dataset dai

#CUDA_VISIBLE_DEVICES=3 WANDB_NAME="2stream_bs64_seg3_multi" python test-dai-2stream.py \
#                     --dataset rachel --modality RGB \
#                     --arch BNInception --num_segments 3 \
#                     --resume /dresden/gpu2/tl6012/TRN-sweep/model/TRN_rachel_RGB_BNInception_TRNmultiscale_segment3_newL3_lossnll_bs64_best.pth.tar \
#                     --resume_of /dresden/gpu2/tl6012/TRN-sweep/model/TRN_rachel_Flow_BNInception_TRNmultiscale_segment3_newL2_lossnll_bs64_best.pth.tar \
#                     --consensus_type TRNmultiscale --batch_size 64 \
#                     --data_length 2 \
#                     --evaluate
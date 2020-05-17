# Temporal Relation Networks
**Introduction**: This repository is to apply TRN on [ASLLRP](http://dev1.cs.rutgers.edu:3000/dai/s/dai) dataset. The goal is to recognize sign labels, limited in lexical signs. In addition to TRN framework, keypoint detection, hand detection and left and right hand recognition is also included. 

**Note**: always use `git clone --recursive https://github.com/metalbubble/TRN-pytorch` to clone this project
Otherwise you will not be able to use the inception series CNN architecture.

![framework](http://relation.csail.mit.edu/framework_trn.png)
### Requirements
- PyTorch 1.3
- [Weights & Biases](https://www.wandb.com/)
- [Detectron2](https://github.com/facebookresearch/detectron2)
### Data preparation
#### Raw RGB image 
1. Extract frames.
```
python extract_frame.py
```
2. Cut videos to utterance clips. 
```bash
python cut_video_to_clip.py
```
3. Delete blank frames.
```
python delete_unvalid_frame.py
```
4. Generate vocabulary and handshape list.
```
python generate_vocabulary.py
```
5. Parse annotation xml file and generate json file for each utterance video. Note: you can also include handshape annotation in this step, while I do so in `process_dataset_dai_hand.py`
```
python parse_sign.py
```
6. Cut the utterance videos into sign videos, and reorganize them according to the label. Note: since the label name migh include '/', it's necessary to replace '/' with 'or'.
```
python GenerateClassificationFolders.py
```
7. Generate category list and train/test list. Each row in train/test list is `[video_id, num_frames, class_idx]`.
```
python process_dataset_dai.py
```
8. Add function `return_dai` in dataset_video.py

#### Optical flow image
To compute optical flow image of each frame, run `python extract_of.py`


#### Hand detection

1. Pretrain on Egohands dataset. 
```
python Hand/train_detectron2_ego.py
```
2. Finetune on my annotated images from ASLLRP dataset
```
python Hand/findtune_asl_detectron2.py
```
#### Body keypoint detection
```
python Hand/train_detectron2_keypoint.py
```
#### Left right hand recognition
Simply associate left right wrist keypoint with detected bounding boxes. Save in root/dai_lexical_handbox, each file is the bounding boxes in [left, right] order.
```
python Hand/classify_leftright_asl_detectron2.py
```
Then we are able to generate train/test list in which each row is  `[video_id, num_frames, class_idx, right_start_hs, left_start_hs, right_end_hs, left_end_hs]`.
```
python process_dataset_dai_hand.py
```
### Code

Core code to implement the Temporal Relation Network module is [TRNmodule](TRNmodule.py). It is plug-and-play on top of the TSN.

### Training and Testing

* The command to train single scale TRN using full frame. To train on hand image, use `main-hand.py`. To jointly train, use `main-handjoint.py`.

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
                     --dataset dai --modality RGB \
                     --arch BNInception --num_segments 3 \
                     --consensus_type TRN --batch_size 64 
```

* The command to train multi-scale TRN
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
                     --dataset dai --modality RGB \
                     --arch BNInception --num_segments 3 \
                     --consensus_type TRNmultiscale --batch_size 64 
```

* The command to test the single scale TRN

```bash
python test_video_dai.py \
       --frame_folder $FRAME_FOLDER \
       --test_segments 3 \
       --consensus_type TRN \
       --weight $WEIGHT_PATH \
       --arch BNInception \
       --dataset dai
```

* The command to test the single scale 2-stream TRN

```bash
python test-dai-2stream.py \
        --dataset rachel --modality RGB \
        --arch BNInception --num_segments 3 \
        --resume $RGB_WEIGHTPATH \
        --resume_of $OF_WEIGHTPATH \
        --consensus_type TRN --batch_size 64 \
        --data_length 3 \
        --evaluate
```

### Reference:
B. Zhou, A. Andonian, and A. Torralba. Temporal Relational Reasoning in Videos. European Conference on Computer Vision (ECCV), 2018. [PDF](https://arxiv.org/pdf/1711.08496.pdf)
```
@article{zhou2017temporalrelation,
    title = {Temporal Relational Reasoning in Videos},
    author = {Zhou, Bolei and Andonian, Alex and Oliva, Aude and Torralba, Antonio},
    journal={European Conference on Computer Vision},
    year={2018}
}
```


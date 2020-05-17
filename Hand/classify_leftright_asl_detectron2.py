import os
import numpy as np
import cv2
import ntpath
import random
import itertools
import pandas as pd
from detectron2.structures import BoxMode, Keypoints
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2 import model_zoo
import datetime

_KEYPOINT_THRESHOLD = 0.05
img_dir = '/dresden/gpu2/tl6012/data/ASL/frames_hand_ann/images/'
keypoint_names = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
    "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]

def iou(bb_test,bb_gt):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)

def get_asl_dicts(split):

    df = pd.read_csv(os.path.join('/dresden/gpu2/tl6012/data/ASL/frames_hand_ann/data', '{}_labels.csv'.format(split)))
    #classes = sorted(df.class.unique().tolist())
    #print('classes ', classes)
    #assert(len(classes) == 2)

    dataset_dicts = []
    for image_id, img_name in enumerate(df.filename.unique()):

        record = {}

        image_df = df[df.filename == img_name]

        file_path = os.path.join(img_dir, split, img_name)
        record["file_name"] = file_path
        record["image_id"] = image_id
        record["height"] = int(image_df.iloc[0].height)
        record["width"] = int(image_df.iloc[0].width)

        objs = []
        for _, row in image_df.iterrows():
            xmin = int(row.xmin)
            ymin = int(row.ymin)
            xmax = int(row.xmax)
            ymax = int(row.ymax)

            poly = [
                (xmin, ymin), (xmax, ymin),
                (xmax, ymax), (xmin, ymax)
            ]
            poly = list(itertools.chain.from_iterable(poly))

            obj = {
                "bbox": [xmin, ymin, xmax, ymax],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,#classes.index(row.classname),
                "iscrowd": 0
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


for d in ["train", "test"]:
  DatasetCatalog.register("hand_" + d, lambda d=d: get_asl_dicts(d))
  MetadataCatalog.get("hand_" + d).set(thing_classes=['hand'])

hand_metadata = MetadataCatalog.get("hand_train")

#visulization
# dataset_dicts = get_asl_dicts("train")
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=hand_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow('vis ', vis.get_image()[:, :, ::-1])
#     k = cv2.waitKey(0)
#     if k == 27:
#         break

##################### train #####################################
class CocoTrainer(DefaultTrainer):

  @classmethod
  def build_evaluator(cls, cfg, dataset_name, output_folder=None):

    if output_folder is None:
        os.makedirs("coco_eval", exist_ok=True)
        output_folder = "coco_eval"

    return COCOEvaluator(dataset_name, cfg, False, output_folder)


cfg = get_cfg()

cfg.merge_from_file(
  model_zoo.get_config_file(
    "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
  )
)

#cfg.MODEL.WEIGHTS = '/ajax/users/tl601/projects/handtracking/output/model_final.pth'
cfg.OUTPUT_DIR = '/dresden/gpu2/tl6012/asl/detectron2/finetune/'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")

cfg.DATASETS.TRAIN = ("hand_train",)
cfg.DATASETS.TEST = ("hand_test",)
cfg.DATALOADER.NUM_WORKERS = 4

cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 100
cfg.SOLVER.MAX_ITER = 150
cfg.SOLVER.STEPS = (100, 150)
cfg.SOLVER.GAMMA = 0.05


cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.TEST.EVAL_PERIOD = 50
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor_box = DefaultPredictor(cfg)
#############################################################

cfg = get_cfg()

cfg.merge_from_file(
  model_zoo.get_config_file(
    "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
  )
)
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
  "COCO-Keypoints/keypoint_rcnn_X_101_32x8d_FPN_3x.yaml"
)
cfg.OUTPUT_DIR = '/dresden/gpu2/tl6012/asl/detectron2/finetune_kp/'

cfg.DATASETS.TRAIN = ("asl_kp_train",)
cfg.DATASETS.TEST = ("asl_kp_test",)
cfg.DATALOADER.NUM_WORKERS = 4

cfg.SOLVER.IMS_PER_BATCH = 16
cfg.SOLVER.BASE_LR = 0.001
cfg.SOLVER.WARMUP_ITERS = 1000
cfg.SOLVER.MAX_ITER = 1500
cfg.SOLVER.STEPS = (1000, 1500)
cfg.SOLVER.GAMMA = 0.05


cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

cfg.TEST.EVAL_PERIOD = 500
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
predictor_kp = DefaultPredictor(cfg)

############# evaluate ########################


################ evaluation and visualization on test set #####################

save_dir = "/dresden/gpu2/tl6012/data/ASL/dai_lexical_handbox/"
os.makedirs(save_dir, exist_ok=True)
filename_imglist_val = '/ajax/users/tl601/projects/TRN-pytorch/dai/val_videofolder_20.txt'
filename_imglist_train = '/ajax/users/tl601/projects/TRN-pytorch/dai/train_videofolder_20.txt'
img_dir = '/dresden/gpu2/tl6012/data/ASL/isolated_signs/'
test_image_paths = list(open(filename_imglist_val)) + list(open(filename_imglist_train))
print('total frames ', len(test_image_paths))
hands = ['left', 'right']
for clothing_image in test_image_paths:
  file_paths = os.path.join(img_dir, clothing_image.strip().split(' ')[0])
  for file_path in os.listdir(file_paths):
      file_name = os.path.join(save_dir, clothing_image.strip().split(' ')[0])
      os.makedirs(file_name, exist_ok=True)
      savename = os.path.join(file_name, os.path.basename(file_path).split('.')[0])
      if os.path.exists(savename+'.npy'):
          print('exists ', savename)
          continue
      print(savename)

      os.makedirs(file_name, exist_ok=True)
      file_path = os.path.join(file_paths, file_path)

      im = cv2.imread(file_path)
      height, width, channels = im.shape
      outputs = predictor_box(im)
      instances = outputs["instances"].to("cpu")
      boxes = instances.pred_boxes if instances.has("pred_boxes") else None
      boxes = boxes.tensor.numpy()
      scores = instances.scores.numpy() if instances.has("scores") else None

      outputs = predictor_kp(im)
      instances = outputs["instances"].to("cpu")

      # process keypoints
      keypoints = instances.pred_keypoints if instances.has("pred_keypoints") else None
      if isinstance(keypoints, Keypoints):
          keypoints = keypoints.tensor
      keypoints = np.asarray(keypoints)
      keypoints = keypoints[0]
      #print('kepoints ', keypoints.shape, '\t', boxes.shape)
      visible = {}
      for idx, keypoint in enumerate(keypoints):
          # draw keypoint
          x, y, prob = keypoint
          #if prob > _KEYPOINT_THRESHOLD:
          if idx in [9,10]:
            cv2.circle(im, (x,y), radius=5, color=(0,0,255), thickness=-1)
          keypoint_name = keypoint_names[idx]
          visible[keypoint_name] = (x, y)

      # generate boxes around left & right wrist
      leftwrist_box = (int(visible['left_wrist'][0]-60), int(visible['left_wrist'][1]-60),
                       int(visible['left_wrist'][0]+60), int(visible['left_wrist'][1]+60))
      rightwrist_box = (int(visible['right_wrist'][0] - 60), int(visible['right_wrist'][1] - 60),
                       int(visible['right_wrist'][0] + 60), int(visible['right_wrist'][1] + 60))

      ious = [iou(boxes[i], b) for i in range(min(boxes.shape[0], 2)) for b in [leftwrist_box, rightwrist_box]]
      #print('ious ', ious)
      preds_inds = [np.argmax(ious[p:(p+2)]) for p in range(0,len(ious),2)]
      #print('preds_ints ', preds_inds)
      preds = ['left' if i == 0 else 'right' for i in preds_inds]
      #print(preds)

      # process same prediction
      if len(preds) == 2 and preds[0] == preds[1]:
          keep = np.argmax([abs(ious[p+1]-ious[p])for p in range(0,len(ious),2)])
          if keep == 0: # keep the first preds
              preds = [preds[0]]
              preds += [i for i in hands if i not in preds]
          elif keep == 1:
              kept = preds[1]
              preds = [i for i in hands if i != preds[1]]
              preds += [kept]
          else:
              print('wrong')
              exit()

      #print('modi ', preds)
      if len(preds) == 2:
        left_box = boxes[preds.index('left')]
        right_box = boxes[preds.index('right')]
        save_boxes = np.vstack((left_box, right_box))
        assert(save_boxes.shape == (2,4))
      else:
        if boxes.shape[0] == 1:
            save_boxes = boxes[0].reshape(1,4)
        elif boxes.shape[0] == 0:
            save_boxes = np.zeros((1,4))

        assert(save_boxes.shape == (1,4))

      np.save(savename, save_boxes)

      # visualize
      # default_font_size = max(
      #         np.sqrt(height * width) // 90, 10 // 1.
      #     )
      # for i in range(min(boxes.shape[0], 2)):
      #     x0, y0, x1, y1 = boxes[i]
      #     x0, x1, y0, y1 = int(max(0, x0)), int(min(x1, width-1)), int(max(y0, 0)), int(min(y1, height-1))
      #     cv2.rectangle(im, (x0, y0), (x1, y1), (77, 255, 9), 3, 1)
      #
      #     text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
      #
      #     horiz_align = "left"
      #     height_ratio = (y1 - y0) / np.sqrt(height * width)
      #     font_size = (
      #           np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
      #           * 0.5
      #           * default_font_size
      #     )
      #     cv2.putText(im, preds[i], text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
      #
      #
      # file_name = '_'.join([ntpath.basename(clothing_image.strip().split(' ')[0]), os.path.basename(file_path)])
      # print(im.shape, ' file_name ', file_name)
      # if not cv2.imwrite(os.path.join(save_dir,file_name), im):
      #     raise Exception("Could not write image")

print('finished')
#################################################################################


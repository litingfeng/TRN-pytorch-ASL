import os
import numpy as np
import cv2
import ntpath
import random
import itertools
import pandas as pd
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2 import model_zoo
import datetime

img_dir = '/dresden/gpu2/tl6012/data/ASL/frames_hand_ann/images/'


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

cfg.MODEL.WEIGHTS = '/dresden/gpu2/tl6012/asl/detectron2/train_ego/model_final.pth'
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

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


############# evaluate ########################

cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.85
predictor = DefaultPredictor(cfg)

################ evaluation and visualization on test set ######################
evaluator = COCOEvaluator("hand_test", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "hand_test")
inference_on_dataset(trainer.model, val_loader, evaluator)

os.makedirs("/dresden/gpu2/tl6012/data/ASL/frames_hand_ann/images_crop/train", exist_ok=True)
test_df = pd.read_csv(os.path.join('/dresden/gpu2/tl6012/data/ASL/frames_hand_ann/data', '{}_labels.csv'.format('train')))
test_image_paths = test_df.filename.unique()
for clothing_image in test_image_paths:
  file_path = os.path.join(img_dir, 'train', clothing_image)
  im = cv2.imread(file_path)
  height, width, channels = im.shape
  outputs = predictor(im)
  instances = outputs["instances"].to("cpu")
  boxes = instances.pred_boxes if instances.has("pred_boxes") else None
  boxes = boxes.tensor.numpy()
  scores = instances.scores.numpy() if instances.has("scores") else None


  for i in range(min(boxes.shape[0], 2)):
      x0, y0, x1, y1 = boxes[i]
      x0, x1, y0, y1 = max(0, x0), min(x1, width-1), max(y0, 0), min(y1, height-1)
      im_crop = im[int(y0):int(y1), int(x0):int(x1), :]

      file_name = ntpath.basename(clothing_image)
      print('file_name ', file_name)
      write_res = cv2.imwrite(f'/dresden/gpu2/tl6012/data/ASL/frames_hand_ann/images_crop/train/{file_name}', im_crop)

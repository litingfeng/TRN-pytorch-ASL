import os
import numpy as np
import cv2
import glob
import ntpath
import random
import itertools
import pandas as pd
from detectron2.structures import BoxMode, Boxes, RotatedBoxes
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances, load_coco_json

#img_dir = '/dresden/gpu2/tl6012/data/EgoHands/images/'
img_dir = '/dresden/gpu2/tl6012/data/ASL/frames_hand_ann/images/'
root_dir = '/dresden/gpu2/tl6012/data/ASL/frames_hand_ann/data/'
keypoint_names = ["nose","left_eye","right_eye","left_ear","right_ear","left_shoulder",
    "right_shoulder","left_elbow","right_elbow","left_wrist","right_wrist",
    "left_hip","right_hip","left_knee","right_knee","left_ankle","right_ankle"]
keypoint_flip_map = [
     ("left_eye", "right_eye"),
     ("left_ear", "right_ear"),
     ("left_shoulder", "right_shoulder"),
     ("left_elbow", "right_elbow"),
     ("left_wrist", "right_wrist"),
     ("left_hip", "right_hip"),
     ("left_knee", "right_knee"),
     ("left_ankle", "right_ankle"),
 ]
KEYPOINT_CONNECTION_RULES = [
    # face
    ("left_ear", "left_eye", (102, 204, 255)),
    ("right_ear", "right_eye", (51, 153, 255)),
    ("left_eye", "nose", (102, 0, 204)),
    ("nose", "right_eye", (51, 102, 255)),
    # upper-body
    ("left_shoulder", "right_shoulder", (255, 128, 0)),
    ("left_shoulder", "left_elbow", (153, 255, 204)),
    ("right_shoulder", "right_elbow", (128, 229, 255)),
    ("left_elbow", "left_wrist", (153, 255, 153)),
    ("right_elbow", "right_wrist", (102, 255, 224)),
    # lower-body
    # ("left_hip", "right_hip", (255, 102, 0)),
    # ("left_hip", "left_knee", (255, 255, 77)),
    # ("right_hip", "right_knee", (153, 255, 204)),
    # ("left_knee", "left_ankle", (191, 255, 128)),
    # ("right_knee", "right_ankle", (255, 195, 77)),
]

def get_asl_dicts(split):

    #df = pd.read_csv(os.path.join(img_dir, split, '{}_labels.csv'.format(split)))
    df = pd.read_csv(os.path.join('/dresden/gpu2/tl6012/data/ASL/frames_hand_ann/data', '{}_labels.csv'.format(split)))
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
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


# for d in ["train", "test"]:
#   DatasetCatalog.register("hand_" + d, lambda d=d: get_egohands_dicts(d))
#   MetadataCatalog.get("hand_" + d).set(thing_classes=['hand'])


for d in ['train', 'test']:
    register_coco_instances("asl_kp_{}".format(d), {}, root_dir+"asl_keypoint_{}.json".format(d), img_dir+"{}".format(d))
    MetadataCatalog.get("asl_kp_" + d).set(keypoint_names=keypoint_names)
    MetadataCatalog.get("asl_kp_" + d).set(keypoint_flip_map=keypoint_flip_map)
    MetadataCatalog.get("asl_kp_" + d).set(keypoint_connection_rules=KEYPOINT_CONNECTION_RULES)



hand_metadata = MetadataCatalog.get("asl_kp_test")
dataset_dicts = load_coco_json(root_dir+"asl_keypoint_test.json", img_dir+"test", 'asl_kp_test')
#visulization
# for d in random.sample(dataset_dicts, 3):
#     img = cv2.imread(d["file_name"])
#     visualizer = Visualizer(img[:, :, ::-1], metadata=hand_metadata, scale=0.5)
#     vis = visualizer.draw_dataset_dict(d)
#     cv2.imshow('vis ', vis.get_image()[:, :, ::-1])
#     k = cv2.waitKey(0)
#     if k == 27:
#         break
#
# exit()

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

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = CocoTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

################# evaluation #####################################
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
predictor = DefaultPredictor(cfg)

# sample visualization
for d in random.sample(dataset_dicts, 3):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=hand_metadata,
                   scale=0.8,
                   #instance_mode=ColorMode.IMAGE
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow('vis ', v.get_image()[:, :, ::-1])
    k = cv2.waitKey(0)
    if k == 27:
        break
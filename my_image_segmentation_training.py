import os
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch
)
from detectron2.data import build_detection_train_loader
from detectron2.modeling import build_model
from detectron2.utils import comm
from yolov7.utils.allreduce_norm import all_reduce_norm
from yolov7.config import add_yolo_config
from yolov7.data.dataset_mapper import MyDatasetMapper, MyDatasetMapper2
from yolov7.evaluation.coco_evaluation import COCOMaskEvaluator

"""
Script used for training instance segmentation, i.e. SparseInst.
"""
from detectron2.data.datasets.coco import load_coco_json, register_coco_instances
from train_det import Trainer, setup

def register_custom_datasets():
	##facemask dataset
    DATASET_ROOT = "./datasets/balloon_dataset/balloon/"
    ANN_ROOT = DATASET_ROOT
    TRAIN_PATH = os.path.join(DATASET_ROOT, "train")
    VAL_PATH = os.path.join(DATASET_ROOT, "val")
    TRAIN_JSON = os.path.join(ANN_ROOT, "train/via_region_data.json")
    VAL_JSON = os.path.join(ANN_ROOT, "val/via_region_data.json")

    register_coco_instances("balloon_train",{}, TRAIN_JSON, TRAIN_PATH)
    register_coco_instances("balloon_val", {}, VAL_JSON, VAL_PATH)

	##ADD YOUR DATASET CONFIG HERE
	##dataset names registerd must be unique, different than any of above
#register_custom_datasets()

import numpy as np
import os, json, cv2, random
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

DATASET_ROOT = "./balloon_dataset/balloon/"
img_dir = "./balloon_dataset/balloon/"
ANN_ROOT = DATASET_ROOT
TRAIN_PATH = os.path.join(DATASET_ROOT, "train")
VAL_PATH = os.path.join(DATASET_ROOT, "val")
TRAIN_JSON = os.path.join(ANN_ROOT, "train/via_region_data.json")
VAL_JSON = os.path.join(ANN_ROOT, "val/via_region_data.json")

import json
import detectron2
def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)
        
    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px,py)]
            poly = [p for x in poly for p in x]
            
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts
for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d:get_balloon_dicts("balloon_dataset/balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])

balloon_metadata = MetadataCatalog.get("balloon_train")
class Trainer(DefaultTrainer):

    custom_mapper = None

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOMaskEvaluator(dataset_name, output_dir=output_folder)

    @classmethod
    def build_train_loader(cls, cfg):
        cls.custom_mapper = MyDatasetMapper(cfg, True)
        return build_detection_train_loader(cfg, mapper=cls.custom_mapper)

    @classmethod
    def build_model(cls, cfg):
        model = build_model(cfg)
        return model

    def run_step(self):
        self._trainer.iter = self.iter
        self._trainer.run_step()
        if comm.get_world_size() == 1:
            self.model.update_iter(self.iter)
        else:
            self.model.module.update_iter(self.iter)

def setup(args):
    cfg = get_cfg()
    add_yolo_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

import torch, detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog
from detectron2.utils.visualizer import Visualizer
import random
import cv2
from detectron2.data import MetadataCatalog
import os
from matplotlib import pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

register_coco_instances("line_dataset_train", {}, "data/line_annotations.json", "data/")
dataset_dicts = DatasetCatalog.get("line_dataset_train")

for d in dataset_dicts:  # Adjust number of samples as needed
    print(d['file_name'])
    img = cv2.imread(d["file_name"])
    if d["file_name"] == "data/K.2.jpeg":
    
        visualizer = Visualizer(img[:, :, ::-1]x, metadata=MetadataCatalog.get("line_dataset_train"), scale=1)
        vis = visualizer.draw_dataset_dict(d)
        original_image = vis.get_image()[:, :, ::-1]
        plt.imshow(original_image)
        plt.show()

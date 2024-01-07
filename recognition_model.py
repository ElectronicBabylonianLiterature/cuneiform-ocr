# Copyright (c) OpenMMLab. All rights reserved.
from random import random

import matplotlib

from line_detection import line_detection

# from mmcls import init_model, inference_model

matplotlib.use("tkAgg")

import torch
import torchvision.transforms.functional
from matplotlib import pyplot as plt
from torch import nn


class Recognition(nn.Module):
    def __init__(self, detection_model, classification_model, classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.detection_model = detection_model
        self.data_preprocessor = self.detection_model.data_preprocessor
        self.classification_model = classification_model
        self.CLASSES = classes
        # this sorting is because thats the way the model was trained in mmcls
        self.CLASSES_CLASSIFICATION = sorted(self.CLASSES)
        self.mapping = {self.CLASSES_CLASSIFICATION.index(a): self.CLASSES.index(a) for a in
                        self.CLASSES_CLASSIFICATION}

    def show(self, image, points):
        # for debugging
        fig, ax = plt.subplots(frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax, aspect="auto")
        ax.imshow(image, cmap="Greys", aspect="auto")
        for point in points:
            ax.plot([x[0] for x in point], [x[1] for x in point], color="red", linewidth=1)
        plt.show()

    def test_step(self, data):
        result = self.detection_model.test_step(data)
        labels = []
        scores = []
        assert len(result) == 1
        _centroids = []
        scale_width, scale_height = result[0].scale_factor
        for counter, bbox in enumerate(result[0].pred_instances.bboxes):
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            x, y, w, h = x * scale_width, y * scale_height, w * scale_width, h * scale_height
            x, y, w, h = list(map(int, [y, x, h, w]))
            _centroids.append([y + h / 2, x + w / 2])

            crop = torchvision.transforms.functional.crop(data["inputs"][0], x-10, y-10, w+10, h+10)
            img = torchvision.transforms.Resize((380, 380))(crop)
            img = img.flip(0)
            img = img.float()
            mean = torch.Tensor([[[86.6589]], [[67.9274]], [[53.7833]]])
            std = torch.Tensor([[[68.9897]], [[57.2049]], [[48.2306]]])
            img = (img - mean) / std
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img = img.unsqueeze(0).to(device)

            _result = self.classification_model.predict(img, None)[0]
            pred = _result.pred_label.item()
            pred = self.mapping[pred]

            labels.append(pred)
            scores.append(1)
        result[0].pred_instances.labels = torch.Tensor(labels).long()
        result[0].pred_instances.scores = torch.Tensor(scores)
        #result[0].img_path.split("/")[-1].split(".")[:-1]
        # generate random string for name
        name = "".join([str(int(random() * 100)) for _ in range(100)])
        #self.do_line_detection(_centroids, labels, name)
        return result


    def do_line_detection(self, _centroids, labels, name):
        ordering, points_ordered = line_detection(torch.Tensor(_centroids), [self.CLASSES[label] for label in labels])
        # write ordering to file
        with open(f"inference_output/{name}_ordering.txt", "w") as f:
            for ordering_elem in ordering:
                print(" ".join(ordering_elem))
                f.write(" ".join(ordering_elem) + "\n")
            print("-"* 50)

# Copyright (c) OpenMMLab. All rights reserved.
import shutil
from random import random

import matplotlib
import numpy as np

from line_detection import SequentialRANSAC

#from mmcls import init_model, inference_model

matplotlib.use("tkAgg")
import argparse
import os
import os.path as osp

import torch
import torchvision.transforms.functional
from matplotlib import pyplot as plt, cm
from mmengine.config import Config
from mmengine.runner import Runner
from torch import nn
from torchvision.utils import save_image

import torchvision.transforms.functional as F

from mmdet.engine.hooks.utils import trigger_visualization_hook


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    args = parser.parse_args()
    os.environ['LOCAL_RANK'] = "0"
    return args

def show(inputs):
    for i in range(inputs.shape[0]):
        show = torch.clone(inputs).cpu()[i]
        show = show.permute(1, 2, 0)
        plt.imshow(show)
        plt.show()

def classification_config(config, checkpoint):
    cfg = Config.fromfile(config)
    cfg.load_from = checkpoint
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    return runner.model

def line_detection(centroids, classes):
    centroids = centroids.numpy()
    try:
        rns = SequentialRANSAC().fit(centroids)
        ordering = rns.ordering_.tolist()
        line_assignment = rns.labels_.tolist()
    except ValueError as e:
        print(f"RANSAC failed in batch elem {img_indx}")
        ordering = None
        line_assignment = None
    points_ordered = [[centroids.tolist()[order] for order in ordering] for ordering in rns.ordering__]
    ordering = [[classes[order] for order in order_list] for order_list in rns.ordering__]
    asd = sum(len(order) for order in ordering)
    return ordering, points_ordered


class Recognition(nn.Module):
    def __init__(self, detection_model, classification_model):
        super().__init__()
        #shutil.rmtree("debug")
        #os.mkdir("debug")
        #self.cls = init_model("configs/efficient_net.py", "checkpoints/efficient_net/epoch_100.pth")
        self.detection_model = detection_model
        self.classification_model = classification_model
        #self.CLASSES = ['ABZ579', 'ABZ13', 'ABZ342', 'ABZ70', 'ABZ461', 'ABZ480', 'ABZ142', 'ABZ1', 'ABZ231', 'ABZ533', 'ABZ449', 'ABZ318', 'ABZ75', 'ABZ61', 'ABZ381', 'ABZ354', 'ABZ139', 'ABZ597', 'ABZ536', 'ABZ308', 'ABZ330', 'ABZ328', 'ABZ15', 'ABZ86', 'ABZ214', 'ABZ545', 'ABZ73', 'ABZ295', 'ABZ55', 'ABZ335', 'ABZ371', 'ABZ537', 'ABZ457', 'ABZ68', 'ABZ151', 'ABZ69', 'ABZ353', 'ABZ5', 'ABZ366', 'ABZ296', 'ABZ411', 'ABZ84', 'ABZ396', 'ABZ206', 'ABZ58', 'ABZ376', 'ABZ324', 'ABZ99', 'ABZ384', 'ABZ59', 'ABZ532', 'ABZ334', 'ABZ589', 'ABZ145', 'ABZ383', 'ABZ586', 'ABZ343', 'ABZ74', 'ABZ399', 'ABZ212', 'ABZ211', 'ABZ7', 'ABZ78', 'ABZ367', 'ABZ38', 'ABZ319', 'ABZ115', 'ABZ85', 'ABZ322', 'ABZ207', 'ABZ144', 'ABZ112', 'ABZ97', 'ABZ427', 'ABZ60', 'ABZ79', 'ABZ80', 'ABZ52', 'ABZ312', 'ABZ142a', 'ABZ232', 'ABZ535', 'ABZ314', 'ABZ331', 'ABZ167', 'ABZ128', 'ABZ172', 'ABZ6', 'ABZ575', 'ABZ331e+152i', 'ABZ554', 'ABZ134', 'ABZ339', 'ABZ465', 'ABZ12', 'ABZ306', 'ABZ397', 'ABZ570', 'ABZ2', 'ABZ147', 'ABZ148', 'ABZ440', 'ABZ401', 'ABZ230', 'ABZ441', 'ABZ104', 'ABZ472', 'ABZ313', 'ABZ595', 'ABZ298', 'ABZ412', 'ABZ455', 'ABZ62', 'ABZ101', 'ABZ468', 'ABZ471', 'ABZ111', 'ABZ483', 'ABZ538', 'ABZ87', 'ABZ143', 'ABZ565', 'ABZ205', 'ABZ126', 'ABZ50', 'ABZ72', 'ABZ152', 'ABZ138', 'ABZ393', 'ABZ406', 'ABZ307', 'ABZ124', 'ABZ94', 'ABZ164', 'ABZ398', 'ABZ529', 'ABZ559', 'ABZ131', 'ABZ437', 'ABZ56', 'ABZ9', 'ABZ191']
        #self.CLASSES = [*self.CLASSES, "unknown"]
        self.CLASSES =  [
    'ABZ579',
    'ABZ13',
    'ABZ480',
    'ABZ70',
    'ABZ342',
    'ABZ597',
    'ABZ461',
    'ABZ142',
    'ABZ381',
    'ABZ1',
    'ABZ61',
    'ABZ318',
    'ABZ533',
    'ABZ231',
    'ABZ449',
    'ABZ75',
    'ABZ354',
    'ABZ545',
    'ABZ139',
    'ABZ330',
    'ABZ536',
    'ABZ308',
    'ABZ86',
    'ABZ15',
    'ABZ328',
    'ABZ214',
    'ABZ73',
    'ABZ295',
    'ABZ55',
    'ABZ537',
    'ABZ69',
    'ABZ371',
    'ABZ296',
    'ABZ457',
    'ABZ151',
    'ABZ411',
    'ABZ68',
    'ABZ335',
    'ABZ366',
    'ABZ5',
    'ABZ324',
    'ABZ396',
    'ABZ353',
    'ABZ99',
    'ABZ206',
    'ABZ84',
    'ABZ532',
    'ABZ376',
    'ABZ58',
    'ABZ384',
    'ABZ74',
    'ABZ334',
    'ABZ59',
    'ABZ383',
    'ABZ145',
    'ABZ399',
    'ABZ7',
    'ABZ589',
    'ABZ586',
    'ABZ97',
    'ABZ211',
    'ABZ343',
    'ABZ367',
    'ABZ52',
    'ABZ212',
    'ABZ85',
    'ABZ115',
    'ABZ319',
    'ABZ207',
    'ABZ78',
    'ABZ144',
    'ABZ465',
    'ABZ38',
    'ABZ570',
    'ABZ322',
    'ABZ331',
    'ABZ60',
    'ABZ427',
    'ABZ112',
    'ABZ80',
    'ABZ314',
    'ABZ79',
    'ABZ142a',
    'ABZ232',
    'ABZ312',
    'ABZ535',
    'ABZ554',
    'ABZ595',
    'ABZ128',
    'ABZ339',
    'ABZ12',
    'ABZ172',
    'ABZ331e+152i',
    'ABZ147',
    'ABZ575',
    'ABZ167',
    'ABZ230',
    'ABZ279',
    'ABZ401',
    'ABZ306',
    'ABZ468',
    'ABZ6',
    'ABZ472',
    'ABZ148',
    'ABZ2',
    'ABZ104',
    'ABZ313',
    'ABZ397',
    'ABZ134',
    'ABZ412',
    'ABZ441',
    'ABZ62',
    'ABZ455',
    'ABZ440',
    'ABZ471',
    'ABZ111',
    'ABZ538',
    'ABZ72',
    'ABZ101',
    'ABZ393',
    'ABZ50',
    'ABZ298',
    'ABZ437',
    'ABZ94',
    'ABZ143',
    'ABZ483',
    'ABZ205',
    'ABZ565',
    'ABZ191',
    'ABZ124',
    'ABZ152',
    'ABZ87',
    'ABZ138',
    'ABZ559',
    'ABZ164',
    'ABZ126',
    'ABZ598a',
    'ABZ195',
    'ABZ307',
    'ABZ9',
    'ABZ556',
]
        #not_found_class = ["SignClassNotInImageClassificationTrainData"]
        # this sorting is because thats the way the model was trained in mmcls
        self.CLASSES_CLASSIFICATION = sorted(self.CLASSES)
        self.mapping = {self.CLASSES_CLASSIFICATION.index(a): self.CLASSES.index(a) for a in self.CLASSES_CLASSIFICATION}

    def show(self, image, points):
        fig, ax = plt.subplots(frameon=False)
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        fig.add_axes(ax, aspect="auto")
        ax.imshow(image, cmap="Greys", aspect="auto")
        for point in points:
            ax.plot([x[0] for x in point], [x[1] for x in point], color="red", linewidth=1)
        plt.show()

    def test_step(self, data):
        #plt.imshow(data["inputs"][0].permute(1, 2, 0))
        #plt.show()
        result = self.detection_model.test_step(data)
        #result[0].pred_instances = result[0].gt_instances.clone()
        labels = []
        scores = []
        assert len(result) == 1
        _centroids = []
        scale_width, scale_height = result[0].scale_factor
        for counter, bbox in enumerate(result[0].pred_instances.bboxes):
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            x, y, w, h = x * scale_width, y * scale_height, w * scale_width, h * scale_height
            x, y, w, h = list(map(int, [y, x, h, w]))
            _centroids.append([ y + h / 2, x + w / 2])

            crop = torchvision.transforms.functional.crop(data["inputs"][0], x,y,w,h)

            #abz = self.CLASSES[result[0].gt_instances.labels[counter].item()]
            #path = f"debug/{abz}_{counter}_{random()}.png"
            #save_image(crop.flip(0) / 255, path)
            #result__ = inference_model(self.cls, path)

            #resized1 = F.interpolate(crop, size=(380,380))
            img = torchvision.transforms.Resize((380, 380))(crop)
            img = img.flip(0)
            img = img.float()
            mean = torch.Tensor([[[86.6589]], [[67.9274]], [[53.7833]]])
            std = torch.Tensor([[[68.9897]], [[57.2049]], [[48.2306]]])
            img = (img - mean) / std

            #show = data["inputs"][0].permute(1, 2, 0)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img = img.unsqueeze(0).to(device)

            _result = self.classification_model.predict(img, None)[0]
            pred = _result.pred_label.item()
            pred = self.mapping[pred]

            labels.append(pred)
            scores.append(1)
        result[0].pred_instances.labels = torch.Tensor(labels).long()
        result[0].pred_instances.scores = torch.Tensor(scores)
        """
        ordering, points_ordered = line_detection(torch.Tensor(_centroids), [self.CLASSES[label] for label in labels])
        self.show(data["inputs"][0].permute(1, 2, 0), points_ordered)
        
        # write ordering to file
        name = result[0].img_path.split("/")[-1].split(".")[:-1]
        with open(f"signs/{'.'.join(name)}_ordering.txt", "w") as f:
            for ordering_elem in ordering:
                print(" ".join(ordering_elem))
                f.write(" ".join(ordering_elem) + "\n")
        """
        return result


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    cfg.launcher = 'none'
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    detection_model = runner.model

    classification_model = classification_config(
       "configs/efficient_net.py",
       "checkpoints/epoch_150.pth"
    )

    runner.model = Recognition(detection_model, classification_model)
    runner.model.eval()
    runner.test()


if __name__ == '__main__':
    main()



# Copyright (c) OpenMMLab. All rights reserved.
import shutil
from random import random

import matplotlib
from mmcls import init_model, inference_model

matplotlib.use("tkAgg")
import argparse
import os
import os.path as osp

import torch
import torchvision.transforms.functional
from matplotlib import pyplot as plt
from mmengine.config import Config
from mmengine.runner import Runner
from torch import nn
from torchvision.utils import save_image
import torch.nn.functional as F

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

class Recognition(nn.Module):
    def __init__(self, detection_model, classification_model):
        super().__init__()
        shutil.rmtree("debug")
        os.mkdir("debug")
        #self.cls = init_model("configs/efficient_net.py", "checkpoints/efficient_net/epoch_100.pth")
        self.detection_model = detection_model
        self.classification_model = classification_model
        self.CLASSES = [
            'ABZ579', 'ABZ13', 'ABZ342', 'ABZ70', 'ABZ461', 'ABZ142', 'ABZ480', 'ABZ1',
            'ABZ231', 'ABZ533', 'ABZ449', 'ABZ318', 'ABZ75', 'ABZ61', 'ABZ354',
            'ABZ139', 'ABZ381', 'ABZ597', 'ABZ536', 'ABZ308', 'ABZ330', 'ABZ328',
            'ABZ86', 'ABZ15', 'ABZ214', 'ABZ545', 'ABZ73', 'ABZ295', 'ABZ55', 'ABZ335',
            'ABZ371', 'ABZ151', 'ABZ457', 'ABZ537', 'ABZ69', 'ABZ353', 'ABZ68', 'ABZ5',
            'ABZ296', 'ABZ84', 'ABZ366', 'ABZ411', 'ABZ396', 'ABZ206', 'ABZ58',
            'ABZ324', 'ABZ376', 'ABZ99', 'ABZ384', 'ABZ59', 'ABZ532', 'ABZ334',
            'ABZ589', 'ABZ383', 'ABZ343', 'ABZ586', 'ABZ399', 'ABZ74', 'ABZ211',
            'ABZ145', 'ABZ7', 'ABZ212', 'ABZ78', 'ABZ367', 'ABZ38', 'ABZ319', 'ABZ85',
            'ABZ115', 'ABZ322', 'ABZ97', 'ABZ144', 'ABZ112', 'ABZ427', 'ABZ60',
            'ABZ207', 'ABZ79', 'ABZ80', 'ABZ232', 'ABZ142a', 'ABZ312', 'ABZ52',
            'ABZ331', 'ABZ128', 'ABZ314', 'ABZ535', 'ABZ575', 'ABZ134', 'ABZ465',
            'ABZ167', 'ABZ172', 'ABZ339', 'ABZ6', 'ABZ331e+152i', 'ABZ306', 'ABZ12',
            'ABZ2', 'ABZ148', 'ABZ397', 'ABZ554', 'ABZ570', 'ABZ441', 'ABZ147',
            'ABZ472', 'ABZ104', 'ABZ440', 'ABZ230', 'ABZ595', 'ABZ455', 'ABZ313',
            'ABZ298', 'ABZ412', 'ABZ62', 'ABZ468', 'ABZ101', 'ABZ111', 'ABZ483',
            'ABZ538', 'ABZ471', 'ABZ87', 'ABZ143', 'ABZ565', 'ABZ205', 'ABZ152',
            'ABZ72', 'ABZ138', 'ABZ401', 'ABZ50', 'ABZ406', 'ABZ307', 'ABZ126',
            'ABZ124', 'ABZ164', 'ABZ529', 'ABZ559', 'ABZ94', 'ABZ437', 'ABZ56',
            'ABZ393', 'ABZ398', "No"
        ]
        self.CLASSES_CLASSIFICATION = ['ABZ1', 'ABZ101', 'ABZ104', 'ABZ111', 'ABZ112', 'ABZ115', 'ABZ12', 'ABZ124', 'ABZ126', 'ABZ128', 'ABZ13', 'ABZ134', 'ABZ138', 'ABZ139', 'ABZ142', 'ABZ142a', 'ABZ143', 'ABZ144', 'ABZ145', 'ABZ147', 'ABZ148', 'ABZ15', 'ABZ151', 'ABZ152', 'ABZ164', 'ABZ167', 'ABZ172', 'ABZ2', 'ABZ205', 'ABZ206', 'ABZ207', 'ABZ211', 'ABZ212', 'ABZ214', 'ABZ230', 'ABZ231', 'ABZ232', 'ABZ295', 'ABZ296', 'ABZ298', 'ABZ306', 'ABZ307', 'ABZ308', 'ABZ312', 'ABZ313', 'ABZ314', 'ABZ318', 'ABZ319', 'ABZ322', 'ABZ324', 'ABZ328', 'ABZ330', 'ABZ331', 'ABZ331e+152i', 'ABZ334', 'ABZ335', 'ABZ339', 'ABZ342', 'ABZ343', 'ABZ353', 'ABZ354', 'ABZ366', 'ABZ367', 'ABZ371', 'ABZ376', 'ABZ38', 'ABZ381', 'ABZ383', 'ABZ384', 'ABZ393', 'ABZ396', 'ABZ397', 'ABZ398', 'ABZ399', 'ABZ401', 'ABZ406', 'ABZ411', 'ABZ412', 'ABZ427', 'ABZ437', 'ABZ440', 'ABZ441', 'ABZ449', 'ABZ455', 'ABZ457', 'ABZ461', 'ABZ465', 'ABZ468', 'ABZ471', 'ABZ472', 'ABZ480', 'ABZ483', 'ABZ5', 'ABZ50', 'ABZ52', 'ABZ529', 'ABZ532', 'ABZ533', 'ABZ535', 'ABZ536', 'ABZ537', 'ABZ538', 'ABZ545', 'ABZ55', 'ABZ554', 'ABZ559', 'ABZ56', 'ABZ565', 'ABZ570', 'ABZ575', 'ABZ579', 'ABZ58', 'ABZ586', 'ABZ589', 'ABZ59', 'ABZ595', 'ABZ597', 'ABZ6', 'ABZ60', 'ABZ61', 'ABZ62', 'ABZ68', 'ABZ69', 'ABZ7', 'ABZ70', 'ABZ72', 'ABZ73', 'ABZ74', 'ABZ75', 'ABZ78', 'ABZ79', 'ABZ80', 'ABZ84', 'ABZ85', 'ABZ86', 'ABZ87', 'ABZ94', 'ABZ97', 'ABZ99', "No"]
        self.mapping = {self.CLASSES_CLASSIFICATION.index(a): self.CLASSES.index(a) for a in self.CLASSES_CLASSIFICATION }


    def test_step(self, data):
        #plt.imshow(data["inputs"][0].permute(1, 2, 0))
        #plt.show()
        result = self.detection_model.test_step(data)
        #result[0].pred_instances = result[0].gt_instances.clone()
        labels = []
        scores = []
        assert len(result) == 1
        scale_width, scale_height = result[0].scale_factor
        for counter, bbox in enumerate(result[0].pred_instances.bboxes):
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            x, y, w, h = x * scale_width, y * scale_height, w * scale_width, h * scale_height
            x, y, w, h = list(map(int, [y, x, h, w]))

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

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img = img.unsqueeze(0).to(device)

            _result = self.classification_model.predict(img, None)[0]
            pred = _result.pred_label.item()
            pred = self.mapping[pred]
            #show(img)
            labels.append(pred)
            scores.append(1)
        result[0].pred_instances.labels = torch.Tensor(labels).long()
        result[0].pred_instances.scores = torch.Tensor(scores)
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
       "checkpoints/efficient_net/epoch_100.pth"
    )

    runner.model = Recognition(detection_model, classification_model)
    runner.model.eval()
    runner.test()


if __name__ == '__main__':
    main()



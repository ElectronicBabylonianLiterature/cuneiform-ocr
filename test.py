# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib
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


def classification_config(config, checkpoint):
    cfg = Config.fromfile(config)
    cfg.load_from = checkpoint
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()
    return runner.model

class Recognition(nn.Module):
    def __init__(self, detection_model, classification_model):
        super().__init__()
        self.detection_model = detection_model
        self.classification_model = classification_model


    def test_step(self, data):
        #plt.imshow(data["inputs"][0].permute(1, 2, 0))
        #plt.show()
        result = self.detection_model.test_step(data)
        result[0].pred_instances = result[0].gt_instances.clone()
        #result[0].pred_instances = result[0].gt_instances
        #result[0].pred_instances.labels = torch.Tensor([0] * len(result[0].pred_instances)).long()
        #result[0].pred_instances.scores = torch.Tensor([1.0] * len(result[0].pred_instances))
        #return result
        labels = []
        scores = []
        assert len(result) == 1
        scale_width, scale_height = result[0].scale_factor
        for bbox in result[0].pred_instances.bboxes:
            x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
            x, y, w, h = x * scale_width, y * scale_height, w * scale_width, h * scale_height
            x, y, w, h = list(map(int, [x,y,w,h]))

            crop = torchvision.transforms.functional.crop(data["inputs"][0], x,y,w,h)

            #plt.imshow(crop.permute(1, 2, 0))
            #plt.show()
            #resized1 = F.interpolate(crop, size=(380,380))

            img = torchvision.transforms.Resize((380, 380))(crop)
            #plt.imshow(img.permute(1, 2, 0))
            #plt.show()

            img = img.flip(0)
            #plt.imshow(img.permute(1, 2, 0))
            #plt.show()

            img = img.float()
            mean = torch.Tensor([[[86.6589]], [[67.9274]], [[53.7833]]])
            std = torch.Tensor([[[68.9897]], [[57.2049]], [[48.2306]]])
            img = (img - mean) / std
            #plt.imshow(img.permute(1, 2, 0))
            #plt.show()

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            img = img.unsqueeze(0).to(device)
            _result = self.classification_model.predict(img, None)[0]
            pred = _result.pred_label.item()
            labels.append(pred)
            scores.append(0.99)
            #scores.append(_result.pred_score[pred].item())
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
       "checkpoints/epoch_100.pth"
    )

    runner.model = Recognition(detection_model, classification_model)
    runner.model.eval()
    runner.test()


if __name__ == '__main__':
    main()



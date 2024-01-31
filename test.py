# Copyright (c) OpenMMLab. All rights reserved.

import matplotlib
import mmcv

from mmdet.apis import inference_detector
from recognition_model import Recognition

# from mmcls import init_model, inference_model

matplotlib.use("tkAgg")
import argparse
import os
import os.path as osp

from mmengine.config import Config
from mmengine.runner import Runner

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
    classes = cfg._cfg_dict["classes"]
    return runner.model, classes


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

    classification_model, classes = classification_config(
        "configs/efficient_net.py",
        "checkpoints/efficient-net/efficient-net-24/epoch_300.pth"
    )

    runner.model = Recognition(detection_model, classification_model, classes, cfg)
    """
    runner.model.eval()
    img = mmcv.imread('data/coco/val2017/BM.33016-0.jpg', channel_order='rgb')
    result = inference_detector(runner.model, img)
    # init the visualizer(execute this block only once)
    from mmdet.registry import VISUALIZERS
    visualizer = VISUALIZERS.build(runner.model.cfg.visualizer)
    visualizer.add_datasample(
        'result',
        img,
        data_sample=result,
        draw_gt=False,
        wait_time=0,
    )
    visualizer.show()
    """
    runner.test()


if __name__ == '__main__':
    main()



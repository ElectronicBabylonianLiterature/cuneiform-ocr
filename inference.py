# Copyright (c) OpenMMLab. All rights reserved.
from urllib import parse
import matplotlib
import mmcv
from bleualign.align import Aligner
from mmdet.apis import inference_detector
from recognition_model import Recognition

# from mmcls import init_model, inference_model

matplotlib.use("tkAgg")
import argparse
import os
import os.path as osp
from pymongo import MongoClient

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


def create_model():
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
        "checkpoints/efficient_net/epoch_500.pth"
    )
    print(classes)
    runner.model = Recognition(detection_model, classification_model, classes, cfg)
    runner.model.eval()
    return runner.model

import requests
BASE_URL = "https://www.ebl.lmu.de/api/fragments"
def download_image(url, filename):
    respone = requests.get(url, stream=True)
    respone.raw.decode_content = True
    with open(filename, "wb") as outfile:
        outfile.write(respone.content)


if __name__ == '__main__':
    import fragments
    model = create_model()
    client  = MongoClient(fragments.CONNECTION)
    database = client["ebl"]
    fragmentarium = database["fragments"]
  
    for fragment in fragments.FRAGMENT:
        
        _id = parse.quote(fragment)
        print(fragment)
        url = f"{BASE_URL}/{_id}/photo"
        fragment_cursor = fragmentarium.find({"_id": _id}, {"signs":1})
        for doc in fragment_cursor:
            src = doc['signs']
        src = src.split('\n')
        download_image(url, f"temp_images/{fragment}.jpg")
        img = mmcv.imread(f"temp_images/{fragment}.jpg", channel_order='rgb')
        pred = inference_detector(model, img)

        print(pred)
        print(src)
        os.remove(f"temp_images/{fragment}.jpg")



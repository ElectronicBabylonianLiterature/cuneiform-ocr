#!/bin/bash

set -ex

cd $HOME
ls -la

# Set up environment

pip install -r $HOME/cuneiform-ocr/requirements-data-retrieval.txt

pip install "numpy<2.0.0"
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -U openmim
mim install "mmocr==1.0.1"
mim install "mmcv==2.0.0"
pip install albumentations==1.3.1
cd ~/cuneiform-ocr/mmdetection
pip install -e .
pip install future tensorboard
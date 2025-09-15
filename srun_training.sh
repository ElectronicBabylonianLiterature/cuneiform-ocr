#!/bin/bash
# setup-env.sh: Set up environment for cuneiform-ocr

# Exit immediately if a command exits with a non-zero status.
set -e
# Print commands and their arguments as they are executed.
set -x

cd $HOME
ls -la
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/cuneiform-ocr/mmdetection:$PYTHONPATH

# Set up environment
pip install "numpy<2.0.0"
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2
pip install -U openmim
mim install "mmocr==1.0.1"
mim install "mmcv==2.0.0"
pip install albumentations==1.3.1
cd ~/cuneiform-ocr/mmdetection
pip install -e .
pip install "numpy<2.0.0"

# Create symlinks
cd ~/cuneiform-ocr/
rm checkpoint_config
ln -s ~/detr-2024 checkpoint_config
ls -alh ~/cuneiform-ocr/

cd ~/cuneiform-ocr/mmdetection
rm data
ln -s ~/ready-for-training/coco-recognition/data data
ls -alh ~/cuneiform-ocr/mmdetection

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import mmcv; print(mmcv.__version__)"
python -c "import mmocr; print(mmocr.__version__)"
python -c "import mmdet; print(mmdet.__version__)"

# Run training
cd ~/cuneiform-ocr/
cd mmdetection
python tools/train.py ../checkpoint_config/detr.py
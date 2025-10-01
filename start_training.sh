#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e
# Print commands and their arguments as they are executed.
set -x

cd $HOME
ls -la
export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/cuneiform-ocr/mmdetection:$PYTHONPATH
pip install "numpy<2.0.0"

# Create symlinks
cd ~/cuneiform-ocr/
ls -alh ~/cuneiform-ocr/

cd ~/cuneiform-ocr/mmdetection
rm data
COCO_DATA="${COCO_DATA:-$HOME/ready-for-training/coco-recognition/data}"
ln -s $COCO_DATA data
echo "using COCO data from $COCO_DATA"
sleep 2
ls -alh ~/cuneiform-ocr/mmdetection

# Verify installation
python -c "import torch; print(torch.__version__)"
python -c "import mmcv; print(mmcv.__version__)"
python -c "import mmocr; print(mmocr.__version__)"
python -c "import mmdet; print(mmdet.__version__)"


# Determine number of GPUs
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())")
echo "Using $GPU_COUNT GPUs"
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,roundup_power2_divisions:4,garbage_collection_threshold:0.6

# export CUDA_LAUNCH_BLOCKING=1

# Run training
cd ~/cuneiform-ocr/
cd mmdetection
if [ $GPU_COUNT -gt 1 ]; then
    # multi-GPU training
    python -m torch.distributed.launch --nproc_per_node=$GPU_COUNT --master_port=29500 \
        tools/train.py ../configs/detr.py
else
    # single-GPU training
    python tools/train.py ../configs/detr.py
fi
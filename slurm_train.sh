#!/bin/bash
#SBATCH -p lrz-dgx-a100-80x8 
#SBATCH --gres=gpu:1
#SBATCH -o log_%j.log                   # File to store standard output
#SBATCH -e log_%j.log                   # File to store standard error
#SBATCH --time=24:00:00                 # Set a time limit


echo "Start on $(hostname) at $(date)"  # Run outside of srun

export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/cuneiform-ocr/mmdetection:$PYTHONPATH

cd $HOME/cuneiform-ocr/

srun set_up_env.sh        # Run inside of srun

# === setup training data ===

export COCO_DATA= # path to COCO data, e.g. $HOME/ready-for-training/coco-recognition/data

# =======================


echo "Start on $(hostname) at $(date)"  # Run outside of srun

export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/cuneiform-ocr/mmdetection:$PYTHONPATH

cd $HOME/cuneiform-ocr/

srun set_up_env.sh        # Run inside of srun

# === download data ===
# cd $HOME
# git clone https://github.com/ElectronicBabylonianLiterature/cuneiform-ocr-data.git
# cd cuneiform-ocr-data
# git checkout re-train-ocr-model
# git pull

export COCO_DATA=$HOME/coco-recognitioin-2025-09-25/data-coco
# =======================


srun start_training.sh          # run training as least once
exit_code=$?
echo "Training script finished with exit code $exit_code"

# Retry training if killed by OOM
while true; do
    srun start_training.sh
    exit_code=$?
    echo "Training script finished with exit code $exit_code"
    if [ $exit_code -eq 0 ]; then
        echo "Training completed successfully"
        break
    elif [ $exit_code -eq 137 ]; then
        echo "Training killed by OOM (exit code 137), retrying..."
        sleep 30
    else
        echo "Training failed with exit code $exit_code, stopping"
        break
    fi
done


echo "End on $(hostname) at $(date)"    # Run outside of srun
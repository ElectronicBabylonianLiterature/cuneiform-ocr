#!/bin/bash
#SBATCH -p lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH -o log_anno_%j.log                   # File to store standard output
#SBATCH -e log_anno_%j.log                   # File to store standard error
#SBATCH --time=24:00:00                 # Set a time limit

echo "Start on $(hostname) at $(date)"  # Run outside of srun

export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/cuneiform-ocr/mmdetection:$PYTHONPATH

cd $HOME/cuneiform-ocr/

srun set_up_env.sh


srun pip install "numpy<2.0.0"

srun python do_annotations.py

echo "End on $(hostname) at $(date)"

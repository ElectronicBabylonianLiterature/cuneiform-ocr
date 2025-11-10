#!/bin/bash
#SBATCH -p lrz-dgx-1-p100x8 
#SBATCH --gres=gpu:1
#SBATCH -o log_anno_exp-3%j.log                   # File to store standard output
#SBATCH -e log_anno_exp-3%j.log                   # File to store standard error
#SBATCH --time=24:00:00                 # Set a time limit

echo "Start on $(hostname) at $(date)"  # Run outside of srun

export PATH=$HOME/.local/bin:$PATH
export PYTHONPATH=$HOME/cuneiform-ocr/mmdetection:$PYTHONPATH

cd $HOME/erc-src/cuneiform-ocr/

srun set_up_env.sh


srun pip install "numpy<2.0.0"

srun TAG=exp-3 THRESHOLD=0.7 Y_THRESHOLD=35 python do_annotations.py


echo "End on $(hostname) at $(date)"

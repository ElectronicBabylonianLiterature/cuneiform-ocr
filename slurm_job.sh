#!/bin/bash
#SBATCH -p lrz-v100x2                   # Select partition (use sinfo)
#SBATCH --gres=gpu:1                    # Request 1 GPU
#SBATCH -o log_%j.out                   # File to store standard output
#SBATCH -e log_%j.err                   # File to store standard error
#SBATCH --time=24:00:00                 # Set a time limit


echo "Start on $(hostname) at $(date)"  # Run outside of srun
cd $HOME/cuneiform-ocr/
srun srun_training.sh          # Run inside of srun
echo "End on $(hostname) at $(date)"    # Run outside of srun
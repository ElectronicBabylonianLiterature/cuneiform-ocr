"""Run python train_with_cuneimae.py to fine-tune MAE using the shared train.py workflow.

To resume, set RESUME = OUTPUT_DIR / f"last_{RUN_TAG}.pt".
"""

from functools import partial
from pathlib import Path

import torch

import train as base_train
from model import build_model


DEBUG_SUBSET = False
OUTPUT_DIR = Path(__file__).resolve().parent / "output_with_cuneimae"
MAE_CHECKPOINT = Path(__file__).resolve().parent / "pretrained" / "checkpoint-10.pth"
RESUME = None
RUN_TAG = "aug-cunei-1"

# TRAIN_CONFIG = dict(base_train.TRAIN_CONFIG)
TRAIN_CONFIG = {
    "epochs": 1000,
    "batch_size": 2,
    "eval_batch_size": 1,
    "grad_accum_steps": 4,
    "lr": 2e-4,
    "lr_backbone": 2e-5,
    "weight_decay": 1e-4,
    "lr_drop_epoch": 40,
    "max_grad_norm": 0.1,
    "eval_interval": 5,
    "num_workers": 16,
    "amp_dtype": torch.float16,  # One shared configuration; V100 supports FP16 but not BF16.
    "seed": 42,
    "save_interval": 5,
    "max_detections": 300,
}

def main():
    base_train.main(
        model_builder=partial(build_model, backbone="cuneimae", mae_checkpoint=MAE_CHECKPOINT),
        train_config=TRAIN_CONFIG,
        debug_subset=DEBUG_SUBSET,
        output_dir=OUTPUT_DIR,
        resume=RESUME,
        run_tag=RUN_TAG,
        tag_checkpoints=True,
    )


if __name__ == "__main__":
    main()

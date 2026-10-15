"""Run python test_with_cuneimae.py using the shared test.py evaluation and visualization workflow."""

from functools import partial
from pathlib import Path

import test as base_test
from model import build_model
from train import checkpoint_path
from train_with_cuneimae import RUN_TAG as TRAIN_RUN_TAG


DEBUG_SUBSET = base_test.DEBUG_SUBSET
OUTPUT_DIR = Path(__file__).resolve().parent / "output_with_cuneimae"
RUN_TAG = TRAIN_RUN_TAG  # Change this value to test a model from another run tag.
CHECKPOINT = checkpoint_path(OUTPUT_DIR, "best.pt", RUN_TAG)
TEST_CONFIG = dict(base_test.TEST_CONFIG)


def main():
    base_test.main(
        model_builder=partial(build_model, backbone="cuneimae"),
        test_config=TEST_CONFIG,
        debug_subset=DEBUG_SUBSET,
        output_dir=OUTPUT_DIR,
        checkpoint_path=CHECKPOINT,
    )


if __name__ == "__main__":
    main()

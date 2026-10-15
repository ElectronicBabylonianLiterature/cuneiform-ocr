#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate cuneiform-ocr-new

tensorboard --logdir "$SCRIPT_DIR/output/tensorboard" --port 6006 &
BASE_PID=$!
trap 'kill "$BASE_PID" 2>/dev/null || true' EXIT

tensorboard --logdir "$SCRIPT_DIR/output_with_cuneimae/tensorboard" --port 6007

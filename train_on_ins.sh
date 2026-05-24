#!/bin/bash

max_attempts=5
attempt=1

export COCO_DATA="~/erc-work-data/coco-recognition-2026-4-no-extraction/data"

while [ $attempt -le $max_attempts ]; do
    echo "Starting training attempt $attempt..."
    ./start_training.sh
    if [ $? -eq 0 ]; then
        echo "Training succeeded on attempt $attempt."
        exit 0
    else
        echo "Training failed on attempt $attempt."
        ((attempt++))
        sleep 5  
    fi
done

echo "Training failed after $max_attempts attempts."
exit 1
#!/bin/bash

max_attempts=5
attempt=1

while [ $attempt -le $max_attempts ]; do
    ./start_training.sh
    if [ $? -eq 0 ]; then
        echo "Training succeeded on attempt $attempt."
        exit 0
    else
        echo "Training failed on attempt $attempt."
        ((attempt++))
    fi
done

echo "Training failed after $max_attempts attempts."
exit 1
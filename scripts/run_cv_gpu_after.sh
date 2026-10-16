#!/usr/bin/env bash
# Start the GPU lane of scripts/run_cv.py once process $1 (an earlier training job) has exited.
cd "$(dirname "$0")/.."
while kill -0 "$1" 2>/dev/null; do sleep 60; done
exec ~/anaconda3/envs/kaggle/bin/python -W ignore -m scripts.run_cv --lane gpu

#!/bin/bash
conda activate diffsheg

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
OMP_NUM_THREADS=10 CUDA_VISIBLE_DEVICES=0 python -u test_realtime_wrapper.py

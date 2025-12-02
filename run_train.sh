#!/bin/bash
# Training startup script with correct CUDA library paths

export LD_LIBRARY_PATH=/home/paul/Waste-Detection-System/.venv/lib/python3.10/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH

cd /home/paul/Waste-Detection-System
/home/paul/Waste-Detection-System/.venv/bin/python train.py "$@"

#!/bin/bash 

source ~/miniconda3/etc/profile.d/conda.sh
conda activate NeuroEvolution
python $WORK/NeuroEvolution/NeuroEvolution/train/torch_train.py

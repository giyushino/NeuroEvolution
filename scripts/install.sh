#!/bin/bash 

source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n NeuroEvolution python=3.13
conda activate NeuroEvolution

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install --upgrade "jax[cuda12]"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"
pip install -e .
rm -rf "$PROJECT_ROOT/NeuroEvolution.egg-info"

#!/bin/bash
source /home/nikoskot/.bashrc

conda activate base
conda activate dev

module load CUDA/11.8.0
module load cuDNN/8.6.0.163-CUDA-11.8.0
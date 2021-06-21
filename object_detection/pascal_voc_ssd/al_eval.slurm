#!/bin/bash
#SBATCH --job-name=SSD_Eval
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=100:00:00
#SBATCH --output=result/evaluation-%j
#SBATCH --exclude=node2

source activate alenv
python3 eval.py


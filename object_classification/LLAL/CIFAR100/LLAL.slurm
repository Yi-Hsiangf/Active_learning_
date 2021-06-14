#!/bin/bash
#SBATCH --job-name=LLAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=72:00:00

source activate alenv
python3 main.py

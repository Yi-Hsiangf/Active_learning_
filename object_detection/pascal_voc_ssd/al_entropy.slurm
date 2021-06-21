#!/bin/bash
#SBATCH --job-name=SSD_En
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=300:00:00
#SBATCH --output=result/entropy-%j
#SBATCH --exclude=node2

source activate alenv
python3 train.py --Acq_func=Entropy --lr=1e-4


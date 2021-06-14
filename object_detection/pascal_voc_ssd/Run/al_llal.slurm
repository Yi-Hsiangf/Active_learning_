#!/bin/bash
#SBATCH --job-name=SSD_LLAL
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=50:00:00
#SBATCH --output=result/llal-%j
#SBATCH --nodelist=node9


source activate alenv
python3 train.py --Acq_func=LLAL --lr=1e-4


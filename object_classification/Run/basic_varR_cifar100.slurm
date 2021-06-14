#!/bin/bash
#SBATCH --job-name=B_VarR
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=Basic_result/VarR_cifar100-%j
#SBATCH --exclude=node2

source activate alenv
python3 ../main.py --func VarR --dataset cifar100 --method Basic

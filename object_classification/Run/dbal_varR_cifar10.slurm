#!/bin/bash
#SBATCH --job-name=DB_VarR
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --output=DBAL_result/VarR_cifar10-%j
#SBATCH --nodelist=node8

source activate alenv
python3 ../main.py --func VarR --dataset cifar10 --method DBAL

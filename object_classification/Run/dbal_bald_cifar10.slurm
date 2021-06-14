#!/bin/bash
#SBATCH --job-name=DB_BALD
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=20:00:00
#SBATCH --output=DBAL_result/BALD_cifar10-%j
#SBATCH --exclude=node2

source activate alenv
python3 ../main.py --func BALD --dataset cifar10 --method DBAL

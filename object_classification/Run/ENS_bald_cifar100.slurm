#!/bin/bash
#SBATCH --job-name=ENS_BALD
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --output=ENS_result/BALD_cifar100-%j
#SBATCH --nodelist=node9

source activate alenv
python3 ../main.py --func BALD --dataset cifar100 --method ENS

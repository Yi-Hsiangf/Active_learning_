#!/bin/bash
#SBATCH --job-name=S_Random
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --output=Simple_result/Random_cifar100-%j
#SBATCH --nodelist=node9

source activate alenv
python3 ../main.py --func Random --dataset cifar100 --method Simple

#!/bin/bash
#SBATCH --job-name=B_Entropy
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:12G
#SBATCH --cpus-per-task=3
#SBATCH --mem=24G
#SBATCH --time=72:00:00
#SBATCH --output=Basic_result/Entropy_cifar10-%j
#SBATCH --exclude=node2

source activate alenv
python3 ../main.py --func Entropy --dataset cifar10 --method Basic

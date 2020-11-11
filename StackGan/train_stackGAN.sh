#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
module load cuDNN/7.6.4.38-gcccuda-2019a
module load TensorFlow/2.1.0-fosscuda-2019b-Python-3.7.4

python main.py --epochs 1200 --data_dir /home/s2965690/datasets/birds



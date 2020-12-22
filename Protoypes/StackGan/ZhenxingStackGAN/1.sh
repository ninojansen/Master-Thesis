#!/bin/bash
#SBATCH --time=02:59:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --job-name=Stack

module purge
module load Python/3.6.4-foss-2018a
module load PyTorch/1.2.0-fosscuda-2019a-Python-3.7.2
module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4 
pip install --user torchvision-0.4.0-cp37-cp37m-manylinux1_x86_64.whl
pip install --user opencv_python-4.1.1.26-cp37-cp37m-manylinux1_x86_64.whl
pip install --user tensorboard-1.10.0-py3-none-any.whl
pip install --user EasyDict
pip install --user nltk
pip install --user tensorboardX

echo Starting Python program
python main.py


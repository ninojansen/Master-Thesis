#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --job-name=CNN_VQA

module purge
module load PyTorch/1.6.0-fosscuda-2019b-Python-3.7.4
module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4 

for lr in 0.0002 0.002 0.2; do
    for ef in sbert_reduced phoc_reduced bow; do
        for n_hidden in 64 128 256 512; do
            python vqa.py --cfg cfg/easyVQA/default.yml  --progress_bar_refresh_rate 0 --data_dir /data/s2965690/datasets/ExtEasyVQA/ --outdir /data/s2965690/vqa_experiment_1 \
                --cnn_type resnet18 --ef_type $ef --n_hidden $n_hidden --lr $lr --attention #--fast_dev_run
        done
    done
done
# --data_dir /data/s2965690/datasets/ExtEasyVQA/
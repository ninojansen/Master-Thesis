#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --job-name=BOTTOMATTENTION_VQA

module purge
module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4 
module load CUDA/10.1.243-GCC-8.3.0


for ef in sbert_full sbert_reduced phoc_full phoc_reduced bow; do
    for lr in 0.002; do
        for n_hidden in 256; do
            python vqa.py --cfg cfg/easyVQA/default.yml --progress_bar_refresh_rate 0 --data_dir /data/s2965690/datasets/ExtEasyVQA/ --outdir /data/s2965690/$name \
                --cnn_type frcnn --ef_type $ef --n_hidden $n_hidden --lr $lr --attention --config_name bottom_attention
        done
    done
done

#--data_dir /data/s2965690/datasets/ExtEasyVQA/
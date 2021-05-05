#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --job-name=PRETRAINED_VQA

module purge
module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4 
module load CUDA/10.1.243-GCC-8.3.0

for ef in sbert_full sbert_reduced phoc_full phoc_reduced bow; do
    python vqa.py --cfg cfg/easyVQA/default.yml --data_dir /data/s2965690/datasets/ExtEasyVQA/ --outdir /data/s2965690/$name --progress_bar_refresh_rate 0 \
        --cnn_type vgg16_flat --ef_type $ef --config_name language_only --type language
done

# ADD THESE LINES FOR PERE
# --data_dir /data/s2965690/datasets/ExtEasyVQA/ --outdir /data/s2965690/$name --progress_bar_refresh_rate 0
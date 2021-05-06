#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --job-name=EVAL_VQA

module purge
module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4 
module load CUDA/10.1.243-GCC-8.3.0

data_dir=/data/s2965690/datasets/ExtEasyVQA/
exp_dir=/data/s2965690/vqa_experiment_final/
out_dir=/data/s2965690/results

# data_dir=/home/nino/Documents/Datasets/ExtEasyVQA
# exp_dir=/home/nino/Documents/Models/VQA/vqa_experiment_final/
# out_dir=/home/nino/Dropbox/Documents/Master/Thesis/Results

names=(
    # cnn
    # top_attention
    # bottem_attention
    # pretrained
    vision_only
    language_only
)

for name in "${names[@]}"; do
    python eval.py --ckpt $exp_dir$name --config_name $name --data_dir $data_dir --outdir $out_dir 
done
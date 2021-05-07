#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --partition=gpushort
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --job-name=EVAL_VQA

module purge
module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4 
module load CUDA/10.1.243-GCC-8.3.0

name=ig_results1

data_dir=/data/s2965690/datasets/ExtEasyVQA/
exp_dir=/data/s2965690/ig_experiment_final/
out_dir=/data/s2965690/results/ig
vqa_ckpt=/data/s2965690/vqa_experiment_final/top_attention

# data_dir=/home/nino/Documents/Datasets/ExtEasyVQA
# exp_dir=/home/nino/Documents/Models/IG/ig_experiment_final/
# out_dir=/home/nino/Dropbox/Documents/Master/Thesis/Results/IG
# vqa_ckpt=/home/nino/Documents/Models/VQA/vqa_experiment_final/top_attention

python eval.py --ckpt $exp_dir --config_name $name --data_dir $data_dir --outdir $out_dir --vqa_ckpt $vqa_ckpt

#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=16G
#SBATCH --job-name=FINETUNE_VQA

module purge
module load scikit-image/0.16.2-fosscuda-2019b-Python-3.7.4 
module load CUDA/10.1.243-GCC-8.3.0


if [ $gate = true ]
then
python cycle.py --cfg cfg/finetune_vqa.yml --vqa_ckpt $vqa_ckpt --ig_ckpt $ig_ckpt --loss $loss --type vqa --gating --data_dir /data/s2965690/datasets/ExtEasyVQA/ --outdir /data/s2965690/$name --progress_bar_refresh_rate 0 
fi


python cycle.py --cfg cfg/finetune_vqa.yml --vqa_ckpt $vqa_ckpt --ig_ckpt $ig_ckpt --loss $loss --type vqa --data_dir /data/s2965690/datasets/ExtEasyVQA/ --outdir /data/s2965690/$name --progress_bar_refresh_rate 0  

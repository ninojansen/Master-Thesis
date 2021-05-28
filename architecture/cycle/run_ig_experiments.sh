#!/bin/bash

name=cycle_ig_final
export name

vqa_ckpt=/data/s2965690/vqa_experiment_final/top_attention/ef=phoc_reduced_nhidden=256_lr=0.002/checkpoints/epoch=10-step=41249.ckpt
ig_ckpt=/data/s2965690/ig_experiment_final/phoc_reduced/non_pretrained_05-05_13:00:37/checkpoints/epoch=399-step=149999.ckpt
finetune_ckpt=/data/s2965690/vqa_cycle_final2/finetune_vqa/cycle_full_True_25-05_07:07:02/checkpoints/epoch=99-step=374999.ckpt
# vqa_ckpt=/home/nino/Documents/Models/VQA/vqa_experiment_final/top_attention/ef=sbert_reduced_nhidden=256_lr=0.002/checkpoints/epoch=8-step=33749.ckpt
# ig_ckpt=/home/nino/Dropbox/Documents/Master/Thesis/architecture/image_generation/output/sbert_reduced/non_pretrained_04-05_11:52:43/checkpoints/epoch=4-step=1502.ckpt

export vqa_ckpt
export ig_ckpt
export finetune_ckpt

loss_arr=(
    vqa_only
    full
)

for loss in "${loss_arr[@]}"; do
    export loss
    sbatch finetune_ig_train.sh
done
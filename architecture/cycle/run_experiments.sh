#!/bin/bash

vqa_ckpt=/home/nino/Documents/Models/VQA/vqa_experiment_final/top_attention/ef=sbert_reduced_nhidden=256_lr=0.002/checkpoints/epoch=8-step=33749.ckpt
ig_ckpt=/home/nino/Dropbox/Documents/Master/Thesis/architecture/image_generation/output/sbert_reduced/non_pretrained_04-05_11:52:43/checkpoints/epoch=4-step=1502.ckpt

export vqa_ckpt
export ig_ckpt

loss_arr=(
    vqa_only
    cns_only
    full
    full_coeff
)

for loss in "${loss_arr[@]}"; do
    export loss
    sh finetune_vqa_train.sh
done
#!/bin/bash

name=vqa_cycle_30_final1
export name
gate=false
export gate

epochs=30
export epochs

vqa_ckpt=/data/s2965690/vqa_experiment_final1/top_attention/ef=sbert_reduced_nhidden=256_lr=0.002/checkpoints/epoch=8-step=33749.ckpt
ig_ckpt=/data/s2965690/ig_experiment_image3/sbert_reduced/pretrained_21-05_20:36:25/checkpoints/epoch=399-step=149999.ckpt

# vqa_ckpt=/home/nino/Documents/Models/VQA/vqa_experiment_final/top_attention/ef=sbert_reduced_nhidden=256_lr=0.002/checkpoints/epoch=8-step=33749.ckpt
# ig_ckpt=/home/nino/Dropbox/Documents/Master/Thesis/architecture/image_generation/output/sbert_reduced/non_pretrained_04-05_11:52:43/checkpoints/epoch=4-step=1502.ckpt

export vqa_ckpt
export ig_ckpt

loss_arr=(
    vqa_only
    cns_only
)

for loss in "${loss_arr[@]}"; do
    export loss
    sbatch finetune_vqa_train.sh
done

loss_arr=(
    full
    full_coeff
)
for loss in "${loss_arr[@]}"; do
    export loss
    gate=true
    export gate
    sbatch finetune_vqa_train.sh
done
# loss_arr=(
#     vqa_only
#     full
# )

# for loss in "${loss_arr[@]}"; do
#     export loss
#     sbatch finetune_ig_train.sh
# done
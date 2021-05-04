#!/bin/bash
name=ig_experiment128
type=no_pretrain
export type
export name

ef_types=(
    sbert_reduced
    # sbert_full
    # phoc_full
    # phoc_reduced
    # bow    
    )

vqa_ckpts=(
/home/nino/Dropbox/Documents/Master/Thesis/architecture/visual_question_answering/output/easy_vqa_sbert/ef=sbert_reduced_nhidden=256_lr=0.0002/checkpoints/epoch=3-step=16009.ckpt
/home/nino/Dropbox/Documents/Master/Thesis/architecture/visual_question_answering/output/easy_vqa_sbert/ef=sbert_reduced_nhidden=256_lr=0.0002/checkpoints/epoch=3-step=16009.ckpt
/home/nino/Dropbox/Documents/Master/Thesis/architecture/visual_question_answering/output/easy_vqa_sbert/ef=sbert_reduced_nhidden=256_lr=0.0002/checkpoints/epoch=3-step=16009.ckpt
/home/nino/Dropbox/Documents/Master/Thesis/architecture/visual_question_answering/output/easy_vqa_sbert/ef=sbert_reduced_nhidden=256_lr=0.0002/checkpoints/epoch=3-step=16009.ckpt
/home/nino/Dropbox/Documents/Master/Thesis/architecture/visual_question_answering/output/easy_vqa_sbert/ef=sbert_reduced_nhidden=256_lr=0.0002/checkpoints/epoch=3-step=16009.ckpt
)    
# vqa_ckpts=(/home/nino/Dropbox/Documents/Master/Thesis/architecture/visual_question_answering/output/easy_vqa_sbert/ef=sbert_reduced_nhidden=256_lr=0.0002/checkpoints/epoch=3-step=16009.ckpt \
# /home/nino/Dropbox/Documents/Master/Thesis/architecture/visual_question_answering/output/easy_vqa_sbert/ef=sbert_reduced_nhidden=256_lr=0.0002/checkpoints/epoch=3-step=16009.ckpt \
# /home/nino/Dropbox/Documents/Master/Thesis/architecture/visual_question_answering/output/easy_vqa_sbert/ef=sbert_reduced_nhidden=256_lr=0.0002/checkpoints/epoch=3-step=16009.ckpt \
# /home/nino/Dropbox/Documents/Master/Thesis/architecture/visual_question_answering/output/easy_vqa_sbert/ef=sbert_reduced_nhidden=256_lr=0.0002/checkpoints/epoch=3-step=16009.ckpt \ 
# /home/nino/Dropbox/Documents/Master/Thesis/architecture/visual_question_answering/output/easy_vqa_sbert/ef=sbert_reduced_nhidden=256_lr=0.0002/checkpoints/epoch=3-step=16009.ckpt )

for i in ${!ef_types[*]}; do
    ef_type="${ef_types[i]}"
    vqa_ckpt="${vqa_ckpts[i]}"
    export ef_type
    export vqa_ckpt
    sbatch image_generation_train.sh
done

# phoc_full phoc_reduced bow sbert_full 
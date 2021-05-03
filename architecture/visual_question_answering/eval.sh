#!/bin/bash
#data_dir=/data/s2965690/datasets/ExtEasyVQA/
data_dir=/home/nino/Documents/Datasets/ExtEasyVQA
exp_dir=/home/nino/Documents/Models/VQA/vqa_experiment_final/
out_dir=/home/nino/Dropbox/Documents/Master/Thesis/Results

for name in cnn top_attention bottom_attention pretrained; do
    python eval.py --ckpt $exp_dir$name --config_name $name --data_dir $data_dir --outdir $out_dir
done
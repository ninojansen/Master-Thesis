#!/bin/bash

name=vqa_experiment_final
export name

sbatch cnn_gpu_train.sh
sbatch pretrained_gpu_train.sh
sbatch top_attention_gpu_train.sh
sbatch bottom_attention_gpu_train.sh

#!/bin/bash
name=ig_experiment128

export name
for ef in sbert_reduced ; do
    export ef
    sbatch image_generation_train.sh
done

# phoc_full phoc_reduced bow sbert_full 
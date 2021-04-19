#!/bin/bash
name=ig_experiment1

export name
for ef in sbert_full sbert_reduced ; do
    export ef
    sh image_generation_train.sh
done

# phoc_full phoc_reduced bow
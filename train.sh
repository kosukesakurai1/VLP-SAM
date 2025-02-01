#!/bin/bash

torchrun --nproc_per_node=1 train.py \
        --datapath '../Datasets_HSN/' \
        --weightpath 'weights/' \
        --benchmark 'coco' \
        --logpath 'coco_CS-ViT-B16_f-0' \
        --backbone 'CS-ViT-B/16' \
        --fold 0 \
        --condition 'mask' \
        --lr 1e-4 \
        --bsz 8 \
        --epochs 50 \
        --use_ignore True \
        --nworker 2 \
        --local_rank 0 \
        --text 'yes'

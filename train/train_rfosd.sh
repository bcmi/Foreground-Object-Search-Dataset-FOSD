#! /bin/bash

CUDA_VISIBLE_DEVICES=1,2 python -m torch.distributed.launch --nproc_per_node=2 --master_port=29500  train/train.py --distill roiconcat --feature gap  --test_freq 10 --cfg config_rfosd.yaml --testset1 True --testset2 True --num_classes 32
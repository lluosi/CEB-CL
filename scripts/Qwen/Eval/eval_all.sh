#!/bin/bash

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 bash ./scripts/Qwen/Eval/eval_3_iconqa.sh Finetune ./checkpoints/Qwen/IconQA

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 bash ./scripts/Qwen/Eval/eval_1_vizwiz.sh Last ./checkpoints/Qwen/IconQA
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 bash ./scripts/Qwen/Eval/eval_2_imagenet.sh Last ./checkpoints/Qwen/IconQA

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 bash ./scripts/Qwen/Eval/eval_1_vizwiz.sh Finetune ./checkpoints/Qwen/VizWiz
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 bash ./scripts/Qwen/Eval/eval_2_imagenet.sh Finetune ./checkpoints/Qwen/ImageNet

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 bash ./scripts/Qwen/Eval/eval_1_vizwiz.sh Checkpoint2 ./checkpoints/Qwen/ImageNet

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 bash ./scripts/Qwen/Eval_rl_light/eval_2_imagenet.sh Checkpoint1 ./checkpoints/Qwen/VizWiz
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 bash ./scripts/Qwen/Eval_rl_light/eval_3_iconqa.sh Checkpoint1 ./checkpoints/Qwen/VizWiz
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 bash ./scripts/Qwen/Eval_rl_light/eval_3_iconqa.sh Checkpoint2 ./checkpoints/Qwen/ImageNet


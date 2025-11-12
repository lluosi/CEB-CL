#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
DIR=`pwd`

GPUS_PER_NODE=5
NNODES=1
NODE_RANK=0
MASTER_ADDR=localhost
MASTER_PORT=6002

MODEL="./checkpoints/Qwen/Qwen2.5-VL" # Set the path if you do not want to load from huggingface directly
# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
OUTPUT_MODEL_PATH="./checkpoints/Qwen/VizWiz"
PREVIOUS_MODEL_PATH="./checkpoints/Qwen/Qwen2.5-VL"
DATA="playground/Instructions_Light/VizWiz/train_2000.json"
DS_CONFIG_PATH="scripts/zero3_offload.json"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 torchrun $DISTRIBUTED_ARGS ./ETrain/Train/Qwen/train_grpo.py \
    --model_name_or_path $MODEL \
    --data_path $DATA \
    --dataset_use "vizwiz" \
    --use_vllm_for_gen true \
    --use_system_prompt true \
    --max_prompt_length 4096 \
    --max_completion_length 512 \
    --num_generations 8 \
    --bf16 True \
    --fix_vit True \
    --output_dir $OUTPUT_MODEL_PATH \
    --previous_task_model_path $PREVIOUS_MODEL_PATH \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 10000 \
    --weight_decay 0.1 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.01 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --temperature 0.7 \
    --report_to "none" \
    --model_max_length 8192 \
    --lazy_preprocess True \
    --use_lora false \
    --gradient_checkpointing \
    --deepspeed ${DS_CONFIG_PATH} 
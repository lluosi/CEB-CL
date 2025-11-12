#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

if [ ! -n "$1" ] ;then
    STAGE='Finetune'
    ZEROSHOT_FLAG=""
else
    STAGE=$1
    if [ "$STAGE" == "Zero_shot" ]; then
        ZEROSHOT_FLAG="--zero-shot"
    else
        ZEROSHOT_FLAG=""
    fi
fi

if [ ! -n "$2" ] ;then
    MODELPATH='./checkpoints/Qwen/Qwen2.5-VL'
else
    MODELPATH=$2
fi

if [ ! -n "$3" ] ;then
    RESULT_DIR="./results/Qwen/VizWiz"
else
    RESULT_DIR=$3
fi

mkdir -p "${RESULT_DIR}/${STAGE}"


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m ETrain.Eval.Qwen.eval_vllm \
        --model-path $MODELPATH \
        --model-base ./checkpoints/Qwen/Qwen2.5-VL \
        --question-file ./playground/Instructions_Light/VizWiz/val.json \
        --image-folder ./cl_dataset \
        --answers-file $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl \
        --num-chunks $CHUNKS \
        $ZEROSHOT_FLAG \
        --chunk-idx $IDX &
done

wait

output_file=$RESULT_DIR/$STAGE/merge.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $RESULT_DIR/$STAGE/${CHUNKS}_${IDX}.jsonl >> "$output_file"
done

python -m ETrain.Eval.Qwen.general_eval \
    --annotation-file ./playground/Instructions_Light/VizWiz/val.json \
    --result-file $output_file \
    --output-dir $RESULT_DIR/$STAGE \


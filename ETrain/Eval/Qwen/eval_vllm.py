import argparse
import torch
import os
import json

from vllm import LLM, SamplingParams
from tqdm import tqdm
import shortuuid

from ETrain.Models.Qwen import load_pretrained_model, load_pretrained_vllm_model

from PIL import Image
import math

ZERO_SHOT_QUESTION_PROMPT = "{Question} Please answer the question with brief answer or phrase, or correct option given. "
GENERAL_QUESTION_PROMPT =  'Answer the question {Question} Please first thinks about the reasoning process in the mind and then provides the user with the answer. Output the thinking process in <think> </think> and final answer in <answer> </answer> tags.'

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    model_path = os.path.expanduser(args.model_path)
    model_base = os.path.expanduser(args.model_base)

    model, tokenizer, processor = load_pretrained_vllm_model(model_path)
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        top_k=50,
        max_tokens=768,
    )

    layers_hidden_states = []

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions)):
        prompts_text_and_vision = []
        idx = line["question_id"]
        question = line['text']
        qs = question.replace('<image>', '').strip()
        images = []
        image_file = line["image"]
        if '.jpg.jpg' in image_file:
            image_file = image_file.replace('.jpg.jpg', '.jpg')
        image_file = os.path.join(args.image_folder, image_file.replace('./', ''))
        image = Image.open(image_file).convert("RGB")

        images.append(image)

        #print(f"zero_shot is {args.zero_shot}")
        if args.zero_shot:
            prompt = ZERO_SHOT_QUESTION_PROMPT
        else:
            prompt = GENERAL_QUESTION_PROMPT
        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image"} for _ in images],
                    {"type": "text", "text": prompt.format(Question=qs)},
                ],
            }
        ]

        vllm_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # merge text and images
        prompts_text_and_vision.append(
            {
                "prompt": vllm_prompt,
                "multi_modal_data": {"image": images}
            }
        )

        output = model.generate(prompts_text_and_vision, sampling_params=sampling_params, use_tqdm=False)


        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": prompt.format(Question=qs),
                                   "text": output[0].outputs[0].text,
                                   "answer_id": ans_id}) + "\n")
        ans_file.flush()
    ans_file.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--zero-shot", action="store_true")

    args = parser.parse_args()

    eval_model(args)

# image_path是./cl_dataset/
from ETrain.Train.Qwen.prompts import *
from datasets import load_dataset
from functools import partial
import os

from openai import images


def make_conversation(example, image_path=None, use_system_prompt=True, is_grounding=False):
    image_path = "./cl_dataset/"
    QUESTION_PROMPT = GENERAL_QUESTION_PROMPT
    SPEC_QUESTION_PROMPT = QUESTION_PROMPT

    if is_grounding:
        SYSTEM_PROMPT = GROUNDING_SYSTEM_PROMPT
    else:
        SYSTEM_PROMPT = GENERAL_SYSTEM_PROMPT

    images = []

    # multimodal sample
    if "image" in example and example["image"]:
        if isinstance(example["image"], list):
            for item in example["image"]:
                if isinstance(item, str):
                    images.append(os.path.join(image_path, item))
                elif isinstance(item, dict):
                    images.append(os.path.join(image_path, item["path"]))
                else:
                    raise TypeError("Unsupported Format.")
        elif isinstance(example["image"], str):
            images.append(os.path.join(image_path, example["image"]))
        elif isinstance(example["image"], dict):
            images.append(os.path.join(image_path, example["image"]["path"]))
        else:
            raise TypeError("Unsupported Format.")

        if not isinstance(images, list):
            raise TypeError(f"Expected 'images' to be a list, but got {type(images)} instead.")

        # Iterate over the conversations, assuming that the format alternates between "human" and "gpt"
        for i in range(0, len(example["conversations"]), 2):
            # "human" question (index i)
            question = example["conversations"][i]["value"]
            # "gpt" answer (index i+1)
            answer = example["conversations"][i + 1]["value"]

            result = {}
            # Create a prompt based on whether we use a system prompt or not
            if use_system_prompt:
                return {
                    "prompt": [
                        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                        {
                            "role": "user",
                            "content": [
                                *[{"type": "image"} for _ in images],
                                {"type": "text", "text": question},
                            ],
                        },
                        # {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                    ],
                    "image": images,
                    "solution": answer  # Add the answer for evaluation purposes
                }
            else:
                return {
                    "prompt": [
                        {
                            "role": "user",
                            "content": [
                                *[{"type": "image"} for _ in images],
                                {"type": "text", "text": question},
                            ],
                        },
                        # {"role": "assistant","content": [{"type": "text", "text": answer}]},
                    ],
                    "image": images,
                    "solution": answer  # Add the answer for evaluation purposes
                }

    # text-only sample
    else:
        print("text-only")
        if use_system_prompt:
            return {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": example["problem"]},
                ],
            }
        else:
            return {
                "prompt": [
                    {"role": "user", "content": SPEC_QUESTION_PROMPT.format(Question=example["problem"])},
                ],
            }



def make_rl_dataset(args):
    use_system_prompt = True

    dataset = load_dataset('json', data_files=args.data_path)

    is_grounding = False
    dataset = dataset.map(partial(make_conversation, image_path="./cl_dataset/", use_system_prompt=use_system_prompt, is_grounding=is_grounding), load_from_cache_file=False)

    return dataset
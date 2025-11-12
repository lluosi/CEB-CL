# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.

import re

from dataclasses import dataclass, field
import json
import math
import logging
import os
import copy
import math

from datetime import datetime
from collections import deque
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
# from transformers import Trainer, deepspeed
from transformers import Trainer
import deepspeed
from accelerate.utils import DistributedType
from ETrain.Train.Base_trainer import *
from ETrain.Models.Qwen import create_Qwen_model
from ETrain.Dataset.Qwen.data_qwen import make_supervised_data_module
from ETrain.Dataset.Qwen.qwen_data_grpo import make_rl_dataset
from ETrain.Train.Qwen.qwen_grpo_trainer import QwenGRPOTrainer
from ETrain.Train.Qwen.rewards import *


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen-7B")
    previous_task_model_path: Optional[str] = field(default=None)
    task_embedding_dim: Optional[int] = field(default=64)
    expert_num: Optional[int] = field(default=4)


@dataclass
class DataArguments:
    dataset_use: str = field(default="")  # 数据集名称
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frame_pixels: int = field(default=32 * 28 * 28)
    video_min_frame_pixels: int = field(default=4 * 28 * 28)
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = field(
        default=False,
        metadata={"help": "Whether to use lazy preprocessing."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    fix_vit: bool = True
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    group_by_modality_length: bool = field(default=False)

    use_vllm_for_gen: bool = field(default=True)
    use_system_prompt: bool = field(default=True)
    max_prompt_length: int = field(default=2048)
    max_completion_length: int = field(default=256)
    num_generations: int = field(default=8)
    temperature: float = field(default=0.7)
    eval_strategy: str = field(default="no")

@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


local_rank = None

def replace_qwen2_vl_attention_class():
    import transformers
    import transformers.modeling_flash_attention_utils

    transformers.models.qwen2_vl.modeling_qwen2_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_vl.modeling_qwen2_vl.Qwen2VLModel._update_causal_mask = (
        _update_causal_mask
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl._flash_attention_forward = (
        _flash_attention_forward
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLModel._update_causal_mask = (
        _update_causal_mask
    )


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def train():

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    (model_args, data_args, training_args, lora_args,) = parser.parse_args_into_dataclasses()

    if getattr(training_args, 'deepspeed', None) and getattr(lora_args, 'q_lora', False):
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    local_rank = training_args.local_rank

    model, tokenizer, processor, ref_model = create_Qwen_model(training_args, model_args, data_args, lora_args)

    data_args.image_processor = processor.image_processor
    data_args.model_type = "qwen2.5vl"
    if data_args.data_flatten:
        replace_qwen2_vl_attention_class()
    model.config.use_cache = False

    dataset = make_rl_dataset(data_args)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    reward_funcs = [accuracy_reward, format_reward]

    # Start trainner
    trainer = QwenGRPOTrainer(model=model, ref_model=ref_model, processing_class=processor, reward_funcs=reward_funcs, args=training_args, train_dataset = dataset['train'], eval_dataset = dataset['test'] if training_args.eval_strategy != "no" else None,)

    trainer.train()
    trainer.save_state()

    data_args.image_processor.save_pretrained(training_args.output_dir)

    trainer.save_trained_model(training_args, lora_args)


if __name__ == "__main__":
    train()




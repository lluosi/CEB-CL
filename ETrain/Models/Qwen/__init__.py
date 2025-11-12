from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
import sys
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, AutoProcessor
import deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import prepare_model_for_kbit_training
from accelerate.utils import DistributedType
from peft import LoraConfig, prepare_model_for_kbit_training
from peft import get_peft_model
from ETrain.Train.Base_trainer import *
from ETrain.Models.Qwen.tokenization_qwen import *
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from vllm import LLM, SamplingParams

import sys
# sys.path.append('/home/chencheng/Code/Slim_Train')
# from CoIN.peft import PeftModel, TaskType, get_peft_model, CoINMOELoraConfig, WEIGHTS_NAME, set_peft_model_state_dict

def create_Qwen_model_peft(training_args, model_args, data_args, lora_args):
    bnb_model_from_pretrained_args = {}
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    local_rank = training_args.local_rank
    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        # torch_dtype=compute_dtype if training_args.bits == 16 else None,
        torch_dtype=torch.bfloat16,
        # device_map=device_map,
        # trust_remote_code=True,
    )


    if not training_args.use_lora:
        if training_args.fix_vit and hasattr(model,'transformer') and hasattr(model.transformer,'visual'):
            model.transformer.visual.requires_grad_(False)
            if hasattr(model.transformer.visual,'attn_pool'):
                model.transformer.visual.attn_pool.requires_grad_(True)


    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = getattr(tokenizer, "eod_id", None)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)


    if training_args.use_lora:
        if lora_args.q_lora or "chat" in model_args.model_name_or_path.lower() or "instruct" in model_args.model_name_or_path.lower():
            modules_to_save = None
        else:
            modules_to_save = ["wte", "lm_head"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=None  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    return model, tokenizer, processor


def create_Qwen_model(training_args, model_args, data_args, lora_args):
    bnb_model_from_pretrained_args = {}
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    
    local_rank = training_args.local_rank
    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        # torch_dtype=compute_dtype if training_args.bits == 16 else None,
        torch_dtype=torch.bfloat16,
        # device_map=device_map,
        # trust_remote_code=True,
    )

    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_args.previous_task_model_path,
        config=config,
        cache_dir=training_args.cache_dir,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = getattr(tokenizer, "eod_id", None)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    return model, tokenizer, processor, ref_model

def load_pretrained_model(model_path, model_base):
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_base, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    print(f"Model type: {type(model)}")
    if not model_base == model_path:
    #if False:
        print('Loading additional weights...')
        if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
            non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        else:
            assert False, 'non_lora_trainables.bin not found'
        non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith('model.model.') for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)

        from peft import PeftModel
        print('Loading LoRA weights...')
        model = PeftModel.from_pretrained(model, model_path)
        print('Merging LoRA weights...')
        model = model.merge_and_unload()
        print('Model is loaded...')

    tokenizer = AutoTokenizer.from_pretrained(
        model_base,
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    processor = AutoProcessor.from_pretrained(model_base, use_fast=True, trust_remote_code=True)
    processor.num_additional_image_tokens = 1
    try:
        min_pixels = 3136
        max_pixels = 480000
        processor.pad_token_id = tokenizer.pad_token_id
        processor.eos_token_id = tokenizer.eos_token_id
        processor.image_processor.max_pixels = max_pixels
        processor.image_processor.min_pixels = min_pixels
    except:
        pass


    return model, tokenizer, processor

def load_pretrained_vllm_model(model_path):

    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_base, device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
    model = LLM(
        model=model_path,
        gpu_memory_utilization=0.8,
        limit_mm_per_prompt={"image": 2},
        max_model_len=16384,
        enable_prefix_caching=True,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    processor = AutoProcessor.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    processor.num_additional_image_tokens = 1
    try:
        min_pixels = 3136
        max_pixels = 480000
        processor.pad_token_id = tokenizer.pad_token_id
        processor.eos_token_id = tokenizer.eos_token_id
        processor.image_processor.max_pixels = max_pixels
        processor.image_processor.min_pixels = min_pixels
    except:
        pass


    return model, tokenizer, processor
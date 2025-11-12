import os
import torch
import textwrap
from collections import defaultdict

from torch.utils.data import Sampler
from torch import nn
import re


from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from typing import Any, Callable, Optional, Union
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from typing import List, Optional
from ETrain.Train.LLaVA.llava_trainer import LLaVATrainer
from ETrain.Train.Base_trainer import *
from ETrain.Train.Qwen.prompts import *
from peft.utils import WEIGHTS_NAME, SAFETENSORS_WEIGHTS_NAME, set_peft_model_state_dict
from enum import Enum
from safetensors.torch import load_file as safe_load_file  # 需要读取 .safetensors
from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from datasets import Dataset, IterableDataset
from packaging import version
from accelerate.utils.other import is_compiled_module
from accelerate.utils import broadcast_object_list, gather, gather_object, set_seed
import warnings
from unittest.mock import patch
class ShardedDDPOption(str, Enum):
    OFF = "off"
    SIMPLE = "simple"
    ZERO_DP_2 = "zero_dp_2"
    ZERO_DP_3 = "zero_dp_3"


import sys


import copy
from PIL import Image
from CoIN.peft import PeftModel, TaskType, get_peft_model, CoINMOELoraConfig, WEIGHTS_NAME, set_peft_model_state_dict
from trl.import_utils import is_vllm_available

if is_vllm_available():
    from vllm import LLM, SamplingParams


# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def load_model_from_previous_task(model, previous_task_model_path):
    if os.path.exists(os.path.join(previous_task_model_path, 'non_lora_trainables.bin')):
        non_lora_trainables = torch.load(os.path.join(previous_task_model_path, 'non_lora_trainables.bin'),
                                         map_location='cpu')
    else:
        # this is probably from HF Hub
        from huggingface_hub import hf_hub_download
        def load_from_hf(repo_id, filename, subfolder=None):
            cache_file = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                subfolder=subfolder)
            return torch.load(cache_file, map_location='cpu')

        non_lora_trainables = load_from_hf(previous_task_model_path, 'non_lora_trainables.bin')
    non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
    if any(k.startswith('model.model.') for k in non_lora_trainables):
        non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
    model.load_state_dict(non_lora_trainables, strict=False)

    filename = os.path.join(previous_task_model_path, SAFETENSORS_WEIGHTS_NAME)
    adapters_weights = safe_load_file(filename, device="cpu")
    load_result = set_peft_model_state_dict(model, adapters_weights, adapter_name="default")


class QwenGRPOTrainer(LLaVATrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            ref_model : Union[str, PreTrainedModel],
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            args: None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (
            None, None),
            peft_config: Optional["PeftConfig"] = None,
            max_pixels: Optional[int] = 440000,
            min_pixels: Optional[int] = 3136,
            attn_implementation: str = "flash_attention_2",
            use_vllm_for_gen: bool = True,
    ):
        model_id = "qwen2.5-vl"
        self.model_id = model_id
        self.use_vllm = use_vllm_for_gen
        model_init_kwargs = {}
        model_init_kwargs["attn_implementation"] = "flash_attention_2"
        model_init_kwargs["torch_dtype"] = torch.bfloat16
        model_init_kwargs["use_cache"] = False
        model_name_or_path = "./checkpoints/Qwen/Qwen2.5-VL"

        if is_deepspeed_zero3_enabled():
            self.ref_model = ref_model
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        self.reasoning_threshold = 0.70

        # processing_class
        pad_token_id = processing_class.tokenizer.pad_token_id
        processing_class.pad_token_id = pad_token_id
        processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
        processing_class.image_processor.max_pixels = max_pixels
        processing_class.image_processor.min_pixels = min_pixels

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")
        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO]
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.beta = 0.15

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )
        self.args.remove_unused_columns = False

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
                self.ref_model.eval()
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

        if self.use_vllm:
            # use vllm to accelerate
            if self.accelerator.is_main_process:
                if torch.cuda.device_count() == 1:
                    vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                else:
                    vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                    print("*****************")
                    print(f"num processes is {self.accelerator.num_processes+1}")
                # Check that the requested device is available
                if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                    raise ValueError(
                        f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                        "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                        "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                        f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                    )
                # Check that the requested device is not also used for training
                if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                    warnings.warn(
                        f"The requested device {vllm_device} is also being used for training. For higher throughput "
                        "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                        "If this is intentional, you may ignore this warning but should adjust "
                        "`vllm_gpu_memory_utilization` accordingly."
                    )
                # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                # setting (profiling_patch).
                world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                print(vllm_device)
                with world_size_patch, profiling_patch:
                    self.llm = LLM(
                        model=model.name_or_path,
                        device=vllm_device,
                        gpu_memory_utilization=0.8,
                        limit_mm_per_prompt={"image": 2},
                        # tensor_parallel_size=7,
                        max_model_len=16384,
                        # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                        # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                        # This is particularly useful here because we generate completions from the same prompts.
                        enable_prefix_caching=True,
                    )
                self.sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=0.9,
                    top_k=50,
                    max_tokens=self.max_completion_length,
                )

            self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

            # When using vLLM, the main process is responsible for loading the model weights. This can cause process
            # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
            # synchronize all processes after vLLM has been fully initialized.
            self.accelerator.wait_for_everyone()
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=self.num_generations,
                pad_token_id=pad_token_id,
            )
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                temperature=args.temperature,
                num_return_sequences=self.num_generations,
                pad_token_id=pad_token_id,
            )
        self.ref_gen_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
        )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, **inputs):
        logits = model(**inputs).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = inputs['input_ids'][:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)

    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
                self.model, self.accelerator, gather_deepspeed3_params=True
        ) as unwrapped_model:
            src_model = unwrapped_model
            if is_compiled_module(unwrapped_model):
                state_dict = unwrapped_model._orig_mod.state_dict()
            else:
                state_dict = unwrapped_model.state_dict()

        if self.accelerator.is_main_process:
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(state_dict.items())


    def create_optimizer(self):
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
                projector_parameters = [
                    name for name, _ in opt_model.named_parameters() if "merger" in name
                ]
                if self.args.vision_tower_lr is not None and self.args.vision_tower_lr != 0:
                    vision_tower_parameters = [
                        name for name, _ in opt_model.named_parameters() if "visual" in name
                    ]
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                        n in decay_parameters
                                        and n not in projector_parameters
                                        and n not in vision_tower_parameters
                                        and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                        n in decay_parameters
                                        and n not in projector_parameters
                                        and n in vision_tower_parameters
                                        and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.vision_tower_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                        n not in decay_parameters
                                        and n not in projector_parameters
                                        and n not in vision_tower_parameters
                                        and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                        n not in decay_parameters
                                        and n not in projector_parameters
                                        and n in vision_tower_parameters
                                        and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.vision_tower_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                        n in decay_parameters
                                        and n in projector_parameters
                                        and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                        n not in decay_parameters
                                        and n in projector_parameters
                                        and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
                else:
                    optimizer_grouped_parameters = [
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                        n in decay_parameters
                                        and n not in projector_parameters
                                        and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                        n not in decay_parameters
                                        and n not in projector_parameters
                                        and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                        n in decay_parameters
                                        and n in projector_parameters
                                        and p.requires_grad
                                )
                            ],
                            "weight_decay": self.args.weight_decay,
                            "lr": self.args.mm_projector_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in opt_model.named_parameters()
                                if (
                                        n not in decay_parameters
                                        and n in projector_parameters
                                        and p.requires_grad
                                )
                            ],
                            "weight_decay": 0.0,
                            "lr": self.args.mm_projector_lr,
                        },
                    ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def build_reasoning_confidence_input(self, example, reasoning, REASONING_CONFIDENCE_SYSTEM_PROMPT, REASONING_CONFIDENCE_PROMPT):
        user_content = example["prompt"][1]["content"]
        texts = [c["text"] for c in user_content if c["type"] == "text"]
        question = texts[-1] if texts else ""
        new_question_text = REASONING_CONFIDENCE_PROMPT.format(question=question, reasoning = reasoning)
        new_example = {
            "prompt": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": REASONING_CONFIDENCE_SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [
                        *[{"type": "image"} for _ in example["image"]],
                        {"type": "text", "text": new_question_text},
                    ],
                },
            ],
            "image": example["image"],
            "solution": example["solution"],
        }
        return new_example

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]

        if self.use_vllm:
            vllm_prompts_text = copy.deepcopy(prompts_text)
            vllm_prompts = copy.deepcopy(prompts)


        # collect images
        images = []
        for x in inputs:
            if isinstance(x["image"], list):
                for image in x["image"]:
                    current_image = Image.open(image) if isinstance(image, str) else image
                    images.append(current_image)
            else:
                images = [Image.open(x["image"]) if isinstance(x["image"], str) else x["image"]]

        prompt_inputs = self.processing_class(
            text=prompts_text,
            images=images if len(images) > 0 else None,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        use_ref_judge = False
        use_ref_kl = False
        min_clip_kl = 0.20
        if use_ref_judge:
            ref_prompt_inputs = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in prompt_inputs.items()}
            with unwrap_model_for_generation(self.ref_model, self.accelerator) as unwrapped_ref_model, torch.inference_mode():
                unwrapped_ref_model.eval()
                ref_prompt_completion_ids = unwrapped_ref_model.generate(
                    **ref_prompt_inputs, generation_config=self.ref_gen_config
                )
                ref_prompt_len = ref_prompt_inputs["input_ids"].size(1)
                ref_completion_ids = ref_prompt_completion_ids[:, ref_prompt_len:]
                ref_completions = self.processing_class.tokenizer.batch_decode(ref_completion_ids, skip_special_tokens=True)
                pattern = re.compile(r"<think>(.*?)</think>\s*<answer>(.*?)</answer>", re.DOTALL)
                valid_mask = []
                reason_texts_for_ci = []
                for completion_text in ref_completions:
                    m = pattern.fullmatch(completion_text)
                    if not m:
                        valid_mask.append(0)
                        reason_texts_for_ci.append("")
                    else:
                        reason_texts_for_ci.append(m.group(1))
                        valid_mask.append(1)

                confidence_inputs_all = [
                    self.build_reasoning_confidence_input(
                        inputs[i],
                        reason_texts_for_ci[i],
                        REASONING_CONFIDENCE_SYSTEM_PROMPT,
                        REASONING_CONFIDENCE_PROMPT,
                    )
                    for i in range(len(reason_texts_for_ci))
                ]
                confidence_prompts_texts = [
                    maybe_apply_chat_template(ci, self.processing_class)["prompt"]
                    for ci in confidence_inputs_all
                ]
                try:
                    confidence_images = []
                    for ex in inputs:
                        if isinstance(ex["image"], list):
                            per_sample_imgs = []
                            for img in ex["image"]:
                                per_sample_imgs.append(Image.open(img) if isinstance(img, str) else img)
                            confidence_images.append(per_sample_imgs)
                        else:
                            confidence_images.append(
                                [Image.open(ex["image"]) if isinstance(ex["image"], str) else ex["image"]])
                except Exception:
                    confidence_images = None

                confidence_batch = self.processing_class(
                    text=confidence_prompts_texts,
                    images=confidence_images,
                    return_tensors="pt",
                    padding=True,
                    padding_side="left",
                    add_special_tokens=False,
                )
                confidence_batch = super()._prepare_inputs(confidence_batch)
                confidence_out = unwrapped_ref_model.generate(
                    **confidence_batch,
                    generation_config=self.ref_gen_config,
                    use_cache=True,
                    max_new_tokens=1,
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False,
                )
                scores_step0 = confidence_out.scores[0]  # logits
                tok = self.processing_class.tokenizer

                ids_A = tok.encode("A", add_special_tokens=False)
                ids_B = tok.encode("B", add_special_tokens=False)

                def pick_single_token_id(ids, fallback_text):
                    return ids[0] if len(ids) >= 1 else tok.encode(fallback_text, add_special_tokens=False)[0]

                id_A = pick_single_token_id(ids_A, "A")
                id_B = pick_single_token_id(ids_B, "B")

                logits_ab = torch.stack([scores_step0[:, id_A], scores_step0[:, id_B]], dim=1)
                probs_ab = torch.softmax(logits_ab, dim=1)
                p_true_batch = probs_ab[:, 0]
                reasoning_score = []
                confidence = []
                for i in range(p_true_batch.size(0)):
                    r = p_true_batch[i].item()
                    reasoning_score.append(r)
                    if valid_mask[i]:
                        confidence.append(max(r, min_clip_kl) if r < self.reasoning_threshold else 1.0)
                    else:
                        confidence.append(min_clip_kl)

                confidence = torch.tensor(confidence, device=device)

        prompt_inputs = {
            k: v.repeat_interleave(self.num_generations, dim=0) if isinstance(v, torch.Tensor) else v
            for k, v in prompt_inputs.items()
        }

        if self.max_prompt_length is not None:
            prompt_ids = prompt_inputs["input_ids"][:, -self.max_prompt_length:]
            prompt_mask = prompt_inputs["attention_mask"][:, -self.max_prompt_length:]
        else:
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # Generate completions using either vLLM or regular generation
        if self.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            if len(images) > 0 and len(vllm_prompts_text) == len(images):
                # single image
                prompts_text_and_vision = [
                    {"prompt": vllm_prompt, "multi_modal_data": {"image": vllm_image}}
                    for vllm_prompt, vllm_image in zip(vllm_prompts_text, images)
                ]
            elif len(images) > 0 and len(vllm_prompts_text) < len(images):
                num_prompts = len(vllm_prompts_text)
                images_per_prompt = len(images) // len(vllm_prompts_text)
                split_images = [images[i * images_per_prompt: (i + 1) * images_per_prompt] for i in range(num_prompts)]
                # multi image
                prompts_text_and_vision = [
                    {"prompt": vllm_prompt, "multi_modal_data": {"image": img_list}}
                    for vllm_prompt, img_list in zip(vllm_prompts_text, split_images)
                ]
            else:
                prompts_text_and_vision = [{"prompt": vllm_prompt} for vllm_prompt in vllm_prompts_text]

            prompts_text_and_vision = [item for item in prompts_text_and_vision for _ in range(self.num_generations)]
            vllm_prompts = [item for item in vllm_prompts for _ in range(self.num_generations)]

            all_prompts_text_and_vision = gather_object(prompts_text_and_vision)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text_and_vision, sampling_params=self.sampling_params,
                                            use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text_and_vision)

            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(vllm_prompts),
                (self.accelerator.process_index + 1) * len(vllm_prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            prompt_length = prompt_ids.size(1)
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Generate completions
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                if "qwen2-vl" in self.model_id.lower() or "qwen2_vl" in self.model_id.lower() or "qwen2vl" in self.model_id.lower():
                   #     or "qwen2.5-vl" in self.model_id.lower() or "qwen2.5_vl" in self.model_id.lower() or "qwen2.5vl" in self.model_id.lower():  # BUG in Qwen-VL / maybe fixed
                    # Generate N times, each generate one with the temp_generation_config , stack the output_ids to prompt_completion_ids, pad the empty places with number 151613
                    num_generations = self.generation_config.num_return_sequences
                    temp_generation_config = copy.deepcopy(self.generation_config)
                    temp_generation_config.num_return_sequences = 1

                    all_completions = []

                    for i in range(num_generations):  # -1 because we already have one generation
                        completion = unwrapped_model.generate(**prompt_inputs, generation_config=temp_generation_config)
                        all_completions.append(completion)

                    # Stack all completions and pad if needed
                    max_length = max(completion.size(1) for completion in all_completions)
                    padded_completions = []
                    for completion in all_completions:
                        if completion.size(1) < max_length:
                            padding = torch.full((completion.size(0), max_length - completion.size(1)),
                                                 self.processing_class.tokenizer.pad_token_id,
                                                 dtype=completion.dtype,
                                                 device=completion.device)
                            padded_completion = torch.cat([completion, padding], dim=1)
                        else:
                            padded_completion = completion
                        padded_completions.append(padded_completion)

                    # Stack all padded completions
                    prompt_completion_ids = torch.cat(padded_completions, dim=0)
                else:
                    temp_generation_config = copy.deepcopy(self.generation_config)
                    temp_generation_config.num_return_sequences = 1
                    prompt_completion_ids = unwrapped_model.generate(**prompt_inputs,
                                                                     generation_config=temp_generation_config)

            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]


        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        prompt_inputs["input_ids"] = prompt_completion_ids
        prompt_inputs["attention_mask"] = attention_mask

        per_token_logps = self._get_per_token_logps(model, **prompt_inputs)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1:]

        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, **prompt_inputs)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, **prompt_inputs)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]

        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        # Decode the generated completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = [[{"role": "assistant", "content": completion}] for completion in completions]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]


        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(prompts=prompts, completions=completions,
                                                 current_step=self.state.global_step, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)  # 最基本的均值基线advantage

        # x - x.detach() allows for preserving gradients from x
        if use_ref_judge:
            confidence = confidence.repeat_interleave(self.num_generations, dim=0)
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
            if use_ref_kl:
                per_token_loss = -(per_token_loss - self.beta * confidence.unsqueeze(1) * per_token_kl)
            else:
                per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        else:
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
            per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss_batch = (per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)


        loss = loss_batch.mean()


        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()


    def save_trained_model(self, training_args, lora_args):
        if training_args.use_lora:
            state_dict = get_peft_state_maybe_zero_3(
                self.model.named_parameters(), lora_args.lora_bias
            )
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
                self.model.named_parameters()
            )
            if training_args.local_rank == 0 or training_args.local_rank == -1:
                self.model.config.save_pretrained(training_args.output_dir)
                self.model.save_pretrained(training_args.output_dir, state_dict=state_dict)
                torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
        else:
            safe_save_model_for_hf_trainer(trainer=self,
                                           output_dir=training_args.output_dir)
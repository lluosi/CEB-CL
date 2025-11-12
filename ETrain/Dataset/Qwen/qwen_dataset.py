# qwen_dataset.py  —— 使用 AutoProcessor 处理图像的版本
from dataclasses import dataclass
import json
from typing import Dict, Sequence, List, Optional, Union
import torch
from torch.utils.data import Dataset
import transformers
from transformers.trainer_pt_utils import LabelSmoother
from torch.utils.data.dataloader import default_collate
from PIL import Image
import requests
from transformers.models.qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
import re




# 你项目里还在引用
from ETrain.Models.Qwen.modeling_qwen import QWenLMHeadModel

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def rank0_print(local_rank, *args):
    if local_rank == 0:
        print(*args)


def preprocess(
        sources,
        image_paths,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = "You are a helpful assistant.",
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant:", "human": "<|im_start|>user", "gpt":"<|im_start|>assistant:"}
    print("*********image path********")
    print(image_paths)
    print(f"length of image paths: {len(image_paths)}")
    print("*********sources********")
    print(sources)
    print(f"length of sources list: {len(sources)}")
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    if tokenizer.pad_token is None or tokenizer.pad_token == tokenizer.eos_token:
        tokenizer.pad_token = "<|extra_0|>"  # 使用未占用的 reserved token

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if image_paths[i] != "":
            for sentence in source:
                # 如果是用户角色且 value 中没有 <image> 占位符，则添加图像占位符
                if (sentence["from"] == "user" or sentence["from"] == "human") and '<image>' not in sentence["value"]:
                    sentence["value"] = "<|vision_start|><|image_pad|><|vision_end|>" + sentence["value"]
                # 如果是其他角色且 value 中包含 <image>，则将其替换为新占位符
                elif '<image>' in sentence["value"]:
                    sentence["value"] = sentence["value"].replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')

        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end]
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 2) + [im_end]
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]

            if role == '<|im_start|>user':
                _input_id = tokenizer(role).input_ids + tokenizer(sentence["value"]).input_ids + [
                    im_end]
                input_id += _input_id
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 2) + [im_end]
            elif role == '<|im_start|>assistant:':
                _input_id = tokenizer(role).input_ids  + tokenizer(sentence["value"]).input_ids + [
                    im_end]
                input_id += _input_id
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                          _input_id[len(tokenizer(role).input_ids) + 1:-1] + [im_end]
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)

        # —— 插入 image_path 标记 ——
        path_str = image_paths[i]  # image_paths 需与你的 sources 同长度
        markup = f"<path>{path_str}</path>\n"
        markup_ids = tokenizer(markup).input_ids
        input_id += markup_ids
        target += [IGNORE_TOKEN_ID] * len(markup_ids)
        # —— image_path 标记结束 ——

        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    print("the prompt is:")
    decoded = tokenizer.decode([i for i in input_ids[0].tolist() if i != tokenizer.pad_token_id], skip_special_tokens=False)
    print(decoded)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, local_rank: int):
        super(SupervisedDataset, self).__init__()

        rank0_print(local_rank, "Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        image_paths = [
            example["image"] if "image" in example and example["image"] else ""
            for example in raw_data
        ]



        data_dict = preprocess(sources, image_paths, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i].to(torch.int64),
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int, local_rank: int,
                 model: QWenLMHeadModel):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        rank0_print(local_rank, "Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

        self.model = model
        self.config = model.config

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            print('Getting cached data: Index:{} : {}'.format(i, self.cached_data_dict[i]))
            return self.cached_data_dict[i]
        image_paths=[]
        image_path = self.raw_data[i].get("image", "")
        if image_path is None:
            image_path = "NO"  # 避免 None 出现在 DataLoader 中
        image_paths.append(image_path)
        print("********* original image path********")
        print(image_paths)
        ret = preprocess([self.raw_data[i]["conversations"]], image_paths, self.tokenizer, self.max_len)




        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0].to(torch.int64),
            attention_mask=ret["attention_mask"][0],
        )

        self.cached_data_dict[i] = ret

        # print('Getting data: Index:{} : {}'.format(i,self.raw_data[i]["conversations"]))
        return ret


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""


    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, model: QWenLMHeadModel, processor):
        self.tokenizer = tokenizer
        self.model = model
        self.config = model.config
        self.processor = processor

    def __call__(self, batch: Sequence[Dict]):
        batch = default_collate(batch)
        # 还原 image_path 字符串
        prompts = []
        image_paths = []
        # batch['input_ids'] 形状 [B, L]
        for seq_ids in batch["input_ids"]:
            # 1) 去掉 padding
            ids = [i for i in seq_ids.tolist() if i != self.tokenizer.pad_token_id]
            # 2) decode（保留特殊 token）
            text = self.tokenizer.decode(ids, skip_special_tokens=False)
            prompt_text = re.split(r"assistant:", text)[0]
            prompt_text = prompt_text+"assistant:"
            prompts.append(prompt_text)
            # 3) 正则提取
            m = re.search(r"<path>(.*?)</path>", text)
            image_paths.append(m.group(1) if m else "")

        print("==========prompts==============")
        print(prompts)
        print("=======image paths===========")
        print(image_paths)

        images = []
        """
        if torch.any(batch['input_ids'] == self.config.visual['image_start_id']):
            bos_pos = torch.where(batch['input_ids'] == self.config.visual['image_start_id'])
            eos_pos = torch.where(batch['input_ids'] == self.config.visual['image_start_id'] + 1)
            assert (bos_pos[0] == eos_pos[0]).all()
            img_pos = torch.stack((bos_pos[0], bos_pos[1], eos_pos[1]), dim=1)
            image_paths = []
            for i, a, b in img_pos:
                image = batch['input_ids'][i][a + 1: b - 1].tolist()
                image = image[: image.index(self.config.visual['image_start_id'] + 2)]
                image_paths.append(bytes(image).decode('utf-8'))

            for image_path in image_paths:
                if image_path.startswith("http://") or image_path.startswith("https://"):
                    image = Image.open(requests.get(image_path, stream=True).raw)
                else:
                    image = Image.open(image_path)
                image = image.convert("RGB")
                # images.append(self.model.transformer.visual.image_transform(image))
                processed = self.processor(images=image, return_tensors="pt")  # 注意 batch=False
                images.append(processed["pixel_values"][0])  # 单张图像取第0维
            images = torch.stack(images, dim=0)
        else:
            image = torch.zeros(3, 224, 224).to(dtype=self.model.transformer.visual.conv1.weight.dtype)
            images.append(image)
            images = torch.stack(images, dim=0)
        """
        """
        for path in image_paths:
            if path=='':
                pixel = torch.zeros((3, 224, 224))  # 占位图像
            else:
                path="cl_dataset/"+path
                print("the path is" + path)
                image = Image.open(path).convert("RGB")
                pixel = self.processor(images=image, text=[""], return_tensors="pt")
            images.append(pixel)
            print(f"DEBUG pixel shape for path={path!r}:", pixel.shape)
        """
        image_files = []
        for path in image_paths:
            if path == '':
                img = Image.new("RGB", (224, 224), 0)
            else:
                path = "cl_dataset/" + path
                print("the path is" + path)
                img = Image.open(path).convert("RGB")

                #
                img = img.resize((224, 224))

            image_files.append(img)

        out_list = []
        for i in range(5):
            prompt0 = prompts[i]
            img = image_files[i]
            image_path = image_paths[i]
            print(f"prompt:{prompt0}, image_path:{image_path}")

            # processor 支持单条数据处理（注意返回 tensor 维度是否为 batch=1）
            out = self.processor(images=img, text=prompt0, padding=True, return_tensors="pt", max_length=819200)
            # 注意 out 可能返回 dict
            out_list.append(out)

        # 计算最大长度
        max_len = max([o["input_ids"].shape[1] for o in out_list])
        pad_token_id = self.processor.tokenizer.pad_token_id if hasattr(self.processor, 'tokenizer') else 0

        # 补齐 input_ids
        padded_input_ids = [
            torch.nn.functional.pad(
                o["input_ids"],
                (0, max_len - o["input_ids"].shape[1]),
                value=pad_token_id
            )
            for o in out_list
        ]
        batch_input_ids = torch.cat(padded_input_ids, dim=0)

        # 补齐 attention_mask
        padded_attention_mask = [
            torch.nn.functional.pad(
                o["attention_mask"],
                (0, max_len - o["attention_mask"].shape[1]),
                value=0
            )
            for o in out_list
        ]
        batch_attention_mask = torch.cat(padded_attention_mask, dim=0)
        
        batch_pixel_values = torch.cat([o["pixel_values"] for o in out_list], dim=0)
        batch_image_grid_thw = torch.cat([o["image_grid_thw"] for o in out_list], dim=0)

        batch["input_ids"] = batch_input_ids
        batch["attention_mask"] = batch_attention_mask
        batch["pixel_values"] = batch_pixel_values
        batch["image_grid_thw"] = batch_image_grid_thw

        return batch


def make_supervised_data_module(
        tokenizer: transformers.PreTrainedTokenizer, data_args, model, max_len, local_rank: int, processor: object
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print(local_rank, "Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len, local_rank=local_rank, model=model)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, max_len=max_len, local_rank=local_rank)
    else:
        eval_dataset = None
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer, model=model, processor=processor)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)
    # return dict(train_dataset=train_dataset, eval_dataset=eval_dataset) #适配GRPOTrainer
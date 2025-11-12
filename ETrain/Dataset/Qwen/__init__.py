
import re

# Define placeholders for dataset paths
SCIENCEQA = {
    "annotation_path": "/home/liuyuyang/CoIN/playground/Instructions_Qwen/ScienceQA/train.json",
    "data_path": "./cl_dataset",
}

TEXTVQA = {
    "annotation_path": "/home/liuyuyang/CoIN/playground/Instructions_Qwen/TextVQA/train.json",
    "data_path": "./cl_dataset",
}

GQA = {
    "annotation_path": "/home/liuyuyang/CoIN/playground/Instructions_Qwen/GQA/train.json",
    "data_path": "./cl_dataset",
}

GROUNDING = {
    "annotation_path": "/home/liuyuyang/CoIN/playground/Instructions_Qwen/Grounding/train.json",
    "data_path": "./cl_dataset",
}

IMAGENET = {
    "annotation_path": "/home/liuyuyang/CoIN/playground/Instructions_Light/ImageNet/train_3000.json",
    "data_path": "./cl_dataset",
}

OCRVQA = {
    "annotation_path": "/home/liuyuyang/CoIN/playground/Instructions_Qwen/OCRVQA/train.json",
    "data_path": "./cl_dataset",
}

VIZWIZ = {
    "annotation_path": "/home/liuyuyang/CoIN/playground/Instructions_Light/VizWiz/train_2000.json",
    "data_path": "./cl_dataset",
}

VQAV = {
    "annotation_path": "/home/liuyuyang/CoIN/playground/Instructions_Qwen/VQAv2/train.json",
    "data_path": "./cl_dataset",
}

ICONQA = {
    "annotation_path": "/home/liuyuyang/CoIN/playground/Instructions_Light/IconQA/train_3000.json",
    "data_path": "./cl_dataset",
}

MULTITASKS = {
    "annotation_path": "/home/liuyuyang/CoIN/playground/Instructions_Light/multitasks_shuffle.json",
    "data_path": "./cl_dataset",
}


data_dict = {
    "gqa": GQA,
    "grounding": GROUNDING,
    "imagenet": IMAGENET,
    "ocrvqa": OCRVQA,
    "scienceqa": SCIENCEQA,
    "textvqa": TEXTVQA,
    "vizwiz": VIZWIZ,
    "vqav2": VQAV,
    "iconqa": ICONQA,
    "multitasks": MULTITASKS,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list




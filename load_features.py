import os

import argparse
import pickle as pkl
import random

import open_clip
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr, spearmanr
from scipy.stats import kendalltau as kendallr
from tqdm import tqdm

from buona_vista import datasets

import wandb


def rescale(x):
    x = np.array(x)
    print("Mean:", x.mean(), "Std", x.std())
    x = (x - x.mean()) / x.std()
    return 1 / (1 + np.exp(-x))

def get_features(save_features=True):
    with open("buona_vista_sa_index.yml", "r") as f:
        opt = yaml.safe_load(f)   

    val_datasets = {}
    for name, dataset in opt["data"].items():
        val_datasets[name] = getattr(datasets, dataset["type"])(dataset["args"])

    print(open_clip.list_pretrained())
    model, _, _ = open_clip.create_model_and_transforms("RN50",pretrained="openai")
    model = model.to("cuda")
    print("loading succeed")

    texts = [
            "a high quality photo",
            "a low quality photo",
            "a good photo",
            "a bad photo",
    ]
    tokenizer = open_clip.get_tokenizer("RN50")
    text_tokens = tokenizer(texts).to("cuda")
    print(f"Prompt_loading_succeed, {texts}")

    results = {}

    gts, paths = {}, {}

    for val_name, val_dataset in val_datasets.items():
        gts[val_name] = [val_dataset.video_infos[i]["label"] for i in range(len(val_dataset))]

    for val_name, val_dataset in val_datasets.items():
        paths[val_name] = [val_dataset.video_infos[i]["filename"] for i in range(len(val_dataset))]


    visual_features = {}

    for val_name, val_dataset in val_datasets.items():
        if val_name != "val-ltrain" and val_name != "val-l1080p":
            visual_features[val_name] = []
            val_loader = torch.utils.data.DataLoader(
                 val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
            )
            for i, data in enumerate(tqdm(val_loader, desc=f"Evaluating in dataset [{val_name}].")):
                video_input = data["aesthetic"].to("cuda").squeeze(0).transpose(0,1)
                with torch.no_grad():
                    video_features = model.encode_image(video_input)
                    visual_features[val_name].append(video_features.cpu())


    if save_features:
        torch.save(visual_features, "CLIP_vis_features.pt")
        
    return visual_features
        
if __name__ == "__main__":
    get_features()
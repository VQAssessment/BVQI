# Contributed by Teo Haoning Wu, Erli Zhang Karl

import argparse
import glob
import math
import os
import pickle as pkl
from collections import OrderedDict

import decord
import numpy as np
import torch
import torchvision as tv
import yaml
from pyiqa import create_metric
from pyiqa.default_model_configs import DEFAULT_CONFIGS
from pyiqa.utils.img_util import imread2tensor
from pyiqa.utils.registry import ARCH_REGISTRY
from scipy.stats import kendalltau as kendallr
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from torch.nn.functional import interpolate

from buona_vista.datasets import ViewDecompositionDataset
from skvideo.measure import niqe

def rescale(x):
    x = np.array(x)
    x = (x - x.mean()) / x.std()
    return 1 / (1 + np.exp(x))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--opt",
        type=str,
        default="./buona_vista_sn_index.yml",
        help="the option file",
    )

    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="the running device"
    )

    args = parser.parse_args()

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)

    metric_name = "niqe"

    # set up IQA model
    iqa_model = create_metric(metric_name, metric_mode="NR")
    # pbar = tqdm(total=test_img_num, unit='image')
    lower_better = DEFAULT_CONFIGS[metric_name].get("lower_better", False)
    device = args.device
    net_opts = OrderedDict()
    kwargs = {}
    if metric_name in DEFAULT_CONFIGS.keys():
        default_opt = DEFAULT_CONFIGS[metric_name]["metric_opts"]
        net_opts.update(default_opt)
    # then update with custom setting
    net_opts.update(kwargs)
    network_type = net_opts.pop("type")
    net = ARCH_REGISTRY.get(network_type)(**net_opts)
    net = net.to(device)

    for key in opt["data"].keys():
        if "val" not in key and "test" not in key:
            continue

        dopt = opt["data"][key]["args"]

        val_dataset = ViewDecompositionDataset(dopt)

        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
        )

        pr_labels, gt_labels = [], []

        for data in tqdm(val_loader, desc=f"Evaluating in dataset [{key}]."):
            target = (
                data["original"].squeeze(0).transpose(0, 1)
            )  # C, T, H, W to N(T), C, H, W
            if min(target.shape[2:]) < 224:
                target = interpolate(target, scale_factor = 224 / min(target.shape[2:]))
            with torch.no_grad():
                score = net((target.to(device))).mean().item()

                if math.isnan(score):
                    print(score, target.shape)
                    score = max(pr_labels) + 1

            #with open(output_result_csv, "a") as w:
            #    w.write(f'{data["name"][0]}, {score}\n')

            pr_labels.append(score)
            gt_labels.append(data["gt_label"].item())

        pr_labels = rescale(pr_labels)

        s = spearmanr(gt_labels, pr_labels)[0]
        p = pearsonr(gt_labels, pr_labels)[0]
        k = kendallr(gt_labels, pr_labels)[0]
        with open(f"spatial_naturalness_{key}.pkl", "wb") as f:
            pkl.dump({"pr_labels": pr_labels,
                      "gt_labels": gt_labels}, f)
        print(s, p, k)

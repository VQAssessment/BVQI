## Contributed by Teo Haoning Wu, Daniel Annan Wang

import argparse
import pickle as pkl

import clip
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm

from buona_vista import datasets

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--opt",
        type=str,
        default="./buona_vista_sa_index.yml",
        help="the option file",
    )

    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="the running device"
    )

    args = parser.parse_args()

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)

    val_datasets = {}
    for name, dataset in opt["data"].items():
        val_datasets[name] = getattr(datasets, dataset["type"])(dataset["args"])

    print(clip.available_models())
    model, preprocess = clip.load("ViT-B/32")
    model = model.to(args.device)
    print("loading succeed")

    texts = [
        "high quality",
        "low quality",
        "a good photo",
        "a bad photo",
    ]
    text_tokens = clip.tokenize(texts).to(args.device)
    print(f"Prompt_loading_succeed, {texts}")

    results = {}
    prs, gts = [], []

    for val_name, val_dataset in val_datasets.items():
        results[val_name] = {"gt": [], "sa_index": []}
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
        )
        for i, data in enumerate(tqdm(val_loader, desc=f"Evaluating in dataset [{val_name}].")):
            video_frames = data["aesthetic"].squeeze(0)
            image_input = torch.transpose(video_frames, 0, 1).to(args.device)

            with torch.no_grad():
                image_features = model.encode_image(image_input).float()
                text_features = model.encode_text(text_tokens).float()

                logits_per_image, logits_per_text = model(image_input, text_tokens)

            probs_a = logits_per_image.cpu().numpy()

            semantic_affinity_index = 0
            for k in [0, 1]:
                semantic_affinity_index += (
                    torch.from_numpy(probs_a[:, 2 * k : 2 * k + 2])
                    .float()
                    .softmax(1)
                    .mean(0)
                    .numpy()[0]
                )

            results[val_name]["gt"].append(data["gt_label"])
            gts.append(data["gt_label"])

            results[val_name]["sa_index"].append(semantic_affinity_index)
            prs.append(semantic_affinity_index)

            with open("semantic_affinity.pkl", "wb") as f:
                pkl.dump(results, f)
        print(
            "Dataset:",
            val_name,
            "Length:",
            len(val_dataset),
            "SRCC:",
            spearmanr(prs, gts)[0],
            "PLCC:",
            pearsonr(prs, gts)[0],
        )

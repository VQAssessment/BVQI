## Contributed by Teo Haoning Wu, Daniel Annan Wang

import argparse
import pickle as pkl

import open_clip
import numpy as np
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from scipy.stats import kendalltau as kendallr
from tqdm import tqdm

from buona_vista import datasets

def rescale(x):
    x = np.array(x)
    x = (x - x.mean()) / x.std()
    return 1 / (1 + np.exp(-x))

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
    
    parser.add_argument(
        "-l", "--local", action="store_true", help="Use BVQI-Local"
    )

    args = parser.parse_args()

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)

    val_datasets = {}
    for name, dataset in opt["data"].items():
        val_datasets[name] = getattr(datasets, dataset["type"])(dataset["args"])

    print(open_clip.list_pretrained())
    model, _, preprocess = open_clip.create_model_and_transforms("RN50",pretrained="openai")
    model = model.to(args.device)
    print("loading succeed")

    texts = [
        "a high quality photo",
        "a low quality photo",
        "a good photo",
        "a bad photo",
    ]
    tokenizer = open_clip.get_tokenizer("RN50")
    text_tokens = tokenizer(texts).to(args.device)
    print(f"Prompt_loading_succeed, {texts}")

    results = {}
    

    for val_name, val_dataset in val_datasets.items():
        prs, gts = [], []
        results[val_name] = {"gt": [], "sa_index": [], "raw_index": []}
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
        )
        for i, data in enumerate(tqdm(val_loader, desc=f"Evaluating in dataset [{val_name}].")):
            video_frames = data["aesthetic"].squeeze(0)
            image_input = torch.transpose(video_frames, 0, 1).to(args.device)

            with torch.no_grad():
                image_features = model.encode_image(image_input).float() #.mean(0)
                text_features = model.encode_text(text_tokens).float()
                logits_per_image = image_features @ text_features.T
                
                
                #logits_per_image = logits_per_image.softmax(dim=-1)
                #logits_per_image, logits_per_text = model(image_input, text_tokens)

            probs_a = logits_per_image.cpu().numpy()

            semantic_affinity_index = 0
            
            for k in [0,1]:
                pn_pair = torch.from_numpy(probs_a[..., 2 * k : 2 * k + 2]).float().numpy()
                semantic_affinity_index += pn_pair[...,0] - pn_pair[...,1]
            if args.local:
                # Use the local feature after AttnPooling
                prs.append(semantic_affinity_index[1:].mean())
            else:
                # Use the global feature after AttnPooling
                prs.append(semantic_affinity_index[0].mean())
                
            results[val_name]["gt"].append(data["gt_label"][0].item())
            
            gts.append(data["gt_label"][0].item())
            results[val_name]["raw_index"].append(semantic_affinity_index)
            
        prs = rescale(prs)
        with open("semantic_affinity_pubs.pkl", "wb") as f:
            results[val_name]["sa_index"] = prs
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
            "KRCC:",
            kendallr(prs, gts)[0],
        )

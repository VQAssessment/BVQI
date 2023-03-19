import torch
import argparse
import pickle as pkl

import numpy as np
import math
import torch
import yaml
from scipy.stats import pearsonr, spearmanr
from scipy.stats import kendalltau as kendallr
from tqdm import tqdm
from sklearn import decomposition

from buona_vista import datasets
from V1_extraction.gabor_filter import GaborFilters
from V1_extraction.utilities import compute_v1_curvature, compute_discrete_v1_curvature, compute_fragment_v1_curvature


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
        default="buona_vista_tn_index.yml",
        help="the option file",
    )

    parser.add_argument(
        "-d", "--device", type=str, default="cuda", help="the running device"
    )
    
    args = parser.parse_args()
    results = {}
    
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
        
    scale = 6
    orientations = 8
    kernel_size = 39
    row_downsample = 4
    column_downsample = 4

    pca_d = 5
    frame_bs = 32

    gb = GaborFilters(scale,
                      orientations, (kernel_size - 1) // 2,
                      row_downsample,
                      column_downsample,
                      device=args.device
                     )

    val_datasets = {}
    for name, dataset in opt["data"].items():
        
        val_datasets[name] = getattr(datasets, dataset["type"])(dataset["args"])
          
    for val_name, val_dataset in val_datasets.items():
        prs, gts = [], []
        results[val_name] = {"gt": [], "tn_index": []}
        val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
        )
        
        for i, data in enumerate(tqdm(val_loader, desc=f"Evaluating in dataset [{val_name}].")):
            video_frames = data["aesthetic"].squeeze(0).to(args.device)
            video_frames = torch.transpose(video_frames, 0, 1).mean(1,keepdim=True)
            zero_frames = torch.zeros(video_frames.shape).to(args.device)
            complex_frames = torch.stack((video_frames, zero_frames), -1)
            video_frames = torch.view_as_complex(complex_frames)
            v1_features  = []
            for i in range((video_frames.shape[0] - 1) // frame_bs):
                these_frames = video_frames[i * frame_bs:(i+1)* frame_bs]
                with torch.no_grad():
                    these_features = gb(these_frames)
                v1_features.append(these_features)
            last_start = ((video_frames.shape[0] - 1) // frame_bs) * frame_bs
            v1_features += [gb(video_frames[last_start:])]
            v1_features = torch.cat(v1_features, 0)
            print(v1_features.shape)
            v1_features = torch.nan_to_num(v1_features)
            v1_features = v1_features.cpu().numpy()
            pca = decomposition.PCA()
            print(v1_features.shape)
            v1_features = pca.fit_transform(v1_features)
            v1_PCA = v1_features[:, :pca_d]
            print(v1_PCA.shape)
            v1_score = compute_v1_curvature(v1_PCA)
            temporal_naturalness_index = math.log(np.mean(v1_score))
            #print(temporal_naturalness_index)
            
            results[val_name]["tn_index"].append(temporal_naturalness_index)
            if not np.isnan(temporal_naturalness_index):
                prs.append(temporal_naturalness_index)
                gts.append(data["gt_label"][0].item())            
            print(spearmanr(prs, gts)[0])
            
        # Sigmoid-like Rescaling
        prs = rescale(prs)
        results[val_name]["tn_index"] = rescale(results[val_name]["tn_index"])
        
        with open("temporal_naturalness_pubs.pkl", "wb") as f:
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
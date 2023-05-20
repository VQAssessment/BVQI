import os, glob

import argparse
import pickle as pkl
import random

from copy import deepcopy

import open_clip
import numpy as np
import torch
import torch.nn as nn
import yaml
from scipy.stats import pearsonr, spearmanr
from scipy.stats import kendalltau as kendallr
from tqdm import tqdm

from buona_vista import datasets

from load_features import get_features

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.transformer.get_cast_dtype()
        self.attn_mask = clip_model.attn_mask

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        return x

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.in_ln = nn.Linear(in_channels, hidden_channels, bias=False)
        self.out_ln = nn.Linear(hidden_channels, out_channels, bias=False)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self.bn = nn.BatchNorm2d(1, affine=False)
        
    def forward(self, x):
        bef_norm = self.out_ln(self.dropout(self.gelu(self.in_ln(x)))).squeeze(-1)
        return (torch.sigmoid(self.bn(bef_norm[:, None, :, :])))
    
class FFN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.ln = nn.Linear(in_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(1, affine=False)
        
    def forward(self, x):
        bef_norm = self.ln(x).squeeze(-1)
        return (torch.sigmoid(self.bn(bef_norm[:, None, :, :])))
               
class VisualFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, indices=None):
        super().__init__()
        if indices == None:
            indices = range(len(sn[dataset_name]))
            print("Using all indices:", indices)
        self.temporal = [tn2[dataset_name][ind] for ind in indices]
        self.spatial = [sn[dataset_name][ind] for ind in indices]
        self.clip_visual_features = [visual_features[dataset_name][ind] for ind in indices]
        self.gts = [gts[dataset_name][ind] for ind in indices]
        
    def __getitem__(self, index):
        return self.clip_visual_features[index], self.spatial[index], self.temporal[index], self.gts[index]
    def __len__(self):
        return len(self.gts)
    
class FastVisualFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, indices=None):
        super().__init__()
        if indices == None:
            indices = range(len(sn[dataset_name]))
            print("Using all indices:", indices)
        self.temporal = [tn2[dataset_name][ind] for ind in indices]
        self.spatial = [sn[dataset_name][ind] for ind in indices]
        self.clip_visual_features = [visual_features[dataset_name][ind] for ind in indices]
        self.fast_visual_features = [fast_visual_features[dataset_name]["feats"][ind] for ind in indices]
        self.gts = [gts[dataset_name][ind] for ind in indices]
        
    def __getitem__(self, index):
        return self.clip_visual_features[index], self.spatial[index], self.temporal[index], self.gts[index], self.fast_visual_features[index].reshape(4,1,768)
    def __len__(self):
        return len(self.gts)
    
class SimpleFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, indices):
        super().__init__()
        #self.temporal = [tn2[dataset_name][ind] for ind in indices]
        #self.spatial = [sn[dataset_name][ind] for ind in indices]
        self.clip_visual_features = [visual_features[dataset_name][ind] for ind in indices]
        self.gts = [gts[dataset_name][ind] for ind in indices]
        
    def __getitem__(self, index):
        return self.clip_visual_features[index], self.gts[index]
    def __len__(self):
        return len(self.gts)
    
class BVQI(nn.Module):
    """
        Modified CLIP, which combined prompt tuning and feature adaptation.
        The spatial and temporal naturalnesses are fed as final features.
        Implcit features is also optional fed into the model.
    """
    def __init__(self, text_tokens, embedding, n_pairs=2,implicit=False, optimizable_encoder=None):
        
        super().__init__()
        self.n_pairs = n_pairs
        self.device = "cuda"
        self.implicit = implicit
        if self.implicit:
            self.implicit_mlp = MLP(1024,64,1)
        self.tokenized_prompts = text_tokens
        #self.text_encoder = TextEncoder(clip_model)
        
        if optimizable_encoder is not None:
            print("Optimizing the text encoder.")
            self.optimizable_encoder = deepcopy(text_encoder)
            for param in self.optimizable_encoder.parameters():
                param.requires_grad = True
        
        if n_ctx > 0:
            self.ctx = nn.Parameter(embedding[:, 1:1+n_ctx].clone())
        else:
            self.register_buffer("ctx", embedding[:, 1:1, :])
            print("Disabled Context Prompt")
        self.register_buffer("prefix", embedding[:, :1, :].clone())  # SOS
        self.register_buffer("suffix", embedding[:, 1 + n_ctx:, :].clone())# CLS, EOS
        
        self.prefix.requires_grad = False
        self.suffix.requires_grad = False
        self.dropout = nn.Dropout(0.5)
        
        self.final_ln = nn.Linear(n_pairs+2+implicit,1,bias=False)
        print(self.final_ln)
        torch.nn.init.constant_(self.final_ln.weight, 1)
        
        
        n_prompts = self.get_text_prompts()
        self.text_feats = text_encoder(n_prompts.cuda(), self.tokenized_prompts)
        
    def get_text_prompts(self):
        return torch.cat(
                [
                    self.prefix,  # (n_cls, 1, dim)
                    self.ctx,     # (n_cls, n_ctx, dim)
                    self.suffix,  # (n_cls, *, dim)
                ],
                dim=1,
        )
        
    
    def forward(self, vis_feat, sn_ind=None, tn_ind=None, train=True):
        n_prompts = self.get_text_prompts()
        if train:
            if hasattr(self, "optimizable_encoder"):
                text_feats = self.optimizable_encoder(n_prompts, self.tokenized_prompts)
            else:
                text_feats = text_encoder(n_prompts, self.tokenized_prompts)
            self.text_feats = text_feats 
        else:
            text_feats = self.text_feats
            
        vis_feats = vis_feat[:,1:].to(self.device)
        if self.implicit:
            sa_ind = [self.implicit_mlp(vis_feats).mean((-1,-2,-3))]
        else:
            sa_ind = []
        self.vis_feats = vis_feats 
        logits = 2 * self.dropout(self.vis_feats) @ text_feats.T
        
        final_feats = [sn_ind.to(self.device), tn_ind.to(self.device)]
    
        for k in range(self.n_pairs):
            pn_pair = logits[..., 2 * k : 2 * k + 2].float() #.softmax(-1)[...,0]
            sa_ind += [torch.sigmoid(pn_pair[...,0] - pn_pair[...,1]).mean((-1,-2))]
         
        final_feats += sa_ind
        final_feats = torch.stack(final_feats, -1).float()
        return final_feats, self.final_ln(final_feats).flatten()
    
    
    def metrics(self, feats, outputs, gt):
        np_feats = feats.mean(-1).detach().cpu().numpy()
        np_outputs = outputs.detach().cpu().numpy()
        np_gt = gt.numpy()
        return spearmanr(np_feats, np_gt)[0], spearmanr(np_outputs, np_gt)[0]
        
def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

def max_plcc_loss(y_pred, y):
    return sum(plcc_loss(y_pred[:,i], y) for i in range(y_pred.shape[-1])) / y_pred.shape[-1]

def rescale(x):
    x = np.array(x)
    print("Mean:", x.mean(), "Std", x.std())
    x = (x - x.mean()) / x.std()
    return 1 / (1 + np.exp(-x))

def count_parameters(model):
    for name, module in model.named_children():
        print(name, "|", sum(p.numel() for p in module.parameters() if p.requires_grad))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def encode_text_prompts(prompts):
    text_tokens = tokenizer(prompts).to("cuda")
    with torch.no_grad():
        embedding = model.token_embedding(text_tokens)
        text_features = model.encode_text(text_tokens).float()
    return text_tokens, embedding, text_features



if __name__ == "__main__":
    

    parser = argparse.ArgumentParser(description='Hyper-parameters')
    parser.add_argument('--n_pairs', type=int, default=2, help='Number of pairs')
    parser.add_argument("-i", '--implicit', action="store_true", help='Use implicit prompts')
    parser.add_argument('-c', '--n_ctx', type=int, default=1, help='Number of context')


    args = parser.parse_args()

    n_pairs = args.n_pairs
    implicit = args.implicit
    n_ctx = args.n_ctx
    
    with open("buona_vista_sa_index.yml", "r") as f:
        opt = yaml.safe_load(f)  
    
    val_datasets = {}
    for name, dataset in opt["data"].items():
        val_datasets[name] = getattr(datasets, dataset["type"])(dataset["args"])

    print("Loading model")
    model, _, preprocess = open_clip.create_model_and_transforms("RN50",pretrained="openai")
    model = model.to("cuda")
    tokenizer = open_clip.get_tokenizer("RN50")
    
    
    print("Loading features")

    results = {}
    
    gts, paths = {}, {}

    for val_name, val_dataset in val_datasets.items():
        gts[val_name] = [val_dataset.video_infos[i]["label"] for i in range(len(val_dataset))]

    for val_name, val_dataset in val_datasets.items():
        paths[val_name] = [val_dataset.video_infos[i]["filename"] for i in range(len(val_dataset))]
        
    if not glob.glob("CLIP_vis_features.pt"):
        visual_features = get_features()
    visual_features = torch.load("CLIP_vis_features.pt")
    
    
    backend = "Matlab" # Matlab | Pytorch

    if backend == "Matlab":
        with open("naturalnesses_matlab_results.pkl","rb") as f:
            matlab_results = pkl.load(f)
            sn = matlab_results["spatial"]
            tn2 = matlab_results["temporal"]

    else:
        sn, tn2 = {}, {}
        for val_name in visual_features:
            with open(f"spatial_naturalness_{val_name}.pkl","rb") as infile:
                sn[val_name] = pkl.load(infile)["pr_labels"]

            with open("temporal_naturalness_pubs.pkl","rb") as infile:
                tn = pkl.load(infile)

            tn2[val_name] = tn[f"{val_name}"]["tn_index"]
            

    context = " ".join(["X"] * n_ctx)
    prompts = [
            f"a {context} high quality photo",
            f"a {context} low quality photo",
            f"a {context} good photo",
            f"a {context} bad photo",
    ]
    
    

    print(n_pairs, implicit)
    
    text_encoder = TextEncoder(model)
    
    print(f'The model has {count_parameters(model):,} trainable parameters')

        
    text_tokens, embedding, text_feats = encode_text_prompts(prompts)
    snames = ["val-cvd2014", "val-kv1k", "val-livevqc", "val-ytugc", ]
    
    print("Start training")
    
    for sname in snames:
        best_srccs, best_plccs = [], []
        cross_snames = [] #name for name in snames if name != sname]
        best_srccs_cross, best_plccs_cross = {}, {}
        for cname in cross_snames:
            best_srccs_cross[cname], best_plccs_cross[cname] = [], []
        
        for split in range(10):
            bvqi = BVQI(text_tokens, embedding, n_pairs=n_pairs, implicit=implicit).cuda()
            print(f'The model has {count_parameters(bvqi):,} trainable parameters')
            optimizer = torch.optim.AdamW(bvqi.parameters(),lr=1e-3)

            random.seed((split+1)*42)
            train_indices = random.sample(range(len(gts[sname])), int(0.8 * len(gts[sname])))
            train_dataset = VisualFeatureDataset(sname, indices=train_indices)
            train_dataloader =  torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

            val_indices = [ind for ind in range(len(gts[sname])) if ind not in train_indices]
            val_dataset = VisualFeatureDataset(sname, indices=val_indices)
            val_dataloader =  torch.utils.data.DataLoader(val_dataset, batch_size=16)
            
            
            cross_test_dataloaders = {}

            for cname in cross_snames:
                test_dataset = VisualFeatureDataset(cname)
                cross_test_dataloaders[cname] = torch.utils.data.DataLoader(test_dataset, batch_size=16)

            val_prs, val_gts = [], []
            for data in (val_dataloader):
                with torch.no_grad():
                    vis_feat, sn_ind, tn_ind, gt = data
                    _, res = bvqi(vis_feat, sn_ind, tn_ind, train=False)
                val_prs.extend(list(res.cpu().numpy()))
                val_gts.extend(list(gt.cpu().numpy()))

            print(f"Split {split}, Bef Training SRCC:", spearmanr(val_prs,val_gts)[0], "Bef Training PLCC:", pearsonr(val_prs,val_gts)[0])
            best_srcc, best_plcc = -1, -1
            srccs_cross, plccs_cross = {}, {}
            
            for epoch in tqdm(range(30)):
                #print(f"Epoch {epoch}:")
                bvqi.train()



                for data in (train_dataloader):
                    optimizer.zero_grad()
                    vis_feat, sn_ind, tn_ind, gt = data
                    feats, res = bvqi(vis_feat, sn_ind, tn_ind)
                        
                    loss = plcc_loss(res, gt.cuda().float()) #+ 0.3 * rank_loss(res, gt.cuda().float())
                    #aux_loss = max_plcc_loss(feats[...,2:], gt.cuda().float())
                    #loss += 0.3 * aux_loss
                    loss.backward()
                    optimizer.step()

                bvqi.eval()

                #val_prs, val_gts = [], []
                #for data in (train_dataloader):
                #    with torch.no_grad():
                #        vis_feat, sn_ind, tn_ind, gt = data
                #        _, res = bvqi(vis_feat, sn_ind, tn_ind)
                #    val_prs.extend(list(res.cpu().numpy()))
                #    val_gts.extend(list(gt.cpu().numpy()))

                #print("Train Spearman:", spearmanr(val_prs,val_gts)[0], "Train Pearson:", pearsonr(val_prs,val_gts)[0])

                val_prs, val_gts = [], []
                for data in (val_dataloader):
                    with torch.no_grad():
                        vis_feat, sn_ind, tn_ind, gt = data
                        _, res = bvqi(vis_feat, sn_ind, tn_ind, train=False)
                    val_prs.extend(list(res.cpu().numpy()))
                    val_gts.extend(list(gt.cpu().numpy()))

                srcc, plcc = spearmanr(val_prs,val_gts)[0], pearsonr(val_prs,val_gts)[0]

                if srcc + plcc > best_srcc + best_plcc:
                    best_srcc = srcc
                    best_plcc = plcc
                    
                    test_prs, test_gts = {}, {}
                    for cname, test_dataloader in cross_test_dataloaders.items():
                        test_prs[cname], test_gts[cname] = [], []
                        for data in (test_dataloader):
                            with torch.no_grad():
                                vis_feat, sn_ind, tn_ind, gt = data
                                _, res = bvqi(vis_feat, sn_ind, tn_ind, train=False)
                            test_prs[cname].extend(list(res.cpu().numpy()))
                            test_gts[cname].extend(list(gt.cpu().numpy()))

                        csrcc, cplcc = spearmanr(test_prs[cname],test_gts[cname])[0], pearsonr(test_prs[cname],test_gts[cname])[0]
                
                        srccs_cross[cname] = csrcc
                        plccs_cross[cname] = cplcc
                #print("Val Spearman:", srcc, "Val Pearson:", plcc, "Best Spearman:", best_srcc, "Best Pearson:", best_plcc, )

            best_srccs.append(best_srcc)
            best_plccs.append(best_plcc)
            
            print("Best SRCC:", best_srcc, "Best PLCC:", best_plcc)
            
            for cname in cross_snames:
                print(f"{cname} SRCC:", srccs_cross[cname], f"{cname} PLCC:", plccs_cross[cname])
                
                best_srccs_cross[cname] += [srccs_cross[cname]]
                best_plccs_cross[cname] += [plccs_cross[cname]]
            
                
        print(f"After training in 10 splits with seeds {[(i+1)*42 for i in range(10)]}:")    
        print(sname, "Avg Best SRCC:", np.mean(best_srccs), "Avg Best PLCC:", np.mean(best_plccs))
        print(f"Cross dataset performance:")  
        print("Cross SRCC", [(key, np.mean(values)) for key, values in best_srccs_cross.items()])
        print("Cross PLCC", [(key, np.mean(values)) for key, values in best_plccs_cross.items()])
             
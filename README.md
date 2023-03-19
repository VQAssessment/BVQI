# BVQI (Zero-shot Blind Video Quality Index)

*Pytorch-accelerated Codebase.* 
The Official Repository for BUONA-VISTA, a robust zero-shot Video Quality Index. Accepted by ICME2023.


## Introduction

In this work, we introduce an explicit semantic affinity index for opinion-unaware Video Quality Assessment (a.k.a. Zero-shot VQA) using text-prompts in the contrastive language-image pre-training (CLIP) model. We also aggregate it with different traditional low-level naturalness indexes through gaussian normalization and sigmoid rescaling strategies. Composed of aggregated semantic and technical metrics, the proposed Blind Unified Opinion-Unaware Video Quality Index via Semantic and Technical Metric Aggregation (BUONA-VISTA) outperforms existing opinion-unaware VQA methods significantly by **at least 20% improvements**, and is more robust than opinion-aware approaches. Extensive studies have validated the effectiveness of each part of the proposed BUONA-VISTA quality index.

![](figs/buona_vista.png)

## Installation

### Install OpenCLIP

To make local semantic affinity index available, the OpenCLIP should be installed as follows (or equivalently):

```
git clone https://github.com/mlfoundations/open_clip.git
cd open_clip
sed -i '92s/return x\[0\]/return x/' src/open_clip/modified_resnet.py 
pip install -e .
```

### Install BVQI

Then you need to install the BVQI codebase.

```
cd ..
git clone https://github.com/VQAssessment/BVQI.git
cd BVQI
pip install -e .
```


## Usage

Extract Semantic Affinity Index:

```
python semantic_affinity.py
```

If you would like to use the local semantic affinity index, please add `-l` after the command.
The results will be **improved** as follows:

|       | KoNViD-1k | CVD2014 | LIVE-VQC | YouTube-UGC (SA-index-only) |
| ----  |    ----   |   ---- |  ----   |   ---- |
| SROCC | 0.772 (0.760 for global, +1.6%) | 0.746 (0.740 for global, +0.8%) | 0.794 (0.784 for global, +1.4%) | 0.603 (0.585 for global, +3.0%)|
| PLCC  | 0.772 (0.760 for global, +1.6%) | 0.768 (0.763 for global, +0.7%) | 0.803 (0.794 for global, +1.1%) | 0.612 (0.606 for global, +1.0%)|

In the next version, we will add the visualization tool to visualize the **local quality maps** from perspective of all three indices.

Extract Spatial Naturalness Index:

```
python spatial_naturalness.py
```

Extract Temporal Naturalness Index:

```
python temporal_naturalness.py
```


Evaluate the Aggregated Results

See *combine.ipynb*


## Note: Possible Performance Drop while Totally Using this Codebase

The Code for Temporal Naturalness Index is slightly different from the original version (with only V1 curvature), therefore we might experience some performance drop. We will try to include the code for LGN curvature computation in the following versions. **Still, we provided the `naturalnesses_matlab_results.pkl` to assist you reproduce our results with MatLab-obtained SN and TN indexes.**

Here shows performance of the Codebase (Performance of Original Paper with MatLab Code):

|       | KoNViD-1k | CVD2014 | LIVE-VQC | 
| ----  |    ----   |   ---- |  ----   |   
| SROCC | 0.758 (0.760 in paper) | 0.683 (0.740 in paper) | 0.772 (0.784 in paper) | 
| PLCC  | 0.755 (0.760 in paper) | 0.708 (0.763 in paper) | 0.784 (0.794 in paper) |

*With GPU, the acceleration is around 10x compared with original MatLab version. *


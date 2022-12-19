# BUONA-VISTA (Zero-shot VQA)

The Official Repository for BUONA-VISTA, a robust zero-shot Video Quality Index.
Incomplete Code Right Now.

## Introduction

In this work, we introduce an explicit semantic affinity index for opinion-unaware Video Quality Assessment (a.k.a. Zero-shot VQA) using text-prompts in the contrastive language-image pre-training (CLIP) model. We also aggregate it with different traditional low-level naturalness indexes through gaussian normalization and sigmoid rescaling strategies. Composed of aggregated semantic and technical metrics, the proposed Blind Unified Opinion-Unaware Video Quality Index via Semantic and Technical Metric Aggregation (BUONA-VISTA) outperforms existing opinion-unaware VQA methods significantly by **at least 20% improvements**, and is more robust than opinion-aware approaches. Extensive studies have validated the effectiveness of each part of the proposed BUONA-VISTA quality index.

![](figs/buona_vista.png)

## To-Do

- [ ] Pytorch Code for Temporal Naturalness Index ([TPQI](https://github.com/UOLMM/TPQI-VQA))
- [x] ~~Pytorch Code for Spatial Naturalness Index (re-aligned NIQE)~~
- [x] ~~Pytorch Code for Semantic Affinity Index (CLIP-based)~~

## Usage

Extract Semantic Affinity Index:

```
python semantic_affinity.py
```

Extract Spatial Naturalness Index:

```
python spatial_naturalness.py
```


*(To-Do) Extract Temporal Naturalness Index*

*(To-Do) Evaluate the Aggregated Results*

## Results

![](figs/results.png)


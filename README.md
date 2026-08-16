<div align="center">

# VCP-CLIP+

### Stabilizing and Optimizing VCP-CLIP with Minimal Architectural Changes

Official PyTorch implementation of
**"VCP-CLIP+: Stabilizing and Optimizing VCP-CLIP with Minimal Architectural Changes"**

**Junhyeok Im · Hanhoon Park**

*Electronics*, 2026, 15, 2058

[![Paper](https://img.shields.io/badge/Paper-Electronics-2F7D32?style=for-the-badge)](https://doi.org/10.3390/electronics15102058)
[![DOI](https://img.shields.io/badge/DOI-10.3390%2Felectronics15102058-blue?style=for-the-badge)](https://doi.org/10.3390/electronics15102058)
[![PyTorch](https://img.shields.io/badge/PyTorch-Implementation-EE4C2C?style=for-the-badge\&logo=pytorch\&logoColor=white)](https://pytorch.org/)

</div>

---

## Overview

**VCP-CLIP+** is an improved version of **VCP-CLIP** for **Zero-Shot Anomaly Segmentation (ZSAS)**.

Instead of introducing additional heavy architectural components, VCP-CLIP+ revisits the original VCP-CLIP framework and focuses on improving its **training stability**, **optimization consistency**, and **visual conditioning** through four simple yet effective modifications.

| Modification                                 | Description                                                                                                                    |
| :------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- |
| **Fixed Temperature Scaling**                | Replaces the learnable temperature parameters with a fixed value for more stable similarity estimation and optimization.       |
| **Unified Anomaly Map Optimization (UniOp)** | Directly optimizes the fused anomaly map with a learnable fusion weight, reducing the mismatch between training and inference. |
| **Loss Rebalancing (LoReb)**                 | Adaptively balances Focal and Dice losses through a learnable weighting mechanism.                                             |
| **Image-Conditioned Direct Prompting (IDP)** | Directly incorporates global image information into the text prompt using image-conditioned visual tokens.                     |

These modifications preserve the overall VCP-CLIP framework while introducing **negligible additional computational overhead**.

For detailed experimental results, ablation studies, and analyses, please refer to our paper.

---

## Abstract

Zero-shot anomaly segmentation (ZSAS) has significantly advanced with the emergence of vision–language models such as CLIP. Among recent approaches for ZSAS, VCP-CLIP introduced visual context prompting (VCP) and demonstrated impressive zero-shot localization capability without class-specific training. However, we revisit VCP-CLIP and find room for supplementation and improvement in the VCP-CLIP framework. In this study, we upgrade VCP-CLIP with simple yet effective modifications designed to enhance pixel-level localization and image-level reliability. Specifically, we propose: (1) a fixed temperature scaling scheme that improves consistency in similarity estimation and stability in training; (2) a learnable anomaly map fusion scheme that adaptively and optimally aggregates anomaly cues from complementary branches; (3) an adaptive loss weighting mechanism that balances segmentation objectives; and (4) an image-conditioned direct prompting module that directly injects visual context information to the text prompts. With minimal architectural changes, our upgraded model, dubbed VCP-CLIP+, achieved high performance improvements over VCP-CLIP on the ZSAS benchmark datasets, outperforming other state-of-the-art CLIP-based ZSAS methods in both pixel-level and image-level anomaly detection.

---

## Qualitative Results

The proposed modifications progressively improve anomaly localization, producing more coherent and spatially complete anomaly maps across diverse object and defect types.

<img width="1007" height="1051" alt="image" src="https://github.com/user-attachments/assets/66c982ee-6547-4336-9879-6d5b85860ff2" />

For detailed quantitative results, ablation studies, and further analyses, please refer to the [paper](https://doi.org/10.3390/electronics15102058).

---

## Prerequisites 🛠️

### Installation

#### 1. Clone this repository

```bash
git clone https://github.com/augustinus2000/VCP-CLIP-plus.git
cd VCP-CLIP-plus
```

#### 2. Create a Conda environment

```bash
conda create -n vcp_plus python=3.9 -y
conda activate vcp_plus
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Data Preparation

VCP-CLIP+ follows the dataset structure and preprocessing pipeline of the original VCP-CLIP implementation.

### MVTec-AD and VisA

#### 1. Download the datasets

Download the original datasets:

* [MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)
* [VisA](https://github.com/amazon-science/spot-diff)

The datasets can be downloaded to any desired location before preprocessing.

#### Original MVTec-AD structure

```text
path1
└── mvtec
    └── bottle
        ├── train
        │   └── good
        │       └── 000.png
        ├── test
        │   ├── good
        │   │   └── 000.png
        │   ├── anomaly1
        │   │   └── 000.png
        │   └── anomaly2
        │       └── 000.png
        └── ground_truth
            ├── anomaly1
            │   └── 000_mask.png
            └── anomaly2
                └── 000_mask.png
```

#### Original VisA structure

```text
path2
└── visa
    ├── candle
    │   └── Data
    │       ├── Images
    │       │   ├── Anomaly
    │       │   │   └── 000.JPG
    │       │   └── Normal
    │       │       └── 0000.JPG
    │       └── Masks
    │           └── Anomaly
    │               └── 000.png
    └── split_csv
        ├── 1cls.csv
        └── 1cls.xlsx
```

### 2. Standardize MVTec-AD and VisA

Run:

```bash
python dataset/make_dataset_new.py
```

This generates the standardized datasets under:

```text
./dataset/mvisa/data/mvtec
./dataset/mvisa/data/visa
```

Then generate the metadata files:

```bash
python dataset/make_meta.py
```

which produces:

```text
./dataset/mvisa/data/meta_mvtec.json
./dataset/mvisa/data/meta_visa.json
```

### Standardized Dataset Structure

```text
./dataset/mvisa/data
├── visa
│   └── candle
│       ├── train
│       │   └── good
│       │       └── visa_0000_000502.bmp
│       ├── test
│       │   ├── good
│       │   │   └── visa_0011_000934.bmp
│       │   └── anomaly
│       │       └── visa_000_001000.bmp
│       └── ground_truth
│           └── anomaly
│               └── visa_000_001000.png
│
├── mvtec
│   └── bottle
│       ├── train
│       │   └── good
│       │       └── mvtec_000000.bmp
│       ├── test
│       │   ├── good
│       │   │   └── mvtec_good_000272.bmp
│       │   └── anomaly
│       │       └── mvtec_broken_large_000209.bmp
│       └── ground_truth
│           └── anomaly
│               └── mvtec_broken_large_000209.png
│
├── meta_mvtec.json
└── meta_visa.json
```

> **Note**
>
> In addition to MVTec-AD and VisA, other anomaly detection datasets such as **BTAD** and **MPDD** can also be used for training and testing as long as their directory structures follow the same standardized format.

---

## Run Experiments 🚀

### 1. Prepare Pre-trained Weights

#### CLIP Backbone

VCP-CLIP+ uses the OpenAI CLIP **ViT-L/14@336px** backbone by default.

Create the directory:

```bash
mkdir -p pretrained_weight
```

Then download the pretrained CLIP weights:

```bash
wget https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt \
  -P ./pretrained_weight/
```

### VCP-CLIP+ Checkpoint

A pretrained VCP-CLIP+ checkpoint trained on **VisA** is available in the [Releases](https://github.com/augustinus2000/VCP-CLIP-plus/releases) section.

After downloading the checkpoint, place it at:

```text
./weights/vcpclip_plus_visa.pth
```

---

### 2. Training

To train VCP-CLIP+:

```bash
bash train.sh
```

---

### 3. Testing and Visualization

To evaluate the trained model and generate visualization results:

```bash
bash test.sh
```

---

## Citation

If you find **VCP-CLIP+** useful in your research, please consider citing our paper:

```bibtex
@article{im2026vcpclipplus,
  title     = {VCP-CLIP+: Stabilizing and Optimizing VCP-CLIP with Minimal Architectural Changes},
  author    = {Im, Junhyeok and Park, Hanhoon},
  journal   = {Electronics},
  volume    = {15},
  number    = {10},
  pages     = {2058},
  year      = {2026},
  publisher = {MDPI},
  doi       = {10.3390/electronics15102058}
}
```

VCP-CLIP+ is built upon the original **VCP-CLIP** framework and its official implementation. If you use this repository, please also consider citing the original VCP-CLIP work:

```bibtex
@article{qu2024vcpclipvisualcontextprompting,
  title         = {VCP-CLIP: A visual context prompting model for zero-shot anomaly segmentation},
  author        = {Zhen Qu and Xian Tao and Mukesh Prasad and Fei Shen and Zhengtao Zhang and Xinyi Gong and Guiguang Ding},
  year          = {2024},
  eprint        = {2407.12276},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2407.12276}
}
```

---

## Acknowledgements

This project is built upon the official implementation of **VCP-CLIP**:

**Zhen Qu, Xian Tao, Mukesh Prasad, Fei Shen, Zhengtao Zhang, Xinyi Gong, and Guiguang Ding**,
*VCP-CLIP: A Visual Context Prompting Model for Zero-Shot Anomaly Segmentation*, ECCV 2024.

Official repository:
https://github.com/xiaozhen228/VCP-CLIP

We sincerely thank the authors of VCP-CLIP for their excellent work and for making their implementation publicly available.

---

## License

This project is derived from the original VCP-CLIP implementation, which is released under the **MIT License**.

Please refer to the original VCP-CLIP repository and the licenses of the corresponding dependencies and datasets for their respective terms.

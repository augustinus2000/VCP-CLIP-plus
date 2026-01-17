# VCP-CLIP+

This repository provides the official implementation of VCP-CLIP+,  
an improved variant of VCP-CLIP that stabilizes training and optimizes performance with minimal architectural changes.

> 📌 The corresponding paper (VCP-CLIP+: Stabilizing and Optimizing VCP-CLIP with Minimal Architectural Changes) is currently **under review**.  
> Therefore, **abstract, detailed method description, and experimental results are intentionally omitted** from this repository.  
> This repo focuses on the **code**, a minimal set of **pretrained weights**, and **instructions to reproduce training & inference**.

---

## ✅ Prerequisites 🛠️

### Installation

#### 1. Clone this repository

```bash
git clone https://github.com/augustinus2000/VCP-CLIP-plus.git
cd VCP-CLIP-plus
```

#### 2. Create environment (recommended)

```bash
conda create -n vcpclip python=3.9 -y
conda activate vcpclip
```

#### 3. Install python dependencies

```bash
pip install -r requirements.txt
```

---

## Data Preparation

VCP-CLIP+ uses the same dataset structure and preprocessing steps as the original VCP-CLIP.

### MVTec-AD and VisA

#### 1. Download and prepare the original **[MVTec-AD](https://www.mvtec.com/company/research/datasets/mvtec-ad/)** and **[VisA](https://github.com/amazon-science/spot-diff)** datasets to any desired path.  
   The original dataset format is as follows:

```bash
path1
├── mvtec
    ├── bottle
        ├── train
            ├── good
                ├── 000.png
        ├── test
            ├── good
                ├── 000.png
            ├── anomaly1
                ├── 000.png
            ├── anomaly2
                ├── 000.png
        ├── ground_truth
            ├── anomaly1
                ├── 000_mask.png
            ├── anomaly2
                ├── 000_mask.png
```


```bash
path2
├── visa
    ├── candle
        ├── Data
            ├── Images
                ├── Anomaly
                    ├── 000.JPG
                ├── Normal
                    ├── 0000.JPG
            ├── Masks
                ├── Anomaly
                    ├── 000.png
    ├── split_csv
        ├── 1cls.csv
        ├── 1cls.xlsx
```

#### 2. Standardize the MVTec-AD and VisA datasets

Run the following script to generate standardized dataset folders:
```bash
python dataset/make_dataset_new.py
```

This will generate:
```bash
./dataset/mvisa/data/mvtec
./dataset/mvisa/data/visa
```

Then generate the metadata JSON files:
```bash
python dataset/make_meta.py
```

Which produces:
```bash
./dataset/mvisa/data/meta_mvtec.json
./dataset/mvisa/data/meta_visa.json
```

#### Standardized Dataset Structure
```bash
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

> **Note:**  
> In addition to MVTec-AD and VisA, other datasets such as **BTAD** and **MPDD** can also be  
> used for training and testing.  
> As long as their directory structures follow the same format, VCP-CLIP+ will run without issues.

---

## Run Experiments

### 1. Prepare the pre-trained weights

1. **Download the CLIP backbone weights**  
   VCP-CLIP+ uses CLIP models provided by OpenAI.  
   The default backbone is **ViT-L/14-336**.

   You may download CLIP weights automatically through the code,  
   or manually place them under: ./pretrained_weight/


2. **(Optional) Download our pretrained weight**  
   A pretrained VCP-CLIP+ checkpoint (trained on **VisA**) is available in the  
👉 **[Releases](https://github.com/augustinus2000/VCP-CLIP-plus/releases)** section.

This checkpoint was obtained from the version of VCP-CLIP+ that is  
**most closely aligned with the full-model configuration described in our paper**  
(currently under review).  
Although exact performance numbers are not provided here, this weight can be  
used for inference, visualization, and reproducing the qualitative behavior  
of the full VCP-CLIP+ model.

After downloading, place the file under:

./weights/vcpclip_plus_visa.pth

---

### 2. Training

```bash
bash train.sh
```

### 3. Testing and visualizing on the unseen products

```bash
bash test.sh
```
   



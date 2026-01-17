# VCP-CLIP+

This repository provides the official implementation of VCP-CLIP+,  
an improved variant of VCP-CLIP that stabilizes training and optimizes performance with minimal architectural changes.

> 📌 The corresponding paper (VCP-CLIP+: Stabilizing and Optimizing VCP-CLIP with Minimal Architectural Changes) is currently **under review**.  
> Therefore, **0--abstract, detailed method description, and experimental results are intentionally omitted** from this repository.  
> This repo focuses on the **code**, a minimal set of **pretrained weights**, and **instructions to reproduce training & inference**.

---

## ✅ Prerequisites 🛠️

### Installation

#### Clone this repository

```bash
git clone https://github.com/augustinus2000/VCP-CLIP-plus.git
cd VCP-CLIP-plus
```

---

## 🛠️ 실험 환경 설정
본 프로젝트는 Conda 가상환경 + PyTorch CUDA 12.8 환경을 기준으로 합니다. (RTX5090 GPU 사용)
VSCode 터미널 또는 일반 터미널에서 아래 명령어를 실행하세요.

### 1) Conda 가상환경 생성 및 활성화

```bash
conda create -n samexporter python=3.10 -y
conda activate samexporter
```

### 2) PyTorch + CUDA 12.8 설치

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```
* CUDA 12.8 및 cuDNN 자동 포함
* 시스템 CUDA Toolkit / cuDNN 설치 불필요
* 5090 GPU에서 안정적으로 동작

---


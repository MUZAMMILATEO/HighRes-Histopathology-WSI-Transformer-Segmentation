# 🧬 HighRes-Histopathology-WSI-Transformer-Segmentation

> **A high-resolution transformer-based framework for efficient and precise segmentation of histopathology whole-slide images (WSIs)** - packaged with full Docker reproducibility for seamless dataset preparation, training, and evaluation 🧠💻

---

## 🧪 Overview

**HighRes-Histopathology-WSI-Transformer-Segmentation** provides an end-to-end deep learning pipeline for histopathology WSI segmentation.  
It leverages **transformer-based contextual encoding** and **boundary-aware learning** to achieve high accuracy and robust generalization across tissue slides.

The project is **containerized with Docker** to ensure fully reproducible experiments from dataset preparation to model evaluation, and supports **GPU acceleration** via CUDA.

---

## 🚀 Features

✅ End-to-end pipeline for WSI segmentation  
✅ Transformer backbone with boundary & dilation-aware training  
✅ Docker-based reproducibility (no dependency headaches 🐳)  
✅ Configurable training hyperparameters and flexible dataset options  
✅ Visual sanity checks for colored mask generation  
✅ Evaluation and overlay visualization for predictions  

---

## 🧰 Prerequisites

Before proceeding, ensure the following resources are available in your environment:

- **Pretrained Backbone Weights:**
Download the pretrained backbone weights for model initialization from [this link](https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth) and place the file at:
  ```bash
  <repo_root>/
  ```

- **Dataset Location:**
Store the raw Whole Slide Images (WSIs) at:
  ```bash
  <repo_root>/
  └── datasets/
      └── raw/
          ├── Training/                 # Training WSIs or tiles
          │   ├── sample_001.png
          │   ├── sample_001_mask.png
          │   ├── sample_002.png
          │   ├── sample_002_mask.png
          │   └── ...
          ├── Validation/               # Validation set
          │   ├── slide_101.png
          │   ├── slide_101_mask.png
          │   └── ...
          └── Extra/                    # Optional: test or unseen slides
              ├── slide_201.png
              ├── slide_201_mask.png
              └── ...

  ```

- **Download Pretrained WSI Checkpoint (Optional)**
For convenience, the fully trained model weights on the prostate WSI dataset are also available for download from the following link:

👉 [Download Trained FCBFormer WSI Model (.pt)](https://drive.google.com/file/d/16UWLpIgQWkI_bCbEVdeCs9XgUkreh8us/view?usp=sharing)

---

## Option A - Using Docker

Make sure you have the following installed on your system:

- 🐋 **Docker** (≥ 20.10)
- 🧠 **NVIDIA Container Toolkit** for GPU support  
  ```bash
  sudo apt install -y nvidia-container-toolkit
  sudo systemctl restart docker
  ```

---

## 📦 Pull Docker Image

Pull the prebuilt Docker image from Docker Hub:

```bash
docker pull khanm2004/fcbformer:conda-cu111
```

Check that it’s available:
```bash
docker images | grep fcbformer
```

---

## 🧑‍💻 Quick Start

### 1️⃣ Verify PyTorch GPU Setup

Run this quick check inside the container:

```bash
docker run --rm -it --gpus all \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/outputs:/workspace/outputs \
  fcbformer:conda \
  conda run -n fcbformer python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
  ```

Expected output:
```bash
    2.3.0+cu111 True
```

### 2️⃣ Prepare Dataset for Training

```bash
docker run --rm -it \
  -u $(id -u):$(id -g) \
  -v /path/to/datasets:/workspace/datasets \
  fcbformer:conda \
  conda run -n fcbformer python Data_prep/prepare_dataset.py \
    --root /workspace/datasets \
    --out-name tiles_512_o64 \
    --patch-size 256 \
    --overlap 64
```
This will generate processed tiles and manifests for model training.

### 3️⃣ Visualize Colored Mask Previews 🎨

```bash
docker run --rm -it \
  -u $(id -u):$(id -g) \
  -v /path/to/datasets:/workspace/datasets \
  fcbformer:conda \
  conda run -n fcbformer python Data_prep/make_colored_previews.py \
    --mask-dir /workspace/datasets/processed/tiles_512_o64/train/masks \
    --limit 50
```

### 4️⃣ Sanity Check the Dataset 🩺

```bash
docker run --rm -it \
  -u $(id -u):$(id -g) \
  -v /path/to/datasets:/workspace/datasets \
  fcbformer:conda \
  conda run -n fcbformer python Data_prep/check_manifests.py \
    --manifests-dir /workspace/datasets/processed/manifests \
    --samples 5
```

---

## 🏋️‍♂️ Training the Model

Train your transformer-based segmentation model on the prepared dataset:

```bash
docker run --rm -it --gpus all \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/datasets:/home/khanm2004/FCBFormer/datasets \
  -v /path/to/outputs:/workspace/outputs \
  fcbformer:conda \
  conda run -n fcbformer python train.py \
    --dataset AIRA \
    --data-root /workspace/datasets \
    --epochs 60 \
    --batch-size 4 \
    --learning-rate 3e-4 \
    --img-size 256 \
    --boundary-weight 0.7 \
    --tumor-dil-weight 0.7 \
    --benign-dil-weight 0.7 \
    --dil-iters 5 \
    --dil-kernel 3 \
    --prob-power 1.0
```

Training logs and checkpoints are automatically saved in:
```bash
/workspace/outputs/<timestamp>/
```

---

## 📈 Model Evaluation (on validation data)

Evaluate the trained model on validation/test sets:

```bash
docker run --rm -it --gpus all \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/datasets:/home/khanm2004/FCBFormer/datasets \
  -v /path/to/outputs:/workspace/outputs \
  fcbformer:conda \
  conda run -n fcbformer python eval.py \
    --data-root /workspace/datasets \
    --checkpoint /workspace/outputs/20251004-183051/best_FCBFormer.pt \
    --img-size 256 \
    --batch-size 4 \
    --save-png \
    --out-dir /workspace/outputs/EvalWSI
```

---

## 🧩 Evaluation on Extra Slides

```bash
docker run --rm -it --gpus all \
  -v /path/to/datasets:/workspace/datasets \
  -v /path/to/datasets:/home/khanm2004/FCBFormer/datasets \
  -v /path/to/outputs:/workspace/outputs \
  fcbformer:conda \
  conda run -n fcbformer python eval.py \
    --data-root /workspace/datasets \
    --checkpoint /workspace/outputs/20251004-183051/best_FCBFormer.pt \
    --img-size 256 \
    --batch-size 4 \
    --tiles-manifest /workspace/datasets/processed/manifests/extra.csv \
    --out-dir /workspace/outputs/extra_preds \
    --overlay-alpha 0.4
```

This generates overlayed predictions for unseen WSIs.

---

## 📂 Directory Structure

```bash
FCBFormer/
├── Data_prep/                     # Dataset preparation utilities
│   ├── prepare_dataset.py
│   ├── make_colored_previews.py
│   └── check_manifests.py
├── train.py                       # Main training entry point
├── eval.py                        # Evaluation script
├── pvt_v2_b3.pth                  # Initial weights
├── datasets/
│   ├── raw/                       # Original WSIs or tiles
│   └── processed/                 # Preprocessed and tiled data
└── outputs/
    ├── <timestamp>/               # Model checkpoints and logs
    ├── EvalWSI/                   # Evaluation results on validation/test data
    └── extra_preds/               # Predictions for extra/unseen slides
...
```

---

## 🛠️ Option B - Manual Setup (Conda)

Use this if you prefer running locally without Docker. The code and flags mirror the Docker commands.

### 1) Clone the repository

```bash
git clone https://github.com/MUZAMMILATEO/HighRes-Histopathology-WSI-Transformer-Segmentation.git
cd HighRes-Histopathology-WSI-Transformer-Segmentation
```

### 2) Create environment
```bash
# From repo root
conda env create -f environment.yml
conda activate fcbformer
```

### 3) One-shot sanity check
```bash
python - <<'PY'
import torch, torchvision, timm, cv2, sklearn, skimage, tqdm, numpy as np
print("CUDA available:", torch.cuda.is_available())
print("PyTorch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("torchvision:", torchvision.__version__)
print("timm:", timm.__version__)
print("opencv:", cv2.__version__)
print("sklearn:", sklearn.__version__, "skimage:", skimage.__version__, "numpy:", np.__version__)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
```
### 4) Prepare data

Keep raw delivery intact and generate a processed set by running 

```bash
python Data_prep/prepare_dataset.py \
  --root ./datasets \
  --out-name tiles_512_o64 \
  --patch-size 256 \
  --overlap 64
```

### 5) Visualize colored mask previews & Sanity-check manifests

```bash
python Data_prep/make_colored_previews.py \
  --mask-dir ./datasets/processed/tiles_512_o64/train/masks \
  --limit 50

python Data_prep/check_manifests.py \
  --manifests-dir ./datasets/processed/manifests \
  --samples 5
```

### 6) Train

```bash
python train.py \
  --dataset AIRA \
  --data-root ./datasets \
  --epochs 60 \
  --batch-size 4 \
  --learning-rate 3e-4 \
  --img-size 256 \
  --boundary-weight 0.7 \
  --tumor-dil-weight 0.7 \
  --benign-dil-weight 0.7 \
  --dil-iters 5 \
  --dil-kernel 3 \
  --prob-power 1.0
```

Artifacts are saved in
```bash
./outputs/<timestamp>/
```

### 7) Evaluate (val/test)

```bash
python eval.py \
  --data-root ./datasets \
  --checkpoint ./outputs/<timestamp>/best_FCBFormer.pt \
  --img-size 256 \
  --batch-size 4 \
  --save-png \
  --out-dir ./outputs/EvalWSI
  ```

### 8) Evaluate on extra slides
```bash
python eval.py \
  --data-root ./datasets \
  --checkpoint ./outputs/<timestamp>/best_FCBFormer.pt \
  --img-size 256 \
  --batch-size 4 \
  --tiles-manifest ./datasets/processed/manifests/extra.csv \
  --out-dir ./outputs/extra_preds \
  --overlay-alpha 0.4
```

---

## 📖 Citation
If you use this repository or build upon it in your research, please cite:
```bash
@article{khan2025highreswsi,
  title   = {HighRes-Histopathology-WSI-Transformer-Segmentation:
             A High-Resolution Transformer-Based Framework for Efficient
             and Precise Whole-Slide Image Segmentation},
  author  = {Khan, Muzammil},
  year    = {2025},
  journal = {GitHub Repository},
  howpublished = {\url{https://github.com/khanm2004/HighRes-Histopathology-WSI-Transformer-Segmentation}}
}
```

## 🙏 Acknowledgement

This implementation builds upon the **FCBFormer** architecture from the following repository:  
🔗 [ESandML/FCBFormer](https://github.com/ESandML/FCBFormer)

We gratefully acknowledge the original authors for making their work available to the research community.




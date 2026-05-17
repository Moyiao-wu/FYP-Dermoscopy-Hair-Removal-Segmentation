# Solving Hair-Occlusion in Dermoscopy: A Three-Stage Pipeline Integrating SAM, LaMa, and UltraLight VM-UNet

## Project Overview
This repository contains the official implementation of an automated end-to-end pipeline designed to mitigate the impact of hair occlusion in dermoscopic images. By integrating foundation models with efficient state-space architectures, the system detects hair artifacts, restores underlying skin textures, and performs high-precision lesion segmentation.

## Key Contributions
- **Prompt-free Hair Detection:** Fine-tuned SAM-ViT-H with a residual adapter for automated pixel-level hair masking.
- **Semantic Restoration:** Deep skin texture reconstruction using LaMa (Fast Fourier Convolutions) to eliminate hair shadows and artifacts.
- **Efficient Segmentation:** Implementation of UltraLight VM-UNet (based on Vision Mamba) for lightweight, clinical-grade segmentation.
- **Domain Adaptation:** A specialized fine-tuning strategy that overcomes texture distribution shifts between restored and clean images.

## Performance Summary
Evaluated on 500 challenging clinical cases from the ISIC database (393 independent test samples):

Baseline (Occluded): 0.7813 Dice Score.
Proposed Pipeline: 0.8857 Dice Score (up to 0.90+ with full dataset).
Improvement: +10.4% absolute gain in Dice Score and 15.2% boost in Sensitivity.
Error Reduction: 47.7% relative reduction in total segmentation error.

## System Architecture
The pipeline consists of three sequential stages:
1. **Stage 1 (SAM-ViT-H + Adapter):** Generates a precise binary mask of hair strands using dual-input (raw image and residual feature map) processing.
2. **Stage 2 (LaMa Inpainting):** Reconstructs the occluded skin regions using the mask from Stage 1, preserving the anatomical integrity of the lesion.
3. **Stage 3 (UltraLight VM-UNet):** Performs the final lesion segmentation on the restored, hair-free image.

## Environment Setup
The project was developed and tested on **NVIDIA RTX 4090** using **PyTorch 2.0.1** and **CUDA 11.8**.  
Please set up the environment via `environment.yml` and then manually install the Mamba operator:

```bash
pip install causal_conv1d-1.1.0+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm-1.1.1+cu118torch2.0cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
```
##  Project Structure

To optimize storage management and high-speed I/O, the project separates source code from heavy data/weight assets:

```text
FYP_Project/
├── code/                       # Script-only directory (Lightweight)
│   ├── lama/                   # LaMa Inpainting source
│   ├── UltraLight-VM-UNet/     # VM-UNet source & configs
│   └── sam_adapter/            # SAM tuning logic
├── data/                       # Centralized Data Hub (Externalized)
│   ├── vmunt_train_base/       # 3,800 Clean images for training
│   ├── gold_real_d2/           # 500 Paired hair-occlusion benchmark
│   └── skin_lesion/            # High-speed .npy binaries for VM-UNet
├── weights/                    # Centralized Model Hub
│   ├── sam/                    # SAM-ViT-H & Adapter weights
│   ├── lama/                   # bigLama checkpoints
│   └── vmunet/                 # Fine-tuned segmentation weights
├── pipeline_final.py           # Automated 3-stage integration script
└── environment.yml             # Environment configuration

## Dataset Structure
Organize your data directory as follows:

```text
data/
└── gold_real_d2/
    ├── dermoscopic_image/      # Raw hair-occluded images
    ├── overlay/                # Binary hair masks / voids
    ├── lesion_mask/            # Ground truth lesion boundaries
    └── restored_image_final/   # Output from Stage 2
```
## ⚠️ Important: Dataset Availability & Upload Note

Due to the significant storage volume of medical imaging datasets and platform upload limits, the full data assets are organized as follows:

### 1. `vmunt_train_base/` (Omitted)
*   **Status:** Not uploaded to this repository.
*   **Reason:** Contains 3,800 high-resolution images (~30GB+).
*   **Action:** Users must download the original images from the [ISIC Archive](https://www.isic-archive.com/).
### 2. `gold_real_d2/` (Representative Samples)
*   **Status:** Partially uploaded.
*   **Details:** To demonstrate the required file structure and naming conventions, **one representative sample image** has been provided in each subdirectory (e.g., `dermoscopic_image/`, `lesion_mask/`). 
*   **Utility:** These samples allow the `pipeline_final.py` to be tested for code integrity without requiring the full 500-image benchmark immediately.

## Model Weights & Download Links
Due to file size limitations, only the fine-tuned adapter weights are included in this repository. The heavy foundation model backbones must be downloaded separately using the links provided below.
SAM ViT-H : (https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)
Big-LaMa : (https://huggingface.co/smartywu/big-lama/resolve/main/bigLama.pt)

After downloading, you must organize the weights/ directory exactly as shown below for the pipeline_final.py script to function:
```text
weights/
├── sam/
│   ├── sam_vit_h_4b8939.pth      <-- Downloaded from Meta AI
│   └── sam_adapter_vith_best.pth <-- (Included in this repo)
├── lama/
│   ├── bigLama.ckpt              <-- Downloaded from HuggingFace
│   └── config.yaml               <-- (Included in this repo)
└── vmunet/
    └── vmunet_finetuned_fixed_seed.pth <-- (Included in this repo)
```

## Usage
### Running the Full Pipeline
To process a single image through all three stages and generate a visualization:

```bash
python pipeline_final.py
```
The code for the final model comparison and evaluation can be found in FYP_Project/code/UltraLight-VM-UNet/Performance_Testing.ipynb.

## Data Sources
The dermoscopic images, hair masks, and lesion annotations used in this project are derived from the following publicly available resources:

- **ISIC 2017 Challenge Dataset** – [https://challenge.isic-archive.com/data/#2017](https://challenge.isic-archive.com/data/#2017)
- **ISIC 2018 Challenge Dataset** – [https://challenge.isic-archive.com/data/#2018](https://challenge.isic-archive.com/data/#2018)
- **Mendeley Data (Hair Masks & Preprocessed Annotations)** – [10.17632/j5ywpd2p27.2](https://data.mendeley.com/datasets/j5ywpd2p27/2)  

## Acknowledgments
This project builds upon the foundational work of the following research communities and authors. We express our gratitude for their open-source contributions:

- **SAM:** Meta AI Research. [Segment Anything](https://github.com/facebookresearch/segment-anything)
- **LaMa:** Samsung AI Center (Suvorov et al.). [LaMa](https://github.com/advimman/lama)
- **UltraLight VM-UNet:** R. Wu et al. (2025). [UltraLight-VM-UNet](https://github.com/wurenkai/UltraLight-VM-UNet)
- **XJTLU:** Special thanks to the Department of Biosciences & Bioinformatics at Xi'an Jiaotong-Liverpool University for providing the high-performance computing (HPC) resources required for model training.


## Generative AI Utilization Statement
The development of this project was supported by Generative AI (Large Language Models) in a collaborative capacity. Specifically, AI was utilized for:
- **Technical Troubleshooting:** Providing real-time debugging support for complex environment conflicts.
- **Documentation Refinement:** Optimizing script comments and technical documentation for improved clarity and academic rigor.

All core algorithmic decisions, experimental design, model training, and performance evaluations were conducted independently by the author.

**Author:** Yixuan Li  
**University:** Xi'an Jiaotong-Liverpool University (XJTLU)  
**Supervisor:** Dr. Shuihua Wang

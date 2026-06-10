# HistoBIT3D: Virtual 3D H&E Staining from Phase-contrast Back-illumination Interference Tomography

[![arXiv](https://img.shields.io/badge/arXiv-2605.22000-b31b1b.svg)](https://arxiv.org/abs/2605.22000)

Official code repository for:

**Anthony A. Song, Boyan Zhou, Mayank Golhar, Marisa Morakis, Alex Baras, and Nicholas J. Durr**

**Virtual 3D H&E Staining from Phase-contrast Back-illumination Interference Tomography**

---

## Overview

Three-dimensional (3D) histopathology of unprocessed tissue has the potential to transform disease diagnosis and management by enabling volumetric characterization of tissue microarchitecture without destructive sectioning. Back-illumination Interference Tomography (BIT) is a recently developed phase microscopy technique capable of rapid, label-free volumetric imaging of thick biological specimens. However, translating BIT volumes into clinically interpretable hematoxylin and eosin (H&E) images remains challenging due to shift-variant contrast and the lack of quantitative validation benchmarks.

To address these challenges, we introduce **HistoBIT3D**, the first voxel-wise paired dataset of BIT and fluorescence-labeled nuclei volumes for quantitative evaluation of virtual staining. Building on this dataset, we present a novel virtual staining framework that combines:

* **Bidirectional multiscale content consistency** for structural preservation
* **Cross-domain style reuse** for realistic H&E appearance
* **Quantitative 3D validation** using ground-truth fluorescence nuclei distributions

Our method achieves state-of-the-art realism metrics while significantly improving 3D nuclei segmentation accuracy and boundary preservation under zero-shot Cellpose evaluation.

---

## Highlights

* First voxel-wise paired 3D BIT–fluorescence nuclei dataset
* Quantitative validation of virtual staining using ground-truth nuclei distributions
* Structurally faithful virtual H&E generation from label-free phase-contrast microscopy
* Bidirectional multiscale content consistency for preserving tissue morphology
* AdaIN-based cross-domain style reuse for realistic H&E appearance
* Zero-shot Cellpose validation of structural preservation
* Slide-free volumetric histopathology without fixation, sectioning, or chemical staining

---

## Paper

**ArXiv:** https://arxiv.org/abs/2605.22000

If you find this work useful, please consider citing:

```bibtex
@article{song2026histobit3d,
  title={Virtual 3D H\&E Staining from Phase-contrast Back-illumination Interference Tomography},
  author={Song, Anthony A. and Zhou, Boyan and Golhar, Mayank and Morakis, Marisa and Baras, Alex and Durr, Nicholas J.},
  journal={arXiv preprint arXiv:2605.22000},
  year={2026}
}
```

---

# Method Overview

## HistoBIT3D Pipeline

<p align="center">
<img src="https://github.com/user-attachments/assets/e34f62c9-b357-4c5c-b3a3-9ef9ef197602" width="100%">
</p>

**Figure 1.** Overview of the HistoBIT3D pipeline. From left to right: volumetric acquisition of voxel-wise paired BIT and fluorescence nuclei data, virtual H&E generation using our GAN-based framework, fluorescence nuclei segmentation, and quantitative evaluation against ground-truth nuclei distributions.

---

## Network Architecture

<p align="center">
<img src="https://github.com/user-attachments/assets/063dc899-6893-406a-b471-39e658a7904a" width="100%">
</p>

**Figure 2.** Virtual staining architecture incorporating bidirectional multiscale content consistency for structural preservation and AdaIN-based cross-domain style injection for realistic H&E appearance.

---

## Quantitative and Qualitative Results

<p align="center">
<img src="https://github.com/user-attachments/assets/19c31836-697e-4262-8248-e7d5a6740aac" width="100%">
</p>

**Figure 3.** Comparison of virtual H&E results against baseline methods. Zero-shot Cellpose segmentation demonstrates improved preservation of nuclei structure relative to fluorescence ground truth. Additional examples show diverse tissue samples from the HistoBIT3D dataset virtually stained from BIT into H&E.

---

# 3D Virtual Histopathology Examples

The examples below demonstrate volumetric virtual staining of fresh human tissue acquired using our multimodal BIT microscope. These animations highlight the ability of HistoBIT3D to generate spatially consistent virtual H&E volumes while preserving tissue microarchitecture across depth.

## Human Duodenum: BIT → Virtual H&E

Fresh human duodenum specimen obtained during Whipple surgery. Crypt structures remain visible throughout the volume after virtual staining.

<table>
<tr>
<td align="center"><b>BIT Input Volume</b></td>
<td align="center"><b>Virtual H&E Volume</b></td>
</tr>
<tr>
<td>
<img src="https://github.com/user-attachments/assets/4ce83eaf-b8b2-43b0-8dec-cf12252ea2c8" width="100%">
</td>
<td>
<img src="https://github.com/user-attachments/assets/95739260-1845-47b8-ac83-529f14f7abb5" width="100%">
</td>
</tr>
</table>

**Volume dimensions:** ~302 × 211 × 15 µm³
**Axial spacing:** 0.5 µm
**Display:** 4× downsampled

---

## Virtual H&E Volume Rendering

A volumetric rendering generated from the virtually stained tissue volume.

<p align="center">
<img src="https://github.com/user-attachments/assets/a6f0a985-bc56-4ae9-b814-065953683e3c" width="70%">
</p>

**Volume dimensions:** ~302 × 211 × 15 µm³
**Axial spacing:** 0.5 µm
**Display:** 4× downsampled

These examples illustrate the potential of HistoBIT3D for slide-free volumetric histopathology, enabling visualization of tissue architecture in three dimensions without fixation, sectioning, or chemical staining.

---

# Installation

## 1. Create Environment

We recommend creating a dedicated Conda environment based on UVCGANv2.

```bash
conda create -n uvcgan2 python=3.10
conda activate uvcgan2
```

Install all required dependencies before proceeding.

---

# Dataset Preparation

Organize the data using the following directory structure:

```text
DATA_ROOT/
├── BIT/
│   ├── trainA/
│   └── testA/
└── FFPE_HE/
    ├── trainB/
    └── testB/
```

Where:

* `BIT/trainA` contains training BIT images
* `BIT/testA` contains testing BIT images
* `FFPE_HE/trainB` contains training H&E images
* `FFPE_HE/testB` contains testing H&E images

---

# Training

Example scripts are located in:

```text
HistoBIT3D_Code/scripts/example_training/
```

## Step 1: Pretraining

Pretrain the model using:

```bash
python pretrain.py
```

Update dataset paths within the script before launching training.

---

## Step 2: Full Training

Launch training with:

```bash
bash run_train_example.sh
```

Update the training data path inside the script to match your local dataset location.

### Using a Pretrained Checkpoint

To initialize from a pretrained model, edit:

```text
train_3D_embedding_style_content_TM.py
```

and set:

```python
pretrain_root = "/path/to/pretrained/model.ckpt"
```

---

## Default Training Configuration

The default model includes:

* UVCGANv2 backbone
* AdaIN-based style fusion
* Cross-domain style reuse
* 8-channel multiscale content consistency loss
* ViT-based latent representation

---

# Dataset Availability

The HistoBIT3D multimodal microscopy dataset is currently being prepared for public release. We are actively expanding the dataset with additional tissue types, tumor subtypes, and pathological annotations, and will update this repository with download instructions upon release.

For early access inquiries, please contact the authors.

---

# Acknowledgements

This work was supported by the Johns Hopkins Computational Biophotonics Laboratory and collaborators in pathology, urology, and biomedical engineering.

---

# License

Code and data will be released under an open-source license upon publication.

# 3D UVC FlowGAN: Unsupervised Virtual H&E Staining with Spatially Consistent Flow Loss

**Unsupervised 3D Virtual H&E staining using z-axis flow and gradient loss for spatial consistency.**  
This method translates 3D Back-illumination Interference Tomography (BIT) image volumes to virtually stained H&E without the need for 3D histology—only 2D target FFPE H&E slices are required.

---

### 🔬 Original BIT (Left) → Virtual H&E (Right)

**Fresh human duodenum from Whipple surgery**, imaged with our multimodal BIT microscope. Specific structures are crypts. Downsampled 4X, 1 micron z-spacing

<p float="left">
  <img src="https://github.com/user-attachments/assets/4ce83eaf-b8b2-43b0-8dec-cf12252ea2c8" width="45%"/>
  <img src="https://github.com/user-attachments/assets/95739260-1845-47b8-ac83-529f14f7abb5" width="45%"/>
</p>

**BIT Input Volume**: ~302×211×15 µm³  
**Virtual H&E Output**: ~302×211×15 µm³

---

### 🔄 3D Virtual Volume Slices
![slice_rendering](https://github.com/user-attachments/assets/a6f0a985-bc56-4ae9-b814-065953683e3c)


~302×211×15 µm³,0.5 micron z-spacing, Downsampled 4X
---

### 🧠 3D Volume Renderings

<p float="left">
  <img src="https://github.com/user-attachments/assets/7cc56886-e8bc-4561-b022-08afe8cfc32c" width="45%" />
  <img src="https://github.com/user-attachments/assets/1b78d323-4936-4aec-93d1-e8eb557f6fe1" width="45%" />
</p>

---

### 🧾 BIT (top row), BIT-to-virtual H&E (middle row), Ground Truth FFPE H&E Patches for Human Duodenum Tissue (bottom row)
<p float="center">
<img width="512" height="311" alt="Figure_BIT_BIT2vHE_FFPE-HE" src="https://github.com/user-attachments/assets/6398e492-af05-476c-a766-cdb68c1b47f1" />
</p>
---

### 📜 References

- **Proceedings**: [NTM 2025 - NTh1C.3](https://opg.optica.org/abstract.cfm?URI=NTM-2025-NTh1C.3)  
- PDF: `ntm-2025-nth1c.3.pdf` (see repository)
- **Original architecture inspiration**: [UVCGANv2: Rethinking CycleGAN for Scientific Image Translation][uvcgan2_paper]
- **Conference Submission in Preparation**

---

## 🧠 Overview

`3D UVC FlowGAN` builds on [UVCGANv2](https://github.com/uvcgan/uvcgan2) to enable **unpaired 3D→2D virtual staining** using:

- Enhanced **3D generators** with volumetric convolution
- **Flow and gradient consistency loss** across z-slices
- Only 3D *source domain* (e.g., BIT) required
- Only 2D *target domain* (e.g., FFPE H&E) required

---

## 📁 Dataset Format

Your dataset should follow the format:

```bash
BIT/              # Input 3D volumes
  trainA/
  testA/

FFPE_HE/          # Target 2D H&E images
  trainB/
  testB/

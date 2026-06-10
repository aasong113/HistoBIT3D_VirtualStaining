# HistoBIT3D_VirtualStaining Training Guide
## 1. Environment Setup

Create and activate a UVCGANv2 Conda environment following the installation instructions provided by the UVCGANv2 repository.

```bash
conda create -n uvcgan2 python=3.10
conda activate uvcgan2
```

Install all required dependencies before proceeding.

---

## 2. Dataset Structure

Organize the training data using the following directory structure:

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

* `BIT/trainA` contains training BIT images.
* `BIT/testA` contains testing BIT images.
* `FFPE_HE/trainB` contains training H&E images.
* `FFPE_HE/testB` contains testing H&E images.

---

## 3. Pretraining

Example training scripts are provided in:

```text
HistoBIT3D_Code/scripts/example_training/
```

Before training the full model, pretrain the network using:

```bash
python pretrain.py
```

Update the dataset paths in the script configuration so that they point to your dataset root directory.

---

## 4. Training

Launch training using:

```bash
bash run_train_example.sh
```

Before running, update the training data directory in the script to match your dataset location.

### Using a Pretrained Model

To initialize training from a pretrained checkpoint:

1. Open:

```text
train_3D_embedding_style_content_TM.py
```

2. Set the pretrained checkpoint path:

```python
pretrain_root = "/path/to/pretrained/model.ckpt"
```

### Default Training Configuration

The default model configuration includes:

* 8-channel multiscale content consistency loss
* AdaIN-based style fusion
* Cross-domain style reuse
* UVCGANv2 backbone

---

## 5. Data Availability

We are currently working on making the multimodal microscopy training dataset used in this work public and will update it with a link to the data soon. Thanks for your patience.

# MVR—DTI: A Multimodal Molecular Visual Representation Learning for Drug-Target Interaction Prediction

## Overview

MVR (Multi-modal Visual Representation) is a deep learning framework for Drug-Target Interaction (DTI) prediction. It integrates multiple modalities to represent molecules and proteins.

---



## Large Files (hosted on Zenodo)

This repository does not include large pretrained model weights and datasets.
Please download them manually and place them into the corresponding directories as follows:

1️⃣ From Zenodo

Download all required files from:
👉 https://zenodo.org/records/19641106

After downloading, place the files into:

molformer/ → IBM MolFormer pretrained weights
vit_model/ → ViT pretrained weights for molecular frame feature extraction
data/ → DTI dataset files
2️⃣ Protein Language Model (ProtT5)

Download ProtT5-XL-UniRef50 from Hugging Face:

👉 https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main

Place the downloaded files into:

prot_t5_xl_uniref50/ → ProtT5-XL-UniRef50 pretrained model weights
📌 Notes
Make sure directory names match exactly as above.
These models are required for running inference and training.

---

## Installation

Install dependencies (all required packages are specified in requirements.txt)
requirements.txt

---

## Usage

### Step 1: Generate molecular video data

Run from the `video/` directory:

```bash
cd video

# Generate 3D conformers from SMILES (RDKit + MMFF optimization)
python 7.py

# Render rotating molecular video frames (PyMOL → PNG sequences)
python 8.py
```

This produces `video/comformer/train/` (SDF files) and `video/video-data/train/` (PNG frames per molecule).

### Step 2: Extract precomputed protein PLM features

```bash
cd code
python extract_unique_features.py
```

Extracts unique protein embeddings using ProtT5-XL (or ESM-2) and saves them to `../preprocessed_features/T5-XL/unique_protein_features.npy`.

### Step 3: Extract precomputed molecule ViT features

```bash
cd code
python extract_unique_vit_features.py
```

Extracts ViT features for all unique molecules from their video frames and saves them to `../preprocessed_features/unique_vit_features.pt`.

### Step 4: Train the model

```bash
cd code
python main.py
```

Results are saved under:

```
output/result/{dataset}/{split}/seed_{seed}/
    ├── best_model_epoch_X.pth
    ├── result_metrics.pt
    ├── val_metrics_per_epoch.csv
    ├── visualization.csv
    ├── valid_markdowntable.txt
    ├── test_markdowntable.txt
    └── train_markdowntable.txt
```

---

## Cross-dataset Experiments

For generalization experiments on BindingDB and PDBbind datasets:

```bash
# BindingDB (binary classification)
cd cross/bind/code
python main.py

# PDBbind (binding affinity regression)
cd cross/PDB/code
python main.py
```

---



# MVR: Multi-modal Visual Representation for Drug-Target Interaction Prediction

## Overview

MVR (Multi-modal Visual Representation) is a deep learning framework for Drug-Target Interaction (DTI) prediction. It integrates multiple modalities to represent molecules and proteins:

- **MolFormer** — SMILES-based molecular tokenization
- **ViT (Vision Transformer)** — 3D rotating molecular video frames rendered with PyMOL
- **ProtT5-XL** — Protein language model embeddings
- **Knowledge Graph embeddings** — Entity-level KG representations for drugs and proteins
- **BiIntention cross-modal fusion** — Bidirectional cross-attention between drug and protein features



---

## Repository Structure

```
MVR/
├── code/                        # Main experiment (primary dataset)
│   ├── main.py                  # Entry point: 5-seed training loop
│   ├── trainer.py               # Training / evaluation logic
│   ├── models.py                # MVR model architecture
│   ├── dataloader.py            # DTIDataset
│   ├── configs.py               # Hyperparameter defaults (YACS)
│   ├── utils.py                 # Utilities (seed, collate, mkdir)
│   ├── ACmix.py                 # ACmix attention-conv protein encoder
│   ├── Intention.py             # BiIntention cross-modal fusion module
│   ├── extract_unique_features.py     # Offline protein PLM feature extraction
│   └── extract_unique_vit_features.py # Offline molecule ViT feature extraction
├── video/
│   ├── 7.py                     # Step 1: Generate 3D conformers (RDKit → SDF)
│   └── 8.py                     # Step 2: Render molecular video frames (PyMOL → PNG)
├── cross/
│   ├── bind/                    # Cross-dataset experiment: BindingDB
│   │   ├── code/                # Same structure as code/
│   │   ├── data/                # df_train_cleaned.csv, df_test_cleaned.csv
│   │   └── video/               # 7.py, 8.py for BindingDB molecules
│   └── PDB/                     # Cross-dataset experiment: PDBbind (regression)
│       ├── code/
│       ├── data/
│       └── video/
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Large Files (hosted on Zenodo)

The following directories contain large pretrained model weights and datasets that are **not included in this repository**. Please download them from Zenodo and place them at the paths shown below:

| Directory | Contents | Notes |
|-----------|----------|-------|
| `molformer/` | IBM MolFormer pretrained weights | SMILES tokenizer + encoder |
| `prot_t5_xl_uniref50/` | ProtT5-XL-UniRef50 model weights | Default protein language model https://huggingface.co/Rostlab/prot_t5_xl_uniref50/tree/main | 
| `vit_model/` | ViT pretrained weights | Used for molecular frame feature extraction |
| `data/` | DTI dataset files  | Primary dataset |

> **Zenodo DOI:** `[TODO: https://zenodo.org/records/19641106]`


---

## Installation

```bash
# 1. Create conda environment (Python 3.8+)
conda create -n mvr python=3.8
conda activate mvr

# 2. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 3. Install DGL (adjust CUDA version as needed)
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html

# 4. Install remaining dependencies
pip install -r requirements.txt

# 5. Install PyMOL (for molecular video rendering)
# Option A: via conda
conda install -c conda-forge pymol-open-source
# Option B: commercial PyMOL — follow https://pymol.org/
```

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



## Requirements

See `requirements.txt`. Core dependencies:

- Python ≥ 3.8
- PyTorch ≥ 1.12
- DGL + dgllife
- HuggingFace Transformers
- RDKit
- PyMOL (for video generation only)

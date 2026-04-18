import torch
import os
from tqdm import tqdm
from transformers import ViTModel
from torchvision import transforms
from PIL import Image
import pandas as pd
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def extract_unique_vit_features():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    print("="*80)
    print("Extracting unique molecule ViT features")
    print("="*80)

    print("\nLoading ViT model...")
    vit_model = ViTModel.from_pretrained('../vit_model').to(device)
    vit_model.eval()
    for param in vit_model.parameters():
        param.requires_grad = False
    print("ViT model loaded\n")

    print("Loading datasets...")
    train = pd.read_csv('../data/train.txt', sep='\t')
    test = pd.read_csv('../data/test.txt', sep='\t')
    val = pd.read_csv('../data/val.txt', sep='\t')

    with open('../data/gene_info_dics.pkl', 'rb') as fp:
        protein_info = pickle.load(fp)
    with open('../data/drug_info_dics.pkl', 'rb') as fp:
        drug_info = pickle.load(fp)

    def preprocess_df(df):
        df['head'] = df['head'].str.split('::').str[1]
        df['SMILES'] = df['head'].apply(lambda x: drug_info.get(x, {}).get('SMILES'))
        df['tail'] = df['tail'].str.split('::').str[1]
        df['Sequence'] = df['tail'].apply(lambda x: protein_info.get(x, {}).get('Sequence'))
        df = df.dropna(subset=['SMILES', 'Sequence'])
        df = df[['SMILES', 'Sequence', 'Label', 'head', 'tail', 'UP', 'relation']]
        df['head'] = df['head'].apply(lambda x: f"Compound::{x}")
        df['tail'] = df['tail'].apply(lambda x: f"Gene::{x}")
        df.columns = ['SMILES', 'Protein', 'Y', 'head', 'tail', 'UP', 'relation']
        return df.reset_index(drop=True)

    df_train = preprocess_df(train)
    df_test = preprocess_df(test)
    df_val = preprocess_df(val)

    df_train1 = pd.read_csv('../data/df_train.csv', sep=',')
    df_test1 = pd.read_csv('../data/df_test.csv', sep=',')
    df_val1 = pd.read_csv('../data/df_val.csv', sep=',')

    common_smiles = set(df_train.iloc[:, 0]).intersection(set(df_train1.iloc[:, 0]))
    df_train = df_train[df_train.iloc[:, 0].isin(common_smiles)].reset_index(drop=True)

    common_smiles1 = set(df_test.iloc[:, 0]).intersection(set(df_test1.iloc[:, 0]))
    df_test = df_test[df_test.iloc[:, 0].isin(common_smiles1)].reset_index(drop=True)

    common_smiles2 = set(df_val.iloc[:, 0]).intersection(set(df_val1.iloc[:, 0]))
    df_val = df_val[df_val.iloc[:, 0].isin(common_smiles2)].reset_index(drop=True)

    print(f"Train samples: {len(df_train):,}")
    print(f"Val samples: {len(df_val):,}")
    print(f"Test samples: {len(df_test):,}")
    total_samples = len(df_train) + len(df_val) + len(df_test)
    print(f"Total samples: {total_samples:,}\n")

    print("="*80)
    print("Step 1: Build global unique molecule list")
    print("="*80)

    all_smiles_series = pd.concat([df_train['SMILES'], df_val['SMILES'], df_test['SMILES']])
    unique_smiles = all_smiles_series.unique().tolist()
    smiles_to_idx = {smiles: idx for idx, smiles in enumerate(unique_smiles)}

    print(f"Unique molecules: {len(unique_smiles):,}")
    print(f"Total samples: {len(all_smiles_series):,}")
    print(f"Redundancy: {(1 - len(unique_smiles) / len(all_smiles_series)) * 100:.1f}%")

    smiles_to_sample_info = {}

    for idx, row in df_train.iterrows():
        smiles = row['SMILES']
        if smiles not in smiles_to_sample_info:
            smiles_to_sample_info[smiles] = {
                'dataset': 'train',
                'index': idx,
                'root': r"../video/video-data/train"
            }

    for idx, row in df_val.iterrows():
        smiles = row['SMILES']
        if smiles not in smiles_to_sample_info:
            smiles_to_sample_info[smiles] = {
                'dataset': 'val',
                'index': idx,
                'root': r"../video/video-data/val"
            }

    for idx, row in df_test.iterrows():
        smiles = row['SMILES']
        if smiles not in smiles_to_sample_info:
            smiles_to_sample_info[smiles] = {
                'dataset': 'test',
                'index': idx,
                'root': r"../video/video-data/test"
            }

    print(f"SMILES-to-sample mapping built\n")

    print("="*80)
    print("Step 2: Extract unique molecule ViT features")
    print("="*80)

    mol_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    num_frames = 60
    vit_dim = 768
    num_unique = len(unique_smiles)

    bytes_per_feature = num_frames * vit_dim * 4
    total_size_gb = (num_unique * bytes_per_feature) / (1024**3)
    print(f"\nEstimated output size: {total_size_gb:.2f} GB (float32)")
    print(f"Savings vs. per-sample: {(1 - num_unique / total_samples) * 100:.1f}%\n")

    all_features = []

    print(f"Extracting features for {num_unique:,} unique molecules...")
    for smiles in tqdm(unique_smiles, desc="Extracting features"):
        info = smiles_to_sample_info[smiles]
        sample_dir = os.path.join(info['root'], str(info['index']))

        try:
            frame_files = sorted([os.path.join(sample_dir, f)
                                  for f in os.listdir(sample_dir)
                                  if f.endswith(".png")])[:num_frames]

            frames = []
            for path in frame_files:
                img = Image.open(path).convert("RGB")
                img = mol_transform(img)
                frames.append(img)

            mol_frames = torch.stack(frames).to(device)

            with torch.no_grad():
                frame_features = []
                for t in range(num_frames):
                    frame = mol_frames[t:t+1]
                    outputs = vit_model(frame)
                    frame_features.append(outputs.last_hidden_state[:, 0, :].cpu())
                features = torch.cat(frame_features, dim=0)
                all_features.append(features)

        except Exception as e:
            print(f"\nWarning: failed to process SMILES '{smiles}': {e}")
            all_features.append(torch.zeros(num_frames, vit_dim))

    unique_vit_features = torch.stack(all_features)

    print(f"\nFeature extraction complete")
    print(f"  Shape: {unique_vit_features.shape}")
    print(f"  Size: {unique_vit_features.numel() * 4 / (1024**3):.2f} GB")

    print("\n" + "="*80)
    print("Step 3: Save features and index mappings")
    print("="*80)

    output_dir = '../preprocessed_features'
    os.makedirs(output_dir, exist_ok=True)

    unique_vit_path = os.path.join(output_dir, 'unique_vit_features.pt')
    torch.save(unique_vit_features, unique_vit_path)
    print(f"Saved unique ViT features: {unique_vit_path}")

    smiles_mapping_path = os.path.join(output_dir, 'smiles_to_idx.pkl')
    with open(smiles_mapping_path, 'wb') as f:
        pickle.dump(smiles_to_idx, f)
    print(f"Saved SMILES index mapping: {smiles_mapping_path}")

    def save_smiles_indices(df, split_name):
        smiles_indices = df['SMILES'].map(smiles_to_idx).values
        output_path = os.path.join(output_dir, f'{split_name}_smiles_indices.npy')
        np.save(output_path, smiles_indices)
        print(f"Saved {split_name} SMILES indices: {output_path}")

    save_smiles_indices(df_train, 'train')
    save_smiles_indices(df_val, 'val')
    save_smiles_indices(df_test, 'test')

    print("\n" + "="*80)
    print("All ViT features extracted successfully.")
    print("="*80)

    print("\nMemory comparison:")
    original_size_gb = (total_samples * bytes_per_feature) / (1024**3)
    print(f"  Per-sample approach: {original_size_gb:.2f} GB")
    print(f"  Unique-only approach: {total_size_gb:.2f} GB")
    print(f"  Savings: {original_size_gb - total_size_gb:.2f} GB ({(1 - total_size_gb/original_size_gb)*100:.1f}%)")

    print("\nNext steps:")
    print("1. Ensure main.py has use_unique_features=True")
    print("2. Run training: python main.py --use_esm2 --use_precomputed_vit")
    print("="*80)


if __name__ == '__main__':
    extract_unique_vit_features()

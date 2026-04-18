import torch
import pandas as pd
import pickle
from transformers import EsmTokenizer, EsmModel, T5Tokenizer, T5EncoderModel
from tqdm import tqdm
import os
import numpy as np
from configs import get_cfg_defaults

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

print("\n" + "="*80)
print("Step 1: Load BindingDB data and build global unique entity lists")
print("="*80)

df_train = pd.read_csv('../data/df_train_cleaned.csv')
df_test = pd.read_csv('../data/df_test_cleaned.csv')

df_train = df_train.rename(columns={
    'compound_iso_smiles': 'SMILES',
    'target_sequence': 'Protein',
    'label': 'Y'
})
df_test = df_test.rename(columns={
    'compound_iso_smiles': 'SMILES',
    'target_sequence': 'Protein',
    'label': 'Y'
})

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

print(f"Train samples: {len(df_train):,}")
print(f"Test samples: {len(df_test):,}")
print(f"Total samples: {len(df_train) + len(df_test):,}")

all_proteins_series = pd.concat([df_train['Protein'], df_test['Protein']])
unique_proteins = all_proteins_series.unique().tolist()
protein_to_idx = {protein: idx for idx, protein in enumerate(unique_proteins)}

print(f"\nUnique proteins: {len(unique_proteins):,}")
print(f"Total samples: {len(all_proteins_series):,}")
print(f"Redundancy: {(1 - len(unique_proteins) / len(all_proteins_series)) * 100:.1f}%")

all_smiles_series = pd.concat([df_train['SMILES'], df_test['SMILES']])
unique_smiles = all_smiles_series.unique().tolist()
smiles_to_idx = {smiles: idx for idx, smiles in enumerate(unique_smiles)}

print(f"Unique molecules: {len(unique_smiles):,}")
print(f"Total samples: {len(all_smiles_series):,}")
print(f"Redundancy: {(1 - len(unique_smiles) / len(all_smiles_series)) * 100:.1f}%")

print("\n" + "="*80)
print("Step 2: Save global index mappings")
print("="*80)

cfg = get_cfg_defaults()
output_base = cfg.PRECOMPUTED.PLM_FEATURE_DIR

model_path = cfg.PROTEIN.MODEL_PATH
model_type = cfg.PROTEIN.MODEL_TYPE
plm_hidden_dim = cfg.PROTEIN.HIDDEN_DIM
max_length = cfg.PROTEIN.MAX_LENGTH

model_path_lower = model_path.lower()
if 'esm2' in model_path_lower or model_type == 'esm2':
    model_type = 'esm2'
    model_name = os.path.basename(os.path.normpath(model_path)).split('_')[2]
elif 'prott5' in model_path_lower or 't5' in model_path_lower or model_type == 'prott5':
    model_type = 'prott5'
    model_name = 'T5-XL'
else:
    raise ValueError(f"Unknown model type in path: {model_path}")

output_dir = os.path.join(output_base, model_name)
os.makedirs(output_dir, exist_ok=True)

protein_mapping_path = os.path.join(output_dir, 'protein_to_idx.pkl')
with open(protein_mapping_path, 'wb') as f:
    pickle.dump(protein_to_idx, f)
print(f"Saved protein index mapping: {protein_mapping_path}")

smiles_mapping_path = os.path.join(output_dir, 'smiles_to_idx.pkl')
with open(smiles_mapping_path, 'wb') as f:
    pickle.dump(smiles_to_idx, f)
print(f"Saved molecule index mapping: {smiles_mapping_path}")

def save_indices(df, split_name, protein_to_idx, smiles_to_idx, save_dir):
    protein_indices = df['Protein'].map(protein_to_idx).values
    smiles_indices = df['SMILES'].map(smiles_to_idx).values
    np.save(os.path.join(save_dir, f'{split_name}_protein_indices.npy'), protein_indices)
    np.save(os.path.join(save_dir, f'{split_name}_smiles_indices.npy'), smiles_indices)
    print(f"Saved {split_name} index arrays: {len(protein_indices):,} samples")

save_indices(df_train, 'train', protein_to_idx, smiles_to_idx, output_dir)
save_indices(df_test, 'test', protein_to_idx, smiles_to_idx, output_dir)

print("\n" + "="*80)
print("Step 3: Extract unique protein PLM features")
print("="*80)

print(f"\n{'='*60}")
print(f"Loading Protein Language Model")
print(f"{'='*60}")
print(f"  Model type: {model_type.upper()}")
print(f"  Model path: {model_path}")
print(f"  Model size: {model_name}")
print(f"  Hidden dim: {plm_hidden_dim}")

if model_type == 'esm2':
    tokenizer = EsmTokenizer.from_pretrained(model_path)
    model = EsmModel.from_pretrained(model_path).to(device)
elif model_type == 'prott5':
    tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)
    model = T5EncoderModel.from_pretrained(model_path).to(device)
else:
    raise ValueError(f"Unsupported model type: {model_type}")

model.eval()
print("Model loaded successfully")
print(f"{'='*60}\n")

def extract_unique_protein_features(unique_proteins, output_path, batch_size=8):
    num_proteins = len(unique_proteins)

    if model_type == 'prott5':
        protein_sequences = [' '.join(list(seq.replace(' ', ''))) for seq in unique_proteins]
    else:
        protein_sequences = unique_proteins

    single_sample_mb = (max_length * plm_hidden_dim * 4) / (1024 * 1024)
    total_size_gb = (num_proteins * single_sample_mb) / 1024

    print(f"\nExtracting {model_type.upper()} features (unique proteins)...")
    print(f"Unique proteins: {num_proteins:,}")
    print(f"Batch size: {batch_size}, Max length: {max_length}")
    print(f"Estimated output size: {total_size_gb:.2f} GB (float32)")
    print(f"Memory optimization: using memmap (peak RAM: ~{batch_size * single_sample_mb / 1024:.2f} GB)")

    output_path_npy = output_path.replace('.pt', '.npy')
    os.makedirs(os.path.dirname(output_path_npy), exist_ok=True)

    features_mmap = np.memmap(
        output_path_npy,
        dtype='float32',
        mode='w+',
        shape=(num_proteins, max_length, plm_hidden_dim)
    )

    current_idx = 0
    with torch.no_grad():
        for i in tqdm(range(0, num_proteins, batch_size), desc="Extracting features"):
            batch_seqs = protein_sequences[i:i+batch_size]

            encoded = tokenizer(
                batch_seqs,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length
            ).to(device)

            outputs = model(**encoded)
            features = outputs.last_hidden_state.cpu().numpy()

            batch_size_actual = features.shape[0]
            features_mmap[current_idx:current_idx+batch_size_actual] = features
            current_idx += batch_size_actual

            del encoded, outputs, features
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    features_mmap.flush()

    print(f"Saved to {output_path_npy}")
    print(f"  Shape: {features_mmap.shape}")
    print(f"  Disk size: {os.path.getsize(output_path_npy) / (1024**3):.2f} GB")

    return features_mmap

print(f"\nSaving features to {output_dir}/")

unique_protein_features_path = os.path.join(output_dir, 'unique_protein_features.npy')
extract_unique_protein_features(unique_proteins, unique_protein_features_path)

print("\n" + "="*80)
print(f"All {model_type.upper()} features extracted successfully.")
print(f"  Saved to: {output_dir}/")
print(f"  Format: .npy (NumPy memmap)")
print("="*80)

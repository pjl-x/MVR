from models import BINDTI
from time import time
from utils import set_seed, graph_collate_func, mkdir
from configs import get_cfg_defaults
from dataloader import DTIDataset
from torch.utils.data import DataLoader
from trainer import Trainer
import torch
import argparse
import warnings, os
import random
import pandas as pd
from datetime import datetime
import pickle
import numpy as np
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer

cuda_id = 0
device = torch.device(f'cuda:{cuda_id}' if torch.cuda.is_available() else 'cpu')
parser = argparse.ArgumentParser(description="BINDTI for DTI prediction")
parser.add_argument('--data', type=str, metavar='TASK', help='dataset', default='sample')
parser.add_argument('--split', default='random', type=str, metavar='S', help="split task",
                    choices=['random', 'random1', 'random2', 'random3', 'random4'])
parser.add_argument('--use_precomputed_vit', action='store_true',  default=True,
                    help='Use precomputed ViT features instead of computing on-the-fly')
parser.add_argument('--vit_feature_dir', type=str, default='../preprocessed_features',
                    help='Directory containing precomputed ViT features')
parser.add_argument('--use_esm2', action='store_true', default=True,
                    help='Use ESM-2 features for protein representation')
parser.add_argument('--no_esm2', dest='use_esm2', action='store_false',
                    help='Disable ESM-2, use original ProteinACmix instead')
parser.add_argument('--esm2_feature_dir', type=str, default='../preprocessed_features',
                    help='Directory containing precomputed PLM features (ProtT5/ESM-2)')
parser.add_argument('--use_unique_features', action='store_true', default=True,
                    help='Use unique entity features (Plan A optimization)')
args = parser.parse_args()


df_train = pd.read_csv('../data/df_train_cleaned.csv')
df_test = pd.read_csv('../data/df_test_cleaned.csv')

df_train = df_train.rename(columns={'compound_iso_smiles': 'SMILES', 'target_sequence': 'Protein', 'label': 'Y'})
df_test = df_test.rename(columns={'compound_iso_smiles': 'SMILES', 'target_sequence': 'Protein', 'label': 'Y'})

df_train = df_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

train_pro = {'df': df_train}
test_pro  = {'df': df_test}

print(f"BindingDB Train: {len(df_train):,} samples")
print(f"BindingDB Test:  {len(df_test):,} samples")

if args.use_unique_features:
    print("\n" + "="*80)
    print("Loading unique entity indices from precomputed files")
    print("="*80)

    feature_dir = '../preprocessed_features/T5-XL'

    with open(os.path.join(feature_dir, 'protein_to_idx.pkl'), 'rb') as f:
        protein_to_idx_map = pickle.load(f)
    with open(os.path.join(feature_dir, 'smiles_to_idx.pkl'), 'rb') as f:
        smiles_to_idx_map = pickle.load(f)

    unique_proteins_list = list(protein_to_idx_map.keys())
    unique_smiles_list   = list(smiles_to_idx_map.keys())

    train_pro['protein_indices'] = df_train['Protein'].map(protein_to_idx_map).values
    test_pro['protein_indices']  = df_test['Protein'].map(protein_to_idx_map).values

    train_pro['smiles_indices'] = df_train['SMILES'].map(smiles_to_idx_map).values
    test_pro['smiles_indices']  = df_test['SMILES'].map(smiles_to_idx_map).values

    print(f"Unique proteins: {len(unique_proteins_list):,}")
    print(f"Unique molecules: {len(unique_smiles_list):,}")
    print("="*80 + "\n")

tokenizer = AutoTokenizer.from_pretrained('../molformer', trust_remote_code=True)

smiles_list  = train_pro['df'].iloc[:, 0].tolist()
smiles_list1 = test_pro['df'].iloc[:, 0].tolist()
encoded_inputs  = tokenizer(smiles_list,  padding=True, truncation=True, max_length=128, return_tensors="pt")
encoded_inputs1 = tokenizer(smiles_list1, padding=True, truncation=True, max_length=128, return_tensors="pt")

input_ids      = encoded_inputs["input_ids"]
attention_mask = encoded_inputs["attention_mask"]

input_ids1      = encoded_inputs1["input_ids"]
attention_mask1 = encoded_inputs1["attention_mask"]


SEEDS = random.sample(range(0, 10000), 5)


def run_single_seed(seed):
    torch.cuda.empty_cache()
    warnings.filterwarnings("ignore", message="invalid value encountered in divide")
    cfg = get_cfg_defaults()
    set_seed(seed)
    mkdir(cfg.RESULT.OUTPUT_DIR + f'{args.data}/{args.split}/seed_{seed}')

    print(f"\n{'='*80}")
    print(f"Running seed {seed}")
    print(f"{'='*80}")
    print(f"dataset:{args.data}")
    print(f"Use precomputed ViT features: {args.use_precomputed_vit}")
    print(f"Use ESM-2 features: {args.use_esm2}")
    print(f"Running on: {device}", end="\n\n")

    dataFolder = f'../datasets/{args.data}'
    dataFolder = os.path.join(dataFolder, str(args.split))

    mol_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    precomputed_train_vit = None
    precomputed_test_vit = None

    if args.use_precomputed_vit:
        print("Loading precomputed ViT features...")
        vit_feature_dir = '../preprocessed_features'
        unique_vit_path = os.path.join(vit_feature_dir, 'unique_vit_features.pt')

        if os.path.exists(unique_vit_path):
            unique_vit_features = torch.load(unique_vit_path)
            print(f"Loaded unique molecule ViT features: {unique_vit_features.shape}")
            precomputed_train_vit = unique_vit_features
            precomputed_test_vit  = unique_vit_features
        else:
            print(f"Warning: unique ViT features not found at {unique_vit_path}, will extract on-the-fly")
            args.use_precomputed_vit = False
        print()

    precomputed_train_esm2 = None
    precomputed_test_esm2  = None

    if args.use_esm2:
        print("Loading precomputed Protein Language Model features...")
        model_path = cfg.PROTEIN.MODEL_PATH
        model_type = cfg.PROTEIN.MODEL_TYPE

        model_path_lower = model_path.lower()
        if 'esm2' in model_path_lower or model_type == 'esm2':
            model_name = os.path.basename(os.path.normpath(model_path)).split('_')[2]
        elif 'prott5' in model_path_lower or 't5' in model_path_lower or model_type == 'prott5':
            model_name = 'T5-XL'
        else:
            model_name = os.path.basename(os.path.normpath(model_path))

        plm_dir = os.path.join('../preprocessed_features', model_name)
        max_length = cfg.PROTEIN.MAX_LENGTH
        hidden_dim = cfg.PROTEIN.HIDDEN_DIM

        print(f"  Model type: {model_type.upper()},  Model name: {model_name}")
        print(f"  Feature dir: {plm_dir}")

        def _load_plm_npy(path_npy, n_samples):
            print(f"  Reading {path_npy} into memory...")
            arr = np.memmap(path_npy, dtype='float32', mode='r', shape=(n_samples, max_length, hidden_dim))
            arr_copy = np.array(arr, copy=True, dtype=np.float32)
            tensor = torch.from_numpy(arr_copy)
            _ = tensor.sum()
            print(f"  Memory resident: {tensor.numel() * 4 / (1024**3):.2f} GB")
            return tensor

        unique_protein_path = os.path.join(plm_dir, 'unique_protein_features.npy')
        if not os.path.exists(unique_protein_path):
            raise FileNotFoundError(
                f"Unique protein features not found: {unique_protein_path}\n"
                f"Please run: python extract_unique_features.py"
            )
        n_unique = len(unique_proteins_list)
        print(f"  Unique proteins: {n_unique:,}")
        precomputed_unique_protein = _load_plm_npy(unique_protein_path, n_unique)
        print(f"Loaded unique protein features: {precomputed_unique_protein.shape}")
        precomputed_train_esm2 = precomputed_unique_protein
        precomputed_test_esm2  = precomputed_unique_protein
        print()

    train_dataset = DTIDataset(train_pro, input_ids, attention_mask,
                               mol_data_root=r"../video/video-data/train",
                               mol_transform=mol_transform, num_frames=60,
                               precomputed_vit_features=precomputed_train_vit,
                               precomputed_esm2_features=precomputed_train_esm2,
                               use_unique_features=args.use_unique_features)
    test_dataset = DTIDataset(test_pro, input_ids1, attention_mask1,
                              mol_data_root=r"../video/video-data/test",
                              mol_transform=mol_transform, num_frames=60,
                              precomputed_vit_features=precomputed_test_vit,
                              precomputed_esm2_features=precomputed_test_esm2,
                              use_unique_features=args.use_unique_features)
    print(f'train_dataset: {len(train_dataset):,}')
    print(f'test_dataset:  {len(test_dataset):,}')

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}

    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    test_generator = DataLoader(test_dataset, **params)

    model = BINDTI(device=device, use_precomputed_vit=args.use_precomputed_vit, use_esm2=args.use_esm2, **cfg).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, opt, device, training_generator, test_generator, args.data, args.split,
                      seed=seed, **cfg)
    result = trainer.train()

    seed_dir = os.path.join(cfg.RESULT.OUTPUT_DIR, f"{args.data}/{args.split}/seed_{seed}")
    with open(os.path.join(seed_dir, "model_architecture.txt"), "w") as wf:
        wf.write(str(model))
    with open(os.path.join(seed_dir, "config.txt"), "w") as wf:
        wf.write(str(dict(cfg)))

    print(f"\nDirectory for saving result: {seed_dir}")
    print(f'Seed {seed} done.')

    return result


def main():
    print(f"Generated seeds: {SEEDS}")
    all_results = {}
    for seed in SEEDS:
        result = run_single_seed(seed)
        all_results[seed] = result
    print(f"\n{'='*80}")
    print("All seeds completed.")
    for seed, res in all_results.items():
        print(f"  seed {seed}: AUROC={res.get('auroc', 'N/A'):.4f}  AUPRC={res.get('auprc', 'N/A'):.4f}")
    print(f"{'='*80}")
    return all_results


if __name__ == '__main__':
    print(f"start time: {datetime.now()}")
    s = time()
    main()
    e = time()
    print(f"end time: {datetime.now()}")
    print(f"Total running time: {round(e - s, 2)}s")

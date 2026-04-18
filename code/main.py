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


train = pd.read_csv('../data/train.txt', sep='\t')
test = pd.read_csv('../data/test.txt', sep='\t')
val = pd.read_csv('../data/val.txt', sep='\t')
with open('../data/gene_info_dics.pkl', 'rb') as fp:
    protein_info = pickle.load(fp)

with open('../data/drug_info_dics.pkl', 'rb') as fp:
    drug_info = pickle.load(fp)

train['head'] = train['head'].str.split('::').str[1]
train['SMILES'] = train['head'].apply(lambda x: drug_info.get(x, {}).get('SMILES'))
train['tail'] = train['tail'].str.split('::').str[1]
train['Sequence'] = train['tail'].apply(lambda x: protein_info.get(x, {}).get('Sequence'))
df_train = train.dropna(subset=['SMILES', 'Sequence'])
df_train = df_train[['SMILES', 'Sequence', 'Label', 'head', 'tail', 'UP', 'relation']]
df_train['head'] = df_train['head'].apply(lambda x: f"Compound::{x}")
df_train['tail'] = df_train['tail'].apply(lambda x: f"Gene::{x}")
df_train.columns = ['SMILES', 'Protein', 'Y', 'head', 'tail', 'UP', 'relation']
df_train = df_train.reset_index(drop=True)

test['head'] = test['head'].str.split('::').str[1]
test['SMILES'] = test['head'].apply(lambda x: drug_info.get(x, {}).get('SMILES'))
test['tail'] = test['tail'].str.split('::').str[1]
test['Sequence'] = test['tail'].apply(lambda x: protein_info.get(x, {}).get('Sequence'))
df_test = test.dropna(subset=['SMILES', 'Sequence'])
df_test = df_test[['SMILES', 'Sequence', 'Label', 'head', 'tail', 'UP', 'relation']]
df_test['head'] = df_test['head'].apply(lambda x: f"Compound::{x}")
df_test['tail'] = df_test['tail'].apply(lambda x: f"Gene::{x}")
df_test.columns = ['SMILES', 'Protein', 'Y', 'head', 'tail', 'UP', 'relation']
df_test = df_test.reset_index(drop=True)

val['head'] = val['head'].str.split('::').str[1]
val['SMILES'] = val['head'].apply(lambda x: drug_info.get(x, {}).get('SMILES'))
val['tail'] = val['tail'].str.split('::').str[1]
val['Sequence'] = val['tail'].apply(lambda x: protein_info.get(x, {}).get('Sequence'))
df_val = val.dropna(subset=['SMILES', 'Sequence'])
df_val = df_val[['SMILES', 'Sequence', 'Label', 'head', 'tail', 'UP', 'relation']]
df_val['head'] = df_val['head'].apply(lambda x: f"Compound::{x}")
df_val['tail'] = df_val['tail'].apply(lambda x: f"Gene::{x}")
df_val.columns = ['SMILES', 'Protein', 'Y', 'head', 'tail', 'UP', 'relation']
df_val = df_val.reset_index(drop=True)


def get_kg_embs():
    entity = pd.read_csv('../data/entity_to_id.tsv.gz', sep='\t')
    entity_dic = dict(zip(entity['label'], entity['id']))
    entity_representation = np.load('../data/entity_representation.npy')
    return entity_dic, entity_representation


def get_d_kg(entity_dic, entity_representation, drug):
    if drug in entity_dic.keys():
        kg_id = entity_dic[drug]
        kg = entity_representation[kg_id]
    else:
        kg = np.zeros((200,))
    return kg


def get_p_kg(entity_dic, entity_representation, protein, **config):
    if protein in entity_dic.keys():
        kg_id = entity_dic[protein]
        kg = entity_representation[kg_id]
    else:
        kg = np.zeros((200,))
    return kg


entity_dic, entity_representation = get_kg_embs()

train_pro = {}
test_pro = {}
val_pro = {}

unique = []
for item in df_train['head'].unique():
    unique.append(get_d_kg(entity_dic, entity_representation, item))
    unique_dict = dict(zip(df_train['head'].unique(), unique))
    train_pro['D_kg'] = [unique_dict.get(i, None) for i in df_train['head']]

unique = []
for item in df_train['tail'].unique():
    unique.append(get_p_kg(entity_dic, entity_representation, item))
    unique_dict = dict(zip(df_train['tail'].unique(), unique))
    train_pro['P_kg'] = [unique_dict.get(i, None) for i in df_train['tail']]
train_pro['df'] = df_train

unique = []
for item in df_test['head'].unique():
    unique.append(get_d_kg(entity_dic, entity_representation, item))
    unique_dict = dict(zip(df_test['head'].unique(), unique))
    test_pro['D_kg'] = [unique_dict.get(i, None) for i in df_test['head']]

unique = []
for item in df_test['tail'].unique():
    unique.append(get_p_kg(entity_dic, entity_representation, item))
    unique_dict = dict(zip(df_test['tail'].unique(), unique))
    test_pro['P_kg'] = [unique_dict.get(i, None) for i in df_test['tail']]

test_pro['df'] = df_test

unique = []
for item in df_val['head'].unique():
    unique.append(get_d_kg(entity_dic, entity_representation, item))
    unique_dict = dict(zip(df_val['head'].unique(), unique))
    val_pro['D_kg'] = [unique_dict.get(i, None) for i in df_val['head']]

unique = []
for item in df_val['tail'].unique():
    unique.append(get_p_kg(entity_dic, entity_representation, item))
    unique_dict = dict(zip(df_val['tail'].unique(), unique))
    val_pro['P_kg'] = [unique_dict.get(i, None) for i in df_val['tail']]

val_pro['df'] = df_val


df_train1 = pd.read_csv('../data/df_train.csv', sep=',')
df_test1 = pd.read_csv('../data/df_test.csv', sep=',')
df_val1 = pd.read_csv('../data/df_val.csv', sep=',')

df_train = train_pro['df']
common_smiles = set(df_train.iloc[:, 0]).intersection(set(df_train1.iloc[:, 0]))
mask_train = df_train.iloc[:, 0].isin(common_smiles)
filtered_indices_train = df_train[mask_train].index.tolist()

df_train = df_train[mask_train].reset_index(drop=True)
train_pro['df'] = df_train
train_pro['D_kg'] = [train_pro['D_kg'][i] for i in filtered_indices_train]
train_pro['P_kg'] = [train_pro['P_kg'][i] for i in filtered_indices_train]

df_test = test_pro['df']
common_smiles1 = set(df_test.iloc[:, 0]).intersection(set(df_test1.iloc[:, 0]))
mask_test = df_test.iloc[:, 0].isin(common_smiles1)
filtered_indices_test = df_test[mask_test].index.tolist()

df_test = df_test[mask_test].reset_index(drop=True)
test_pro['df'] = df_test
test_pro['D_kg'] = [test_pro['D_kg'][i] for i in filtered_indices_test]
test_pro['P_kg'] = [test_pro['P_kg'][i] for i in filtered_indices_test]

df_val = val_pro['df']
common_smiles2 = set(df_val.iloc[:, 0]).intersection(set(df_val1.iloc[:, 0]))
mask_val = df_val.iloc[:, 0].isin(common_smiles2)
filtered_indices_val = df_val[mask_val].index.tolist()

df_val = df_val[mask_val].reset_index(drop=True)
val_pro['df'] = df_val
val_pro['D_kg'] = [val_pro['D_kg'][i] for i in filtered_indices_val]
val_pro['P_kg'] = [val_pro['P_kg'][i] for i in filtered_indices_val]

if args.use_unique_features:
    print("\n" + "="*80)
    print("Building global unique entity indices")
    print("="*80)

    all_proteins_series = pd.concat([df_train['Protein'], df_val['Protein'], df_test['Protein']])
    unique_proteins_list = all_proteins_series.unique().tolist()
    protein_to_idx_map = {protein: idx for idx, protein in enumerate(unique_proteins_list)}

    train_pro['protein_indices'] = df_train['Protein'].map(protein_to_idx_map).values
    val_pro['protein_indices'] = df_val['Protein'].map(protein_to_idx_map).values
    test_pro['protein_indices'] = df_test['Protein'].map(protein_to_idx_map).values

    print(f"Unique proteins: {len(unique_proteins_list):,}")
    print(f"Total samples: {len(all_proteins_series):,}")
    print(f"Redundancy: {(1 - len(unique_proteins_list) / len(all_proteins_series)) * 100:.1f}%")

    all_smiles_series = pd.concat([df_train['SMILES'], df_val['SMILES'], df_test['SMILES']])
    unique_smiles_list = all_smiles_series.unique().tolist()
    smiles_to_idx_map = {smiles: idx for idx, smiles in enumerate(unique_smiles_list)}

    train_pro['smiles_indices'] = df_train['SMILES'].map(smiles_to_idx_map).values
    val_pro['smiles_indices'] = df_val['SMILES'].map(smiles_to_idx_map).values
    test_pro['smiles_indices'] = df_test['SMILES'].map(smiles_to_idx_map).values

    print(f"Unique molecules: {len(unique_smiles_list):,}")
    print(f"Total samples: {len(all_smiles_series):,}")
    print(f"Redundancy: {(1 - len(unique_smiles_list) / len(all_smiles_series)) * 100:.1f}%")
    print("="*80 + "\n")

tokenizer = AutoTokenizer.from_pretrained('../molformer', trust_remote_code=True)

smiles_list = train_pro['df'].iloc[:, 0].tolist()
smiles_list1 = test_pro['df'].iloc[:, 0].tolist()
smiles_list2 = val_pro['df'].iloc[:, 0].tolist()
encoded_inputs = tokenizer(smiles_list, padding=True, truncation=True, max_length=128, return_tensors="pt")
encoded_inputs1 = tokenizer(smiles_list1, padding=True, truncation=True, max_length=128, return_tensors="pt")
encoded_inputs2 = tokenizer(smiles_list2, padding=True, truncation=True, max_length=128, return_tensors="pt")

input_ids = encoded_inputs["input_ids"]
attention_mask = encoded_inputs["attention_mask"]

input_ids1 = encoded_inputs1["input_ids"]
attention_mask1 = encoded_inputs1["attention_mask"]

input_ids2 = encoded_inputs2["input_ids"]
attention_mask2 = encoded_inputs2["attention_mask"]


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
    precomputed_val_vit = None
    precomputed_test_vit = None

    if args.use_precomputed_vit:
        print("Loading precomputed ViT features...")
        print(f"  Use unique features: {args.use_unique_features}")

        if args.use_unique_features:
            unique_vit_path = os.path.join(args.vit_feature_dir, 'unique_vit_features.pt')

            if os.path.exists(unique_vit_path):
                unique_vit_features = torch.load(unique_vit_path)
                print(f"Loaded unique molecule ViT features: {unique_vit_features.shape}")
                print(f"  Unique molecules: {len(unique_smiles_list):,}")
                precomputed_train_vit = unique_vit_features
                precomputed_val_vit = unique_vit_features
                precomputed_test_vit = unique_vit_features
            else:
                print(f"Warning: unique ViT features not found at {unique_vit_path}")
                print(f"  Falling back to per-split loading")
                print(f"  Please run: python extract_unique_vit_features.py")
                args.use_unique_features = False

        if not args.use_unique_features:
            train_vit_path = os.path.join(args.vit_feature_dir, 'train_vit_features.pt')
            val_vit_path = os.path.join(args.vit_feature_dir, 'val_vit_features.pt')
            test_vit_path = os.path.join(args.vit_feature_dir, 'test_vit_features.pt')

            if os.path.exists(train_vit_path):
                precomputed_train_vit = torch.load(train_vit_path)
                print(f"Loaded train ViT features: {precomputed_train_vit.shape}")
            else:
                print(f"Warning: {train_vit_path} not found, will compute on-the-fly")
                args.use_precomputed_vit = False

            if os.path.exists(val_vit_path):
                precomputed_val_vit = torch.load(val_vit_path)
                print(f"Loaded val ViT features: {precomputed_val_vit.shape}")

            if os.path.exists(test_vit_path):
                precomputed_test_vit = torch.load(test_vit_path)
                print(f"Loaded test ViT features: {precomputed_test_vit.shape}")
        print()

    precomputed_train_esm2 = None
    precomputed_val_esm2 = None
    precomputed_test_esm2 = None

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

        plm_dir = os.path.join(args.esm2_feature_dir, model_name)

        print(f"  Model type: {model_type.upper()}")
        print(f"  Model name: {model_name}")
        print(f"  Feature dir: {plm_dir}")
        print(f"  Use unique features: {args.use_unique_features}")

        max_length = cfg.PROTEIN.MAX_LENGTH
        hidden_dim = cfg.PROTEIN.HIDDEN_DIM

        def _load_plm_npy(path_npy, n_samples):
            print(f"  Reading {path_npy} into memory...")
            arr = np.memmap(path_npy, dtype='float32', mode='r', shape=(n_samples, max_length, hidden_dim))
            arr_copy = np.array(arr, copy=True, dtype=np.float32)
            tensor = torch.from_numpy(arr_copy)
            print(f"  Forcing all pages into physical memory...")
            _ = tensor.sum()
            print(f"  Memory resident: {tensor.numel() * 4 / (1024**3):.2f} GB")
            return tensor

        if args.use_unique_features:
            unique_protein_path = os.path.join(plm_dir, 'unique_protein_features.npy')

            if os.path.exists(unique_protein_path):
                n_unique = len(unique_proteins_list)
                print(f"\nLoading unique protein features...")
                print(f"  Unique proteins: {n_unique:,}")
                precomputed_unique_protein = _load_plm_npy(unique_protein_path, n_unique)
                print(f"Loaded unique protein features: {precomputed_unique_protein.shape}")
                precomputed_train_esm2 = precomputed_unique_protein
                precomputed_val_esm2 = precomputed_unique_protein
                precomputed_test_esm2 = precomputed_unique_protein
            else:
                raise FileNotFoundError(
                    f"Unique protein features not found: {unique_protein_path}\n"
                    f"Please run: python extract_unique_features.py"
                )
        else:
            train_plm_path = os.path.join(plm_dir, 'train_esm2_features.pt')
            val_plm_path = os.path.join(plm_dir, 'val_esm2_features.pt')
            test_plm_path = os.path.join(plm_dir, 'test_esm2_features.pt')

            train_plm_path_npy = train_plm_path.replace('.pt', '.npy')
            val_plm_path_npy = val_plm_path.replace('.pt', '.npy')
            test_plm_path_npy = test_plm_path.replace('.pt', '.npy')

            if os.path.exists(train_plm_path):
                precomputed_train_esm2 = torch.load(train_plm_path)
                print(f"Loaded train features (.pt): {precomputed_train_esm2.shape}")
            elif os.path.exists(train_plm_path_npy):
                n_train = len(train_pro['df'])
                print(f"Loading train features from .npy (memmap, shape=({n_train}, {max_length}, {hidden_dim}))...")
                precomputed_train_esm2 = _load_plm_npy(train_plm_path_npy, n_train)
                print(f"Loaded train features (.npy): {precomputed_train_esm2.shape}")
            else:
                raise FileNotFoundError(f"PLM features not found at {train_plm_path} or {train_plm_path_npy}. Please run feature extraction first.")

            if os.path.exists(val_plm_path):
                precomputed_val_esm2 = torch.load(val_plm_path)
                print(f"Loaded val features (.pt): {precomputed_val_esm2.shape}")
            elif os.path.exists(val_plm_path_npy):
                n_val = len(val_pro['df'])
                precomputed_val_esm2 = _load_plm_npy(val_plm_path_npy, n_val)
                print(f"Loaded val features (.npy): {precomputed_val_esm2.shape}")

            if os.path.exists(test_plm_path):
                precomputed_test_esm2 = torch.load(test_plm_path)
                print(f"Loaded test features (.pt): {precomputed_test_esm2.shape}")
            elif os.path.exists(test_plm_path_npy):
                n_test = len(test_pro['df'])
                precomputed_test_esm2 = _load_plm_npy(test_plm_path_npy, n_test)
                print(f"Loaded test features (.npy): {precomputed_test_esm2.shape}")
        print()

    train_dataset = DTIDataset(train_pro, input_ids, attention_mask, mol_data_root=r"../video/video-data/train",
                               mol_transform=mol_transform, num_frames=60, precomputed_vit_features=precomputed_train_vit,
                               precomputed_esm2_features=precomputed_train_esm2, use_unique_features=args.use_unique_features)
    print(f'train_dataset:{len(train_dataset)}')
    val_dataset = DTIDataset(val_pro, input_ids2, attention_mask2, mol_data_root=r"../video/video-data/val",
                             mol_transform=mol_transform, num_frames=60, precomputed_vit_features=precomputed_val_vit,
                             precomputed_esm2_features=precomputed_val_esm2, use_unique_features=args.use_unique_features)
    test_dataset = DTIDataset(test_pro, input_ids1, attention_mask1, mol_data_root=r"../video/video-data/test",
                              mol_transform=mol_transform, num_frames=60, precomputed_vit_features=precomputed_test_vit,
                              precomputed_esm2_features=precomputed_test_esm2, use_unique_features=args.use_unique_features)

    params = {'batch_size': cfg.SOLVER.BATCH_SIZE, 'shuffle': True, 'num_workers': cfg.SOLVER.NUM_WORKERS,
              'drop_last': True, 'collate_fn': graph_collate_func}

    training_generator = DataLoader(train_dataset, **params)
    params['shuffle'] = False
    params['drop_last'] = False
    val_generator = DataLoader(val_dataset, **params)
    test_generator = DataLoader(test_dataset, **params)

    model = BINDTI(device=device, use_precomputed_vit=args.use_precomputed_vit, use_esm2=args.use_esm2, **cfg).to(device=device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    torch.backends.cudnn.benchmark = True

    trainer = Trainer(model, opt, device, training_generator, val_generator, test_generator, args.data, args.split,
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

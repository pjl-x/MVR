import os
import traceback
from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
from tqdm import tqdm

df_train = pd.read_csv('../data/df_test_cleaned.csv', sep=',')

folder_path = './comformer/train'
os.makedirs(folder_path, exist_ok=True)


def mol2sdf(molecule, sdf_save_path):
    writer = Chem.SDWriter(sdf_save_path)
    writer.write(molecule)
    writer.close()


def generate_3d_comformer(smiles, sdf_save_path, mmffVariant="MMFF94", randomSeed=0, maxIters=5000, increment=2,
                          optim_count=10, save_force=False):
    count = 0
    res = None
    while count < optim_count:
        try:
            m = Chem.MolFromSmiles(smiles)
            if m is None:
                return False

            m3d = Chem.AddHs(m)

            if save_force:
                try:
                    AllChem.EmbedMolecule(m3d, randomSeed=randomSeed)
                    res = AllChem.MMFFOptimizeMolecule(m3d, mmffVariant=mmffVariant, maxIters=maxIters)
                    m3d = Chem.RemoveHs(m3d)
                except Exception as e:
                    res = -1
                    m3d = Chem.RemoveHs(m3d)
                    mol2sdf(m3d, sdf_save_path)
            else:
                AllChem.EmbedMolecule(m3d, randomSeed=randomSeed)
                res = AllChem.MMFFOptimizeMolecule(m3d, mmffVariant=mmffVariant, maxIters=maxIters)
                m3d = Chem.RemoveHs(m3d)
        except Exception as e:
            res = -1

        if res == 1:
            maxIters *= increment
            count += 1
            continue
        elif res == 0:
            mol2sdf(m3d, sdf_save_path)
            return True
        elif res == -1:
            break
        else:
            break

    if save_force and res != 0:
        mol2sdf(m3d, sdf_save_path)
        return True

    return False


print(f"Dataset size: {len(df_train)} entries. Generating 3D conformers...")

successful_indices = []
failed_indices = []
successful_count = 0

for original_idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Processing"):
    smiles = row.iloc[0]

    sdf_save_path = os.path.join(folder_path, f"molecule_{successful_count}_3d.sdf")

    success = generate_3d_comformer(
        smiles=smiles,
        sdf_save_path=sdf_save_path,
        mmffVariant="MMFF94",
        randomSeed=42,
        maxIters=5000,
        increment=2,
        optim_count=10,
        save_force=False
    )

    if success:
        successful_indices.append(original_idx)
        successful_count += 1
    else:
        failed_indices.append(original_idx)

df_clean = df_train.loc[successful_indices].copy()
df_clean.reset_index(drop=True, inplace=True)
clean_csv_path = os.path.join(folder_path, "df_train_cleaned.csv")
df_clean.to_csv(clean_csv_path, index=False)

if len(failed_indices) > 0:
    df_failed = df_train.loc[failed_indices].copy()
    failed_csv_path = os.path.join(folder_path, "df_train_failed.csv")
    df_failed.to_csv(failed_csv_path, index=False)
else:
    failed_csv_path = "none"

print("\n" + "=" * 50)
print(f"Done.")
print(f"Total: {len(df_train)}")
print(f"Succeeded: {len(df_clean)}")
print(f"Failed: {len(failed_indices)}")
print("-" * 50)
print(f"Clean dataset saved to: {clean_csv_path}")
print(f"Failed records saved to: {failed_csv_path}")
print(f"Next step: run 8.py to render molecular video frames (range={len(df_clean)})")
print("=" * 50)

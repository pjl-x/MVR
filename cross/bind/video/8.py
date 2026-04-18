import os
import sys
import time
import pandas as pd
from tqdm import tqdm

import __main__
__main__.pymol_argv = ['pymol', '-qc']
import pymol

pymol.finish_launching()

sdf_folder = './comformer/train'
output_base_folder = './video-data/train'
os.makedirs(output_base_folder, exist_ok=True)

clean_csv_path = os.path.join(sdf_folder, "df_train_cleaned.csv")
df_clean = pd.read_csv(clean_csv_path)
total_molecules = len(df_clean)

pymol.cmd.bg_color("white")
pymol.cmd.set("opaque_background", "on")
pymol.cmd.set("grid_mode", "off")
pymol.cmd.set("ray_shadow", 0)
pymol.cmd.set("ray_trace_mode", 0)
pymol.cmd.hide("everything", "hydro")
pymol.cmd.set("stick_ball", "on")
pymol.cmd.set("stick_ball_ratio", 3.5)
pymol.cmd.set("stick_radius", 0.15)
pymol.cmd.set("sphere_scale", 0.2)
pymol.cmd.set("valence", 1)
pymol.cmd.set("valence_mode", 0)
pymol.cmd.set("valence_size", 0.1)

print(f"Rendering {total_molecules} molecules...")

for mol_index in tqdm(range(total_molecules), desc="Rendering Images"):

    sdf_filepath = os.path.join(sdf_folder, f"molecule_{mol_index}_3d.sdf")

    if not os.path.exists(sdf_filepath):
        print(f"\nWarning: {sdf_filepath} not found, skipping.")
        continue

    output_folder = os.path.join(output_base_folder, str(mol_index))
    os.makedirs(output_folder, exist_ok=True)

    pymol.cmd.load(sdf_filepath, "current_mol")

    pymol.cmd.set("auto_color", 0)
    pymol.cmd.util.cbag("current_mol")

    image_index = 0

    for axis in ["X", "Y", "Z"]:
        for angle in range(0, 360, 18):
            output_path = os.path.join(output_folder, f"{image_index}.png")

            pymol.cmd.png(output_path, width=640, height=480, dpi=72, ray=0)

            pymol.cmd.rotate(axis, 18, "current_mol")

            time.sleep(0.05)

            image_index += 1

    pymol.cmd.delete("current_mol")

print(f"\nAll frames saved to: {output_base_folder}")

pymol.cmd.quit()

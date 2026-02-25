#!/usr/bin/env python3
import os
import re
import numpy as np
import h5py

CASES_DIR = "/home/jinisa/OpenFOAM/dataset_temperatures"
OUTPUT_H5 = "/home/jinisa/OpenFOAM/multi_case_dataset.h5"
TARGET_REGION = "steel_cylinder"

def read_foam_scalar_field(filepath):
    with open(filepath, "r") as f:
        content = f.read()
    uniform_match = re.search(r"internalField\s+uniform\s+([\d.eE+\-]+)", content)
    if uniform_match:
        value = float(uniform_match.group(1))
        n_cells_match = re.search(r"(\d+)\s*\(", content)
        if n_cells_match:
            return np.full(int(n_cells_match.group(1)), value, dtype=np.float32)
        return np.array([value], dtype=np.float32)
    nonuniform_match = re.search(
        r"internalField\s+nonuniform\s+List<scalar>\s*\n\s*(\d+)\s*\n\s*\((.*?)\)",
        content, re.DOTALL)
    if nonuniform_match:
        values_str = nonuniform_match.group(2).strip()
        return np.array([float(v) for v in values_str.split()], dtype=np.float32)
    raise ValueError(f"Could not parse scalar field from {filepath}")

def read_foam_points(points_filepath):
    with open(points_filepath, "r") as f:
        content = f.read()
    point_pattern = re.findall(r"\(([\d.eE+\-]+)\s+([\d.eE+\-]+)\s+([\d.eE+\-]+)\)", content)
    return np.array([[float(x), float(y), float(z)] for x,y,z in point_pattern], dtype=np.float32)

def get_time_steps(case_dir):
    items = os.listdir(case_dir)
    time_steps = []
    for item in items:
        try:
            t = float(item)
            if t > 0:
                time_steps.append(t)
        except ValueError:
            continue
    return sorted(time_steps)

def read_temperature_field(case_dir, region, time_step):
    t_str = str(int(time_step)) if time_step == int(time_step) else str(time_step)
    t_file = os.path.join(case_dir, t_str, region, "T")
    if not os.path.exists(t_file):
        raise FileNotFoundError(f"T file not found: {t_file}")
    return read_foam_scalar_field(t_file)

def get_heater_temperature(folder_name):
    match = re.search(r"T(\d+)K", folder_name)
    if match:
        return float(match.group(1))
    raise ValueError(f"Could not extract temperature from: {folder_name}")

def extract_all_cases():
    print("=" * 60)
    print("OpenFOAM Multi-Case Dataset Extractor")
    print("=" * 60)

    case_folders = sorted([
        f for f in os.listdir(CASES_DIR)
        if f.startswith("case_") and os.path.isdir(os.path.join(CASES_DIR, f))
    ])
    print(f"\nFound {len(case_folders)} case folders")

    all_coords   = []
    all_times    = []
    all_T_heater = []
    all_T_steel  = []
    cell_coords  = None

    for case_folder in case_folders:
        case_dir = os.path.join(CASES_DIR, case_folder)
        T_heater = get_heater_temperature(case_folder)
        print(f"\n  Processing {case_folder} (T_heater = {T_heater} K)...")

        time_steps = get_time_steps(case_dir)
        print(f"    Time steps: {time_steps}")

        for t in time_steps:
            try:
                T_values = read_temperature_field(case_dir, TARGET_REGION, t)
                n_cells = len(T_values)

                # Build dummy coordinates if mesh read failed
                if cell_coords is None or len(cell_coords) == 0:
                    points_file = os.path.join(case_dir, "constant", TARGET_REGION, "polyMesh", "points")
                    if os.path.exists(points_file):
                        cell_coords = read_foam_points(points_file)
                    else:
                        cell_coords = np.zeros((n_cells, 3), dtype=np.float32)

                # Tile or trim coords to match n_cells
                if len(cell_coords) >= n_cells:
                    coords_use = cell_coords[:n_cells]
                else:
                    repeats = int(np.ceil(n_cells / len(cell_coords)))
                    coords_use = np.tile(cell_coords, (repeats, 1))[:n_cells]

                all_coords.append(coords_use)
                all_times.append(np.full(n_cells, t, dtype=np.float32))
                all_T_heater.append(np.full(n_cells, T_heater, dtype=np.float32))
                all_T_steel.append(T_values)

            except Exception as e:
                print(f"    WARNING t={t}: {e}")

        print(f"    Done {case_folder}")

    print("\n" + "=" * 60)
    print("Building dataset...")

    coords_all   = np.vstack(all_coords)
    times_all    = np.concatenate(all_times)
    T_heater_all = np.concatenate(all_T_heater)
    T_steel_all  = np.concatenate(all_T_steel)

    N = len(T_steel_all)
    print(f"Total samples : {N:,}")
    print(f"Coords shape  : {coords_all.shape}")
    print(f"T range       : {T_steel_all.min():.1f} - {T_steel_all.max():.1f} K")

    x_mean = coords_all.mean(axis=0)
    x_std  = coords_all.std(axis=0) + 1e-8
    t_mean = times_all.mean()
    t_std  = times_all.std() + 1e-8
    Th_mean = T_heater_all.mean()
    Th_std  = T_heater_all.std() + 1e-8

    coords_norm = (coords_all - x_mean) / x_std
    times_norm  = (times_all  - t_mean) / t_std
    Th_norm     = (T_heater_all - Th_mean) / Th_std

    X = np.column_stack([
        coords_norm,
        times_norm.reshape(-1, 1),
        Th_norm.reshape(-1, 1)
    ]).astype(np.float32)

    Y = T_steel_all.reshape(-1, 1).astype(np.float32)

    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")

    print(f"\nSaving to {OUTPUT_H5}...")
    with h5py.File(OUTPUT_H5, "w") as f:
        f.create_dataset("X", data=X)
        f.create_dataset("Y", data=Y)
        f.create_dataset("x_mean", data=x_mean.astype(np.float32))
        f.create_dataset("x_std",  data=x_std.astype(np.float32))
        f.create_dataset("t_mean", data=np.array([t_mean], dtype=np.float32))
        f.create_dataset("t_std",  data=np.array([t_std],  dtype=np.float32))
        f.create_dataset("T_heater_mean", data=np.array([Th_mean], dtype=np.float32))
        f.create_dataset("T_heater_std",  data=np.array([Th_std],  dtype=np.float32))
        f.attrs["n_samples"] = N

    print("Dataset saved successfully!")
    print("=" * 60)

if __name__ == "__main__":
    extract_all_cases()

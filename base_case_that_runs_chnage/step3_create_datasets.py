#!/usr/bin/env python3
"""
Step 3: Create .h5 datasets from VTK files
Run AFTER OpenFOAM simulations finish
"""

import os
import json
import glob
import numpy as np
import h5py
import pyvista as pv

BASE_CASE = "/workspace/rise_furnace/base_case_that_runs_chnage"
OUTPUT_DIR = "/workspace/rise_furnace/parameter_study"


def read_simulation(case_dir, params):
    vtk_dir = os.path.join(case_dir, "VTK")
    if not os.path.exists(vtk_dir):
        print(f"    VTK/ not found in {case_dir}")
        return None

    series_files = glob.glob(os.path.join(vtk_dir, "*.series"))
    if not series_files:
        print(f"    No .series file in {vtk_dir}")
        return None

    with open(series_files[0]) as f:
        series = json.load(f)

    entries = series["files"]
    file_key = "file" if "file" in entries[0] else "name"

    vtm_files = [os.path.join(vtk_dir, e[file_key]) for e in entries]
    times = np.array([float(e.get("time", i)) for i, e in enumerate(entries)], dtype=np.float32)

    mb0 = pv.read(vtm_files[0])
    steel_key = None
    for k in mb0.keys():
        if "steel" in k.lower():
            steel_key = k
            break
    if steel_key is None:
        print(f"    Steel region not found")
        return None

    steel_mb0 = mb0[steel_key]
    internal_key = None
    for k in steel_mb0.keys():
        if "internal" in k.lower():
            internal_key = k
            break
    if internal_key is None:
        for k, block in steel_mb0.items():
            if isinstance(block, pv.UnstructuredGrid):
                internal_key = k
                break
    if internal_key is None:
        print(f"    Internal block not found")
        return None

    coords = steel_mb0[internal_key].cell_centers().points.astype(np.float32)

    all_T = []
    for vtm_path in vtm_files:
        mb = pv.read(vtm_path)
        T = mb[steel_key][internal_key]["T"].astype(np.float32)
        all_T.append(T)
    all_T = np.stack(all_T, axis=0)

    h5_path = os.path.join(case_dir, "steel_cylinder_T_timeseries.h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("coords", data=coords)
        f.create_dataset("times", data=times)
        f.create_dataset("T", data=all_T)
        for key, val in params.items():
            f.attrs[key] = val

    print(f"    Saved: {h5_path}")
    print(f"    {coords.shape[0]} cells, {len(times)} times, T=[{all_T.min():.1f}, {all_T.max():.1f}] K")

    return {
        "coords": coords, "times": times, "T": all_T,
        "T_set": float(params["T_set"]), "kappa": float(params["kappa"]),
        "Cp": float(params["Cp"]), "rho": float(params["rho"]),
        "h5_path": h5_path,
    }


def main():
    print("=" * 60)
    print("STEP 3: CREATING DATASETS FROM VTK FILES")
    print("=" * 60)

    manifest_path = os.path.join(OUTPUT_DIR, "case_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_data = []

    print(f"\nLoading base case (T_set=1000)...")
    base_h5 = os.path.join(BASE_CASE, "steel_cylinder_T_timeseries.h5")
    if os.path.isfile(base_h5):
        with h5py.File(base_h5, "r") as f:
            base = {
                "coords": f["coords"][:].astype(np.float32),
                "times": f["times"][:].astype(np.float32),
                "T": f["T"][:].astype(np.float32),
                "T_set": 1000.0, "kappa": 80.0, "Cp": 450.0, "rho": 7800.0,
                "h5_path": base_h5,
            }
        all_data.append(base)
        print(f"  OK: T={base['T'].shape}")
    else:
        print(f"  Base case .h5 not found!")
        return

    for m in manifest:
        if m.get("status") == "completed" and m["case"] == "base_case_that_runs_chnage":
            continue
        if m["case"] == "base_case_that_runs_chnage":
            continue

        case_dir = os.path.join(OUTPUT_DIR, m["case"])
        print(f"\nProcessing: {m['case']}...")

        vtk_dir = os.path.join(case_dir, "VTK")
        if not os.path.exists(vtk_dir):
            print(f"  No VTK/ — skipping")
            continue

        params = {"T_set": m["T_set"], "kappa": m["kappa"], "Cp": m["Cp"], "rho": m["rho"]}
        sim = read_simulation(case_dir, params)
        if sim:
            all_data.append(sim)
            m["status"] = "completed"

    if len(all_data) > 1:
        print(f"\n{'='*60}")
        print(f"CREATING COMBINED DATASET")
        print(f"{'='*60}")

        all_X, all_Y = [], []
        for sim in all_data:
            coords, times, T_data = sim["coords"], sim["times"], sim["T"]
            n_cells = coords.shape[0]
            for ti in range(len(times)):
                X_block = np.stack([
                    coords[:, 0], coords[:, 1], coords[:, 2],
                    np.full(n_cells, times[ti], dtype=np.float32),
                    np.full(n_cells, sim["T_set"], dtype=np.float32),
                    np.full(n_cells, sim["kappa"], dtype=np.float32),
                    np.full(n_cells, sim["Cp"], dtype=np.float32),
                    np.full(n_cells, sim["rho"], dtype=np.float32),
                ], axis=1)
                all_X.append(X_block)
                all_Y.append(T_data[ti, :].reshape(-1, 1))

        X = np.concatenate(all_X, axis=0).astype(np.float32)
        Y = np.concatenate(all_Y, axis=0).astype(np.float32)

        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        Y_mean = float(Y.mean())
        Y_std = float(Y.std()) + 1e-8
        X_norm = ((X - X_mean) / X_std).astype(np.float32)
        Y_norm = ((Y - Y_mean) / Y_std).astype(np.float32)

        cols = ["x", "y", "z", "t", "T_set", "kappa", "Cp", "rho"]
        path = os.path.join(OUTPUT_DIR, "combined_dataset.h5")
        with h5py.File(path, "w") as f:
            f.create_dataset("X_raw", data=X)
            f.create_dataset("Y_raw", data=Y)
            f.create_dataset("X_norm", data=X_norm)
            f.create_dataset("Y_norm", data=Y_norm)
            f.create_dataset("X_mean", data=X_mean)
            f.create_dataset("X_std", data=X_std)
            f.create_dataset("Y_mean", data=np.float32(Y_mean))
            f.create_dataset("Y_std", data=np.float32(Y_std))
            f.attrs["columns"] = json.dumps(cols)
            f.attrs["n_simulations"] = len(all_data)
            f.attrs["total_points"] = X.shape[0]

        print(f"\n  Combined: {X.shape} -> {Y.shape}")
        print(f"  Columns: {cols}")
        for sim in all_data:
            print(f"    T_set={sim['T_set']:.0f}, k={sim['kappa']}, Cp={sim['Cp']}, rho={sim['rho']}")
        print(f"  Saved: {path}")
        print(f"  Total: {X.shape[0]:,} points")
    elif len(all_data) == 1:
        print(f"\nOnly base case available. Need OpenFOAM results first!")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE! Next: python3 train_pinn_with_Tset.py")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()

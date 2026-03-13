#!/usr/bin/env python3
"""
Step 3 (v2): Build training dataset for:

  T(x, y, z, t, T_set, cx, cy, cz, radius, height, volume, mass, kappa, Cp, rho)

Cylinder geometry from .geo:
  Disk(45) = {cx=0, cy, cz, radius, radius}
  Rotate Pi/2 around y-axis
  Extrude {height, 0, 0}

Feature columns (15 total):
  x, y, z          - spatial position of cell center [m]
  t                 - simulation time [s]
  T_set             - heater temperature [K]
  cx, cy, cz        - cylinder disk center position [m]
  radius            - cylinder radius [m]
  height            - cylinder extrusion length along x [m]
  volume            - pi*r^2*h [m^3]   (derived, captures size)
  mass              - rho*volume [kg]  (derived, captures thermal inertia)
  kappa             - thermal conductivity [W/m·K]
  Cp                - specific heat [J/kg·K]
  rho               - density [kg/m^3]

Target (1):
  T                 - temperature [K]
"""

import os
import re
import json
import glob
import math
import numpy as np
import h5py
import pyvista as pv

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
BASE_CASE  = "/home/openfoam/rise_furnace/base_case_that_runs_chnage"
OUTPUT_DIR = "/home/openfoam/rise_furnace/parameter_study_v3"

# -------------------------------------------------------
# Feature / target definition
# (order here = column order in X matrix — never change
#  after training starts, or models become incompatible)
# -------------------------------------------------------
FEATURE_COLS = [
    # --- coordinates ---
    "x", "y", "z",
    # --- time ---
    "t",
    # --- boundary condition ---
    "T_set",
    # --- cylinder position (from Disk(45) center) ---
    "cx", "cy", "cz",
    # --- cylinder geometry ---
    "radius", "height",
    # --- derived geometry (help the model learn scale) ---
    "volume", "mass",
    # --- material properties ---
    "kappa", "Cp", "rho",
]
TARGET_COL = "T"

N_FEATURES = len(FEATURE_COLS)   # = 15

# -------------------------------------------------------
# Base case cylinder params  (from .geo file)
# -------------------------------------------------------
BASE_CYLINDER = {
    "T_set":  1000.0,
    "cx":     0.0,      # Fixed at x=0
    "cy":     0.18,
    "cz":     0.195,
    "radius": 0.05,
    "height": 0.10,
    "kappa":  80.0,
    "Cp":     450.0,
    "rho":    7800.0,
}
BASE_CYLINDER["volume"] = math.pi * BASE_CYLINDER["radius"]**2 * BASE_CYLINDER["height"]
BASE_CYLINDER["mass"]   = BASE_CYLINDER["rho"] * BASE_CYLINDER["volume"]


# -------------------------------------------------------
# VTK helpers
# -------------------------------------------------------
def find_steel_internal(multiblock: pv.MultiBlock):
    """
    Navigate: top-level -> steel block -> internal mesh (UnstructuredGrid).
    Returns UnstructuredGrid or None.
    """
    # Find steel region block
    steel_key = None
    for k in multiblock.keys():
        if "steel" in k.lower():
            steel_key = k
            break
    if steel_key is None:
        return None, None

    steel_mb = multiblock[steel_key]

    # Find internal mesh inside the steel block
    for k in steel_mb.keys():
        if "internal" in k.lower():
            return steel_key, steel_mb[k]

    # Fallback: first UnstructuredGrid
    for k in steel_mb.keys():
        blk = steel_mb[k]
        if isinstance(blk, pv.UnstructuredGrid):
            return steel_key, blk

    return steel_key, None


def read_steel_timeseries(case_dir: str):
    """
    Read steel_cylinder temperature time series from VTK output.

    Returns:
        coords  : np.ndarray (n_cells, 3)   cell-center coordinates
        times   : np.ndarray (n_times,)     simulation times [s]
        T_array : np.ndarray (n_times, n_cells)  temperature [K]
    or None on failure.
    """
    vtk_dir = os.path.join(case_dir, "VTK")
    if not os.path.exists(vtk_dir):
        print(f"    [WARN] No VTK/ in {case_dir}")
        return None

    series_files = glob.glob(os.path.join(vtk_dir, "*.series"))
    if not series_files:
        print(f"    [WARN] No .series file in {vtk_dir}")
        # Try to find any .vtm file as fallback
        vtm_files = glob.glob(os.path.join(vtk_dir, "*.vtm"))
        if vtm_files:
            print(f"    Found {len(vtm_files)} .vtm files, using first")
            # Create a simple series file
            entries = [{"name": os.path.basename(f), "time": float(i)} 
                      for i, f in enumerate(sorted(vtm_files))]
            series_path = os.path.join(vtk_dir, "case.series")
            with open(series_path, "w") as f:
                json.dump({"files": entries}, f)
            series_files = [series_path]
        else:
            return None

    with open(series_files[0]) as f:
        series = json.load(f)

    entries  = series["files"]
    fkey     = "file" if "file" in entries[0] else "name"
    vtm_list = [os.path.join(vtk_dir, e[fkey]) for e in entries]
    times    = np.array(
        [float(e.get("time", i)) for i, e in enumerate(entries)],
        dtype=np.float64,
    )

    print(f"    Time steps : {len(times)}  "
          f"t=[{times[0]:.1f}, {times[-1]:.1f}] s")

    # ---- Coordinates from t=0 ----
    mb0           = pv.read(vtm_list[0])
    steel_key, ug = find_steel_internal(mb0)
    if ug is None:
        print(f"    [ERROR] Steel internal block not found")
        print(f"    Available blocks: {list(mb0.keys())}")
        return None

    coords = ug.cell_centers().points.astype(np.float64)
    print(f"    Steel cells: {coords.shape[0]}")

    # ---- T over all time steps ----
    T_frames = []
    for vtm_path in vtm_list:
        mb  = pv.read(vtm_path)
        _, ug_t = find_steel_internal(mb)
        if ug_t is None or "T" not in ug_t.cell_data:
            print(f"    [WARN] Missing T in {os.path.basename(vtm_path)}")
            T_frames.append(np.full(coords.shape[0], np.nan, dtype=np.float64))
        else:
            T_frames.append(ug_t.cell_data["T"].astype(np.float64))

    T_array = np.stack(T_frames, axis=0)   # (n_times, n_cells)
    print(f"    T range    : [{np.nanmin(T_array):.1f}, "
          f"{np.nanmax(T_array):.1f}] K")

    return coords, times, T_array


# -------------------------------------------------------
# Load cylinder params
# -------------------------------------------------------
def load_cylinder_params(case_dir: str, manifest_entry: dict = None) -> dict:
    """
    Priority:
      1. cylinder_params.json  (written by step1, most accurate)
      2. manifest entry
      3. BASE_CYLINDER fallback
    """
    json_path = os.path.join(case_dir, "cylinder_params.json")
    if os.path.isfile(json_path):
        with open(json_path) as f:
            p = json.load(f)
        # Ensure derived fields
        if "volume" not in p or "mass" not in p:
            if "radius" in p and "height" in p and "rho" in p:
                p["volume"] = math.pi * p["radius"]**2 * p["height"]
                p["mass"] = p["rho"] * p["volume"]
        return p

    if manifest_entry is not None:
        p = {k: float(manifest_entry[k])
             for k in FEATURE_COLS
             if k not in ("x","y","z","t") and k in manifest_entry}
        if "volume" not in p and "radius" in p and "height" in p:
            p["volume"] = math.pi * p["radius"]**2 * p["height"]
        if "mass" not in p and "rho" in p and "volume" in p:
            p["mass"] = p["rho"] * p["volume"]
        return p

    print(f"    [WARN] No params found — using BASE_CYLINDER")
    return dict(BASE_CYLINDER)


# -------------------------------------------------------
# Build feature matrix for one simulation
# -------------------------------------------------------
def build_feature_matrix(
    coords: np.ndarray,
    times:  np.ndarray,
    T_array: np.ndarray,
    cyl: dict,
) -> tuple:
    """
    Returns:
        X : (N, N_FEATURES)  float32
        Y : (N, 1)           float32
    where N = n_cells * n_times
    """
    n_cells = coords.shape[0]
    blocks_X, blocks_Y = [], []

    for ti, t_val in enumerate(times):

        # Skip time steps with NaN temperature
        if np.any(np.isnan(T_array[ti])):
            continue

        # Build feature matrix with proper dtypes
        X_block = np.column_stack([
            # spatial - keep as float64 for precision
            coords[:, 0],                                               # x
            coords[:, 1],                                               # y
            coords[:, 2],                                               # z
            # temporal
            np.full(n_cells, t_val,           dtype=np.float64),       # t
            # boundary condition
            np.full(n_cells, cyl["T_set"],    dtype=np.float32),       # T_set
            # cylinder position
            np.full(n_cells, cyl.get("cx", 0.0), dtype=np.float64),    # cx
            np.full(n_cells, cyl["cy"],       dtype=np.float64),       # cy
            np.full(n_cells, cyl["cz"],       dtype=np.float64),       # cz
            # cylinder geometry
            np.full(n_cells, cyl["radius"],   dtype=np.float64),       # radius
            np.full(n_cells, cyl["height"],   dtype=np.float64),       # height
            # derived
            np.full(n_cells, cyl["volume"],   dtype=np.float64),       # volume
            np.full(n_cells, cyl["mass"],     dtype=np.float64),       # mass
            # material
            np.full(n_cells, cyl["kappa"],    dtype=np.float32),       # kappa
            np.full(n_cells, cyl["Cp"],       dtype=np.float32),       # Cp
            np.full(n_cells, cyl["rho"],      dtype=np.float32),       # rho
        ]).astype(np.float32)  # Convert to float32 for ML efficiency

        Y_block = T_array[ti, :].reshape(-1, 1).astype(np.float32)

        blocks_X.append(X_block)
        blocks_Y.append(Y_block)

    X = np.concatenate(blocks_X, axis=0)
    Y = np.concatenate(blocks_Y, axis=0)
    return X, Y


# -------------------------------------------------------
# Save per-case HDF5  (coords + times + T, with params)
# -------------------------------------------------------
def save_case_h5(case_dir, coords, times, T_array, cyl):
    h5_path = os.path.join(case_dir, "steel_cylinder_T_timeseries.h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("coords",  data=coords)
        f.create_dataset("times",   data=times)
        f.create_dataset("T",       data=T_array)
        for k, v in cyl.items():
            f.attrs[k] = float(v)
    print(f"    Cached: {h5_path}")
    return h5_path


def load_case_h5(case_dir):
    """Load pre-cached per-case HDF5."""
    h5_path = os.path.join(case_dir, "steel_cylinder_T_timeseries.h5")
    if not os.path.isfile(h5_path):
        return None
    with h5py.File(h5_path) as f:
        coords  = f["coords"][:].astype(np.float64)
        times   = f["times"][:].astype(np.float64)
        T_array = f["T"][:].astype(np.float64)
        cyl     = {k: float(v) for k, v in f.attrs.items()}
    return coords, times, T_array, cyl


# -------------------------------------------------------
# Print stats table
# -------------------------------------------------------
def print_stats(X, Y):
    print(f"\n  {'Feature':<12} {'Mean':>10} {'Std':>10} "
          f"{'Min':>10} {'Max':>10}")
    print("  " + "-" * 54)
    for i, col in enumerate(FEATURE_COLS):
        print(f"  {col:<12} {X[:,i].mean():>10.4g} "
              f"{X[:,i].std():>10.4g} "
              f"{X[:,i].min():>10.4g} "
              f"{X[:,i].max():>10.4g}")
    print(f"  {'T (target)':<12} {Y.mean():>10.4g} "
          f"{Y.std():>10.4g} "
          f"{Y.min():>10.4g} "
          f"{Y.max():>10.4g}")


# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    print("=" * 60)
    print("STEP 3 (v2): DATASET WITH CYLINDER FEATURES")
    print("=" * 60)
    print(f"\nFeatures ({N_FEATURES}):")
    for i, c in enumerate(FEATURE_COLS):
        print(f"  [{i:02d}] {c}")
    print(f"Target : {TARGET_COL}")

    # ---- Load manifest ----
    manifest_path = os.path.join(OUTPUT_DIR, "case_manifest.json")
    with open(manifest_path) as f:
        manifest = json.load(f)

    all_X, all_Y = [], []
    n_ok         = 0
    case_summary = []

    # ================================================================
    # 1.  Base case
    # ================================================================
    print(f"\n{'='*40}")
    print("Base case (T_set=1000, base geometry)")
    print(f"{'='*40}")

    # Try cache first
    cached = load_case_h5(BASE_CASE)
    if cached:
        coords, times, T_array, cyl_loaded = cached
        # Merge with BASE_CYLINDER (cache may not have all keys)
        cyl = dict(BASE_CYLINDER)
        cyl.update(cyl_loaded)
        print(f"  Loaded from HDF5 cache")
    else:
        result = read_steel_timeseries(BASE_CASE)
        if result is None:
            print("  [ERROR] Cannot load base case — aborting!")
            return
        coords, times, T_array = result
        cyl = dict(BASE_CYLINDER)
        save_case_h5(BASE_CASE, coords, times, T_array, cyl)

    X_b, Y_b = build_feature_matrix(coords, times, T_array, cyl)
    all_X.append(X_b)
    all_Y.append(Y_b)
    n_ok += 1
    case_summary.append({"case": "base", **cyl, "n_rows": X_b.shape[0]})
    print(f"  Rows: {X_b.shape[0]:,}")

    # ================================================================
    # 2.  Parameter study cases
    # ================================================================
    for m in manifest:
        if m["case"] == "base_case_that_runs_chnage":
            continue

        case_dir = os.path.join(OUTPUT_DIR, m["case"])

        print(f"\n{'='*40}")
        print(f"[{m['idx']:03d}] {m['case']}")

        # ---- Try cached HDF5 ----
        cached = load_case_h5(case_dir)
        if cached:
            coords, times, T_array, cyl_loaded = cached
            cyl = load_cylinder_params(case_dir, m)
            cyl.update(cyl_loaded)   # HDF5 attrs take priority
            print(f"  Loaded from HDF5 cache")
        else:
            # ---- Check VTK exists ----
            result = read_steel_timeseries(case_dir)
            if result is None:
                print(f"  [ERROR] VTK read failed")
                continue
            coords, times, T_array = result
            cyl = load_cylinder_params(case_dir, m)
            save_case_h5(case_dir, coords, times, T_array, cyl)

        print(f"  T_set={cyl['T_set']:.0f}K | "
              f"cx={cyl.get('cx', 0.0):.3f} cy={cyl['cy']:.3f} cz={cyl['cz']:.3f} | "
              f"r={cyl['radius']*1e3:.1f}mm h={cyl['height']*1e3:.1f}mm | "
              f"k={cyl['kappa']:.0f} Cp={cyl['Cp']:.0f} rho={cyl['rho']:.0f} | "
              f"V={cyl['volume']*1e6:.2f}cm³")

        X_i, Y_i = build_feature_matrix(coords, times, T_array, cyl)
        all_X.append(X_i)
        all_Y.append(Y_i)
        n_ok += 1
        m["status"] = "completed"
        case_summary.append({"case": m["case"], **cyl, "n_rows": X_i.shape[0]})
        print(f"  Rows: {X_i.shape[0]:,}")

    # ================================================================
    # 3.  Combine & normalise
    # ================================================================
    if n_ok == 0:
        print("\n[ERROR] No simulation data loaded!")
        return

    print(f"\n{'='*60}")
    print(f"COMBINING {n_ok} simulations ...")

    X = np.concatenate(all_X, axis=0).astype(np.float32)
    Y = np.concatenate(all_Y, axis=0).astype(np.float32)

    # Normalisation stats
    X_mean = X.mean(axis=0).astype(np.float32)
    X_std  = (X.std(axis=0) + 1e-8).astype(np.float32)
    Y_mean = float(Y.mean())
    Y_std  = float(Y.std()) + 1e-8

    X_norm = ((X - X_mean) / X_std).astype(np.float32)
    Y_norm = ((Y - Y_mean) / Y_std).astype(np.float32)

    # Print stats
    print_stats(X, Y)

    # ================================================================
    # 4.  Save combined HDF5
    # ================================================================
    out_path = os.path.join(OUTPUT_DIR, "dataset_cylinder_features.h5")
    with h5py.File(out_path, "w") as f:

        # ---- Raw data ----
        f.create_dataset("X_raw",  data=X, compression="gzip", chunks=True)
        f.create_dataset("Y_raw",  data=Y, compression="gzip", chunks=True)

        # ---- Normalised data ----
        f.create_dataset("X_norm", data=X_norm, compression="gzip", chunks=True)
        f.create_dataset("Y_norm", data=Y_norm, compression="gzip", chunks=True)

        # ---- Normalisation stats (REQUIRED for inference) ----
        f.create_dataset("X_mean", data=X_mean)
        f.create_dataset("X_std",  data=X_std)
        f.create_dataset("Y_mean", data=np.float32(Y_mean))
        f.create_dataset("Y_std",  data=np.float32(Y_std))

        # ---- Metadata ----
        f.attrs["feature_cols"]     = json.dumps(FEATURE_COLS)
        f.attrs["target_col"]       = TARGET_COL
        f.attrs["n_simulations"]    = n_ok
        f.attrs["total_points"]     = int(X.shape[0])
        f.attrs["n_features"]       = N_FEATURES
        f.attrs["case_summary"]     = json.dumps(case_summary)

        # ---- Per-simulation index (for train/val split by simulation) ----
        # Store start index of each simulation so you can split without
        # data leakage (don't split mid-simulation!)
        sim_starts = np.array(
            [0] + list(np.cumsum([b.shape[0] for b in all_X[:-1]])),
            dtype=np.int64
        )
        f.create_dataset("sim_start_indices", data=sim_starts)
        f.create_dataset(
            "sim_n_rows",
            data=np.array([b.shape[0] for b in all_X], dtype=np.int64)
        )

    print(f"\n  Saved : {out_path}")
    print(f"  Shape : X={X.shape}, Y={Y.shape}")
    print(f"  Size  : {os.path.getsize(out_path)/1e6:.1f} MB")

    # ---- Update manifest ----
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ================================================================
    # 5.  Summary
    # ================================================================
    print(f"\n{'='*60}")
    print(f"DONE!  {n_ok} simulations -> {X.shape[0]:,} training points")
    print(f"{'='*60}")
    print(f"\nDataset file: {out_path}")
    print(f"\nFeature index reference (for model input):")
    for i, c in enumerate(FEATURE_COLS):
        print(f"  X[:, {i:02d}] = {c}")
    print(f"\nHow to load in training script:")
    print(f"""
  import h5py, json
  with h5py.File("{out_path}") as f:
      X_norm       = f["X_norm"][:]
      Y_norm       = f["Y_norm"][:]
      X_mean       = f["X_mean"][:]
      X_std        = f["X_std"][:]
      Y_mean       = float(f["Y_mean"][()])
      Y_std        = float(f["Y_std"][()])
      feature_cols = json.loads(f.attrs["feature_cols"])
      sim_starts   = f["sim_start_indices"][:]

  # Train/val split by simulation (avoid data leakage)
  n_sims    = len(sim_starts)
  val_sims  = [n_sims - 1]   # last simulation as validation
  val_mask  = np.zeros(len(X_norm), dtype=bool)
  for s in val_sims:
      start = sim_starts[s]
      end   = sim_starts[s+1] if s+1 < n_sims else len(X_norm)
      val_mask[start:end] = True

  X_train, Y_train = X_norm[~val_mask], Y_norm[~val_mask]
  X_val,   Y_val   = X_norm[val_mask],  Y_norm[val_mask]
    """)
    print(f"\nNext: python3 train_pinn_with_cylinder.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
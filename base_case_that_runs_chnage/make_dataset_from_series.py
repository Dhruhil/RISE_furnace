#!/usr/bin/env python3
"""
make_dataset_from_series.py

Usage: run from the case directory, e.g.
    python3 make_dataset_from_series.py

Requires: pyvista, h5py, numpy
"""

import os
import json
import numpy as np
import pyvista as pv
import h5py


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    case_name = os.path.basename(base_dir.rstrip(os.sep))

    # Path to the global foamToVTK time-series JSON
    series_path = os.path.join(base_dir, "VTK", f"{case_name}-regions.vtm.series")
    if not os.path.isfile(series_path):
        raise FileNotFoundError(f"Cannot find series file: {series_path}")

    # --- Read time-series description ---
    with open(series_path, "r") as f:
        series = json.load(f)

    entries = series["files"]
    if not entries:
        raise RuntimeError("No entries in VTK time-series file.")

    # Determine which key stores the VTM filename ('file' or 'name')
    file_key = None
    for k in ("file", "name"):
        if k in entries[0]:
            file_key = k
            break
    if file_key is None:
        raise KeyError(f"No 'file' or 'name' key in time-series entries: {entries[0]}")

    # Time key (usually 'time')
    time_key = "time" if "time" in entries[0] else None

    vtk_dir = os.path.dirname(series_path)
    vtm_files = []
    times = []

    for i, e in enumerate(entries):
        fname = e[file_key]
        vtm_files.append(os.path.join(vtk_dir, fname))
        if time_key is not None and time_key in e:
            times.append(float(e[time_key]))
        else:
            times.append(float(i))

    times = np.array(times, dtype=np.float32)
    n_times = len(times)

    print(f"Found {n_times} time steps")
    print("First 3 VTM files:")
    for vf in vtm_files[:3]:
        print("  ", vf)

    # --- Inspect first VTM to find region and internal block ---
    mb0 = pv.read(vtm_files[0])
    if not isinstance(mb0, pv.MultiBlock):
        raise RuntimeError(f"Expected a MultiBlock at top-level, got {type(mb0)}")

    region_keys = list(mb0.keys())
    print("Top-level regions:", region_keys)

    # Choose the steel region: look for 'steel' in the region name
    steel_region_key = None
    for k in region_keys:
        if "steel" in k.lower():
            steel_region_key = k
            break
    if steel_region_key is None:
        steel_region_key = "steel_cylinder"
        if steel_region_key not in region_keys:
            raise KeyError(f"Cannot find steel region. Available regions: {region_keys}")

    print("Using region:", repr(steel_region_key))

    steel_mb0 = mb0[steel_region_key]
    subblock_keys = list(steel_mb0.keys())
    print("Sub-blocks in steel region:", subblock_keys)

    # Choose internal volume block: look for 'internal'
    internal_key = None
    for k in subblock_keys:
        if "internal" in k.lower():
            internal_key = k
            break

    # If not found, fall back to first UnstructuredGrid
    if internal_key is None:
        for k, block in steel_mb0.items():
            if isinstance(block, pv.UnstructuredGrid):
                internal_key = k
                break

    if internal_key is None:
        raise RuntimeError(
            f"Could not find an internal volume block in region '{steel_region_key}'. "
            f"Sub-blocks: {subblock_keys}"
        )

    print("Using internal block:", repr(internal_key))

    # --- Geometry from first time step ---
    ugrid0 = steel_mb0[internal_key]
    centers0 = ugrid0.cell_centers()
    coords = centers0.points.astype(np.float32)  # (N_cells, 3)
    n_cells = coords.shape[0]

    # --- Extract T for all time steps ---
    all_T = []

    for vtm_path, t in zip(vtm_files, times):
        print(f"Reading {vtm_path} at time={t}")
        mb = pv.read(vtm_path)
        steel_mb = mb[steel_region_key]
        ugrid = steel_mb[internal_key]

        centers = ugrid.cell_centers().points.astype(np.float32)
        if centers.shape != coords.shape or not np.allclose(centers, coords):
            raise RuntimeError(
                "Cell centers changed between time steps; mesh is not fixed in time."
            )

        if "T" not in ugrid.array_names:
            raise RuntimeError(
                f"No 'T' field found in {vtm_path}. "
                f"Available arrays: {list(ugrid.array_names)}"
            )

        T = ugrid["T"].astype(np.float32)  # (N_cells,)
        if T.shape[0] != n_cells:
            raise RuntimeError(
                f"Length of T ({T.shape[0]}) != n_cells ({n_cells}) in {vtm_path}"
            )

        all_T.append(T)

    all_T = np.stack(all_T, axis=0)  # (N_t, N_cells)

    print("coords shape:", coords.shape)
    print("times shape:", times.shape)
    print("T shape:", all_T.shape)

    # --- Save time-series HDF5: coords, times, T ---
    ts_path = os.path.join(base_dir, "steel_cylinder_T_timeseries.h5")
    with h5py.File(ts_path, "w") as f:
        f.create_dataset("coords", data=coords)   # (N_cells, 3)
        f.create_dataset("times", data=times)     # (N_t,)
        f.create_dataset("T", data=all_T)         # (N_t, N_cells)

    print("Wrote", ts_path)

    # --- Flatten into supervised dataset X=[x,y,z,t], Y=T ---
    N_t, N_cells = all_T.shape

    # Repeat coords for each time, repeat times for each cell
    coords_rep = np.repeat(coords[None, :, :], N_t, axis=0)       # (N_t, N_cells, 3)
    times_rep  = np.repeat(times[:, None, None], N_cells, axis=1)  # (N_t, N_cells, 1)

    X = np.concatenate([coords_rep, times_rep], axis=2).reshape(-1, 4)  # (N_t*N_cells, 4)
    x = X[:, :3]
    tcol = X[:, 3:4]

    # Normalization
    x_mean = x.mean(axis=0)
    x_std  = x.std(axis=0)
    t_mean = float(tcol.mean())
    t_std  = float(tcol.std())

    x_hat = (x - x_mean) / x_std
    t_hat = (tcol - t_mean) / t_std

    X_norm = np.hstack([x_hat, t_hat])   # (N_t*N_cells, 4)
    Y = all_T.reshape(-1, 1)             # (N_t*N_cells, 1)

    # --- Save flat dataset ---
    flat_path = os.path.join(base_dir, "flat_XY_dataset.h5")
    with h5py.File(flat_path, "w") as f:
        f.create_dataset("X", data=X_norm)
        f.create_dataset("Y", data=Y)
        f.create_dataset("x_mean", data=x_mean)
        f.create_dataset("x_std", data=x_std)
        f.create_dataset("t_mean", data=np.array(t_mean, dtype=np.float32))
        f.create_dataset("t_std", data=np.array(t_std, dtype=np.float32))

    print("Wrote", flat_path)


if __name__ == "__main__":
    main()
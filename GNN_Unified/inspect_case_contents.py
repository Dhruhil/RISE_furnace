"""Show all features/data stored per case in the HDF5 file."""
import h5py
import numpy as np
import re

H5_PATH = "dataset_v2_all_regions_clean.h5"

with h5py.File(H5_PATH, "r") as f:
    case_keys = sorted([k for k in f.keys() if k.startswith("case_")])
    print(f"Total cases: {len(case_keys)}\n")
    print("=" * 80)

    # Show first 3 cases in full detail
    for ck in case_keys[:3]:
        grp = f[ck]
        print(f"\n### {ck} ###")

        # --- Metadata stored as HDF5 attributes ---
        print("\n  [ATTRIBUTES / metadata]")
        for k, v in grp.attrs.items():
            print(f"    {k} = {v}")

        # --- Parse REAL cylinder params from the name ---
        name = str(grp.attrs.get("name", ""))
        cx_m = re.search(r"cx(-?\d+)mm", name)
        cy_m = re.search(r"cy(\d+)mm", name)
        cz_m = re.search(r"cz(\d+)mm", name)
        r_m  = re.search(r"r(\d+)mm", name)
        h_m  = re.search(r"h(\d+)mm", name)
        k_m  = re.search(r"k(\d+)", name)
        cp_m = re.search(r"Cp(\d+)", name)
        rh_m = re.search(r"rho(\d+)", name)

        print("\n  [PARSED FROM NAME — true per-case parameters]")
        if cx_m: print(f"    cx    = {int(cx_m.group(1))/1000:.3f} m")
        if cy_m: print(f"    cy    = {int(cy_m.group(1))/1000:.3f} m")
        if cz_m: print(f"    cz    = {int(cz_m.group(1))/1000:.3f} m")
        if r_m:  print(f"    radius= {int(r_m.group(1))/1000:.3f} m")
        if h_m:  print(f"    height= {int(h_m.group(1))/1000:.3f} m")
        if k_m:  print(f"    kappa = {int(k_m.group(1))} W/m/K")
        if cp_m: print(f"    Cp    = {int(cp_m.group(1))} J/kg/K")
        if rh_m: print(f"    rho   = {int(rh_m.group(1))} kg/m^3")

        # --- Datasets stored per case ---
        print("\n  [DATASETS per case]")
        if "times" in grp:
            t = grp["times"][:]
            print(f"    times:  shape={t.shape}  range=[{t[0]:.1f}, {t[-1]:.1f}] s")

        # --- Per-region data ---
        print("\n  [REGIONS]")
        for region_name in grp.keys():
            if region_name == "times":
                continue
            region = grp[region_name]
            if "coords" in region and "T" in region:
                coords = region["coords"]
                T = region["T"]
                mean_xyz = coords[:].mean(axis=0)
                T_min = np.nanmin(T[:])
                T_max = np.nanmax(T[:])
                print(f"    {region_name:15s}  "
                      f"nodes={coords.shape[0]:>5}  "
                      f"timesteps={T.shape[0]:>4}  "
                      f"center=({mean_xyz[0]:+.3f}, {mean_xyz[1]:+.3f}, {mean_xyz[2]:+.3f})  "
                      f"T=[{T_min:.0f}, {T_max:.0f}] K")

        print("\n" + "-" * 80)

    # --- Compact summary of remaining cases ---
    print(f"\n\n### SUMMARY OF ALL {len(case_keys)} CASES ###\n")
    print(f"  {'case':>7} | {'T_set':>6} | {'cx(mm)':>7} | {'cy(mm)':>7} | "
          f"{'steel nodes':>11} | {'T range (K)':>14}")
    print(f"  {'-'*7}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*11}-+-{'-'*14}")

    for ck in case_keys:
        grp = f[ck]
        name = str(grp.attrs.get("name", "?"))
        T_set = float(grp.attrs.get("T_set", 0))

        cx_m = re.search(r"cx(-?\d+)mm", name)
        cy_m = re.search(r"cy(\d+)mm", name)
        cx_str = cx_m.group(1) if cx_m else "?"
        cy_str = cy_m.group(1) if cy_m else "?"

        steel = grp.get("steel_cylinder")
        if steel is not None:
            n_nodes = steel["coords"].shape[0]
            T = steel["T"]
            T_str = f"[{np.nanmin(T[:]):.0f}, {np.nanmax(T[:]):.0f}]"
        else:
            n_nodes = 0
            T_str = "no steel"

        print(f"  {ck[-3:]:>7} | {T_set:>6.0f} | {cx_str:>7} | {cy_str:>7} | "
              f"{n_nodes:>11} | {T_str:>14}")

    print("\nDone.")

"""Verify that steel cylinder is at different positions across cases."""
import re
import numpy as np
import h5py

H5_PATH = "/mimer/NOBACKUP/groups/revar/GNN_Unified/dataset_v2_all_regions_clean.h5"

with h5py.File(H5_PATH, "r") as f:
    case_keys = sorted([k for k in f.keys() if k.startswith("case_")])
    print(f"Total cases in HDF5: {len(case_keys)}\n")

    # Header
    print(f"{'case':>4} | {'name (cx,cy from name)':<35} | "
          f"{'mesh cx':>9} | {'mesh cy':>9} | {'mesh cz':>9} | {'match?':>7}")
    print("-" * 110)

    mismatches = 0
    for ck in case_keys:
        grp = f[ck]
        name = str(grp.attrs.get("name", "?"))

        # Parse cx, cy from filename
        cx_m = re.search(r"cx(-?\d+)mm", name)
        cy_m = re.search(r"cy(\d+)mm", name)
        cz_m = re.search(r"cz(\d+)mm", name)
        cx_name = int(cx_m.group(1)) / 1000.0 if cx_m else None
        cy_name = int(cy_m.group(1)) / 1000.0 if cy_m else None
        cz_name = int(cz_m.group(1)) / 1000.0 if cz_m else None

        # Actual mesh center
        coords = grp["steel_cylinder"]["coords"][:]
        mesh_cx = coords[:, 0].mean()
        mesh_cy = coords[:, 1].mean()
        mesh_cz = coords[:, 2].mean()

        # For cx: cylinder is extruded along +x from cx with height 0.10,
        # so the cell-center mean x ≈ cx + height/2 = cx + 0.05
        expected_mesh_cx = (cx_name + 0.05) if cx_name is not None else None

        match = "OK"
        if expected_mesh_cx is not None and abs(mesh_cx - expected_mesh_cx) > 0.005:
            match = "MISMATCH"
            mismatches += 1

        label = f"cx={cx_name:+.3f} cy={cy_name:.3f} cz={cz_name:.3f}" \
                if cx_name is not None else name[:35]

        print(f"{ck[-3:]} | {label:<35} | "
              f"{mesh_cx:+.4f} m | {mesh_cy:+.4f} m | {mesh_cz:+.4f} m | {match:>7}")

    # Summary of unique mesh positions
    print()
    print("=" * 60)
    print("SUMMARY: Unique cylinder positions in HDF5 mesh")
    print("=" * 60)
    mesh_cxs, mesh_cys, mesh_czs = [], [], []
    for ck in case_keys:
        coords = f[ck]["steel_cylinder"]["coords"][:]
        mesh_cxs.append(round(coords[:, 0].mean(), 3))
        mesh_cys.append(round(coords[:, 1].mean(), 3))
        mesh_czs.append(round(coords[:, 2].mean(), 3))

    print(f"Unique mesh cx centers: {sorted(set(mesh_cxs))}")
    print(f"Unique mesh cy centers: {sorted(set(mesh_cys))}")
    print(f"Unique mesh cz centers: {sorted(set(mesh_czs))}")
    print(f"\nMismatches (filename vs mesh): {mismatches} / {len(case_keys)}")

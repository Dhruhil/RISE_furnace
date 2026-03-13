#!/usr/bin/env python3
"""
generate_and_run_parameter_study.py

PHASE 1: Vary T_set only (default steel)
PHASE 2: Add material variations later (uncomment)

Automatically:
  1. Creates OpenFOAM cases
  2. Runs simulations
  3. Converts to VTK
  4. Creates .h5 datasets
  5. Creates combined dataset
"""

import os
import re
import shutil
import json
import glob
import subprocess
import numpy as np
import h5py

try:
    import pyvista as pv
    HAVE_PYVISTA = True
except ImportError:
    HAVE_PYVISTA = False
    print("WARNING: pyvista not installed!")

BASE_CASE = "/workspace/rise_furnace/base_case_that_runs_chnage"
OUTPUT_DIR = "/workspace/rise_furnace/parameter_study"

HEATER_REGIONS = [
    "brick_heater",
    "heater_1", "heater_2", "heater_3", "heater_4",
    "heater_5", "heater_6", "heater_7", "heater_8",
]

PHASE_1_CASES = [
    {"T_set": 900,  "kappa": 80, "Cp": 450, "rho": 7800, "molWeight": 195},
    {"T_set": 950,  "kappa": 80, "Cp": 450, "rho": 7800, "molWeight": 195},
    {"T_set": 1050, "kappa": 80, "Cp": 450, "rho": 7800, "molWeight": 195},
    {"T_set": 1100, "kappa": 80, "Cp": 450, "rho": 7800, "molWeight": 195},
]

CASES = PHASE_1_CASES

THERMO_TEMPLATE = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      thermophysicalProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

thermoType
{{
    type            heSolidThermo;
    mixture         pureMixture;
    transport       constIso;
    thermo          hConst;
    equationOfState rhoConst;
    specie          specie;
    energy          sensibleEnthalpy;
}}

mixture
{{
    specie
    {{
        molWeight   {molWeight};
    }}

    transport
    {{
        kappa   {kappa};
    }}

    thermodynamics
    {{
        Hf      0;
        Cp      {Cp};
    }}

    equationOfState
    {{
        rho     {rho};
    }}
}}

// ************************************************************************* //
"""


def case_name(p):
    return f"Tset{p['T_set']}_k{p['kappa']}_Cp{p['Cp']}_rho{p['rho']}"


def modify_heater_T(filepath, T_set):
    with open(filepath, "r") as f:
        content = f.read()
    content_new = re.sub(r'(uniform\s+)\d+(\.\d+)?', f'\\g<1>{T_set}', content)
    with open(filepath, "w") as f:
        f.write(content_new)
    return content != content_new


def modify_steel(case_dir, p):
    path = os.path.join(case_dir, "constant", "steel_cylinder", "thermophysicalProperties")
    with open(path, "w") as f:
        f.write(THERMO_TEMPLATE.format(**p))
    return True


def clean_case(case_dir):
    removed = 0
    for item in os.listdir(case_dir):
        item_path = os.path.join(case_dir, item)
        if os.path.isdir(item_path):
            try:
                if float(item) > 0:
                    shutil.rmtree(item_path)
                    removed += 1
            except ValueError:
                pass

    for folder in ["VTK", "log_files", "results", "figures"]:
        path = os.path.join(case_dir, folder)
        if os.path.exists(path):
            shutil.rmtree(path)
            removed += 1

    for pattern in ["log*", "*.h5", "pinn*.py", "PINN*.py", "make_dataset*.py", "*.foam"]:
        for f in glob.glob(os.path.join(case_dir, pattern)):
            if os.path.isfile(f):
                os.remove(f)
                removed += 1

    return removed


def step1_create(base_dir, output_dir, p):
    name = case_name(p)
    cdir = os.path.join(output_dir, name)
    if os.path.exists(cdir):
        shutil.rmtree(cdir)

    print(f"\n{'='*60}")
    print(f"STEP 1: Creating {name}")
    print(f"  T_set={p['T_set']}, kappa={p['kappa']}, Cp={p['Cp']}, rho={p['rho']}")
    print(f"{'='*60}")

    shutil.copytree(base_dir, cdir)
    n = clean_case(cdir)
    print(f"  Cleaned {n} old items")

    print(f"  Modifying heaters -> uniform {p['T_set']}:")
    for region in HEATER_REGIONS:
        T_file = os.path.join(cdir, "0", region, "T")
        if os.path.isfile(T_file):
            modify_heater_T(T_file, p["T_set"])
            print(f"    OK 0/{region}/T")

    print(f"  Modifying steel properties:")
    modify_steel(cdir, p)
    print(f"    OK kappa={p['kappa']}, Cp={p['Cp']}, rho={p['rho']}")

    print(f"  Verifying:")
    for region in HEATER_REGIONS:
        T_file = os.path.join(cdir, "0", region, "T")
        if os.path.isfile(T_file):
            with open(T_file) as f:
                if f"uniform {p['T_set']}" not in f.read():
                    print(f"    WRONG: 0/{region}/T")

    print(f"    All verified!")
    return cdir, name


def step2_run(cdir, name):
    print(f"\n  STEP 2: Running {name}...")
    allrun = os.path.join(cdir, "Allrun")
    if not os.path.isfile(allrun):
        print(f"    Allrun not found")
        return False
    os.chmod(allrun, 0o755)
    try:
        result = subprocess.run(
            ["./Allrun"], cwd=cdir,
            capture_output=True, text=True, timeout=7200
        )
        with open(os.path.join(cdir, "log.allrun"), "w") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n" + result.stderr)
        if result.returncode == 0:
            print(f"    Simulation completed")
            return True
        else:
            print(f"    Failed (code {result.returncode})")
            for line in result.stdout.strip().split("\n")[-3:]:
                print(f"      {line}")
            return False
    except subprocess.TimeoutExpired:
        print(f"    Timeout (>2h)")
        return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def step3_vtk(cdir, name):
    print(f"  STEP 3: Converting to VTK...")
    try:
        result = subprocess.run(
            ["foamToVTK", "-allRegions"], cwd=cdir,
            capture_output=True, text=True, timeout=600
        )
        with open(os.path.join(cdir, "log.foamToVTK"), "w") as f:
            f.write(result.stdout)
        if result.returncode == 0:
            print(f"    VTK done")
            return True
        else:
            print(f"    VTK failed")
            return False
    except Exception as e:
        print(f"    Error: {e}")
        return False


def step4_dataset(cdir, name, p):
    if not HAVE_PYVISTA:
        print(f"    pyvista missing")
        return None
    print(f"  STEP 4: Creating .h5 dataset...")
    vtk_dir = os.path.join(cdir, "VTK")
    if not os.path.exists(vtk_dir):
        print(f"    VTK/ not found")
        return None
    series_files = glob.glob(os.path.join(vtk_dir, "*.series"))
    if not series_files:
        print(f"    No .series file")
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
    coords = steel_mb0[internal_key].cell_centers().points.astype(np.float32)
    all_T = []
    for vtm_path in vtm_files:
        mb = pv.read(vtm_path)
        T = mb[steel_key][internal_key]["T"].astype(np.float32)
        all_T.append(T)
    all_T = np.stack(all_T, axis=0)
    h5_path = os.path.join(cdir, "steel_cylinder_T_timeseries.h5")
    with h5py.File(h5_path, "w") as f:
        f.create_dataset("coords", data=coords)
        f.create_dataset("times", data=times)
        f.create_dataset("T", data=all_T)
        for key, val in p.items():
            f.attrs[key] = val
    print(f"    Saved: {h5_path}")
    print(f"      {coords.shape[0]} cells, {len(times)} times, T=[{all_T.min():.1f}, {all_T.max():.1f}] K")
    return {
        "coords": coords, "times": times, "T": all_T,
        "T_set": float(p["T_set"]), "kappa": float(p["kappa"]),
        "Cp": float(p["Cp"]), "rho": float(p["rho"]),
        "h5_path": h5_path,
    }


def step5_combine(all_data, output_dir):
    print(f"\n{'='*60}")
    print(f"STEP 5: COMBINED DATASET")
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
    path = os.path.join(output_dir, "combined_dataset.h5")
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
    print(f"\n  Dataset: {X.shape} -> {Y.shape}")
    print(f"  Columns: {cols}")
    for sim in all_data:
        print(f"    T_set={sim['T_set']:.0f}, k={sim['kappa']}, Cp={sim['Cp']}, rho={sim['rho']}")
    print(f"\n  Saved: {path}")
    print(f"  Total: {X.shape[0]:,} points")
    return path


def main():
    print("=" * 60)
    print("PARAMETER STUDY PIPELINE")
    print("=" * 60)
    print(f"\nPhase 1: {len(CASES)} new cases")
    for p in CASES:
        print(f"  T_set={p['T_set']}, k={p['kappa']}, Cp={p['Cp']}, rho={p['rho']}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_data = []
    manifest = []

    print(f"\n{'='*60}")
    print(f"LOADING BASE CASE (T_set=1000)")
    print(f"{'='*60}")
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
        manifest.append({"case": "base", "T_set": 1000, "kappa": 80, "Cp": 450, "rho": 7800, "status": "completed"})
        print(f"  Loaded: T={base['T'].shape}, range=[{base['T'].min():.1f}, {base['T'].max():.1f}] K")
    else:
        print(f"  {base_h5} not found! Run make_dataset_from_series.py first!")
        return

    for p in CASES:
        name = case_name(p)
        cdir, name = step1_create(BASE_CASE, OUTPUT_DIR, p)
        if not step2_run(cdir, name):
            manifest.append({**p, "case": name, "status": "sim_failed"})
            continue
        if not step3_vtk(cdir, name):
            manifest.append({**p, "case": name, "status": "vtk_failed"})
            continue
        sim = step4_dataset(cdir, name, p)
        if sim:
            all_data.append(sim)
            manifest.append({**p, "case": name, "status": "completed"})
        else:
            manifest.append({**p, "case": name, "status": "h5_failed"})

    if len(all_data) > 1:
        step5_combine(all_data, OUTPUT_DIR)

    with open(os.path.join(OUTPUT_DIR, "case_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n{'='*60}")
    print(f"FINAL SUMMARY")
    print(f"{'='*60}")
    for m in manifest:
        icon = "OK" if m["status"] == "completed" else "FAIL"
        print(f"  {m['case']:<40} T_set={m['T_set']:>5} {icon}")
    done = sum(1 for m in manifest if m["status"] == "completed")
    print(f"\nCompleted: {done}/{len(manifest)}")
    print(f"Combined dataset: {OUTPUT_DIR}/combined_dataset.h5")

if __name__ == "__main__":
    main()

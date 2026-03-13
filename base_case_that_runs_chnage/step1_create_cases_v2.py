#!/usr/bin/env python3
"""
Step 1 (v2): Create OpenFOAM cases with different parameters.

From the .geo file analysis:
  Disk(45) = {cx_yz_x, cy_yz_y, cz_yz_z, radius, radius};
  → center at (0, 0.18, 0.195), radius=0.05
  Extrude {height, 0, 0} → height=0.1 along x-axis

Parameters varied:
  - T_set         : heater temperature [K]
  - cy            : cylinder center y-position [m]
  - cz            : cylinder center z-position [m]
  - radius        : cylinder disk radius [m]
  - height        : extrusion length along x [m]
  - kappa         : thermal conductivity [W/m·K]
  - Cp            : specific heat [J/kg·K]
  - rho           : density [kg/m³]

Fixed (from geo):
  - cx = 0.0      : disk starts at x=0 (Rotate then Extrude along +x)
  - Rotation: Pi/2 around y-axis at z=0.195
"""

import os
import re
import shutil
import json
import glob
import math
import numpy as np

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
BASE_CASE="/home/openfoam/rise_furnace/base_case_that_runs_chnage"
OUTPUT_DIR = "/home/openfoam/rise_furnace/parameter_study_v3"

GEO_FILENAME = "rise_furnace_mid_part_base_case_coarse.geo"

HEATER_REGIONS = [
    "brick_heater",
    "heater_1", "heater_2", "heater_3", "heater_4",
    "heater_5", "heater_6", "heater_7", "heater_8",
]

# -------------------------------------------------------
# Furnace geometry bounds (from geo file)
# inner_box ~ x:[0,0.206], y:[0,0.36], z:[0,0.39]
# Keep cylinder safely inside inner_box
# -------------------------------------------------------
FURNACE = {
    "x_min": 0.0,   "x_max": 0.206,
    "y_min": 0.0,   "y_max": 0.36,
    "z_min": 0.0,   "z_max": 0.39,
}

# -------------------------------------------------------
# Base case values (read directly from .geo file)
# -------------------------------------------------------
BASE_PARAMS = {
    "T_set":  1000.0,
    "cx":     0.0,        # disk origin x (fixed, extrude along +x)
    "cy":     0.12,       # disk origin y  ← Disk(45) = {0, 0.18, ...}
    "cz":     0.195,      # disk origin z  ← Disk(45) = {..., 0.195, ...}
    "radius": 0.05,       # ← Disk(45) = {..., 0.05, 0.05}
    "height": 0.1,        # ← Extrude {0.1, 0, 0}
    "kappa":  80.0,
    "Cp":     450.0,
    "rho":    7800.0,
    "mol_weight": 195,
}
BASE_PARAMS["volume"] = math.pi * BASE_PARAMS["radius"]**2 * BASE_PARAMS["height"]
BASE_PARAMS["mass"]   = BASE_PARAMS["rho"] * BASE_PARAMS["volume"]

# -------------------------------------------------------
# Parameter ranges
# Cylinder must fit inside inner_box:
#   radius in [0.03, 0.06]
#   height in [0.06, 0.14]  (along x: cx=0 to cx+height <= 0.206)
#   cy - radius >= y_min=0.065 (brick_heater boundary)
#   cy + radius <= y_max=0.295 (brick_heater boundary)
#   cz in [radius+eps, z_max-radius-eps]
# -------------------------------------------------------
# T_SET_VALUES  = [900, 950, 1000, 1050, 1100]       # K

# CY_VALUES     = [0.15, 0.18, 0.21]                 # m  (y-center of disk)
# CZ_VALUES     = [0.15, 0.195, 0.24]                # m  (z-center of disk)

# RADIUS_VALUES = [0.035, 0.05, 0.06]               # m
# HEIGHT_VALUES = [0.07,  0.10, 0.13]               # m  (extrude along x)

# KAPPA_VALUES  = [40.0,  60.0,  80.0]              # W/m·K
# CP_VALUES     = [400.0, 450.0, 500.0]             # J/kg·K
# RHO_VALUES    = [7600.0, 7800.0, 8000.0]          # kg/m³

T_SET_VALUES  = [900]       # K

CY_VALUES     = [0.18]                 # m  (y-center of disk)
CZ_VALUES     = [0.195]                # m  (z-center of disk)

RADIUS_VALUES = [0.05]               # m
HEIGHT_VALUES = [0.10]               # m  (extrude along x)

KAPPA_VALUES  = [60.0]              # W/m·K
CP_VALUES     = [450.0]             # J/kg·K
RHO_VALUES    = [7800.0]          # kg/m³

MOL_WEIGHT    = 195                          # fixed

# N_LHS_SAMPLES = 30   # number of parameter study cases
N_LHS_SAMPLES = 1  # number of parameter study cases


# -------------------------------------------------------
# Thermophysical properties template
# -------------------------------------------------------
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
        molWeight   {mol_weight};
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


# -------------------------------------------------------
# Latin Hypercube Sampling
# -------------------------------------------------------
def latin_hypercube_samples(param_ranges: dict, n_samples: int, seed: int = 42):
    """
    param_ranges: {name: [v1, v2, ...]}  discrete lists
    Returns list of dicts, one per sample.
    """
    rng    = np.random.default_rng(seed)
    names  = list(param_ranges.keys())
    n_dims = len(names)

    # LHS: for each dimension, permute n_samples intervals
    samples = np.zeros((n_samples, n_dims))
    for d in range(n_dims):
        perm          = rng.permutation(n_samples)
        samples[:, d] = (perm + rng.uniform(size=n_samples)) / n_samples

    result = []
    for row in samples:
        case = {}
        for d, name in enumerate(names):
            choices   = param_ranges[name]
            idx       = int(row[d] * len(choices))
            idx       = min(idx, len(choices) - 1)
            case[name] = choices[idx]
        result.append(case)
    return result


def build_cases_lhs(n_samples=N_LHS_SAMPLES, seed=42):
    param_ranges = {
        "T_set":  T_SET_VALUES,
        "cy":     CY_VALUES,
        "cz":     CZ_VALUES,
        "radius": RADIUS_VALUES,
        "height": HEIGHT_VALUES,
        "kappa":  KAPPA_VALUES,
        "Cp":     CP_VALUES,
        "rho":    RHO_VALUES,
    }
    raw_cases = latin_hypercube_samples(param_ranges, n_samples, seed)

    cases = []
    for p in raw_cases:
        p["cx"]         = 0.0          # fixed: disk at x=0, extruded along +x
        p["mol_weight"] = MOL_WEIGHT
        p["volume"]     = math.pi * p["radius"]**2 * p["height"]
        p["mass"]       = p["rho"] * p["volume"]

        # Safety check: cylinder must fit inside furnace
        if not validate_geometry(p):
            continue
        cases.append(p)

    print(f"  LHS generated {len(raw_cases)} samples, "
          f"{len(cases)} passed geometry validation")
    return cases


def validate_geometry(p):
    """
    Check cylinder fits inside the furnace inner_box.
    inner_box from geo: x:[0,0.206], y:[0.065,0.295], z:[0,0.39]
    Cylinder: disk center (cx=0, cy, cz), radius, extruded height along +x
    """
    r = p["radius"]
    h = p["height"]

    # x: extrude from cx=0 to cx+h
    if p["cx"] + h > FURNACE["x_max"] - 0.01:
        return False

    # y: disk center ± radius
    if p["cy"] - r < FURNACE["y_min"] + 0.07:   # keep clear of brick_heater
        return False
    if p["cy"] + r > FURNACE["y_max"] - 0.07:
        return False

    # z: disk center ± radius
    if p["cz"] - r < FURNACE["z_min"] + 0.01:
        return False
    if p["cz"] + r > FURNACE["z_max"] - 0.01:
        return False

    return True


# -------------------------------------------------------
# Case naming
# -------------------------------------------------------
def case_name(p: dict, idx: int) -> str:
    return (
        f"case{idx:03d}"
        f"_Tset{p['T_set']:.0f}"
        f"_cy{p['cy']*1e3:.0f}mm"
        f"_cz{p['cz']*1e3:.0f}mm"
        f"_r{p['radius']*1e3:.0f}mm"
        f"_h{p['height']*1e3:.0f}mm"
        f"_k{p['kappa']:.0f}"
        f"_Cp{p['Cp']:.0f}"
        f"_rho{p['rho']:.0f}"
    )


# -------------------------------------------------------
# Clean old time folders
# -------------------------------------------------------
def clean_case(case_dir: str) -> int:
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
    for pattern in ["log*", "*.h5", "pinn*.py",
                    "PINN*.py", "make_dataset*.py", "*.foam"]:
        for fp in glob.glob(os.path.join(case_dir, pattern)):
            if os.path.isfile(fp):
                os.remove(fp)
                removed += 1
    return removed


# -------------------------------------------------------
# Patch the .geo file  (geometry + re-mesh needed)
# -------------------------------------------------------
def patch_geo_file(case_dir: str, p: dict) -> bool:
    """
    Patch the Gmsh .geo file for:
      Disk(45) = {cx, cy, cz, radius, radius};
      Extrude {height, 0, 0} { ... Surface{45} ... }

    Uses line-by-line patching to avoid DOTALL corruption.
    """
    geo_src = os.path.join(BASE_CASE, GEO_FILENAME)
    geo_dst = os.path.join(case_dir, GEO_FILENAME)

    if not os.path.isfile(geo_src):
        print(f"    [WARN] .geo not found: {geo_src}")
        return False

    with open(geo_src, "r") as f:
        lines = f.readlines()

    n_disk = 0
    n_ext  = 0

    # ----------------------------------------------------------------
    # Strategy:
    #   Pass 1 – patch Disk(45) line (single line, no context needed)
    #   Pass 2 – find the Extrude block that contains Surface{45}
    #            and patch ONLY its opening "Extrude {<num>, 0, 0} {"
    # ----------------------------------------------------------------

    # ---- Pass 1: Disk(45) ----
    disk_pat = re.compile(
        r"(Disk\s*\(\s*45\s*\)\s*=\s*\{)"
        r"[^}]+"          # everything inside braces
        r"(\};)"
    )
    new_lines = []
    for line in lines:
        m = disk_pat.search(line)
        if m:
            new_line = disk_pat.sub(
                rf"\g<1>"
                f"{p['cx']}, {p['cy']}, {p['cz']}, "
                f"{p['radius']}, {p['radius']}"
                rf"\g<2>",
                line
            )
            new_lines.append(new_line)
            n_disk += 1
        else:
            new_lines.append(line)

    # ---- Pass 2: Extrude containing Surface{45} ----
    # Find the Extrude block that references Surface{45}.
    # The block looks like:
    #   Extrude {0.1, 0, 0} {
    #     Curve{69}; Surface{45}; Layers {10};
    #   }
    # We scan for "Extrude {..." lines and check whether the
    # subsequent lines (until closing '}') contain "Surface{45}".
    # Then we patch ONLY the Extrude opening line.

    extrude_open_pat = re.compile(
        r"^(\s*Extrude\s*\{)\s*"
        r"([\d.\-e+]+)"          # group 2 = the x-distance (height)
        r"(\s*,\s*0\s*,\s*0\s*\}\s*\{.*)$"  # group 3 = rest of line
    )

    surface45_pat = re.compile(r"Surface\s*\{\s*45\s*\}")

    result_lines = []
    i = 0
    while i < len(new_lines):
        line = new_lines[i]
        m = extrude_open_pat.match(line)
        if m:
            # Look ahead to check if Surface{45} appears in this block
            # Collect lines until matching closing brace
            depth   = line.count('{') - line.count('}')
            block   = [line]
            j       = i + 1
            while j < len(new_lines) and depth > 0:
                block.append(new_lines[j])
                depth += new_lines[j].count('{') - new_lines[j].count('}')
                j += 1

            block_text = "".join(block)
            if surface45_pat.search(block_text):
                # Patch the opening line only
                patched_open = extrude_open_pat.sub(
                    rf"\g<1>{p['height']}\g<3>",
                    line
                )
                result_lines.append(patched_open)
                result_lines.extend(block[1:])  # rest of block unchanged
                n_ext += 1
                i = j
                continue

        result_lines.append(line)
        i += 1

    assert n_disk == 1, f"Disk(45) patched {n_disk} times (expected 1)"
    assert n_ext  == 1, f"Extrude patched {n_ext} times (expected 1)"

    with open(geo_dst, "w") as f:
        f.writelines(result_lines)

    ok = (n_disk == 1) and (n_ext == 1)
    print(f"    Geo patched: Disk(45) hits={n_disk}, "
          f"Extrude hits={n_ext}  -> {'OK' if ok else 'PARTIAL'}")
    print(f"    Disk(45) = {{{p['cx']}, {p['cy']}, {p['cz']}, "
          f"{p['radius']}, {p['radius']}}}")
    print(f"    Extrude  = {{{p['height']}, 0, 0}}")
    return ok

# -------------------------------------------------------
# Write cylinder params JSON  (used by step3)
# -------------------------------------------------------
def write_cylinder_params(case_dir: str, p: dict):
    keys = ["T_set", "cx", "cy", "cz", "radius", "height",
            "volume", "mass", "kappa", "Cp", "rho", "mol_weight"]
    params = {k: float(p[k]) for k in keys if k in p}
    out = os.path.join(case_dir, "cylinder_params.json")
    with open(out, "w") as f:
        json.dump(params, f, indent=2)


# -------------------------------------------------------
# Update Allmesh to re-run gmsh for new geometry
# -------------------------------------------------------
def patch_allmesh(case_dir: str, geo_filename: str):
    """
    Write a clean correct Allmesh with gmsh in the right position.
    This REPLACES the broken patch approach completely.
    
    Correct structure:
      #!/bin/sh
      . ${WM_PROJECT_DIR}/bin/tools/RunFunctions   <- NO cd line
      gmsh -3 ...geo -o ...msh -format msh2        <- BEFORE blockMesh
      runApplication blockMesh
      ...
    """
    allmesh_path = os.path.join(case_dir, "Allmesh")
    msh_file     = geo_filename.replace(".geo", ".msh")

    # Read original Allmesh from BASE_CASE to get exact content
    base_allmesh = os.path.join(BASE_CASE, "Allmesh")

    if os.path.isfile(base_allmesh):
        with open(base_allmesh, "r") as f:
            original_lines = f.readlines()
    else:
        print(f"  [WARN] Base Allmesh not found, writing minimal version")
        original_lines = []

    # Build new Allmesh line by line
    new_lines = []
    gmsh_inserted = False

    for line in original_lines:
        stripped = line.strip()

        # Skip the broken cd line entirely
        if stripped.startswith('cd "${0%/*}"') or \
           stripped.startswith("cd '${0%/*}'") or \
           stripped.startswith("cd ${0%/*}"):
            # Do NOT add this line
            print(f"  [Allmesh] Removed broken cd line")
            continue

        # Add every other line normally
        new_lines.append(line)

        # After RunFunctions source line -> insert gmsh
        if not gmsh_inserted and "RunFunctions" in stripped:
            new_lines.append("\n")
            new_lines.append(
                f"# Regenerate mesh from patched .geo\n"
            )
            new_lines.append(
                f"gmsh -3 {geo_filename} "
                f"-o {msh_file} "
                f"-format msh2\n"
            )
            new_lines.append("\n")
            gmsh_inserted = True
            print(f"  [Allmesh] gmsh inserted after RunFunctions -> OK")

    # If original_lines was empty or RunFunctions not found
    if not gmsh_inserted:
        new_lines = [
            "#!/bin/sh\n",
            "\n",
            ". ${WM_PROJECT_DIR:?}/bin/tools/RunFunctions\n",
            "#------------------------------------------\n",
            "\n",
            f"# Regenerate mesh from patched .geo\n",
            f"gmsh -3 {geo_filename} -o {msh_file} -format msh2\n",
            "\n",
            "runApplication blockMesh\n",
            "runApplication topoSet\n",
            "rm log.topoSet\n",
            "runApplication topoSet -dict system/topoSetDict.f1\n",
            "restore0Dir\n",
            "\n",
            "runApplication splitMeshRegions -cellZones -overwrite\n",
            "\n",
            "for region in $(foamListRegions solid)\n",
            "do\n",
            "    rm -f 0/$region/{nut,alphat,epsilon,k,U,p_rgh}\n",
            "    rm -f processor*/0/$region/{nut,alphat,epsilon,k,U,p_rgh}\n",
            "done\n",
            "\n",
            "for region in $(foamListRegions)\n",
            "do\n",
            "    runApplication -s $region changeDictionary -region $region\n",
            "done\n",
            "\n",
            "runApplication createBaffles -region rightFluid -overwrite\n",
            "\n",
            "echo\n",
            'echo "End"\n',
            "\n",
            "#------------------------------------------\n",
        ]
        print(f"  [Allmesh] Written from template -> OK")

    with open(allmesh_path, "w") as f:
        f.writelines(new_lines)

    # Make executable
    os.chmod(allmesh_path, 0o755)
    print(f"  [Allmesh] File written and made executable")

# -------------------------------------------------------
# Main
# -------------------------------------------------------
def main():
    print("=" * 60)
    print("STEP 1 (v2): CASES WITH CYLINDER GEOMETRY VARIATIONS")
    print("=" * 60)
    print(f"\nBase case geo analysis:")
    print(f"  Disk(45) = {{0, 0.18, 0.195, 0.05, 0.05}}")
    print(f"  Rotate Pi/2 around y-axis at z=0.195")
    print(f"  Extrude {{0.1, 0, 0}}  -> height=0.1 along x-axis")
    print(f"\nVariable parameters:")
    print(f"  T_set  : {T_SET_VALUES} K")
    print(f"  cy     : {CY_VALUES} m")
    print(f"  cz     : {CZ_VALUES} m")
    print(f"  radius : {RADIUS_VALUES} m")
    print(f"  height : {HEIGHT_VALUES} m")
    print(f"  kappa  : {KAPPA_VALUES} W/m·K")
    print(f"  Cp     : {CP_VALUES} J/kg·K")
    print(f"  rho    : {RHO_VALUES} kg/m³")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---- Generate LHS cases ----
    cases = build_cases_lhs(N_LHS_SAMPLES)

    if not cases:
        print("[ERROR] No valid cases generated!")
        return

    # ---- Manifest: start with base case ----
    manifest = [{
        "idx":    0,
        "case":   "base_case_that_runs_chnage",
        "status": "completed",
        **{k: float(v) if isinstance(v, (int, float, np.floating))
           else v for k, v in BASE_PARAMS.items()},
    }]

    print(f"\n{'='*60}")
    print(f"Creating {len(cases)} parameter study cases ...")
    print(f"{'='*60}")

    for idx, p in enumerate(cases, start=1):
        name = case_name(p, idx)
        cdir = os.path.join(OUTPUT_DIR, name)

        print(f"\n[{idx:03d}/{len(cases)}] {name}")
        print(f"  T_set={p['T_set']:.0f}K | "
              f"cy={p['cy']:.3f}m cz={p['cz']:.3f}m | "
              f"r={p['radius']*1e3:.1f}mm h={p['height']*1e3:.1f}mm | "
              f"k={p['kappa']:.0f} Cp={p['Cp']:.0f} rho={p['rho']:.0f} | "
              f"V={p['volume']*1e6:.2f}cm³ m={p['mass']*1e3:.1f}g")

        # ---- Copy base case ----
        if os.path.exists(cdir):
            shutil.rmtree(cdir)
        shutil.copytree(BASE_CASE, cdir)
        n = clean_case(cdir)
        print(f"  Cleaned {n} old items")

        # ---- 1. Heater temperatures (0/heater_*/T) ----
        for region in HEATER_REGIONS:
            T_file = os.path.join(cdir, "0", region, "T")
            if os.path.isfile(T_file):
                with open(T_file, "r") as f:
                    c = f.read()
                c = re.sub(
                    r'(uniform\s+)\d+(\.\d+)?',
                    f'\\g<1>{p["T_set"]:.1f}',
                    c
                )
                with open(T_file, "w") as f:
                    f.write(c)
        print(f"  OK heater T = {p['T_set']:.0f} K")

        # ---- 2. Steel thermophysical properties ----
        thermo_path = os.path.join(
            cdir, "constant", "steel_cylinder", "thermophysicalProperties"
        )
        with open(thermo_path, "w") as f:
            f.write(THERMO_TEMPLATE.format(**p))
        print(f"  OK thermophysicalProperties "
              f"(k={p['kappa']}, Cp={p['Cp']}, rho={p['rho']})")

        # ---- 3. Patch .geo file for new cylinder geometry ----
        patch_geo_file(cdir, p)

        # ---- 4. Ensure Allmesh re-runs gmsh ----
        patch_allmesh(cdir, GEO_FILENAME)

        # ---- 5. Save cylinder_params.json ----
        write_cylinder_params(cdir, p)

        # ---- 6. Manifest entry ----
        manifest.append({
            "idx":    idx,
            "case":   name,
            "status": "ready",
            **{k: float(v) if isinstance(v, (int, float, np.floating))
               else v for k, v in p.items()},
        })

    # ---- Save manifest ----
    manifest_path = os.path.join(OUTPUT_DIR, "case_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest: {manifest_path}  ({len(manifest)} entries)")

    # ---- Generate OpenFOAM run script ----
    run_script = os.path.join(OUTPUT_DIR, "run_all_openfoam.sh")
    with open(run_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Run INSIDE OpenFOAM Docker container\n")
        f.write("# docker run -it \\\n")
        f.write("#   -v ~/OpenFOAM/rise_furnace:/home/openfoam/rise_furnace \\\n")
        f.write("#   microfluidica/openfoam:2412 bash\n\n")
        f.write("set -e\n\n")
        f.write(f'BASE_DIR="/home/openfoam/rise_furnace/parameter_study_v2"\n\n')
        for idx, p in enumerate(cases, start=1):
            name      = case_name(p, idx)
            foam_path = f"$BASE_DIR/{name}"
            f.write(f'echo ""\n')
            f.write(f'echo "========================================"\n')
            f.write(f'echo "[{idx:03d}/{len(cases)}] {name}"\n')
            f.write(f'echo "========================================"\n')
            f.write(f'cd {foam_path}\n')
            f.write(f'bash Allmesh   # remesh with new cylinder geometry\n')
            f.write(f'bash Allrun\n')
            f.write(f'foamToVTK -allRegions\n')
            f.write(f'echo "Done: {name}"\n\n')
    os.chmod(run_script, 0o755)
    print(f"Run script: {run_script}")

    # ---- Summary table ----
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(cases)} cases")
    print(f"{'='*60}")
    print(f"{'#':>4} {'T_set':>6} {'cy[mm]':>7} {'cz[mm]':>7} "
          f"{'r[mm]':>6} {'h[mm]':>6} {'k':>4} {'Cp':>5} "
          f"{'rho':>5} {'V[cm3]':>7}")
    print("-" * 70)
    for idx, p in enumerate(cases, start=1):
        print(f"{idx:>4} {p['T_set']:>6.0f} "
              f"{p['cy']*1e3:>7.1f} {p['cz']*1e3:>7.1f} "
              f"{p['radius']*1e3:>6.1f} {p['height']*1e3:>6.1f} "
              f"{p['kappa']:>4.0f} {p['Cp']:>5.0f} "
              f"{p['rho']:>5.0f} {p['volume']*1e6:>7.2f}")

    print(f"\n{'='*60}")
    print(f"NEXT STEPS:")
    print(f"  1. Start OpenFOAM container and run:")
    print(f"       bash {run_script}")
    print(f"  2. Then build datasets:")
    print(f"       python3 step3_create_datasets_v2.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
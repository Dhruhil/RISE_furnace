#!/usr/bin/env python3
"""
Step 1: Create OpenFOAM cases with different T_set
Run this in PhysicsNeMo container
"""

import os
import re
import shutil
import json
import glob

BASE_CASE = "/workspace/rise_furnace/base_case_that_runs_chnage"
OUTPUT_DIR = "/workspace/rise_furnace/parameter_study"

HEATER_REGIONS = [
    "brick_heater",
    "heater_1", "heater_2", "heater_3", "heater_4",
    "heater_5", "heater_6", "heater_7", "heater_8",
]

CASES = [
    {"T_set": 900,  "kappa": 80, "Cp": 450, "rho": 7800, "molWeight": 195},
    # {"T_set": 950,  "kappa": 80, "Cp": 450, "rho": 7800, "molWeight": 195},
    # {"T_set": 1050, "kappa": 80, "Cp": 450, "rho": 7800, "molWeight": 195},
    # {"T_set": 1100, "kappa": 80, "Cp": 450, "rho": 7800, "molWeight": 195},
]

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


def main():
    print("=" * 60)
    print("STEP 1: CREATING OPENFOAM CASES")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = []

    # Base case
    manifest.append({
        "case": "base_case_that_runs_chnage",
        "T_set": 1000, "kappa": 80, "Cp": 450, "rho": 7800,
        "status": "completed"
    })

    for p in CASES:
        name = case_name(p)
        cdir = os.path.join(OUTPUT_DIR, name)

        if os.path.exists(cdir):
            shutil.rmtree(cdir)

        print(f"\nCreating: {name}")
        print(f"  T_set={p['T_set']}, kappa={p['kappa']}, Cp={p['Cp']}, rho={p['rho']}")

        # Copy
        shutil.copytree(BASE_CASE, cdir)

        # Clean
        n = clean_case(cdir)
        print(f"  Cleaned {n} old items")

        # Modify heaters in 0/
        for region in HEATER_REGIONS:
            T_file = os.path.join(cdir, "0", region, "T")
            if os.path.isfile(T_file):
                with open(T_file, "r") as f:
                    content = f.read()
                content_new = re.sub(r'(uniform\s+)\d+(\.\d+)?', f'\\g<1>{p["T_set"]}', content)
                with open(T_file, "w") as f:
                    f.write(content_new)
                print(f"  OK 0/{region}/T -> uniform {p['T_set']}")

        # Modify steel properties
        thermo_path = os.path.join(cdir, "constant", "steel_cylinder", "thermophysicalProperties")
        with open(thermo_path, "w") as f:
            f.write(THERMO_TEMPLATE.format(**p))
        print(f"  OK constant/steel_cylinder/thermophysicalProperties")

        # Verify
        ok = True
        for region in HEATER_REGIONS:
            T_file = os.path.join(cdir, "0", region, "T")
            if os.path.isfile(T_file):
                with open(T_file) as f:
                    if f"uniform {p['T_set']}" not in f.read():
                        print(f"  WRONG: 0/{region}/T")
                        ok = False

        if ok:
            print(f"  VERIFIED!")

        manifest.append({**p, "case": name, "status": "ready"})

    # Save manifest
    with open(os.path.join(OUTPUT_DIR, "case_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Create OpenFOAM run script
    run_script = os.path.join(OUTPUT_DIR, "run_all_openfoam.sh")
    with open(run_script, "w") as f:
        f.write("#!/bin/bash\n")
        f.write("# Run this INSIDE the OpenFOAM Docker container\n")
        f.write("# docker run -it -v ~/OpenFOAM/rise_furnace:/home/openfoam/rise_furnace microfluidica/openfoam:2412 bash\n")
        f.write("# Then: cd /home/openfoam/rise_furnace/parameter_study && bash run_all_openfoam.sh\n\n")

        for p in CASES:
            name = case_name(p)
            # Use OpenFOAM container path
            foam_path = f"/home/openfoam/rise_furnace/parameter_study/{name}"
            f.write(f'echo "========================================"\n')
            f.write(f'echo "Running: {name} (T_set={p["T_set"]})"\n')
            f.write(f'echo "========================================"\n')
            f.write(f'cd {foam_path}\n')
            f.write(f'./Allrun\n')
            f.write(f'foamToVTK -allRegions\n')
            f.write(f'echo "Done: {name}"\n\n')

    os.chmod(run_script, 0o755)

    # Summary
    print(f"\n{'='*60}")
    print(f"CASES CREATED SUCCESSFULLY!")
    print(f"{'='*60}")
    print(f"\nCases:")
    for m in manifest:
        print(f"  {m['case']:<40} T_set={m['T_set']}")

    print(f"\nFiles:")
    print(f"  {os.path.join(OUTPUT_DIR, 'case_manifest.json')}")
    print(f"  {run_script}")

    print(f"\n{'='*60}")
    print(f"NEXT STEPS:")
    print(f"{'='*60}")
    print(f"""
  1. Open a NEW terminal

  2. Start OpenFOAM Docker container:
     docker run -it \\
       -v ~/OpenFOAM/rise_furnace:/home/openfoam/rise_furnace \\
       microfluidica/openfoam:2412 bash

  3. Inside OpenFOAM container, run:
     cd /home/openfoam/rise_furnace/parameter_study
     bash run_all_openfoam.sh

  4. Wait for all simulations to finish

  5. Come back to THIS terminal (PhysicsNeMo) and run:
     python3 step3_create_datasets.py
    """)


if __name__ == "__main__":
    main()

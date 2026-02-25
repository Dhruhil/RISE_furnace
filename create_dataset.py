#!/usr/bin/env python3
"""
OpenFOAM Heat Treatment Dataset Generator
==========================================
Creates 21 simulation cases with heater temperatures
varying from 900K to 1100K in steps of 10K.

Usage:
    python3 create_dataset.py

Requirements:
    pip install numpy scipy
"""

import os
import re
import shutil

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# Path to your base OpenFOAM case
BASE_CASE_DIR = "/home/jinisa/OpenFOAM/Case_910"

# Where to store all generated cases
OUTPUT_DIR = "/home/jinisa/OpenFOAM/dataset_temperatures"

# Temperature range: 900K to 1100K in steps of 10K → 21 cases
TEMP_START = 900
TEMP_END   = 1100
TEMP_STEP  = 10

# Heater regions whose T file will be changed
HEATER_REGIONS = [
    "brick_heater",
    "heater_1",
    "heater_2",
    "heater_3",
    "heater_4",
    "heater_5",
    "heater_6",
    "heater_7",
    "heater_8",
]

# Files and folders to EXCLUDE when copying base case (keeps cases clean)
EXCLUDE_ITEMS = [
    "dataset",
    "dataset_temperatures",
    "VTK",
    "figures",
    "flat_XY_dataset.h5",
    "steel_cylinder_T_timeseries.h5",
    "pinn_heat3d_model.pth",
    "pinn_h5_training.png",
    "log.chtMultiRegionFoam",
    "log.viewFactorsGen.inner_box",
    "log_files",
    "create_dataset.py",
    "create_dataset.pyc",
    "make_dataset_from_series.py",
    "Load_h5_Dataset.py",
    "PINN1.py",
    ".pyc",
]

# Also exclude old timestep folders (numbered folders like 10, 20, 30...)
EXCLUDE_TIMESTEP_FOLDERS = True

# ==============================================================================
# COPY BASE CASE (clean copy, no old results)
# ==============================================================================

def should_exclude(name):
    """Check if a file/folder should be excluded from the copy."""
    if name in EXCLUDE_ITEMS:
        return True
    if EXCLUDE_TIMESTEP_FOLDERS:
        try:
            int(name)  # if folder name is a number, it's a timestep folder
            return True
        except ValueError:
            pass
    return False


def copy_base_case(dest_dir):
    """Copy base case to destination, excluding unnecessary files."""
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    os.makedirs(dest_dir)

    for item in os.listdir(BASE_CASE_DIR):
        if should_exclude(item):
            continue
        src = os.path.join(BASE_CASE_DIR, item)
        dst = os.path.join(dest_dir, item)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


# ==============================================================================
# EDIT TEMPERATURE IN HEATER T FILES
# ==============================================================================

def edit_heater_temperature(case_dir, region, temperature):
    """
    Set the initial temperature in 0/<region>/T file.
    Changes both internalField and all boundary condition values.
    """
    t_file = os.path.join(case_dir, "0", region, "T")

    if not os.path.exists(t_file):
        # Also check 0_orig
        t_file_orig = os.path.join(case_dir, "0_orig", region, "T")
        if os.path.exists(t_file_orig):
            t_file = t_file_orig
        else:
            print(f"    Warning: T file not found for {region}, skipping.")
            return

    with open(t_file, 'r') as f:
        content = f.read()

    # Replace internalField temperature value
    new_content = re.sub(
        r'(internalField\s+uniform\s+)[\d.]+(\s*;)',
        rf'\g<1>{temperature}\g<2>',
        content
    )

    # Replace all boundary condition uniform temperature values
    new_content = re.sub(
        r'(value\s+uniform\s+)[\d.]+(\s*;)',
        rf'\g<1>{temperature}\g<2>',
        new_content
    )

    with open(t_file, 'w') as f:
        f.write(new_content)


# ==============================================================================
# SAVE PARAMETER LOG
# ==============================================================================

def save_parameter_log(all_params):
    """Save all parameter combinations to CSV."""
    log_path = os.path.join(OUTPUT_DIR, "parameter_log.csv")
    header = "case_id,heater_temperature_K,case_folder"
    rows = [header]
    for p in all_params:
        rows.append(f"{p['case_id']:04d},{p['temperature']},{p['case_folder']}")
    with open(log_path, 'w') as f:
        f.write('\n'.join(rows))
    print(f"Parameter log saved to: {log_path}\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("OpenFOAM Heat Treatment Dataset Generator")
    print("Heater Temperature Sweep: 900K to 1100K (step 10K)")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate temperature list: 900, 910, 920, ..., 1100
    temperatures = list(range(TEMP_START, TEMP_END + TEMP_STEP, TEMP_STEP))
    print(f"\nTotal cases to generate: {len(temperatures)}")
    print(f"Temperatures: {temperatures}\n")

    all_params = []
    success = 0
    failed = 0

    for i, temp in enumerate(temperatures):
        case_folder = f"case_{i:04d}_T{temp}K"
        case_dir = os.path.join(OUTPUT_DIR, case_folder)

        print(f"  [{i+1:02d}/{len(temperatures)}] Setting up {case_folder} (T = {temp} K)...")

        try:
            # Copy clean base case
            copy_base_case(case_dir)

            # Edit temperature in all heater T files
            for region in HEATER_REGIONS:
                edit_heater_temperature(case_dir, region, temp)

            all_params.append({
                "case_id": i,
                "temperature": temp,
                "case_folder": case_folder
            })
            success += 1

        except Exception as e:
            print(f"    ERROR: {e}")
            failed += 1

    # Save log
    save_parameter_log(all_params)

    # Summary
    print("=" * 60)
    print(f"Done! {success} cases prepared, {failed} failed.")
    print(f"\nCases saved in: {OUTPUT_DIR}")
    print(f"\nNext steps:")
    print(f"  1. Verify a case:  cat {OUTPUT_DIR}/case_0000_T900K/0/heater_1/T")
    print(f"  2. Test one case:  cd {OUTPUT_DIR}/case_0000_T900K && ./Allrun")
    print(f"  3. Run all cases by adding AUTO_RUN = True to this script")
    print("=" * 60)


if __name__ == "__main__":
    main()


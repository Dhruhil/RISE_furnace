"""
Inspect HDF5 dataset to see how long each case actually runs.
"""
import h5py
import numpy as np
from configs.fno_config import CONFIG

cfg = CONFIG
h5_path = cfg.dataset_path
print(f"Inspecting: {h5_path}\n")

with h5py.File(h5_path, "r") as f:
    # Find all case groups
    case_keys = sorted([k for k in f.keys() if k.startswith("case_")])
    print(f"Total cases: {len(case_keys)}\n")

    # Collect time info per case
    print(f"  {'Case':>10}  {'N_times':>8}  {'t_first':>9}  {'t_last':>9}  {'duration':>10}")
    print(f"  {'-'*60}")

    t_lasts = []
    n_times_list = []

    for k in case_keys:
        if "times" in f[k]:
            times = f[k]["times"][:]
            n_t = len(times)
            t_first = float(times[0])
            t_last  = float(times[-1])
            duration = t_last - t_first
            t_lasts.append(t_last)
            n_times_list.append(n_t)
            print(f"  {k:>10}  {n_t:>8}  {t_first:>7.1f}s  {t_last:>7.1f}s  {duration:>8.1f}s")

    # Summary statistics
    print(f"\n{'='*60}")
    print("  SUMMARY ACROSS ALL CASES")
    print(f"{'='*60}")
    t_lasts = np.array(t_lasts)
    n_times_list = np.array(n_times_list)
    print(f"  N_times      → min={n_times_list.min()}  "
          f"median={int(np.median(n_times_list))}  "
          f"max={n_times_list.max()}")
    print(f"  Last time    → min={t_lasts.min():.1f}s  "
          f"median={np.median(t_lasts):.1f}s  "
          f"max={t_lasts.max():.1f}s")

    # How many cases reach each milestone?
    print(f"\n  Cases reaching milestones:")
    for ms in [2760, 3000, 3200, 3460, 3500, 3600]:
        n = (t_lasts >= ms).sum()
        print(f"    {n:>3} of {len(t_lasts)} cases reach t = {ms}s "
              f"({100*n/len(t_lasts):.0f}%)")

    # Outliers — cases that ended early
    print(f"\n  Cases that ended BEFORE 3460s (Phase 2 cutoff):")
    early = [(k, t) for k, t in zip(case_keys, t_lasts) if t < 3460]
    if not early:
        print("    None — all cases reach Phase 2 end ✓")
    else:
        for k, t in early:
            print(f"    {k}: ended at {t:.1f}s")

    # Test sim subset
    test_sim_indices = [13, 17, 28, 31, 35, 3, 14]
    print(f"\n  TEST SET cases:")
    for idx in test_sim_indices:
        k = f"case_{idx:03d}"
        if k in f and "times" in f[k]:
            times = f[k]["times"][:]
            print(f"    {k}: {len(times)} steps, ends at {times[-1]:.1f}s")

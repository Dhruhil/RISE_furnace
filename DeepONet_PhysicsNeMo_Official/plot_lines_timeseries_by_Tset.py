#!/usr/bin/env python3
"""
Time-series line plots: steel cylinder mean temperature vs time.
One plot per T_set group, showing 5 cases each.
Output in Celsius (matches engineering convention for heat treatment).
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import re

# ──────────────── CONFIG ────────────────
H5_PATH           = "dataset_v2_all_regions_clean.h5"
OUT_DIR           = Path("plots_timeseries")
OUT_DIR.mkdir(exist_ok=True)

TARGET_TSETS_K    = [1173, 1223, 1273, 1323, 1373]   # 900, 950, 1000, 1050, 1100 °C
N_CASES_PER_PLOT  = 5
RNG_SEED          = 42

# ──────────────── STYLE ────────────────
mpl.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   9,
    'figure.dpi':        120,
    'savefig.dpi':       220,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
})

# ──────────────── HELPERS ────────────────
def parse_name(name):
    t  = int(re.search(r'Tset(\d+)',   name).group(1))
    cx = int(re.search(r'cx(-?\d+)mm', name).group(1))
    cy = int(re.search(r'cy(-?\d+)mm', name).group(1))
    return t, cx, cy

# ──────────────── LOAD DATA ────────────────
print(f"Loading dataset: {H5_PATH}")

cases_by_tset = {}
with h5py.File(H5_PATH, 'r') as f:
    for cid in sorted(f.keys()):
        if not cid.startswith('case_'):
            continue
        c = f[cid]
        try:
            t_set, cx, cy = parse_name(c.attrs['name'])
        except Exception:
            continue
        cases_by_tset.setdefault(t_set, []).append(cid)

print(f"\n═══ T_set distribution ═══")
for t in sorted(cases_by_tset.keys()):
    print(f"  T_set = {t} K ({t-273}°C):  {len(cases_by_tset[t])} cases")

# ──────────────── PLOT FUNCTION ────────────────
def plot_one_tset(T_set_K, case_keys, h5):
    n = len(case_keys)
    T_set_C = T_set_K - 273

    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n))

    for i, case_key in enumerate(case_keys):
        c = h5[case_key]
        times   = c['times'][:]
        T_steel = c['steel_cylinder']['T'][:]      # (n_t, n_cells)  in Kelvin
        T_mean  = T_steel.mean(axis=1) - 273.15    # convert to °C

        try:
            _, cx, cy = parse_name(c.attrs['name'])
            label = (f"{case_key.replace('case_','c')}  "
                     f"cx={cx:+4d}mm  cy={cy:3d}mm")
        except:
            label = case_key

        ax.plot(times, T_mean, color=cmap[i], lw=1.8, label=label)

    # T_set reference line
    ax.axhline(T_set_C, color='red', ls='--', lw=1.2, alpha=0.7,
               label=f'T_set = {T_set_C}°C')

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Steel cylinder mean temperature [°C]')
    ax.set_title(f'Steel cylinder heating — T_set = {T_set_K} K ({T_set_C}°C)  '
                 f'(n={n} cases)',
                 fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)

    out = OUT_DIR / f'lines_Tset_{T_set_K}K_{T_set_C}C.png'
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out.name}")

# ──────────────── GENERATE INDIVIDUAL PLOTS ────────────────
print(f"\nGenerating individual T_set plots ...")
rng = np.random.default_rng(RNG_SEED)

with h5py.File(H5_PATH, 'r') as f:
    for tset_k in TARGET_TSETS_K:
        if tset_k not in cases_by_tset:
            print(f"  SKIP: no cases at T_set = {tset_k} K")
            continue

        all_cases = cases_by_tset[tset_k]
        n_pick = min(N_CASES_PER_PLOT, len(all_cases))
        if len(all_cases) > n_pick:
            picked = sorted(rng.choice(all_cases, size=n_pick, replace=False))
        else:
            picked = all_cases

        plot_one_tset(tset_k, picked, f)

    # ─── Combined plot: all T_sets, one case each ───
    print(f"\nGenerating combined plot ...")
    fig, ax = plt.subplots(figsize=(11, 6))
    cmap = plt.cm.plasma(np.linspace(0.15, 0.85, len(TARGET_TSETS_K)))

    for i, tset_k in enumerate(TARGET_TSETS_K):
        if tset_k not in cases_by_tset:
            continue
        case_key = sorted(cases_by_tset[tset_k])[0]
        c = f[case_key]
        times   = c['times'][:]
        T_steel = c['steel_cylinder']['T'][:]
        T_mean  = T_steel.mean(axis=1) - 273.15
        T_set_C = tset_k - 273

        ax.plot(times, T_mean, color=cmap[i], lw=2.0,
                label=f'T_set = {tset_k} K ({T_set_C}°C)')
        ax.axhline(T_set_C, color=cmap[i], ls=':', lw=0.8, alpha=0.5)

    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Steel cylinder mean temperature [°C]')
    ax.set_title('Steel cylinder heating curves across all T_set values',
                 fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'lines_combined_all_Tsets.png')
    plt.close()
    print(f"  Saved: lines_combined_all_Tsets.png")

print(f"\n═══ All plots saved in: {OUT_DIR.absolute()}")

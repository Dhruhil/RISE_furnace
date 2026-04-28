#!/usr/bin/env python3
"""
Two-panel time-series plots:
  LEFT:  Steel cylinder SURFACE temperature evolution
  RIGHT: Steel cylinder CENTRE temperature evolution
One figure per T_set, 5 cases each.
All plots truncated to t <= 3600s.
Legend shows final temperature for each case at t = 3600s.

Surface = cells within 2 mm of outermost radial position
Centre  = cells within 7.5 mm of cylinder axis (y, z)
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

TARGET_TSETS_K    = [1173, 1223, 1273, 1323, 1373]
N_CASES_PER_PLOT  = 5
RNG_SEED          = 42

SURFACE_DEPTH_MM  = 2.0
CENTRE_RADIUS_MM  = 7.5

T_MAX_S           = 3600.0   # ← truncate all data to t <= 3600s

# ──────────────── STYLE ────────────────
mpl.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.labelsize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   8,
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

def get_region_masks(coords):
    y_c = (coords[:, 1].min() + coords[:, 1].max()) / 2
    z_c = (coords[:, 2].min() + coords[:, 2].max()) / 2
    r_m = np.sqrt((coords[:, 1] - y_c) ** 2 + (coords[:, 2] - z_c) ** 2) * 1000
    r_max = r_m.max()
    centre_mask  = r_m < CENTRE_RADIUS_MM
    surface_mask = r_m > (r_max - SURFACE_DEPTH_MM)
    return centre_mask, surface_mask

# ──────────────── LOAD DATA ────────────────
print(f"Loading dataset: {H5_PATH}")
print(f"Truncating data to t <= {T_MAX_S:.0f}s")

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

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)
    ax_surf, ax_ctr = axes
    cmap = plt.cm.viridis(np.linspace(0.15, 0.85, n))

    for i, case_key in enumerate(case_keys):
        c = h5[case_key]
        times   = c['times'][:]
        coords  = c['steel_cylinder']['coords'][:]
        T_steel = c['steel_cylinder']['T'][:]

        # ─── Truncate to t <= T_MAX_S ───
        time_mask = times <= T_MAX_S
        times_cropped = times[time_mask]
        T_steel_cropped = T_steel[time_mask]

        centre_mask, surface_mask = get_region_masks(coords)
        if surface_mask.sum() == 0 or centre_mask.sum() == 0:
            continue

        T_surf = T_steel_cropped[:, surface_mask].mean(axis=1) - 273.15
        T_ctr  = T_steel_cropped[:, centre_mask].mean(axis=1)  - 273.15

        try:
            _, cx, cy = parse_name(c.attrs['name'])
            short = case_key.replace('case_', 'c')
            geo = f"cx={cx:+4d}mm cy={cy:3d}mm"
        except:
            short = case_key
            geo = ""

        # Final temp at the truncated end (closest to 3600s)
        T_surf_final = T_surf[-1]
        T_ctr_final  = T_ctr[-1]
        t_final      = times_cropped[-1]

        label_surf = f"{short}  {geo}  →  {T_surf_final:6.1f}°C"
        label_ctr  = f"{short}  {geo}  →  {T_ctr_final:6.1f}°C"

        ax_surf.plot(times_cropped, T_surf, color=cmap[i], lw=1.8, label=label_surf)
        ax_ctr.plot(times_cropped,  T_ctr,  color=cmap[i], lw=1.8, label=label_ctr)

    # T_set reference line
    for ax in axes:
        ax.axhline(T_set_C, color='red', ls='--', lw=1.2, alpha=0.7,
                   label=f'T_set = {T_set_C}°C')

    # X-axis fixed range
    for ax in axes:
        ax.set_xlim(0, T_MAX_S)

    # Labels and titles
    ax_surf.set_xlabel('Time [s]')
    ax_ctr.set_xlabel('Time [s]')
    ax_surf.set_ylabel('Temperature [°C]')

    avg_n_surf = "~" + str(int(np.mean([
        get_region_masks(h5[ck]['steel_cylinder']['coords'][:])[1].sum()
        for ck in case_keys
    ])))
    avg_n_ctr  = "~" + str(int(np.mean([
        get_region_masks(h5[ck]['steel_cylinder']['coords'][:])[0].sum()
        for ck in case_keys
    ])))

    ax_surf.set_title(f'SURFACE temperature\n'
                      f'(within {SURFACE_DEPTH_MM:.0f} mm of outermost, {avg_n_surf} cells)',
                      fontweight='bold', fontsize=11)
    ax_ctr.set_title(f'CENTRE temperature\n'
                     f'(within {CENTRE_RADIUS_MM:.1f} mm of axis, {avg_n_ctr} cells)',
                     fontweight='bold', fontsize=11)

    # Legend with monospace font for clean alignment
    leg_props = dict(loc='lower right', fontsize=8,
                     framealpha=0.95, prop={'family': 'monospace', 'size': 8})
    ax_surf.legend(**leg_props)
    ax_ctr.legend(**leg_props)

    fig.suptitle(f'Steel cylinder: SURFACE vs CENTRE heating  '
                 f'|  T_set = {T_set_K} K ({T_set_C}°C)  |  n = {n} cases  '
                 f'|  Time window: 0–{T_MAX_S:.0f}s',
                 fontsize=13, fontweight='bold', y=1.02)

    out = OUT_DIR / f'lines_2panel_Tset_{T_set_K}K_{T_set_C}C.png'
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    print(f"  Saved: {out.name}")

# ──────────────── GENERATE PLOTS ────────────────
print(f"\nGenerating two-panel plots ...")
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

print(f"\n═══ All plots saved in: {OUT_DIR.absolute()}")

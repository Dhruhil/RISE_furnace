#!/usr/bin/env python3
"""
One plot per T_set: all cases shown as lines on single figure.
COLOR encodes cx position, LINESTYLE encodes cy position.
Left panel = centre T, right panel = surface T.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
import re
from pathlib import Path

# ──────────────── CONFIG ────────────────
H5_PATH = "dataset_v2_all_regions_clean.h5"
OUT_DIR = Path("../plots_lines_unified")
OUT_DIR.mkdir(exist_ok=True)

SHELL_FRACTION = 0.15  # inner/outer 15%

# Parameter grid
CX_VALUES = [-140, -100, -60, -20, 0, 20, 60, 100, 140]
CY_VALUES = [120, 150, 180, 210, 240]

# Style encoding maps
# Use viridis-like 9-step palette for cx (perceptually uniform)
cx_cmap = plt.cm.viridis
CX_COLORS = {
    cx: cx_cmap(i / (len(CX_VALUES) - 1))
    for i, cx in enumerate(CX_VALUES)
}

# Linestyles for cy (5 distinct styles)
CY_STYLES = {
    120: '-',            # solid
    150: '--',           # dashed
    180: '-.',           # dashdot
    210: ':',            # dotted
    240: (0, (5, 1, 1, 1)),  # custom: dash-dot-dot
}

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
})

# ──────────────── HELPERS ────────────────
def parse_name(name):
    t  = int(re.search(r'Tset(\d+)',   name).group(1))
    cx = int(re.search(r'cx(-?\d+)mm', name).group(1))
    cy = int(re.search(r'cy(-?\d+)mm', name).group(1))
    return t, cx, cy

def split_cells_by_radius(coords):
    y_c = (coords[:,1].min() + coords[:,1].max()) / 2
    z_c = (coords[:,2].min() + coords[:,2].max()) / 2
    r = np.sqrt((coords[:,1]-y_c)**2 + (coords[:,2]-z_c)**2)
    rmax = r.max()
    return r <= rmax*SHELL_FRACTION, r >= rmax*(1-SHELL_FRACTION)

# ──────────────── LOAD ────────────────
print("Loading cases from HDF5 ...")
all_cases = []
with h5py.File(H5_PATH, 'r') as f:
    for cid in sorted(f.keys()):
        if not cid.startswith('case_'): continue
        c = f[cid]
        t_set, cx_mm, cy_mm = parse_name(c.attrs['name'])
        times  = c['times'][:]
        coords = c['steel_cylinder']['coords'][:]
        T_hist = c['steel_cylinder']['T'][:]
        c_mask, s_mask = split_cells_by_radius(coords)
        all_cases.append({
            'case_id':   cid,
            'T_set':     t_set, 'cx': cx_mm, 'cy': cy_mm,
            'times':     times,
            'T_centre':  np.nanmean(T_hist[:, c_mask], axis=1) - 273.15,
            'T_surface': np.nanmean(T_hist[:, s_mask], axis=1) - 273.15,
        })

T_set_values = sorted(set(c['T_set'] for c in all_cases))
print(f"Loaded {len(all_cases)} cases.")

# ──────────────── PLOT FUNCTION ────────────────
def plot_unified(cases, T_set_K, out_path):
    n = len(cases)
    T_set_C = T_set_K - 273.15

    fig, (ax_c, ax_s) = plt.subplots(1, 2, figsize=(16, 7),
                                       sharey=True,
                                       gridspec_kw={'wspace': 0.12})

    # Plot all lines on both panels
    for case in cases:
        color = CX_COLORS.get(case['cx'], 'gray')
        style = CY_STYLES.get(case['cy'], '-')

        ax_c.plot(case['times'], case['T_centre'],
                  color=color, linestyle=style,
                  linewidth=1.6, alpha=0.85, zorder=3)
        ax_s.plot(case['times'], case['T_surface'],
                  color=color, linestyle=style,
                  linewidth=1.6, alpha=0.85, zorder=3)

    # T_set reference line on both panels
    for ax in (ax_c, ax_s):
        ax.axhline(T_set_C, color='#4A1B0C', linestyle=':',
                   linewidth=1.5, alpha=0.7, zorder=2)
        # Box label inside plot
        ax.text(0.02, T_set_C + 20, f'T_set = {T_set_C:.0f}°C',
                transform=ax.get_yaxis_transform(),
                color='#4A1B0C', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='white', edgecolor='#4A1B0C',
                          linewidth=0.8, alpha=0.9))

    # Titles
    ax_c.set_title(f'CENTRE temperature (inner {int(SHELL_FRACTION*100)}%)',
                   fontweight='bold')
    ax_s.set_title(f'SURFACE temperature (outer {int(SHELL_FRACTION*100)}%)',
                   fontweight='bold')

    # Labels
    ax_c.set_xlabel('Time (s)')
    ax_s.set_xlabel('Time (s)')
    ax_c.set_ylabel('Steel cylinder temperature (°C)')

    # Grid
    for ax in (ax_c, ax_s):
        ax.grid(alpha=0.25, linestyle='--', linewidth=0.5)
        ax.set_xlim(0, None)

    # ─── BUILD TWO-PART LEGEND ───
    # Part 1: cx colours
    legend_cx = [
        Line2D([0], [0], color=CX_COLORS[cx], linewidth=2.5,
               label=f'cx = {cx:+d} mm')
        for cx in CX_VALUES
    ]
    # Part 2: cy linestyles
    legend_cy = [
        Line2D([0], [0], color='black', linestyle=CY_STYLES[cy],
               linewidth=2, label=f'cy = {cy} mm')
        for cy in CY_VALUES
    ]

    # Place two legends: left side for cx, right side for cy
    leg1 = ax_s.legend(handles=legend_cx, title='Colour: cx position',
                        loc='upper left', bbox_to_anchor=(1.02, 1.0),
                        frameon=True, framealpha=0.95, fontsize=9,
                        title_fontsize=10)
    leg1.get_title().set_fontweight('bold')
    ax_s.add_artist(leg1)

    leg2 = ax_s.legend(handles=legend_cy, title='Style: cy position',
                        loc='upper left', bbox_to_anchor=(1.02, 0.45),
                        frameon=True, framealpha=0.95, fontsize=9,
                        title_fontsize=10)
    leg2.get_title().set_fontweight('bold')

    # Super title
    fig.suptitle(
        f'Steel cylinder temperature evolution — T_set = {T_set_K} K ({T_set_C:.0f}°C)\n'
        f'{n} converged cases',
        fontsize=14, fontweight='bold', y=1.00
    )

    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path.name}  ({n} cases)")

# ──────────────── GENERATE PLOTS ────────────────
print("\nGenerating unified line plots (one per T_set)...")
for T_set in T_set_values:
    sub = [c for c in all_cases if c['T_set'] == T_set]
    if not sub: continue
    out_path = OUT_DIR / f'unified_Tset_{T_set}K_{T_set-273}C.png'
    plot_unified(sub, T_set, out_path)

# ──────────────── SUMMARY ────────────────
print("\n" + "="*70)
print("  UNIFIED LINE PLOT SUMMARY")
print("="*70)
print(f"{'T_set (K)':>10} {'°C':>5} {'cases':>6}  "
      f"{'cx values present':>35}")
print("-"*70)
for T_set in T_set_values:
    sub = [c for c in all_cases if c['T_set'] == T_set]
    if not sub: continue
    cxs = sorted(set(c['cx'] for c in sub))
    print(f"{T_set:>10} {T_set-273:>5} {len(sub):>6}  "
          f"{str(cxs):>35}")

print(f"\nAll plots saved in: {OUT_DIR.absolute()}")

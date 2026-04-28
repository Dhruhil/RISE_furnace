#!/usr/bin/env python3
"""
Generate 5 separate figures, one per T_set value.
Each figure has 2 panels:
  LEFT:  Steel cylinder CENTRE temperature vs time
  RIGHT: Steel cylinder SURFACE temperature vs time
All cases at that T_set are plotted as individual lines.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import re
from pathlib import Path

# ════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════
H5_PATH = "dataset_v2_all_regions_clean.h5"

# Output folder at the Simulating_Heat_Treatment... root
OUT_DIR = Path("../plots_temperature_evolution")
OUT_DIR.mkdir(exist_ok=True)

SHELL_FRACTION = 0.15      # inner 15% = centre; outer 15% = surface
TOP_N_HIGHLIGHT = 3        # bold the top N hottest cases

# ════════════════════════════════════════════════════════════════
#  PUBLICATION-QUALITY STYLE
# ════════════════════════════════════════════════════════════════
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
    'axes.linewidth':    0.9,
    'axes.edgecolor':    '#333333',
    'axes.grid':         True,
    'grid.alpha':        0.25,
    'grid.linestyle':    '--',
    'grid.linewidth':    0.5,
})

# Yellow → Red colormap (low T = pale yellow, high T = dark red)
yellow_red = LinearSegmentedColormap.from_list(
    'yellow_red',
    [(0.0, '#FFF5B0'), (0.2, '#FFE066'), (0.4, '#FFB347'),
     (0.6, '#FF7F2A'), (0.8, '#E63B1F'), (1.0, '#8B0000')],
    N=256
)

# ════════════════════════════════════════════════════════════════
#  HELPERS
# ════════════════════════════════════════════════════════════════
def parse_name(name):
    t  = int(re.search(r'Tset(\d+)',   name).group(1))
    cx = int(re.search(r'cx(-?\d+)mm', name).group(1))
    cy = int(re.search(r'cy(-?\d+)mm', name).group(1))
    return t, cx, cy

def split_cells_by_radius(coords):
    """
    Split cylinder cells into inner-core (centre) and outer-shell (surface) masks.
    Uses radial distance from the cylinder's axis (which runs along x).
    """
    y_c = (coords[:, 1].min() + coords[:, 1].max()) / 2
    z_c = (coords[:, 2].min() + coords[:, 2].max()) / 2
    r = np.sqrt((coords[:, 1] - y_c)**2 + (coords[:, 2] - z_c)**2)
    r_max = r.max()
    centre_mask  = r <= r_max * SHELL_FRACTION           # inner 15%
    surface_mask = r >= r_max * (1 - SHELL_FRACTION)     # outer 15%
    return centre_mask, surface_mask

# ════════════════════════════════════════════════════════════════
#  LOAD DATA (time series per case)
# ════════════════════════════════════════════════════════════════
print("Loading cases from HDF5 ...")
all_cases = []
with h5py.File(H5_PATH, 'r') as f:
    for cid in sorted(f.keys()):
        if not cid.startswith('case_'): continue
        c = f[cid]
        t_set, cx_mm, cy_mm = parse_name(c.attrs['name'])
        times  = c['times'][:]                          # (n_times,)
        coords = c['steel_cylinder']['coords'][:]        # (n_cells, 3)
        T_hist = c['steel_cylinder']['T'][:]             # (n_times, n_cells)

        centre_mask, surface_mask = split_cells_by_radius(coords)

        # Per-timestep mean of centre and surface cells
        T_centre  = np.nanmean(T_hist[:, centre_mask],  axis=1) - 273.15
        T_surface = np.nanmean(T_hist[:, surface_mask], axis=1) - 273.15

        all_cases.append({
            'case_id':   cid,
            'T_set':     t_set,
            'cx':        cx_mm,
            'cy':        cy_mm,
            'times':     times,
            'T_centre':  T_centre,
            'T_surface': T_surface,
            'T_final':   float(T_surface[-1]),   # for colouring
        })

T_set_values = sorted(set(c['T_set'] for c in all_cases))
print(f"Loaded {len(all_cases)} cases. T_set values found: {T_set_values} K")

# ════════════════════════════════════════════════════════════════
#  PLOTTING
# ════════════════════════════════════════════════════════════════
def plot_one_Tset(cases, T_set_K, out_path):
    """Two-panel figure (centre + surface) for one T_set."""
    n = len(cases)
    T_set_C = T_set_K - 273.15

    fig, (ax_centre, ax_surface) = plt.subplots(
        1, 2, figsize=(15, 6.5), sharey=True
    )

    # Order cases by final surface temperature so the colorbar is meaningful
    cases_sorted = sorted(cases, key=lambda c: c['T_final'])
    T_finals = [c['T_final'] for c in cases_sorted]

    # Colour normalisation (local scale per T_set — shows variation best)
    if len(T_finals) >= 2:
        vmin = min(T_finals) - 2
        vmax = max(T_finals) + 2
    else:
        vmin, vmax = T_finals[0] - 20, T_finals[0] + 20
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Identify top-N hottest for highlighting
    top_hottest = set(c['case_id'] for c in cases_sorted[-TOP_N_HIGHLIGHT:])

    # ─── Plot each case on both panels ───
    for case in cases_sorted:
        color = yellow_red(norm(case['T_final']))
        is_top = case['case_id'] in top_hottest
        lw = 2.0 if is_top else 0.9
        alpha = 1.0 if is_top else 0.7
        zorder = 5 if is_top else 3

        ax_centre.plot(case['times'], case['T_centre'],
                       color=color, linewidth=lw, alpha=alpha, zorder=zorder)
        ax_surface.plot(case['times'], case['T_surface'],
                        color=color, linewidth=lw, alpha=alpha, zorder=zorder)

    # ─── Annotate top-3 hottest with case labels on surface panel ───
    for case in cases_sorted[-TOP_N_HIGHLIGHT:]:
        label = case['case_id'].replace('case_', 'c')
        # Annotate near the end of the line
        x_last = case['times'][-1]
        y_last = case['T_surface'][-1]
        ax_surface.annotate(
            f"{label} (cx={case['cx']},cy={case['cy']})",
            xy=(x_last, y_last),
            xytext=(8, 0), textcoords='offset points',
            fontsize=8, color='#2C0D05', fontweight='bold',
            va='center'
        )

    # ─── T_set reference line on both panels ───
    for ax in (ax_centre, ax_surface):
        ax.axhline(T_set_C, color='#4A1B0C', linestyle=':',
                   linewidth=1.3, alpha=0.7, zorder=2,
                   label=f'T_set = {T_set_C:.0f}°C')

    # ─── Axis formatting ───
    ax_centre.set_title(f'CENTRE temperature (inner {int(SHELL_FRACTION*100)}% of cylinder)',
                        fontweight='bold')
    ax_surface.set_title(f'SURFACE temperature (outer {int(SHELL_FRACTION*100)}% of cylinder)',
                         fontweight='bold')

    ax_centre.set_xlabel('Time (s)')
    ax_surface.set_xlabel('Time (s)')
    ax_centre.set_ylabel('Steel cylinder temperature (°C)')

    for ax in (ax_centre, ax_surface):
        ax.set_xlim(0, None)
        ax.legend(loc='lower right', framealpha=0.9)

    # ─── Shared colorbar on the right side ───
    sm = ScalarMappable(cmap=yellow_red, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax_centre, ax_surface],
                        shrink=0.85, pad=0.02)
    cbar.set_label('Final surface T (°C)', fontsize=11)

    # ─── Super title ───
    fig.suptitle(
        f'Steel cylinder temperature evolution — T_set = {T_set_K} K ({T_set_C:.0f}°C)\n'
        f'{n} converged cases',
        fontsize=14, fontweight='bold', y=1.02
    )

    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path.name}  ({n} cases)")

# ════════════════════════════════════════════════════════════════
#  GENERATE ALL 5 PLOTS
# ════════════════════════════════════════════════════════════════
print("\nGenerating 5 temperature-evolution figures...")
for T_set in T_set_values:
    sub = [c for c in all_cases if c['T_set'] == T_set]
    if not sub:
        print(f"  Skipped T_set = {T_set} K (no cases)")
        continue
    T_set_C = T_set - 273
    out_path = OUT_DIR / f'temperature_evolution_Tset_{T_set}K_{T_set_C}C.png'
    plot_one_Tset(sub, T_set, out_path)

# ════════════════════════════════════════════════════════════════
#  SUMMARY TABLE
# ════════════════════════════════════════════════════════════════
print("\n" + "="*80)
print("  SUMMARY PER T_SET (final surface temperature)")
print("="*80)
print(f"{'T_set (K)':>10} {'T_set (°C)':>11} {'N cases':>9} "
      f"{'T_surf_min':>11} {'T_surf_max':>11} {'T_surf_mean':>12}")
print("-"*80)
for T_set in T_set_values:
    sub = [c for c in all_cases if c['T_set'] == T_set]
    if not sub: continue
    T_finals = [c['T_final'] for c in sub]
    print(f"{T_set:>10} {T_set-273:>11} {len(sub):>9} "
          f"{min(T_finals):>11.1f} {max(T_finals):>11.1f} "
          f"{np.mean(T_finals):>12.1f}")

print(f"\nAll plots saved in: {OUT_DIR.absolute()}")
print("\nDone.")

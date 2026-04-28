#!/usr/bin/env python3
"""
V5: Clean layout.
- T_set shown in super title only (no floating labels)
- Ranked HOTTEST at top -> COLDEST at bottom
- Dotted reference line at T_set value (unlabeled)
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import re
from pathlib import Path

# ──────────────── CONFIG ────────────────
H5_PATH        = "dataset_v2_all_regions_clean.h5"
OUT_DIR        = Path("../plots_bar_V5")
OUT_DIR.mkdir(exist_ok=True)

TARGET_TIME       = 3460.0
SURFACE_DEPTH_MM  = 2.0
CENTRE_RADIUS_MM  = 7.5

# ──────────────── STYLE ────────────────
mpl.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   9,
    'figure.dpi':        120,
    'savefig.dpi':       220,
    'savefig.bbox':      'tight',
})

yellow_red = LinearSegmentedColormap.from_list(
    'yellow_red',
    [(0.00, '#FFFFCC'), (0.15, '#FFEB84'), (0.30, '#FFC947'),
     (0.45, '#FD9833'), (0.60, '#F65E24'), (0.75, '#D62F1C'),
     (0.90, '#A50F15'), (1.00, '#67000D')],
    N=256
)

# ──────────────── HELPERS ────────────────
def parse_name(name):
    t  = int(re.search(r'Tset(\d+)',   name).group(1))
    cx = int(re.search(r'cx(-?\d+)mm', name).group(1))
    cy = int(re.search(r'cy(-?\d+)mm', name).group(1))
    return t, cx, cy

def get_region_masks(coords):
    y_c = (coords[:,1].min() + coords[:,1].max()) / 2
    z_c = (coords[:,2].min() + coords[:,2].max()) / 2
    r_m = np.sqrt((coords[:,1]-y_c)**2 + (coords[:,2]-z_c)**2) * 1000
    r_max = r_m.max()
    return r_m < CENTRE_RADIUS_MM, r_m > (r_max - SURFACE_DEPTH_MM), r_max

# ──────────────── LOAD DATA ────────────────
print(f"Loading at t = {TARGET_TIME:.0f}s ...")
all_records = []
with h5py.File(H5_PATH, 'r') as f:
    for cid in sorted(f.keys()):
        if not cid.startswith('case_'): continue
        c = f[cid]
        t_set, cx, cy = parse_name(c.attrs['name'])
        times  = c['times'][:]
        coords = c['steel_cylinder']['coords'][:]
        T_hist = c['steel_cylinder']['T'][:]

        idx = int(np.argmin(np.abs(times - TARGET_TIME)))
        T_field = T_hist[idx]

        c_mask, s_mask, r_max = get_region_masks(coords)
        if c_mask.sum() == 0 or s_mask.sum() == 0:
            continue

        all_records.append({
            'case':      cid,
            'T_set':     t_set, 'cx': cx, 'cy': cy,
            'T_surf':    float(np.nanmean(T_field[s_mask])) - 273.15,
            'T_centre':  float(np.nanmean(T_field[c_mask])) - 273.15,
            'n_surface': int(s_mask.sum()),
            'n_centre':  int(c_mask.sum()),
        })

T_set_values = sorted(set(r['T_set'] for r in all_records))
print(f"Loaded {len(all_records)} cases.")

# ──────────────── PLOT FUNCTION ────────────────
def plot_two_panel(records, T_set_K, out_path):
    n = len(records)
    T_set_C = T_set_K - 273.15

    # ═══ KEY: sort HOTTEST FIRST so it appears at TOP of chart ═══
    # matplotlib's barh draws y=0 at bottom by default
    # We place hottest at y=n-1 (top), coldest at y=0 (bottom)
    records_sorted = sorted(records, key=lambda r: -r['T_surf'])  # hottest first
    # Reverse for plotting (matplotlib puts index 0 at bottom by default)
    # So we'll explicitly invert y-axis below

    T_surf_list   = [r['T_surf']   for r in records_sorted]
    T_centre_list = [r['T_centre'] for r in records_sorted]
    all_T = T_surf_list + T_centre_list
    vmin, vmax = min(all_T), max(all_T)
    if vmax - vmin < 30:
        mid = (vmax + vmin) / 2
        vmin, vmax = mid - 20, mid + 20
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Figure height
    fig_height = max(8.0, 0.40 * n + 5.0)
    fig = plt.figure(figsize=(17, fig_height))

    # Reserve top 20% for super title (critical for small-case plots)
    gs = fig.add_gridspec(
        1, 2,
        left=0.08, right=0.91,
        top=0.80,
        bottom=0.08, wspace=0.28
    )
    ax_s = fig.add_subplot(gs[0, 0])
    ax_c = fig.add_subplot(gs[0, 1], sharey=ax_s)

    # y positions: 0 = hottest (we'll invert axis to put y=0 at top)
    y = np.arange(n)
    labels = [
        f"c{r['case'].replace('case_','')}  cx={r['cx']:+4d}  cy={r['cy']:3d}"
        for r in records_sorted
    ]

    # ═══ LEFT PANEL: SURFACE ═══
    colors_s = [yellow_red(norm(t)) for t in T_surf_list]
    ax_s.barh(y, T_surf_list, color=colors_s,
              edgecolor='black', linewidth=0.5, height=0.78, zorder=3)

    for i, t_val in enumerate(T_surf_list):
        ax_s.text(t_val + (vmax-vmin)*0.005, i, f'{t_val:.1f}',
                  va='center', ha='left', fontsize=8,
                  fontweight='bold', color='#2C0D05', zorder=4)

    # T_set reference line (unlabeled, just visual aid)
    ax_s.axvline(T_set_C, color='#4A1B0C', linestyle=':',
                 linewidth=1.8, alpha=0.65, zorder=2)

    ax_s.set_yticks(y)
    ax_s.set_yticklabels(labels, family='monospace', fontsize=8)
    ax_s.set_xlabel(f'SURFACE T at t = {TARGET_TIME:.0f}s (°C)', fontsize=11)
    avg_n_s = int(np.mean([r["n_surface"] for r in records_sorted]))
    ax_s.set_title(
        f'SURFACE temperature\n'
        f'(cells within {SURFACE_DEPTH_MM:.0f} mm of outermost, ~{avg_n_s} cells)',
        fontweight='bold', fontsize=11, pad=15
    )
    ax_s.set_xlim(vmin - (vmax-vmin)*0.04, vmax + (vmax-vmin)*0.14)
    ax_s.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)
    ax_s.set_axisbelow(True)
    ax_s.invert_yaxis()  # ← hottest (index 0) ends up at TOP

    # ═══ RIGHT PANEL: CENTRE ═══
    colors_c = [yellow_red(norm(t)) for t in T_centre_list]
    ax_c.barh(y, T_centre_list, color=colors_c,
              edgecolor='black', linewidth=0.5, height=0.78, zorder=3)

    for i, t_val in enumerate(T_centre_list):
        ax_c.text(t_val + (vmax-vmin)*0.005, i, f'{t_val:.1f}',
                  va='center', ha='left', fontsize=8,
                  fontweight='bold', color='#2C0D05', zorder=4)

    ax_c.axvline(T_set_C, color='#4A1B0C', linestyle=':',
                 linewidth=1.8, alpha=0.65, zorder=2)

    ax_c.set_xlabel(f'CENTRE T at t = {TARGET_TIME:.0f}s (°C)', fontsize=11)
    avg_n_c = int(np.mean([r["n_centre"] for r in records_sorted]))
    ax_c.set_title(
        f'CENTRE temperature\n'
        f'(cells within {CENTRE_RADIUS_MM:.1f} mm of axis, ~{avg_n_c} cells)',
        fontweight='bold', fontsize=11, pad=15
    )
    ax_c.set_xlim(vmin - (vmax-vmin)*0.04, vmax + (vmax-vmin)*0.14)
    ax_c.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5, zorder=1)
    ax_c.set_axisbelow(True)
    # ax_c shares y with ax_s, so inversion propagates automatically

    # ═══ COLORBAR ═══
    cbar_ax = fig.add_axes([0.925, 0.15, 0.012, 0.55])
    sm = ScalarMappable(cmap=yellow_red, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f'T at t={TARGET_TIME:.0f}s (°C)', fontsize=10)

    # ═══ SUPER TITLE (contains T_set) ═══
    hot = records_sorted[0]   # hottest is first in sorted list
    cld = records_sorted[-1]  # coldest is last
    fig.text(0.5, 0.95,
        f'Steel cylinder: SURFACE vs CENTRE temperature at t = {TARGET_TIME:.0f} s '
        f'({TARGET_TIME/60:.0f} min)  |  T_set = {T_set_K} K ({T_set_C:.0f}°C)',
        ha='center', va='top', fontsize=13, fontweight='bold'
    )
    fig.text(0.5, 0.915,
        f'{n} cases, ranked HOTTEST (top) to COLDEST (bottom) by surface T '
        f'— dotted line = T_set',
        ha='center', va='top', fontsize=11
    )
    fig.text(0.5, 0.885,
        f"hottest: c{hot['case'].replace('case_','')} "
        f"(cx={hot['cx']:+d}, cy={hot['cy']}) surf={hot['T_surf']:.1f}°C  |  "
        f"coldest: c{cld['case'].replace('case_','')} "
        f"(cx={cld['cx']:+d}, cy={cld['cy']}) surf={cld['T_surf']:.1f}°C",
        ha='center', va='top', fontsize=10, color='#4A1B0C'
    )

    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path.name}  ({n} cases)")

# ──────────────── GENERATE ALL PLOTS ────────────────
print(f"\nGenerating bar charts at t = {TARGET_TIME:.0f}s ...")
for T_set in T_set_values:
    sub = [r for r in all_records if r['T_set'] == T_set]
    if not sub: continue
    out_path = OUT_DIR / f'bar_V5_Tset_{T_set}K_{T_set-273}C_t{int(TARGET_TIME)}s.png'
    plot_two_panel(sub, T_set, out_path)

# ──────────────── SUMMARY ────────────────
print("\n" + "="*90)
print(f"  SUMMARY AT t = {TARGET_TIME:.0f}s")
print("="*90)
print(f"{'T_set (K)':>10} {'°C':>5} {'N':>4}  "
      f"{'surf_min':>9} {'surf_max':>9} "
      f"{'ctr_min':>9} {'ctr_max':>9} "
      f"{'max dT':>8}")
print("-"*90)
for T_set in T_set_values:
    sub = [r for r in all_records if r['T_set'] == T_set]
    if not sub: continue
    Ts = [r['T_surf'] for r in sub]
    Tc = [r['T_centre'] for r in sub]
    dT = [r['T_surf'] - r['T_centre'] for r in sub]
    print(f"{T_set:>10} {T_set-273:>5} {len(sub):>4}  "
          f"{min(Ts):>9.1f} {max(Ts):>9.1f} "
          f"{min(Tc):>9.1f} {max(Tc):>9.1f} "
          f"{max(dT):>8.2f}")

print(f"\nAll plots saved in: {OUT_DIR.absolute()}")

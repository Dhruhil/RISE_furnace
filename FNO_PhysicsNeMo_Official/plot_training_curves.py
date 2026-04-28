#!/usr/bin/env python3
"""
FNO training curves — 3-panel figure for thesis.
  Panel 1: Loss curves (Train data + Val data)
  Panel 2: Validation MAE [K]
  Panel 3: Validation R²
Best epoch marked with vertical dashed line.

Log column format:
  Ep | TrWLoss | TrData | TrPhys | VaWLoss | VaData | MAE | Steel | R2 | lam | w2 | Time
"""
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ──────────────── CONFIG ────────────────
LOG_PATH  = Path("outputs/FNO_v5_FIX_150ep_20260425_1040/logs/fno_v5_6494264.log")
OUT_DIR   = Path("outputs/FNO_v5_FIX_150ep_20260425_1040/evaluation/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────── STYLE ────────────────
mpl.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    12,
    'axes.labelsize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'figure.dpi':        120,
    'savefig.dpi':       220,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'grid.linestyle':    '--',
})

# ──────────────── PARSE LOG ────────────────
print(f"Parsing log: {LOG_PATH}")
records = []
# Pattern matches lines like:
#   82 |   0.09646 |   0.00121 |  31.61001 |   0.00226 |   0.00284 |   8.26 |   2.21 |  0.9923 | 0.0030 | 0.50 |   840s *
# (the trailing " *" or "" is optional)
pattern = re.compile(
    r'^\s*(\d+)\s*\|'        # 1 Ep
    r'\s*([\d.e+\-]+)\s*\|'  # 2 TrWLoss
    r'\s*([\d.e+\-]+)\s*\|'  # 3 TrData
    r'\s*([\d.e+\-]+)\s*\|'  # 4 TrPhys
    r'\s*([\d.e+\-]+)\s*\|'  # 5 VaWLoss
    r'\s*([\d.e+\-]+)\s*\|'  # 6 VaData
    r'\s*([\d.e+\-]+)\s*\|'  # 7 MAE
    r'\s*([\d.e+\-]+)\s*\|'  # 8 Steel
    r'\s*([\d.e+\-]+)\s*\|'  # 9 R2
    r'\s*([\d.e+\-]+)\s*\|'  # 10 lam
    r'\s*([\d.e+\-]+)\s*\|'  # 11 w2
    r'\s*(\d+)s\s*\*?\s*$'   # 12 Time (with optional * marker)
)
with open(LOG_PATH) as f:
    for line in f:
        m = pattern.match(line)
        if m:
            records.append({
                'epoch':    int(m.group(1)),
                'tr_data':  float(m.group(3)),
                'va_data':  float(m.group(6)),
                'mae_k':    float(m.group(7)),
                'steel_k':  float(m.group(8)),
                'r2':       float(m.group(9)),
            })

df = pd.DataFrame(records)
print(f"Parsed {len(df)} epochs")

# Best epoch — using overall MAE (could also use steel_k for thesis focus)
best_idx   = df['mae_k'].idxmin()
best_epoch = int(df.loc[best_idx, 'epoch'])
best_mae   = df.loc[best_idx, 'mae_k']
best_r2    = df.loc[best_idx, 'r2']

print(f"Best val MAE: {best_mae:.3f} K at epoch {best_epoch}")
print(f"Best val R² (at best MAE): {best_r2:.4f}")
print(f"Final val MAE: {df.iloc[-1]['mae_k']:.3f} K")

COLOR_BEST = '#2ca02c'  # forest green

# ════════════════════════════════════════════════════════════════════
# 3-PANEL FIGURE
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# ─── Panel 1: Loss curves (log-y) ───
ax = axes[0]
ax.semilogy(df['epoch'], df['tr_data'], color='#1f77b4', lw=1.8,
            label='Training loss')
ax.semilogy(df['epoch'], df['va_data'], color='#d62728', lw=1.8,
            label='Validation loss')
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss (log scale)')
ax.set_title('Loss curves', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, which='both', alpha=0.3, ls='--')

# ─── Panel 2: Validation MAE ───
ax = axes[1]
ax.plot(df['epoch'], df['mae_k'], color='#d62728', lw=1.8,
        label='Validation MAE')
ax.axvline(best_epoch, color=COLOR_BEST, ls='--', lw=1.5, alpha=0.85,
           label=f'Best epoch ({best_epoch}): MAE = {best_mae:.2f} K')
ax.axhline(best_mae, color=COLOR_BEST, ls=':', lw=1.0, alpha=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation MAE [K]')
ax.set_title('Validation MAE', fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3, ls='--')

# ─── Panel 3: Validation R² ───
ax = axes[2]
ax.plot(df['epoch'], df['r2'], color='#9467bd', lw=1.8, label='Validation R²')
ax.axvline(best_epoch, color=COLOR_BEST, ls='--', lw=1.5, alpha=0.85,
           label=f'Best epoch ({best_epoch}): R² = {best_r2:.4f}')
ax.axhline(best_r2, color=COLOR_BEST, ls=':', lw=1.0, alpha=0.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('Validation R²')
ax.set_title('Validation R²', fontweight='bold')
ax.set_ylim(0.93, 1.005)
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3, ls='--')

# Super-title
fig.suptitle(
    f'FNO training curves — 150 epochs   |   '
    f'Best val MAE = {best_mae:.2f} K @ epoch {best_epoch}',
    fontsize=13, fontweight='bold', y=1.02)

plt.tight_layout()
out = OUT_DIR / 'training_curves_3panel.png'
plt.savefig(out)
plt.close()
print(f"\n  Saved: {out}")

print(f"\n═══ Done. Plot: {out} ═══")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import sys
sys.path.insert(0, "/mimer/NOBACKUP/groups/revar/jinis/python_packages")

EXP_XLSX = "/mimer/NOBACKUP/groups/revar/jinis/DeepONet_Unified/experimental_cleaned.xlsx"
OUT_DIR  = Path("/mimer/NOBACKUP/groups/revar/jinis/DeepONet_Unified/outputs/plots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    "font.family":"sans-serif","font.size":11,
    "axes.titlesize":12,"axes.labelsize":11,
    "xtick.labelsize":10,"ytick.labelsize":10,"legend.fontsize":10,
    "figure.dpi":150,"savefig.dpi":300,"savefig.bbox":"tight",
    "axes.linewidth":0.9,"xtick.direction":"in","ytick.direction":"in",
    "axes.grid":True,"grid.alpha":0.3,"grid.linewidth":0.5,
    "lines.linewidth":1.8,
})

df_raw = pd.read_excel(EXP_XLSX, sheet_name='Lab furnace results', header=None)
data   = df_raw.iloc[4:,[1,2,3,4]].copy()
data.columns = ['time_h','furnace_C','center_C','surface_C']
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data = data.dropna(subset=['time_h','center_C','surface_C']).reset_index(drop=True)
data['time_s'] = data['time_h'] * 3600.0

early = data[data['time_s'] < 600].copy()
dT_dt = np.gradient(early['surface_C'].values, early['time_s'].values)
insertion_idx = int(np.argmax(dT_dt))
data = data.iloc[insertion_idx:].reset_index(drop=True)
data['time_s'] = data['time_s'] - data['time_s'].iloc[0]
data['time_h'] = data['time_s'] / 3600.0

print(f"Rows: {len(data)}")
print(f"Time: {data['time_h'].max():.2f} h")
print(f"Final centre:  {data['center_C'].iloc[-1]:.1f} C")
print(f"Final surface: {data['surface_C'].iloc[-1]:.1f} C")

EXP_COLOR = "#CC5500"
t_h = data['time_h'].values
dT  = data['surface_C'].values - data['center_C'].values

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
    gridspec_kw={"hspace":0.10, "height_ratios":[1.6, 0.8]})

ax1.set_title("Temperature vs. time, RISE furnace — Experimental", fontsize=12, pad=10)
ax1.plot(t_h, data['surface_C'], color=EXP_COLOR, ls="-",  lw=2.0, label="Surface, experiment")
ax1.plot(t_h, data['center_C'],  color=EXP_COLOR, ls="--", lw=2.0, label="Centre, experiment")
ax1.set_ylabel("Temperature [°C]")
ax1.legend(loc="lower right", frameon=True, framealpha=0.95)
ax1.set_ylim(0, 1100)

ax2.plot(t_h, dT, color=EXP_COLOR, ls="-", lw=1.8, label="ΔT = Surface − Centre")
ax2.set_xlabel("Time [h]")
ax2.set_ylabel("ΔT [°C]")
ax2.legend(loc="upper right", frameon=True, framealpha=0.95)
ax2.set_xlim(0, t_h.max())
ax2.set_ylim(0, 35)

plt.subplots_adjust(left=0.09, right=0.97, top=0.94, bottom=0.09, hspace=0.10)
plt.savefig(OUT_DIR / "experiment_only.png")
plt.savefig(OUT_DIR / "experiment_only.pdf")
plt.close()
print("Saved!")

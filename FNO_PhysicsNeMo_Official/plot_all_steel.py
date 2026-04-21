import json, argparse, os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--h5", default="dataset_all_regions_66cases.h5")
parser.add_argument("--outdir", default="plots_all_cases")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

with h5py.File(args.h5, "r") as f:
    n_cases = int(f.attrs["n_cases"])
    regions_list = json.loads(f.attrs["regions"])
    print(f"Cases: {n_cases}, Regions: {regions_list}")

    fig1, ax1 = plt.subplots(figsize=(16, 9))
    t_set_all = []

    for ci in range(n_cases):
        grp = f[f"case_{ci:03d}"]
        T_set = float(grp.attrs["T_set"])
        times = grp["times"][:].astype(np.float32)
        t_set_all.append(T_set)
        if "steel_cylinder" not in grp:
            continue
        T = grp["steel_cylinder"]["T"][:].astype(np.float32)
        T_mean = T.mean(axis=1)
        color = plt.cm.plasma((T_set - 800) / 400)
        ax1.plot(times, T_mean, lw=1.0, color=color, alpha=0.8)

    sm = plt.cm.ScalarMappable(cmap="plasma",
         norm=plt.Normalize(vmin=min(t_set_all), vmax=max(t_set_all)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label("T_set [K]", fontsize=12)
    for ts in sorted(set(t_set_all)):
        ax1.axhline(ts, color="red", ls=":", lw=0.5, alpha=0.3)
    ax1.set_xlabel("Time [s]", fontsize=13)
    ax1.set_ylabel("Steel cylinder — mean T [K]", fontsize=13)
    ax1.set_title(f"OpenFOAM ground truth — steel cylinder — {n_cases} cases", fontsize=14)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(f"{args.outdir}/all_cases_steel_cylinder.png", dpi=200)
    print(f"Saved: {args.outdir}/all_cases_steel_cylinder.png")
    plt.close(fig1)

    for ci in range(n_cases):
        grp = f[f"case_{ci:03d}"]
        name = str(grp.attrs["name"])
        T_set = float(grp.attrs["T_set"])
        times = grp["times"][:].astype(np.float32)
        fig, ax = plt.subplots(figsize=(14, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, 12))
        ri = 0
        for region in regions_list:
            if region not in grp:
                continue
            T = grp[region]["T"][:].astype(np.float32)
            T_mean = T.mean(axis=1)
            lw = 2.5 if region == "steel_cylinder" else 1.0
            ls = "-" if region == "steel_cylinder" else "--"
            ax.plot(times, T_mean, lw=lw, ls=ls, color=colors[ri],
                    label=f"{region} ({T_mean[-1]:.0f}K)", alpha=0.9)
            ri += 1
        ax.axhline(T_set, color="red", ls="-", lw=2, alpha=0.5,
                   label=f"T_set = {T_set:.0f}K")
        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_ylabel("Mean temperature [K]", fontsize=12)
        ax.set_title(f"Case {ci:03d} — {name}\nT_set = {T_set:.0f}K", fontsize=11)
        ax.legend(fontsize=7, ncol=3, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{args.outdir}/case_{ci:03d}_all_regions.png", dpi=150)
        plt.close(fig)
        print(f"  case_{ci:03d} done")

    fig3, ax3 = plt.subplots(figsize=(14, 6))
    case_ids, t_set_vals, t_final_vals = [], [], []
    for ci in range(n_cases):
        grp = f[f"case_{ci:03d}"]
        T_set = float(grp.attrs["T_set"])
        if "steel_cylinder" not in grp:
            continue
        T = grp["steel_cylinder"]["T"][:].astype(np.float32)
        case_ids.append(ci)
        t_set_vals.append(T_set)
        t_final_vals.append(T[-1].mean())
    x = np.arange(len(case_ids))
    ax3.bar(x, t_set_vals, width=0.4, label="T_set (target)", color="tomato", alpha=0.7)
    ax3.bar(x + 0.4, t_final_vals, width=0.4, label="Steel T at t=4000s", color="steelblue", alpha=0.7)
    ax3.set_xticks(x + 0.2)
    ax3.set_xticklabels([f"{ci:03d}" for ci in case_ids], rotation=90, fontsize=6)
    ax3.set_xlabel("Case", fontsize=12)
    ax3.set_ylabel("Temperature [K]", fontsize=12)
    ax3.set_title("T_set vs actual steel cylinder temperature at t=4000s", fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")
    fig3.tight_layout()
    fig3.savefig(f"{args.outdir}/tset_vs_steel_final.png", dpi=200)
    print(f"Saved: {args.outdir}/tset_vs_steel_final.png")
    plt.close(fig3)

print(f"\nAll plots in: {args.outdir}/")

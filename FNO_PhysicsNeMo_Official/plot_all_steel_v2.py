import json, argparse, os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--h5", default="dataset_all_regions_66cases.h5")
parser.add_argument("--outdir", default="plots_all_cases_v2")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

with h5py.File(args.h5, "r") as f:
    n_cases = int(f.attrs["n_cases"])
    regions_list = json.loads(f.attrs["regions"])
    print(f"Cases: {n_cases}, Regions: {regions_list}")

    # ── Plot 1: All cases steel cylinder with case names ─────────
    fig1, ax1 = plt.subplots(figsize=(18, 10))
    t_set_all = []

    for ci in range(n_cases):
        grp = f[f"case_{ci:03d}"]
        name = str(grp.attrs["name"])
        T_set = float(grp.attrs["T_set"])
        times = grp["times"][:].astype(np.float32)
        t_set_all.append(T_set)
        if "steel_cylinder" not in grp:
            continue
        T = grp["steel_cylinder"]["T"][:].astype(np.float32)
        T_mean = T.mean(axis=1)
        color = plt.cm.plasma((T_set - 800) / 400)
        ax1.plot(times, T_mean, lw=1.0, color=color, alpha=0.8)

        # Add case name at the end of each curve
        short_name = name.replace("mm_", "").replace("_rho", " rho").replace("_Cp", " Cp").replace("_k", " k")
        ax1.annotate(f"c{ci:03d} T={T_set:.0f}", xy=(times[-1], T_mean[-1]),
                     fontsize=4, color=color, alpha=0.8, va="center",
                     xytext=(5, 0), textcoords="offset points")

    sm = plt.cm.ScalarMappable(cmap="plasma",
         norm=plt.Normalize(vmin=min(t_set_all), vmax=max(t_set_all)))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax1)
    cbar.set_label("T_set [K]", fontsize=12)
    for ts in sorted(set(t_set_all)):
        ax1.axhline(ts, color="red", ls=":", lw=0.5, alpha=0.3)
        ax1.text(50, ts + 5, f"T_set={ts:.0f}K", fontsize=7, color="red", alpha=0.5)
    ax1.set_xlabel("Time [s]", fontsize=13)
    ax1.set_ylabel("Steel cylinder — mean T [K]", fontsize=13)
    ax1.set_title(f"OpenFOAM ground truth — steel cylinder — {n_cases} cases\n"
                  f"Red dotted lines = T_set targets | Curves = actual steel temperature",
                  fontsize=14)
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(f"{args.outdir}/all_cases_steel_cylinder.png", dpi=200)
    print(f"Saved: {args.outdir}/all_cases_steel_cylinder.png")
    plt.close(fig1)

    # ── Plot 2: Per-case all regions with proper name ────────────
    for ci in range(n_cases):
        grp = f[f"case_{ci:03d}"]
        name = str(grp.attrs["name"])
        T_set = float(grp.attrs["T_set"])
        times = grp["times"][:].astype(np.float32)

        # Parse case name for readable title
        parts = name.split("_")
        title_parts = []
        for p in parts:
            if p.startswith("case"):
                title_parts.append(p)
            elif p.startswith("Tset"):
                title_parts.append(f"T_set={p[4:]}K")
            elif p.startswith("cy"):
                title_parts.append(f"cy={p[2:]}")
            elif p.startswith("cz"):
                title_parts.append(f"cz={p[2:]}")
            elif p.startswith("r") and p[1:].replace("mm","").isdigit():
                title_parts.append(f"r={p[1:]}")
            elif p.startswith("h") and p[1:].replace("mm","").isdigit():
                title_parts.append(f"h={p[1:]}")
            elif p.startswith("k") and p[1:].isdigit():
                title_parts.append(f"kappa={p[1:]} W/mK")
            elif p.startswith("Cp"):
                title_parts.append(f"Cp={p[2:]} J/kgK")
            elif p.startswith("rho"):
                title_parts.append(f"rho={p[3:]} kg/m3")
        nice_title = " | ".join(title_parts)

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
                    label=f"{region} (final: {T_mean[-1]:.0f}K)", alpha=0.9)
            ri += 1

        ax.axhline(T_set, color="red", ls="-", lw=2, alpha=0.5,
                   label=f"T_set = {T_set:.0f}K (target)")
        ax.set_xlabel("Time [s]", fontsize=12)
        ax.set_ylabel("Mean temperature [K]", fontsize=12)
        ax.set_title(nice_title, fontsize=11)
        ax.legend(fontsize=7, ncol=3, loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{args.outdir}/case_{ci:03d}_all_regions.png", dpi=150)
        plt.close(fig)
        print(f"  case_{ci:03d} done — {nice_title}")

    # ── Plot 3: Bar chart with case names ────────────────────────
    fig3, ax3 = plt.subplots(figsize=(20, 8))
    case_ids, t_set_vals, t_final_vals, case_labels = [], [], [], []
    for ci in range(n_cases):
        grp = f[f"case_{ci:03d}"]
        name = str(grp.attrs["name"])
        T_set = float(grp.attrs["T_set"])
        if "steel_cylinder" not in grp:
            continue
        T = grp["steel_cylinder"]["T"][:].astype(np.float32)
        case_ids.append(ci)
        t_set_vals.append(T_set)
        t_final_vals.append(T[-1].mean())
        case_labels.append(f"c{ci:03d}\nT={T_set:.0f}")

    x = np.arange(len(case_ids))
    gap = np.array(t_set_vals) - np.array(t_final_vals)

    ax3.bar(x, t_set_vals, width=0.4, label="T_set (target)", color="tomato", alpha=0.7)
    ax3.bar(x + 0.4, t_final_vals, width=0.4, label="Steel T at t=4000s", color="steelblue", alpha=0.7)

    # Add gap labels on top
    for i in range(len(case_ids)):
        ax3.text(x[i] + 0.2, max(t_set_vals[i], t_final_vals[i]) + 15,
                 f"-{gap[i]:.0f}K", fontsize=5, ha="center", color="darkred", rotation=90)

    ax3.set_xticks(x + 0.2)
    ax3.set_xticklabels(case_labels, rotation=0, fontsize=5)
    ax3.set_xlabel("Case", fontsize=12)
    ax3.set_ylabel("Temperature [K]", fontsize=12)
    ax3.set_title(f"T_set vs actual steel cylinder temperature at t=4000s — {len(case_ids)} cases\n"
                  f"Red numbers show the gap (how far below T_set)", fontsize=13)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis="y")
    fig3.tight_layout()
    fig3.savefig(f"{args.outdir}/tset_vs_steel_final.png", dpi=200)
    print(f"\nSaved: {args.outdir}/tset_vs_steel_final.png")
    plt.close(fig3)

print(f"\nAll plots in: {args.outdir}/")

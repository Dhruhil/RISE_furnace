"""
3D FNO Rollout Evaluation — MAE and R² only.
Focus: steel_cylinder + inner_box.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.fno_config import CONFIG
from data.dataset import FNO3DDataset, REGION_IDS, HEATER_REGIONS
from models.fno_model import HeatTreatmentFNO3D
from models.rollout import rollout_per_region


FOCUS_REGIONS = ["steel_cylinder", "inner_box"]


def run_fno3d_evaluation(model, cfg, device="cuda", n_sims=None):
    dataset = FNO3DDataset(cfg.dataset_path, cfg, "test", "evaluation")
    sim_indices = dataset.sim_indices
    if n_sims is not None:
        sim_indices = sim_indices[:n_sims]

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  3D FNO ROLLOUT EVALUATION")
    print(f"  {len(sim_indices)} test sims | grid {cfg.grid_x}x{cfg.grid_y}x{cfg.grid_z}")
    print(f"  In-horizon    : 200-{cfg.train_time_end:.0f}s (within training time range)")
    print(f"  Extrapolation : {cfg.train_time_end:.0f}-{cfg.predict_time_end:.0f}s (beyond training horizon)")
    print(f"{sep}")

    all_results = {}
    focus_p1 = {r: [] for r in FOCUS_REGIONS}
    focus_p2 = {r: [] for r in FOCUS_REGIONS}

    # ── In-horizon table ──────────────────────────────────────
    print(f"\n  IN-HORIZON (200-{cfg.train_time_end:.0f}s)")
    print(f"  {'Sim':>4}  {'Region':>16}  {'MAE':>9}  {'R²':>9}")
    print(f"  {'-'*48}")

    for sim_i in sim_indices:
        results = rollout_per_region(model, dataset, sim_i, device=device)
        all_results[f"sim_{sim_i}"] = results

        for region in FOCUS_REGIONS:
            if region not in results:
                continue
            r = results[region]
            print(f"  {sim_i:>4}  {region:>16}  {r['mae_p1']:>7.2f}K  {r['r2_p1']:>9.4f}")
            focus_p1[region].append(r)
            if not np.isnan(r['mae_p2']):
                focus_p2[region].append(r)
        print()

    # ── Extrapolation table ───────────────────────────────────
    print(f"\n  EXTRAPOLATION ({cfg.train_time_end:.0f}-{cfg.predict_time_end:.0f}s)")
    print(f"  {'Sim':>4}  {'Region':>16}  {'MAE':>9}  {'R²':>9}")
    print(f"  {'-'*48}")

    for sim_i in sim_indices:
        results = all_results[f"sim_{sim_i}"]
        for region in FOCUS_REGIONS:
            if region not in results:
                continue
            r = results[region]
            if np.isnan(r['mae_p2']):
                print(f"  {sim_i:>4}  {region:>16}  {'N/A':>9}  {'N/A':>9}")
            else:
                print(f"  {sim_i:>4}  {region:>16}  {r['mae_p2']:>7.2f}K  {r['r2_p2']:>9.4f}")
        print()

    # ── Summary across sims ───────────────────────────────────
    print(f"\n{sep}")
    print(f"  FOCUS REGIONS — MEAN ACROSS {len(sim_indices)} TEST SIMS")
    print(f"{sep}")

    summary = {}
    for region in FOCUS_REGIONS:
        if not focus_p1[region]:
            continue
        rs = focus_p1[region]
        print(f"\n  {region}:")
        print(f"    In-horizon (200-{cfg.train_time_end:.0f}s, within training time range):")
        print(f"      MAE = {np.mean([r['mae_p1'] for r in rs]):>7.2f} ± {np.std([r['mae_p1'] for r in rs]):.2f} K")
        print(f"      R²  = {np.mean([r['r2_p1']  for r in rs]):>7.4f}")

        summary[region] = {"in_horizon": {
            "mae_mean": float(np.mean([r['mae_p1'] for r in rs])),
            "mae_std":  float(np.std ([r['mae_p1'] for r in rs])),
            "r2_mean":  float(np.mean([r['r2_p1']  for r in rs])),
        }}

        if focus_p2[region]:
            rs2 = focus_p2[region]
            print(f"    Extrapolation ({cfg.train_time_end:.0f}-{cfg.predict_time_end:.0f}s, beyond training horizon):")
            print(f"      MAE = {np.mean([r['mae_p2'] for r in rs2]):>7.2f} ± {np.std([r['mae_p2'] for r in rs2]):.2f} K")
            print(f"      R²  = {np.mean([r['r2_p2']  for r in rs2]):>7.4f}")
            summary[region]["extrapolation"] = {
                "mae_mean": float(np.mean([r['mae_p2'] for r in rs2])),
                "mae_std":  float(np.std ([r['mae_p2'] for r in rs2])),
                "r2_mean":  float(np.mean([r['r2_p2']  for r in rs2])),
            }

    # ── Outlier check ─────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  OUTLIER CHECK — steel_cylinder (in-horizon)")
    print(f"{sep}")
    steel_maes = []
    for sim_i in sim_indices:
        r = all_results[f"sim_{sim_i}"].get("steel_cylinder", {})
        if r:
            steel_maes.append((sim_i, r["mae_p1"]))
    if steel_maes:
        median_in = np.median([m for _, m in steel_maes])
        for sim_i, mae in steel_maes:
            sim = dataset._simulations[sim_i]
            flag = "  ⚠ OUTLIER" if mae > 2 * median_in else ""
            print(f"  sim_{sim_i:<3}  T_set={sim['T_set']:>5.0f}K  MAE={mae:>6.2f}K{flag}")

    # ── Save JSON ─────────────────────────────────────────────
    out_path = f"{cfg.output_dir}/evaluation/fno_rollout_results.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"per_sim": all_results, "summary": summary}, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print(f"{sep}\n")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_sims", type=int, default=None)
    args = parser.parse_args()

    cfg = CONFIG
    ckpt_path = f"{cfg.checkpoint_dir}/best_model.pt"
    print(f"  Loading: {ckpt_path}")
    model = HeatTreatmentFNO3D.load(ckpt_path, cfg, args.device)
    run_fno3d_evaluation(model, cfg, device=args.device, n_sims=args.n_sims)


if __name__ == "__main__":
    main()

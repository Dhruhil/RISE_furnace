"""
DeepONet rollout evaluation — saves per-step trajectories + aggregate metrics.
Output structure mirrors FNO eval JSON for direct comparison plotting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")

from configs.deeponet_config import CONFIG
from data.dataset import get_deeponet_eval_dataset
from models.deeponet_model import HeatTreatmentDeepONet
from models.rollout import rollout_deeponet
from utils.checkpoint import load_best
from utils.metrics import compute_metrics


REGIONS = ["steel_cylinder", "inner_box", "outer_box",
           "heater_1", "heater_2", "heater_3", "heater_4",
           "heater_5", "heater_6", "heater_7", "heater_8",
           "brick_heater"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_sims", type=int, default=7)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--output_dir", default=None,
                        help="Output dir for JSON. Defaults to cfg.output_dir/evaluation/")
    args = parser.parse_args()

    cfg = CONFIG
    device = torch.device(args.device)
    ds = get_deeponet_eval_dataset(cfg)

    model = HeatTreatmentDeepONet(cfg).to(device)
    ckpt = args.ckpt or f"{cfg.checkpoint_dir}/best.pt"
    load_best(model, ckpt, device)

    sep = "=" * 70
    print(f"\n{sep}\n  DeepONet rollout — test set\n{sep}\n")

    n_train = int(cfg.train_time_end / cfg.dt)
    start_t = 20
    p1_steps = n_train - start_t  # number of steps in Phase 1

    # ─── Per-sim aggregate metrics + step trajectories ───
    per_sim = {}
    p1_mae_list, p1_r2_list = [], []
    p2_mae_list, p2_r2_list = [], []
    per_region_p1 = {r: [] for r in REGIONS}
    per_region_p2 = {r: [] for r in REGIONS}

    # ─── Phase 2 per-step trajectory (averaged across sims) ───
    p2_traj_steel_per_sim = []   # list of (abs_t, mae) tuples
    p2_traj_inner_per_sim = []

    sim_list = ds.sim_indices[:args.n_sims]
    for si, sim_i in enumerate(sim_list):
        sim = ds._simulations[sim_i]
        T_pred, T_true = rollout_deeponet(model, ds, sim_i, device=device, start_t=start_t)
        n_rollout = T_pred.shape[0]
        times = sim["times"]

        # Phase boundaries (in rollout step indices)
        p1_end = min(n_rollout, p1_steps)
        p2_end = n_rollout

        # Aggregate Phase 1 / Phase 2
        if p1_end > 0:
            m1 = compute_metrics(T_pred[:p1_end].reshape(-1), T_true[:p1_end].reshape(-1))
            p1_mae_list.append(m1["mae"]); p1_r2_list.append(m1["r2"])
        if p2_end > p1_end:
            m2 = compute_metrics(T_pred[p1_end:p2_end].reshape(-1), T_true[p1_end:p2_end].reshape(-1))
            p2_mae_list.append(m2["mae"]); p2_r2_list.append(m2["r2"])

        # Per-sim per-region per-step trajectory
        sim_key = f"sim_{sim_i}"
        per_sim[sim_key] = {"n_rollout": int(n_rollout),
                            "T_set": float(sim["T_set"])}

        for region, (a, b) in sim["region_slices"].items():
            # Per-step MAE over full rollout
            step_mae = np.mean(np.abs(T_pred[:, a:b] - T_true[:, a:b]), axis=1).tolist()

            # Phase 1 / Phase 2 aggregates
            r_pred1 = T_pred[:p1_end, a:b].reshape(-1)
            r_true1 = T_true[:p1_end, a:b].reshape(-1)
            mae_p1 = compute_metrics(r_pred1, r_true1)["mae"] if r_pred1.size else None

            r_pred2 = T_pred[p1_end:p2_end, a:b].reshape(-1)
            r_true2 = T_true[p1_end:p2_end, a:b].reshape(-1)
            mae_p2 = compute_metrics(r_pred2, r_true2)["mae"] if r_pred2.size else None

            per_sim[sim_key][region] = {
                "n_cells": int(b - a),
                "n_steps": int(n_rollout),
                "mae_p1": float(mae_p1) if mae_p1 is not None else None,
                "mae_p2": float(mae_p2) if mae_p2 is not None else None,
                "step_mae": [float(x) for x in step_mae],
            }
            if mae_p1 is not None:
                per_region_p1[region].append(mae_p1)
            if mae_p2 is not None:
                per_region_p2[region].append(mae_p2)

            # Phase 2 trajectory for steel and inner_box
            if p2_end > p1_end:
                p2_step_mae = np.mean(np.abs(T_pred[p1_end:p2_end, a:b] - T_true[p1_end:p2_end, a:b]), axis=1)
                # Time axis: step k in p2 → times[start_t + p1_end + k]
                p2_abs_t = times[start_t + p1_end : start_t + p2_end].astype(float)
                if region == "steel_cylinder":
                    p2_traj_steel_per_sim.append((p2_abs_t, p2_step_mae))
                elif region == "inner_box":
                    p2_traj_inner_per_sim.append((p2_abs_t, p2_step_mae))

        print(f"  sim {si+1}/{len(sim_list)} done  "
              f"(n_rollout={n_rollout}, T_set={sim['T_set']:.0f} K)")

    # ─── Aggregate Phase 2 per-step trajectory ───
    def aggregate_p2(traj_list):
        if not traj_list:
            return None
        # Find shortest trajectory length to align
        min_len = min(len(t[0]) for t in traj_list)
        # Use first sim's time axis (truncated)
        abs_t = traj_list[0][0][:min_len].tolist()
        # Stack and average MAE
        mae_stack = np.stack([t[1][:min_len] for t in traj_list])
        return {
            "abs_t": [float(x) for x in abs_t],
            "mae_mean": [float(x) for x in mae_stack.mean(axis=0)],
            "mae_std":  [float(x) for x in mae_stack.std(axis=0)],
        }

    phase2_per_step = {
        "steel_cylinder": aggregate_p2(p2_traj_steel_per_sim),
        "inner_box":      aggregate_p2(p2_traj_inner_per_sim),
    }

    # ─── Build full results dict (mirrors FNO structure) ───
    results = {
        "summary": {
            "phase1": {
                "mean_mae": float(np.mean(p1_mae_list)) if p1_mae_list else None,
                "mean_r2":  float(np.mean(p1_r2_list))  if p1_r2_list  else None,
            },
            "phase2": {
                "mean_mae": float(np.mean(p2_mae_list)) if p2_mae_list else None,
                "mean_r2":  float(np.mean(p2_r2_list))  if p2_r2_list  else None,
            },
            "per_region": {
                r: {
                    "p1": float(np.mean(per_region_p1[r])) if per_region_p1[r] else None,
                    "p2": float(np.mean(per_region_p2[r])) if per_region_p2[r] else None,
                }
                for r in REGIONS
            },
        },
        "per_sim": per_sim,
        "phase2_per_step": phase2_per_step,
    }

    # ─── Save JSON ───
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(cfg.output_dir) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "deeponet_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # ─── Print summary ───
    s = results["summary"]
    print(f"\n{sep}\n  DEEPONET SUMMARY — ALL REGIONS\n{sep}")
    if s["phase1"]["mean_mae"] is not None:
        print(f"  Phase 1 (training):     MAE={s['phase1']['mean_mae']:.2f} K  "
              f"R2={s['phase1']['mean_r2']:.4f}")
    if s["phase2"]["mean_mae"] is not None:
        print(f"  Phase 2 (verification): MAE={s['phase2']['mean_mae']:.2f} K  "
              f"R2={s['phase2']['mean_r2']:.4f}")
    print("\n  Per-region MAE:")
    for r, v in s["per_region"].items():
        s1 = f"{v['p1']:.2f} K" if v["p1"] is not None else "  N/A"
        s2 = f"{v['p2']:.2f} K" if v["p2"] is not None else "  N/A"
        print(f"    {r:>16}:  P1={s1}  P2={s2}")
    print(f"\n  Saved: {json_path}")


if __name__ == "__main__":
    main()

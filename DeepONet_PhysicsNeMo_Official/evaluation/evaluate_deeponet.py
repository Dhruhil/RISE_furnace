"""
DeepONet rollout evaluation.

Saves per-step error trajectories alongside the aggregate metrics so
the same JSON powers both the headline numbers in the thesis tables
and the rollout curves in Section 5.3. Output structure mirrors the
FNO eval JSON exactly — same keys, same shapes — which keeps the
plotting script architecture-agnostic and the comparison numbers in
the thesis fair across the three surrogates.
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


# All 12 regions get scored, but only steel_cylinder and inner_box
# end up in the thesis figures since the others are either
# clamped to T_set (heaters, brick) or quasi-static (outer_box).
REGIONS = ["steel_cylinder", "inner_box", "outer_box",
           "heater_1", "heater_2", "heater_3", "heater_4",
           "heater_5", "heater_6", "heater_7", "heater_8",
           "brick_heater"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n_sims", type=int, default=7,
                        help="Number of test sims to evaluate (default 7 = full test set)")
    parser.add_argument("--ckpt", default=None,
                        help="Path to best.pt (overrides cfg.checkpoint_dir)")
    parser.add_argument("--output_dir", default=None,
                        help="Output dir for JSON. Defaults to cfg.output_dir/evaluation/")
    args = parser.parse_args()

    cfg = CONFIG
    device = torch.device(args.device)
    ds = get_deeponet_eval_dataset(cfg)

    # ---- load the trained checkpoint -------------------------------
    model = HeatTreatmentDeepONet(cfg).to(device)
    # Default to the best-model checkpoint inside the config's
    # checkpoint dir; --ckpt lets the SLURM job point at any other.
    ckpt = args.ckpt or f"{cfg.checkpoint_dir}/best.pt"
    load_best(model, ckpt, device)

    sep = "=" * 70
    print(f"\n{sep}\n  DeepONet rollout — test set\n{sep}\n")

    # ---- phase boundaries (in rollout-step indices) ----------------
    # cfg.train_time_end / cfg.dt gives the Phase-1 boundary in
    # absolute steps; subtracting start_t converts it to a rollout-
    # step index since the rollout starts at t=200s (= step 20).
    n_train = int(cfg.train_time_end / cfg.dt)
    start_t = 20
    p1_steps = n_train - start_t

    # ---- accumulators for the aggregate metrics --------------------
    per_sim = {}
    p1_mae_list, p1_r2_list = [], []
    p2_mae_list, p2_r2_list = [], []
    per_region_p1 = {r: [] for r in REGIONS}
    per_region_p2 = {r: [] for r in REGIONS}

    # Per-step trajectories for the two focus regions — collected
    # across all sims, then averaged into a single (mean, std)
    # curve at the end. These feed the rollout figures.
    p2_traj_steel_per_sim = []
    p2_traj_inner_per_sim = []

    # ---- main loop over test sims ----------------------------------
    sim_list = ds.sim_indices[:args.n_sims]
    for si, sim_i in enumerate(sim_list):
        sim = ds._simulations[sim_i]
        T_pred, T_true = rollout_deeponet(model, ds, sim_i, device=device, start_t=start_t)
        n_rollout = T_pred.shape[0]
        times = sim["times"]

        # Phase 1 = in-distribution (training-window) portion;
        # Phase 2 = temporal-extrapolation portion past train_time_end.
        # Some short sims don't reach Phase 2; the >0 guards below
        # keep that case from blowing up.
        p1_end = min(n_rollout, p1_steps)
        p2_end = n_rollout

        # ---- aggregate Phase 1 / Phase 2 metrics -------------------
        if p1_end > 0:
            m1 = compute_metrics(T_pred[:p1_end].reshape(-1), T_true[:p1_end].reshape(-1))
            p1_mae_list.append(m1["mae"]); p1_r2_list.append(m1["r2"])
        if p2_end > p1_end:
            m2 = compute_metrics(T_pred[p1_end:p2_end].reshape(-1), T_true[p1_end:p2_end].reshape(-1))
            p2_mae_list.append(m2["mae"]); p2_r2_list.append(m2["r2"])

        # ---- per-region per-step trajectory ------------------------
        sim_key = f"sim_{sim_i}"
        per_sim[sim_key] = {"n_rollout": int(n_rollout),
                            "T_set": float(sim["T_set"])}

        for region, (a, b) in sim["region_slices"].items():
            # Per-step MAE across the full rollout for this region —
            # what the rollout figures plot directly.
            step_mae = np.mean(np.abs(T_pred[:, a:b] - T_true[:, a:b]), axis=1).tolist()

            # Phase 1 / Phase 2 aggregates for this region. None when
            # there's no data in that phase (covers the short-sim case).
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

            # Phase 2 trajectory captured separately for the two focus
            # regions so the cross-sim averaging below has a clean
            # alignment by absolute time.
            if p2_end > p1_end:
                p2_step_mae = np.mean(np.abs(T_pred[p1_end:p2_end, a:b] - T_true[p1_end:p2_end, a:b]), axis=1)
                # Map rollout step k in Phase 2 back to absolute time:
                # times[start_t + p1_end + k]
                p2_abs_t = times[start_t + p1_end : start_t + p2_end].astype(float)
                if region == "steel_cylinder":
                    p2_traj_steel_per_sim.append((p2_abs_t, p2_step_mae))
                elif region == "inner_box":
                    p2_traj_inner_per_sim.append((p2_abs_t, p2_step_mae))

        print(f"  sim {si+1}/{len(sim_list)} done  "
              f"(n_rollout={n_rollout}, T_set={sim['T_set']:.0f} K)")

    # ---- aggregate Phase-2 per-step trajectory --------------------
    def aggregate_p2(traj_list):
        """
        Average a list of (abs_t, mae) trajectories into one
        (mean, std) curve. Sims sometimes finish a step short of
        each other, so the trajectories get clipped to the shortest
        common length before stacking.
        """
        if not traj_list:
            return None
        min_len = min(len(t[0]) for t in traj_list)
        # Use the first sim's time axis (truncated) — all sims share
        # the same OpenFOAM dt so the time grids line up.
        abs_t = traj_list[0][0][:min_len].tolist()
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

    # ---- build the output dict (mirrors the FNO eval structure) ----
    # Three top-level blocks: summary (table-ready aggregates),
    # per_sim (every metric for every case), and phase2_per_step
    # (the trajectory data the plotting script picks up).
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

    # ---- write JSON ------------------------------------------------
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = Path(cfg.output_dir) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "deeponet_evaluation.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # ---- print summary --------------------------------------------
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
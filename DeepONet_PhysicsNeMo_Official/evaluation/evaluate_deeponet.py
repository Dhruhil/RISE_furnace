"""
Rollout evaluation — same split as FNO (phase 1: training window, phase 2:
verification window t > train_time_end).
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
    parser.add_argument("--n_sims", type=int, default=5)
    parser.add_argument("--ckpt",   default=None)
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

    p1_mae, p1_r2 = [], []
    p2_mae, p2_r2 = [], []
    per_region_mae = {r: {"p1": [], "p2": []} for r in REGIONS}

    sim_list = ds.sim_indices[:args.n_sims]
    for si, sim_i in enumerate(sim_list):
        sim = ds._simulations[sim_i]
        T_pred, T_true = rollout_deeponet(model, ds, sim_i, device=device, start_t=20)

        n_rollout = T_pred.shape[0]
        # Phase 1: within training window
        p1_end = min(n_rollout, n_train - 20)
        p2_end = n_rollout

        p1_pred = T_pred[:p1_end].reshape(-1)
        p1_true = T_true[:p1_end].reshape(-1)
        if p1_pred.size:
            m1 = compute_metrics(p1_pred, p1_true)
            p1_mae.append(m1["mae"]); p1_r2.append(m1["r2"])

        if p2_end > p1_end:
            p2_pred = T_pred[p1_end:p2_end].reshape(-1)
            p2_true = T_true[p1_end:p2_end].reshape(-1)
            if p2_pred.size:
                m2 = compute_metrics(p2_pred, p2_true)
                p2_mae.append(m2["mae"]); p2_r2.append(m2["r2"])

        # Per-region
        for region, (a, b) in sim["region_slices"].items():
            r_pred1 = T_pred[:p1_end, a:b].reshape(-1)
            r_true1 = T_true[:p1_end, a:b].reshape(-1)
            if r_pred1.size:
                per_region_mae[region]["p1"].append(
                    compute_metrics(r_pred1, r_true1)["mae"])
            if p2_end > p1_end:
                r_pred2 = T_pred[p1_end:p2_end, a:b].reshape(-1)
                r_true2 = T_true[p1_end:p2_end, a:b].reshape(-1)
                if r_pred2.size:
                    per_region_mae[region]["p2"].append(
                        compute_metrics(r_pred2, r_true2)["mae"])

        print(f"  sim {si+1}/{len(sim_list)} done  "
              f"(n_rollout={n_rollout}, T_set={sim['T_set']:.0f} K)")

    summary = {
        "phase1": {"mean_mae": float(np.mean(p1_mae)) if p1_mae else None,
                   "mean_r2":  float(np.mean(p1_r2))  if p1_r2  else None},
        "phase2": {"mean_mae": float(np.mean(p2_mae)) if p2_mae else None,
                   "mean_r2":  float(np.mean(p2_r2))  if p2_r2  else None},
        "per_region": {
            r: {"p1": float(np.mean(v["p1"])) if v["p1"] else None,
                "p2": float(np.mean(v["p2"])) if v["p2"] else None}
            for r, v in per_region_mae.items()
        },
    }

    out = Path(cfg.output_dir) / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "deeponet_evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{sep}\n  DEEPONET SUMMARY — ALL REGIONS\n{sep}")
    p1 = summary["phase1"]; p2 = summary["phase2"]
    if p1["mean_mae"] is not None:
        print(f"  Phase 1 (training):     MAE={p1['mean_mae']:.2f} K  "
              f"R2={p1['mean_r2']:.4f}")
    if p2["mean_mae"] is not None:
        print(f"  Phase 2 (verification): MAE={p2['mean_mae']:.2f} K  "
              f"R2={p2['mean_r2']:.4f}")
    print("\n  Per-region MAE:")
    for r, v in summary["per_region"].items():
        s1 = f"{v['p1']:.2f} K" if v["p1"] is not None else "  N/A"
        s2 = f"{v['p2']:.2f} K" if v["p2"] is not None else "  N/A"
        print(f"    {r:>16}:  P1={s1}  P2={s2}")


if __name__ == "__main__":
    main()

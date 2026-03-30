"""
GNN Rollout Evaluation — per-region MAE for T_next prediction.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.base_config import CONFIG
from data.dataset_unified import UnifiedDataset, REGION_IDS, HEATER_REGIONS
from models.meshgraphnet import HeatTreatmentGNN
from torch_geometric.data import Data


def rollout_gnn(model, dataset, sim_i, device, start_t=20):
    """Autoregressive rollout on unified graph."""
    sim = dataset._simulations[sim_i]
    edge_index, edge_attr = dataset._graphs[sim_i]
    total = sim["total_nodes"]
    T_set = sim["T_set"]
    times = sim["times"]

    n_times = len(times)
    T_all = np.zeros((n_times, total), dtype=np.float32)
    for region, rdata in sim["region_data"].items():
        o = rdata["offset"]
        n = rdata["n_cells"]
        for t in range(n_times):
            T_all[t, o:o+n] = rdata["T_array"][t]

    T_cur = T_all[start_t].copy()
    n_rollout = n_times - start_t
    T_pred_all = np.zeros((n_rollout, total), dtype=np.float32)
    T_pred_all[0] = T_cur

    all_coords = sim["all_coords"]
    all_rids = sim["all_region_ids"]
    heater_rids = {REGION_IDS[r] for r in HEATER_REGIONS if r in REGION_IDS}
    is_heater = np.array([1.0 if int(all_rids[j]) in heater_rids else 0.0
                          for j in range(total)], dtype=np.float32)

    T_mean = dataset.T_mean
    T_std = dataset.T_std

    model.eval()
    with torch.no_grad():
        for step in range(1, n_rollout):
            t_idx = start_t + step
            if t_idx >= n_times:
                break
            t_val = times[t_idx - 1]

            T_norm = (T_cur - T_mean) / T_std
            Tset_norm = (T_set - T_mean) / T_std
            t_norm = t_val / 4000.0

            node_feats = np.column_stack([
                all_coords[:, 0], all_coords[:, 1], all_coords[:, 2],
                T_norm,
                np.full(total, Tset_norm, dtype=np.float32),
                all_rids / 11.0,
                np.full(total, t_norm, dtype=np.float32),
                is_heater,
                np.full(total, sim.get("cx", 0.0) / 0.206, dtype=np.float32),
                np.full(total, sim.get("cy", 0.18) / 0.36, dtype=np.float32),
                np.full(total, sim.get("cz", 0.195) / 0.39, dtype=np.float32),
                np.full(total, sim.get("radius", 0.05) / 0.10, dtype=np.float32),
                np.full(total, sim.get("height", 0.10) / 0.20, dtype=np.float32),
                np.full(total, sim.get("kappa", 60.0) / 100.0, dtype=np.float32),
                np.full(total, sim.get("Cp", 450.0) / 1000.0, dtype=np.float32),
                np.full(total, sim.get("rho", 7800.0) / 10000.0, dtype=np.float32),
            ]).astype(np.float32)

            batch = Data(
                x=torch.tensor(node_feats, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_attr,
            ).to(device)

            pred = model(batch)
            T_next = pred.squeeze(-1).cpu().numpy() * T_std + T_mean

            heater_mask = is_heater > 0.5
            if t_idx < n_times:
                T_next[heater_mask] = T_all[t_idx][heater_mask]

            T_pred_all[step] = T_next
            T_cur = T_next.copy()

    return T_pred_all, T_all[start_t:start_t + n_rollout]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_sims", type=int, default=5)
    args = parser.parse_args()

    cfg = CONFIG
    cfg.node_in_features = 16
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*80}")
    print(f"  GNN ROLLOUT EVALUATION — Per-Region Accuracy")
    print(f"  Phase 1: 0-3200s | Phase 2: 3200-4000s")
    print(f"{'='*80}\n")

    test_ds = UnifiedDataset(cfg.all_regions_dataset_path, cfg, "test", "evaluation")

    ckpt_path = "outputs/checkpoints_unified/best_model.pt"
    print(f"  Loading model from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = HeatTreatmentGNN(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Model loaded (epoch {ckpt.get('epoch', '?')})\n")

    start_t = 20
    n_train_steps = 320
    all_results = {}
    region_names = {v: k for k, v in REGION_IDS.items()}

    n_eval = min(args.n_sims, len(test_ds.sim_indices))

    for i, sim_i in enumerate(test_ds.sim_indices[:n_eval]):
        sim = test_ds._simulations[sim_i]
        all_rids = sim["all_region_ids"]
        print(f"  Sim {sim_i} (T_set={sim['T_set']:.0f}K)")
        print(f"  {'Region':>18} | {'Cells':>6} | {'P1 MAE':>8} | {'P2 MAE':>8}")
        print(f"  {'-'*50}")

        t0 = time.time()
        T_pred, T_true = rollout_gnn(model, test_ds, sim_i, device, start_t)
        rt = time.time() - t0

        p1_end = min(n_train_steps - start_t, T_pred.shape[0])
        metrics = {}

        for rid in range(12):
            rname = region_names.get(rid, f"r{rid}")
            mask = (all_rids == rid)
            if mask.sum() == 0:
                continue
            is_h = rname in HEATER_REGIONS
            p1 = float(np.mean(np.abs(T_pred[1:p1_end, mask] - T_true[1:p1_end, mask]))) if p1_end > 1 else 0
            p2_data = T_pred[p1_end:, mask] - T_true[p1_end:, mask]
            p2 = float(np.mean(np.abs(p2_data))) if p2_data.size > 0 else float('nan')
            metrics[rname] = {"n_cells": int(mask.sum()), "p1_mae": p1, "p2_mae": p2, "is_heater": is_h}

            p2s = f"{p2:.2f}K" if not np.isnan(p2) else "N/A"
            tag = " (BC)" if is_h else ""
            print(f"  {rname:>18} | {mask.sum():>6} | {p1:>7.2f}K | {p2s:>8}{tag}")

        nh = {k: v for k, v in metrics.items() if not v['is_heater']}
        if nh:
            p1a = np.mean([v['p1_mae'] for v in nh.values()])
            p2v = [v['p2_mae'] for v in nh.values() if not np.isnan(v['p2_mae'])]
            p2a = np.mean(p2v) if p2v else float('nan')
            print(f"  {'NON-HEATER AVG':>18} |        | {p1a:>7.2f}K | {p2a:.2f}K")
        print(f"  Rollout: {rt:.1f}s ({T_pred.shape[0]} steps)\n")
        all_results[f"sim_{sim_i}"] = metrics

    print(f"{'='*80}")
    print(f"  SUMMARY — GNN ROLLOUT")
    print(f"{'='*80}")
    for rname in ["steel_cylinder", "inner_box", "outer_box"]:
        p1v, p2v = [], []
        for sk, m in all_results.items():
            if rname in m:
                p1v.append(m[rname]["p1_mae"])
                if not np.isnan(m[rname]["p2_mae"]):
                    p2v.append(m[rname]["p2_mae"])
        if p1v:
            p2s = f"{np.mean(p2v):.2f}K" if p2v else "N/A"
            print(f"  {rname:>18}: P1={np.mean(p1v):.2f}K  P2={p2s}")

    out_path = "outputs/evaluation/gnn_rollout_results.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()

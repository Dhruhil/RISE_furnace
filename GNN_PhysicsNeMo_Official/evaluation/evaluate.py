"""
GNN rollout evaluation — per-region MAE for next-step temperature
predictions across the held-out test set.

Patched to take --checkpoint and --output_dir flags so the same
script works for any of the trained checkpoints.
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import torch

# Make the project importable when running this from the eval/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.base_config import CONFIG
from data.dataset_unified import UnifiedDataset, REGION_IDS, HEATER_REGIONS
from configs.base_config import REGION_MATERIALS
from models.meshgraphnet import HeatTreatmentGNN
from torch_geometric.data import Data


def rollout_gnn(model, dataset, sim_i, device, start_t=20):
    """
    Run an autoregressive rollout on the unified graph for one sim.

    The model predicts T_next from T_current, then that prediction
    gets fed back in as the next T_current. Heater cells are clamped
    to their ground-truth values at every step (Dirichlet BC).
    """
    sim = dataset._simulations[sim_i]
    edge_index, edge_attr = dataset._graphs[sim_i]
    total = sim["total_nodes"]
    T_set = sim["T_set"]
    times = sim["times"]

    # ---- ground-truth field stacked across time --------------------
    n_times = len(times)
    T_all = np.zeros((n_times, total), dtype=np.float32)
    for region, rdata in sim["region_data"].items():
        o = rdata["offset"]
        n = rdata["n_cells"]
        for t in range(n_times):
            T_all[t, o:o+n] = rdata["T_array"][t]

    # Start the rollout from the OpenFOAM ground truth at start_t
    T_cur = T_all[start_t].copy()
    n_rollout = n_times - start_t
    T_pred_all = np.zeros((n_rollout, total), dtype=np.float32)
    T_pred_all[0] = T_cur

    all_coords = sim["all_coords"]
    all_rids = sim["all_region_ids"]

    # Pre-compute the heater mask once — gets reused at every step
    # to clamp the heater cells back to their ground truth.
    heater_rids = {REGION_IDS[r] for r in HEATER_REGIONS if r in REGION_IDS}
    is_heater = np.array([1.0 if int(all_rids[j]) in heater_rids else 0.0
                          for j in range(total)], dtype=np.float32)

    # Per-region material features — must match the rescaling used
    # in dataset.__getitem__ exactly, otherwise the model sees a
    # different feature distribution at inference time.
    _kappa_feat = np.zeros(total, dtype=np.float32)
    _Cp_feat    = np.zeros(total, dtype=np.float32)
    _rho_feat   = np.zeros(total, dtype=np.float32)
    for _rname, _rdata in sim["region_data"].items():
        _o = _rdata["offset"]
        _n = _rdata["n_cells"]
        _mat = REGION_MATERIALS.get(_rname, {"kappa": 80.0, "Cp": 450.0, "rho": 7800.0})
        _kappa_feat[_o:_o + _n] = _mat["kappa"] / 100.0
        _Cp_feat   [_o:_o + _n] = _mat["Cp"]    / 1000.0
        _rho_feat  [_o:_o + _n] = _mat["rho"]   / 10000.0

    T_mean = dataset.T_mean
    T_std = dataset.T_std

    # ---- main rollout loop -----------------------------------------
    model.eval()
    with torch.no_grad():
        for step in range(1, n_rollout):
            t_idx = start_t + step
            if t_idx >= n_times:
                break
            # Feed time stamp from the previous step (model predicts
            # the *next* state given the current one)
            t_val = times[t_idx - 1]

            T_norm = (T_cur - T_mean) / T_std
            Tset_norm = (T_set - T_mean) / T_std
            t_norm = t_val / 4000.0

            # Same 16-feature stack as in dataset.__getitem__ —
            # any drift between training and inference features will
            # quietly tank the rollout, so keep these in sync.
            node_feats = np.column_stack([
                all_coords[:, 0], all_coords[:, 1], all_coords[:, 2],
                T_norm,
                np.full(total, Tset_norm, dtype=np.float32),
                all_rids / 11.0,
                np.full(total, t_norm, dtype=np.float32),
                is_heater,
                np.full(total, sim["cx"] / 0.206, dtype=np.float32),
                np.full(total, sim["cy"] / 0.36, dtype=np.float32),
                np.full(total, sim["cz"] / 0.39, dtype=np.float32),
                np.full(total, sim["radius"] / 0.10, dtype=np.float32),
                np.full(total, sim["height"] / 0.20, dtype=np.float32),
                _kappa_feat,
                _Cp_feat,
                _rho_feat,
            ]).astype(np.float32)

            batch = Data(
                x=torch.tensor(node_feats, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_attr,
            ).to(device)

            pred = model(batch)
            # Denormalise back to physical Kelvin for the next step
            T_next = pred.squeeze(-1).cpu().numpy() * T_std + T_mean

            # Clamp heater cells to the ground truth (Dirichlet BC).
            # This mirrors how OpenFOAM enforces T = T_set on the
            # heaters and the brick heater at every solver step.
            heater_mask = is_heater > 0.5
            if t_idx < n_times:
                T_next[heater_mask] = T_all[t_idx][heater_mask]

            T_pred_all[step] = T_next
            T_cur = T_next.copy()

    return T_pred_all, T_all[start_t:start_t + n_rollout]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--n_sims",     type=int, default=None)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint .pt file (e.g., outputs/.../checkpoints/best_model.pt)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save evaluation JSON")
    parser.add_argument("--start_t", type=int, default=20,
                        help="Starting timestep for rollout (default 20)")
    parser.add_argument("--n_train_steps", type=int, default=276,
                        help="Number of in-horizon timesteps (Phase 1 boundary)")
    args = parser.parse_args()

    cfg = CONFIG
    cfg.node_in_features = 16   # 16-dim node features (see dataset)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*80}")
    print(f"  GNN ROLLOUT EVALUATION — Per-Region Accuracy")
    print(f"  Phase 1: 0-2760s | Phase 2: 2760-3460s")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*80}\n")

    # Load the test split — same stratified split that was used
    # at training time, so the held-out cases match exactly.
    test_ds = UnifiedDataset(cfg.all_regions_dataset_path, cfg, "test", "evaluation")

    # ---- load the trained checkpoint -------------------------------
    print(f"  Loading model from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = HeatTreatmentGNN(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Model loaded (epoch {ckpt.get('epoch', '?')})\n")

    all_results = {}
    region_names = {v: k for k, v in REGION_IDS.items()}

    # Run rollouts for either the full test set or just the first
    # --n_sims cases (handy for quick sanity checks during dev)
    n_eval = len(test_ds.sim_indices) if args.n_sims is None else min(args.n_sims, len(test_ds.sim_indices))

    for i, sim_i in enumerate(test_ds.sim_indices[:n_eval]):
        sim = test_ds._simulations[sim_i]
        all_rids = sim["all_region_ids"]
        print(f"  Sim {sim_i} (T_set={sim['T_set']:.0f}K)")
        print(f"  {'Region':>18} | {'Cells':>6} | {'P1 MAE':>8} | {'P2 MAE':>8}")
        print(f"  {'-'*50}")

        t0 = time.time()
        T_pred, T_true = rollout_gnn(model, test_ds, sim_i, device, args.start_t)
        rt = time.time() - t0

        # Phase 1 = in-distribution, Phase 2 = temporal extrapolation
        p1_end = min(args.n_train_steps - args.start_t, T_pred.shape[0])
        metrics = {}

        # ---- per-region MAE breakdown ------------------------------
        for rid in range(12):
            rname = region_names.get(rid, f"r{rid}")
            mask = (all_rids == rid)
            if mask.sum() == 0:
                continue
            is_h = rname in HEATER_REGIONS

            # Skip step 0 (that's the seeded ground-truth field)
            p1 = float(np.mean(np.abs(T_pred[1:p1_end, mask] - T_true[1:p1_end, mask]))) if p1_end > 1 else 0
            p2_data = T_pred[p1_end:, mask] - T_true[p1_end:, mask]
            p2 = float(np.mean(np.abs(p2_data))) if p2_data.size > 0 else float('nan')
            metrics[rname] = {"n_cells": int(mask.sum()), "p1_mae": p1, "p2_mae": p2, "is_heater": is_h}

            p2s = f"{p2:.2f}K" if not np.isnan(p2) else "N/A"
            # (BC) marker = heater region clamped via Dirichlet BC,
            # so its MAE will always be ~0 by construction
            tag = " (BC)" if is_h else ""
            print(f"  {rname:>18} | {mask.sum():>6} | {p1:>7.2f}K | {p2s:>8}{tag}")

        # Average across non-heater regions only — the heaters are
        # clamped, so including them would just push the average
        # artificially close to zero.
        nh = {k: v for k, v in metrics.items() if not v['is_heater']}
        if nh:
            p1a = np.mean([v['p1_mae'] for v in nh.values()])
            p2v = [v['p2_mae'] for v in nh.values() if not np.isnan(v['p2_mae'])]
            p2a = np.mean(p2v) if p2v else float('nan')
            print(f"  {'NON-HEATER AVG':>18} |        | {p1a:>7.2f}K | {p2a:.2f}K")
        print(f"  Rollout: {rt:.1f}s ({T_pred.shape[0]} steps)\n")

        all_results[f"sim_{sim_i}"] = {
            "T_set": float(sim['T_set']),
            "n_steps": int(T_pred.shape[0]),
            "rollout_time_s": float(rt),
            "regions": metrics,
        }

    # ---- summary table across all evaluated sims --------------------
    print(f"{'='*80}")
    print(f"  SUMMARY — GNN ROLLOUT")
    print(f"{'='*80}")
    summary = {}
    # Only the three predicted regions matter — the rest are clamped
    for rname in ["steel_cylinder", "inner_box", "outer_box"]:
        p1v, p2v = [], []
        for sk, sim_data in all_results.items():
            if rname in sim_data["regions"]:
                p1v.append(sim_data["regions"][rname]["p1_mae"])
                if not np.isnan(sim_data["regions"][rname]["p2_mae"]):
                    p2v.append(sim_data["regions"][rname]["p2_mae"])
        if p1v:
            p2_mean = float(np.mean(p2v)) if p2v else float('nan')
            p1_mean = float(np.mean(p1v))
            p1_std  = float(np.std(p1v))
            p2_std  = float(np.std(p2v)) if p2v else float('nan')
            summary[rname] = {
                "p1_mae_mean": p1_mean,
                "p1_mae_std":  p1_std,
                "p2_mae_mean": p2_mean,
                "p2_mae_std":  p2_std,
                "n_sims": len(p1v),
            }
            p2s = f"{p2_mean:.2f}±{p2_std:.2f}K" if p2v else "N/A"
            print(f"  {rname:>18}: P1={p1_mean:.2f}±{p1_std:.2f}K  P2={p2s}")

    # ---- dump everything to a single JSON --------------------------
    # The summary block is what gets reported in the thesis tables;
    # the per_sim block is kept around for plotting and audits.
    output_data = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get('epoch', None),
        "phase1_start_s": 0,
        "phase1_end_s": 2760,
        "phase2_start_s": 2760,
        "phase2_end_s": 3460,
        "start_t": args.start_t,
        "n_train_steps": args.n_train_steps,
        "n_test_sims": n_eval,
        "summary": summary,
        "per_sim": all_results,
    }

    out_path = output_dir / "gnn_rollout_results.json"
    with open(out_path, "w") as f:
        json.dump(output_data, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
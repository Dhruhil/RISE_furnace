"""
Unified GNN Rollout Evaluation — Phase 1 (0-3200s) + Phase 2 (3200-4000s).
Autoregressive rollout on unified multi-region graph.
"""
from __future__ import annotations
import sys, json, argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, "/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official")

from configs.base_config import CONFIG
from models.meshgraphnet import HeatTreatmentGNN

import importlib.util
spec = importlib.util.spec_from_file_location(
    "dataset_unified",
    "/mimer/NOBACKUP/groups/revar/GNN_Unified/data/dataset_unified.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
UnifiedDataset = mod.UnifiedDataset
HEATER_REGIONS = mod.HEATER_REGIONS
REGION_IDS = mod.REGION_IDS


def rollout_unified(model, dataset, sim_i, device):
    """Full autoregressive rollout on unified graph for one simulation."""
    model.eval()
    sim = dataset._simulations[sim_i]
    edge_index, edge_attr = dataset._graphs[sim_i]
    times = sim["times"]
    T_set = sim["T_set"]
    total_nodes = sim["total_nodes"]
    all_coords = sim["all_coords"]
    all_rids = sim["all_region_ids"]
    # Compute is_heater from region data
    all_is_heater = np.zeros(total_nodes, dtype=np.float32)
    for region, rdata in sim["region_data"].items():
        if region in HEATER_REGIONS:
            o = rdata["offset"]
            n = rdata["n_cells"]
            all_is_heater[o:o+n] = 1.0

    dT_std = dataset.dT_std
    dT_mean = dataset.dT_mean
    T_mean = dataset.T_mean
    T_std = dataset.T_std

    t_start = 20
    n_t = sim["n_times"]

    # Initial temperature
    T_current = np.zeros(total_nodes, dtype=np.float32)
    for region, rdata in sim["region_data"].items():
        o = rdata["offset"]
        n = rdata["n_cells"]
        T_current[o:o+n] = rdata["T_array"][t_start]

    results = {"phase1": {}, "phase2": {}}

    with torch.no_grad():
        for t_i in range(t_start, n_t - 1):
            t_val = times[t_i]

            # Build node features
            T_norm = (T_current - T_mean) / (T_std + 1e-8)
            Tset_norm = (T_set - T_mean) / (T_std + 1e-8)
            t_norm = t_val / 4000.0

            node_feats = np.column_stack([
                all_coords[:, 0],
                all_coords[:, 1],
                all_coords[:, 2],
                T_norm,
                np.full(total_nodes, Tset_norm, dtype=np.float32),
                all_rids / 11.0,
                np.full(total_nodes, t_norm, dtype=np.float32),
            ]).astype(np.float32)

            # Create batch
            from torch_geometric.data import Data
            data = Data(
                x=torch.tensor(node_feats, dtype=torch.float32),
                edge_index=edge_index,
                edge_attr=edge_attr.float(),
            ).to(device)

            # Predict
            pred = model(data)
            dT_pred = pred.squeeze(-1).cpu().numpy() * dT_std + dT_mean

            # Update temperature
            T_pred = T_current + dT_pred

            # Clamp heaters to T_set
            T_pred = np.where(all_is_heater > 0.5, T_set, T_pred)

            # Ground truth
            T_true = np.zeros(total_nodes, dtype=np.float32)
            for region, rdata in sim["region_data"].items():
                o = rdata["offset"]
                n = rdata["n_cells"]
                T_true[o:o+n] = rdata["T_array"][t_i + 1]

            # Per-region errors
            phase = "phase1" if t_val <= 3200 else "phase2"
            for region, rdata in sim["region_data"].items():
                o = rdata["offset"]
                n = rdata["n_cells"]
                mae = np.abs(T_pred[o:o+n] - T_true[o:o+n]).mean()
                if region not in results[phase]:
                    results[phase][region] = []
                results[phase][region].append(float(mae))

            # Use prediction for next step
            T_current = T_pred

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    cfg = CONFIG
    cfg.node_in_features = 7
    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt_path = args.checkpoint or "/mimer/NOBACKUP/groups/revar/GNN_Unified/outputs/checkpoints/best_model.pt"
    print(f"\n  Loading unified GNN checkpoint: {ckpt_path}")

    model = HeatTreatmentGNN(cfg).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    epoch = ckpt.get("epoch", 0)
    mae = ckpt.get("metrics", {}).get("mae", 0)
    print(f"  Loaded epoch {epoch}, val_MAE={mae:.3f}K")

    # Load test dataset
    test_ds = UnifiedDataset(
        "/mimer/NOBACKUP/groups/revar/GNN_Unified/dataset_all_regions.h5",
        cfg, "test", "evaluation")

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  UNIFIED GNN ROLLOUT EVALUATION")
    print(f"{sep}")

    all_p1, all_p2 = {}, {}
    for sim_i in test_ds.sim_indices:
        results = rollout_unified(model, test_ds, sim_i, device)

        p1_mae = np.mean([np.mean(v) for v in results["phase1"].values()]) if results["phase1"] else 0
        p2_mae = np.mean([np.mean(v) for v in results["phase2"].values()]) if results["phase2"] else 0
        print(f"    Sim {sim_i:>3}  P1={p1_mae:.2f}K  P2={p2_mae:.2f}K")

        for phase_name, phase_data in [("phase1", results["phase1"]), ("phase2", results["phase2"])]:
            target = all_p1 if phase_name == "phase1" else all_p2
            for region, maes in phase_data.items():
                if region not in target:
                    target[region] = []
                target[region].extend(maes)

    # Summary
    print(f"\n{sep}")
    print(f"  SUMMARY — Unified GNN Rollout")
    print(f"{sep}")

    overall_p1, overall_p2 = [], []
    for region in sorted(set(list(all_p1.keys()) + list(all_p2.keys()))):
        p1 = np.mean(all_p1.get(region, [0]))
        p2 = np.mean(all_p2.get(region, [0]))
        if region in HEATER_REGIONS:
            print(f"    {region:20s}: P1={p1:.2f}K  P2=N/A (clamped)")
        else:
            print(f"    {region:20s}: P1={p1:.2f}K  P2={p2:.2f}K")
            overall_p1.extend(all_p1.get(region, []))
            overall_p2.extend(all_p2.get(region, []))

    p1_mae = np.mean(overall_p1) if overall_p1 else 0
    p2_mae = np.mean(overall_p2) if overall_p2 else 0
    print(f"\n  Phase 1 (0-3200s):     MAE={p1_mae:.2f}K")
    print(f"  Phase 2 (3200-4000s):  MAE={p2_mae:.2f}K")
    print(f"{sep}")


if __name__ == "__main__":
    main()

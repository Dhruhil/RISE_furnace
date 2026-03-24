"""Evaluate WITHOUT heater clamping — GNN predicts all 12 regions."""
import sys, torch, numpy as np
sys.path.insert(0, '.')
from configs.base_config import CONFIG
from data.dataset_all_regions import AllRegionsDataset
from models.meshgraphnet import HeatTreatmentGNN
from utils.metrics import compute_metrics
from torch_geometric.data import Data, Batch

cfg = CONFIG
cfg.node_in_features = 7
device = 'cuda'
ckpt = f'{cfg.checkpoint_dir}_allregions/best_model.pt'
model = HeatTreatmentGNN.load(ckpt, cfg, device)
model.eval()
dataset = AllRegionsDataset(cfg.all_regions_dataset_path, cfg, 'test', 'evaluation')

n_train = cfg.n_train_steps
start_t = 40
p1_end = n_train - start_t

region_data = {}
for sim_i in dataset.sim_indices:
    sim = dataset._simulations[sim_i]
    n_times = sim["n_times"]
    T_set = sim["T_set"]
    times = sim["times"]
    n_steps = n_times - start_t - 1

    for region, rdata in sim["region_data"].items():
        n_cells = len(rdata["T_array"][start_t])
        region_id = rdata["region_id"]
        coords = rdata["coords"]
        T_max_r = sim["region_T_max"][region]
        T_min_r = sim["region_T_min"][region]

        edge_index, edge_attr = dataset._graphs[sim_i][region]
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.float().to(device)

        T_current = rdata["T_array"][start_t].copy().astype(np.float64)
        T_rollout = np.zeros((n_steps + 1, n_cells), dtype=np.float32)
        T_rollout[0] = T_current

        with torch.no_grad():
            for step in range(n_steps):
                t_idx = start_t + step
                t_val = float(times[t_idx])
                t_norm = t_val / 4000.0
                T_norm = ((T_current - dataset.T_mean) / (dataset.T_std + 1e-8)).astype(np.float32)
                Tset_norm = float((T_set - dataset.T_mean) / (dataset.T_std + 1e-8))
                node_feats = np.column_stack([
                    coords[:, 0], coords[:, 1], coords[:, 2], T_norm,
                    np.full(n_cells, Tset_norm, dtype=np.float32),
                    np.full(n_cells, region_id/11, dtype=np.float32),
                    np.full(n_cells, t_norm, dtype=np.float32),
                ]).astype(np.float32)
                x = torch.tensor(node_feats, dtype=torch.float32, device=device)
                batch = Batch.from_data_list([Data(x=x, edge_index=edge_index, edge_attr=edge_attr)]).to(device)
                dT_norm = model(batch).squeeze(-1).reshape(-1).cpu().numpy()
                dT = dT_norm * dataset.dT_std + dataset.dT_mean
                T_current = T_current + dT
                T_current = np.clip(T_current, T_min_r, T_max_r)
                T_rollout[step + 1] = T_current.astype(np.float32)

        T_true = rdata["T_array"][start_t: start_t + n_steps + 1]
        if region not in region_data:
            region_data[region] = {"p1p":[], "p1t":[], "p2p":[], "p2t":[]}
        ns = T_rollout.shape[0]
        p1s = min(p1_end + 1, ns)
        region_data[region]["p1p"].append(T_rollout[:p1s].ravel())
        region_data[region]["p1t"].append(T_true[:p1s].ravel())
        if p1_end < ns and p1_end < T_true.shape[0]:
            gt_end = min(ns, T_true.shape[0])
            region_data[region]["p2p"].append(T_rollout[p1_end:gt_end].ravel())
            region_data[region]["p2t"].append(T_true[p1_end:gt_end].ravel())

print()
print("=" * 80)
print("  NO CLAMPING — GNN predicts ALL 12 regions")
print("=" * 80)
header = "  {:>16}  {:>8}  {:>8}  {:>8}  {:>8}".format("Region", "P1 MAE", "P1 R2", "P2 MAE", "P2 R2")
print(header)
print("  " + "-" * 55)

all_p1p, all_p1t, all_p2p, all_p2t = [], [], [], []
for region, rd in region_data.items():
    m1 = compute_metrics(np.concatenate(rd["p1p"]), np.concatenate(rd["p1t"]))
    all_p1p.append(np.concatenate(rd["p1p"]))
    all_p1t.append(np.concatenate(rd["p1t"]))
    if rd["p2p"]:
        m2 = compute_metrics(np.concatenate(rd["p2p"]), np.concatenate(rd["p2t"]))
        all_p2p.append(np.concatenate(rd["p2p"]))
        all_p2t.append(np.concatenate(rd["p2t"]))
        print("  {:>16}  {:>7.2f}K  {:>8.4f}  {:>7.2f}K  {:>8.4f}".format(
            region, m1["mae"], m1["r2"], m2["mae"], m2["r2"]))
    else:
        print("  {:>16}  {:>7.2f}K  {:>8.4f}      N/A       N/A".format(
            region, m1["mae"], m1["r2"]))

m1 = compute_metrics(np.concatenate(all_p1p), np.concatenate(all_p1t))
m2 = compute_metrics(np.concatenate(all_p2p), np.concatenate(all_p2t))
print()
print("  OVERALL (all 12 regions predicted):")
print("    Phase 1: MAE={:.2f}K  R2={:.6f}".format(m1["mae"], m1["r2"]))
print("    Phase 2: MAE={:.2f}K  R2={:.6f}".format(m2["mae"], m2["r2"]))
print("=" * 80)

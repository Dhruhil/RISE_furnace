import sys, torch, numpy as np
sys.path.insert(0, '.')
from configs.base_config import CONFIG
from data.dataset_all_regions import AllRegionsDataset
from models.meshgraphnet import HeatTreatmentGNN
from models.rollout_all_regions import rollout_all_regions
from utils.metrics import compute_metrics

cfg = CONFIG
cfg.node_in_features = 7
device = 'cuda'
ckpt = f'{cfg.checkpoint_dir}_allregions/best_model.pt'
model = HeatTreatmentGNN.load(ckpt, cfg, device)
dataset = AllRegionsDataset(cfg.all_regions_dataset_path, cfg, 'test', 'evaluation')

n_train = cfg.n_train_steps
start_t = 40
p1_end = n_train - start_t

region_data = {}
for sim_i in dataset.sim_indices:
    results = rollout_all_regions(model, dataset, sim_i, start_t=start_t, device=device)
    for region, (T_pred, T_true) in results.items():
        if region not in region_data:
            region_data[region] = {"p1p":[], "p1t":[], "p2p":[], "p2t":[]}
        ns = T_pred.shape[0]
        p1s = min(p1_end + 1, ns)
        region_data[region]["p1p"].append(T_pred[:p1s].ravel())
        region_data[region]["p1t"].append(T_true[:p1s].ravel())
        if p1_end < ns and p1_end < T_true.shape[0]:
            gt_end = min(ns, T_true.shape[0])
            region_data[region]["p2p"].append(T_pred[p1_end:gt_end].ravel())
            region_data[region]["p2t"].append(T_true[p1_end:gt_end].ravel())

print()
print("=" * 80)
print("  FULL RESULTS — Phase 1 (0-3200s) and Phase 2 (3200-4000s)")
print("=" * 80)
header = "  {:>16}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}".format(
    "Region", "P1 MAE", "P1 R2", "P2 MAE", "P2 R2", "Clamped")
print(header)
print("  " + "-" * 60)

all_p1p, all_p1t, all_p2p, all_p2t = [], [], [], []
pred_p1p, pred_p1t, pred_p2p, pred_p2t = [], [], [], []

for region, rd in region_data.items():
    m1 = compute_metrics(np.concatenate(rd["p1p"]), np.concatenate(rd["p1t"]))
    clamped = "YES" if m1["mae"] < 0.01 else "no"
    all_p1p.append(np.concatenate(rd["p1p"]))
    all_p1t.append(np.concatenate(rd["p1t"]))
    if m1["mae"] > 0.01:
        pred_p1p.append(np.concatenate(rd["p1p"]))
        pred_p1t.append(np.concatenate(rd["p1t"]))
    if rd["p2p"]:
        m2 = compute_metrics(np.concatenate(rd["p2p"]), np.concatenate(rd["p2t"]))
        all_p2p.append(np.concatenate(rd["p2p"]))
        all_p2t.append(np.concatenate(rd["p2t"]))
        if m1["mae"] > 0.01:
            pred_p2p.append(np.concatenate(rd["p2p"]))
            pred_p2t.append(np.concatenate(rd["p2t"]))
        row = "  {:>16}  {:>7.2f}K  {:>8.4f}  {:>7.2f}K  {:>8.4f}  {:>8}".format(
            region, m1["mae"], m1["r2"], m2["mae"], m2["r2"], clamped)
    else:
        row = "  {:>16}  {:>7.2f}K  {:>8.4f}      N/A       N/A  {:>8}".format(
            region, m1["mae"], m1["r2"], clamped)
    print(row)

print()
print("=" * 80)
m_all1 = compute_metrics(np.concatenate(all_p1p), np.concatenate(all_p1t))
m_all2 = compute_metrics(np.concatenate(all_p2p), np.concatenate(all_p2t))
m_pr1 = compute_metrics(np.concatenate(pred_p1p), np.concatenate(pred_p1t))
m_pr2 = compute_metrics(np.concatenate(pred_p2p), np.concatenate(pred_p2t))
print("  ALL REGIONS:")
print("    Phase 1 (0-3200s):    MAE={:.2f}K   R2={:.6f}".format(m_all1["mae"], m_all1["r2"]))
print("    Phase 2 (3200-4000s): MAE={:.2f}K   R2={:.6f}".format(m_all2["mae"], m_all2["r2"]))
print()
print("  PREDICTED ONLY (steel + inner_box + outer_box):")
print("    Phase 1 (0-3200s):    MAE={:.2f}K   R2={:.6f}".format(m_pr1["mae"], m_pr1["r2"]))
print("    Phase 2 (3200-4000s): MAE={:.2f}K   R2={:.6f}".format(m_pr2["mae"], m_pr2["r2"]))
print("=" * 80)

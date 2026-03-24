#!/bin/bash
#SBATCH --job-name=gnn_test
#SBATCH --account=NAISS2026-4-525
#SBATCH --output=/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs/test_%j.log
#SBATCH --error=/mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official/outputs/logs/test_err_%j.log
#SBATCH --time=01:00:00
#SBATCH --gpus-per-node=A100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4

cd /mimer/NOBACKUP/groups/revar/GNN_PhysicsNeMo_Official

apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
  python -c "
import sys, torch, numpy as np, json
sys.path.insert(0, '.')
from configs.base_config import CONFIG
from data.dataset_all_regions import AllRegionsDataset
from models.meshgraphnet import HeatTreatmentGNN
from models.rollout_all_regions import rollout_all_regions
from utils.metrics import compute_metrics

cfg = CONFIG
device = 'cuda'
ckpt = f'{cfg.checkpoint_dir}_allregions/best_model.pt'

print('Loading model...')
cfg.node_in_features = 7
model = HeatTreatmentGNN.load(ckpt, cfg, device)

print('Loading dataset...')
dataset = AllRegionsDataset(cfg.all_regions_dataset_path, cfg, 'test', 'evaluation')

sim_i = dataset.sim_indices[0]
sim = dataset._simulations[sim_i]
print(f'Test sim: {sim[\"name\"]}  T_set={sim[\"T_set\"]:.0f}K')

print('Running rollout...')
results = rollout_all_regions(model, dataset, sim_i, start_t=40, device=device)

n_train = cfg.n_train_steps
start_t = 40
p1_end = n_train - start_t

print()
print('=' * 70)
print(f'  RESULTS — Phase 1 (0-3200s) and Phase 2 (3200-4000s)')
print('=' * 70)
print(f'  {\"Region\":>16}  {\"P1 MAE (K)\":>12}  {\"P2 MAE (K)\":>12}  {\"Clamped?\":>10}')
print(f'  {\"-\"*55}')

for region, (T_pred, T_true) in results.items():
    ns = T_pred.shape[0]
    p1s = min(p1_end + 1, ns)
    m1 = compute_metrics(T_pred[:p1s].ravel(), T_true[:p1s].ravel())

    if p1_end < ns and p1_end < T_true.shape[0]:
        gt_end = min(ns, T_true.shape[0])
        m2 = compute_metrics(T_pred[p1_end:gt_end].ravel(), T_true[p1_end:gt_end].ravel())
    else:
        m2 = {'mae': 0.0}

    clamped = 'YES' if m1['mae'] < 0.01 else 'no'
    print(f'  {region:>16}  {m1[\"mae\"]:>12.2f}  {m2[\"mae\"]:>12.2f}  {clamped:>10}')

# Summary
all_p1, all_p2 = [], []
for region, (T_pred, T_true) in results.items():
    ns = T_pred.shape[0]
    p1s = min(p1_end + 1, ns)
    m1 = compute_metrics(T_pred[:p1s].ravel(), T_true[:p1s].ravel())
    all_p1.append(m1['mae'])
    if p1_end < ns and p1_end < T_true.shape[0]:
        gt_end = min(ns, T_true.shape[0])
        m2 = compute_metrics(T_pred[p1_end:gt_end].ravel(), T_true[p1_end:gt_end].ravel())
        all_p2.append(m2['mae'])

# Predicted regions only (exclude clamped)
pred_p1 = [m for m in all_p1 if m > 0.01]
pred_p2 = [m for m in all_p2 if m > 0.01]

print()
print(f'  OVERALL (all regions):      P1={np.mean(all_p1):.2f}K  P2={np.mean(all_p2):.2f}K')
print(f'  PREDICTED ONLY (3 regions): P1={np.mean(pred_p1):.2f}K  P2={np.mean(pred_p2):.2f}K')
print()

# Verify checks
print('=' * 70)
print('  VERIFICATION CHECKS')
print('=' * 70)
heaters_ok = all(m < 0.01 for r, (Tp, Tt) in results.items()
                 for m in [compute_metrics(Tp[:min(p1_end+1,Tp.shape[0])].ravel(),
                           Tt[:min(p1_end+1,Tt.shape[0])].ravel())['mae']]
                 if 'heater' in r or r == 'brick_heater')
print(f'  Heaters clamped (0.00K):    {\"PASS\" if heaters_ok else \"FAIL\"} ')

outer_ok = any(r == 'outer_box' and compute_metrics(
    Tp[:min(p1_end+1,Tp.shape[0])].ravel(),
    Tt[:min(p1_end+1,Tt.shape[0])].ravel())['mae'] < 20
    for r, (Tp, Tt) in results.items())
print(f'  outer_box < 20K:            {\"PASS\" if outer_ok else \"FAIL (was 48K before fix)\"}')

steel_ok = any(r == 'steel_cylinder' and compute_metrics(
    Tp[:min(p1_end+1,Tp.shape[0])].ravel(),
    Tt[:min(p1_end+1,Tt.shape[0])].ravel())['mae'] < 10
    for r, (Tp, Tt) in results.items())
print(f'  steel_cylinder < 10K:       {\"PASS\" if steel_ok else \"FAIL\"}')
print()
"

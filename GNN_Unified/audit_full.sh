#!/bin/bash
echo "============================================================"
echo "  GNN_Unified — FULL AUDIT"
echo "============================================================"

echo ""
echo "### 1. Physics loss structure ###"
grep -nE "physics_loss|w_conduction|w_convection|w_radiation|sigma_sb|epsilon" \
    train_unified.py configs/base_config.py 2>/dev/null | head -40

echo ""
echo "### 2. Pushforward logic ###"
grep -nE "pushforward|x2\[|batch\.x\[:, 6\]|T_pred1|T_pred2" \
    train_unified.py 2>/dev/null | head -20

echo ""
echo "### 3. Heater clamping (is T_set enforced?) ###"
grep -nE "clamp|is_heater|HEATER_REGIONS|T_set" \
    train_unified.py data/dataset_unified.py 2>/dev/null | head -20

echo ""
echo "### 4. Edge feature construction ###"
grep -nE "edge_attr|edge_index|knn_graph|cKDTree" \
    data/dataset_unified.py 2>/dev/null | head -20

echo ""
echo "### 5. Target normalization (y) ###"
grep -nE "self\.y|batch\.y|dT_mean|dT_std|T_tp" \
    data/dataset_unified.py train_unified.py 2>/dev/null | head -20

echo ""
echo "### 6. NaN handling ###"
grep -nE "isnan|nan_to_num|NaN" \
    train_unified.py data/dataset_unified.py 2>/dev/null

echo ""
echo "### 7. t_start (first timestep used) ###"
grep -nE "t_start|range\(20" \
    data/dataset_unified.py evaluation/evaluate_unified.py 2>/dev/null

echo ""
echo "### 8. Final config values ###"
grep -E "^\s+(n_epochs|batch_size|learning_rate|hidden_features|n_message_passing|graph_k|w_|dt:|t_total|train_time|predict_time)" \
    configs/base_config.py

echo ""
echo "### 9. Dataset path (final verification) ###"
grep "all_regions_dataset_path" configs/base_config.py

echo ""
echo "### 10. NaN check on HDF5 ###"
module load h5py/3.12.1-foss-2024a 2>/dev/null
python3 << 'PYEOF' 2>/dev/null
import h5py, numpy as np
try:
    with h5py.File("/mimer/NOBACKUP/groups/revar/Dataset_k8/dataset_all_regions_clean.h5") as f:
        n = int(f.attrs["n_cases"])
        total_nan = 0
        for ci in range(n):
            g = f[f"case_{ci:03d}"]
            for r in g.keys():
                if isinstance(g[r], h5py.Group) and "T" in g[r]:
                    T = g[r]["T"][:]
                    total_nan += int(np.isnan(T).sum())
        print(f"  Total NaN values across all cases/regions: {total_nan}")
        print("  (should be 0 for clean dataset)" if total_nan == 0 else "  WARNING: NaNs present")
except Exception as e:
    print(f"  Error checking NaN: {e}")
PYEOF

echo ""
echo "============================================================"
echo "  AUDIT COMPLETE"
echo "============================================================"

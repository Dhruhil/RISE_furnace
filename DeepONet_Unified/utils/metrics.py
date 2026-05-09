"""
utils/metrics.py
----------------
Evaluation metrics. Mirrors GNN_Unified/utils/metrics.py.
"""

import torch
import numpy as np
from typing import Dict
from configs.base_config import TARGET_REGIONS


def relative_l2(pred: torch.Tensor, true: torch.Tensor) -> float:
    return (torch.norm(pred - true) / (torch.norm(true) + 1e-8)).item()


def compute_metrics(
    pred_K: np.ndarray,
    true_K: np.ndarray,
    region_idx: np.ndarray,
) -> Dict:
    """
    Compute MAE, RMSE, R², Rel-L2 overall and per region.

    Parameters
    ----------
    pred_K, true_K : (N,) arrays in Kelvin
    region_idx     : (N,) int array, index into TARGET_REGIONS
    """
    overall = _scalar_metrics(pred_K, true_K)
    overall["name"] = "overall"

    per_region = []
    for r_idx, region in enumerate(TARGET_REGIONS):
        mask = (region_idx == r_idx)
        if mask.sum() == 0:
            continue
        m = _scalar_metrics(pred_K[mask], true_K[mask])
        m["name"] = region
        per_region.append(m)

    return {"overall": overall, "per_region": per_region}


def _scalar_metrics(pred: np.ndarray, true: np.ndarray) -> Dict:
    mae   = float(np.mean(np.abs(pred - true)))
    rmse  = float(np.sqrt(np.mean((pred - true) ** 2)))
    rel   = float(np.linalg.norm(pred - true) / (np.linalg.norm(true) + 1e-8))
    ss_r  = float(np.sum((true - pred) ** 2))
    ss_t  = float(np.sum((true - true.mean()) ** 2))
    r2    = 1.0 - ss_r / (ss_t + 1e-8)
    w5    = float(np.mean(np.abs(pred - true) < 5.0)  * 100)
    w10   = float(np.mean(np.abs(pred - true) < 10.0) * 100)
    return dict(MAE=mae, RMSE=rmse, RelL2=rel, R2=r2,
                within5K=w5, within10K=w10)


def print_metrics(metrics: Dict):
    ov = metrics["overall"]
    print(f"\n{'='*55}")
    print(f"  Overall  MAE={ov['MAE']:.3f}K  RMSE={ov['RMSE']:.3f}K  "
          f"R²={ov['R2']:.4f}  RelL2={ov['RelL2']*100:.2f}%")
    print(f"{'='*55}")
    for m in metrics["per_region"]:
        print(f"  {m['name']:<18} MAE={m['MAE']:.3f}K  "
              f"RMSE={m['RMSE']:.3f}K  R²={m['R2']:.4f}  "
              f"RelL2={m['RelL2']*100:.2f}%")

# DeepONet — Heat Treatment Digital Twin

### Master's Thesis — Simulating Heat Treatment of Cast Metal Products using OpenFOAM and AI

**Deep Operator Network** surrogate for temperature prediction across all
furnace regions. Wraps the official NVIDIA PhysicsNeMo DeepONet with a
pure-PyTorch fallback, mirroring the structure of
`FNO_PhysicsNeMo_Official/` and `GNN_PhysicsNeMo_Official/`.

---

## Architecture

```
Branch input (current T field on sensor lattice):
    (batch, 6, n_sensors=2160)
        channels: [T_norm, region_id/11, is_heater,
                   kappa/100, Cp/1000, rho/10000]
Branch scalars: (batch, 2) = [T_set_norm, time/t_total]

Trunk input (query points):
    (batch, n_query=4096, 8)
        features: [x, y, z, region_id/11, is_heater,
                   kappa/100, Cp/1000, rho/10000]

                  ↓
     Branch MLP → b ∈ R^128     Trunk MLP → t ∈ R^128
                  ↓
         G(u)(y) = <b, t> + bias
                  ↓
Output: (batch, n_query) = normalised T_next
```

## Dataset

Uses `dataset_v2_all_regions_clean.h5` (78 cases), identical to FNO and GNN.

```
attrs: n_cases, regions
case_XXX/
    attrs: name, T_set
    times
    steel_cylinder/{coords, T}
    inner_box/{coords, T}
    heater_1 … heater_8/{coords, T}
    brick_heater/{coords, T}
    outer_box/{coords, T}
```

## How to run on Alvis

```bash
cd /mimer/NOBACKUP/groups/revar/DeepONet_PhysicsNeMo_Official
sbatch run_sanity_test_deeponet.sh    # quick GPU check
sbatch run_alvis_deeponet.sh          # full 100-epoch training
sbatch run_eval_deeponet.sh           # rollout evaluation
```

Inside a running container:
```bash
apptainer exec --nv /mimer/NOBACKUP/groups/revar/physicsnemo_25.06.sif \
    python -u train.py --epochs 100 --lr 1e-3 --batch 4 --lam 0.003
```

## Folder layout (matches FNO)

```
DeepONet_PhysicsNeMo_Official/
├── configs/deeponet_config.py
├── data/dataset.py
├── models/
│   ├── deeponet_model.py
│   └── rollout.py
├── training/
│   ├── train.py
│   ├── loss.py
│   └── scheduler.py
├── evaluation/evaluate_deeponet.py
├── utils/
│   ├── checkpoint.py
│   ├── logging.py
│   └── metrics.py
├── outputs/{checkpoints,logs,evaluation,plots,rollout_results}/
├── train.py                     # top-level entry (matches FNO)
├── run_alvis_deeponet.sh
├── run_sanity_test_deeponet.sh
├── run_eval_deeponet.sh
└── README.md
```

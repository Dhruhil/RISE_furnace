"""
Save per-step T(t) arrays for thesis Figure RQ3 — true vs prediction.
Saves MESH-LEVEL temperatures (not grid) by interpolating FNO grid
predictions back to the original OpenFOAM mesh cells.
"""
from __future__ import annotations
import sys, time, argparse
from pathlib import Path
import numpy as np
import torch
import h5py
from scipy.interpolate import NearestNDInterpolator

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.fno_config import CONFIG
from data.dataset import FNO3DDataset, REGION_IDS, HEATER_REGIONS
from models.fno_model import HeatTreatmentFNO3D
from models.rollout import rollout_fno3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--n_sims",     type=int, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_t",    type=int, default=20)
    args = parser.parse_args()

    cfg = CONFIG
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rollout_temps.h5"

    print(f"\n{'='*80}")
    print(f"  FNO ROLLOUT — saving full mesh temperatures (interpolated grid -> mesh)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output:     {out_path}")
    print(f"{'='*80}\n")

    print(f"  Loading model from {args.checkpoint}")
    model = HeatTreatmentFNO3D.load(args.checkpoint, cfg, args.device)
    model.eval()

    dataset = FNO3DDataset(cfg.dataset_path, cfg, "test", "evaluation")
    sim_indices = dataset.sim_indices
    if args.n_sims is not None:
        sim_indices = sim_indices[:args.n_sims]

    grid_points = dataset.grid_points  # (n_grid, 3) - the FNO regular grid coords

    with h5py.File(out_path, "w") as f:
        f.attrs["model"]      = "FNO"
        f.attrs["checkpoint"] = str(args.checkpoint)
        f.attrs["start_t"]    = args.start_t
        f.attrs["dt"]         = float(cfg.dt)

        for i, sim_i in enumerate(sim_indices):
            sim = dataset._simulations[sim_i]
            print(f"  [{i+1}/{len(sim_indices)}] Sim {sim_i} (T_set={sim['T_set']:.0f}K)")
            t0 = time.time()
            # rollout_fno3d returns grid predictions (n_steps, gx, gy, gz)
            T_pred_grid, _ = rollout_fno3d(model, dataset, sim_i,
                                            device=device, start_t=args.start_t)
            rt = time.time() - t0
            print(f"      Rollout: {rt:.1f}s, grid shape={T_pred_grid.shape}")

            # Now interpolate grid -> mesh cells
            n_steps = T_pred_grid.shape[0]
            coords = sim["coords"]            # (n_cells, 3)
            n_cells = coords.shape[0]

            # Build mesh-level T_pred and T_true
            T_pred_mesh = np.zeros((n_steps, n_cells), dtype=np.float32)
            T_true_mesh = np.zeros((n_steps, n_cells), dtype=np.float32)

            print(f"      Interpolating to {n_cells} mesh cells...")
            for step in range(n_steps):
                t_idx = args.start_t + step
                if t_idx >= sim["n_times"]:
                    break
                # Ground truth on mesh
                T_true_mesh[step] = sim["T_all"][t_idx]
                # Interpolate grid prediction to mesh
                interp = NearestNDInterpolator(grid_points,
                                                T_pred_grid[step].ravel())
                T_pred_mesh[step] = interp(coords).astype(np.float32)

            # Heater override (boundary condition)
            heater_rids = {REGION_IDS[r] for r in HEATER_REGIONS if r in REGION_IDS}
            region_id = np.zeros(n_cells, dtype=np.int32)
            for rname, (s, e) in sim["region_slices"].items():
                region_id[s:e] = REGION_IDS.get(rname, -1)
            heater_mask = np.array([rid in heater_rids for rid in region_id])
            for step in range(n_steps):
                t_idx = args.start_t + step
                if t_idx < sim["n_times"]:
                    T_pred_mesh[step, heater_mask] = sim["T_all"][t_idx, heater_mask]

            times = sim["times"][args.start_t : args.start_t + n_steps]

            steel_id = REGION_IDS["steel_cylinder"]
            air_id   = REGION_IDS["inner_box"]
            outer_id = REGION_IDS["outer_box"]
            is_steel = (region_id == steel_id)
            is_air   = (region_id == air_id)
            is_outer = (region_id == outer_id)

            grp = f.create_group(f"sim_{sim_i}")
            grp.attrs["T_set"]  = float(sim["T_set"])
            for k in ["cx", "cy", "cz", "radius", "height"]:
                if k in sim:
                    grp.attrs[k] = float(sim[k])

            grp.create_dataset("times",     data=np.asarray(times, dtype=np.float32))
            grp.create_dataset("T_pred",    data=T_pred_mesh,
                               compression="gzip", compression_opts=4)
            grp.create_dataset("T_true",    data=T_true_mesh,
                               compression="gzip", compression_opts=4)
            grp.create_dataset("is_steel",  data=is_steel)
            grp.create_dataset("is_air",    data=is_air)
            grp.create_dataset("is_outer",  data=is_outer)
            grp.create_dataset("region_id", data=region_id)

    print(f"\n  Saved: {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1e6:.1f} MB\n")


if __name__ == "__main__":
    main()

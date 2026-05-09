"""
Save the FNO rollout as MESH-LEVEL temperature arrays — interpolates
the FNO's grid predictions back onto the original OpenFOAM mesh cells.

Companion to save_rollout_native_grid.py: that one writes the raw
voxel grid output, this one writes the mesh-level version that
plots cleanly against the OpenFOAM ground truth and against the
GNN predictions on the same mesh. The output of this script is what
feeds into the thesis Figure 5.4 (RQ3 prediction-vs-truth curves).

Pipeline per sim:
  1. Run the FNO rollout on its native voxel grid.
  2. Nearest-neighbour interpolate each grid output onto the mesh
     cells of that sim.
  3. Clamp the heater cells back to the OpenFOAM ground truth so
     the Dirichlet BC is preserved (matches what the GNN evaluator
     does, so the two pipelines stay comparable).
  4. Bundle predictions, ground truth, region masks, and metadata
     into one HDF5 group per sim.
"""
from __future__ import annotations
import sys, time, argparse
from pathlib import Path
import numpy as np
import torch
import h5py
from scipy.interpolate import NearestNDInterpolator

# Make the project importable when running this from the eval/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.fno_config import CONFIG
from data.dataset import FNO3DDataset, REGION_IDS, HEATER_REGIONS
from models.fno_model import HeatTreatmentFNO3D
from models.rollout import rollout_fno3d


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--n_sims",     type=int, default=None,
                        help="Limit to first N test sims (default: all)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to the trained FNO best_model.pt")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to write rollout_temps.h5")
    parser.add_argument("--start_t",    type=int, default=20,
                        help="Rollout start step (default 20 = t=200s)")
    args = parser.parse_args()

    cfg = CONFIG
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Make sure the target dir exists before the long rollout starts —
    # cheap up front, saves a head-scratch later if the path was wrong.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rollout_temps.h5"

    print(f"\n{'='*80}")
    print(f"  FNO ROLLOUT — saving full mesh temperatures (interpolated grid -> mesh)")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output:     {out_path}")
    print(f"{'='*80}\n")

    # ---- load the trained checkpoint -------------------------------
    print(f"  Loading model from {args.checkpoint}")
    model = HeatTreatmentFNO3D.load(args.checkpoint, cfg, args.device)
    model.eval()

    # Same test split used everywhere else, so the cases line up
    # with the standard FNO and GNN evaluation runs.
    dataset = FNO3DDataset(cfg.dataset_path, cfg, "test", "evaluation")
    sim_indices = dataset.sim_indices
    if args.n_sims is not None:
        # Quick-pass mode for sanity checks during dev
        sim_indices = sim_indices[:args.n_sims]

    # The FNO operates on a regular voxel grid — these are the
    # (Gx*Gy*Gz, 3) coordinates of those grid points, used as the
    # source points for the grid->mesh interpolation below.
    grid_points = dataset.grid_points

    # ---- run rollouts and dump everything to one HDF5 file --------
    with h5py.File(out_path, "w") as f:
        # Top-level attrs — useful for downstream scripts that want
        # to know which model produced this file and how to interpret
        # the time axis without re-reading the config.
        f.attrs["model"]      = "FNO"
        f.attrs["checkpoint"] = str(args.checkpoint)
        f.attrs["start_t"]    = args.start_t
        f.attrs["dt"]         = float(cfg.dt)

        for i, sim_i in enumerate(sim_indices):
            sim = dataset._simulations[sim_i]
            print(f"  [{i+1}/{len(sim_indices)}] Sim {sim_i} (T_set={sim['T_set']:.0f}K)")
            t0 = time.time()

            # rollout_fno3d returns grid predictions of shape
            # (n_steps, gx, gy, gz) — that's the native FNO output.
            T_pred_grid, _ = rollout_fno3d(model, dataset, sim_i,
                                            device=device, start_t=args.start_t)
            rt = time.time() - t0
            print(f"      Rollout: {rt:.1f}s, grid shape={T_pred_grid.shape}")

            # ---- grid -> mesh interpolation -------------------------
            # Each step gets its own NearestNDInterpolator. Construction
            # cost is dominated by the kd-tree build, which is fast for
            # the grid sizes here (~58k voxels), so doing it in a tight
            # loop is fine.
            n_steps = T_pred_grid.shape[0]
            coords = sim["coords"]            # (n_cells, 3)
            n_cells = coords.shape[0]

            T_pred_mesh = np.zeros((n_steps, n_cells), dtype=np.float32)
            T_true_mesh = np.zeros((n_steps, n_cells), dtype=np.float32)

            print(f"      Interpolating to {n_cells} mesh cells...")
            for step in range(n_steps):
                t_idx = args.start_t + step
                if t_idx >= sim["n_times"]:
                    # Last few sims occasionally have a missing tail;
                    # bail out before reading off the end of T_all.
                    break
                # Ground truth straight off the mesh (no interpolation
                # needed since OpenFOAM dumped it on the mesh in the
                # first place).
                T_true_mesh[step] = sim["T_all"][t_idx]

                # FNO prediction lives on the voxel grid; project it
                # back onto the mesh cell centres.
                interp = NearestNDInterpolator(grid_points,
                                                T_pred_grid[step].ravel())
                T_pred_mesh[step] = interp(coords).astype(np.float32)

            # ---- heater Dirichlet override --------------------------
            # Heaters and the brick heater are clamped to T_set in
            # OpenFOAM. The grid->mesh interpolation can smear that
            # boundary across cells near the heater edges, so re-apply
            # the ground truth on those cells. This mirrors what the
            # GNN evaluator does, keeping the comparison fair.
            heater_rids = {REGION_IDS[r] for r in HEATER_REGIONS if r in REGION_IDS}
            region_id = np.zeros(n_cells, dtype=np.int32)
            for rname, (s, e) in sim["region_slices"].items():
                region_id[s:e] = REGION_IDS.get(rname, -1)
            heater_mask = np.array([rid in heater_rids for rid in region_id])
            for step in range(n_steps):
                t_idx = args.start_t + step
                if t_idx < sim["n_times"]:
                    T_pred_mesh[step, heater_mask] = sim["T_all"][t_idx, heater_mask]

            # Slice the time axis to match the rollout length
            times = sim["times"][args.start_t : args.start_t + n_steps]

            # Per-region mesh masks for the three predicted regions —
            # what the plotting / metrics scripts use to slice into
            # T_pred_mesh and T_true_mesh.
            steel_id = REGION_IDS["steel_cylinder"]
            air_id   = REGION_IDS["inner_box"]
            outer_id = REGION_IDS["outer_box"]
            is_steel = (region_id == steel_id)
            is_air   = (region_id == air_id)
            is_outer = (region_id == outer_id)

            # ---- write this sim's group to the HDF5 -----------------
            grp = f.create_group(f"sim_{sim_i}")
            grp.attrs["T_set"]  = float(sim["T_set"])
            # Geometry attrs — useful when filtering rollouts by
            # cylinder position later in the analysis notebooks.
            for k in ["cx", "cy", "cz", "radius", "height"]:
                if k in sim:
                    grp.attrs[k] = float(sim[k])

            # gzip-4 keeps the file size under control without
            # slowing the write down too much. Most of the bytes
            # go into the T_pred / T_true float arrays.
            grp.create_dataset("times",     data=np.asarray(times, dtype=np.float32))
            grp.create_dataset("T_pred",    data=T_pred_mesh,
                               compression="gzip", compression_opts=4)
            grp.create_dataset("T_true",    data=T_true_mesh,
                               compression="gzip", compression_opts=4)
            # Boolean masks aren't gzipped — they're already tiny
            # and the compression overhead isn't worth it.
            grp.create_dataset("is_steel",  data=is_steel)
            grp.create_dataset("is_air",    data=is_air)
            grp.create_dataset("is_outer",  data=is_outer)
            grp.create_dataset("region_id", data=region_id)

    print(f"\n  Saved: {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1e6:.1f} MB\n")


if __name__ == "__main__":
    main()
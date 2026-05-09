"""
Save the FNO rollout on its NATIVE 3D grid — no mesh interpolation
applied at the output side.

Background: the FNO predicts on a regular voxel grid, but the
standard evaluation script interpolates those predictions back
onto the OpenFOAM mesh so the comparison against the GNN stays
apples-to-apples. That extra interpolation step adds a layer of
nearest-neighbour averaging that can hide what the FNO actually
produces.

This script writes the raw grid-level predictions to disk so the
FNO output can be inspected directly. Useful for figuring out
whether an apparent under-prediction is an artefact of the
grid->mesh interpolation or something the architecture itself
is doing — that distinction matters for the discussion in
Section 6.3 of the thesis.
"""
from __future__ import annotations
import sys, time, argparse
from pathlib import Path
import numpy as np
import torch
import h5py

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
                        help="Where to write rollout_temps_native_grid.h5")
    parser.add_argument("--start_t",    type=int, default=20,
                        help="Rollout start step (default 20 = t=200s)")
    args = parser.parse_args()

    cfg = CONFIG
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Make sure the target dir exists before the long rollout starts —
    # cheap up front, saves a head-scratch later if the path was wrong.
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rollout_temps_native_grid.h5"

    print(f"\n{'='*80}")
    print(f"  FNO ROLLOUT — saving NATIVE GRID temperatures (no mesh interpolation)")
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

    Gx, Gy, Gz = dataset.grid_shape
    print(f"  Grid shape: {Gx} x {Gy} x {Gz} = {Gx*Gy*Gz} voxels\n")

    # ---- run rollouts and dump everything to one HDF5 file --------
    with h5py.File(out_path, "w") as f:
        # Top-level attrs — mostly useful for downstream scripts that
        # want to know which model produced this file and how to
        # interpret the time axis without re-reading the config.
        f.attrs["model"]      = "FNO_native_grid"
        f.attrs["checkpoint"] = str(args.checkpoint)
        f.attrs["start_t"]    = args.start_t
        f.attrs["dt"]         = float(cfg.dt)
        f.attrs["grid_shape"] = (Gx, Gy, Gz)

        for i, sim_i in enumerate(sim_indices):
            sim = dataset._simulations[sim_i]
            static = dataset._static_grids[sim_i]
            print(f"  [{i+1}/{len(sim_indices)}] Sim {sim_i} (T_set={sim['T_set']:.0f}K)")

            t0 = time.time()
            T_pred_grid, T_true_grid = rollout_fno3d(
                model, dataset, sim_i, device=device, start_t=args.start_t)
            rt = time.time() - t0
            print(f"      Rollout: {rt:.1f}s, grid shape={T_pred_grid.shape}")

            # Build region masks directly on the voxel grid — these
            # are the masks needed when comparing FNO output without
            # ever touching the original OpenFOAM mesh.
            region_id_grid = static["interp_fields"]["region_id"].squeeze(-1)
            # Shape: (Gx, Gy, Gz)

            steel_id  = REGION_IDS["steel_cylinder"]
            air_id    = REGION_IDS["inner_box"]
            outer_id  = REGION_IDS["outer_box"]

            is_steel_grid = (region_id_grid == steel_id)
            is_air_grid   = (region_id_grid == air_id)
            is_outer_grid = (region_id_grid == outer_id)

            # Quick sanity check on the masks — voxel counts here
            # should match across runs since the geometry is fixed.
            n_steel_voxels = is_steel_grid.sum()
            n_air_voxels   = is_air_grid.sum()
            print(f"      Grid masks: steel={n_steel_voxels} voxels, "
                  f"air={n_air_voxels} voxels")

            # Slice the time axis to match the rollout length —
            # rollout starts at start_t, runs for T_pred_grid.shape[0] steps.
            times = sim["times"][args.start_t : args.start_t + T_pred_grid.shape[0]]

            grp = f.create_group(f"sim_{sim_i}")
            grp.attrs["T_set"] = float(sim["T_set"])
            # Geometry attrs — useful when filtering rollouts by
            # cylinder position later in the analysis notebooks.
            for k in ["cx", "cy", "cz", "radius", "height"]:
                if k in sim:
                    grp.attrs[k] = float(sim[k])
            grp.attrs["n_steel_voxels"] = int(n_steel_voxels)
            grp.attrs["n_air_voxels"]   = int(n_air_voxels)

            # gzip-4 keeps the file size under control without slowing
            # the write down too much. Most of the bytes go into the
            # T_pred_grid / T_true_grid float arrays.
            grp.create_dataset("times",         data=np.asarray(times, dtype=np.float32))
            grp.create_dataset("T_pred_grid",   data=T_pred_grid.astype(np.float32),
                               compression="gzip", compression_opts=4)
            grp.create_dataset("T_true_grid",   data=T_true_grid.astype(np.float32),
                               compression="gzip", compression_opts=4)
            # Boolean masks aren't gzipped — they're already tiny
            # and the compression overhead isn't worth it.
            grp.create_dataset("is_steel_grid", data=is_steel_grid)
            grp.create_dataset("is_air_grid",   data=is_air_grid)
            grp.create_dataset("is_outer_grid", data=is_outer_grid)
            grp.create_dataset("region_id_grid", data=region_id_grid.astype(np.int32))

    print(f"\n  Saved: {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1e6:.1f} MB\n")


if __name__ == "__main__":
    main()
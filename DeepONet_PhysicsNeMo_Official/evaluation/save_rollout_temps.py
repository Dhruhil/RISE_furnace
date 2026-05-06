"""
Save per-step T(t) arrays for thesis Figure RQ3 — true vs prediction.
Uses the same rollout_deeponet from models/rollout.py.
"""
from __future__ import annotations
import sys, time, argparse
from pathlib import Path
import numpy as np
import torch
import h5py

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.deeponet_config import CONFIG
from data.dataset import get_deeponet_eval_dataset
from models.deeponet_model import HeatTreatmentDeepONet
from models.rollout import rollout_deeponet
from utils.checkpoint import load_best


REGION_NAMES = {
    "steel_cylinder": 0, "inner_box": 1, "outer_box": 2,
    "heater_1": 3, "heater_2": 4, "heater_3": 5, "heater_4": 6,
    "heater_5": 7, "heater_6": 8, "heater_7": 9, "heater_8": 10,
    "brick_heater": 11,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device",     default="cuda")
    parser.add_argument("--n_sims",     type=int, default=None)
    parser.add_argument("--ckpt",       type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--start_t",    type=int, default=20)
    args = parser.parse_args()

    cfg = CONFIG
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "rollout_temps.h5"

    print(f"\n{'='*80}")
    print(f"  DeepONet ROLLOUT — saving full mesh temperatures for thesis plotting")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Output:     {out_path}")
    print(f"{'='*80}\n")

    print(f"  Loading model from {args.ckpt}")
    model = HeatTreatmentDeepONet(cfg).to(device)
    load_best(model, args.ckpt, device)
    model.eval()

    ds = get_deeponet_eval_dataset(cfg)
    sim_indices = ds.sim_indices
    if args.n_sims is not None:
        sim_indices = sim_indices[:args.n_sims]

    with h5py.File(out_path, "w") as f:
        f.attrs["model"]      = "DeepONet"
        f.attrs["checkpoint"] = str(args.ckpt)
        f.attrs["start_t"]    = args.start_t
        f.attrs["dt"]         = float(cfg.dt)

        for i, sim_i in enumerate(sim_indices):
            sim = ds._simulations[sim_i]
            print(f"  [{i+1}/{len(sim_indices)}] Sim {sim_i} (T_set={sim['T_set']:.0f}K)")
            t0 = time.time()
            T_pred, T_true = rollout_deeponet(model, ds, sim_i,
                                               device=device, start_t=args.start_t)
            rt = time.time() - t0
            print(f"      Rollout: {rt:.1f}s, shape={T_pred.shape}")

            times = sim["times"][args.start_t : args.start_t + T_pred.shape[0]]

            # region_id is already on each cell in DeepONet sim
            region_id = np.asarray(sim["region_id"], dtype=np.int32)

            steel_id = REGION_NAMES["steel_cylinder"]
            air_id   = REGION_NAMES["inner_box"]
            outer_id = REGION_NAMES["outer_box"]
            is_steel = (region_id == steel_id)
            is_air   = (region_id == air_id)
            is_outer = (region_id == outer_id)

            grp = f.create_group(f"sim_{sim_i}")
            grp.attrs["T_set"] = float(sim["T_set"])
            for k in ["cx", "cy", "cz", "radius", "height"]:
                if k in sim:
                    grp.attrs[k] = float(sim[k])

            grp.create_dataset("times",     data=np.asarray(times, dtype=np.float32))
            grp.create_dataset("T_pred",    data=T_pred.astype(np.float32),
                               compression="gzip", compression_opts=4)
            grp.create_dataset("T_true",    data=T_true.astype(np.float32),
                               compression="gzip", compression_opts=4)
            grp.create_dataset("is_steel",  data=is_steel)
            grp.create_dataset("is_air",    data=is_air)
            grp.create_dataset("is_outer",  data=is_outer)
            grp.create_dataset("region_id", data=region_id)

    print(f"\n  Saved: {out_path}")
    print(f"  File size: {out_path.stat().st_size / 1e6:.1f} MB\n")


if __name__ == "__main__":
    main()

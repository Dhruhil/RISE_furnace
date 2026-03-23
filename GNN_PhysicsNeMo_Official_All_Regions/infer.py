"""
Top-level inference entry point.

BUGS FIXED vs old version:
  1. Old version passed extra_steps= to run_inference() but the new
     inference/infer.py no longer takes extra_steps — it uses Option A logic.
  2. Default target_time updated to 3500s (in verification window) to
     demonstrate Option A rather than training window prediction.

Usage:
    python infer.py --sim_idx 0 --target_time 3500   # verification window
    python infer.py --sim_idx 0 --target_time 2000   # training window
    python infer.py --sim_idx 2 --target_time 4000   # end of simulation
"""

import argparse
import sys
sys.path.insert(0, ".")

import torch
from configs.base_config import CONFIG
from inference.infer import run_inference


def main():
    parser = argparse.ArgumentParser(
        description="GNN inference — predict at any time in 0–4000s"
    )
    parser.add_argument("--checkpoint",  default=None)
    parser.add_argument("--sim_idx",     type=int,   default=0)
    parser.add_argument("--target_time", type=float, default=3500.0,
                        help=(
                            "Target time [s]. "
                            "Use 3200–4000 to test the verification window "
                            "(unseen during training)."
                        ))
    parser.add_argument("--device",      default=None)
    args = parser.parse_args()

    cfg    = CONFIG
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = args.checkpoint or f"{cfg.checkpoint_dir}/best_model.pt"
    save_dir = (
        f"{cfg.output_dir}/predictions/"
        f"sim{args.sim_idx:03d}_t{int(args.target_time)}s"
    )

    run_inference(
        cfg             = cfg,
        checkpoint_path = ckpt,
        sim_idx         = args.sim_idx,
        target_time     = args.target_time,
        device          = device,
        save_dir        = save_dir,
    )


if __name__ == "__main__":
    main()

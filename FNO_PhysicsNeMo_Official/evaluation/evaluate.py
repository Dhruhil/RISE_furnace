"""
Full test-set evaluation for FNO — ALL REGIONS.
Master's Thesis: Simulating Heat Treatment using OpenFOAM and AI

Usage:
    python evaluation/evaluate.py
    python evaluation/evaluate.py --checkpoint outputs/checkpoints/best_model.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.fno_config import CONFIG
from models.fno_model import HeatTreatmentFNO
from train import run_verification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    cfg    = CONFIG
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = args.checkpoint or f"{cfg.checkpoint_dir}/best_model.pt"

    print(f"\n  Loading FNO checkpoint: {ckpt}")
    model = HeatTreatmentFNO.load(ckpt, cfg, device)
    run_verification(model, cfg, device, f"{cfg.output_dir}/evaluation")


if __name__ == "__main__":
    main()

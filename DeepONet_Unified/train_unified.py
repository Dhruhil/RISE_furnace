"""
train_unified.py
----------------
Main training entry point for DeepONet_Unified.
Mirrors GNN_Unified/train_unified.py exactly.

Usage:
    python train_unified.py [--epochs 200] [--lr 5e-5] [--batch 4096] [--lam 0.003]
"""

import os
import sys
import argparse
import torch
import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from configs.base_config import (
    DATA_PATH, CKPT_DIR, LOG_DIR, OUTPUT_DIR,
    EPOCHS, BATCH_SIZE, LR, LAMBDA_PHYSICS,
)
from models.deeponet import PhysicsNeMoDeepONet
from data.dataset_unified import build_dataloaders
from training.train import train


def parse_args():
    p = argparse.ArgumentParser(description="Train DeepONet_Unified")
    p.add_argument("--epochs",    type=int,   default=EPOCHS)
    p.add_argument("--lr",        type=float, default=LR)
    p.add_argument("--batch",     type=int,   default=BATCH_SIZE)
    p.add_argument("--lam",       type=float, default=LAMBDA_PHYSICS,
                   help="Physics loss coefficient")
    p.add_argument("--data",      type=str,   default=DATA_PATH)
    p.add_argument("--ckpt_dir",  type=str,   default=CKPT_DIR)
    p.add_argument("--log_dir",   type=str,   default=LOG_DIR)
    p.add_argument("--seed",      type=int,   default=42)
    return p.parse_args()


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  DeepONet_Unified — NVIDIA PhysicsNeMo DeepONetArch")
    print("=" * 60)
    print(f"  Device  : {device}")
    print(f"  Epochs  : {args.epochs}")
    print(f"  LR      : {args.lr}")
    print(f"  Batch   : {args.batch}")
    print(f"  Lambda  : {args.lam}")
    print(f"  Data    : {args.data}")
    print("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────
    train_loader, val_loader, test_loader, normalizer, test_keys = \
        build_dataloaders(args.data, args.batch, seed=args.seed)

    # ── Model ─────────────────────────────────────────────────────────────
    model = PhysicsNeMoDeepONet()
    print(f"\n  Model params: {model.count_parameters():,}")

    # ── Train ─────────────────────────────────────────────────────────────
    ckpt_mgr = train(
        model        = model,
        train_loader = train_loader,
        val_loader   = val_loader,
        normalizer   = normalizer,
        epochs       = args.epochs,
        lr           = args.lr,
        batch_size   = args.batch,
        ckpt_dir     = args.ckpt_dir,
        log_dir      = args.log_dir,
        device       = device,
        lambda_physics = args.lam,
    )

    # ── Final evaluation on test set ──────────────────────────────────────
    print("\n[test] Running final evaluation on test set...")
    from evaluation.evaluate import evaluate
    _, _, normalizer = ckpt_mgr.load_best()
    metrics = evaluate(model, test_loader, normalizer, device)

    print("\n[Done] Training and evaluation complete.")
    print(f"       Checkpoints saved to: {args.ckpt_dir}")


if __name__ == "__main__":
    main()

"""
Top-level training entry point.

BUGS FIXED vs old version:
  1. No --train_time_end CLI arg — users can't override temporal split from CLI
  2. No --hidden / --layers args for quick hyperparameter sweeps
  3. cfg attributes referenced that don't exist (now matched to new BaseConfig)

Usage:
    python train.py
    python train.py --epochs 300 --lr 5e-4
    python train.py --train_time_end 2400   # use 60% for training
"""

import argparse
import sys
sys.path.insert(0, ".")

from configs.base_config import BaseConfig, CONFIG
from training.train import main as _train_main


def main():
    parser = argparse.ArgumentParser(description="Train PhysicsNeMo MeshGraphNet")
    parser.add_argument("--dataset",         default=None, help="Override dataset path")
    parser.add_argument("--epochs",          type=int,   default=None)
    parser.add_argument("--lr",              type=float, default=None)
    parser.add_argument("--batch",           type=int,   default=None)
    parser.add_argument("--device",          default=None)
    parser.add_argument("--hidden",          type=int,   default=None)
    parser.add_argument("--layers",          type=int,   default=None)
    parser.add_argument("--train_time_end",  type=float, default=None,
                        help="End of training window [s] (default 3200)")
    parser.add_argument("--wandb",           action="store_true")
    args = parser.parse_args()

    cfg = CONFIG
    if args.dataset:        cfg.dataset_path          = args.dataset
    if args.epochs:         cfg.n_epochs              = args.epochs
    if args.lr:             cfg.learning_rate         = args.lr
    if args.batch:          cfg.batch_size            = args.batch
    if args.device:         cfg.device                = args.device
    if args.hidden:         cfg.hidden_features       = args.hidden
    if args.layers:         cfg.n_message_passing_layers = args.layers
    if args.train_time_end: cfg.train_time_end        = args.train_time_end
    if args.wandb:          cfg.use_wandb             = True

    _train_main(cfg)


if __name__ == "__main__":
    main()

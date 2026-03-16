"""
Top-level training entry point for the Heat Treatment GNN surrogate.

Usage:
    python train.py
    python train.py --epochs 300 --lr 5e-4
"""
import argparse
import sys
sys.path.insert(0, ".")

from configs.base_config import BaseConfig, CONFIG
from training.train import main as _train_main


def main():
    parser = argparse.ArgumentParser(description="Train MeshGraphNet surrogate")
    parser.add_argument("--dataset", default=None, help="Override dataset path")
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--batch",   type=int,   default=None)
    parser.add_argument("--device",  default=None)
    parser.add_argument("--wandb",   action="store_true")
    args = parser.parse_args()

    cfg = CONFIG
    if args.dataset: cfg.dataset_path    = args.dataset
    if args.epochs:  cfg.n_epochs        = args.epochs
    if args.lr:      cfg.learning_rate   = args.lr
    if args.batch:   cfg.batch_size      = args.batch
    if args.device:  cfg.device          = args.device
    if args.wandb:   cfg.use_wandb       = True

    _train_main(cfg)


if __name__ == "__main__":
    main()
"""
training/train.py
-----------------
Core training loop for DeepONet_Unified.
Mirrors GNN_Unified/training/train.py.
"""

import os
import csv
import torch
import numpy as np
from tqdm import tqdm

from training.loss import DeepONetLoss
from utils.metrics import compute_metrics, print_metrics
from utils.checkpoint import CheckpointManager
from configs.base_config import (
    TARGET_REGIONS, GRAD_CLIP, LOG_EVERY, SAVE_EVERY,
    LR_PATIENCE, LR_FACTOR, LR_MIN, LAMBDA_PHYSICS,
)


def train(
    model,
    train_loader,
    val_loader,
    normalizer: dict,
    epochs: int,
    lr: float,
    batch_size: int,
    ckpt_dir: str,
    log_dir: str,
    device: torch.device,
    lambda_physics: float = LAMBDA_PHYSICS,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=LR_PATIENCE, factor=LR_FACTOR, min_lr=LR_MIN
    )
    criterion = DeepONetLoss(lambda_physics=lambda_physics)
    ckpt_mgr  = CheckpointManager(ckpt_dir, model, optimizer, scheduler)

    log_path = os.path.join(log_dir, "training_log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"] +
                        [f"rel_l2_{r}" for r in TARGET_REGIONS])

    print(f"\nStarting training — {epochs} epochs")
    print(f"{'Epoch':>6} {'Train':>10} {'Val':>10} {'LR':>10}")
    print("-" * 50)

    for epoch in range(1, epochs + 1):

        # ── Train ─────────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for a_b, x_b, u_b in train_loader:
            a_b, x_b, u_b = a_b.to(device), x_b.to(device), u_b.to(device)
            optimizer.zero_grad()
            pred = model(a_b, x_b)
            loss = criterion(pred, u_b, x_b, a_b, normalizer, step=epoch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))

        # ── Validate ──────────────────────────────────────────────────────
        model.eval()
        val_losses, preds_all, trues_all, ridx_all = [], [], [], []
        with torch.no_grad():
            for a_b, x_b, u_b in val_loader:
                a_b, x_b, u_b = a_b.to(device), x_b.to(device), u_b.to(device)
                pred = model(a_b, x_b)
                loss = criterion(pred, u_b, x_b, a_b, normalizer)
                val_losses.append(loss.item())

                # Denormalize for metrics
                pred_K = pred.cpu().numpy() * normalizer["u_std"] + normalizer["u_mean"]
                true_K = u_b.cpu().numpy()  * normalizer["u_std"] + normalizer["u_mean"]
                preds_all.append(pred_K.flatten())
                trues_all.append(true_K.flatten())
                ridx_all.append(x_b[:, 4:8].argmax(dim=1).cpu().numpy())

        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        lr_cur = optimizer.param_groups[0]["lr"]

        # Per-region Rel-L2
        preds_K = np.concatenate(preds_all)
        trues_K = np.concatenate(trues_all)
        ridx    = np.concatenate(ridx_all)
        metrics = compute_metrics(preds_K, trues_K, ridx)
        rel_l2s = {m["name"]: m["RelL2"] for m in metrics["per_region"]}

        print(f"{epoch:>6d} {train_loss:>10.6f} {val_loss:>10.6f} {lr_cur:>10.2e}  "
              + "  ".join(f"{r}: {rel_l2s.get(r, -1):.4f}" for r in TARGET_REGIONS))

        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, train_loss, val_loss, lr_cur] +
                             [rel_l2s.get(r, -1) for r in TARGET_REGIONS])

        if epoch % SAVE_EVERY == 0:
            is_best = ckpt_mgr.save(epoch, val_loss, normalizer)
            if is_best:
                print(f"  ✓ Best model saved (val_loss={val_loss:.6f})")

    # Final save
    ckpt_mgr.save(epochs, val_loss, normalizer, tag="_final")
    print(f"\nTraining complete. Best val loss: {ckpt_mgr.best_loss:.6f}")
    return ckpt_mgr

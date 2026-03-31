"""
Comprehensive Hyperparameter Finder v3 for 3D FNO
Master's Thesis: Simulating Heat Treatment using OpenFOAM and AI

Sequential search: best from each stage feeds into the next.
Final combo validation: top 5 combinations tested with 5 epochs.
Total runtime: ~2.5-3 hours on A40.
"""
import sys, math, json, time, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

from configs.fno_config import CONFIG, FNOConfig
from data.dataset import FNO3DDataset
from models.fno_model import HeatTreatmentFNO3D
from utils.metrics import compute_metrics
from torch.utils.data import DataLoader


def quick_train(train_loader, val_loader, T_mean, T_std, cfg,
                lr, weight_decay, physics_lam, n_epochs=3, device="cuda"):
    """Train for a few epochs and return metrics."""
    model = HeatTreatmentFNO3D(cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_steel = float("inf")
    best_mae = float("inf")
    train_losses = []
    val_losses = []
    final_m = {}

    for epoch in range(1, n_epochs + 1):
        model.train()
        total_loss, nb = 0.0, 0
        for batch in train_loader:
            x, y, T_cur, T_next_gt, weight = [
                b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
            optimizer.zero_grad()
            pred = model(x)
            loss_data = F.mse_loss(pred, y)

            if physics_lam > 1e-8:
                T_pred_norm = pred.squeeze(1)
                Tset_norm = x[:, 1]
                is_heater = x[:, 4]
                non_heater = (1.0 - is_heater)
                overshoot = F.relu(T_pred_norm - Tset_norm) * non_heater
                L_phys = overshoot.pow(2).mean()
                loss = (1 - physics_lam) * loss_data + physics_lam * L_phys
            else:
                loss = loss_data

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss_data.item()
            nb += 1
        train_losses.append(total_loss / max(nb, 1))

        model.eval()
        all_pred, all_true = [], []
        steel_pred, steel_true = [], []
        val_loss_total, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x, y, T_cur, T_next_gt, weight = [
                    b.to(device) if isinstance(b, torch.Tensor) else b for b in batch]
                pred = model(x)
                vloss = F.mse_loss(pred, y)
                val_loss_total += vloss.item()
                vn += 1
                T_pred_K = (pred.squeeze(1).cpu().numpy() * T_std + T_mean).ravel()
                T_true_K = T_next_gt.cpu().numpy().ravel()
                all_pred.append(T_pred_K)
                all_true.append(T_true_K)
                rid = x[:, 2].cpu().numpy()
                is_steel = (rid < 0.05)
                if is_steel.any():
                    steel_pred.append((pred.squeeze(1).cpu().numpy() * T_std + T_mean)[is_steel])
                    steel_true.append(T_next_gt.cpu().numpy()[is_steel])

        val_loss = val_loss_total / max(vn, 1)
        val_losses.append(val_loss)
        final_m = compute_metrics(np.concatenate(all_pred), np.concatenate(all_true))
        steel_mae = float(np.mean(np.abs(np.concatenate(steel_pred) - np.concatenate(steel_true)))) if steel_pred else 999.0

        if val_loss < best_val: best_val = val_loss
        if steel_mae < best_steel: best_steel = steel_mae
        if final_m["mae"] < best_mae: best_mae = final_m["mae"]

    del model
    torch.cuda.empty_cache()

    return {
        "best_val_loss": best_val,
        "best_steel_mae": best_steel,
        "best_mae": best_mae,
        "final_r2": final_m.get("r2", 0),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "converging": val_losses[-1] < val_losses[0],
    }


def build_loaders(cfg, batch_size):
    kw = dict(num_workers=2, pin_memory=True)
    train_ds = FNO3DDataset(cfg.dataset_path, cfg, "train", "training")
    val_ds = FNO3DDataset(cfg.dataset_path, cfg, "val", "training")
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=True, **kw),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, **kw),
            train_ds.T_mean, train_ds.T_std)


def print_header():
    print(f"  {'Config':>14} | {'ValLoss':>10} | {'SteelMAE':>8} | {'MAE':>7} | {'R2':>8} | {'Conv':>4} | {'Time':>5}")
    print(f"  {'-'*72}")


def print_row(label, m, dt):
    conv = "Yes" if m["converging"] else "No"
    print(f"  {label:>14} | {m['best_val_loss']:>10.6f} | {m['best_steel_mae']:>8.2f}K | "
          f"{m['best_mae']:>7.2f}K | {m['final_r2']:>8.4f} | {conv:>4} | {dt:>4.0f}s")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = CONFIG
    t_start = time.time()

    sep = "=" * 75
    print(f"\n{sep}")
    print(f"  COMPREHENSIVE HYPERPARAMETER FINDER v3 — 3D FNO")
    print(f"  Device: {device}")
    print(f"  3 epochs per config | Sequential search + final combo test")
    print(f"{sep}\n")

    all_results = {}

    # Build dataset ONCE
    print("  Loading dataset...")
    t0 = time.time()
    train_loader_8, val_loader_8, T_mean, T_std = build_loaders(cfg, 8)
    print(f"  Done in {time.time()-t0:.0f}s\n")

    # ─── 1. Learning Rate ──────────────────────────────────────
    print("  ══════ 1. LEARNING RATE ══════")
    print_header()
    lr_results = {}
    for lr in [2e-3, 1e-3, 7e-4, 5e-4, 3e-4, 1e-4, 5e-5, 1e-5]:
        t0 = time.time()
        m = quick_train(train_loader_8, val_loader_8, T_mean, T_std, cfg,
                        lr=lr, weight_decay=1e-5, physics_lam=0.0005, device=str(device))
        lr_results[str(lr)] = m
        print_row(f"lr={lr:.0e}", m, time.time()-t0)
    best_lr = float(min(lr_results, key=lambda k: lr_results[k]["best_val_loss"]))
    all_results["best_lr"] = best_lr
    print(f"\n  >>> Best LR: {best_lr:.1e}\n")

    # ─── 2. Batch Size ─────────────────────────────────────────
    print("  ══════ 2. BATCH SIZE ══════")
    print_header()
    batch_results = {}
    for bs in [2, 4, 8, 16]:
        try:
            t0 = time.time()
            tl, vl = (train_loader_8, val_loader_8) if bs == 8 else build_loaders(cfg, bs)[:2]
            m = quick_train(tl, vl, T_mean, T_std, cfg,
                            lr=best_lr, weight_decay=1e-5, physics_lam=0.0005, device=str(device))
            batch_results[str(bs)] = m
            print_row(f"batch={bs}", m, time.time()-t0)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  {'batch='+str(bs):>14} | OOM — skipped")
                torch.cuda.empty_cache()
            else:
                raise
    best_batch = int(min(batch_results, key=lambda k: batch_results[k]["best_val_loss"]))
    all_results["best_batch"] = best_batch
    print(f"\n  >>> Best batch: {best_batch}\n")

    if best_batch != 8:
        tl_best, vl_best, _, _ = build_loaders(cfg, best_batch)
    else:
        tl_best, vl_best = train_loader_8, val_loader_8

    # ─── 3. Weight Decay ───────────────────────────────────────
    print("  ══════ 3. WEIGHT DECAY ══════")
    print_header()
    wd_results = {}
    for wd in [0.0, 1e-6, 1e-5, 1e-4, 1e-3]:
        t0 = time.time()
        m = quick_train(tl_best, vl_best, T_mean, T_std, cfg,
                        lr=best_lr, weight_decay=wd, physics_lam=0.0005, device=str(device))
        wd_results[str(wd)] = m
        print_row(f"wd={wd:.0e}", m, time.time()-t0)
    best_wd = float(min(wd_results, key=lambda k: wd_results[k]["best_val_loss"]))
    all_results["best_weight_decay"] = best_wd
    print(f"\n  >>> Best weight decay: {best_wd:.1e}\n")

    # ─── 4. Physics Lambda ─────────────────────────────────────
    print("  ══════ 4. PHYSICS LAMBDA ══════")
    print_header()
    lam_results = {}
    for lam in [0.0, 0.0001, 0.0003, 0.0005, 0.001, 0.003, 0.01]:
        t0 = time.time()
        m = quick_train(tl_best, vl_best, T_mean, T_std, cfg,
                        lr=best_lr, weight_decay=best_wd, physics_lam=lam, device=str(device))
        lam_results[str(lam)] = m
        print_row(f"lam={lam:.4f}", m, time.time()-t0)
    best_lam = float(min(lam_results, key=lambda k: lam_results[k]["best_val_loss"]))
    all_results["best_physics_lambda"] = best_lam
    print(f"\n  >>> Best physics lambda: {best_lam}\n")

    # ─── 5. FNO Layers ─────────────────────────────────────────
    print("  ══════ 5. FNO LAYERS ══════")
    print_header()
    layer_results = {}
    for nl in [2, 3, 4, 5]:
        t0 = time.time()
        cfg_t = FNOConfig()
        cfg_t.fno_layers = nl
        cfg_t.batch_size = best_batch
        m = quick_train(tl_best, vl_best, T_mean, T_std, cfg_t,
                        lr=best_lr, weight_decay=best_wd, physics_lam=best_lam, device=str(device))
        layer_results[str(nl)] = m
        print_row(f"layers={nl}", m, time.time()-t0)
    best_layers = int(min(layer_results, key=lambda k: layer_results[k]["best_val_loss"]))
    all_results["best_layers"] = best_layers
    print(f"\n  >>> Best layers: {best_layers}\n")

    # ─── 6. Latent Dimension ───────────────────────────────────
    print("  ══════ 6. LATENT DIMENSION ══════")
    print_header()
    latent_results = {}
    for lat in [16, 32, 48, 64, 96]:
        try:
            t0 = time.time()
            cfg_t = FNOConfig()
            cfg_t.fno_layers = best_layers
            cfg_t.fno_latent = lat
            cfg_t.fno_decoder_layer_size = lat
            cfg_t.batch_size = best_batch
            m = quick_train(tl_best, vl_best, T_mean, T_std, cfg_t,
                            lr=best_lr, weight_decay=best_wd, physics_lam=best_lam, device=str(device))
            latent_results[str(lat)] = m
            print_row(f"latent={lat}", m, time.time()-t0)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  {'latent='+str(lat):>14} | OOM — skipped")
                torch.cuda.empty_cache()
            else:
                raise
    best_latent = int(min(latent_results, key=lambda k: latent_results[k]["best_val_loss"]))
    all_results["best_latent"] = best_latent
    print(f"\n  >>> Best latent: {best_latent}\n")

    # ─── 7. FINAL COMBO VALIDATION (5 epochs) ─────────────────
    print("  ══════ 7. FINAL COMBO VALIDATION (5 epochs) ══════")
    print_header()

    # Top combo from sequential search
    combos = [
        {"name": "Sequential best",
         "lr": best_lr, "batch": best_batch, "wd": best_wd,
         "lam": best_lam, "layers": best_layers, "latent": best_latent},
        # Slightly different LR
        {"name": "LR x0.5",
         "lr": best_lr * 0.5, "batch": best_batch, "wd": best_wd,
         "lam": best_lam, "layers": best_layers, "latent": best_latent},
        # Slightly different LR
        {"name": "LR x2",
         "lr": best_lr * 2, "batch": best_batch, "wd": best_wd,
         "lam": best_lam, "layers": best_layers, "latent": best_latent},
        # Bigger model
        {"name": "Bigger model",
         "lr": best_lr, "batch": best_batch, "wd": best_wd,
         "lam": best_lam, "layers": min(best_layers+1, 5),
         "latent": min(best_latent+16, 96)},
        # No physics
        {"name": "No physics",
         "lr": best_lr, "batch": best_batch, "wd": best_wd,
         "lam": 0.0, "layers": best_layers, "latent": best_latent},
    ]

    combo_results = {}
    for c in combos:
        try:
            t0 = time.time()
            cfg_t = FNOConfig()
            cfg_t.fno_layers = c["layers"]
            cfg_t.fno_latent = c["latent"]
            cfg_t.fno_decoder_layer_size = c["latent"]
            cfg_t.batch_size = c["batch"]
            if c["batch"] != best_batch:
                tl_c, vl_c, _, _ = build_loaders(cfg_t, c["batch"])
            else:
                tl_c, vl_c = tl_best, vl_best
            m = quick_train(tl_c, vl_c, T_mean, T_std, cfg_t,
                            lr=c["lr"], weight_decay=c["wd"], physics_lam=c["lam"],
                            n_epochs=5, device=str(device))
            combo_results[c["name"]] = {**m, **c}
            print_row(c["name"], m, time.time()-t0)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  {c['name']:>14} | OOM — skipped")
                torch.cuda.empty_cache()
            else:
                raise

    best_combo_name = min(combo_results, key=lambda k: combo_results[k]["best_val_loss"])
    best_combo = combo_results[best_combo_name]
    all_results["combo_validation"] = {k: {kk: vv for kk, vv in v.items()
        if kk not in ("train_losses", "val_losses")} for k, v in combo_results.items()}
    all_results["best_combo"] = best_combo_name

    # ─── Final Summary ─────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n{sep}")
    print(f"  FINAL RECOMMENDED HYPERPARAMETERS")
    print(f"  (Best combo: {best_combo_name})")
    print(f"{sep}")
    print(f"  Learning rate:      {best_combo.get('lr', best_lr):.1e}")
    print(f"  Batch size:         {best_combo.get('batch', best_batch)}")
    print(f"  Weight decay:       {best_combo.get('wd', best_wd):.1e}")
    print(f"  Physics lambda:     {best_combo.get('lam', best_lam)}")
    print(f"  FNO layers:         {best_combo.get('layers', best_layers)}")
    print(f"  Latent dimension:   {best_combo.get('latent', best_latent)}")
    print(f"")
    print(f"  Steel MAE:          {best_combo['best_steel_mae']:.2f}K")
    print(f"  Overall MAE:        {best_combo['best_mae']:.2f}K")
    print(f"  R²:                 {best_combo['final_r2']:.4f}")
    print(f"")
    print(f"  Total search time:  {total_time/60:.1f} min")
    print(f"")
    print(f"  === COMMANDS FOR FINAL TRAINING ===")
    bl = best_combo.get('layers', best_layers)
    blt = best_combo.get('latent', best_latent)
    blr = best_combo.get('lr', best_lr)
    bbs = best_combo.get('batch', best_batch)
    bwd = best_combo.get('wd', best_wd)
    blam = best_combo.get('lam', best_lam)
    print(f"  # 1. Edit fno_config.py:")
    print(f"  sed -i 's/fno_layers:.*int = .*/fno_layers:       int = {bl}/' configs/fno_config.py")
    print(f"  sed -i 's/fno_latent:.*int = .*/fno_latent:       int = {blt}/' configs/fno_config.py")
    print(f"  sed -i 's/fno_decoder_layer_size:.*int = .*/fno_decoder_layer_size: int = {blt}/' configs/fno_config.py")
    print(f"  #")
    print(f"  # 2. Edit train.py physics lambda:")
    print(f"  # Change get_physics_lambda to: return {blam}")
    print(f"  #")
    print(f"  # 3. SLURM command:")
    print(f"  python -u train.py --epochs 150 --lr {blr:.1e} --batch {bbs}")
    print(f"  # Add weight_decay={bwd:.1e} in train.py optimizer line")
    print(f"{sep}\n")

    # Save
    out_path = "outputs/hp_finder_v3_results.json"
    Path("outputs").mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()

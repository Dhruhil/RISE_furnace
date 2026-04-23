"""PI-DeepONet training loop — autograd physics, matched to FNO/GNN."""
from __future__ import annotations
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, ".")
from configs.deeponet_config import CONFIG
from data.dataset import get_deeponet_dataloaders
from models.deeponet_model import HeatTreatmentDeepONet
from training.scheduler import build_scheduler
from training.loss import DeepONetLoss
from utils.checkpoint import CheckpointManager
from utils.metrics import compute_metrics
from utils.logging import setup_logging


def get_physics_lambda(epoch, n_epochs):
    p = min(1.0, epoch / max(n_epochs, 1))
    cosine = 0.5 * (1 + math.cos(math.pi * p))
    return 0.0005 + (0.003 - 0.0005) * cosine


def get_pushforward_weight(epoch, n_epochs):
    warmup_end = int(n_epochs * 0.10)
    if epoch <= warmup_end: return 0.0
    return 1.0 * (epoch - warmup_end) / (n_epochs - warmup_end)


def get_warmup_lr(epoch, base_lr, warmup_epochs=5):
    if epoch <= warmup_epochs:
        return base_lr * (0.1 + 0.9 * epoch / warmup_epochs)
    return base_lr


def _build_trunk_with_grad(xyz, region_id, is_heater, kappa, Cp, rho):
    static = torch.stack([
        region_id, is_heater,
        kappa / 100.0, Cp / 1000.0, rho / 10000.0,
    ], dim=-1)
    return torch.cat([xyz, static], dim=-1)


def train_one_epoch(model, loader, optimizer, criterion, device,
                    T_mean, T_std, grad_clip, w2, lam, dt=10.0,
                    noise_std=0.03, t_total=3460.0):
    model.train()
    totals = {"loss":0,"data":0,"phys":0,"cond":0,"conv":0,"rad":0,"overshoot":0,"pf":0}
    n = 0
    for batch_ in loader:
        (branch, scalars, _trunk_old, y, T_cur_K, T_next_gt, w,
         xyz, rid, is_heat, kappa, Cp, rho) = [
            b.to(device, non_blocking=True) if isinstance(b, torch.Tensor) else b
            for b in batch_]
        if noise_std > 0:
            branch = branch.clone()
            branch[:,0:1,:] = branch[:,0:1,:] + torch.randn_like(branch[:,0:1,:])*noise_std
        use_phys = lam > 1e-6
        xyz_grad = xyz.clone().detach().requires_grad_(use_phys)
        trunk = _build_trunk_with_grad(xyz_grad, rid, is_heat, kappa, Cp, rho)

        optimizer.zero_grad()
        pred1 = model(branch, scalars, trunk)
        pred_next = None
        if use_phys:
            scalars_next = scalars.clone()
            scalars_next[:,1] = scalars_next[:,1] + dt / t_total
            pred_next = model(branch, scalars_next, trunk)

        T_set_K = scalars[:,0] * T_std + T_mean
        loss1, parts = criterion(
            pred_norm=pred1, target_norm=y, weight=w,
            T_set=T_set_K, T_mean=T_mean, T_std=T_std,
            pred_next_norm=pred_next,
            xyz=xyz_grad if use_phys else None,
            T_cur_K=T_cur_K if use_phys else None,
            region_id=rid if use_phys else None,
            is_heater=is_heat if use_phys else None,
            kappa=kappa if use_phys else None,
            Cp=Cp if use_phys else None,
            rho=rho if use_phys else None,
            dt=dt,
        )

        loss = loss1
        pf_val = 0.0
        if w2 > 1e-6:
            with torch.no_grad():
                shift = pred1.mean(dim=1, keepdim=True).unsqueeze(1)
            branch_pf = branch.clone()
            branch_pf[:,0:1,:] = branch_pf[:,0:1,:] + shift
            trunk_pf = _build_trunk_with_grad(
                xyz_grad.detach(), rid, is_heat, kappa, Cp, rho)
            pred_pf = model(branch_pf, scalars, trunk_pf)
            loss2 = ((pred_pf - y).pow(2) * w).sum() / (w.sum() + 1e-8)
            loss = loss1 + w2 * loss2
            pf_val = float(loss2.detach())

        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        totals["loss"] += float(loss.detach())
        totals["data"] += parts["data"]
        totals["phys"] += parts.get("physics",0.0)
        totals["cond"] += parts.get("cond",0.0)
        totals["conv"] += parts.get("conv",0.0)
        totals["rad"]  += parts.get("rad",0.0)
        totals["overshoot"] += parts.get("overshoot",0.0)
        totals["pf"]   += pf_val
        n += 1
    for k in totals: totals[k] /= max(n,1)
    return totals


@torch.no_grad()
def evaluate(model, loader, criterion, device, T_mean, T_std):
    model.eval()
    losses, maes, r2s = [], [], []
    for batch_ in loader:
        (branch, scalars, trunk, y, _T_cur, T_gt, w, *_rest) = [
            b.to(device) if isinstance(b, torch.Tensor) else b for b in batch_]
        pred = model(branch, scalars, trunk)
        T_set_K = scalars[:,0] * T_std + T_mean
        loss, _ = criterion(
            pred_norm=pred, target_norm=y, weight=w,
            T_set=T_set_K, T_mean=T_mean, T_std=T_std)
        losses.append(float(loss))
        pred_K = pred.cpu().numpy() * T_std + T_mean
        true_K = T_gt.cpu().numpy()
        m = compute_metrics(pred_K.reshape(-1), true_K.reshape(-1))
        maes.append(m["mae"]); r2s.append(m["r2"])
    return float(np.mean(losses)), float(np.mean(maes)), float(np.mean(r2s))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lam", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--no_pushforward", action="store_true")
    parser.add_argument("--no_physics", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    cfg = CONFIG
    if args.epochs: cfg.n_epochs = args.epochs
    if args.lr: cfg.learning_rate = args.lr
    if args.batch: cfg.batch_size = args.batch
    LAM_OVERRIDE = args.lam

    device = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu"))

    sep = "=" * 72
    print(f"\n{sep}\n  PI-DeepONet — Heat Treatment (autograd physics)\n{sep}")
    print(f"  Device:  {device}   Batch: {cfg.batch_size}   Epochs: {cfg.n_epochs}")
    print(f"  Sensors: {cfg.sensor_grid_x}x{cfg.sensor_grid_y}x{cfg.sensor_grid_z} = {cfg.n_sensors}")
    print(f"  Query points/sample: {cfg.n_query_points}")
    print(f"  Latent dim: {cfg.latent_dim}")
    if args.no_physics:
        print(f"  Physics: OFF (--no_physics)")
    elif LAM_OVERRIDE is not None:
        print(f"  Physics: FIXED lam={LAM_OVERRIDE}")
    else:
        print(f"  Physics: cosine 0.003 -> 0.0005 (autograd Laplacian + Newton + SB)")
    print(f"  Pushforward: " + ("OFF" if args.no_pushforward else "w2 ramp 0 -> 1.0 (10% warmup)"))
    print(f"  LR warmup: linear 5 epochs")
    print(f"  Dataset: {cfg.dataset_path}\n{sep}\n")

    logger = setup_logging(cfg)
    train_loader, val_loader, train_ds, _ = get_deeponet_dataloaders(cfg)
    T_mean, T_std = train_ds.T_mean, train_ds.T_std
    print(f"  Stats: T_mean={T_mean:.1f} K   T_std={T_std:.1f} K\n")

    model = HeatTreatmentDeepONet(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = DeepONetLoss(lambda_physics=cfg.lambda_physics)
    ckpt_mgr = CheckpointManager(cfg.checkpoint_dir)

    if args.test:
        print("=== SANITY TEST ===")
        batch_ = next(iter(train_loader))
        (br, sc, tr, y, Tc, Tn, w, xyz, rid, ish, k, cp, rho) = [
            b.to(device) if isinstance(b, torch.Tensor) else b for b in batch_]
        print(f"  branch: {tuple(br.shape)}  scalars: {tuple(sc.shape)}  trunk: {tuple(tr.shape)}")
        print(f"  xyz: {tuple(xyz.shape)}  (raw metres, range=[{xyz.min():.3f},{xyz.max():.3f}])")
        print(f"  kappa range=[{k.min():.1f},{k.max():.1f}]  Cp=[{cp.min():.1f},{cp.max():.1f}]  rho=[{rho.min():.1f},{rho.max():.1f}]")
        xyz_g = xyz.clone().detach().requires_grad_(True)
        trunk = _build_trunk_with_grad(xyz_g, rid, ish, k, cp, rho)
        pred1 = model(br, sc, trunk)
        sc_next = sc.clone(); sc_next[:,1] += 10.0/3460.0
        pred_next = model(br, sc_next, trunk)
        print(f"  pred1: {tuple(pred1.shape)}  range=[{pred1.min().item():.3f},{pred1.max().item():.3f}]")
        T_set_K = sc[:,0] * T_std + T_mean
        loss, parts = criterion(
            pred_norm=pred1, target_norm=y, weight=w,
            T_set=T_set_K, T_mean=T_mean, T_std=T_std,
            pred_next_norm=pred_next, xyz=xyz_g, T_cur_K=Tc,
            region_id=rid, is_heater=ish, kappa=k, Cp=cp, rho=rho, dt=10.0)
        print(f"\n  Loss: {float(loss):.4f}  Data: {parts['data']:.4f}  Physics: {parts.get('physics',0):.4f}")
        print(f"    cond={parts.get('cond',0):.4f}  conv={parts.get('conv',0):.4f}  "
              f"rad={parts.get('rad',0):.4f}  overshoot={parts.get('overshoot',0):.4f}")
        loss.backward()
        print(f"\n  Backward pass: OK")
        print(f"  xyz.grad range: [{xyz_g.grad.min().item():.3e},{xyz_g.grad.max().item():.3e}]")
        print("\n  Schedule preview (100 epochs):")
        for ep in [1, 10, 11, 50, 100]:
            print(f"    ep {ep:>3}: w2={get_pushforward_weight(ep,100):.3f}  "
                  f"lam={get_physics_lambda(ep,100):.4f}  lr={get_warmup_lr(ep,1.0):.3f}")
        print("=== OK ===")
        return

    history = {"train_loss":[],"val_loss":[],"val_mae":[],"val_r2":[],
               "lr":[],"w2":[],"lam":[],"pf_loss":[],
               "tr_data":[],"tr_phys":[],"L_cond":[],"L_conv":[],"L_rad":[]}

    print(f"  {'Ep':>4} | {'TrLoss':>9} | {'TrData':>9} | {'Cond':>8} | {'Conv':>8} | "
          f"{'Rad':>8} | {'VaLoss':>9} | {'MAE[K]':>8} | {'R2':>7} | "
          f"{'lam':>7} | {'w2':>6} | {'LR':>9} | {'t[s]':>6}")
    print("  " + "-" * 130)

    best_mae = float("inf")
    t0 = time.time()
    for epoch in range(1, cfg.n_epochs + 1):
        if args.no_physics: lam = 0.0
        elif LAM_OVERRIDE is not None: lam = LAM_OVERRIDE
        else: lam = get_physics_lambda(epoch, cfg.n_epochs)
        w2 = 0.0 if args.no_pushforward else get_pushforward_weight(epoch, cfg.n_epochs)
        lr = get_warmup_lr(epoch, cfg.learning_rate, warmup_epochs=max(5, cfg.n_epochs//10))
        for pg in optimizer.param_groups: pg["lr"] = lr
        criterion.lambda_physics = lam

        t_ep = time.time()
        tr = train_one_epoch(model, train_loader, optimizer, criterion, device,
                             T_mean, T_std, cfg.grad_clip, w2, lam,
                             dt=cfg.dt, t_total=cfg.t_total)
        val_loss, val_mae, val_r2 = evaluate(model, val_loader, criterion, device, T_mean, T_std)
        if epoch > 5: scheduler.step(val_loss)
        dt_ep = time.time() - t_ep
        lr_now = optimizer.param_groups[0]["lr"]

        for k, v in [("train_loss",tr["loss"]),("tr_data",tr["data"]),
                     ("tr_phys",tr["phys"]),("L_cond",tr["cond"]),
                     ("L_conv",tr["conv"]),("L_rad",tr["rad"]),
                     ("val_loss",val_loss),("val_mae",val_mae),
                     ("val_r2",val_r2),("lr",lr_now),("w2",w2),
                     ("lam",lam),("pf_loss",tr["pf"])]:
            history[k].append(v)

        print(f"  {epoch:>4d} | {tr['loss']:>9.4f} | {tr['data']:>9.4f} | "
              f"{tr['cond']:>8.4f} | {tr['conv']:>8.4f} | {tr['rad']:>8.4f} | "
              f"{val_loss:>9.4f} | {val_mae:>8.3f} | {val_r2:>7.4f} | "
              f"{lam:>7.4f} | {w2:>6.3f} | {lr_now:>9.2e} | {dt_ep:>6.1f}")

        if val_mae < best_mae:
            best_mae = val_mae
            ckpt_mgr.save_best(model, optimizer, scheduler, epoch, {"mae":val_mae,"r2":val_r2})
        if epoch % cfg.save_every_n_epochs == 0:
            ckpt_mgr.save_epoch(model, optimizer, scheduler, epoch, {"mae":val_mae,"r2":val_r2})

    total = time.time() - t0
    with open(f"{cfg.output_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n  Training done in {total/60:.1f} min ({total/3600:.2f} hrs)")
    print(f"  Best val MAE: {best_mae:.3f} K")


if __name__ == "__main__":
    main()

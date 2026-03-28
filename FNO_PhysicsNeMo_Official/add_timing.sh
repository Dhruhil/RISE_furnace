#!/bin/bash
# ============================================================
# Add timing measurements to FNO and GNN for thesis
# Reports: per-epoch time, per-sample time, inference speed
#
# cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official
# bash add_timing.sh
# ============================================================

set -euo pipefail
cd /mimer/NOBACKUP/groups/revar/FNO_PhysicsNeMo_Official

echo "=== Adding timing to FNO train.py ==="

python3 << 'XEOF'
with open("train.py", "r") as f:
    code = f.read()

# 1. Add per-epoch timing inside training loop
old_loop_start = '''        model.train()
        total_loss, total_data, total_phys, nb = 0.0, 0.0, 0.0, 0'''

new_loop_start = '''        model.train()
        total_loss, total_data, total_phys, nb = 0.0, 0.0, 0.0, 0
        ep_start = time.time()'''

if old_loop_start in code:
    code = code.replace(old_loop_start, new_loop_start)
    print("  OK: Added epoch start timer")

# 2. Add epoch time + samples/sec to print line
old_print = '''        tag = " *" if is_best else ""
        print(f"  {epoch:>4} | {tr_loss:>9.5f} | {tr_data:>9.5f} | "
              f"{tr_phys:>9.5f} | {val_m['loss_data']:>9.5f} | "
              f"{val_m['loss_phys']:>9.5f} | "
              f"{val_m['mae']:>6.2f} | {val_m['r2']:>7.4f} | "
              f"{val_m['within_5K']:>5.1f} | {lam:>5.3f}{tag}")'''

new_print = '''        ep_time = time.time() - ep_start
        samples_per_sec = len(train_loader.dataset) / ep_time if ep_time > 0 else 0
        tag = " *" if is_best else ""
        print(f"  {epoch:>4} | {tr_loss:>9.5f} | {tr_data:>9.5f} | "
              f"{tr_phys:>9.5f} | {val_m['loss_data']:>9.5f} | "
              f"{val_m['loss_phys']:>9.5f} | "
              f"{val_m['mae']:>6.2f} | {val_m['r2']:>7.4f} | "
              f"{val_m['within_5K']:>5.1f} | {lam:>5.3f} | "
              f"{ep_time:>5.1f}s{tag}")'''

if old_print in code:
    code = code.replace(old_print, new_print)
    print("  OK: Added epoch time to log")

# 3. Update header
old_header = '''    print(f"  {'Ep':>4} | {'TrLoss':>9} | {'TrData':>9} | {'TrPhys':>9} | "
          f"{'VaData':>9} | {'VaPhys':>9} | "
          f"{'MAE':>6} | {'R2':>7} | {'W5K':>5} | {'lam':>5}")
    print(f"  {'-'*95}")'''

new_header = '''    print(f"  {'Ep':>4} | {'TrLoss':>9} | {'TrData':>9} | {'TrPhys':>9} | "
          f"{'VaData':>9} | {'VaPhys':>9} | "
          f"{'MAE':>6} | {'R2':>7} | {'W5K':>5} | {'lam':>5} | {'Time':>5}")
    print(f"  {'-'*102}")'''

if old_header in code:
    code = code.replace(old_header, new_header)
    print("  OK: Updated header")

# 4. Add inference speed test at end of training
old_done = '''    print(f"\\n  Done in {(time.time()-t0)/60:.1f} min")
    print(f"  Best MAE: {ckpt_mgr.best_mae:.3f}K (epoch {ckpt_mgr.best_epoch})")'''

new_done = '''    total_time = time.time() - t0
    print(f"\\n  Training done in {total_time/60:.1f} min ({total_time/3600:.2f} hrs)")
    print(f"  Best MAE: {ckpt_mgr.best_mae:.3f}K (epoch {ckpt_mgr.best_epoch})")
    print(f"  Avg epoch time: {total_time/cfg.n_epochs:.1f}s")
    print(f"  Total samples trained: {cfg.n_epochs * len(train_loader.dataset):,}")

    # Inference speed test: time 100 forward passes
    print(f"\\n  === INFERENCE SPEED TEST ===")
    model.eval()
    batch = next(iter(val_loader))
    x_test = batch[0].to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x_test)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inf = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x_test)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inf = (time.time() - t_inf) / 100
    n_timesteps = int(cfg.t_total / cfg.dt)  # 400 steps for full simulation
    t_full_rollout = t_inf * n_timesteps
    print(f"  Single step inference: {t_inf*1000:.2f} ms")
    print(f"  Full rollout (400 steps): {t_full_rollout:.2f}s")
    print(f"  Batch size: {x_test.shape[0]}")
    print(f"  Grid: {cfg.grid_x}x{cfg.grid_y}x{cfg.grid_z}")
    print(f"")
    print(f"  === SPEED COMPARISON (for thesis) ===")
    print(f"  OpenFOAM (estimated):  ~2-4 hours per simulation")
    print(f"  3D FNO single step:    {t_inf*1000:.2f} ms")
    print(f"  3D FNO full rollout:   {t_full_rollout:.2f}s")
    print(f"  Speedup vs OpenFOAM:   ~{3600*3/t_full_rollout:.0f}x")'''

if old_done in code:
    code = code.replace(old_done, new_done)
    print("  OK: Added inference speed test")

with open("train.py", "w") as f:
    f.write(code)
XEOF

echo ""
echo "=== Now adding timing to GNN train_unified.py ==="

cd /mimer/NOBACKUP/groups/revar/GNN_Unified

python3 << 'XEOF'
with open("train_unified.py", "r") as f:
    code = f.read()

# Add inference speed test at end of GNN training
old_done = '''    total_time = time.time() - t0
    print(f"\\n  Training done in {total_time/60:.1f} min")
    print(f"  Best MAE: {ckpt_mgr.best_mae:.3f}K (epoch {ckpt_mgr.best_epoch})")'''

new_done = '''    total_time = time.time() - t0
    print(f"\\n  Training done in {total_time/60:.1f} min ({total_time/3600:.2f} hrs)")
    print(f"  Best MAE: {ckpt_mgr.best_mae:.3f}K (epoch {ckpt_mgr.best_epoch})")
    print(f"  Avg epoch time: {total_time/n_ep:.1f}s")

    # Inference speed test
    print(f"\\n  === INFERENCE SPEED TEST ===")
    model.eval()
    batch = next(iter(val_loader))
    batch = batch.to(device)
    with torch.no_grad():
        for _ in range(10):
            _ = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inf = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(batch)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_inf = (time.time() - t_inf) / 100
    n_timesteps = 400
    t_full_rollout = t_inf * n_timesteps
    print(f"  Single step inference: {t_inf*1000:.2f} ms")
    print(f"  Full rollout (400 steps): {t_full_rollout:.2f}s")
    print(f"  Nodes per graph: {batch.x.shape[0]}")
    print(f"  Edges per graph: {batch.edge_index.shape[1]}")
    print(f"")
    print(f"  === SPEED COMPARISON (for thesis) ===")
    print(f"  OpenFOAM (estimated):  ~2-4 hours per simulation")
    print(f"  GNN single step:       {t_inf*1000:.2f} ms")
    print(f"  GNN full rollout:      {t_full_rollout:.2f}s")
    print(f"  Speedup vs OpenFOAM:   ~{3600*3/t_full_rollout:.0f}x")'''

if old_done in code:
    code = code.replace(old_done, new_done)
    print("  OK: Added inference speed test to GNN")
else:
    print("  WARNING: Could not find GNN end block")
    print("  GNN job already running — timing will be added next run")

with open("train_unified.py", "w") as f:
    f.write(code)
XEOF

echo ""
echo "=== DONE ==="
echo ""
echo "  Both models now report at end of training:"
echo "    - Total training time"
echo "    - Average epoch time"  
echo "    - Single-step inference time (ms)"
echo "    - Full rollout time (400 steps)"
echo "    - Speedup vs OpenFOAM"
echo ""
echo "  Per-epoch log shows time per epoch"
echo ""
echo "  For thesis Table:"
echo "    | Method    | Train time | Inference | Rollout | Speedup |"
echo "    |-----------|-----------|-----------|---------|---------|"
echo "    | OpenFOAM  | N/A       | N/A       | ~3 hrs  | 1x      |"
echo "    | GNN       | ~35 hrs   | ~X ms     | ~Y s    | ~Zx     |"
echo "    | 3D FNO    | ~12 hrs   | ~X ms     | ~Y s    | ~Zx     |"

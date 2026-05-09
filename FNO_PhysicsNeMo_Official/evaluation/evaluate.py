"""
3D FNO rollout evaluation — MAE, R^2, per-step trajectory, and
tolerance-band coverage on the held-out test set.

Reports the same set of metrics as the GNN evaluation script so the
two pipelines can be compared side-by-side in the thesis tables.
The focus is on the steel cylinder (the engineering target) and the
inner air cavity (the convective driver) — the outer enclosure
barely moves and the heaters are clamped, so neither is informative.

Output: one JSON file with per-sim rollouts + an aggregated summary
+ Phase-2 per-step trajectories for the rollout figures.
"""
from __future__ import annotations
import sys, json, argparse
from pathlib import Path
import numpy as np
import torch

# Make the project importable when running this from the eval/ folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.fno_config import CONFIG
from data.dataset import FNO3DDataset, REGION_IDS, HEATER_REGIONS
from models.fno_model import HeatTreatmentFNO3D
from models.rollout import rollout_per_region


# Two regions the thesis focuses on. The outer enclosure is
# quasi-static so its MAE is uninformative, and the heaters are
# clamped to T_set so reporting their "MAE" is meaningless.
FOCUS_REGIONS = ["steel_cylinder", "inner_box"]


def run_fno3d_evaluation(model, cfg, device="cuda", n_sims=None, output_dir=None):
    """
    Run the autoregressive rollout on every test sim and collect
    per-region accuracy metrics for both rollout phases.

    Phase 1 (in-horizon):    t in [200, train_time_end]   — within
                             the training window, measures
                             single-step / short-horizon stability
    Phase 2 (extrapolation): t in (train_time_end, predict_time_end] —
                             beyond the training horizon, measures
                             how the model holds up across the full
                             326-step rollout reported in the thesis
    """
    # Same dataset class used at training time, just with the test
    # split selected. The "evaluation" mode flag tells the dataset
    # to expose all timesteps instead of capping at n_train_steps.
    dataset = FNO3DDataset(cfg.dataset_path, cfg, "test", "evaluation")
    sim_indices = dataset.sim_indices
    if n_sims is not None:
        # Quick-pass mode for sanity checks during dev — limits
        # the rollout to the first --n_sims test cases.
        sim_indices = sim_indices[:n_sims]

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  3D FNO ROLLOUT EVALUATION")
    print(f"  {len(sim_indices)} test sims | grid {cfg.grid_x}x{cfg.grid_y}x{cfg.grid_z}")
    print(f"  In-horizon    : 200-{cfg.train_time_end:.0f}s (within training time range)")
    print(f"  Extrapolation : {cfg.train_time_end:.0f}-{cfg.predict_time_end:.0f}s (beyond training horizon)")
    print(f"{sep}")

    all_results = {}
    focus_p1 = {r: [] for r in FOCUS_REGIONS}
    focus_p2 = {r: [] for r in FOCUS_REGIONS}

    # ---- in-horizon table ------------------------------------------
    # Phase-1 rollout, one row per (sim, region). The rollout itself
    # happens once in rollout_per_region(); the loop below just pulls
    # out the Phase-1 numbers and prints them.
    print(f"\n  IN-HORIZON (200-{cfg.train_time_end:.0f}s)")
    print(f"  {'Sim':>4}  {'Region':>16}  {'MAE':>9}  {'R²':>9}  {'<5K%':>6}  {'<20K%':>7}")
    print(f"  {'-'*64}")

    for sim_i in sim_indices:
        results = rollout_per_region(model, dataset, sim_i, device=device)
        all_results[f"sim_{sim_i}"] = results

        for region in FOCUS_REGIONS:
            if region not in results:
                continue
            r = results[region]
            print(f"  {sim_i:>4}  {region:>16}  {r['mae_p1']:>7.2f}K  {r['r2_p1']:>9.4f}"
                  f"  {r['within_5K_p1']:>5.1f}%  {r['within_20K_p1']:>6.1f}%")
            focus_p1[region].append(r)
            # Skip Phase-2 entries that came back as NaN — happens
            # for short test cases that don't span the full horizon.
            if not np.isnan(r['mae_p2']):
                focus_p2[region].append(r)
        print()

    # ---- extrapolation table ---------------------------------------
    # Phase-2 numbers from the same rollouts. Printed separately so
    # the in-horizon vs extrapolation comparison is easy to read.
    print(f"\n  EXTRAPOLATION ({cfg.train_time_end:.0f}-{cfg.predict_time_end:.0f}s)")
    print(f"  {'Sim':>4}  {'Region':>16}  {'MAE':>9}  {'R²':>9}  {'<5K%':>6}  {'<20K%':>7}")
    print(f"  {'-'*64}")

    for sim_i in sim_indices:
        results = all_results[f"sim_{sim_i}"]
        for region in FOCUS_REGIONS:
            if region not in results:
                continue
            r = results[region]
            if np.isnan(r['mae_p2']):
                print(f"  {sim_i:>4}  {region:>16}  {'N/A':>9}  {'N/A':>9}")
            else:
                print(f"  {sim_i:>4}  {region:>16}  {r['mae_p2']:>7.2f}K  {r['r2_p2']:>9.4f}"
                      f"  {r['within_5K_p2']:>5.1f}%  {r['within_20K_p2']:>6.1f}%")
        print()

    # ---- per-step Phase 2 progression ------------------------------
    # Per-step MAE trajectory across the extrapolation window. This
    # is what produces the "MAE vs rollout time" curves in Section
    # 5.3 of the thesis — error growth gets very visible here.
    print(f"\n{sep}")
    print(f"  PHASE 2 PER-STEP MAE — STEEL CYLINDER (mean across {len(sim_indices)} sims)")
    print(f"  Each row = one rollout step (Δt = {cfg.dt:.0f} s)")
    print(f"{sep}")
    print(f"  {'abs_t':>6}  {'rollout_dt':>10}  {'MAE':>8}  {'std':>8}")
    print(f"  {'-'*40}")

    # Strip the first 20 timesteps off n_train_steps because the
    # rollout starts at t=200s (= step 20), not t=0s.
    n_train_steps = cfg.n_train_steps - 20
    per_step_phase2 = {region: [] for region in FOCUS_REGIONS}

    for sim_i in sim_indices:
        results = all_results[f"sim_{sim_i}"]
        for region in FOCUS_REGIONS:
            if region not in results:
                continue
            full_step_mae = np.array(results[region]["step_mae"])
            # Phase 2 starts where Phase 1 ends, so slice from
            # n_train_steps onwards.
            if len(full_step_mae) > n_train_steps:
                p2_mae = full_step_mae[n_train_steps:]
                per_step_phase2[region].append(p2_mae)

    # Steel cylinder per-step printout
    if per_step_phase2["steel_cylinder"]:
        # Different sims may end up with slightly different P2
        # lengths if the OpenFOAM run cut off a step early — clip
        # to the shortest one before stacking.
        min_len = min(len(x) for x in per_step_phase2["steel_cylinder"])
        steel_p2 = np.stack([x[:min_len] for x in per_step_phase2["steel_cylinder"]])
        mean_steel = steel_p2.mean(axis=0)
        std_steel  = steel_p2.std(axis=0)

        # Print every step (10s increments) so the rollout dynamics
        # are visible without needing the JSON file
        for step in range(min_len):
            abs_t = cfg.train_time_end + (step + 1) * cfg.dt
            rollout_dt = (n_train_steps + step + 1) * cfg.dt
            print(f"  {abs_t:>5.0f}s  {rollout_dt:>9.0f}s  "
                  f"{mean_steel[step]:>6.2f}K  {std_steel[step]:>6.2f}K")

    # Same per-step printout for the inner cavity
    print(f"\n{sep}")
    print(f"  PHASE 2 PER-STEP MAE — INNER CAVITY (mean across {len(sim_indices)} sims)")
    print(f"{sep}")
    print(f"  {'abs_t':>6}  {'rollout_dt':>10}  {'MAE':>8}  {'std':>8}")
    print(f"  {'-'*40}")

    if per_step_phase2["inner_box"]:
        min_len = min(len(x) for x in per_step_phase2["inner_box"])
        air_p2 = np.stack([x[:min_len] for x in per_step_phase2["inner_box"]])
        mean_air = air_p2.mean(axis=0)
        std_air  = air_p2.std(axis=0)

        for step in range(min_len):
            abs_t = cfg.train_time_end + (step + 1) * cfg.dt
            rollout_dt = (n_train_steps + step + 1) * cfg.dt
            print(f"  {abs_t:>5.0f}s  {rollout_dt:>9.0f}s  "
                  f"{mean_air[step]:>6.2f}K  {std_air[step]:>6.2f}K")

    # ---- summary across sims ---------------------------------------
    # Mean +/- std across the test set for each focus region. These
    # are the numbers that get pulled into Table 5.2 of the thesis.
    print(f"\n{sep}")
    print(f"  FOCUS REGIONS — MEAN ACROSS {len(sim_indices)} TEST SIMS")
    print(f"{sep}")

    summary = {}
    for region in FOCUS_REGIONS:
        if not focus_p1[region]:
            continue
        rs = focus_p1[region]
        print(f"\n  {region}:")
        print(f"    In-horizon (200-{cfg.train_time_end:.0f}s):")
        print(f"      MAE         = {np.mean([r['mae_p1'] for r in rs]):>7.2f} ± {np.std([r['mae_p1'] for r in rs]):.2f} K")
        print(f"      R²          = {np.mean([r['r2_p1']  for r in rs]):>7.4f}")
        print(f"      Within  5 K = {np.mean([r['within_5K_p1']  for r in rs]):>5.1f}%")
        print(f"      Within 10 K = {np.mean([r['within_10K_p1'] for r in rs]):>5.1f}%")
        print(f"      Within 20 K = {np.mean([r['within_20K_p1'] for r in rs]):>5.1f}%")
        print(f"      Rel. MAE    = {np.mean([r['rel_mae_pct_p1'] for r in rs]):>5.2f}%")

        summary[region] = {"in_horizon": {
            "mae_mean":      float(np.mean([r['mae_p1'] for r in rs])),
            "mae_std":       float(np.std ([r['mae_p1'] for r in rs])),
            "r2_mean":       float(np.mean([r['r2_p1']  for r in rs])),
            "within_5K":     float(np.mean([r['within_5K_p1']  for r in rs])),
            "within_10K":    float(np.mean([r['within_10K_p1'] for r in rs])),
            "within_20K":    float(np.mean([r['within_20K_p1'] for r in rs])),
            "rel_mae_pct":   float(np.mean([r['rel_mae_pct_p1'] for r in rs])),
        }}

        if focus_p2[region]:
            rs2 = focus_p2[region]
            print(f"    Extrapolation ({cfg.train_time_end:.0f}-{cfg.predict_time_end:.0f}s):")
            print(f"      MAE         = {np.mean([r['mae_p2'] for r in rs2]):>7.2f} ± {np.std([r['mae_p2'] for r in rs2]):.2f} K")
            # R^2 is reported but flagged because in the late-rollout
            # plateau regime the ground-truth variance collapses, which
            # makes R^2 unstable even when the absolute error is small.
            print(f"      R²          = {np.mean([r['r2_p2']  for r in rs2]):>7.4f}    (note: pathological in low-variance regime)")
            print(f"      Within  5 K = {np.mean([r['within_5K_p2']  for r in rs2]):>5.1f}%")
            print(f"      Within 10 K = {np.mean([r['within_10K_p2'] for r in rs2]):>5.1f}%")
            print(f"      Within 20 K = {np.mean([r['within_20K_p2'] for r in rs2]):>5.1f}%")
            print(f"      Rel. MAE    = {np.mean([r['rel_mae_pct_p2'] for r in rs2]):>5.2f}%")
            summary[region]["extrapolation"] = {
                "mae_mean":      float(np.mean([r['mae_p2'] for r in rs2])),
                "mae_std":       float(np.std ([r['mae_p2'] for r in rs2])),
                "r2_mean":       float(np.mean([r['r2_p2']  for r in rs2])),
                "within_5K":     float(np.mean([r['within_5K_p2']  for r in rs2])),
                "within_10K":    float(np.mean([r['within_10K_p2'] for r in rs2])),
                "within_20K":    float(np.mean([r['within_20K_p2'] for r in rs2])),
                "rel_mae_pct":   float(np.mean([r['rel_mae_pct_p2'] for r in rs2])),
            }

    # ---- outlier check ---------------------------------------------
    # Quick scan for sims whose Phase-1 MAE is more than 2x the
    # ensemble median — usually the high-T_set cases that the
    # convergence filter under-represented (Section 5.2 in the thesis).
    print(f"\n{sep}")
    print(f"  OUTLIER CHECK — steel_cylinder (in-horizon)")
    print(f"{sep}")
    steel_maes = []
    for sim_i in sim_indices:
        r = all_results[f"sim_{sim_i}"].get("steel_cylinder", {})
        if r:
            steel_maes.append((sim_i, r["mae_p1"]))
    if steel_maes:
        median_in = np.median([m for _, m in steel_maes])
        for sim_i, mae in steel_maes:
            sim = dataset._simulations[sim_i]
            flag = "  ⚠ OUTLIER" if mae > 2 * median_in else ""
            print(f"  sim_{sim_i:<3}  T_set={sim['T_set']:>5.0f}K  MAE={mae:>6.2f}K{flag}")

    # ---- save JSON (with per-step trajectories) -------------------
    # Three blocks in the output: per_sim (every metric for every
    # case), summary (table-ready aggregates), and phase2_per_step
    # (the trajectory data the plotting script picks up).
    out_dir = output_dir if output_dir else f"{cfg.output_dir}/evaluation"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = f"{out_dir}/fno_rollout_results.json"

    # Build the per-step block ahead of time so the JSON dump
    # at the bottom stays a one-liner.
    phase2_per_step = {}
    for region in FOCUS_REGIONS:
        if per_step_phase2[region]:
            min_len = min(len(x) for x in per_step_phase2[region])
            stack = np.stack([x[:min_len] for x in per_step_phase2[region]])
            phase2_per_step[region] = {
                "abs_t":       [cfg.train_time_end + (s + 1) * cfg.dt for s in range(min_len)],
                "rollout_dt":  [(n_train_steps + s + 1) * cfg.dt for s in range(min_len)],
                "mae_mean":    stack.mean(axis=0).tolist(),
                "mae_std":     stack.std(axis=0).tolist(),
                "n_sims":      len(per_step_phase2[region]),
            }

    print(f"\n  Saving results to: {out_path}")
    with open(out_path, "w") as f:
        json.dump({
            "per_sim":           all_results,
            "summary":           summary,
            "phase2_per_step":   phase2_per_step,
        }, f, indent=2, default=str)
    print(f"  Saved: {out_path}")
    print(f"{sep}\n")
    return all_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_sims", type=int, default=None,
                        help="Limit to first N test sims (default: all)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to best_model.pt (overrides cfg.checkpoint_dir)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output folder for JSON results")
    args = parser.parse_args()

    cfg = CONFIG
    # Default to the best-model checkpoint inside the config's
    # output dir; --checkpoint lets the SLURM job point at any other.
    ckpt_path = args.checkpoint if args.checkpoint else f"{cfg.checkpoint_dir}/best_model.pt"
    print(f"  Loading: {ckpt_path}")
    model = HeatTreatmentFNO3D.load(ckpt_path, cfg, args.device)
    run_fno3d_evaluation(model, cfg, device=args.device,
                         n_sims=args.n_sims, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
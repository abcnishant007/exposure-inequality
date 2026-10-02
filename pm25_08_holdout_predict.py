#!/usr/bin/env python3
"""
Evaluate true held-out sensors (never seen in training) using the trained model.
Assumes a holdout list exists at PROJECT/holdout_sensors.txt (one sensor_index per line).
Produces:
  single_images_fixed/holdout_diurnal_logpm25_level_oos.pdf
  single_images_fixed/holdout_diurnal_logS_shape_only.pdf
  single_images_fixed/holdout_error_summary.pdf

Run from repo root:
  python pm25_08_holdout_predict.py --project bayesian_fusion_after_20260223_keep_median_only_Oklahoma
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import qr
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans


def load_holdout_list(project: Path) -> List[int]:
    f = project / "holdout_sensors.txt"
    if not f.exists():
        raise FileNotFoundError(f"holdout_sensors.txt not found in {project}")
    return sorted(int(line.strip()) for line in f.read_text().splitlines() if line.strip())


def main():
    parser = argparse.ArgumentParser(description="Evaluate holdout sensors with trained model")
    parser.add_argument("--project", default=".", help="Scenario folder (with run_pipeline.sh and trace file)")
    parser.add_argument(
        "--holdout-sensor",
        type=int,
        default=None,
        help="If set, evaluate only this single holdout sensor_id.",
    )
    parser.add_argument(
        "--out-dir",
        default="single_images_fixed",
        help="Output directory relative to --project (or absolute path). Default: single_images_fixed",
    )
    parser.add_argument(
        "--only-binmedian-plot",
        action="store_true",
        help="Only generate the bin-median normalised shape profile plot (plus metrics CSV/JSON).",
    )
    parser.add_argument(
        "--clean-plot",
        action="store_true",
        help="For single-sensor bin-median plot: remove titles/legend/axis labels/ticks (thumbnail-friendly).",
    )
    parser.add_argument(
        "--example-plot",
        action="store_true",
        help="For single-sensor bin-median plot: style with one legend/axis for map example (no xlabel, keep xticks).",
    )
    parser.add_argument(
        "--mu-baseline-mode",
        choices=["grid", "nearest_sensor_posterior", "grid_plus_nearest_delta", "grid_plus_basis_delta"],
        default="grid",
        help="How to set mu_baseline for a holdout sensor. "
             "'grid' uses log(B(x*)) from the satellite baseline grid. "
             "'nearest_sensor_posterior' borrows posterior mu_baseline from nearest training sensor. "
             "'grid_plus_nearest_delta' uses log(B(x*)) + (mu_baseline_near - log(B(x_near))). "
             "'grid_plus_basis_delta' uses log(B(x*)) + delta(x*), where delta is a least-squares fit "
             "of (mu_baseline - log(B)) onto the spatial basis.",
    )
    args = parser.parse_args()

    project = Path(args.project).resolve()
    if not (project / "trace_hierarchical_interactive_fixed.nc").exists():
        raise SystemExit("Trace file not found; run the pipeline first")

    # Many per-city `pm25_data_loader.py` implementations load inputs via relative paths.
    # Ensure those reads resolve against the project directory even when this script is
    # invoked from the repo root (e.g. when generating per-city thumbnails).
    os.chdir(project)

    holdouts = load_holdout_list(project)
    if args.holdout_sensor is not None:
        if args.holdout_sensor not in holdouts:
            raise SystemExit(f"--holdout-sensor {args.holdout_sensor} not found in {project / 'holdout_sensors.txt'}")
        holdouts = [int(args.holdout_sensor)]
    print(f"Holdout sensors: {holdouts}")
    print(f"mu_baseline mode: {args.mu_baseline_mode}")

    # Temporarily disable holdout filtering to load full data (for heldout obs)
    old_env = os.environ.pop("HOLDOUT_SENSORS_FILE", None)

    import sys
    sys.path.insert(0, str(project))
    from pm25_data_loader import load_data  # type: ignore
    from pm25_config import LENGTH_SCALE, N_TIME_BASIS, N_SPATIAL_BASIS, K

    # Load full data (includes holdouts) for observations
    df_full, df_sensors_full, df_static, sensor_order_full, S_full, K_full = load_data(verbose=True)

    # Helper: nearest baseline B(x) from the static grid (used for holdout baselines)
    grid_coords = df_static[["latitude", "longitude"]].to_numpy()
    grid_tree = cKDTree(grid_coords)
    # Now load training-only data by re-applying holdout env var
    if old_env:
        os.environ["HOLDOUT_SENSORS_FILE"] = old_env
    elif (project / "holdout_sensors.txt").exists():
        os.environ["HOLDOUT_SENSORS_FILE"] = str(project / "holdout_sensors.txt")
    else:
        raise SystemExit("No holdout env or file set; cannot load training data")

    df_train, df_sensors_train, df_static_train, sensor_order_train, S_train, _ = load_data(verbose=True)

    # Restore env
    if old_env:
        os.environ["HOLDOUT_SENSORS_FILE"] = old_env
    else:
        os.environ.pop("HOLDOUT_SENSORS_FILE", None)

    # Build training coords and adaptive basis
    sensor_coords = df_sensors_train[["latitude", "longitude"]].to_numpy()
    n_spatial_basis = min(N_SPATIAL_BASIS, S_train)
    # Basis centers from training sensors
    km2 = KMeans(n_clusters=n_spatial_basis, random_state=42, n_init=10)
    basis_centers = km2.fit(sensor_coords).cluster_centers_
    dist_s = cdist(sensor_coords, basis_centers)
    Phi_raw = np.exp(-0.5 * (dist_s / LENGTH_SCALE) ** 2)
    Q_s, R = qr(Phi_raw, mode="economic")
    Phi_sensors = Q_s
    # Map raw RBF features at x* into the orthonormalised QR basis used in training.
    # R can become (near-)singular for extreme LENGTH_SCALE values; use a pseudo-inverse
    # so sweeps over h don't crash.
    R_inv = np.linalg.pinv(R)

    # KD-tree for nearest training sensor lookups (baseline and bias borrowing)
    train_coords = sensor_coords
    tree = cKDTree(train_coords)

    # Temporal basis (must match training in pm25_01_fusion.py)
    time_centers = np.arange(N_TIME_BASIS) * (K / N_TIME_BASIS)  # 0, K/n_T, ...
    time_scale = K / (N_TIME_BASIS * 1.65)
    k_idx = np.arange(K)[:, None]
    center = time_centers[None, :]
    dist = np.abs(k_idx - center)
    dist_wrapped = np.minimum(dist, K - dist)
    temporal_basis = np.exp(-0.5 * (dist_wrapped / time_scale) ** 2)
    temporal_basis = temporal_basis / temporal_basis.sum(axis=0, keepdims=True)

    # Load trace
    from pm25_config import N_PRED_SAMPLES
    trace = az.from_netcdf(project / "trace_hierarchical_interactive_fixed.nc")
    post = trace.posterior
    n_total = int(post.sizes["chain"] * post.sizes["draw"])

    rng = np.random.default_rng(42)
    sel = rng.choice(n_total, size=min(N_PRED_SAMPLES, n_total), replace=False)
    sel = np.sort(sel)

    def flat(x):
        s = x.shape
        return x.reshape(s[0] * s[1], *s[2:])

    log_S_flat = flat(post["log_S"].values)[sel]          # (n_samp, S_train, K)
    w_flat = flat(post["w"].values)[sel]                  # (n_samp, N_TIME_BASIS, n_spatial_basis)

    # Keep basis construction consistent with posterior width if config/sensor count drifted.
    n_spatial_basis_post = int(w_flat.shape[2])
    if n_spatial_basis_post != n_spatial_basis:
        n_spatial_basis = min(n_spatial_basis_post, S_train)
        km2 = KMeans(n_clusters=n_spatial_basis, random_state=42, n_init=10)
        basis_centers = km2.fit(sensor_coords).cluster_centers_
        dist_s = cdist(sensor_coords, basis_centers)
        Phi_raw = np.exp(-0.5 * (dist_s / LENGTH_SCALE) ** 2)
        Q_s, R = qr(Phi_raw, mode="economic")
        Phi_sensors = Q_s
        R_inv = np.linalg.pinv(R)

    mu_flat = flat(post["mu"].values)[sel]                # (n_samp, K)
    mu_bl_flat = flat(post["mu_baseline"].values)[sel]    # (n_samp, S_train)
    b_flat = flat(post["b"].values)[sel]                  # (n_samp, S_train)
    sigma_flat = flat(post["sigma_s"].values)[sel]        # (n_samp, S_train)
    n_samp = len(sel)

    # Precompute log(B(x_s)) for training sensors for delta = mu_baseline - log(B).
    baseline_train = df_sensors_train["baseline"].to_numpy(dtype=float)
    baseline_train = np.maximum(baseline_train, 1e-12)
    log_baseline_train = np.log(baseline_train)  # (S_train,)

    BIN_HOURS = np.arange(K) * 0.5

    loso_results = []

    for hid in holdouts:
        # Observations for this holdout
        obs = df_full[df_full['sensor_index'] == hid].copy()
        if obs.empty:
            print(f"Holdout sensor {hid} has no observations; skipping")
            continue
        y_obs = obs['log_pm25'].values
        k_obs = obs['bin'].values
        bin_hours_obs = obs['bin'].values * 0.5

        # Mean-over-bins normalization: after load_data() collapses to one median per sensor x bin,
        # there are no true per-day observations left. So this normalizes the 48-bin monthly-median
        # profile by its mean over bins.
        if 'datetime_utc' in obs.columns:
            obs['date'] = pd.to_datetime(obs['datetime_utc']).dt.date
        else:
            obs['date'] = 0  # fallback single day
        day_mean = obs.groupby('date')['pm2.5_atm'].transform('mean')
        obs['pm_norm_mean48'] = obs['pm2.5_atm'] / day_mean.replace(0, np.nan)
        # Sensor coords from df_sensors_full
        coord_row = df_sensors_full[df_sensors_full['sensor_index'] == hid][['latitude','longitude']]
        if coord_row.empty:
            print(f"Holdout sensor {hid} missing coords; skipping")
            continue
        coord = coord_row.to_numpy()  # shape (1,2)

        # Baseline at holdout location: use satellite baseline B(x*) from nearest grid cell.
        grid_dist, grid_idx = grid_tree.query(coord)
        grid_idx = int(np.atleast_1d(grid_idx)[0])
        baseline_holdout = float(df_static.iloc[grid_idx]["baseline"])
        baseline_holdout = max(baseline_holdout, 1e-12)
        log_baseline_holdout = float(np.log(baseline_holdout))

        # Nearest training sensor (for mu_baseline delta / bias / noise borrowing)
        _, nearest_idx = tree.query(coord)  # nearest training sensor index
        nearest_idx = int(np.atleast_1d(nearest_idx)[0])

        # Project coord into basis (used for r(x,k) and optionally mu_baseline delta(x))
        dist_held = cdist(coord, basis_centers)
        phi_raw_h = np.exp(-0.5 * (dist_held / LENGTH_SCALE) ** 2)
        phi_held = phi_raw_h @ R_inv  # (1, N_SPATIAL_BASIS)

        if args.mu_baseline_mode == "grid":
            mu_bl_holdout = np.full((n_samp,), log_baseline_holdout, dtype=float)
        elif args.mu_baseline_mode == "nearest_sensor_posterior":
            mu_bl_holdout = mu_bl_flat[:, nearest_idx]
        elif args.mu_baseline_mode == "grid_plus_nearest_delta":
            baseline_near = float(df_sensors_train.iloc[nearest_idx]["baseline"])
            baseline_near = max(baseline_near, 1e-12)
            log_baseline_near = float(np.log(baseline_near))
            delta = mu_bl_flat[:, nearest_idx] - log_baseline_near
            mu_bl_holdout = log_baseline_holdout + delta
        elif args.mu_baseline_mode == "grid_plus_basis_delta":
            # Fit delta(s) = mu_baseline(s) - log(B(x_s)) onto the spatial basis and evaluate at x*.
            # Phi_sensors is Q from QR -> columns are orthonormal, so LS coefficients are alpha = Phi^T delta.
            delta_samp = mu_bl_flat - log_baseline_train[None, :]  # (n_samp, S_train)
            alpha = delta_samp @ Phi_sensors  # (n_samp, N_SPATIAL_BASIS)
            delta_holdout = (alpha @ phi_held.T).reshape(-1)  # (n_samp,)
            mu_bl_holdout = log_baseline_holdout + delta_holdout
        else:
            raise SystemExit(f"Unknown --mu-baseline-mode: {args.mu_baseline_mode}")

        A = np.einsum('pij,xj->pi', w_flat, phi_held)   # (n_samp, N_TIME_BASIS)
        r_raw_held = np.einsum('pi,ki->pk', A, temporal_basis)  # (n_samp, K)

        A_all = np.einsum('pij,sj->pis', w_flat, Phi_sensors)
        r_all = np.einsum('pis,ki->psk', A_all, temporal_basis)
        r_mean = r_all.mean(axis=1)  # (n_samp, K)
        r_centered_held = r_raw_held - r_mean

        Z_held = mu_flat + r_centered_held
        log_sum = np.log(np.exp(Z_held - Z_held.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True)) + Z_held.max(axis=1, keepdims=True)
        log_S_held = Z_held - log_sum + np.log(K)

        # Level prediction: baseline from satellite at x*, bias from nearest training sensor
        b_near     = b_flat[:, nearest_idx]            # (n_samp,)
        log_pm25_pred = mu_bl_holdout[:, None] + log_S_held + b_near[:, None]  # (n_samp, K)

        # No-bias/no-baseline prediction
        log_pm25_pred_nobias = log_S_held  # (n_samp, K)

        y_pred = log_pm25_pred[:, k_obs].mean(axis=0)
        rmse = float(np.sqrt(np.mean((y_obs - y_pred)**2)))
        mae  = float(np.mean(np.abs(y_obs - y_pred)))
        r2   = float(r2_score(y_obs, y_pred))

        y_pred_nb = log_pm25_pred_nobias[:, k_obs].mean(axis=0)
        rmse_nb = float(np.sqrt(np.mean((y_obs - y_pred_nb)**2)))
        mae_nb  = float(np.mean(np.abs(y_obs - y_pred_nb)))
        r2_nb   = float(r2_score(y_obs, y_pred_nb))

        pred_q = np.percentile(log_pm25_pred, [5, 95], axis=0)  # (2, K)
        pred_q_nb = np.percentile(log_pm25_pred_nobias, [5, 95], axis=0)  # (2, K)

        # predicted shape (S) and empirical shape (median of mean-normalised obs)
        shape_samples = np.exp(log_S_held)  # (n_samp, K), already mean-preserving
        shape_mean = shape_samples.mean(axis=0)
        shape_q95 = np.percentile(shape_samples, [2.5, 97.5], axis=0)

        # Normalised by daily mean (per instruction)
        emp_shape_profile = np.full(K, np.nan)
        for k in range(K):
            vals_norm = obs.loc[obs['bin'] == k, 'pm_norm_mean48'].values
            vals_norm = vals_norm[np.isfinite(vals_norm)]
            if len(vals_norm):
                emp_shape_profile[k] = np.median(vals_norm)

        # Empirical bin medians (linear) and bin-median normalisation
        emp_bin_median = np.full(K, np.nan)
        for k in range(K):
            vals_lin = obs[obs['bin'] == k]['pm2.5_atm'].values
            if len(vals_lin):
                emp_bin_median[k] = np.median(vals_lin)
        bin_median_mean = np.nanmean(emp_bin_median)
        emp_shape_bin_median = emp_bin_median / bin_median_mean if np.isfinite(bin_median_mean) and bin_median_mean > 0 else np.full(K, np.nan)

        # R^2 for the normalised plots (computed on normalised quantities, per-bin).
        r2_shape_norm = np.nan
        valid = np.isfinite(emp_shape_profile)
        if valid.sum() > 1:
            r2_shape_norm = float(r2_score(emp_shape_profile[valid], shape_mean[valid]))

        r2_shape_binmedian_norm = np.nan
        valid = np.isfinite(emp_shape_bin_median)
        if valid.sum() > 1:
            r2_shape_binmedian_norm = float(r2_score(emp_shape_bin_median[valid], shape_mean[valid]))

        # Level predictions in linear space (with borrowed baseline/bias)
        pm_samples = np.exp(log_pm25_pred)
        pm_pred_mean = pm_samples.mean(axis=0)
        pm_pred_q = np.percentile(pm_samples, [5, 95], axis=0)

        # Level predictions without bias/baseline
        pm_samples_nb = np.exp(log_pm25_pred_nobias)
        pm_pred_mean_nb = pm_samples_nb.mean(axis=0)
        pm_pred_q_nb = np.percentile(pm_samples_nb, [5, 95], axis=0)

        # empirical profile
        emp_profile = np.full(K, np.nan)
        for k in range(K):
            vals = obs[obs['bin']==k]['log_pm25'].values
            if len(vals):
                emp_profile[k] = vals.mean()

        # ---------------- Shape-only metrics ----------------
        # Normalise predicted log S and empirical shapes by their mean over bins
        pred_profile = log_pm25_pred.mean(axis=0)
        pred_profile_nb = log_pm25_pred_nobias.mean(axis=0)
        pred_shape = pred_profile - np.nanmean(pred_profile)
        pred_shape_nb = pred_profile_nb - np.nanmean(pred_profile_nb)

        emp_shape_log = emp_profile - np.nanmean(emp_profile)
        shape_mask = np.isfinite(emp_shape_log)
        if shape_mask.sum() > 1:
            shape_rmse = float(np.sqrt(np.mean((emp_shape_log[shape_mask] - pred_shape[shape_mask])**2)))
            shape_mae = float(np.mean(np.abs(emp_shape_log[shape_mask] - pred_shape[shape_mask])))
            shape_r2 = float(r2_score(emp_shape_log[shape_mask], pred_shape[shape_mask]))
        else:
            shape_rmse, shape_mae, shape_r2 = np.nan, np.nan, np.nan

        # Linear obs for level plot (bin medians, un-normalised)
        emp_pm_bin_median = emp_bin_median.copy()

        loso_results.append({
            'sensor_id': hid,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'rmse_nb': rmse_nb,
            'mae_nb': mae_nb,
            'r2_nb': r2_nb,
            'shape_rmse': shape_rmse,
            'shape_mae': shape_mae,
            'shape_r2': shape_r2,
            'pred_profile': log_pm25_pred.mean(axis=0),
            'pred_q': pred_q,
            'pred_profile_nb': pred_profile_nb,
            'pred_q_nb': pred_q_nb,
            'emp_profile': emp_profile,
            'shape_mean': shape_mean,
            'shape_q95': shape_q95,
            'emp_shape': emp_shape_profile,
            'emp_shape_bin_median': emp_shape_bin_median,
            'r2_shape_norm': r2_shape_norm,
            'r2_shape_binmedian_norm': r2_shape_binmedian_norm,
            'pm_pred_mean': pm_pred_mean,
            'pm_pred_q': pm_pred_q,
            'pm_pred_mean_nb': pm_pred_mean_nb,
            'pm_pred_q_nb': pm_pred_q_nb,
            'emp_pm_bin_median': emp_pm_bin_median,
            'scatter_log': (bin_hours_obs, y_obs),
            'scatter_norm': (bin_hours_obs, obs['pm_norm_mean48'].values),
            'nearest_train_idx': int(nearest_idx),
        })
        print(f"Holdout {hid}: n={len(y_obs)} RMSE={rmse:.3f} MAE={mae:.3f} R2={r2:.3f} | NoBias RMSE={rmse_nb:.3f} R2={r2_nb:.3f} | ShapeRMSE={shape_rmse:.3f} ShapeR2={shape_r2:.3f}")

    if not loso_results:
        raise SystemExit("No holdout sensors evaluated")

    out_dir = Path(args.out_dir)
    OUT_DIR = out_dir if out_dir.is_absolute() else (project / out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    single_holdout_mode = args.holdout_sensor is not None
    clean_plot = bool(args.clean_plot) and single_holdout_mode
    example_plot = bool(args.example_plot) and single_holdout_mode and (not clean_plot)

    # Output names (also save legacy filenames for backwards compatibility)
    FN_DIURNAL_LOGPM25_LEVEL_OOS = "holdout_diurnal_logpm25_level_oos.pdf"
    FN_DIURNAL_LOGS_SHAPE_ONLY = "holdout_diurnal_logS_shape_only.pdf"
    LEGACY_FN_DIURNAL_TRUE_OOS = "holdout_diurnal_profiles_true_out_of_sample.pdf"
    LEGACY_FN_DIURNAL_NO_BIAS = "holdout_diurnal_profiles_no_bias.pdf"

    # Plot diurnal profiles (log space) with raw-point scatter
    n = len(loso_results)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    if not args.only_binmedian_plot:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharey=False)
        axes = np.atleast_1d(axes).flatten()
        for ax, res in zip(axes, loso_results):
            pred = res['pred_profile']
            q05, q95 = res['pred_q']
            emp = res['emp_profile']
            valid = ~np.isnan(emp)
            ax.fill_between(BIN_HOURS, q05, q95, color='#1f77b4', alpha=0.2, label='Predictive 90% CI')
            ax.plot(BIN_HOURS, pred, '#1f77b4', lw=2, label='Predicted')
            ax.plot(BIN_HOURS[valid], emp[valid], 'o', color='#d62728', ms=3, alpha=0.7, label='Observed mean')
            # raw observations
            sc_x, sc_y = res['scatter_log']
            ax.scatter(sc_x, sc_y, color='gold', alpha=0.15, s=8, label='Obs (all)')
            ax.set_title(f"Sensor {res['sensor_id']}  RMSE={res['rmse']:.3f}  R²={res['r2']:.3f}", fontsize=9)
            ax.set_xticks([0,6,12,18,24])
            ax.set_xlabel('Hour', fontsize=8)
            ax.set_ylabel('log PM2.5', fontsize=8)
        for ax in axes[len(loso_results):]:
            ax.set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, fontsize=7)
        fig.suptitle('Holdout Sensors: Predicted vs Observed Diurnal Profile (no refit, true holdout)', y=1.02)
        fig.tight_layout()
        fig.savefig(OUT_DIR / FN_DIURNAL_LOGPM25_LEVEL_OOS, dpi=200, bbox_inches='tight')
        fig.savefig(OUT_DIR / LEGACY_FN_DIURNAL_TRUE_OOS, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Plot diurnal profiles WITHOUT bias/baseline borrowing (pure log_S)
    if not args.only_binmedian_plot:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharey=False)
        axes = np.atleast_1d(axes).flatten()
        for ax, res in zip(axes, loso_results):
            pred = res['pred_profile_nb']
            q05, q95 = res['pred_q_nb']
            emp = res['emp_profile']
            valid = ~np.isnan(emp)
            ax.fill_between(BIN_HOURS, q05, q95, color='#1f77b4', alpha=0.2, label='Predictive 90% CI (no bias)')
            ax.plot(BIN_HOURS, pred, '#1f77b4', lw=2, label='Predicted (no bias)')
            ax.plot(BIN_HOURS[valid], emp[valid], 'o', color='#d62728', ms=3, alpha=0.7, label='Observed mean')
            sc_x, sc_y = res['scatter_log']
            ax.scatter(sc_x, sc_y, color='gold', alpha=0.15, s=8, label='Obs (all)')
            ax.set_title(f"Sensor {res['sensor_id']}  RMSE(nb)={res['rmse_nb']:.3f}  R²(nb)={res['r2_nb']:.3f}", fontsize=9)
            ax.set_xticks([0,6,12,18,24])
            ax.set_xlabel('Hour', fontsize=8)
            ax.set_ylabel('log PM2.5', fontsize=8)
        for ax in axes[len(loso_results):]:
            ax.set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        axes[0].legend(handles, labels, fontsize=7)
        fig.suptitle('Holdout Sensors: Diurnal Profile without bias/baseline borrowing', y=1.02)
        fig.tight_layout()
        fig.savefig(OUT_DIR / FN_DIURNAL_LOGS_SHAPE_ONLY, dpi=200, bbox_inches='tight')
        fig.savefig(OUT_DIR / LEGACY_FN_DIURNAL_NO_BIAS, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # Shape-only profiles (mean-normalised)
    if not args.only_binmedian_plot:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharey=True)
        axes = np.atleast_1d(axes).flatten()
        for ax, res in zip(axes, loso_results):
            shape = res['shape_mean']
            q2p5, q97p5 = res['shape_q95']
            emp_shape = res['emp_shape']
            valid = ~np.isnan(emp_shape)
            ax.fill_between(BIN_HOURS, q2p5, q97p5, color='#1f77b4', alpha=0.2, label='Predictive 95% CI')
            ax.plot(BIN_HOURS, shape, '#1f77b4', lw=2, label='Predictive shape mean')
            if valid.any():
                ax.plot(BIN_HOURS[valid], emp_shape[valid], 'o', color='#d62728', ms=3, alpha=0.8, label='Observed median (norm)')
            ax.set_ylim(0.2, 3.0)
            ax.set_title(f"Sensor {res['sensor_id']}  R²(norm)={res['r2_shape_norm']:.3f}", fontsize=9)
            ax.set_xticks([0,6,12,18,24])
            ax.set_xlabel('Hour', fontsize=8)
            ax.set_ylabel('Normalised PM2.5', fontsize=8)
        for ax in axes[len(loso_results):]:
            ax.set_visible(False)
        axes[0].legend(fontsize=7)
        fig.suptitle('Holdout Sensors: Shape-only profiles (mean normalised)', y=1.02)
        fig.tight_layout()
        fig.savefig(OUT_DIR / 'holdout_shape_profiles_true_out_of_sample.pdf', dpi=200, bbox_inches='tight')
        plt.close(fig)

    # Shape-only profiles with bin-median normalisation (mean of 48-bin medians = 1) + raw scatter
    # (monthly-median profile normalized by mean over bins).
    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharey=True)
    axes = np.atleast_1d(axes).flatten()
    for ax, res in zip(axes, loso_results):
        shape = res['shape_mean']
        q2p5, q97p5 = res['shape_q95']
        emp_norm = res['emp_shape_bin_median']
        valid = ~np.isnan(emp_norm)
        ax.fill_between(BIN_HOURS, q2p5, q97p5, color='#1f77b4', alpha=0.2, label='Predictive 95% CI')
        ax.plot(BIN_HOURS, shape, '#1f77b4', lw=2, label='Predictive shape mean')
        if valid.any():
            ax.plot(BIN_HOURS[valid], emp_norm[valid], 'o', color='#d62728', ms=3, alpha=0.8, label='Observed median / mean(bin medians)')
        sc_x, sc_y = res['scatter_norm']
        ax.scatter(sc_x, sc_y, color='gold', alpha=0.15, s=8, label='Obs (all, mean48-norm)')
        ax.set_xlim(0, 24)
        ax.set_ylim(0.2, 3.0)
        if not single_holdout_mode:
            ax.set_title(f"Sensor {res['sensor_id']}  R²(bin-med norm)={res['r2_shape_binmedian_norm']:.3f}", fontsize=9)
        else:
            ax.set_title("")
        if clean_plot:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_xlabel("")
            ax.set_ylabel("")
            for spine in ax.spines.values():
                spine.set_visible(False)
        elif example_plot:
            ax.set_xlabel("")
            ax.set_ylabel("Normalised PM2.5\n(bin medians)", fontsize=10)
            xt = [3, 6, 9, 12, 15, 18, 21]
            ax.set_xticks(xt)
            ax.set_xticklabels([f"{h:02d}:00" for h in xt], fontsize=9)
            ax.tick_params(axis="x", length=3.5, width=0.8)
            ax.tick_params(axis="y", labelsize=9)
        else:
            ax.set_xticks([0, 6, 12, 18, 24])
            ax.set_xlabel('Hour', fontsize=8)
            ax.set_ylabel('Normalised PM2.5 (bin medians)', fontsize=8)
    for ax in axes[len(loso_results):]:
        ax.set_visible(False)
    if not single_holdout_mode:
        axes[0].legend(fontsize=7)
        fig.suptitle('Holdout Sensors: Shape-only (bin-median normalised)', y=1.02)
    elif example_plot:
        # One legend for the example plot, larger for readability on the US map.
        axes[0].legend(fontsize=10, loc="lower left", frameon=False)

    # Thumbnail-friendly transparent background (only for clean single-sensor plot).
    if clean_plot:
        fig.patch.set_alpha(0.0)
        fig.patch.set_facecolor("none")
        for ax in axes:
            try:
                ax.set_facecolor("none")
            except Exception:
                pass
    fig.tight_layout()
    if single_holdout_mode:
        out_pdf = OUT_DIR / "best_holdout_shape_profiles_binmedian_norm_true_out_of_sample.pdf"
        out_png = OUT_DIR / "best_holdout_shape_profiles_binmedian_norm_true_out_of_sample.png"
        out_npz = OUT_DIR / "best_holdout_shape_profile_binmedian_norm_data.npz"
    else:
        out_pdf = OUT_DIR / "holdout_shape_profiles_binmedian_norm_true_out_of_sample.pdf"
        out_png = None
    fig.savefig(out_pdf, dpi=200, bbox_inches='tight')
    if out_png is not None:
        fig.savefig(out_png, dpi=200, bbox_inches='tight', transparent=bool(clean_plot))
    plt.close(fig)

    # Export the underlying 1-sensor data to allow re-styling plots later without re-running the
    # full holdout evaluation logic. This is only produced in single-holdout mode.
    if single_holdout_mode:
        try:
            res0 = loso_results[0]
            q2p5, q97p5 = res0["shape_q95"]
            sc_x, sc_y = res0["scatter_norm"]
            np.savez(
                out_npz,
                sensor_id=int(res0["sensor_id"]),
                r2_shape_binmedian_norm=float(res0.get("r2_shape_binmedian_norm", np.nan)),
                bin_hours=np.asarray(BIN_HOURS, dtype=float),
                pred_shape_mean=np.asarray(res0["shape_mean"], dtype=float),
                pred_shape_q2p5=np.asarray(q2p5, dtype=float),
                pred_shape_q97p5=np.asarray(q97p5, dtype=float),
                obs_shape_binmedian_norm=np.asarray(res0["emp_shape_bin_median"], dtype=float),
                scatter_x=np.asarray(sc_x, dtype=float),
                scatter_y=np.asarray(sc_y, dtype=float),
            )
            print(f"Wrote plot data: {out_npz}")
        except Exception:
            pass

    # Level (µg/m³) profiles using B(x) × S(x,k)
    if not args.only_binmedian_plot:
        fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharey=False)
        axes = np.atleast_1d(axes).flatten()
        for ax, res in zip(axes, loso_results):
            pm_mean = res['pm_pred_mean']
            q05, q95 = res['pm_pred_q']
            emp_pm = res['emp_pm_bin_median']
            valid = ~np.isnan(emp_pm)
            ax.fill_between(BIN_HOURS, q05, q95, color='#1f77b4', alpha=0.2, label='Predictive 90% CI')
            ax.plot(BIN_HOURS, pm_mean, '#1f77b4', lw=2, label='Predictive mean')
            if valid.any():
                ax.plot(BIN_HOURS[valid], emp_pm[valid], 'o', color='#d62728', ms=3, alpha=0.8, label='Observed bin median')
            ax.set_title(f"Sensor {res['sensor_id']}  RMSE={res['rmse']:.3f}  R²={res['r2']:.3f}", fontsize=9)
            ax.set_xticks([0,6,12,18,24])
            ax.set_xlabel('Hour', fontsize=8)
            ax.set_ylabel('PM2.5 (µg/m³)', fontsize=8)
        for ax in axes[len(loso_results):]:
            ax.set_visible(False)
        axes[0].legend(fontsize=7)
        fig.suptitle('Holdout Sensors: PM level profiles (B(x) × S(x,k))', y=1.02)
        fig.tight_layout()
        fig.savefig(OUT_DIR / 'holdout_level_profiles_true_out_of_sample.pdf', dpi=200, bbox_inches='tight')
        plt.close(fig)

    # Summary bars
    if not args.only_binmedian_plot:
        fig, ax = plt.subplots(figsize=(max(6, 0.6*n), 4))
        rmses = [r['rmse'] for r in loso_results]
        maes  = [r['mae'] for r in loso_results]
        r2s   = [r['r2']  for r in loso_results]
        labels = [f"H{r['sensor_id']}" for r in loso_results]
        x = np.arange(n)
        ax.bar(x - 0.2, rmses, 0.4, label='RMSE', color='#1f77b4', alpha=0.8)
        ax.bar(x + 0.2, maes,  0.4, label='MAE',  color='#ff7f0e', alpha=0.8)
        ax.plot(x, r2s, 'k^', label='R²', markersize=6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel('Error (log scale)')
        ax.set_title('Holdout CV: Prediction error at sensors excluded from training\n(no refit)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / 'holdout_error_summary_true_out_of_sample.pdf', dpi=200, bbox_inches='tight')
        plt.close(fig)

    # Shape-only error summary (bin-median normalised): RMSE / MAE / R² only
    if not args.only_binmedian_plot:
        fig, ax = plt.subplots(figsize=(max(6, 0.6*n), 4))
        shape_rmse = []
        shape_mae = []
        shape_r2 = []
        labels = []
        for res in loso_results:
            pred = res['shape_mean']
            obs = res['emp_shape_bin_median']
            valid = ~np.isnan(obs)
            if not valid.any():
                shape_rmse.append(np.nan)
                shape_mae.append(np.nan)
                shape_r2.append(np.nan)
            else:
                obs_v = obs[valid]
                pred_v = pred[valid]
                shape_rmse.append(float(np.sqrt(np.mean((obs_v - pred_v)**2))))
                shape_mae.append(float(np.mean(np.abs(obs_v - pred_v))))
                shape_r2.append(float(r2_score(obs_v, pred_v)) if len(obs_v) > 1 else np.nan)
            labels.append(f"H{res['sensor_id']}")
        x = np.arange(n)
        ax.bar(x - 0.25, shape_rmse, 0.25, label='RMSE (shape)', color='#1f77b4', alpha=0.8)
        ax.bar(x,        shape_mae,  0.25, label='MAE (shape)',  color='#ff7f0e', alpha=0.8)
        ax.bar(x + 0.25, shape_r2,   0.25, label='R² (shape)',   color='#2ca02c', alpha=0.8)
        ax.set_ylabel('Shape metrics (norm units / R²)')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=8)
        ax.set_title('Holdout CV: Shape-only error (bin-median normalised)')
        fig.tight_layout()
        fig.savefig(OUT_DIR / 'holdout_shape_error_summary_true_out_of_sample.pdf', dpi=200, bbox_inches='tight')
        plt.close(fig)

    print("Holdout summary:")
    for res in loso_results:
        print(
            f"  Sensor {res['sensor_id']}: RMSE={res['rmse']:.3f}  MAE={res['mae']:.3f}  R2={res['r2']:.3f}  "
            f"NoBias RMSE={res['rmse_nb']:.3f} R2={res['r2_nb']:.3f}  "
            f"ShapeRMSE={res.get('shape_rmse', float('nan')):.3f}  "
            f"ShapeMAE={res.get('shape_mae', float('nan')):.3f}  "
            f"ShapeR2={res.get('shape_r2', float('nan')):.3f}"
        )

    shape_rmses = np.array([r.get("shape_rmse", np.nan) for r in loso_results], dtype=float)
    shape_maes = np.array([r.get("shape_mae", np.nan) for r in loso_results], dtype=float)
    shape_r2s = np.array([r.get("shape_r2", np.nan) for r in loso_results], dtype=float)

    def _nan_summary(x: np.ndarray) -> dict:
        return {
            "mean": float(np.nanmean(x)) if np.isfinite(x).any() else float("nan"),
            "median": float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan"),
        }

    agg = {
        "length_scale": float(LENGTH_SCALE),
        "mu_baseline_mode": str(args.mu_baseline_mode),
        "n_holdouts": int(len(loso_results)),
        "shape_rmse": _nan_summary(shape_rmses),
        "shape_mae": _nan_summary(shape_maes),
        "shape_r2": _nan_summary(shape_r2s),
    }

    print(
        "Aggregate shape metrics (centered log profile): "
        f"RMSE mean={agg['shape_rmse']['mean']:.4f} median={agg['shape_rmse']['median']:.4f} | "
        f"MAE mean={agg['shape_mae']['mean']:.4f} median={agg['shape_mae']['median']:.4f} | "
        f"R2 mean={agg['shape_r2']['mean']:.4f} median={agg['shape_r2']['median']:.4f}"
    )

    if single_holdout_mode:
        out_summary = project / "best_holdout_shape_metrics_summary.json"
    else:
        out_summary = project / "holdout_shape_metrics_summary.json"
    out_summary.write_text(json.dumps(agg, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {out_summary}")

    # Persist per-sensor holdout metrics so downstream plots can draw whiskers
    # without relying on parsing stdout logs.
    per_sensor_rows = []
    for res in loso_results:
        per_sensor_rows.append(
            {
                "sensor_id": int(res["sensor_id"]),
                "rmse": float(res["rmse"]),
                "mae": float(res["mae"]),
                "r2": float(res["r2"]),
                "rmse_nb": float(res["rmse_nb"]),
                "mae_nb": float(res["mae_nb"]),
                "r2_nb": float(res["r2_nb"]),
                "shape_rmse": float(res.get("shape_rmse", np.nan)),
                "shape_mae": float(res.get("shape_mae", np.nan)),
                "shape_r2": float(res.get("shape_r2", np.nan)),
                "r2_shape_norm": float(res.get("r2_shape_norm", np.nan)),
                "r2_shape_binmedian_norm": float(res.get("r2_shape_binmedian_norm", np.nan)),
                "nearest_train_idx": int(res.get("nearest_train_idx", -1)),
            }
        )
    if single_holdout_mode:
        out_per_sensor = project / "best_holdout_shape_metrics_per_sensor.csv"
    else:
        out_per_sensor = project / "holdout_shape_metrics_per_sensor.csv"
    pd.DataFrame(per_sensor_rows).to_csv(out_per_sensor, index=False)
    print(f"Wrote {out_per_sensor}")

    if args.only_binmedian_plot:
        if single_holdout_mode:
            print("Saved best-holdout bin-median shape plot (.pdf + .png) plus best_holdout_shape_metrics_per_sensor.csv")
        else:
            print("Saved holdout_shape_profiles_binmedian_norm_true_out_of_sample.pdf plus holdout_shape_metrics_per_sensor.csv")
    else:
        print(
            f"Saved {FN_DIURNAL_LOGPM25_LEVEL_OOS} and {FN_DIURNAL_LOGS_SHAPE_ONLY} "
            f"(and legacy diurnal filenames) plus holdout_error_summary_true_out_of_sample.pdf"
        )


if __name__ == "__main__":
    main()

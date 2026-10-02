#!/usr/bin/env python
"""
Compute basic Gelman-Rubin R-hat summaries from multiple Phase 2 runs.

Usage:
  python tools/compute_rhat_phase2.py --run-dirs /path/to/runA /path/to/runB ...

Assumptions:
  - Each run dir contains exactly one NPZ at traces/*_phase2_posterior.npz
  - All runs share the same draw shapes.

Outputs a small table of mean/max R-hat for key arrays.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np


def find_npz(run_dir: Path) -> Path:
    traces = run_dir / "traces"
    if not traces.is_dir():
        raise FileNotFoundError(f"missing traces dir: {traces}")
    npzs = list(traces.glob("*_phase2_posterior.npz"))
    if len(npzs) != 1:
        raise FileNotFoundError(f"expected 1 npz in {traces}, found {len(npzs)}")
    return npzs[0]


def _rhat(chains: np.ndarray) -> np.ndarray:
    # chains shape: (m, n, ...)
    m, n = chains.shape[0], chains.shape[1]
    chain_means = chains.mean(axis=1)
    overall_mean = chain_means.mean(axis=0)
    B = n * ((chain_means - overall_mean) ** 2).sum(axis=0) / (m - 1)
    W = ((chains - chain_means[:, None, ...]) ** 2).sum(axis=1) / (m * (n - 1))
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    rhat = np.sqrt(var_hat / W)
    return rhat


def summarize(name: str, arr: np.ndarray):
    flat = arr.reshape(-1)
    print(f"{name:20s} mean={flat.mean():.3f}  max={flat.max():.3f}")


def main(run_dirs: list[Path]):
    npz_paths = [find_npz(rd) for rd in run_dirs]
    arrays = [np.load(p) for p in npz_paths]
    keys = [
        "sigma_s_draws",
        "tau_draws",
        "mu_tilde_draws",
        "r_sensor_draws",
        "b_draws",
    ]
    for key in keys:
        if any(key not in arr for arr in arrays):
            print(f"[skip] {key} not present in all runs")
            continue
        stacked = np.stack([arr[key] for arr in arrays], axis=0)  # (chains, draws, ...)
        rhat = _rhat(stacked)
        summarize(key, rhat)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True, help="Phase2 run roots")
    args = ap.parse_args()
    main([Path(p) for p in args.run_dirs])

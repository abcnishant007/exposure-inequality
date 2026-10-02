#!/usr/bin/env python
"""
Stack multiple Phase 2 NPZ traces and emit R-hat summaries plus simple trace plots.

Usage:
  python tools/stack_phase2_traces.py --run-dirs runA runB runC --out-root ./combined_sa

Outputs (written to out-root):
  combined_phase2_rhat.txt   - mean/max R-hat per array
    plots/trace_{key}.pdf      - trace plots for scalar summaries of key arrays
    plots/rhat_by_k_tau.pdf    - optional R-hat by k for tau if k-dim present
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def find_npz(run_dir: Path) -> Path:
    traces = run_dir / "traces"
    npzs = list(traces.glob("*_phase2_posterior.npz"))
    if len(npzs) != 1:
        raise FileNotFoundError(f"expected 1 npz in {traces}, found {len(npzs)}")
    return npzs[0]


def _rhat(chains: np.ndarray) -> np.ndarray:
    m, n = chains.shape[0], chains.shape[1]
    chain_means = chains.mean(axis=1)
    overall_mean = chain_means.mean(axis=0)
    B = n * ((chain_means - overall_mean) ** 2).sum(axis=0) / (m - 1)
    W = ((chains - chain_means[:, None, ...]) ** 2).sum(axis=1) / (m * (n - 1))
    var_hat = ((n - 1) / n) * W + (1 / n) * B
    return np.sqrt(var_hat / W)


def summarize_rhat(key: str, arr: np.ndarray) -> str:
    flat = arr.reshape(-1)
    return f"{key:20s} mean={flat.mean():.3f}  max={flat.max():.3f}"


def trace_plot(key: str, chains: np.ndarray, out_dir: Path):
    # For large arrays, plot a simple scalar summary per draw: mean over all dims.
    series = chains.reshape(chains.shape[0], chains.shape[1], -1).mean(axis=2)  # (chains, draws)
    fig, ax = plt.subplots(figsize=(6, 3))
    for i in range(series.shape[0]):
        ax.plot(series[i], label=f"chain {i+1}", lw=0.7)
    ax.set_title(f"{key} (mean over elements)")
    ax.set_xlabel("draw")
    ax.legend(frameon=False)
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"trace_{key}.pdf", dpi=200)
    plt.close(fig)


def maybe_plot_rhat_by_k(key: str, rhat: np.ndarray, out_dir: Path):
    if key != "tau_draws":
        return
    if rhat.ndim != 1:
        # Expect shape (K,) after flattening chains/draws
        if rhat.ndim > 1:
            rhat1d = rhat.reshape(-1)
        else:
            return
    else:
        rhat1d = rhat
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(1, rhat1d.size + 1), rhat1d, marker="o", lw=0.7)
    ax.set_xlabel("k (1-based)")
    ax.set_ylabel("R-hat")
    ax.set_title("R-hat by k (tau)")
    plt.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "rhat_by_k_tau.pdf", dpi=200)
    plt.close(fig)


def main(run_dirs: list[Path], out_root: Path):
    out_root.mkdir(parents=True, exist_ok=True)
    npz_paths = [find_npz(rd) for rd in run_dirs]
    arrays = [np.load(p) for p in npz_paths]
    keys = ["sigma_s_draws", "tau_draws", "mu_tilde_draws", "r_sensor_draws", "b_draws"]

    lines = []
    for key in keys:
        if any(key not in arr for arr in arrays):
            continue
        stacked = np.stack([arr[key] for arr in arrays], axis=0)  # (chains, draws, ...)
        rhat = _rhat(stacked)
        lines.append(summarize_rhat(key, rhat))
        trace_plot(key, stacked, out_root / "plots")
        maybe_plot_rhat_by_k(key, rhat, out_root / "plots")

    (out_root / "combined_phase2_rhat.txt").write_text("\n".join(lines))
    print("Wrote", out_root / "combined_phase2_rhat.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dirs", nargs="+", required=True, help="Phase2 run roots")
    ap.add_argument("--out-root", required=True, help="Output directory for summaries/plots")
    args = ap.parse_args()
    main([Path(p) for p in args.run_dirs], Path(args.out_root))

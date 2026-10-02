#!/usr/bin/env python
"""
Compare uncertainty surfaces (quantile widths) between two runs.

Computes Pearson correlation of p95-p05 (and p75-p25 if present) across grid cells
for a chosen time bin or for the time-collapsed mean surface.

Usage:
  python tools/compare_surface_uncertainty.py --run-a /path/to/runA --run-b /path/to/runB --bin 20
  python tools/compare_surface_uncertainty.py --run-a ... --run-b ... --mean
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def load_quantiles(run_dir: Path, use_mean: bool, k: int | None) -> pd.DataFrame:
    csv_path = run_dir / "outputs" / f"{run_dir.name}_dynamic_surface_quantiles.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path)
    # If there's no 'k' column, assume the file is already a single (mean or fixed-bin) surface.
    if "k" not in df.columns:
        if not use_mean and k is not None:
            raise ValueError("quantiles file has no 'k' column; use --mean for collapsed surfaces")
        return df
    if use_mean:
        return df[df["k"] == "mean"]
    if k is None:
        raise ValueError("bin k must be provided when not using --mean")
    return df[df["k"] == k]


def corr_width(df_a: pd.DataFrame, df_b: pd.DataFrame, p_hi: str, p_lo: str) -> float:
    wa = df_a[p_hi] - df_a[p_lo]
    wb = df_b[p_hi] - df_b[p_lo]
    return pearsonr(wa, wb)


def main(run_a: Path, run_b: Path, k: int | None, use_mean: bool):
    df_a = load_quantiles(run_a, use_mean, k)
    df_b = load_quantiles(run_b, use_mean, k)
    shared_cols = set(df_a.columns) & set(df_b.columns)
    required = {"pm25_p05", "pm25_p95"}
    if not required.issubset(shared_cols):
        raise ValueError("quantile columns pm25_p05/pm25_p95 not found in both runs")
    print(f"Comparing widths for {'mean' if use_mean else f'bin {k}'}")
    w95_corr, w95_p = corr_width(df_a, df_b, "pm25_p95", "pm25_p05")
    print(f"corr(p95-p05): {w95_corr:.3f}  p={w95_p:.3g}")
    if {"pm25_p75", "pm25_p25"}.issubset(shared_cols):
        w75_corr, w75_p = corr_width(df_a, df_b, "pm25_p75", "pm25_p25")
        print(f"corr(p75-p25): {w75_corr:.3f}  p={w75_p:.3g}")

    # Save scatter plots
    wa = df_a["pm25_p95"] - df_a["pm25_p05"]
    wb = df_b["pm25_p95"] - df_b["pm25_p05"]
    title_suffix = "mean" if use_mean else f"k{k}"
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(wa, wb, s=3, alpha=0.3)
    ax.set_xlabel(f"{run_a.name} width p95-p05")
    ax.set_ylabel(f"{run_b.name} width p95-p05")
    ax.set_title(f"Width corr (p95-p05) {title_suffix}: r={w95_corr:.3f}")
    plt.tight_layout()
    fig.savefig(f"width_corr_p95_p05_{title_suffix}.pdf", dpi=200)
    plt.close(fig)

    # Spatial heatmap of width differences (run A - run B)
    def plot_map(diff_series, label, fname):
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(df_a["lon"], df_a["lat"], c=diff_series, s=4, cmap="coolwarm", alpha=0.9)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.set_title(f"{label} (A-B) {title_suffix}")
        plt.colorbar(sc, ax=ax, label=label)
        plt.tight_layout()
        fig.savefig(fname, dpi=200)
        plt.close(fig)

    plot_map(wa - wb, "width p95-p05", f"width_map_p95_p05_diff_{title_suffix}.pdf")
    # Individual width maps
    plot_map(wa, f"{run_a.name} width p95-p05", f"width_map_p95_p05_{run_a.name}_{title_suffix}.pdf")
    plot_map(wb, f"{run_b.name} width p95-p05", f"width_map_p95_p05_{run_b.name}_{title_suffix}.pdf")

    if {"pm25_p75", "pm25_p25"}.issubset(shared_cols):
        wa = df_a["pm25_p75"] - df_a["pm25_p25"]
        wb = df_b["pm25_p75"] - df_b["pm25_p25"]
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(wa, wb, s=3, alpha=0.3)
        ax.set_xlabel(f"{run_a.name} width p75-p25")
        ax.set_ylabel(f"{run_b.name} width p75-p25")
        ax.set_title(f"Width corr (p75-p25) {title_suffix}: r={w75_corr:.3f}")
        plt.tight_layout()
        fig.savefig(f"width_corr_p75_p25_{title_suffix}.pdf", dpi=200)
        plt.close(fig)
        plot_map(wa - wb, "width p75-p25", f"width_map_p75_p25_diff_{title_suffix}.pdf")
        plot_map(wa, f"{run_a.name} width p75-p25", f"width_map_p75_p25_{run_a.name}_{title_suffix}.pdf")
        plot_map(wb, f"{run_b.name} width p75-p25", f"width_map_p75_p25_{run_b.name}_{title_suffix}.pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-a", required=True, help="Run root A")
    ap.add_argument("--run-b", required=True, help="Run root B")
    ap.add_argument("--bin", type=int, help="Time bin k (0-based) to compare")
    ap.add_argument("--mean", action="store_true", help="Use time-collapsed mean surface")
    args = ap.parse_args()
    main(Path(args.run_a), Path(args.run_b), args.bin, args.mean)

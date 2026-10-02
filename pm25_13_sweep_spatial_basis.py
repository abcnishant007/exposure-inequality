#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List


def _parse_n_values(args: argparse.Namespace) -> List[int]:
    default_values = [2, 5, 10, 15, 20, 25, 30, 35]
    if args.n_values:
        return [int(x) for x in args.n_values.split(",") if x.strip()]
    if args.n_min is None and args.n_max is None:
        return default_values
    if args.n_min is None or args.n_max is None:
        raise SystemExit(
            "Provide either --n-values, both --n-min and --n-max, or neither to use the default list."
        )
    if args.n_min < 1 or args.n_max < 1:
        raise SystemExit("--n-min/--n-max must be >= 1.")
    if args.n_min > args.n_max:
        raise SystemExit("--n-min must be <= --n-max.")
    return list(range(int(args.n_min), int(args.n_max) + 1))


def _run(cmd: List[str], cwd: Path, env: dict) -> None:
    print(f"\n$ (cd {cwd}) {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _get_s_train(project: Path, holdout_file: Path, env: dict) -> int:
    code = (
        "from pm25_data_loader import load_data\n"
        "df_dynamic, df_sensors, df_static, sensor_order, S, K = load_data(verbose=False)\n"
        "print(S)\n"
    )
    out = subprocess.check_output(
        [sys.executable, "-c", code],
        cwd=str(project),
        env=env,
        text=True,
    ).strip()
    try:
        return int(out.splitlines()[-1].strip())
    except Exception as exc:
        raise SystemExit(f"Could not parse S from load_data() output: {out}") from exc


def _plot(project: Path, rows: list[dict], title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print(f"Skipping plot (matplotlib not available): {exc}")
        return

    xs = np.array([r["n_spatial_basis"] for r in rows], dtype=int)
    rmse_med = np.array([r["shape_rmse_median"] for r in rows], dtype=float)
    mae_med = np.array([r["shape_mae_median"] for r in rows], dtype=float)
    r2_med = np.array([r["shape_r2_median"] for r in rows], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    l1 = ax.plot(xs, rmse_med, "o-", label="Shape RMSE (median)")
    l2 = ax.plot(xs, mae_med, "o-", label="Shape MAE (median)")

    ax_r2 = ax.twinx()
    tab_green = "tab:green"
    l3 = ax_r2.plot(xs, r2_med, "o-", label="Shape R² (median)", color=tab_green)

    ax.set_xlabel("N_SPATIAL_BASIS (PM25_N_SPATIAL_BASIS)")
    ax.set_ylabel("RMSE / MAE")
    ax_r2.set_ylabel("R²", color=tab_green)
    ax_r2.tick_params(axis="y", colors=tab_green)

    ax.grid(True, alpha=0.3)

    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="best")

    ax.set_xticks(xs)
    ax.set_xticklabels([str(int(x)) for x in xs], rotation=0)
    ax.set_title(title)
    fig.tight_layout()

    plot_path = project / "spatial_basis_sweep_holdout_shape_metrics.pdf"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {plot_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep PM25_N_SPATIAL_BASIS and report aggregate holdout shape metrics."
    )
    parser.add_argument("--project", default="bayesian_fusion/bayesian_fusion_Boston", help="Scenario folder")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Skip running the sweep; read the existing JSON and (re)generate the PNG plot.",
    )
    parser.add_argument(
        "--json",
        type=str,
        help="Optional path to an existing sweep JSON (defaults to <project>/spatial_basis_sweep_holdout_shape_metrics.json).",
    )
    parser.add_argument(
        "--n-values",
        help="Comma-separated list of N_SPATIAL_BASIS values (e.g. 4,8,12,16). Overrides --n-min/--n-max.",
    )
    parser.add_argument("--n-min", type=int, help="Min N_SPATIAL_BASIS (inclusive) when using range sweep")
    parser.add_argument("--n-max", type=int, help="Max N_SPATIAL_BASIS (inclusive) when using range sweep")
    parser.add_argument("--holdout-count", type=int, default=10, help="Holdout sensor count (if generating list)")
    parser.add_argument("--seed", type=int, default=55, help="Seed for holdout list (if generating list)")
    parser.add_argument(
        "--regen-holdouts",
        action="store_true",
        help="Regenerate holdout_sensors.txt before sweeping",
    )
    parser.add_argument(
        "--mu-baseline-mode",
        choices=["grid", "nearest_sensor_posterior", "grid_plus_nearest_delta", "grid_plus_basis_delta"],
        default="grid",
        help="Pass-through to pm25_08_holdout_predict.py",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    project = (repo_root / args.project).resolve()
    if not project.exists():
        raise SystemExit(f"Project folder not found: {project}")

    out_path = Path(args.json).expanduser().resolve() if args.json else (project / "spatial_basis_sweep_holdout_shape_metrics.json")
    if args.plot_only:
        if not out_path.exists():
            raise SystemExit(f"Sweep JSON not found: {out_path}")
        out = json.loads(out_path.read_text())
        rows = out.get("results", [])
        if not rows:
            raise SystemExit(f"No results found in JSON: {out_path}")
        print(f"Loaded sweep results: {out_path}")
        _plot(
            project,
            rows,
            title=f"Holdout shape metrics (median) vs N_SPATIAL_BASIS ({args.project}, mu_baseline={out.get('mu_baseline_mode','?')})",
        )
        return

    holdout_file = project / "holdout_sensors.txt"
    if args.regen_holdouts or not holdout_file.exists():
        _run(
            [
                sys.executable,
                str(repo_root / "pm25_holdout_train_test.py"),
                "--project",
                str(project),
                "--holdout-count",
                str(args.holdout_count),
                "--seed",
                str(args.seed),
            ],
            cwd=repo_root,
            env=os.environ.copy(),
        )

    if not holdout_file.exists():
        raise SystemExit(f"Holdout list missing: {holdout_file}")

    base_env = os.environ.copy()
    base_env["HOLDOUT_SENSORS_FILE"] = str(holdout_file)
    s_train = _get_s_train(project, holdout_file, base_env)
    print(f"S_train (after holdout removal): {s_train}")

    n_values = _parse_n_values(args)
    n_values = sorted(set(n_values))
    n_values_ok = [n for n in n_values if 1 <= n <= s_train]
    n_values_bad = [n for n in n_values if n not in n_values_ok]
    if n_values_bad:
        print(f"Skipping invalid N_SPATIAL_BASIS values (must be 1..{s_train}): {n_values_bad}")
    if not n_values_ok:
        raise SystemExit("No valid N_SPATIAL_BASIS values to sweep after applying guardrails.")

    print(f"N_SPATIAL_BASIS sweep values ({len(n_values_ok)}): {n_values_ok}")
    print(f"Using holdouts: {holdout_file}")

    rows = []
    for n_spatial in n_values_ok:
        env = os.environ.copy()
        env["PM25_N_SPATIAL_BASIS"] = str(n_spatial)
        env["HOLDOUT_SENSORS_FILE"] = str(holdout_file)

        _run([sys.executable, "pm25_01_fusion.py"], cwd=project, env=env)
        _run(
            [
                sys.executable,
                str(repo_root / "pm25_08_holdout_predict.py"),
                "--project",
                str(project),
                "--mu-baseline-mode",
                args.mu_baseline_mode,
            ],
            cwd=project,
            env=env,
        )

        summary_path = project / "holdout_shape_metrics_summary.json"
        if not summary_path.exists():
            raise SystemExit(f"Expected summary JSON missing: {summary_path}")
        summary = json.loads(summary_path.read_text())

        row = {
            "n_spatial_basis": int(n_spatial),
            "shape_rmse_mean": summary["shape_rmse"]["mean"],
            "shape_rmse_median": summary["shape_rmse"]["median"],
            "shape_mae_mean": summary["shape_mae"]["mean"],
            "shape_mae_median": summary["shape_mae"]["median"],
            "shape_r2_mean": summary["shape_r2"]["mean"],
            "shape_r2_median": summary["shape_r2"]["median"],
        }
        rows.append(row)

        print(
            f"N={n_spatial:3d} | "
            f"shapeRMSE mean/med={row['shape_rmse_mean']:.4f}/{row['shape_rmse_median']:.4f} | "
            f"shapeMAE mean/med={row['shape_mae_mean']:.4f}/{row['shape_mae_median']:.4f} | "
            f"shapeR2 mean/med={row['shape_r2_mean']:.4f}/{row['shape_r2_median']:.4f}"
        )

    out = {
        "project": str(project),
        "holdout_file": str(holdout_file),
        "mu_baseline_mode": args.mu_baseline_mode,
        "s_train": int(s_train),
        "results": rows,
    }
    out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"\nWrote sweep results: {out_path}")

    _plot(
        project,
        rows,
        title=f"Holdout shape metrics (median) vs N_SPATIAL_BASIS ({args.project}, mu_baseline={args.mu_baseline_mode})",
    )


if __name__ == "__main__":
    main()

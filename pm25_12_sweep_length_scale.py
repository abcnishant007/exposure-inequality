#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List

import numpy as np


def _parse_h_values(args: argparse.Namespace) -> List[float]:
    if args.h_values:
        return [float(x) for x in args.h_values.split(",") if x.strip()]
    if args.h_min is None or args.h_max is None:
        raise SystemExit("Provide either --h-values or both --h-min and --h-max.")
    if args.n <= 1:
        raise SystemExit("--n must be >= 2 when using --h-min/--h-max.")
    return [float(x) for x in np.geomspace(args.h_min, args.h_max, args.n)]


def _run(cmd: List[str], cwd: Path, env: dict) -> None:
    print(f"\n$ (cd {cwd}) {' '.join(cmd)}")
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep PM25_LENGTH_SCALE (h) and report aggregate holdout shape metrics."
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
        help="Optional path to an existing sweep JSON (defaults to <project>/length_scale_sweep_holdout_shape_metrics.json).",
    )
    parser.add_argument(
        "--h-values",
        help="Comma-separated list of h values (e.g. 0.03,0.05,0.08,0.1). Overrides --h-min/--h-max.",
    )
    parser.add_argument("--h-min", type=float, help="Min h for geometric sweep (inclusive)")
    parser.add_argument("--h-max", type=float, help="Max h for geometric sweep (inclusive)")
    parser.add_argument("--n", type=int, default=10, help="Number of h values for geometric sweep")
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

    out_path = Path(args.json).expanduser().resolve() if args.json else (project / "length_scale_sweep_holdout_shape_metrics.json")
    if args.plot_only:
        if not out_path.exists():
            raise SystemExit(f"Sweep JSON not found: {out_path}")
        out = json.loads(out_path.read_text())
        rows = out.get("results", [])
        if not rows:
            raise SystemExit(f"No results found in JSON: {out_path}")
        print(f"Loaded sweep results: {out_path}")
        # fall through to plotting at the end of main()
    else:
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

        h_values = _parse_h_values(args)
        print(f"h sweep values ({len(h_values)}): {h_values}")
        print(f"Using holdouts: {holdout_file}")

        rows = []
        for h in h_values:
            env = os.environ.copy()
            env["PM25_LENGTH_SCALE"] = str(h)
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
            rows.append(
                {
                    "h": float(h),
                    "shape_rmse_mean": summary["shape_rmse"]["mean"],
                    "shape_rmse_median": summary["shape_rmse"]["median"],
                    "shape_mae_mean": summary["shape_mae"]["mean"],
                    "shape_mae_median": summary["shape_mae"]["median"],
                    "shape_r2_mean": summary["shape_r2"]["mean"],
                    "shape_r2_median": summary["shape_r2"]["median"],
                }
            )

            print(
                f"h={h:g} | "
                f"shapeRMSE mean/med={summary['shape_rmse']['mean']:.4f}/{summary['shape_rmse']['median']:.4f} | "
                f"shapeMAE mean/med={summary['shape_mae']['mean']:.4f}/{summary['shape_mae']['median']:.4f} | "
                f"shapeR2 mean/med={summary['shape_r2']['mean']:.4f}/{summary['shape_r2']['median']:.4f}"
            )

        out = {
            "project": str(project),
            "holdout_file": str(holdout_file),
            "mu_baseline_mode": args.mu_baseline_mode,
            "results": rows,
        }
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"\nWrote sweep results: {out_path}")

    # Plot summary curves (optional dependency)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping plot (matplotlib not available): {exc}")
        return

    hs = np.array([r["h"] for r in rows], dtype=float)
    rmse_med = np.array([r["shape_rmse_median"] for r in rows], dtype=float)
    mae_med = np.array([r["shape_mae_median"] for r in rows], dtype=float)
    r2_med = np.array([r["shape_r2_median"] for r in rows], dtype=float)

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    l1 = ax.plot(hs, rmse_med, "o-", label="Shape RMSE (median)")
    l2 = ax.plot(hs, mae_med, "o-", label="Shape MAE (median)")

    ax_r2 = ax.twinx()
    tab_green = "tab:green"
    l3 = ax_r2.plot(hs, r2_med, "o-", label="Shape R² (median)", color=tab_green)
    ax.set_xlabel("h (PM25_LENGTH_SCALE)")
    ax.set_ylabel("RMSE / MAE")
    ax_r2.set_ylabel("R²", color=tab_green)
    ax_r2.tick_params(axis="y", colors=tab_green)
    ax.grid(True, alpha=0.3)
    lines = l1 + l2 + l3
    labels = [ln.get_label() for ln in lines]
    ax.legend(lines, labels, loc="best")

    ax.set_xscale("log")
    ax.set_xticks(hs)
    ax.set_xticklabels([f"{h:.4f}" for h in hs], rotation=45, ha="right")

    ax.set_title(f"Holdout shape metrics (median) vs h ({args.project}, mu_baseline={args.mu_baseline_mode})")
    fig.tight_layout()
    plot_path = project / "length_scale_sweep_holdout_shape_metrics.pdf"
    fig.savefig(plot_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plot: {plot_path}")


if __name__ == "__main__":
    main()

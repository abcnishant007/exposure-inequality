#!/usr/bin/env python3
"""
Train/test split by sensors for true spatial holdout.
- Picks holdout sensors (random sample or provided file).
- Writes the holdout list to a file.
- Optionally runs the project's pipeline with HOLDOUT_SENSORS_FILE set, so the model
  is trained *excluding* those sensors and all downstream diagnostics run on the
  reduced training set. You can later evaluate the trained model on the holdout
  sensors by writing a separate evaluation script (not included here).

Usage examples:
  python pm25_holdout_train_test.py --project bayesian_fusion_after_20260223_keep_median_only_Oklahoma \
      --holdout-count 2 --seed 42

  python pm25_holdout_train_test.py --project bayesian_fusion_after_20260223_keep_median_only_Boston \
      --holdout-file my_holdouts.txt --run-pipeline

Notes:
- This script does not refit inside multiple folds; it performs a single split.
- Pipeline refit can be very slow; use --run-pipeline only when ready.
"""

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


def _cap_holdout_count(n_sensors: int, holdout_count: int, min_train_sensors: int) -> Tuple[int, str]:
    """Cap holdout_count so at least min_train_sensors remain for training."""
    if n_sensors <= 0:
        return 0, "no sensors available"
    if min_train_sensors < 1:
        min_train_sensors = 1
    max_holdout = max(n_sensors - min_train_sensors, 1)
    if holdout_count > max_holdout:
        return max_holdout, f"capped holdout_count from {holdout_count} to {max_holdout} (n_sensors={n_sensors}, min_train_sensors={min_train_sensors})"
    if holdout_count < 1:
        return 1, f"raised holdout_count from {holdout_count} to 1 (n_sensors={n_sensors})"
    return holdout_count, ""


def choose_holdouts(project: Path, holdout_count: int, seed: int, *, min_train_sensors: int) -> List[int]:
    sys.path.insert(0, str(project))
    import pm25_data_loader  # type: ignore

    df_dynamic, df_sensors, df_static, sensor_order, S, K = pm25_data_loader.load_data(verbose=True)
    sensors = sorted(df_sensors["sensor_index"].unique().tolist())
    capped, msg = _cap_holdout_count(len(sensors), int(holdout_count), int(min_train_sensors))
    if msg:
        print(f"Warning: {msg}")
    holdout_count = capped
    random.seed(seed)
    return sorted(random.sample(sensors, holdout_count))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, help="Path to scenario folder (contains run_pipeline.sh)")
    parser.add_argument("--holdout-count", type=int, default=8, help="Number of sensors to hold out (ignored if --holdout-file given)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for sampling holdouts")
    parser.add_argument("--holdout-file", type=Path, help="Optional file with sensor_index values (one per line) to hold out")
    parser.add_argument(
        "--min-train-sensors",
        type=int,
        default=3,
        help="Ensure at least this many sensors remain in training (caps --holdout-count if needed).",
    )
    parser.add_argument("--run-pipeline", action="store_true", help="Run run_pipeline.sh with HOLDOUT_SENSORS_FILE set")
    args = parser.parse_args()

    project = Path(args.project).resolve()
    if not (project / "run_pipeline.sh").exists():
        raise SystemExit(f"run_pipeline.sh not found in {project}")

    # Many per-city `pm25_data_loader.py` implementations load inputs via relative paths.
    # Ensure those reads resolve against the project directory even when this script is
    # invoked from outside the project.
    os.chdir(project)

    if args.holdout_file:
        holdouts = [int(line.strip()) for line in args.holdout_file.read_text().splitlines() if line.strip()]
    else:
        holdouts = choose_holdouts(project, args.holdout_count, args.seed, min_train_sensors=args.min_train_sensors)

    out_file = project / "holdout_sensors.txt"
    out_file.write_text("\n".join(str(s) for s in holdouts) + "\n")
    print(f"Wrote {len(holdouts)} holdout sensors to {out_file}")

    if args.run_pipeline:
        env = os.environ.copy()
        env["HOLDOUT_SENSORS_FILE"] = str(out_file)
        cmd = ["bash", "run_pipeline.sh"]
        print(f"Running pipeline in {project} with HOLDOUT_SENSORS_FILE={out_file} ...")
        subprocess.run(cmd, cwd=project, env=env, check=True)


if __name__ == "__main__":
    main()

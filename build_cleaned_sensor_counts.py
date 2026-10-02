#!/usr/bin/env python3
"""Build cleaned sensor counts per MSA from purpleair_history cleaned CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build cleaned sensor counts per MSA.")
    p.add_argument(
        "--history-dir",
        default="EDA_PM25/data/purpleair_history",
        help="Directory containing <msa-slug>_<YYYY-MM> subfolders.",
    )
    p.add_argument(
        "--samples-csv",
        default="EDA_PM25/data/msa_sensor_samples.csv",
        help="CSV with msa_slug and % sampled (unfiltered counts).")
    p.add_argument(
        "--out",
        default="EDA_PM25/data/cleaned_sensor_counts.csv",
        help="Output CSV path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    hist_dir = Path(args.history_dir)
    out_path = Path(args.out)
    samples_path = Path(args.samples_csv)

    if not hist_dir.exists():
        raise SystemExit(f"History dir not found: {hist_dir}")

    samples = None
    if samples_path.exists():
        samples = pd.read_csv(samples_path)
        if "% sampled" in samples.columns:
            samples = samples.rename(columns={"% sampled": "sample_pct"})

    rows = []
    for folder in sorted(hist_dir.iterdir()):
        if not folder.is_dir():
            continue
        if "_" not in folder.name:
            continue
        slug, month = folder.name.rsplit("_", 1)
        cleaned_files = sorted(folder.glob("cleaned_*.csv"))
        if not cleaned_files:
            continue
        cleaned = cleaned_files[-1]
        try:
            df = pd.read_csv(cleaned, usecols=["sensor_index"])
        except Exception:
            continue
        count = df["sensor_index"].dropna().nunique()
        rows.append({
            "msa_slug": slug,
            "month": month,
            "cleaned_file": cleaned.name,
            "cleaned_sensor_count": int(count),
        })

    out = pd.DataFrame(rows)

    if samples is not None and "msa_slug" in samples.columns:
        out = out.merge(samples[["msa_slug", "sensor_count", "sample_pct"]], on="msa_slug", how="left")
        out = out.rename(columns={"sensor_count": "unfiltered_sensor_count"})

    cols = [
        "msa_slug",
        "month",
        "cleaned_file",
        "cleaned_sensor_count",
        "unfiltered_sensor_count",
        "sample_pct",
    ]
    cols = [c for c in cols if c in out.columns]
    out = out[cols]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} rows={len(out)}")


if __name__ == "__main__":
    main()

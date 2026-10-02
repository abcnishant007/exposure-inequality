#!/usr/bin/env python3
"""Build a combined GeoJSON of sensors present in cleaned PurpleAir data.

For each folder in EDA_PM25/data/purpleair_history/<msa-slug>_<YYYY-MM>/,
read cleaned_*.csv, collect sensor_index values, and join against the
corresponding MSA sensor GeoJSON to pull lat/lon. Output a combined
FeatureCollection with msa_slug in properties.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, Set

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine cleaned PurpleAir sensors into a single GeoJSON.")
    p.add_argument(
        "--history-dir",
        default="EDA_PM25/data/purpleair_history",
        help="Directory containing <msa-slug>_<YYYY-MM> subfolders.",
    )
    p.add_argument(
        "--geojson-dir",
        default="EDA_PM25/data/msa_sensor_list_purple_air",
        help="Directory containing MSA sensor list GeoJSONs.",
    )
    p.add_argument(
        "--out",
        default="EDA_PM25/data/cleaned_sensors_by_msa.geojson",
        help="Output GeoJSON path.",
    )
    return p.parse_args()


def iter_cleaned_files(history_dir: Path) -> Iterable[Path]:
    for folder in history_dir.iterdir():
        if not folder.is_dir():
            continue
        for cleaned in folder.glob("cleaned_*.csv"):
            yield cleaned


def folder_to_slug(folder: Path) -> str | None:
    name = folder.name
    if "_" not in name:
        return None
    return name.rsplit("_", 1)[0]


def main() -> None:
    args = parse_args()
    history_dir = Path(args.history_dir)
    geojson_dir = Path(args.geojson_dir)
    out_path = Path(args.out)

    if not history_dir.exists():
        raise SystemExit(f"History dir not found: {history_dir}")
    if not geojson_dir.exists():
        raise SystemExit(f"GeoJSON dir not found: {geojson_dir}")

    # Collect sensor indices per msa slug
    sensors_by_slug: Dict[str, Set[int]] = {}
    for cleaned in iter_cleaned_files(history_dir):
        slug = folder_to_slug(cleaned.parent)
        if not slug:
            continue
        if not (geojson_dir / f"{slug}.geojson").exists():
            # skip folders that don't match a known MSA slug
            continue
        try:
            df = pd.read_csv(cleaned, usecols=["sensor_index"])
        except Exception:
            continue
        sensor_ids = set(int(x) for x in df["sensor_index"].dropna().unique())
        sensors_by_slug.setdefault(slug, set()).update(sensor_ids)

    features = []
    for slug, sensor_ids in sorted(sensors_by_slug.items()):
        geo_path = geojson_dir / f"{slug}.geojson"
        with geo_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for feat in data.get("features", []):
            props = feat.get("properties", {})
            sid = props.get("sensor_index")
            if sid is None:
                continue
            try:
                sid_int = int(sid)
            except Exception:
                continue
            if sid_int not in sensor_ids:
                continue
            new_props = dict(props)
            new_props["msa_slug"] = slug
            new_props["sensor_index"] = sid_int
            features.append({
                "type": "Feature",
                "geometry": feat.get("geometry"),
                "properties": new_props,
            })

    out = {
        "type": "FeatureCollection",
        "features": features,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f)

    print(f"Wrote {len(features)} features to {out_path}")


if __name__ == "__main__":
    main()

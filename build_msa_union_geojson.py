#!/usr/bin/env python3
"""Combine all MSA polygons into a single GeoJSON FeatureCollection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Combine all MSA GeoJSONs into one file.")
    p.add_argument(
        "--msa-dir",
        default="EDA_PM25/data/geojson_msa",
        help="Directory containing MSA GeoJSON files.",
    )
    p.add_argument(
        "--out",
        default="EDA_PM25/data/geojson_msa_all.geojson",
        help="Output GeoJSON path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    msa_dir = Path(args.msa_dir)
    out_path = Path(args.out)

    if not msa_dir.exists():
        raise SystemExit(f"MSA dir not found: {msa_dir}")

    features = []
    for geo in sorted(msa_dir.glob("*.geojson")):
        with geo.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for feat in data.get("features", []):
            props = dict(feat.get("properties", {}) or {})
            props.setdefault("msa_slug", geo.stem)
            features.append({
                "type": "Feature",
                "geometry": feat.get("geometry"),
                "properties": props,
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

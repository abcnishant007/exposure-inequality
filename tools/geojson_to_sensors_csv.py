"""Convert a Point GeoJSON FeatureCollection to a sensors CSV.

The output schema matches the Boston sensor list used elsewhere in this repo:
sensor_index, latitude, longitude, name
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _parse_points(features: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract point records with lon/lat, sensor_index, and name."""
    out: List[Dict[str, Any]] = []
    for feat in features:
        if not isinstance(feat, dict):
            continue
        geom = feat.get("geometry") or {}
        if geom.get("type") != "Point":
            continue
        coords = geom.get("coordinates") or []
        if not (isinstance(coords, list) and len(coords) >= 2):
            continue
        lon, lat = coords[:2]
        props = feat.get("properties") or {}
        sensor_idx = props.get("sensor_index", "")
        name = props.get("name", "")
        out.append(
            {
                "sensor_index": sensor_idx,
                "latitude": lat,
                "longitude": lon,
                "name": name,
            }
        )
    return out


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Export sensors CSV from a Point GeoJSON.")
    p.add_argument("--input", required=True, help="GeoJSON path (FeatureCollection of Points).")
    p.add_argument("--out", required=True, help="Destination CSV path.")
    args = p.parse_args(argv)

    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    geo = json.loads(in_path.read_text())
    features = geo.get("features") or []
    rows = _parse_points(features)
    if not rows:
        raise SystemExit("No Point features with coordinates found in the input GeoJSON.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sensor_index", "latitude", "longitude", "name"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Wrote {len(rows)} sensors to {out_path}")


if __name__ == "__main__":
    main()

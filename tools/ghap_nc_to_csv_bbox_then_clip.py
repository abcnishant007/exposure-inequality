"""
Fast GHAP NetCDF -> CSV using bbox prefilter, then polygon clip.

Workflow:
1) Compute bbox from the provided ROI GeoJSON.
2) Run ghap_nc_to_kepler_csv.py with the bbox (fast, memory-friendly).
3) Post-filter the resulting CSV to the exact ROI polygon.

Output columns: lon, lat, baseline[, date] (same as ghap_nc_to_kepler_csv).
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running from repo root without package install
import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse helpers from existing scripts to avoid divergence.
from tools.ghap_nc_to_kepler_csv import (  # type: ignore
    _roi_geojson_outer_rings,
    _points_in_polygons,
    main as nc_to_csv_main,
)


def _bbox_geojson_from_polygon(path: Path) -> Path:
    geo = json.loads(path.read_text())
    rings = _roi_geojson_outer_rings(geo)
    all_lon = np.concatenate([r[:, 0] for r in rings])
    all_lat = np.concatenate([r[:, 1] for r in rings])
    min_lon, max_lon = float(np.min(all_lon)), float(np.max(all_lon))
    min_lat, max_lat = float(np.min(all_lat)), float(np.max(all_lat))
    bbox_geo = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [min_lon, min_lat],
                            [max_lon, min_lat],
                            [max_lon, max_lat],
                            [min_lon, max_lat],
                            [min_lon, min_lat],
                        ]
                    ],
                },
                "properties": {"name": "bbox"},
            }
        ],
    }
    tmp = Path(tempfile.gettempdir()) / (path.stem + "_bbox_tmp.geojson")
    tmp.write_text(json.dumps(bbox_geo))
    return tmp


def _clip_csv_to_polygon(csv_path: Path, roi_path: Path, out_path: Path) -> None:
    df = pd.read_csv(csv_path)
    rings = _roi_geojson_outer_rings(json.loads(roi_path.read_text()))
    mask = _points_in_polygons(
        df["lon"].to_numpy(dtype=float),
        df["lat"].to_numpy(dtype=float),
        rings,
    )
    clipped = df[mask].reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clipped.to_csv(out_path, index=False)
    print(f"Clipped {csv_path.name}: kept {clipped.shape[0]} of {df.shape[0]} rows -> {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fast GHAP NetCDF to CSV: bbox prefilter, then polygon clip."
    )
    p.add_argument("--input", required=True, help="Path to GHAP .nc file or directory of .nc files.")
    p.add_argument("--roi-geojson", required=True, help="ROI polygon GeoJSON (Polygon/MultiPolygon).")
    p.add_argument("--out", required=True, help="Final clipped CSV path.")
    p.add_argument("--spacing-km", type=float, default=1.0, help="Subsampling spacing in km (default 1.0).")
    p.add_argument("--max-points", type=int, default=600000, help="Cap points before polygon clip (default 600k).")
    p.add_argument("--include-date", action="store_true", help="Include date column parsed from filenames.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    input_path = Path(args.input)
    roi_path = Path(args.roi_geojson)
    out_path = Path(args.out)

    # Step 1: bbox from polygon
    bbox_geo = _bbox_geojson_from_polygon(roi_path)

    # Step 2: fast extraction using bbox
    bbox_out = out_path.parent / (out_path.stem + "_bbox.csv")
    nc_to_csv_main(
        [
            "--input",
            str(input_path),
            "--out",
            str(bbox_out),
            "--roi-geojson",
            str(bbox_geo),
            "--spacing-km",
            str(args.spacing_km),
            "--max-points",
            str(args.max_points),
        ]
        + (["--include-date"] if args.include_date else [])
    )

    # Step 3: clip bbox CSV to polygon
    _clip_csv_to_polygon(bbox_out, roi_path, out_path)


if __name__ == "__main__":
    main()

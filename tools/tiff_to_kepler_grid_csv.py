"""
conda run -n odmatrix python tools/tiff_to_kepler_grid_csv.py --tiff /path/to/pm25_baseline.tif --out /tmp/city_grid_1km.csv --spacing-km 1.0
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import json

try:
    import numpy as np
    import pandas as pd
    import rasterio
except ModuleNotFoundError as e:  # pragma: no cover
    missing = str(e).split("No module named", 1)[-1].strip().strip("'").strip('"')
    raise SystemExit(
        f"Missing dependency {missing!r}. Run this script inside the project environment "
        f"(e.g., `conda run -n odmatrix python tools/tiff_to_kepler_grid_csv.py ...`)."
    ) from e


def _approx_m_per_deg_lon(lat_deg: float) -> float:
    return 111_320.0 * max(1e-6, math.cos(math.radians(lat_deg)))


def _approx_m_per_deg_lat(_: float) -> float:
    return 111_320.0


def _roi_geojson_outer_rings(geo: dict) -> list[np.ndarray]:
    """
    Extract outer rings from a GeoJSON object.

    Supports Feature / FeatureCollection / Polygon / MultiPolygon. Holes are ignored.
    Returns a list of arrays shaped (N, 2) with columns [lon, lat].
    """

    def _geom_from_obj(obj: dict) -> dict:
        if obj.get("type") == "Feature":
            return obj.get("geometry") or {}
        return obj

    def _geoms(obj: dict) -> list[dict]:
        obj = _geom_from_obj(obj)
        if obj.get("type") == "FeatureCollection":
            feats = obj.get("features") or []
            if not feats:
                raise ValueError("roi GeoJSON FeatureCollection has no features")
            out: list[dict] = []
            for feat in feats:
                geom = _geom_from_obj(feat or {})
                if geom:
                    out.append(geom)
            return out
        return [obj]

    rings: list[np.ndarray] = []
    for geom in _geoms(geo):
        gtype = geom.get("type")
        coords = geom.get("coordinates") or []
        if gtype == "Polygon":
            if coords and coords[0]:
                ring0 = np.asarray(coords[0], dtype=float)
                if ring0.ndim == 2 and ring0.shape[1] >= 2:
                    rings.append(ring0[:, :2])
        elif gtype == "MultiPolygon":
            for poly in coords:
                if poly and poly[0]:
                    ring0 = np.asarray(poly[0], dtype=float)
                    if ring0.ndim == 2 and ring0.shape[1] >= 2:
                        rings.append(ring0[:, :2])
        else:
            raise ValueError(f"roi GeoJSON must be Polygon/MultiPolygon; got type={gtype}")
    if not rings:
        raise ValueError("roi GeoJSON has no polygon coordinates")
    return rings


def _points_in_polygon(lon: np.ndarray, lat: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """
    Vectorized ray-casting point-in-polygon test for one outer ring.

    Returns a boolean mask for points strictly inside (boundary behavior is not guaranteed).
    """
    x = np.asarray(lon, dtype=float)
    y = np.asarray(lat, dtype=float)
    poly = np.asarray(ring, dtype=float)
    if poly.ndim != 2 or poly.shape[1] < 2:
        raise ValueError("Invalid polygon ring coordinates")

    px = poly[:, 0]
    py = poly[:, 1]
    if px.size < 3:
        return np.zeros_like(x, dtype=bool)
    if px[0] != px[-1] or py[0] != py[-1]:
        px = np.concatenate([px, px[:1]])
        py = np.concatenate([py, py[:1]])

    inside = np.zeros_like(x, dtype=bool)
    x0 = px[:-1]
    y0 = py[:-1]
    x1 = px[1:]
    y1 = py[1:]

    for xa, ya, xb, yb in zip(x0, y0, x1, y1):
        if yb == ya:
            continue
        crosses = (ya > y) != (yb > y)
        if not np.any(crosses):
            continue
        x_intersect = (xb - xa) * (y - ya) / (yb - ya) + xa
        inside ^= crosses & (x < x_intersect)
    return inside


def _points_in_polygons(lon: np.ndarray, lat: np.ndarray, rings: list[np.ndarray], *, chunk_size: int = 200_000) -> np.ndarray:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    out = np.zeros(lon.shape[0], dtype=bool)
    for start in range(0, lon.shape[0], int(chunk_size)):
        end = min(start + int(chunk_size), lon.shape[0])
        mask = np.zeros(end - start, dtype=bool)
        for ring in rings:
            mask |= _points_in_polygon(lon[start:end], lat[start:end], ring)
        out[start:end] = mask
    return out


def _choose_stride(
    *,
    ds: rasterio.io.DatasetReader,
    target_spacing_m: float,
) -> tuple[int, int]:
    t = ds.transform
    px_x = float(abs(t.a))
    px_y = float(abs(t.e))
    if px_x <= 0 or px_y <= 0:
        return (1, 1)

    crs = ds.crs
    if crs is not None and bool(getattr(crs, "is_projected", False)):
        stride_x = max(1, int(round(target_spacing_m / px_x)))
        stride_y = max(1, int(round(target_spacing_m / px_y)))
        return (stride_y, stride_x)

    # Geographic CRS: px sizes are in degrees. Convert to meters approximately at raster center latitude.
    center_row = ds.height // 2
    center_col = ds.width // 2
    _, center_lat = rasterio.transform.xy(t, center_row, center_col)
    m_per_deg_lon = _approx_m_per_deg_lon(float(center_lat))
    m_per_deg_lat = _approx_m_per_deg_lat(float(center_lat))
    stride_x = max(1, int(round(target_spacing_m / (px_x * m_per_deg_lon))))
    stride_y = max(1, int(round(target_spacing_m / (px_y * m_per_deg_lat))))
    return (stride_y, stride_x)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export a lon/lat grid CSV from a baseline GeoTIFF for Kepler.")
    p.add_argument("--tiff", required=True, help="Path to baseline GeoTIFF (e.g., weekday mean PM25).")
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--spacing-km", type=float, default=1.0, help="Approximate grid spacing in km (default: 1.0).")
    p.add_argument(
        "--lonlat-only",
        action="store_true",
        help="Write only lon/lat columns (omit raster pixel value). By default, includes a 'baseline' column.",
    )
    p.add_argument(
        "--roi-geojson",
        default="",
        help="Optional GeoJSON file path (Polygon/MultiPolygon/Feature/FeatureCollection) to filter points inside ROI.",
    )
    p.add_argument("--max-points", type=int, default=0, help="Optional cap on points (0 = no cap).")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    tiff_path = Path(args.tiff).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    spacing_km = float(args.spacing_km)
    if spacing_km <= 0:
        raise SystemExit("--spacing-km must be positive.")
    spacing_m = spacing_km * 1000.0

    with rasterio.open(tiff_path) as ds:
        arr = ds.read(1).astype(float)
        nodata = ds.nodata
        if nodata is not None:
            arr = np.where(arr == float(nodata), np.nan, arr)
        # Treat non-positive as invalid (matches bayesian_fusion baseline handling).
        arr = np.where(arr > 0, arr, np.nan)

        valid = np.isfinite(arr)
        if not bool(valid.any()):
            raise SystemExit("No valid pixels found (after nodata/non-positive filtering).")

        stride_y, stride_x = _choose_stride(ds=ds, target_spacing_m=spacing_m)

        rows, cols = np.where(valid)
        if stride_y > 1:
            rows_cols = (rows // stride_y, cols)
            keep = (rows_cols[0] * stride_y) == rows
            rows = rows[keep]
            cols = cols[keep]
        if stride_x > 1:
            rows_cols = (rows, cols // stride_x)
            keep = (rows_cols[1] * stride_x) == cols
            rows = rows[keep]
            cols = cols[keep]

        if rows.size == 0:
            raise SystemExit("No pixels left after subsampling; try smaller --spacing-km.")

        xs, ys = rasterio.transform.xy(ds.transform, rows, cols)
        df = pd.DataFrame({"lon": np.asarray(xs, dtype=float), "lat": np.asarray(ys, dtype=float)})
        if not bool(args.lonlat_only):
            df["baseline"] = arr[rows, cols].astype(float)

    if args.max_points and int(args.max_points) > 0 and df.shape[0] > int(args.max_points):
        df = df.sample(n=int(args.max_points), random_state=42).reset_index(drop=True)

    if str(args.roi_geojson).strip():
        geo_path = Path(str(args.roi_geojson)).expanduser().resolve()
        geo = json.loads(geo_path.read_text())
        rings = _roi_geojson_outer_rings(geo)
        inside = _points_in_polygons(df["lon"].to_numpy(dtype=float), df["lat"].to_numpy(dtype=float), rings)
        df = df[inside].reset_index(drop=True)
        if df.empty:
            raise SystemExit(f"No grid points remain inside ROI polygon: {geo_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Wrote {df.shape[0]} points to {out_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

try:
    import netCDF4 as nc  # type: ignore
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as e:  # pragma: no cover
    missing = str(e).split("No module named", 1)[-1].strip().strip("'").strip('"')
    raise SystemExit(
        f"Missing dependency {missing!r}. Run inside your project environment "
        f"(e.g., `conda run -n odmatrix python tools/ghap_nc_to_kepler_csv.py ...`)."
    ) from e


def _roi_geojson_outer_rings(geo: dict) -> list[np.ndarray]:
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
    x = np.asarray(lon, dtype=float)
    y = np.asarray(lat, dtype=float)
    poly = np.asarray(ring, dtype=float)
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


def _extract_date_from_name(path: Path) -> str | None:
    m = re.search(r"(19|20)\\d{6}", path.name)
    return m.group(0) if m else None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert GHAP PM2.5 netCDF(s) to a Kepler-ready CSV (lon,lat,value).")
    p.add_argument("--input", required=True, help="Path to a GHAP .nc file or a folder containing .nc files.")
    p.add_argument("--out", required=True, help="Output CSV path.")
    p.add_argument("--var", default="PM2.5", help="Variable name in netCDF (default: PM2.5).")
    p.add_argument("--lon-var", default="lon", help="Longitude variable name (default: lon).")
    p.add_argument("--lat-var", default="lat", help="Latitude variable name (default: lat).")
    p.add_argument("--spacing-km", type=float, default=1.0, help="Approximate subsampling spacing in km (default: 1.0).")
    p.add_argument("--roi-geojson", default="", help="Optional ROI GeoJSON polygon to keep points inside.")
    p.add_argument("--max-points", type=int, default=0, help="Optional cap on output points (0 = no cap).")
    p.add_argument("--include-date", action="store_true", help="Include a 'date' column parsed from the filename.")
    return p.parse_args(argv)


def _stride_from_lonlat(lon: np.ndarray, lat: np.ndarray, *, spacing_km: float) -> tuple[int, int]:
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.size < 2 or lat.size < 2:
        return (1, 1)
    dlon = float(np.median(np.abs(np.diff(lon))))
    dlat = float(np.median(np.abs(np.diff(lat))))
    lat0 = float(np.median(lat))
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(1e-6, np.cos(np.deg2rad(lat0)))
    px_m_x = dlon * m_per_deg_lon
    px_m_y = dlat * m_per_deg_lat
    target_m = float(spacing_km) * 1000.0
    stride_x = max(1, int(round(target_m / max(1e-9, px_m_x))))
    stride_y = max(1, int(round(target_m / max(1e-9, px_m_y))))
    return (stride_y, stride_x)


def _load_one_nc(path: Path, *, var: str, lon_var: str, lat_var: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = nc.Dataset(str(path))
    try:
        data = np.array(ds.variables[var][:], dtype=float)
        lon = np.array(ds.variables[lon_var][:], dtype=float)
        lat = np.array(ds.variables[lat_var][:], dtype=float)
    finally:
        ds.close()
    # GHAP convention: missing=65535
    data = np.where(data == 65535, np.nan, data)
    return data, lon, lat


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    in_path = Path(args.input).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()

    if in_path.is_dir():
        nc_files = sorted(in_path.glob("*.nc"))
    else:
        nc_files = [in_path]
    if not nc_files:
        raise SystemExit(f"No .nc files found under {in_path}")

    frames: list[pd.DataFrame] = []
    rings: list[np.ndarray] | None = None
    roi_bbox: tuple[float, float, float, float] | None = None
    if str(args.roi_geojson).strip():
        geo = json.loads(Path(str(args.roi_geojson)).expanduser().resolve().read_text())
        rings = _roi_geojson_outer_rings(geo)
        all_lon = np.concatenate([r[:, 0] for r in rings])
        all_lat = np.concatenate([r[:, 1] for r in rings])
        roi_bbox = (float(np.min(all_lon)), float(np.max(all_lon)), float(np.min(all_lat)), float(np.max(all_lat)))

    for f in nc_files:
        data, lon, lat = _load_one_nc(f, var=str(args.var), lon_var=str(args.lon_var), lat_var=str(args.lat_var))
        if data.ndim != 2:
            raise SystemExit(f"Expected 2D {args.var} in {f}, got shape={data.shape}")
        if data.shape != (lat.size, lon.size):
            raise SystemExit(f"Shape mismatch in {f}: data {data.shape} vs (lat,lon)=({lat.size},{lon.size})")

        # Optional bbox prefilter to reduce memory before meshgrid.
        if roi_bbox is not None:
            min_lon, max_lon, min_lat, max_lat = roi_bbox
            lon_mask = (lon >= min_lon) & (lon <= max_lon)
            lat_mask = (lat >= min_lat) & (lat <= max_lat)
            if not lon_mask.any() or not lat_mask.any():
                continue
            data = data[np.ix_(lat_mask, lon_mask)]
            lon = lon[lon_mask]
            lat = lat[lat_mask]

        stride_y, stride_x = _stride_from_lonlat(lon, lat, spacing_km=float(args.spacing_km))
        yy = np.arange(0, lat.size, stride_y, dtype=int)
        xx = np.arange(0, lon.size, stride_x, dtype=int)
        sub = data[np.ix_(yy, xx)]
        lon_sub = lon[xx]
        lat_sub = lat[yy]
        LON, LAT = np.meshgrid(lon_sub, lat_sub)

        df = pd.DataFrame(
            {
                "lon": LON.reshape(-1).astype(float),
                "lat": LAT.reshape(-1).astype(float),
                "baseline": sub.reshape(-1).astype(float),
            }
        )
        df = df[np.isfinite(df["baseline"].to_numpy(dtype=float))].copy()
        if rings is not None and not df.empty:
            inside = _points_in_polygons(df["lon"].to_numpy(dtype=float), df["lat"].to_numpy(dtype=float), rings)
            df = df[inside].copy()
    if bool(args.include_date):
        d = _extract_date_from_name(f)
        df["date"] = d if d is not None else ""
    frames.append(df)

    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["lon", "lat", "baseline"])
    if args.max_points and int(args.max_points) > 0 and out.shape[0] > int(args.max_points):
        out = out.sample(n=int(args.max_points), random_state=42).reset_index(drop=True)
    if out.empty:
        raise SystemExit("No output points (after missing-value filtering and ROI filtering).")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Wrote {out.shape[0]} rows to {out_path}")


if __name__ == "__main__":
    main()

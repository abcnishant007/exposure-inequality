"""
Lightweight NetCDF -> GeoTIFF converter for GHAP PM2.5 daily rasters.
Uses rasterio instead of GDAL/osgeo (which isn't in the env).
Outputs one GeoTIFF per input .nc, preserving 1 km grid and WGS84.
"""
from __future__ import annotations

import sys
from pathlib import Path

import netCDF4 as nc
import numpy as np
import rasterio
from rasterio.transform import from_origin

def main(inp: str, out_dir: str | None = None, var: str = "PM2.5", lon_var: str = "lon", lat_var: str = "lat") -> None:
    in_path = Path(inp).expanduser().resolve()
    if in_path.is_dir():
        files = sorted(in_path.glob("*.nc"))
    else:
        files = [in_path]
    if not files:
        raise SystemExit(f"No .nc files found under {in_path}")

    out_root = Path(out_dir).expanduser().resolve() if out_dir else (in_path if in_path.is_dir() else in_path.parent)
    out_root.mkdir(parents=True, exist_ok=True)

    for f in files:
        with nc.Dataset(str(f)) as ds:
            data = np.array(ds.variables[var][:], dtype=np.float32)
            lon = np.array(ds.variables[lon_var][:], dtype=np.float64)
            lat = np.array(ds.variables[lat_var][:], dtype=np.float64)
        # GHAP missing flag
        data = np.where(data == 65535, np.nan, data)

        # Resolution and orientation
        lon_res = float(np.median(np.diff(lon))) if lon.size > 1 else 0.01
        lat_res = abs(float(np.median(np.diff(lat)))) if lat.size > 1 else 0.01

        lat_increasing = lat.size > 1 and (lat[1] > lat[0])
        if lat_increasing:
            data = np.flipud(data)
            lat_max = float(lat.max())
        else:
            lat_max = float(lat[0])
        lon_min = float(lon.min())

        transform = from_origin(lon_min - lon_res / 2.0, lat_max + lat_res / 2.0, lon_res, lat_res)

        profile = {
            "driver": "GTiff",
            "height": data.shape[0],
            "width": data.shape[1],
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:4326",
            "transform": transform,
            "nodata": np.nan,
        }

        out_path = out_root / f.name.replace(".nc", ".tif")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data, 1)
        print(f"wrote {out_path}")

if __name__ == "__main__":
    # usage: python tools/ghap_nc_to_geotiff_rio.py INPUT_NC_OR_DIR [OUT_DIR]
    args = sys.argv[1:]
    if not args:
        raise SystemExit("Usage: python tools/ghap_nc_to_geotiff_rio.py INPUT_NC_OR_DIR [OUT_DIR]")
    inp = args[0]
    out_dir = args[1] if len(args) > 1 else None
    main(inp, out_dir)

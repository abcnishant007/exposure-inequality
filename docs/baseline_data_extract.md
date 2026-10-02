## Baseline data extraction for a new city (1 km CSV)

Goal: create a Kepler-ready baseline PM2.5 CSV (lon, lat, baseline[, date]) for a new metro using the GHAP daily NetCDF tiles, and produce a matching sensors CSV from the PurpleAir point list.

### Prerequisites
- Activate project env: `conda run -n odmatrix …`
- GHAP daily NetCDFs placed in a folder (e.g., `GHAP_Data/GHAP_PM2.5_D1K_202204_V1`).
- A metro ROI GeoJSON in `data/geojson_msa/` (Polygon/MultiPolygon).
- (Optional) PurpleAir sensor point GeoJSON in `data/msa_sensor_list_purple_air/`.

### 1) Baseline grid CSV from GHAP NetCDF
Use the existing converter; it subsamples to ~1 km, clips to ROI, handles missing=65535, and can add a date column parsed from filename.

```bash
conda run -n odmatrix python tools/ghap_nc_to_kepler_csv.py \
  --input GHAP_Data/GHAP_PM2.5_D1K_202204_V1/GHAP_PM2.5_D1K_YYYYMMDD_V1.nc \
  --out /tmp/<city>_baseline.csv \
  --roi-geojson data/geojson_msa/<city-msa>.geojson \
  --spacing-km 1 \
  --include-date \
  --max-points 400000
```
Notes:
- For multiple days, point `--input` at the folder; it will concatenate daily grids (with optional `date` column).
- Use `--max-points` to cap output and avoid huge CSVs.
- If disk space is tight, write to `/tmp/` and preview with `head` only (these files can be large).

### 2) Sensors CSV from PurpleAir GeoJSON
Generates the standard sensor list schema (`sensor_index,latitude,longitude,name`).

```bash
python tools/geojson_to_sensors_csv.py \
  --input data/msa_sensor_list_purple_air/<city-msa>.geojson \
  --out bayesian_fusion/bayesian_fusion_<city>/<city-msa>_sensors.csv
```

### 3) Quick checks
- `ls -lh /tmp/<city>_baseline.csv` to verify size.
- `head -n 5 /tmp/<city>_baseline.csv` (never open full file).
- `head -n 5 bayesian_fusion/bayesian_fusion_<city>/<city-msa>_sensors.csv` to confirm schema.

### Troubleshooting
- `ModuleNotFoundError: netCDF4` → run inside `odmatrix` env.
- `No space left on device` → free space or write to `/tmp/` and lower `--max-points`.
- `No output points` → ROI may be outside raster extent; confirm the GeoJSON bounds.

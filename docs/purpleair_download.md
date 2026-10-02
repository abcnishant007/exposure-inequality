# PurpleAir Download Quickstart

Purpose: show how to fetch PurpleAir history and produce `cleaned_*.csv` for any metro area using the existing scripts in this repo.

## Prerequisites
- PurpleAir API key available as env var `PURPLE_AIR_API_KEY` or passed via `--api-key`.
- Python deps for the scripts already installed.

## 1) Generate the sensor list GeoJSON for your metro
- The script grabs the metro polygon, computes its bounding box, calls the PurpleAir sensors API, and saves a FeatureCollection with `sensor_index` for each sensor.

Command (load API key from `.env` first):
```
set -a; source .env; set +a
python EDA_PM25/MSA_50_get_sensor_list.py --limit 50
```
Notes:
- `--limit` is how many MSAs from the built‑in TOP_50_MSAS list to process. Default 50; Oklahoma City is inside that set. You can lower it as long as it still includes your target metro.
- Output lives in `data/msa_sensor_list_purple_air/<slug>.geojson` (e.g., `oklahoma-city-ok.geojson`).

## 2) Download and clean history for a month
- Uses the cached sensor list; downloads per‑sensor history, aggregates to `raw_*.csv`, then applies range/MAD/step filters to produce `cleaned_*.csv`.

Example for Oklahoma City, April 2022:
```
python EDA_PM25/download_purpleair_history.py \
  --geojson data/msa_sensor_list_purple_air/oklahoma-city-ok.geojson \
  --month 2022-04 \
  --output-dir data/purpleair_history/oklahoma_city_2022-04
```

Key flags (optional):
- `--pm-min/--pm-max` (default 0, 500) physical range.
- `--k-mad` (default 6) robust z-score cutoff per sensor-day.
- `--step-max` (default 150) max 30‑min jump allowed.
- `--rate-limit` / `--sleep` to throttle API calls.

Outputs in the chosen `--output-dir`:
- `raw_<start>_<end>.csv` (all downloaded rows)
- `cleaned_<start>_<end>.csv` (after filters)

## 3) Regenerate combined files without re-downloading
- If per-sensor CSVs already exist, re-run aggregation only:
```
python EDA_PM25/download_purpleair_history.py \
  --geojson data/msa_sensor_list_purple_air/oklahoma-city-ok.geojson \
  --month 2022-04 \
  --output-dir data/purpleair_history/oklahoma_city_2022-04 \
  --aggregate-only
```

## 4) Verify file contents quickly
- Use `head` to check columns without loading large CSVs:
```
head -n 5 data/purpleair_history/oklahoma_city_2022-04/cleaned_20220401_20220501.csv
```

# New City Checklist

This checklist documents the steps to add a new city and run the pipeline safely.

## 1) Place the licensed Replica CSV
- Put the licensed city CSV in your local input folder, for example:
  `/path/to/project_inputs/<city-folder>/replica-<city>.csv`

## 2) Create 10K + 100K person-complete samples
Run `make_person_subsample_das.py` to generate `_10000.csv` and `_100000.csv`:

```bash
python scripts/make_person_subsample_das.py \
  --full-das /path/to/project_inputs/<city-folder>/replica-<city>.csv \
  --out-dir /path/to/project_inputs/<city-folder> \
  --sizes 10000,100000 \
  --prefix <city>_spring_2023_thursday
```

This creates:
- `<city>_spring_2023_thursday_10000.csv`
- `<city>_spring_2023_thursday_100000.csv`

## 3) Update the city config
Edit `configs/config_<city>.yaml`:

- Set the activity file:
  `activity_data.path` -> the desired sample file

Example (100K):
```
path: "/path/to/project_inputs/<city-folder>/<city>_spring_2023_thursday_100000.csv"
```

- Ensure outputs are on SSD (Results root only):
```
results_root: "/path/to/project_outputs/results_<city>"
temp_root: "/path/to/project_outputs/temp_processing_<city>"
repo_results_root: "/path/to/project_outputs/results_<city>"
```

## 4) Create output directories
```bash
mkdir -p /path/to/project_outputs/results_<city>
mkdir -p /path/to/project_outputs/temp_processing_<city>
```

## 5) Run the pipeline
```bash
python -u main_pipeline.py --config config.yaml
```

## Notes
- The pipeline should **not modify** `input_data` during runtime.
- Use 10K first for quick validation, then switch to 100K.
- Make sure the raw CSV has required coordinate columns:
  `origin_bgrp_lat_2020`, `origin_bgrp_lng_2020`,
  `destination_bgrp_lat_2020`, `destination_bgrp_lng_2020`,
  `trip_taker_home_bgrp_lat_2020`, `trip_taker_home_bgrp_lng_2020`.

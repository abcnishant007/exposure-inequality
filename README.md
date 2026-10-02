# Exposure Inequality Release Pipeline

This repository contains the release pipeline for reconstructing population exposure inequality from mobility, PM2.5, sensor, and cohort inputs. It is a cleaned code release, not the full private working repository used during paper development.

Raw city-scale inputs, licensed Replica population mobility data, manuscript LaTeX, figure-generation workspaces, debug folders, archives, derived paper-result tables, and large intermediate artifacts are not included. Small public lookup tables are included where appropriate.

## Repository Contents

- Core analysis pipeline: `main_pipeline.py` and helper modules for DAS processing, exposure calculation, cohort statistics, comparisons, and visualizations.
- PM2.5 preparation and validation helper scripts.
- `tools/`: lightweight conversion, diagnostics, and uncertainty utilities.
- `docs/`: Sphinx source documentation.
- `data/NAICS_industry.csv`: lightweight industry lookup data.

## Environment Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate odmatrix
```

If the environment already exists, update it:

```bash
conda env update -f environment.yml --prune
conda activate odmatrix
```

## Configure a Run

Copy the example config and edit paths for your licensed and third-party inputs:

```bash
cp config.yaml.example config.yaml
```

At minimum, update:

- `study_area.geojson_path`
- `activity_data.path`
- PM2.5 input paths under `pollution`
- `outputs.results_root` and `outputs.temp_root`

The example config is intentionally non-runnable until these input paths are replaced.

## Run the Analysis Pipeline

```bash
python main_pipeline.py --config config.yaml
```

You can also pass an activity file override:

```bash
python main_pipeline.py --config config.yaml --activity-data /path/to/activity.csv
```

The pipeline writes analysis outputs under the configured `outputs.results_root` and temporary streaming/intermediate files under `outputs.temp_root`.

## Build the Documentation

```bash
sphinx-build -b html docs docs/_build/html
open docs/_build/html/index.html
```

The generated `_build` directory is ignored by git.

## Data

See `DATA_AVAILABILITY.md` for what is included, what must be recollected or licensed, and what is required to reproduce full city-scale estimates and paper results.

## Citation

This repository contains code accompanying our manuscript:

> Nishant Kumar, Khoa D. Vo, Robbie M. Parks, Swapnil Mishra, and Prateek Bansal. 2026. "Daily activity reshapes air-pollution exposure and inequality." arXiv link to be updated.

If you find the code or data products in this repository useful for your research, please remember to cite our paper.

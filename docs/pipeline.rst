Pipeline Modules (Quick Map)
============================

This repo has a few generations of scripts; the “current” end-to-end run is the DuckDB-based pipeline.

Main entrypoint
---------------

- ``main_pipeline.py``: ``DuckDBExposurePipeline`` orchestrates the run (config loading, temp/results dirs, module wiring, cohort loops).

Module 1: DAS → stay points (data prep)
---------------------------------------

- ``das_processor.py``: ``DuckDBDASProcessor`` loads the large activity CSV into DuckDB (with cleaning + optional bbox filter) and converts per-person trip records into time-stamped “stay points”.

Module 2: pollution lookup + exposure
-------------------------------------

- ``exposure_calculator.py``: ``DuckDBExposureCalculator`` loads pollution inputs and computes per-person exposure:
  - Static mode: masked GeoTIFF raster + KD-tree accelerated lookups.
  - Dynamic mode: 30-minute ``(lon, lat, pm25, timestamp_utc)`` CSV + KD-trees per time slice.

Comparison + visualization outputs
----------------------------------

- ``visualization.py``: builds the activity-vs-home comparison table and a compact per-person summary dictionary.
- ``utils.py``: shared helpers for config, caching/fingerprinting, and building lightweight visualization payload JSON.
- ``visualization_summary.py``: renders mean/median curves + sample-individual plots from the precomputed visualization JSON (no recomputation).

Cohorts + reporting
-------------------

- ``config.yaml`` (copied from ``config.yaml.example``): cohort groups are expressed as DuckDB ``WHERE`` filters over ``activity_data``.
- ``cohort_statistics.py``: computes descriptive stats (daily distributions, time-window exposure, peaks, threshold durations) per cohort/group.
- ``cohort_comparator.py``: flattens cohort stats JSON into comparison CSVs and simple plots.

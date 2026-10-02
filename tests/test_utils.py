from __future__ import annotations

import hashlib

from utils import compute_config_md5, compute_pollution_fingerprint, load_config, normalize_exposure_metric


def test_load_config_adds_default_output_paths(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
study_area:
  geojson_path: inputs/example.geojson
pollution:
  mode: dynamic
activity_data:
  path: inputs/activity.csv
""",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["outputs"]["results_root"] == "results"
    assert config["outputs"]["temp_root"] == "temp_processing"
    assert config["outputs"]["repo_results_root"] == "results"


def test_compute_config_md5_hashes_raw_file_bytes(tmp_path):
    config_path = tmp_path / "config.yaml"
    payload = b"pollution:\n  mode: static\n"
    config_path.write_bytes(payload)

    assert compute_config_md5(config_path) == hashlib.md5(payload).hexdigest()


def test_normalize_exposure_metric_accepts_known_metrics_and_defaults_unknown():
    assert normalize_exposure_metric(" cumulative ") == "cumulative"
    assert normalize_exposure_metric("TWAC") == "twac"
    assert normalize_exposure_metric("") == "cumulative"
    assert normalize_exposure_metric("daily_average") == "cumulative"


def test_pollution_fingerprint_changes_with_input_state(tmp_path):
    dynamic_a = tmp_path / "dynamic_a.csv"
    dynamic_b = tmp_path / "dynamic_b.csv"
    dynamic_a.write_text("lon,lat,pm25\n-1,1,8\n", encoding="utf-8")
    dynamic_b.write_text("lon,lat,pm25\n-1,1,8\n-2,2,9\n", encoding="utf-8")

    base_cfg = {
        "mode": "dynamic",
        "travel_exposure": 0.5,
        "travel_exposure_type": "origin_destination_mean",
        "dynamic_csv_path": str(dynamic_a),
    }
    changed_cfg = {**base_cfg, "dynamic_csv_path": str(dynamic_b)}

    assert compute_pollution_fingerprint(base_cfg) == compute_pollution_fingerprint(dict(base_cfg))
    assert compute_pollution_fingerprint(base_cfg) != compute_pollution_fingerprint(changed_cfg)

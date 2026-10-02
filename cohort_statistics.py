#!/usr/bin/env python3
"""
Cohort Statistics Module

Generates additional statistical summaries for the full population and
each configured cohort without re-running exposure calculations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import duckdb

TIME_WINDOWS = [
    ("early_morning", 0, 6),
    ("morning_commute", 6, 9),
    ("midday", 11, 14),
    ("evening_commute", 16, 19),
    ("night", 19, 24),
]

THRESHOLDS = [35.0, 55.0, 75.0]  # μg/m³ concentration thresholds


def generate_cohort_statistics(
    detailed_exposure: pd.DataFrame,
    comparison_results: pd.DataFrame,
    cohort_entries: List[Dict],
    base_output_dir: str = "results",
    duckdb_path: Optional[str] = None,
    config_hash: Optional[str] = None,
) -> None:
    """
    Generate JSON summaries with descriptive statistics for each cohort.
    """
    detailed_glob = None
    if isinstance(detailed_exposure, dict) and detailed_exposure.get('type') == 'parquet_glob':
        detailed_glob = detailed_exposure.get('path')
    elif detailed_exposure is None or detailed_exposure.empty:
        print("No detailed exposure data available for cohort statistics")
        return
    if comparison_results is None or comparison_results.empty:
        print("No comparison results available for cohort statistics")
        return
    comparison_results = comparison_results.copy()
    comparison_results['person_id'] = comparison_results['person_id'].astype(str)

    os.makedirs(base_output_dir, exist_ok=True)

    total_cohorts = len(cohort_entries)
    for i, cohort in enumerate(cohort_entries, start=1):
        label = cohort.get('label', 'cohort')
        output_root = cohort.get('output_root', base_output_dir)
        person_ids = cohort.get('person_ids')
        print(f"Cohort statistics [{i}/{total_cohorts}]: {label}")

        subset_comparison = comparison_results
        if person_ids:
            person_ids_set = set(person_ids)
            subset_comparison = comparison_results[comparison_results['person_id'].isin(person_ids_set)].copy()

        if subset_comparison.empty:
            print(f"No comparison data for cohort '{label}', skipping stats")
            continue

        if detailed_glob:
            stats = _compute_statistics_from_parquet(
                detailed_glob, subset_comparison, person_ids,
                duckdb_path=duckdb_path, config_hash=config_hash
            )
        else:
            subset_details = detailed_exposure
            if person_ids:
                person_ids_set = set(person_ids)
                subset_details = detailed_exposure[detailed_exposure['person_id'].isin(person_ids_set)].copy()
            stats = _compute_statistics(subset_details, subset_comparison)
        os.makedirs(output_root, exist_ok=True)
        output_path = os.path.join(output_root, f"cohort_stats_{label}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved cohort statistics for '{label}' to {output_path}")


def _compute_statistics(stay_df: pd.DataFrame, daily_df: pd.DataFrame) -> Dict:
    stats: Dict[str, Dict] = {}
    stats['population_count'] = int(daily_df['person_id'].nunique())

    stats['daily_activity_exposure'] = _describe_series(daily_df['activity_exposure'])
    stats['daily_home_exposure'] = _describe_series(daily_df['home_exposure'])
    ratio_series = daily_df['activity_exposure'] / daily_df['home_exposure'].replace(0, np.nan)
    stats['daily_exposure_ratio'] = _describe_series(ratio_series.replace([np.inf, -np.inf], np.nan).dropna())

    time_window_stats = _compute_time_window_exposure(stay_df, daily_df['person_id'].unique())
    stats['time_windows'] = time_window_stats

    peak_stats = _compute_peak_exposure(stay_df)
    stats['peak_exposure'] = peak_stats

    threshold_stats = _compute_duration_above_thresholds(stay_df)
    stats['duration_above_thresholds_minutes'] = threshold_stats

    return stats


def _compute_statistics_from_parquet(detailed_glob: str, daily_df: pd.DataFrame,
                                     person_ids: Optional[List[str]], duckdb_path: Optional[str] = None,
                                     config_hash: Optional[str] = None) -> Dict:
    stats: Dict[str, Dict] = {}
    stats['population_count'] = int(daily_df['person_id'].nunique())

    stats['daily_activity_exposure'] = _describe_series(daily_df['activity_exposure'])
    stats['daily_home_exposure'] = _describe_series(daily_df['home_exposure'])
    ratio_series = daily_df['activity_exposure'] / daily_df['home_exposure'].replace(0, np.nan)
    stats['daily_exposure_ratio'] = _describe_series(ratio_series.replace([np.inf, -np.inf], np.nan).dropna())

    conn = duckdb.connect(duckdb_path) if duckdb_path else duckdb.connect(':memory:')
    temp_root = Path(os.environ.get('TEMP_PROCESSING_ROOT', 'temp_processing'))
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        conn.execute(f"PRAGMA temp_directory='{temp_root.as_posix()}'")
    except Exception:
        pass

    all_person_ids = daily_df['person_id'].astype(str).unique()
    target_persons_df = pd.DataFrame({'person_id': all_person_ids})
    conn.register('target_persons', target_persons_df)

    if duckdb_path:
        base_source = "detailed_exposure"
    else:
        base_source = f"read_parquet('{detailed_glob}', union_by_name=true)"
    if person_ids:
        subset_df = pd.DataFrame({'person_id': [str(pid) for pid in person_ids]})
        conn.register('subset_ids', subset_df)
        source = f"(SELECT d.* FROM {base_source} d JOIN subset_ids s USING(person_id))"
    else:
        source = f"(SELECT * FROM {base_source})"

    if duckdb_path:
        _ensure_time_window_bins(conn, config_hash)
        time_window_stats = _compute_time_window_exposure_from_bins(conn)
    else:
        time_window_stats = _compute_time_window_exposure_sql(conn, source)
    stats['time_windows'] = time_window_stats

    peak_stats = _compute_peak_exposure_sql(conn, source)
    stats['peak_exposure'] = peak_stats

    threshold_stats = _compute_duration_above_thresholds_sql(conn, source)
    stats['duration_above_thresholds_minutes'] = threshold_stats

    conn.unregister('target_persons')
    conn.close()
    return stats


def _describe_sql(conn: duckdb.DuckDBPyConnection, values_sql: str, value_col: str = "value") -> Dict[str, float]:
    row = conn.execute(f"""
        SELECT
            AVG({value_col}) AS mean,
            MEDIAN({value_col}) AS median,
            COALESCE(STDDEV_SAMP({value_col}), 0.0) AS std,
            MIN({value_col}) AS min,
            MAX({value_col}) AS max,
            QUANTILE_CONT({value_col}, 0.25) AS p25,
            QUANTILE_CONT({value_col}, 0.75) AS p75
        FROM ({values_sql}) q
    """).fetchone()
    if row is None or row[0] is None:
        return {}
    return {
        'mean': float(row[0]),
        'median': float(row[1]),
        'std': float(row[2]),
        'min': float(row[3]),
        'max': float(row[4]),
        'p25': float(row[5]),
        'p75': float(row[6]),
    }


def _describe_series(series: pd.Series) -> Dict[str, float]:
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if series.empty:
        return {}
    arr = series.to_numpy(dtype=float)
    return {
        'mean': float(np.mean(arr)),
        'median': float(np.median(arr)),
        'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        'min': float(np.min(arr)),
        'max': float(np.max(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75))
    }


def _compute_time_window_exposure(stay_df: pd.DataFrame, person_ids: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Compute exposure accumulated within each time window using 5-minute increments.
    """
    if stay_df.empty:
        return {}

    conn = duckdb.connect(':memory:')
    temp_root = Path(os.environ.get('TEMP_PROCESSING_ROOT', 'temp_processing'))
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        conn.execute(f"PRAGMA temp_directory='{temp_root.as_posix()}'")
    except Exception:
        pass
    conn.register('detailed_exposure', stay_df)

    incremental_sql = """
        WITH base AS (
            SELECT
                person_id,
                start_time,
                end_time,
                CASE 
                    WHEN location_type = 'travel' AND travel_exposure IS NOT NULL 
                        THEN travel_exposure
                    ELSE pollution_concentration
                END AS exposure_rate
            FROM detailed_exposure
            WHERE start_time IS NOT NULL AND end_time IS NOT NULL
        ),
        expanded AS (
            SELECT
                person_id,
                gs.bin_start,
                LEAST(end_time, gs.bin_start + INTERVAL '5 minutes') AS overlap_end,
                GREATEST(start_time, gs.bin_start) AS overlap_start,
                exposure_rate
            FROM base
            CROSS JOIN LATERAL generate_series(
                date_trunc('minute', start_time),
                end_time,
                INTERVAL '5 minutes'
            ) AS gs(bin_start)
        ),
        valid AS (
            SELECT
                person_id,
                bin_start,
                exposure_rate,
                GREATEST(0, EXTRACT(EPOCH FROM overlap_end - overlap_start) / 3600.0) AS overlap_hours
            FROM expanded
            WHERE overlap_end > overlap_start
        )
        SELECT
            person_id,
            CAST(DATEDIFF('minute', date_trunc('day', bin_start), bin_start) / 5 AS INTEGER) AS bin_index,
            SUM(exposure_rate * overlap_hours) AS exposure_increment
        FROM valid
        GROUP BY person_id, bin_index
    """

    bin_df = conn.execute(incremental_sql).fetchdf()
    conn.unregister('detailed_exposure')
    conn.close()

    if bin_df.empty:
        return {}

    bin_df['bin_hour'] = bin_df['bin_index'] * (5.0 / 60.0)
    results = {}
    all_persons = pd.Index(person_ids)

    for name, start_hour, end_hour in TIME_WINDOWS:
        window_mask = (bin_df['bin_hour'] >= start_hour) & (bin_df['bin_hour'] < end_hour)
        window_df = bin_df[window_mask]
        per_person = window_df.groupby('person_id')['exposure_increment'].sum()
        per_person = per_person.reindex(all_persons, fill_value=0.0)
        results[name] = _describe_series(per_person)

    return results


def _compute_time_window_exposure_sql(conn: duckdb.DuckDBPyConnection, source_sql: str) -> Dict[str, Dict[str, float]]:
    incremental_sql = f"""
        WITH base AS (
            SELECT
                person_id,
                start_time,
                end_time,
                CASE
                    WHEN location_type = 'travel' AND travel_exposure IS NOT NULL
                        THEN travel_exposure
                    ELSE pollution_concentration
                END AS exposure_rate
            FROM {source_sql}
            WHERE start_time IS NOT NULL AND end_time IS NOT NULL
        ),
        expanded AS (
            SELECT
                person_id,
                gs.bin_start,
                LEAST(end_time, gs.bin_start + INTERVAL '5 minutes') AS overlap_end,
                GREATEST(start_time, gs.bin_start) AS overlap_start,
                exposure_rate
            FROM base
            CROSS JOIN LATERAL generate_series(
                date_trunc('minute', start_time),
                end_time,
                INTERVAL '5 minutes'
            ) AS gs(bin_start)
        ),
        valid AS (
            SELECT
                person_id,
                bin_start,
                exposure_rate,
                GREATEST(0, EXTRACT(EPOCH FROM overlap_end - overlap_start) / 3600.0) AS overlap_hours
            FROM expanded
            WHERE overlap_end > overlap_start
        )
        SELECT
            person_id,
            CAST(DATEDIFF('minute', date_trunc('day', bin_start), bin_start) / 5 AS INTEGER) AS bin_index,
            SUM(exposure_rate * overlap_hours) AS exposure_increment
        FROM valid
        GROUP BY person_id, bin_index
    """
    conn.execute(f"CREATE OR REPLACE TEMP TABLE tmp_bins AS {incremental_sql}")
    results = {}
    for name, start_hour, end_hour in TIME_WINDOWS:
        start_bin = int(start_hour * 12)
        end_bin = int(end_hour * 12)
        values_sql = f"""
            WITH per_person AS (
                SELECT
                    person_id,
                    SUM(exposure_increment) AS value
                FROM tmp_bins
                WHERE bin_index >= {start_bin} AND bin_index < {end_bin}
                GROUP BY person_id
            )
            SELECT
                t.person_id,
                COALESCE(p.value, 0.0) AS value
            FROM target_persons t
            LEFT JOIN per_person p USING(person_id)
        """
        results[name] = _describe_sql(conn, values_sql)
    conn.execute("DROP TABLE IF EXISTS tmp_bins")
    return results


def _ensure_time_window_bins(conn: duckdb.DuckDBPyConnection, config_hash: Optional[str]) -> None:
    """Build 5-min exposure bins table once per config hash."""
    temp_root = Path(os.environ.get('TEMP_PROCESSING_ROOT', 'temp_processing'))
    temp_root.mkdir(parents=True, exist_ok=True)
    manifest_path = temp_root / 'duckdb_intermediates' / 'time_window_bins_manifest.json'
    expected = {'config_hash': config_hash or 'na', 'table': 'time_window_bins_5min'}
    try:
        if manifest_path.exists():
            current = json.loads(manifest_path.read_text())
            if current == expected:
                return
    except Exception:
        pass
    conn.execute("DROP TABLE IF EXISTS time_window_bins_5min")
    conn.execute("""
        CREATE TABLE time_window_bins_5min AS
        WITH base AS (
            SELECT
                person_id,
                start_time,
                end_time,
                CASE
                    WHEN location_type = 'travel' AND travel_exposure IS NOT NULL
                        THEN travel_exposure
                    ELSE pollution_concentration
                END AS exposure_rate
            FROM detailed_exposure
            WHERE start_time IS NOT NULL AND end_time IS NOT NULL
        ),
        expanded AS (
            SELECT
                person_id,
                gs.bin_start,
                LEAST(end_time, gs.bin_start + INTERVAL '5 minutes') AS overlap_end,
                GREATEST(start_time, gs.bin_start) AS overlap_start,
                exposure_rate
            FROM base
            CROSS JOIN LATERAL generate_series(
                date_trunc('minute', start_time),
                end_time,
                INTERVAL '5 minutes'
            ) AS gs(bin_start)
        ),
        valid AS (
            SELECT
                person_id,
                bin_start,
                exposure_rate,
                GREATEST(0, EXTRACT(EPOCH FROM overlap_end - overlap_start) / 3600.0) AS overlap_hours
            FROM expanded
            WHERE overlap_end > overlap_start
        )
        SELECT
            person_id,
            CAST(DATEDIFF('minute', date_trunc('day', bin_start), bin_start) / 5 AS INTEGER) AS bin_index,
            SUM(exposure_rate * overlap_hours) AS exposure_increment
        FROM valid
        GROUP BY person_id, bin_index
    """)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bins_person ON time_window_bins_5min(person_id)")
    except Exception:
        pass
    try:
        manifest_path.write_text(json.dumps(expected, indent=2))
    except Exception:
        pass


def _compute_time_window_exposure_from_bins(conn: duckdb.DuckDBPyConnection) -> Dict[str, Dict[str, float]]:
    results = {}
    for name, start_hour, end_hour in TIME_WINDOWS:
        start_bin = int(start_hour * 12)
        end_bin = int(end_hour * 12)
        values_sql = f"""
            WITH per_person AS (
                SELECT
                    person_id,
                    SUM(exposure_increment) AS value
                FROM time_window_bins_5min
                WHERE bin_index >= {start_bin} AND bin_index < {end_bin}
                GROUP BY person_id
            )
            SELECT
                t.person_id,
                COALESCE(p.value, 0.0) AS value
            FROM target_persons t
            LEFT JOIN per_person p USING(person_id)
        """
        results[name] = _describe_sql(conn, values_sql)
    return results


def _compute_peak_exposure(stay_df: pd.DataFrame) -> Dict[str, float]:
    if stay_df.empty:
        return {}
    temp = stay_df.copy()
    temp['exposure_rate'] = np.where(
        temp['location_type'] == 'travel',
        temp['travel_exposure'],
        temp['pollution_concentration']
    )
    temp['exposure_rate'] = temp['exposure_rate'].fillna(0.0)
    per_person_peak = temp.groupby('person_id')['exposure_rate'].max()
    return _describe_series(per_person_peak)


def _compute_peak_exposure_sql(conn: duckdb.DuckDBPyConnection, source_sql: str) -> Dict[str, float]:
    values_sql = f"""
        WITH per_person AS (
            SELECT
                person_id,
                MAX(CASE
                    WHEN location_type = 'travel' AND travel_exposure IS NOT NULL
                        THEN travel_exposure
                    ELSE pollution_concentration
                END) AS value
            FROM {source_sql}
            GROUP BY person_id
        )
        SELECT
            t.person_id,
            COALESCE(p.value, 0.0) AS value
        FROM target_persons t
        LEFT JOIN per_person p USING(person_id)
    """
    return _describe_sql(conn, values_sql)


def _compute_duration_above_thresholds(stay_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    if stay_df.empty:
        return {}
    temp = stay_df.copy()
    temp['exposure_rate'] = np.where(
        temp['location_type'] == 'travel',
        temp['travel_exposure'],
        temp['pollution_concentration']
    )
    temp['exposure_rate'] = temp['exposure_rate'].fillna(0.0)
    temp['duration_hours'] = temp['duration_hours'].fillna(0.0)

    results = {}
    persons = temp['person_id'].unique()
    for threshold in THRESHOLDS:
        mask = temp['exposure_rate'] >= threshold
        duration = temp[mask].groupby('person_id')['duration_hours'].sum() * 60.0  # minutes
        duration = duration.reindex(persons, fill_value=0.0)
        results[f">={threshold}"] = _describe_series(duration)
    return results


def _compute_duration_above_thresholds_sql(conn: duckdb.DuckDBPyConnection, source_sql: str) -> Dict[str, Dict[str, float]]:
    results = {}
    for threshold in THRESHOLDS:
        values_sql = f"""
            WITH per_person AS (
                SELECT
                    person_id,
                    SUM(CASE
                        WHEN (CASE
                                WHEN location_type = 'travel' AND travel_exposure IS NOT NULL
                                    THEN travel_exposure
                                ELSE pollution_concentration
                              END) >= {threshold}
                        THEN COALESCE(duration_hours, 0.0)
                        ELSE 0.0
                    END) * 60.0 AS value
                FROM {source_sql}
                GROUP BY person_id
            )
            SELECT
                t.person_id,
                COALESCE(p.value, 0.0) AS value
            FROM target_persons t
            LEFT JOIN per_person p USING(person_id)
        """
        results[str(threshold)] = _describe_sql(conn, values_sql)
    return results

#!/usr/bin/env python3
"""
Utility Module for DuckDB-based exposure analysis pipeline.

Contains helper functions, file management, and data validation utilities.
"""

import pandas as pd
import numpy as np
import yaml
import json
import os
import warnings
import duckdb
import hashlib
import shutil
from typing import List
from datetime import datetime
from pathlib import Path
warnings.filterwarnings('ignore')

VALID_EXPOSURE_METRICS = {'cumulative', 'twac'}
BIN_DURATION_HOURS = 5.0 / 60.0  # 5-minute increments
VISUALIZATION_CACHE_SCHEMA_VERSION = 2

_PERSON_INDEX_CACHE = None
_PERSON_INDEX_HASH = None


def _file_signature(path):
    if not path:
        return {'path': '', 'missing': True}
    try:
        stat = os.stat(path)
        return {
            'path': os.path.abspath(path),
            'mtime': stat.st_mtime,
            'size': stat.st_size
        }
    except OSError:
        return {
            'path': os.path.abspath(path),
            'missing': True
        }


def compute_config_md5(config_path: str) -> str:
    """Return MD5 hash of the raw config file contents (safety-first cache key)."""
    with open(config_path, 'rb') as f:
        data = f.read()
    return hashlib.md5(data).hexdigest()


def compute_pollution_fingerprint(pollution_cfg):
    """Return a short fingerprint that captures pollution input state."""
    cfg = pollution_cfg or {}
    mode = cfg.get('mode', 'static')
    payload = {
        'mode': mode,
        'travel_exposure': cfg.get('travel_exposure'),
        'travel_exposure_type': cfg.get('travel_exposure_type')
    }
    if mode == 'dynamic':
        payload['dynamic_csv'] = _file_signature(cfg.get('dynamic_csv_path'))
        payload['dynamic_interval_minutes'] = cfg.get('dynamic_interval_minutes', 30)
    elif mode == 'bayesian':
        payload['bayesian_csv'] = _file_signature(cfg.get('bayesian_csv_path'))
        payload['bayesian_column'] = cfg.get('bayesian_column')
        payload['bayesian_low_column'] = cfg.get('bayesian_low_column')
        payload['bayesian_high_column'] = cfg.get('bayesian_high_column')
        payload['dynamic_interval_minutes'] = cfg.get('dynamic_interval_minutes', 30)
    else:
        payload['raster'] = _file_signature(cfg.get('raster_path'))
        payload['grid_resolution'] = cfg.get('grid_resolution', 'raw')
    encoded = json.dumps(payload, sort_keys=True).encode('utf-8')
    return hashlib.md5(encoded).hexdigest()


def _load_json(path: Path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            return {}
    return {}


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2))


def _compute_identifier_hash(values):
    digest = hashlib.md5()
    for value in values:
        digest.update(str(value).encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


def _compute_home_factor_hash(person_ids, exposure_calc):
    if exposure_calc is None:
        return "na"
    home_factor_map = getattr(exposure_calc, "home_factor_map", {}) or {}
    digest = hashlib.md5()
    for pid in person_ids:
        digest.update(str(pid).encode('utf-8'))
        digest.update(b'\0')
        factor = float(home_factor_map.get(str(pid), 1.0))
        digest.update(f"{factor:.10f}".encode('utf-8'))
        digest.update(b'\0')
    return digest.hexdigest()


def _load_person_index(cache_root: Path, metadata_hash: str):
    global _PERSON_INDEX_CACHE, _PERSON_INDEX_HASH
    if _PERSON_INDEX_CACHE is not None and _PERSON_INDEX_HASH == metadata_hash:
        return _PERSON_INDEX_CACHE
    index_path = cache_root / 'person_index_all.parquet'
    if not index_path.exists():
        return {}
    df = pd.read_parquet(index_path)
    mapping = dict(zip(df['person_id'].astype(str), df['row_index'].astype(int)))
    _PERSON_INDEX_CACHE = mapping
    _PERSON_INDEX_HASH = metadata_hash
    return mapping


def _memmap_stat(memmap_obj, indices, mode, progress_label=None):
    if memmap_obj is None or len(indices) == 0:
        bin_count = memmap_obj.shape[1] if memmap_obj is not None else 288
        return np.zeros(bin_count, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    bin_count = memmap_obj.shape[1]
    result = np.zeros(bin_count, dtype=np.float64)
    iterator = range(bin_count)
    if progress_label:
        from tqdm import tqdm
        iterator = tqdm(iterator, desc=progress_label, unit="bin", leave=False)
    for bin_idx in iterator:
        column = np.asarray(memmap_obj[indices, bin_idx], dtype=np.float64)
        if column.size == 0:
            continue
        if mode == 'mean':
            result[bin_idx] = column.mean()
        else:
            result[bin_idx] = np.median(column)
    return result


def _chunk_cumulative_sql(exposure_field="pollution_concentration", travel_field="travel_exposure",
                          source_table="detailed_exposure"):
    """
    Build DuckDB SQL to aggregate per-person cumulative exposure curves.

    IMPORTANT: exposure_field / travel_field must be *concentration* (µg/m³),
    not pre-multiplied exposure totals. Duration weighting is handled inside
    the query via overlap_hours.
    """
    return f"""
        WITH base AS (
            SELECT
                person_id,
                start_time,
                end_time,
                COALESCE({exposure_field}, {travel_field}) AS exposure_rate
            FROM {source_table}
            WHERE start_time IS NOT NULL AND end_time IS NOT NULL
        ),
        floored AS (
            SELECT
                person_id,
                start_time,
                end_time,
                exposure_rate,
                date_trunc('minute', start_time) - INTERVAL (EXTRACT(MINUTE FROM start_time)::INT % 5) MINUTE AS floored_start
            FROM base
        ),
        expanded AS (
            SELECT
                person_id,
                gs.bin_start,
                LEAST(end_time, gs.bin_start + INTERVAL '5 minutes') AS overlap_end,
                GREATEST(start_time, gs.bin_start) AS overlap_start,
                exposure_rate
            FROM floored
            CROSS JOIN LATERAL generate_series(
                floored_start,
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
        ),
        binned AS (
            SELECT
                person_id,
                CAST(DATEDIFF('minute', date_trunc('day', bin_start), bin_start) / 5 AS INTEGER) AS bin_index,
                SUM(exposure_rate * overlap_hours) AS exposure_value
            FROM valid
            GROUP BY person_id, bin_index
        ),
        persons AS (
            SELECT DISTINCT person_id FROM base
        ),
        bins AS (
            SELECT range AS bin_index FROM range(288)
        ),
        filled AS (
            SELECT
                persons.person_id,
                bins.bin_index,
                COALESCE(binned.exposure_value, 0) AS exposure_value
            FROM persons
            CROSS JOIN bins
            LEFT JOIN binned USING (person_id, bin_index)
        )
        SELECT
            person_id,
            bin_index,
            SUM(exposure_value) OVER (
                PARTITION BY person_id
                ORDER BY bin_index
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS cumulative_exposure,
            exposure_value AS exposure_increment
        FROM filled
        ORDER BY person_id, bin_index;
    """


def _chunk_home_sql(source_table="detailed_exposure"):
    return f"""
        SELECT person_id, longitude, latitude FROM (
            SELECT
                person_id,
                longitude,
                latitude,
                start_time,
                ROW_NUMBER() OVER (
                    PARTITION BY person_id
                    ORDER BY start_time
                ) AS rn
            FROM {source_table}
            WHERE location_type = 'origin' AND longitude IS NOT NULL AND latitude IS NOT NULL
        )
        WHERE rn = 1;
    """


def _compute_chunk_curves(chunk_path: Path):
    conn = duckdb.connect()
    chunk_str = str(chunk_path).replace("'", "''")
    conn.execute(f"""
        CREATE OR REPLACE TEMP VIEW detailed_exposure AS
        SELECT * FROM read_parquet('{chunk_str}', union_by_name=true)
    """)
    curves = conn.execute(_chunk_cumulative_sql(
        exposure_field="pollution_concentration",
        travel_field="travel_exposure"
    )).fetchdf()
    cols = conn.execute("PRAGMA table_info('detailed_exposure')").fetchdf()['name'].tolist()
    curves_low = pd.DataFrame()
    curves_high = pd.DataFrame()
    if 'exposure_low' in cols:
        travel_low_field = 'travel_exposure_low' if 'travel_exposure_low' in cols else 'travel_exposure'
        curves_low = conn.execute(_chunk_cumulative_sql(
            exposure_field="pollution_low",
            travel_field=travel_low_field
        )).fetchdf()
    if 'exposure_high' in cols:
        travel_high_field = 'travel_exposure_high' if 'travel_exposure_high' in cols else 'travel_exposure'
        curves_high = conn.execute(_chunk_cumulative_sql(
            exposure_field="pollution_high",
            travel_field=travel_high_field
        )).fetchdf()
    homes = conn.execute(_chunk_home_sql()).fetchdf()
    conn.close()
    if not curves.empty:
        curves['person_id'] = curves['person_id'].astype(str)
    if not curves_low.empty:
        curves_low['person_id'] = curves_low['person_id'].astype(str)
        curves = curves.merge(
            curves_low[['person_id','bin_index','cumulative_exposure','exposure_increment']].rename(
                columns={'cumulative_exposure':'cumulative_exposure_low','exposure_increment':'exposure_increment_low'}
            ),
            on=['person_id','bin_index'], how='left'
        )
    if not curves_high.empty:
        curves_high['person_id'] = curves_high['person_id'].astype(str)
        curves = curves.merge(
            curves_high[['person_id','bin_index','cumulative_exposure','exposure_increment']].rename(
                columns={'cumulative_exposure':'cumulative_exposure_high','exposure_increment':'exposure_increment_high'}
            ),
            on=['person_id','bin_index'], how='left'
        )
    if not homes.empty:
        homes['person_id'] = homes['person_id'].astype(str)
    return curves, homes


def _compute_chunk_curves_from_db(conn: duckdb.DuckDBPyConnection, person_ids: List[str]):
    subset_df = pd.DataFrame({'person_id': [str(pid) for pid in person_ids]})
    conn.register('subset_ids', subset_df)
    source_view = "subset_detailed_exposure"
    conn.execute(f"""
        CREATE OR REPLACE TEMP VIEW {source_view} AS
        SELECT d.* FROM detailed_exposure d
        JOIN subset_ids s USING(person_id)
    """)
    curves = conn.execute(_chunk_cumulative_sql(
        exposure_field="pollution_concentration",
        travel_field="travel_exposure",
        source_table=source_view
    )).fetchdf()
    cols = conn.execute(f"PRAGMA table_info('{source_view}')").fetchdf()['name'].tolist()
    curves_low = pd.DataFrame()
    curves_high = pd.DataFrame()
    if 'exposure_low' in cols:
        travel_low_field = 'travel_exposure_low' if 'travel_exposure_low' in cols else 'travel_exposure'
        curves_low = conn.execute(_chunk_cumulative_sql(
            exposure_field="pollution_low",
            travel_field=travel_low_field,
            source_table=source_view
        )).fetchdf()
    if 'exposure_high' in cols:
        travel_high_field = 'travel_exposure_high' if 'travel_exposure_high' in cols else 'travel_exposure'
        curves_high = conn.execute(_chunk_cumulative_sql(
            exposure_field="pollution_high",
            travel_field=travel_high_field,
            source_table=source_view
        )).fetchdf()
    homes = conn.execute(_chunk_home_sql(source_table=source_view)).fetchdf()
    conn.execute(f"DROP VIEW IF EXISTS {source_view}")
    conn.unregister('subset_ids')
    if not curves.empty:
        curves['person_id'] = curves['person_id'].astype(str)
    if not curves_low.empty:
        curves_low['person_id'] = curves_low['person_id'].astype(str)
        curves = curves.merge(
            curves_low[['person_id','bin_index','cumulative_exposure','exposure_increment']].rename(
                columns={'cumulative_exposure':'cumulative_exposure_low','exposure_increment':'exposure_increment_low'}
            ),
            on=['person_id','bin_index'], how='left'
        )
    if not curves_high.empty:
        curves_high['person_id'] = curves_high['person_id'].astype(str)
        curves = curves.merge(
            curves_high[['person_id','bin_index','cumulative_exposure','exposure_increment']].rename(
                columns={'cumulative_exposure':'cumulative_exposure_high','exposure_increment':'exposure_increment_high'}
            ),
            on=['person_id','bin_index'], how='left'
        )
    if not homes.empty:
        homes['person_id'] = homes['person_id'].astype(str)
    return curves, homes


def _build_visualization_cache(cache_root: Path, person_ids, chunk_files,
                               dynamic_home_enabled, exposure_calc,
                               bin_count=288, duckdb_path: str = None,
                               chunk_size: int = 5000):
    cache_root = Path(cache_root)
    if cache_root.exists():
        try:
            shutil.rmtree(cache_root)
        except FileNotFoundError:
            pass
    cache_root.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_root / 'all_metadata.json'
    person_ids = [str(pid) for pid in person_ids]
    num_persons = len(person_ids)
    activity_file = 'activity_curves_all.dat'
    increment_file = 'activity_increments_all.dat'
    home_activity_file = 'home_activity_all.dat'
    home_increment_file = 'home_increments_all.dat'
    index_file = cache_root / 'person_index_all.parquet'
    # Detect availability of low/high exposure fields
    has_low = has_high = False
    if duckdb_path:
        conn_probe = duckdb.connect(duckdb_path)
        probe_cols = conn_probe.execute("PRAGMA table_info('detailed_exposure')").fetchdf()['name'].tolist()
        has_low = 'exposure_low' in probe_cols
        has_high = 'exposure_high' in probe_cols
        conn_probe.close()
    elif chunk_files:
        conn_probe = duckdb.connect()
        probe_path = str(chunk_files[0]).replace("'", "''")
        probe_cols = conn_probe.execute(f"SELECT * FROM read_parquet('{probe_path}', union_by_name=true) LIMIT 0").fetchdf().columns.tolist()
        has_low = 'exposure_low' in probe_cols
        has_high = 'exposure_high' in probe_cols
        conn_probe.close()

    metadata = {
        'person_hash': _compute_identifier_hash(person_ids),
        'person_count': num_persons,
        'bin_count': bin_count,
        'cache_schema_version': VISUALIZATION_CACHE_SCHEMA_VERSION,
        'activity_file': activity_file,
        'increment_file': increment_file,
        'home_activity_file': home_activity_file if dynamic_home_enabled and exposure_calc is not None else '',
        'home_increment_file': home_increment_file if dynamic_home_enabled and exposure_calc is not None else '',
        'activity_low_file': 'activity_curves_low.dat' if has_low else '',
        'activity_high_file': 'activity_curves_high.dat' if has_high else '',
        'home_activity_low_file': 'home_activity_low.dat' if (dynamic_home_enabled and has_low) else '',
        'home_activity_high_file': 'home_activity_high.dat' if (dynamic_home_enabled and has_high) else '',
        'dynamic_home_enabled': bool(dynamic_home_enabled and exposure_calc is not None),
        'home_factor_hash': _compute_home_factor_hash(person_ids, exposure_calc) if dynamic_home_enabled and exposure_calc is not None else 'na',
        'chunk_count': len(chunk_files),
        'created_at': datetime.utcnow().isoformat(),
        'complete': False
    }
    _write_json(metadata_path, metadata)

    from tqdm import tqdm
    activity_mem = np.memmap(cache_root / activity_file, dtype='float32', mode='w+', shape=(num_persons, bin_count))
    increment_mem = np.memmap(cache_root / increment_file, dtype='float32', mode='w+', shape=(num_persons, bin_count))
    activity_low_mem = activity_high_mem = None
    if has_low:
        activity_low_mem = np.memmap(cache_root / metadata['activity_low_file'], dtype='float32', mode='w+', shape=(num_persons, bin_count))
    if has_high:
        activity_high_mem = np.memmap(cache_root / metadata['activity_high_file'], dtype='float32', mode='w+', shape=(num_persons, bin_count))
    pid_to_index = {pid: idx for idx, pid in enumerate(person_ids)}
    home_coords = {}

    if duckdb_path:
        conn = duckdb.connect(duckdb_path)
        total_chunks = (num_persons + chunk_size - 1) // chunk_size
        for i in tqdm(range(total_chunks), desc="Building visualization cache", unit="chunk"):
            start = i * chunk_size
            end = min(start + chunk_size, num_persons)
            curves, homes = _compute_chunk_curves_from_db(conn, person_ids[start:end])
            if not curves.empty:
                for pid, grp in curves.groupby('person_id', sort=False):
                    idx = pid_to_index.get(pid)
                    if idx is None:
                        continue
                    bin_idx = grp['bin_index'].to_numpy(dtype=int)
                    cumulative = np.zeros(bin_count, dtype=np.float32)
                    increments = np.zeros(bin_count, dtype=np.float32)
                    cumulative[bin_idx] = grp['cumulative_exposure'].to_numpy(dtype=np.float32)
                    increments[bin_idx] = grp['exposure_increment'].to_numpy(dtype=np.float32)
                    activity_mem[idx, :] = cumulative
                    increment_mem[idx, :] = increments
                    if has_low and activity_low_mem is not None and 'cumulative_exposure_low' in grp.columns:
                        cumulative_low = np.zeros(bin_count, dtype=np.float32)
                        cumulative_low[bin_idx] = grp['cumulative_exposure_low'].to_numpy(dtype=np.float32)
                        activity_low_mem[idx, :] = cumulative_low
                    if has_high and activity_high_mem is not None and 'cumulative_exposure_high' in grp.columns:
                        cumulative_high = np.zeros(bin_count, dtype=np.float32)
                        cumulative_high[bin_idx] = grp['cumulative_exposure_high'].to_numpy(dtype=np.float32)
                        activity_high_mem[idx, :] = cumulative_high
            if dynamic_home_enabled and exposure_calc is not None and not homes.empty:
                for _, row in homes.iterrows():
                    pid = row['person_id']
                    if pid not in home_coords:
                        lon, lat = row['longitude'], row['latitude']
                        if not (pd.isna(lon) or pd.isna(lat)):
                            home_coords[pid] = (float(lon), float(lat))
        conn.close()
    else:
        for chunk_path in tqdm(chunk_files, desc="Building visualization cache", unit="chunk"):
            curves, homes = _compute_chunk_curves(chunk_path)
            if not curves.empty:
                for pid, grp in curves.groupby('person_id', sort=False):
                    idx = pid_to_index.get(pid)
                    if idx is None:
                        continue
                    bin_idx = grp['bin_index'].to_numpy(dtype=int)
                    cumulative = np.zeros(bin_count, dtype=np.float32)
                    increments = np.zeros(bin_count, dtype=np.float32)
                    cumulative[bin_idx] = grp['cumulative_exposure'].to_numpy(dtype=np.float32)
                    increments[bin_idx] = grp['exposure_increment'].to_numpy(dtype=np.float32)
                    activity_mem[idx, :] = cumulative
                    increment_mem[idx, :] = increments
                    if has_low and activity_low_mem is not None and 'cumulative_exposure_low' in grp.columns:
                        cumulative_low = np.zeros(bin_count, dtype=np.float32)
                        cumulative_low[bin_idx] = grp['cumulative_exposure_low'].to_numpy(dtype=np.float32)
                        activity_low_mem[idx, :] = cumulative_low
                    if has_high and activity_high_mem is not None and 'cumulative_exposure_high' in grp.columns:
                        cumulative_high = np.zeros(bin_count, dtype=np.float32)
                        cumulative_high[bin_idx] = grp['cumulative_exposure_high'].to_numpy(dtype=np.float32)
                        activity_high_mem[idx, :] = cumulative_high
            if dynamic_home_enabled and exposure_calc is not None and not homes.empty:
                for _, row in homes.iterrows():
                    pid = row['person_id']
                    if pid not in home_coords:
                        lon, lat = row['longitude'], row['latitude']
                        if not (pd.isna(lon) or pd.isna(lat)):
                            home_coords[pid] = (float(lon), float(lat))

    activity_mem.flush()
    increment_mem.flush()
    if activity_low_mem is not None:
        activity_low_mem.flush()
    if activity_high_mem is not None:
        activity_high_mem.flush()
    pd.DataFrame({
        'person_id': person_ids,
        'row_index': np.arange(num_persons, dtype=np.int64)
    }).to_parquet(index_file, index=False)

    if dynamic_home_enabled and exposure_calc is not None:
        base_ts = pd.Timestamp("2000-01-01 00:00:00")
        home_factor_map = getattr(exposure_calc, "home_factor_map", {}) or {}
        fallback_pen = float(getattr(exposure_calc, "pen_prox_table", {}).get('work_indoor', {}).get('pen', {}).get('gm', 1.0))
        fallback_prox = float(getattr(exposure_calc, "pen_prox_table", {}).get('work_indoor', {}).get('prox', {}).get('gm', 1.0))
        fallback_factor = fallback_pen * fallback_prox
        home_activity_mem = np.memmap(cache_root / home_activity_file, dtype='float32', mode='w+', shape=(num_persons, bin_count))
        home_increment_mem = np.memmap(cache_root / home_increment_file, dtype='float32', mode='w+', shape=(num_persons, bin_count))
        home_activity_low_mem = home_activity_high_mem = None
        if has_low and metadata.get('home_activity_low_file'):
            home_activity_low_mem = np.memmap(cache_root / metadata['home_activity_low_file'], dtype='float32', mode='w+', shape=(num_persons, bin_count))
        if has_high and metadata.get('home_activity_high_file'):
            home_activity_high_mem = np.memmap(cache_root / metadata['home_activity_high_file'], dtype='float32', mode='w+', shape=(num_persons, bin_count))
        for pid, idx in tqdm(pid_to_index.items(), desc="Computing home curves", total=num_persons, unit="person"):
            coords = home_coords.get(pid)
            home_factor = float(home_factor_map.get(pid, fallback_factor))
            inc = np.zeros(bin_count, dtype=np.float32)
            inc_low = np.zeros(bin_count, dtype=np.float32) if has_low and home_activity_low_mem is not None else None
            inc_high = np.zeros(bin_count, dtype=np.float32) if has_high and home_activity_high_mem is not None else None
            if coords:
                lon, lat = coords
                for b in range(bin_count):
                    ts = base_ts + pd.Timedelta(minutes=5 * b)
                    if has_low or has_high:
                        val_mid, val_lo, val_hi = exposure_calc.get_pollution_at_location(lon, lat, ts, return_low_high=True)
                    else:
                        val_mid = exposure_calc.get_pollution_at_location(lon, lat, ts)
                        val_lo = val_hi = val_mid
                    if not pd.isna(val_mid):
                        inc[b] = float(val_mid) * home_factor * BIN_DURATION_HOURS
                    if inc_low is not None and not pd.isna(val_lo):
                        inc_low[b] = float(val_lo) * home_factor * BIN_DURATION_HOURS
                    if inc_high is not None and not pd.isna(val_hi):
                        inc_high[b] = float(val_hi) * home_factor * BIN_DURATION_HOURS
            home_increment_mem[idx, :] = inc
            home_activity_mem[idx, :] = np.cumsum(inc, dtype=np.float32)
            if inc_low is not None:
                home_activity_low_mem[idx, :] = np.cumsum(inc_low, dtype=np.float32)
            if inc_high is not None:
                home_activity_high_mem[idx, :] = np.cumsum(inc_high, dtype=np.float32)
        home_increment_mem.flush()
        home_activity_mem.flush()
        if home_activity_low_mem is not None:
            home_activity_low_mem.flush()
        if home_activity_high_mem is not None:
            home_activity_high_mem.flush()

    metadata['complete'] = True
    _write_json(metadata_path, metadata)
    global _PERSON_INDEX_CACHE, _PERSON_INDEX_HASH
    _PERSON_INDEX_CACHE = None
    _PERSON_INDEX_HASH = None


def _ensure_visualization_cache(cache_root: Path, comparison_results, temp_root: Path,
                                dynamic_home_enabled: bool, exposure_calc,
                                pollution_fingerprint: str, duckdb_path: str = None):
    cache_root = Path(cache_root)
    metadata_path = cache_root / 'all_metadata.json'
    person_ids = comparison_results['person_id'].astype(str).tolist()
    person_hash = _compute_identifier_hash(person_ids)
    metadata = _load_json(metadata_path)
    manifest_path = Path(temp_root) / 'duckdb_intermediates' / 'exposure_manifest.json'
    exposure_manifest = _load_json(manifest_path)
    chunk_files = []
    if not duckdb_path:
        if exposure_manifest.get('complete'):
            chunk_entries = exposure_manifest.get('chunks', {})
            for key in sorted(chunk_entries, key=lambda x: int(x)):
                chunk_files.append(Path(chunk_entries[key]['detailed']))
        else:
            chunk_files = sorted((Path(temp_root) / 'duckdb_intermediates').glob('exposure_detailed_*.parquet'))
    expected_dynamic = bool(dynamic_home_enabled and exposure_calc is not None)
    expected_home_factor_hash = _compute_home_factor_hash(person_ids, exposure_calc) if expected_dynamic else 'na'
    if (metadata.get('person_hash') == person_hash
            and metadata.get('person_count') == len(person_ids)
            and metadata.get('complete')
            and metadata.get('cache_schema_version') == VISUALIZATION_CACHE_SCHEMA_VERSION
            and metadata.get('pollution_fingerprint') == pollution_fingerprint
            and metadata.get('dynamic_home_enabled') == expected_dynamic
            and metadata.get('home_factor_hash') == expected_home_factor_hash):
        return metadata
    if not duckdb_path and not chunk_files:
        print("No exposure chunk files found for visualization cache.")
        return None
    if duckdb_path:
        print("Building visualization cache from DuckDB table...")
    else:
        print(f"Building visualization cache from {len(chunk_files)} exposure chunk files...")
    _build_visualization_cache(
        cache_root, person_ids, chunk_files, dynamic_home_enabled, exposure_calc,
        duckdb_path=duckdb_path
    )
    metadata = _load_json(metadata_path)
    metadata['person_hash'] = person_hash
    metadata['person_count'] = len(person_ids)
    metadata['cache_schema_version'] = VISUALIZATION_CACHE_SCHEMA_VERSION
    metadata['pollution_fingerprint'] = pollution_fingerprint
    metadata['dynamic_home_enabled'] = expected_dynamic
    metadata['home_factor_hash'] = expected_home_factor_hash
    metadata['complete'] = True
    _write_json(metadata_path, metadata)
    return metadata


def normalize_exposure_metric(metric: str) -> str:
    """Return a supported exposure metric key."""
    if not metric:
        return 'cumulative'
    metric_key = str(metric).strip().lower()
    return metric_key if metric_key in VALID_EXPOSURE_METRICS else 'cumulative'


def exposure_metric_units(metric: str) -> str:
    """Return the unit label for the requested metric."""
    metric_key = normalize_exposure_metric(metric)
    return 'μg/m³·hours' if metric_key == 'cumulative' else 'μg/m³'


def exposure_metric_label(metric: str) -> str:
    """Return a readable label for printing summaries."""
    metric_key = normalize_exposure_metric(metric)
    return 'Cumulative exposure' if metric_key == 'cumulative' else 'Time-weighted average concentration'


def exposure_home_fallback(travel_value: float, metric: str) -> float:
    """Return the fallback home exposure given the output metric."""
    metric_key = normalize_exposure_metric(metric)
    return travel_value * 24.0 if metric_key == 'cumulative' else travel_value


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Loaded configuration from {config_path}")
        config.setdefault('outputs', {})
        outputs = config['outputs']
        outputs.setdefault('results_root', 'results')
        outputs.setdefault('temp_root', 'temp_processing')
        outputs.setdefault('repo_results_root', outputs.get('results_root', 'results'))
        return config
    except Exception as e:
        print(f"Error loading config file {config_path}: {e}")
        return None


def ensure_detailed_exposure_db(parquet_glob: str, temp_root: Path) -> Path:
    """
    Materialize detailed exposure parquet into a DuckDB database table with indexes.
    Reuses the DB if inputs haven't changed.
    """
    temp_root = Path(temp_root or os.environ.get('TEMP_PROCESSING_ROOT', 'temp_processing'))
    intermediate_dir = temp_root / 'duckdb_intermediates'
    intermediate_dir.mkdir(parents=True, exist_ok=True)
    db_path = intermediate_dir / 'detailed_exposure.duckdb'
    manifest_path = intermediate_dir / 'detailed_exposure_db_manifest.json'

    def _list_files(glob_path: str) -> List[Path]:
        glob_path = str(glob_path)
        parent = Path(glob_path).parent
        pattern = Path(glob_path).name
        return sorted(parent.glob(pattern))

    files = _list_files(parquet_glob)
    file_info = []
    for f in files:
        try:
            st = f.stat()
            file_info.append({'path': str(f), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns})
        except OSError:
            file_info.append({'path': str(f), 'size': -1, 'mtime_ns': -1})

    expected = {
        'glob': parquet_glob,
        'files': file_info,
        'table': 'detailed_exposure',
    }

    try:
        if manifest_path.exists() and db_path.exists():
            current = json.loads(manifest_path.read_text())
            if current == expected:
                return db_path
    except Exception:
        pass

    conn = duckdb.connect(db_path.as_posix())
    try:
        conn.execute(f"PRAGMA temp_directory='{temp_root.as_posix()}'")
    except Exception:
        pass
    conn.execute("DROP TABLE IF EXISTS detailed_exposure")
    conn.execute(f"""
        CREATE TABLE detailed_exposure AS
        SELECT * FROM read_parquet('{parquet_glob}', union_by_name=true)
    """)
    try:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_detailed_person ON detailed_exposure(person_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_detailed_start ON detailed_exposure(start_time)")
    except Exception:
        pass
    conn.close()

    try:
        manifest_path.write_text(json.dumps(expected, indent=2))
    except Exception:
        pass
    return db_path


def _resolve_results_path(path):
    results_root = Path(os.environ.get("RESULTS_ROOT", "results"))
    path = Path(path)
    if not path.is_absolute():
        path = results_root / path
    return path


def save_results(comparison_results, detailed_exposure, output_dir=None):
    """
    Save pipeline results to files.
    """
    if output_dir is None:
        output_dir = Path(os.environ.get("RESULTS_ROOT", "results"))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = output_dir / "duckdb_comparison_results.csv"
    detailed_path = output_dir / "duckdb_detailed_exposure.csv"
    comparison_results.to_csv(comparison_path, index=False)
    if isinstance(detailed_exposure, dict) and detailed_exposure.get('type') == 'parquet_glob':
        glob_path = detailed_exposure.get('path')
        if glob_path:
            conn = duckdb.connect(':memory:')
            conn.execute(f"""
                COPY (SELECT * FROM read_parquet('{glob_path}', union_by_name=true))
                TO '{detailed_path.as_posix()}'
                (HEADER, DELIMITER ',');
            """)
            conn.close()
        else:
            raise ValueError("detailed_exposure parquet glob missing")
    else:
        detailed_exposure.to_csv(detailed_path, index=False)

    print("Results saved:")
    print(f"- {comparison_path}")
    print(f"- {detailed_path}")
    print("- duckdb_exposure_comparison.pdf")


def print_detailed_summary(comparison_results, detailed_exposure, config):
    """
    Print detailed summary of the analysis.
    """
    metric = normalize_exposure_metric(config.get('analysis', {}).get('exposure_metric', 'cumulative'))
    units = exposure_metric_units(metric)
    metric_label = exposure_metric_label(metric).lower()

    print("\n=== DUCKDB EXPOSURE ANALYSIS SUMMARY ===")
    print(f"Number of individuals analyzed: {len(comparison_results)}")
    print(f"Mean daily {metric_label} (activity-based): {comparison_results['activity_exposure'].mean():.2f} {units}")
    print(f"Mean daily {metric_label} (home-only): {comparison_results['home_exposure'].mean():.2f} {units}")
    print(f"Exposure ratio (activity/home): {comparison_results['activity_exposure'].mean() / comparison_results['home_exposure'].mean():.2f}x")
    
    # Additional statistics
    print(f"Activity exposure range: {comparison_results['activity_exposure'].min():.2f} to {comparison_results['activity_exposure'].max():.2f} {units}")
    print(f"Home exposure range: {comparison_results['home_exposure'].min():.2f} to {comparison_results['home_exposure'].max():.2f} {units}")
    
    # Pollution concentration statistics
    if isinstance(detailed_exposure, dict) and detailed_exposure.get('type') == 'parquet_glob':
        glob_path = detailed_exposure.get('path')
        if glob_path:
            conn = duckdb.connect(':memory:')
            stats = conn.execute(f"""
                SELECT
                    MIN(pollution_concentration) AS min_val,
                    MAX(pollution_concentration) AS max_val,
                    AVG(pollution_concentration) AS mean_val
                FROM read_parquet('{glob_path}', union_by_name=true)
                WHERE pollution_concentration IS NOT NULL
            """).fetchdf()
            conn.close()
            if not stats.empty:
                row = stats.iloc[0]
                print(f"Pollution concentration range: {row['min_val']:.3f} to {row['max_val']:.3f} μg/m³")
                print(f"Mean pollution concentration: {row['mean_val']:.3f} μg/m³")
        else:
            print("Pollution concentration range: n/a")
            print("Mean pollution concentration: n/a")
    else:
        valid_pollution = detailed_exposure['pollution_concentration'].dropna()
        print(f"Pollution concentration range: {valid_pollution.min():.3f} to {valid_pollution.max():.3f} μg/m³")
        print(f"Mean pollution concentration: {valid_pollution.mean():.3f} μg/m³")
    
    # Configuration summary
    print(f"Grid resolution: {config['pollution'].get('grid_resolution', 'raw')}")
    print(f"Chunk size: {config.get('performance', {}).get('chunk_size', 1000)}")


def print_sample_dictionary(comparison_dict):
    """
    Print a sample of the dictionary format for debugging.
    """
    print(f"\n=== SAMPLE DICTIONARY FORMAT (First 2 PIDs) ===")
    sample_pids = list(comparison_dict.keys())[:2]
    
    for pid in sample_pids:
        person_data = comparison_dict[pid]
        summary = person_data.get('summary', {})
        metadata = person_data.get('metadata', {})
        metric = metadata.get('metric', 'cumulative')
        units = metadata.get('units', 'μg/m³·hours')
        activity_label = "Activity exposure" if metric == 'cumulative' else "Activity TWAC"
        home_label = "Home exposure" if metric == 'cumulative' else "Home TWAC"
        print(f"\nPID {pid}:")
        print(f"  {activity_label}: {summary.get('total_activity_exposure', 0):.2f} {units}")
        print(f"  {home_label}: {summary.get('total_home_exposure', 0):.2f} {units}")
        print(f"  Ratio: {summary.get('exposure_ratio', 0):.2f}x")


def save_dictionary_as_json(comparison_dict, output_path=None):
    """
    Save the comparison dictionary as a proper JSON file for easy viewing in Firefox.
    Uses streaming JSON writing to handle large files for 5M+ people.
    """
    if output_path is None:
        output_path = "pipeline_results.json"
    output_path = _resolve_results_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Saving full population data to JSON: {len(comparison_dict)} individuals")
    
    # Use streaming JSON writing to handle large files
    with open(output_path, 'w') as f:
        f.write('{')  # Start JSON object
        first = True
        for pid, person_data in comparison_dict.items():
            if not first:
                f.write(',')
            first = False
            f.write(f'"{pid}":')
            json.dump(person_data, f, indent=None, separators=(',', ':'), default=str)
        f.write('}')  # End JSON object
    
    print(f"Saved full population dictionary as valid JSON: {output_path}")
    print(f"File contains data for {len(comparison_dict)} individuals")
    print(f"You can open this file in Firefox to view the structured data.")


def cleanup_intermediate_files():
    """
    Clean up intermediate files created during processing.
    PRESERVES intermediate stay points files for future runs.
    """
    print("Cleaning up intermediate files (preserving stay points)...")
    intermediate_files = [f for f in os.listdir('.') if f.startswith('intermediate_')]
    for file in intermediate_files:
        try:
            os.remove(file)
            print(f"Removed {file}")
        except Exception as e:
            print(f"Could not remove {file}: {e}")
    
    # Clean up temporary processing directory but PRESERVE stay points
    temp_dir = Path(os.environ.get('TEMP_PROCESSING_ROOT', 'temp_processing'))
    if temp_dir.exists():
        try:
            shutil.rmtree(temp_dir, ignore_errors=False)
            print(f"Removed temporary directory: {temp_dir}")
        except FileNotFoundError:
            # Directory vanished between exists() check and rmtree; ignore.
            pass
        except Exception as e:
            print(f"Could not clean {temp_dir}: {e}")


def save_visualization_data(detailed_exposure, comparison_results,
                            output_path=None,
                            sample_size=10, random_seed=42,
                            person_ids=None, label=None,
                            debug_person_ids=None,
                            debug_output_dir=None,
                            exposure_metric='cumulative',
                            exposure_calc=None,
                            duckdb_path=None,
                            memory_limit=None,
                            pollution_fingerprint=None,
                            person_naics_map=None):
    """Export visualization-ready statistics using cached exposure data."""
    if comparison_results is None or comparison_results.empty:
        print("Comparison results unavailable for visualization export")
        return None

    if output_path is None:
        output_path = "visualization_data.json"
    output_path = _resolve_results_path(output_path)

    if debug_output_dir is None:
        debug_output_dir = "debug"
    debug_output_dir = _resolve_results_path(debug_output_dir)

    output_parent = Path(output_path).parent
    output_parent.mkdir(parents=True, exist_ok=True)

    metric = normalize_exposure_metric(exposure_metric)
    units = exposure_metric_units(metric)
    dynamic_home_enabled = (
        exposure_calc is not None and
        getattr(exposure_calc, "data_mode", "static") in {"dynamic", "bayesian"}
    )

    comparison_results = comparison_results.copy()
    comparison_results['person_id'] = comparison_results['person_id'].astype(str)

    subset_comparison = comparison_results
    if person_ids:
        subset_set = set(str(pid) for pid in person_ids)
        subset_comparison = comparison_results[comparison_results['person_id'].isin(subset_set)].copy()
        if subset_comparison.empty:
            print(f"No matching exposure data for cohort {label or ''}")
            return None

    subset_ids = subset_comparison['person_id'].tolist()
    if not subset_ids:
        print(f"No individuals found for cohort {label or ''}")
        return None

    temp_root = Path(os.environ.get('TEMP_PROCESSING_ROOT', 'temp_processing'))
    cache_root = temp_root / 'visualization_cache'
    cache_metadata = _ensure_visualization_cache(
        cache_root,
        comparison_results,
        temp_root,
        dynamic_home_enabled,
        exposure_calc,
        pollution_fingerprint or 'na',
        duckdb_path=duckdb_path
    )
    if not cache_metadata:
        print("Unable to prepare visualization cache; skipping export.")
        return None

    num_persons_total = cache_metadata.get('person_count', len(comparison_results))
    bin_count = cache_metadata.get('bin_count', 288)
    activity_mem = np.memmap(cache_root / cache_metadata['activity_file'], dtype='float32',
                             mode='r', shape=(num_persons_total, bin_count))
    increment_mem = np.memmap(cache_root / cache_metadata['increment_file'], dtype='float32',
                              mode='r', shape=(num_persons_total, bin_count))
    activity_low_mem = activity_high_mem = None
    home_activity_low_mem = home_activity_high_mem = None
    if cache_metadata.get('activity_low_file'):
        activity_low_mem = np.memmap(cache_root / cache_metadata['activity_low_file'], dtype='float32',
                                     mode='r', shape=(num_persons_total, bin_count))
    if cache_metadata.get('activity_high_file'):
        activity_high_mem = np.memmap(cache_root / cache_metadata['activity_high_file'], dtype='float32',
                                      mode='r', shape=(num_persons_total, bin_count))
    if cache_metadata.get('home_activity_low_file'):
        home_activity_low_mem = np.memmap(cache_root / cache_metadata['home_activity_low_file'], dtype='float32',
                                          mode='r', shape=(num_persons_total, bin_count))
    if cache_metadata.get('home_activity_high_file'):
        home_activity_high_mem = np.memmap(cache_root / cache_metadata['home_activity_high_file'], dtype='float32',
                                           mode='r', shape=(num_persons_total, bin_count))
    home_activity_mem = None
    home_increment_mem = None
    if cache_metadata.get('home_activity_file') and cache_metadata.get('home_increment_file'):
        home_activity_mem = np.memmap(cache_root / cache_metadata['home_activity_file'], dtype='float32',
                                      mode='r', shape=(num_persons_total, bin_count))
        home_increment_mem = np.memmap(cache_root / cache_metadata['home_increment_file'], dtype='float32',
                                       mode='r', shape=(num_persons_total, bin_count))

    index_map = _load_person_index(cache_root, cache_metadata.get('person_hash'))
    try:
        subset_indices = np.array([index_map[pid] for pid in subset_ids], dtype=np.int64)
    except KeyError as exc:
        print(f"Missing cached exposures for person {exc}; rebuild cache.")
        return None

    progress_prefix = label if label and label != "all_individuals" else None
    def _progress(step: str):
        return f"{progress_prefix}: {step}" if progress_prefix else None

    activity_mean = _memmap_stat(activity_mem, subset_indices, 'mean', _progress("activity mean"))
    activity_median = _memmap_stat(activity_mem, subset_indices, 'median', _progress("activity median"))
    activity_low_mean = activity_low_median = None
    activity_high_mean = activity_high_median = None
    if activity_low_mem is not None:
        activity_low_mean = _memmap_stat(activity_low_mem, subset_indices, 'mean', _progress("activity low mean"))
        activity_low_median = _memmap_stat(activity_low_mem, subset_indices, 'median', _progress("activity low median"))
    if activity_high_mem is not None:
        activity_high_mean = _memmap_stat(activity_high_mem, subset_indices, 'mean', _progress("activity high mean"))
        activity_high_median = _memmap_stat(activity_high_mem, subset_indices, 'median', _progress("activity high median"))
    activity_mean_increment = _memmap_stat(increment_mem, subset_indices, 'mean', _progress("activity mean increment"))
    activity_median_increment = _memmap_stat(increment_mem, subset_indices, 'median', _progress("activity median increment"))

    home_totals = subset_comparison['home_exposure'].fillna(0).values
    home_totals_map = dict(zip(subset_comparison['person_id'], home_totals))

    if home_activity_mem is not None and home_increment_mem is not None:
        home_mean_curve = _memmap_stat(home_activity_mem, subset_indices, 'mean', _progress("home mean"))
        home_median_curve = _memmap_stat(home_activity_mem, subset_indices, 'median', _progress("home median"))
        home_mean_increment = _memmap_stat(home_increment_mem, subset_indices, 'mean', _progress("home mean increment"))
        home_median_increment = _memmap_stat(home_increment_mem, subset_indices, 'median', _progress("home median increment"))
        home_mean_low_curve = home_mean_curve.copy()
        home_mean_high_curve = home_mean_curve.copy()
        if home_activity_low_mem is not None:
            home_mean_low_curve = _memmap_stat(home_activity_low_mem, subset_indices, 'mean', _progress("home low mean"))
            home_mean_low_curve = np.concatenate(([0.0], home_mean_low_curve))
        if home_activity_high_mem is not None:
            home_mean_high_curve = _memmap_stat(home_activity_high_mem, subset_indices, 'mean', _progress("home high mean"))
            home_mean_high_curve = np.concatenate(([0.0], home_mean_high_curve))
    else:
        fractions = (np.arange(1, bin_count + 1) / bin_count)
        frac_ext = np.concatenate(([0.0], fractions))
        home_mean_total = np.mean(home_totals) if len(home_totals) else 0.0
        home_median_total = np.median(home_totals) if len(home_totals) else 0.0
        home_mean_curve = home_mean_total * frac_ext
        home_median_curve = home_median_total * frac_ext
        home_mean_increment = np.concatenate(([0.0], np.ones(bin_count) * (home_mean_total / bin_count if bin_count else 0.0)))
        home_median_increment = np.concatenate(([0.0], np.ones(bin_count) * (home_median_total / bin_count if bin_count else 0.0)))
        home_mean_low_curve = home_mean_curve.copy()
        home_mean_high_curve = home_mean_curve.copy()

    unique_ids = subset_ids.copy()
    rng = np.random.default_rng(random_seed)
    sample_sz = min(sample_size, len(unique_ids))
    debug_ids = [str(pid) for pid in (debug_person_ids or [])]
    forced_ids = [pid for pid in debug_ids if pid in unique_ids]
    remaining_ids = [pid for pid in unique_ids if pid not in forced_ids]
    chosen_ids = list(forced_ids)
    if sample_sz > len(chosen_ids):
        additional = sample_sz - len(chosen_ids)
        if remaining_ids:
            drawn = rng.choice(remaining_ids, size=min(additional, len(remaining_ids)), replace=False).tolist()
            chosen_ids.extend(drawn)
    sample_ids = chosen_ids

    elapsed_vector = np.maximum((np.arange(1, bin_count + 1, dtype=float) * BIN_DURATION_HOURS), BIN_DURATION_HOURS)
    if metric == 'twac':
        activity_mean = np.divide(activity_mean, elapsed_vector, out=np.zeros_like(activity_mean), where=elapsed_vector > 0)
        activity_median = np.divide(activity_median, elapsed_vector, out=np.zeros_like(activity_median), where=elapsed_vector > 0)
        activity_mean_increment = activity_mean_increment / BIN_DURATION_HOURS
        activity_median_increment = activity_median_increment / BIN_DURATION_HOURS
        if home_activity_mem is not None and home_increment_mem is not None:
            home_mean_curve = np.divide(home_mean_curve, elapsed_vector, out=np.zeros_like(home_mean_curve), where=elapsed_vector > 0)
            home_median_curve = np.divide(home_median_curve, elapsed_vector, out=np.zeros_like(home_median_curve), where=elapsed_vector > 0)
            home_mean_increment = home_mean_increment / BIN_DURATION_HOURS
            home_median_increment = home_median_increment / BIN_DURATION_HOURS

    activity_mean_series = np.concatenate(([0.0], activity_mean))
    activity_median_series = np.concatenate(([0.0], activity_median))
    activity_mean_increment_series = np.concatenate(([0.0], activity_mean_increment))
    activity_median_increment_series = np.concatenate(([0.0], activity_median_increment))

    if home_activity_mem is not None and home_increment_mem is not None:
        home_mean_series = np.concatenate(([0.0], home_mean_curve))
        home_median_series = np.concatenate(([0.0], home_median_curve))
        home_mean_increment_series = np.concatenate(([0.0], home_mean_increment))
        home_median_increment_series = np.concatenate(([0.0], home_median_increment))
    else:
        home_mean_series = home_mean_curve.tolist()
        home_median_series = home_median_curve.tolist()
        home_mean_increment_series = home_mean_increment.tolist()
        home_median_increment_series = home_median_increment.tolist()

    # Normalize to lists
    home_mean_series = np.asarray(home_mean_series, dtype=float).tolist()
    home_median_series = np.asarray(home_median_series, dtype=float).tolist()
    home_mean_increment_series = np.asarray(home_mean_increment_series, dtype=float).tolist()
    home_median_increment_series = np.asarray(home_median_increment_series, dtype=float).tolist()

    sample_curves = []
    for pid in sample_ids:
        idx = index_map[pid]
        act_vals = np.asarray(activity_mem[idx, :], dtype=np.float64)
        inc_vals = np.asarray(increment_mem[idx, :], dtype=np.float64)
        act_low_vals = np.asarray(activity_low_mem[idx, :], dtype=np.float64) if activity_low_mem is not None else None
        act_high_vals = np.asarray(activity_high_mem[idx, :], dtype=np.float64) if activity_high_mem is not None else None
        if metric == 'twac':
            with np.errstate(divide='ignore', invalid='ignore'):
                act_vals = np.divide(act_vals, elapsed_vector, out=np.zeros_like(act_vals), where=elapsed_vector > 0)
            inc_vals = inc_vals / BIN_DURATION_HOURS
            if act_low_vals is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    act_low_vals = np.divide(act_low_vals, elapsed_vector, out=np.zeros_like(act_low_vals), where=elapsed_vector > 0)
            if act_high_vals is not None:
                with np.errstate(divide='ignore', invalid='ignore'):
                    act_high_vals = np.divide(act_high_vals, elapsed_vector, out=np.zeros_like(act_high_vals), where=elapsed_vector > 0)
        entry = {
            'person_id': str(pid),
            'activity_curve': np.concatenate(([0.0], act_vals)).tolist(),
            'activity_increment': np.concatenate(([0.0], inc_vals)).tolist()
        }
        if person_naics_map:
            naics_label = person_naics_map.get(str(pid))
            if naics_label and naics_label != "nan":
                entry['naics_group'] = str(naics_label)
        if act_low_vals is not None:
            entry['activity_curve_low'] = np.concatenate(([0.0], act_low_vals)).tolist()
        if act_high_vals is not None:
            entry['activity_curve_high'] = np.concatenate(([0.0], act_high_vals)).tolist()
        if home_activity_mem is not None and home_increment_mem is not None:
            home_vals = np.asarray(home_activity_mem[idx, :], dtype=np.float64)
            home_inc_vals = np.asarray(home_increment_mem[idx, :], dtype=np.float64)
            home_low_vals = np.asarray(home_activity_low_mem[idx, :], dtype=np.float64) if home_activity_low_mem is not None else None
            home_high_vals = np.asarray(home_activity_high_mem[idx, :], dtype=np.float64) if home_activity_high_mem is not None else None
            if metric == 'twac':
                with np.errstate(divide='ignore', invalid='ignore'):
                    home_vals = np.divide(home_vals, elapsed_vector, out=np.zeros_like(home_vals), where=elapsed_vector > 0)
                home_inc_vals = home_inc_vals / BIN_DURATION_HOURS
                if home_low_vals is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        home_low_vals = np.divide(home_low_vals, elapsed_vector, out=np.zeros_like(home_low_vals), where=elapsed_vector > 0)
                if home_high_vals is not None:
                    with np.errstate(divide='ignore', invalid='ignore'):
                        home_high_vals = np.divide(home_high_vals, elapsed_vector, out=np.zeros_like(home_high_vals), where=elapsed_vector > 0)
            entry['home_curve'] = np.concatenate(([0.0], home_vals)).tolist()
            entry['home_increment'] = np.concatenate(([0.0], home_inc_vals)).tolist()
            if home_low_vals is not None:
                entry['home_curve_low'] = np.concatenate(([0.0], home_low_vals)).tolist()
            if home_high_vals is not None:
                entry['home_curve_high'] = np.concatenate(([0.0], home_high_vals)).tolist()
        else:
            home_total = home_totals_map.get(pid, 0.0)
            if metric == 'twac':
                entry['home_curve'] = np.full(bin_count + 1, home_total).tolist()
                entry['home_increment'] = np.full(bin_count + 1, home_total).tolist()
            else:
                fractions = (np.arange(1, bin_count + 1) / bin_count)
                frac_ext = np.concatenate(([0.0], fractions))
                entry['home_curve'] = (home_total * frac_ext).tolist()
                entry['home_increment'] = np.concatenate(([0.0], np.ones(bin_count) * (home_total / bin_count if bin_count else 0.0))).tolist()
        sample_curves.append(entry)

    time_points = (np.arange(0, bin_count + 1, dtype=float) * BIN_DURATION_HOURS).tolist()
    home_mean_values = np.asarray(home_mean_series, dtype=np.float64).tolist()
    home_median_values = np.asarray(home_median_series, dtype=np.float64).tolist()
    home_mean_increment_values = np.asarray(home_mean_increment_series, dtype=np.float64).tolist()
    home_median_increment_values = np.asarray(home_median_increment_series, dtype=np.float64).tolist()

    visualization_payload = {
        'time_points': time_points,
        'activity_mean': activity_mean_series.tolist(),
        'activity_mean_increment': activity_mean_increment_series.tolist(),
        'activity_median': activity_median_series.tolist(),
        'activity_median_increment': activity_median_increment_series.tolist(),
        'home_mean': home_mean_values,
        'home_mean_increment': home_mean_increment_values,
        'home_median': home_median_values,
        'home_median_increment': home_median_increment_values,
        'metadata': {
            'individuals_included': int(len(subset_ids)),
            'sample_size': int(len(sample_curves)),
            'cohort': label or "all_individuals",
            'exposure_metric': metric,
            'units': units
        },
        'sample_curves': sample_curves
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(visualization_payload, f)

    if debug_person_ids:
        os.makedirs(debug_output_dir, exist_ok=True)
        debug_set = set(debug_person_ids)
        for curve in sample_curves:
            if curve['person_id'] in debug_set:
                debug_path = Path(debug_output_dir) / f"debug_timeseries_{curve['person_id']}.json"
                with open(debug_path, 'w', encoding='utf-8') as f:
                    json.dump(curve, f, indent=2)
                print(f"Saved debug time series for {curve['person_id']} to {debug_path}")

    print(f"Saved visualization summary to {output_path}")
    return str(output_path)


def export_exposure_timeseries(comparison_results,
                               exposure_calc=None,
                               exposure_metric='cumulative',
                               output_dir=None,
                               pollution_fingerprint=None,
                               person_naics_map=None,
                               duckdb_path=None,
                               per_person_filename='exposure_timeseries_per_person.parquet',
                               aggregated_filename='exposure_timeseries_aggregated.csv',
                               chunk_size=5000):
    """Persist per-person and aggregated exposure time series (activity vs home).

    Outputs:
      - aggregated CSV with mean/median cumulative and incremental curves + deltas.
      - per-person Parquet with cumulative & incremental curves, home curves, and deltas.
    """

    if comparison_results is None or comparison_results.empty:
        print("No comparison results available; skipping timeseries export")
        return

    metric = normalize_exposure_metric(exposure_metric)
    output_root = Path(output_dir) if output_dir else Path(os.environ.get('RESULTS_ROOT', 'results'))
    output_root.mkdir(parents=True, exist_ok=True)
    agg_path = output_root / aggregated_filename
    person_path = output_root / per_person_filename

    # Reuse visualization cache (builds if missing)
    temp_root = Path(os.environ.get('TEMP_PROCESSING_ROOT', 'temp_processing'))
    cache_root = temp_root / 'visualization_cache'
    pollution_fingerprint = pollution_fingerprint or 'na'
    dynamic_home_enabled = (
        exposure_calc is not None and
        getattr(exposure_calc, "data_mode", "static") in {"dynamic", "bayesian"}
    )
    cache_metadata = _ensure_visualization_cache(
        cache_root,
        comparison_results,
        temp_root,
        dynamic_home_enabled,
        exposure_calc,
        pollution_fingerprint,
        duckdb_path=duckdb_path
    )
    if not cache_metadata:
        print("Could not prepare visualization cache; timeseries export skipped")
        return

    bin_count = cache_metadata.get('bin_count', 288)
    total_persons = cache_metadata.get('person_count', len(comparison_results))
    time_points = np.arange(0, bin_count + 1, dtype=float) * BIN_DURATION_HOURS
    elapsed_vector = np.maximum((np.arange(1, bin_count + 1, dtype=float) * BIN_DURATION_HOURS), BIN_DURATION_HOURS)

    activity_mem = np.memmap(cache_root / cache_metadata['activity_file'], dtype='float32', mode='r', shape=(total_persons, bin_count))
    increment_mem = np.memmap(cache_root / cache_metadata['increment_file'], dtype='float32', mode='r', shape=(total_persons, bin_count))
    activity_low_mem = activity_high_mem = None
    if cache_metadata.get('activity_low_file'):
        activity_low_mem = np.memmap(cache_root / cache_metadata['activity_low_file'], dtype='float32', mode='r', shape=(total_persons, bin_count))
    if cache_metadata.get('activity_high_file'):
        activity_high_mem = np.memmap(cache_root / cache_metadata['activity_high_file'], dtype='float32', mode='r', shape=(total_persons, bin_count))

    home_activity_mem = home_increment_mem = None
    home_activity_low_mem = home_activity_high_mem = None
    if cache_metadata.get('home_activity_file') and cache_metadata.get('home_increment_file'):
        home_activity_mem = np.memmap(cache_root / cache_metadata['home_activity_file'], dtype='float32', mode='r', shape=(total_persons, bin_count))
        home_increment_mem = np.memmap(cache_root / cache_metadata['home_increment_file'], dtype='float32', mode='r', shape=(total_persons, bin_count))
        if cache_metadata.get('home_activity_low_file'):
            home_activity_low_mem = np.memmap(cache_root / cache_metadata['home_activity_low_file'], dtype='float32', mode='r', shape=(total_persons, bin_count))
        if cache_metadata.get('home_activity_high_file'):
            home_activity_high_mem = np.memmap(cache_root / cache_metadata['home_activity_high_file'], dtype='float32', mode='r', shape=(total_persons, bin_count))

    index_map = _load_person_index(cache_root, cache_metadata.get('person_hash', 'na'))
    subset_ids = comparison_results['person_id'].astype(str).tolist()
    try:
        subset_indices = np.array([index_map[pid] for pid in subset_ids], dtype=np.int64)
    except KeyError as exc:
        print(f"Missing person in cache: {exc}; timeseries export aborted")
        return

    # Aggregated curves
    activity_mean = _memmap_stat(activity_mem, subset_indices, 'mean')
    activity_median = _memmap_stat(activity_mem, subset_indices, 'median')
    activity_mean_inc = _memmap_stat(increment_mem, subset_indices, 'mean')
    activity_median_inc = _memmap_stat(increment_mem, subset_indices, 'median')

    home_totals_map = dict(zip(comparison_results['person_id'].astype(str), comparison_results['home_exposure'].fillna(0).values))
    if home_activity_mem is not None and home_increment_mem is not None:
        home_mean_curve = _memmap_stat(home_activity_mem, subset_indices, 'mean')
        home_median_curve = _memmap_stat(home_activity_mem, subset_indices, 'median')
        home_mean_inc = _memmap_stat(home_increment_mem, subset_indices, 'mean')
        home_median_inc = _memmap_stat(home_increment_mem, subset_indices, 'median')
    else:
        fractions = (np.arange(1, bin_count + 1, dtype=float) / bin_count)
        frac_ext = fractions
        home_mean_total = np.mean(list(home_totals_map.values())) if home_totals_map else 0.0
        home_median_total = np.median(list(home_totals_map.values())) if home_totals_map else 0.0
        home_mean_curve = home_mean_total * frac_ext
        home_median_curve = home_median_total * frac_ext
        home_mean_inc = np.ones(bin_count, dtype=float) * (home_mean_total / bin_count if bin_count else 0.0)
        home_median_inc = np.ones(bin_count, dtype=float) * (home_median_total / bin_count if bin_count else 0.0)

    if metric == 'twac':
        activity_mean = np.divide(activity_mean, elapsed_vector, out=np.zeros_like(activity_mean), where=elapsed_vector > 0)
        activity_median = np.divide(activity_median, elapsed_vector, out=np.zeros_like(activity_median), where=elapsed_vector > 0)
        activity_mean_inc = activity_mean_inc / BIN_DURATION_HOURS
        activity_median_inc = activity_median_inc / BIN_DURATION_HOURS
        home_mean_curve = np.divide(home_mean_curve, elapsed_vector, out=np.zeros_like(home_mean_curve), where=elapsed_vector > 0)
        home_median_curve = np.divide(home_median_curve, elapsed_vector, out=np.zeros_like(home_median_curve), where=elapsed_vector > 0)
        home_mean_inc = home_mean_inc / BIN_DURATION_HOURS
        home_median_inc = home_median_inc / BIN_DURATION_HOURS

    delta_mean = activity_mean - home_mean_curve
    delta_median = activity_median - home_median_curve
    delta_mean_inc = activity_mean_inc - home_mean_inc
    delta_median_inc = activity_median_inc - home_median_inc

    agg_df = pd.DataFrame({
        'bin_index': np.arange(bin_count, dtype=int),
        'time_hours': time_points[1:],  # bin endpoints
        'activity_mean': activity_mean,
        'home_mean': home_mean_curve,
        'delta_mean': delta_mean,
        'activity_median': activity_median,
        'home_median': home_median_curve,
        'delta_median': delta_median,
        'activity_mean_increment': activity_mean_inc,
        'home_mean_increment': home_mean_inc,
        'delta_mean_increment': delta_mean_inc,
        'activity_median_increment': activity_median_inc,
        'home_median_increment': home_median_inc,
        'delta_median_increment': delta_median_inc,
    })
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved aggregated exposure timeseries to {agg_path}")

    # Per-person curves (long, chunked)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        print("pyarrow not available; skipping per-person timeseries parquet export")
        return

    writer = None
    bin_indices = np.arange(bin_count, dtype=int)
    bin_hours = (bin_indices + 1) * BIN_DURATION_HOURS
    naics_map = person_naics_map or {}
    home_totals_series = comparison_results.set_index('person_id')['home_exposure']

    from tqdm import tqdm
    chunk_starts = range(0, len(subset_ids), chunk_size)
    for start in tqdm(chunk_starts, desc="Per-person timeseries", unit="chunk"):
        chunk_ids = subset_ids[start:start + chunk_size]
        rows = []
        for pid in chunk_ids:
            idx = index_map[pid]
            act_curve = np.asarray(activity_mem[idx, :], dtype=np.float64)
            act_inc = np.asarray(increment_mem[idx, :], dtype=np.float64)
            home_curve = home_inc = None
            home_low = home_high = None
            if home_activity_mem is not None and home_increment_mem is not None:
                home_curve = np.asarray(home_activity_mem[idx, :], dtype=np.float64)
                home_inc = np.asarray(home_increment_mem[idx, :], dtype=np.float64)
                if home_activity_low_mem is not None:
                    home_low = np.asarray(home_activity_low_mem[idx, :], dtype=np.float64)
                if home_activity_high_mem is not None:
                    home_high = np.asarray(home_activity_high_mem[idx, :], dtype=np.float64)
            else:
                home_total = float(home_totals_series.get(pid, 0.0))
                frac = (np.arange(1, bin_count + 1, dtype=float) / bin_count)
                home_curve = home_total * frac
                home_inc = np.ones(bin_count, dtype=float) * (home_total / bin_count if bin_count else 0.0)
            if metric == 'twac':
                with np.errstate(divide='ignore', invalid='ignore'):
                    act_curve = np.divide(act_curve, elapsed_vector, out=np.zeros_like(act_curve), where=elapsed_vector > 0)
                    act_inc = act_inc / BIN_DURATION_HOURS
                    home_curve = np.divide(home_curve, elapsed_vector, out=np.zeros_like(home_curve), where=elapsed_vector > 0)
                    home_inc = home_inc / BIN_DURATION_HOURS
                    if home_low is not None:
                        home_low = np.divide(home_low, elapsed_vector, out=np.zeros_like(home_low), where=elapsed_vector > 0)
                    if home_high is not None:
                        home_high = np.divide(home_high, elapsed_vector, out=np.zeros_like(home_high), where=elapsed_vector > 0)

            delta_curve = act_curve - home_curve
            delta_inc = act_inc - home_inc
            naics_val = naics_map.get(pid)
            df_chunk = pd.DataFrame({
                'person_id': pid,
                'naics_code': naics_val,
                'bin_index': bin_indices,
                'time_hours': bin_hours,
                'activity_cumulative': act_curve,
                'home_cumulative': home_curve,
                'delta_cumulative': delta_curve,
                'activity_increment': act_inc,
                'home_increment': home_inc,
                'delta_increment': delta_inc,
            })
            if activity_low_mem is not None:
                act_low = np.asarray(activity_low_mem[idx, :], dtype=np.float64)
                if metric == 'twac':
                    with np.errstate(divide='ignore', invalid='ignore'):
                        act_low = np.divide(act_low, elapsed_vector, out=np.zeros_like(act_low), where=elapsed_vector > 0)
                df_chunk['activity_cumulative_low'] = act_low
            if activity_high_mem is not None:
                act_high = np.asarray(activity_high_mem[idx, :], dtype=np.float64)
                if metric == 'twac':
                    with np.errstate(divide='ignore', invalid='ignore'):
                        act_high = np.divide(act_high, elapsed_vector, out=np.zeros_like(act_high), where=elapsed_vector > 0)
                df_chunk['activity_cumulative_high'] = act_high
            if home_low is not None:
                df_chunk['home_cumulative_low'] = home_low
            if home_high is not None:
                df_chunk['home_cumulative_high'] = home_high
            rows.append(df_chunk)
        if not rows:
            continue
        out_df = pd.concat(rows, ignore_index=True)
        table = pa.Table.from_pandas(out_df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(person_path, table.schema, compression="snappy")
        writer.write_table(table)
    if writer is not None:
        writer.close()
        print(f"Saved per-person exposure timeseries to {person_path}")
    else:
        print("No per-person timeseries written (no rows)")

    return str(agg_path), (str(person_path) if writer is not None else None)
    return output_path

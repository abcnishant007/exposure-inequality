#!/usr/bin/env python3
"""
Exposure Calculator Module for DuckDB-based exposure analysis pipeline.

Handles pollution data loading and exposure calculation from stay points.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from tqdm import tqdm
import duckdb
import os
import warnings
import json
import hashlib
import shutil
from pathlib import Path
from scipy.spatial import KDTree
warnings.filterwarnings('ignore')

# Default PEN/PROX parameters (GM, GSD) by microenvironment (used if config missing)
DEFAULT_PEN_PROX_TABLE = {
    "work_indoor": {
        "pen": {"gm": 0.385, "gsd": 1.69},
        "prox": {"gm": 1.0, "gsd": 1.0},
    },
    "work_outdoor_near_road": {
        "pen": {"gm": 1.0, "gsd": 1.0},
        "prox": {"gm": 1.803, "gsd": 2.97},
    },
    "school_indoor": {
        "pen": {"gm": 0.385, "gsd": 1.69},
        "prox": {"gm": 1.0, "gsd": 1.0},
    },
    "school_outdoor": {
        "pen": {"gm": 1.0, "gsd": 1.0},
        "prox": {"gm": 1.0, "gsd": 1.0},
    },
}


class DuckDBExposureCalculator:
    """
    Module 2: Calculate pollution exposure with DuckDB for memory efficiency.
    """
    
    def __init__(self, pollution_raster_path, study_area_geojson_path, grid_resolution='raw',
                 travel_exposure=0.5, travel_exposure_type='mid-point-teleportation',
                 data_mode='static', dynamic_csv_path=None, bayesian_csv_path=None, 
                 bayesian_column='pm25_p50', bayesian_low_column=None, bayesian_high_column=None,
                 hapem_cfg=None, travel_hapem=None, pen_prox_cfg=None):
        self.pollution_raster_path = pollution_raster_path
        self.study_area_geojson_path = study_area_geojson_path
        self.grid_resolution = grid_resolution
        self.travel_exposure = travel_exposure  # Exposure concentration during travel (μg/m³)
        # 'mid-point-teleportation' (default), 'hardcoded', 'home_exposure',
        # 'origin_destination_mean', 'centroid_ambient'
        self.travel_exposure_type = travel_exposure_type
        travel_hapem = travel_hapem or {}
        self.travel_pen = float(travel_hapem.get('pen', 1.0))
        self.travel_prox = float(travel_hapem.get('prox', 1.0))
        pen_prox_cfg = pen_prox_cfg or {}
        self.pen_prox_sampling = str(pen_prox_cfg.get('sampling', 'stochastic')).lower()
        self.pen_prox_seed = int(pen_prox_cfg.get('seed', 42))
        self.school_p_out = float(pen_prox_cfg.get('school_p_outdoor', 0.2))
        table_cfg = pen_prox_cfg.get('table', {})
        # Merge table_cfg over defaults (shallow merge per microenv)
        self.pen_prox_table = DEFAULT_PEN_PROX_TABLE.copy()
        for k, v in table_cfg.items():
            if isinstance(v, dict):
                merged = self.pen_prox_table.get(k, {}).copy()
                merged.update(v)
                self.pen_prox_table[k] = merged
            else:
                self.pen_prox_table[k] = v
        self.pollution_data = None
        self.masked_pollution = None
        self.masked_transform = None
        self.data_mode = (data_mode or 'static').lower()
        self.dynamic_csv_path = dynamic_csv_path
        self.bayesian_csv_path = bayesian_csv_path
        self.bayesian_column = bayesian_column
        self.bayesian_low_column = bayesian_low_column
        self.bayesian_high_column = bayesian_high_column
        # HAPEM-like attenuation config (optional; currently only for work stays)
        default_hapem = {
            "enable_work_bernoulli": False,
            "school_outdoor_fraction": 0.0,
            "enable_school_bernoulli": False,
        }
        # Support nested hapem config {work:{...}, school:{...}} from config.yaml
        if hapem_cfg:
            flat = {}
            work_cfg = hapem_cfg.get("work", {}) if isinstance(hapem_cfg, dict) else {}
            school_cfg = hapem_cfg.get("school", {}) if isinstance(hapem_cfg, dict) else {}
            if work_cfg:
                flat["enable_work_bernoulli"] = work_cfg.get("enable_bernoulli", default_hapem["enable_work_bernoulli"])
            if school_cfg:
                flat["school_outdoor_fraction"] = school_cfg.get("outdoor_fraction", default_hapem["school_outdoor_fraction"])
                flat["enable_school_bernoulli"] = school_cfg.get("enable_bernoulli", default_hapem["enable_school_bernoulli"])
            # also allow flat keys
            for k,v in hapem_cfg.items():
                if k in default_hapem:
                    flat[k]=v
            default_hapem.update(flat)
        self.hapem_cfg = default_hapem
        # Enforce expected-value mode only in deterministic sampling
        if self.pen_prox_sampling == 'stochastic':
            if not self.hapem_cfg.get("enable_work_bernoulli", True):
                raise ValueError("Expected-value HAPEM mode (work) is only allowed with deterministic sampling.")
            if not self.hapem_cfg.get("enable_school_bernoulli", True):
                raise ValueError("Expected-value HAPEM mode (school) is only allowed with deterministic sampling.")
        self.dynamic_time_slices = {}
        self.dynamic_time_index = np.array([], dtype=int)
        self._lookup_cache = {}
        self.home_pen_map = {}
        self.home_prox_map = {}
        self.home_factor_map = {}

        # Set time resolution based on data mode
        if self.data_mode == 'bayesian':
            self.dynamic_interval_minutes = 30
        elif self.data_mode == 'dynamic':
            self.dynamic_interval_minutes = 30
        else:
            self.dynamic_interval_minutes = 5

        self.home_locations = {}

        if self.data_mode == 'dynamic':
            loaded = self._load_dynamic_pollution()
            if not loaded:
                print("Falling back to static raster mode due to dynamic load failure.")
                self.data_mode = 'static'
                self._load_pollution_data()
        elif self.data_mode == 'bayesian':
            loaded = self._load_bayesian_pollution()
            if not loaded:
                print("Falling back to static raster mode due to Bayesian load failure.")
                self.data_mode = 'static'
                self._load_pollution_data()
        else:
            self._load_pollution_data()

    def _sample_lognormal(self, gm: float, gsd: float, rng: np.random.Generator, deterministic: bool) -> float:
        if deterministic or gsd == 1.0:
            return float(gm)
        return float(rng.lognormal(mean=np.log(gm), sigma=np.log(gsd)))

    def _get_pen_prox_cfg(self, microenv_key: str, fallback_keys=None):
        fallback_keys = fallback_keys or []
        keys = [microenv_key] + list(fallback_keys)
        for key in keys:
            cfg = self.pen_prox_table.get(key, {})
            if isinstance(cfg, dict) and 'pen' in cfg and 'prox' in cfg:
                return cfg['pen'], cfg['prox']
        return {'gm': 1.0, 'gsd': 1.0}, {'gm': 1.0, 'gsd': 1.0}

    def _build_person_pen_prox(self, stay_points: pd.DataFrame, temp_root: Path):
        """Precompute per-person work/school microenvironment factors and outdoor flags."""
        # Get one probability per person (first non-null)
        prob_df = (
            stay_points[['person_id', 'p_outdoor_work']]
            .dropna(subset=['p_outdoor_work'])
            .drop_duplicates(subset='person_id')
        )
        prob_map = dict(zip(prob_df['person_id'].astype(str), prob_df['p_outdoor_work'].astype(float)))

        rng = np.random.default_rng(self.pen_prox_seed)
        deterministic = self.pen_prox_sampling != 'stochastic'

        records = []
        pen_map = {}
        prox_map = {}
        outdoor_map = {}
        school_pen_map = {}
        school_prox_map = {}
        school_outdoor_map = {}
        home_pen_map = {}
        home_prox_map = {}

        for pid in stay_points['person_id'].astype(str).unique():
            p_out = prob_map.get(pid)
            if p_out is None or np.isnan(p_out):
                p_out = 0.0
            if deterministic and not self.hapem_cfg.get("enable_work_bernoulli", True):
                pen_in = self.pen_prox_table['work_indoor']['pen']['gm']
                prox_in = self.pen_prox_table['work_indoor']['prox']['gm']
                pen_out = self.pen_prox_table['work_outdoor_near_road']['pen']['gm']
                prox_out = self.pen_prox_table['work_outdoor_near_road']['prox']['gm']
                pen_val = float(p_out * pen_out + (1.0 - p_out) * pen_in)
                prox_val = float(p_out * prox_out + (1.0 - p_out) * prox_in)
                is_outdoor = bool(p_out >= 0.5)
            else:
                is_outdoor = bool(rng.binomial(1, p_out))
                if is_outdoor:
                    pen_cfg = self.pen_prox_table['work_outdoor_near_road']['pen']
                    prox_cfg = self.pen_prox_table['work_outdoor_near_road']['prox']
                else:
                    pen_cfg = self.pen_prox_table['work_indoor']['pen']
                    prox_cfg = self.pen_prox_table['work_indoor']['prox']

                pen_val = self._sample_lognormal(pen_cfg['gm'], pen_cfg['gsd'], rng, deterministic)
                prox_val = self._sample_lognormal(prox_cfg['gm'], prox_cfg['gsd'], rng, deterministic)

            pen_map[pid] = pen_val
            prox_map[pid] = prox_val
            outdoor_map[pid] = is_outdoor

            # School draw (per person)
            if deterministic and not self.hapem_cfg.get("enable_school_bernoulli", True):
                p_school_out = float(self.school_p_out)
                s_pen_in = self.pen_prox_table['school_indoor']['pen']['gm']
                s_prox_in = self.pen_prox_table['school_indoor']['prox']['gm']
                s_pen_out = self.pen_prox_table['school_outdoor']['pen']['gm']
                s_prox_out = self.pen_prox_table['school_outdoor']['prox']['gm']
                s_pen_val = float(p_school_out * s_pen_out + (1.0 - p_school_out) * s_pen_in)
                s_prox_val = float(p_school_out * s_prox_out + (1.0 - p_school_out) * s_prox_in)
                is_school_outdoor = bool(p_school_out >= 0.5)
            else:
                is_school_outdoor = bool(rng.binomial(1, self.school_p_out))
                if is_school_outdoor:
                    s_pen_cfg = self.pen_prox_table['school_outdoor']['pen']
                    s_prox_cfg = self.pen_prox_table['school_outdoor']['prox']
                else:
                    s_pen_cfg = self.pen_prox_table['school_indoor']['pen']
                    s_prox_cfg = self.pen_prox_table['school_indoor']['prox']
                s_pen_val = self._sample_lognormal(s_pen_cfg['gm'], s_pen_cfg['gsd'], rng, deterministic)
                s_prox_val = self._sample_lognormal(s_prox_cfg['gm'], s_prox_cfg['gsd'], rng, deterministic)

            school_pen_map[pid] = s_pen_val
            school_prox_map[pid] = s_prox_val
            school_outdoor_map[pid] = is_school_outdoor

            # Home indoor factor (sampled or deterministic via same sampling mode)
            h_pen_cfg, h_prox_cfg = self._get_pen_prox_cfg(
                'home_indoor',
                fallback_keys=['school_indoor', 'work_indoor']
            )
            h_pen_val = self._sample_lognormal(h_pen_cfg['gm'], h_pen_cfg['gsd'], rng, deterministic)
            h_prox_val = self._sample_lognormal(h_prox_cfg['gm'], h_prox_cfg['gsd'], rng, deterministic)
            home_pen_map[pid] = h_pen_val
            home_prox_map[pid] = h_prox_val

            records.append({
                'person_id': pid,
                'p_outdoor_work': p_out,
                'work_outdoor_flag': is_outdoor,
                'pen': pen_val,
                'prox': prox_val,
                'school_outdoor_flag': is_school_outdoor,
                'school_pen': s_pen_val,
                'school_prox': s_prox_val,
                'home_pen': h_pen_val,
                'home_prox': h_prox_val,
                'home_factor': h_pen_val * h_prox_val,
                'sampling_mode': self.pen_prox_sampling,
                'seed': self.pen_prox_seed,
            })

        df = pd.DataFrame(records)
        results_root = Path(os.environ.get('RESULTS_ROOT', 'results'))
        results_root.mkdir(parents=True, exist_ok=True)
        out_path = results_root / 'per_person_pen_prox.csv'
        try:
            df.to_csv(out_path, index=False)
        except Exception:
            pass
        return (
            outdoor_map,
            pen_map,
            prox_map,
            school_outdoor_map,
            school_pen_map,
            school_prox_map,
            home_pen_map,
            home_prox_map,
            out_path,
        )

    def _build_person_pen_prox_from_chunks(self, chunk_files, temp_root: Path):
        """Precompute per-person PEN/PROX using streaming stay-point chunks."""
        rng = np.random.default_rng(self.pen_prox_seed)
        deterministic = self.pen_prox_sampling != 'stochastic'

        prob_map = {}
        person_ids = set()
        for path in chunk_files:
            try:
                df = pd.read_parquet(path, columns=['person_id', 'p_outdoor_work'])
            except Exception:
                df = pd.read_parquet(path)
            if df.empty:
                continue
            df['person_id'] = df['person_id'].astype(str)
            person_ids.update(df['person_id'].unique().tolist())
            for pid, p_out in zip(df['person_id'].values, df['p_outdoor_work'].values):
                if pid in prob_map:
                    continue
                if p_out is None or (isinstance(p_out, float) and np.isnan(p_out)):
                    continue
                prob_map[pid] = float(p_out)

        records = []
        pen_map = {}
        prox_map = {}
        outdoor_map = {}
        school_pen_map = {}
        school_prox_map = {}
        school_outdoor_map = {}
        home_pen_map = {}
        home_prox_map = {}

        for pid in person_ids:
            p_out = prob_map.get(pid, 0.0)
            if deterministic and not self.hapem_cfg.get("enable_work_bernoulli", True):
                pen_in = self.pen_prox_table['work_indoor']['pen']['gm']
                prox_in = self.pen_prox_table['work_indoor']['prox']['gm']
                pen_out = self.pen_prox_table['work_outdoor_near_road']['pen']['gm']
                prox_out = self.pen_prox_table['work_outdoor_near_road']['prox']['gm']
                pen_val = float(p_out * pen_out + (1.0 - p_out) * pen_in)
                prox_val = float(p_out * prox_out + (1.0 - p_out) * prox_in)
                is_outdoor = bool(p_out >= 0.5)
            else:
                is_outdoor = bool(rng.binomial(1, p_out))
                if is_outdoor:
                    pen_cfg = self.pen_prox_table['work_outdoor_near_road']['pen']
                    prox_cfg = self.pen_prox_table['work_outdoor_near_road']['prox']
                else:
                    pen_cfg = self.pen_prox_table['work_indoor']['pen']
                    prox_cfg = self.pen_prox_table['work_indoor']['prox']

                pen_val = self._sample_lognormal(pen_cfg['gm'], pen_cfg['gsd'], rng, deterministic)
                prox_val = self._sample_lognormal(prox_cfg['gm'], prox_cfg['gsd'], rng, deterministic)

            pen_map[pid] = pen_val
            prox_map[pid] = prox_val
            outdoor_map[pid] = is_outdoor

            if deterministic and not self.hapem_cfg.get("enable_school_bernoulli", True):
                p_school_out = float(self.school_p_out)
                s_pen_in = self.pen_prox_table['school_indoor']['pen']['gm']
                s_prox_in = self.pen_prox_table['school_indoor']['prox']['gm']
                s_pen_out = self.pen_prox_table['school_outdoor']['pen']['gm']
                s_prox_out = self.pen_prox_table['school_outdoor']['prox']['gm']
                s_pen_val = float(p_school_out * s_pen_out + (1.0 - p_school_out) * s_pen_in)
                s_prox_val = float(p_school_out * s_prox_out + (1.0 - p_school_out) * s_prox_in)
                is_school_outdoor = bool(p_school_out >= 0.5)
            else:
                is_school_outdoor = bool(rng.binomial(1, self.school_p_out))
                if is_school_outdoor:
                    s_pen_cfg = self.pen_prox_table['school_outdoor']['pen']
                    s_prox_cfg = self.pen_prox_table['school_outdoor']['prox']
                else:
                    s_pen_cfg = self.pen_prox_table['school_indoor']['pen']
                    s_prox_cfg = self.pen_prox_table['school_indoor']['prox']
                s_pen_val = self._sample_lognormal(s_pen_cfg['gm'], s_pen_cfg['gsd'], rng, deterministic)
                s_prox_val = self._sample_lognormal(s_prox_cfg['gm'], s_prox_cfg['gsd'], rng, deterministic)

            school_pen_map[pid] = s_pen_val
            school_prox_map[pid] = s_prox_val
            school_outdoor_map[pid] = is_school_outdoor

            h_pen_cfg, h_prox_cfg = self._get_pen_prox_cfg(
                'home_indoor',
                fallback_keys=['school_indoor', 'work_indoor']
            )
            h_pen_val = self._sample_lognormal(h_pen_cfg['gm'], h_pen_cfg['gsd'], rng, deterministic)
            h_prox_val = self._sample_lognormal(h_prox_cfg['gm'], h_prox_cfg['gsd'], rng, deterministic)
            home_pen_map[pid] = h_pen_val
            home_prox_map[pid] = h_prox_val

            records.append({
                'person_id': pid,
                'p_outdoor_work': p_out,
                'work_outdoor_flag': is_outdoor,
                'pen': pen_val,
                'prox': prox_val,
                'school_outdoor_flag': is_school_outdoor,
                'school_pen': s_pen_val,
                'school_prox': s_prox_val,
                'home_pen': h_pen_val,
                'home_prox': h_prox_val,
                'home_factor': h_pen_val * h_prox_val,
                'sampling_mode': self.pen_prox_sampling,
                'seed': self.pen_prox_seed,
            })

        df = pd.DataFrame(records)
        results_root = Path(os.environ.get('RESULTS_ROOT', 'results'))
        results_root.mkdir(parents=True, exist_ok=True)
        out_path = results_root / 'per_person_pen_prox.csv'
        try:
            df.to_csv(out_path, index=False)
        except Exception:
            pass
        return (
            outdoor_map,
            pen_map,
            prox_map,
            school_outdoor_map,
            school_pen_map,
            school_prox_map,
            home_pen_map,
            home_prox_map,
            out_path,
        )

    def _build_home_locations_from_chunks(self, chunk_files):
        """Collect per-person home locations from stay-point chunks (origin stays)."""
        home_locations = {}
        for path in chunk_files:
            try:
                df = pd.read_parquet(path, columns=['person_id', 'location_type', 'longitude', 'latitude', 'start_time'])
            except Exception:
                df = pd.read_parquet(path)
            if df.empty:
                continue
            origin_stays = df[df['location_type'] == 'origin'].copy()
            if origin_stays.empty:
                continue
            origin_stays.sort_values('start_time', inplace=True)
            firsts = origin_stays.drop_duplicates(subset='person_id', keep='first')
            for _, row in firsts.iterrows():
                pid = str(row['person_id'])
                if pid in home_locations:
                    continue
                if pd.isna(row['longitude']) or pd.isna(row['latitude']):
                    home_locations[pid] = None
                    continue
                home_locations[pid] = (float(row['longitude']), float(row['latitude']))
        return home_locations
    
    def _load_pollution_data(self):
        """Load and mask pollution data to study area."""
        print(f"Loading pollution data with {self.grid_resolution} resolution...")
        
        # Load study area
        study_area = gpd.read_file(self.study_area_geojson_path)
        
        # Load and mask pollution data
        try:
            with rasterio.open(self.pollution_raster_path) as src:
                out_image, out_transform = mask(
                    src, 
                    study_area.geometry, 
                    crop=True, 
                    filled=True
                )
                self.masked_pollution = out_image[0]
                self.masked_transform = out_transform
            print("Pollution data loaded and masked")
        except Exception as e:
            print(f"Warning: Could not mask pollution data: {e}")
            with rasterio.open(self.pollution_raster_path) as src:
                self.masked_pollution = src.read(1)
                self.masked_transform = src.transform
            print("Using full pollution raster")
        
        # Build spatial index for fast pollution lookups
        self._build_spatial_index()

    def _load_dynamic_pollution(self):
        """Load dynamic 30-min pollution CSV and build spatial indices per time slice."""
        if not self.dynamic_csv_path or not os.path.exists(self.dynamic_csv_path):
            print(f"Dynamic pollution CSV '{self.dynamic_csv_path}' not found.")
            return False
        try:
            df = pd.read_csv(self.dynamic_csv_path, parse_dates=["timestamp_utc"])
        except Exception as e:
            print(f"Failed to read dynamic pollution CSV: {e}")
            return False

        required_cols = {"lon", "lat", "pm25", "timestamp_utc"}
        missing = required_cols - set(df.columns)
        if missing:
            print(f"Dynamic pollution CSV missing columns: {missing}")
            return False

        df = df.dropna(subset=["lon", "lat", "pm25", "timestamp_utc"]).copy()
        if df.empty:
            print("Dynamic pollution CSV has no valid rows after dropping NaNs.")
            return False

        df["minute_of_day"] = df["timestamp_utc"].dt.hour * 60 + df["timestamp_utc"].dt.minute
        self.dynamic_time_slices = {}

        for minute, g in df.groupby("minute_of_day"):
            coords = g[["lon", "lat"]].to_numpy(dtype=float)
            values = g["pm25"].to_numpy(dtype=float)
            valid = ~np.isnan(values)
            coords = coords[valid]
            values = values[valid]
            if coords.size == 0:
                continue
            tree = KDTree(coords)
            self.dynamic_time_slices[int(minute)] = {
                "tree": tree,
                "values": values
            }

        self.dynamic_time_index = np.array(sorted(self.dynamic_time_slices.keys()), dtype=int)
        if self.dynamic_time_index.size == 0:
            print("Dynamic pollution CSV yielded no usable time slices.")
            return False

        print(f"Loaded dynamic pollution rasters for {len(self.dynamic_time_index)} time slices.")
        return True
    
    def _load_bayesian_pollution(self):
        """Load Bayesian fusion CSV with quantiles and build spatial indices per time slice."""
        if not self.bayesian_csv_path or not os.path.exists(self.bayesian_csv_path):
            print(f"Bayesian pollution CSV '{self.bayesian_csv_path}' not found.")
            return False
        try:
            df = pd.read_csv(self.bayesian_csv_path)
        except Exception as e:
            print(f"Failed to read Bayesian pollution CSV: {e}")
            return False

        # Check for required columns
        # Accept both lon/lat and longitude/latitude, and timestamp or timestamp_utc
        if 'timestamp_utc' not in df.columns and 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'timestamp_utc'})
        if 'lon' not in df.columns and 'longitude' in df.columns:
            df = df.rename(columns={'longitude': 'lon'})
        if 'lat' not in df.columns and 'latitude' in df.columns:
            df = df.rename(columns={'latitude': 'lat'})

        required_cols = {"lon", "lat", "timestamp_utc"}
        missing = required_cols - set(df.columns)
        if missing:
            print(f"Bayesian pollution CSV missing required columns: {missing}")
            return False

        # Check if the requested bayesian_column exists
        if self.bayesian_column not in df.columns:
            print(f"Bayesian column '{self.bayesian_column}' not found in CSV.")
            print(f"Available columns: {list(df.columns)}")
            # Try to fall back to 'pm25' if available (mean file)
            if 'pm25' in df.columns:
                print(f"Falling back to 'pm25' column (mean)")
                self.bayesian_column = 'pm25'
            else:
                # Try to find any pm25-related column
                pm25_cols = [col for col in df.columns if 'pm25' in col]
                if pm25_cols:
                    print(f"Available PM25 columns: {pm25_cols}")
                    # Prefer pm25_p50 if available
                    if 'pm25_p50' in pm25_cols:
                        self.bayesian_column = 'pm25_p50'
                        print(f"Using '{self.bayesian_column}' as default")
                    else:
                        self.bayesian_column = pm25_cols[0]
                        print(f"Using first available PM25 column: '{self.bayesian_column}'")
                else:
                    print("No PM25-related columns found in Bayesian CSV.")
                    return False

        # Identify optional low/high columns
        low_col = self.bayesian_low_column
        high_col = self.bayesian_high_column
        if low_col and low_col not in df.columns:
            print(f"Low quantile column '{low_col}' not found; disabling low quantile output.")
            low_col = None
        if high_col and high_col not in df.columns:
            print(f"High quantile column '{high_col}' not found; disabling high quantile output.")
            high_col = None

        keep_cols = ["lon", "lat", self.bayesian_column, "timestamp_utc"]
        if low_col:
            keep_cols.append(low_col)
        if high_col:
            keep_cols.append(high_col)

        # Ensure timestamp_utc parsed
        df = df.dropna(subset=["lon", "lat", self.bayesian_column, "timestamp_utc"]).copy()
        try:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors='coerce', utc=True)
        except Exception:
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors='coerce')
        df = df.dropna(subset=["timestamp_utc"])
        if df.empty:
            print(f"Bayesian pollution CSV has no valid rows after dropping NaNs for column '{self.bayesian_column}'.")
            return False

        df["minute_of_day"] = df["timestamp_utc"].dt.hour * 60 + df["timestamp_utc"].dt.minute
        self.dynamic_time_slices = {}

        for minute, g in df.groupby("minute_of_day"):
            coords = g[["lon", "lat"]].to_numpy(dtype=float)
            values = g[self.bayesian_column].to_numpy(dtype=float)
            valid = ~np.isnan(values)
            coords = coords[valid]
            values = values[valid]
            if coords.size == 0:
                continue
            entry = {
                "tree": KDTree(coords),
                "values": values
            }
            if low_col:
                entry["values_low"] = g[low_col].to_numpy(dtype=float)[valid]
            if high_col:
                entry["values_high"] = g[high_col].to_numpy(dtype=float)[valid]
            self.dynamic_time_slices[int(minute)] = entry

        self.dynamic_time_index = np.array(sorted(self.dynamic_time_slices.keys()), dtype=int)
        if self.dynamic_time_index.size == 0:
            print("Bayesian pollution CSV yielded no usable time slices.")
            return False

        print(f"Loaded Bayesian pollution data (column: '{self.bayesian_column}') for {len(self.dynamic_time_index)} time slices.")
        return True
    
    def _build_spatial_index(self):
        """Build KD-tree spatial index for fast pollution lookups."""
        if self.masked_pollution is None:
            return
            
        print("Building spatial index for fast pollution lookups...")
        
        # Get all valid grid coordinates and their pollution values
        valid_coords = []
        pollution_values = []
        
        rows, cols = self.masked_pollution.shape
        for row in range(rows):
            for col in range(cols):
                pollution_val = self.masked_pollution[row, col]
                if not np.isnan(pollution_val) and pollution_val > 0:
                    # Convert grid coordinates to geographic coordinates
                    lon, lat = rasterio.transform.xy(self.masked_transform, row, col)
                    valid_coords.append([lon, lat])
                    pollution_values.append(pollution_val)
        
        if valid_coords:
            # Build KD-tree for fast nearest neighbor search
            self.kdtree = KDTree(valid_coords)
            self.kdtree_pollution_values = np.array(pollution_values)
            print(f"Built spatial index with {len(valid_coords)} valid pollution points")
        else:
            print("Warning: No valid pollution points found for spatial index")
            self.kdtree = None
            self.kdtree_pollution_values = None
    
    def _cache_key(self, longitude, latitude, timestamp):
        if pd.isna(longitude) or pd.isna(latitude):
            return None
        minute = -1
        if timestamp is not None and not (isinstance(timestamp, float) and np.isnan(timestamp)):
            if not isinstance(timestamp, pd.Timestamp):
                try:
                    timestamp = pd.to_datetime(timestamp)
                except Exception:
                    timestamp = None
            if timestamp is not None and not pd.isna(timestamp):
                minute = int(timestamp.hour * 60 + timestamp.minute)
        return (round(float(longitude), 5), round(float(latitude), 5), minute)

    def get_pollution_at_location(self, longitude, latitude, timestamp=None, return_low_high=False):
        """
        Get pollution value at specific geographic coordinates.
        Uses KD-tree spatial index for fast lookups when available.
        """
        if pd.isna(longitude) or pd.isna(latitude):
            return (np.nan, np.nan, np.nan) if return_low_high else np.nan

        cache_key = self._cache_key(longitude, latitude, timestamp)
        if cache_key and cache_key in self._lookup_cache:
            val = self._lookup_cache[cache_key]
            if return_low_high:
                if isinstance(val, tuple):
                    return val
                else:
                    return (val, val, val)
            else:
                if isinstance(val, tuple):
                    return val[0]
                return val

        # For Bayesian/dynamic mode, try dynamic lookup first
        if self.data_mode in ['dynamic', 'bayesian'] and self.dynamic_time_index.size > 0:
            result = self._get_dynamic_pollution(longitude, latitude, timestamp, return_low_high=return_low_high)
            if (not return_low_high and not np.isnan(result)) or (return_low_high and not np.isnan(result[0])):
                if cache_key:
                    self._lookup_cache[cache_key] = result
                return result
            # If dynamic lookup returns NaN, fall back to static raster
            print(f"DEBUG: Dynamic lookup failed for ({longitude:.4f}, {latitude:.4f}), falling back to static raster")
        
        # DEBUG MODE: Return fixed value for testing (easy to identify if accidentally enabled)
        # if hasattr(self, 'debug_mode') and self.debug_mode:
        #    return -0.05  # Fixed value to easily identify if debug mode is accidentally enabled
            
        if self.masked_pollution is None:
            return (np.nan, np.nan, np.nan) if return_low_high else np.nan
        
        # Try KD-tree spatial index first (100x faster)
        if hasattr(self, 'kdtree') and self.kdtree is not None:
            try:
                # Find nearest pollution point using KD-tree
                distance, idx = self.kdtree.query([longitude, latitude])
                pollution_value = self.kdtree_pollution_values[idx]
                result = (pollution_value, pollution_value, pollution_value) if return_low_high else pollution_value
                if cache_key:
                    self._lookup_cache[cache_key] = result
                return result
            except Exception as e:
                # Fall back to rasterio method if KD-tree fails
                pass
        
        # Fallback to original rasterio method
        try:
            row, col = rasterio.transform.rowcol(
                self.masked_transform, 
                [longitude], 
                [latitude]
            )
            row, col = int(row[0]), int(col[0])
            
            if (0 <= row < self.masked_pollution.shape[0] and 
                0 <= col < self.masked_pollution.shape[1]):
                value = self.masked_pollution[row, col]
                result = value if not np.isnan(value) and value > 0 else np.nan
                if cache_key and not np.isnan(result):
                    self._lookup_cache[cache_key] = (result, result, result) if return_low_high else result
                return (result, result, result) if return_low_high else result
            else:
                return (np.nan, np.nan, np.nan) if return_low_high else np.nan
        except Exception as e:
            return (np.nan, np.nan, np.nan) if return_low_high else np.nan

    def _get_dynamic_pollution(self, longitude, latitude, timestamp, return_low_high=False):
        if self.dynamic_time_index.size == 0:
            return (np.nan, np.nan, np.nan) if return_low_high else np.nan
        if timestamp is None or (isinstance(timestamp, float) and np.isnan(timestamp)):
            minutes = 0
        else:
            if not isinstance(timestamp, pd.Timestamp):
                try:
                    timestamp = pd.to_datetime(timestamp)
                except Exception:
                    timestamp = None
            if timestamp is None or pd.isna(timestamp):
                minutes = 0
            else:
                minutes = int(timestamp.hour * 60 + timestamp.minute)
        minutes %= (24 * 60)
        idx = np.abs(self.dynamic_time_index - minutes).argmin()
        slot = int(self.dynamic_time_index[idx])
        slice_info = self.dynamic_time_slices.get(slot)
        if not slice_info:
            return (np.nan, np.nan, np.nan) if return_low_high else np.nan
        try:
            distance, tree_idx = slice_info["tree"].query([longitude, latitude])
            values = slice_info["values"]
            if 0 <= tree_idx < len(values):
                base = float(values[tree_idx])
                if return_low_high:
                    low = float(slice_info.get("values_low", values)[tree_idx]) if "values_low" in slice_info else base
                    high = float(slice_info.get("values_high", values)[tree_idx]) if "values_high" in slice_info else base
                    return (base, low, high)
                return base
        except Exception:
            return (np.nan, np.nan, np.nan) if return_low_high else np.nan
        return (np.nan, np.nan, np.nan) if return_low_high else np.nan

    def _split_stay_points(self, df):
        if df.empty:
            return df
        records = []
        interval = max(1, int(self.dynamic_interval_minutes))
        for _, row in df.iterrows():
            start = pd.to_datetime(row['start_time'])
            end = pd.to_datetime(row['end_time'])
            if pd.isna(start) or pd.isna(end):
                continue
            total_minutes = (end - start).total_seconds() / 60.0
            if total_minutes <= 0:
                continue
            current = start
            remaining = total_minutes
            while remaining > 1e-6:
                segment_minutes = min(interval, remaining)
                seg = row.copy()
                seg['start_time'] = current
                seg['end_time'] = current + pd.Timedelta(minutes=segment_minutes)
                seg['duration_hours'] = segment_minutes / 60.0
                records.append(seg)
                current = current + pd.Timedelta(minutes=segment_minutes)
                remaining -= segment_minutes
        if not records:
            return pd.DataFrame(columns=df.columns)
        return pd.DataFrame(records)

    def _lookup_home_travel_exposure(self, person_id, timestamp):
        coords = self.home_locations.get(str(person_id))
        if coords:
            return self.get_pollution_at_location(coords[0], coords[1], timestamp)
        return self.travel_exposure
    
    def calculate_exposure_chunked(self, stay_points, chunk_size=10000,
                                   temp_root=None, pollution_fingerprint=None,
                                   stream_results=True):
        """
        Calculate pollution exposure from stay points in chunks.
        Uses DuckDB for final aggregation to avoid loading everything into memory.
        
        Parameters:
        -----------
        stay_points : pandas.DataFrame
            Stay points with actual locations and durations
        chunk_size : int
            Number of stay points to process in each chunk
            
        Returns:
        --------
        pandas.DataFrame
            Exposure calculations for each stay point
        """
        chunk_files = None
        if isinstance(stay_points, (list, tuple)) and stay_points and not isinstance(stay_points, pd.DataFrame):
            chunk_files = [Path(p) for p in stay_points]
        else:
            stay_points = stay_points.copy()
            stay_points['person_id'] = stay_points['person_id'].astype(str)
        self.home_locations = {}
        temp_root = Path(temp_root or os.environ.get('TEMP_PROCESSING_ROOT', 'temp_processing'))
        intermediate_dir = temp_root / 'duckdb_intermediates'
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = intermediate_dir / 'exposure_manifest.json'

        def _hash_identifier_sequence(values):
            digest = hashlib.md5()
            for value in values:
                digest.update(str(value).encode('utf-8'))
                digest.update(b'\0')
            return digest.hexdigest()

        def _hash_files(paths):
            digest = hashlib.md5()
            for path in paths:
                try:
                    stat = path.stat()
                    digest.update(str(path).encode('utf-8'))
                    digest.update(str(stat.st_mtime_ns).encode('utf-8'))
                    digest.update(str(stat.st_size).encode('utf-8'))
                    digest.update(b'\0')
                except OSError:
                    digest.update(str(path).encode('utf-8'))
                    digest.update(b'\0')
            return digest.hexdigest()

        def _load_manifest():
            if manifest_path.exists():
                try:
                    return json.loads(manifest_path.read_text())
                except json.JSONDecodeError:
                    return {}
            return {}

        def _write_manifest(data):
            manifest_path.write_text(json.dumps(data, indent=2))

        def _reset_manifest():
            for pattern in ('exposure_detailed_*.parquet', 'exposure_summary_*.parquet'):
                for file in intermediate_dir.glob(pattern):
                    try:
                        file.unlink()
                    except OSError:
                        pass
            if manifest_path.exists():
                manifest_path.unlink()

        if chunk_files:
            unique_persons = None
            person_hash = _hash_files(chunk_files)
        else:
            unique_persons = stay_points['person_id'].unique()
            person_hash = _hash_identifier_sequence(unique_persons)

        manifest = _load_manifest()
        expected = {
            'chunk_size': int(chunk_size),
            'person_hash': person_hash,
            'person_count': int(len(unique_persons)) if unique_persons is not None else -1,
            'pollution_fingerprint': pollution_fingerprint or 'na'
        }
        if not manifest or any(manifest.get(k) != v for k, v in expected.items()):
            _reset_manifest()
            manifest = {**expected, 'chunks': {}, 'complete': False}
            _write_manifest(manifest)
        else:
            manifest.setdefault('chunks', {})
            manifest['complete'] = False
            _write_manifest(manifest)

        print("Calculating exposure from stay points (chunked with DuckDB)...")
        print(f"Travel exposure type: {self.travel_exposure_type}")
        log_path = Path(temp_root or os.environ.get('TEMP_PROCESSING_ROOT', 'temp_processing')) / "exposure_log.txt"
        def _log(msg: str):
            print(msg)
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
        _log("=== Exposure run start ===")
        _log(f"Travel exposure type: {self.travel_exposure_type}")
        if self.travel_exposure_type == 'mid-point-hapem':
            _log(f"Travel HAPEM factors: pen={self.travel_pen}, prox={self.travel_prox}, factor={self.travel_pen * self.travel_prox}")
        elif self.travel_exposure_type == 'mid-point-teleportation':
            _log(f"Travel legacy factor: {self.travel_exposure}")

        # Precompute per-person PEN/PROX draws for work and school microenvironments
        if chunk_files:
            (work_outdoor_map, work_pen_map, work_prox_map,
             school_outdoor_map, school_pen_map, school_prox_map,
             home_pen_map, home_prox_map,
             penprox_path) = self._build_person_pen_prox_from_chunks(chunk_files, temp_root)
        else:
            (work_outdoor_map, work_pen_map, work_prox_map,
             school_outdoor_map, school_pen_map, school_prox_map,
             home_pen_map, home_prox_map,
             penprox_path) = self._build_person_pen_prox(stay_points, temp_root)
        self.home_pen_map = home_pen_map
        self.home_prox_map = home_prox_map
        self.home_factor_map = {pid: float(home_pen_map[pid] * home_prox_map[pid]) for pid in home_pen_map}
        n_persons_penprox = len(work_outdoor_map)
        n_outdoor_persons = sum(1 for v in work_outdoor_map.values() if v)
        n_school_outdoor = sum(1 for v in school_outdoor_map.values() if v)
        _log(f"PEN/PROX file: {penprox_path}, persons={n_persons_penprox}, work_outdoor_draws={n_outdoor_persons}, work_indoor_draws={n_persons_penprox - n_outdoor_persons}, school_outdoor_draws={n_school_outdoor}, sampling={self.pen_prox_sampling}, school_p_out={self.school_p_out}")
        
        # Pre-compute home pollution values for ALL people if using home_exposure type
        home_locations = {}
        if self.travel_exposure_type == 'home_exposure':
            print("Pre-computing home pollution values for all individuals...")
            if chunk_files:
                home_locations = self._build_home_locations_from_chunks(chunk_files)
                # Assign fallback for missing persons
                missing_persons = set(work_outdoor_map.keys()) - set(home_locations.keys())
                for person_id in missing_persons:
                    home_locations[str(person_id)] = None
                self.home_locations = home_locations
            else:
                unique_persons = stay_points['person_id'].astype(str).unique()
                origin_stays = stay_points[stay_points['location_type'] == 'origin'].copy()
                origin_stays.sort_values('start_time', inplace=True)
                home_candidates = origin_stays.drop_duplicates(subset='person_id', keep='first')
                print(f"Found {len(home_candidates)} individuals with identifiable home origin stays")
                if not home_candidates.empty:
                    coord_dict = (home_candidates[["person_id","longitude","latitude"]]
                                  .set_index("person_id").to_dict("index"))
                    for pid, vals in coord_dict.items():
                        if pd.isna(vals["longitude"]) or pd.isna(vals["latitude"]):
                            continue
                        home_locations[str(pid)] = (float(vals["longitude"]), float(vals["latitude"]))
                missing_persons = set(unique_persons) - set(home_locations.keys())
                for person_id in missing_persons:
                    home_locations[str(person_id)] = None
                self.home_locations = home_locations
                sample_home_values = []
                for _, row in home_candidates.head(5).iterrows():
                    pid = str(row['person_id'])
                    loc = home_locations.get(pid)
                    if loc:
                        sample_home_values.append(
                            (pid, self.get_pollution_at_location(loc[0], loc[1], row.get('start_time')))
                        )
                print(f"DEBUG: Sample home pollution values: {sample_home_values}")
        
        # Process in chunks by INDIVIDUALS (not stay points) to avoid fragmentation
        total_persons = len(work_outdoor_map)
        print(f"Processing {total_persons:,} unique individuals in chunks...")
        _log(f"Unique individuals: {total_persons:,}, chunk_size={chunk_size}")
        
        total_travel_rows_before_split = 0
        total_travel_rows_after_split = 0
        total_travel_splits = 0
        total_work_rows = 0
        total_work_outdoor_rows = 0
        total_school_rows = 0

        if chunk_files:
            chunk_iter = list(enumerate(chunk_files))
        else:
            chunk_iter = []
            for chunk_start in range(0, len(unique_persons), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(unique_persons))
                chunk_persons = unique_persons[chunk_start:chunk_end]
                chunk_idx = chunk_start // chunk_size
                chunk_iter.append((chunk_idx, chunk_persons))

        for chunk_idx, chunk_source in tqdm(chunk_iter, desc="Calculating exposure"):
            chunk_key = str(chunk_idx)
            detailed_filename = intermediate_dir / f"exposure_detailed_{chunk_idx:05d}.parquet"
            summary_filename = intermediate_dir / f"exposure_summary_{chunk_idx:05d}.parquet"
            chunk_meta = manifest['chunks'].get(chunk_key)
            if chunk_meta and chunk_meta.get('status') == 'complete':
                if Path(chunk_meta['detailed']).exists() and Path(chunk_meta['summary']).exists():
                    continue
            if chunk_files:
                chunk = pd.read_parquet(chunk_source)
                chunk['person_id'] = chunk['person_id'].astype(str)
            else:
                chunk_persons = chunk_source
                chunk = stay_points[stay_points['person_id'].isin(chunk_persons)].copy()
            if self.data_mode in ['dynamic', 'bayesian']:
                chunk = self._split_stay_points(chunk)
            # If using mid-point-teleportation/hapem, split each travel row into two half-duration legs
            if self.travel_exposure_type in ['mid-point-teleportation', 'mid-point-hapem']:
                new_rows = []
                travel_split_count = 0
                for _, r in chunk.iterrows():
                    if r.get('location_type') != 'travel':
                        new_rows.append(r)
                        continue
                    total_travel_rows_before_split += 1
                    # Guard against missing times or zero/negative duration
                    start = r.get('start_time')
                    end = r.get('end_time')
                    dur_h = r.get('duration_hours')
                    if pd.isna(start) or pd.isna(end) or pd.isna(dur_h) or dur_h <= 0:
                        new_rows.append(r)
                        continue
                    try:
                        mid = start + (end - start) / 2
                    except Exception:
                        new_rows.append(r)
                        continue
                    half_hours = dur_h / 2.0
                    # First half at origin
                    r1 = r.copy()
                    r1['end_time'] = mid
                    r1['duration_hours'] = half_hours
                    r1['longitude'] = r.get('origin_longitude', r.get('longitude'))
                    r1['latitude'] = r.get('origin_latitude', r.get('latitude'))
                    # Second half at destination
                    r2 = r.copy()
                    r2['start_time'] = mid
                    r2['duration_hours'] = half_hours
                    r2['longitude'] = r.get('destination_longitude', r.get('longitude'))
                    r2['latitude'] = r.get('destination_latitude', r.get('latitude'))
                    new_rows.extend([r1, r2])
                    travel_split_count += 1
                if travel_split_count:
                    chunk = pd.DataFrame(new_rows, columns=chunk.columns)
                    total_travel_rows_after_split += len(chunk[chunk['location_type'] == 'travel'])
                    total_travel_splits += travel_split_count
            if chunk.empty:
                continue
            
            # Get pollution at each actual location
            if self.data_mode == 'bayesian':
                # Get median and quantile pollution values
                vals = chunk.apply(
                    lambda row: self.get_pollution_at_location(
                        row['longitude'], row['latitude'], row.get('start_time'), return_low_high=True
                    ), axis=1
                )
                chunk[['pollution_concentration', 'pollution_low', 'pollution_high']] = pd.DataFrame(vals.tolist(), index=chunk.index)
            else:
                chunk['pollution_concentration'] = chunk.apply(
                    lambda row: self.get_pollution_at_location(
                        row['longitude'], row['latitude'], row.get('start_time')
                    ), axis=1
                )
            chunk.sort_values(['person_id', 'start_time'], inplace=True)

            # Initialize travel exposure columns to avoid KeyErrors
            if 'travel_exposure' not in chunk.columns:
                chunk['travel_exposure'] = np.nan
            if self.data_mode == 'bayesian':
                if 'travel_exposure_low' not in chunk.columns:
                    chunk['travel_exposure_low'] = np.nan
                if 'travel_exposure_high' not in chunk.columns:
                    chunk['travel_exposure_high'] = np.nan

            # Apply work microenvironment PEN/PROX per person
            work_rows = chunk['microenv'] == 'work_indoor'
            if work_rows.any():
                total_work_rows += int(work_rows.sum())

                def _work_penprox(row):
                    pid = row['person_id']
                    is_out = work_outdoor_map.get(pid, False)
                    pen = work_pen_map.get(pid, self.pen_prox_table['work_indoor']['pen']['gm'])
                    prox = work_prox_map.get(pid, self.pen_prox_table['work_indoor']['prox']['gm'])
                    factor = pen * prox
                    return factor, is_out

                factors, flags = zip(*chunk.loc[work_rows].apply(_work_penprox, axis=1))
                factors = np.array(factors, dtype=float)
                flags = np.array(flags, dtype=bool)
                chunk.loc[work_rows, 'pollution_concentration'] *= factors
                if self.data_mode == 'bayesian':
                    chunk.loc[work_rows, 'pollution_low'] *= factors
                    chunk.loc[work_rows, 'pollution_high'] *= factors
                total_work_outdoor_rows += int(flags.sum())
                _log(f"Work PEN/PROX applied: work_rows={len(factors)}, outdoor_rows={int(flags.sum())}, indoor_rows={len(factors)-int(flags.sum())}")
            # Apply school microenvironment PEN/PROX per person (Bernoulli p=0.2)
            school_rows = chunk['microenv'] == 'school_indoor'
            if school_rows.any():
                total_school_rows += int(school_rows.sum())

                def _school_penprox(row):
                    pid = row['person_id']
                    is_out = school_outdoor_map.get(pid, False)
                    pen = school_pen_map.get(pid, self.pen_prox_table['school_indoor']['pen']['gm'])
                    prox = school_prox_map.get(pid, self.pen_prox_table['school_indoor']['prox']['gm'])
                    factor = pen * prox
                    return factor, is_out

                factors, flags = zip(*chunk.loc[school_rows].apply(_school_penprox, axis=1))
                factors = np.array(factors, dtype=float)
                flags = np.array(flags, dtype=bool)
                chunk.loc[school_rows, 'pollution_concentration'] *= factors
                if self.data_mode == 'bayesian':
                    chunk.loc[school_rows, 'pollution_low'] *= factors
                    chunk.loc[school_rows, 'pollution_high'] *= factors
                _log(f"School PEN/PROX applied: school_rows={len(factors)}, outdoor_rows={int(flags.sum())}, indoor_rows={len(factors)-int(flags.sum())}, p_outdoor_fixed=0.2")

            # Apply home indoor PEN/PROX per person
            home_rows = chunk['microenv'] == 'home_indoor'
            if home_rows.any():
                def _home_penprox(row):
                    pid = row['person_id']
                    pen = home_pen_map.get(pid, self.pen_prox_table.get('work_indoor', {}).get('pen', {}).get('gm', 1.0))
                    prox = home_prox_map.get(pid, self.pen_prox_table.get('work_indoor', {}).get('prox', {}).get('gm', 1.0))
                    return pen * prox

                home_factors = np.array(chunk.loc[home_rows].apply(_home_penprox, axis=1), dtype=float)
                chunk.loc[home_rows, 'pollution_concentration'] *= home_factors
                if self.data_mode == 'bayesian':
                    chunk.loc[home_rows, 'pollution_low'] *= home_factors
                    chunk.loc[home_rows, 'pollution_high'] *= home_factors
                _log(f"Home PEN/PROX applied: home_rows={len(home_factors)}")

            elif self.travel_exposure_type == 'mid-point-teleportation':
                # After splitting travel into two legs, use ambient at each leg and scale by legacy factor.
                factor = self.travel_exposure
                chunk['travel_exposure'] = np.where(
                    chunk['location_type'] == 'travel',
                    chunk['pollution_concentration'] * factor,
                    np.nan
                )
                if self.data_mode == 'bayesian':
                    chunk['travel_exposure_low'] = np.where(
                        chunk['location_type'] == 'travel',
                        chunk.get('pollution_low', np.nan) * factor,
                        np.nan
                    )
                    chunk['travel_exposure_high'] = np.where(
                        chunk['location_type'] == 'travel',
                        chunk.get('pollution_high', np.nan) * factor,
                        np.nan
                    )
            elif self.travel_exposure_type == 'mid-point-hapem':
                # After splitting travel into two legs, apply HAPEM on-road PEN*PROX factor to ambient.
                factor = self.travel_pen * self.travel_prox
                chunk['travel_exposure'] = np.where(
                    chunk['location_type'] == 'travel',
                    chunk['pollution_concentration'] * factor,
                    np.nan
                )
                if self.data_mode == 'bayesian':
                    chunk['travel_exposure_low'] = np.where(
                        chunk['location_type'] == 'travel',
                        chunk.get('pollution_low', np.nan) * factor,
                        np.nan
                    )
                    chunk['travel_exposure_high'] = np.where(
                        chunk['location_type'] == 'travel',
                        chunk.get('pollution_high', np.nan) * factor,
                        np.nan
                    )
            elif self.travel_exposure_type == 'centroid_ambient':
                # Travel uses ambient concentration evaluated at the travel segment's representative
                # coordinate (currently set as the origin-destination centroid in staypoint generation).
                chunk['travel_exposure'] = np.where(
                    chunk['location_type'] == 'travel',
                    chunk['pollution_concentration'],
                    np.nan
                )
                if self.data_mode == 'bayesian':
                    chunk['travel_exposure_low'] = np.where(
                        chunk['location_type'] == 'travel',
                        chunk.get('pollution_low', np.nan),
                        np.nan
                    )
                    chunk['travel_exposure_high'] = np.where(
                        chunk['location_type'] == 'travel',
                        chunk.get('pollution_high', np.nan),
                        np.nan
                    )
            elif self.travel_exposure_type == 'origin_destination_mean':
                chunk['prev_poll'] = chunk.groupby('person_id')['pollution_concentration'].shift(1)
                chunk['next_poll'] = chunk.groupby('person_id')['pollution_concentration'].shift(-1)
                chunk['travel_exposure'] = np.where(
                    chunk['location_type'] == 'travel',
                    chunk[['prev_poll', 'next_poll']].mean(axis=1, skipna=True),
                    np.nan
                )
                chunk['travel_exposure'].fillna(self.travel_exposure, inplace=True)
                chunk.drop(columns=['prev_poll', 'next_poll'], inplace=True)
                if self.data_mode == 'bayesian':
                    chunk['travel_exposure_low'] = chunk['travel_exposure']
                    chunk['travel_exposure_high'] = chunk['travel_exposure']
            else:  # hardcoded
                chunk['travel_exposure'] = chunk.apply(
                    lambda row: self.travel_exposure if row['location_type'] == 'travel' else np.nan,
                    axis=1
                )
                if self.data_mode == 'bayesian':
                    chunk['travel_exposure_low'] = chunk['travel_exposure']
                    chunk['travel_exposure_high'] = chunk['travel_exposure']

            # Fallback: if any travel_exposure still NaN for travel rows, use pollution_concentration * factor
            nan_travel = (chunk['location_type'] == 'travel') & chunk['travel_exposure'].isna()
            if nan_travel.any():
                chunk.loc[nan_travel, 'travel_exposure'] = chunk.loc[nan_travel, 'pollution_concentration'] * self.travel_exposure
                if self.data_mode == 'bayesian':
                    chunk.loc[nan_travel, 'travel_exposure_low'] = chunk.loc[nan_travel, 'pollution_low'] * self.travel_exposure
                    chunk.loc[nan_travel, 'travel_exposure_high'] = chunk.loc[nan_travel, 'pollution_high'] * self.travel_exposure
            
            # DEBUG: Print sample travel exposure values to verify they're not 0
            travel_segments = chunk[chunk['location_type'] == 'travel']
            if len(travel_segments) > 0:
                sample_travel = travel_segments.head(3)
                print(f"DEBUG: Sample travel exposure values: {sample_travel['travel_exposure'].tolist()}")
            
            # Calculate exposure (concentration × duration)
            # For travel segments, use travel_exposure; for others use pollution_concentration
            chunk['exposure'] = chunk.apply(
                lambda row: (
                    row['travel_exposure'] * row['duration_hours'] 
                    if row['location_type'] == 'travel' 
                    else row['pollution_concentration'] * row['duration_hours']
                ), axis=1
            )
            if self.data_mode == 'bayesian':
                chunk['exposure_low'] = chunk.apply(
                    lambda row: (
                        row.get('travel_exposure_low', np.nan) * row['duration_hours']
                        if row['location_type'] == 'travel'
                        else row.get('pollution_low', np.nan) * row['duration_hours']
                    ), axis=1
                )
                chunk['exposure_high'] = chunk.apply(
                    lambda row: (
                        row.get('travel_exposure_high', np.nan) * row['duration_hours']
                        if row['location_type'] == 'travel'
                        else row.get('pollution_high', np.nan) * row['duration_hours']
                    ), axis=1
                )

            # Chunk-level logging for diagnostics
            n_persons_chunk = chunk['person_id'].nunique()
            travel_rows = chunk['location_type'] == 'travel'
            work_rows = chunk['microenv'] == 'work_indoor'
            school_rows = chunk['microenv'] == 'school_indoor'
            _log(f"Chunk stats: persons={n_persons_chunk}, rows={len(chunk)}, travel_rows={int(travel_rows.sum())}, "
                 f"travel_nan_exposure={int(chunk.loc[travel_rows, 'travel_exposure'].isna().sum())}, "
                 f"work_rows={int(work_rows.sum())}, school_rows={int(school_rows.sum())}")
            
            # Pre-aggregate person totals within this chunk (no cross-file aggregation needed)
            chunk_summary = chunk.groupby('person_id').agg({
                'exposure': 'sum',
                'duration_hours': 'sum',
                **({'exposure_low': 'sum', 'exposure_high': 'sum'} if 'exposure_low' in chunk.columns else {})
            }).reset_index()
            chunk_summary.rename(columns={'exposure': 'total_exposure'}, inplace=True)
            if 'exposure_low' in chunk_summary.columns:
                chunk_summary.rename(columns={
                    'exposure_low': 'total_exposure_low',
                    'exposure_high': 'total_exposure_high'
                }, inplace=True)
            
            chunk.to_parquet(detailed_filename, index=False)
            chunk_summary.to_parquet(summary_filename, index=False)
            if chunk_files:
                person_start = -1
                person_end = -1
            else:
                person_start = int(chunk_start)
                person_end = int(chunk_end)
            manifest['chunks'][chunk_key] = {
                'detailed': str(detailed_filename.resolve()),
                'summary': str(summary_filename.resolve()),
                'person_start': person_start,
                'person_end': person_end,
                'status': 'complete'
            }
            _write_manifest(manifest)

        manifest['complete'] = True
        _write_manifest(manifest)

        # Aggregate summary for audit trail
        _log(
            f"Travel split summary: travel_rows_before_split={total_travel_rows_before_split:,}, "
            f"travel_rows_after_split={total_travel_rows_after_split:,}, travel_segments_split={total_travel_splits:,}"
        )
        _log(
            f"HAPEM summary: work_rows={total_work_rows:,}, work_rows_outdoor={total_work_outdoor_rows:,}, "
            f"school_rows={total_school_rows:,}"
        )
        
        # Use DuckDB for fast concatenation (no complex aggregation needed)
        print("Concatenating results using DuckDB...")
        
        # Create a DuckDB connection for fast file reading
        conn = duckdb.connect(':memory:')
        if not intermediate_dir.exists():
            return pd.DataFrame(), pd.DataFrame()

        detailed_glob = str(intermediate_dir / 'exposure_detailed_*.parquet').replace("'", "''")
        summary_glob = str(intermediate_dir / 'exposure_summary_*.parquet').replace("'", "''")

        detailed_files = sorted(intermediate_dir.glob('exposure_detailed_*.parquet'))
        summary_files = sorted(intermediate_dir.glob('exposure_summary_*.parquet'))
        print(f"Concatenating {len(detailed_files)} detailed files and {len(summary_files)} summary files...")

        if not detailed_files or not summary_files:
            return pd.DataFrame(), pd.DataFrame()

        detailed_exposure = None
        if not stream_results:
            detailed_exposure = conn.execute(f"""
                SELECT * FROM read_parquet('{detailed_glob}', union_by_name=true)
                ORDER BY person_id, start_time
            """).fetchdf()
            if not detailed_exposure.empty:
                detailed_exposure['person_id'] = detailed_exposure['person_id'].astype(str)

        exposure_summary = conn.execute(f"""
            SELECT * FROM read_parquet('{summary_glob}', union_by_name=true)
            ORDER BY person_id
        """).fetchdf()
        if not exposure_summary.empty:
            exposure_summary['person_id'] = exposure_summary['person_id'].astype(str)
        
        conn.close()
        
        if stream_results:
            detailed_exposure = {
                'type': 'parquet_glob',
                'path': detailed_glob,
            }
            print(f"Aggregation complete: streamed detailed exposure via {detailed_glob}, "
                  f"{len(exposure_summary):,} summary rows")
        else:
            print(f"Aggregation complete: {len(detailed_exposure):,} detailed rows, {len(exposure_summary):,} summary rows")
        
        return exposure_summary, detailed_exposure
    
    def _load_intermediate_exposure(self):
        """
        Load intermediate exposure results from files.
        """
        exposure_files = []
        intermediate_dir = str(Path(os.environ.get('TEMP_PROCESSING_ROOT', 'temp_processing')) / 'duckdb_intermediates')
        if os.path.exists(intermediate_dir):
            exposure_files = [os.path.join(intermediate_dir, f) for f in os.listdir(intermediate_dir) 
                            if f.startswith('exposure_') and f.endswith('.parquet')]
        if not exposure_files:
            return pd.DataFrame()
        
        print(f"Loading {len(exposure_files)} intermediate exposure files...")
        chunks = []
        for file in exposure_files:
            chunks.append(pd.read_parquet(file))
        
        return pd.concat(chunks, ignore_index=True)

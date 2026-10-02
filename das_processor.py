#!/usr/bin/env python3
"""
DAS Processor Module for DuckDB-based exposure analysis pipeline.

Handles DAS data loading, cleaning, and conversion to stay points with proper time allocation.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import duckdb
import os
import csv
import warnings
from pathlib import Path
from typing import Dict
warnings.filterwarnings('ignore')


class DuckDBDASProcessor:
    """
    Module 1: Process DAS data using DuckDB for memory-efficient processing.
    """
    
    def __init__(self, duckdb_connection, config):
        self.conn = duckdb_connection
        self.config = config
        self.stay_points = None
        self.person_work_outdoor_prob = None
        self.debug_mode = bool(self.config.get('debug', {}).get('enabled', False))

    def set_person_work_outdoor_probs(self, mapping: Dict[str, float]) -> None:
        self.person_work_outdoor_prob = mapping or {}

    @staticmethod
    def _haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        # Compute great-circle distance in meters.
        r = 6371000.0
        phi1 = np.radians(lat1)
        phi2 = np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlmb = np.radians(lon2 - lon1)
        a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2.0) ** 2
        return float(2.0 * r * np.arctan2(np.sqrt(a), np.sqrt(max(0.0, 1.0 - a))))

    def _infer_microenv(self,
                        location_type: str,
                        longitude: float,
                        latitude: float,
                        home_longitude: float,
                        home_latitude: float,
                        primary_mode: str,
                        purpose_anchor: str,
                        home_threshold_m: float) -> tuple:
        """
        Return (microenv, microenv_rule, dist_to_home_m).
        Deterministic and debuggable: home match first; then travel mode; then purpose.
        """
        if location_type == "travel":
            # Travel segments are not "home" even if a centroid happens to be close.
            dist_m = np.nan
            mode = (str(primary_mode).strip().lower() if primary_mode is not None else "")
            if mode in {"private_auto", "auto_passenger", "on_demand_auto"}:
                return "in_vehicle_on_road_private", "mode_primary", dist_m
            if mode in {"public_transit"}:
                return "in_vehicle_on_road_transit", "mode_primary", dist_m
            if mode in {"walking", "biking"}:
                return "outdoor_near_road", "mode_primary", dist_m
            return "travel_unknown", "mode_unknown", dist_m

        dist_m = np.nan
        if not pd.isna(longitude) and not pd.isna(latitude) and not pd.isna(home_longitude) and not pd.isna(home_latitude):
            dist_m = self._haversine_m(float(longitude), float(latitude), float(home_longitude), float(home_latitude))
            if dist_m <= float(home_threshold_m):
                return "home_indoor", "home_distance", dist_m

        purpose = (str(purpose_anchor).strip().lower() if purpose_anchor is not None else "")
        if purpose == "work":
            return "work_indoor", "purpose_anchor", dist_m
        if purpose == "school":
            # For now, treat school as fully indoor (no outdoor-work probability applied).
            return "school_indoor", "purpose_anchor", dist_m
        return "other_indoor", "purpose_anchor", dist_m

    def create_duckdb_schema(self, activity_data_path):
        """
        Create DuckDB schema and load data for efficient querying.
        Uses pure Python line-by-line processing with tqdm progress tracking.
        """
        print("Setting up DuckDB schema for memory-efficient processing...")
        
        # Check if table already exists with data
        table_exists = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'activity_data'"
        ).fetchone()[0]
        
        if table_exists:
            print("Activity data table already exists, checking data freshness...")
            existing_records = self.conn.execute(
                "SELECT COUNT(*) FROM activity_data"
            ).fetchone()[0]
            
            # Get file modification times
            data_mtime = os.path.getmtime(activity_data_path)
            temp_root = Path(os.environ.get("TEMP_PROCESSING_ROOT", "temp_processing"))
            cleaned_csv = temp_root / "cleaned_activity_data.csv"
            cleaned_mtime = os.path.getmtime(cleaned_csv) if cleaned_csv.exists() else 0
                
            if existing_records > 0 and data_mtime <= cleaned_mtime:
                print(f"Using existing DuckDB table with {existing_records:,} records")
                return existing_records
        
        import tempfile
        print("Cleaning data with Python line-by-line processing...")
        
        # Create temporary files in configured temp root
        temp_dir = Path(os.environ.get("TEMP_PROCESSING_ROOT", "temp_processing"))
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        final_path = str(temp_dir / "cleaned_activity_data.csv")
        
        try:
            # Get total lines for progress tracking
            print("Counting total lines...")
            total_lines = 0
            with open(activity_data_path, 'r', encoding='utf-8') as f:
                for _ in f:
                    total_lines += 1
            
            print(f"Processing {total_lines:,} lines...")
            
            # Process file line by line with progress tracking
            bbox_cfg = self.config.get('activity_data', {}).get('bbox_filter', {})
            excluded_ids = set()
            if bbox_cfg.get('enabled'):
                print("Applying bounding box filter...")
                excluded_ids = self._collect_oob_persons(activity_data_path, bbox_cfg)
                print(f"Excluded {len(excluded_ids):,} individuals outside bbox")
            
            with open(activity_data_path, 'r', encoding='utf-8') as infile, \
                 open(final_path, 'w', encoding='utf-8', newline='') as outfile:
                
                reader = csv.reader(infile)
                writer = csv.writer(outfile)
                header = next(reader)
                writer.writerow(header)
                
                cfg_pid_col = self.config['activity_data']['person_id_column']
                if cfg_pid_col in header:
                    person_col_name = cfg_pid_col
                elif 'trip_taker_person_id' in header:
                    person_col_name = 'trip_taker_person_id'
                elif 'person_id' in header:
                    person_col_name = 'person_id'
                else:
                    raise ValueError("No person id column found (expected trip_taker_person_id or person_id)")
                self.person_id_column = person_col_name
                person_col = header.index(person_col_name)
                
                for row in tqdm(reader, total=total_lines-1, desc="Cleaning data"):
                    person_id = row[person_col]
                    if person_id in excluded_ids:
                        continue
                    cleaned_row = []
                    for cell in row:
                        cell = cell.replace('Out of Region', 'nan')
                        cell = cell.replace('Does not have work/school location', 'nan')
                        cell = cell.replace('Visitor (no home location)', 'nan')
                        cleaned_row.append(cell)
                    writer.writerow(cleaned_row)
            
            print("Data cleaning completed successfully")
            
            # Load the cleaned data into DuckDB (raw as VARCHARs)
            types_map = {self.person_id_column: 'VARCHAR'}
            self.conn.execute(f"""
                CREATE OR REPLACE TABLE activity_data_raw AS 
                SELECT * FROM read_csv_auto(
                    '{final_path}',
                    types={types_map},
                    all_varchar=true
                )
            """)
            
        except Exception as e:
            print(f"Warning: Python-based cleaning failed: {e}")
            print("Falling back to original file without cleaning...")
            # Fall back to original file
            final_path = activity_data_path
            
            # Load the original data into DuckDB (raw as VARCHARs)
            types_map = {self.person_id_column: 'VARCHAR'}
            self.conn.execute(f"""
                CREATE OR REPLACE TABLE activity_data_raw AS 
                SELECT * FROM read_csv_auto(
                    '{final_path}',
                    types={types_map},
                    all_varchar=true
                )
            """)

        # Build a typed activity_data table from raw strings using semantic casting
        try:
            cols = self.conn.execute("PRAGMA table_info('activity_data_raw')").fetchdf()['name'].tolist()
            # Columns that should be numeric (DOUBLE) based on meaning
            numeric_cols = {
                'trip_duration_minutes',
                'trip_distance_miles',
                'trip_taker_age',
                'trip_taker_individual_income',
                'trip_taker_household_income',
                'trip_taker_household_size',
                'origin_bgrp_lng_2020',
                'origin_bgrp_lat_2020',
                'destination_bgrp_lng_2020',
                'destination_bgrp_lat_2020',
                'trip_taker_home_bgrp_lng_2020',
                'trip_taker_home_bgrp_lat_2020',
                'trip_taker_work_bgrp_lng_2020',
                'trip_taker_work_bgrp_lat_2020',
            }

            # Tokens that should be treated as NULL for numeric fields
            null_tokens = {
                'nan',
                'NaN',
                'Out of Region',
                'Buffer Region',
                'Visitor (no home location)',
                'Visitor (no work/school location)',
                'Does not have work/school location',
                '',
            }
            null_list = ", ".join(["'" + t.replace("'", "''") + "'" for t in sorted(null_tokens)])

            def _clean_expr(col: str) -> str:
                col_quoted = f"\"{col}\""
                return (
                    f"CASE WHEN {col_quoted} IS NULL THEN NULL "
                    f"WHEN TRIM({col_quoted}) IN ({null_list}) THEN NULL "
                    f"ELSE {col_quoted} END"
                )

            select_exprs = []
            for c in cols:
                if c in numeric_cols:
                    select_exprs.append(f'TRY_CAST({_clean_expr(c)} AS DOUBLE) AS "{c}"')
                else:
                    select_exprs.append(f'"{c}"')

            select_sql = ",\n                ".join(select_exprs)
            self.conn.execute(f"""
                CREATE OR REPLACE TABLE activity_data AS
                SELECT
                    {select_sql}
                FROM activity_data_raw
            """)
        except Exception as e:
            print(f"Warning: failed to build typed activity_data table: {e}")
            print("Falling back to raw string table for activity_data...")
            self.conn.execute("CREATE OR REPLACE TABLE activity_data AS SELECT * FROM activity_data_raw")
        
        # Create indexes for efficient querying
        self.conn.execute(f"""
            CREATE INDEX IF NOT EXISTS idx_person_id ON activity_data({self.person_id_column})
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_start_time ON activity_data(trip_start_time)
        """)

        # Build a person-level attributes table for fast cohort filtering.
        try:
            cols = self.conn.execute("PRAGMA table_info('activity_data')").fetchdf()['name'].tolist()
            person_cols = [
                'trip_taker_race_ethnicity',
                'trip_taker_sex',
                'trip_taker_age',
                'trip_taker_individual_income',
                'trip_taker_education',
                'trip_taker_industry',
                'trip_taker_household_id',
                'trip_taker_home_bgrp_lng_2020',
                'trip_taker_home_bgrp_lat_2020',
            ]
            present_cols = [c for c in person_cols if c in cols]
            if present_cols:
                select_exprs = ",\n                ".join([f"MAX({c}) AS {c}" for c in present_cols])
                self.conn.execute(f"""
                    CREATE OR REPLACE TABLE person_attributes AS
                    SELECT
                        CAST({self.person_id_column} AS VARCHAR) AS person_id,
                        {select_exprs}
                    FROM activity_data
                    GROUP BY {self.person_id_column}
                """)
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_person_attr_id ON person_attributes(person_id)")
                for c in present_cols:
                    try:
                        self.conn.execute(f"CREATE INDEX IF NOT EXISTS idx_person_attr_{c} ON person_attributes({c})")
                    except Exception:
                        pass
        except Exception as e:
            print(f"Warning: could not build person_attributes table: {e}")

        # Build cohort membership table (one row per person per cohort group).
        try:
            self._build_person_cohorts()
        except Exception as e:
            print(f"Warning: could not build person_cohorts table: {e}")
        
        # Get total count for progress tracking
        total_count = self.conn.execute("SELECT COUNT(*) FROM activity_data").fetchone()[0]
        print(f"Loaded {total_count:,} activity records into DuckDB")
        
        return total_count
    
    def get_unique_persons(self):
        """
        Get unique person IDs from DuckDB.
        """
        col = getattr(self, "person_id_column", "trip_taker_person_id")
        result = self.conn.execute(f"""
            SELECT DISTINCT CAST({col} AS VARCHAR) AS trip_taker_person_id
            FROM activity_data 
            ORDER BY trip_taker_person_id
        """).fetchall()
        return [str(row[0]) for row in result]
    
    def get_person_activities(self, person_id):
        """
        Get activities for a specific person from DuckDB.
        """
        col = getattr(self, "person_id_column", "trip_taker_person_id")
        query = f"""
            SELECT 
                CAST({col} AS VARCHAR) AS trip_taker_person_id,
                trip_start_time,
                trip_end_time,
                trip_duration_minutes,
                trip_distance_miles,
                trip_purpose,
                primary_mode,
                TRY_CAST(origin_bgrp_lng_2020 AS DOUBLE) as origin_bgrp_lng_2020,
                TRY_CAST(origin_bgrp_lat_2020 AS DOUBLE) as origin_bgrp_lat_2020,
                TRY_CAST(destination_bgrp_lng_2020 AS DOUBLE) as destination_bgrp_lng_2020,
                TRY_CAST(destination_bgrp_lat_2020 AS DOUBLE) as destination_bgrp_lat_2020,
                TRY_CAST(trip_taker_home_bgrp_lng_2020 AS DOUBLE) as trip_taker_home_bgrp_lng_2020,
                TRY_CAST(trip_taker_home_bgrp_lat_2020 AS DOUBLE) as trip_taker_home_bgrp_lat_2020
            FROM activity_data 
            WHERE {col} = '{person_id}'
            ORDER BY trip_start_time
        """
        df = self.conn.execute(query).fetchdf()
        if not df.empty:
            df['trip_taker_person_id'] = df['trip_taker_person_id'].astype(str)
        return df

    def get_person_ids_by_filter(self, filter_expression):
        """
        Get person IDs that satisfy a configurable WHERE clause on activity_data.
        """
        if not filter_expression:
            return []
        col = getattr(self, "person_id_column", "trip_taker_person_id")
        try:
            person_table_exists = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'person_attributes'"
            ).fetchone()[0] > 0
            if person_table_exists:
                query = f"""
                    SELECT CAST(person_id AS VARCHAR) AS trip_taker_person_id
                    FROM person_attributes
                    WHERE {filter_expression}
                    ORDER BY trip_taker_person_id
                """
            else:
                query = f"""
                    SELECT DISTINCT CAST({col} AS VARCHAR) AS trip_taker_person_id
                    FROM activity_data
                    WHERE {filter_expression}
                    ORDER BY trip_taker_person_id
                """
            rows = self.conn.execute(query).fetchall()
            return [str(row[0]) for row in rows]
        except Exception as e:
            print(f"Error evaluating cohort filter '{filter_expression}': {e}")
            return []

    def get_person_ids_by_group(self, cohort: str, group: str, filter_expression: str | None = None):
        """
        Get person IDs for a cohort/group. Uses person_cohorts table if available; falls back to filter.
        """
        if not cohort or not group:
            return []
        try:
            person_table_exists = self.conn.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_name = 'person_cohorts'"
            ).fetchone()[0] > 0
            if person_table_exists:
                rows = self.conn.execute(
                    """
                    SELECT person_id
                    FROM person_cohorts
                    WHERE cohort = ? AND grp = ?
                    ORDER BY person_id
                    """,
                    [cohort, group],
                ).fetchall()
                return [str(row[0]) for row in rows]
        except Exception as e:
            print(f"Warning: cohort lookup failed for {cohort}:{group}: {e}")
        if filter_expression:
            return self.get_person_ids_by_filter(filter_expression)
        return []

    def _build_person_cohorts(self):
        analysis_cfg = self.config.get('analysis', {})
        cohorts = analysis_cfg.get('cohorts', [])
        if not cohorts:
            return
        # Ensure person_attributes exists
        person_table_exists = self.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'person_attributes'"
        ).fetchone()[0] > 0
        if not person_table_exists:
            return

        self.conn.execute("DROP TABLE IF EXISTS person_cohorts")
        self.conn.execute("""
            CREATE TABLE person_cohorts (
                person_id VARCHAR,
                cohort VARCHAR,
                grp VARCHAR
            )
        """)
        for cohort in cohorts:
            cohort_name = cohort.get('name', 'cohort')
            for group in cohort.get('groups', []):
                group_name = group.get('name')
                filter_expr = group.get('filter')
                if not group_name or not filter_expr:
                    continue
                try:
                    self.conn.execute(f"""
                        INSERT INTO person_cohorts
                        SELECT person_id, '{cohort_name}', '{group_name}'
                        FROM person_attributes
                        WHERE {filter_expr}
                    """)
                except Exception as e:
                    print(f"Warning: could not materialize cohort {cohort_name}:{group_name}: {e}")
                    continue
        try:
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_person_cohorts ON person_cohorts(person_id)")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_person_cohorts_group ON person_cohorts(cohort, grp)")
        except Exception:
            pass

    
    def process_das_to_stay_points(self, chunk_size=1000, stream_to_disk=True):
        """
        Convert DAS data to stay points with CORRECTED algorithm.
        Only destination stays, no origin stays. Travel time completely ignored.
        Processes individuals in chunks to avoid memory issues.
        
        Parameters:
        -----------
        chunk_size : int
            Number of individuals to process in each chunk
            
        Returns:
        --------
        pandas.DataFrame
            Stay points with actual locations and proper time durations
        """
        print("Processing DAS to stay points with CORRECTED algorithm (DuckDB)...")

        # Simple logger that also mirrors to exposure_log.txt for run auditing
        temp_root = Path(os.environ.get("TEMP_PROCESSING_ROOT", "temp_processing"))
        log_path = temp_root / "exposure_log.txt"
        def _log(msg: str):
            print(msg)
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(log_path, "a") as f:
                    f.write(msg + "\n")
            except Exception:
                pass
        
        # Get unique persons
        unique_persons = self.get_unique_persons()
        print(f"Processing {len(unique_persons):,} individuals in chunks of {chunk_size}...")
        _log(f"DAS -> staypoints: total individuals before filters = {len(unique_persons):,}")
        
        all_stay_points = [] if not stream_to_disk else None
        rejected_individuals = []  # time mismatch
        nan_rejected_individuals = []  # bad coords or non-numeric casts
        processed_count = 0
        
        intermediate_dir = temp_root / "duckdb_intermediates"
        intermediate_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing intermediate files and skip processing if they exist
        print("Checking for existing intermediate chunks...")
        existing_chunks = {}
        total_chunks = (len(unique_persons) + chunk_size - 1) // chunk_size
        
        # First, check which chunks already exist
        existing_chunk_nums = []
        for chunk_num in range(total_chunks):
            intermediate_path = intermediate_dir / f"stay_points_{chunk_num}.parquet"
            if intermediate_path.exists():
                existing_chunk_nums.append(chunk_num)
                if not stream_to_disk:
                    try:
                        existing_chunks[chunk_num] = pd.read_parquet(intermediate_path)
                        print(f"Found existing chunk {chunk_num}")
                    except Exception as e:
                        print(f"Error loading {intermediate_path}: {e}")
        
        print(f"Found {len(existing_chunk_nums)} existing chunks out of {total_chunks} total chunks")
        
        # Process in chunks
        label_cfg = self.config.get("microenv_labeling", {})
        home_threshold_m = float(label_cfg.get("home_distance_threshold_m", 100.0))

        # Counters for audit trail
        dropped_bbox_nan = 0
        dropped_time_mismatch = 0
        kept_individuals = 0
        for chunk_start in tqdm(range(0, len(unique_persons), chunk_size), desc="Processing chunks"):
            chunk_end = min(chunk_start + chunk_size, len(unique_persons))
            chunk_persons = unique_persons[chunk_start:chunk_end]
            
            chunk_num = chunk_start // chunk_size
            
            # Skip processing if chunk already exists
            if chunk_num in existing_chunks or chunk_num in existing_chunk_nums:
                # Use existing chunk data (skip recompute)
                if not stream_to_disk and chunk_num in existing_chunks:
                    all_stay_points.extend(existing_chunks[chunk_num].to_dict('records'))
                processed_count += len(chunk_persons)
                kept_individuals += len(chunk_persons)
                continue  # Skip processing this chunk
            
            # Process chunk normally if no existing file
            chunk_stay_points = []
            
            for person_id in chunk_persons:
                try:
                    # Get person's activities from DuckDB
                    person_activities = self.get_person_activities(person_id)
                    
                    if len(person_activities) == 0:
                        continue
                    
                    # Check for NaN values in coordinates - REJECT if any NaN found
                    has_nan = person_activities[['origin_bgrp_lng_2020','origin_bgrp_lat_2020',
                                                 'destination_bgrp_lng_2020','destination_bgrp_lat_2020',
                                                 'trip_taker_home_bgrp_lng_2020','trip_taker_home_bgrp_lat_2020']].isna().any(axis=None)
                    
                    if has_nan:
                        nan_rejected_individuals.append({'person_id': person_id, 'reason': 'nan_coords_or_non_numeric'})
                        dropped_bbox_nan += 1
                        continue  # Skip this person entirely
                    
                    # Convert time columns to datetime objects
                    person_activities = person_activities.copy()
                    # Fast, tolerant HH:MM:SS parsing; coerce bad rows to NaT to avoid slow regex fallback
                    person_activities['trip_start_datetime'] = pd.to_datetime(
                        person_activities['trip_start_time'].astype(str).str.slice(0,8),
                        format='%H:%M:%S', errors='coerce'
                    )
                    person_activities['trip_end_datetime'] = pd.to_datetime(
                        person_activities['trip_end_time'].astype(str).str.slice(0,8),
                        format='%H:%M:%S', errors='coerce'
                    )
                    # Drop rows with unparsable times (rare; avoids downstream issues)
                    person_activities = person_activities.dropna(subset=['trip_start_datetime','trip_end_datetime'])
                    if len(person_activities) == 0:
                        continue
                    
                    person_activities = person_activities.sort_values('trip_start_datetime')
                    
                    # Get person's home coordinates
                    home_longitude = person_activities.iloc[0]['trip_taker_home_bgrp_lng_2020']
                    home_latitude = person_activities.iloc[0]['trip_taker_home_bgrp_lat_2020']
                    p_outdoor_work = None
                    if self.person_work_outdoor_prob is not None:
                        p_outdoor_work = self.person_work_outdoor_prob.get(str(person_id))
                    
                    # Initialize day boundaries
                    day_start = pd.to_datetime('00:00:00', format='%H:%M:%S')
                    day_end = pd.to_datetime('23:59:59', format='%H:%M:%S')
                    
                    # Track all stay points for this person (NEW ALGORITHM)
                    person_stay_points = []
                    
                    # Process each trip according to the NEW rules
                    for i, activity in person_activities.iterrows():
                        # RULE 1: Before the first trip - from midnight until first trip starts
                        if i == 0:  # First trip of the day
                            duration_hours = (activity['trip_start_datetime'] - day_start).total_seconds() / 3600
                            if duration_hours > 0:
                                microenv, microenv_rule, dist_m = self._infer_microenv(
                                    "origin",
                                    activity['origin_bgrp_lng_2020'],
                                    activity['origin_bgrp_lat_2020'],
                                    home_longitude,
                                    home_latitude,
                                    activity.get("primary_mode"),
                                    activity.get("trip_purpose"),
                                    home_threshold_m,
                                )
                                person_stay_points.append({
                                    'person_id': person_id,
                                    'location_type': 'origin',
                                    'longitude': activity['origin_bgrp_lng_2020'],
                                    'latitude': activity['origin_bgrp_lat_2020'],
                                    'start_time': day_start,
                                    'end_time': activity['trip_start_datetime'],
                                    'duration_hours': duration_hours,
                                    'activity_type': f"pre_{activity['trip_purpose']}",
                                    'trip_purpose': activity.get("trip_purpose"),
                                    'primary_mode': activity.get("primary_mode"),
                                    'home_longitude': home_longitude,
                                    'home_latitude': home_latitude,
                                    'dist_to_home_m': dist_m,
                                    'microenv': microenv,
                                    'microenv_rule': microenv_rule,
                                    'p_outdoor_work': p_outdoor_work if microenv == "work_indoor" else np.nan,
                                })
                        
                        # RULE 2: During each trip - travel time (use travel exposure value)
                        # Create travel segments with a centroid coordinate (no route choice model).
                        # Exposure during travel is still controlled by exposure_calculator.travel_exposure_type.
                        travel_duration_hours = activity['trip_duration_minutes'] / 60
                        if travel_duration_hours > 0:
                            o_lon = float(activity['origin_bgrp_lng_2020'])
                            o_lat = float(activity['origin_bgrp_lat_2020'])
                            d_lon = float(activity['destination_bgrp_lng_2020'])
                            d_lat = float(activity['destination_bgrp_lat_2020'])
                            centroid_lon = 0.5 * (o_lon + d_lon)
                            centroid_lat = 0.5 * (o_lat + d_lat)
                            microenv, microenv_rule, dist_m = self._infer_microenv(
                                "travel",
                                centroid_lon,
                                centroid_lat,
                                home_longitude,
                                home_latitude,
                                activity.get("primary_mode"),
                                activity.get("trip_purpose"),
                                home_threshold_m,
                            )
                            person_stay_points.append({
                                'person_id': person_id,
                                'location_type': 'travel',
                                'longitude': centroid_lon,
                                'latitude': centroid_lat,
                                'start_time': activity['trip_start_datetime'],
                                'end_time': activity['trip_end_datetime'],
                                'duration_hours': travel_duration_hours,
                                'activity_type': f"travel_{activity['trip_purpose']}",
                                'trip_purpose': activity.get("trip_purpose"),
                                'primary_mode': activity.get("primary_mode"),
                                'origin_longitude': o_lon,
                                'origin_latitude': o_lat,
                                'destination_longitude': d_lon,
                                'destination_latitude': d_lat,
                                'home_longitude': home_longitude,
                                'home_latitude': home_latitude,
                                'dist_to_home_m': dist_m,
                                'microenv': microenv,
                                'microenv_rule': microenv_rule,
                                'p_outdoor_work': np.nan,
                            })
                        
                        # RULE 3: Between trips - after one trip ends and before next starts
                        if i > 0:  # Not the first trip
                            prev_activity = person_activities.iloc[i - 1]
                            duration_hours = (activity['trip_start_datetime'] - prev_activity['trip_end_datetime']).total_seconds() / 3600
                            if duration_hours > 0:
                                microenv, microenv_rule, dist_m = self._infer_microenv(
                                    "origin",
                                    prev_activity['destination_bgrp_lng_2020'],
                                    prev_activity['destination_bgrp_lat_2020'],
                                    home_longitude,
                                    home_latitude,
                                    prev_activity.get("primary_mode"),
                                    prev_activity.get("trip_purpose"),
                                    home_threshold_m,
                                )
                                person_stay_points.append({
                                    'person_id': person_id,
                                    'location_type': 'origin',
                                    'longitude': prev_activity['destination_bgrp_lng_2020'],
                                    'latitude': prev_activity['destination_bgrp_lat_2020'],
                                    'start_time': prev_activity['trip_end_datetime'],
                                    'end_time': activity['trip_start_datetime'],
                                    'duration_hours': duration_hours,
                                    'activity_type': f"stay_{prev_activity['trip_purpose']}",
                                    'trip_purpose': prev_activity.get("trip_purpose"),
                                    'primary_mode': prev_activity.get("primary_mode"),
                                    'home_longitude': home_longitude,
                                    'home_latitude': home_latitude,
                                    'dist_to_home_m': dist_m,
                                    'microenv': microenv,
                                    'microenv_rule': microenv_rule,
                                    'p_outdoor_work': p_outdoor_work if microenv == "work_indoor" else np.nan,
                                })
                        
                        # RULE 4: After the last trip - from end of last trip until midnight
                        if i == len(person_activities) - 1:  # Last trip of the day
                            duration_hours = (day_end - activity['trip_end_datetime']).total_seconds() / 3600
                            if duration_hours > 0:
                                microenv, microenv_rule, dist_m = self._infer_microenv(
                                    "destination",
                                    activity['destination_bgrp_lng_2020'],
                                    activity['destination_bgrp_lat_2020'],
                                    home_longitude,
                                    home_latitude,
                                    activity.get("primary_mode"),
                                    activity.get("trip_purpose"),
                                    home_threshold_m,
                                )
                                person_stay_points.append({
                                    'person_id': person_id,
                                    'location_type': 'destination',
                                    'longitude': activity['destination_bgrp_lng_2020'],
                                    'latitude': activity['destination_bgrp_lat_2020'],
                                    'start_time': activity['trip_end_datetime'],
                                    'end_time': day_end,
                                    'duration_hours': duration_hours,
                                    'activity_type': f"post_{activity['trip_purpose']}",
                                    'trip_purpose': activity.get("trip_purpose"),
                                    'primary_mode': activity.get("primary_mode"),
                                    'home_longitude': home_longitude,
                                    'home_latitude': home_latitude,
                                    'dist_to_home_m': dist_m,
                                    'microenv': microenv,
                                    'microenv_rule': microenv_rule,
                                    'p_outdoor_work': p_outdoor_work if microenv == "work_indoor" else np.nan,
                                })
                    
                    # Calculate total time allocated so far
                    total_allocated_time = sum([stay['duration_hours'] for stay in person_stay_points])
                    
                    # Note: Travel segments are now created in Rule 2, so no remaining time should be travel time
                    # The remaining time is just small rounding errors due to time calculations
                    remaining_time = 24 - total_allocated_time
                    
                    # Small rounding errors in time calculations are normal and expected
                    # No need to print warnings for these minor discrepancies
                    
                    # Final sanity check - should be approximately 24 hours (allow small rounding errors)
                    person_total_time = sum([stay['duration_hours'] for stay in person_stay_points])
                    if abs(person_total_time - 24) > 0.1:  # Increased tolerance to 0.1 hours (6 minutes)
                        rejected_individuals.append({
                            'person_id': person_id,
                            'reason': 'time_mismatch',
                            'total_time': person_total_time,
                            'num_activities': len(person_activities)
                        })
                        dropped_time_mismatch += 1
                        # Don't add this person's data if time doesn't match
                        continue
                    
                    # Add this person's stay points to the chunk
                    chunk_stay_points.extend(person_stay_points)
                    processed_count += 1
                    kept_individuals += 1
                    
                except Exception as e:
                    print(f"Error processing person {person_id}: {e}")
                    continue
            
            # Add chunk stay points to main list (only if keeping in memory)
            if not stream_to_disk:
                all_stay_points.extend(chunk_stay_points)
            
            # Save this chunk's results immediately with correct chunk number
            self._save_intermediate_results(
                chunk_stay_points,
                chunk_num,
                intermediate_dir,
                verbose=self.debug_mode,
            )
        
        if stream_to_disk:
            self.stay_points = pd.DataFrame()
            print("Stay points written to disk per chunk (streaming mode).")
        else:
            self.stay_points = pd.DataFrame(all_stay_points)
            if not self.stay_points.empty:
                self.stay_points['person_id'] = self.stay_points['person_id'].astype(str)
            print(f"Created {len(self.stay_points):,} stay points with proper durations")
        print(f"Processed {processed_count:,} individuals successfully (out of {len(unique_persons):,} after bbox)")
        _log(f"DAS -> staypoints complete: kept={kept_individuals:,}, dropped_bbox_nan={dropped_bbox_nan:,}, dropped_time_mismatch={dropped_time_mismatch:,}, existing_chunks_reused={len(existing_chunks):,}")
        # Persist rejected persons for audit
        results_root = Path(os.environ.get('RESULTS_ROOT', 'results'))
        results_root.mkdir(parents=True, exist_ok=True)
        if nan_rejected_individuals or rejected_individuals:
            rej_path = results_root / 'rejected_persons.csv'
            import pandas as _pd
            rows = []
            rows.extend(nan_rejected_individuals)
            rows.extend(rejected_individuals)
            _pd.DataFrame(rows).to_csv(rej_path, index=False)
            _log(f"Rejected persons written to {rej_path} (count={len(rows)})")
        if not stream_to_disk and not self.stay_points.empty:
            microenv_counts = self.stay_points['microenv'].value_counts().to_dict()
            print("Microenvironment row counts:", ", ".join([f"{k}:{v}" for k,v in microenv_counts.items()]))
            _log("Microenvironment row counts: " + ", ".join([f"{k}:{v}" for k,v in microenv_counts.items()]))
        
        # Print rejection summary
        if rejected_individuals:
            print(f"Rejected {len(rejected_individuals):,} individuals with invalid time allocations")
            print("Rejected individuals summary:")
            for rejected in rejected_individuals[:5]:  # Show first 5
                print(f"  Person {rejected['person_id']}: {rejected['total_time']:.2f} hours, {rejected['num_activities']} activities")
        
        if nan_rejected_individuals:
            print(f"Rejected {len(nan_rejected_individuals):,} individuals with NaN values in coordinates")
        
        if stream_to_disk:
            # Return ordered list of chunk files for downstream streaming.
            chunk_files = [
                str(intermediate_dir / f"stay_points_{chunk_num}.parquet")
                for chunk_num in range(total_chunks)
                if (intermediate_dir / f"stay_points_{chunk_num}.parquet").exists()
            ]
            return chunk_files
        return self.stay_points
    
    def _save_intermediate_results(self, stay_points, chunk_num, intermediate_dir: Path, verbose: bool = False):
        """
        Save intermediate results to avoid memory issues.
        Uses explicit chunk numbering for consistent file naming.
        """
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        filename = intermediate_dir / f"stay_points_{chunk_num}.parquet"
        
        # Only save if file doesn't exist
        if not filename.exists():
            pd.DataFrame(stay_points).to_parquet(str(filename), index=False)
            if verbose:
                print(f"Saved new intermediate results for chunk {chunk_num}")
        else:
            if verbose:
                print(f"Using existing intermediate file for chunk {chunk_num}")

    def _collect_oob_persons(self, csv_path, bbox_cfg):
        coordinate_columns = self.config['activity_data']['coordinate_columns']
        person_col_name = self.config['activity_data']['person_id_column']
        min_lon = bbox_cfg.get('min_longitude', -180)
        max_lon = bbox_cfg.get('max_longitude', 180)
        min_lat = bbox_cfg.get('min_latitude', -90)
        max_lat = bbox_cfg.get('max_latitude', 90)
        audit_records = {}

        def parse_coord(value):
            try:
                return float(value)
            except (ValueError, TypeError):
                return None

        def coord_in_bounds(lon, lat):
            if lon is None or lat is None:
                return False
            return min_lon <= lon <= max_lon and min_lat <= lat <= max_lat

        excluded = set()
        with open(csv_path, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            person_idx = header.index(person_col_name)
            try:
                cols = {key: header.index(val) for key, val in coordinate_columns.items()}
            except ValueError as exc:
                raise ValueError(f"Coordinate column missing from CSV header: {exc}")

            for row in reader:
                person_id = row[person_idx]
                origin_lon = parse_coord(row[cols['origin_longitude']])
                origin_lat = parse_coord(row[cols['origin_latitude']])
                dest_lon = parse_coord(row[cols['destination_longitude']])
                dest_lat = parse_coord(row[cols['destination_latitude']])
                home_lon = parse_coord(row[cols['home_longitude']])
                home_lat = parse_coord(row[cols['home_latitude']])

                origin_ok = coord_in_bounds(origin_lon, origin_lat)
                dest_ok = coord_in_bounds(dest_lon, dest_lat)
                home_ok = coord_in_bounds(home_lon, home_lat)
                removed = not (origin_ok and dest_ok and home_ok)

                entry = audit_records.setdefault(person_id, {
                    'origin_lon': origin_lon,
                    'origin_lat': origin_lat,
                    'destination_lon': dest_lon,
                    'destination_lat': dest_lat,
                    'home_lon': home_lon,
                    'home_lat': home_lat,
                    'origin_outside': False,
                    'destination_outside': False,
                    'home_outside': False,
                    'removed': False
                })
                entry['origin_outside'] = entry['origin_outside'] or not origin_ok
                entry['destination_outside'] = entry['destination_outside'] or not dest_ok
                entry['home_outside'] = entry['home_outside'] or not home_ok
                entry['removed'] = entry['removed'] or removed

                if removed:
                    excluded.add(person_id)

        temp_dir = Path(os.environ.get('TEMP_PROCESSING_ROOT', 'temp_processing'))
        temp_dir.mkdir(parents=True, exist_ok=True)
        audit_path = temp_dir / 'bbox_filter_audit.csv'
        with open(audit_path, 'w', encoding='utf-8', newline='') as audit_file:
            writer = csv.writer(audit_file)
            writer.writerow([
                'person_id',
                'origin_lon', 'origin_lat', 'origin_outside',
                'destination_lon', 'destination_lat', 'destination_outside',
                'home_lon', 'home_lat', 'home_outside',
                'removed'
            ])
            for person_id, data in audit_records.items():
                status = 'kicked_out' if data['removed'] else 'retained'
                writer.writerow([
                    person_id,
                    data['origin_lon'], data['origin_lat'], data['origin_outside'],
                    data['destination_lon'], data['destination_lat'], data['destination_outside'],
                    data['home_lon'], data['home_lat'], data['home_outside'],
                    status
                ])
        print(f"Saved bbox filter audit to {audit_path}")

        return excluded

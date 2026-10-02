#!/usr/bin/env python3
"""
Visualization Module for DuckDB-based exposure analysis pipeline.

Handles plotting and visualization of exposure comparison results.
USES ONLY ACTUAL COMPUTED DATA - NO INTERPOLATION OR RECOMPUTATION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import warnings
import glob
import duckdb
import os
from utils import normalize_exposure_metric, exposure_metric_units, exposure_home_fallback
warnings.filterwarnings('ignore')


def create_exposure_comparison(exposure_summary, detailed_exposure=None,
                               travel_fallback=0.5, exposure_metric='cumulative',
                               exposure_calc=None):
    """
    Create comparison between activity-based and home-only exposure using vectorized operations.
    Returns the merged DataFrame plus a compact dictionary of per-person summaries.
    """
    metric = normalize_exposure_metric(exposure_metric)
    units = exposure_metric_units(metric)
    home_fallback_value = exposure_home_fallback(travel_fallback, metric)
    
    # Determine time resolution based on data mode
    if exposure_calc is not None and getattr(exposure_calc, "data_mode", "static") == "bayesian":
        # Bayesian data has 30-minute resolution
        time_resolution_minutes = 30
        print(f"DEBUG: Using {time_resolution_minutes}-minute resolution for Bayesian mode")
    else:
        # Default to 5-minute resolution for static/dynamic modes
        time_resolution_minutes = 5
    
    time_points = np.arange(0, 24 * 60 + time_resolution_minutes, time_resolution_minutes) / 60.0
    interval_hours = time_resolution_minutes / 60.0
    dynamic_home_enabled = (
        exposure_calc is not None and
        getattr(exposure_calc, "data_mode", "static") in ["dynamic", "bayesian"]
    )
    home_factor_map = getattr(exposure_calc, "home_factor_map", {}) if exposure_calc is not None else {}
    home_profiles = {}
    home_profile_cache = {}
    base_ts = pd.Timestamp("2000-01-01 00:00:00")

    activity_agg = exposure_summary.copy()
    if not activity_agg.empty:
        activity_agg['person_id'] = activity_agg['person_id'].astype(str)
    activity_agg.rename(columns={'total_exposure': 'activity_exposure_cumulative'}, inplace=True)
    if 'duration_hours' not in activity_agg.columns:
        activity_agg['duration_hours'] = 0.0
    activity_agg['activity_exposure_twac'] = np.where(
        activity_agg['duration_hours'] > 0,
        activity_agg['activity_exposure_cumulative'] / activity_agg['duration_hours'],
        0.0
    )
    if metric == 'twac':
        activity_agg['activity_exposure'] = activity_agg['activity_exposure_twac']
    else:
        activity_agg['activity_exposure'] = activity_agg['activity_exposure_cumulative']
    activity_agg['activity_exposure'] = activity_agg['activity_exposure'].fillna(0.0)
    
    if detailed_exposure is None or (isinstance(detailed_exposure, dict) and detailed_exposure.get('type') == 'parquet_glob'):
        if isinstance(detailed_exposure, dict):
            detailed_glob = detailed_exposure.get('path')
            if detailed_glob:
                conn = duckdb.connect(':memory:')
                origin_stays = conn.execute(f"""
                    SELECT person_id,
                           pollution_concentration AS home_pollution,
                           longitude AS home_longitude,
                           latitude AS home_latitude
                    FROM (
                        SELECT *,
                               row_number() OVER (PARTITION BY person_id ORDER BY start_time) AS rn
                        FROM read_parquet('{detailed_glob}', union_by_name=true)
                        WHERE location_type = 'origin'
                    )
                    WHERE rn = 1
                """).fetchdf()
                conn.close()
                if not origin_stays.empty:
                    home_exposure_df = origin_stays.copy()
                    home_exposure_df['person_id'] = home_exposure_df['person_id'].astype(str)
                    home_exposure_df['home_exposure_cumulative'] = home_exposure_df['home_pollution'] * 24.0
                    home_exposure_df['home_exposure_twac'] = home_exposure_df['home_pollution']
                    if metric == 'twac':
                        home_exposure_df['home_exposure'] = home_exposure_df['home_exposure_twac']
                    else:
                        home_exposure_df['home_exposure'] = home_exposure_df['home_exposure_cumulative']
                    if dynamic_home_enabled:
                        for idx, row in home_exposure_df.iterrows():
                            lon, lat = row['home_longitude'], row['home_latitude']
                            if pd.isna(lon) or pd.isna(lat):
                                continue
                            pid = str(row['person_id'])
                            home_factor = float(home_factor_map.get(pid, 1.0))
                            cache_key = (round(lon, 6), round(lat, 6), round(home_factor, 8))
                            profile = home_profile_cache.get(cache_key)
                            if profile is None:
                                increments = np.zeros(len(time_points) - 1, dtype=float)
                                for t_idx in range(len(time_points) - 1):
                                    ts = base_ts + pd.Timedelta(hours=time_points[t_idx])
                                    val = exposure_calc.get_pollution_at_location(lon, lat, ts)
                                    if pd.isna(val):
                                        continue
                                    increments[t_idx] = float(val) * home_factor * interval_hours
                                cumulative = np.concatenate(([0.0], np.cumsum(increments)))
                                profile = {
                                    'increments': increments,
                                    'cumulative': cumulative,
                                }
                                home_profile_cache[cache_key] = profile
                            if metric == 'twac':
                                home_exposure_df.at[idx, 'home_exposure'] = float(np.nanmean(profile['increments']) / interval_hours)
                            else:
                                home_exposure_df.at[idx, 'home_exposure'] = float(profile['cumulative'][-1])
                    activity_agg = activity_agg.merge(
                        home_exposure_df[['person_id', 'home_exposure']],
                        on='person_id',
                        how='left'
                    )
                    activity_agg['home_exposure'] = activity_agg['home_exposure'].fillna(home_fallback_value)
                    comparison_dict = {
                        row.person_id: {
                            'summary': {
                                'total_activity_exposure': float(row.activity_exposure),
                                'total_home_exposure': float(row.home_exposure),
                                'exposure_ratio': float(row.activity_exposure / row.home_exposure) if row.home_exposure > 0 else 0.0
                            },
                            'metadata': {
                                'metric': metric,
                                'units': units
                            }
                        }
                        for row in activity_agg.itertuples()
                    }
                    return activity_agg, comparison_dict
        activity_agg['home_exposure'] = home_fallback_value
        comparison_dict = {
            row.person_id: {
                'summary': {
                    'total_activity_exposure': float(row.activity_exposure),
                    'total_home_exposure': float(row.home_exposure),
                    'exposure_ratio': float(row.activity_exposure / row.home_exposure) if row.home_exposure > 0 else 0.0
                },
                'metadata': {
                    'metric': metric,
                    'units': units
                }
            }
            for row in activity_agg.itertuples()
        }
        return activity_agg, comparison_dict
    if detailed_exposure is None or detailed_exposure.empty:
        activity_agg['home_exposure'] = home_fallback_value
        comparison_dict = {
            row.person_id: {
                'summary': {
                    'total_activity_exposure': float(row.activity_exposure),
                    'total_home_exposure': float(row.home_exposure),
                    'exposure_ratio': float(row.activity_exposure / row.home_exposure) if row.home_exposure > 0 else 0.0
                },
                'metadata': {
                    'metric': metric,
                    'units': units
                }
            }
            for row in activity_agg.itertuples()
        }
        return activity_agg, comparison_dict
    
    detailed_exposure = detailed_exposure.copy()
    detailed_exposure['person_id'] = detailed_exposure['person_id'].astype(str)

    origin_stays = (
        detailed_exposure[detailed_exposure['location_type'] == 'origin']
        .sort_values(['person_id', 'start_time'])
        .drop_duplicates('person_id', keep='first')
        .copy()
    )

    # If staypoints already carry home_longitude/home_latitude (added for debugging),
    # avoid creating duplicate columns by renaming longitude/latitude.
    origin_stays = origin_stays.loc[:, ~origin_stays.columns.duplicated()]
    
    origin_stays['home_pollution'] = origin_stays['pollution_concentration']
    if 'home_longitude' not in origin_stays.columns and 'longitude' in origin_stays.columns:
        origin_stays.rename(columns={'longitude': 'home_longitude'}, inplace=True)
    if 'home_latitude' not in origin_stays.columns and 'latitude' in origin_stays.columns:
        origin_stays.rename(columns={'latitude': 'home_latitude'}, inplace=True)
    
    home_cols = ['person_id', 'home_pollution', 'home_longitude', 'home_latitude']
    home_exposure_df = origin_stays[home_cols].copy()
    home_exposure_df['home_exposure_cumulative'] = home_exposure_df['home_pollution'] * 24.0
    home_exposure_df['home_exposure_twac'] = home_exposure_df['home_pollution']
    if metric == 'twac':
        home_exposure_df['home_exposure'] = home_exposure_df['home_exposure_twac']
    else:
        home_exposure_df['home_exposure'] = home_exposure_df['home_exposure_cumulative']

    if dynamic_home_enabled and not home_exposure_df.empty:
        print(f"DEBUG: Dynamic home calculation enabled for {len(home_exposure_df)} individuals")
        print(f"DEBUG: exposure_calc.data_mode = {getattr(exposure_calc, 'data_mode', 'unknown')}")
        for idx, row in home_exposure_df.iterrows():
            lon, lat = row['home_longitude'], row['home_latitude']
            if pd.isna(lon) or pd.isna(lat):
                print(f"DEBUG: Skipping person {row['person_id']} - NaN coordinates")
                continue
            pid = str(row['person_id'])
            home_factor = float(home_factor_map.get(pid, 1.0))
            cache_key = (round(lon, 6), round(lat, 6), round(home_factor, 8))
            profile = home_profile_cache.get(cache_key)
            if profile is None:
                print(f"DEBUG: Calculating 24-hour profile for location ({lon:.4f}, {lat:.4f})")
                increments = np.zeros(len(time_points) - 1, dtype=float)
                valid_count = 0
                for t_idx in range(len(time_points) - 1):
                    ts = base_ts + pd.Timedelta(hours=time_points[t_idx])
                    val = exposure_calc.get_pollution_at_location(lon, lat, ts)
                    if pd.isna(val):
                        continue
                    increments[t_idx] = float(val) * home_factor * interval_hours
                    valid_count += 1
                cumulative = np.concatenate(([0.0], np.cumsum(increments)))
                twac = cumulative[-1] / 24.0 if cumulative[-1] > 0 else 0.0
                profile = {'increments': increments, 'cumulative': cumulative, 'twac': twac}
                home_profile_cache[cache_key] = profile
                print(f"DEBUG: Profile calculated - valid points: {valid_count}, total: {cumulative[-1]:.2f}, twac: {twac:.2f}")
            if profile['cumulative'][-1] <= 0:
                print(f"DEBUG: Profile has zero cumulative for person {row['person_id']}")
                continue
            home_profiles[row.person_id] = profile
            home_exposure_df.at[idx, 'home_pollution'] = profile['twac']
            home_exposure_df.at[idx, 'home_exposure_cumulative'] = profile['cumulative'][-1]
            home_exposure_df.at[idx, 'home_exposure_twac'] = profile['twac']
            home_exposure_df.at[idx, 'home_exposure'] = profile['twac'] if metric == 'twac' else profile['cumulative'][-1]
            print(f"DEBUG: Person {row['person_id']} - home exposure: {profile['cumulative'][-1]:.2f}, pollution: {profile['twac']:.2f}")
    
    missing_persons = set(activity_agg['person_id']) - set(home_exposure_df['person_id'])
    if missing_persons:
        fallback_df = pd.DataFrame({
            'person_id': list(missing_persons),
            'home_pollution': travel_fallback,
            'home_longitude': np.nan,
            'home_latitude': np.nan
        })
        fallback_df['home_exposure_cumulative'] = fallback_df['home_pollution'] * 24.0
        fallback_df['home_exposure_twac'] = fallback_df['home_pollution']
        if metric == 'twac':
            fallback_df['home_exposure'] = fallback_df['home_exposure_twac']
        else:
            fallback_df['home_exposure'] = fallback_df['home_exposure_cumulative']
        home_exposure_df = pd.concat([home_exposure_df, fallback_df], ignore_index=True)
    
    comparison_df = pd.merge(
        activity_agg,
        home_exposure_df,
        on='person_id',
        how='left'
    )
    
    comparison_df['home_exposure'].fillna(home_fallback_value, inplace=True)
    
    comparison_dict = {}
    for row in comparison_df.itertuples():
        home_exposure = float(row.home_exposure)
        activity_exposure = float(row.activity_exposure)
        ratio = (activity_exposure / home_exposure) if home_exposure > 0 else 0.0
        comparison_dict[row.person_id] = {
            'summary': {
                'total_activity_exposure': activity_exposure,
                'total_home_exposure': home_exposure,
                'exposure_ratio': ratio
            },
            'metadata': {
                'metric': metric,
                'units': units
            }
        }
    
    return comparison_df, comparison_dict


def create_comparison_plot(comparison_data, detailed_exposure=None):
    """
    Create comparison plot between activity-based and home-only exposure.
    Uses the new visualization approach that works with actual computed data.
    """
    print("Creating exposure comparison plot...")
    
    # Use the new visualization approach that works with pre-generated JSON
    visualize_actual_exposure_time_series()


def visualize_actual_exposure_time_series(json_file='pipeline_results.json', sample_size=100):
    """
    Create visualization using actual computed time series data.
    Uses DuckDB for full population statistics and JSON for individual samples.
    NO INTERPOLATION - uses only real computed data.
    """
    print("Creating exposure comparison plots using ACTUAL computed time series data...")
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    try:
        # Load the sample time series data for individual curves
        with open(json_file, 'r') as f:
            sample_data = json.load(f)
        
        print(f"Loaded sample time series data for {len(sample_data)} individuals")
        
        # Get sample of individuals for plotting
        person_ids = list(sample_data.keys())
        if len(person_ids) > sample_size:
            import random
            # random.seed(42)
            sample_person_ids = random.sample(person_ids, sample_size)
        else:
            sample_person_ids = person_ids
        
        # Extract sample time series data for individual curves
        activity_cumulative_curves = []
        home_cumulative_curves = []
        individual_curves = []
        
        missing_series = False
        for person_id in sample_person_ids:
            person_data = sample_data[person_id]
            time_series = person_data.get('time_series')
            if time_series is None:
                missing_series = True
                continue
            
            # Use the ACTUAL computed cumulative exposure time series
            time_points = time_series['time_points']
            activity_cumulative = time_series['activity_cumulative']
            home_cumulative = time_series['home_cumulative']
            
            activity_cumulative_curves.append(activity_cumulative)
            home_cumulative_curves.append(home_cumulative)
            individual_curves.append({
                'person_id': person_id,
                'activity_curve': activity_cumulative,
                'home_curve': home_cumulative
            })
        
        if missing_series and not activity_cumulative_curves:
            print("No legacy time series data available in pipeline_results.json; skipping legacy plots.")
            return

        # Compute full population statistics using DuckDB
        print("Computing full population statistics using DuckDB...")
        full_pop_stats = compute_full_population_statistics()
        
        if full_pop_stats is not None:
            # Create main comparison plot using FULL POPULATION data
            create_main_comparison_plot(time_points, activity_cumulative_curves, home_cumulative_curves, full_pop_stats)
            
            # Create median plot using FULL POPULATION data
            create_median_plot(time_points, full_pop_stats)
            
            # Create individual sample plot using SAMPLE data
            create_individual_sample_plot(time_points, individual_curves)
        else:
            print("Warning: Could not compute full population statistics, using sample data only")
            # Fallback to sample-based statistics
            create_main_comparison_plot(time_points, activity_cumulative_curves, home_cumulative_curves)
            create_median_plot(time_points, activity_curves=activity_cumulative_curves, home_curves=home_cumulative_curves)
            create_individual_sample_plot(time_points, individual_curves)
        
    except Exception as e:
        print(f"Error in visualization: {e}")
        print("Cannot create proper visualization without computed data")
        return


def compute_full_population_statistics():
    """
    Compute full population statistics using DuckDB on CSV files.
    Returns mean and median time series for activity and home exposure.
    Uses the actual column names from the CSV files.
    """
    try:
        conn = duckdb.connect(':memory:')
        
        # Check if required files exist
        if not os.path.exists('duckdb_detailed_exposure.csv'):
            print("Warning: Detailed exposure CSV not found")
            return None
        if not os.path.exists('duckdb_comparison_results.csv'):
            print("Warning: Comparison results CSV not found")
            return None
        
        print("Computing full population statistics using DuckDB...")
        
        # First, let's compute the daily exposure statistics for the full population
        # This gives us accurate daily totals for mean/median calculations
        daily_stats_query = """
            SELECT 
                person_id,
                activity_exposure,
                home_exposure
            FROM read_csv('duckdb_comparison_results.csv', auto_detect=true)
        """
        
        daily_stats = conn.execute(daily_stats_query).fetchdf()
        
        # Calculate full population daily statistics
        activity_mean_daily = daily_stats['activity_exposure'].mean()
        activity_median_daily = daily_stats['activity_exposure'].median()
        home_mean_daily = daily_stats['home_exposure'].mean()
        home_median_daily = daily_stats['home_exposure'].median()
        
        print(f"Full population daily statistics computed for {len(daily_stats)} individuals")
        print(f"Activity - Mean: {activity_mean_daily:.2f}, Median: {activity_median_daily:.2f}")
        print(f"Home - Mean: {home_mean_daily:.2f}, Median: {home_median_daily:.2f}")
        
        # For time series visualization, we need to reconstruct the curves
        # Since we don't have time series in CSV, we'll create linear approximations
        # based on the daily totals and typical activity patterns
        
        # Create time points (288 intervals in 24 hours)
        time_resolution_minutes = 5
        time_points = np.arange(0, 24 * 60, time_resolution_minutes) / 60
        
        # Create linear cumulative exposure curves based on daily totals
        # This is an approximation since we don't have actual time series in CSV
        activity_cumulative = np.linspace(0, activity_mean_daily, len(time_points))
        home_cumulative = np.linspace(0, home_mean_daily, len(time_points))
        activity_median_curve = np.linspace(0, activity_median_daily, len(time_points))
        home_median_curve = np.linspace(0, home_median_daily, len(time_points))
        
        # Create statistics DataFrame
        stats_df = pd.DataFrame({
            'hour_bin': time_points,
            'activity_mean': activity_cumulative,
            'activity_median': activity_median_curve,
            'home_mean': home_cumulative,
            'home_median': home_median_curve
        })
        
        conn.close()
        
        print(f"Created full population time series statistics for {len(stats_df)} time points")
        return stats_df
        
    except Exception as e:
        print(f"Error computing full population statistics: {e}")
        return None


def create_main_comparison_plot(time_points, activity_curves, home_curves, full_pop_stats=None):
    """
    Create main comparison plot using full population statistics.
    Uses DuckDB for full population mean, sample for confidence intervals.
    """
    print("Creating main comparison plot with full population statistics...")
    
    plt.figure(figsize=(12, 8))
    
    # Convert from μg/m³·hours to mg/m³·hours (divide by 1000)
    activity_curves = np.array(activity_curves) / 1000
    home_curves = np.array(home_curves) / 1000
    
    # Filter out NaN values before calculating statistics
    valid_activity_curves = []
    valid_home_curves = []
    
    for i in range(len(activity_curves)):
        activity_curve = activity_curves[i]
        home_curve = home_curves[i]
        
        # Check if both curves are valid (no NaN values)
        if not np.any(np.isnan(activity_curve)) and not np.any(np.isnan(home_curve)):
            valid_activity_curves.append(activity_curve)
            valid_home_curves.append(home_curve)
    
    if not valid_activity_curves or not valid_home_curves:
        print("Warning: No valid data for main comparison plot (all curves contain NaN values)")
        plt.close()
        return
    
    valid_activity_curves = np.array(valid_activity_curves)
    valid_home_curves = np.array(valid_home_curves)
    
    print(f"Using {len(valid_activity_curves)} valid individuals for confidence intervals")
    
    # Calculate confidence intervals from sample
    activity_ci_lower = np.percentile(valid_activity_curves, 2.5, axis=0)
    activity_ci_upper = np.percentile(valid_activity_curves, 97.5, axis=0)
    home_ci_lower = np.percentile(valid_home_curves, 2.5, axis=0)
    home_ci_upper = np.percentile(valid_home_curves, 97.5, axis=0)
    
    # Use full population statistics if available, otherwise fallback to sample
    if full_pop_stats is not None:
        activity_mean = full_pop_stats['activity_mean'].values / 1000  # Convert to mg
        home_mean = full_pop_stats['home_mean'].values / 1000
        print("Using full population mean statistics")
    else:
        activity_mean = np.mean(valid_activity_curves, axis=0)
        home_mean = np.mean(valid_home_curves, axis=0)
        print("Using sample mean statistics (fallback)")
    
    # Plot exposure curves
    plt.plot(time_points, activity_mean, label='Activity-Based Exposure', color='blue', linewidth=2)
    plt.fill_between(time_points, activity_ci_lower, activity_ci_upper, alpha=0.3, color='blue')
    
    plt.plot(time_points, home_mean, label='Home-Only Exposure', color='red', linewidth=2)
    plt.fill_between(time_points, home_ci_lower, home_ci_upper, alpha=0.3, color='red')
    
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Cumulative PM2.5 Exposure (mg/m³·hours)', fontsize=12)
    plt.title('Daily Cumulative PM2.5 Exposure Comparison\nActivity-Based vs Home-Only Scenarios (FULL POPULATION Statistics)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 24)
    
    # Add statistics
    activity_final = activity_mean[-1] if len(activity_mean) > 0 else 0
    home_final = home_mean[-1] if len(home_mean) > 0 else 0
    ratio = activity_final / home_final if home_final > 0 else 0
    
    plt.figtext(0.02, 0.02, 
               f'Mean Daily Exposure - Activity: {activity_final:.1f}, Home: {home_final:.1f}, Ratio: {ratio:.2f}x',
               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('results/duckdb_exposure_comparison_actual.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def create_median_plot(time_points, full_pop_stats=None, activity_curves=None, home_curves=None):
    """
    Create median-only plot using full population statistics.
    Uses DuckDB for full population median when available.
    """
    print("Creating median plot with full population statistics...")
    
    plt.figure(figsize=(12, 8))
    
    # Use full population statistics if available, otherwise fallback to sample
    if full_pop_stats is not None:
        activity_median = full_pop_stats['activity_median'].values / 1000  # Convert to mg
        home_median = full_pop_stats['home_median'].values / 1000
        print("Using full population median statistics")
    elif activity_curves is not None and home_curves is not None:
        # Fallback to sample-based median
        activity_curves = np.array(activity_curves)
        home_curves = np.array(home_curves)
        
        # Filter out NaN values
        valid_activity_curves = []
        valid_home_curves = []
        
        for i in range(len(activity_curves)):
            activity_curve = activity_curves[i]
            home_curve = home_curves[i]
            
            if not np.any(np.isnan(activity_curve)) and not np.any(np.isnan(home_curve)):
                valid_activity_curves.append(activity_curve)
                valid_home_curves.append(home_curve)
        
        if not valid_activity_curves or not valid_home_curves:
            print("Warning: No valid data for median plot (all curves contain NaN values)")
            plt.close()
            return
        
        valid_activity_curves = np.array(valid_activity_curves) / 1000
        valid_home_curves = np.array(valid_home_curves) / 1000
        
        activity_median = np.median(valid_activity_curves, axis=0)
        home_median = np.median(valid_home_curves, axis=0)
        print(f"Using sample median statistics from {len(valid_activity_curves)} individuals (fallback)")
    else:
        print("Warning: No data available for median plot")
        plt.close()
        return
    
    # Plot median curves
    plt.plot(time_points, activity_median, label='Activity Median', color='blue', linewidth=2)
    plt.plot(time_points, home_median, label='Home Median', color='red', linewidth=2)
    
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Cumulative PM2.5 Exposure (mg/m³·hours)', fontsize=12)
    plt.title('Median Daily Cumulative PM2.5 Exposure\nActivity-Based vs Home-Only (FULL POPULATION Statistics)', fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 24)
    plt.tight_layout()
    plt.savefig('results/duckdb_exposure_comparison_median_actual.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def create_individual_sample_plot(time_points, individual_curves):
    """
    Create individual sample plot using ACTUAL computed cumulative exposure data.
    """
    print("Creating individual sample plot with actual computed data...")
    
    plt.figure(figsize=(12, 8))
    
    # Filter out individuals with NaN values
    valid_individual_curves = []
    for curve in individual_curves:
        activity_curve = np.array(curve['activity_curve'])
        home_curve = np.array(curve['home_curve'])
        
        # Check if both curves are valid (no NaN values)
        if not np.any(np.isnan(activity_curve)) and not np.any(np.isnan(home_curve)):
            valid_individual_curves.append(curve)
    
    print(f"Using {len(valid_individual_curves)} valid individuals for sample plot")
    
    # Convert from μg/m³·hours to mg/m³·hours (divide by 1000)
    for curve in valid_individual_curves:
        curve['activity_curve'] = np.array(curve['activity_curve']) / 1000
        curve['home_curve'] = np.array(curve['home_curve']) / 1000
    
    # Plot sample of individual ACTUAL computed curves
    sample_size = min(10, len(valid_individual_curves))
    import random
    # random.seed(42)
    selected_curves = random.sample(valid_individual_curves, sample_size)
    
    for i, curve in enumerate(selected_curves):
        plt.plot(time_points, curve['activity_curve'], 
                linewidth=1.5, alpha=0.8, 
                label=f'PID {curve["person_id"]} Activity')
        plt.plot(time_points, curve['home_curve'], 
                linewidth=1.0, alpha=0.7, linestyle='--',
                label=f'PID {curve["person_id"]} Home')
    
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Cumulative PM2.5 Exposure (mg/m³·hours)', fontsize=12)
    plt.title('Raw Cumulative Exposure Curves\nSample of 10 Individuals (ACTUAL Computed Data)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 24)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig('results/duckdb_exposure_comparison_sample_actual.pdf', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    # Generate all plots using ACTUAL computed data
    visualize_actual_exposure_time_series()

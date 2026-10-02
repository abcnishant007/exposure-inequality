#!/usr/bin/env python3
"""
Main Pipeline Module for DuckDB-based exposure analysis pipeline.

Orchestrates the complete pipeline by coordinating all modules.
"""

import time
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
import duckdb
import warnings
import os
from pathlib import Path
import gc
warnings.filterwarnings('ignore')

# Import the refactored modules
from das_processor import DuckDBDASProcessor
from exposure_calculator import DuckDBExposureCalculator
from visualization import create_exposure_comparison
from visualization_summary import generate_summary_plots
from cohort_statistics import generate_cohort_statistics
from cohort_comparator import generate_cohort_comparisons
from utils import (
    load_config,
    save_results,
    export_exposure_timeseries,
    print_detailed_summary,
    print_sample_dictionary,
    save_dictionary_as_json,
    cleanup_intermediate_files,
    save_visualization_data,
    normalize_exposure_metric,
    compute_pollution_fingerprint,
    ensure_detailed_exposure_db,
    compute_config_md5,
)
from work_outdoor_model import WorkOutdoorConfig, compute_work_outdoor_probs


class DuckDBExposurePipeline:
    """
    Main pipeline orchestrating DuckDB-based memory-efficient processing.
    """
    
    def __init__(self, config_path='config.yaml'):
        self.config = load_config(config_path)
        self.config_path = config_path
        outputs_cfg = self.config.get('outputs', {})
        self.results_root = Path(outputs_cfg.get('results_root', 'results')).resolve()
        self.temp_root = Path(outputs_cfg.get('temp_root', 'temp_processing')).resolve()
        os.environ.setdefault('RESULTS_ROOT', str(self.results_root))
        os.environ.setdefault('TEMP_PROCESSING_ROOT', str(self.temp_root))
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.temp_root.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(':memory:')
        self.das_processor = DuckDBDASProcessor(self.conn, self.config)
        output_cfg = self.config.get('analysis', {}).get('output', {})
        analysis_cfg = self.config.get('analysis', {})
        self.debug_mode = self.config.get('debug', {}).get('enabled', False)
        self.sample_size = output_cfg.get('visualization_sample_size', 10)
        self.random_seed = output_cfg.get('visualization_random_seed', 42)
        try:
            self.person_plot_prob = float(output_cfg.get('person_plot_prob', output_cfg.get('plot_person_prob', 1.0)))
        except (TypeError, ValueError):
            self.person_plot_prob = 1.0
        self.person_plot_prob = max(0.0, min(1.0, self.person_plot_prob))
        self.exposure_metric = normalize_exposure_metric(analysis_cfg.get('exposure_metric', 'cumulative'))
        pollution_cfg = self.config.get('pollution', {})
        self.pollution_fingerprint = compute_pollution_fingerprint(pollution_cfg)
        self.exposure_calculator = DuckDBExposureCalculator(
            pollution_cfg.get('raster_path'),
            self.config['study_area']['geojson_path'],
            pollution_cfg.get('grid_resolution', 'raw'),
            pollution_cfg.get('travel_exposure', 0.5),
            pollution_cfg.get('travel_exposure_type', 'mid-point-teleportation'),
            data_mode=pollution_cfg.get('mode', 'static'),
            dynamic_csv_path=pollution_cfg.get('dynamic_csv_path'),
            bayesian_csv_path=pollution_cfg.get('bayesian_csv_path'),
            bayesian_column=pollution_cfg.get('bayesian_column', 'pm25_p50'),
            bayesian_low_column=pollution_cfg.get('bayesian_low_column', None),
            bayesian_high_column=pollution_cfg.get('bayesian_high_column', None),
            hapem_cfg=pollution_cfg.get('hapem', {}),
            travel_hapem=pollution_cfg.get('travel_hapem', {}),
            pen_prox_cfg=pollution_cfg.get('pen_prox', {})
        )
        # Pass debug mode to exposure calculator
        self.exposure_calculator.debug_mode = self.debug_mode
    
    def run_pipeline(self, activity_data_path=None):
        """
        Run the complete DuckDB-based exposure analysis pipeline.
        
        Parameters:
        -----------
        activity_data_path : str, optional
            Path to the activity data CSV file (overrides config)
            
        Returns:
        --------
        dict
            Results from all pipeline stages
        """
        print("Starting DUCKDB exposure analysis pipeline (memory-efficient)...")
        
        # Use provided path or config path
        if activity_data_path is None:
            activity_data_path = self.config['activity_data']['path']
        
        # Set up DuckDB schema
        total_records = self.das_processor.create_duckdb_schema(activity_data_path)

        # Optional: compute per-person probability of outdoor work during the workday.
        work_cfg = self.config.get("work_outdoor_model", {})
        person_naics_map = {}
        if work_cfg.get("enabled"):
            pid_col = self.config.get("activity_data", {}).get("person_id_column", "trip_taker_person_id")
            try:
                oews_path = work_cfg.get("oews_nat4d_path")
                onet_path = work_cfg.get("onet_work_context_path")
                if not oews_path or not Path(oews_path).exists():
                    raise FileNotFoundError(f"Missing OEWS file: {oews_path}")
                if not onet_path or not Path(onet_path).exists():
                    raise FileNotFoundError(f"Missing O*NET Work Context file: {onet_path}")
                person_industry = self.conn.execute(
                    f"SELECT DISTINCT CAST({pid_col} AS VARCHAR) AS person_id, trip_taker_industry "
                    "FROM activity_data"
                ).fetchdf()
                model_cfg = WorkOutdoorConfig(
                    oews_nat4d_path=Path(oews_path),
                    onet_work_context_path=Path(onet_path),
                    outdoor_elements=list(work_cfg.get("outdoor_elements", [
                        "Outdoors, Exposed to All Weather Conditions",
                        "Outdoors, Under Cover",
                    ])),
                    outdoor_category_threshold=int(work_cfg.get("outdoor_category_threshold", 4)),
                    combine_elements=str(work_cfg.get("combine_elements", "union")),
                    oews_occ_group=str(work_cfg.get("oews_occ_group", "detailed")),
                )
                person_probs, sector_probs, missingness = compute_work_outdoor_probs(person_industry, model_cfg)

                # Persist outputs (written under results_root, which is user-configured).
                out_person = Path(work_cfg.get("output_person_params_csv", self.results_root / "person_work_outdoor_probs.csv"))
                out_sector = Path(work_cfg.get("output_sector_probs_csv", self.results_root / "sector_outdoor_probs.csv"))
                out_miss = Path(work_cfg.get("output_missingness_csv", self.results_root / "work_outdoor_missingness.csv"))
                out_person.parent.mkdir(parents=True, exist_ok=True)
                person_probs.to_csv(out_person, index=False)
                sector_probs.to_csv(out_sector, index=False)
                missingness.to_csv(out_miss, index=False)

                # Attach to staypoint generation for debugging / downstream microenv logic.
                mapping = dict(zip(person_probs["person_id"].astype(str), person_probs["p_outdoor_work"]))
                self.das_processor.set_person_work_outdoor_probs(mapping)
                # Store NAICS labels for plots (prefer group, fall back to raw).
                naics_group = person_probs["naics_group"].astype(str)
                naics_raw = person_probs["naics_raw"].astype(str)
                person_naics_map = dict(
                    zip(
                        person_probs["person_id"].astype(str),
                        [g if g and g != "nan" else r for g, r in zip(naics_group, naics_raw)],
                    )
                )
                print(f"Wrote work-outdoor mapping: {out_person}")
            except Exception as exc:
                raise RuntimeError(f"work_outdoor_model enabled but failed: {exc}") from exc
        
        # Get chunk sizes from config
        chunk_size = self.config.get('performance', {}).get('chunk_size', 1000)
        exposure_chunk_size = self.config.get('performance', {}).get('exposure_chunk_size', 10000)
        
        # Module 1: Process DAS to stay points with proper durations using DuckDB
        stream_stays = self.config.get('performance', {}).get('stream_stay_points', True)
        stay_points = self.das_processor.process_das_to_stay_points(
            chunk_size=chunk_size,
            stream_to_disk=stream_stays,
        )
        if stream_stays:
            print("Stay points will be streamed from disk during exposure calculation.")
        
        # Module 2: Calculate exposure with actual coordinates in chunks
        stream_results = self.config.get('performance', {}).get('stream_exposure_outputs', True)
        exposure_summary, detailed_exposure = self.exposure_calculator.calculate_exposure_chunked(
            stay_points, chunk_size=exposure_chunk_size,
            temp_root=self.temp_root,
            pollution_fingerprint=self.pollution_fingerprint,
            stream_results=stream_results
        )
        detailed_db_path = None
        if isinstance(detailed_exposure, dict) and detailed_exposure.get('type') == 'parquet_glob':
            detailed_db_path = ensure_detailed_exposure_db(detailed_exposure.get('path'), self.temp_root)
        
        debug_cfg = self.config.get('debug', {})
        debug_persons = []
        if debug_cfg.get('enabled'):
            debug_persons = [str(pid) for pid in debug_cfg.get('person_ids', [])]
        
        cohort_entries = [{
            'label': 'all_individuals',
            'cohort': 'overall',
            'group': 'all',
            'output_root': str(self.results_root),
            'person_ids': None,
            'debug_persons': debug_persons
        }]
        
        # Create comparison between activity-based and home-only exposure
        comparison_results, comparison_dict = create_exposure_comparison(
            exposure_summary,
            detailed_exposure,
            travel_fallback=self.config['pollution'].get('travel_exposure', 0.5),
            exposure_metric=self.exposure_metric,
            exposure_calc=self.exposure_calculator
        )

        # Attach NAICS info (if computed) and exposure deltas
        if person_naics_map:
            comparison_results['naics_code'] = comparison_results['person_id'].map(person_naics_map).fillna('')
        comparison_results['exposure_delta'] = comparison_results['activity_exposure'] - comparison_results['home_exposure']
        comparison_results['exposure_ratio'] = np.where(
            comparison_results['home_exposure'] > 0,
            comparison_results['activity_exposure'] / comparison_results['home_exposure'],
            np.nan,
        )
        # Old visualization used create_comparison_plot; keep reference for possible rollback
        # create_comparison_plot(comparison_results, detailed_exposure)
        
        # Prepare lightweight visualization payloads (mean/median/sample curves)
        viz_summary_path = save_visualization_data(
            detailed_exposure,
            comparison_results,
            label="all_individuals",
            sample_size=self.sample_size,
            random_seed=self.random_seed,
            debug_person_ids=debug_persons,
            exposure_metric=self.exposure_metric,
            exposure_calc=self.exposure_calculator,
            pollution_fingerprint=self.pollution_fingerprint,
            person_naics_map=person_naics_map,
            duckdb_path=str(detailed_db_path) if detailed_db_path else None,
        )
        
        # Generate visualization from summarized payload (no recomputation)
        if viz_summary_path:
            generate_summary_plots(
                viz_summary_path,
                output_dir=str(self.results_root),
                person_plot_prob=self.person_plot_prob,
                person_plot_seed=self.random_seed,
            )

            # Export full-resolution exposure time series (activity vs home) and deltas
            export_exposure_timeseries(
                comparison_results,
                exposure_calc=self.exposure_calculator,
                exposure_metric=self.exposure_metric,
                output_dir=self.results_root,
                pollution_fingerprint=self.pollution_fingerprint,
                person_naics_map=person_naics_map,
                duckdb_path=str(detailed_db_path) if detailed_db_path else None,
            )
        
        # Generate cohort-specific visualizations if configured
        analysis_cfg = self.config.get('analysis', {})
        cohort_sets = analysis_cfg.get('cohorts')
        if cohort_sets:
            for cohort in tqdm(cohort_sets, desc="Cohorts"):
                cohort_label = cohort.get('name', 'cohort')
                configured_root = cohort.get('output_root')
                if configured_root is None:
                    cohort_root = self.results_root / cohort_label
                else:
                    cohort_root = Path(configured_root)
                    if not cohort_root.is_absolute():
                        cohort_root = self.results_root / cohort_root
                os.makedirs(cohort_root, exist_ok=True)
                for group in cohort.get('groups', []):
                    group_name = group.get('name')
                    group_filter = group.get('filter')
                    if not group_name or not group_filter:
                        continue
                    suffix = f"_{cohort_label}_{group_name}"
                    print(f"Preparing cohort visualization for '{cohort_label}:{group_name}'...")
                    cohort_person_ids = self.das_processor.get_person_ids_by_group(
                        cohort_label, group_name, group_filter
                    )
                    if not cohort_person_ids:
                        print(f"No individuals matched cohort '{cohort_label}:{group_name}'")
                        continue
                    cohort_output = os.path.join(cohort_root, f"visualization_data_{group_name}.json")
                    cohort_viz = save_visualization_data(
                        detailed_exposure,
                        comparison_results,
                        output_path=cohort_output,
                        sample_size=group.get('sample_size', self.sample_size),
                        random_seed=group.get('random_seed', self.random_seed),
                        person_ids=cohort_person_ids,
                        label=f"{cohort_label}:{group_name}",
                        debug_person_ids=[pid for pid in debug_persons if pid in cohort_person_ids],
                        debug_output_dir=os.path.join(cohort_root, "debug"),
                        exposure_metric=self.exposure_metric,
                        exposure_calc=self.exposure_calculator,
                        pollution_fingerprint=self.pollution_fingerprint,
                        duckdb_path=str(detailed_db_path) if detailed_db_path else None,
                    )
                    if cohort_viz:
                        generate_summary_plots(
                            cohort_viz,
                            output_dir=cohort_root,
                            suffix=suffix,
                            person_plot_prob=self.person_plot_prob,
                            person_plot_seed=self.random_seed,
                        )
                        cohort_entries.append({
                            'label': suffix.strip('_'),
                            'cohort': cohort_label,
                            'group': group_name,
                            'output_root': cohort_root,
                            'person_ids': cohort_person_ids,
                            'debug_persons': [pid for pid in debug_persons if pid in cohort_person_ids]
                        })
                    # Explicitly release cohort-scoped objects between groups.
                    del cohort_person_ids, cohort_viz
                    gc.collect()
        else:
            group_config_root = analysis_cfg.get('group_output_root')
            group_configs = analysis_cfg.get('groups', [])
            if group_configs:
                root = Path(group_config_root) if group_config_root else Path("custom_groups")
                if not root.is_absolute():
                    root = self.results_root / root
                os.makedirs(root, exist_ok=True)
                for group in group_configs:
                    group_name = group.get('name')
                    group_filter = group.get('filter')
                    if not group_name or not group_filter:
                        continue
                    print(f"Preparing cohort visualization for '{group_name}'...")
                    cohort_person_ids = self.das_processor.get_person_ids_by_group(
                        'custom_groups', group_name, group_filter
                    )
                    if not cohort_person_ids:
                        print(f"No individuals matched cohort '{group_name}'")
                        continue
                    cohort_output = os.path.join(root, f"visualization_data_{group_name}.json")
                    cohort_viz = save_visualization_data(
                        detailed_exposure,
                        comparison_results,
                        output_path=cohort_output,
                        sample_size=group.get('sample_size', self.sample_size),
                        random_seed=group.get('random_seed', self.random_seed),
                        person_ids=cohort_person_ids,
                        label=group_name,
                        debug_person_ids=[pid for pid in debug_persons if pid in cohort_person_ids],
                        debug_output_dir=os.path.join(root, "debug"),
                        exposure_metric=self.exposure_metric,
                        exposure_calc=self.exposure_calculator,
                        pollution_fingerprint=self.pollution_fingerprint,
                        duckdb_path=str(detailed_db_path) if detailed_db_path else None,
                    )
                    if cohort_viz:
                        generate_summary_plots(
                            cohort_viz,
                            output_dir=root,
                            suffix=f"_{group_name}",
                            person_plot_prob=self.person_plot_prob,
                            person_plot_seed=self.random_seed,
                        )
                        cohort_entries.append({
                            'label': group_name,
                            'cohort': 'custom_groups',
                            'group': group_name,
                            'output_root': root,
                            'person_ids': cohort_person_ids,
                            'debug_persons': [pid for pid in debug_persons if pid in cohort_person_ids]
                        })
                    # Explicitly release cohort-scoped objects between groups.
                    del cohort_person_ids, cohort_viz
                    gc.collect()
        
        config_hash = compute_config_md5(self.config_path)
        generate_cohort_statistics(
            detailed_exposure,
            comparison_results,
            cohort_entries,
            base_output_dir=str(self.results_root),
            duckdb_path=str(detailed_db_path) if detailed_db_path else None,
            config_hash=config_hash,
        )
        generate_cohort_comparisons(
            cohort_entries,
            output_dir=str(self.results_root / "cohort_comparisons"),
        )
        
        # Save results
        save_results(comparison_results, detailed_exposure)
        
        # Print detailed summary
        print_detailed_summary(comparison_results, detailed_exposure, self.config)
        
        # Print sample of dictionary format for debugging
        print_sample_dictionary(comparison_dict)
        
        # Save dictionary as proper JSON file for easy viewing
        save_dictionary_as_json(comparison_dict)
        
        # Clean up intermediate files
        cleanup_intermediate_files()
        
        # Close DuckDB connection
        self.conn.close()
        
        stay_points_result = stay_points if isinstance(stay_points, pd.DataFrame) else None
        return {
            'stay_points': stay_points_result,
            'exposure_summary': exposure_summary,
            'comparison': comparison_results,
            'config': self.config
        }


def main():
    """Run the complete DuckDB-based exposure analysis pipeline."""

    parser = argparse.ArgumentParser(description="Run the DuckDB exposure analysis pipeline.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    # Initialize and run pipeline with config
    pipeline = DuckDBExposurePipeline(args.config)
    results = pipeline.run_pipeline()
    
    print("\n=== DUCKDB PIPELINE COMPLETED SUCCESSFULLY ===")
    print("Features included:")
    print("- DuckDB-based memory-efficient processing")
    print("- Chunked processing for 4M+ individuals")
    print("- No loading entire dataset into RAM")
    print("- Intermediate file management")
    print("- Configurable chunk sizes")
    print("- Same accuracy as best pipeline")


if __name__ == "__main__":
    starttime = time.time()

    main()

    endtime = time.time()
    print('Time taken in seconds:', round(endtime - starttime,2))

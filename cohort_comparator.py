#!/usr/bin/env python3
"""
Cohort comparison utilities.

Reads the per-group cohort statistics JSON files and produces
side-by-side summaries for each cohort (e.g., sex, income).
"""

import json
import os
from collections import defaultdict

import pandas as pd
import matplotlib.pyplot as plt


def _flatten_stats(stats_dict):
    flat = {}
    for key, value in stats_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, dict):
                    for metric, metric_val in sub_val.items():
                        flat[f"{key}.{sub_key}.{metric}"] = metric_val
                else:
                    flat[f"{key}.{sub_key}"] = sub_val
        else:
            flat[key] = value
    return flat


def generate_cohort_comparisons(cohort_entries, output_dir="results/cohort_comparisons"):
    """
    Build CSV comparisons for each cohort using the computed stats JSON files
    and generate quick bar charts for key metrics.
    """
    os.makedirs(output_dir, exist_ok=True)
    cohorts = defaultdict(list)
    
    for entry in cohort_entries:
        cohort_name = entry.get('cohort', 'overall')
        group_name = entry.get('group', entry.get('label', 'group'))
        label = entry.get('label', group_name)
        root = entry.get('output_root', 'results')
        stats_path = os.path.join(root, f"cohort_stats_{label}.json")
        if os.path.exists(stats_path):
            cohorts[cohort_name].append((group_name, stats_path))
    
    key_metrics = [
        ("daily_activity_exposure.mean", "daily_activity_exposure.std", "Daily Activity Exposure (µg/m³·h)"),
        ("daily_home_exposure.mean", "daily_home_exposure.std", "Daily Home Exposure (µg/m³·h)"),
        ("daily_exposure_ratio.mean", "daily_exposure_ratio.std", "Activity/Home Ratio"),
    ]
    
    for cohort_name, items in cohorts.items():
        if len(items) < 2:
            continue  # nothing to compare
        
        rows = []
        for group_name, stats_path in items:
            try:
                with open(stats_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                flat = _flatten_stats(stats)
                flat['group'] = group_name
                rows.append(flat)
            except Exception as exc:
                print(f"Warning: Could not load {stats_path}: {exc}")
        
        if not rows:
            continue
        
        df = pd.DataFrame(rows).set_index('group')
        comparison_path = os.path.join(output_dir, f"{cohort_name}_comparison.csv")
        df.to_csv(comparison_path)
        print(f"Saved cohort comparison for '{cohort_name}' to {comparison_path}")
        
        for mean_col, std_col, title in key_metrics:
            if mean_col not in df.columns:
                continue
            plt.figure(figsize=(8, 4))
            means = df[mean_col]
            errors = None
            if std_col in df.columns:
                n = df.get('population_count')
                if n is not None:
                    errors = 1.96 * (df[std_col] / n**0.5)
            means.plot(kind='bar', color='steelblue', alpha=0.8, yerr=errors, capsize=4)
            plt.ylabel(title)
            plt.title(f"{title} – {cohort_name} cohort")
            plt.grid(axis='y', alpha=0.2)
            plt.tight_layout()
            plot_path = os.path.join(output_dir, f"{cohort_name}_{mean_col.replace('.', '_')}.pdf")
            plt.savefig(plot_path, dpi=300)
            plt.close()

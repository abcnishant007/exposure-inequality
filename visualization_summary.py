#!/usr/bin/env python3
"""
Lightweight visualization module that consumes precomputed exposure summaries.

Generates:
    1. Mean cumulative exposure comparison (activity vs. home).
    2. Median cumulative exposure comparison.
    3. Sample of randomly selected individual curves.

All heavy aggregation is completed during the main pipeline; this module only reads
the compact summary JSON and renders plots.
"""

import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def _ensure_results_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _downsample_curves(sample_curves: List[Dict], prob: float, seed: int = 42) -> List[Dict]:
    if not sample_curves:
        return []
    if prob >= 1.0:
        return sample_curves
    if prob <= 0.0:
        return []
    rng = np.random.default_rng(seed)
    keep = rng.random(len(sample_curves)) < prob
    out = [c for c, k in zip(sample_curves, keep, strict=False) if k]
    if not out:
        out = [sample_curves[int(rng.integers(0, len(sample_curves)))]]
    return out


def generate_summary_plots(summary_path: str = 'results/visualization_data.json',
                           output_dir: str = 'results',
                           suffix: str = '',
                           person_plot_prob: float = 1.0,
                           person_plot_seed: int = 42) -> None:
    """Load the precomputed visualization summary and render all plots."""
    if not os.path.exists(summary_path):
        print(f"Visualization summary not found at {summary_path}")
        return

    with open(summary_path, 'r', encoding='utf-8') as f:
        summary_data = json.load(f)

    time_points = np.array(summary_data['time_points'], dtype=float)
    activity_mean = np.array(summary_data['activity_mean'], dtype=float)
    home_mean = np.array(summary_data['home_mean'], dtype=float)
    activity_median = np.array(summary_data['activity_median'], dtype=float)
    home_median = np.array(summary_data['home_median'], dtype=float)
    sample_curves = summary_data.get('sample_curves', [])
    metadata = summary_data.get('metadata', {})
    exposure_metric = metadata.get('exposure_metric', 'cumulative')
    units = metadata.get('units', 'μg/m³·hours')

    _ensure_results_dir(output_dir)

    plotted_curves = _downsample_curves(sample_curves, person_plot_prob, seed=person_plot_seed)

    _plot_mean_curves(time_points, activity_mean, home_mean, output_dir, suffix, exposure_metric, units)
    _plot_median_curves(time_points, activity_median, home_median, output_dir, suffix, exposure_metric, units)
    _plot_quantile_band_curves(time_points, plotted_curves, output_dir, suffix, exposure_metric, units)
    _plot_sample_curves(time_points, plotted_curves, output_dir, suffix, exposure_metric, units)
    _plot_individual_cumulative(time_points, plotted_curves, output_dir, suffix, exposure_metric, units)
    _plot_individual_incremental(time_points, plotted_curves, output_dir, suffix, exposure_metric, units)


def _plot_mean_curves(time_points: np.ndarray, activity_mean: np.ndarray, home_mean: np.ndarray,
                      output_dir: str, suffix: str, metric: str, units: str) -> None:
    metric_key = metric if metric in {'cumulative', 'twac'} else 'cumulative'
    ylabel = "Cumulative PM2.5 Exposure" if metric_key == 'cumulative' else "Time-Weighted Avg PM2.5"
    title_metric = "Cumulative Exposure" if metric_key == 'cumulative' else "Time-Weighted Average (TWAC)"
    plt.figure(figsize=(12, 8))
    plt.plot(time_points, activity_mean, label='Activity Mean', color='steelblue', linewidth=2)
    plt.plot(time_points, home_mean, label='Home Mean', color='tomato', linewidth=2)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel(f'{ylabel} ({units})', fontsize=12)
    plt.title(f'Daily Mean {title_metric}\nActivity-Based vs Home-Only', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(0, 24)
    ymax = max(float(np.nanmax(activity_mean)), float(np.nanmax(home_mean)), 0.1)
    plt.ylim(0, ymax * 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'exposure_mean_comparison{suffix}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_median_curves(time_points: np.ndarray, activity_median: np.ndarray, home_median: np.ndarray,
                        output_dir: str, suffix: str, metric: str, units: str) -> None:
    metric_key = metric if metric in {'cumulative', 'twac'} else 'cumulative'
    ylabel = "Cumulative PM2.5 Exposure" if metric_key == 'cumulative' else "Time-Weighted Avg PM2.5"
    title_metric = "Cumulative Exposure" if metric_key == 'cumulative' else "Time-Weighted Average (TWAC)"
    plt.figure(figsize=(12, 8))
    plt.plot(time_points, activity_median, label='Activity Median', color='steelblue', linewidth=2)
    plt.plot(time_points, home_median, label='Home Median', color='tomato', linewidth=2)
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel(f'{ylabel} ({units})', fontsize=12)
    plt.title(f'Daily Median {title_metric}\nActivity-Based vs Home-Only', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(0, 24)
    ymax = max(float(np.nanmax(activity_median)), float(np.nanmax(home_median)), 0.1)
    plt.ylim(0, ymax * 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'exposure_median_comparison{suffix}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_quantile_band_curves(time_points: np.ndarray, sample_curves: List[Dict],
                               output_dir: str, suffix: str, metric: str, units: str) -> None:
    """
    Plot median curves with 5–95% and 25–75% bands computed from sampled individuals.
    This shows both the full spread and an interquartile-style core band.
    """
    if not sample_curves:
        return
    metric_key = metric if metric in {'cumulative', 'twac'} else 'cumulative'
    ylabel = "Cumulative PM2.5 Exposure" if metric_key == 'cumulative' else "Time-Weighted Avg PM2.5"
    title_metric = "Cumulative Exposure" if metric_key == 'cumulative' else "Time-Weighted Average (TWAC)"

    def _stack(key: str):
        arrs = []
        for curve in sample_curves:
            vals = curve.get(key)
            if vals is None:
                continue
            arr = np.array(vals, dtype=float)
            if len(arr) == len(time_points):
                arrs.append(arr)
        return np.array(arrs) if arrs else None

    activity_arr = _stack('activity_curve')
    home_arr = _stack('home_curve')
    if activity_arr is None or home_arr is None or activity_arr.size == 0 or home_arr.size == 0:
        return

    act_p50 = np.nanmedian(activity_arr, axis=0)
    act_p05 = np.nanpercentile(activity_arr, 5, axis=0)
    act_p25 = np.nanpercentile(activity_arr, 25, axis=0)
    act_p75 = np.nanpercentile(activity_arr, 75, axis=0)
    act_p95 = np.nanpercentile(activity_arr, 95, axis=0)

    home_p50 = np.nanmedian(home_arr, axis=0)
    home_p05 = np.nanpercentile(home_arr, 5, axis=0)
    home_p25 = np.nanpercentile(home_arr, 25, axis=0)
    home_p75 = np.nanpercentile(home_arr, 75, axis=0)
    home_p95 = np.nanpercentile(home_arr, 95, axis=0)

    plt.figure(figsize=(12, 8))
    plt.plot(time_points, act_p50, label='Activity Median', color='steelblue', linewidth=2)
    # Core band (25–75%) slightly darker than the outer 5–95% band.
    plt.fill_between(time_points, act_p25, act_p75, color='steelblue', alpha=0.25, label='Activity 25–75%')
    plt.fill_between(time_points, act_p05, act_p95, color='steelblue', alpha=0.12, label='Activity 5–95%')

    plt.plot(time_points, home_p50, label='Home Median', color='tomato', linewidth=2)
    plt.fill_between(time_points, home_p25, home_p75, color='tomato', alpha=0.25, label='Home 25–75%')
    plt.fill_between(time_points, home_p05, home_p95, color='tomato', alpha=0.12, label='Home 5–95%')

    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel(f'{ylabel} ({units})', fontsize=12)
    plt.title(f'Median with 5–95% Bands\nActivity-Based vs Home-Only {title_metric}', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.xlim(0, 24)
    ymax = max(
        float(np.nanmax(act_p95)),
        float(np.nanmax(home_p95)),
        0.1
    )
    plt.ylim(0, ymax * 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'exposure_median_band_comparison{suffix}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


def _plot_individual_incremental(time_points: np.ndarray, sample_curves: List[Dict],
                                 output_dir: str, suffix: str,
                                 metric: str, units: str) -> None:
    """
    Create non-cumulative plots for each sampled individual showing 5-minute exposure increments.

    Align increments to bin centres so the curve starts at zero and the first jump
    occurs at the end of the first 5-minute window, matching intuition.
    """
    if not sample_curves:
        return

    # Bin geometry
    if len(time_points) < 2:
        return
    bin_width = time_points[1] - time_points[0]
    bin_centres = time_points[1:] - bin_width / 2.0

    for curve in sample_curves:
        if 'activity_increment' in curve and 'home_increment' in curve:
            activity_incremental = np.array(curve['activity_increment'], dtype=float)
            home_incremental = np.array(curve['home_increment'], dtype=float)
        else:
            # Fallback for older JSON files
            activity_curve = np.array(curve['activity_curve'], dtype=float)
            home_curve = np.array(curve['home_curve'], dtype=float)
            activity_incremental = np.diff(np.concatenate(([0.0], activity_curve)))
            home_incremental = np.diff(np.concatenate(([0.0], home_curve)))

        # Drop the leading 0 so lengths match bin centres (one value per 5-min bin)
        if activity_incremental.size == time_points.size:
            activity_incremental = activity_incremental[1:]
        if home_incremental.size == time_points.size:
            home_incremental = home_incremental[1:]

        act_low_inc = act_high_inc = home_low_inc = home_high_inc = None
        if 'activity_curve_low' in curve and 'activity_curve_high' in curve:
            act_low = np.array(curve['activity_curve_low'], dtype=float)
            act_high = np.array(curve['activity_curve_high'], dtype=float)
            if act_low.size == time_points.size and act_high.size == time_points.size:
                act_low_inc = np.diff(act_low)
                act_high_inc = np.diff(act_high)
        if 'home_curve_low' in curve and 'home_curve_high' in curve:
            h_low = np.array(curve['home_curve_low'], dtype=float)
            h_high = np.array(curve['home_curve_high'], dtype=float)
            if h_low.size == time_points.size and h_high.size == time_points.size:
                home_low_inc = np.diff(h_low)
                home_high_inc = np.diff(h_high)

        plt.figure(figsize=(12, 6))
        plt.step(bin_centres, activity_incremental, where='mid', label='Activity (5-min)', color='steelblue', linewidth=1.5)
        plt.step(bin_centres, home_incremental, where='mid', label='Home (5-min)', color='tomato', linewidth=1.5, linestyle='--')

        # Activity - Home incremental difference (per bin)
        delta_incremental = activity_incremental - home_incremental
        plt.step(bin_centres, delta_incremental, where='mid', label='Activity − Home (5-min)', color='seagreen', linewidth=1.6, linestyle=':')

        if act_low_inc is not None and act_high_inc is not None:
            plt.fill_between(bin_centres, act_low_inc, act_high_inc, color='steelblue', alpha=0.2, step='mid', label='Activity band')
        if home_low_inc is not None and home_high_inc is not None:
            plt.fill_between(bin_centres, home_low_inc, home_high_inc, color='tomato', alpha=0.2, step='mid', label='Home band')

        # Difference uncertainty band: simple bound-wise subtraction (high-high, low-low)
        if act_low_inc is not None and act_high_inc is not None and home_low_inc is not None and home_high_inc is not None:
            delta_low_inc = act_low_inc - home_low_inc
            delta_high_inc = act_high_inc - home_high_inc
            plt.fill_between(bin_centres, delta_low_inc, delta_high_inc, color='seagreen', alpha=0.18, step='mid', label='(A−H) band')
        plt.xlabel('Hour of Day', fontsize=12)
        ylabel = "Per-bin PM2.5 Exposure (5-min)" if metric == 'cumulative' else "5-Minute Avg PM2.5"
        plt.ylabel(f'{ylabel} ({units})', fontsize=12)
        naics_label = curve.get("naics_group")
        naics_text = f" (NAICS {naics_label})" if naics_label else ""
        plt.title(f'5-Minute Exposure Metrics – Person {curve["person_id"]}{naics_text}', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 24)
        ymax = max(float(np.nanmax(activity_incremental)), float(np.nanmax(home_incremental)), float(np.nanmax(delta_incremental)), 0.01)
        ymin = min(float(np.nanmin(activity_incremental)), float(np.nanmin(home_incremental)), float(np.nanmin(delta_incremental)), 0.0)
        pad = (ymax - ymin) * 0.1 if ymax > ymin else 0.1
        plt.ylim(ymin - pad, ymax + pad)
        plt.legend(fontsize=10)
        plt.tight_layout()
        filename = os.path.join(output_dir, f"exposure_incremental_{curve['person_id']}{suffix}.pdf")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

def _plot_individual_cumulative(time_points: np.ndarray, sample_curves: List[Dict],
                                output_dir: str, suffix: str,
                                metric: str, units: str) -> None:
    """
    Save one cumulative plot per sampled individual: activity vs home.
    """
    if not sample_curves:
        return

    metric_key = metric if metric in {'cumulative', 'twac'} else 'cumulative'
    ylabel = "Cumulative PM2.5 Exposure" if metric_key == 'cumulative' else "Time-Weighted Avg PM2.5"
    title = 'Individual Cumulative Exposure' if metric_key == 'cumulative' else 'Individual TWAC'

    for curve in sample_curves:
        person_id = curve.get('person_id', 'person')
        activity_curve = np.array(curve.get('activity_curve', []), dtype=float)
        home_curve = np.array(curve.get('home_curve', []), dtype=float)
        if activity_curve.size != time_points.size or home_curve.size != time_points.size:
            continue

        plt.figure(figsize=(10, 6))
        act_low = np.array(curve.get('activity_curve_low', []), dtype=float) if 'activity_curve_low' in curve else None
        act_high = np.array(curve.get('activity_curve_high', []), dtype=float) if 'activity_curve_high' in curve else None
        home_low = np.array(curve.get('home_curve_low', []), dtype=float) if 'home_curve_low' in curve else None
        home_high = np.array(curve.get('home_curve_high', []), dtype=float) if 'home_curve_high' in curve else None
        plt.plot(time_points, activity_curve, label='Activity', color='steelblue', linewidth=2)
        if act_low is not None and act_high is not None and act_low.size == activity_curve.size and act_high.size == activity_curve.size:
            plt.fill_between(time_points, act_low, act_high, color='steelblue', alpha=0.2, label='Activity band')
        plt.plot(time_points, home_curve, label='Home', color='tomato', linewidth=2, linestyle='--')
        if home_low is not None and home_high is not None and home_low.size == home_curve.size and home_high.size == home_curve.size:
            plt.fill_between(time_points, home_low, home_high, color='tomato', alpha=0.2, label='Home band')

        # Activity - Home cumulative difference
        delta_curve = activity_curve - home_curve
        plt.plot(time_points, delta_curve, label='Activity − Home', color='seagreen', linewidth=2, linestyle=':')

        # Difference uncertainty band: bound-wise subtraction (high-high, low-low)
        if (act_low is not None and act_high is not None and home_low is not None and home_high is not None
                and act_low.size == activity_curve.size and home_low.size == home_curve.size):
            delta_low = act_low - home_low
            delta_high = act_high - home_high
            plt.fill_between(time_points, delta_low, delta_high, color='seagreen', alpha=0.18, label='(A−H) band')
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel(f'{ylabel} ({units})', fontsize=12)
        naics_label = curve.get("naics_group")
        naics_text = f" (NAICS {naics_label})" if naics_label else ""
        plt.title(f"{title} — {person_id}{naics_text}", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.xlim(0, 24)
        ymax = max(float(np.nanmax(activity_curve)), float(np.nanmax(home_curve)), 0.1)
        plt.ylim(0, ymax * 1.1)
        plt.legend(fontsize=11)
        plt.tight_layout()
        outfile = os.path.join(output_dir, f"exposure_cumulative_{person_id}{suffix}.pdf")
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()


def _plot_sample_curves(time_points: np.ndarray, sample_curves: List[Dict],
                        output_dir: str, suffix: str,
                        metric: str, units: str) -> None:
    if not sample_curves:
        print("No sample curves found in summary data; skipping sample plot")
        return

    plt.figure(figsize=(12, 8))
    max_samples = min(10, len(sample_curves))
    for idx in range(max_samples):
        curve = sample_curves[idx]
        activity_curve = np.array(curve['activity_curve'], dtype=float)
        home_curve = np.array(curve['home_curve'], dtype=float)
        plt.plot(time_points, activity_curve, linewidth=1.5, label=f"{curve['person_id']} Activity")
        plt.plot(time_points, home_curve, linewidth=1.0, linestyle='--', label=f"{curve['person_id']} Home")

    plt.xlabel('Hour of Day', fontsize=12)
    ylabel = "Cumulative PM2.5 Exposure" if metric == 'cumulative' else "Time-Weighted Avg PM2.5"
    plt.ylabel(f'{ylabel} ({units})', fontsize=12)
    title = 'Sample Individual Cumulative Exposure Curves' if metric == 'cumulative' else 'Sample Individual TWAC Curves'
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 24)
    ymax = 0.1
    for curve in sample_curves[:max_samples]:
        activity_curve = np.array(curve['activity_curve'], dtype=float)
        home_curve = np.array(curve['home_curve'], dtype=float)
        ymax = max(ymax, float(np.nanmax(activity_curve)), float(np.nanmax(home_curve)))
    plt.ylim(0, ymax * 1.1)
    plt.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'exposure_sample_curves{suffix}.pdf'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    generate_summary_plots()

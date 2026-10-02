#!/usr/bin/env python
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_map(lon, lat, values, title, fname, cmap):
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(lon, lat, c=values, s=3, cmap=cmap, alpha=0.8)
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    ax.set_title(title)
    plt.colorbar(sc, ax=ax, label="PM2.5 width")
    plt.tight_layout()
    fig.savefig(fname, dpi=250)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True, help="Path to run root containing outputs/*quantiles.csv")
    args = ap.parse_args()
    root = Path(args.run_root)
    qcsv = next(root.glob("outputs/*_dynamic_surface_quantiles.csv"))
    df = pd.read_csv(qcsv)
    lon, lat = df["lon"], df["lat"]
    w95 = df["pm25_p95"] - df["pm25_p05"]
    plot_map(lon, lat, w95, "Uncertainty width (p95 - p05)", "width_map_p95_p05_mean.pdf", "inferno")
    if {"pm25_p75","pm25_p25"}.issubset(df.columns):
        w75 = df["pm25_p75"] - df["pm25_p25"]
        plot_map(lon, lat, w75, "Uncertainty width (p75 - p25)", "width_map_p75_p25_mean.pdf", "viridis")
    print("Saved width maps in current directory.")

if __name__ == "__main__":
    main()

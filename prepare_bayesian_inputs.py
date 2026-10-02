#!/usr/bin/env python3
"""Populate bayesian_fusion city folders with cleaned CSV + sensors CSV.

- Copies cleaned_*.csv from purpleair_history/<msa>_2023-04/ into the city folder
  using the expected naming pattern cleaned_20230401_20230501_<city>.csv
- Generates <msa-slug>_sensors.csv from msa_sensor_list_purple_air

Does NOT create baseline CSVs (GHAP required).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import shutil

special_prefixes = {
    ('san', 'francisco'): 'san_francisco',
    ('san', 'jose'): 'san_jose',
    ('san', 'diego'): 'san_diego',
    ('san', 'antonio'): 'san_antonio',
    ('los', 'angeles'): 'los_angeles',
    ('las', 'vegas'): 'las_vegas',
    ('new', 'york'): 'new_york',
    ('st', 'louis'): 'st_louis',
    ('salt', 'lake', 'city'): 'salt_lake_city',
    ('virginia', 'beach'): 'virginia_beach',
    ('grand', 'rapids'): 'grand_rapids',
    ('kansas', 'city'): 'kansas_city',
}

special_prefixes_camel = {
    ('san', 'francisco'): 'SanFrancisco',
    ('san', 'jose'): 'SanJose',
    ('san', 'diego'): 'SanDiego',
    ('san', 'antonio'): 'SanAntonio',
    ('los', 'angeles'): 'LosAngeles',
    ('las', 'vegas'): 'LasVegas',
    ('new', 'york'): 'NewYork',
    ('st', 'louis'): 'StLouis',
    ('salt', 'lake', 'city'): 'SaltLakeCity',
    ('virginia', 'beach'): 'VirginiaBeach',
    ('grand', 'rapids'): 'GrandRapids',
    ('kansas', 'city'): 'KansasCity',
}


def msa_to_city_key(slug: str) -> str:
    parts = slug.split('-')
    if len(parts) >= 2 and tuple(parts[:2]) in special_prefixes:
        return special_prefixes[tuple(parts[:2])]
    if len(parts) >= 3 and tuple(parts[:3]) in special_prefixes:
        return special_prefixes[tuple(parts[:3])]
    return parts[0]


def msa_to_folder(slug: str) -> str:
    parts = slug.split('-')
    if len(parts) >= 2 and tuple(parts[:2]) in special_prefixes_camel:
        return special_prefixes_camel[tuple(parts[:2])]
    if len(parts) >= 3 and tuple(parts[:3]) in special_prefixes_camel:
        return special_prefixes_camel[tuple(parts[:3])]
    return parts[0].capitalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate Bayesian-fusion city folders with cleaned PurpleAir CSVs and sensor CSVs."
    )
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project/input root containing the expected folders.")
    parser.add_argument("--base", type=Path, help="Bayesian-fusion city folder root. Defaults to ROOT/bayesian_fusion.")
    parser.add_argument(
        "--history",
        type=Path,
        help="PurpleAir history root. Defaults to ROOT/EDA_PM25/data/purpleair_history.",
    )
    parser.add_argument(
        "--geojson-dir",
        type=Path,
        help="MSA sensor-list GeoJSON root. Defaults to ROOT/EDA_PM25/data/msa_sensor_list_purple_air.",
    )
    parser.add_argument(
        "--converter",
        type=Path,
        default=Path(__file__).resolve().parent / "tools" / "geojson_to_sensors_csv.py",
        help="Path to geojson_to_sensors_csv.py.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    base = (args.base or root / "bayesian_fusion").resolve()
    hist = (args.history or root / "EDA_PM25" / "data" / "purpleair_history").resolve()
    geojson_dir = (args.geojson_dir or root / "EDA_PM25" / "data" / "msa_sensor_list_purple_air").resolve()
    converter = args.converter.resolve()

    slugs = sorted(p.stem for p in geojson_dir.glob('*.geojson'))
    copied = 0
    sensors_built = 0

    for slug in slugs:
        folder = base / f'bayesian_fusion_{msa_to_folder(slug)}'
        if folder.name.startswith('_bayesian_fusion'):
            continue
        if not folder.exists():
            continue

        city = msa_to_city_key(slug)
        target_cleaned = folder / f'cleaned_20230401_20230501_{city}.csv'
        target_sensors = folder / f'{slug}_sensors.csv'

        # Copy cleaned CSV from purpleair_history
        hist_folder = hist / f'{slug}_2023-04'
        cleaned_src = None
        if hist_folder.exists():
            cleaned_files = sorted(hist_folder.glob('cleaned_*.csv'))
            if cleaned_files:
                cleaned_src = cleaned_files[-1]
        if cleaned_src and not target_cleaned.exists():
            shutil.copy2(cleaned_src, target_cleaned)
            copied += 1

        # Build sensors CSV
        if not target_sensors.exists():
            geojson = geojson_dir / f'{slug}.geojson'
            if geojson.exists() and converter.exists():
                import subprocess
                subprocess.run(
                    [sys.executable, str(converter), '--input', str(geojson), '--out', str(target_sensors)],
                    check=False,
                )
                if target_sensors.exists():
                    sensors_built += 1

    print(f'copied_cleaned={copied} sensors_built={sensors_built}')


if __name__ == '__main__':
    main()

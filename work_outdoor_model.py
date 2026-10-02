#!/usr/bin/env python3
"""
Work-outdoor probability model (deterministic; no Bayesian fitting).

Goal: Attach a probability of outdoor work during the workday to each person,
based on:
  - DAS person industry (NAICS, possibly aggregated in the synthetic data),
  - OEWS (OEWS_nat4d_M2022_dl.csv): industry-by-occupation employment weights,
  - O*NET Work Context: outdoor-related work context category distributions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def _safe_numeric(series: pd.Series) -> pd.Series:
    # OEWS fields often contain commas and suppression markers like * / **.
    s = series.astype(str).str.replace(",", "", regex=False)
    s = s.replace({"*": np.nan, "**": np.nan, "nan": np.nan})
    return pd.to_numeric(s, errors="coerce")


def normalize_soc_code(code: str) -> Optional[str]:
    """Normalize SOC-like codes to the 7-char form '11-1011' when possible."""
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return None
    s = str(code).strip()
    if not s:
        return None
    # O*NET uses e.g. '11-1011.00'; OEWS uses '11-1011'
    if s.endswith(".00"):
        s = s[:-3]
    # Keep only typical SOC patterns.
    m = re.match(r"^\d{2}-\d{4}$", s)
    if m:
        return s
    # Some rows may contain 2- or 3-digit groupings (e.g., 11-0000). We skip them
    # in the baseline to avoid double-counting / ambiguity.
    return None


def parse_naics_group(value: str) -> Tuple[Optional[str], str]:
    """
    Parse DAS `trip_taker_industry` into a 2-digit sector or a combined-sector group.

    Returns (naics_group, reason):
      - naics_group: e.g. '62', '31_33', '44_45', '48_49', or None
      - reason: 'ok', 'not_working', 'missing', 'unparsed'
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, "missing"
    s = str(value).strip().lower()
    if not s or s == "nan":
        return None, "missing"
    if s == "not_working":
        return None, "not_working"

    # Common aggregated sector buckets in the DAS.
    if s.startswith("naics31_33"):
        return "31_33", "ok"
    if s.startswith("naics44_45"):
        return "44_45", "ok"
    if s.startswith("naics48_49"):
        return "48_49", "ok"

    # Extract first run of 2-6 digits; use first two digits as sector.
    m = re.search(r"(\d{2,6})", s)
    if not m:
        return None, "unparsed"
    digits = m.group(1)
    if len(digits) < 2:
        return None, "unparsed"
    return digits[:2], "ok"


@dataclass
class WorkOutdoorConfig:
    oews_nat4d_path: Path
    onet_work_context_path: Path
    outdoor_elements: List[str]
    outdoor_category_threshold: int = 4
    combine_elements: str = "union"  # 'union' or 'mean'
    oews_occ_group: str = "detailed"  # prefer detailed SOC rows


class MissingnessTracker:
    def __init__(self) -> None:
        self.rows: List[Dict[str, object]] = []

    def add(self, step: str, metric: str, value: object) -> None:
        self.rows.append({"step": step, "metric": metric, "value": value})

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def compute_soc_outdoor_metric(cfg: WorkOutdoorConfig, tracker: MissingnessTracker) -> pd.DataFrame:
    onet_path = Path(cfg.onet_work_context_path)
    df = pd.read_csv(
        onet_path,
        encoding="utf-8-sig",
        usecols=["O*NET-SOC Code", "Element Name", "Scale ID", "Category", "Data Value"],
    )
    tracker.add("onet_load", "rows", int(len(df)))

    df = df[(df["Scale ID"] == "CXP") & (df["Element Name"].isin(cfg.outdoor_elements))].copy()
    tracker.add("onet_filter", "rows_cxp_outdoor", int(len(df)))

    df["soc"] = df["O*NET-SOC Code"].map(normalize_soc_code)
    df["Category"] = pd.to_numeric(df["Category"], errors="coerce")
    df["Data Value"] = pd.to_numeric(df["Data Value"], errors="coerce")
    df = df.dropna(subset=["soc", "Category", "Data Value"])
    tracker.add("onet_clean", "rows_valid", int(len(df)))

    thresh = int(cfg.outdoor_category_threshold)
    df["is_outdoor_cat"] = df["Category"] >= thresh
    # Avoid groupby.apply for performance + future pandas compatibility.
    agg = (
        df.loc[df["is_outdoor_cat"]]
        .groupby(["soc", "Element Name"], as_index=False)["Data Value"]
        .sum()
        .rename(columns={"Data Value": "p_outdoor"})
    )
    agg["p_outdoor"] = agg["p_outdoor"].astype(float) / 100.0

    # Pivot elements into columns and combine.
    piv = agg.pivot(index="soc", columns="Element Name", values="p_outdoor")
    piv = piv.reindex(columns=cfg.outdoor_elements)
    tracker.add("onet_metric", "unique_soc", int(piv.shape[0]))
    tracker.add("onet_metric", "soc_with_all_elements", int(piv.dropna().shape[0]))

    if cfg.combine_elements == "mean":
        p_soc = piv.mean(axis=1, skipna=False)
    else:
        # Union-style: 1 - prod_e (1 - p_e)
        p_soc = 1.0 - (1.0 - piv).prod(axis=1, skipna=False)

    out = pd.DataFrame({"soc": p_soc.index.astype(str), "p_outdoor_soc": p_soc.values.astype(float)})
    out = out.dropna(subset=["p_outdoor_soc"])
    tracker.add("onet_metric", "soc_with_metric", int(len(out)))
    return out


def compute_sector_soc_weights(cfg: WorkOutdoorConfig, tracker: MissingnessTracker) -> Tuple[pd.DataFrame, pd.DataFrame]:
    oews_path = Path(cfg.oews_nat4d_path)
    df = pd.read_csv(
        oews_path,
        encoding="utf-8-sig",
        usecols=["NAICS", "OCC_CODE", "O_GROUP", "TOT_EMP"],
        dtype={"NAICS": str, "OCC_CODE": str, "O_GROUP": str, "TOT_EMP": str},
    )
    tracker.add("oews_load", "rows", int(len(df)))

    if cfg.oews_occ_group:
        df = df[df["O_GROUP"] == cfg.oews_occ_group].copy()
        tracker.add("oews_filter", f"rows_o_group_{cfg.oews_occ_group}", int(len(df)))

    df["tot_emp"] = _safe_numeric(df["TOT_EMP"])
    before = len(df)
    df = df.dropna(subset=["NAICS", "OCC_CODE", "tot_emp"])
    tracker.add("oews_clean", "rows_dropped_bad_tot_emp", int(before - len(df)))

    df["sector"] = df["NAICS"].astype(str).str.slice(0, 2)
    # Normalize SOC codes; keep only detailed SOCs (e.g., 11-1011).
    df["soc"] = df["OCC_CODE"].map(normalize_soc_code)
    before_soc = len(df)
    df = df.dropna(subset=["soc"])
    tracker.add("oews_clean", "rows_dropped_non_detailed_soc", int(before_soc - len(df)))

    df = df.groupby(["sector", "soc"], as_index=False)["tot_emp"].sum()
    tracker.add("oews_weights", "unique_sectors", int(df["sector"].nunique()))
    tracker.add("oews_weights", "unique_socs", int(df["soc"].nunique()))

    sector_totals = df.groupby("sector", as_index=False)["tot_emp"].sum().rename(columns={"tot_emp": "sector_tot_emp"})
    weights = df.merge(sector_totals, on="sector", how="left")
    weights["w_soc_given_sector"] = weights["tot_emp"] / weights["sector_tot_emp"]
    return weights[["sector", "soc", "w_soc_given_sector"]], sector_totals


def compute_sector_outdoor_probs(
    weights: pd.DataFrame,
    soc_outdoor: pd.DataFrame,
    tracker: MissingnessTracker,
) -> pd.DataFrame:
    merged = weights.merge(soc_outdoor, on="soc", how="left", indicator=True)
    tracker.add("join_oews_onet", "rows", int(len(merged)))
    tracker.add("join_oews_onet", "rows_matched", int((merged["_merge"] == "both").sum()))
    tracker.add("join_oews_onet", "rows_unmatched", int((merged["_merge"] != "both").sum()))
    merged = merged[merged["_merge"] == "both"].copy()

    merged["contrib"] = merged["w_soc_given_sector"] * merged["p_outdoor_soc"]
    sector = merged.groupby("sector", as_index=False)["contrib"].sum().rename(columns={"contrib": "p_outdoor_work"})
    tracker.add("sector_probs", "sectors_with_prob", int(len(sector)))
    return sector


def assign_person_probs(
    person_industry: pd.DataFrame,
    sector_probs: pd.DataFrame,
    sector_totals: pd.DataFrame,
    tracker: MissingnessTracker,
) -> pd.DataFrame:
    # sector_probs: sector -> p_outdoor_work
    sector_map = dict(zip(sector_probs["sector"].astype(str), sector_probs["p_outdoor_work"].astype(float)))
    tot_map = dict(zip(sector_totals["sector"].astype(str), sector_totals["sector_tot_emp"].astype(float)))

    def _combined_prob(group: str) -> Optional[float]:
        # group like '31_33' or '44_45'
        parts = group.split("_")
        if len(parts) != 2:
            return None
        a, b = parts
        # include both endpoints and anything in between if numeric and consecutive (31_33 => 31,32,33)
        try:
            lo, hi = int(a), int(b)
        except ValueError:
            return None
        sectors = [f"{s:02d}" for s in range(lo, hi + 1)]
        num = 0.0
        den = 0.0
        for s in sectors:
            p = sector_map.get(s)
            w = tot_map.get(s)
            if p is None or w is None:
                continue
            num += p * w
            den += w
        return (num / den) if den > 0 else None

    probs: List[float] = []
    reasons: List[str] = []
    for _, row in person_industry.iterrows():
        g = row["naics_group"]
        r = row["naics_reason"]
        if r == "not_working":
            probs.append(np.nan)
            reasons.append("not_working")
            continue
        if g is None or (isinstance(g, float) and np.isnan(g)):
            probs.append(np.nan)
            reasons.append(r if r != "ok" else "missing_group")
            continue
        g = str(g)
        if "_" in g:
            p = _combined_prob(g)
            if p is None:
                probs.append(np.nan)
                reasons.append("unmapped_combined_group")
            else:
                probs.append(float(p))
                reasons.append("ok")
        else:
            p = sector_map.get(g)
            if p is None:
                probs.append(np.nan)
                reasons.append("sector_not_covered")
            else:
                probs.append(float(p))
                reasons.append("ok")

    out = person_industry.copy()
    out["p_outdoor_work"] = probs
    out["p_outdoor_reason"] = reasons
    tracker.add("person_assign", "persons_total", int(out["person_id"].nunique()))
    tracker.add("person_assign", "persons_with_prob", int(out["p_outdoor_work"].notna().sum()))
    tracker.add("person_assign", "persons_missing_prob", int(out["p_outdoor_work"].isna().sum()))
    return out[["person_id", "naics_raw", "naics_group", "p_outdoor_work", "p_outdoor_reason"]]


def build_person_industry_table(person_df: pd.DataFrame, tracker: MissingnessTracker) -> pd.DataFrame:
    df = person_df.copy()
    df["person_id"] = df["person_id"].astype(str)
    df["naics_raw"] = df["trip_taker_industry"]
    parsed = df["naics_raw"].map(parse_naics_group)
    df["naics_group"] = [x[0] for x in parsed]
    df["naics_reason"] = [x[1] for x in parsed]
    tracker.add("das_industry", "persons_total", int(df["person_id"].nunique()))
    tracker.add("das_industry", "persons_not_working", int((df["naics_reason"] == "not_working").sum()))
    tracker.add("das_industry", "persons_parsed", int((df["naics_reason"] == "ok").sum()))
    tracker.add("das_industry", "persons_unparsed_or_missing", int((df["naics_reason"].isin(["missing", "unparsed"])).sum()))
    return df[["person_id", "naics_raw", "naics_group", "naics_reason"]]


def compute_work_outdoor_probs(
    person_df: pd.DataFrame,
    cfg: WorkOutdoorConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    tracker = MissingnessTracker()
    person_industry = build_person_industry_table(person_df, tracker)
    soc_outdoor = compute_soc_outdoor_metric(cfg, tracker)
    weights, sector_totals = compute_sector_soc_weights(cfg, tracker)
    sector_probs = compute_sector_outdoor_probs(weights, soc_outdoor, tracker)
    person_probs = assign_person_probs(person_industry, sector_probs, sector_totals, tracker)
    return person_probs, sector_probs, tracker.to_frame()

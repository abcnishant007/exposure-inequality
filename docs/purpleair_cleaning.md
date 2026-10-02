# PurpleAir Cleaning Parameters

This describes the filters applied when `EDA_PM25/download_purpleair_history.py` builds the `cleaned_*.csv` output.

## Defaults
- `pm_min = 0.0`
- `pm_max = 500.0`
- `k_mad = 6.0`
- `step_max = 150.0`

## What Each Parameter Does

### `pm_min` and `pm_max`
After parsing the PM field (default `pm2.5_atm`), any rows outside the range are dropped:
- Keep rows where `pm_min <= pm2.5_atm <= pm_max`.

### MAD-based outlier filter (`k_mad`)
For each **sensor** and **day**, the script computes:
- Median PM for that sensor/day
- MAD (median absolute deviation) for that sensor/day
- Robust scale = `1.4826 * MAD`

Then it computes a robust z-score:
- `z_robust = |pm - median| / robust_scale`

Rows are **kept** if:
- `z_robust <= k_mad`, or
- `robust_scale` is 0/NaN (no variability), in which case rows are kept.

### Step-change filter (`step_max`)
After sorting by time for each sensor, it computes:
- `pm_diff = |pm(t) - pm(t-1)|`

Rows are **kept** if:
- `pm_diff` is NaN (first row), or
- `pm_diff <= step_max`

This removes sharp jumps that are likely sensor spikes.

## Output Files
- `raw_YYYYMMDD_YYYYMMDD.csv`: combined, unfiltered readings.
- `cleaned_YYYYMMDD_YYYYMMDD.csv`: combined, filtered readings using the rules above.

## Timestamp Conversion
The raw API provides `time_stamp` in Unix epoch seconds. The script converts it to UTC:
```python
df["datetime_utc"] = pd.to_datetime(df.pop("time_stamp"), unit="s", utc=True)
```

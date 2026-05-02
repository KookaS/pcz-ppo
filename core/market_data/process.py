"""process.py: minimum-defensible data quality utilities.

What's here:
- ``detect_gaps`` — flags bars whose timestamp doesn't follow the previous
  by exactly the bar interval. Adds an ``is_gap`` boolean column. Inserts
  *no* synthetic rows; consumers decide how to handle gaps. Forward-fill
  is **never** applied here — it would imply trades you couldn't have made.
- ``flag_outliers`` — flags bars whose log-return |z-score| over a rolling
  window exceeds threshold ``k``. One opinionated heuristic, documented
  limitations (it does not detect fat-fingered prints, exchange glitches,
  or correlated multi-bar anomalies).
- ``validate_schema`` — asserts a DataFrame matches the expected pyarrow
  schema by column names + dtypes.
- ``coerce_dtypes`` — applies the dtype map; raises on conversion failure.

What's NOT here (deliberate):
- Forward-fill / interpolation of price gaps (dangerous; off by default).
- Bid-ask reconstruction (needs L2 data we don't yet ingest).
- Trade-side imbalance features (microstructure transformation, premature
  here — belongs to a feature-engineering layer downstream).
- Robust statistics for adversarial data (research scope).
- Anomaly detection beyond rolling z-score (research scope).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from .schema import BARS_DTYPES, INTERVAL_TO_MS


def detect_gaps(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Add ``is_gap`` column flagging non-contiguous bars.

    A bar is flagged if its ``timestamp_ns`` is not equal to the previous
    bar's ``timestamp_ns + interval_ns``. The first bar is never flagged
    (no previous bar to compare).

    Operates on a copy; does not modify the input.
    """
    if interval not in INTERVAL_TO_MS:
        raise ValueError(f"unknown interval {interval!r}")
    interval_ns = INTERVAL_TO_MS[interval] * 1_000_000
    out = df.copy().sort_values("timestamp_ns").reset_index(drop=True)
    if out.empty:
        out["is_gap"] = pd.Series(dtype="boolean")
        return out
    diffs = out["timestamp_ns"].diff()
    is_gap = (diffs != interval_ns) & diffs.notna()
    out["is_gap"] = is_gap.astype("boolean")
    out.loc[out.index[0], "is_gap"] = False
    return out


def flag_outliers(
    df: pd.DataFrame,
    k: float = 5.0,
    window: int = 60,
    price_col: str = "close",
) -> pd.DataFrame:
    """Add ``is_outlier`` column based on rolling z-score of log-returns.

    Args:
        df: DataFrame conforming to ``BARS_SCHEMA``.
        k: Threshold; |z| > k flags the bar. Default 5.0 (very conservative —
            flags only severe deviations).
        window: Rolling window size in bars for mean/std estimation.
        price_col: Column to use for return computation.

    Returns:
        Copy of df with ``is_outlier`` column populated.

    Limitations (read before relying on this):
    - Detects single-bar log-return spikes only. Multi-bar drifts and
      regime-shift signatures are not detected.
    - The first ``window`` bars are NOT flagged (insufficient history).
    - Wash-trade prints, exchange glitches, and fat-fingered orders that
      revert within one bar may not produce a flagged log return on the
      bar timeframe (they show up in trades, not bars).
    - Sensitivity to `k` is high. Default k=5 is calibrated against
      Gaussian; real returns are heavier-tailed so k=5 is roughly the
      0.99th percentile in practice on minute-bar BTC.
    """
    out = df.copy().sort_values("timestamp_ns").reset_index(drop=True)
    if out.empty or len(out) <= window:
        out["is_outlier"] = pd.Series([False] * len(out), dtype="boolean")
        return out
    log_ret = np.log(out[price_col].astype("float64")).diff()
    rolling_mean = log_ret.rolling(window=window, min_periods=window).mean()
    rolling_std = log_ret.rolling(window=window, min_periods=window).std()
    z = (log_ret - rolling_mean) / rolling_std.replace(0.0, np.nan)
    is_outlier = (z.abs() > k) & z.notna()
    out["is_outlier"] = is_outlier.astype("boolean").fillna(False)
    return out


def validate_schema(df: pd.DataFrame, schema: pa.Schema) -> None:
    """Raise ``ValueError`` if ``df`` doesn't match ``schema``.

    Checks: every required (non-nullable) column is present, dtypes match
    the schema's pyarrow types within pandas-arrow equivalence.
    """
    expected_cols = {f.name for f in schema}
    actual_cols = set(df.columns)
    missing = expected_cols - actual_cols
    if missing:
        raise ValueError(f"DataFrame missing columns required by schema: {sorted(missing)}")
    for f in schema:
        if f.name not in df.columns:
            continue
        col = df[f.name]
        actual_pa = pa.array(col).type
        if not _pa_types_compatible(actual_pa, f.type):
            raise ValueError(f"column {f.name!r}: schema dtype {f.type} but DataFrame dtype {actual_pa}")


def _pa_types_compatible(actual: pa.DataType, expected: pa.DataType) -> bool:
    if actual.equals(expected):
        return True
    if pa.types.is_integer(actual) and pa.types.is_integer(expected):
        return True
    if pa.types.is_floating(actual) and pa.types.is_floating(expected):
        return True
    if pa.types.is_string(actual) and pa.types.is_string(expected):
        return True
    return pa.types.is_boolean(actual) and pa.types.is_boolean(expected)


def coerce_dtypes(df: pd.DataFrame, dtype_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Cast columns to the dtype map (defaults to ``BARS_DTYPES``).

    Skips columns not in the input. Raises on conversion failure.
    """
    target = dtype_map if dtype_map is not None else BARS_DTYPES
    available = {k: v for k, v in target.items() if k in df.columns}
    return df.astype(available)

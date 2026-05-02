"""Data-quality utilities: detect_gaps, flag_outliers, validate_schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from core.market_data import (
    BARS_SCHEMA,
    coerce_dtypes,
    detect_gaps,
    flag_outliers,
    validate_schema,
)
from core.market_data.process import _pa_types_compatible
from core.market_data.schema import INTERVAL_TO_MS


def _make_bars(n: int, interval_ms: int = 60_000, start_ns: int = 1_700_000_000_000_000_000) -> pd.DataFrame:
    interval_ns = interval_ms * 1_000_000
    timestamps = [start_ns + i * interval_ns for i in range(n)]
    rng = np.random.default_rng(42)
    closes = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame(
        {
            "timestamp_ns": pd.Series(timestamps, dtype="int64"),
            "venue": pd.Series(["binance"] * n, dtype="string"),
            "symbol": pd.Series(["BTCUSDT"] * n, dtype="string"),
            "interval": pd.Series(["1m"] * n, dtype="string"),
            "open": closes,
            "high": closes + 0.2,
            "low": closes - 0.2,
            "close": closes,
            "volume_base": np.full(n, 1.0),
            "volume_quote": closes,
            "n_trades": np.full(n, 5, dtype="int64"),
            "taker_buy_volume_base": np.full(n, 0.5),
            "taker_buy_volume_quote": closes / 2,
            "is_gap": pd.Series([pd.NA] * n, dtype="boolean"),
            "is_outlier": pd.Series([pd.NA] * n, dtype="boolean"),
        }
    )


def test_detect_gaps_no_gaps_in_dense_series():
    df = _make_bars(n=100)
    out = detect_gaps(df, interval="1m")
    assert (out["is_gap"] == False).all()  # noqa: E712 (pandas boolean compare)


def test_detect_gaps_flags_missing_bar():
    df = _make_bars(n=10)
    df_with_gap = df.drop(index=5).reset_index(drop=True)
    out = detect_gaps(df_with_gap, interval="1m")
    assert out["is_gap"].sum() == 1
    flagged = out[out["is_gap"]].iloc[0]
    assert flagged["timestamp_ns"] == df.iloc[6]["timestamp_ns"]


def test_detect_gaps_unknown_interval_raises():
    df = _make_bars(n=5)
    with pytest.raises(ValueError, match="unknown interval"):
        detect_gaps(df, interval="bogus")


def test_detect_gaps_first_bar_never_flagged():
    df = _make_bars(n=10)
    out = detect_gaps(df, interval="1m")
    assert bool(out.iloc[0]["is_gap"]) is False


def test_flag_outliers_clean_series_has_few_flags():
    df = _make_bars(n=200)
    out = flag_outliers(df, k=5.0, window=60)
    assert out["is_outlier"].sum() <= 2


def test_flag_outliers_detects_inserted_spike():
    df = _make_bars(n=200)
    df = df.copy()
    df.loc[150, "close"] = df.loc[149, "close"] * 1.20
    out = flag_outliers(df, k=4.0, window=60)
    assert bool(out.iloc[150]["is_outlier"]) is True


def test_flag_outliers_window_warmup_not_flagged():
    df = _make_bars(n=200)
    out = flag_outliers(df, k=5.0, window=60)
    assert out.iloc[:60]["is_outlier"].sum() == 0


def test_validate_schema_passes_for_well_formed_df():
    df = _make_bars(n=10)
    df = coerce_dtypes(df)
    validate_schema(df, BARS_SCHEMA)


def test_validate_schema_detects_missing_column():
    df = _make_bars(n=10).drop(columns=["close"])
    with pytest.raises(ValueError, match="missing columns"):
        validate_schema(df, BARS_SCHEMA)


def test_pa_types_compatible_int_variants():
    assert _pa_types_compatible(pa.int64(), pa.int32())
    assert _pa_types_compatible(pa.float32(), pa.float64())
    assert not _pa_types_compatible(pa.int64(), pa.string())


def test_interval_round_trip():
    assert "1m" in INTERVAL_TO_MS
    assert "5m" in INTERVAL_TO_MS
    assert INTERVAL_TO_MS["5m"] == 5 * INTERVAL_TO_MS["1m"]


def test_detect_gaps_empty_df():
    from core.market_data import BARS_DTYPES

    df = pd.DataFrame({k: pd.Series(dtype=v) for k, v in BARS_DTYPES.items()})
    out = detect_gaps(df, interval="1m")
    assert len(out) == 0
    assert "is_gap" in out.columns


def test_detect_gaps_single_row_is_gap_false_not_na():
    df = _make_bars(n=1)
    out = detect_gaps(df, interval="1m")
    val = out.iloc[0]["is_gap"]
    assert not pd.isna(val)
    assert bool(val) is False


def test_detect_gaps_unsorted_input_sorts_first():
    df = _make_bars(n=10)
    df = df.iloc[[2, 0, 1, 4, 3, 5, 6, 8, 7, 9]].reset_index(drop=True)
    out = detect_gaps(df, interval="1m")
    assert out["timestamp_ns"].is_monotonic_increasing
    assert out["is_gap"].sum() == 0


def test_flag_outliers_constant_prices_zero_flags():
    df = _make_bars(n=200)
    df = df.copy()
    for c in ("open", "high", "low", "close"):
        df[c] = 100.0
    out = flag_outliers(df, k=5.0, window=60)
    assert out["is_outlier"].sum() == 0


def test_flag_outliers_n_equals_window_zero_flags():
    df = _make_bars(n=60)
    out = flag_outliers(df, k=5.0, window=60)
    assert out["is_outlier"].sum() == 0

"""Schema definitions are stable, importable, and parquet-roundtrip-compatible."""

from __future__ import annotations

import io

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from core.market_data import (
    BARS_DTYPES,
    BARS_SCHEMA,
    INTERVAL_TO_MS,
    TRADES_DTYPES,
    TRADES_SCHEMA,
)


def _make_minimal_bars_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp_ns": pd.Series([1_000_000_000_000_000_000], dtype="int64"),
            "venue": pd.Series(["binance"], dtype="string"),
            "symbol": pd.Series(["BTCUSDT"], dtype="string"),
            "interval": pd.Series(["1m"], dtype="string"),
            "open": pd.Series([100.0], dtype="float64"),
            "high": pd.Series([101.0], dtype="float64"),
            "low": pd.Series([99.0], dtype="float64"),
            "close": pd.Series([100.5], dtype="float64"),
            "volume_base": pd.Series([1.5], dtype="float64"),
            "volume_quote": pd.Series([150.0], dtype="float64"),
            "n_trades": pd.Series([10], dtype="int64"),
            "taker_buy_volume_base": pd.Series([0.7], dtype="float64"),
            "taker_buy_volume_quote": pd.Series([70.5], dtype="float64"),
            "is_gap": pd.Series([False], dtype="boolean"),
            "is_outlier": pd.Series([False], dtype="boolean"),
        }
    )


def test_bars_schema_is_pyarrow_schema():
    assert isinstance(BARS_SCHEMA, pa.Schema)
    assert "timestamp_ns" in BARS_SCHEMA.names
    assert BARS_SCHEMA.field("timestamp_ns").type == pa.int64()


def test_trades_schema_is_pyarrow_schema():
    assert isinstance(TRADES_SCHEMA, pa.Schema)
    assert "trade_id" in TRADES_SCHEMA.names
    assert TRADES_SCHEMA.field("price").type == pa.float64()


def test_bars_dtypes_keys_match_schema():
    assert set(BARS_DTYPES) == set(BARS_SCHEMA.names)


def test_trades_dtypes_keys_match_schema():
    assert set(TRADES_DTYPES) == set(TRADES_SCHEMA.names)


def test_interval_to_ms_known_values():
    assert INTERVAL_TO_MS["1m"] == 60_000
    assert INTERVAL_TO_MS["1h"] == 3_600_000
    assert INTERVAL_TO_MS["1d"] == 86_400_000


def test_bars_roundtrip_parquet():
    df = _make_minimal_bars_df()
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, schema=BARS_SCHEMA)
    buf.seek(0)
    table = pq.read_table(buf)
    assert table.num_rows == 1
    out = table.to_pandas()
    assert out.iloc[0]["close"] == 100.5
    assert out.iloc[0]["symbol"] == "BTCUSDT"

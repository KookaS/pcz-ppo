"""schema.py: unified market-data parquet schemas.

Two independent schemas — one for OHLCV bars, one for tick-by-tick trades.
Bars are the lab-calibration unit (small, dense, easy to backtest); trades
are the microstructure unit (huge, sparse, needed only for fill-model
calibration). Most consumers will use bars.

Conventions:
- All timestamps are int64 nanoseconds since Unix epoch UTC.
- Bar timestamp = bar OPEN time, not close. Close-time is implicit
  (open_time + interval). This matches Binance's klines convention.
- All prices and sizes are float64. (Decimal would be safer for accounting
  but float is universally supported and adequate for research backtests
  where 1e-9 errors are not material.)
- `venue` and `symbol` are str — keep keys lowercase by convention
  (`binance`, `polymarket`) to avoid case-sensitivity bugs.
- `is_gap` and `is_outlier` are nullable booleans, set by the processing
  stage (NULL until a quality check stamps them). They are NOT set by the
  ingestion adapter — separation of concerns.
"""

from __future__ import annotations

import pyarrow as pa

BARS_DTYPES: dict[str, str] = {
    "timestamp_ns": "int64",
    "venue": "string",
    "symbol": "string",
    "interval": "string",
    "open": "float64",
    "high": "float64",
    "low": "float64",
    "close": "float64",
    "volume_base": "float64",
    "volume_quote": "float64",
    "n_trades": "int64",
    "taker_buy_volume_base": "float64",
    "taker_buy_volume_quote": "float64",
    "is_gap": "boolean",
    "is_outlier": "boolean",
}

BARS_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("interval", pa.string(), nullable=False),
        pa.field("open", pa.float64(), nullable=False),
        pa.field("high", pa.float64(), nullable=False),
        pa.field("low", pa.float64(), nullable=False),
        pa.field("close", pa.float64(), nullable=False),
        pa.field("volume_base", pa.float64(), nullable=False),
        pa.field("volume_quote", pa.float64(), nullable=False),
        pa.field("n_trades", pa.int64(), nullable=False),
        pa.field("taker_buy_volume_base", pa.float64(), nullable=True),
        pa.field("taker_buy_volume_quote", pa.float64(), nullable=True),
        pa.field("is_gap", pa.bool_(), nullable=True),
        pa.field("is_outlier", pa.bool_(), nullable=True),
    ]
)

TRADES_DTYPES: dict[str, str] = {
    "timestamp_ns": "int64",
    "venue": "string",
    "symbol": "string",
    "trade_id": "int64",
    "price": "float64",
    "size": "float64",
    "is_buyer_maker": "boolean",
}

TRADES_SCHEMA: pa.Schema = pa.schema(
    [
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("venue", pa.string(), nullable=False),
        pa.field("symbol", pa.string(), nullable=False),
        pa.field("trade_id", pa.int64(), nullable=False),
        pa.field("price", pa.float64(), nullable=False),
        pa.field("size", pa.float64(), nullable=False),
        pa.field("is_buyer_maker", pa.bool_(), nullable=False),
    ]
)

INTERVAL_TO_MS: dict[str, int] = {
    "1s": 1_000,
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
}

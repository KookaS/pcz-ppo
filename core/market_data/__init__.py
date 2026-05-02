"""market_data: venue-agnostic historical/recent market data ingestion.

Single concrete adapter today (Binance public REST). Additional venues
(Polymarket public REST, Coinbase, equities CSV) become drop-in additions
when their respective backlog stories unblock — the schema is the
abstraction, not a base class.

Public interface::

    from core.market_data import BinanceClient, BARS_SCHEMA, TRADES_SCHEMA
    from core.market_data import detect_gaps, flag_outliers, validate_schema

    client = BinanceClient()
    bars = client.fetch_bars("BTCUSDT", "1m", start="2026-04-29T00:00:00Z",
                             end="2026-04-29T01:00:00Z")
    bars = detect_gaps(bars, interval="1m")
    bars = flag_outliers(bars, k=5.0)
    bars.to_parquet("artifacts/market-making/data/btcusdt_1m.parquet")

CLI::

    uv run python -m core.market_data fetch \\
        --venue binance --kind bars --symbol BTCUSDT --interval 1m \\
        --start 2026-04-29T00:00:00Z --end 2026-04-29T01:00:00Z \\
        --output artifacts/market-making/data/btcusdt_1m.parquet
"""

from .binance import BinanceClient
from .process import (
    coerce_dtypes,
    detect_gaps,
    flag_outliers,
    validate_schema,
)
from .schema import (
    BARS_DTYPES,
    BARS_SCHEMA,
    INTERVAL_TO_MS,
    TRADES_DTYPES,
    TRADES_SCHEMA,
)

__all__ = [
    "BinanceClient",
    "BARS_SCHEMA",
    "BARS_DTYPES",
    "TRADES_SCHEMA",
    "TRADES_DTYPES",
    "INTERVAL_TO_MS",
    "detect_gaps",
    "flag_outliers",
    "validate_schema",
    "coerce_dtypes",
]

"""cli.py: command-line entry point for market-data fetching.

Examples::

    # 1 hour of BTCUSDT 1m bars (offline backfill)
    python -m core.market_data fetch \\
        --venue binance --kind bars \\
        --symbol BTCUSDT --interval 1m \\
        --start 2026-04-29T00:00:00Z --end 2026-04-29T01:00:00Z \\
        --output artifacts/market-making/data/btcusdt_1m.parquet

    # Latest bar (for "is the API alive?" smoke checks)
    python -m core.market_data fetch \\
        --venue binance --kind latest-bar \\
        --symbol BTCUSDT --interval 1m \\
        --output -

    # Latest aggTrade
    python -m core.market_data fetch \\
        --venue binance --kind latest-trade \\
        --symbol BTCUSDT --output -
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .binance import BinanceClient
from .process import detect_gaps, flag_outliers, validate_schema
from .schema import BARS_SCHEMA, TRADES_SCHEMA

KIND_BARS = "bars"
KIND_TRADES = "trades"
KIND_LATEST_BAR = "latest-bar"
KIND_LATEST_TRADE = "latest-trade"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="core.market_data")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fetch = sub.add_parser("fetch", help="Fetch market data and write parquet (or stdout).")
    fetch.add_argument("--venue", choices=["binance"], default="binance")
    fetch.add_argument(
        "--kind",
        choices=[KIND_BARS, KIND_TRADES, KIND_LATEST_BAR, KIND_LATEST_TRADE],
        default=KIND_BARS,
    )
    fetch.add_argument("--symbol", required=True)
    fetch.add_argument("--interval", default="1m", help="Bar interval (1m, 5m, 1h, ...).")
    fetch.add_argument("--start", help="ISO-8601 UTC, required for --kind=bars/trades.")
    fetch.add_argument("--end", help="ISO-8601 UTC, required for --kind=bars/trades.")
    fetch.add_argument(
        "--output",
        required=True,
        help="Path to parquet file, or '-' to print head() to stdout.",
    )
    fetch.add_argument(
        "--no-quality",
        action="store_true",
        help="Skip detect_gaps + flag_outliers (raw output).",
    )
    fetch.add_argument(
        "--outlier-k",
        type=float,
        default=5.0,
        help="Outlier z-score threshold (default 5.0).",
    )

    args = parser.parse_args(argv)
    if args.cmd != "fetch":
        parser.error(f"unknown command {args.cmd}")
        return 2

    client = BinanceClient()
    df = _dispatch(client, args)
    if args.kind in (KIND_BARS, KIND_LATEST_BAR) and not args.no_quality:
        df = detect_gaps(df, interval=args.interval)
        df = flag_outliers(df, k=args.outlier_k)

    schema = BARS_SCHEMA if args.kind in (KIND_BARS, KIND_LATEST_BAR) else TRADES_SCHEMA
    if not df.empty:
        validate_schema(df, schema)

    _write(df, args.output, schema)
    return 0


def _dispatch(client: BinanceClient, args: argparse.Namespace) -> pd.DataFrame:
    if args.kind == KIND_LATEST_BAR:
        return client.fetch_latest_bar(args.symbol, args.interval)
    if args.kind == KIND_LATEST_TRADE:
        return client.fetch_latest_trade(args.symbol)
    if not (args.start and args.end):
        sys.stderr.write("--start and --end are required for --kind=bars/trades\n")
        sys.exit(2)
    if args.kind == KIND_BARS:
        return client.fetch_bars(args.symbol, args.interval, args.start, args.end)
    if args.kind == KIND_TRADES:
        return client.fetch_trades(args.symbol, args.start, args.end)
    raise ValueError(f"unknown kind {args.kind!r}")


def _write(df: pd.DataFrame, output: str, schema) -> None:
    if output == "-":
        if df.empty:
            print("(empty result)")
            return
        print(f"# {len(df)} rows")
        print(df.head(10).to_string(index=False))
        return
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False, schema=schema)
    print(f"wrote {len(df)} rows → {out_path}")


if __name__ == "__main__":
    sys.exit(main())

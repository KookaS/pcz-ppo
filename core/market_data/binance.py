"""binance.py: Binance public REST client for klines + aggTrades.

No authentication required for these endpoints. Rate limit: 1200 weight/min/IP;
klines and aggTrades are 1 weight per call (limit ≤ 1000 rows per call).

The client is intentionally minimal — no WebSocket, no signed endpoints, no
account APIs. It supports four operations:

    fetch_bars(symbol, interval, start, end)        — paginated historical bars
    fetch_trades(symbol, start, end)                — paginated historical trades
    fetch_latest_bar(symbol, interval)              — single most-recent bar
    fetch_latest_trade(symbol)                      — single most-recent trade

All return a pandas DataFrame conforming to ``BARS_SCHEMA`` /
``TRADES_SCHEMA`` from ``schema.py``.

Latency caveat: REST polling has 1-2s lag from "true now"; this client is
fine for offline backfill and "recent" queries but is NOT suitable for live
quote-management. WebSocket support is documented as future work in
``docs/pcz-ppo/data-pipeline.md``.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from typing import Any

import pandas as pd
import requests

from .schema import BARS_DTYPES, INTERVAL_TO_MS, TRADES_DTYPES

DEFAULT_BASE_URL = "https://api.binance.com"
KLINES_PATH = "/api/v3/klines"
AGG_TRADES_PATH = "/api/v3/aggTrades"
DEFAULT_TIMEOUT_S = 15.0
MAX_LIMIT = 1000
VENUE = "binance"


class BinanceAPIError(RuntimeError):
    """Raised when Binance returns a non-200 or shape-invalid response."""


def _parse_iso8601(s: str | datetime) -> datetime:
    """Parse an ISO-8601 string or pass through a datetime; assume UTC if naive."""
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=UTC)
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _to_ms(dt: datetime | str) -> int:
    return int(_parse_iso8601(dt).timestamp() * 1000)


class BinanceClient:
    """Thin Binance public REST client.

    Args:
        base_url: Override the API base URL (e.g. for testnet or a mock server).
        session: Optional ``requests.Session`` for connection reuse / mocking.
        request_interval_s: Min seconds between requests (default 0.05 →
            ~20 req/s, well under the 1200 weight/min limit for klines).
        timeout_s: Per-request HTTP timeout.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        session: requests.Session | None = None,
        request_interval_s: float = 0.05,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ):
        self.base_url = base_url.rstrip("/")
        self.session = session if session is not None else requests.Session()
        self.request_interval_s = float(request_interval_s)
        self.timeout_s = float(timeout_s)
        self._last_request_t = 0.0

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_t
        wait = self.request_interval_s - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_t = time.monotonic()

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        self._throttle()
        url = f"{self.base_url}{path}"
        resp = self.session.get(url, params=params, timeout=self.timeout_s)
        if resp.status_code != 200:
            raise BinanceAPIError(f"GET {url} params={params} → HTTP {resp.status_code}: {resp.text[:300]}")
        try:
            return resp.json()
        except ValueError as exc:
            raise BinanceAPIError(f"GET {url} returned non-JSON: {resp.text[:200]}") from exc

    @staticmethod
    def _klines_rows_to_df(rows: list[list[Any]], symbol: str, interval: str) -> pd.DataFrame:
        if not rows:
            return _empty_bars_df()
        cols = [
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume_base",
            "close_time_ms",
            "volume_quote",
            "n_trades",
            "taker_buy_volume_base",
            "taker_buy_volume_quote",
            "_ignore",
        ]
        df = pd.DataFrame(rows, columns=cols)
        out = pd.DataFrame(
            {
                "timestamp_ns": (df["open_time_ms"].astype("int64") * 1_000_000),
                "venue": VENUE,
                "symbol": symbol,
                "interval": interval,
                "open": df["open"].astype("float64"),
                "high": df["high"].astype("float64"),
                "low": df["low"].astype("float64"),
                "close": df["close"].astype("float64"),
                "volume_base": df["volume_base"].astype("float64"),
                "volume_quote": df["volume_quote"].astype("float64"),
                "n_trades": df["n_trades"].astype("int64"),
                "taker_buy_volume_base": df["taker_buy_volume_base"].astype("float64"),
                "taker_buy_volume_quote": df["taker_buy_volume_quote"].astype("float64"),
                "is_gap": pd.Series([pd.NA] * len(df), dtype="boolean"),
                "is_outlier": pd.Series([pd.NA] * len(df), dtype="boolean"),
            }
        )
        return out.astype({k: v for k, v in BARS_DTYPES.items() if k in out.columns})

    @staticmethod
    def _trades_rows_to_df(rows: list[dict[str, Any]], symbol: str) -> pd.DataFrame:
        if not rows:
            return _empty_trades_df()
        df = pd.DataFrame(rows)
        out = pd.DataFrame(
            {
                "timestamp_ns": (df["T"].astype("int64") * 1_000_000),
                "venue": VENUE,
                "symbol": symbol,
                "trade_id": df["a"].astype("int64"),
                "price": df["p"].astype("float64"),
                "size": df["q"].astype("float64"),
                "is_buyer_maker": df["m"].astype("boolean"),
            }
        )
        return out.astype({k: v for k, v in TRADES_DTYPES.items() if k in out.columns})

    def fetch_bars(
        self,
        symbol: str,
        interval: str,
        start: datetime | str,
        end: datetime | str,
        max_pages: int = 10_000,
    ) -> pd.DataFrame:
        """Fetch all bars in [start, end] inclusive on bar ``open_time`` (UTC).

        Note: Binance's klines endpoint is inclusive on both endpoints,
        filtering by ``open_time``. Practical consequence: a 1-hour fetch
        with 1m interval returns **61** bars (open_times at minute 0, 1, ...,
        60), not 60. The bar at ``open_time == end_ms`` covers the period
        [end_ms, end_ms + interval) — its data extends past the requested
        end. If the caller needs strict half-open [start, end), filter the
        returned DataFrame on ``timestamp_ns < end_ns`` downstream.

        Args:
            symbol: Native venue symbol (e.g. ``"BTCUSDT"``).
            interval: Bar interval string (``"1m"``, ``"5m"``, ``"1h"``, ...).
            start: ISO-8601 string or timezone-aware datetime (UTC).
            end: ISO-8601 string or timezone-aware datetime (UTC). Must be > start.
            max_pages: Hard cap on pagination iterations (raises
                ``BinanceAPIError`` if exceeded). Default 10000.
        """
        if interval not in INTERVAL_TO_MS:
            raise ValueError(f"unknown interval {interval!r}; options: {sorted(INTERVAL_TO_MS)}")
        start_ms = _to_ms(start)
        end_ms = _to_ms(end)
        if end_ms <= start_ms:
            raise ValueError(f"end ({end_ms}) must be > start ({start_ms})")

        chunks: list[pd.DataFrame] = []
        cur = start_ms
        for _ in range(max_pages):
            data = self._get(
                KLINES_PATH,
                {
                    "symbol": symbol,
                    "interval": interval,
                    "startTime": cur,
                    "endTime": end_ms,
                    "limit": MAX_LIMIT,
                },
            )
            if not data:
                break
            chunk = self._klines_rows_to_df(data, symbol=symbol, interval=interval)
            chunks.append(chunk)
            last_open_ms = int(data[-1][0])
            cur = last_open_ms + INTERVAL_TO_MS[interval]
            if len(data) < MAX_LIMIT or cur >= end_ms:
                break
        else:
            raise BinanceAPIError(
                f"fetch_bars exceeded max_pages={max_pages} for {symbol} {interval} {start_ms}-{end_ms}"
            )
        if not chunks:
            return _empty_bars_df()
        return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["timestamp_ns"], keep="first")

    def fetch_trades(
        self,
        symbol: str,
        start: datetime | str,
        end: datetime | str,
        max_pages: int = 10_000,
    ) -> pd.DataFrame:
        """Fetch all aggregated trades in [start, end] (UTC). Paginated."""
        start_ms = _to_ms(start)
        end_ms = _to_ms(end)
        if end_ms <= start_ms:
            raise ValueError(f"end ({end_ms}) must be > start ({start_ms})")

        chunks: list[pd.DataFrame] = []
        cur = start_ms
        for _ in range(max_pages):
            data = self._get(
                AGG_TRADES_PATH,
                {
                    "symbol": symbol,
                    "startTime": cur,
                    "endTime": min(cur + 60 * 60 * 1000, end_ms),
                    "limit": MAX_LIMIT,
                },
            )
            if not data:
                cur += 60 * 60 * 1000
                if cur >= end_ms:
                    break
                continue
            chunk = self._trades_rows_to_df(data, symbol=symbol)
            chunks.append(chunk)
            last_T = int(data[-1]["T"])
            cur = last_T + 1
            if cur >= end_ms:
                break
        else:
            raise BinanceAPIError(f"fetch_trades exceeded max_pages={max_pages} for {symbol} {start_ms}-{end_ms}")
        if not chunks:
            return _empty_trades_df()
        return pd.concat(chunks, ignore_index=True).drop_duplicates(subset=["trade_id"], keep="first")

    def fetch_latest_bar(self, symbol: str, interval: str) -> pd.DataFrame:
        """Single most-recent (possibly-still-open) bar."""
        if interval not in INTERVAL_TO_MS:
            raise ValueError(f"unknown interval {interval!r}")
        data = self._get(KLINES_PATH, {"symbol": symbol, "interval": interval, "limit": 1})
        return self._klines_rows_to_df(data, symbol=symbol, interval=interval)

    def fetch_latest_trade(self, symbol: str) -> pd.DataFrame:
        """Single most-recent aggregated trade.

        Note: REST polling has 1-2s lag. For sub-second freshness, use a
        WebSocket client (not implemented in this module).
        """
        end_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
        start_ms = end_ms - 60_000
        data = self._get(
            AGG_TRADES_PATH,
            {"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1},
        )
        if not data:
            return _empty_trades_df()
        return self._trades_rows_to_df(data[-1:], symbol=symbol)


def _empty_bars_df() -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in BARS_DTYPES.items()})


def _empty_trades_df() -> pd.DataFrame:
    return pd.DataFrame({k: pd.Series(dtype=v) for k, v in TRADES_DTYPES.items()})

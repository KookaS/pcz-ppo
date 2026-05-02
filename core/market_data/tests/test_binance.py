"""BinanceClient: HTTP-mocked unit tests; no live API calls."""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from core.market_data import BARS_SCHEMA, TRADES_SCHEMA, BinanceClient
from core.market_data.binance import BinanceAPIError, _to_ms


def _mock_response(payload, status_code: int = 200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = payload
    resp.text = str(payload)
    return resp


def _mock_klines_row(open_time_ms: int, close: float = 100.0):
    open_p = close - 0.1
    high = close + 0.2
    low = close - 0.2
    return [
        open_time_ms,
        f"{open_p:.8f}",
        f"{high:.8f}",
        f"{low:.8f}",
        f"{close:.8f}",
        "1.5",
        open_time_ms + 60_000 - 1,
        f"{1.5 * close:.8f}",
        7,
        "0.7",
        f"{0.7 * close:.8f}",
        "0",
    ]


def _mock_aggtrade(t_ms: int, agg_id: int, price: float):
    return {
        "a": agg_id,
        "p": f"{price:.8f}",
        "q": "0.01",
        "f": agg_id * 10,
        "l": agg_id * 10 + 1,
        "T": t_ms,
        "m": True,
        "M": True,
    }


def test_to_ms_iso8601():
    assert _to_ms("2026-04-29T00:00:00Z") == int(pd.Timestamp("2026-04-29T00:00:00Z").timestamp() * 1000)


def test_fetch_bars_paginates_and_dedupes():
    session = MagicMock()
    page1 = [_mock_klines_row(t * 60_000, close=100.0 + t * 0.1) for t in range(1000)]
    page2 = [_mock_klines_row((1000 + t) * 60_000, close=200.0 + t * 0.1) for t in range(500)]
    session.get = MagicMock(side_effect=[_mock_response(page1), _mock_response(page2)])
    client = BinanceClient(session=session, request_interval_s=0.0)
    df = client.fetch_bars(
        "BTCUSDT",
        "1m",
        start="1970-01-01T00:00:00Z",
        end="1970-01-02T00:00:00Z",
    )
    assert len(df) == 1500
    assert (df["symbol"] == "BTCUSDT").all()
    assert (df["interval"] == "1m").all()
    assert (df["venue"] == "binance").all()
    assert df["timestamp_ns"].is_monotonic_increasing
    assert set(df.columns) == set(BARS_SCHEMA.names)


def test_fetch_bars_empty_response():
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response([]))
    client = BinanceClient(session=session, request_interval_s=0.0)
    df = client.fetch_bars(
        "BTCUSDT",
        "1m",
        start="1970-01-01T00:00:00Z",
        end="1970-01-02T00:00:00Z",
    )
    assert df.empty
    assert set(df.columns) == set(BARS_SCHEMA.names)


def test_fetch_bars_unknown_interval():
    client = BinanceClient(session=MagicMock(), request_interval_s=0.0)
    with pytest.raises(ValueError, match="unknown interval"):
        client.fetch_bars("BTCUSDT", "bogus", "1970-01-01T00:00:00Z", "1970-01-02T00:00:00Z")


def test_fetch_bars_inverted_window():
    client = BinanceClient(session=MagicMock(), request_interval_s=0.0)
    with pytest.raises(ValueError, match="must be > start"):
        client.fetch_bars("BTCUSDT", "1m", "2026-04-29T01:00:00Z", "2026-04-29T00:00:00Z")


def test_http_error_raises():
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response({"code": -1121}, status_code=400))
    client = BinanceClient(session=session, request_interval_s=0.0)
    with pytest.raises(BinanceAPIError):
        client.fetch_latest_bar("INVALID", "1m")


def test_fetch_trades_schema():
    session = MagicMock()
    base_t = 1_700_000_000_000
    page = [_mock_aggtrade(base_t + i, agg_id=i, price=100.0 + i * 0.001) for i in range(50)]
    session.get = MagicMock(return_value=_mock_response(page))
    client = BinanceClient(session=session, request_interval_s=0.0)
    df = client.fetch_trades(
        "BTCUSDT",
        start="2023-11-14T00:00:00Z",
        end="2023-11-14T00:01:00Z",
    )
    assert len(df) == 50
    assert set(df.columns) == set(TRADES_SCHEMA.names)
    assert df["is_buyer_maker"].dtype.name == "boolean"


def test_fetch_latest_bar_single_row():
    session = MagicMock()
    session.get = MagicMock(return_value=_mock_response([_mock_klines_row(1_700_000_000_000, close=37000.0)]))
    client = BinanceClient(session=session, request_interval_s=0.0)
    df = client.fetch_latest_bar("BTCUSDT", "1m")
    assert len(df) == 1
    assert df.iloc[0]["close"] == 37000.0

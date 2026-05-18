"""price_feed.py — mock yfinance 기반 단위 테스트.

실 네트워크 호출은 test_data_live.py (별도, 옵션) 에서.
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


@pytest.fixture
def mock_yf_history():
    """yfinance Ticker.history 가짜 응답."""
    idx = pd.date_range("2026-05-01", periods=5, freq="B", tz="America/New_York")
    df = pd.DataFrame({
        "Open":  [100, 101, 102, 103, 104],
        "High":  [101, 102, 103, 104, 105],
        "Low":   [99,  100, 101, 102, 103],
        "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
        "Adj Close": [100.5, 101.5, 102.5, 103.5, 104.5],
        "Volume":[1_000_000] * 5,
    }, index=idx)
    return df


def test_get_daily_ohlcv_normalizes(tmp_path, monkeypatch, mock_yf_history):
    """yfinance 응답이 표준 컬럼명으로 변환 + tz-naive index."""
    from src.data import _common, price_feed
    monkeypatch.setattr(_common, "CACHE_DIR", tmp_path)

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_yf_history

    with patch.object(price_feed.yf, "Ticker", return_value=mock_ticker):
        df = price_feed.get_daily_ohlcv("CRCL", "2026-05-01", "2026-05-05")

    assert not df.empty
    assert set(df.columns) >= {"open", "high", "low", "close", "volume"}
    assert df.index.tz is None
    assert df["close"].iloc[0] == 100.5


def test_get_daily_ohlcv_uses_cache(tmp_path, monkeypatch, mock_yf_history):
    """두 번째 호출은 캐시에서 (yfinance 다시 호출 X)."""
    from src.data import _common, price_feed
    monkeypatch.setattr(_common, "CACHE_DIR", tmp_path)

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = mock_yf_history

    with patch.object(price_feed.yf, "Ticker", return_value=mock_ticker):
        df1 = price_feed.get_daily_ohlcv("CRCL", "2026-05-01", "2026-05-05")
        df2 = price_feed.get_daily_ohlcv("CRCL", "2026-05-01", "2026-05-05")

    # yfinance Ticker.history는 첫 호출만
    assert mock_ticker.history.call_count == 1
    assert df1.equals(df2)


def test_get_intraday_validates_interval():
    from src.data import price_feed
    with pytest.raises(ValueError):
        price_feed.get_intraday("CRCL", interval="2m", days=5)


def test_get_current_price_fallback_to_history(tmp_path, monkeypatch):
    """fast_info 실패 시 history 1m last close로 fallback."""
    from src.data import _common, price_feed
    monkeypatch.setattr(_common, "CACHE_DIR", tmp_path)

    idx = pd.date_range("2026-05-16 09:30", periods=3, freq="1min", tz="America/New_York")
    intraday_df = pd.DataFrame({
        "Open":[114, 114.5, 114.8],
        "High":[114.5, 115, 115.2],
        "Low":[113.5, 114, 114.5],
        "Close":[114.2, 114.7, 115.0],
        "Volume":[10000, 12000, 15000],
    }, index=idx)

    mock_ticker = MagicMock()
    # fast_info를 빈 dict로 — 모든 키 lookup이 None 반환 → fallback 트리거
    mock_ticker.fast_info = {}
    mock_ticker.history.return_value = intraday_df

    # cachetools cache가 활성화돼 있으면 결과를 캐싱하므로 cache 우회를 위해 새 ticker 사용
    with patch.object(price_feed.yf, "Ticker", return_value=mock_ticker):
        price = price_feed._get_current_price_uncached("MOCKSYM")

    assert price == 115.0

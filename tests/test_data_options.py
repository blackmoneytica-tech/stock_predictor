"""options_chain.py — HV / IV Rank 단위 테스트."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


def _mock_long_history(days: int = 300, start_price: float = 100.0):
    """가상의 long daily history (HV / IV Rank 계산용)."""
    idx = pd.date_range(
        end="2026-05-16", periods=days, freq="B", tz="America/New_York",
    )
    n = len(idx)
    rng = np.random.default_rng(seed=42)
    rets = rng.normal(0.0005, 0.02, n)  # ~32% annualized HV
    prices = start_price * np.exp(np.cumsum(rets))
    return pd.DataFrame({
        "Open": prices, "High": prices, "Low": prices,
        "Close": prices, "Adj Close": prices, "Volume": [1e6] * n,
    }, index=idx)


def test_historic_volatility_reasonable_range(tmp_path, monkeypatch):
    """HV가 0.1~1.0 합리적 범위에 있어야."""
    from src.data import _common, options_chain, price_feed
    monkeypatch.setattr(_common, "CACHE_DIR", tmp_path)

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_long_history(days=60)
    with patch.object(price_feed.yf, "Ticker", return_value=mock_ticker):
        hv = options_chain.get_historic_volatility("MOCK", lookback_days=30)

    assert 0.05 < hv < 2.0, f"HV out of range: {hv}"


def test_iv_rank_neutral_when_insufficient_data(tmp_path, monkeypatch):
    """30일 미만 데이터면 0.5 (중립) 반환."""
    from src.data import _common, options_chain, price_feed
    monkeypatch.setattr(_common, "CACHE_DIR", tmp_path)

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = _mock_long_history(days=5)
    with patch.object(price_feed.yf, "Ticker", return_value=mock_ticker):
        rank = options_chain.get_iv_rank("MOCK")

    assert rank == 0.5


def test_get_options_chain_structure(tmp_path, monkeypatch):
    """체인 dict가 {expiration: {strike: {...}}} 구조."""
    from src.data import _common, options_chain
    monkeypatch.setattr(_common, "CACHE_DIR", tmp_path)

    calls = pd.DataFrame({
        "strike": [100.0, 105.0, 110.0],
        "openInterest": [500, 1000, 200],
        "volume": [100, 200, 50],
        "impliedVolatility": [0.5, 0.45, 0.42],
    })
    puts = pd.DataFrame({
        "strike": [100.0, 105.0, 110.0],
        "openInterest": [300, 800, 100],
        "volume": [50, 100, 20],
        "impliedVolatility": [0.55, 0.50, 0.48],
    })

    mock_chain = MagicMock()
    mock_chain.calls = calls
    mock_chain.puts = puts

    mock_ticker = MagicMock()
    mock_ticker.option_chain.return_value = mock_chain

    with patch.object(options_chain.yf, "Ticker", return_value=mock_ticker):
        chain = options_chain.get_options_chain("CRCL", "2026-05-23")

    assert "2026-05-23" in chain
    strikes = chain["2026-05-23"]
    assert set(strikes.keys()) == {100.0, 105.0, 110.0}
    assert strikes[105.0]["call_oi"] == 1000
    assert strikes[105.0]["put_oi"] == 800
    # iv는 call/put 평균
    assert abs(strikes[105.0]["iv"] - 0.475) < 1e-6

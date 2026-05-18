"""catalyst.py — 이벤트 통합 + pre_event_rally 테스트."""
from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


def test_get_pre_event_rally_pct(tmp_path, monkeypatch):
    """30일 전 100 → 130 = +30% 누적 수익률."""
    from src.data import _common, catalyst, price_feed
    monkeypatch.setattr(_common, "CACHE_DIR", tmp_path)

    idx = pd.date_range(end="2026-05-15", periods=30, freq="B", tz="America/New_York")
    df = pd.DataFrame({
        "Open": [100] * 30, "High": [100] * 30, "Low": [100] * 30,
        "Close": list(pd.Series(range(100, 130))),
        "Adj Close": list(pd.Series(range(100, 130))),
        "Volume": [1e6] * 30,
    }, index=idx)

    mock_ticker = MagicMock()
    mock_ticker.history.return_value = df

    with patch.object(price_feed.yf, "Ticker", return_value=mock_ticker):
        rally = catalyst.get_pre_event_rally_pct("CRCL", date(2026, 5, 15))

    # 100 → 129 ≈ +29%
    assert 0.25 < rally < 0.32


def test_get_upcoming_events_includes_fomc(tmp_path, monkeypatch):
    """horizon 안에 FOMC 일정이 들어가는지."""
    from src.data import catalyst

    # 2026-05-06 FOMC가 있는데 today=2026-05-01, horizon=10이면 포함
    class FakeDate(date):
        @classmethod
        def today(cls):
            return date(2026, 5, 1)

    monkeypatch.setattr(catalyst, "date", FakeDate)
    # earnings/option 호출 무력화
    monkeypatch.setattr(catalyst, "get_next_earnings_date", lambda t: None)

    fake_ticker = MagicMock()
    fake_ticker.options = ()
    with patch.object(catalyst.yf, "Ticker", return_value=fake_ticker):
        events = catalyst.get_upcoming_events("CRCL", horizon_days=10)

    fomcs = [e for e in events if e["type"] == "FOMC"]
    assert len(fomcs) == 1
    assert fomcs[0]["date"] == date(2026, 5, 6)

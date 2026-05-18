"""Alpha Vantage options client + walk_forward 통합 테스트."""
from __future__ import annotations

import os
from datetime import date
from unittest.mock import patch


def test_av_options_no_key_raises():
    """ALPHA_VANTAGE_KEY 없으면 명확한 에러."""
    os.environ.pop("ALPHA_VANTAGE_KEY", None)
    os.environ.pop("ALPHA_VANTAGE_API_KEY", None)
    from src.data.alpha_vantage_options import fetch_historical_options_raw

    try:
        fetch_historical_options_raw("CRCL", "2026-05-01", use_cache=False)
        assert False, "should have raised"
    except RuntimeError as e:
        assert "ALPHA_VANTAGE_KEY" in str(e)


def test_av_chain_parsing():
    """raw contracts → nested chain dict 변환 검증."""
    from src.data import alpha_vantage_options as av

    mock_contracts = [
        {"expiration": "2026-05-23", "strike": "115.00", "type": "call",
         "open_interest": "500", "volume": "100", "implied_volatility": "0.85"},
        {"expiration": "2026-05-23", "strike": "115.00", "type": "put",
         "open_interest": "300", "volume": "50", "implied_volatility": "0.92"},
        {"expiration": "2026-05-23", "strike": "120.00", "type": "call",
         "open_interest": "1000", "volume": "200", "implied_volatility": "0.80"},
    ]

    with patch.object(av, "fetch_historical_options_raw", return_value=mock_contracts):
        chain = av.get_chain_at("CRCL", date(2026, 5, 16), horizon_days=5)

    assert "2026-05-23" in chain
    s115 = chain["2026-05-23"][115.0]
    assert s115["call_oi"] == 500
    assert s115["put_oi"] == 300
    # iv = avg(0.85, 0.92)
    assert abs(s115["iv"] - 0.885) < 1e-6


def test_av_chain_picks_horizon_appropriate_expiration():
    """horizon_days보다 가까운 만기는 skip하고 그 이상 첫 만기 선택."""
    from src.data import alpha_vantage_options as av

    mock = [
        {"expiration": "2026-05-18", "strike": "100.00", "type": "call",
         "open_interest": "1", "volume": "0", "implied_volatility": "0.5"},
        {"expiration": "2026-05-30", "strike": "100.00", "type": "call",
         "open_interest": "1", "volume": "0", "implied_volatility": "0.5"},
        {"expiration": "2026-06-20", "strike": "100.00", "type": "call",
         "open_interest": "1", "volume": "0", "implied_volatility": "0.5"},
    ]

    with patch.object(av, "fetch_historical_options_raw", return_value=mock):
        # 5/16 기준 5일 horizon → 5/21 이후 첫 만기 = 5/30
        chain = av.get_chain_at("X", date(2026, 5, 16), horizon_days=5)

    assert "2026-05-30" in chain
    assert "2026-05-18" not in chain


def test_walk_forward_falls_back_when_av_unavailable():
    """AV 키 없으면 build_data_at가 더미 chain으로 fallback (예외 X)."""
    os.environ.pop("ALPHA_VANTAGE_KEY", None)

    # OHLCV 호출도 mock — yfinance 호출 회피
    import pandas as pd
    idx = pd.date_range("2025-12-01", "2026-05-08", freq="B")
    fake_ohlcv = pd.DataFrame({
        "open": [100.0] * len(idx),
        "high": [105.0] * len(idx),
        "low": [95.0] * len(idx),
        "close": [100.0] * len(idx),
        "adj_close": [100.0] * len(idx),
        "volume": [1_000_000] * len(idx),
    }, index=idx)

    from src.backtest import walk_forward
    with patch.object(walk_forward, "get_daily_ohlcv", return_value=fake_ohlcv), \
         patch.object(walk_forward, "get_insider_activity", return_value={
             "insider_buys_30d": 0, "insider_sells_30d": 0,
             "insider_buys_6m": 0, "insider_sells_6m": 0,
             "recent_sells_prices": [], "recent_buys_prices": [],
         }):
        data = walk_forward.build_data_at("CRCL", date(2026, 5, 8), use_options=True)

    # 더미 chain — 정확히 3 strikes
    chain = data["options_chain"]
    exp = data["target_expiration"]
    assert len(chain[exp]) == 3, f"dummy chain should have 3 strikes, got {len(chain[exp])}"

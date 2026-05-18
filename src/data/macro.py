"""Macro data — FRED API + 매크로 발표 시계열 + 종목 베타.

FRED 주요 시리즈:
- DFF       : Fed Funds Effective Rate
- DGS10     : 10Y Treasury Yield
- DGS2      : 2Y Treasury Yield
- DTWEXBGS  : Dollar Index (Broad)
- VIXCLS    : VIX
- CPIAUCSL  : CPI
- PPIACO    : PPI (All Commodities)
- PAYEMS    : NFP
- DCOILWTICO: WTI Crude
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ._common import FRED_LIMITER, cache_load, cache_store, env, normalize_date, retry
from ._time import utcnow_naive
from .price_feed import get_daily_ohlcv


SERIES_CATALOG = {
    "DFF": "Fed Funds Rate",
    "DGS10": "10Y Treasury",
    "DGS2": "2Y Treasury",
    "DTWEXBGS": "USD Index",
    "VIXCLS": "VIX",
    "CPIAUCSL": "CPI",
    "PPIACO": "PPI",
    "PAYEMS": "NFP",
    "DCOILWTICO": "WTI",
}


# ── FRED client (lazy) ───────────────────────────────────────
_fred_client = None


def _get_fred():
    global _fred_client
    if _fred_client is not None:
        return _fred_client
    api_key = env("FRED_API_KEY")
    if not api_key:
        raise RuntimeError("FRED_API_KEY 환경변수 미설정 (.env 참조)")
    try:
        from fredapi import Fred
    except ImportError as e:
        raise RuntimeError("fredapi 미설치 — pip install fredapi") from e
    _fred_client = Fred(api_key=api_key)
    return _fred_client


# ── 시계열 fetch ──────────────────────────────────────────────
@retry(max_attempts=3, base_delay=1.0)
def get_fred_series(
    series_id: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_cache: bool = True,
) -> pd.Series:
    """FRED 시계열.

    Args:
        series_id: 'DGS10', 'CPIAUCSL' 등
        start, end: 'YYYY-MM-DD' (None이면 1년 전 ~ 오늘)
    """
    if start is None:
        start = (utcnow_naive() - timedelta(days=365)).strftime("%Y-%m-%d")
    else:
        start = normalize_date(start)
    if end is None:
        end = utcnow_naive().strftime("%Y-%m-%d")
    else:
        end = normalize_date(end)

    cache_key = f"fred:{series_id}:{start}:{end}"
    if use_cache:
        cached = cache_load("macro_fred", cache_key, ttl_seconds=86400)
        if cached is not None and not cached.empty:
            return cached.iloc[:, 0] if isinstance(cached, pd.DataFrame) else cached

    FRED_LIMITER.wait()
    fred = _get_fred()
    series = fred.get_series(series_id, observation_start=start, observation_end=end)
    series.name = series_id

    if use_cache and not series.empty:
        cache_store("macro_fred", cache_key, series.to_frame())

    return series


# ── 최근 매크로 발표 + surprise ───────────────────────────────
def get_recent_macro_releases(days_back: int = 7) -> List[Dict]:
    """최근 매크로 발표 + actual vs consensus surprise.

    MVP: FRED actual만 사용 (consensus 없음 → surprise=0 placeholder).
    실제 surprise는 Trading Economics / Investing.com calendar API 필요 (유료).

    Returns:
        [{date, series, value, surprise}, ...]
    """
    end = utcnow_naive()
    start = end - timedelta(days=days_back + 30)  # buffer

    releases = []
    for series_id in ("CPIAUCSL", "PPIACO", "PAYEMS"):
        try:
            s = get_fred_series(
                series_id,
                start.strftime("%Y-%m-%d"),
                end.strftime("%Y-%m-%d"),
            )
            if s.empty:
                continue
            # 최근 days_back 일 이내 발표만
            recent = s[s.index >= pd.Timestamp(end - timedelta(days=days_back))]
            for date, value in recent.items():
                releases.append({
                    "date": date.date(),
                    "series": series_id,
                    "value": float(value),
                    "surprise": 0.0,  # TODO: consensus 비교 통합
                })
        except Exception:
            continue

    return releases


# ── 종목 매크로 베타 (회귀 분석) ─────────────────────────────
def estimate_macro_beta(
    ticker: str,
    factor_series: str = "DGS10",
    lookback_days: int = 252,
) -> float:
    """종목 수익률을 factor 변화율에 회귀하여 베타 추정.

    beta = cov(stock, factor) / var(factor)
    """
    end = utcnow_naive()
    start = end - timedelta(days=lookback_days + 30)

    stock_df = get_daily_ohlcv(ticker, start, end)
    if stock_df.empty or len(stock_df) < 30:
        return float("nan")
    stock_ret = stock_df["close"].pct_change().dropna()

    factor = get_fred_series(
        factor_series,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )
    factor_ret = factor.pct_change().dropna()

    # 인덱스 정렬
    if factor.index.tz is not None:
        factor_ret.index = factor_ret.index.tz_localize(None)
    aligned = pd.concat([stock_ret, factor_ret], axis=1, join="inner").dropna()
    if len(aligned) < 30:
        return float("nan")

    s, f = aligned.iloc[:, 0], aligned.iloc[:, 1]
    cov = np.cov(s, f, ddof=0)[0, 1]
    var = np.var(f, ddof=0)
    return float(cov / var) if var > 0 else float("nan")

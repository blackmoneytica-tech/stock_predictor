"""Price feed — yfinance 기본 + 캐시 + rate limit + retry.

함수:
    get_daily_ohlcv(ticker, start, end) -> DataFrame
    get_intraday(ticker, interval, days) -> DataFrame
    get_current_price(ticker) -> float

캐시 정책 (config/default.yaml):
    daily OHLCV: 1시간 TTL
    intraday: 5분 TTL
    current_price: cachetools 60초 (메모리)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional, Union

import pandas as pd
import yfinance as yf

from ._common import (
    YFINANCE_LIMITER,
    cache_load,
    cache_store,
    normalize_date,
    retry,
)
from ._time import utcnow_naive

DateLike = Union[str, datetime]


# ── 일별 OHLCV ────────────────────────────────────────────────
@retry(max_attempts=3, base_delay=1.0)
def get_daily_ohlcv(
    ticker: str,
    start: DateLike,
    end: Optional[DateLike] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """일별 OHLCV (yfinance Ticker.history).

    Args:
        ticker: 'CRCL', 'AAPL' 등
        start: 'YYYY-MM-DD' or datetime
        end: 미지정 시 오늘
        use_cache: True면 1h TTL 캐시 활용

    Returns:
        columns: open, high, low, close, volume, adj_close
        index: DatetimeIndex (tz-aware UTC → naive로 변환)
    """
    start_s = normalize_date(start)
    end_s = normalize_date(end) if end else utcnow_naive().strftime("%Y-%m-%d")
    cache_key = f"daily:{ticker}:{start_s}:{end_s}"

    if use_cache:
        cached = cache_load("ohlcv_daily", cache_key, ttl_seconds=3600)
        if cached is not None:
            return cached

    YFINANCE_LIMITER.wait()
    df = yf.Ticker(ticker).history(
        start=start_s,
        end=end_s,
        interval="1d",
        auto_adjust=False,
        actions=False,
    )

    if df.empty:
        return _empty_ohlcv()

    df = _normalize_ohlcv_frame(df)
    if use_cache:
        cache_store("ohlcv_daily", cache_key, df)
    return df


# ── 인트라데이 ────────────────────────────────────────────────
@retry(max_attempts=3, base_delay=1.0)
def get_intraday(
    ticker: str,
    interval: str = "5m",
    days: int = 5,
    use_cache: bool = True,
) -> pd.DataFrame:
    """인트라데이 OHLCV.

    yfinance 제약:
        1m: 7일까지만
        5m/15m/30m/60m: 60일까지
        1h: 730일

    Args:
        interval: '1m', '5m', '15m', '30m', '60m', '1h'
        days: 룩백 일수
    """
    valid = {"1m", "5m", "15m", "30m", "60m", "1h"}
    if interval not in valid:
        raise ValueError(f"interval must be one of {valid}")

    end = utcnow_naive()
    start = end - timedelta(days=days)
    cache_key = (
        f"intraday:{ticker}:{interval}:"
        f"{start.strftime('%Y-%m-%d')}:{end.strftime('%Y-%m-%d')}"
    )

    if use_cache:
        cached = cache_load("ohlcv_intraday", cache_key, ttl_seconds=300)
        if cached is not None:
            return cached

    YFINANCE_LIMITER.wait()
    df = yf.Ticker(ticker).history(
        period=f"{days}d",
        interval=interval,
        auto_adjust=False,
        actions=False,
        prepost=True,
    )

    if df.empty:
        return _empty_ohlcv()

    df = _normalize_ohlcv_frame(df)
    if use_cache:
        cache_store("ohlcv_intraday", cache_key, df)
    return df


# ── 현재가 ────────────────────────────────────────────────────
# cachetools TTLCache — 60초 메모리 캐시 (다중 호출 절약)
try:
    from cachetools import TTLCache, cached
    _PRICE_CACHE = TTLCache(maxsize=200, ttl=60)
    _USE_TTL_CACHE = True
except ImportError:
    _USE_TTL_CACHE = False


def _get_current_price_uncached(ticker: str) -> float:
    YFINANCE_LIMITER.wait()
    t = yf.Ticker(ticker)
    price: Optional[float] = None

    # 1) fast_info (yfinance 0.2.40+ 권장 경로)
    try:
        fi = t.fast_info
        for attr in ("last_price", "regular_market_price", "lastPrice"):
            if isinstance(fi, dict):
                v = fi.get(attr)
            else:
                v = getattr(fi, attr, None)
            if v is not None and not pd.isna(v):
                price = float(v)
                break
    except Exception:
        pass

    # 2) fallback: 1d 1m 히스토리 마지막 close
    if price is None:
        hist = t.history(period="1d", interval="1m", prepost=True)
        if not hist.empty:
            price = float(hist["Close"].iloc[-1])

    if price is None or pd.isna(price):
        raise RuntimeError(f"현재가 fetch 실패: {ticker}")
    return float(price)


if _USE_TTL_CACHE:
    @cached(_PRICE_CACHE)
    @retry(max_attempts=3, base_delay=0.5)
    def get_current_price(ticker: str) -> float:
        """현재가 (60초 메모리 캐시)."""
        return _get_current_price_uncached(ticker)
else:
    @retry(max_attempts=3, base_delay=0.5)
    def get_current_price(ticker: str) -> float:
        return _get_current_price_uncached(ticker)


# ── 내부 헬퍼 ─────────────────────────────────────────────────
def _normalize_ohlcv_frame(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance DataFrame을 표준 형태로 변환."""
    df = df.rename(columns={
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })
    # tz-aware → naive (parquet 안정성)
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    # 표준 컬럼만 유지
    cols = [c for c in ["open", "high", "low", "close", "adj_close", "volume"]
            if c in df.columns]
    return df[cols].copy()


def _empty_ohlcv() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["open", "high", "low", "close", "adj_close", "volume"],
        index=pd.DatetimeIndex([], name="Date"),
    )

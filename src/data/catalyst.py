"""Catalyst calendar — Earnings + FOMC + 옵션 만기 + 종목 특이 이벤트.

MVP source:
- Earnings: yfinance .calendar / .earnings_dates
- FOMC: 정적 일정 (Fed.gov 수동 동기화)
- 옵션 만기: yfinance Ticker.options
- 종목 특이: 수동 입력 (config/events.yaml 향후 추가)

핵심: pre_event_rally_pct (sell-the-news 판정용).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf

from ._common import YFINANCE_LIMITER, cache_load, cache_store, retry
from ._time import utcnow_naive
from .price_feed import get_daily_ohlcv


# ── FOMC 정적 일정 (2026) — Fed.gov 수동 동기화 ──────────────
# 새 회의 결정 시 갱신 필요
FOMC_DATES_2026 = [
    date(2026, 1, 28),
    date(2026, 3, 18),
    date(2026, 5, 6),
    date(2026, 6, 17),
    date(2026, 7, 29),
    date(2026, 9, 16),
    date(2026, 10, 28),
    date(2026, 12, 9),
]


# ── Earnings (yfinance) ───────────────────────────────────────
@retry(max_attempts=3, base_delay=1.0)
def get_next_earnings_date(ticker: str) -> Optional[date]:
    """다음 실적 발표일.

    yfinance .earnings_dates는 과거+미래를 함께 반환. 미래 첫 항목 선택.
    """
    YFINANCE_LIMITER.wait()
    t = yf.Ticker(ticker)

    try:
        df = t.get_earnings_dates(limit=8)
    except Exception:
        df = None

    if df is None or df.empty:
        # fallback: .calendar
        try:
            cal = t.calendar
            if isinstance(cal, dict) and "Earnings Date" in cal:
                ed = cal["Earnings Date"]
                if isinstance(ed, list) and ed:
                    return ed[0] if isinstance(ed[0], date) else None
        except Exception:
            pass
        return None

    today = utcnow_naive()
    if df.index.tz is not None:
        df = df.tz_localize(None) if hasattr(df, "tz_localize") else df
        df.index = df.index.tz_localize(None)
    future = df[df.index >= today]
    if future.empty:
        return None
    return future.index[0].date() if hasattr(future.index[0], "date") else future.index[0]


def get_upcoming_events(
    ticker: str,
    horizon_days: int = 30,
) -> List[Dict]:
    """다가오는 이벤트 통합 (earnings + FOMC + 옵션 만기).

    Returns:
        [{date, type, expected_impact, expected_direction, source}, ...]
    """
    today = date.today()
    end = today + timedelta(days=horizon_days)
    events: List[Dict] = []

    # 1) Earnings
    eps_date = get_next_earnings_date(ticker)
    if eps_date and today <= eps_date <= end:
        events.append({
            "date": eps_date,
            "type": "EARNINGS",
            "expected_impact": 0.08,    # 통상 ±8% implied move
            "expected_direction": 0,    # 중립 (catalyst 모듈이 sell-news 보정)
            "source": "yfinance",
        })

    # 2) FOMC
    for fomc in FOMC_DATES_2026:
        if today <= fomc <= end:
            events.append({
                "date": fomc,
                "type": "FOMC",
                "expected_impact": 0.04,
                "expected_direction": 0,
                "source": "static",
            })

    # 3) 옵션 만기 (가장 가까운 1개)
    try:
        YFINANCE_LIMITER.wait()
        opts = yf.Ticker(ticker).options
        if opts:
            exp_s = opts[0]
            exp_d = datetime.strptime(exp_s, "%Y-%m-%d").date()
            if today <= exp_d <= end:
                events.append({
                    "date": exp_d,
                    "type": "OPTION_EXPIRATION",
                    "expected_impact": 0.02,
                    "expected_direction": 0,
                    "source": "yfinance",
                })
    except Exception:
        pass

    events.sort(key=lambda e: e["date"])
    return events


# ── Pre-event 랠리 측정 (sell-news 판정) ─────────────────────
def get_pre_event_rally_pct(
    ticker: str,
    event_date: date,
    lookback_days: int = 30,
) -> float:
    """event_date 직전 lookback_days 누적 수익률.

    +0.30 이상이면 sell-the-news 80% base rate 트리거.
    """
    end_dt = datetime.combine(event_date, datetime.min.time())
    start_dt = end_dt - timedelta(days=lookback_days + 5)
    df = get_daily_ohlcv(ticker, start_dt, end_dt)
    if df.empty or len(df) < 5:
        return 0.0

    closes = df["close"].dropna()
    if len(closes) < 2:
        return 0.0
    return float((closes.iloc[-1] - closes.iloc[0]) / closes.iloc[0])

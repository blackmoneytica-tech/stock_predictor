"""position_sizing.py 보조 — macro 카테고리/RS grade 계산.

trade-journal "매크로 탭"이 사용하는 룰을 시스템 내부에서 재현 (백테스트 검증):
  - drawdown (52w high 기준): strong_buy ≤ -20%, deep -20~-15%,
    trap -15~-10%, buy_zone -10~-3%, safe > -3%
  - RS grade (SPY 대비 20일 초과수익): very_strong ≥ +5%, strong 0~+5%,
    weak -5~0%, very_weak < -5%
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Optional, Tuple

import pandas as pd


def classify_macro_cat_rs(
    ticker: str,
    current_price: float,
    ohlcv: Optional[pd.DataFrame],
    as_of_date: Optional[date] = None,
) -> Tuple[str, str]:
    """(cat, rs_grade) 반환. 실패 시 ('', '')."""
    if ohlcv is None or ohlcv.empty or not current_price:
        return "", ""

    if as_of_date:
        ts = pd.Timestamp(as_of_date)
        hist = ohlcv[ohlcv.index <= ts]
    else:
        hist = ohlcv

    if len(hist) < 22:
        return "", ""

    # 52주 high
    wk52 = hist.tail(252)
    wk52_high = float(wk52["high"].max()) if "high" in wk52.columns else float(wk52["close"].max())
    if wk52_high <= 0:
        return "", ""
    dd_pct = (current_price - wk52_high) / wk52_high * 100

    # SPY 대비 20일 상대강도
    try:
        from ..data.price_feed import get_daily_ohlcv
        start = hist.index[-1].date() - timedelta(days=40)
        end = hist.index[-1].date() + timedelta(days=1)
        spy = get_daily_ohlcv("SPY", start, end)
        if not spy.empty and len(hist) >= 21 and len(spy) >= 21:
            t20 = float(hist["close"].iloc[-21])
            t_chg20 = (current_price - t20) / t20 * 100
            spy_cur = float(spy["close"].iloc[-1])
            spy_20 = float(spy["close"].iloc[-21])
            spy_chg20 = (spy_cur - spy_20) / spy_20 * 100
            rel_chg20 = t_chg20 - spy_chg20
        else:
            rel_chg20 = 0
    except Exception:
        rel_chg20 = 0

    # Category
    if dd_pct <= -20:
        cat = "strong_buy"
    elif dd_pct <= -15:
        cat = "deep"
    elif dd_pct <= -10:
        cat = "trap"
    elif dd_pct <= -3:
        cat = "buy_zone"
    else:
        cat = "safe"

    # RS grade
    if rel_chg20 >= 5:
        rs_grade = "very_strong"
    elif rel_chg20 >= 0:
        rs_grade = "strong"
    elif rel_chg20 >= -5:
        rs_grade = "weak"
    else:
        rs_grade = "very_weak"

    return cat, rs_grade

"""Finnhub API client — EPS estimates + 다음 earnings 발표일.

용도:
- 다음 earnings 발표일 정확하게 (yfinance보다 reliable)
- EPS estimate trend (rising/falling) — beat 확률 사전 시그널
- 키 발급: https://finnhub.io/register (이메일만, 무료 60 req/min)

free tier 한계:
- /stock/earnings: 무료 (분기별 actual + estimate)
- /stock/recommendation: 무료 (analyst trend)
- /calendar/earnings: 무료 (전체 시장 calendar)

설정: .env FINNHUB_KEY=...
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import requests

from ._common import RateLimiter, cache_load, cache_store, env, retry

FINNHUB_BASE = "https://finnhub.io/api/v1"
# Free tier: 60 req/min
FINNHUB_LIMITER = RateLimiter(max_per_minute=60)


def _api_key() -> str:
    key = env("FINNHUB_KEY") or env("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError(
            "FINNHUB_KEY 미설정. 무료 발급: https://finnhub.io/register"
        )
    return key


# ── EPS estimates + actuals ─────────────────────────────────
@retry(max_attempts=3, base_delay=1.0)
def get_earnings_history(ticker: str, use_cache: bool = True) -> List[Dict]:
    """분기별 actual + estimate EPS.

    Returns:
        [{period (date), actual, estimate, surprise, surprisePercent}, ...]
        최근 발표 순.
    """
    ticker = ticker.upper()
    cache_key = f"finnhub_earnings:{ticker}"
    if use_cache:
        cached = cache_load("finnhub_earnings", cache_key, ttl_seconds=86400)
        if cached is not None and not cached.empty:
            return cached.to_dict(orient="records")

    FINNHUB_LIMITER.wait()
    r = requests.get(
        f"{FINNHUB_BASE}/stock/earnings",
        params={"symbol": ticker, "token": _api_key()},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return []

    if data and use_cache:
        import pandas as pd
        cache_store("finnhub_earnings", cache_key, pd.DataFrame(data))
    return data


@retry(max_attempts=3, base_delay=1.0)
def get_earnings_calendar(
    ticker: str,
    from_date: date,
    to_date: date,
    use_cache: bool = True,
) -> List[Dict]:
    """earnings calendar — 다가오는 발표일 + estimate."""
    ticker = ticker.upper()
    cache_key = f"finnhub_cal:{ticker}:{from_date.isoformat()}:{to_date.isoformat()}"
    if use_cache:
        cached = cache_load("finnhub_cal", cache_key, ttl_seconds=43200)
        if cached is not None and not cached.empty:
            return cached.to_dict(orient="records")

    FINNHUB_LIMITER.wait()
    r = requests.get(
        f"{FINNHUB_BASE}/calendar/earnings",
        params={
            "symbol": ticker,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "token": _api_key(),
        },
        timeout=15,
    )
    r.raise_for_status()
    data = r.json().get("earningsCalendar", [])
    if data and use_cache:
        import pandas as pd
        cache_store("finnhub_cal", cache_key, pd.DataFrame(data))
    return data


# ── Analyst recommendation trend ─────────────────────────────
@retry(max_attempts=3, base_delay=1.0)
def get_recommendation_trend(ticker: str, use_cache: bool = True) -> List[Dict]:
    """analyst buy/hold/sell 추천 trend (월별).

    Returns:
        [{period (YYYY-MM-DD), strongBuy, buy, hold, sell, strongSell}, ...]
    """
    ticker = ticker.upper()
    cache_key = f"finnhub_rec:{ticker}"
    if use_cache:
        cached = cache_load("finnhub_rec", cache_key, ttl_seconds=86400)
        if cached is not None and not cached.empty:
            return cached.to_dict(orient="records")

    FINNHUB_LIMITER.wait()
    r = requests.get(
        f"{FINNHUB_BASE}/stock/recommendation",
        params={"symbol": ticker, "token": _api_key()},
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return []

    if data and use_cache:
        import pandas as pd
        cache_store("finnhub_rec", cache_key, pd.DataFrame(data))
    return data


# ── 통합 시그널 ────────────────────────────────────────────
def get_earnings_signals(ticker: str, as_of: date) -> Dict:
    """종합 earnings 시그널 — as_of 시점 기준.

    Returns:
        {
            next_earnings_date: date or None,
            days_to_earnings: int,
            past_beat_rate: float,         # 최근 4 분기 beat 비율
            past_surprise_pct_avg: float,  # 최근 4 분기 평균 surprise %
            estimate_trend: str,            # 'rising' / 'falling' / 'flat'
            analyst_sentiment: float,       # -1~+1 (strong sell ~ strong buy)
            beat_probability_proxy: float,  # 0~1 (heuristic)
        }
    """
    out = {
        "next_earnings_date": None,
        "days_to_earnings": 999,
        "past_beat_rate": 0.5,
        "past_surprise_pct_avg": 0.0,
        "estimate_trend": "flat",
        "analyst_sentiment": 0.0,
        "beat_probability_proxy": 0.5,
    }

    if not env("FINNHUB_KEY") and not env("FINNHUB_API_KEY"):
        return out

    # 1) 과거 4 분기 beat rate
    try:
        history = get_earnings_history(ticker)
        # as_of 이전 발표만 (lookahead 방지)
        past = []
        for h in history:
            try:
                p_date = datetime.fromisoformat(h["period"]).date()
                if p_date < as_of:
                    past.append(h)
            except (KeyError, ValueError):
                continue
        past = past[:4]  # 최근 4
        if past:
            beats = sum(1 for h in past if (h.get("surprise") or 0) > 0)
            out["past_beat_rate"] = beats / len(past)
            avg_surp_pct = sum((h.get("surprisePercent") or 0) for h in past) / len(past)
            out["past_surprise_pct_avg"] = avg_surp_pct
    except Exception:
        pass

    # 2) 다음 발표일
    try:
        cal = get_earnings_calendar(
            ticker,
            as_of,
            as_of + timedelta(days=120),
        )
        future = []
        for c in cal:
            try:
                cd = datetime.fromisoformat(c["date"]).date()
                if cd >= as_of:
                    future.append((cd, c))
            except (KeyError, ValueError):
                continue
        if future:
            future.sort(key=lambda x: x[0])
            next_date, next_c = future[0]
            out["next_earnings_date"] = next_date.isoformat()
            out["days_to_earnings"] = (next_date - as_of).days
    except Exception:
        pass

    # 3) Analyst recommendation
    try:
        recs = get_recommendation_trend(ticker)
        # as_of 이전 가장 최근 month
        past_recs = []
        for r in recs:
            try:
                p_date = datetime.fromisoformat(r["period"]).date()
                if p_date <= as_of:
                    past_recs.append((p_date, r))
            except (KeyError, ValueError):
                continue
        if past_recs:
            past_recs.sort(key=lambda x: -x[0].toordinal())
            latest = past_recs[0][1]
            total = sum(latest.get(k, 0) for k in
                        ("strongBuy", "buy", "hold", "sell", "strongSell"))
            if total > 0:
                weighted = (
                    latest.get("strongBuy", 0) * 1.0
                    + latest.get("buy", 0) * 0.5
                    + latest.get("hold", 0) * 0.0
                    + latest.get("sell", 0) * -0.5
                    + latest.get("strongSell", 0) * -1.0
                ) / total
                out["analyst_sentiment"] = weighted
    except Exception:
        pass

    # 4) Beat probability proxy
    # heuristic: past_beat_rate × 0.6 + analyst_sentiment × 0.3 + (rising estimates) × 0.1
    proxy = (
        out["past_beat_rate"] * 0.6
        + (out["analyst_sentiment"] + 1) / 2 * 0.3
        + 0.1  # estimate_trend는 별도 fetch 필요 — flat 가정
    )
    out["beat_probability_proxy"] = float(min(1.0, max(0.0, proxy)))

    return out

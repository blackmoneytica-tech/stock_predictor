"""News sentiment — Finnhub /company-news + 키워드 기반 score.

alert/worker.js의 NEWS_NEGATIVE/POSITIVE_KEYWORDS 그대로 이식.
무료 Finnhub 60 req/min.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List

import requests

from ._common import RateLimiter, cache_load, cache_store, env, retry

FH_BASE = "https://finnhub.io/api/v1"
FH_LIMITER = RateLimiter(max_per_minute=60)


# alert/worker.js NEWS_NEGATIVE_KEYWORDS 이식
NEGATIVE_KEYWORDS = [
    'lawsuit', 'sec investigation', 'sec probe', 'fraud', 'investigation',
    'bankruptcy', 'delisting', 'going concern', 'restated', 'restatement',
    'short report', 'short seller', 'hindenburg', 'muddy waters',
    'fbi', 'doj', 'subpoena', 'whistleblower', 'lay off', 'layoff',
    'downgrade', 'cuts guidance', 'cuts forecast', 'recall', 'breach',
    'data breach', 'hack', 'cyber attack', 'falsified', 'misleading',
    'plunges', 'tumbles', 'crashes', 'slumps', 'misses estimates',
]
POSITIVE_KEYWORDS = [
    'beats', 'beats estimates', 'beats expectations', 'raises guidance',
    'upgrade', 'partnership', 'acquired', 'acquires', 'expands',
    'new deal', 'wins contract', 'awarded', 'launches', 'breakthrough',
    'approval', 'approved', 'fda approval', 'patent', 'breakthrough',
    'surges', 'rallies', 'soars', 'jumps', 'record high',
    'analyst raise', 'price target raised', 'overweight', 'buy rating',
]


def _api_key() -> str:
    key = env("FINNHUB_KEY") or env("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_KEY 미설정")
    return key


@retry(max_attempts=3, base_delay=1.0)
def fetch_company_news(
    symbol: str,
    from_date: date,
    to_date: date,
    use_cache: bool = True,
) -> List[Dict]:
    """Finnhub /company-news."""
    symbol = symbol.upper()
    cache_key = f"news:{symbol}:{from_date.isoformat()}:{to_date.isoformat()}"
    if use_cache:
        cached = cache_load("news", cache_key, ttl_seconds=7200)  # 2h
        if cached is not None and not cached.empty:
            return cached.to_dict(orient="records")

    FH_LIMITER.wait()
    r = requests.get(
        f"{FH_BASE}/company-news",
        params={
            "symbol": symbol,
            "from": from_date.isoformat(),
            "to": to_date.isoformat(),
            "token": _api_key(),
        },
        timeout=15,
    )
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, list):
        return []
    if data and use_cache:
        import pandas as pd
        cache_store("news", cache_key, pd.DataFrame(data))
    return data


def compute_sentiment(news_items: List[Dict]) -> Dict:
    """헤드라인 + summary 키워드 점수 + 가중 평균.

    Returns:
        {
            score: -10 ~ +10 (정규화),
            n_items: 갯수,
            n_negative: 부정 헤드라인,
            n_positive: 긍정 헤드라인,
            top_negative: [...],
            top_positive: [...],
        }
    """
    if not news_items:
        return {
            "score": 0.0, "n_items": 0,
            "n_negative": 0, "n_positive": 0,
            "top_negative": [], "top_positive": [],
        }

    weighted_score = 0.0
    n_neg = 0
    n_pos = 0
    top_neg, top_pos = [], []

    # 최근 헤드라인일수록 가중 (exponential decay)
    now_ts = datetime.now().timestamp()
    for item in news_items:
        ts = item.get("datetime", 0)
        try:
            ts = float(ts)
        except (TypeError, ValueError):
            ts = 0
        # 최근 24시간 내 = 1.0 weight, 7일 전 = 0.3
        age_hours = max(0, (now_ts - ts) / 3600)
        recency_w = max(0.3, 1.0 - age_hours / 168)  # 168h = 7d

        text = (
            (item.get("headline") or "")
            + " " + (item.get("summary") or "")
        ).lower()

        neg_hits = sum(1 for k in NEGATIVE_KEYWORDS if k in text)
        pos_hits = sum(1 for k in POSITIVE_KEYWORDS if k in text)

        # 항목 점수 = (pos - 2*neg) (부정에 더 큰 weight)
        item_score = pos_hits - 2 * neg_hits
        weighted_score += item_score * recency_w

        if neg_hits > pos_hits:
            n_neg += 1
            if len(top_neg) < 3:
                top_neg.append({
                    "headline": item.get("headline", "")[:120],
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                })
        elif pos_hits > neg_hits:
            n_pos += 1
            if len(top_pos) < 3:
                top_pos.append({
                    "headline": item.get("headline", "")[:120],
                    "source": item.get("source", ""),
                    "url": item.get("url", ""),
                })

    # -10 ~ +10 정규화 (5 = 강한 시그널)
    score = max(-10, min(10, weighted_score))
    return {
        "score": float(score),
        "n_items": len(news_items),
        "n_negative": n_neg,
        "n_positive": n_pos,
        "top_negative": top_neg,
        "top_positive": top_pos,
    }


def get_news_sentiment(symbol: str, days_back: int = 7) -> Dict:
    """편의 함수 — 최근 N일 sentiment."""
    if not env("FINNHUB_KEY") and not env("FINNHUB_API_KEY"):
        return {"score": 0.0, "n_items": 0, "n_negative": 0, "n_positive": 0,
                "top_negative": [], "top_positive": []}
    to_d = date.today()
    from_d = to_d - timedelta(days=days_back)
    try:
        items = fetch_company_news(symbol, from_d, to_d)
        return compute_sentiment(items)
    except Exception as e:
        return {"score": 0.0, "n_items": 0, "n_negative": 0, "n_positive": 0,
                "top_negative": [], "top_positive": [], "error": str(e)}

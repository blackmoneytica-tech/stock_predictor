"""Alpha Vantage HISTORICAL_OPTIONS — 무료 historical 옵션 체인.

핵심: 무료 키로 historical 옵션 chain (strike/IV/OI/Greeks) 받을 수 있는 유일한 API.

제한:
- Free tier: 25 req/day, 5 req/min
- date 파라미터로 특정 영업일 체인 받음 (그 날 EOD)
- demo 키는 IBM only + date 없을 때만 동작

발급: https://www.alphavantage.co/support/#api-key (이메일 입력 20초)
.env 에 ALPHA_VANTAGE_KEY 설정.

반환 필드 (모든 contract에 대해):
- contractID, symbol, expiration, strike, type (call/put)
- last, mark, bid, ask, bid_size, ask_size
- volume, open_interest
- date (조회 날짜)
- implied_volatility
- delta, gamma, theta, vega, rho
"""
from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

from ._common import (
    RateLimiter,
    cache_load,
    cache_store,
    env,
    normalize_date,
    retry,
)


AV_BASE = "https://www.alphavantage.co/query"
# Free tier: 5 req/min, 25 req/day
AV_LIMITER = RateLimiter(max_per_minute=5)


def _api_key() -> str:
    key = env("ALPHA_VANTAGE_KEY") or env("ALPHA_VANTAGE_API_KEY")
    if not key:
        raise RuntimeError(
            "ALPHA_VANTAGE_KEY 미설정. 무료 키 발급: "
            "https://www.alphavantage.co/support/#api-key"
        )
    return key


@retry(max_attempts=3, base_delay=2.0)
def fetch_historical_options_raw(
    symbol: str,
    date_str: Optional[str] = None,
    use_cache: bool = True,
) -> List[Dict]:
    """단일 날짜의 모든 옵션 contract (raw).

    Args:
        date_str: 'YYYY-MM-DD'. None이면 가장 최근 EOD.

    Returns:
        contract dict 리스트 (strike/type/IV/OI/Greeks).
    """
    symbol = symbol.upper()
    cache_key = f"av_opt:{symbol}:{date_str or 'latest'}"
    if use_cache:
        cached = cache_load("av_options", cache_key, ttl_seconds=7 * 86400)
        if cached is not None and not cached.empty:
            return cached.to_dict(orient="records")

    AV_LIMITER.wait()
    params = {
        "function": "HISTORICAL_OPTIONS",
        "symbol": symbol,
        "apikey": _api_key(),
    }
    if date_str:
        params["date"] = normalize_date(date_str)

    r = requests.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    body = r.json()

    if "Information" in body and "data" not in body:
        # rate limit / 25/day 초과 / demo 제한
        msg = body["Information"]
        if "demo" in msg.lower() or "premium" in msg.lower():
            raise RuntimeError(f"AV 제한: {msg[:200]}")
        # 일일 한도 초과 등 — 빈 리스트 반환 (조용한 degrade)
        return []

    if "Note" in body:
        # rate limit hit
        time.sleep(15)
        raise RuntimeError(f"AV rate limited: {body['Note'][:120]}")

    data = body.get("data", [])
    if data and use_cache:
        cache_store("av_options", cache_key, pd.DataFrame(data))
    return data


# ── 실시간 (15분 지연) ───────────────────────────────────────
@retry(max_attempts=3, base_delay=2.0)
def fetch_realtime_options_raw(
    symbol: str,
    require_greeks: bool = False,
    use_cache: bool = True,
) -> List[Dict]:
    """현재(15분 지연) 옵션 chain.

    REALTIME_OPTIONS는 entitlement='delayed' 무료 (15분 지연).
    require_greeks=True면 'entitlement=realtime' 시도 → 유료 필요.
    """
    symbol = symbol.upper()
    cache_key = f"av_rt_opt:{symbol}:{int(require_greeks)}"
    if use_cache:
        # 실시간이지만 15분 지연이므로 cache TTL 15분
        cached = cache_load("av_realtime_options", cache_key, ttl_seconds=900)
        if cached is not None and not cached.empty:
            return cached.to_dict(orient="records")

    AV_LIMITER.wait()
    params = {
        "function": "REALTIME_OPTIONS",
        "symbol": symbol,
        "apikey": _api_key(),
    }
    if require_greeks:
        params["require_greeks"] = "true"

    r = requests.get(AV_BASE, params=params, timeout=30)
    r.raise_for_status()
    body = r.json()

    if "Information" in body and "data" not in body:
        msg = body["Information"]
        raise RuntimeError(f"AV REALTIME_OPTIONS 제한: {msg[:200]}")

    if "Note" in body:
        time.sleep(15)
        raise RuntimeError(f"AV rate limited: {body['Note'][:120]}")

    data = body.get("data", [])
    if data and use_cache:
        cache_store("av_realtime_options", cache_key, pd.DataFrame(data))
    return data


def get_realtime_chain(
    symbol: str,
    horizon_days: int = 5,
) -> Dict:
    """현재 시점 옵션 체인 (15분 지연).

    horizon_days에 가장 가까운 (≥) 만기 선택 후 nested dict 반환.
    실패 시 빈 dict.
    """
    from datetime import datetime, timedelta

    contracts = fetch_realtime_options_raw(symbol)
    if not contracts:
        return {}

    target = (datetime.utcnow() + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
    exps = sorted({c["expiration"] for c in contracts if c.get("expiration")})
    future = [e for e in exps if e >= target]
    chosen = future[0] if future else (exps[-1] if exps else None)
    if chosen is None:
        return {}

    by_strike: Dict[float, Dict] = {}
    for c in contracts:
        if c.get("expiration") != chosen:
            continue
        try:
            strike = float(c["strike"])
        except (KeyError, TypeError, ValueError):
            continue
        side = c.get("type", "").lower()
        if side not in ("call", "put"):
            continue
        slot = by_strike.setdefault(strike, {
            "call_oi": 0, "put_oi": 0,
            "call_volume": 0, "put_volume": 0,
            "call_iv": float("nan"), "put_iv": float("nan"),
        })
        oi = _to_int(c.get("open_interest"))
        vol = _to_int(c.get("volume"))
        iv = _to_float(c.get("implied_volatility"))
        if side == "call":
            slot["call_oi"] = oi
            slot["call_volume"] = vol
            slot["call_iv"] = iv
        else:
            slot["put_oi"] = oi
            slot["put_volume"] = vol
            slot["put_iv"] = iv

    for slot in by_strike.values():
        ivs = [v for v in (slot["call_iv"], slot["put_iv"]) if not pd.isna(v)]
        slot["iv"] = sum(ivs) / len(ivs) if ivs else float("nan")

    return {chosen: by_strike}


def get_chain_at(
    symbol: str,
    as_of: date,
    horizon_days: int = 5,
) -> Dict:
    """walk-forward 백테스트용 — as_of 시점 옵션 체인.

    horizon_days에 가장 가까운 (≥) 만기 선택 후 nested dict 반환:
        {expiration: {strike: {call_oi, put_oi, call_iv, put_iv, iv, ...}}}

    실패 시 빈 dict 반환 (caller가 더미 chain으로 fallback).
    """
    date_str = normalize_date(as_of)
    contracts = fetch_historical_options_raw(symbol, date_str)
    if not contracts:
        return {}

    # 만기 선택
    target = as_of + timedelta(days=horizon_days)
    exps = sorted({c["expiration"] for c in contracts if c.get("expiration")})
    future_exps = [e for e in exps if e >= target.strftime("%Y-%m-%d")]
    if not future_exps:
        # horizon보다 가까운 만기만 있으면 가장 가까운 것
        if not exps:
            return {}
        chosen = exps[-1]
    else:
        chosen = future_exps[0]

    # strike별 call/put 통합
    by_strike: Dict[float, Dict] = {}
    for c in contracts:
        if c.get("expiration") != chosen:
            continue
        try:
            strike = float(c["strike"])
        except (KeyError, TypeError, ValueError):
            continue
        side = c.get("type", "").lower()
        if side not in ("call", "put"):
            continue
        slot = by_strike.setdefault(strike, {
            "call_oi": 0, "put_oi": 0,
            "call_volume": 0, "put_volume": 0,
            "call_iv": float("nan"), "put_iv": float("nan"),
        })
        oi = _to_int(c.get("open_interest"))
        vol = _to_int(c.get("volume"))
        iv = _to_float(c.get("implied_volatility"))
        if side == "call":
            slot["call_oi"] = oi
            slot["call_volume"] = vol
            slot["call_iv"] = iv
        else:
            slot["put_oi"] = oi
            slot["put_volume"] = vol
            slot["put_iv"] = iv

    # 평균 IV
    for slot in by_strike.values():
        ivs = [v for v in (slot["call_iv"], slot["put_iv"]) if not pd.isna(v)]
        slot["iv"] = sum(ivs) / len(ivs) if ivs else float("nan")

    return {chosen: by_strike}


def _to_int(v) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return 0


def _to_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


# ── ATM IV 추출 헬퍼 (백테스트용) ───────────────────────────
def get_atm_iv_at(symbol: str, as_of: date, current_price: float) -> float:
    """as_of 시점의 가장 가까운 만기 ATM IV."""
    chain = get_chain_at(symbol, as_of, horizon_days=5)
    if not chain:
        return float("nan")
    exp, strikes = next(iter(chain.items()))
    if not strikes:
        return float("nan")
    atm = min(strikes.keys(), key=lambda s: abs(s - current_price))
    iv = strikes[atm].get("iv", float("nan"))
    return iv if not pd.isna(iv) else float("nan")

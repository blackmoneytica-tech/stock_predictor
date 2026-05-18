"""Marketdata.app options chain — yfinance가 OI/IV placeholder 반환할 때 대체.

장점:
- 무료 100 req/day, 이메일 가입 (https://www.marketdata.app/)
- options chain 1 req = 단일 만기 전체 strikes (OI/IV/Greeks 포함)
- yfinance와 달리 장 closed/주말에도 마지막 정확한 EOD OI/IV 제공
- 한국 거주자 가입 가능 (OPRA non-pro agreement 무료)

설정: .env에 MARKETDATA_KEY=<your_token>

엔드포인트:
  GET https://api.marketdata.app/v1/options/chain/{symbol}/
    ?expiration=YYYY-MM-DD
    &token=<key>

응답 (column arrays):
  optionSymbol, expiration, strike, side (call/put),
  bid, ask, mid, last, volume, openInterest,
  iv, delta, gamma, theta, vega, intrinsicValue, extrinsicValue, inTheMoney
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
    retry,
)

MD_BASE = "https://api.marketdata.app/v1"
# 100 req/day = 분당 1.5 conservative
MD_LIMITER = RateLimiter(max_per_minute=15)


def _api_key() -> str:
    key = env("MARKETDATA_KEY") or env("MARKETDATA_API_KEY")
    if not key:
        raise RuntimeError(
            "MARKETDATA_KEY 미설정. 무료 가입: https://www.marketdata.app/ "
            "(이메일만, 100 req/day)"
        )
    return key


@retry(max_attempts=3, base_delay=2.0)
def fetch_chain_raw(
    symbol: str,
    expiration: Optional[str] = None,
    use_cache: bool = True,
) -> List[Dict]:
    """단일 만기의 옵션 체인 raw (call+put).

    Args:
        symbol: 'CRCL', 'NVDA' 등
        expiration: 'YYYY-MM-DD' (None이면 'all' — 무료 한도 빠르게 소진)
    """
    symbol = symbol.upper()
    cache_key = f"md_chain:{symbol}:{expiration or 'all'}"
    if use_cache:
        # 6시간 TTL — API budget 100/day 보호 (매시간 cron + 워치 20종 가능)
        cached = cache_load("md_options", cache_key, ttl_seconds=21600)
        if cached is not None and not cached.empty:
            return cached.to_dict(orient="records")

    MD_LIMITER.wait()
    url = f"{MD_BASE}/options/chain/{symbol}/"
    params = {"token": _api_key()}
    if expiration:
        params["expiration"] = expiration

    r = requests.get(url, params=params, timeout=30)
    if r.status_code == 429:
        time.sleep(30)
        raise RuntimeError("MD rate limited (429)")
    r.raise_for_status()
    body = r.json()

    if body.get("s") != "ok":
        # status error
        err = body.get("errmsg") or str(body)[:200]
        raise RuntimeError(f"MD API error: {err}")

    # column-array → row-dict 변환
    n = len(body.get("strike", []))
    rows = []
    keys = [
        "optionSymbol", "expiration", "strike", "side",
        "bid", "ask", "mid", "last", "volume", "openInterest",
        "iv", "delta", "gamma", "theta", "vega",
        "intrinsicValue", "extrinsicValue", "inTheMoney",
    ]
    for i in range(n):
        row = {}
        for k in keys:
            arr = body.get(k, [])
            row[k] = arr[i] if i < len(arr) else None
        rows.append(row)

    if rows and use_cache:
        cache_store("md_options", cache_key, pd.DataFrame(rows))
    return rows


def get_chain(
    symbol: str,
    expiration: str,
    use_cache: bool = True,
) -> Dict:
    """단일 만기 → nested dict {expiration: {strike: {call_oi, put_oi, ...}}}.

    yfinance get_options_chain과 동일 schema.
    """
    rows = fetch_chain_raw(symbol, expiration, use_cache=use_cache)
    if not rows:
        return {}

    by_strike: Dict[float, Dict] = {}
    for r in rows:
        try:
            strike = float(r["strike"])
        except (KeyError, TypeError, ValueError):
            continue
        side = (r.get("side") or "").lower()
        if side not in ("call", "put"):
            continue
        slot = by_strike.setdefault(strike, {
            "call_oi": 0, "put_oi": 0,
            "call_volume": 0, "put_volume": 0,
            "call_iv": float("nan"), "put_iv": float("nan"),
        })
        oi = _to_int(r.get("openInterest"))
        vol = _to_int(r.get("volume"))
        iv = _to_float(r.get("iv"))
        if side == "call":
            slot["call_oi"] = oi
            slot["call_volume"] = vol
            slot["call_iv"] = iv
            slot["call_delta"] = _to_float(r.get("delta"))
            slot["call_gamma"] = _to_float(r.get("gamma"))
        else:
            slot["put_oi"] = oi
            slot["put_volume"] = vol
            slot["put_iv"] = iv
            slot["put_delta"] = _to_float(r.get("delta"))
            slot["put_gamma"] = _to_float(r.get("gamma"))

    # 평균 IV
    for slot in by_strike.values():
        ivs = [v for v in (slot["call_iv"], slot["put_iv"])
               if not pd.isna(v) and v > 0.01]
        slot["iv"] = sum(ivs) / len(ivs) if ivs else float("nan")

    return {expiration: by_strike}


@retry(max_attempts=3, base_delay=1.0)
def list_expirations(symbol: str) -> List[str]:
    """사용 가능한 만기 리스트 (1 req)."""
    symbol = symbol.upper()
    cache_key = f"md_exps:{symbol}"
    cached = cache_load("md_exps", cache_key, ttl_seconds=86400)
    if cached is not None and not cached.empty:
        return cached["expiration"].tolist()

    MD_LIMITER.wait()
    url = f"{MD_BASE}/options/expirations/{symbol}/"
    r = requests.get(url, params={"token": _api_key()}, timeout=15)
    r.raise_for_status()
    body = r.json()
    if body.get("s") != "ok":
        return []
    exps = body.get("expirations", [])
    if exps:
        cache_store("md_exps", cache_key, pd.DataFrame({"expiration": exps}))
    return exps


def pick_horizon_expiration(symbol: str, horizon_days: int = 5) -> Optional[str]:
    exps = list_expirations(symbol)
    if not exps:
        return None
    from datetime import datetime, timedelta
    target = (datetime.utcnow() + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
    future = [e for e in exps if e >= target]
    return future[0] if future else exps[-1]


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

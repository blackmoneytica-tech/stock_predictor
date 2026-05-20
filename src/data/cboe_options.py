"""CBOE Options Chain — 무료, 한국 IP OK, 가입 불필요, 옵션 거래소 본가 데이터.

엔드포인트: https://cdn.cboe.com/api/global/delayed_quotes/options/{SYMBOL}.json
- 15분 지연 실시간 (장 close 후엔 EOD)
- 풀 OI / Volume / IV / Greeks (delta/gamma/vega/theta) / bid/ask
- 모든 만기 + 모든 strike 한 번에 (수천 개)

응답 구조:
  {timestamp, symbol, data: {
    current_price, bid, ask, open, high, low, close,
    options: [{
      option: "MSTR260522C00030000",  # SYMBOL YYMMDD C/P STRIKEx1000(8자리)
      bid, ask, bid_size, ask_size,
      iv, open_interest, volume, delta, gamma, vega, theta, rho,
      last_trade_price, last_trade_time, ...
    }, ...]
  }}
"""
from __future__ import annotations

import re
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests

from ._common import RateLimiter, cache_load, cache_store, retry

CBOE_BASE = "https://cdn.cboe.com/api/global/delayed_quotes/options"
# Rate-limit 자율 (CBOE는 명시 안 함, 보수적으로 분당 20)
CBOE_LIMITER = RateLimiter(max_per_minute=20)

# option symbol regex: SYMBOL YYMMDD C|P STRIKE(8자리 = strike × 1000)
_OPT_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


def _parse_option_symbol(opt: str):
    """MSTR260522C00030000 → (ticker='MSTR', exp='2026-05-22', side='call', strike=30.0)."""
    m = _OPT_RE.match(opt)
    if not m:
        return None
    ticker, ymd, cp, strike_raw = m.groups()
    try:
        yr, mo, dy = int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6])
        # YY → 20YY (2000년대)
        yr += 2000 if yr < 70 else 1900
        exp_date = date(yr, mo, dy).isoformat()
    except Exception:
        return None
    side = "call" if cp == "C" else "put"
    strike = int(strike_raw) / 1000.0
    return ticker, exp_date, side, strike


@retry(max_attempts=3, base_delay=1.5)
def fetch_chain_raw(symbol: str, use_cache: bool = True) -> Dict:
    """CBOE 전체 옵션 chain JSON (모든 만기 포함)."""
    symbol = symbol.upper()
    cache_key = f"cboe_chain:{symbol}"
    if use_cache:
        cached = cache_load("cboe_options", cache_key, ttl_seconds=21600)  # 6시간
        if cached is not None and not cached.empty:
            return cached.to_dict(orient="records")

    CBOE_LIMITER.wait()
    url = f"{CBOE_BASE}/{symbol}.json"
    r = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; stock_predictor/2.0)"},
        timeout=20,
    )
    if r.status_code == 429:
        time.sleep(30)
        raise RuntimeError("CBOE rate limited (429)")
    r.raise_for_status()
    payload = r.json()
    data = payload.get("data") or {}
    opts = data.get("options") or []
    current_price = data.get("current_price")

    rows = []
    for o in opts:
        sym = o.get("option") or ""
        parsed = _parse_option_symbol(sym)
        if not parsed:
            continue
        _, exp_date, side, strike = parsed
        rows.append({
            "expiration": exp_date,
            "strike": strike,
            "side": side,
            "bid": o.get("bid", 0),
            "ask": o.get("ask", 0),
            "openInterest": o.get("open_interest", 0),
            "volume": o.get("volume", 0),
            "iv": o.get("iv", 0),
            "delta": o.get("delta", 0),
            "gamma": o.get("gamma", 0),
            "vega": o.get("vega", 0),
            "theta": o.get("theta", 0),
            "last_trade_price": o.get("last_trade_price", 0),
            "last_trade_time": o.get("last_trade_time", ""),
            "_current_price": current_price,
        })

    if rows and use_cache:
        cache_store("cboe_options", cache_key, pd.DataFrame(rows))
    return rows


def list_expirations(symbol: str) -> List[str]:
    """사용 가능한 만기일 (오름차순)."""
    rows = fetch_chain_raw(symbol)
    exps = sorted(set(r["expiration"] for r in rows if r.get("expiration")))
    return exps


def get_chain(
    symbol: str, expiration: str, use_cache: bool = True,
) -> Dict:
    """단일 만기 → {expiration: {strike: {call_oi, put_oi, call_volume, ...}}}.

    yfinance/Marketdata get_options_chain과 동일 schema.
    """
    rows = fetch_chain_raw(symbol, use_cache=use_cache)
    if not rows:
        return {}

    by_strike: Dict[float, Dict] = {}
    for r in rows:
        if r["expiration"] != expiration:
            continue
        try:
            strike = float(r["strike"])
        except (TypeError, ValueError):
            continue
        slot = by_strike.setdefault(strike, {
            "call_oi": 0, "put_oi": 0,
            "call_volume": 0, "put_volume": 0,
            "call_iv": float("nan"), "put_iv": float("nan"),
            "call_delta": float("nan"), "put_delta": float("nan"),
            "call_gamma": float("nan"), "put_gamma": float("nan"),
        })
        side = r["side"]
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

    # 평균 IV (call_iv + put_iv)
    for slot in by_strike.values():
        ivs = [v for v in (slot["call_iv"], slot["put_iv"])
               if v == v and v > 0.001]  # not NaN, > tiny placeholder
        slot["iv"] = sum(ivs) / len(ivs) if ivs else float("nan")

    return {expiration: by_strike}


def pick_horizon_expiration(symbol: str, horizon_days: int = 5) -> Optional[str]:
    """horizon 이후 가장 가까운 만기."""
    exps = list_expirations(symbol)
    if not exps:
        return None
    target = (datetime.utcnow() + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
    future = [e for e in exps if e >= target]
    return future[0] if future else exps[-1]


def get_current_price(symbol: str) -> Optional[float]:
    """CBOE response의 current_price (참고용 — yfinance 우선)."""
    rows = fetch_chain_raw(symbol)
    if rows:
        cp = rows[0].get("_current_price")
        return float(cp) if cp else None
    return None


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

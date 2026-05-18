"""Options chain — yfinance 옵션 체인 + IV / HV / IV Rank.

핵심 데이터:
- strike별 call_oi, put_oi, call_iv, put_iv, call_volume, put_volume
- ATM IV (Module 2의 implied_move 계산용)
- Historic Volatility (30d realized)
- IV Rank (1년 IV 분포 내 위치)

검증 케이스 (CRCL 5/15):
- IV: 84.75%, HV: 119.96% → HV/IV = 1.42 (옵션 underpriced)
- IV Rank: 19.34% (낮음 → 보호 비용 저렴)
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf

from ._common import YFINANCE_LIMITER, cache_load, cache_store, retry
from ._time import utcnow_naive
from .price_feed import get_daily_ohlcv


# ── 만기 목록 ─────────────────────────────────────────────────
@retry(max_attempts=3, base_delay=1.0)
def list_expirations(ticker: str) -> List[str]:
    """사용 가능한 옵션 만기 (YYYY-MM-DD 문자열 리스트)."""
    YFINANCE_LIMITER.wait()
    t = yf.Ticker(ticker)
    return list(t.options)  # tuple → list


def pick_expiration_for_horizon(
    ticker: str,
    horizon_days: int = 5,
) -> Optional[str]:
    """예측 horizon에 가장 가까운 (≥) 만기 선택.

    5일 예측이면 다음 금요일 만기 또는 그 이상.
    """
    exps = list_expirations(ticker)
    if not exps:
        return None
    target = utcnow_naive() + timedelta(days=horizon_days)
    target_s = target.strftime("%Y-%m-%d")
    # horizon 이상인 첫 만기
    future = [e for e in exps if e >= target_s]
    return future[0] if future else exps[-1]


# ── 옵션 체인 ─────────────────────────────────────────────────
@retry(max_attempts=3, base_delay=1.0)
def get_options_chain(
    ticker: str,
    expiration: str,
    use_cache: bool = True,
) -> Dict[str, Dict]:
    """단일 만기의 옵션 체인.

    Returns:
        {
            expiration: {
                strike: {
                    'call_oi', 'put_oi',
                    'call_volume', 'put_volume',
                    'call_iv', 'put_iv',
                    'iv'  # avg(call_iv, put_iv) ATM 추정용
                }
            }
        }
    """
    cache_key = f"options:{ticker}:{expiration}"
    if use_cache:
        cached = cache_load("options_chain", cache_key, ttl_seconds=1800)
        if cached is not None:
            # parquet 라운드트립 후 복원
            return _df_to_chain_dict(cached, expiration)

    YFINANCE_LIMITER.wait()
    chain = yf.Ticker(ticker).option_chain(expiration)
    calls = chain.calls
    puts = chain.puts

    def _safe_int(s, default: int = 0) -> int:
        if s.empty:
            return default
        v = s.iloc[0]
        return int(v) if pd.notna(v) else default

    def _safe_float(s) -> float:
        if s.empty:
            return np.nan
        v = s.iloc[0]
        return float(v) if pd.notna(v) else np.nan

    # 데이터 품질 검사 (장 closed면 OI=0, IV=0.00001 placeholder)
    total_oi = (calls["openInterest"].fillna(0).sum()
                + puts["openInterest"].fillna(0).sum())
    median_iv = pd.concat([
        calls["impliedVolatility"].dropna(),
        puts["impliedVolatility"].dropna(),
    ]).median()
    oi_unavailable = total_oi < 1
    iv_unavailable = pd.isna(median_iv) or median_iv < 0.01

    # strike 통합
    all_strikes = sorted(set(calls["strike"].tolist()) | set(puts["strike"].tolist()))
    rows = []
    for k in all_strikes:
        c = calls[calls["strike"] == k]
        p = puts[puts["strike"] == k]
        c_iv = _safe_float(c["impliedVolatility"])
        p_iv = _safe_float(p["impliedVolatility"])
        ivs = [v for v in (c_iv, p_iv) if not np.isnan(v) and v > 0.01]
        c_vol = _safe_int(c["volume"])
        p_vol = _safe_int(p["volume"])
        c_oi = _safe_int(c["openInterest"])
        p_oi = _safe_int(p["openInterest"])

        # OI 부재 시 volume을 OI proxy로 사용 (장 closed 한계 대응)
        if oi_unavailable:
            c_oi_used = c_vol
            p_oi_used = p_vol
        else:
            c_oi_used = c_oi
            p_oi_used = p_oi

        rows.append({
            "strike": float(k),
            "call_oi": c_oi_used,
            "put_oi": p_oi_used,
            "call_volume": c_vol,
            "put_volume": p_vol,
            "call_oi_raw": c_oi,  # 디버그용
            "put_oi_raw": p_oi,
            "call_iv": c_iv if c_iv > 0.01 else np.nan,
            "put_iv": p_iv if p_iv > 0.01 else np.nan,
            "iv": float(np.mean(ivs)) if ivs else np.nan,
            "_oi_unavailable": oi_unavailable,
            "_iv_unavailable": iv_unavailable,
        })

    df = pd.DataFrame(rows).set_index("strike")
    if use_cache:
        cache_store("options_chain", cache_key, df)

    return _df_to_chain_dict(df, expiration)


def _df_to_chain_dict(df: pd.DataFrame, expiration: str) -> Dict[str, Dict]:
    """DataFrame → Module 2가 기대하는 nested dict."""
    return {
        expiration: {
            float(strike): row.dropna().to_dict()
            for strike, row in df.iterrows()
        }
    }


# ── Historic Volatility (실현 변동성) ────────────────────────
def get_historic_volatility(
    ticker: str,
    lookback_days: int = 30,
) -> float:
    """30일 실현 변동성 (annualized).

    HV = log_return.std() × √252
    """
    end = utcnow_naive()
    start = end - timedelta(days=lookback_days + 10)  # 영업일 buffer
    df = get_daily_ohlcv(ticker, start, end)
    if df.empty or len(df) < 10:
        return float("nan")

    closes = df["close"].dropna().tail(lookback_days)
    log_ret = np.log(closes / closes.shift(1)).dropna()
    return float(log_ret.std() * np.sqrt(252))


# ── IV Rank (1년 IV 분포 내 위치) ────────────────────────────
def get_iv_rank(ticker: str, lookback_days: int = 252) -> float:
    """IV Rank — 현재 ATM IV가 지난 1년 IV의 어디쯤?

    yfinance는 IV 시계열을 직접 제공하지 않으므로 **HV proxy** 사용:
        IV ≈ HV (approx) for ranking 목적
    더 정확한 IV Rank는 Polygon/Barchart 유료 데이터 필요.

    Returns:
        0~1 (0 = 1년 최저, 1 = 1년 최고)
    """
    end = utcnow_naive()
    start = end - timedelta(days=lookback_days + 30)
    df = get_daily_ohlcv(ticker, start, end)
    if df.empty or len(df) < 30:
        return 0.5  # 데이터 부족 시 중립

    closes = df["close"].dropna()
    log_ret = np.log(closes / closes.shift(1)).dropna()
    rolling_hv = log_ret.rolling(20).std() * np.sqrt(252)
    rolling_hv = rolling_hv.dropna().tail(lookback_days)

    if rolling_hv.empty:
        return 0.5

    current_hv = rolling_hv.iloc[-1]
    rank = (rolling_hv < current_hv).mean()
    return float(rank)


# ── ATM IV 추출 (편의 함수) ───────────────────────────────────
def get_atm_iv(
    ticker: str,
    expiration: str,
    current_price: float,
) -> float:
    """ATM strike의 IV."""
    chain = get_options_chain(ticker, expiration)
    strikes = list(chain[expiration].keys())
    if not strikes:
        return float("nan")
    atm = min(strikes, key=lambda s: abs(s - current_price))
    return float(chain[expiration][atm].get("iv", float("nan")))

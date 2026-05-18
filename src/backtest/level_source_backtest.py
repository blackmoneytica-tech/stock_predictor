"""가격대 source별 hit rate 검증.

가설:
  각 source가 만드는 가격대 (supply/demand zone, 옵션 OI strike,
  인사이더 ceiling, swing pivot, Max Pain, VWAP, SMA)가 실제로
  reverse(반전) 일으켰는가?

방법:
  T 시점에 각 source의 가격대 추출 → T+1~T+10 가격이 그 zone에 닿으면
  3% 이내 반전(bounce) 또는 깨짐(break) 분류.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..data.price_feed import get_daily_ohlcv
from ..data.realtime_options import get_realtime_chain


# ── 가격대 source extractor ─────────────────────────────────
def extract_levels_at(
    ticker: str,
    as_of: date,
    ohlcv: pd.DataFrame,
    current_price: float,
    option_chain: Optional[Dict] = None,
) -> List[Dict]:
    """가능한 모든 가격대 source 추출.

    Returns:
        [{source, price (or low,high), side}, ...]
        side: 'above' (저항) / 'below' (지지) / 'magnet' (자석)
    """
    levels = []

    # 1. Volume Profile demand/supply zones (검증 76.7% bounce)
    from ..modules.demand_supply import compute_volume_profile, extract_zones
    profile = compute_volume_profile(ohlcv, lookback_days=90, num_bins=50)
    zones = extract_zones(profile, ohlcv, current_price, {})
    for z in zones[:6]:
        levels.append({
            "source": "vol_profile",
            "low": z["low"], "high": z["high"],
            "price": (z["low"] + z["high"]) / 2,
            "strength": z["strength"],
            "side": z["side"],  # demand=below, supply=above
        })

    # 2. POC + VAH/VAL (자석)
    levels.append({
        "source": "poc", "low": profile["poc"], "high": profile["poc"],
        "price": profile["poc"], "strength": 5.0, "side": "magnet",
    })
    levels.append({
        "source": "vah", "low": profile["vah"], "high": profile["vah"],
        "price": profile["vah"], "strength": 3.0,
        "side": "above" if profile["vah"] > current_price else "below",
    })
    levels.append({
        "source": "val", "low": profile["val"], "high": profile["val"],
        "price": profile["val"], "strength": 3.0,
        "side": "above" if profile["val"] > current_price else "below",
    })

    # 3. 옵션 OI 큰 strikes (call → 저항, put → 지지)
    if option_chain:
        for exp, strikes in option_chain.items():
            if not strikes:
                continue
            # Top 3 call OI (보통 저항)
            call_oi_sorted = sorted(
                strikes.items(),
                key=lambda kv: -kv[1].get("call_oi", 0),
            )[:3]
            for k, v in call_oi_sorted:
                if v.get("call_oi", 0) < 100:
                    continue
                levels.append({
                    "source": "call_oi", "low": k, "high": k,
                    "price": k, "strength": min(8, v["call_oi"] / 500),
                    "side": "above" if k > current_price else "below",
                })
            # Top 3 put OI (지지)
            put_oi_sorted = sorted(
                strikes.items(),
                key=lambda kv: -kv[1].get("put_oi", 0),
            )[:3]
            for k, v in put_oi_sorted:
                if v.get("put_oi", 0) < 100:
                    continue
                levels.append({
                    "source": "put_oi", "low": k, "high": k,
                    "price": k, "strength": min(8, v["put_oi"] / 500),
                    "side": "above" if k > current_price else "below",
                })
            break

    # 4. Swing high/low (20일 pivot)
    if len(ohlcv) >= 30:
        hi_20 = float(ohlcv["high"].tail(20).max())
        lo_20 = float(ohlcv["low"].tail(20).min())
        levels.append({
            "source": "swing_high_20d", "low": hi_20, "high": hi_20,
            "price": hi_20, "strength": 4.0,
            "side": "above" if hi_20 > current_price else "below",
        })
        levels.append({
            "source": "swing_low_20d", "low": lo_20, "high": lo_20,
            "price": lo_20, "strength": 4.0,
            "side": "above" if lo_20 > current_price else "below",
        })

    # 5. SMA 50 / 200 (자석)
    if len(ohlcv) >= 50:
        sma50 = float(ohlcv["close"].rolling(50).mean().iloc[-1])
        levels.append({
            "source": "sma_50", "low": sma50, "high": sma50,
            "price": sma50, "strength": 3.0,
            "side": "above" if sma50 > current_price else "below",
        })
    if len(ohlcv) >= 200:
        sma200 = float(ohlcv["close"].rolling(200).mean().iloc[-1])
        levels.append({
            "source": "sma_200", "low": sma200, "high": sma200,
            "price": sma200, "strength": 4.0,
            "side": "above" if sma200 > current_price else "below",
        })

    # 6. ATR × N stops (volatility-adjusted)
    tr = pd.concat([
        ohlcv["high"] - ohlcv["low"],
        (ohlcv["high"] - ohlcv["close"].shift()).abs(),
        (ohlcv["low"] - ohlcv["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr14 = float(tr.rolling(14).mean().iloc[-1])
    if not np.isnan(atr14):
        for mult, name in [(1.5, "atr_1_5"), (3.0, "atr_3")]:
            up = current_price + atr14 * mult
            dn = current_price - atr14 * mult
            levels.append({
                "source": name, "low": up, "high": up,
                "price": up, "strength": 2.0, "side": "above",
            })
            levels.append({
                "source": name, "low": dn, "high": dn,
                "price": dn, "strength": 2.0, "side": "below",
            })

    return levels


# ── Hit rate 백테스트 ───────────────────────────────────────
def classify_hit(
    level: Dict,
    future: pd.DataFrame,
    touch_pct: float = 0.005,
    bounce_pct: float = 0.02,  # 2% 반전 = bounce 인정
    break_pct: float = 0.02,
) -> Tuple[str, Optional[date]]:
    """T+1~T+10 future가 level에 닿았을 때 bounce/break 분류."""
    z_low = level["low"]
    z_high = level["high"]
    side = level["side"]

    touch_lo = z_low * (1 - touch_pct)
    touch_hi = z_high * (1 + touch_pct)

    touched_idx = None
    for idx, row in future.iterrows():
        if row["low"] <= touch_hi and row["high"] >= touch_lo:
            touched_idx = idx
            break
    if touched_idx is None:
        return ("no_touch", None)

    after = future[future.index >= touched_idx]
    if len(after) < 2:
        return ("tagged", touched_idx.date())

    # side에 따라 bounce/break 방향 결정
    if side == "below":  # demand/지지
        bounce_target = z_high * (1 + bounce_pct)
        break_target = z_low * (1 - break_pct)
        for _, row in after.iterrows():
            if row["high"] >= bounce_target:
                return ("bounce", touched_idx.date())
            if row["low"] <= break_target:
                return ("break", touched_idx.date())
    elif side == "above":  # supply/저항
        bounce_target = z_low * (1 - bounce_pct)
        break_target = z_high * (1 + break_pct)
        for _, row in after.iterrows():
            if row["low"] <= bounce_target:
                return ("bounce", touched_idx.date())
            if row["high"] >= break_target:
                return ("break", touched_idx.date())
    else:  # magnet — 양방향 OK
        bounce_above = z_high * (1 + bounce_pct)
        bounce_below = z_low * (1 - bounce_pct)
        for _, row in after.iterrows():
            if row["high"] >= bounce_above or row["low"] <= bounce_below:
                return ("bounce", touched_idx.date())

    return ("tagged", touched_idx.date())


def run_source_backtest(
    tickers: List[str],
    snapshot_dates: List[date],
    horizon_days: int = 10,
    verbose: bool = True,
) -> pd.DataFrame:
    """각 ticker × snapshot × source별 hit/break 분류."""
    results = []

    for ticker in tickers:
        if verbose:
            print(f"[{ticker}]", flush=True)
        try:
            earliest = min(snapshot_dates) - timedelta(days=400)
            latest = max(snapshot_dates) + timedelta(days=horizon_days + 5)
            full = get_daily_ohlcv(ticker, earliest, latest)
            if full.empty:
                continue
            # 옵션 chain 1회 (현재 chain, lookahead bias 약간 — OI는 slow)
            option_chain = None
            try:
                option_chain = get_realtime_chain(ticker, horizon_days=10)
            except Exception:
                pass
        except Exception as e:
            if verbose:
                print(f"  SKIP fetch: {e}", flush=True)
            continue

        for snap in snapshot_dates:
            ts_snap = pd.Timestamp(snap)
            at_or_before = full[full.index <= ts_snap]
            after = full[full.index > ts_snap].head(horizon_days)
            if len(at_or_before) < 50 or after.empty:
                continue
            cur = float(at_or_before["close"].iloc[-1])

            levels = extract_levels_at(
                ticker, snap, at_or_before, cur, option_chain,
            )

            for level in levels:
                outcome, touch_date = classify_hit(level, after)
                results.append({
                    "ticker": ticker,
                    "snapshot": snap,
                    "source": level["source"],
                    "side": level["side"],
                    "price": round(level["price"], 2),
                    "strength": round(level["strength"], 2),
                    "current_price": round(cur, 2),
                    "dist_pct": round((level["price"] - cur) / cur * 100, 2),
                    "outcome": outcome,
                    "touch_date": touch_date,
                })
        if verbose:
            n_this = sum(1 for r in results if r["ticker"] == ticker)
            print(f"  done: {n_this} levels", flush=True)

    return pd.DataFrame(results)


def analyze_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Source별 bounce rate."""
    touched = df[df["outcome"].isin(["bounce", "break"])].copy()
    if touched.empty:
        return pd.DataFrame()

    by_source = (
        touched.groupby("source")
        .agg(
            n=("outcome", "count"),
            bounce=("outcome", lambda s: (s == "bounce").mean()),
            n_above=("side", lambda s: (s == "above").sum()),
            n_below=("side", lambda s: (s == "below").sum()),
            avg_dist=("dist_pct", lambda s: s.abs().mean()),
        )
        .reset_index()
        .sort_values("bounce", ascending=False)
    )
    return by_source

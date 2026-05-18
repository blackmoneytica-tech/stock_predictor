"""매물대 강도 백테스트 — 강도 분위수별 지지/저항 적중률 검증.

가설:
  H1: 강한 매물대 (강도 상위 25%) = 지지/저항으로 작용 (가격 반등)
  H2: 약한 매물대 (강도 하위 50%) = 가격이 통과 (돌파)
  H3: 옵션 OI 큰 strike와 일치하는 매물대 = 더 강한 지지/저항

방법:
  1. T0 시점에 종목 매물대 (zones) 추출
  2. T0~T+30 가격 시계열 추적
  3. 가격이 zone에 닿으면:
     - "touch" 이벤트
     - 이후 5일 내 반등(bounce) vs 돌파(break) 분류
  4. 강도 분위수별 bounce rate 측정
  5. bounce rate가 강도 따라 증가하면 H1 검증
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.price_feed import get_daily_ohlcv
from ..data.realtime_options import get_realtime_chain
from ..modules.demand_supply import compute_volume_profile, extract_zones


def run_zone_strength_backtest(
    tickers: List[str],
    snapshot_dates: List[date],
    horizon_days: int = 5,
    touch_threshold_pct: float = 0.005,  # zone 0.5% 이내 진입 = touch
    bounce_threshold_pct: float = 0.015,  # 1.5% 반등 = bounce
    break_threshold_pct: float = 0.015,   # zone 반대편 1.5% 진입 = break
    profile_lookback_days: int = 90,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Args:
        tickers: 분석 종목 리스트
        snapshot_dates: 매물대 추출 시점들 (월 1회 등)
        horizon_days: zone touch 이후 추적 기간
        touch_threshold_pct: zone 근접 임계
        bounce_threshold_pct: 반등 인정 임계
        break_threshold_pct: 돌파 인정 임계

    Returns:
        DataFrame [ticker, snapshot_date, zone_id, zone_side, zone_low, zone_high,
                   strength, touch_date, outcome ('bounce'/'break'/'no_touch'/'pending')]
    """
    results = []

    for ticker in tickers:
        # 1회 fetch — 전체 기간 (earliest snapshot - lookback ~ latest + horizon)
        try:
            earliest = min(snapshot_dates) - timedelta(days=profile_lookback_days + 30)
            latest = max(snapshot_dates) + timedelta(days=horizon_days + 5)
            full = get_daily_ohlcv(ticker, earliest, latest)
            if full.empty:
                if verbose:
                    print(f"  {ticker} SKIP: empty OHLCV", flush=True)
                continue

            # 옵션 chain 1회 (yfinance만, 현재 시점)
            option_oi = {}
            try:
                chain = get_realtime_chain(ticker, horizon_days=horizon_days)
                if chain:
                    exp = next(iter(chain))
                    for s, slot in chain[exp].items():
                        oi = slot.get("call_oi", 0) + slot.get("put_oi", 0)
                        if oi > 0:
                            option_oi[float(s)] = int(oi)
            except Exception:
                pass

            for snap_date in snapshot_dates:
                try:
                    events = _process_ticker_snapshot_cached(
                        ticker, snap_date, horizon_days,
                        touch_threshold_pct, bounce_threshold_pct, break_threshold_pct,
                        profile_lookback_days, full, option_oi,
                    )
                    results.extend(events)
                    if verbose:
                        n_touched = sum(1 for e in events if e["outcome"] != "no_touch")
                        print(
                            f"  {ticker} @ {snap_date}: {len(events)} zones, "
                            f"{n_touched} touched",
                            flush=True,
                        )
                except Exception as e:
                    if verbose:
                        print(f"  {ticker} @ {snap_date} SKIP: {e}", flush=True)
        except Exception as e:
            if verbose:
                print(f"  {ticker} SKIP (fetch): {e}", flush=True)

    return pd.DataFrame(results)


def _process_ticker_snapshot_cached(
    ticker: str,
    snap_date: date,
    horizon_days: int,
    touch_pct: float,
    bounce_pct: float,
    break_pct: float,
    lookback: int,
    full_ohlcv: pd.DataFrame,
    option_oi: Dict[float, int],
) -> List[Dict]:
    """Cached version — ticker별 fetch 없이 사전 데이터 사용."""
    ohlcv_at = full_ohlcv[full_ohlcv.index <= pd.Timestamp(snap_date)]
    if len(ohlcv_at) < 30:
        return []
    current_price = float(ohlcv_at["close"].iloc[-1])

    profile = compute_volume_profile(ohlcv_at, lookback_days=lookback, num_bins=50)
    zones = extract_zones(profile, ohlcv_at, current_price, option_oi)

    future = full_ohlcv[full_ohlcv.index > pd.Timestamp(snap_date)].head(horizon_days)
    if future.empty:
        return []

    events = []
    for i, zone in enumerate(zones[:10]):
        outcome, touch_date = _classify_outcome(
            zone, future, touch_pct, bounce_pct, break_pct,
        )
        events.append({
            "ticker": ticker,
            "snapshot_date": snap_date,
            "zone_id": i,
            "zone_side": zone["side"],
            "zone_low": round(zone["low"], 2),
            "zone_high": round(zone["high"], 2),
            "zone_center": round(zone["center"], 2),
            "strength": round(zone["strength"], 3),
            "volume_pct": round(zone["volume_pct"], 4),
            "recency": round(zone["recency"], 3),
            "option_boost": round(zone["option_boost"], 3),
            "current_price_at_snap": round(current_price, 2),
            "distance_pct": round(
                (zone["center"] - current_price) / current_price * 100, 2,
            ),
            "touch_date": touch_date,
            "outcome": outcome,
        })
    return events


def _process_ticker_snapshot(
    ticker: str,
    snap_date: date,
    horizon_days: int,
    touch_pct: float,
    bounce_pct: float,
    break_pct: float,
    lookback: int,
) -> List[Dict]:
    """단일 (ticker, snapshot_date) 처리."""
    # snap_date 이전 lookback 영업일까지의 OHLCV
    start = snap_date - timedelta(days=lookback + 30)
    ohlcv_full = get_daily_ohlcv(
        ticker, start, snap_date + timedelta(days=horizon_days + 5),
    )
    if ohlcv_full.empty:
        return []

    # snap_date 시점 cutoff (lookahead 방지)
    ohlcv_at = ohlcv_full[ohlcv_full.index <= pd.Timestamp(snap_date)]
    if len(ohlcv_at) < 30:
        return []
    current_price = float(ohlcv_at["close"].iloc[-1])

    # 옵션 OI (옵션은 historical 없어서 현재 chain만 가능)
    # 백테스트엔 lookahead 위험 있지만 OI는 slowly evolving → 영향 작다고 가정
    option_oi = {}
    try:
        chain = get_realtime_chain(ticker, horizon_days=horizon_days)
        if chain:
            exp = next(iter(chain))
            for s, slot in chain[exp].items():
                oi = slot.get("call_oi", 0) + slot.get("put_oi", 0)
                if oi > 0:
                    option_oi[float(s)] = int(oi)
    except Exception:
        pass

    # Zones 추출
    profile = compute_volume_profile(ohlcv_at, lookback_days=lookback, num_bins=50)
    zones = extract_zones(profile, ohlcv_at, current_price, option_oi)

    # T+1 ~ T+horizon_days 가격 시계열 (zone touch 분석용)
    future = ohlcv_full[ohlcv_full.index > pd.Timestamp(snap_date)].head(horizon_days)
    if future.empty:
        return []

    events = []
    for i, zone in enumerate(zones[:10]):  # 강도 top 10만
        outcome, touch_date = _classify_outcome(
            zone, future, touch_pct, bounce_pct, break_pct,
        )
        events.append({
            "ticker": ticker,
            "snapshot_date": snap_date,
            "zone_id": i,
            "zone_side": zone["side"],
            "zone_low": round(zone["low"], 2),
            "zone_high": round(zone["high"], 2),
            "zone_center": round(zone["center"], 2),
            "strength": round(zone["strength"], 3),
            "volume_pct": round(zone["volume_pct"], 4),
            "recency": round(zone["recency"], 3),
            "option_boost": round(zone["option_boost"], 3),
            "current_price_at_snap": round(current_price, 2),
            "distance_pct": round(
                (zone["center"] - current_price) / current_price * 100, 2,
            ),
            "touch_date": touch_date,
            "outcome": outcome,
        })
    return events


def _classify_outcome(
    zone: Dict,
    future: pd.DataFrame,
    touch_pct: float,
    bounce_pct: float,
    break_pct: float,
) -> tuple:
    """zone touch + bounce/break/no_touch 분류.

    Returns:
        (outcome: 'bounce'/'break'/'no_touch'/'tagged_only', touch_date or None)
    """
    z_low = zone["low"]
    z_high = zone["high"]
    z_center = zone["center"]

    # touch zone — 가격이 zone에 닿거나 통과
    touch_zone_low = z_low * (1 - touch_pct)
    touch_zone_high = z_high * (1 + touch_pct)

    touched_idx = None
    for idx, row in future.iterrows():
        if row["low"] <= touch_zone_high and row["high"] >= touch_zone_low:
            touched_idx = idx
            break

    if touched_idx is None:
        return ("no_touch", None)

    # touch 이후 가격 추적
    after = future[future.index >= touched_idx]
    if len(after) < 2:
        return ("tagged_only", touched_idx.date())

    # zone side에 따라 bounce/break 방향 다름
    # demand zone (지지) → 위쪽으로 반등 = bounce, 아래로 깨짐 = break
    # supply zone (저항) → 아래쪽으로 반등 = bounce, 위로 돌파 = break
    if zone["side"] == "demand":
        bounce_target = z_high * (1 + bounce_pct)
        break_target = z_low * (1 - break_pct)
        for _, row in after.iterrows():
            if row["high"] >= bounce_target:
                return ("bounce", touched_idx.date())
            if row["low"] <= break_target:
                return ("break", touched_idx.date())
    else:  # supply
        bounce_target = z_low * (1 - bounce_pct)
        break_target = z_high * (1 + break_pct)
        for _, row in after.iterrows():
            if row["low"] <= bounce_target:
                return ("bounce", touched_idx.date())
            if row["high"] >= break_target:
                return ("break", touched_idx.date())

    return ("tagged_only", touched_idx.date())


# ── 분석 메트릭 ──────────────────────────────────────────────
def analyze_results(df: pd.DataFrame) -> Dict:
    """강도 분위수별 bounce rate 계산."""
    if df.empty:
        return {"n": 0}

    touched = df[df["outcome"].isin(["bounce", "break"])].copy()
    if touched.empty:
        return {
            "n_total": len(df),
            "n_touched": 0,
            "summary": "no zones touched",
        }

    # 강도 quartile — duplicates drop 시 라벨 미스매치 회피
    try:
        touched["strength_q"] = pd.qcut(
            touched["strength"], q=4,
            labels=["Q1_weak", "Q2", "Q3", "Q4_strong"],
        )
    except ValueError:
        # 동일 값 많아 bin 줄어든 경우 — 자동 라벨
        touched["strength_q"] = pd.qcut(
            touched["strength"], q=4, duplicates="drop",
        ).astype(str)

    bounce_by_q = (
        touched.groupby("strength_q", observed=True)
        .agg(
            n=("outcome", "count"),
            bounce_rate=("outcome", lambda s: (s == "bounce").mean()),
            avg_strength=("strength", "mean"),
            avg_volume_pct=("volume_pct", "mean"),
        )
        .reset_index()
    )

    # demand/supply 분리
    by_side = (
        touched.groupby(["zone_side", "strength_q"], observed=True)
        .agg(
            n=("outcome", "count"),
            bounce_rate=("outcome", lambda s: (s == "bounce").mean()),
        )
        .reset_index()
    )

    # option_boost effect
    touched["has_option_oi"] = touched["option_boost"] > 1.05
    by_oi = (
        touched.groupby("has_option_oi", observed=True)
        .agg(
            n=("outcome", "count"),
            bounce_rate=("outcome", lambda s: (s == "bounce").mean()),
        )
        .reset_index()
    )

    # recency effect
    touched["recency_bin"] = pd.cut(
        touched["recency"], bins=[0, 0.6, 1.0, 2.0],
        labels=["old", "mid", "recent"],
    )
    by_recency = (
        touched.groupby("recency_bin", observed=True)
        .agg(
            n=("outcome", "count"),
            bounce_rate=("outcome", lambda s: (s == "bounce").mean()),
        )
        .reset_index()
    )

    return {
        "n_total": len(df),
        "n_touched": len(touched),
        "n_bounce": int((touched["outcome"] == "bounce").sum()),
        "n_break": int((touched["outcome"] == "break").sum()),
        "overall_bounce_rate": float((touched["outcome"] == "bounce").mean()),
        "by_strength_quartile": bounce_by_q.to_dict(orient="records"),
        "by_side_quartile": by_side.to_dict(orient="records"),
        "by_option_oi": by_oi.to_dict(orient="records"),
        "by_recency": by_recency.to_dict(orient="records"),
    }


def print_analysis(metrics: Dict) -> None:
    print(f"\n====== Zone Strength Backtest ======")
    if metrics.get("n_total", 0) == 0:
        print("  no data")
        return

    print(f"  Total zones:    {metrics['n_total']}")
    print(f"  Touched zones:  {metrics['n_touched']}")
    if metrics["n_touched"] == 0:
        return
    print(f"  Bounces:        {metrics['n_bounce']}")
    print(f"  Breaks:         {metrics['n_break']}")
    print(f"  Overall bounce: {metrics['overall_bounce_rate']:.1%}")

    print("\n  By strength quartile (H1: stronger zone -> higher bounce):")
    for r in metrics["by_strength_quartile"]:
        print(
            f"    {str(r['strength_q']):12s}  n={r['n']:3d}  "
            f"bounce={r['bounce_rate']:.1%}  "
            f"avg_str={r['avg_strength']:.2f}  vol_pct={r['avg_volume_pct']:.3f}"
        )

    print("\n  By side × strength:")
    for r in metrics["by_side_quartile"]:
        print(
            f"    {r['zone_side']:8s} {str(r['strength_q']):12s}  "
            f"n={r['n']:3d}  bounce={r['bounce_rate']:.1%}"
        )

    print("\n  By option OI (H3: zones aligned with high OI strikes):")
    for r in metrics["by_option_oi"]:
        flag = "with_OI" if r["has_option_oi"] else "no_OI"
        print(f"    {flag:10s}  n={r['n']:3d}  bounce={r['bounce_rate']:.1%}")

    print("\n  By recency (recent volume vs older):")
    for r in metrics["by_recency"]:
        print(
            f"    {str(r['recency_bin']):8s}  n={r['n']:3d}  "
            f"bounce={r['bounce_rate']:.1%}"
        )

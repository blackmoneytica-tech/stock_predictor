"""Max Pain 수렴 가설 검증.

가설: 옵션 만기 임박 시 가격 → Max Pain으로 수렴.

방법:
1. 각 (ticker, snap)에서:
   - 그 시점 옵션 chain에서 max_pain + DTE(만기까지 일수) 추출
   - 만기일 actual close price → expiry_price
   - signed_dist = (cur - max_pain) / cur (음수면 가격이 max_pain 아래)
   - expiry_dist = (expiry_price - max_pain) / expiry_price
   - 수렴 여부: |signed_dist| > |expiry_dist|

2. 분석:
   A. DTE bucket별 평균 수렴률
   B. signed_dist bucket별 수렴 (가격이 max_pain 위/아래일 때 다른가?)
   C. 만기일까지 actual return (long vs short)
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def collect_max_pain_events(
    tickers: List[str], snapshot_dates: List[date],
    verbose: bool = True,
) -> pd.DataFrame:
    from ..data.insider import get_insider_activity
    from ..data.price_feed import get_daily_ohlcv
    from ..system import StockPredictionSystem
    from .walk_forward import build_data_at

    system = StockPredictionSystem()
    rows = []

    if verbose:
        print("[prefetch] sector/macro ETFs...", flush=True)
    earliest = min(snapshot_dates) - timedelta(days=400)
    latest = max(snapshot_dates) + timedelta(days=60)
    for etf in ('XLK','XLF','XLE','XLV','XLI','XLY','XLP','XLU','XLRE',
                'XLB','XLC','SPY','QQQ','IWM','^VIX','HYG','LQD'):
        try:
            get_daily_ohlcv(etf, earliest, latest)
        except Exception:
            pass
    try:
        from ..data.sector_macro import compute_macro_breadth_at
        for snap in snapshot_dates:
            try:
                compute_macro_breadth_at(snap)
            except Exception:
                pass
    except ImportError:
        pass

    for ticker in tickers:
        if verbose:
            print(f"[{ticker}]", flush=True)
        try:
            full = get_daily_ohlcv(ticker, earliest, latest)
            if full.empty:
                continue
            insider = get_insider_activity(ticker, months_back=12)
        except Exception:
            continue

        for snap in snapshot_dates:
            ts_snap = pd.Timestamp(snap)
            at_or_before = full[full.index <= ts_snap]
            after = full[full.index > ts_snap]
            if at_or_before.empty or len(after) < 2:
                continue
            actual_today = float(full.loc[at_or_before.index[-1], "close"])

            # 분석 시점 (snap)에 시스템 호출 → max_pain + DTE
            try:
                data = build_data_at(
                    ticker, snap, horizon_days=5, use_macro=False,
                    insider_cache=insider,
                )
                result = system.analyze(ticker, horizon_days=5, data=data)
            except Exception:
                continue

            opt = result.modules['options'].details
            max_pain = opt.get('max_pain')
            dte = opt.get('days_to_expiration')
            exp_date_str = opt.get('expiration_date')
            if not max_pain or not dte or not exp_date_str or max_pain <= 0:
                continue

            try:
                exp_date = pd.Timestamp(exp_date_str)
            except Exception:
                continue

            # 만기일 close (실제 가격)
            # 가장 가까운 trading day 사용
            on_or_after_exp = full[full.index >= exp_date]
            if on_or_after_exp.empty:
                continue
            expiry_idx = on_or_after_exp.index[0]
            expiry_price = float(full.loc[expiry_idx, "close"])
            actual_days = (expiry_idx.date() - at_or_before.index[-1].date()).days

            signed_dist = (actual_today - max_pain) / actual_today * 100  # %
            expiry_signed_dist = (expiry_price - max_pain) / expiry_price * 100
            abs_dist_reduced = abs(signed_dist) - abs(expiry_signed_dist)  # 양수 = 수렴
            converged = abs_dist_reduced > 0
            # 가격이 max_pain 방향으로 움직였나
            direction_match = (
                (signed_dist > 0 and expiry_price < actual_today) or
                (signed_dist < 0 and expiry_price > actual_today)
            )
            ret_to_expiry = (expiry_price - actual_today) / actual_today * 100

            rows.append({
                "ticker": ticker,
                "as_of": snap.isoformat(),
                "expiry_date": exp_date_str,
                "cur": round(actual_today, 2),
                "max_pain": round(max_pain, 2),
                "expiry_price": round(expiry_price, 2),
                "dte_target": int(dte),  # 분석 시점 시스템이 본 DTE
                "actual_days_to_expiry": int(actual_days),
                "signed_dist_pct": round(signed_dist, 2),
                "expiry_signed_dist_pct": round(expiry_signed_dist, 2),
                "abs_dist_reduced_pct": round(abs_dist_reduced, 2),
                "converged": bool(converged),
                "direction_match": bool(direction_match),
                "ret_to_expiry_pct": round(ret_to_expiry, 2),
            })

    return pd.DataFrame(rows)


def analyze_max_pain(df: pd.DataFrame):
    if df.empty:
        print("결과 없음")
        return

    n = len(df)
    print(f"\n=== 전체 {n} events ({df['ticker'].nunique()} tickers) ===")
    print(f"converged (|dist| 감소) rate: {df['converged'].mean():.1%}")
    print(f"direction_match (max_pain 방향으로 이동) rate: {df['direction_match'].mean():.1%}")
    print(f"평균 |signed_dist| at snap: {df['signed_dist_pct'].abs().mean():.2f}%")
    print(f"평균 |signed_dist| at expiry: {df['expiry_signed_dist_pct'].abs().mean():.2f}%")
    print()

    # --- A. DTE bucket별 ---
    print("--- A. DTE (만기까지 일수) bucket별 수렴률 ---")
    df['dte_bucket'] = pd.cut(
        df['actual_days_to_expiry'],
        bins=[-1, 3, 7, 14, 30, 1000],
        labels=["1~3일", "4~7일", "8~14일", "15~30일", "30일+"],
    )
    g = df.groupby('dte_bucket', observed=False).agg(
        n=('converged', 'size'),
        converged_rate=('converged', 'mean'),
        dir_match_rate=('direction_match', 'mean'),
        avg_dist_reduced=('abs_dist_reduced_pct', 'mean'),
        avg_ret_to_expiry=('ret_to_expiry_pct', 'mean'),
    )
    for c in ['converged_rate', 'dir_match_rate']:
        g[c] = g[c].apply(lambda x: f"{x:.1%}")
    for c in ['avg_dist_reduced', 'avg_ret_to_expiry']:
        g[c] = g[c].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())
    print()

    # --- B. signed_dist bucket별 (가격이 max_pain 위/아래) ---
    print("--- B. 현재가 vs Max Pain 거리 bucket별 ---")
    df['dist_bucket'] = pd.cut(
        df['signed_dist_pct'],
        bins=[-100, -10, -5, -2, 2, 5, 10, 100],
        labels=["<-10%", "-10~-5%", "-5~-2%", "±2%", "+2~+5%", "+5~+10%", ">+10%"],
    )
    g = df.groupby('dist_bucket', observed=False).agg(
        n=('converged', 'size'),
        converged_rate=('converged', 'mean'),
        dir_match_rate=('direction_match', 'mean'),
        avg_ret_to_expiry=('ret_to_expiry_pct', 'mean'),
    )
    for c in ['converged_rate', 'dir_match_rate']:
        g[c] = g[c].apply(lambda x: f"{x:.1%}")
    g['avg_ret_to_expiry'] = g['avg_ret_to_expiry'].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())
    print()

    # --- C. DTE ≤ 7일에서 거리 큰 종목 (max_pain까지 도달 못한 것) ---
    print("--- C. DTE ≤7일 + |거리| ≥ 3% → max_pain까지 진짜 끌릴까? ---")
    near = df[(df['actual_days_to_expiry'] <= 7) & (df['signed_dist_pct'].abs() >= 3)]
    if len(near) > 0:
        # 가격이 max_pain 위인 경우 → 만기까지 down 예상
        above = near[near['signed_dist_pct'] > 0]
        below = near[near['signed_dist_pct'] < 0]
        print(f"  현재가 > Max Pain (가격이 떨어져야 수렴):")
        if len(above):
            print(f"    n={len(above)}, 실제 down 비율: {(above['ret_to_expiry_pct']<0).mean():.1%}, "
                  f"avg ret: {above['ret_to_expiry_pct'].mean():+.2f}%")
        print(f"  현재가 < Max Pain (가격이 올라가야 수렴):")
        if len(below):
            print(f"    n={len(below)}, 실제 up 비율: {(below['ret_to_expiry_pct']>0).mean():.1%}, "
                  f"avg ret: {below['ret_to_expiry_pct'].mean():+.2f}%")
    print()

    # --- D. 큰 거리 vs 작은 거리: 가설 더 적합한 case? ---
    print("--- D. 큰 거리(|dist|≥5%) vs 작은 거리 → 수렴률 차이 ---")
    for label, mask in [
        ("거리 ≥10%", df['signed_dist_pct'].abs() >= 10),
        ("거리 5~10%", df['signed_dist_pct'].abs().between(5, 10)),
        ("거리 2~5%", df['signed_dist_pct'].abs().between(2, 5)),
        ("거리 <2% (이미 근접)", df['signed_dist_pct'].abs() < 2),
    ]:
        sub = df[mask]
        if len(sub) < 5:
            continue
        c = sub['converged'].mean()
        d = sub['direction_match'].mean()
        r = sub['ret_to_expiry_pct'].mean()
        print(f"  {label:<20s} n={len(sub):>4d} converged={c:.1%} dir_match={d:.1%} avg_ret={r:+.2f}%")
    print()

    # --- E. 무작위 baseline 비교 ---
    print("--- E. Baseline 비교 (수렴이 진짜 자석 효과인지) ---")
    # 가격이 어디로 갈지 50/50이라면 |dist|는 평균 같거나 약간 커야 (random walk variance)
    # 수렴 진짜라면 |dist| 줄어듦 비율 > 50%
    conv_rate = df['converged'].mean()
    dir_rate = df['direction_match'].mean()
    print(f"  Max Pain 자석 효과 검증:")
    print(f"  - |거리| 감소: {conv_rate:.1%} (50% 초과 시 자석 효과 있음)")
    print(f"  - 방향 일치: {dir_rate:.1%}")
    if conv_rate > 0.55:
        print(f"  → ✅ 자석 효과 의미 있음 (random walk 50% 대비)")
    elif conv_rate > 0.50:
        print(f"  → 🟡 자석 효과 약함")
    else:
        print(f"  → ❌ 자석 효과 부정 (random walk보다 나쁨)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers",
        default="NVDA,AMD,TSLA,AAPL,MSFT,GOOGL,META,AMZN,CRCL,MSTR,COIN,HOOD",
    )
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    end_date = date.today() - timedelta(days=20)  # 만기일까지 fully observed 보장
    snaps = []
    d = end_date
    while len(snaps) < args.days:
        if d.weekday() < 5:
            snaps.append(d)
        d -= timedelta(days=1)
    snaps.sort()

    print(f"tickers ({len(tickers)}): {tickers}")
    print(f"snapshots: {snaps[0]} ~ {snaps[-1]} ({len(snaps)} BD)")

    df = collect_max_pain_events(tickers, snaps)
    if df.empty:
        print("no events")
        return

    out = Path(__file__).resolve().parents[2] / "data" / "results" / "max_pain_convergence.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n[saved] {out} ({len(df)} events)\n")

    analyze_max_pain(df)


if __name__ == "__main__":
    main()

"""2020 봄 폭락 backtest — 극단 약세장 + V-반등 검증.

기간: 2020-02-19 ~ 2020-05-31
  - 2020-02-19: ATH ($339 SPY) — 폭락 시작점
  - 2020-03-23: 저점 ($222 SPY, -34%)
  - 2020-05: V-반등 진행 중

이 구간:
  - 강세장도 약세장도 단순 분류 어려운 극단 변동성
  - VIX 80+ (역사적 최고)
  - Sweet Spot 진짜 robust인지 가장 극단 검증
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def collect_2020(tickers: List[str], n_snapshots: int = 40, verbose: bool = True):
    """2020 봄 폭락 + V-반등 데이터 수집."""
    from ..data.insider import get_insider_activity
    from ..data.price_feed import get_daily_ohlcv
    from ..system import StockPredictionSystem
    from .walk_forward import build_data_at

    system = StockPredictionSystem()
    rows = []

    start, end = date(2020, 2, 19), date(2020, 5, 31)
    all_bd = pd.bdate_range(start, end)
    step = max(1, len(all_bd) // n_snapshots)
    snaps = [d.date() for d in all_bd[::step]][:n_snapshots]

    if verbose:
        print(f"[prefetch] ETFs + macro for {snaps[0]} ~ {snaps[-1]}", flush=True)
    earliest = snaps[0] - timedelta(days=400)
    latest = snaps[-1] + timedelta(days=15)
    for etf in ('XLK','XLF','XLE','XLV','XLI','XLY','XLP','XLU','XLRE',
                'XLB','XLC','SPY','QQQ','IWM','^VIX','HYG','LQD'):
        try:
            get_daily_ohlcv(etf, earliest, latest)
        except Exception:
            pass
    try:
        from ..data.sector_macro import compute_macro_breadth_at
        for snap in snaps:
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

        for snap in snaps:
            ts_snap = pd.Timestamp(snap)
            after = full[full.index > ts_snap]
            at_or_before = full[full.index <= ts_snap]
            if at_or_before.empty or len(after) < 10:
                continue
            actual_today = float(full.loc[at_or_before.index[-1], "close"])
            actuals = {}
            for h in (1, 3, 5, 10):
                future = full[full.index > ts_snap]
                actuals[h] = float(future["close"].iloc[h - 1]) if len(future) >= h else None
            if any(v is None for v in actuals.values()):
                continue

            try:
                data = build_data_at(
                    ticker, snap, horizon_days=5, use_macro=True,
                    insider_cache=insider,
                )
                pred = system.analyze(ticker, horizon_days=5, data=data)
            except Exception:
                continue

            ss_dict = getattr(pred, "sweet_spot", None) or {}
            mc_dict = getattr(pred, "module_consensus", None) or {}
            row = {
                "ticker": ticker,
                "as_of": snap.isoformat(),
                "cur": round(actual_today, 2),
                "composite_score": round(pred.composite_score, 3),
                "confidence": round(pred.confidence, 3),
                "ev_pct_5d": round(
                    (pred.expected_value - actual_today) / actual_today * 100, 3
                ),
                "macro_mode": (data.get("macro_breadth") or {}).get("mode", "?"),
                "recommended_size": getattr(pred, "recommended_size", 0.0),
                "sweet_spot_active": bool(ss_dict.get("active", False)),
                "module_tier": mc_dict.get("tier", "noise"),
            }
            for name, m in pred.modules.items():
                row[f"mod_{name}"] = round(m.score, 3)
            for h in (1, 3, 5, 10):
                ret = (actuals[h] - actual_today) / actual_today * 100
                row[f"actual_ret_{h}d"] = round(ret, 3)
            rows.append(row)

    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame):
    if df.empty:
        print("no data")
        return

    print(f"\n=== 전체 {len(df)} events ===")
    print(f"period: {df['as_of'].min()} ~ {df['as_of'].max()}")
    print(f"\nmacro 분포:")
    print(df['macro_mode'].value_counts().to_string())

    print(f"\nactual returns:")
    for h in [1, 5, 10]:
        s = df[f"actual_ret_{h}d"].dropna()
        print(f"  {h}d: win {(s>0).mean():.1%}, avg {s.mean():+.2f}%, n={len(s)}")

    # Sweet spot
    print(f"\n=== Sweet Spot (Tier 1 검증) ===")
    sweet = df[df['sweet_spot_active']]
    print(f"  적중: {len(sweet)}/{len(df)} ({len(sweet)/len(df)*100:.1f}%)")
    if len(sweet):
        for h in [1, 5, 10]:
            s = sweet[f"actual_ret_{h}d"].dropna()
            base = df[f"actual_ret_{h}d"].dropna()
            print(f"  {h}d: win {(s>0).mean():.1%} (baseline {(base>0).mean():.1%}), "
                  f"avg {s.mean():+.2f}% (baseline {base.mean():+.2f}%)")

    # Module tier
    print(f"\n=== Module tier (5d) ===")
    for tier in df['module_tier'].value_counts().index:
        sub = df[df['module_tier'] == tier].dropna(subset=['actual_ret_5d'])
        if len(sub) < 3:
            continue
        win = (sub['actual_ret_5d'] > 0).mean()
        avg = sub['actual_ret_5d'].mean()
        print(f"  {tier:<28s} n={len(sub):>4d}, win {win:.1%}, avg {avg:+.2f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers",
        default="NVDA,AMD,TSLA,AAPL,MSFT,GOOGL,META,AMZN,NFLX,SPY",
    )
    parser.add_argument("--snapshots", type=int, default=40)
    args = parser.parse_args()
    tickers = [t.strip().upper() for t in args.tickers.split(",")]

    print(f"tickers ({len(tickers)}): {tickers}")
    print(f"snapshots: {args.snapshots}")

    df = collect_2020(tickers, n_snapshots=args.snapshots)
    if df.empty:
        print("no events")
        return

    out = Path(__file__).resolve().parents[2] / "data" / "results" / "bear_2020.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n[saved] {out} ({len(df)} events)")
    analyze(df)


if __name__ == "__main__":
    main()

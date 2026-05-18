"""MA family comparison backtest — SMA vs EMA vs WMA vs Ensemble.

목적: trend_score가 어떤 moving average에서 가장 좋은 예측력을 갖는지.

방법:
- 동일 ticker × 동일 snapshot 시점에서 4번 system.analyze 호출 (각 variant)
- expected_value의 방향 vs actual next-day return 비교
- variant별 directional accuracy + mean |error| + Sharpe-ish (mean/std of pred_ret) + by-context breakdown

caveat:
- trend_score는 11개 모듈 중 하나 (technical, weight 0.15). 영향력 제한적.
- 그래도 같은 데이터·같은 모듈에서 MA 공식만 바꿔서 비교 → 마진 효과 측정 가능.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List

import numpy as np
import pandas as pd

from ..data.insider import get_insider_activity
from ..data.price_feed import get_daily_ohlcv
from ..system import StockPredictionSystem
from .walk_forward import build_data_at


VARIANTS = ('sma', 'ema', 'wma', 'ensemble')


def run_ma_variant_backtest(
    tickers: List[str],
    snapshot_dates: List[date],
    horizon_days: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """N × M × 4 variants prediction. 동일 data로 _ma_variant override만 다르게."""
    system = StockPredictionSystem()
    results = []

    # Prefetch ETFs / macro (variant 무관 — 한 번만)
    if verbose:
        print("[prefetch] sector + macro ETFs...", flush=True)
    earliest_etf = min(snapshot_dates) - timedelta(days=400)
    latest_etf = max(snapshot_dates) + timedelta(days=horizon_days + 5)
    for etf in ('XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY',
                'XLP', 'XLU', 'XLRE', 'XLB', 'XLC',
                'SPY', 'QQQ', 'IWM', '^VIX', 'HYG', 'LQD'):
        try:
            get_daily_ohlcv(etf, earliest_etf, latest_etf)
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
    if verbose:
        print("[prefetch] done", flush=True)

    for ticker in tickers:
        if verbose:
            print(f"[{ticker}]", flush=True)

        try:
            earliest = min(snapshot_dates) - timedelta(days=400)
            latest = max(snapshot_dates) + timedelta(days=horizon_days + 5)
            full = get_daily_ohlcv(ticker, earliest, latest)
            if full.empty:
                continue
            insider_cache = get_insider_activity(ticker, months_back=12)
        except Exception as e:
            if verbose:
                print(f"  SKIP fetch: {e}", flush=True)
            continue

        for snap in snapshot_dates:
            ts_snap = pd.Timestamp(snap)
            at_or_before = full[full.index <= ts_snap]
            after = full[full.index > ts_snap]
            if at_or_before.empty or after.empty:
                continue
            as_of_idx = at_or_before.index[-1]
            next_idx = after.index[0]
            as_of_date = as_of_idx.date()
            next_date = next_idx.date()
            if (next_date - as_of_date).days > 5:
                continue

            actual_today = float(full.loc[as_of_idx, "close"])
            actual_next = float(full.loc[next_idx, "close"])
            actual_ret = (actual_next - actual_today) / actual_today * 100

            try:
                data = build_data_at(
                    ticker, as_of_date,
                    horizon_days=horizon_days,
                    use_macro=True,
                    insider_cache=insider_cache,
                )
            except Exception as e:
                if verbose:
                    print(f"  {as_of_date} build fail: {e}", flush=True)
                continue

            for variant in VARIANTS:
                data['_ma_variant'] = variant
                try:
                    pred = system.analyze(ticker, horizon_days=horizon_days, data=data)
                except Exception as e:
                    if verbose:
                        print(f"  {as_of_date} {variant} FAIL: {e}", flush=True)
                    continue
                pred_ret = (pred.expected_value - pred.current_price) / pred.current_price * 100

                THRES = 0.3
                if abs(pred_ret) < THRES and abs(actual_ret) > 0.5:
                    dir_correct = False
                else:
                    dir_correct = (pred_ret > 0) == (actual_ret > 0)

                results.append({
                    "ticker": ticker,
                    "as_of": as_of_date.isoformat(),
                    "variant": variant,
                    "pred_ret": round(pred_ret, 3),
                    "actual_ret": round(actual_ret, 3),
                    "abs_err": round(abs(pred_ret - actual_ret), 3),
                    "dir_correct": bool(dir_correct),
                    "composite_score": round(pred.composite_score, 3),
                    "confidence": round(pred.confidence, 3),
                    "tech_trend": round(
                        pred.modules['technical'].details.get('trend_score', 0), 3
                    ),
                    "macro_mode": (data.get("macro_breadth") or {}).get("mode", "?"),
                })

    return pd.DataFrame(results)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """variant별 directional accuracy + bias + signal/noise."""
    if df.empty:
        return pd.DataFrame()
    rows = []
    for v in VARIANTS:
        sub = df[df['variant'] == v]
        if sub.empty:
            continue
        n = len(sub)
        acc = float(sub['dir_correct'].mean())
        mae = float(sub['abs_err'].mean())
        mean_pred = float(sub['pred_ret'].mean())
        std_pred = float(sub['pred_ret'].std())
        # 같은 행에서 actual과 pred의 상관계수
        corr = float(sub[['pred_ret', 'actual_ret']].corr().iloc[0, 1])
        # PnL proxy: long-only when pred_ret > 0.3, short when < -0.3
        signal = sub['pred_ret'].clip(-1, 1).where(sub['pred_ret'].abs() > 0.3, 0)
        signal = np.sign(signal)
        pnl = (signal * sub['actual_ret']).mean()
        # trend score 평균
        mean_trend = float(sub['tech_trend'].mean())
        rows.append({
            "variant": v,
            "n": n,
            "dir_acc": round(acc, 3),
            "abs_err_pct": round(mae, 3),
            "mean_pred_ret": round(mean_pred, 3),
            "pearson_corr": round(corr, 3),
            "long_short_pnl_per_trade": round(pnl, 3),
            "mean_trend_score": round(mean_trend, 2),
        })
    return pd.DataFrame(rows)


def main():
    """CLI: python -m src.backtest.ma_variant_comparison."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers", default="NVDA,AMD,TSLA,AAPL,MSFT,GOOGL,META,AMZN,NFLX,CRCL",
        help="comma-separated tickers"
    )
    parser.add_argument("--days", type=int, default=30, help="how many snapshot days back")
    parser.add_argument("--horizon", type=int, default=1, help="prediction horizon days")
    parser.add_argument("--end", default=None, help="last snapshot YYYY-MM-DD (default: today-2)")
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    if args.end:
        end_date = date.fromisoformat(args.end)
    else:
        # 어제는 actual 못 잡을 수 있으니 -2일
        end_date = date.today() - timedelta(days=2)

    # 영업일 가까운 25개 (주말 제외 추정)
    snaps = []
    d = end_date
    while len(snaps) < args.days:
        if d.weekday() < 5:  # Mon-Fri
            snaps.append(d)
        d -= timedelta(days=1)
    snaps.sort()

    print(f"tickers: {tickers}", flush=True)
    print(f"snapshots: {snaps[0]} ~ {snaps[-1]} ({len(snaps)} business days)", flush=True)

    df = run_ma_variant_backtest(
        tickers, snaps, horizon_days=args.horizon, verbose=True,
    )

    if df.empty:
        print("\n결과 없음 (모든 snapshot 실패)")
        return

    # 저장
    from pathlib import Path
    out = Path(__file__).resolve().parents[2] / "data" / "results" / "ma_variants.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n[saved] {out} ({len(df)} rows)", flush=True)

    # 요약
    print("\n=== Variant 비교 ===")
    summary = summarize(df)
    print(summary.to_string(index=False))

    # context별 (macro_mode 분기)
    print("\n=== macro_mode × variant ===")
    pivot = df.groupby(['macro_mode', 'variant'])['dir_correct'].agg(['mean', 'count'])
    print(pivot.to_string())


if __name__ == "__main__":
    main()

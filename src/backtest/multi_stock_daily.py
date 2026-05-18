"""다종목 daily walk-forward — N 종목 × M 일 = N*M 예측.

목적:
- 1-day horizon 시스템의 통계적 적중률 검증
- ticker별 OHLCV/Finnhub 1회 fetch + 시점별 cutoff
- 결과: directional accuracy / MAE / bias / by-ticker breakdown
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data.insider import get_insider_activity
from ..data.price_feed import get_daily_ohlcv
from ..system import StockPredictionSystem
from .walk_forward import build_data_at


def run_multi_stock_daily(
    tickers: List[str],
    snapshot_dates: List[date],
    horizon_days: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """N 종목 × M 시점 = N*M 예측.

    각 (ticker, as_of)에서 다음 영업일 등락 예측 + actual 비교.
    """
    system = StockPredictionSystem()
    results = []

    # Prefetch sector + macro ETFs (모든 시점 공유)
    if verbose:
        print("[prefetch] sector + macro ETFs...", flush=True)
    earliest_etf = min(snapshot_dates) - timedelta(days=400)
    latest_etf = max(snapshot_dates) + timedelta(days=horizon_days + 5)
    for etf in ('XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY',
                'XLP', 'XLU', 'XLRE', 'XLB', 'XLC',
                'SPY', 'QQQ', 'IWM', '^VIX', '^VIX9D', 'HYG', 'LQD'):
        try:
            get_daily_ohlcv(etf, earliest_etf, latest_etf)
        except Exception:
            pass

    # Prefetch macro_breadth per snapshot (ticker 무관)
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

        # ticker별 1회 fetch
        try:
            earliest = min(snapshot_dates) - timedelta(days=400)
            latest = max(snapshot_dates) + timedelta(days=horizon_days + 5)
            full = get_daily_ohlcv(ticker, earliest, latest)
            if full.empty:
                if verbose:
                    print(f"  SKIP: empty OHLCV", flush=True)
                continue
            insider_cache = get_insider_activity(ticker, months_back=12)
        except Exception as e:
            if verbose:
                print(f"  SKIP fetch: {e}", flush=True)
            continue

        for snap in snapshot_dates:
            # as_of 영업일 + 다음 영업일 actual 찾기
            ts_snap = pd.Timestamp(snap)
            at_or_before = full[full.index <= ts_snap]
            after = full[full.index > ts_snap]
            if at_or_before.empty or after.empty:
                continue
            as_of_idx = at_or_before.index[-1]
            next_idx = after.index[0]
            as_of_date = as_of_idx.date()
            next_date = next_idx.date()

            # 1일 미만 차이만 (영업일 next 1)
            if (next_date - as_of_date).days > 5:
                continue  # 너무 큰 갭 = 휴장 등

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
                pred = system.analyze(ticker, horizon_days=horizon_days, data=data)
                pred_ret = (pred.expected_value - pred.current_price) / pred.current_price * 100
            except Exception as e:
                if verbose:
                    print(f"  {as_of_date} FAIL: {e}", flush=True)
                continue

            # Directional with 0.3% noise threshold
            THRES = 0.3
            if abs(pred_ret) < THRES and abs(actual_ret) > 0.5:
                dir_correct = False
            else:
                dir_correct = (pred_ret > 0) == (actual_ret > 0)

            results.append({
                "ticker": ticker,
                "as_of": as_of_date.isoformat(),
                "next": next_date.isoformat(),
                "cur": round(actual_today, 2),
                "pred_close": round(pred.expected_value, 2),
                "actual_close": round(actual_next, 2),
                "pred_ret_pct": round(pred_ret, 3),
                "actual_ret_pct": round(actual_ret, 3),
                "abs_err_usd": round(abs(pred.expected_value - actual_next), 3),
                "score": round(pred.composite_score, 3),
                "confidence": round(pred.confidence, 3),
                "directional_bias": pred.directional_bias,
                "dir_correct": bool(dir_correct),
                # 추가 context
                "post_catalyst_within": data.get("post_catalyst_within_days", 999),
                "pre_rally_pct": round(data.get("pre_catalyst_rally_pct", 0), 3),
                "macro_mode": (data.get("macro_breadth") or {}).get("mode", "?"),
                "days_to_earnings": data.get("days_to_earnings", 999),
                "beat_proxy": round(data.get("beat_probability_proxy", 0.5), 3),
            })
        if verbose:
            n_ticker = sum(1 for r in results if r["ticker"] == ticker)
            correct_ticker = sum(1 for r in results if r["ticker"] == ticker and r["dir_correct"])
            if n_ticker > 0:
                print(f"  done: {correct_ticker}/{n_ticker} correct", flush=True)

    return pd.DataFrame(results)


def analyze(df: pd.DataFrame) -> Dict:
    """전체 + by-ticker 분석."""
    if df.empty:
        return {"n": 0}

    n = len(df)
    correct = int(df["dir_correct"].sum())
    mae = float(df["abs_err_usd"].mean())
    bias = float((df["pred_ret_pct"] - df["actual_ret_pct"]).mean())

    # By ticker
    by_ticker = (
        df.groupby("ticker")
        .agg(
            n=("dir_correct", "count"),
            correct=("dir_correct", "sum"),
            mae=("abs_err_usd", "mean"),
        )
        .reset_index()
    )
    by_ticker["acc"] = by_ticker["correct"] / by_ticker["n"]

    # By confidence bucket
    df = df.copy()
    df["conf_bin"] = pd.cut(
        df["confidence"],
        bins=[0, 0.4, 0.5, 0.6, 0.75],
        labels=["<0.4", "0.4-0.5", "0.5-0.6", "0.6-0.75"],
    )
    by_conf = (
        df.groupby("conf_bin", observed=True)
        .agg(n=("dir_correct", "count"), acc=("dir_correct", "mean"))
        .reset_index()
    )

    # By macro mode
    by_mode = (
        df.groupby("macro_mode")
        .agg(n=("dir_correct", "count"), acc=("dir_correct", "mean"))
        .reset_index()
    )

    # Catalyst-active vs not
    df["has_catalyst"] = df["post_catalyst_within"] <= 5
    by_cat = (
        df.groupby("has_catalyst")
        .agg(n=("dir_correct", "count"), acc=("dir_correct", "mean"))
        .reset_index()
    )

    return {
        "n": n,
        "correct": correct,
        "accuracy": correct / n if n else 0,
        "mae_usd": mae,
        "bias_pct": bias,
        "by_ticker": by_ticker.to_dict("records"),
        "by_confidence": by_conf.to_dict("records"),
        "by_macro_mode": by_mode.to_dict("records"),
        "by_catalyst": by_cat.to_dict("records"),
    }


def print_report(df: pd.DataFrame) -> None:
    m = analyze(df)
    if not m.get("n"):
        print("no data")
        return
    print(f"\n====== Multi-stock daily walk-forward ======")
    print(f"  N predictions:     {m['n']}")
    print(f"  Directional acc:   {m['accuracy']:.1%}  ({m['correct']}/{m['n']})")
    print(f"  MAE:               ${m['mae_usd']:.2f}")
    print(f"  Bias (pred-actual): {m['bias_pct']:+.3f}%")

    print("\n  By ticker (top 15 by accuracy):")
    by_t = sorted(m["by_ticker"], key=lambda r: -r["acc"])[:15]
    for r in by_t:
        print(f"    {r['ticker']:8s}  n={int(r['n']):2d}  acc={r['acc']:.0%}  mae=${r['mae']:.2f}")

    print("\n  Bottom 5 tickers:")
    bot = sorted(m["by_ticker"], key=lambda r: r["acc"])[:5]
    for r in bot:
        print(f"    {r['ticker']:8s}  n={int(r['n']):2d}  acc={r['acc']:.0%}  mae=${r['mae']:.2f}")

    print("\n  By confidence bucket:")
    for r in m["by_confidence"]:
        print(f"    {str(r['conf_bin']):10s}  n={int(r['n']):3d}  acc={r['acc']:.1%}")

    print("\n  By macro mode:")
    for r in m["by_macro_mode"]:
        print(f"    {r['macro_mode']:14s}  n={int(r['n']):3d}  acc={r['acc']:.1%}")

    print("\n  By catalyst (post_catalyst<=5):")
    for r in m["by_catalyst"]:
        flag = "with_cat" if r["has_catalyst"] else "no_cat"
        print(f"    {flag:10s}  n={int(r['n']):3d}  acc={r['acc']:.1%}")

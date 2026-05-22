"""2022 약세장 backtest — 검증된 룰의 robustness 검증.

기간: 2022-01-03 ~ 2022-12-30 (SPY -19%, QQQ -33%)

검증 룰 (강세장 100일 sample에서 발견):
  1. Verified Rules + Position Sizing — 1d Sharpe 2.51
  2. 모듈 합의 n_bull≥5 — 65.7% win, +4.89%
  3. IV<30% — 10d 93% win
  4. 1d × BEAR macro + 시스템 신호 — 64% win
  5. Sweet spot (contrarian) — 14개 robust filter

한계: yfinance/CBOE options chain은 historical 제공 X (현재만).
옵션 모듈 score는 약식 (options_data_unavailable=True로 처리),
다른 10개 모듈은 정확히 historical 데이터로 계산.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def collect_2022(tickers: List[str], n_snapshots: int = 60, verbose: bool = True):
    """2022 약세장 데이터 수집."""
    from ..data.insider import get_insider_activity
    from ..data.price_feed import get_daily_ohlcv
    from ..system import StockPredictionSystem
    from .walk_forward import build_data_at

    system = StockPredictionSystem()
    rows = []

    # 2022 영업일 균등 분포로 snapshot
    start, end = date(2022, 1, 3), date(2022, 12, 30)
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
        except Exception as e:
            if verbose:
                print(f"  SKIP fetch: {e}", flush=True)
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
            except Exception as e:
                if verbose:
                    print(f"  {snap} fail: {e}", flush=True)
                continue

            ss_dict = getattr(pred, "sweet_spot", None) or {}
            mc_dict = getattr(pred, "module_consensus", None) or {}
            row = {
                "ticker": ticker,
                "as_of": snap.isoformat(),
                "cur": round(actual_today, 2),
                "composite_score": round(pred.composite_score, 3),
                "confidence": round(pred.confidence, 3),
                "ev_pct_5d": round((pred.expected_value - actual_today) / actual_today * 100, 3),
                "macro_mode": (data.get("macro_breadth") or {}).get("mode", "?"),
                "recommended_size": getattr(pred, "recommended_size", 0.0),
                "sweet_spot_active": bool(ss_dict.get("active", False)),
                "module_tier": mc_dict.get("tier", "noise"),
            }
            # raw 모듈 score
            for name, m in pred.modules.items():
                row[f"mod_{name}"] = round(m.score, 3)
            # actual returns
            for h in (1, 3, 5, 10):
                ret = (actuals[h] - actual_today) / actual_today * 100
                row[f"actual_ret_{h}d"] = round(ret, 3)
            rows.append(row)

    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame):
    """2022 약세장 데이터로 5개 검증 룰 평가."""
    if df.empty:
        print("no data")
        return

    print(f"\n=== 전체 {len(df)} events ({df['ticker'].nunique()} tickers) ===")
    print(f"period: {df['as_of'].min()} ~ {df['as_of'].max()}")
    print(f"\nmacro 분포:")
    print(df['macro_mode'].value_counts().to_string())

    print(f"\nactual returns (baseline):")
    for h in [1, 5, 10]:
        col = f"actual_ret_{h}d"
        s = df[col].dropna()
        print(f"  {h}d: win {(s>0).mean():.1%}, avg {s.mean():+.2f}%, n={len(s)}")

    # 모듈 카운팅
    mod_cols = [c for c in df.columns if c.startswith("mod_")]
    df['n_bull'] = df[mod_cols].apply(lambda r: sum(1 for v in r if v > 1), axis=1)
    df['n_bear'] = df[mod_cols].apply(lambda r: sum(1 for v in r if v < -1), axis=1)

    # ─── 룰 1: Verified Rules + Position Sizing ───
    print("\n\n=== 룰 1: Verified Rules + Sizing (1d horizon) ===")
    # Direction: BEAR/CHOPPY에서만 시스템 신호, 그 외 baseline long
    # Sizing: macro × horizon × signal 강도
    s = df.dropna(subset=['actual_ret_1d']).copy()

    def dir_fn(macro, ev, h=1):
        m = (macro or "?").upper()
        if h == 1 and m in ("BEAR", "CHOPPY"):
            if abs(ev) < 0.3:
                return 0
            return 1 if ev > 0 else 0
        return 1

    def size_fn(macro, ev, conf, h=1):
        m = (macro or "?").upper()
        sig_strong = abs(ev) > 0.5 and conf >= 0.5
        if h == 1:
            if m == "BEAR":
                return 1.5 if sig_strong else 0.5
            if m == "CHOPPY":
                return 1.2 if sig_strong else 0.4
            if m in ("BULL", "STRONG_BULL", "STRONG_BEAR"):
                return 0.8
            return 0.4
        return 1.0

    s['direction_1d'] = s.apply(lambda r: dir_fn(r['macro_mode'], r['ev_pct_5d']), axis=1)
    s['size_1d'] = s.apply(lambda r: size_fn(r['macro_mode'], r['ev_pct_5d'], r['confidence']), axis=1)
    s['pnl_1d_verified'] = s['direction_1d'] * s['size_1d'] * s['actual_ret_1d']

    base_pnl_1d = s['actual_ret_1d'].mean()
    base_total = s['actual_ret_1d'].sum()
    ver_pnl = s['pnl_1d_verified'].mean()
    ver_total = s['pnl_1d_verified'].sum()
    ver_win = (s['pnl_1d_verified'] > 0).mean()
    base_win = (s['actual_ret_1d'] > 0).mean()
    ver_std = s['pnl_1d_verified'].std()
    sharpe = (ver_pnl / ver_std * np.sqrt(252)) if ver_std > 0 else 0
    base_std = s['actual_ret_1d'].std()
    base_sharpe = (base_pnl_1d / base_std * np.sqrt(252)) if base_std > 0 else 0

    print(f"  baseline (always long): win {base_win:.1%}, avg {base_pnl_1d:+.3f}%, total {base_total:+.1f}%, Sharpe {base_sharpe:.2f}")
    print(f"  verified+sizing:        win {ver_win:.1%}, avg {ver_pnl:+.3f}%, total {ver_total:+.1f}%, Sharpe {sharpe:.2f}")
    print(f"  → uplift: Sharpe {sharpe - base_sharpe:+.2f}, total {ver_total - base_total:+.1f}%")

    # ─── 룰 2: 모듈 합의 n_bull≥5 ───
    print("\n=== 룰 2: 모듈 합의 n_bull≥5 (5d) ===")
    sub = df.dropna(subset=['actual_ret_5d'])
    base = (sub['actual_ret_5d'] > 0).mean()
    base_avg = sub['actual_ret_5d'].mean()
    high_consensus = sub[sub['n_bull'] >= 5]
    if len(high_consensus) > 0:
        win = (high_consensus['actual_ret_5d'] > 0).mean()
        avg = high_consensus['actual_ret_5d'].mean()
        print(f"  n=  {len(high_consensus)} | win {win:.1%} (baseline {base:.1%}) | avg {avg:+.2f}% (baseline {base_avg:+.2f}%)")
    else:
        print("  n_bull≥5 sample 없음")

    # ─── 룰 3: IV<30% ─── (옵션 데이터 약함이라 skip 또는 추정)
    print("\n=== 룰 3: IV<30% ===")
    print("  옵션 chain historical 없음 → 약식 평가 X (강세장 sample에서 검증된 룰)")

    # ─── 룰 4: 1d × BEAR macro + 시스템 신호 ───
    print("\n=== 룰 4: 1d × BEAR + 시스템 신호 (raw direction) ===")
    bear = sub[sub['macro_mode'].isin(['BEAR', 'STRONG_BEAR'])]
    if len(bear):
        # 시스템 신호 따라가기 (raw direction)
        bear_buy = bear[bear['ev_pct_5d'] > 0.3]
        bear_sell = bear[bear['ev_pct_5d'] < -0.3]
        print(f"  전체 BEAR/STRONG_BEAR: n={len(bear)}, baseline 1d win {(bear['actual_ret_1d']>0).mean():.1%}")
        if len(bear_buy):
            win = (bear_buy['actual_ret_1d'] > 0).mean()
            avg = bear_buy['actual_ret_1d'].mean()
            print(f"  BEAR + ev>+0.3% (long): n={len(bear_buy)}, win {win:.1%}, avg {avg:+.3f}%")
        if len(bear_sell):
            # short = -actual_ret 만약 system short signal 정말 alpha?
            win = (bear_sell['actual_ret_1d'] < 0).mean()  # actual 음수면 short 이김
            avg = -bear_sell['actual_ret_1d'].mean()
            print(f"  BEAR + ev<-0.3% (system short 검증 — 영구 금지 룰): n={len(bear_sell)}, 가상 short win {win:.1%}, avg {avg:+.3f}%")

    # ─── 룰 5: Sweet spot ───
    print("\n=== 룰 5: Sweet spot (contrarian) ===")
    sweet = df[df['sweet_spot_active'] == True]  # noqa
    print(f"  sweet spot 적중 sample: n={len(sweet)}")
    if len(sweet):
        for h in [1, 5, 10]:
            ss = sweet.dropna(subset=[f'actual_ret_{h}d'])
            if len(ss):
                win = (ss[f'actual_ret_{h}d'] > 0).mean()
                avg = ss[f'actual_ret_{h}d'].mean()
                print(f"  {h}d: win {win:.1%}, avg {avg:+.2f}% (n={len(ss)})")

    # ─── module_tier별 ───
    print("\n=== Module tier 분포 + outcome (5d) ===")
    for tier in df['module_tier'].unique():
        sub_t = df[(df['module_tier'] == tier)].dropna(subset=['actual_ret_5d'])
        if len(sub_t) < 5:
            continue
        win = (sub_t['actual_ret_5d'] > 0).mean()
        avg = sub_t['actual_ret_5d'].mean()
        print(f"  {tier:<30s} n={len(sub_t):>4d}, win {win:.1%}, avg {avg:+.2f}%")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers",
        default="NVDA,AMD,TSLA,AAPL,MSFT,META,AMZN,NFLX,GOOGL,MSTR",
    )
    parser.add_argument("--snapshots", type=int, default=60)
    parser.add_argument("--sanity", action="store_true",
                        help="3종 × 30일 빠른 sanity")
    args = parser.parse_args()

    if args.sanity:
        tickers = ["NVDA", "AMD", "MSFT"]
        snapshots = 30
    else:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        snapshots = args.snapshots

    print(f"tickers ({len(tickers)}): {tickers}")
    print(f"snapshots target: {snapshots}")

    df = collect_2022(tickers, n_snapshots=snapshots)
    if df.empty:
        print("no events")
        return

    out = Path(__file__).resolve().parents[2] / "data" / "results" / "bear_2022.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n[saved] {out} ({len(df)} events)")

    analyze(df)


if __name__ == "__main__":
    main()

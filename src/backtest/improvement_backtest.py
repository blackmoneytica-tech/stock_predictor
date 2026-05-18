"""예측 성공률 향상 백테스트 — 다양한 필터/보정 조합 검증.

비교 후보:
A) raw                — 시스템 raw pred_ret
B) bias_corrected     — pred_ret - rolling_bias(last 100 예측)
C) magnitude filter   — |pred_ret| > threshold일 때만 trade
D) macro_aligned      — pred 방향과 macro_mode 방향 일치만
E) score_strong       — composite_score 절대값 큰 것만
F) multi_horizon      — 1d/3d/5d 같은 방향 (3 horizons 분석 필요)
G) combo              — bias_corrected + magnitude + macro_aligned

각 strategy에 대해:
- n_trades (trade signal 발생 횟수)
- win_rate (long → actual>0, short → actual<0)
- mean_pnl (per trade, %)
- total_pnl (모든 trade 합)
- baseline (always long) 대비 alpha
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


def run_backtest_data_collection(
    tickers: List[str],
    snapshot_dates: List[date],
    horizons: List[int] = [1, 3, 5],
    verbose: bool = True,
) -> pd.DataFrame:
    """multi-horizon 데이터 수집 — N 종목 × M 시점 × K horizon."""
    system = StockPredictionSystem()
    results = []

    # Prefetch
    if verbose:
        print("[prefetch] ETFs + macro...", flush=True)
    earliest_etf = min(snapshot_dates) - timedelta(days=400)
    latest_etf = max(snapshot_dates) + timedelta(days=max(horizons) + 5)
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
            latest = max(snapshot_dates) + timedelta(days=max(horizons) + 5)
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
            if at_or_before.empty:
                continue
            as_of_idx = at_or_before.index[-1]
            as_of_date = as_of_idx.date()
            actual_today = float(full.loc[as_of_idx, "close"])

            # actuals for each horizon
            actuals = {}
            for h in horizons:
                # 영업일 기준 h개 뒤
                future = full[full.index > ts_snap]
                if len(future) < h:
                    actuals[h] = None
                else:
                    fut_idx = future.index[h - 1]
                    actuals[h] = float(full.loc[fut_idx, "close"])
            if any(v is None for v in actuals.values()):
                continue

            # build data once
            try:
                base_data = build_data_at(
                    ticker, as_of_date,
                    horizon_days=horizons[0],
                    use_macro=True,
                    insider_cache=insider_cache,
                )
            except Exception as e:
                if verbose:
                    print(f"  {as_of_date} build fail: {e}", flush=True)
                continue

            # 같은 data로 horizon만 바꿔서 분석
            preds_by_h = {}
            for h in horizons:
                try:
                    pred = system.analyze(ticker, horizon_days=h, data=base_data)
                    preds_by_h[h] = pred
                except Exception:
                    preds_by_h[h] = None
            if any(v is None for v in preds_by_h.values()):
                continue

            row = {
                "ticker": ticker,
                "as_of": as_of_date.isoformat(),
                "cur": round(actual_today, 2),
                "macro_mode": (base_data.get("macro_breadth") or {}).get("mode", "?"),
                "composite_score": round(preds_by_h[horizons[0]].composite_score, 3),
                "confidence": round(preds_by_h[horizons[0]].confidence, 3),
                "directional_bias": preds_by_h[horizons[0]].directional_bias,
            }
            for h in horizons:
                pred = preds_by_h[h]
                pred_ret = (pred.expected_value - pred.current_price) / pred.current_price * 100
                actual_ret = (actuals[h] - actual_today) / actual_today * 100
                row[f"pred_ret_{h}d"] = round(pred_ret, 3)
                row[f"actual_ret_{h}d"] = round(actual_ret, 3)
                row[f"dir_correct_{h}d"] = (pred_ret > 0) == (actual_ret > 0)

            results.append(row)

    return pd.DataFrame(results)


def evaluate_strategies(df: pd.DataFrame, horizon: int = 1) -> pd.DataFrame:
    """다양한 strategy의 win_rate / pnl / n_trades."""
    if df.empty:
        return pd.DataFrame()

    pred_col = f"pred_ret_{horizon}d"
    actual_col = f"actual_ret_{horizon}d"
    s = df.copy()
    s = s.dropna(subset=[pred_col, actual_col]).reset_index(drop=True)
    s['date'] = pd.to_datetime(s['as_of'])
    s = s.sort_values('date').reset_index(drop=True)

    # rolling bias = 직전 100 예측 평균 (look-ahead 방지)
    s['rolling_bias'] = s[pred_col].shift(1).rolling(100, min_periods=20).mean()
    s['pred_debiased'] = s[pred_col] - s['rolling_bias'].fillna(0)

    # multi-horizon agreement
    if 'pred_ret_1d' in s.columns and 'pred_ret_3d' in s.columns and 'pred_ret_5d' in s.columns:
        s['agree_up'] = (s['pred_ret_1d'] > 0) & (s['pred_ret_3d'] > 0) & (s['pred_ret_5d'] > 0)
        s['agree_dn'] = (s['pred_ret_1d'] < 0) & (s['pred_ret_3d'] < 0) & (s['pred_ret_5d'] < 0)
    else:
        s['agree_up'] = False
        s['agree_dn'] = False

    macro_bull = s['macro_mode'].isin(['BULL', 'STRONG_BULL'])
    macro_bear = s['macro_mode'].isin(['BEAR', 'STRONG_BEAR'])

    def trade(name, signal, side):
        """signal: bool series. side: 'long' or 'short' or array-like of ±1."""
        sub = s[signal].copy()
        if len(sub) == 0:
            return None
        if isinstance(side, str):
            if side == 'long':
                sub['pnl'] = sub[actual_col]
            elif side == 'short':
                sub['pnl'] = -sub[actual_col]
            else:
                raise ValueError(f"unknown side string: {side}")
        else:
            side_arr = np.asarray(side)
            if len(side_arr) == len(s):
                # signal mask와 같은 길이의 ±1 배열
                sig_arr = side_arr[signal.values]
            else:
                sig_arr = side_arr
            sub['pnl'] = sig_arr * sub[actual_col].values
        return {
            "strategy": name,
            "n": len(sub),
            "win_rate": float((sub['pnl'] > 0).mean()),
            "avg_pnl": float(sub['pnl'].mean()),
            "total_pnl": float(sub['pnl'].sum()),
            "median_pnl": float(sub['pnl'].median()),
        }

    rows = []

    # 0) baseline: 매번 long
    rows.append({
        "strategy": "baseline (always long)",
        "n": len(s),
        "win_rate": float((s[actual_col] > 0).mean()),
        "avg_pnl": float(s[actual_col].mean()),
        "total_pnl": float(s[actual_col].sum()),
        "median_pnl": float(s[actual_col].median()),
    })

    # A) raw direction
    direction_raw = np.sign(s[pred_col])
    rows.append(trade("A_raw_direction", s[pred_col].abs() > 0.1, direction_raw))

    # B) bias-corrected direction
    direction_debiased = np.sign(s['pred_debiased'])
    rows.append(trade("B_bias_corrected", s['pred_debiased'].abs() > 0.3, direction_debiased))

    # C) magnitude filter (raw)
    for thr in [0.5, 1.0, 2.0]:
        rows.append(trade(
            f"C_raw_long_>{thr}%", s[pred_col] > thr, 'long',
        ))
        rows.append(trade(
            f"C_raw_short_<-{thr}%", s[pred_col] < -thr, 'short',
        ))

    # D) magnitude filter (debiased)
    for thr in [0.3, 0.5, 1.0]:
        rows.append(trade(
            f"D_debiased_long_>{thr}%", s['pred_debiased'] > thr, 'long',
        ))
        rows.append(trade(
            f"D_debiased_short_<-{thr}%", s['pred_debiased'] < -thr, 'short',
        ))

    # E) macro-aligned
    rows.append(trade("E_long_in_BULL", macro_bull, 'long'))
    rows.append(trade(
        "E_long_in_BULL_w_raw>0.5", macro_bull & (s[pred_col] > 0.5), 'long',
    ))
    rows.append(trade(
        "E_long_in_BULL_w_debiased>0.3",
        macro_bull & (s['pred_debiased'] > 0.3), 'long',
    ))
    rows.append(trade(
        "E_short_in_BEAR_w_raw<-0.5",
        macro_bear & (s[pred_col] < -0.5), 'short',
    ))

    # F) multi-horizon agreement
    rows.append(trade("F_agree_up_long", s['agree_up'], 'long'))
    rows.append(trade("F_agree_dn_short", s['agree_dn'], 'short'))

    # G) combo: bias-corrected + magnitude + macro
    rows.append(trade(
        "G_combo_long",
        macro_bull & (s['pred_debiased'] > 0.3),
        'long',
    ))
    rows.append(trade(
        "G_combo_short",
        macro_bear & (s['pred_debiased'] < -0.3),
        'short',
    ))

    # H) score-based
    rows.append(trade(
        "H_score>2_long", s['composite_score'] > 2, 'long',
    ))
    rows.append(trade(
        "H_score<-2_short", s['composite_score'] < -2, 'short',
    ))

    out = pd.DataFrame([r for r in rows if r is not None])
    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers",
        default="NVDA,AMD,TSLA,AAPL,MSFT,GOOGL,META,AMZN,CRCL,MSTR,COIN,HOOD,NFLX,PLTR,SMCI",
    )
    parser.add_argument("--days", type=int, default=40)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    if args.end:
        end_date = date.fromisoformat(args.end)
    else:
        end_date = date.today() - timedelta(days=8)  # 5d horizon이므로 더 보수적

    snaps = []
    d = end_date
    while len(snaps) < args.days:
        if d.weekday() < 5:
            snaps.append(d)
        d -= timedelta(days=1)
    snaps.sort()

    print(f"tickers ({len(tickers)}): {tickers}", flush=True)
    print(f"snapshots: {snaps[0]} ~ {snaps[-1]} ({len(snaps)} BD)", flush=True)
    print(f"horizons: 1d, 3d, 5d (multi-horizon agreement 측정)", flush=True)

    df = run_backtest_data_collection(tickers, snaps, horizons=[1, 3, 5])
    if df.empty:
        print("결과 없음")
        return

    from pathlib import Path
    out = Path(__file__).resolve().parents[2] / "data" / "results" / "improvement_data.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n[saved] {out} ({len(df)} rows)\n", flush=True)

    for h in [1, 3, 5]:
        print(f"\n{'=' * 70}")
        print(f"=== Horizon {h}d strategy 비교 ===")
        print(f"{'=' * 70}")
        eval_df = evaluate_strategies(df, horizon=h)
        eval_df['win_rate'] = eval_df['win_rate'].apply(lambda x: f"{x:.1%}")
        eval_df['avg_pnl'] = eval_df['avg_pnl'].apply(lambda x: f"{x:+.2f}%")
        eval_df['total_pnl'] = eval_df['total_pnl'].apply(lambda x: f"{x:+.1f}%")
        eval_df['median_pnl'] = eval_df['median_pnl'].apply(lambda x: f"{x:+.2f}%")
        print(eval_df.to_string(index=False))


if __name__ == "__main__":
    main()

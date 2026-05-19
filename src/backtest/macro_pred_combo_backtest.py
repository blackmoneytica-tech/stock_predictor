"""매크로 탭 vs 예측 탭 vs 결합 — alpha 검증.

매크로 시그널 (trade-journal 매크로 탭 규칙 그대로 재현):
  - drawdown (52w high 기준): strong_buy ≤ -20%, deep -20~-15%,
    trap -15~-10%, buy_zone -10~-3%, safe > -3%
  - RS grade (SPY 대비 20일 초과수익): very_strong ≥ +5%, strong 0~+5%,
    weak -5~0%, very_weak < -5%
  - 매크로 BUY signal = cat in (strong_buy, deep, buy_zone) AND rs_grade in (strong, very_strong)

예측 시그널 (시스템 11모듈 ev_pct):
  - 예측 BUY = ev_pct > 0.3%
  - 예측 SELL = ev_pct < -0.3%

비교 전략:
  A. baseline always long
  B. 매크로만 (cat in buy/deep/strong_buy → long)
  C. 매크로 + RS (cat buy + rs strong/very_strong → long)
  D. 예측만 (ev_pct > 0.3 → long)
  E. AND (매크로 buy AND 예측 buy → long, concordance)
  F. OR (매크로 또는 예측 buy → long)
  G. 매크로 우선 + 예측 무시
  H. 예측 우선 + 매크로 trap이면 skip
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def compute_macro_signal_per_snapshot(
    ohlcv: pd.DataFrame, spy_ohlcv: pd.DataFrame, snap_date,
) -> Dict:
    """단일 시점에서 매크로 신호 계산."""
    ts_snap = pd.Timestamp(snap_date)
    hist = ohlcv[ohlcv.index <= ts_snap]
    spy_hist = spy_ohlcv[spy_ohlcv.index <= ts_snap]
    if len(hist) < 60 or len(spy_hist) < 60:
        return None

    cur = float(hist["close"].iloc[-1])
    # 52주 high
    wk52 = hist.tail(252)
    wk52_high = float(wk52["high"].max())
    dd_pct = (cur - wk52_high) / wk52_high * 100

    # 20일 ticker chg vs SPY chg (relChg20d)
    if len(hist) >= 21:
        t_chg20 = (cur - float(hist["close"].iloc[-21])) / float(hist["close"].iloc[-21]) * 100
        spy_cur = float(spy_hist["close"].iloc[-1])
        spy_20 = float(spy_hist["close"].iloc[-21]) if len(spy_hist) >= 21 else spy_cur
        spy_chg20 = (spy_cur - spy_20) / spy_20 * 100
        rel_chg20 = t_chg20 - spy_chg20
    else:
        rel_chg20 = 0

    # 5일 chg
    if len(hist) >= 6:
        t_chg5 = (cur - float(hist["close"].iloc[-6])) / float(hist["close"].iloc[-6]) * 100
        spy_5 = float(spy_hist["close"].iloc[-6]) if len(spy_hist) >= 6 else float(spy_hist["close"].iloc[-1])
        spy_chg5 = (float(spy_hist["close"].iloc[-1]) - spy_5) / spy_5 * 100
        rel_chg5 = t_chg5 - spy_chg5
    else:
        rel_chg5 = 0

    # Category (drawdown 기반)
    if dd_pct <= -20:
        cat = "strong_buy"
    elif dd_pct <= -15:
        cat = "deep"
    elif dd_pct <= -10:
        cat = "trap"
    elif dd_pct <= -3:
        cat = "buy_zone"
    else:
        cat = "safe"

    # RS grade (rel_chg20d 기반)
    if rel_chg20 >= 5:
        rs_grade = "very_strong"
    elif rel_chg20 >= 0:
        rs_grade = "strong"
    elif rel_chg20 >= -5:
        rs_grade = "weak"
    else:
        rs_grade = "very_weak"

    return {
        "dd_pct": round(dd_pct, 2),
        "cat": cat,
        "rel_chg5": round(rel_chg5, 2),
        "rel_chg20": round(rel_chg20, 2),
        "rs_grade": rs_grade,
    }


def collect_macro_signals(df_predictions: pd.DataFrame) -> pd.DataFrame:
    """기존 module_scores.parquet의 (ticker, as_of)에 매크로 시그널 추가."""
    from ..data.price_feed import get_daily_ohlcv

    tickers = df_predictions['ticker'].unique()
    dates = pd.to_datetime(df_predictions['as_of'])
    earliest = (dates.min() - timedelta(days=400)).date()
    latest = (dates.max() + timedelta(days=10)).date()

    print("[fetch] SPY...", flush=True)
    spy = get_daily_ohlcv("SPY", earliest, latest)
    if spy.empty:
        print("SPY empty — abort")
        return df_predictions

    ohlcv_cache = {}
    print(f"[fetch] {len(tickers)} tickers OHLCV...", flush=True)
    for t in tickers:
        try:
            ohlcv_cache[t] = get_daily_ohlcv(t, earliest, latest)
        except Exception as e:
            print(f"  {t} skip: {e}")
            ohlcv_cache[t] = pd.DataFrame()

    print("[compute] macro signals per (ticker, snap)...", flush=True)
    new_rows = []
    for _, row in df_predictions.iterrows():
        t = row['ticker']
        snap = date.fromisoformat(row['as_of']) if isinstance(row['as_of'], str) else row['as_of']
        ohlcv = ohlcv_cache.get(t)
        if ohlcv is None or ohlcv.empty:
            continue
        sig = compute_macro_signal_per_snapshot(ohlcv, spy, snap)
        if sig is None:
            continue
        merged = dict(row)
        merged.update(sig)
        new_rows.append(merged)

    return pd.DataFrame(new_rows)


def simulate_combo_strategies(df: pd.DataFrame, horizon: int = 5):
    """6+개 전략의 win rate / PnL / Sharpe 비교."""
    actual_col = f"actual_ret_{horizon}d"
    if actual_col not in df.columns:
        # ev_pct_5d만 있는 케이스, actual은 actual_ret_5d
        actual_col = f"actual_ret_5d"
    s = df.dropna(subset=[actual_col, 'cat', 'rs_grade']).copy()
    if s.empty:
        print("No data")
        return None
    s['date'] = pd.to_datetime(s['as_of'])
    s = s.sort_values('date').reset_index(drop=True)

    # 매크로 buy 카테고리
    macro_cats_buy = ['strong_buy', 'deep', 'buy_zone']
    rs_strong = ['strong', 'very_strong']

    # 신호 정의
    s['macro_buy_dd'] = s['cat'].isin(macro_cats_buy)
    s['macro_buy_dd_rs'] = s['macro_buy_dd'] & s['rs_grade'].isin(rs_strong)
    s['macro_trap'] = s['cat'] == 'trap'
    s['pred_buy'] = s['ev_pct_5d'] > 0.3
    s['pred_sell'] = s['ev_pct_5d'] < -0.3

    def strat(name, mask):
        sub = s[mask].copy()
        if len(sub) == 0:
            return {"name": name, "n": 0}
        sub['pnl'] = sub[actual_col]  # long-only PnL
        return {
            "name": name,
            "n": len(sub),
            "win_rate": round(float((sub['pnl'] > 0).mean()), 3),
            "avg_pnl": round(float(sub['pnl'].mean()), 3),
            "total_pnl": round(float(sub['pnl'].sum()), 1),
            "median_pnl": round(float(sub['pnl'].median()), 3),
            "std": round(float(sub['pnl'].std()), 3),
            "sharpe": round(float(sub['pnl'].mean() / sub['pnl'].std() * np.sqrt(252 / horizon))
                            if sub['pnl'].std() > 0 else 0, 2),
        }

    results = [
        strat("A. baseline always_long", pd.Series([True] * len(s))),
        strat("B. macro only (DD buy cat)", s['macro_buy_dd']),
        strat("C. macro + RS strong",       s['macro_buy_dd_rs']),
        strat("D. pred only (ev>0.3%)",     s['pred_buy']),
        strat("E. AND (macro & pred)",      s['macro_buy_dd'] & s['pred_buy']),
        strat("F. OR (macro | pred)",       s['macro_buy_dd'] | s['pred_buy']),
        strat("G. macro priority (macro & not pred_sell)",
              s['macro_buy_dd'] & ~s['pred_sell']),
        strat("H. pred-first (pred & not trap)",
              s['pred_buy'] & ~s['macro_trap']),
        strat("I. macro + RS + pred (가장 엄격)",
              s['macro_buy_dd_rs'] & s['pred_buy']),
        # disconcord 분석
        strat("X1. concord (macro buy & pred buy)",
              s['macro_buy_dd'] & s['pred_buy']),
        strat("X2. macro buy + pred SELL (disconcord)",
              s['macro_buy_dd'] & s['pred_sell']),
        strat("X3. macro safe/trap + pred buy",
              ~s['macro_buy_dd'] & s['pred_buy']),
        strat("X4. macro trap + pred buy (위험!)",
              s['macro_trap'] & s['pred_buy']),
    ]
    return pd.DataFrame([r for r in results if r])


def main():
    """기존 module_scores.parquet 활용."""
    pred_path = Path("data/results/module_scores.parquet")
    if not pred_path.exists():
        print(f"ERROR: {pred_path} 없음. full_simulation.py --collect 먼저 실행")
        return
    df = pd.read_parquet(pred_path)
    print(f"loaded {len(df)} prediction rows from {pred_path}")
    print(f"date range: {df['as_of'].min()} ~ {df['as_of'].max()}")
    print(f"tickers: {df['ticker'].nunique()}")

    # 매크로 시그널 추가
    combined_path = Path("data/results/macro_pred_combined.parquet")
    if combined_path.exists():
        print(f"[loaded] {combined_path}")
        combined = pd.read_parquet(combined_path)
    else:
        combined = collect_macro_signals(df)
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(combined_path, index=False)
        print(f"[saved] {combined_path} ({len(combined)} rows)")

    # 매크로 카테고리 분포
    print(f"\n=== Macro category 분포 ===")
    print(combined['cat'].value_counts().to_dict())
    print(f"\n=== RS grade 분포 ===")
    print(combined['rs_grade'].value_counts().to_dict())

    # cross-tab
    print(f"\n=== category × RS grade ===")
    ct = pd.crosstab(combined['cat'], combined['rs_grade'])
    print(ct)

    # 각 horizon별 strategy 비교
    for h in [1, 5]:
        print(f"\n{'='*78}")
        print(f"=== Horizon {h}d combo simulation ===")
        print('='*78)
        results = simulate_combo_strategies(combined, horizon=h)
        if results is not None:
            print(results.to_string(index=False))

            # 분석 메시지
            base = results[results['name'].str.contains('baseline')].iloc[0]
            print(f"\n  baseline (always long): win {base['win_rate']:.1%}, avg {base['avg_pnl']:+.2f}%, Sharpe {base['sharpe']:.2f}")
            for name in ["B. macro only (DD buy cat)", "D. pred only (ev>0.3%)",
                         "E. AND (macro & pred)", "I. macro + RS + pred"]:
                r = results[results['name'] == name]
                if len(r):
                    r = r.iloc[0]
                    delta_win = r['win_rate'] - base['win_rate']
                    delta_pnl = r['avg_pnl'] - base['avg_pnl']
                    print(f"  {name}: win {r['win_rate']:.1%} ({delta_win:+.1%}p), "
                          f"avg {r['avg_pnl']:+.2f}% ({delta_pnl:+.2f}p), "
                          f"Sharpe {r['sharpe']:.2f}, n={r['n']}")


if __name__ == "__main__":
    main()

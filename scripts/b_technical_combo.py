"""B: F5 + technical signal 결합 백테스트.

가설: F5 단독은 213-univ 에서 alpha 거의 0. 다른 OHLCV 기반 signal 과 결합 시 alpha?

추가 signal (모두 OHLCV 만 사용 — F1 module score 와 달리 5y backtest 가능):
  T1  RSI < 35           (oversold momentum)
  T2  5d ret < -5%       (recent drop)
  T3  20d ret < -10%     (medium drawdown)
  T4  price < SMA50      (downtrend — mean reversion 후보)
  T5  price > SMA200     (long-term uptrend filter — quality)
  T6  vol surge          (today vol > 1.5× 20d avg)
  T7  52w low zone       (price within 10% of 52w low)
  T8  BB squeeze         (BB width < 20th percentile)
  T9  MA crossover up    (SMA50 > SMA200, golden cross)
  T10 close > VAH-5%     (price approaching VAH — momentum)

각 single signal + F5 결합, 그리고 multi-signal stacking 측정.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats
from datetime import date, timedelta

from src.data.price_feed import get_daily_ohlcv
from src.data.universe import get_universe


def section(t): print(f"\n{'='*65}\n{t}\n{'='*65}")


def compute_technical_signals(hist):
    """OHLCV 만으로 가능한 모든 technical signal 일괄 계산."""
    df = hist.copy()
    closes = df["close"]
    highs = df["high"]
    lows = df["low"]
    vols = df["volume"]

    # RSI 14
    delta = closes.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    # Returns
    df["ret_5d"] = closes.pct_change(5) * 100
    df["ret_20d"] = closes.pct_change(20) * 100

    # MAs
    df["sma50"] = closes.rolling(50).mean()
    df["sma200"] = closes.rolling(200).mean()

    # Volume surge
    df["vol_20d_avg"] = vols.rolling(20).mean()
    df["vol_ratio"] = vols / df["vol_20d_avg"]

    # 52w low / high
    df["52w_low"] = lows.rolling(252).min()
    df["52w_high"] = highs.rolling(252).max()
    df["dist_52w_low"] = (closes - df["52w_low"]) / df["52w_low"]

    # Bollinger Band width
    bb_mid = closes.rolling(20).mean()
    bb_std = closes.rolling(20).std()
    df["bb_width"] = (bb_std * 4) / bb_mid  # 2σ 양쪽 = 4σ
    df["bb_width_pct"] = df["bb_width"].rolling(252).rank(pct=True)

    return df


def compute_hv_iv_rank(hist):
    log_ret = np.log(hist["close"] / hist["close"].shift(1))
    rolling_hv = log_ret.rolling(20).std() * np.sqrt(252)
    ranks = []
    arr = rolling_hv.values
    for i in range(len(arr)):
        start = max(0, i - 251)
        win = arr[start:i + 1]
        win = win[~np.isnan(win)]
        if len(win) < 30:
            ranks.append(np.nan)
        else:
            ranks.append(float((win < win[-1]).mean()))
    return pd.Series(ranks, index=hist.index)


def build_panel():
    """모든 213-종 universe 에 대해 technical signal + iv_rank + forward returns 계산."""
    section("0. 213 universe — OHLCV + technical signal panel 생성")
    universe = get_universe("full")
    print(f"  universe: {len(universe)} tickers")
    rows = []
    for i, t in enumerate(universe):
        try:
            start = date.today() - timedelta(days=int(5 * 365) + 30)
            hist = get_daily_ohlcv(t, start=start, end=date.today())
            if hist is None or len(hist) < 250:
                continue
            tech = compute_technical_signals(hist)
            tech["iv_rank"] = compute_hv_iv_rank(hist)
            # forward returns
            for h in [1, 5, 10]:
                tech[f"fwd_{h}d_pct"] = (tech["close"].shift(-h) / tech["close"] - 1) * 100
            tech["ticker"] = t
            tech["date"] = tech.index
            rows.append(tech)
            if (i + 1) % 30 == 0:
                print(f"    {i+1}/{len(universe)} loaded")
        except Exception as e:
            pass
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.dropna(subset=["iv_rank", "rsi", "ret_5d", "ret_20d", "fwd_5d_pct"])
    print(f"  panel: {len(panel)} rows")
    return panel


def add_signal_flags(p):
    """각 row 에 technical signal flag (binary)."""
    p["T1_rsi35"] = p["rsi"] < 35
    p["T2_5d_drop"] = p["ret_5d"] < -5
    p["T3_20d_drawdown"] = p["ret_20d"] < -10
    p["T4_below_sma50"] = p["close"] < p["sma50"]
    p["T5_above_sma200"] = p["close"] > p["sma200"]
    p["T6_vol_surge"] = p["vol_ratio"] > 1.5
    p["T7_52w_low_zone"] = p["dist_52w_low"] < 0.10
    p["T8_bb_squeeze"] = p["bb_width_pct"] < 0.20
    p["T9_golden_cross"] = p["sma50"] > p["sma200"]
    p["F5"] = p["iv_rank"] < 0.20
    return p


def block_stats(rets, h_days=5):
    rets = pd.Series(rets).dropna()
    if len(rets) < 5:
        return None
    wins = int((rets > 0).sum())
    win = wins / len(rets)
    avg = rets.mean()
    std = rets.std()
    sharpe = (avg / std * np.sqrt(252 / h_days)) if std > 0 else 0
    p = stats.binomtest(wins, len(rets), 0.5, alternative="greater").pvalue
    return dict(n=len(rets), win=win, avg=avg, sharpe=sharpe, p=p)


def single_signal_test(p):
    section("1. 단독 signal alpha (5d)")
    print(f"  baseline (전체): n={len(p)}  avg={p['fwd_5d_pct'].mean():+.2f}%")
    cols = [c for c in p.columns if c.startswith("T") or c == "F5"]
    print(f"\n  {'signal':<22} {'active n':<10} {'win%':<8} {'avg%':<8} {'Sharpe':<7} {'p<.5':<7}")
    for c in cols:
        sub = p[p[c]]["fwd_5d_pct"]
        st = block_stats(sub)
        if st: print(f"  {c:<22} {st['n']:<10} {st['win']:.1%}    {st['avg']:+.2f}%   {st['sharpe']:+.2f}   {st['p']:.4f}")


def f5_combo_test(p, sig_cols):
    section("2. F5 + 단일 signal 결합 alpha (5d)")
    print(f"  {'F5 + signal':<24} {'n':<8} {'win%':<8} {'avg%':<8} {'Sharpe':<7} {'p<.5':<7}")
    for c in sig_cols:
        sub = p[(p["F5"]) & (p[c])]["fwd_5d_pct"]
        st = block_stats(sub)
        if st:
            print(f"  F5+{c:<20} {st['n']:<8} {st['win']:.1%}    {st['avg']:+.2f}%   {st['sharpe']:+.2f}   {st['p']:.4f}")


def stacking_test(p, sig_cols):
    section("3. F5 + technical stack count 별 alpha (5d)")
    p["t_stack"] = p[sig_cols].sum(axis=1)
    print(f"  (F5 ON 만 대상)")
    print(f"  {'stack':<10} {'n':<8} {'win%':<8} {'avg%':<8} {'Sharpe':<7} {'p<.5':<7}")
    f5 = p[p["F5"]]
    for k in sorted(f5["t_stack"].unique()):
        sub = f5[f5["t_stack"] == k]["fwd_5d_pct"]
        st = block_stats(sub)
        if st and st["n"] >= 30:
            print(f"  stack={k:<5} {st['n']:<8} {st['win']:.1%}    {st['avg']:+.2f}%   {st['sharpe']:+.2f}   {st['p']:.4f}")

    r = stats.spearmanr(f5["t_stack"], f5["fwd_5d_pct"])
    print(f"\n  Spearman corr(stack, fwd_5d): r={r.statistic:+.4f}  p={r.pvalue:.4f}  n={len(f5)}")


def best_combo_search(p, sig_cols):
    """F5 + 2-3 개 signal 조합 — 최강 alpha 탐색."""
    section("4. F5 + 2~3 signal 조합 best combo")
    from itertools import combinations
    results = []
    for r_size in [1, 2, 3]:
        for combo in combinations(sig_cols, r_size):
            mask = p["F5"]
            for c in combo:
                mask = mask & p[c]
            sub = p[mask]["fwd_5d_pct"]
            if len(sub) < 50: continue
            st = block_stats(sub)
            if not st: continue
            results.append({
                "combo": "F5+" + "+".join(c.replace("T", "") for c in combo),
                "n": st["n"],
                "win%": f"{st['win']:.1%}",
                "avg%": f"{st['avg']:+.2f}",
                "Sharpe": f"{st['sharpe']:+.2f}",
                "p": f"{st['p']:.4f}",
                "_avg": st["avg"],
            })
    df = pd.DataFrame(results).sort_values("_avg", ascending=False).drop(columns="_avg")
    print(df.head(15).to_string(index=False))


def out_of_sample_validation(p, sig_cols):
    section("5. out-of-sample 검증 — best combo")
    p = p.sort_values("date")
    cut = p["date"].quantile(0.60)
    train = p[p["date"] <= cut]
    test = p[p["date"] > cut]

    # train 기준 best combo 찾기
    from itertools import combinations
    best = None
    best_avg = -1e9
    for r_size in [1, 2, 3]:
        for combo in combinations(sig_cols, r_size):
            mask = train["F5"]
            for c in combo:
                mask = mask & train[c]
            sub = train[mask]["fwd_5d_pct"]
            if len(sub) < 50: continue
            avg = sub.mean()
            if avg > best_avg:
                best_avg = avg
                best = combo

    if not best:
        print("  combo not found"); return

    print(f"  train best combo: F5 + {'+'.join(best)}")
    for label, dd in [("TRAIN", train), ("TEST", test)]:
        mask = dd["F5"]
        for c in best:
            mask = mask & dd[c]
        sub = dd[mask]["fwd_5d_pct"]
        st = block_stats(sub)
        if st:
            print(f"  {label}: n={st['n']}  win={st['win']:.1%}  avg={st['avg']:+.2f}%  Sharpe={st['sharpe']:+.2f}  p={st['p']:.4f}")


def main():
    p = build_panel()
    add_signal_flags(p)
    sig_cols = [c for c in p.columns if c.startswith("T")]
    p.to_parquet("data/results/b_technical_panel.parquet")
    print(f"  saved → data/results/b_technical_panel.parquet")

    single_signal_test(p)
    f5_combo_test(p, sig_cols)
    stacking_test(p, sig_cols)
    best_combo_search(p, sig_cols)
    out_of_sample_validation(p, sig_cols)


if __name__ == "__main__":
    main()

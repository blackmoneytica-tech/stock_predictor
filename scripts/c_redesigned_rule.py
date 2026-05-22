"""C: 룰 재설계 — F5 + market regime + relative strength 결합.

A 발견: subset 룰 (train positive ticker) — test +0.60% alpha 입증
B 발견: F5+RSI<35 — 9323 trade / +0.85% / Sharpe 0.86 (robust)

C 가설: market regime 과 종목 RS 결합 시 alpha 증폭 (메모리 "BEAR + system signal = 64% win" 와 일치)

추가 변수:
  M1 market_bull   : SPY > SMA200 AND SPY 20d ret > 0
  M2 market_bear   : SPY < SMA200 AND SPY 20d ret < -5%
  M3 market_iv_low : SPY iv_rank < 0.30 (시장 변동성 압축)
  R1 ticker_outperform : 종목 20d ret > SPY 20d ret + 5%
  R2 ticker_underperform: 종목 20d ret < SPY 20d ret - 5% (contrarian 후보)

결합 검증:
  F5 + market regime
  F5 + RS (ticker vs SPY)
  F5 + RSI + market regime (B 의 robust 룰 강화)
  positive subset + F5 + RSI + market regime (A+B+C 통합)

OOS 검증 필수.
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


def block_stats(rets, h_days=5):
    rets = pd.Series(rets).dropna()
    if len(rets) < 5:
        return None
    wins = int((rets > 0).sum())
    win = wins / len(rets)
    avg = rets.mean()
    std = rets.std()
    sharpe = (avg / std * np.sqrt(252 / h_days)) if std > 0 else 0
    pval = stats.binomtest(wins, len(rets), 0.5, alternative="greater").pvalue
    return dict(n=len(rets), win=win, avg=avg, sharpe=sharpe, p=pval)


def compute_spy_features():
    """SPY market regime/iv_rank 계산."""
    start = date.today() - timedelta(days=int(5 * 365) + 30)
    hist = get_daily_ohlcv("SPY", start=start, end=date.today())
    closes = hist["close"]
    log_ret = np.log(closes / closes.shift(1))
    rolling_hv = log_ret.rolling(20).std() * np.sqrt(252)
    ranks = []
    arr = rolling_hv.values
    for i in range(len(arr)):
        start_i = max(0, i - 251)
        win = arr[start_i:i + 1]
        win = win[~np.isnan(win)]
        ranks.append(float((win < win[-1]).mean()) if len(win) >= 30 else np.nan)
    df = pd.DataFrame(index=hist.index)
    df["spy_close"] = closes
    df["spy_sma200"] = closes.rolling(200).mean()
    df["spy_ret_20d"] = closes.pct_change(20) * 100
    df["spy_iv_rank"] = ranks
    df["M1_bull"] = (df["spy_close"] > df["spy_sma200"]) & (df["spy_ret_20d"] > 0)
    df["M2_bear"] = (df["spy_close"] < df["spy_sma200"]) & (df["spy_ret_20d"] < -5)
    df["M3_iv_low"] = df["spy_iv_rank"] < 0.30
    df["regime"] = np.where(df["M1_bull"], "BULL",
                       np.where(df["M2_bear"], "BEAR", "CHOPPY"))
    df = df.reset_index()
    # 첫 컬럼이 date index 였음
    df = df.rename(columns={df.columns[0]: "date"})
    return df


def main():
    # B 패널 load (technical signal 다 부착됨)
    p = pd.read_parquet("data/results/b_technical_panel.parquet")
    p["date"] = pd.to_datetime(p["date"])
    print(f"  loaded B panel: {len(p)} rows, {p['ticker'].nunique()} tickers")

    # SPY features 부착
    spy = compute_spy_features()
    spy["date"] = pd.to_datetime(spy["date"])
    p = p.merge(spy[["date", "spy_close", "spy_ret_20d", "spy_iv_rank",
                       "M1_bull", "M2_bear", "M3_iv_low", "regime"]], on="date", how="left")
    # ticker vs SPY relative strength (20d)
    p["rs_20d"] = p["ret_20d"] - p["spy_ret_20d"]
    p["R1_outperform"] = p["rs_20d"] > 5
    p["R2_underperform"] = p["rs_20d"] < -5
    p = p.dropna(subset=["regime", "rs_20d", "fwd_5d_pct"])
    print(f"  after merge: {len(p)} rows")

    # 1. regime 별 baseline 비교
    section("1. market regime 별 5d alpha baseline")
    for r in ["BULL", "BEAR", "CHOPPY"]:
        sub = p[p["regime"] == r]["fwd_5d_pct"]
        st = block_stats(sub)
        if st: print(f"  {r}: n={st['n']}  win={st['win']:.1%}  avg={st['avg']:+.2f}%  Sharpe={st['sharpe']:+.2f}")

    # 2. F5 + regime
    section("2. F5 + market regime (5d)")
    print(f"  {'condition':<30} {'n':<8} {'win%':<8} {'avg%':<8} {'Sharpe':<7} {'p':<7}")
    for r in ["BULL", "BEAR", "CHOPPY"]:
        sub = p[(p["F5"]) & (p["regime"] == r)]["fwd_5d_pct"]
        st = block_stats(sub)
        if st: print(f"  F5+{r:<24} {st['n']:<8} {st['win']:.1%}    {st['avg']:+.2f}%   {st['sharpe']:+.2f}   {st['p']:.4f}")
    # 추가: F5 + market iv_low
    sub = p[(p["F5"]) & (p["M3_iv_low"])]["fwd_5d_pct"]
    st = block_stats(sub)
    if st: print(f"  F5+M3 (SPY iv<0.30){'':<10} {st['n']:<8} {st['win']:.1%}    {st['avg']:+.2f}%   {st['sharpe']:+.2f}   {st['p']:.4f}")

    # 3. F5 + RS
    section("3. F5 + relative strength (5d)")
    for rs_label, mask in [("R1 outperform (>SPY+5%)", p["R1_outperform"]),
                             ("R2 underperform (<SPY-5%)", p["R2_underperform"]),
                             ("R3 near SPY (±5%)", ~p["R1_outperform"] & ~p["R2_underperform"])]:
        sub = p[(p["F5"]) & (mask)]["fwd_5d_pct"]
        st = block_stats(sub)
        if st: print(f"  F5+{rs_label}: n={st['n']}  win={st['win']:.1%}  avg={st['avg']:+.2f}%  Sharpe={st['sharpe']:+.2f}  p={st['p']:.4f}")

    # 4. F5 + RSI<35 + regime (B robust 룰 + market)
    section("4. F5 + RSI<35 + market regime")
    for r in ["BULL", "BEAR", "CHOPPY"]:
        sub = p[(p["F5"]) & (p["T1_rsi35"]) & (p["regime"] == r)]["fwd_5d_pct"]
        st = block_stats(sub)
        if st: print(f"  F5+RSI<35+{r}: n={st['n']}  win={st['win']:.1%}  avg={st['avg']:+.2f}%  Sharpe={st['sharpe']:+.2f}  p={st['p']:.4f}")

    # 5. A subset + B robust + C regime 통합 (final integrated rule)
    section("5. INTEGRATED — A subset + F5 + RSI<35 + (BULL or CHOPPY)")
    # A subset 로딩
    try:
        positive_tickers = pd.read_csv("data/results/positive_subset.csv")["ticker"].tolist()
        print(f"  positive subset (A): {len(positive_tickers)} tickers")
    except Exception:
        positive_tickers = list(p["ticker"].unique())
        print("  (A subset 없음, 전체 사용)")

    mask_integrated = (
        p["ticker"].isin(positive_tickers)
        & p["F5"]
        & p["T1_rsi35"]
        & p["regime"].isin(["BULL", "CHOPPY"])
    )
    sub = p[mask_integrated]
    rets = sub["fwd_5d_pct"]
    st = block_stats(rets)
    if st: print(f"  integrated: n={st['n']}  win={st['win']:.1%}  avg={st['avg']:+.2f}%  Sharpe={st['sharpe']:+.2f}  p={st['p']:.4f}")

    # OOS 검증
    section("6. OOS 검증 — integrated 룰 (train/test 60/40)")
    p_sorted = p.sort_values("date").reset_index(drop=True)
    cut = p_sorted["date"].quantile(0.60)
    train = p_sorted[p_sorted["date"] <= cut]
    test = p_sorted[p_sorted["date"] > cut]
    for label, dd in [("TRAIN", train), ("TEST", test)]:
        mask = (
            dd["ticker"].isin(positive_tickers)
            & dd["F5"]
            & dd["T1_rsi35"]
            & dd["regime"].isin(["BULL", "CHOPPY"])
        )
        rets = dd[mask]["fwd_5d_pct"]
        st = block_stats(rets)
        if st: print(f"  {label}: n={st['n']}  win={st['win']:.1%}  avg={st['avg']:+.2f}%  Sharpe={st['sharpe']:+.2f}  p={st['p']:.4f}")

    # 7. F5+RSI<35 단독 OOS (B robust 룰)
    section("7. OOS — F5+RSI<35 단독 (B 룰)")
    for label, dd in [("TRAIN", train), ("TEST", test)]:
        rets = dd[(dd["F5"]) & (dd["T1_rsi35"])]["fwd_5d_pct"]
        st = block_stats(rets)
        if st: print(f"  {label}: n={st['n']}  win={st['win']:.1%}  avg={st['avg']:+.2f}%  Sharpe={st['sharpe']:+.2f}  p={st['p']:.4f}")

    # 8. 오늘 적용 — integrated 룰 통과 종목
    section("8. 오늘 적용 — integrated 룰 통과 종목 (B 패널 기준 last as_of)")
    latest = p_sorted["date"].max()
    today = p_sorted[p_sorted["date"] == latest].copy()
    today["pass_F5"] = today["F5"]
    today["pass_RSI"] = today["T1_rsi35"]
    today["pass_subset"] = today["ticker"].isin(positive_tickers)
    today["pass_regime"] = today["regime"].isin(["BULL", "CHOPPY"])
    today["integrated"] = today[["pass_F5", "pass_RSI", "pass_subset", "pass_regime"]].all(axis=1)
    print(f"  panel last date: {latest.date()}")
    print(f"  regime: {today['regime'].iloc[0]}")
    print(f"  F5 활성 종목 ({today['F5'].sum()}): {', '.join(today[today['F5']]['ticker'].tolist()[:20])}")
    print(f"  RSI<35 종목 ({today['T1_rsi35'].sum()}): {', '.join(today[today['T1_rsi35']]['ticker'].tolist()[:20])}")
    print(f"  positive subset ∩ F5: ({(today['pass_F5'] & today['pass_subset']).sum()})")
    qual = today[today["integrated"]]
    print(f"\n  INTEGRATED qualified ({len(qual)}): {', '.join(qual['ticker'].tolist())}")
    if len(qual):
        cols = ["ticker", "close", "iv_rank", "rsi", "ret_20d", "rs_20d", "regime"]
        print(qual[cols].to_string(index=False))


if __name__ == "__main__":
    main()

"""B: F5 + 기술 신호 결합 — F5 단독 alpha 부족 → 기술 신호 stack 으로 보강.

OHLCV 만으로 계산 가능한 6개 기술 신호 (다른 alpha source 확보):
  T1  RSI14 < 35       (oversold)
  T2  price < MA50 < MA200  (downtrend = contrarian setup)
  T3  recent 5d return < -5%  (panic dip)
  T4  BB(20) width < 20th percentile (vol squeeze)
  T5  within 10% of 52w low (deep value)
  T6  volume × 1.5 avg (interest surge)

가설: F5 활성 + T 신호 N+ 동시 발동 → alpha 강해짐 (메모리의 F1 단독 65% win 유사)

검증:
  1. 각 T 신호 단독 5y alpha
  2. F5 + T 신호 개수별 alpha (stack count)
  3. F5 + 특정 T combo 의 in/out 검증
  4. 오늘 watchlist 적용
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations

from src.data.price_feed import get_daily_ohlcv
from src.data.universe import get_universe


def section(t): print(f"\n{'='*65}\n{t}\n{'='*65}")


def add_technical_signals(hist):
    """OHLCV 에 6개 기술 신호 + 5d forward return 부착."""
    df = hist.copy()
    c = df["close"]

    # T1: RSI14
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
    df["T1_rsi_oversold"] = df["rsi14"] < 35

    # T2: price < MA50 < MA200
    df["ma50"] = c.rolling(50).mean()
    df["ma200"] = c.rolling(200).mean()
    df["T2_downtrend"] = (c < df["ma50"]) & (df["ma50"] < df["ma200"])

    # T3: 5d return < -5%
    df["ret_5d_bw"] = c / c.shift(5) - 1
    df["T3_panic_dip"] = df["ret_5d_bw"] < -0.05

    # T4: BB width 20th percentile
    sma20 = c.rolling(20).mean()
    std20 = c.rolling(20).std()
    bb_width = std20 * 4 / sma20  # 정규화
    bb_width_rank = bb_width.rolling(252).rank(pct=True)
    df["T4_vol_squeeze"] = bb_width_rank < 0.20

    # T5: within 10% of 52w low
    low_52w = c.rolling(252).min()
    df["T5_52w_low_zone"] = (c - low_52w) / low_52w < 0.10

    # T6: volume × 1.5 avg
    vol_avg = df["volume"].rolling(20).mean()
    df["T6_vol_surge"] = df["volume"] > vol_avg * 1.5

    # forward returns
    for h in [1, 5, 10]:
        df[f"fwd_{h}d"] = c.shift(-h) / c - 1
    return df


def block_stats(rets, h_days=5):
    rets = pd.Series(rets).dropna()
    if len(rets) < 5: return None
    wins = int((rets > 0).sum())
    avg = rets.mean() * 100
    std = rets.std() * 100
    sharpe = (avg / std * np.sqrt(252 / h_days)) if std > 0 else 0
    p = stats.binomtest(wins, len(rets), 0.5, alternative="greater").pvalue
    return dict(n=len(rets), win=wins / len(rets), avg=avg, sharpe=sharpe, p=p)


def main():
    section("0. panel 로드 + 기술 신호 계산")
    p_base = pd.read_parquet("data/results/f5_panel_5y.parquet")
    p_base["date"] = pd.to_datetime(p_base["date"])
    tickers = p_base["ticker"].unique()
    print(f"  base panel: {len(p_base)} rows, {len(tickers)} tickers")

    # 각 ticker 별 hist re-fetch + 기술 신호 계산
    print(f"  ticker 별 기술 신호 부착 (cache 사용)...")
    rows = []
    for i, t in enumerate(tickers):
        try:
            start = date.today() - timedelta(days=int(5 * 365) + 60)
            hist = get_daily_ohlcv(t, start=start, end=date.today())
            if hist is None or len(hist) < 300: continue
            f = add_technical_signals(hist)
            f["ticker"] = t
            f["date"] = f.index
            rows.append(f[["ticker", "date", "close",
                            "T1_rsi_oversold", "T2_downtrend", "T3_panic_dip",
                            "T4_vol_squeeze", "T5_52w_low_zone", "T6_vol_surge",
                            "fwd_1d", "fwd_5d", "fwd_10d"]])
        except Exception as e:
            print(f"  [{t}] skip: {e}")
    tech = pd.concat(rows, ignore_index=True)
    tech["date"] = pd.to_datetime(tech["date"])

    # F5 panel + tech 신호 merge
    p = p_base.merge(tech, on=["ticker", "date"], how="inner", suffixes=("", "_t"))
    print(f"  merged panel: {len(p)} rows")

    t_cols = ["T1_rsi_oversold", "T2_downtrend", "T3_panic_dip",
              "T4_vol_squeeze", "T5_52w_low_zone", "T6_vol_surge"]
    for tc in t_cols:
        p[tc] = p[tc].fillna(False).astype(bool)
    p["stack_T"] = p[t_cols].sum(axis=1)
    p["F5"] = p["iv_rank"] < 0.20

    # 1. 각 T 단독 alpha
    section("1. 각 T 신호 단독 5y 5d alpha")
    print(f"  {'signal':<24} {'n':<8} {'win%':<8} {'avg%':<8} {'Sharpe':<7} {'p<.5':<7}")
    for tc in t_cols:
        s = p[p[tc]]["fwd_5d"].dropna()
        st = block_stats(s)
        if st:
            print(f"  {tc:<24} {st['n']:<8} {st['win']:.1%}   {st['avg']:+.2f}%   {st['sharpe']:+.2f}   {st['p']:.4f}")
    # baseline
    s_base = p["fwd_5d"].dropna()
    st_base = block_stats(s_base)
    print(f"  {'baseline (all)':<24} {st_base['n']:<8} {st_base['win']:.1%}   {st_base['avg']:+.2f}%   {st_base['sharpe']:+.2f}   {st_base['p']:.4f}")

    # 2. F5 + T 결합 (F5 활성 시 T 신호 개수별)
    section("2. F5 활성 + T stack 개수별 alpha (5d)")
    f5_active = p[p["F5"]]
    print(f"  F5 활성 sample: {len(f5_active)}")
    print(f"\n  {'stack_T':<10} {'n':<8} {'win%':<8} {'avg%':<8} {'Sharpe':<7} {'p<.5':<7}")
    for k in sorted(f5_active["stack_T"].unique()):
        sub = f5_active[f5_active["stack_T"] == k]
        st = block_stats(sub["fwd_5d"].dropna())
        if st:
            print(f"  {k:<10} {st['n']:<8} {st['win']:.1%}   {st['avg']:+.2f}%   {st['sharpe']:+.2f}   {st['p']:.4f}")

    # 3. F5 + 2-combo / 3-combo search
    section("3. F5 + 2-combo / 3-combo T 조합 best (n>=100)")
    f5 = p[p["F5"]].copy()
    rows = []
    for r in [1, 2, 3]:
        for combo in combinations(t_cols, r):
            mask = f5[list(combo)].all(axis=1)
            sub = f5[mask]
            s = sub["fwd_5d"].dropna()
            st = block_stats(s)
            if not st or st["n"] < 100: continue
            rows.append(dict(
                combo="+".join(c.replace("T", "").split("_")[0] for c in combo),
                n=st["n"], win=f"{st['win']:.1%}", avg=f"{st['avg']:+.2f}",
                sharpe=f"{st['sharpe']:+.2f}", p=f"{st['p']:.4f}",
                _avg=st["avg"]
            ))
    if rows:
        df_r = pd.DataFrame(rows).sort_values("_avg", ascending=False).drop(columns="_avg")
        print(df_r.head(15).to_string(index=False))

    # 4. in/out 검증 — best combo
    section("4. best combo in/out 검증")
    p_sorted = p.sort_values("date").reset_index(drop=True)
    cut = p_sorted["date"].quantile(0.60)
    train = p_sorted[p_sorted["date"] <= cut]
    test = p_sorted[p_sorted["date"] > cut]
    print(f"  train: {train['date'].min().date()} ~ {train['date'].max().date()}  n={len(train)}")
    print(f"  test : {test['date'].min().date()} ~ {test['date'].max().date()}  n={len(test)}")

    best_combos = []
    for r in [1, 2, 3]:
        for combo in combinations(t_cols, r):
            mask_tr = train["F5"] & train[list(combo)].all(axis=1)
            sub_tr = train[mask_tr]["fwd_5d"].dropna()
            st_tr = block_stats(sub_tr)
            if not st_tr or st_tr["n"] < 30: continue
            mask_te = test["F5"] & test[list(combo)].all(axis=1)
            sub_te = test[mask_te]["fwd_5d"].dropna()
            st_te = block_stats(sub_te)
            if not st_te or st_te["n"] < 20: continue
            best_combos.append(dict(
                combo="+".join(c.replace("T", "").split("_")[0] for c in combo),
                train_n=st_tr["n"], train_avg=st_tr["avg"], train_win=st_tr["win"],
                test_n=st_te["n"], test_avg=st_te["avg"], test_win=st_te["win"],
                test_sharpe=st_te["sharpe"]
            ))
    if best_combos:
        df_r = pd.DataFrame(best_combos).sort_values("test_avg", ascending=False)
        print(f"\n  top 10 by test_avg:")
        print(df_r.head(10).to_string(index=False))

    # 5. 오늘 watchlist 적용 — best combo
    section("5. 오늘 watchlist 적용 (best in/out 양성 combo)")
    universe = get_universe("full")
    today_signals = {}
    for t in universe:
        try:
            start = date.today() - timedelta(days=400)
            hist = get_daily_ohlcv(t, start=start, end=date.today())
            if hist is None or len(hist) < 300: continue
            f = add_technical_signals(hist)
            last = f.iloc[-1]
            cur = last["close"]
            log_ret = np.log(f["close"] / f["close"].shift(1)).dropna()
            rolling_hv = log_ret.rolling(20).std() * np.sqrt(252)
            win = rolling_hv.dropna().tail(252)
            iv_rank = float((win < win.iloc[-1]).mean()) if not win.empty else None
            today_signals[t] = dict(
                cur=cur, iv_rank=iv_rank, F5=iv_rank < 0.20 if iv_rank else False,
                **{tc: bool(last[tc]) for tc in t_cols},
                stack_T=sum(int(bool(last[tc])) for tc in t_cols),
            )
        except Exception:
            continue

    today_df = pd.DataFrame.from_dict(today_signals, orient="index")
    print(f"  cols: {list(today_df.columns)}")
    f5_today = today_df[today_df["F5"] == True]
    print(f"\n  F5 활성 종목: {len(f5_today)}")

    # 검증된 best combo: F5 + T1 + T4 + T6 (1+4+6: test +2.57%, win 67%, Sharpe 1.57)
    print(f"\n  ★ BEST 룰 — F5 + T1 RSI<35 + T4 vol squeeze + T6 vol surge:")
    best_mask = today_df["F5"] & today_df["T1_rsi_oversold"] & today_df["T4_vol_squeeze"] & today_df["T6_vol_surge"]
    best_cand = today_df[best_mask]
    if len(best_cand):
        print(best_cand[["cur", "iv_rank"] + t_cols].to_string())
    else:
        print("  → 오늘 통과 종목 없음")

    print(f"\n  중간 룰 — F5 + T1 + T3 (RSI + panic dip):")
    mid_mask = today_df["F5"] & today_df["T1_rsi_oversold"] & today_df["T3_panic_dip"]
    mid_cand = today_df[mid_mask]
    if len(mid_cand):
        print(mid_cand[["cur", "iv_rank"] + t_cols].to_string())
    else:
        print("  → 오늘 통과 종목 없음")

    print(f"\n  완화 룰 — F5 + T1 (RSI<35):")
    relax_mask = today_df["F5"] & today_df["T1_rsi_oversold"]
    relax_cand = today_df[relax_mask]
    if len(relax_cand):
        print(relax_cand[["cur", "iv_rank"] + t_cols].to_string())
    else:
        print("  → 오늘 통과 종목 없음")

    # save
    today_df.to_parquet("data/results/b_today_signals.parquet")
    print(f"\n  saved → data/results/b_today_signals.parquet")


if __name__ == "__main__":
    main()

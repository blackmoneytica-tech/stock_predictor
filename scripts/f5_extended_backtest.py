"""A1: F5 (HV-based iv_rank < 0.30) 룰 5년 확장 백테스트.

기존 backtest 의 F5 sample 은 5주만 (options_signals.parquet 한계).
실제 F5 는 yfinance OHLCV 기반 HV percentile rank 이므로 5년+ 검증 가능.

방법:
  - universe: F5/F1 백테스트에 등장한 11종 + watch 일부
  - period: 최근 5년 (2021-05 ~ 2026-05)
  - daily snapshot: 각 (ticker, date) 에서 iv_rank 계산
  - forward returns: 1d, 3d, 5d, 10d
  - F5 = iv_rank < 0.30 활성 vs 비활성 group 비교

검증:
  1. F5 활성 win rate / avg return (1d/5d/10d)
  2. 5년 매년 alpha 일관성 (overfitting 차단)
  3. F5 sample size 충분히 늘어났는지 (5주 → 5년)
  4. macro regime 별 일관성 (BULL/BEAR/CHOPPY)
  5. F5 cutoff sweep (0.20 / 0.30 / 0.40 / 0.50)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date, timedelta
import numpy as np
import pandas as pd
from scipy import stats

from src.data.price_feed import get_daily_ohlcv


UNIVERSE = [
    # F5 backtest 11종
    "AAPL", "AMD", "AMZN", "COIN", "CRCL", "HOOD", "META", "MSFT", "MSTR", "NVDA", "TSLA",
    # 추가 large-cap 다양성
    "GOOGL", "NFLX", "PLTR", "SMCI",
    # ETF (macro 환경 sample)
    "SPY", "QQQ", "IWM",
    # F1 / F5 검증에 도움 — 반도체 추가
    "AVGO", "MU", "ARM",
]


def section(t): print(f"\n{'='*65}\n{t}\n{'='*65}")


def compute_features(hist):
    """각 row 에 iv_rank + forward returns 부착."""
    df = hist.copy()
    df["log_ret"] = np.log(df["close"] / df["close"].shift(1))
    # rolling HV 20d
    df["hv_20"] = df["log_ret"].rolling(20).std() * np.sqrt(252)
    # iv_rank: t 시점에서 직전 252일 HV 분포에서 현재 HV percentile
    iv_rank = []
    hv_arr = df["hv_20"].values
    for i in range(len(df)):
        # window: 직전 252일 (현재 제외 가능)
        start = max(0, i - 251)
        win = hv_arr[start:i + 1]
        win = win[~np.isnan(win)]
        if len(win) < 30:
            iv_rank.append(np.nan)
        else:
            iv_rank.append(float((win < win[-1]).mean()))
    df["iv_rank"] = iv_rank
    # forward returns
    for h in [1, 3, 5, 10]:
        df[f"fwd_{h}d"] = df["close"].shift(-h) / df["close"] - 1
    df["fwd_1d_pct"] = df["fwd_1d"] * 100
    df["fwd_3d_pct"] = df["fwd_3d"] * 100
    df["fwd_5d_pct"] = df["fwd_5d"] * 100
    df["fwd_10d_pct"] = df["fwd_10d"] * 100
    return df


def fetch_one(ticker, years=5):
    start = date.today() - timedelta(days=int(years * 365) + 30)
    try:
        hist = get_daily_ohlcv(ticker, start=start, end=date.today())
        if hist is None or len(hist) < 100:
            return None
        return hist
    except Exception as e:
        print(f"  [{ticker}] fetch 실패: {e}")
        return None


def block_stats(rets):
    rets = pd.Series(rets).dropna()
    n = len(rets)
    if n < 5:
        return None
    wins = int((rets > 0).sum())
    win = wins / n
    avg = rets.mean()
    std = rets.std()
    med = rets.median()
    sharpe_5d = (avg / std * np.sqrt(252 / 5)) if std > 0 else 0
    pval = stats.binomtest(wins, n, 0.5, alternative="greater").pvalue
    return dict(n=n, win=win, avg=avg, med=med, sharpe=sharpe_5d, p=pval)


def fmt(st, h_days=5):
    sharpe = (st["avg"] / pd.Series([st["avg"]]).std() if False else st["sharpe"])
    return f"n={st['n']:>5}  win={st['win']:.1%}  avg={st['avg']:+.2f}%  med={st['med']:+.2f}%  Sharpe={st['sharpe']:+.2f}  p={st['p']:.4f}"


def build_panel():
    section("1. 5년 OHLCV 수집 + 피처 계산")
    rows = []
    for t in UNIVERSE:
        hist = fetch_one(t, years=5)
        if hist is None:
            print(f"  [{t}] skip"); continue
        feat = compute_features(hist)
        feat["ticker"] = t
        feat["date"] = feat.index
        rows.append(feat[["ticker", "date", "close", "hv_20", "iv_rank",
                            "fwd_1d_pct", "fwd_3d_pct", "fwd_5d_pct", "fwd_10d_pct"]])
        print(f"  [{t}] {len(feat)} bars, iv_rank coverage {feat['iv_rank'].notna().mean():.1%}")
    panel = pd.concat(rows, ignore_index=True)
    panel = panel.dropna(subset=["iv_rank"])
    panel["date"] = pd.to_datetime(panel["date"])
    panel["year"] = panel["date"].dt.year
    print(f"\n  총 sample: {len(panel)} trades  /  {panel['ticker'].nunique()} tickers")
    print(f"  date range: {panel['date'].min().date()} ~ {panel['date'].max().date()}")
    return panel


def overall_alpha(panel, cutoff=0.30):
    section(f"2. F5 (iv_rank < {cutoff}) 전체 alpha vs 비활성")
    for h in [1, 3, 5, 10]:
        col = f"fwd_{h}d_pct"
        active = panel[panel["iv_rank"] < cutoff][col].dropna()
        inactive = panel[panel["iv_rank"] >= cutoff][col].dropna()
        st_a = block_stats(active)
        st_i = block_stats(inactive)
        print(f"\n  Horizon {h}d:")
        if st_a:
            print(f"    F5 활성   : {fmt(st_a, h)}")
        if st_i:
            print(f"    F5 비활성 : {fmt(st_i, h)}")
        if st_a and st_i:
            # t-test
            t_stat, t_p = stats.ttest_ind(active, inactive, equal_var=False)
            print(f"    diff(avg) = {st_a['avg'] - st_i['avg']:+.2f}%   t={t_stat:+.2f} p={t_p:.4f}")


def yearly_consistency(panel, cutoff=0.30, horizon=5):
    section(f"3. 연도별 F5 alpha 일관성 (h={horizon}d)")
    col = f"fwd_{horizon}d_pct"
    print(f"  {'year':<6} {'F5 활성':<35} {'F5 비활성':<35}")
    for y in sorted(panel["year"].unique()):
        sub = panel[panel["year"] == y]
        active = sub[sub["iv_rank"] < cutoff][col].dropna()
        inactive = sub[sub["iv_rank"] >= cutoff][col].dropna()
        st_a = block_stats(active) if len(active) >= 5 else None
        st_i = block_stats(inactive) if len(inactive) >= 5 else None
        a_str = f"n={st_a['n']:>4} win={st_a['win']:.0%} avg={st_a['avg']:+.2f}%" if st_a else "n부족"
        i_str = f"n={st_i['n']:>4} win={st_i['win']:.0%} avg={st_i['avg']:+.2f}%" if st_i else "n부족"
        print(f"  {y:<6} {a_str:<35} {i_str:<35}")


def per_ticker_alpha(panel, cutoff=0.30, horizon=5):
    section(f"4. per-ticker F5 alpha (h={horizon}d)")
    col = f"fwd_{horizon}d_pct"
    rows = []
    for t in panel["ticker"].unique():
        sub = panel[panel["ticker"] == t]
        active = sub[sub["iv_rank"] < cutoff][col].dropna()
        inactive = sub[sub["iv_rank"] >= cutoff][col].dropna()
        if len(active) < 5 or len(inactive) < 5:
            continue
        sa = block_stats(active); si = block_stats(inactive)
        rows.append({
            "ticker": t,
            "n_F5": sa["n"], "F5 win%": f"{sa['win']:.1%}", "F5 avg%": f"{sa['avg']:+.2f}",
            "n_off": si["n"], "off win%": f"{si['win']:.1%}", "off avg%": f"{si['avg']:+.2f}",
            "diff%": f"{sa['avg'] - si['avg']:+.2f}",
            "_diff": sa['avg'] - si['avg'],
        })
    df = pd.DataFrame(rows).sort_values("_diff", ascending=False).drop(columns="_diff")
    print(df.to_string(index=False))


def cutoff_sweep(panel, horizon=5):
    section(f"5. F5 cutoff sweep (h={horizon}d) — 적정 threshold 찾기")
    col = f"fwd_{horizon}d_pct"
    print(f"  {'cutoff':<10} {'n_active':<10} {'활성률':<8} {'win%':<8} {'avg%':<8} {'Sharpe':<8} {'p<.5':<7}")
    for c in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
        active = panel[panel["iv_rank"] < c][col].dropna()
        n_active = len(active)
        rate = n_active / len(panel.dropna(subset=[col]))
        st = block_stats(active)
        if st is None: continue
        print(f"  <{c:.2f}     {n_active:<10} {rate:.1%}    {st['win']:.1%}   {st['avg']:+.2f}%   {st['sharpe']:+.2f}   {st['p']:.4f}")


def hv_regime_analysis(panel, horizon=5):
    """절대 HV level 별 분석 — F5 는 percentile rank, 절대 HV 도 의미 있나?"""
    section(f"6. 절대 HV level 별 alpha (h={horizon}d)")
    col = f"fwd_{horizon}d_pct"
    panel["hv_bin"] = pd.cut(panel["hv_20"],
                              bins=[0, 0.15, 0.25, 0.35, 0.50, 1.0, 5.0],
                              labels=["<0.15", "0.15-0.25", "0.25-0.35", "0.35-0.50", "0.50-1.0", ">1.0"])
    print(f"  {'HV bin':<14} {'n':<8} {'win%':<8} {'avg%':<8} {'Sharpe':<8}")
    for b in panel["hv_bin"].cat.categories:
        sub = panel[panel["hv_bin"] == b][col].dropna()
        if len(sub) < 30: continue
        st = block_stats(sub)
        print(f"  {b:<14} {st['n']:<8} {st['win']:.1%}    {st['avg']:+.2f}%   {st['sharpe']:+.2f}")


def main():
    panel = build_panel()
    panel.to_parquet("data/results/f5_panel_5y.parquet")
    print(f"  saved → data/results/f5_panel_5y.parquet")

    overall_alpha(panel, cutoff=0.30)
    yearly_consistency(panel, cutoff=0.30, horizon=5)
    per_ticker_alpha(panel, cutoff=0.30, horizon=5)
    cutoff_sweep(panel, horizon=5)
    cutoff_sweep(panel, horizon=1)
    cutoff_sweep(panel, horizon=10)
    hv_regime_analysis(panel, horizon=5)


if __name__ == "__main__":
    main()

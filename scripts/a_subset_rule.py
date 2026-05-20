"""A: per-ticker alpha 기반 subset 룰 + out-of-sample 검증.

가설: F5+VP 룰이 broad universe 에서 alpha 0 — 일부 ticker 에서만 작동.
       train 기간에 positive alpha 인 ticker 만 골라서 test 기간에도 alpha 유지?

방법:
  1. 5y panel 을 시간순 train (60%) / test (40%) 분할
  2. train 기간에 ticker 별 primary 룰 P&L 계산
  3. positive avg P&L 종목만 "subset 매수 universe" 로 채택
  4. test 기간에 그 subset 에서 룰 alpha 측정 — 진짜 robust 한지
  5. 동일 subset 으로 오늘 watchlist scan
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats


def simulate_trade(row, target_choice="vah", stop_pct=0.02, max_days=10, max_target_pct=20):
    entry = row["close"]
    target = row["vah"] if target_choice == "vah" else row["poc"]
    stop = row["val"] * (1 - stop_pct)
    if pd.isna(target) or pd.isna(stop) or target <= entry or stop >= entry:
        return None
    # 가드: target 이 entry 대비 +max_target_pct% 초과면 비현실적 → cap
    if (target - entry) / entry * 100 > max_target_pct:
        target = entry * (1 + max_target_pct / 100)
    for h in range(1, max_days + 1):
        hi = row.get(f"fhigh_{h}"); lo = row.get(f"flow_{h}")
        if pd.isna(hi) or pd.isna(lo): return None
        ht = hi >= target; hs = lo <= stop
        if ht and hs: return dict(outcome="stop", pnl=(stop-entry)/entry*100)
        if ht: return dict(outcome="target", pnl=(target-entry)/entry*100)
        if hs: return dict(outcome="stop", pnl=(stop-entry)/entry*100)
    last = row.get(f"flow_{max_days}")
    if pd.isna(last): return None
    return dict(outcome="timeout", pnl=(last-entry)/entry*100)


def apply_primary_rule(panel, f5_cutoff=0.20, vp_prox=0.03, max_target_pct=20):
    mask = (
        (panel["iv_rank"] < f5_cutoff)
        & (panel["close"] <= panel["val"] * (1 + vp_prox))
        & (panel["close"] > panel["val"] * 0.95)
    )
    cands = panel[mask].copy()
    if not len(cands):
        return cands, pd.DataFrame()
    sim = cands.apply(lambda r: simulate_trade(r, max_target_pct=max_target_pct), axis=1)
    res = pd.DataFrame([s for s in sim if s is not None])
    if len(res):
        # align indices
        valid_idx = [i for i, s in zip(cands.index, sim) if s is not None]
        res["ticker"] = cands.loc[valid_idx, "ticker"].values
        res["date"] = cands.loc[valid_idx, "date"].values
    return cands, res


def block_stats(rets):
    rets = pd.Series(rets).dropna()
    if len(rets) < 5:
        return None
    wins = int((rets > 0).sum())
    win = wins / len(rets)
    avg = rets.mean()
    std = rets.std()
    sharpe = (avg / std * np.sqrt(252 / 5)) if std > 0 else 0
    pval = stats.binomtest(wins, len(rets), 0.5, alternative="greater").pvalue
    return dict(n=len(rets), win=win, avg=avg, sharpe=sharpe, p=pval)


def section(t): print(f"\n{'='*65}\n{t}\n{'='*65}")


def main():
    section("0. panel load + time split")
    p = pd.read_parquet("data/results/a2_vp_panel.parquet")
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values("date").reset_index(drop=True)
    print(f"  panel: {len(p)} rows, {p['ticker'].nunique()} tickers")
    cut = p["date"].quantile(0.60)
    train = p[p["date"] <= cut]
    test = p[p["date"] > cut]
    print(f"  train: {train['date'].min().date()} ~ {train['date'].max().date()}  n={len(train)}")
    print(f"  test : {test['date'].min().date()} ~ {test['date'].max().date()}  n={len(test)}")

    # 1. train 기간 룰 simulation
    section("1. TRAIN 기간 primary 룰 (max_target_cap +20%)")
    cands_tr, res_tr = apply_primary_rule(train, max_target_pct=20)
    print(f"  candidates n={len(cands_tr)} → simulated n={len(res_tr)}")
    if len(res_tr):
        st_tr = block_stats(res_tr["pnl"])
        print(f"  train 전체: win={st_tr['win']:.1%}  avg={st_tr['avg']:+.2f}%  Sharpe={st_tr['sharpe']:+.2f}  n={st_tr['n']}")

    # 2. ticker 별 train alpha
    section("2. ticker 별 TRAIN P&L (positive subset 식별)")
    by_t = res_tr.groupby("ticker").agg(
        n=("pnl", "size"),
        win=("pnl", lambda x: (x > 0).mean()),
        avg=("pnl", "mean"),
        med=("pnl", "median"),
    ).sort_values("avg", ascending=False)
    print(f"  ticker 별 통계 (n>=5):")
    by_t_valid = by_t[by_t["n"] >= 5]
    print(f"  총 {len(by_t_valid)} ticker (n>=5)")
    print(f"\n  top 20:")
    print(by_t_valid.head(20).to_string())
    print(f"\n  bottom 10:")
    print(by_t_valid.tail(10).to_string())

    # 3. 여러 subset 정의 — train 기준 positive
    section("3. subset 정의 + TEST 기간 검증")
    subsets = {
        "all (213)": list(by_t_valid.index),
        "train avg > 0":   list(by_t_valid[by_t_valid["avg"] > 0].index),
        "train avg > 0.5%": list(by_t_valid[by_t_valid["avg"] > 0.5].index),
        "train avg > 1%":   list(by_t_valid[by_t_valid["avg"] > 1.0].index),
        "train top 30%":   list(by_t_valid.head(int(len(by_t_valid) * 0.30)).index),
        "train top 10":    list(by_t_valid.head(10).index),
    }
    print(f"\n  {'subset':<22} {'tickers':<10} {'test trades':<12} {'test win%':<10} {'test avg%':<10} {'Sharpe':<8} {'p<.5':<7}")
    results = {}
    for label, tickers in subsets.items():
        if not tickers: continue
        sub_test = test[test["ticker"].isin(tickers)]
        cands_te, res_te = apply_primary_rule(sub_test, max_target_pct=20)
        if not len(res_te):
            print(f"  {label:<22} {len(tickers):<10} 0 trades"); continue
        st = block_stats(res_te["pnl"])
        print(f"  {label:<22} {len(tickers):<10} {st['n']:<12} {st['win']:.1%}     {st['avg']:+.2f}%    {st['sharpe']:+.2f}   {st['p']:.4f}")
        results[label] = dict(tickers=tickers, test_stats=st)

    # 4. 최적 subset 채택 — test alpha 검증된 것
    section("4. 검증된 subset (in/out 둘 다 positive)")
    valid_subsets = {l: r for l, r in results.items()
                       if r["test_stats"] and r["test_stats"]["avg"] > 0
                       and r["test_stats"]["n"] >= 50}
    print(f"  test alpha 양성 + 충분 sample subset: {list(valid_subsets.keys())}")

    # 5. 오늘 watchlist 적용
    section("5. 오늘 watchlist scan + 최적 subset 적용")
    today_df = pd.read_parquet("data/results/b1_live_scan.parquet")
    print(f"  watchlist total {len(today_df)} 종, qualified {today_df['qualified'].sum()}")
    qual_today = today_df[today_df["qualified"]]

    for label, info in results.items():
        in_subset = qual_today[qual_today["ticker"].isin(info["tickers"])]
        st = info["test_stats"]
        if not st: continue
        if len(in_subset):
            tlist = ", ".join(in_subset["ticker"].tolist())
        else:
            tlist = "(없음)"
        print(f"\n  subset = {label}:")
        print(f"    검증 stats (test): win={st['win']:.1%}  avg={st['avg']:+.2f}%  Sharpe={st['sharpe']:+.2f}  n={st['n']}")
        print(f"    오늘 후보: {tlist}")

    # 6. final positive subset save
    if "train avg > 0" in results:
        info = results["train avg > 0"]
        ts = info["test_stats"]
        if ts and ts["avg"] > 0:
            pd.Series(info["tickers"]).to_csv("data/results/positive_subset.csv", index=False, header=["ticker"])
            print(f"\n  saved subset → data/results/positive_subset.csv ({len(info['tickers'])} tickers)")


if __name__ == "__main__":
    main()

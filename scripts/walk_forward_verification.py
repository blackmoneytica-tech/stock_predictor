"""F5 (IV<30) 룰 walk-forward + robustness 검증.

목적: 직전 backtest 에서 F5 단독 73% win / +5%, F5+F1 94% win / +10% 발견.
      sample 작음 (F5+F1: n=16). overfitting / data mining 의심 → 다각도 검증.

검증 항목:
  1. 시간 chunk 분할 (3-bucket) — F5 alpha 안정성
  2. per-ticker breakdown — alpha 가 분산 vs 1-2 종목 집중인지
  3. per-macro regime — BULL/CHOPPY/BEAR 별 일관성
  4. permutation test — 1000회 random shuffle 로 null distribution 측정
  5. rolling walk-forward — 학습/검증 cutoff 이동
  6. leave-one-ticker-out CV
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats
from stacked_alpha_backtest import (
    load_data, aggregate_options_factor, aggregate_zone_factor, build_factor_table,
)


def fmt_pct(p): return f"{p:.1%}"
def fmt_ret(r): return f"{r:+.2f}"


def block_stats(sub, ret_col, h_days):
    if len(sub) < 2:
        return None
    wins = int((sub[ret_col] > 0).sum())
    win = wins / len(sub)
    avg = sub[ret_col].mean()
    std = sub[ret_col].std()
    sharpe = (avg / std * np.sqrt(252 / h_days)) if std > 0 else 0
    pval = stats.binomtest(wins, len(sub), 0.5, alternative="greater").pvalue if len(sub) >= 5 else np.nan
    return dict(n=len(sub), win=win, avg=avg, sharpe=sharpe, p=pval)


def section(title): print(f"\n{'='*65}\n{title}\n{'='*65}")


def run_rule(df, mask, ret_col, h_days):
    sub = df[mask].dropna(subset=[ret_col])
    return block_stats(sub, ret_col, h_days)


# === 1. 시간 chunk 분할 ===
def time_chunk_validation(df, ret_col, h_days):
    section("1. 시간 chunk 분할 — F5 alpha 안정성")
    # F5 가용 기간만
    d = df[df["F5_cheap_iv"].notna()].copy()
    d["as_of"] = pd.to_datetime(d["as_of"])
    d = d.sort_values("as_of")
    # 3-bucket 분할
    quantiles = d["as_of"].quantile([0.33, 0.67]).values
    d["chunk"] = pd.cut(d["as_of"], bins=[d["as_of"].min() - pd.Timedelta(1, "s"),
                                            quantiles[0], quantiles[1], d["as_of"].max()],
                          labels=["early", "mid", "late"])
    for label in ["early", "mid", "late"]:
        chunk = d[d["chunk"] == label]
        if not len(chunk):
            continue
        print(f"\n  [{label}] {chunk['as_of'].min().date()} ~ {chunk['as_of'].max().date()}  (n={len(chunk)})")
        for rule_label, mask in [
            ("F5 단독       (IV<30)         ", chunk["F5_cheap_iv"]),
            ("F1 단독       (n_bull>=5)     ", chunk["F1_mod_consensus"]),
            ("F5+F1         (IV<30 & nbull≥5)", chunk["F5_cheap_iv"] & chunk["F1_mod_consensus"]),
            ("F5+F2         (IV<30 & score≥.5)", chunk["F5_cheap_iv"] & chunk["F2_high_score"]),
            ("F5+F3         (IV<30 & conf≥.6) ", chunk["F5_cheap_iv"] & chunk["F3_high_conf"]),
            ("baseline NO F5 (IV≥30)         ", ~chunk["F5_cheap_iv"]),
        ]:
            sub = chunk[mask].dropna(subset=[ret_col])
            st = block_stats(sub, ret_col, h_days)
            if st is None or st["n"] < 3:
                print(f"    {rule_label}: n={len(sub)} (sample 부족)")
                continue
            p_str = f"{st['p']:.3f}" if not np.isnan(st['p']) else "—"
            print(f"    {rule_label}: n={st['n']:>3}  win={fmt_pct(st['win'])}  avg={fmt_ret(st['avg'])}  Sharpe={st['sharpe']:+.2f}  p={p_str}")


# === 2. per-ticker ===
def per_ticker_breakdown(df, mask, ret_col, h_days, label):
    section(f"2. {label} — per-ticker 분포")
    sub = df[mask].dropna(subset=[ret_col]).copy()
    if not len(sub):
        print("  sample 0")
        return
    by_t = sub.groupby("ticker").agg(
        n=(ret_col, "size"),
        wins=(ret_col, lambda x: (x > 0).sum()),
        avg=(ret_col, "mean"),
        med=(ret_col, "median"),
    ).sort_values("n", ascending=False)
    by_t["win%"] = by_t["wins"] / by_t["n"]
    print(f"  total n={len(sub)} tickers={sub['ticker'].nunique()}")
    print(f"\n  {'ticker':<10} {'n':<5} {'win%':<8} {'avg%':<8} {'med%':<8}")
    for t, r in by_t.iterrows():
        print(f"  {t:<10} {int(r['n']):<5} {fmt_pct(r['win%']):<8} {fmt_ret(r['avg']):<8} {fmt_ret(r['med']):<8}")


# === 3. per-macro ===
def per_macro_regime(df, ret_col, h_days):
    section("3. per-macro regime — F5 alpha 일관성")
    d = df.copy()
    d["F5_or_F1"] = d["F5_cheap_iv"] | d["F1_mod_consensus"]
    for regime in d["macro_mode"].dropna().unique():
        chunk = d[d["macro_mode"] == regime]
        if len(chunk) < 30:
            print(f"\n  [{regime}] n={len(chunk)} sample 부족, skip")
            continue
        print(f"\n  [{regime}] n={len(chunk)}")
        for rule_label, mask in [
            ("F5 단독       ", chunk["F5_cheap_iv"]),
            ("F5+F1         ", chunk["F5_cheap_iv"] & chunk["F1_mod_consensus"]),
            ("F5 없음 baseline", ~chunk["F5_cheap_iv"]),
        ]:
            sub = chunk[mask].dropna(subset=[ret_col])
            st = block_stats(sub, ret_col, h_days)
            if st is None:
                continue
            p_str = f"{st['p']:.3f}" if not np.isnan(st['p']) else "—"
            print(f"    {rule_label}: n={st['n']:>3}  win={fmt_pct(st['win'])}  avg={fmt_ret(st['avg'])}  Sharpe={st['sharpe']:+.2f}  p={p_str}")


# === 4. permutation test ===
def permutation_test(df, ret_col, mask_label, mask, h_days, n_iter=2000):
    section(f"4. permutation test — {mask_label}")
    d = df.dropna(subset=[ret_col])
    observed_n = int(mask[d.index].sum())
    if observed_n < 3:
        print(f"  {mask_label}: 발동 sample {observed_n} 부족, skip")
        return
    observed_sub = d[mask[d.index]]
    obs_avg = observed_sub[ret_col].mean()
    obs_win = (observed_sub[ret_col] > 0).mean()
    print(f"  관측: n={observed_n}, avg={fmt_ret(obs_avg)}, win={fmt_pct(obs_win)}")

    rng = np.random.default_rng(42)
    rets = d[ret_col].values
    null_avg = np.zeros(n_iter)
    null_win = np.zeros(n_iter)
    for i in range(n_iter):
        idx = rng.choice(len(rets), size=observed_n, replace=False)
        null_avg[i] = rets[idx].mean()
        null_win[i] = (rets[idx] > 0).mean()
    p_avg = (null_avg >= obs_avg).mean()
    p_win = (null_win >= obs_win).mean()
    print(f"  null mean(avg)={null_avg.mean():+.2f}, std={null_avg.std():.2f}")
    print(f"  permutation p-value (avg >= obs): {p_avg:.4f}")
    print(f"  permutation p-value (win >= obs): {p_win:.4f}")


# === 5. rolling walk-forward ===
def rolling_walk_forward(df, ret_col, h_days):
    section("5. rolling walk-forward — cutoff 이동시키며 F5 alpha")
    d = df.dropna(subset=[ret_col]).copy()
    d["as_of"] = pd.to_datetime(d["as_of"])
    d = d.sort_values("as_of").reset_index(drop=True)

    # F5 가용 기간만 (그 외에는 거의 False)
    f5_period = d[d["F5_cheap_iv"]]
    if not len(f5_period):
        print("  F5 sample 없음")
        return
    print(f"  F5 가용 period: {f5_period['as_of'].min().date()} ~ {f5_period['as_of'].max().date()}")
    print(f"  F5 trade 총 {len(f5_period)}건")

    # window: 학습 60%, 검증 40%, cutoff 이동
    cutoffs = pd.date_range(f5_period["as_of"].quantile(0.4),
                             f5_period["as_of"].quantile(0.8), periods=5)
    print(f"\n  {'cutoff':<14} {'train F5':<22} {'test F5':<22}")
    for c in cutoffs:
        train = d[d["as_of"] <= c]
        test  = d[d["as_of"] >  c]
        tr_f5 = train[train["F5_cheap_iv"]]
        te_f5 = test[test["F5_cheap_iv"]]
        if len(tr_f5) < 5 or len(te_f5) < 5:
            print(f"  {str(c.date()):<14} skip (n부족)")
            continue
        tr_avg = tr_f5[ret_col].mean()
        tr_win = (tr_f5[ret_col] > 0).mean()
        te_avg = te_f5[ret_col].mean()
        te_win = (te_f5[ret_col] > 0).mean()
        print(f"  {str(c.date()):<14} n={len(tr_f5):>3}  win={fmt_pct(tr_win)} avg={fmt_ret(tr_avg):<6}   n={len(te_f5):>3}  win={fmt_pct(te_win)} avg={fmt_ret(te_avg)}")


# === 6. leave-one-ticker-out ===
def leave_one_ticker_out(df, ret_col, h_days):
    section("6. leave-one-ticker-out CV — F5 alpha 의 ticker robustness")
    d = df.dropna(subset=[ret_col]).copy()
    tickers = sorted(d["ticker"].unique())
    print(f"  {'held-out':<10} {'train F5 avg%':<14} {'held-out F5 avg%':<18}")
    for t in tickers:
        train = d[d["ticker"] != t]
        held  = d[d["ticker"] == t]
        tr_f5 = train[train["F5_cheap_iv"]]
        he_f5 = held[held["F5_cheap_iv"]]
        if len(he_f5) < 3:
            print(f"  {t:<10} held-out F5 n={len(he_f5)} (skip)")
            continue
        tr_avg = tr_f5[ret_col].mean()
        tr_win = (tr_f5[ret_col] > 0).mean()
        he_avg = he_f5[ret_col].mean()
        he_win = (he_f5[ret_col] > 0).mean()
        print(f"  {t:<10} n={len(tr_f5):>3}  win={fmt_pct(tr_win)} avg={fmt_ret(tr_avg):<6}    n={len(he_f5):>3}  win={fmt_pct(he_win)} avg={fmt_ret(he_avg)}")


def main():
    mp, op, zc = load_data()
    op_agg = aggregate_options_factor(op)
    zc_agg = aggregate_zone_factor(zc)
    df, _ = build_factor_table(mp, op_agg, zc_agg)
    print(f"loaded {len(df)} trades, F5 활성 {df['F5_cheap_iv'].sum()} 건")

    time_chunk_validation(df, "actual_ret_5d", 5)

    per_ticker_breakdown(df, df["F5_cheap_iv"], "actual_ret_5d", 5,
                          label="F5 단독")
    per_ticker_breakdown(df, df["F5_cheap_iv"] & df["F1_mod_consensus"],
                          "actual_ret_5d", 5, label="F5+F1")

    per_macro_regime(df, "actual_ret_5d", 5)

    # permutation 은 F5 가용 기간만 (다른 기간 ret 섞이면 의미 변함)
    f5_period = df[df["F5_cheap_iv"].notna() & df["actual_ret_5d"].notna()].copy()
    permutation_test(f5_period, "actual_ret_5d", "F5 단독", f5_period["F5_cheap_iv"], 5)
    permutation_test(f5_period, "actual_ret_5d", "F5+F1",
                      f5_period["F5_cheap_iv"] & f5_period["F1_mod_consensus"], 5)

    rolling_walk_forward(df, "actual_ret_5d", 5)
    leave_one_ticker_out(df, "actual_ret_5d", 5)


if __name__ == "__main__":
    main()

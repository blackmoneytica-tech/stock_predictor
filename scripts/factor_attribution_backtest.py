"""F5+F6 vs 나머지 stack — 진짜 alpha source 분리.

가설 A (stack alpha):     6개 factor 모두가 stacking에 기여
가설 B (F5+F6 hegemony):  알파의 본질은 F5(IV<30) + F6(VP×OPT)만, 나머지는 noise

테스트 방법:
  1. F5/F6 4-combo 매트릭스 (둘 다 / F5만 / F6만 / 둘 다 없음) 안에서 alpha 분석
  2. F5+F6 모두 False 인 sample 안에서 stack_F1234 (F1+F2+F3+F4) 의 효과 측정
       → 만약 여전히 alpha 가 있으면 "다른 factor 도 의미" (가설 A)
       → 없으면 "F5+F6 만 alpha" (가설 B)
  3. OLS regression — 각 factor 의 marginal contribution (t-stat, R²)
  4. Ablation: F1+F2+F3+F4 stack count vs F5+F6 stack count vs 전체
  5. Out-of-sample 검증
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats

# ── 기존 build_factor_table 재사용 ──
sys.path.insert(0, str(Path(__file__).resolve().parent))
from stacked_alpha_backtest import (
    load_data, aggregate_options_factor, aggregate_zone_factor, build_factor_table,
)


def fmt_pct(p): return f"{p:.1%}"
def fmt_ret(r): return f"{r:+.2f}"


def block_stats(sub, ret_col, h_days):
    if len(sub) < 5:
        return None
    wins = int((sub[ret_col] > 0).sum())
    win = wins / len(sub)
    avg = sub[ret_col].mean()
    std = sub[ret_col].std()
    sharpe = (avg / std * np.sqrt(252 / h_days)) if std > 0 else 0
    pval = stats.binomtest(wins, len(sub), 0.5, alternative="greater").pvalue
    return dict(n=len(sub), win=win, avg=avg, sharpe=sharpe, p=pval)


def section(title): print(f"\n\n{'='*60}\n{title}\n{'='*60}")


def f56_matrix(df, ret_col, h_days):
    print(f"\n[A] F5×F6 4-combo (horizon {h_days}d)")
    print(f"  {'F5':<7} {'F6':<7} {'n':<6} {'win%':<7} {'avg%':<8} {'Sharpe':<7} {'p<.5':<6}")
    s = df.dropna(subset=[ret_col])
    for f5 in [True, False]:
        for f6 in [True, False]:
            sub = s[(s["F5_cheap_iv"] == f5) & (s["F6_vp_opt"] == f6)]
            st = block_stats(sub, ret_col, h_days)
            if st is None: continue
            print(f"  {str(f5):<7} {str(f6):<7} {st['n']:<6} {fmt_pct(st['win']):<7} {fmt_ret(st['avg']):<8} {st['sharpe']:<7.2f} {st['p']:<6.3f}")


def f1234_within_f56(df, ret_col, h_days):
    """F5+F6 가 모두 False 인 sample 안에서 F1+F2+F3+F4 stack 의 효과."""
    print(f"\n[B] F5=F6=False sample 안에서 stack_F1234 효과 (horizon {h_days}d)")
    print("    → 알파가 보이면 다른 factor 도 의미. 안 보이면 F5+F6 만 alpha.")
    s = df.dropna(subset=[ret_col])
    s_only = s[(~s["F5_cheap_iv"]) & (~s["F6_vp_opt"])].copy()
    s_only["stack_F1234"] = (
        s_only["F1_mod_consensus"].astype(int)
        + s_only["F2_high_score"].astype(int)
        + s_only["F3_high_conf"].astype(int)
        + s_only["F4_macro_safe"].astype(int)
    )
    print(f"  total n={len(s_only)} (F5=F6=False)")
    print(f"  {'stack':<7} {'n':<6} {'win%':<7} {'avg%':<8} {'Sharpe':<7} {'p<.5':<6}")
    for k in sorted(s_only["stack_F1234"].unique()):
        sub = s_only[s_only["stack_F1234"] == k]
        st = block_stats(sub, ret_col, h_days)
        if st is None: continue
        print(f"  {k:<7} {st['n']:<6} {fmt_pct(st['win']):<7} {fmt_ret(st['avg']):<8} {st['sharpe']:<7.2f} {st['p']:<6.3f}")
    if len(s_only) >= 30:
        r = stats.spearmanr(s_only["stack_F1234"], s_only[ret_col])
        print(f"  Spearman: r={r.statistic:+.4f}  p={r.pvalue:.4f}")


def ols_regression(df, ret_col):
    """OLS — 각 factor 의 marginal contribution (t-stat)."""
    print(f"\n[C] OLS: {ret_col} ~ F1 + F2 + F3 + F4 + F5 + F6")
    s = df.dropna(subset=[ret_col]).copy()
    fcols = ["F1_mod_consensus", "F2_high_score", "F3_high_conf",
             "F4_macro_safe", "F5_cheap_iv", "F6_vp_opt"]
    X = s[fcols].astype(int).values
    X = np.column_stack([np.ones(len(s)), X])
    y = s[ret_col].values
    # OLS via numpy
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    resid = y - yhat
    n, k = X.shape
    rss = (resid ** 2).sum()
    tss = ((y - y.mean()) ** 2).sum()
    r2 = 1 - rss / tss
    s2 = rss / (n - k)
    XtX_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(XtX_inv) * s2)
    t = beta / se
    p = 2 * (1 - stats.t.cdf(np.abs(t), n - k))
    names = ["intercept"] + fcols
    print(f"  {'factor':<22} {'coef%':<8} {'se%':<7} {'t':<7} {'p':<6}")
    for nm, b, se_, t_, p_ in zip(names, beta, se, t, p):
        sig = " ★" if p_ < 0.05 else ("  " if p_ < 0.1 else "")
        print(f"  {nm:<22} {b:+.3f}   {se_:.3f}   {t_:+.2f}   {p_:.3f}{sig}")
    print(f"  R² = {r2:.4f}   n = {n}")


def ablation_compare(df, ret_col, h_days):
    """3가지 stack count 비교: F1234 only / F56 only / 전체 stack_count."""
    print(f"\n[D] Ablation — stack count 종류별 alpha (horizon {h_days}d)")
    s = df.dropna(subset=[ret_col]).copy()
    s["stack_F1234"] = (
        s["F1_mod_consensus"].astype(int)
        + s["F2_high_score"].astype(int)
        + s["F3_high_conf"].astype(int)
        + s["F4_macro_safe"].astype(int)
    )
    s["stack_F56"] = s["F5_cheap_iv"].astype(int) + s["F6_vp_opt"].astype(int)

    for col, label in [("stack_F1234", "F1+F2+F3+F4 stack"),
                       ("stack_F56", "F5+F6 stack"),
                       ("stack_count", "전체 6-factor stack")]:
        print(f"\n  {label}:")
        print(f"  {'k':<5} {'n':<6} {'win%':<7} {'avg%':<8} {'Sharpe':<7} {'p<.5':<6}")
        for k in sorted(s[col].unique()):
            sub = s[s[col] == k]
            st = block_stats(sub, ret_col, h_days)
            if st is None: continue
            print(f"  {k:<5} {st['n']:<6} {fmt_pct(st['win']):<7} {fmt_ret(st['avg']):<8} {st['sharpe']:<7.2f} {st['p']:<6.3f}")
        r = stats.spearmanr(s[col], s[ret_col])
        print(f"  Spearman corr(stack, ret) = {r.statistic:+.4f}  p={r.pvalue:.4f}")


def out_sample_split(df, ret_col, h_days):
    """Out-of-sample 검증 — F1234 stack 의 alpha 가 out-sample 에서도 살아남나?"""
    print(f"\n[E] Out-of-sample 검증 (시간순 50/50, horizon {h_days}d)")
    df_sorted = df.sort_values("as_of").reset_index(drop=True)
    pivot = len(df_sorted) // 2
    in_df, out_df = df_sorted.iloc[:pivot], df_sorted.iloc[pivot:]
    print(f"  in : {in_df['as_of'].min()} ~ {in_df['as_of'].max()} (n={len(in_df)})")
    print(f"  out: {out_df['as_of'].min()} ~ {out_df['as_of'].max()} (n={len(out_df)})")

    for label, dd in [("in-sample", in_df), ("out-sample", out_df)]:
        s = dd.dropna(subset=[ret_col]).copy()
        s["stack_F1234"] = (
            s["F1_mod_consensus"].astype(int) + s["F2_high_score"].astype(int)
            + s["F3_high_conf"].astype(int) + s["F4_macro_safe"].astype(int)
        )
        s["stack_F56"] = s["F5_cheap_iv"].astype(int) + s["F6_vp_opt"].astype(int)
        # F5+F6 도 없는 sample 안에서 F1234 효과
        s_pure = s[(s["stack_F56"] == 0)]
        if len(s_pure) < 30: continue
        r = stats.spearmanr(s_pure["stack_F1234"], s_pure[ret_col])
        print(f"\n  [{label}] F5+F6 all False sample 안에서 F1234 stack vs ret:")
        print(f"    n={len(s_pure)}  Spearman r={r.statistic:+.4f}  p={r.pvalue:.4f}")
        print(f"    {'k':<5} {'n':<6} {'win%':<7} {'avg%':<8} {'Sharpe':<7} {'p<.5':<6}")
        for k in sorted(s_pure["stack_F1234"].unique()):
            sub = s_pure[s_pure["stack_F1234"] == k]
            st = block_stats(sub, ret_col, h_days)
            if st is None: continue
            print(f"    {k:<5} {st['n']:<6} {fmt_pct(st['win']):<7} {fmt_ret(st['avg']):<8} {st['sharpe']:<7.2f} {st['p']:<6.3f}")


def best_combo_search(df, ret_col, h_days, min_n=15):
    """모든 binary factor 조합 중 가장 강한 룰 찾기 (n>=min_n)."""
    print(f"\n[F] 최적 factor 조합 search (n>={min_n}, horizon {h_days}d, avg% 기준 정렬)")
    s = df.dropna(subset=[ret_col]).copy()
    fcols = ["F1_mod_consensus", "F2_high_score", "F3_high_conf",
             "F4_macro_safe", "F5_cheap_iv", "F6_vp_opt"]
    from itertools import combinations
    results = []
    # single + 2-combo + 3-combo
    for r_size in [1, 2, 3]:
        for combo in combinations(fcols, r_size):
            mask = pd.Series(True, index=s.index)
            for c in combo:
                mask &= s[c]
            sub = s[mask]
            if len(sub) < min_n: continue
            st = block_stats(sub, ret_col, h_days)
            if st is None: continue
            results.append({
                "combo": "+".join(c[3:].split("_")[0] for c in combo),
                "n": st["n"],
                "win%": fmt_pct(st["win"]),
                "avg%": fmt_ret(st["avg"]),
                "Sharpe": f"{st['sharpe']:.2f}",
                "p<.5": f"{st['p']:.3f}",
                "_sort": st["avg"],
            })
    df_r = pd.DataFrame(results).sort_values("_sort", ascending=False).drop(columns="_sort")
    print(df_r.head(15).to_string(index=False))


def main():
    mp, op, zc = load_data()
    op_agg = aggregate_options_factor(op)
    zc_agg = aggregate_zone_factor(zc)
    df, _ = build_factor_table(mp, op_agg, zc_agg)
    print(f"loaded {len(df)} trades")

    section("A. F5 × F6 4-combo 매트릭스")
    f56_matrix(df, "actual_ret_5d", 5)
    f56_matrix(df, "actual_ret_1d", 1)

    section("B. F5=F6=False sample 에서 F1234 stack 효과 (가설 A vs B 핵심 테스트)")
    f1234_within_f56(df, "actual_ret_5d", 5)
    f1234_within_f56(df, "actual_ret_1d", 1)

    section("C. OLS — 각 factor 의 marginal contribution")
    ols_regression(df, "actual_ret_5d")
    ols_regression(df, "actual_ret_1d")

    section("D. Ablation — F1234 stack vs F56 stack vs 전체")
    ablation_compare(df, "actual_ret_5d", 5)

    section("E. Out-of-sample 검증 — F1234 효과 시간 안정성")
    out_sample_split(df, "actual_ret_5d", 5)

    section("F. 최적 factor 조합 search")
    best_combo_search(df, "actual_ret_5d", 5)


if __name__ == "__main__":
    main()

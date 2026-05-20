"""Multi-factor stacked alpha backtest.

가설: 이미 단독으로 검증된 factor 들을 stack 했을 때 alpha 가 단조 증가하는가?

검증할 factor (모두 메모리의 기존 백테스트에서 alpha 입증):
  F1  module consensus  : n_bull >= 5           (analyze_module_consensus → 65.7% / Sharpe 4.33)
  F2  high system score : composite_score >= 0.5
  F3  high confidence   : confidence >= 0.6
  F4  macro safe        : macro_mode NOT in {BEAR, STRONG_BEAR}
                          (단순 contrarian 아닌 baseline-aligned 영역 — alpha discovery 결과)
  F5  cheap options     : iv_rank < 30          (options_signals → 93% / +17.7% 10d)
  F6  VP×OPT confluence : any zone with has_vp ∧ has_opt ∧ n_sources >= 2
                          (zone_confluence → 65% bounce / +5.69%)

데이터:
  macro_pred_combined  1499 trades  (2025-12-23 ~ 2026-05-11)
  options_signals       324 events  (2026-03-30 ~ 2026-05-05)
  zone_confluence      1055 events  (2026-03-30 ~ 2026-05-08)

inner-join (ticker, as_of) 후 stack_count vs outcome 측정.
in-sample / out-of-sample 분할로 overfit 검증.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from scipy import stats


def binom_pvalue(wins, n, p0=0.5):
    if n == 0:
        return np.nan
    return stats.binomtest(wins, n, p0, alternative="greater").pvalue


def load_data():
    mp = pd.read_parquet("data/results/macro_pred_combined.parquet")
    op = pd.read_parquet("data/results/options_signals.parquet")
    zc = pd.read_parquet("data/results/zone_confluence.parquet")
    return mp, op, zc


def aggregate_zone_factor(zc):
    """zone_confluence -> 1 row per (ticker, as_of) with F6 boolean."""
    zc["is_vp_opt"] = zc["has_vp"] & zc["has_opt"] & (zc["n_sources"] >= 2)
    agg = zc.groupby(["ticker", "as_of"], as_index=False)["is_vp_opt"].max()
    agg.rename(columns={"is_vp_opt": "F6_vp_opt"}, inplace=True)
    return agg


def aggregate_options_factor(op):
    """options_signals -> 1 row per (ticker, as_of) with F5 boolean."""
    op = op.copy()
    op["F5_cheap_iv"] = op["iv_rank"] < 30
    agg = op.groupby(["ticker", "as_of"], as_index=False)["F5_cheap_iv"].max()
    return agg


def build_factor_table(mp, op_agg, zc_agg):
    df = mp.copy()
    mod_cols = [c for c in df.columns if c.startswith("mod_")]
    df["n_bull"] = (df[mod_cols] > 1.0).sum(axis=1)

    df["F1_mod_consensus"] = df["n_bull"] >= 5
    df["F2_high_score"] = df["composite_score"] >= 0.5
    df["F3_high_conf"] = df["confidence"] >= 0.6
    df["F4_macro_safe"] = ~df["macro_mode"].isin(["BEAR", "STRONG_BEAR"])

    # join options + zone factors (NaN -> False if no data for that ticker-date)
    df = df.merge(op_agg, on=["ticker", "as_of"], how="left")
    df = df.merge(zc_agg, on=["ticker", "as_of"], how="left")
    df["F5_cheap_iv"] = df["F5_cheap_iv"].fillna(False).astype(bool)
    df["F6_vp_opt"] = df["F6_vp_opt"].fillna(False).astype(bool)

    fcols = ["F1_mod_consensus", "F2_high_score", "F3_high_conf",
             "F4_macro_safe", "F5_cheap_iv", "F6_vp_opt"]
    df["stack_count"] = df[fcols].sum(axis=1).astype(int)
    return df, fcols


def stack_table(df, label, horizon="5d"):
    ret_col = f"actual_ret_{horizon}"
    up_col = f"actual_up_{horizon}"
    s = df.dropna(subset=[ret_col]).copy()
    rows = []
    for n in sorted(s["stack_count"].unique()):
        sub = s[s["stack_count"] == n]
        if len(sub) < 5:
            continue
        wins = int((sub[ret_col] > 0).sum())
        win_rate = wins / len(sub)
        avg = sub[ret_col].mean()
        std = sub[ret_col].std()
        med = sub[ret_col].median()
        h_days = int(horizon.replace("d", ""))
        sharpe = (avg / std * np.sqrt(252 / h_days)) if std > 0 else 0
        pval = binom_pvalue(wins, len(sub))
        rows.append({
            "stack": n, "n": len(sub),
            "win%": f"{win_rate:.1%}",
            "avg%": f"{avg:+.2f}",
            "med%": f"{med:+.2f}",
            "Sharpe": f"{sharpe:.2f}",
            "p<0.5": f"{pval:.3f}" if not np.isnan(pval) else "—",
        })
    print(f"\n=== {label} · horizon={horizon} ===")
    print(pd.DataFrame(rows).to_string(index=False))


def per_factor_table(df, fcols, horizon="5d"):
    ret_col = f"actual_ret_{horizon}"
    s = df.dropna(subset=[ret_col]).copy()
    rows = []
    for c in fcols:
        for val in [True, False]:
            sub = s[s[c] == val]
            if len(sub) < 5:
                continue
            wins = int((sub[ret_col] > 0).sum())
            win_rate = wins / len(sub)
            avg = sub[ret_col].mean()
            std = sub[ret_col].std()
            h_days = int(horizon.replace("d", ""))
            sharpe = (avg / std * np.sqrt(252 / h_days)) if std > 0 else 0
            pval = binom_pvalue(wins, len(sub))
            rows.append({
                "factor": c, "active": val,
                "n": len(sub),
                "win%": f"{win_rate:.1%}",
                "avg%": f"{avg:+.2f}",
                "Sharpe": f"{sharpe:.2f}",
                "p<0.5": f"{pval:.3f}" if not np.isnan(pval) else "—",
            })
    print(f"\n=== Per-factor 단독 효과 · horizon={horizon} ===")
    print(pd.DataFrame(rows).to_string(index=False))


def in_out_split(df, frac=0.5):
    df = df.sort_values("as_of").reset_index(drop=True)
    pivot = int(len(df) * frac)
    return df.iloc[:pivot], df.iloc[pivot:]


def main():
    mp, op, zc = load_data()
    print(f"loaded mp={len(mp)} / op={len(op)} / zc={len(zc)}")
    op_agg = aggregate_options_factor(op)
    zc_agg = aggregate_zone_factor(zc)
    print(f"aggregated  op_agg={len(op_agg)} / zc_agg={len(zc_agg)}")

    df, fcols = build_factor_table(mp, op_agg, zc_agg)
    print(f"\nfactor table built — {len(df)} trades total")
    print("\nfactor 활성 비율:")
    for c in fcols:
        print(f"  {c}: {df[c].mean():.1%}  ({df[c].sum()} of {len(df)})")
    print(f"\nstack_count 분포:")
    print(df["stack_count"].value_counts().sort_index().to_string())

    # --- 단독 factor 검증 (기존 메모리 백테스트 재확인) ---
    per_factor_table(df, fcols, horizon="1d")
    per_factor_table(df, fcols, horizon="5d")

    # --- 전체 sample 의 stack vs outcome ---
    stack_table(df, "ALL · 전체 1499 trade", horizon="1d")
    stack_table(df, "ALL · 전체 1499 trade", horizon="5d")

    # --- in-sample / out-of-sample (시간순 50/50) ---
    in_df, out_df = in_out_split(df, 0.5)
    print(f"\n\n=== Train/Test 시간순 분할 ===")
    print(f"in-sample  ({in_df['as_of'].min()} ~ {in_df['as_of'].max()}): {len(in_df)} trades")
    print(f"out-sample ({out_df['as_of'].min()} ~ {out_df['as_of'].max()}): {len(out_df)} trades")
    stack_table(in_df, "IN-SAMPLE (학습)", horizon="5d")
    stack_table(out_df, "OUT-OF-SAMPLE (검증)", horizon="5d")
    stack_table(in_df, "IN-SAMPLE", horizon="1d")
    stack_table(out_df, "OUT-OF-SAMPLE", horizon="1d")

    # --- 결론: 단조 증가 검정 ---
    s5 = df.dropna(subset=["actual_ret_5d"])
    spearman = stats.spearmanr(s5["stack_count"], s5["actual_ret_5d"])
    s1 = df.dropna(subset=["actual_ret_1d"])
    spearman1 = stats.spearmanr(s1["stack_count"], s1["actual_ret_1d"])
    print(f"\n\n=== 단조 증가 검정 (Spearman corr) ===")
    print(f"  1d ret vs stack_count : r={spearman1.statistic:.4f}  p={spearman1.pvalue:.4f}  n={len(s1)}")
    print(f"  5d ret vs stack_count : r={spearman.statistic:.4f}  p={spearman.pvalue:.4f}  n={len(s5)}")

    # Top stack count 종목 list (오늘 적용 가능한지 보기 위해)
    print(f"\n\n=== 최근 30일 stack_count >= 3 종목 (참고용) ===")
    recent = df[df["as_of"] >= df["as_of"].max()]  # latest date
    print(f"  latest as_of: {df['as_of'].max()}")
    top_recent = df[df["stack_count"] >= 3].sort_values(["as_of", "stack_count"], ascending=[False, False]).head(30)
    cols = ["ticker", "as_of", "stack_count"] + fcols + ["composite_score", "confidence", "ev_pct_5d", "macro_mode"]
    print(top_recent[cols].to_string(index=False))


if __name__ == "__main__":
    main()

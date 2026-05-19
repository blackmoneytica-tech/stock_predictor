"""Grid search 결과의 robust 검증 — in-sample/out-of-sample 분리.

filter_grid_search 결과는 1499 rows 전체에 돌린 것 = overfitting 위험.
이 스크립트는 각 best 룰에 대해 in-sample (앞 50%)에서 발견 → out-sample (뒤 50%)에서 검증.

만약 in 좋고 out 망함 → overfit
in/out 비슷 → robust alpha
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


# filter_grid_search에서 발견한 best 후보들
CANDIDATE_RULES = [
    {
        "name": "[1]macro=safe_only",
        "filter": lambda d: d["cat"] == "safe",
    },
    {
        "name": "[2]strong_buy + very_weak + conf≥0.6",
        "filter": lambda d: (d["cat"] == "strong_buy") & (d["rs_grade"] == "very_weak") & (d["confidence"] >= 0.6),
    },
    {
        "name": "[3]safe + strong_or_better",
        "filter": lambda d: (d["cat"] == "safe") & (d["rs_grade"].isin(["strong", "very_strong"])),
    },
    {
        "name": "4. strong_buy + very_weak + conf≥0.5",
        "filter": lambda d: (d["cat"] == "strong_buy") & (d["rs_grade"] == "very_weak") & (d["confidence"] >= 0.5),
    },
    {
        "name": "5. safe + very_strong",
        "filter": lambda d: (d["cat"] == "safe") & (d["rs_grade"] == "very_strong"),
    },
    {
        "name": "M1. macro module Q4",
        "filter": lambda d: d["mod_macro"] >= d["mod_macro"].quantile(0.75),
    },
    {
        "name": "M2. sentiment module Q4",
        "filter": lambda d: d["mod_sentiment"] >= d["mod_sentiment"].quantile(0.75),
    },
    {
        "name": "Combo. macro_safe + sentiment Q3",
        "filter": lambda d: (d["cat"] == "safe") & (d["mod_sentiment"] >= d["mod_sentiment"].quantile(0.5)),
    },
    {
        "name": "Combo. macro Q4 + safe",
        "filter": lambda d: (d["cat"] == "safe") & (d["mod_macro"] >= d["mod_macro"].quantile(0.75)),
    },
    {
        "name": "Combo. macro Q4 + sentiment Q4",
        "filter": lambda d: (d["mod_macro"] >= d["mod_macro"].quantile(0.75)) &
                           (d["mod_sentiment"] >= d["mod_sentiment"].quantile(0.75)),
    },
    {
        "name": "baseline (always long)",
        "filter": lambda d: pd.Series([True] * len(d), index=d.index),
    },
]


def evaluate_part(df: pd.DataFrame, horizon: int, label: str) -> list:
    """각 후보의 in/out 분리 평가 결과."""
    actual_col = f"actual_ret_{horizon}d"
    results = []
    for rule in CANDIDATE_RULES:
        mask = rule["filter"](df)
        sub = df[mask].dropna(subset=[actual_col])
        if len(sub) < 10:
            continue
        pnl = sub[actual_col]
        win = float((pnl > 0).mean())
        avg = float(pnl.mean())
        sh = pnl.mean() / pnl.std() * np.sqrt(252 / horizon) if pnl.std() > 0 else 0
        results.append({
            "rule": rule["name"],
            "phase": label,
            "n": len(sub),
            "win": win,
            "avg": avg,
            "sharpe": sh,
        })
    return results


def main():
    p = Path("data/results/macro_pred_combined.parquet")
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["as_of"])
    df = df.sort_values("date").reset_index(drop=True)
    cut = len(df) // 2
    in_sample = df.iloc[:cut].copy()
    out_sample = df.iloc[cut:].copy()
    print(f"in-sample (n={len(in_sample)}): {in_sample['as_of'].min()} ~ {in_sample['as_of'].max()}")
    print(f"out-sample (n={len(out_sample)}): {out_sample['as_of'].min()} ~ {out_sample['as_of'].max()}")

    for horizon in [1, 5]:
        in_res = evaluate_part(in_sample, horizon, "in")
        out_res = evaluate_part(out_sample, horizon, "out")
        df_in = pd.DataFrame(in_res).set_index("rule")
        df_out = pd.DataFrame(out_res).set_index("rule")
        merged = df_in.join(df_out, lsuffix="_in", rsuffix="_out", how="inner")

        # 정렬: out_sample sharpe
        merged = merged.sort_values("sharpe_out", ascending=False)

        # 포맷
        for c in ["win_in", "win_out"]:
            merged[c] = merged[c].apply(lambda x: f"{x:.1%}")
        for c in ["avg_in", "avg_out"]:
            merged[c] = merged[c].apply(lambda x: f"{x:+.2f}%")
        for c in ["sharpe_in", "sharpe_out"]:
            merged[c] = merged[c].apply(lambda x: f"{x:+.2f}")
        merged = merged[["n_in", "win_in", "avg_in", "sharpe_in",
                         "n_out", "win_out", "avg_out", "sharpe_out"]]
        merged.columns = ["n_in", "win_in", "avg_in", "Sh_in", "n_out", "win_out", "avg_out", "Sh_out"]
        print(f"\n{'='*90}")
        print(f"=== {horizon}d in-sample vs out-sample 비교 (out Sharpe 내림) ===")
        print('='*90)
        print(merged.to_string())


if __name__ == "__main__":
    main()

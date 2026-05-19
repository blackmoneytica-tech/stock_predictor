"""Win × Avg 동시 극대화 — 다차원 필터 grid search.

기존 macro_pred_combined.parquet 활용 (1499 rows × 매크로+예측+11모듈).

Grid 차원:
1. ev_pct threshold (0.0, 0.3, 0.5, 1.0, 1.5)
2. confidence threshold (0.0, 0.5, 0.6, 0.7)
3. composite_score threshold (0, 1, 2, 3)
4. macro cat 필터 (all / not_trap / safe_only / buy_zone_only / deep_only / strong_buy_only)
5. RS grade 필터 (all / strong_or_better / very_strong_only / not_very_weak)
6. 11개 module score 필터 (top-K module score 합이 임계값 이상)

목표:
  - n >= 50 (통계적 sample 보장)
  - Pareto: win 50%+ AND avg +1%+ 둘 다 만족
  - 비교 baseline: 5d × pred+not_trap = 50.7%/+1.55% Sharpe 1.28 (n=73)
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def evaluate(s: pd.DataFrame, horizon: int = 5) -> dict:
    actual_col = f"actual_ret_{horizon}d"
    if len(s) == 0:
        return None
    pnl = s[actual_col]
    return {
        "n": len(s),
        "win": float((pnl > 0).mean()),
        "avg": float(pnl.mean()),
        "median": float(pnl.median()),
        "std": float(pnl.std()),
        "sharpe": float(pnl.mean() / pnl.std() * np.sqrt(252 / horizon)) if pnl.std() > 0 else 0,
        "max_dd": float((pnl.cumsum() - pnl.cumsum().cummax()).min()),
    }


def grid_search(df: pd.DataFrame, horizon: int = 5, min_n: int = 50) -> pd.DataFrame:
    """다차원 grid search. n>=min_n인 조합만 저장."""
    s = df.dropna(subset=[f"actual_ret_{horizon}d", "cat", "rs_grade"]).copy()

    ev_thrs = [-99, 0.0, 0.3, 0.5, 1.0, 1.5]
    conf_thrs = [0.0, 0.5, 0.6, 0.7]
    score_thrs = [-99, 0.0, 1.0, 2.0, 3.0]
    macro_filters = {
        "all": None,
        "not_trap": lambda d: d["cat"] != "trap",
        "buy_or_deep": lambda d: d["cat"].isin(["buy_zone", "deep", "strong_buy"]),
        "deep_or_buyzone": lambda d: d["cat"].isin(["deep", "buy_zone"]),
        "buy_zone_only": lambda d: d["cat"] == "buy_zone",
        "deep_only": lambda d: d["cat"] == "deep",
        "strong_buy_only": lambda d: d["cat"] == "strong_buy",
        "safe_only": lambda d: d["cat"] == "safe",
    }
    rs_filters = {
        "all": None,
        "not_very_weak": lambda d: d["rs_grade"] != "very_weak",
        "strong_or_better": lambda d: d["rs_grade"].isin(["strong", "very_strong"]),
        "very_strong": lambda d: d["rs_grade"] == "very_strong",
        "weak_only": lambda d: d["rs_grade"] == "weak",
        "very_weak_only": lambda d: d["rs_grade"] == "very_weak",
    }

    results = []
    for ev_t in ev_thrs:
        ev_mask = s["ev_pct_5d"] > ev_t if ev_t > -50 else pd.Series([True] * len(s))
        for c_t in conf_thrs:
            c_mask = s["confidence"] >= c_t
            for sc_t in score_thrs:
                sc_mask = s["composite_score"] > sc_t if sc_t > -50 else pd.Series([True] * len(s))
                for mf_name, mf_fn in macro_filters.items():
                    m_mask = mf_fn(s) if mf_fn else pd.Series([True] * len(s))
                    for rf_name, rf_fn in rs_filters.items():
                        r_mask = rf_fn(s) if rf_fn else pd.Series([True] * len(s))
                        mask = ev_mask & c_mask & sc_mask & m_mask & r_mask
                        if mask.sum() < min_n:
                            continue
                        sub = s[mask]
                        m = evaluate(sub, horizon)
                        if m is None:
                            continue
                        results.append({
                            "ev>": ev_t if ev_t > -50 else None,
                            "conf>=": c_t,
                            "score>": sc_t if sc_t > -50 else None,
                            "macro": mf_name,
                            "rs": rf_name,
                            **m,
                        })

    df_res = pd.DataFrame(results)
    return df_res


def explore_module_score(df: pd.DataFrame, horizon: int = 5, min_n: int = 50):
    """각 11개 모듈 score 단독 필터 효과."""
    s = df.dropna(subset=[f"actual_ret_{horizon}d"]).copy()
    actual_col = f"actual_ret_{horizon}d"
    mod_cols = [c for c in s.columns if c.startswith("mod_")]
    print(f"\n--- 11개 모듈 score top-quantile 단독 필터 (horizon {horizon}d) ---")
    print(f"{'module':<25s} {'thr':>8s} {'n':>5s} {'win':>7s} {'avg':>9s} {'sharpe':>8s}")
    for mc in mod_cols:
        # top 25% 잡기
        thr = s[mc].quantile(0.75)
        if pd.isna(thr):
            continue
        sub = s[s[mc] >= thr]
        if len(sub) < min_n:
            continue
        pnl = sub[actual_col]
        win = float((pnl > 0).mean())
        avg = float(pnl.mean())
        sh = pnl.mean() / pnl.std() * np.sqrt(252 / horizon) if pnl.std() > 0 else 0
        # 부호 판단 매수 strategy
        name = mc.replace("mod_", "")
        print(f"{name:<25s} {thr:>+8.2f} {len(sub):>5d} {win:>6.1%} {avg:>+8.2f}% {sh:>+7.2f}")


def main():
    p = Path("data/results/macro_pred_combined.parquet")
    if not p.exists():
        print(f"ERROR: {p} 없음. macro_pred_combo_backtest.py 먼저")
        return
    df = pd.read_parquet(p)
    print(f"loaded {len(df)} rows ({df['ticker'].nunique()} tickers, {df['as_of'].min()} ~ {df['as_of'].max()})")

    for horizon in [1, 5]:
        print(f"\n{'='*78}")
        print(f"=== Grid search Horizon {horizon}d (n_min=50, win+avg 동시 극대화) ===")
        print(f"{'='*78}")
        res = grid_search(df, horizon=horizon, min_n=50)
        if res.empty:
            print(f"  결과 없음 (n_min=50 안 채움)")
            continue

        # 정렬 기준 1: win+avg/4 통합 score (Pareto-like)
        res["score"] = res["win"] + res["avg"] / 4.0
        res = res.sort_values("score", ascending=False)

        # 포맷
        out = res.head(15).copy()
        out["win"] = out["win"].apply(lambda x: f"{x:.1%}")
        out["avg"] = out["avg"].apply(lambda x: f"{x:+.2f}%")
        out["sharpe"] = out["sharpe"].apply(lambda x: f"{x:+.2f}")
        out["max_dd"] = out["max_dd"].apply(lambda x: f"{x:+.0f}%")
        out = out[["ev>", "conf>=", "score>", "macro", "rs", "n", "win", "avg", "sharpe", "max_dd"]]
        print("\nTop 15 (win + avg/4 통합 score 내림):")
        print(out.to_string(index=False))

        # Pareto: win >= 0.5 AND avg >= 1.0
        print(f"\n--- {horizon}d Pareto (win ≥ 50% AND avg ≥ +1%) ---")
        pareto = res[(res["win"] >= 0.5) & (res["avg"] >= 1.0)].head(10).copy()
        if pareto.empty:
            print("  Pareto 조건 만족 없음")
        else:
            pareto["win"] = pareto["win"].apply(lambda x: f"{x:.1%}")
            pareto["avg"] = pareto["avg"].apply(lambda x: f"{x:+.2f}%")
            pareto["sharpe"] = pareto["sharpe"].apply(lambda x: f"{x:+.2f}")
            pareto = pareto[["ev>", "conf>=", "score>", "macro", "rs", "n", "win", "avg", "sharpe"]]
            print(pareto.to_string(index=False))

        # 11개 모듈 단독 top-quantile
        explore_module_score(df, horizon=horizon, min_n=50)


if __name__ == "__main__":
    main()

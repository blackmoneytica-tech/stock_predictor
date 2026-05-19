"""Win rate + avg PnL 둘 다 높이는 sweet spot을 multi-factor grid search.

데이터: macro_pred_combined.parquet (1499 rows × ticker × snapshot)
필드:
  - composite_score, confidence, ev_pct_5d
  - macro_mode (BULL/BEAR/CHOPPY/STRONG_*)
  - cat (drawdown), rs_grade (RS), rel_chg5/20
  - mod_* (11 모듈 score)
  - actual_ret_1d/3d/5d

차원 (각 종목별):
  A. composite_score buckets
  B. confidence buckets
  C. macro_mode
  D. drawdown cat
  E. RS grade
  F. specific module score thresholds (best modules from learn_aggregator)
  G. dist_pct (drawdown from 52w high)

목표:
- EV = win_rate × |avg_winner| - (1-win_rate) × |avg_loser|
- win + avg PnL 둘 다 baseline 대비 상승하는 조합 search

in/out-sample 분리해서 overfit 차단.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


def search_sweet_spots(
    df: pd.DataFrame, horizon: int = 5, min_n_in: int = 20,
):
    """Multi-factor filter grid search.

    in-sample(앞 50%)에서 best filter 찾고, out-sample(뒤 50%)에서 검증.
    """
    actual_col = f"actual_ret_{horizon}d"
    s = df.dropna(subset=[actual_col]).copy()
    s['date'] = pd.to_datetime(s['as_of'])
    s = s.sort_values('date').reset_index(drop=True)
    cut = len(s) // 2
    in_df = s.iloc[:cut]
    out_df = s.iloc[cut:]

    print(f"in-sample: n={len(in_df)} ({in_df['date'].min().date()} ~ {in_df['date'].max().date()})")
    print(f"out-sample: n={len(out_df)} ({out_df['date'].min().date()} ~ {out_df['date'].max().date()})")
    print(f"baseline (in-sample) win {(in_df[actual_col]>0).mean():.1%}, avg {in_df[actual_col].mean():+.3f}%")
    print(f"baseline (out-sample) win {(out_df[actual_col]>0).mean():.1%}, avg {out_df[actual_col].mean():+.3f}%")
    print()

    # 필터 옵션 정의
    macro_groups = {
        "any":      lambda d: pd.Series([True] * len(d), index=d.index),
        "BULL":     lambda d: d['macro_mode'].isin(['BULL', 'STRONG_BULL']),
        "CHOPPY":   lambda d: d['macro_mode'] == 'CHOPPY',
        "BEAR":     lambda d: d['macro_mode'].isin(['BEAR', 'STRONG_BEAR']),
        "non_BEAR": lambda d: ~d['macro_mode'].isin(['BEAR', 'STRONG_BEAR']),
    }
    cat_groups = {
        "any":          lambda d: pd.Series([True] * len(d), index=d.index),
        "buy_cat":      lambda d: d['cat'].isin(['strong_buy', 'deep', 'buy_zone']),
        "not_trap":     lambda d: d['cat'] != 'trap',
        "buy_zone_only": lambda d: d['cat'] == 'buy_zone',
        "deep_only":    lambda d: d['cat'] == 'deep',
        "strong_buy":   lambda d: d['cat'] == 'strong_buy',
    }
    rs_groups = {
        "any":          lambda d: pd.Series([True] * len(d), index=d.index),
        "strong":       lambda d: d['rs_grade'].isin(['strong', 'very_strong']),
        "weak":         lambda d: d['rs_grade'].isin(['weak', 'very_weak']),
        "not_very_weak": lambda d: d['rs_grade'] != 'very_weak',
        "very_strong":  lambda d: d['rs_grade'] == 'very_strong',
    }
    score_filters = {
        "any":          lambda d: pd.Series([True] * len(d), index=d.index),
        "score>+2":     lambda d: d['composite_score'] > 2,
        "score>+1":     lambda d: d['composite_score'] > 1,
        "score>0":      lambda d: d['composite_score'] > 0,
        "score<0":      lambda d: d['composite_score'] < 0,
        "score<-1":     lambda d: d['composite_score'] < -1,
        "score<-2":     lambda d: d['composite_score'] < -2,
    }
    conf_filters = {
        "any":  lambda d: pd.Series([True] * len(d), index=d.index),
        ">0.5": lambda d: d['confidence'] > 0.5,
        ">0.6": lambda d: d['confidence'] > 0.6,
        ">0.7": lambda d: d['confidence'] > 0.7,
    }
    ev_filters = {
        "any":      lambda d: pd.Series([True] * len(d), index=d.index),
        "ev>+0.3": lambda d: d['ev_pct_5d'] > 0.3,
        "ev>+1.0": lambda d: d['ev_pct_5d'] > 1.0,
        "ev<-0.3": lambda d: d['ev_pct_5d'] < -0.3,
        "ev<-1.0": lambda d: d['ev_pct_5d'] < -1.0,
    }
    rsi_filters = {
        "any":      lambda d: pd.Series([True] * len(d), index=d.index),
        "<35":      lambda d: d.get('mod_mean_reversion', pd.Series([0]*len(d), index=d.index)) > 2,
        ">2":       lambda d: d.get('mod_short_squeeze', pd.Series([0]*len(d), index=d.index)) > 2,
        "<-2":      lambda d: d.get('mod_short_squeeze', pd.Series([0]*len(d), index=d.index)) < -2,
    }

    # Grid search
    print(f"=== Multi-factor grid search ({horizon}d) — in-sample ===")
    candidates = []
    for (mn, mf), (cn, cf), (rn, rf), (sn, sf), (en, ef), (qn, qf) in product(
        macro_groups.items(), cat_groups.items(), rs_groups.items(),
        score_filters.items(), ev_filters.items(), conf_filters.items(),
    ):
        try:
            mask_in = mf(in_df) & cf(in_df) & rf(in_df) & sf(in_df) & ef(in_df) & qf(in_df)
        except Exception:
            continue
        sub_in = in_df[mask_in]
        if len(sub_in) < min_n_in:
            continue
        win_in = (sub_in[actual_col] > 0).mean()
        avg_in = sub_in[actual_col].mean()
        # in-sample sweet spot 후보: win >= 55% AND avg >= 1%
        if win_in >= 0.55 and avg_in >= 1.0:
            candidates.append({
                "filter": f"macro={mn}, cat={cn}, rs={rn}, score={sn}, ev={en}, conf={qn}",
                "n_in": len(sub_in),
                "win_in": win_in,
                "avg_in": avg_in,
                "_masks": (mn, cn, rn, sn, en, qn),
            })

    if not candidates:
        print("No sweet spot found (win>=55%, avg>=1% 동시 충족 필터 없음)")
        return

    print(f"\n총 {len(candidates)} 개 후보 (in-sample win>=55%, avg>=1%)\n")

    # in-sample 기준 EV (= win × avg) 상위 30 검토
    cands_df = pd.DataFrame(candidates)
    cands_df['ev_in'] = cands_df['win_in'] * cands_df['avg_in']
    cands_df = cands_df.sort_values('ev_in', ascending=False).head(30).reset_index(drop=True)

    # 각 후보를 out-sample에서 검증
    out_rows = []
    for _, c in cands_df.iterrows():
        mn, cn, rn, sn, en, qn = c['_masks']
        mask_out = (macro_groups[mn](out_df) & cat_groups[cn](out_df)
                    & rs_groups[rn](out_df) & score_filters[sn](out_df)
                    & ev_filters[en](out_df) & conf_filters[qn](out_df))
        sub_out = out_df[mask_out]
        n_out = len(sub_out)
        if n_out == 0:
            win_out = float('nan')
            avg_out = float('nan')
        else:
            win_out = (sub_out[actual_col] > 0).mean()
            avg_out = sub_out[actual_col].mean()
        out_rows.append({
            "filter": c['filter'],
            "n_in": c['n_in'],
            "win_in": f"{c['win_in']:.1%}",
            "avg_in": f"{c['avg_in']:+.2f}%",
            "n_out": n_out,
            "win_out": f"{win_out:.1%}" if not np.isnan(win_out) else "—",
            "avg_out": f"{avg_out:+.2f}%" if not np.isnan(avg_out) else "—",
            "generalize": (
                "✅" if not np.isnan(win_out) and win_out >= 0.55 and avg_out >= 0.5
                else "⚠️" if not np.isnan(win_out) and win_out >= 0.50
                else "❌"
            ),
        })
    result = pd.DataFrame(out_rows)
    print("=== Top 30 candidates with out-sample validation ===")
    print(result.to_string(index=False))
    print()

    # Generalize 잘 되는 것들만
    gen = result[result['generalize'] == '✅']
    print(f"\n✅ Generalize 잘되는 ({len(gen)}개):")
    if len(gen):
        print(gen.to_string(index=False))


def main():
    p = Path("data/results/macro_pred_combined.parquet")
    if not p.exists():
        print(f"ERROR: {p} 없음. macro_pred_combo_backtest.py 먼저 실행")
        return
    df = pd.read_parquet(p)
    print(f"loaded {len(df)} rows from {p}")
    print(f"date range: {df['as_of'].min()} ~ {df['as_of'].max()}\n")

    for h in [1, 5]:
        print(f"\n{'#'*80}")
        print(f"# Horizon {h}d")
        print(f"{'#'*80}\n")
        search_sweet_spots(df, horizon=h, min_n_in=20)


if __name__ == "__main__":
    main()

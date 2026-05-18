"""Walk-forward 평가 — improvement_data.parquet에서:

1. macro_mode 별 strategy 평가 (BULL/CHOPPY/BEAR 각각)
2. in-sample vs out-of-sample 분리 (앞 50% / 뒤 50%) — alpha generalize 검증
3. 시기별 분포 (월별)
4. 최종 권고 룰
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.backtest.improvement_backtest import evaluate_strategies


def summarize_by_macro(df: pd.DataFrame, horizon: int = 5):
    """macro_mode별 baseline vs E_long_in_BULL 등 핵심 strategy."""
    pred_col = f"pred_ret_{horizon}d"
    actual_col = f"actual_ret_{horizon}d"

    print(f"\n--- {horizon}d × macro_mode 분기 ---")
    rows = []
    for mode in df['macro_mode'].unique():
        sub = df[df['macro_mode'] == mode]
        if sub.empty:
            continue
        # baseline (always long)
        baseline_win = (sub[actual_col] > 0).mean()
        baseline_pnl = sub[actual_col].mean()
        # raw direction
        raw_dir = sub.copy()
        raw_dir['pnl'] = np.sign(raw_dir[pred_col]) * raw_dir[actual_col]
        raw_win = (raw_dir['pnl'] > 0).mean()
        raw_pnl = raw_dir['pnl'].mean()

        rows.append({
            "macro_mode": mode,
            "n": len(sub),
            "always_long_win": f"{baseline_win:.1%}",
            "always_long_pnl": f"{baseline_pnl:+.2f}%",
            "system_signal_win": f"{raw_win:.1%}",
            "system_signal_pnl": f"{raw_pnl:+.2f}%",
            "actual_up_pct": f"{(sub[actual_col] > 0).mean():.1%}",
        })
    out = pd.DataFrame(rows).sort_values('n', ascending=False)
    print(out.to_string(index=False))


def in_out_of_sample(df: pd.DataFrame, horizon: int = 5):
    """기간 절반 기준으로 분리. in-sample에서 best strategy가 out-of-sample에서도 best?"""
    s = df.copy().sort_values('as_of').reset_index(drop=True)
    cut = len(s) // 2
    in_df = s.iloc[:cut]
    out_df = s.iloc[cut:]
    in_dates = (in_df['as_of'].min(), in_df['as_of'].max())
    out_dates = (out_df['as_of'].min(), out_df['as_of'].max())
    print(f"\n--- {horizon}d × in/out-of-sample ---")
    print(f"in-sample (n={len(in_df)}): {in_dates[0]} ~ {in_dates[1]}")
    print(f"out-sample (n={len(out_df)}): {out_dates[0]} ~ {out_dates[1]}")
    print()

    in_eval = evaluate_strategies(in_df, horizon=horizon)
    out_eval = evaluate_strategies(out_df, horizon=horizon)

    # 합쳐서 비교
    in_eval = in_eval.set_index('strategy')[['n', 'win_rate', 'avg_pnl']]
    out_eval = out_eval.set_index('strategy')[['n', 'win_rate', 'avg_pnl']]
    in_eval.columns = ['in_n', 'in_win', 'in_pnl']
    out_eval.columns = ['out_n', 'out_win', 'out_pnl']
    merged = in_eval.join(out_eval, how='outer').fillna(0)
    merged['in_win'] = merged['in_win'].apply(lambda x: f"{x:.1%}")
    merged['in_pnl'] = merged['in_pnl'].apply(lambda x: f"{x:+.2f}%")
    merged['out_win'] = merged['out_win'].apply(lambda x: f"{x:.1%}")
    merged['out_pnl'] = merged['out_pnl'].apply(lambda x: f"{x:+.2f}%")
    print(merged.to_string())


def monthly_breakdown(df: pd.DataFrame, horizon: int = 5):
    """월별 macro_mode 분포 + baseline vs E_long_in_BULL win."""
    actual_col = f"actual_ret_{horizon}d"
    s = df.copy()
    s['month'] = pd.to_datetime(s['as_of']).dt.to_period('M')
    print(f"\n--- {horizon}d × 월별 분포 ---")
    g = s.groupby('month').agg(
        n=('macro_mode', 'size'),
        baseline_win=(actual_col, lambda x: (x > 0).mean()),
        baseline_pnl=(actual_col, 'mean'),
        macro_modes=('macro_mode', lambda x: x.mode().iloc[0] if not x.mode().empty else "?"),
    )
    g['baseline_win'] = g['baseline_win'].apply(lambda x: f"{x:.1%}")
    g['baseline_pnl'] = g['baseline_pnl'].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())


def main():
    p = Path("data/results/improvement_data.parquet")
    df = pd.read_parquet(p)
    print(f"loaded {len(df)} rows, {df['ticker'].nunique()} tickers")
    print(f"date range: {df['as_of'].min()} ~ {df['as_of'].max()}")
    print()
    print("macro_mode distribution:")
    print(df['macro_mode'].value_counts())

    for h in [1, 5]:
        print(f"\n{'#' * 70}")
        print(f"# Horizon {h}d")
        print(f"{'#' * 70}")
        summarize_by_macro(df, horizon=h)
        in_out_of_sample(df, horizon=h)

    monthly_breakdown(df, horizon=5)


if __name__ == "__main__":
    main()

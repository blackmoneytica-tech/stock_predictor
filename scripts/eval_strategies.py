"""improvement_data.parquet → strategy 평가 (이미 모은 데이터로)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.backtest.improvement_backtest import evaluate_strategies


def main():
    p = Path("data/results/improvement_data.parquet")
    df = pd.read_parquet(p)
    print(f"loaded {len(df)} rows from {p}")
    print(f"date range: {df['as_of'].min()} ~ {df['as_of'].max()}")
    print(f"tickers: {df['ticker'].nunique()} ({list(df['ticker'].unique())})")
    print()

    for h in [1, 3, 5]:
        print(f"\n{'=' * 75}")
        print(f"=== Horizon {h}d ===")
        print(f"{'=' * 75}")
        eval_df = evaluate_strategies(df, horizon=h)
        eval_df['win_rate'] = eval_df['win_rate'].apply(lambda x: f"{x:.1%}")
        eval_df['avg_pnl'] = eval_df['avg_pnl'].apply(lambda x: f"{x:+.2f}%")
        eval_df['total_pnl'] = eval_df['total_pnl'].apply(lambda x: f"{x:+.1f}%")
        eval_df['median_pnl'] = eval_df['median_pnl'].apply(lambda x: f"{x:+.2f}%")
        print(eval_df.to_string(index=False))


if __name__ == "__main__":
    main()

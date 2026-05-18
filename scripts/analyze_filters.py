"""기존 ma_variants.parquet → 어떤 필터/조건이 acc 50%+ 만드는지 분석.

SMA variant만 사용 (raw 시스템 동작). 다양한 segmentation:
- composite_score 분위수
- confidence 분위수
- score × macro_mode 조합
- score × confidence 결합 필터
- pred_ret magnitude 필터 (|pred_ret| > X일 때만 진입)
- bias 보정 효과 시뮬레이션
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = Path(__file__).resolve().parents[1] / "data" / "results" / "ma_variants.parquet"
    df = pd.read_parquet(p)

    # 1d 또는 5d? 마지막 backtest는 5d였음. 보고 horizon 자동 추정.
    # actual_ret이 1.5%+ 평균이면 5d, 0.5%이하면 1d
    mean_abs = df['actual_ret'].abs().mean()
    horizon = "5d" if mean_abs > 1.5 else "1d"

    # SMA variant만
    s = df[df['variant'] == 'sma'].copy()
    s['abs_pred'] = s['pred_ret'].abs()
    s['abs_score'] = s['composite_score'].abs()
    s['actual_sign'] = np.sign(s['actual_ret'])
    s['pred_sign'] = np.sign(s['pred_ret'])

    print(f"=== Horizon: {horizon}, n={len(s)} ===")
    print(f"전체 acc: {s['dir_correct'].mean():.1%}")
    print(f"mean_pred: {s['pred_ret'].mean():+.2f}%, mean_actual: {s['actual_ret'].mean():+.2f}%")
    print(f"actual UP 비율: {(s['actual_ret'] > 0).mean():.1%}")
    print()

    # 1) composite_score 분위수
    print("--- 1) composite_score 분위수 ---")
    s['score_bucket'] = pd.cut(
        s['composite_score'],
        bins=[-10, -3, -1, 1, 3, 10],
        labels=["score<-3", "-3~-1", "-1~1", "1~3", "score>3"],
    )
    g = s.groupby('score_bucket', observed=False).agg(
        n=('dir_correct', 'size'),
        acc=('dir_correct', 'mean'),
        mean_pred=('pred_ret', 'mean'),
        mean_actual=('actual_ret', 'mean'),
    )
    print(g.round(3).to_string())
    print()

    # 2) confidence 분위수
    print("--- 2) confidence 분위수 ---")
    s['conf_bucket'] = pd.cut(s['confidence'], bins=[0, 0.5, 0.6, 0.7, 1.0])
    g = s.groupby('conf_bucket', observed=False).agg(
        n=('dir_correct', 'size'),
        acc=('dir_correct', 'mean'),
    )
    print(g.round(3).to_string())
    print()

    # 3) |pred_ret| magnitude — 작은 신호 무시
    print("--- 3) |pred_ret| 분위수 (작은 신호 vs 큰 신호) ---")
    s['abs_pred_bucket'] = pd.cut(
        s['abs_pred'],
        bins=[-0.001, 0.3, 1.0, 2.0, 100],
        labels=["<0.3% (noise)", "0.3~1%", "1~2%", ">2% (강)"],
    )
    g = s.groupby('abs_pred_bucket', observed=False).agg(
        n=('dir_correct', 'size'),
        acc=('dir_correct', 'mean'),
        mean_pnl=('actual_ret', lambda x: float(np.mean(np.sign(s.loc[x.index, 'pred_ret']) * x))),
    )
    print(g.round(3).to_string())
    print()

    # 4) macro_mode 별
    print("--- 4) macro_mode 별 ---")
    g = s.groupby('macro_mode').agg(
        n=('dir_correct', 'size'),
        acc=('dir_correct', 'mean'),
        mean_pred=('pred_ret', 'mean'),
        mean_actual=('actual_ret', 'mean'),
    )
    print(g.round(3).sort_values('n', ascending=False).to_string())
    print()

    # 5) 시스템 bias 보정 시뮬레이션
    print("--- 5) Bias 보정 시뮬 (mean_pred 0으로 shift) ---")
    bias = s['pred_ret'].mean()
    s['pred_ret_debiased'] = s['pred_ret'] - bias
    debiased_correct = (
        (np.sign(s['pred_ret_debiased']) == np.sign(s['actual_ret']))
        & (s['pred_ret_debiased'].abs() > 0.3)
    )
    # 노이즈 임계값 미만일 땐 기존 룰
    s['debiased_correct'] = np.where(
        s['pred_ret_debiased'].abs() < 0.3,
        False,  # 신호 약함 = trade X
        np.sign(s['pred_ret_debiased']) == np.sign(s['actual_ret']),
    )
    n_signal = (s['pred_ret_debiased'].abs() >= 0.3).sum()
    acc_signal = (
        np.sign(s.loc[s['pred_ret_debiased'].abs() >= 0.3, 'pred_ret_debiased'])
        == np.sign(s.loc[s['pred_ret_debiased'].abs() >= 0.3, 'actual_ret'])
    ).mean() if n_signal > 0 else float('nan')
    print(f"system bias: {bias:+.3f}% → 보정 후 |pred|>0.3 신호: n={n_signal}, acc={acc_signal:.1%}")
    print()

    # 6) long-only with 강신호 + 정상 macro
    print("--- 6) Long-only 필터 조합 ---")
    filters = [
        ("score > 2", s['composite_score'] > 2),
        ("score > 3", s['composite_score'] > 3),
        ("pred_ret > 1%", s['pred_ret'] > 1.0),
        ("pred_ret > 2%", s['pred_ret'] > 2.0),
        ("conf > 0.65 AND pred_ret > 1%", (s['confidence'] > 0.65) & (s['pred_ret'] > 1.0)),
        ("conf > 0.65 AND score > 2", (s['confidence'] > 0.65) & (s['composite_score'] > 2)),
        ("BULL macro AND pred_ret > 0.5%", (s['macro_mode'] == 'BULL') & (s['pred_ret'] > 0.5)),
        ("BULL macro AND score > 2", (s['macro_mode'] == 'BULL') & (s['composite_score'] > 2)),
        ("score > 3 AND BULL", (s['composite_score'] > 3) & (s['macro_mode'] == 'BULL')),
    ]
    print(f"{'filter':<40s} {'n':>4s} {'win%':>7s} {'avg_pnl':>9s}")
    print("-" * 65)
    # win = actual_ret > 0 (long-only)
    base_winrate = (s['actual_ret'] > 0).mean()
    print(f"{'baseline (모두 long)':<40s} {len(s):>4d} {base_winrate:>6.1%} {s['actual_ret'].mean():>+8.2f}%")
    for name, mask in filters:
        sub = s[mask]
        if len(sub) == 0:
            continue
        win = (sub['actual_ret'] > 0).mean()
        avg = sub['actual_ret'].mean()
        print(f"{name:<40s} {len(sub):>4d} {win:>6.1%} {avg:>+8.2f}%")
    print()

    # 7) short-only 강신호 + BEAR macro
    print("--- 7) Short-only 필터 조합 ---")
    filters_short = [
        ("score < -2", s['composite_score'] < -2),
        ("pred_ret < -1%", s['pred_ret'] < -1.0),
        ("pred_ret < -2%", s['pred_ret'] < -2.0),
        ("score < -2 AND BEAR macro", (s['composite_score'] < -2) & (s['macro_mode'].isin(['BEAR', 'STRONG_BEAR']))),
        ("score < -2 AND not BULL", (s['composite_score'] < -2) & (~s['macro_mode'].isin(['BULL', 'STRONG_BULL']))),
    ]
    print(f"{'filter':<40s} {'n':>4s} {'win%':>7s} {'avg_pnl':>9s}")
    print("-" * 65)
    for name, mask in filters_short:
        sub = s[mask]
        if len(sub) == 0:
            continue
        # short PnL = -actual_ret
        win = (sub['actual_ret'] < 0).mean()
        avg = -sub['actual_ret'].mean()
        print(f"{name:<40s} {len(sub):>4d} {win:>6.1%} {avg:>+8.2f}%")


if __name__ == "__main__":
    main()

"""약세장 alpha 추가 탐색 — Sweet Spot 부족분 보충용 가설 검증.

기존 (강세장+약세장 검증된 alpha):
  - Sweet Spot contrarian — 적중률 약세장 7.8% (drop), 강세장 1.6%

목표: 약세장에서 진입 기회 늘리고 자본 효율↑.

가설:
  1. **VIX spike → 5d 후 mean reversion** (VIX 40+에서 oversold bounce)
  2. **Drawdown 30%+ 종목 (capitulation)** + 5일 반등
  3. **Pair trade: 매크로 약세 + 종목 strong RS** (시장은 빠지지만 종목은 outperform)
  4. **Module tier overhyped_warning** — 2022 검증서 약세장 +1.44% 의외 alpha

기존 bear_2022.parquet (720 events) 활용.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p = Path("data/results/bear_2022.parquet")
    if not p.exists():
        print(f"ERROR: {p} 없음")
        return
    df = pd.read_parquet(p)
    print(f"loaded {len(df)} rows from {p}\n")

    base_5d = df['actual_ret_5d'].dropna()
    base_10d = df['actual_ret_10d'].dropna()
    print(f"Baseline 5d: win {(base_5d>0).mean():.1%}, avg {base_5d.mean():+.2f}%")
    print(f"Baseline 10d: win {(base_10d>0).mean():.1%}, avg {base_10d.mean():+.2f}%")
    print()

    # ── 가설 1: composite_score 매우 낮음 (-3 이하) — capitulation oversold ──
    print("=== 가설 1: 시스템도 매우 약세 예측 (score ≤ -3) → contrarian 반등? ===")
    for thr, label in [(-5, "score≤-5"), (-3, "score≤-3"), (-2, "score≤-2")]:
        sub = df[df['composite_score'] <= thr].dropna(subset=['actual_ret_5d'])
        if len(sub) < 5:
            continue
        win5 = (sub['actual_ret_5d'] > 0).mean()
        avg5 = sub['actual_ret_5d'].mean()
        win10 = (sub['actual_ret_10d'] > 0).mean()
        avg10 = sub['actual_ret_10d'].mean()
        print(f"  {label:<10s} n={len(sub):>4d} | 5d {win5:.1%}/{avg5:+.2f}% | 10d {win10:.1%}/{avg10:+.2f}%")
    print()

    # ── 가설 2: 모듈별 score 분포에서 oversold 종목 (mean_reversion + short_squeeze) ──
    if 'mod_mean_reversion' in df.columns:
        print("=== 가설 2: Mean reversion 모듈 강한 신호 → 약세장 반등? ===")
        for thr, label in [(2, "MR>2"), (3, "MR>3"), (4, "MR>4")]:
            sub = df[df['mod_mean_reversion'] >= thr].dropna(subset=['actual_ret_5d'])
            if len(sub) < 5:
                continue
            win5 = (sub['actual_ret_5d'] > 0).mean()
            avg5 = sub['actual_ret_5d'].mean()
            win10 = (sub['actual_ret_10d'] > 0).mean()
            avg10 = sub['actual_ret_10d'].mean()
            print(f"  {label:<10s} n={len(sub):>4d} | 5d {win5:.1%}/{avg5:+.2f}% | 10d {win10:.1%}/{avg10:+.2f}%")
        print()

    # ── 가설 3: Short squeeze 모듈 강함 (oversold + 공매도 누적) ──
    if 'mod_short_squeeze' in df.columns:
        print("=== 가설 3: Short Squeeze 신호 → 약세장 spike 반등? ===")
        for thr, label in [(2, "SS>2"), (3, "SS>3"), (4, "SS>4")]:
            sub = df[df['mod_short_squeeze'] >= thr].dropna(subset=['actual_ret_5d'])
            if len(sub) < 5:
                continue
            win5 = (sub['actual_ret_5d'] > 0).mean()
            avg5 = sub['actual_ret_5d'].mean()
            print(f"  {label:<10s} n={len(sub):>4d} | 5d {win5:.1%}/{avg5:+.2f}%")
        print()

    # ── 가설 4: Overhyped warning tier (이전 발견: 약세장 +1.44%) ──
    print("=== 가설 4: Overhyped Warning tier (n_bear=0 만장일치 매수) ===")
    ohw = df[df['module_tier'] == 'overhyped_warning'].dropna(subset=['actual_ret_5d'])
    if len(ohw):
        win5 = (ohw['actual_ret_5d'] > 0).mean()
        avg5 = ohw['actual_ret_5d'].mean()
        win10 = (ohw['actual_ret_10d'] > 0).mean()
        avg10 = ohw['actual_ret_10d'].mean()
        print(f"  n={len(ohw)} | 5d {win5:.1%}/{avg5:+.2f}% | 10d {win10:.1%}/{avg10:+.2f}%")
    print()

    # ── 가설 5: Composite score 극단 양수 + 약세장 = 더 큰 mean reversion? ──
    print("=== 가설 5: 매우 강한 매수 신호 (score≥+3) — 약세장에서도 양수? ===")
    for thr in [2, 3, 4]:
        sub = df[df['composite_score'] >= thr].dropna(subset=['actual_ret_5d'])
        if len(sub) < 5:
            continue
        win5 = (sub['actual_ret_5d'] > 0).mean()
        avg5 = sub['actual_ret_5d'].mean()
        print(f"  score≥+{thr} n={len(sub):>4d} | 5d {win5:.1%}/{avg5:+.2f}%")
    print()

    # ── 가설 6: macro_mode별 best entry condition ──
    print("=== 가설 6: macro별 진입 최적화 ===")
    for mode in df['macro_mode'].unique():
        sub = df[df['macro_mode'] == mode]
        if len(sub) < 10:
            continue
        # 5d
        w5 = (sub['actual_ret_5d'] > 0).mean()
        a5 = sub['actual_ret_5d'].mean()
        print(f"  {mode:<15s} n={len(sub):>4d} | baseline 5d {w5:.1%}/{a5:+.2f}%")
        # 그 안에서 sweet spot 적중률
        ss = sub[sub['sweet_spot_active']]
        if len(ss):
            ws = (ss['actual_ret_5d'] > 0).mean()
            avs = ss['actual_ret_5d'].mean()
            print(f"      └ sweet spot n={len(ss):>3d} | {ws:.1%}/{avs:+.2f}%")
    print()

    # ── 가설 7: 다중 조건 combo (모든 가설 결합) ──
    print("=== 가설 7: Multi-factor combo (약세장 sweet spot 보완) ===")
    combos = [
        ("MR>2 + score<-2",
         (df['mod_mean_reversion'] >= 2) & (df['composite_score'] <= -2)),
        ("MR>3 + score<0",
         (df['mod_mean_reversion'] >= 3) & (df['composite_score'] < 0)),
        ("MR>2 + macro BEAR/STRONG_BEAR",
         (df['mod_mean_reversion'] >= 2) & df['macro_mode'].isin(['BEAR','STRONG_BEAR'])),
        ("score>+3 (강매수 신호)",
         df['composite_score'] >= 3),
        ("score<-3 + macro STRONG_BEAR (oversold)",
         (df['composite_score'] <= -3) & (df['macro_mode'] == 'STRONG_BEAR')),
    ]
    for name, mask in combos:
        sub = df[mask].dropna(subset=['actual_ret_5d'])
        if len(sub) < 5:
            continue
        win5 = (sub['actual_ret_5d'] > 0).mean()
        avg5 = sub['actual_ret_5d'].mean()
        win10 = (sub['actual_ret_10d'] > 0).mean() if len(sub.dropna(subset=['actual_ret_10d'])) > 5 else float('nan')
        avg10 = sub['actual_ret_10d'].mean() if not np.isnan(win10) else float('nan')
        marker = "⭐" if win5 > 0.55 and avg5 > 1.0 else ("✅" if win10 > 0.6 else "")
        print(f"  {name:<45s} n={len(sub):>4d} | 5d {win5:.1%}/{avg5:+.2f}% | 10d {win10:.1%}/{avg10:+.2f}% {marker}")


if __name__ == "__main__":
    main()

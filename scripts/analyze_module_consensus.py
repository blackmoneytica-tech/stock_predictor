"""모듈 우세 개수(매수/매도) vs 실제 outcome — composite_score보다 단순 카운팅이 의미있나?

데이터: module_scores.parquet (1499 rows × 11 mod_* score)

검증:
A. 매수 우세 모듈 개수 (mod_X > 1) buckets → win rate, avg PnL
B. 매도 우세 모듈 개수 (mod_X < -1) buckets → win rate
C. (매수 개수, 매도 개수) 매트릭스 — 합의/충돌별 outcome
D. 극단 비교: 모두 매수 우세 vs 모두 매도 우세 → 실제 결과
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

p = Path("data/results/module_scores.parquet")
df = pd.read_parquet(p)

mod_cols = [c for c in df.columns if c.startswith("mod_")]
print(f"loaded {len(df)} rows, {len(mod_cols)} modules: {[c[4:] for c in mod_cols]}\n")

# 모듈 score → bull / neutral / bear count
def count_dir(row, threshold=1.0):
    bull = sum(1 for c in mod_cols if row[c] > threshold)
    bear = sum(1 for c in mod_cols if row[c] < -threshold)
    neutral = len(mod_cols) - bull - bear
    return bull, bear, neutral

df['n_bull'], df['n_bear'], df['n_neutral'] = zip(
    *df.apply(lambda r: count_dir(r), axis=1)
)
df['consensus_net'] = df['n_bull'] - df['n_bear']

print("=== n_bull 분포 ===")
print(df['n_bull'].value_counts().sort_index().to_string())
print("\n=== n_bear 분포 ===")
print(df['n_bear'].value_counts().sort_index().to_string())

# === A. 매수 우세 모듈 개수별 ===
print("\n\n### A. 매수 우세 모듈 개수 (mod_X > 1) → 1d / 5d outcome ###")
for h in [1, 5]:
    actual_col = f"actual_ret_{h}d"
    s = df.dropna(subset=[actual_col]).copy()
    print(f"\n  Horizon {h}d:")
    print(f"  {'n_bull':<10} {'n':<6} {'win%':<8} {'avg%':<8} {'Sharpe':<7}")
    for nb in sorted(s['n_bull'].unique()):
        sub = s[s['n_bull'] == nb]
        if len(sub) < 5:
            continue
        win = (sub[actual_col] > 0).mean()
        avg = sub[actual_col].mean()
        std = sub[actual_col].std()
        sharpe = (avg / std * np.sqrt(252 / h)) if std > 0 else 0
        print(f"  {nb:<10} {len(sub):<6} {win:.1%}   {avg:+.2f}%   {sharpe:.2f}")

# === B. 매도 우세 ===
print("\n\n### B. 매도 우세 모듈 개수 (mod_X < -1) → outcome ###")
for h in [1, 5]:
    actual_col = f"actual_ret_{h}d"
    s = df.dropna(subset=[actual_col]).copy()
    print(f"\n  Horizon {h}d:")
    print(f"  {'n_bear':<10} {'n':<6} {'win%':<8} {'avg%':<8} {'Sharpe':<7}")
    for nb in sorted(s['n_bear'].unique()):
        sub = s[s['n_bear'] == nb]
        if len(sub) < 5:
            continue
        win = (sub[actual_col] > 0).mean()
        avg = sub[actual_col].mean()
        std = sub[actual_col].std()
        sharpe = (avg / std * np.sqrt(252 / h)) if std > 0 else 0
        print(f"  {nb:<10} {len(sub):<6} {win:.1%}   {avg:+.2f}%   {sharpe:.2f}")

# === C. Net consensus (bull - bear) ===
print("\n\n### C. Net consensus (bull - bear) → 5d outcome ###")
actual_col = "actual_ret_5d"
s = df.dropna(subset=[actual_col]).copy()
print(f"  {'net':<8} {'n':<6} {'win%':<8} {'avg%':<8} {'Sharpe':<7}")
for net in sorted(s['consensus_net'].unique()):
    sub = s[s['consensus_net'] == net]
    if len(sub) < 8:
        continue
    win = (sub[actual_col] > 0).mean()
    avg = sub[actual_col].mean()
    std = sub[actual_col].std()
    sharpe = (avg / std * np.sqrt(252 / 5)) if std > 0 else 0
    print(f"  {net:+d}      {len(sub):<6} {win:.1%}   {avg:+.2f}%   {sharpe:.2f}")

# === D. 극단 비교 ===
print("\n\n### D. 극단 비교 (5d horizon) ###")
print("  '모두 매수 동의 (n_bull≥7)' vs '모두 매도 동의 (n_bear≥7)' vs '의견 분산'")
actual_col = "actual_ret_5d"
s = df.dropna(subset=[actual_col]).copy()
groups = {
    "강한 합의 (n_bull≥7)":  s[s['n_bull'] >= 7],
    "약한 합의 (n_bull 5~6)": s[s['n_bull'].between(5, 6)],
    "중립 (n_bull 3~4)":      s[s['n_bull'].between(3, 4)],
    "약한 약세 (n_bear 4~5)": s[(s['n_bear'].between(4, 5)) & (s['n_bull'] <= 3)],
    "강한 약세 (n_bear≥6)":   s[s['n_bear'] >= 6],
    "전체 baseline":          s,
}
print(f"  {'group':<28} {'n':<6} {'win%':<8} {'avg%':<8} {'Sharpe':<7}")
for name, sub in groups.items():
    if len(sub) < 5:
        continue
    win = (sub[actual_col] > 0).mean()
    avg = sub[actual_col].mean()
    std = sub[actual_col].std()
    sharpe = (avg / std * np.sqrt(252 / 5)) if std > 0 else 0
    print(f"  {name:<28} {len(sub):<6} {win:.1%}   {avg:+.2f}%   {sharpe:.2f}")

# === E. n_bull≥7 + macro 상호작용 ===
print("\n\n### E. '강한 매수 합의 (n_bull≥7)' × macro_mode → 진짜 alpha 있나? (5d) ###")
strong_buy_consensus = s[s['n_bull'] >= 7]
print(f"  {'macro':<15} {'n':<6} {'win%':<8} {'avg%':<8} {'Sharpe':<7}")
for mode in strong_buy_consensus['macro_mode'].value_counts().index:
    sub = strong_buy_consensus[strong_buy_consensus['macro_mode'] == mode]
    if len(sub) < 5:
        continue
    win = (sub[actual_col] > 0).mean()
    avg = sub[actual_col].mean()
    std = sub[actual_col].std()
    sharpe = (avg / std * np.sqrt(252 / 5)) if std > 0 else 0
    print(f"  {mode:<15} {len(sub):<6} {win:.1%}   {avg:+.2f}%   {sharpe:.2f}")

# === F. n_bear≥6 (강한 매도 합의) — 정말 손실인지 or contrarian 반등? ===
print("\n\n### F. '강한 매도 합의 (n_bear≥6)' × macro → 손실 or contrarian 반등? (5d) ###")
strong_bear_consensus = s[s['n_bear'] >= 6]
print(f"  {'macro':<15} {'n':<6} {'win%':<8} {'avg%':<8} {'Sharpe':<7}")
for mode in strong_bear_consensus['macro_mode'].value_counts().index:
    sub = strong_bear_consensus[strong_bear_consensus['macro_mode'] == mode]
    if len(sub) < 5:
        continue
    win = (sub[actual_col] > 0).mean()
    avg = sub[actual_col].mean()
    std = sub[actual_col].std()
    sharpe = (avg / std * np.sqrt(252 / 5)) if std > 0 else 0
    print(f"  {mode:<15} {len(sub):<6} {win:.1%}   {avg:+.2f}%   {sharpe:.2f}")

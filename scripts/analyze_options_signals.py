"""저장된 options_signal_events.parquet에서 의미있는 신호 조합 찾기.

historical walls/volume 미작동 → news + iv_rank + max_pain dist + composite로만.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

df = pd.read_parquet("data/results/options_signal_events.parquet")
df = df.dropna(subset=['ret_5d']).copy()
print(f"n={len(df)} events, {df['ticker'].nunique()} tickers")
print(f"date: {df['as_of'].min()} ~ {df['as_of'].max()}")
print(f"baseline 5d: win {(df['ret_5d']>0).mean():.1%}, avg {df['ret_5d'].mean():+.2f}%")
print()

baseline_win = (df['ret_5d']>0).mean()
baseline_avg = df['ret_5d'].mean()

# === 1. IV Rank × outcome ===
print("--- 1. IV Rank × 5d ---")
df['iv_bkt'] = pd.cut(
    df['iv_rank'].fillna(0.5),
    bins=[0, 0.3, 0.5, 0.7, 1.0],
    labels=["저 (<30%)", "중저 (30~50)", "중고 (50~70)", "고 (>70%)"],
)
g = df.groupby('iv_bkt', observed=False).agg(
    n=('ret_5d','size'),
    win=('ret_5d', lambda x: (x>0).mean()),
    avg=('ret_5d','mean'),
)
g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
print(g.to_string())
print()

# === 2. 뉴스 sentiment ===
print("--- 2. 뉴스 sentiment × 5d ---")
df['news_bkt'] = pd.cut(
    df['news_score'],
    bins=[-11, -2, -0.5, 0.5, 2, 11],
    labels=["매우 부정", "부정", "중립", "긍정", "매우 긍정"],
)
g = df.groupby('news_bkt', observed=False).agg(
    n=('ret_5d','size'),
    win=('ret_5d', lambda x: (x>0).mean()),
    avg=('ret_5d','mean'),
)
g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
print(g.to_string())
print()

# === 3. Max Pain 거리 ===
print("--- 3. Max Pain 거리 × 5d ---")
df['mp_bkt'] = pd.cut(
    df['mp_dist_pct'],
    bins=[-100, -5, -2, 2, 5, 100],
    labels=["멀리 아래 <-5%", "-5~-2%", "근접 ±2%", "+2~+5%", "멀리 위 >+5%"],
)
g = df.groupby('mp_bkt', observed=False).agg(
    n=('ret_5d','size'),
    win=('ret_5d', lambda x: (x>0).mean()),
    avg=('ret_5d','mean'),
)
g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
print(g.to_string())
print()

# === 4. Unusual options ===
print("--- 4. Unusual options score × 5d ---")
df['unu_bkt'] = pd.cut(
    df['unusual_score'].fillna(0),
    bins=[-100, -2, -0.5, 0.5, 2, 100],
    labels=["강한 puts", "약한 puts", "중립", "약한 calls", "강한 calls"],
)
g = df.groupby('unu_bkt', observed=False).agg(
    n=('ret_5d','size'),
    win=('ret_5d', lambda x: (x>0).mean()),
    avg=('ret_5d','mean'),
)
g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
print(g.to_string())
print()

# === 5. composite_score (시스템 종합) ===
print("--- 5. composite_score × 5d ---")
df['cs_bkt'] = pd.cut(
    df['composite_score'],
    bins=[-10, -2, -0.5, 0.5, 2, 10],
    labels=["강한 매도", "약한 매도", "중립", "약한 매수", "강한 매수"],
)
g = df.groupby('cs_bkt', observed=False).agg(
    n=('ret_5d','size'),
    win=('ret_5d', lambda x: (x>0).mean()),
    avg=('ret_5d','mean'),
)
g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
print(g.to_string())
print()

# === 6. Multi-factor combo — 진짜 alpha 검색 ===
print("=" * 90)
print(f"--- 6. Multi-factor combo (baseline win {baseline_win:.1%}, avg {baseline_avg:+.2f}%) ---")
print("=" * 90)
combos = [
    # 단일 신호
    ("IV 낮음 (rank<30%)",         df['iv_rank'] < 0.3),
    ("IV 높음 (rank>70%)",         df['iv_rank'] > 0.7),
    ("뉴스 매우 긍정 (>+2)",       df['news_score'] > 2),
    ("뉴스 매우 부정 (<-2)",       df['news_score'] < -2),
    ("뉴스 강력 (|score|>3)",      df['news_score'].abs() > 3),
    ("Max Pain 멀리 아래 (<-5%)",  df['mp_dist_pct'] < -5),
    ("Max Pain 멀리 위 (>+5%)",    df['mp_dist_pct'] > 5),
    ("Unusual calls (>+2)",        df['unusual_score'] > 2),
    ("Unusual puts (<-2)",         df['unusual_score'] < -2),
    ("composite>+1 (매수 신호)",   df['composite_score'] > 1),
    ("composite<-1 (매도 신호)",   df['composite_score'] < -1),

    # 결합
    ("IV<30% + 뉴스 긍정",          (df['iv_rank']<0.3) & (df['news_score']>0.5)),
    ("IV<30% + composite>+1",       (df['iv_rank']<0.3) & (df['composite_score']>1)),
    ("뉴스 긍정 + composite>+1",    (df['news_score']>0.5) & (df['composite_score']>1)),
    ("뉴스 부정 + composite<-1",    (df['news_score']<-0.5) & (df['composite_score']<-1)),
    ("Max Pain<-5% + 뉴스>0 (oversold rebound)",
                                    (df['mp_dist_pct']<-5) & (df['news_score']>0)),
    ("Max Pain>+5% + 뉴스<0 (overhyped pullback)",
                                    (df['mp_dist_pct']>5) & (df['news_score']<0)),
    ("Unusual calls + 뉴스 긍정",   (df['unusual_score']>2) & (df['news_score']>0.5)),
    ("Unusual puts + 뉴스 부정",    (df['unusual_score']<-2) & (df['news_score']<-0.5)),
    ("IV>70% + 뉴스 매우 부정",     (df['iv_rank']>0.7) & (df['news_score']<-2)),
    ("composite>+2 + IV<50%",       (df['composite_score']>2) & (df['iv_rank']<0.5)),
    ("composite<-2 + IV>70%",       (df['composite_score']<-2) & (df['iv_rank']>0.7)),

    # 3-factor
    ("IV<30% + 뉴스>+0.5 + composite>+1",
                                    (df['iv_rank']<0.3) & (df['news_score']>0.5) & (df['composite_score']>1)),
    ("뉴스>+2 + composite>+1 + Max Pain 근접",
                                    (df['news_score']>2) & (df['composite_score']>1) & (df['mp_dist_pct'].abs()<2)),
    ("Max Pain<-5% + composite>0 + 뉴스>0 (deep oversold rebound)",
                                    (df['mp_dist_pct']<-5) & (df['composite_score']>0) & (df['news_score']>0)),
]
print(f"  {'combo':<60s} {'n':<5s} {'win':<8s} {'avg':<10s} {'edge':<22s}")
print("  " + "-"*110)
sweet_combos = []
for name, mask in combos:
    try:
        sub = df[mask.fillna(False)]
    except Exception:
        continue
    n = len(sub)
    if n < 5:
        continue
    win = (sub['ret_5d']>0).mean()
    avg = sub['ret_5d'].mean()
    d_win = win - baseline_win
    d_avg = avg - baseline_avg
    marker = "⭐" if (d_win > 0.05 and d_avg > 0.5) or (d_win < -0.10 and d_avg < -2.0) else "  "
    edge = f"({d_win:+.1%}, {d_avg:+.2f}%)"
    print(f"  {marker}{name:<58s} {n:<5d} {win:.1%}   {avg:+.2f}%   {edge}")
    if d_win > 0.05 and d_avg > 0.5:
        sweet_combos.append({"name": name, "n": n, "win": win, "avg": avg,
                             "edge_win": d_win, "edge_avg": d_avg})

print()
if sweet_combos:
    print(f"=== ⭐ Baseline 초과 sweet combo ({len(sweet_combos)}개): ===")
    for c in sorted(sweet_combos, key=lambda x: -x['edge_avg']):
        print(f"  {c['name']:<60s} n={c['n']}, win={c['win']:.1%}, avg={c['avg']:+.2f}%")
else:
    print("=== 어떤 combo도 baseline 초과 못함 (강세장 baseline 71%/+4.68% 너무 강함) ===")

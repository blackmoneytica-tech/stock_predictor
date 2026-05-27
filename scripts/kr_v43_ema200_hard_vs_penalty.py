"""KR v43 — EMA200 hard filter vs penalty 정밀 비교.

사용자 통찰: hard filter는 약세장에서 EMA200 위 종목이 부족하면
    top-7을 못 채우거나 과집중 → penalty(점수만 깎음)가 종목 수 유지 + 수익 살짝↑?

검증:
    1. 성과: Full + IS/OOS + 2018/2022 약세장 (Total/Sharpe/MDD/Calmar)
    2. 약세장 보유 종목 수: hard가 7개 못 채우는지 (count_log)
    3. 과집중: 종목당 최대 비중 (hard에서 비중 쏠림?)
    4. penalty 강도 민감도 (30/50/100)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from kr_v11_enhanced_modules import load_macro
from kr_v12_integrated_champion import fetch_all_extended
from kr_v34_topk_weight_exit import m
from kr_v41_signal_enhancement import sim_flex_enh


def count_stats(count_log, sd, ed):
    """구간 내 rebal의 평균 종목 수 + 평균 최대비중 + 7개 미달 비율."""
    rows = [(d, z, n, nc, mw) for (d, z, n, nc, mw) in count_log if sd <= d <= ed]
    if not rows: return None
    ns = [r[2] for r in rows]
    mws = [r[4] for r in rows]
    under7 = sum(1 for n in ns if n < 7) / len(ns) * 100
    return {
        'n_rebals': len(rows),
        'avg_holdings': np.mean(ns),
        'min_holdings': min(ns),
        'avg_max_weight': np.mean(mws) * 100,
        'under7_pct': under7,
    }


if __name__ == '__main__':
    import time
    t0 = time.time()
    print('=' * 90)
    print('KR v43 — EMA200 hard filter vs penalty 정밀 비교')
    print('=' * 90)
    print('Loading...')
    data = fetch_all_extended('2014-03-04')
    macro = load_macro('2014-01-01')

    print('Sim (track_counts)...')
    base, base_log = sim_flex_enh(data, macro, top_k=7, track_counts=True)
    hard, hard_log = sim_flex_enh(data, macro, top_k=7, trend_filter='ema200_hard', track_counts=True)
    pen30, pen30_log = sim_flex_enh(data, macro, top_k=7, trend_filter='ema200_pen', trend_penalty=30, track_counts=True)
    pen50, pen50_log = sim_flex_enh(data, macro, top_k=7, trend_filter='ema200_pen', trend_penalty=50, track_counts=True)
    pen100, pen100_log = sim_flex_enh(data, macro, top_k=7, trend_filter='ema200_pen', trend_penalty=100, track_counts=True)

    full = (pd.to_datetime('2015-03-16'), pd.to_datetime('2026-05-26'))
    IS = (pd.to_datetime('2015-03-16'), pd.to_datetime('2020-12-31'))
    OOS = (pd.to_datetime('2021-01-01'), pd.to_datetime('2026-05-26'))
    b18 = (pd.to_datetime('2018-01-01'), pd.to_datetime('2018-12-31'))
    b22 = (pd.to_datetime('2022-01-01'), pd.to_datetime('2022-12-31'))

    def met(s, rng):
        sub = s.loc[rng[0]:rng[1]]
        if len(sub) < 5: return None
        return m(sub / sub.iloc[0])

    # ── 1. 성과 비교 ──
    print('\n=== 1. 성과 (Full / IS / OOS / 약세장) ===')
    sims = [('Baseline', base), ('EMA200 hard', hard),
            ('EMA200 pen30', pen30), ('EMA200 pen50', pen50), ('EMA200 pen100', pen100)]
    print(f'{"전략":<16} {"Full Tot":<10} {"FullSh":<7} {"FullMDD":<9} {"FullCal":<8} {"IS Sh":<7} {"OOS Sh":<7} {"2018":<8} {"2022":<8}')
    print('-' * 90)
    for nm, s in sims:
        rf = met(s, full); ris = met(s, IS); roos = met(s, OOS)
        r18 = met(s, b18); r22 = met(s, b22)
        print(f'{nm:<16} {rf[0]*100:>+8.0f}% {rf[2]:>6.2f} {rf[3]*100:>+7.1f}% {rf[4]:>7.2f} '
              f'{ris[2]:>6.2f} {roos[2]:>6.2f} {r18[0]*100:>+6.1f}% {r22[0]*100:>+6.1f}%')

    # ── 2. 보유 종목 수 (약세장 핵심) ──
    print('\n=== 2. 보유 종목 수 + 과집중 (hard vs penalty) ===')
    print(f'{"전략":<16} {"구간":<14} {"rebal수":<8} {"평균종목":<9} {"최소종목":<9} {"7개미달%":<9} {"평균최대비중"}')
    for nm, log in [('Baseline', base_log), ('EMA200 hard', hard_log),
                     ('EMA200 pen50', pen50_log)]:
        for rng_nm, rng in [('Full', full), ('2018 약세', b18), ('2022 약세', b22)]:
            st = count_stats(log, rng[0], rng[1])
            if st is None: continue
            print(f'{nm:<16} {rng_nm:<14} {st["n_rebals"]:<8} {st["avg_holdings"]:<9.2f} '
                  f'{st["min_holdings"]:<9} {st["under7_pct"]:<8.0f}% {st["avg_max_weight"]:.1f}%')
        print()

    # ── 3. penalty 강도 민감도 ──
    print('=== 3. penalty 강도 민감도 ===')
    print(f'{"penalty":<12} {"Full Tot":<11} {"FullSh":<8} {"FullMDD":<9} {"Calmar":<8} {"2022약세"}')
    for nm, s in [('hard', hard), ('pen30', pen30), ('pen50', pen50), ('pen100', pen100)]:
        rf = met(s, full); r22 = met(s, b22)
        print(f'{nm:<12} {rf[0]*100:>+9.0f}% {rf[2]:>7.2f} {rf[3]*100:>+7.1f}% {rf[4]:>7.2f} {r22[0]*100:>+7.1f}%')

    print(f'\nElapsed: {time.time()-t0:.0f}s')

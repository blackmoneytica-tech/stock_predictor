"""KR v47 — 종목 선정 기준 검증: universe 풀 확장 + 모멘텀 기간.

사용자 질문: 현 7종목 선정 기준이 최선인가? 다른 종목 기회를 놓치나?

검증 (현 라이브 = n50 + mom120 + EMA200 pen30 기준):
    1. universe 풀 확장: top50 → 60/70/85 (거래대금 top N)
    2. 모멘텀 기간: mom120 → mom60/90 (단기 가속 포착)
    3. 조합

목표: 현 기준 대비 수익↑/Sharpe↑/DD↓ robust 개선 있는지. 없으면 현 기준이 최선.
12.2년 + 최근 2년 + 2018/2022 약세장.

⚠️ 데이터 85종목 한정 (n=85가 max). 코스닥/소형주는 미포함 — 별도 데이터 필요.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
from kr_v11_enhanced_modules import load_macro
from kr_v12_integrated_champion import fetch_all_extended
from kr_v34_topk_weight_exit import m
from kr_v41_signal_enhancement import sim_flex_enh


if __name__ == '__main__':
    import time
    t0 = time.time()
    print('=' * 96)
    print('KR v47 — 종목 선정 기준 검증 (universe 풀 + 모멘텀 기간)')
    print('=' * 96)
    print('Loading...')
    data = fetch_all_extended('2014-03-04')
    macro = load_macro('2014-01-01')
    n_stocks = len([c for c in data if c != 'KS200'])
    print(f'  데이터 보유 종목: {n_stocks}')

    # 공통: EMA200 pen30 (현 라이브 기준)
    EMA = dict(trend_filter='ema200_pen', trend_penalty=30)
    variants = [
        ('★ 현 라이브 (n50/mom120)', dict(universe_n=50, mom_lb=120, **EMA)),
        # universe 풀 확장
        ('풀확장 n60', dict(universe_n=60, mom_lb=120, **EMA)),
        ('풀확장 n70', dict(universe_n=70, mom_lb=120, **EMA)),
        ('풀확장 n85 (전체)', dict(universe_n=85, mom_lb=120, **EMA)),
        # 모멘텀 기간 단축
        ('mom60 (단기 가속)', dict(universe_n=50, mom_lb=60, **EMA)),
        ('mom90', dict(universe_n=50, mom_lb=90, **EMA)),
        # 조합
        ('n70 + mom60', dict(universe_n=70, mom_lb=60, **EMA)),
        ('n85 + mom60', dict(universe_n=85, mom_lb=60, **EMA)),
    ]

    sims = {}
    for name, kw in variants:
        print(f'  sim: {name}...')
        sims[name] = sim_flex_enh(data, macro, top_k=7, **kw)

    full = (pd.to_datetime('2015-03-16'), pd.to_datetime('2026-05-26'))
    y2 = (pd.to_datetime('2024-05-27'), pd.to_datetime('2026-05-26'))
    b18 = (pd.to_datetime('2018-01-01'), pd.to_datetime('2018-12-31'))
    b22 = (pd.to_datetime('2022-01-01'), pd.to_datetime('2022-12-31'))

    def met(s, rng):
        sub = s.loc[rng[0]:rng[1]]
        return m(sub / sub.iloc[0]) if len(sub) >= 5 else None
    def ret(s, rng):
        sub = s.loc[rng[0]:rng[1]]
        return (sub.iloc[-1]/sub.iloc[0]-1) if len(sub) >= 5 else None

    print(f'\n{"기준":<28} {"Full Tot":<11} {"FullSh":<7} {"FullMDD":<9} {"Calmar":<8} | {"2Y Tot":<10} {"2022":<8} {"2018"}')
    print('-' * 96)
    base = met(sims['★ 현 라이브 (n50/mom120)'], full)
    for name, _ in variants:
        rf = met(sims[name], full); r2 = ret(sims[name], y2)
        r22 = ret(sims[name], b22); r18 = ret(sims[name], b18)
        mark = ''
        if not name.startswith('★'):
            if rf[0] > base[0] and rf[2] >= base[2] and rf[3] >= base[3]:
                mark = ' ⭐ (전부 개선)'
            elif rf[0] > base[0]*1.05:
                mark = ' (수익↑)'
        print(f'{name:<28} {rf[0]*100:>+9.0f}% {rf[2]:>6.2f} {rf[3]*100:>+7.1f}% {rf[4]:>7.2f} | '
              f'{r2*100:>+8.1f}% {r22*100:>+6.1f}% {r18*100:>+6.1f}%{mark}')

    print(f'\nElapsed: {time.time()-t0:.0f}s')

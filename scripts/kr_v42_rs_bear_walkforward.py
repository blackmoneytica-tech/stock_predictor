"""KR v42 — RS bear-only walk-forward IS/OOS 최종 검증.

RS bear-only (KS200<EMA200일 때만 상대강도 가산)가 robust한지 3축 검증:
    1. 연도별 일관성 — RS bear가 특정 1-2년 우연인지, 여러 해 일관 개선인지
    2. IS/OOS holdout — IS(2015~2020)에서 좋으면 OOS(2021~2026)에서도?
    3. rs_w 민감도 — 0.3/0.5/0.7/1.0 모두 개선이면 파라미터 fit 아님

robust 기준: 약세장 연도 대부분 개선 + OOS 개선 + 파라미터 무관 개선.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from kr_v11_enhanced_modules import load_macro
from kr_v12_integrated_champion import fetch_all_extended
from kr_v30_market_leader_strategy import add_features_v30
from kr_v34_topk_weight_exit import m
from kr_v41_signal_enhancement import sim_flex_enh


if __name__ == '__main__':
    import time
    t0 = time.time()
    print('=' * 92)
    print('KR v42 — RS bear-only walk-forward IS/OOS 최종 검증')
    print('=' * 92)
    print('Loading...')
    data = fetch_all_extended('2014-03-04')
    macro = load_macro('2014-01-01')

    # KS200 regime (연도별 약세장 여부 표시용)
    ksf = add_features_v30(data['KS200']).copy()
    ksf['ema200'] = ksf['close'].ewm(span=200, adjust=False).mean()
    ksf['below'] = ksf['close'] < ksf['ema200']

    print('Sim: baseline + RS bear (w0.3/0.5/0.7/1.0)...')
    base = sim_flex_enh(data, macro, top_k=7)
    rs05 = sim_flex_enh(data, macro, top_k=7, rs_w=0.5, rs_regime='bear')
    rs03 = sim_flex_enh(data, macro, top_k=7, rs_w=0.3, rs_regime='bear')
    rs07 = sim_flex_enh(data, macro, top_k=7, rs_w=0.7, rs_regime='bear')
    rs10 = sim_flex_enh(data, macro, top_k=7, rs_w=1.0, rs_regime='bear')

    def ret_period(s, sd, ed):
        sub = s.loc[sd:ed]
        return (sub.iloc[-1]/sub.iloc[0] - 1) if len(sub) >= 5 else None

    # ============================================================
    # 1. 연도별 일관성
    # ============================================================
    print('\n=== 1. 연도별 Total (baseline vs RS bear w0.5) ===')
    print(f'{"연도":<10} {"KS200 추세":<12} {"Baseline":<11} {"RS bear":<11} {"차이":<10} {"승"}')
    years = list(range(2015, 2027))
    win_count = 0; applicable = 0
    for y in years:
        sd = pd.to_datetime(f'{y}-01-01'); ed = pd.to_datetime(f'{y}-12-31')
        if y == 2015: sd = pd.to_datetime('2015-03-16')
        if y == 2026: ed = pd.to_datetime('2026-05-26')
        rb = ret_period(base, sd, ed); rr = ret_period(rs05, sd, ed)
        if rb is None or rr is None: continue
        # 그 해 KS200 약세장 비중 (RS 작동 구간)
        ksy = ksf['below'].loc[sd:ed]
        bear_pct = ksy.mean() * 100 if len(ksy) > 0 else 0
        trend = f'약세 {bear_pct:.0f}%' if bear_pct > 30 else f'강세 {bear_pct:.0f}%'
        diff = (rr - rb) * 100
        win = '✓' if rr > rb else ('=' if abs(diff) < 0.5 else '✗')
        if rr > rb: win_count += 1
        applicable += 1
        print(f'{y:<10} {trend:<12} {rb*100:>+9.1f}% {rr*100:>+9.1f}% {diff:>+8.1f}%p  {win}')
    print(f'  → RS bear 우위: {win_count}/{applicable}년')

    # ============================================================
    # 2. IS/OOS holdout
    # ============================================================
    print('\n=== 2. IS/OOS holdout ===')
    splits = [
        ('IS (2015-03~2020-12)', '2015-03-16', '2020-12-31'),
        ('OOS (2021-01~2026-05)', '2021-01-01', '2026-05-26'),
    ]
    print(f'{"구간":<26} {"전략":<12} {"Total":<11} {"Sharpe":<8} {"MDD":<9} {"Calmar"}')
    for label, s, e in splits:
        sd, ed = pd.to_datetime(s), pd.to_datetime(e)
        for nm, sim in [('Baseline', base), ('RS bear 0.5', rs05)]:
            sub = sim.loc[sd:ed]
            if len(sub) < 5: continue
            r = m(sub / sub.iloc[0])
            print(f'{label:<26} {nm:<12} {r[0]*100:>+9.1f}% {r[2]:>7.2f} {r[3]*100:>+7.1f}% {r[4]:>7.2f}')
        print()

    # ============================================================
    # 3. rs_w 민감도
    # ============================================================
    print('=== 3. rs_w 민감도 (파라미터 robust 확인) ===')
    print(f'{"rs_w":<10} {"Full Total":<12} {"Full Sh":<9} {"Full MDD":<10} {"2022약세":<10} {"2018약세"}')
    full_s, full_e = pd.to_datetime('2015-03-16'), pd.to_datetime('2026-05-26')
    b22 = (pd.to_datetime('2022-01-01'), pd.to_datetime('2022-12-31'))
    b18 = (pd.to_datetime('2018-01-01'), pd.to_datetime('2018-12-31'))
    for wname, sim in [('baseline', base), ('0.3', rs03), ('0.5', rs05), ('0.7', rs07), ('1.0', rs10)]:
        rf = m(sim.loc[full_s:full_e] / sim.loc[full_s:full_e].iloc[0])
        r22 = ret_period(sim, *b22); r18 = ret_period(sim, *b18)
        print(f'{wname:<10} {rf[0]*100:>+10.0f}% {rf[2]:>8.2f} {rf[3]*100:>+8.1f}% {r22*100:>+8.1f}% {r18*100:>+8.1f}%')

    print(f'\nElapsed: {time.time()-t0:.0f}s')

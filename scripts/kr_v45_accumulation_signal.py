"""KR v45 — "조용한 매집" 신호 검증 (폭등 전 외국인/기관 매집 → 사후 폭발?).

사용자 가설: 기관/외국인이 폭등 전 조용히 매집(가격 횡보 + 수급 유입)하는 신호를 잡으면
            사후 상승을 미리 포착할 수 있나?

데이터: data/investor_flow.csv (Naver 6개월) + 가격
방법:
    1. 각 시점: combo_20d(외국인+기관 20일 누적 순매수), ret_20d(가격 모멘텀), 변동성
    2. "조용한 매집" = combo_20d 순매수 상위 AND ret_20d 횡보(아직 안 폭등)
    3. 사후 +10d/+20d 성과 → 매집 그룹 vs 비매집 비교
    4. 2×2 (수급 부호 × 가격 횡보/급등) 매트릭스

⚠️ 6개월 강세장 표본 — 방향성 힌트. 정량 자동화는 forward 후.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from kr_v12_integrated_champion import fetch_all_extended


if __name__ == '__main__':
    print('=' * 90)
    print('KR v45 — 조용한 매집 신호 검증 (수급 유입 + 가격 횡보 → 사후 폭발?)')
    print('=' * 90)
    print('Loading...')

    flow = pd.read_csv('data/investor_flow.csv', encoding='utf-8-sig', parse_dates=['날짜'])
    flow['code'] = flow['code'].astype(str).str.zfill(6)

    rows = []
    for code, sub in flow.groupby('code'):
        sub = sub.sort_values('날짜').set_index('날짜').copy()
        sub['combo'] = sub['외국인_순매매'] + sub['기관_순매매']
        sub['dv'] = sub['종가'] * sub['거래량']
        # 거래대금 대비 정규화 (종목 규모 무관 비교)
        sub['combo_20d_val'] = sub['combo'].rolling(20).sum() * sub['종가']
        sub['dv_20d'] = sub['dv'].rolling(20).sum()
        sub['combo_20d_pct'] = sub['combo_20d_val'] / sub['dv_20d']
        sub['ret_20d'] = sub['종가'].pct_change(20)
        sub['vol_20d'] = sub['종가'].pct_change().rolling(20).std()
        # 사후
        sub['fwd10'] = sub['종가'].pct_change(10).shift(-10)
        sub['fwd20'] = sub['종가'].pct_change(20).shift(-20)
        for d, r in sub.iterrows():
            if pd.isna(r['combo_20d_pct']) or pd.isna(r['ret_20d']) or pd.isna(r['fwd20']): continue
            rows.append({
                'code': code, 'date': d,
                'combo_20d_pct': r['combo_20d_pct'], 'ret_20d': r['ret_20d'],
                'vol_20d': r['vol_20d'], 'fwd10': r['fwd10'], 'fwd20': r['fwd20'],
            })
    ev = pd.DataFrame(rows)
    print(f'  Total samples: {len(ev):,}')

    # ── 1. combo_20d 순매수 quintile → 사후 (baseline) ──
    print('\n=== 1. 20d 누적 수급 quintile → 사후 (전체) ===')
    ev['q'] = pd.qcut(ev['combo_20d_pct'], 5, labels=False, duplicates='drop')
    print(f'  {"Quintile":<22} {"n":<6} {"+10d":<11} {"+20d":<11} {"음수%(20d)"}')
    for q in sorted(ev['q'].dropna().unique()):
        g = ev[ev['q'] == q]
        lbl = ['Q1 강매도','Q2','Q3 중립','Q4','Q5 강매수'][int(q)]
        print(f'  {lbl:<22} {len(g):<6} {g["fwd10"].mean()*100:>+6.2f}%    {g["fwd20"].mean()*100:>+6.2f}%    {(g["fwd20"]<0).mean()*100:>4.0f}%')

    # ── 2. 조용한 매집 = 수급 순매수 AND 가격 횡보 ──
    print('\n=== 2. "조용한 매집" 2×2 (수급 × 가격 모멘텀) → 사후 +20d ===')
    print(f'  {"조합":<34} {"n":<6} {"+10d":<11} {"+20d":<11} {"음수%"}')
    # 수급: combo_20d_pct 양/음. 가격: ret_20d 횡보(<15%) / 급등(>=15%)
    for lbl, mask in [
        ('💰 매집(수급+) × 횡보(ret<15%)', (ev['combo_20d_pct'] > 0) & (ev['ret_20d'] < 0.15)),
        ('   수급+ × 이미급등(ret>=15%)', (ev['combo_20d_pct'] > 0) & (ev['ret_20d'] >= 0.15)),
        ('   수급- × 횡보', (ev['combo_20d_pct'] <= 0) & (ev['ret_20d'] < 0.15)),
        ('   수급- × 급등', (ev['combo_20d_pct'] <= 0) & (ev['ret_20d'] >= 0.15)),
    ]:
        g = ev[mask]
        if len(g) < 5:
            print(f'  {lbl:<34} {len(g):<6} (부족)'); continue
        print(f'  {lbl:<34} {len(g):<6} {g["fwd10"].mean()*100:>+6.2f}%    {g["fwd20"].mean()*100:>+6.2f}%    {(g["fwd20"]<0).mean()*100:>4.0f}%')

    # ── 3. 강한 매집 (수급 상위 20% + 횡보 + 저변동성) ──
    print('\n=== 3. 강한 매집 신호 정밀 (수급 상위 + 횡보 + 저변동성) → 사후 ===')
    p80 = ev['combo_20d_pct'].quantile(0.80)
    vol_med = ev['vol_20d'].median()
    print(f'  (combo_20d 상위20% 기준={p80*100:.1f}%, 변동성 중앙값={vol_med*100:.2f}%)')
    print(f'  {"신호":<36} {"n":<6} {"+10d":<11} {"+20d":<11} {"음수%"}')
    for lbl, mask in [
        ('전체 baseline', ev['combo_20d_pct'].notna()),
        ('수급 상위20%', ev['combo_20d_pct'] >= p80),
        ('수급 상위20% + 횡보(ret<15%)', (ev['combo_20d_pct'] >= p80) & (ev['ret_20d'] < 0.15)),
        ('수급 상위20% + 횡보 + 저변동성', (ev['combo_20d_pct'] >= p80) & (ev['ret_20d'] < 0.15) & (ev['vol_20d'] < vol_med)),
        ('수급 상위20% + 깊은조정(ret<0)', (ev['combo_20d_pct'] >= p80) & (ev['ret_20d'] < 0)),
    ]:
        g = ev[mask]
        if len(g) < 5:
            print(f'  {lbl:<36} {len(g):<6} (부족)'); continue
        print(f'  {lbl:<36} {len(g):<6} {g["fwd10"].mean()*100:>+6.2f}%    {g["fwd20"].mean()*100:>+6.2f}%    {(g["fwd20"]<0).mean()*100:>4.0f}%')

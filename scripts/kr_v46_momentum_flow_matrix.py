"""KR v46 — 모멘텀 × 수급 매트릭스: "안 산 종목 매수" 통찰 검증.

사용자 통찰: 외국인/기관이 아직 안 산(수급 낮은) 종목이 더 오름 → 매수 연동?
가설 정밀화: 수급-의 의미는 모멘텀에 따라 정반대일 수 있음.
    - 고모멘텀(폭등) + 수급- = 스마트머니 이탈 → 천장 (kr_v44에서 -7% 확인)
    - 저모멘텀(횡보) + 수급- = 아직 관심 못 받음 → 상승 여력 (kr_v45 +8.13% 힌트)

검증: ret_60d(모멘텀) × combo_20d(수급) 3×3 매트릭스 → 사후 +20d.
      → 어느 영역에서 "수급- 매수"가 alpha인지, V25-full(고모멘텀) 통합 가능한지.

⚠️ Naver 6개월 강세장 표본. walk-forward 불가. RS bear 교훈 — 강세장 fit 경계.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from kr_v12_integrated_champion import fetch_all_extended


if __name__ == '__main__':
    print('=' * 88)
    print('KR v46 — 모멘텀 × 수급 매트릭스 ("안 산 종목 매수" 통찰 검증)')
    print('=' * 88)
    print('Loading...')

    flow = pd.read_csv('data/investor_flow.csv', encoding='utf-8-sig', parse_dates=['날짜'])
    flow['code'] = flow['code'].astype(str).str.zfill(6)

    rows = []
    for code, sub in flow.groupby('code'):
        sub = sub.sort_values('날짜').set_index('날짜').copy()
        sub['combo'] = sub['외국인_순매매'] + sub['기관_순매매']
        sub['dv'] = sub['종가'] * sub['거래량']
        sub['combo_20d_pct'] = (sub['combo'].rolling(20).sum() * sub['종가']) / sub['dv'].rolling(20).sum()
        sub['ret_60d'] = sub['종가'].pct_change(60)
        sub['fwd20'] = sub['종가'].pct_change(20).shift(-20)
        for d, r in sub.iterrows():
            if pd.isna(r['combo_20d_pct']) or pd.isna(r['ret_60d']) or pd.isna(r['fwd20']): continue
            rows.append({'code': code, 'date': d, 'mom': r['ret_60d'],
                          'flow': r['combo_20d_pct'], 'fwd20': r['fwd20']})
    ev = pd.DataFrame(rows)
    print(f'  Total samples: {len(ev):,}')

    # 모멘텀 3분위 (저/중/고), 수급 3분위 (매도/중립/매수)
    ev['mom_t'] = pd.qcut(ev['mom'], 3, labels=['저모멘텀', '중모멘텀', '고모멘텀'])
    ev['flow_t'] = pd.qcut(ev['flow'], 3, labels=['수급매도', '수급중립', '수급매수'])

    print('\n=== 3×3 매트릭스: +20d 평균 수익 (괄호=음수%, n) ===')
    print(f'  {"":12}{"수급매도":>16}{"수급중립":>16}{"수급매수":>16}')
    for mt in ['저모멘텀', '중모멘텀', '고모멘텀']:
        line = f'  {mt:<12}'
        for ft in ['수급매도', '수급중립', '수급매수']:
            g = ev[(ev['mom_t'] == mt) & (ev['flow_t'] == ft)]
            if len(g) < 5:
                line += f'{"(부족)":>16}'; continue
            cell = f'{g["fwd20"].mean()*100:+.1f}%({(g["fwd20"]<0).mean()*100:.0f}%,{len(g)})'
            line += f'{cell:>16}'
        print(line)

    # 사용자 통찰 핵심: 저모멘텀 + 수급매도 (안 산 횡보) vs 고모멘텀 + 수급매도 (폭등 이탈)
    print('\n=== 핵심 비교: "수급- 매수" 통찰이 작동하는 영역 ===')
    for lbl, mask in [
        ('저모멘텀 + 수급매도 (안 산 횡보)', (ev['mom_t']=='저모멘텀') & (ev['flow_t']=='수급매도')),
        ('고모멘텀 + 수급매도 (폭등+이탈=천장?)', (ev['mom_t']=='고모멘텀') & (ev['flow_t']=='수급매도')),
        ('고모멘텀 + 수급매수 (V25-full 영역)', (ev['mom_t']=='고모멘텀') & (ev['flow_t']=='수급매수')),
        ('고모멘텀 전체 (현 V25 picks 근사)', ev['mom_t']=='고모멘텀'),
    ]:
        g = ev[mask]
        if len(g) < 5: print(f'  {lbl:<36} (부족)'); continue
        print(f'  {lbl:<36} n={len(g):<5} +20d={g["fwd20"].mean()*100:>+6.2f}% 음수={(g["fwd20"]<0).mean()*100:.0f}%')

    # V25-full 통합 시사: 고모멘텀 내에서 수급 높은 vs 낮은
    print('\n=== V25-full(고모멘텀) 통합 시사: 고모멘텀 내 수급별 ===')
    hi = ev[ev['mom_t']=='고모멘텀']
    for ft in ['수급매도', '수급중립', '수급매수']:
        g = hi[hi['flow_t']==ft]
        if len(g)<5: continue
        print(f'  고모멘텀 × {ft:<8} n={len(g):<5} +20d={g["fwd20"].mean()*100:>+6.2f}% 음수={(g["fwd20"]<0).mean()*100:.0f}%')

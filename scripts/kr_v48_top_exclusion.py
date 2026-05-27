"""KR v48 — V25 picks 중 수급 천장 종목 제외가 나은가? (실전 의사결정 검증)

사용자 질문: LG이노텍·미래에셋처럼 수급 천장(외인·기관 강매도) 종목을
            매수해야 하나, 제외해야 하나?

검증 (Naver 6개월): 고모멘텀(V25 영역) 종목 중 수급 combo_5d로 분리 →
    사후 fwd20 평균 + 음수% + p25(downside) 비교.
    - 천장 제외(수급 양호만) vs 전체 vs 천장만

⚠️ 6개월 강세장 표본. kr_v46에서 수급 매수신호는 cut-noise 결론. 방향성 힌트만.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from kr_v12_integrated_champion import fetch_all_extended


if __name__ == '__main__':
    print('=' * 84)
    print('KR v48 — V25 picks 중 수급 천장 종목 제외 효과 (실전 검증)')
    print('=' * 84)
    print('Loading...')
    flow = pd.read_csv('data/investor_flow.csv', encoding='utf-8-sig', parse_dates=['날짜'])
    flow['code'] = flow['code'].astype(str).str.zfill(6)

    rows = []
    for code, sub in flow.groupby('code'):
        sub = sub.sort_values('날짜').set_index('날짜').copy()
        sub['combo'] = sub['외국인_순매매'] + sub['기관_순매매']
        sub['dv'] = sub['종가'] * sub['거래량']
        sub['combo_5d_pct'] = (sub['combo'].rolling(5).sum()*sub['종가']) / sub['dv'].rolling(5).sum()
        sub['mom60'] = sub['종가'].pct_change(60)
        sub['ret1d'] = sub['종가'].pct_change()
        sub['fwd20'] = sub['종가'].pct_change(20).shift(-20)
        sub['fwd5'] = sub['종가'].pct_change(5).shift(-5)
        for d, r in sub.iterrows():
            if pd.isna(r['mom60']) or pd.isna(r['combo_5d_pct']) or pd.isna(r['fwd20']): continue
            rows.append({'date': d, 'code': code, 'mom60': r['mom60'],
                          'combo': r['combo_5d_pct'], 'ret1d': r['ret1d'],
                          'fwd5': r['fwd5'], 'fwd20': r['fwd20']})
    ev = pd.DataFrame(rows)

    # 고모멘텀 = V25 영역 (mom60 상위 40%)
    hi = ev[ev['mom60'] >= ev['mom60'].quantile(0.6)].copy()
    print(f'고모멘텀(V25 영역) 샘플: {len(hi):,}')

    # 수급 천장 정의: combo_5d 하위 20% (Bottom)
    bottom_thr = hi['combo'].quantile(0.20)
    print(f'수급 천장(Bottom20%) 기준: combo_5d <= {bottom_thr*100:.1f}%')

    def stat(g, lbl):
        f20 = g['fwd20'].dropna(); f5 = g['fwd5'].dropna()
        print(f'  {lbl:<30} n={len(g):<5} +5d={f5.mean()*100:>+5.1f}% '
              f'+20d={f20.mean()*100:>+6.2f}% 음수={(f20<0).mean()*100:>3.0f}% '
              f'p25={f20.quantile(0.25)*100:>+6.1f}%')

    print('\n=== 고모멘텀 종목: 수급 천장 제외 효과 ===')
    stat(hi, '전체 (현 V25 = 천장 포함)')
    stat(hi[hi['combo'] > bottom_thr], '천장 제외 (수급 양호만)')
    stat(hi[hi['combo'] <= bottom_thr], '천장만 (Bottom20%)')

    print('\n=== 추가 세분화: 폭등 동반 여부 (사용자 예시 분석) ===')
    # LG이노텍류 = 고모멘텀 + 최근 폭등 + 천장 / 미래에셋류 = 고모멘텀 + 천장 (폭등X)
    surge = hi['ret1d'] > 0.10
    stat(hi[(hi['combo'] <= bottom_thr) & surge], '천장 + 당일폭등(LG이노텍류)')
    stat(hi[(hi['combo'] <= bottom_thr) & ~surge], '천장 + 폭등X(미래에셋류)')

    print('\n⚠️ 6개월 강세장. kr_v46 = 수급 매수신호 cut-noise. 방향성만.')

"""KR v44 — H-B 유사 폭등 이벤트 + 그 시점 수급 → 사후 성과 (6개월 제한 검증).

가설: 폭등 시 외국인/기관이 함께 매도하면(개인이 떠받친 천장) 사후 약세 → 익절 정당.
      폭등 시 기관/외국인이 매수하면(추세 지속) 사후 강세 → 익절 보류.

데이터: data/investor_flow.csv (Naver 6개월, 50종목 × 120일) + 가격(fetch_all_extended)
방법:
    1. 폭등 이벤트 탐지 (ret_1d > thr AND dv_spike > spike_thr) — 여러 thr
    2. 이벤트 당일 외국인+기관 순매매 부호 (매수/매도) + 5d 누적
    3. 사후 +5d/+10d/+20d 성과
    4. 수급 매수 그룹 vs 매도 그룹 비교 → H-B 수급 차등화 정당성

⚠️ 6개월 강세장 표본, 제한적 — 패턴 힌트만. 정량 자동화는 forward 데이터 성숙 후.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from kr_v12_integrated_champion import fetch_all_extended


if __name__ == '__main__':
    print('=' * 92)
    print('KR v44 — H-B 폭등 이벤트 + 수급 → 사후 성과 (Naver 6개월 제한 검증)')
    print('=' * 92)
    print('Loading...')

    flow = pd.read_csv('data/investor_flow.csv', encoding='utf-8-sig', parse_dates=['날짜'])
    flow['code'] = flow['code'].astype(str).str.zfill(6)
    price_data = fetch_all_extended('2014-03-04')

    # 종목별 이벤트 + 수급 + 사후성과 수집
    events = []
    for code, sub in flow.groupby('code'):
        sub = sub.sort_values('날짜').set_index('날짜').copy()
        sub['ret1d'] = sub['종가'].pct_change()
        sub['dv'] = sub['종가'] * sub['거래량']
        sub['dv5_prev'] = sub['dv'].rolling(5).mean().shift(1)
        sub['dv_spike'] = sub['dv'] / sub['dv5_prev']
        # 수급 (당일 + 5d 누적 외국인+기관 순매매)
        sub['combo'] = sub['외국인_순매매'] + sub['기관_순매매']
        sub['combo_today'] = sub['combo']
        sub['combo_5d'] = sub['combo'].rolling(5).sum()
        # 사후 성과
        sub['fwd5'] = sub['종가'].pct_change(5).shift(-5)
        sub['fwd10'] = sub['종가'].pct_change(10).shift(-10)
        sub['fwd20'] = sub['종가'].pct_change(20).shift(-20)
        for d, row in sub.iterrows():
            if pd.isna(row['ret1d']) or pd.isna(row['dv_spike']): continue
            events.append({
                'code': code, 'date': d,
                'ret1d': row['ret1d'], 'dv_spike': row['dv_spike'],
                'combo_today': row['combo_today'], 'combo_5d': row['combo_5d'],
                'frgn_today': row['외국인_순매매'], 'inst_today': row['기관_순매매'],
                'fwd5': row['fwd5'], 'fwd10': row['fwd10'], 'fwd20': row['fwd20'],
            })
    ev = pd.DataFrame(events)
    print(f'  Total bars: {len(ev):,}')

    # ── 폭등 이벤트 thr별 + 수급 그룹 비교 ──
    for ret_thr, spike_thr in [(0.10, 2.0), (0.15, 2.5), (0.20, 3.0)]:
        spike = ev[(ev['ret1d'] > ret_thr) & (ev['dv_spike'] > spike_thr)].dropna(subset=['fwd5'])
        print(f'\n{"="*70}')
        print(f'=== 폭등 이벤트: ret>{ret_thr*100:.0f}% AND dv_spike>{spike_thr:.1f}x  (n={len(spike)}) ===')
        if len(spike) < 6:
            print('  표본 부족')
            continue

        # 당일 수급 부호로 그룹 분리 (외국인+기관 당일 순매매)
        buy_grp = spike[spike['combo_today'] > 0]
        sell_grp = spike[spike['combo_today'] <= 0]
        print(f'  {"그룹":<28} {"n":<5} {"+5d 평균":<11} {"음수%":<8} {"+10d":<11} {"+20d":<11}')
        for gname, g in [('전체 폭등', spike),
                          ('  └ 외인+기관 당일 매수', buy_grp),
                          ('  └ 외인+기관 당일 매도', sell_grp)]:
            if len(g) < 3:
                print(f'  {gname:<28} {len(g):<5} (부족)'); continue
            f5 = g['fwd5'].dropna(); f10 = g['fwd10'].dropna(); f20 = g['fwd20'].dropna()
            print(f'  {gname:<28} {len(g):<5} {f5.mean()*100:>+6.2f}%    {(f5<0).mean()*100:>4.0f}%   '
                  f'{f10.mean()*100:>+6.2f}%    {f20.mean()*100:>+6.2f}%')

        # 5d 누적 수급 부호로도
        buy5 = spike[spike['combo_5d'] > 0].dropna(subset=['fwd5'])
        sell5 = spike[spike['combo_5d'] <= 0].dropna(subset=['fwd5'])
        print(f'  --- 5d 누적 수급 기준 ---')
        for gname, g in [('  └ 5d 누적 매수', buy5), ('  └ 5d 누적 매도', sell5)]:
            if len(g) < 3:
                print(f'  {gname:<28} {len(g):<5} (부족)'); continue
            f5 = g['fwd5'].dropna()
            print(f'  {gname:<28} {len(g):<5} {f5.mean()*100:>+6.2f}%    {(f5<0).mean()*100:>4.0f}%')

    # ── 결론용: 가장 표본 큰 thr에서 외국인 vs 기관 분리 ──
    print(f'\n{"="*70}')
    print('=== 외국인 vs 기관 분리 (ret>10% AND dv>2x 기준) ===')
    spike = ev[(ev['ret1d'] > 0.10) & (ev['dv_spike'] > 2.0)].dropna(subset=['fwd5'])
    print(f'  {"그룹":<24} {"n":<5} {"+5d 평균":<11} {"음수%":<8} {"+20d"}')
    for gname, mask in [
        ('외국인 매수', spike['frgn_today'] > 0),
        ('외국인 매도', spike['frgn_today'] <= 0),
        ('기관 매수', spike['inst_today'] > 0),
        ('기관 매도', spike['inst_today'] <= 0),
        ('외+기 모두 매수', (spike['frgn_today'] > 0) & (spike['inst_today'] > 0)),
        ('외+기 모두 매도', (spike['frgn_today'] <= 0) & (spike['inst_today'] <= 0)),
    ]:
        g = spike[mask]
        if len(g) < 3:
            print(f'  {gname:<24} {len(g):<5} (부족)'); continue
        f5 = g['fwd5'].dropna(); f20 = g['fwd20'].dropna()
        print(f'  {gname:<24} {len(g):<5} {f5.mean()*100:>+6.2f}%    {(f5<0).mean()*100:>4.0f}%   {f20.mean()*100:>+6.2f}%')

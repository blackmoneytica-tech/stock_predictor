"""KR v49 — "조만간 오를" pre-breakout 신호 캐치 가능한가? (12.2년 검증)

사용자 질문: "지금 강하다"(모멘텀)가 아니라 "조만간 오를 것 같다"를 미리 표시 가능?

후보 신호 (가격 기반 → 12.2년 검증 가능, 수급과 달리 robust):
    1. bb_squeeze (볼린저밴드 수축 = 변동성 압축 → 폭발 직전)
    2. squeeze + 횡보 (저모멘텀, 아직 안 오름) = 조용한 코일링
    3. squeeze + 거래량 수축 (조용한 다지기)
    4. squeeze + breakout (밴드 상단 돌파 시작)

검증: 각 신호 → 사후 fwd20 평균 + 폭발(>20%) 확률 vs baseline.
12.2년 cross-sectional (85종목, survivorship은 KOSPI200이라 대형주 안정).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from kr_v12_integrated_champion import fetch_all_extended
from kr_v30_market_leader_strategy import add_features_v30


if __name__ == '__main__':
    print('=' * 84)
    print('KR v49 — pre-breakout "조만간 오를" 신호 검증 (12.2년)')
    print('=' * 84)
    print('Loading...')
    data = fetch_all_extended('2014-03-04')

    rows = []
    for c in data:
        if c == 'KS200': continue
        df = add_features_v30(data[c])
        if 'bb_squeeze_ratio' not in df.columns: continue
        df = df.copy()
        df['fwd20'] = df['close'].pct_change(20).shift(-20)
        df['fwd10'] = df['close'].pct_change(10).shift(-10)
        for d, r in df.iterrows():
            sq = r.get('bb_squeeze_ratio'); fwd20 = r.get('fwd20')
            if pd.isna(sq) or pd.isna(fwd20): continue
            rows.append({
                'sq': sq, 'breakout': bool(r.get('bb_breakout', False)),
                'ret20': r.get('ret_20d', 0), 'ret60': r.get('ret_60d', 0),
                'vol_accum': r.get('vol_accum_ratio', 1),
                'dist_high': r.get('dist_from_high', 0),
                'fwd10': r.get('fwd10'), 'fwd20': fwd20,
            })
    ev = pd.DataFrame(rows)
    print(f'  샘플: {len(ev):,}')
    base = ev['fwd20'].mean()
    base_surge = (ev['fwd20'] > 0.20).mean()
    print(f'  baseline +20d: {base*100:+.2f}%, 폭발(>20%) 확률 {base_surge*100:.1f}%')

    def stat(g, lbl):
        if len(g) < 30: print(f'  {lbl:<32} n={len(g):<6} (부족)'); return
        f = g['fwd20'].dropna()
        surge = (f > 0.20).mean()
        print(f'  {lbl:<32} n={len(g):<6} +20d={f.mean()*100:>+6.2f}% '
              f'폭발%={surge*100:>4.1f}% 음수={ (f<0).mean()*100:>3.0f}%')

    # 1. squeeze 강도별 (bb_squeeze_ratio 낮을수록 수축)
    print('\n=== 1. 볼린저 수축(squeeze) 강도 → 사후 ===')
    ev['sq_q'] = pd.qcut(ev['sq'], 5, labels=False, duplicates='drop')
    for q in sorted(ev['sq_q'].dropna().unique()):
        lbl = ['Q1 강수축','Q2','Q3','Q4','Q5 확장'][int(q)]
        stat(ev[ev['sq_q']==q], f'squeeze {lbl}')

    # 2. 조용한 코일링: 강수축 + 횡보 + 거래량 수축
    print('\n=== 2. 조용한 코일링 조합 ===')
    sq20 = ev['sq'].quantile(0.20)   # 강수축 기준
    stat(ev[ev['sq'] <= sq20], '강수축(하위20%)')
    stat(ev[(ev['sq']<=sq20) & (ev['ret20'].abs()<0.05)], '강수축 + 횡보(|ret20|<5%)')
    stat(ev[(ev['sq']<=sq20) & (ev['vol_accum']<1.0)], '강수축 + 거래량수축')
    stat(ev[(ev['sq']<=sq20) & ev['breakout']], '강수축 + breakout(밴드돌파)')
    stat(ev[(ev['sq']<=sq20) & (ev['ret20'].abs()<0.05) & ev['breakout']], '강수축 + 횡보 + breakout')

    # 3. breakout 단독
    print('\n=== 3. breakout (밴드 상단 돌파) ===')
    stat(ev[ev['breakout']], 'breakout 전체')
    stat(ev[ev['breakout'] & (ev['ret60']<0.10)], 'breakout + 저모멘텀(ret60<10%)')

    print('\n⚠️ KOSPI200 대형주 12.2년. 폭발=fwd20>20%.')

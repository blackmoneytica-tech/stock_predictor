"""KR v55 — "급등 PANIC" vs "공포 PANIC" 사후 검증.

문제: proxy(EWMA 변동성)는 방향 무관 → 급등도 PANIC(lev 2.0) 분류.
질문: 급등형 PANIC(proxy≥30, DD 하락 아님)에서 lev 2.0이 위험한가 alpha인가?
     → Strict-Panic(DD≤-10%일 때만 lev 2x) 적용 정당성 검증.

검증: PANIC 시점을 lagged 60d DD로 분리 → 사후 KS200 성과.
    - 급등형 PANIC (proxy≥30, DD > -5%)
    - 공포형 PANIC (proxy≥30, DD ≤ -10%)
12.2년 KS200.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from kr_v12_integrated_champion import fetch_all_extended


if __name__ == '__main__':
    print('=' * 78)
    print('KR v55 — 급등 PANIC vs 공포 PANIC 사후 검증')
    print('=' * 78)
    data = fetch_all_extended('2014-03-04')
    ks = data['KS200']['close']

    rets = ks.pct_change()
    proxy = rets.ewm(alpha=0.06).std() * (252**0.5) * 100 * 1.25
    high60 = ks.rolling(60, min_periods=20).max()
    dd60 = (ks / high60 - 1) * 100
    # lag (전일 기준)
    proxy_l = proxy.shift(1); dd_l = dd60.shift(1)
    fwd5 = ks.pct_change(5).shift(-5)
    fwd20 = ks.pct_change(20).shift(-20)

    df = pd.DataFrame({'proxy': proxy_l, 'dd': dd_l, 'fwd5': fwd5, 'fwd20': fwd20}).dropna()

    panic = df[df['proxy'] >= 30]
    print(f'\nPANIC(proxy≥30) 총 일수: {len(panic)} / {len(df)} ({len(panic)/len(df)*100:.0f}%)')

    def stat(g, lbl):
        if len(g) < 20: print(f'  {lbl:<34} n={len(g):<5} (부족)'); return
        print(f'  {lbl:<34} n={len(g):<5} +5d={g["fwd5"].mean()*100:>+6.2f}%(음수{(g["fwd5"]<0).mean()*100:.0f}%) '
              f'+20d={g["fwd20"].mean()*100:>+6.2f}%(음수{(g["fwd20"]<0).mean()*100:.0f}%)')

    print('\n=== PANIC 유형별 사후 KS200 성과 ===')
    stat(df, '전체 (baseline)')
    stat(panic, 'PANIC 전체 (현 lev 2.0)')
    print('  --- DD로 분리 ---')
    stat(panic[panic['dd'] > -5], '급등형 PANIC (DD>-5%, 하락아님)')
    stat(panic[(panic['dd'] <= -5) & (panic['dd'] > -10)], '중간 PANIC (DD -5~-10%)')
    stat(panic[panic['dd'] <= -10], '공포형 PANIC (DD≤-10%, 진짜)')

    print('\n=== 현재 상황 (proxy~72, DD~0%) = 급등형 PANIC ===')
    surge = panic[panic['dd'] > -5]
    print(f'  급등형 PANIC 사후 +20d 음수 비율: {(surge["fwd20"]<0).mean()*100:.0f}%')
    print(f'  공포형 PANIC 사후 +20d 음수 비율: {(panic[panic["dd"]<=-10]["fwd20"]<0).mean()*100:.0f}%')
    print('\n⚠️ 급등형이 사후 나쁘면 → Strict-Panic(DD조건) 적용 = lev 2.0→1.5x 정당.')

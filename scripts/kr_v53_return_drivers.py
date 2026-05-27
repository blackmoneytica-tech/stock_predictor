"""KR v53 — 한국 주식 상승의 driver 분해.

질문: 도대체 한국 주식 상승을 만드는 요소는 무엇인가?
검증 누적 결과: 펀더멘털·수급매수·pre-breakout 다 무효. 그럼 진짜 driver는?

분해:
    1. 시장 베타: 개별 종목 수익 중 시장(KS200)이 설명하는 비율
    2. 글로벌: KS200 자체가 미국(SOX/나스닥)에 얼마나 동조
    3. 섹터: 12.2년 어느 섹터가 상승 주도 (집중도)
    4. 모멘텀 지속성: 추세가 이어지는가 (V25 alpha 근거)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from kr_v12_integrated_champion import fetch_all_extended
from kr_v18_operational_system import SECTOR_MAP


def get_us(sym, start='2014-01-01'):
    import yfinance as yf
    df = yf.download(sym, start=start, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df['Close']


if __name__ == '__main__':
    print('=' * 80)
    print('KR v53 — 한국 주식 상승 driver 분해')
    print('=' * 80)
    print('Loading...')
    data = fetch_all_extended('2014-03-04')
    ks = data['KS200']['close']
    ks_ret = ks.pct_change()

    # ── 1. 시장 베타: 개별 종목 수익을 KS200이 얼마나 설명? ──
    print('\n=== 1. 시장 베타 (개별 종목 수익 중 KS200 설명력 R²) ===')
    r2_list = []
    for c in data:
        if c == 'KS200': continue
        ret = data[c]['close'].pct_change()
        m = pd.DataFrame({'stock': ret, 'mkt': ks_ret}).dropna()
        if len(m) < 250: continue
        corr = m['stock'].corr(m['mkt'])
        r2_list.append(corr**2)
    print(f'  평균 R²(시장 설명력): {np.mean(r2_list)*100:.1f}%  (개별 종목 수익 변동의 절반 가까이가 시장)')
    print(f'  → 즉 개별 알파보다 "시장에 올라타는 것"이 수익의 큰 부분')

    # ── 2. 글로벌: KS200이 미국에 얼마나 동조 ──
    print('\n=== 2. 글로벌 동조 (KS200 vs 미국 전일) ===')
    for nm, sym in [('SOX', '^SOX'), ('NASDAQ', '^IXIC'), ('SP500', '^GSPC')]:
        try:
            us_ret = get_us(sym).pct_change()
            m = pd.DataFrame({'ks': ks_ret, 'us_prev': us_ret.shift(1)}).dropna()
            corr = m['ks'].corr(m['us_prev'])
            print(f'  KS200 ~ {nm} 전일: corr={corr:.3f} (R²={corr**2*100:.1f}%)')
        except Exception as e:
            print(f'  {nm} fail: {e}')

    # ── 3. 섹터 집중: 12.2년 누가 주도 ──
    print('\n=== 3. 섹터별 12.2년 수익 (상승 주도 섹터) ===')
    sec_ret = {}
    for c in data:
        if c == 'KS200': continue
        s = SECTOR_MAP.get(c, 'Other')
        px = data[c]['close']
        tot = px.iloc[-1] / px.iloc[0] - 1
        sec_ret.setdefault(s, []).append(tot)
    sec_avg = {s: np.mean(v) for s, v in sec_ret.items() if len(v) >= 2}
    for s, r in sorted(sec_avg.items(), key=lambda x: -x[1])[:8]:
        print(f'  {s:<10} 평균 {r*100:>+8.0f}% (n={len(sec_ret[s])})')
    ks_tot = ks.iloc[-1]/ks.iloc[0]-1
    print(f'  [KS200 지수: {ks_tot*100:+.0f}%]')

    # ── 4. 모멘텀 지속성: 고모멘텀 → 사후 ──
    print('\n=== 4. 모멘텀 지속성 (추세가 이어지는가 = V25 근거) ===')
    rows = []
    for c in data:
        if c == 'KS200': continue
        px = data[c]['close']
        mom = px.pct_change(120)
        fwd = px.pct_change(20).shift(-20)
        m = pd.DataFrame({'mom': mom, 'fwd': fwd}).dropna()
        rows.append(m)
    allm = pd.concat(rows)
    allm['q'] = pd.qcut(allm['mom'], 5, labels=False, duplicates='drop')
    print(f'  mom120 quintile → 사후 20d:')
    for q in sorted(allm['q'].dropna().unique()):
        g = allm[allm['q']==q]['fwd']
        lbl = ['Q1 최저','Q2','Q3','Q4','Q5 최고모멘텀'][int(q)]
        print(f'    {lbl:<14} 사후 {g.mean()*100:>+5.2f}% (음수 {(g<0).mean()*100:.0f}%)')

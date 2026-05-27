"""KR v50 — 글로벌 시장 연동: 미국/SOX/닛케이가 한국을 예측하나?

가설: 한국은 수출·외국인 의존 → 미국 overnight + SOX(반도체)가 한국 선행.
검증:
    1. 단기(갭): 미국 지수 D-1 수익 → 한국 종목 D 수익 예측력
    2. SOX → 한국 반도체(삼성전자·SK하이닉스) 특화
    3. 중기 선행: SOX 20d 모멘텀 → 한국 반도체 사후 20d

데이터: yfinance(^GSPC/^IXIC/^SOX/^N225) + fdr(한국). 날짜 정렬: 미국 D-1 → 한국 D.
12.2년 가능 → robust 검증.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from kr_v12_integrated_champion import fetch_all_extended


def get_us(sym, start='2014-01-01'):
    import yfinance as yf
    df = yf.download(sym, start=start, progress=False, auto_adjust=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df['Close']


if __name__ == '__main__':
    print('=' * 80)
    print('KR v50 — 글로벌 연동: 미국/SOX → 한국 예측력')
    print('=' * 80)
    print('Loading 한국...')
    data = fetch_all_extended('2014-03-04')
    ks = data['KS200']['close']

    print('Loading 미국/글로벌 (yfinance)...')
    us = {}
    for nm, sym in [('SP500','^GSPC'), ('NASDAQ','^IXIC'), ('SOX','^SOX'), ('Nikkei','^N225')]:
        try:
            us[nm] = get_us(sym)
            print(f'  {nm}: {len(us[nm])} rows')
        except Exception as e:
            print(f'  {nm} fail: {e}')

    # ── 1. 미국 D-1 → 한국(KS200) D 예측력 (상관) ──
    print('\n=== 1. 미국 전일 수익 → 한국 KS200 당일 수익 (상관) ===')
    ks_ret = ks.pct_change()
    for nm, s in us.items():
        us_ret = s.pct_change()
        # 미국 D-1 → 한국 D: 미국을 1일 shift (미국 종가가 한국 다음날 개장 전 정보)
        merged = pd.DataFrame({'us_prev': us_ret.shift(1), 'kr': ks_ret}).dropna()
        if len(merged) < 100: continue
        corr = merged['us_prev'].corr(merged['kr'])
        # 미국 상승일 다음 한국 / 하락일 다음 한국
        up = merged[merged['us_prev'] > 0]['kr'].mean()
        dn = merged[merged['us_prev'] <= 0]['kr'].mean()
        print(f'  {nm:<8} corr={corr:>+.3f} | 미국↑다음날 한국 {up*100:>+.2f}% / 미국↓다음날 {dn*100:>+.2f}% (n={len(merged)})')

    # ── 2. SOX → 한국 반도체 특화 ──
    print('\n=== 2. SOX 전일 → 한국 반도체 당일 (삼성전자·SK하이닉스) ===')
    if 'SOX' in us:
        sox_ret = us['SOX'].pct_change()
        for code, nm in [('005930','삼성전자'), ('000660','SK하이닉스'), ('009150','삼성전기')]:
            if code not in data: continue
            kr_ret = data[code]['close'].pct_change()
            merged = pd.DataFrame({'sox_prev': sox_ret.shift(1), 'kr': kr_ret}).dropna()
            if len(merged) < 100: continue
            corr = merged['sox_prev'].corr(merged['kr'])
            up = merged[merged['sox_prev']>0.01]['kr'].mean()   # SOX +1%↑
            dn = merged[merged['sox_prev']<-0.01]['kr'].mean()  # SOX -1%↓
            print(f'  {nm:<10} corr={corr:>+.3f} | SOX+1%↑→{up*100:>+.2f}% / SOX-1%↓→{dn*100:>+.2f}%')

    # ── 3. 중기 선행: SOX 20d 모멘텀 → 한국 반도체 사후 20d ──
    print('\n=== 3. 중기 선행: SOX 20d 모멘텀 → 한국 반도체 사후 20d ===')
    if 'SOX' in us:
        sox_mom = us['SOX'].pct_change(20)
        for code, nm in [('005930','삼성전자'), ('000660','SK하이닉스')]:
            if code not in data: continue
            kr_fwd = data[code]['close'].pct_change(20).shift(-20)
            merged = pd.DataFrame({'sox_mom': sox_mom, 'kr_fwd': kr_fwd}).dropna()
            if len(merged) < 100: continue
            # SOX 모멘텀 상위/하위 → 한국 사후
            q80 = merged['sox_mom'].quantile(0.8); q20 = merged['sox_mom'].quantile(0.2)
            hi = merged[merged['sox_mom']>=q80]['kr_fwd'].mean()
            lo = merged[merged['sox_mom']<=q20]['kr_fwd'].mean()
            corr = merged['sox_mom'].corr(merged['kr_fwd'])
            print(f'  {nm:<10} corr={corr:>+.3f} | SOX모멘텀 상위20%→한국 사후 {hi*100:>+.2f}% / 하위20%→{lo*100:>+.2f}%')

    print('\n⚠️ 단기 갭(D-1→D)은 진입 타이밍용. 중기 선행이 "조만간 오를" 캐치 핵심.')

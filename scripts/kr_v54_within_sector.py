"""KR v54 — 섹터 내 종목 선택 기준 정교화.

질문: 섹터(반도체 등)가 돌 때, 그 안에서 어떤 종목을 골라야 하나?
한국 상승 = 섹터 사이클(kr_v53) → 섹터 내 leader 선택 기준이 alpha를 더할 수 있나?

기준 후보 (섹터 내 ranking → 섹터 top1 사후 vs 섹터 평균):
    1. mom120 (현 V25)         2. mom60 (단기)
    3. 거래대금 (대장주)          4. SOX 베타 (사이클 민감도)
    5. 저변동성 (안정)           6. RS vs 시장

월말 샘플링, fwd20 사후. 섹터 2+ 종목만.
⚠️ 비대칭 원칙(매수 선택 신호 회의적) 감안 — robust 여부 확인.
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
    print('KR v54 — 섹터 내 종목 선택 기준 검증')
    print('=' * 80)
    print('Loading...')
    data = fetch_all_extended('2014-03-04')
    ks_ret = data['KS200']['close'].pct_change()
    sox_ret = get_us('^SOX').pct_change().shift(1)   # 전일 SOX

    # 종목별 지표 시계열 사전계산
    feats = {}
    for c in data:
        if c == 'KS200': continue
        px = data[c]['close']; vol = data[c]['volume']
        ret = px.pct_change()
        df = pd.DataFrame(index=px.index)
        df['mom120'] = px.pct_change(120)
        df['mom60'] = px.pct_change(60)
        df['dv'] = (px*vol).rolling(20).mean()
        df['vol60'] = ret.rolling(60).std()
        # SOX 베타 (60일): cov(stock, sox)/var(sox)
        merged = pd.DataFrame({'s': ret, 'x': sox_ret}).dropna()
        df['soxbeta'] = merged['s'].rolling(60).cov(merged['x']) / merged['x'].rolling(60).var()
        df['rs'] = px.pct_change(60) - data['KS200']['close'].pct_change(60)
        df['fwd20'] = px.pct_change(20).shift(-20)
        df['sector'] = SECTOR_MAP.get(c, 'Other')
        df['code'] = c
        feats[c] = df

    # 섹터별 종목 (2+)
    sec_codes = {}
    for c in feats:
        s = SECTOR_MAP.get(c, 'Other')
        sec_codes.setdefault(s, []).append(c)
    sec_codes = {s: cs for s, cs in sec_codes.items() if len(cs) >= 3 and s != 'Other'}
    print(f'  대상 섹터(3+종목): {list(sec_codes.keys())}')

    # 월말마다 각 섹터 내 종목 ranking → top1 사후 vs 섹터 평균
    dates = pd.date_range('2015-06-01', '2026-04-30', freq='ME')
    results = {k: [] for k in ['mom120','mom60','dv','soxbeta','low_vol','rs']}
    sector_avg = []
    for d in dates:
        for s, codes in sec_codes.items():
            # 그 시점 유효 종목 데이터
            recs = []
            for c in codes:
                df = feats[c]
                idx = df.index[df.index <= d]
                if len(idx) < 130: continue
                d0 = idx[-1]
                row = df.loc[d0]
                if pd.isna(row['fwd20']) or pd.isna(row['mom120']): continue
                recs.append(row)
            if len(recs) < 3: continue
            rdf = pd.DataFrame(recs).reset_index(drop=True)
            sector_avg.append(rdf['fwd20'].mean())
            # 각 기준 top1의 fwd20
            results['mom120'].append(rdf.loc[rdf['mom120'].idxmax(), 'fwd20'])
            results['mom60'].append(rdf.loc[rdf['mom60'].idxmax(), 'fwd20'])
            results['dv'].append(rdf.loc[rdf['dv'].idxmax(), 'fwd20'])
            if rdf['soxbeta'].notna().any():
                results['soxbeta'].append(rdf.loc[rdf['soxbeta'].idxmax(), 'fwd20'])
            if rdf['vol60'].notna().any():
                results['low_vol'].append(rdf.loc[rdf['vol60'].idxmin(), 'fwd20'])
            if rdf['rs'].notna().any():
                results['rs'].append(rdf.loc[rdf['rs'].idxmax(), 'fwd20'])

    base = np.mean(sector_avg)
    print(f'\n=== 섹터 내 종목 선택 기준 → 사후 fwd20 (섹터 평균 대비) ===')
    print(f'  {"기준":<26} {"top1 사후":<12} {"음수%":<8} {"vs 섹터평균"}')
    print(f'  {"[섹터 평균 (baseline)]":<26} {base*100:>+7.2f}%')
    labels = {'mom120':'mom120 (현 V25)','mom60':'mom60 단기','dv':'거래대금 대장주',
              'soxbeta':'SOX베타 (사이클민감)','low_vol':'저변동성','rs':'RS 시장대비'}
    rank = []
    for k in ['mom120','mom60','dv','soxbeta','low_vol','rs']:
        arr = np.array(results[k])
        if len(arr) < 30: continue
        m = arr.mean(); neg = (arr<0).mean()
        rank.append((labels[k], m, neg, m-base))
    for lbl, m, neg, sp in sorted(rank, key=lambda x:-x[1]):
        print(f'  {lbl:<26} {m*100:>+7.2f}%    {neg*100:>3.0f}%    {sp*100:>+5.2f}%p')

    print('\n⚠️ 매수 선택 신호 — robust 여부는 walk-forward 추가 필요.')

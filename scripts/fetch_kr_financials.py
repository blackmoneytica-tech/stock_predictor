"""KIS 한국 재무 데이터 수집 (KOSPI200 대형주 펀더멘털).

성장성비율(매출/영업이익 증가율) + 재무비율(ROE/부채비율/순이익증가율) 분기별 30기간.
→ data/kr_financials.pkl

⚠️ look-ahead: stac_yymm(분기말) 재무는 공시 시차 있음 → 백테스트 시 +90일 lag 적용.
"""
import sys
import time
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import requests
import pandas as pd
from kis_flow import get_token
import os

from kr_v07_universe_comparison import INDIVIDUAL_STOCKS
from kr_v12_integrated_champion import fetch_all_extended

BASE = 'https://openapi.koreainvestment.com:9443'
OUT = Path(__file__).parent.parent / 'data' / 'kr_financials.pkl'


def fetch_fin(token, code, tr, path):
    ak = os.environ['KIS_APP_KEY']; sk = os.environ['KIS_APP_SECRET']
    h = {'authorization': f'Bearer {token}', 'appkey': ak, 'appsecret': sk,
         'tr_id': tr, 'content-type': 'application/json'}
    p = {'FID_COND_MRKT_DIV_CODE': 'J', 'FID_INPUT_ISCD': code, 'FID_DIV_CLS_CODE': '1'}
    try:
        r = requests.get(f'{BASE}{path}', headers=h, params=p, timeout=15)
        d = r.json()
        if d.get('rt_cd') != '0': return None
        return d.get('output', [])
    except Exception:
        return None


def main():
    token = get_token()
    print('token ok')
    # 데이터 보유 종목 (fetch_all_extended 85종목)
    data = fetch_all_extended('2014-03-04')
    codes = [c for c in data if c != 'KS200']
    print(f'대상 종목: {len(codes)}')

    fin = {}
    t0 = time.time()
    for i, code in enumerate(codes):
        if i % 20 == 0:
            print(f'  [{i}/{len(codes)}] elapsed {time.time()-t0:.0f}s', flush=True)
        growth = fetch_fin(token, code, 'FHKST66430800', '/uapi/domestic-stock/v1/finance/growth-ratio')
        ratio = fetch_fin(token, code, 'FHKST66430300', '/uapi/domestic-stock/v1/finance/financial-ratio')
        time.sleep(0.1)
        rows = {}
        if growth:
            for o in growth:
                ym = o.get('stac_yymm')
                if ym: rows.setdefault(ym, {}).update({
                    'sales_growth': pd.to_numeric(o.get('grs'), errors='coerce'),
                    'op_growth': pd.to_numeric(o.get('bsop_prfi_inrt'), errors='coerce'),
                })
        if ratio:
            for o in ratio:
                ym = o.get('stac_yymm')
                if ym: rows.setdefault(ym, {}).update({
                    'roe': pd.to_numeric(o.get('roe_val'), errors='coerce'),
                    'debt_ratio': pd.to_numeric(o.get('lblt_rate'), errors='coerce'),
                    'ni_growth': pd.to_numeric(o.get('ntin_inrt'), errors='coerce'),
                    'eps': pd.to_numeric(o.get('eps'), errors='coerce'),
                })
        if rows:
            df = pd.DataFrame(rows).T
            df.index = pd.to_datetime(df.index, format='%Y%m', errors='coerce')
            df = df[df.index.notna()].sort_index()
            fin[code] = df

    with open(OUT, 'wb') as f:
        pickle.dump(fin, f)
    print(f'\nSaved: {OUT}')
    print(f'  재무 확보 종목: {len(fin)}')
    if fin:
        sample = list(fin.keys())[0]
        print(f'  예시 {sample}: {len(fin[sample])} 분기, {fin[sample].index[0].date()}~{fin[sample].index[-1].date()}')


if __name__ == '__main__':
    main()

"""코스닥 survivorship-free universe 데이터 수집.

universe = 현재 코스닥 시총 top 300 + 코스닥 상폐(2018+) 전체.
  → 상폐 종목 포함이 survivorship 보정의 핵심.
각 종목 2017-01+ 가격(close, volume) fetch → data/kosdaq_prices.pkl (dict).

상폐 종목은 상폐일까지만 데이터 있음 → 그 시점 이후 universe에서 자동 제외.
"""
import sys
import time
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import FinanceDataReader as fdr
import pandas as pd

OUT = Path(__file__).parent.parent / 'data' / 'kosdaq_prices.pkl'
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    print('Universe 구성...')
    kq = fdr.StockListing('KOSDAQ')
    kq = kq.dropna(subset=['Marcap']).sort_values('Marcap', ascending=False)
    listed_codes = kq['Code'].head(300).tolist()

    dl = fdr.StockListing('KRX-DELISTING')
    dl['DelistingDate'] = pd.to_datetime(dl['DelistingDate'], errors='coerce')
    kq_dl = dl[(dl['Market'] == 'KOSDAQ') & (dl['DelistingDate'] >= '2018-01-01')]
    delisted_codes = kq_dl['Symbol'].dropna().tolist()

    all_codes = list(dict.fromkeys(listed_codes + delisted_codes))   # dedup, 순서 유지
    print(f'  현재 top300: {len(listed_codes)} + 상폐: {len(delisted_codes)} = 통합 {len(all_codes)}')

    prices = {}
    t0 = time.time()
    fail = 0
    for i, code in enumerate(all_codes):
        if i % 50 == 0:
            print(f'  [{i}/{len(all_codes)}] elapsed {time.time()-t0:.0f}s, ok={len(prices)} fail={fail}', flush=True)
        try:
            df = fdr.DataReader(code, '2017-01-01')
            df.columns = [c.lower() for c in df.columns]
            if len(df) > 60 and 'close' in df.columns and 'volume' in df.columns:
                prices[code] = df[['close', 'volume']].copy()
            else:
                fail += 1
            time.sleep(0.05)
        except Exception:
            fail += 1

    with open(OUT, 'wb') as f:
        pickle.dump(prices, f)
    print(f'\nSaved: {OUT}')
    print(f'  수집 종목: {len(prices)} (fail {fail})')
    # 상폐 종목 중 데이터 있는 것
    dl_with_data = sum(1 for c in delisted_codes if c in prices)
    print(f'  상폐 종목 데이터 확보: {dl_with_data}/{len(delisted_codes)}')


if __name__ == '__main__':
    main()

"""KR v51 — 펀더멘털 "사면 안 되는 종목" 필터 검증 (위험 회피 방향).

가설: 매출 역성장·영업이익 적자·고부채 종목은 사후 성과 저조 → hard-reject 정당.
특히 V25 함정: 펀더멘털 나쁜데 모멘텀만 강한 종목 (작전/테마) 제외.

데이터: data/kr_financials.pkl (성장성+재무비율 30분기) + 가격.
look-ahead 방지: 재무(분기말) 공시 시차 +90일 적용 — 그 이후 시점에만 사용.

검증: 펀더멘털 나쁜 종목 사후 fwd60(3개월) vs baseline. + 고모멘텀 교집합(함정).
"""
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from kr_v12_integrated_champion import fetch_all_extended

LAG_DAYS = 90   # 분기말 후 공시 시차


def main():
    print('=' * 80)
    print('KR v51 — 펀더멘털 "사면 안되는 종목" 필터 검증')
    print('=' * 80)
    print('Loading...')
    data = fetch_all_extended('2014-03-04')
    with open('data/kr_financials.pkl', 'rb') as f:
        fin = pickle.load(f)

    rows = []
    for code, df in fin.items():
        if code not in data: continue
        price = data[code]['close']
        # 재무 각 분기 → 공시일(분기말+90일)부터 다음 분기까지 유효
        fin_shifted = df.copy()
        fin_shifted.index = fin_shifted.index + pd.Timedelta(days=LAG_DAYS)
        # 월말 샘플링
        for d in pd.date_range(fin_shifted.index.min(), price.index.max(), freq='ME'):
            avail = fin_shifted[fin_shifted.index <= d]
            if len(avail) == 0: continue
            f = avail.iloc[-1]   # 가장 최근 공시 재무
            # 가격 매칭
            pidx = price.index[price.index <= d]
            if len(pidx) < 120: continue
            d0 = pidx[-1]
            i0 = price.index.get_loc(d0)
            if i0 + 60 >= len(price): continue
            fwd60 = price.iloc[i0+60] / price.iloc[i0] - 1
            mom120 = price.iloc[i0] / price.iloc[i0-120] - 1
            rows.append({
                'code': code, 'date': d0,
                'sales_g': f.get('sales_growth'), 'op_g': f.get('op_growth'),
                'ni_g': f.get('ni_growth'), 'debt': f.get('debt_ratio'), 'roe': f.get('roe'),
                'mom120': mom120, 'fwd60': fwd60,
            })
    ev = pd.DataFrame(rows)
    print(f'  샘플: {len(ev):,} (월말×종목, look-ahead +{LAG_DAYS}d)')
    base = ev['fwd60'].mean()
    print(f'  baseline fwd60(3개월): {base*100:+.2f}%, 음수 {(ev["fwd60"]<0).mean()*100:.0f}%')

    def stat(mask, lbl):
        g = ev[mask].dropna(subset=['fwd60'])
        if len(g) < 30: print(f'  {lbl:<34} n={len(g):<6} (부족)'); return
        f = g['fwd60']
        print(f'  {lbl:<34} n={len(g):<6} fwd60={f.mean()*100:>+6.2f}% 음수={(f<0).mean()*100:>3.0f}% vs base {(f.mean()-base)*100:>+5.1f}%p')

    print('\n=== 1. 단일 펀더멘털 악재 → 사후 (사면 안되는?) ===')
    stat(ev['sales_g'] < 0, '매출 역성장 (sales_g<0)')
    stat(ev['op_g'] < 0, '영업이익 역성장 (op_g<0)')
    stat(ev['ni_g'] < 0, '순이익 역성장 (ni_g<0)')
    stat(ev['debt'] > 200, '고부채 (debt>200%)')
    stat(ev['roe'] < 0, 'ROE 음수 (적자)')
    print('  --- 대조: 펀더멘털 우량 ---')
    stat((ev['sales_g'] > 0) & (ev['op_g'] > 0), '매출+영업이익 동반성장')

    print('\n=== 2. V25 함정: 고모멘텀 + 펀더멘털 악재 ===')
    hi_mom = ev['mom120'] >= ev['mom120'].quantile(0.7)
    stat(hi_mom, '고모멘텀 전체 (V25 영역)')
    stat(hi_mom & (ev['op_g'] < 0), '고모멘텀 + 영업이익 역성장 (함정?)')
    stat(hi_mom & (ev['sales_g'] < 0), '고모멘텀 + 매출 역성장')
    stat(hi_mom & (ev['op_g'] > 0), '고모멘텀 + 영업이익 성장 (건강)')

    print('\n=== 3. 복합 악재 (여러 개 겹침) ===')
    bad = ((ev['sales_g']<0).astype(int) + (ev['op_g']<0).astype(int) +
           (ev['ni_g']<0).astype(int) + (ev['debt']>200).astype(int))
    stat(bad >= 2, '악재 2개+ 겹침')
    stat(bad >= 3, '악재 3개+ 겹침')
    stat(bad == 0, '악재 0개 (클린)')

    print('\n⚠️ KOSPI200 대형주 7.5년. fwd60=3개월 사후.')


if __name__ == '__main__':
    main()

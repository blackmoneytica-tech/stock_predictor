"""코스닥 survivorship-free 모멘텀 백테스트.

data/kosdaq_prices.pkl (상장 top300 + 상폐 통합) 사용.
- 분기별 PIT universe: 그 시점 거래대금(60일 평균) top N
- 매 21일 mom_lb top_k equal rotation
- 보유 중 상폐(데이터 끊김) → 청산 손실 반영 (정직성 핵심)
- 1차 +805%(survivorship 有)가 보정 후 얼마인지 확인

variant: 손절 유무, mom 기간, top_k.
"""
import sys
import pickle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

PKL = Path(__file__).parent.parent / 'data' / 'kosdaq_prices.pkl'
DELIST_LOSS = -0.50   # 보유 중 상폐 시 손실 가정 (정리매매 ~ 보수적 -50%)


def load_prices():
    with open(PKL, 'rb') as f:
        return pickle.load(f)


def backtest(prices, top_n=50, top_k=7, mom_lb=60, rebal_days=21,
              stop_loss=None, lb_dv=60, low_vol_pct=None, require_uptrend=False,
              market_gate_ma=None):
    """survivorship-free 모멘텀 rotation.

    stop_loss: None 또는 음수 (예: -0.20) — 보유 종목 진입가 대비 손절.
    low_vol_pct: universe를 60일 변동성 하위 X%로 정제 (품질 프록시, 0~1).
    require_uptrend: 종가 > 120일 이동평균 종목만 (안정 추세).
    market_gate_ma: 코스닥 지수 < MA(N)면 현금 (약세장 방어). 예: 120.
    """
    # 코스닥 지수 게이트 준비
    kq_close = None
    if market_gate_ma is not None:
        import FinanceDataReader as fdr
        kq = fdr.DataReader('KQ11', '2016-06-01'); kq.columns=[c.lower() for c in kq.columns]
        kq_close = kq['close']
        kq_ma = kq_close.rolling(market_gate_ma).mean()
        kq_gate = (kq_close > kq_ma)   # True=리스크온, False=현금
    # 전체 거래일 (합집합)
    all_dates = sorted(set().union(*[set(df.index) for df in prices.values()]))
    all_dates = [d for d in all_dates if d >= pd.Timestamp('2017-06-01')]

    # 종목별 close/dv 시계열 (빠른 lookup 위해 dict)
    closes = {c: df['close'] for c, df in prices.items()}
    dvs = {c: (df['close'] * df['volume']) for c, df in prices.items()}
    last_date = {c: df.index[-1] for c, df in prices.items()}

    val = 1.0; vals = []; out_dates = []
    holdings = {}        # code → weight
    entry_px = {}        # code → 진입가
    last_rebal = None

    for i, d in enumerate(all_dates):
        # 시장 게이트 (전일 기준 — look-ahead 방지)
        gate_on = True
        if kq_close is not None and i > 0:
            prev = all_dates[i-1]
            if prev in kq_gate.index:
                g = kq_gate.loc[prev]
                gate_on = bool(g) if pd.notna(g) else gate_on
        # 게이트 off → 전량 현금 (holdings 청산)
        if not gate_on and holdings:
            holdings = {}; entry_px = {}

        # PnL (보유 종목 일간 수익) + 상폐 처리
        if holdings and i > 0:
            prev = all_dates[i-1]
            pr_list = []
            delisted = []
            for c, w in list(holdings.items()):
                cs = closes[c]
                if d in cs.index and prev in cs.index:
                    r = cs.loc[d] / cs.loc[prev] - 1
                    pr_list.append(w * r)
                elif d > last_date[c]:
                    # 상폐: 보유 중 데이터 끊김 → 손실 반영 후 제거
                    pr_list.append(w * DELIST_LOSS)
                    delisted.append(c)
                # 거래정지(중간 결측)는 0 처리
            if pr_list:
                val *= (1 + sum(pr_list))
            for c in delisted:
                holdings.pop(c, None); entry_px.pop(c, None)

        # 손절 체크 (진입가 대비)
        if stop_loss is not None and holdings:
            for c in list(holdings.keys()):
                cs = closes[c]
                if d in cs.index and c in entry_px and entry_px[c] > 0:
                    if cs.loc[d] / entry_px[c] - 1 <= stop_loss:
                        holdings.pop(c, None); entry_px.pop(c, None)

        vals.append(val); out_dates.append(d)

        # Rebal (게이트 on일 때만 신규 진입)
        if gate_on and (last_rebal is None or (d - last_rebal).days >= rebal_days):
            # PIT universe: 그 시점 거래대금 60일 평균 top_n
            dv_scores = []
            for c, dv in dvs.items():
                if d not in dv.index: continue
                idx = dv.index.get_loc(d)
                if idx < lb_dv: continue
                avg_dv = dv.iloc[idx-lb_dv:idx].mean()
                if avg_dv > 0: dv_scores.append((c, avg_dv))
            dv_scores.sort(key=lambda x: -x[1])
            universe = [c for c, _ in dv_scores[:top_n]]

            # 품질 프록시: 저변동성 필터 (60일 변동성 하위 X%)
            if low_vol_pct is not None:
                vol_list = []
                for c in universe:
                    cs = closes[c]
                    if d not in cs.index: continue
                    idx = cs.index.get_loc(d)
                    if idx < 60: continue
                    v = cs.iloc[idx-60:idx].pct_change().std()
                    if pd.notna(v): vol_list.append((c, v))
                vol_list.sort(key=lambda x: x[1])   # 변동성 낮은 순
                keep = int(len(vol_list) * low_vol_pct)
                universe = [c for c, _ in vol_list[:keep]]

            # 모멘텀 top_k (+ uptrend 필터)
            mom_scores = []
            for c in universe:
                cs = closes[c]
                if d not in cs.index: continue
                idx = cs.index.get_loc(d)
                if idx < mom_lb: continue
                if require_uptrend and idx >= 120:
                    ma120 = cs.iloc[idx-120:idx].mean()
                    if cs.iloc[idx] < ma120: continue
                m = cs.iloc[idx] / cs.iloc[idx-mom_lb] - 1
                mom_scores.append((c, m))
            mom_scores.sort(key=lambda x: -x[1])
            new_codes = [c for c, _ in mom_scores[:top_k]]

            if new_codes:
                w = 1.0 / len(new_codes)
                holdings = {c: w for c in new_codes}
                entry_px = {c: closes[c].loc[d] for c in new_codes if d in closes[c].index}
            last_rebal = d

    return pd.Series(vals, index=out_dates)


def metrics(s):
    yrs = (s.index[-1]-s.index[0]).days/365.25
    tot = s.iloc[-1]-1
    cagr = s.iloc[-1]**(1/yrs)-1 if yrs>0 else 0
    r = s.pct_change().dropna()
    sh = r.mean()/r.std()*(252**0.5) if r.std()>0 else 0
    dd = (s/s.cummax()-1).min()
    return tot, cagr, sh, dd


if __name__ == '__main__':
    import time
    t0 = time.time()
    print('='*84)
    print('코스닥 survivorship-free 모멘텀 백테스트 (상폐 종목 포함)')
    print('='*84)
    prices = load_prices()
    print(f'종목 수: {len(prices)} (상폐 포함)')

    import FinanceDataReader as fdr
    kq = fdr.DataReader('KQ11','2017-06-01'); kq.columns=[c.lower() for c in kq.columns]
    kq_norm = kq['close']/kq['close'].iloc[0]
    kt, kc, ksh, kdd = metrics(kq_norm)
    print(f'\n코스닥 지수 (벤치마크): Total={kt*100:+.0f}% CAGR={kc*100:+.1f}% Sh={ksh:.2f} MDD={kdd*100:+.1f}%')

    # 시장 게이트 효과: 저변동성50% mom60 top7 기준으로 게이트 유무 비교
    variants = [
        ('저변동성50% (게이트X)', dict(mom_lb=60, top_k=7, low_vol_pct=0.5)),
        ('+ 게이트 MA120', dict(mom_lb=60, top_k=7, low_vol_pct=0.5, market_gate_ma=120)),
        ('+ 게이트 MA60', dict(mom_lb=60, top_k=7, low_vol_pct=0.5, market_gate_ma=60)),
        ('+ 게이트 MA200', dict(mom_lb=60, top_k=7, low_vol_pct=0.5, market_gate_ma=200)),
    ]
    sims = {}
    print(f'\n{"전략":<26} {"Total":<11} {"CAGR":<9} {"Sharpe":<8} {"MDD"}')
    print('-'*84)
    for name, kw in variants:
        s = backtest(prices, top_n=50, **kw)
        sims[name] = s
        t, c, sh, dd = metrics(s)
        print(f'{name:<26} {t*100:>+9.0f}% {c*100:>+7.1f}% {sh:>7.2f} {dd*100:>+7.1f}%')

    # 게이트 MA120의 IS/OOS + 약세장 (robust + 방어 확인)
    best = sims['+ 게이트 MA120']
    def seg(s, sd, ed, lbl):
        sub = s.loc[sd:ed]
        if len(sub)<5: return
        t,c,sh,dd = metrics(sub/sub.iloc[0])
        print(f'  {lbl:<20} Total={t*100:>+7.1f}% Sh={sh:>5.2f} MDD={dd*100:>+6.1f}%')
    print(f'\n=== 게이트 MA120 — IS/OOS + 약세장 방어 확인 ===')
    seg(best,'2017-06-01','2021-12-31','IS(2017-21)')
    seg(best,'2022-01-01','2026-05-31','OOS(2022-26)')
    seg(best,'2018-01-01','2018-12-31','2018 약세')
    seg(best,'2022-01-01','2022-12-31','2022 약세')
    seg(best,'2020-01-01','2020-12-31','2020 회복')

    print(f'\nElapsed: {time.time()-t0:.0f}s')

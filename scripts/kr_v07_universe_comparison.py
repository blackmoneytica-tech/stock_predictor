"""KR v07 — Q1 검증: KODEX 레버리지 only vs 개별종목 vs Sector ETF 비교.

같은 zone framework + DD throttle 하에서 universe만 변경:
    A) KODEX 레버리지 (122630) 단독 + zone leverage
    B) Sector ETF top-3 (현재 v02-v06 baseline)
    C) KOSPI200 개별종목 top-3 (mom60 기준)
    D) KOSPI200 개별종목 top-5
    E) KOSPI200 개별종목 top-7

결론: 어떤 universe가 가장 수익률 + risk-adjusted best인가?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import time

from kr_v02_sector_panic_buy import (
    SECTOR_ETFS, compute_regime,
    zone_lev, KR_BORROW_DAILY, KR_TAX, CASH_DAILY,
)
from kr_v04_sector_pick_alpha import add_features
from kr_v05_multifactor_combo import compute_signals_on_date, rank_scores


# KOSPI200 대형 50종 (avg_dv > 100억)
INDIVIDUAL_STOCKS = [
    '005930', '000660', '005380', '012450', '000270', '068270', '035420',
    '005490', '051910', '028260', '006400', '035720', '105560', '055550',
    '086790', '096770', '017670', '033780', '015760', '009540', '010130',
    '024110', '011200', '042660', '047810', '010140', '011170', '010950',
    '030200', '051900', '032830', '086280', '003670', '011070', '009150',
    '012330', '035250', '066570', '071050', '097950', '006800', '128940',
    '017800',
    # 후기 상장 (2017+)
    '207940', '316140', '018260', '329180', '373220', '251270', '259960',
]


def fetch_all(start='2014-03-04'):
    data = {}
    print(f'Loading sector ETFs ({len(SECTOR_ETFS)})...')
    for code in SECTOR_ETFS:
        try:
            df = fdr.DataReader(code, start)
            df.columns = [c.lower() for c in df.columns]
            data[code] = df
        except: pass
        time.sleep(0.05)
    print(f'Loading individual stocks ({len(INDIVIDUAL_STOCKS)})...')
    for code in INDIVIDUAL_STOCKS:
        try:
            df = fdr.DataReader(code, start)
            df.columns = [c.lower() for c in df.columns]
            if len(df) > 60:
                data[code] = df
        except: pass
        time.sleep(0.05)
    # KS200, KODEX 레버리지
    ks = fdr.DataReader('KS200', start)
    ks.columns = [c.lower() for c in ks.columns]
    data['KS200'] = ks
    lev = fdr.DataReader('122630', start)
    lev.columns = [c.lower() for c in lev.columns]
    data['122630'] = lev
    print(f'Loaded: {len(data)} universes')
    return data


def simulate_universe(data, universe_codes, scorer, top_k=3,
                       dd_throttle_thr=-0.50, throttle_factor=0.5,
                       rebal_days=21, label=''):
    """Universe에서 scorer 기준 top-K rebal + zone framework."""
    ks200 = compute_regime(data['KS200'])
    # Features for each ticker
    feat_data = {}
    for c in universe_codes:
        if c in data:
            feat_data[c] = add_features(data[c])
    all_dates = sorted(ks200.index)
    val = 1.0; vals = []; dates_out = []
    holdings = {}; last_rebal = None; peak = 1.0

    for i, d in enumerate(all_dates):
        if i < 252:
            vals.append(1.0); dates_out.append(d); continue
        v = ks200['vkospi_prev'].get(d, None)
        lev = zone_lev(v)

        cur_dd = val / peak - 1
        if lev > 1 and cur_dd < dd_throttle_thr:
            lev = max(1.0, lev * throttle_factor)

        if lev > 0 and holdings:
            pr = sum(w * (feat_data[c]['ret'].get(d, 0) or 0)
                     for c, w in holdings.items() if c in feat_data)
            cost = max(0, lev - 1) * KR_BORROW_DAILY
            net = lev * pr - cost
        else:
            net = CASH_DAILY if lev == 0 else 0
        val *= (1 + net); val = max(val, 0.01)
        peak = max(peak, val)
        vals.append(val); dates_out.append(d)

        if last_rebal is None or (d - last_rebal).days >= rebal_days:
            if lev > 0:
                # Score each ticker
                scored = []
                for c in universe_codes:
                    if c in feat_data and d in feat_data[c].index:
                        row = feat_data[c].loc[d]
                        try:
                            sc = scorer(row)
                            if sc is not None and not pd.isna(sc):
                                scored.append((c, sc))
                        except: pass
                scored.sort(key=lambda x: -x[1])
                top = scored[:top_k]
                new_holdings = {}
                if top:
                    w = 1.0 / len(top)
                    for c, _ in top:
                        new_holdings[c] = w
                changed = sum(abs(new_holdings.get(c, 0) - holdings.get(c, 0))
                              for c in set(list(holdings.keys()) + list(new_holdings.keys())))
                val *= (1 - KR_TAX * changed / 2)
                holdings = new_holdings
                last_rebal = d
            elif lev == 0 and holdings:
                changed = sum(abs(w) for w in holdings.values())
                val *= (1 - KR_TAX * changed / 2)
                holdings = {}; last_rebal = d

    return pd.Series(vals, index=dates_out)


def simulate_single_lev(data, ticker='122630', dd_throttle_thr=-0.50):
    """KODEX 레버리지 단독 + zone framework."""
    ks200 = compute_regime(data['KS200'])
    lev_df = add_features(data[ticker])
    all_dates = sorted(ks200.index)
    val = 1.0; vals = []; dates_out = []
    peak = 1.0; position = 0  # 0 or 1

    for i, d in enumerate(all_dates):
        if i < 252:
            vals.append(1.0); dates_out.append(d); continue
        v = ks200['vkospi_prev'].get(d, None)
        # Zone: 0 cash / 1 KS200 / 1.5 KS200+lev mix / 2 KODEX lev (122630 already 2x)
        if pd.isna(v):
            lev = 1
        elif v < 15: lev = 0
        elif v < 22.5: lev = 1
        elif v < 30: lev = 1.5
        else: lev = 2

        cur_dd = val / peak - 1
        if lev > 1 and cur_dd < dd_throttle_thr:
            lev = max(1.0, lev * 0.5)

        # KODEX 레버리지 (122630)는 이미 2x KS200.
        # lev=0 cash, lev=1 KS200, lev=1.5 75% lev (effective 1.5x), lev=2 100% lev (effective 2x)
        # 우리는 KS200 시세 사용 + lev factor 적용 (전산상)
        ks_ret = ks200['ret'].get(d, 0) if 'ret' in ks200.columns else 0
        if pd.isna(ks_ret): ks_ret = 0
        if lev == 0:
            net = CASH_DAILY
        else:
            cost = max(0, lev - 1) * KR_BORROW_DAILY
            net = lev * ks_ret - cost
        val *= (1 + net); val = max(val, 0.01)
        peak = max(peak, val)
        vals.append(val); dates_out.append(d)

    return pd.Series(vals, index=dates_out)


def report(s, label):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    total = s.iloc[-1] - 1
    cagr = s.iloc[-1] ** (1/yrs) - 1
    rets = s.pct_change().dropna()
    sharpe = rets.mean()/rets.std() * (252**0.5) if rets.std() > 0 else 0
    dd = (s/s.cummax()-1).min()
    print(f'  {label:55s}  total={total*100:>9.1f}%  CAGR={cagr*100:>5.1f}%  Sh={sharpe:.2f}  DD={dd*100:>5.1f}%')
    return {'total':total,'cagr':cagr,'sharpe':sharpe,'dd':dd}


def walk_forward_simple(simulate_fn, data, n=6):
    ks = data['KS200']
    all_dates = sorted(ks.index)
    win_len = len(all_dates) // n
    rows = []
    for k in range(n):
        s_i, e_i = k*win_len, min((k+1)*win_len, len(all_dates))
        win_dates = all_dates[s_i:e_i]
        if len(win_dates) < 252: continue
        sliced = {kk: vv.loc[win_dates[0]:win_dates[-1]] for kk, vv in data.items()}
        s = simulate_fn(sliced)
        bh = ks.loc[win_dates[0]:win_dates[-1]]
        bh_ret = bh['close'].iloc[-1]/bh['close'].iloc[0] - 1
        kr = s.iloc[-1] - 1
        rets = s.pct_change().dropna()
        sharpe = rets.mean()/rets.std() * (252**0.5) if rets.std() > 0 else 0
        dd = (s/s.cummax()-1).min()
        rows.append({
            'win': k+1, 'start': win_dates[0].date(), 'end': win_dates[-1].date(),
            'bh%': round(bh_ret*100,1), 'kr%': round(kr*100,1),
            'alpha_pp': round((kr-bh_ret)*100,1),
            'sharpe': round(sharpe,2), 'dd%': round(dd*100,1),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print('Loading data...')
    data = fetch_all('2014-03-04')

    # BH KS200 baseline
    ks = data['KS200']
    bh = ks['close'] / ks['close'].iloc[0]
    print()
    print('=== BH baseline ===')
    report(bh, 'BH KS200')
    bh_lev = data['122630']
    bh_lev_norm = bh_lev['close'] / bh_lev['close'].iloc[0]
    report(bh_lev_norm, 'BH KODEX 레버리지 (no zone, just hold)')

    # ---- 5 universe variants ----
    print()
    print('=== Universe comparison (zone + DD@-50% + mom60 scorer + monthly) ===')

    # A) KODEX 레버리지 + zone (no picking, no rotation)
    s_A = simulate_single_lev(data, '122630')
    report(s_A, 'A) KODEX 레버리지 + zone (단순)')

    # B) Sector ETF top-3 (baseline v06)
    sector_codes = [c for c in SECTOR_ETFS if c in data]
    scorer_m60 = lambda r: r.get('ret_60d', 0) or 0
    s_B = simulate_universe(data, sector_codes, scorer_m60, top_k=3)
    report(s_B, 'B) Sector ETF top-3 mom60')

    # C) Individual top-3 mom60
    indiv_codes = [c for c in INDIVIDUAL_STOCKS if c in data]
    s_C = simulate_universe(data, indiv_codes, scorer_m60, top_k=3)
    report(s_C, 'C) Individual top-3 mom60')

    # D) Individual top-5
    s_D = simulate_universe(data, indiv_codes, scorer_m60, top_k=5)
    report(s_D, 'D) Individual top-5 mom60')

    # E) Individual top-7
    s_E = simulate_universe(data, indiv_codes, scorer_m60, top_k=7)
    report(s_E, 'E) Individual top-7 mom60')

    # F) Hybrid: sector + individual mixed
    hybrid_codes = sector_codes + indiv_codes
    s_F = simulate_universe(data, hybrid_codes, scorer_m60, top_k=5)
    report(s_F, 'F) Hybrid (sector+indiv) top-5 mom60')

    # G) Champion (mom120+rs_calm) on individual
    def champion_scorer(r):
        m120 = r.get('ret_120d', 0) or 0
        rs = (r.get('ret_60d', 0) or 0) - (r.get('ret_5d', 0) or 0)
        return m120 + rs  # rank-equivalent at scale
    s_G = simulate_universe(data, indiv_codes, champion_scorer, top_k=3)
    report(s_G, 'G) Individual top-3 Champion (mom120+rs_calm)')

    # ---- Walk-forward for each ----
    print()
    print('=== Walk-forward 6 windows for each ===')
    for label, sim_fn in [
        ('A) KODEX lev only', lambda d: simulate_single_lev(d, '122630')),
        ('B) Sector top-3', lambda d: simulate_universe(d, [c for c in SECTOR_ETFS if c in d], scorer_m60, 3)),
        ('C) Indiv top-3', lambda d: simulate_universe(d, [c for c in INDIVIDUAL_STOCKS if c in d], scorer_m60, 3)),
        ('D) Indiv top-5', lambda d: simulate_universe(d, [c for c in INDIVIDUAL_STOCKS if c in d], scorer_m60, 5)),
    ]:
        wf = walk_forward_simple(sim_fn, data, n=6)
        n_pos = (wf['alpha_pp'] > 0).sum()
        print(f'\n{label}: alpha {n_pos}/{len(wf)}, mean={wf["alpha_pp"].mean():.1f}pp')
        print(wf.to_string(index=False))

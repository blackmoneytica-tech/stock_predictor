"""KR v05 — v04 best 시그널들을 multi-factor combo로 결합 + DD throttle.

v04 best 단일 시그널:
    1. mom120: +2,879% / Sharpe 0.91 (BEST single)
    2. ema20>ema50 golden: +2,707% / Sharpe 0.90
    3. mom60 - mom5: +2,446% / Sharpe 0.88
    4. closest to 52w high: +2,201% / Sharpe 0.89
    5. vol_momentum: +2,115% / Sharpe 0.88

목표: 4-5개 결합으로 alpha 추가 + DD throttle 적용으로 risk-adjusted 더 끌어올리기.

방법: rank-based combo (각 시그널 rank → 합산). 회귀나 weight 찾기는 overfit 위험.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from kr_v02_sector_panic_buy import (
    SECTOR_ETFS, fetch_all, compute_regime,
    zone_lev, KR_BORROW_DAILY, KR_TAX, CASH_DAILY,
)
from kr_v04_sector_pick_alpha import add_features


def compute_signals_on_date(etf_data, ks200, d):
    """모든 ETF에 대해 d 시점 signal dict 반환."""
    ks5 = ks200['close'].pct_change(5).get(d, 0) or 0
    out = {}
    for code, df in etf_data.items():
        if d not in df.index:
            continue
        row = df.loc[d]
        out[code] = {
            'mom120': row.get('ret_120d', 0) or 0,
            'mom60': row.get('ret_60d', 0) or 0,
            'mom20': row.get('ret_20d', 0) or 0,
            'mom5': row.get('ret_5d', 0) or 0,
            'golden': 1 if row.get('ema20_over_ema50') else 0,
            'above_50': 1 if row.get('above_50') else 0,
            'above_200': 1 if row.get('above_200') else 0,
            'dist_52w_high': row.get('dist_52w_high', -1) or -1,
            'vol_momentum': (row.get('vol_momentum', 1.0) or 1.0),
            'rs_calm': (row.get('ret_60d', 0) or 0) - (row.get('ret_5d', 0) or 0),
        }
    return out


def rank_scores(sig_dict, factors, weights):
    """sig_dict {code: {f: val}} → weighted-rank score per code."""
    codes = list(sig_dict.keys())
    if not codes:
        return {}
    # Build score matrix
    scores = {c: 0 for c in codes}
    for f, w in zip(factors, weights):
        vals = [(c, sig_dict[c][f]) for c in codes]
        vals.sort(key=lambda x: x[1])
        # rank 0..n-1
        for rk, (c, v) in enumerate(vals):
            scores[c] += w * rk / max(1, len(codes) - 1)
    return scores


def simulate_multifactor(data, factors, weights, top_k=3,
                          dd_throttle_thr=-0.50, throttle_factor=0.5):
    """Multi-factor backtest with optional DD throttle."""
    ks200 = compute_regime(data['KS200'])
    etf_data = {c: add_features(data[c]) for c in SECTOR_ETFS if c in data}
    all_dates = sorted(ks200.index)
    val = 1.0; vals = []; dates_out = []
    holdings = {}; last_rebal = None; peak = 1.0

    for i, d in enumerate(all_dates):
        if i < 252:
            vals.append(1.0); dates_out.append(d); continue
        v_proxy = ks200['vkospi_prev'].get(d, None)
        lev = zone_lev(v_proxy)

        # DD throttle
        cur_dd = val / peak - 1
        if lev > 1 and cur_dd < dd_throttle_thr:
            lev = max(1.0, lev * throttle_factor)

        # PnL
        if lev > 0 and holdings:
            pr = sum(w * (etf_data[c]['ret'].get(d, 0) or 0)
                     for c, w in holdings.items() if c in etf_data)
            cost = max(0, lev - 1) * KR_BORROW_DAILY
            net = lev * pr - cost
        else:
            net = CASH_DAILY if lev == 0 else 0
        val *= (1 + net); val = max(val, 0.01)
        peak = max(peak, val)
        vals.append(val); dates_out.append(d)

        # Monthly rebal
        rebal_now = (last_rebal is None) or (d.month != last_rebal.month)
        if rebal_now and lev > 0:
            sigs = compute_signals_on_date(etf_data, ks200, d)
            scores = rank_scores(sigs, factors, weights)
            top = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
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
        elif rebal_now and lev == 0 and holdings:
            changed = sum(abs(w) for w in holdings.values())
            val *= (1 - KR_TAX * changed / 2)
            holdings = {}; last_rebal = d

    return pd.Series(vals, index=dates_out)


def report(s, label):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    total = s.iloc[-1] - 1
    cagr = s.iloc[-1] ** (1/yrs) - 1
    rets = s.pct_change().dropna()
    sharpe = rets.mean()/rets.std() * (252**0.5) if rets.std() > 0 else 0
    dd = (s/s.cummax()-1).min()
    print(f'  {label:55s}  total={total*100:>8.1f}%  CAGR={cagr*100:>5.1f}%  Sharpe={sharpe:.2f}  DD={dd*100:.1f}%')
    return {'total':total,'cagr':cagr,'sharpe':sharpe,'dd':dd}


if __name__ == '__main__':
    print('Loading data...')
    data = fetch_all('2014-03-04')

    print('\n=== Single factors (v04 baseline 재현, throttle OFF) ===')
    for f in ['mom120', 'mom60', 'golden', 'dist_52w_high', 'vol_momentum', 'rs_calm']:
        s = simulate_multifactor(data, [f], [1.0], top_k=3, dd_throttle_thr=-999)
        report(s, f'single {f}')

    print('\n=== 2-factor equal-weight combos (throttle OFF) ===')
    pairs = [
        ('mom120', 'golden'), ('mom120', 'dist_52w_high'), ('mom120', 'vol_momentum'),
        ('mom120', 'rs_calm'), ('mom60', 'golden'), ('mom60', 'dist_52w_high'),
        ('golden', 'dist_52w_high'), ('golden', 'vol_momentum'),
        ('dist_52w_high', 'rs_calm'),
    ]
    for f1, f2 in pairs:
        s = simulate_multifactor(data, [f1, f2], [0.5, 0.5], top_k=3, dd_throttle_thr=-999)
        report(s, f'{f1} + {f2}')

    print('\n=== 3-factor combos ===')
    triples = [
        (['mom120', 'golden', 'dist_52w_high'], [1,1,1]),
        (['mom120', 'golden', 'vol_momentum'], [1,1,1]),
        (['mom120', 'dist_52w_high', 'vol_momentum'], [1,1,1]),
        (['mom120', 'rs_calm', 'golden'], [1,1,1]),
        (['mom60', 'mom120', 'golden'], [1,1,1]),
    ]
    for facs, ws in triples:
        s = simulate_multifactor(data, facs, ws, top_k=3, dd_throttle_thr=-999)
        report(s, f'{"+".join(facs)}')

    print('\n=== 4-factor combo + DD throttle scan ===')
    best_facs = ['mom120', 'golden', 'dist_52w_high', 'vol_momentum']
    for thr in [-999, -0.25, -0.35, -0.50]:
        for factor in [0.5, 0.7]:
            s = simulate_multifactor(data, best_facs, [1]*4, top_k=3,
                                       dd_throttle_thr=thr, throttle_factor=factor)
            thr_lbl = 'OFF' if thr == -999 else f'{thr*100:.0f}%'
            report(s, f'4-factor thr={thr_lbl} f={factor}')

    print('\n=== Top-K sensitivity on best 3-factor ===')
    facs = ['mom120', 'golden', 'dist_52w_high']
    for k in [1, 2, 3, 5]:
        s = simulate_multifactor(data, facs, [1]*3, top_k=k, dd_throttle_thr=-0.50)
        report(s, f'mom120+golden+dist52w top-{k}')

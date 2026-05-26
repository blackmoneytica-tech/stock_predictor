"""KR v04 — Sector pick 알고리즘 alpha 파기.

v02에서 sector rotation only (zone=1x)는 +279% < BH +398%로 underperform.
즉 picking 자체가 alpha source가 아님. 한국 시장에서 진짜 alpha 시그널 찾기.

검증 가설 (8종):
    H1. Momentum lookback (5/10/20/60/120/252d) — 어느 horizon이 best?
    H2. Cross-sectional RS norm (KS200 대비, rank-based, z-score)
    H3. Volume momentum (거래대금 급증 = signal?)
    H4. Reversal (oversold bounce) — 한국 mean-reversion 활용
    H5. 52주 신고가 거리 (52w high distance)
    H6. EMA cross (golden cross 시점 진입)
    H7. Pair (top + bottom 동시) — 한국 sector rotation 강함
    H8. Mixed score (multi-factor)

Zone framework는 그대로 유지 (alpha source).
공정 비교: 같은 zone, 같은 rebal, picking만 변경.
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


def add_features(df):
    df = df.copy()
    df['ret'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_10d'] = df['close'].pct_change(10)
    df['ret_20d'] = df['close'].pct_change(20)
    df['ret_60d'] = df['close'].pct_change(60)
    df['ret_120d'] = df['close'].pct_change(120)
    df['ret_252d'] = df['close'].pct_change(252)
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    df['above_50'] = df['close'] > df['ema50']
    df['above_200'] = df['close'] > df['ema200']
    df['ema20_over_ema50'] = df['ema20'] > df['ema50']  # golden cross
    df['dist_52w_high'] = df['close'] / df['close'].rolling(252).max() - 1
    df['dist_52w_low'] = df['close'] / df['close'].rolling(252).min() - 1
    # volume features
    if 'volume' in df.columns:
        df['dollar_vol'] = df['close'] * df['volume']
        df['dv_60d_avg'] = df['dollar_vol'].rolling(60).mean()
        df['dv_5d_avg'] = df['dollar_vol'].rolling(5).mean()
        df['vol_momentum'] = df['dv_5d_avg'] / df['dv_60d_avg']
    return df


def pick_by_scorer(etf_data, ks200, target_date, scorer, top_k=3):
    """Pick sectors by `scorer(row, ks5)`."""
    if target_date not in ks200.index:
        return []
    ks5 = ks200['close'].pct_change(5).get(target_date, 0) or 0
    scored = []
    for code, df in etf_data.items():
        if target_date not in df.index:
            continue
        row = df.loc[target_date]
        try:
            sc = scorer(row, ks5)
        except Exception:
            sc = None
        if sc is None or pd.isna(sc):
            continue
        scored.append((code, sc))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def simulate_with_scorer(data, scorer, top_k=3, label='', reverse=False):
    """Simulate with custom scorer."""
    ks200 = compute_regime(data['KS200'])
    etf_data = {c: add_features(data[c]) for c in SECTOR_ETFS if c in data}
    all_dates = sorted(ks200.index)
    val = 1.0; vals = []; dates_out = []
    holdings = {}; last_rebal = None

    for i, d in enumerate(all_dates):
        if i < 252:  # warm-up for 52w features
            vals.append(1.0); dates_out.append(d); continue
        v_proxy = ks200['vkospi_prev'].get(d, None)
        lev = zone_lev(v_proxy)

        # PnL
        if lev > 0 and holdings:
            port_ret = sum(
                w * (etf_data[c]['ret'].get(d, 0) or 0)
                for c, w in holdings.items() if c in etf_data
            )
            cost = max(0, lev - 1) * KR_BORROW_DAILY
            net = lev * port_ret - cost
        else:
            net = CASH_DAILY if lev == 0 else 0
        val *= (1 + net)
        val = max(val, 0.01)
        vals.append(val); dates_out.append(d)

        # Monthly rebal
        rebal_now = (last_rebal is None) or (d.month != last_rebal.month)
        if rebal_now and lev > 0:
            top = pick_by_scorer(etf_data, ks200, d, scorer, top_k=top_k)
            if reverse:
                # bottom-K (worst) — for reversal hypothesis
                # re-score then take last
                ks5 = ks200['close'].pct_change(5).get(d, 0) or 0
                scored = []
                for code, df in etf_data.items():
                    if d not in df.index:
                        continue
                    row = df.loc[d]
                    try:
                        sc = scorer(row, ks5)
                        if sc is not None and not pd.isna(sc):
                            scored.append((code, sc))
                    except Exception:
                        pass
                scored.sort(key=lambda x: x[1])  # ascending
                top = scored[:top_k]
            new_holdings = {}
            if top:
                w = 1.0 / len(top)
                for code, sc in top:
                    new_holdings[code] = w
            changed = sum(abs(new_holdings.get(c, 0) - holdings.get(c, 0))
                          for c in set(list(holdings.keys()) + list(new_holdings.keys())))
            val *= (1 - KR_TAX * changed / 2)
            holdings = new_holdings
            last_rebal = d
        elif rebal_now and lev == 0 and holdings:
            changed = sum(abs(w) for w in holdings.values())
            val *= (1 - KR_TAX * changed / 2)
            holdings = {}; last_rebal = d

    s = pd.Series(vals, index=dates_out)
    return s


def report(s, label):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    total = s.iloc[-1] - 1
    cagr = s.iloc[-1] ** (1/yrs) - 1
    rets = s.pct_change().dropna()
    sharpe = rets.mean()/rets.std() * (252**0.5) if rets.std() > 0 else 0
    dd = (s/s.cummax()-1).min()
    print(f'  {label:50s}  total={total*100:>8.1f}%  CAGR={cagr*100:>5.1f}%  Sharpe={sharpe:.2f}  DD={dd*100:.1f}%')


if __name__ == '__main__':
    print('Loading data...')
    data = fetch_all('2014-03-04')

    # Baseline
    ks200 = data['KS200']
    print(f'\nBaseline period: {ks200.index[252].date()} ~ {ks200.index[-1].date()}')

    print('\n=== H1. Momentum lookback grid (zone + top3 monthly) ===')
    for lb in [5, 10, 20, 60, 120, 252]:
        s = simulate_with_scorer(data, scorer=lambda r, k5, lb=lb: r.get(f'ret_{lb}d', 0), top_k=3)
        report(s, f'momentum_{lb}d')

    print('\n=== H2. RS norm variants ===')
    s = simulate_with_scorer(data, scorer=lambda r, k5: (r.get('ret_60d', 0) or 0) - 3*k5, top_k=3)
    report(s, 'mom60 - 3*ks5 (relative)')
    s = simulate_with_scorer(data, scorer=lambda r, k5: (r.get('ret_60d', 0) or 0) - (r.get('ret_5d', 0) or 0), top_k=3)
    report(s, 'mom60 - mom5 (recent calm)')

    print('\n=== H3. Volume momentum ===')
    s = simulate_with_scorer(data, scorer=lambda r, k5: r.get('vol_momentum', 1.0) or 1.0, top_k=3)
    report(s, 'volume_momentum (5d/60d DV ratio)')
    s = simulate_with_scorer(data, scorer=lambda r, k5: (r.get('vol_momentum', 1.0) or 1.0) + (r.get('ret_20d', 0) or 0)*10, top_k=3)
    report(s, 'vol_momentum + 20d return')

    print('\n=== H4. Reversal (oversold bounce, top=worst) ===')
    s = simulate_with_scorer(data, scorer=lambda r, k5: r.get('ret_20d', 0) or 0, top_k=3, reverse=True)
    report(s, '20d losers (worst momentum)')
    s = simulate_with_scorer(data, scorer=lambda r, k5: r.get('dist_52w_high', 0) or 0, top_k=3, reverse=True)
    report(s, 'farthest from 52w high')

    print('\n=== H5. 52w high distance ===')
    s = simulate_with_scorer(data, scorer=lambda r, k5: r.get('dist_52w_high', 0) or 0, top_k=3)
    report(s, 'closest to 52w high')

    print('\n=== H6. EMA cross filter ===')
    s = simulate_with_scorer(data, scorer=lambda r, k5: (r.get('ret_60d', 0) or 0) if r.get('above_50') else -999, top_k=3)
    report(s, 'mom60 + above_ema50 filter')
    s = simulate_with_scorer(data, scorer=lambda r, k5: (r.get('ret_60d', 0) or 0) if r.get('ema20_over_ema50') else -999, top_k=3)
    report(s, 'mom60 + ema20>ema50 golden')

    print('\n=== H7. Mixed multi-factor ===')
    def mixed1(r, k5):
        m = r.get('ret_60d', 0) or 0
        a50 = 0.05 if r.get('above_50') else 0
        d52 = r.get('dist_52w_high', 0) or 0
        return m + a50 + 0.3 * d52
    s = simulate_with_scorer(data, scorer=mixed1, top_k=3)
    report(s, 'mom60 + above_50 bonus + dist52w')

    def mixed2(r, k5):
        m20 = r.get('ret_20d', 0) or 0
        m60 = r.get('ret_60d', 0) or 0
        vol = (r.get('vol_momentum', 1.0) or 1.0) - 1.0
        return m20 * 0.4 + m60 * 0.4 + vol * 0.2
    s = simulate_with_scorer(data, scorer=mixed2, top_k=3)
    report(s, 'mom20*.4 + mom60*.4 + vol*.2')

    print('\n=== Top-K sensitivity (best scorer) ===')
    best = lambda r, k5: r.get('ret_60d', 0) or 0
    for k in [1, 2, 3, 5, 7, 11]:
        s = simulate_with_scorer(data, scorer=best, top_k=k)
        report(s, f'mom60 top-{k}')

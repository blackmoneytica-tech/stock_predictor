"""KR v02 — Sector ETF + Panic-Buy zone 통합 백테스트.

설계 (v01 feasibility 결과 반영):
    Universe = liquid Korean sector ETFs (7개) + KODEX 레버리지 (2x 시장)
    Volatility regime = EWMA(20) * 1.25 (synthetic VKOSPI proxy)
    Zone-based leverage (한국 mean-reversion 특성):
        proxy < 15        → CASH (dead zone)
        15 ≤ proxy < 22.5 → standard 1.0x
        22.5 ≤ proxy < 30 → enhanced 1.5x
        proxy ≥ 30        → max panic-buy 2.0x
    Rotation = monthly (거래비용 0.21% drag 최소화)
    Picking = top-K sector ETFs by momentum + RS score

핵심 차이 (vs 미국 G5_22):
    - panic = MAX (미국은 cut)
    - calm = CASH (미국은 lev 1.55x)
    - rotation = monthly (미국은 weekly/daily stateless)
    - leverage 표현 = KODEX 레버리지로 시장 노출만, sector 픽업은 1x
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import FinanceDataReader as fdr
import pandas as pd
import numpy as np

# ---- 파라미터 ----
KR_BORROW_DAILY = 0.075 / 252
KR_TAX = 0.0021  # round-trip
CASH_DAILY = 0.03 / 252

# Liquid sector ETF universe (60일 avg DV > 100억원)
SECTOR_ETFS = {
    '091160': 'KODEX 반도체',
    '139260': 'TIGER 200 IT',
    '305720': 'KODEX 2차전지산업',
    '305540': 'TIGER 2차전지테마',
    '091180': 'KODEX 자동차',
    '091170': 'KODEX 은행',
    '117700': 'KODEX 조선',
    '139220': 'TIGER 200 건설',
    '139230': 'TIGER 200 중공업',
    '364980': 'TIGER KRX2차전지K-뉴딜',
    '102110': 'TIGER 200',         # KOSPI200 broad
}

LEV_2X = '122630'  # KODEX 레버리지 (2x KS200)


def fetch_all(start='2014-03-04'):
    data = {}
    # Sector ETFs
    for code in SECTOR_ETFS:
        try:
            df = fdr.DataReader(code, start)
            df.columns = [c.lower() for c in df.columns]
            data[code] = df
        except Exception as e:
            print(f'  {code}: FAIL {str(e)[:50]}')
    # KS200 for regime
    ks200 = fdr.DataReader('KS200', start)
    ks200.columns = [c.lower() for c in ks200.columns]
    data['KS200'] = ks200
    # 2x lev
    data[LEV_2X] = fdr.DataReader(LEV_2X, start)
    data[LEV_2X].columns = [c.lower() for c in data[LEV_2X].columns]
    return data


def compute_regime(ks200_df):
    """EWMA*1.25 synthetic VKOSPI proxy."""
    df = ks200_df.copy()
    df['ret'] = df['close'].pct_change()
    df['ewma_vol'] = df['ret'].ewm(alpha=0.06).std() * (252 ** 0.5) * 100
    df['vkospi_proxy'] = df['ewma_vol'] * 1.25
    df['vkospi_prev'] = df['vkospi_proxy'].shift(1)
    return df


def compute_etf_features(df):
    """ETF 별 score = momentum + relative strength."""
    df = df.copy()
    df['ret'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_20d'] = df['close'].pct_change(20)
    df['ret_60d'] = df['close'].pct_change(60)
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['above_ema50'] = df['close'] > df['ema50']
    # 60d return is the main momentum signal
    return df


def zone_lev(vkospi_proxy):
    """한국 zone-based leverage."""
    if pd.isna(vkospi_proxy):
        return 1.0
    if vkospi_proxy < 15:
        return 0.0       # CASH dead zone
    if vkospi_proxy < 22.5:
        return 1.0       # standard long
    if vkospi_proxy < 30:
        return 1.5       # enhanced
    return 2.0           # panic max buy


def pick_top_etfs(data, etfs, target_date, top_k=3, ks5_ret=0):
    """Top-K sector ETFs by score on target_date."""
    scored = []
    for code in etfs:
        if code not in data:
            continue
        df = data[code]
        if target_date not in df.index:
            continue
        row = df.loc[target_date]
        score = 0
        # momentum 60d
        m60 = row.get('ret_60d', 0)
        if pd.notna(m60):
            score += m60 * 100
        # ema50 위 +10
        if row.get('above_ema50') == True:
            score += 10
        # RS vs KS200 (5d)
        m5 = row.get('ret_5d', 0)
        if pd.notna(m5):
            score += (m5 - ks5_ret) * 50
        scored.append((code, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def simulate_strategy(data, label, lev_fn=None, monthly_rebal=True,
                       top_k=3, panic_buy=True):
    """
    lev_fn(vkospi_proxy) → leverage. None이면 zone_lev 사용 (panic_buy=True).
                                       panic_buy=False면 미국 G5_22 그대로.
    """
    ks200 = compute_regime(data['KS200'])
    # ETF features
    etf_data = {c: compute_etf_features(data[c]) for c in SECTOR_ETFS if c in data}
    # 공통 dates
    all_dates = sorted(ks200.index)

    val = 1.0
    vals = []
    dates_out = []
    holdings = {}  # {code: weight}
    last_rebal = None
    prev_v = None

    for i, d in enumerate(all_dates):
        if i < 60:  # warm-up
            vals.append(1.0); dates_out.append(d); continue
        # Regime
        v_proxy = ks200['vkospi_prev'].get(d, None)
        if lev_fn is not None:
            lev = lev_fn(v_proxy)
        else:
            lev = zone_lev(v_proxy)

        # Daily PnL on current holdings
        port_ret = 0
        if lev > 0:
            for code, w in holdings.items():
                if code in etf_data and d in etf_data[code].index:
                    r = etf_data[code]['ret'].get(d, 0)
                    if pd.notna(r):
                        port_ret += w * r
            # leverage applied
            cost = max(0, lev - 1) * KR_BORROW_DAILY
            net = lev * port_ret - cost
        else:
            net = CASH_DAILY

        val *= (1 + net)
        val = max(val, 0.01)
        vals.append(val); dates_out.append(d)

        # Monthly rebal: last business day of month or every 21 trading days
        rebal_now = False
        if monthly_rebal:
            if last_rebal is None:
                rebal_now = True
            elif (d.month != last_rebal.month) or ((d - last_rebal).days >= 21):
                rebal_now = True
        else:
            # weekly
            if last_rebal is None or (d - last_rebal).days >= 5:
                rebal_now = True

        if rebal_now and lev > 0:
            ks5 = ks200['close'].pct_change(5).get(d, 0) or 0
            top = pick_top_etfs(etf_data, list(SECTOR_ETFS.keys()), d,
                                 top_k=top_k, ks5_ret=ks5)
            new_holdings = {}
            if len(top) > 0:
                weight = 1.0 / len(top)
                for code, sc in top:
                    new_holdings[code] = weight
            # Tax: 변경된 weight 합계
            changed = 0
            for c in set(list(holdings.keys()) + list(new_holdings.keys())):
                changed += abs(new_holdings.get(c, 0) - holdings.get(c, 0))
            val *= (1 - KR_TAX * changed / 2)  # round-trip 절반 적용
            holdings = new_holdings
            last_rebal = d
        elif rebal_now and lev == 0:
            # 청산
            if holdings:
                changed = sum(abs(w) for w in holdings.values())
                val *= (1 - KR_TAX * changed / 2)
                holdings = {}
                last_rebal = d
        prev_v = v_proxy

    s = pd.Series(vals, index=dates_out)
    return s


def metrics(s, label):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    total = s.iloc[-1] - 1
    cagr = s.iloc[-1] ** (1 / yrs) - 1
    rets = s.pct_change().dropna()
    sharpe = rets.mean() / rets.std() * (252 ** 0.5) if rets.std() > 0 else 0
    dd = (s / s.cummax() - 1).min()
    print(f'{label:50s}  total={total*100:>8.1f}%  CAGR={cagr*100:>5.1f}%  '
          f'Sharpe={sharpe:.2f}  DD={dd*100:.1f}%')
    return {'total': total, 'cagr': cagr, 'sharpe': sharpe, 'dd': dd}


if __name__ == '__main__':
    print('=' * 90)
    print('KR v02 — Sector ETF + Panic-Buy zone backtest')
    print('=' * 90)

    print('Loading data...')
    data = fetch_all('2014-03-04')
    print(f'Universe: {len([c for c in SECTOR_ETFS if c in data])}/{len(SECTOR_ETFS)} ETFs')
    print(f'KS200: {data["KS200"].index[0].date()} ~ {data["KS200"].index[-1].date()}')

    # Baseline: BH KS200
    ks200 = data['KS200']
    bh = ks200['close'] / ks200['close'].iloc[0]
    print()
    print('=== Baselines ===')
    metrics(bh, 'BH KS200')

    # KR v02 main variants
    print()
    print('=== KR v02 variants (monthly rebal, top-3 sector picks) ===')
    s_kr = simulate_strategy(data, 'KR-PanicBuy', top_k=3, monthly_rebal=True)
    metrics(s_kr, 'KR-PanicBuy zone (CASH<15, 1x, 1.5x, 2x>30)')

    # No-panic baseline: always 1.0x leverage (sector rotation only)
    s_flat = simulate_strategy(data, 'Flat1x', lev_fn=lambda v: 1.0, top_k=3)
    metrics(s_flat, 'Sector rotation only (no zone, lev=1.0x)')

    # US-style mistake: dead zone에서 lev, panic에서 cut
    def us_style(v):
        if pd.isna(v): return 1.0
        if v < 18.5: return 1.55
        if v <= 22.5: return 2.0
        return 0.5  # panic cut
    s_us = simulate_strategy(data, 'US-style', lev_fn=us_style, top_k=3)
    metrics(s_us, 'US G5_22-style (calm 1.55x, panic cut)')

    # Top-1 concentration vs Top-5 diversification
    s_kr1 = simulate_strategy(data, 'KR-Top1', top_k=1, monthly_rebal=True)
    metrics(s_kr1, 'KR-PanicBuy top1 concentration')
    s_kr5 = simulate_strategy(data, 'KR-Top5', top_k=5, monthly_rebal=True)
    metrics(s_kr5, 'KR-PanicBuy top5 diversified')

    # Weekly vs monthly rotation
    s_kr_w = simulate_strategy(data, 'KR-Weekly', top_k=3, monthly_rebal=False)
    metrics(s_kr_w, 'KR-PanicBuy weekly rotation (cost drag)')

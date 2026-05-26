"""KR v19 — Accumulation/Reversal alpha source 5종 검증.

가설:
    H1. Volume Accumulation: 5d/60d 거래대금 비율 ↑ + 가격 변동성 낮음
    H2. 52w Low Bounce: 52w low 근처 + 거래량 spike + 단기 return 양수 turn
    H3. OBV Divergence: 30d 가격 하락 but OBV 상승 (bullish divergence)
    H4. Bollinger Squeeze Breakout: BB 폭 압축 후 위로 돌파
    H5. Oversold Rebound: RSI <35 → 50 + mom120 음수에서 양수 turn

V12 framework 그대로 (zone + macro gate + DD throttle + monthly rebal)
Picking 알고리즘만 변경.

Comparison metrics:
    - Total return
    - Sharpe
    - DD
    - WF 6 windows alpha 비율
    - vs V12 mom120 baseline
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from kr_v02_sector_panic_buy import compute_regime, KR_BORROW_DAILY, KR_TAX, CASH_DAILY
from kr_v07_universe_comparison import INDIVIDUAL_STOCKS
from kr_v11_enhanced_modules import (
    SECTOR_MAP, apply_sector_cap, load_macro, macro_gate,
    dd_multistage_lev, pit_universe,
)
from kr_v12_integrated_champion import (
    zone_base_lev, fetch_all_extended,
)


# ============================================================
# Features extension (가설별 시그널)
# ============================================================
def add_features_v19(df):
    """V12 features + 가설별 신규 features."""
    df = df.copy()
    df['ret'] = df['close'].pct_change()
    df['ret_5d'] = df['close'].pct_change(5)
    df['ret_20d'] = df['close'].pct_change(20)
    df['ret_60d'] = df['close'].pct_change(60)
    df['ret_120d'] = df['close'].pct_change(120)

    # H1. Volume Accumulation
    if 'volume' in df.columns:
        df['dv'] = df['close'] * df['volume']
        df['dv_5d_avg'] = df['dv'].rolling(5).mean()
        df['dv_60d_avg'] = df['dv'].rolling(60).mean()
        df['vol_accum_ratio'] = df['dv_5d_avg'] / df['dv_60d_avg']
    # 가격 변동성 (낮을수록 횡보 = 매집)
    df['price_std_20d'] = df['ret'].rolling(20).std()

    # H2. 52w low distance + 거래량 spike
    df['low_52w'] = df['close'].rolling(252).min()
    df['high_52w'] = df['close'].rolling(252).max()
    df['dist_from_low'] = df['close'] / df['low_52w'] - 1
    df['dist_from_high'] = df['close'] / df['high_52w'] - 1
    df['ret_3d'] = df['close'].pct_change(3)

    # H3. OBV
    if 'volume' in df.columns:
        df['obv_diff'] = df['volume'] * np.sign(df['ret'].fillna(0))
        df['obv'] = df['obv_diff'].cumsum()
        df['obv_30d_chg'] = df['obv'].diff(30)
        df['price_30d_chg'] = df['close'].pct_change(30)

    # H4. Bollinger Squeeze
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['bb_std'] = df['close'].rolling(20).std()
    df['bb_width'] = (df['bb_std'] * 2) / df['ema20']
    df['bb_width_avg60'] = df['bb_width'].rolling(60).mean()
    df['bb_squeeze_ratio'] = df['bb_width'] / df['bb_width_avg60']
    df['bb_upper'] = df['ema20'] + 2 * df['bb_std']
    df['bb_breakout'] = df['close'] > df['bb_upper'].shift(1)  # yesterday upper 돌파

    # H5. RSI
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_5d_chg'] = df['rsi'].diff(5)

    # mom turn signal (음→양 전환)
    df['mom60_prev30d'] = df['ret_60d'].shift(30)
    df['mom_turn'] = (df['mom60_prev30d'] < 0) & (df['ret_60d'] > 0)

    return df


# ============================================================
# 가설별 scoring function
# ============================================================
def score_h1_volume_accumulation(row):
    """거래량 비율 (5d/60d) ↑ + 가격 변동성 낮음 + return slightly positive."""
    var = row.get('vol_accum_ratio', 1) or 1
    pstd = row.get('price_std_20d', 0.02) or 0.02
    r20 = row.get('ret_20d', 0) or 0
    # 거래량 +50% 이상 + 변동성 낮음 + 20d return -5%~+10%
    if pd.isna(var) or pd.isna(pstd): return None
    score = (var - 1) * 50  # 거래량 가산점
    score += (0.02 - pstd) * 1000  # 변동성 낮을수록 가점 (max 20pt)
    score += min(r20 * 50, 5) - abs(r20 - 0.025) * 20  # 0~5% 사이 return에 보너스
    return score


def score_h2_52w_low_bounce(row):
    """52w low 근처 + 단기 return 양수 turn + 거래량 spike."""
    dlo = row.get('dist_from_low', 0) or 0
    r3 = row.get('ret_3d', 0) or 0
    var = row.get('vol_accum_ratio', 1) or 1
    if pd.isna(dlo) or pd.isna(r3): return None
    # 52w low + 10% 이내 + 3d return 양수 + 거래량 spike
    if dlo > 0.20: return None  # 너무 멀면 제외
    score = max(0, 0.20 - dlo) * 100  # 가까울수록 가점
    score += r3 * 100  # 단기 반등
    score += (var - 1) * 30
    return score


def score_h3_obv_divergence(row):
    """OBV 상승 + 가격 하락 (bullish divergence)."""
    obv_chg = row.get('obv_30d_chg', 0) or 0
    price_chg = row.get('price_30d_chg', 0) or 0
    if pd.isna(obv_chg) or pd.isna(price_chg): return None
    # OBV 양수 + 가격 음수 = divergence (강한 매집)
    if obv_chg > 0 and price_chg < 0:
        # divergence 강도 = obv_chg 크기 + 가격 음수 강도
        return abs(price_chg) * 100 + obv_chg / 1e9 * 0.1
    return None  # divergence 없으면 제외


def score_h4_bb_squeeze_breakout(row):
    """BB 폭 압축 후 위로 돌파."""
    sq = row.get('bb_squeeze_ratio', 1) or 1
    breakout = row.get('bb_breakout', False)
    r5 = row.get('ret_5d', 0) or 0
    if pd.isna(sq): return None
    # squeeze 0.7 이하 + breakout + 5d return 양수
    if sq > 0.8: return None
    score = (1 - sq) * 100  # squeeze 강할수록 가점
    if breakout: score += 30
    score += r5 * 50
    return score


def score_h5_oversold_rebound(row):
    """RSI <35에서 회복 + mom turn."""
    rsi = row.get('rsi', 50) or 50
    rsi_chg = row.get('rsi_5d_chg', 0) or 0
    mom_turn = row.get('mom_turn', False)
    if pd.isna(rsi): return None
    # RSI 30-50 range + RSI 상승 중 + mom turn = oversold bounce
    if rsi < 30 or rsi > 55: return None
    score = (50 - abs(rsi - 40)) * 2  # 35-45 최고점
    score += rsi_chg * 5  # 회복 강도
    if mom_turn: score += 20  # mom turn 보너스
    return score


# Baseline V12 scorer
def score_v12_mom_ensemble(row):
    """V12 baseline: mom 90/120/150 평균."""
    scores = []
    for lb in (90, 120, 150):
        v = row.get(f'ret_{lb}d', None)
        if v is not None and not pd.isna(v):
            scores.append(v)
    return sum(scores)/len(scores) if scores else None


# Combo scorers
def score_combo_mom_h1(row):
    """mom120 + volume accumulation."""
    m = row.get('ret_120d', 0)
    h1 = score_h1_volume_accumulation(row)
    if pd.isna(m) or h1 is None: return None
    return m * 100 + h1 * 0.3  # mom dominant


def score_combo_mom_h2(row):
    """mom120 + 52w low bounce filter."""
    m = row.get('ret_120d', 0)
    dlo = row.get('dist_from_low', 0) or 0
    # 52w low에 너무 가까운 종목은 mom120과 어울리지 않음 (mom120 양수면 high 근처)
    # 대신 52w low에서 회복 중인 종목 (dlo 10-50%)에 가점
    if pd.isna(m): return None
    score = m * 100
    if 0.10 < dlo < 0.50:
        score += 20  # healthy bounce zone
    return score


def score_combo_mom_h4(row):
    """mom120 + BB squeeze breakout."""
    m = row.get('ret_120d', 0)
    h4 = score_h4_bb_squeeze_breakout(row)
    if pd.isna(m): return None
    score = m * 100
    if h4 is not None: score += h4 * 0.5
    return score


# ============================================================
# Simulator
# ============================================================
def sim_v19(data, macro, scorer, top_k=7, sector_cap=3, rebal_days=21):
    """V12 framework + custom scorer."""
    ks200 = compute_regime(data['KS200'])
    feat_data = {c: add_features_v19(data[c]) for c in data if c != 'KS200'}
    all_dates = sorted(ks200.index)

    val = 1.0; vals = []; dates_out = []
    holdings = {}; last_rebal = None; peak = 1.0; pit_cache = {}

    for i, d in enumerate(all_dates):
        if i < 252:
            vals.append(1.0); dates_out.append(d); continue
        v = ks200['vkospi_prev'].get(d, None)
        base_lev = zone_base_lev(v)
        if d in macro.index:
            g = macro_gate(macro.loc[d], ks_row=None)
            if g == 'crisis': base_lev = 0
            elif g == 'caution': base_lev = min(base_lev, 1.0)
        cur_dd = val/peak - 1
        lev = dd_multistage_lev(base_lev, cur_dd)

        if lev > 0 and holdings:
            pr = sum(w * (feat_data[c]['ret'].get(d, 0) or 0)
                     for c, w in holdings.items() if c in feat_data)
            cost = max(0, lev-1) * KR_BORROW_DAILY
            net = lev * pr - cost
        else:
            net = CASH_DAILY if lev == 0 else 0
        val *= (1 + net); val = max(val, 0.01)
        peak = max(peak, val)
        vals.append(val); dates_out.append(d)

        if last_rebal is None or (d - last_rebal).days >= rebal_days:
            if lev > 0:
                qk = (d.year, (d.month-1)//3)
                if qk not in pit_cache:
                    pit_cache[qk] = pit_universe(data, d, n=50, lookback_days=60)
                universe = pit_cache[qk]
                scored = []
                for c in universe:
                    if c in feat_data and d in feat_data[c].index:
                        row = feat_data[c].loc[d]
                        sc = scorer(row)
                        if sc is not None and not pd.isna(sc):
                            scored.append((c, sc))
                scored.sort(key=lambda x: -x[1])
                scored = apply_sector_cap(scored, max_per_sector=sector_cap)
                top = scored[:top_k]
                new_h = {c: 1.0/len(top) for c, _ in top} if top else {}
                changed = sum(abs(new_h.get(c, 0) - holdings.get(c, 0))
                              for c in set(list(holdings.keys()) + list(new_h.keys())))
                val *= (1 - KR_TAX * changed/2)
                holdings = new_h; last_rebal = d
            elif lev == 0 and holdings:
                changed = sum(abs(w) for w in holdings.values())
                val *= (1 - KR_TAX * changed/2)
                holdings = {}; last_rebal = d

    return pd.Series(vals, index=dates_out)


def m(s):
    if len(s) < 2: return None
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    total = s.iloc[-1] - 1
    cagr = s.iloc[-1] ** (1/yrs) - 1
    r = s.pct_change().dropna()
    sh = r.mean()/r.std() * (252**0.5) if r.std() > 0 else 0
    dd = (s/s.cummax()-1).min()
    return total, cagr, sh, dd


def wf_alpha(s, bh, n=6):
    dates = sorted(s.index)
    win_len = len(dates) // n
    rows = []
    for k in range(n):
        s_i, e_i = k*win_len, min((k+1)*win_len, len(dates))
        wd = dates[s_i:e_i]
        if len(wd) < 100: continue
        a, b = s.loc[wd[0]], s.loc[wd[-1]]
        c, d_ = bh.loc[wd[0]], bh.loc[wd[-1]]
        rows.append({
            'win': k+1,
            'strat%': round((b/a-1)*100, 1),
            'bh%': round((d_/c-1)*100, 1),
            'alpha_pp': round(((b/a)-(d_/c))*100, 1),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print('=' * 90)
    print('KR v19 — Accumulation/Reversal Hypotheses (5종 + Combo)')
    print('=' * 90)
    print('\nLoading...')
    data = fetch_all_extended('2014-03-04')
    macro = load_macro('2014-01-01')
    ks = data['KS200']
    s_bh = ks['close'] / ks['close'].iloc[0]

    hypotheses = [
        ('V12 Baseline (mom_ensemble)', score_v12_mom_ensemble),
        ('H1. Volume Accumulation', score_h1_volume_accumulation),
        ('H2. 52w Low Bounce', score_h2_52w_low_bounce),
        ('H3. OBV Divergence', score_h3_obv_divergence),
        ('H4. BB Squeeze Breakout', score_h4_bb_squeeze_breakout),
        ('H5. Oversold Rebound (RSI)', score_h5_oversold_rebound),
        ('Combo Mom + H1 (vol-confirmed)', score_combo_mom_h1),
        ('Combo Mom + H2 (52w bounce)', score_combo_mom_h2),
        ('Combo Mom + H4 (squeeze)', score_combo_mom_h4),
    ]

    results = []
    for name, scorer in hypotheses:
        print(f'\n=== {name} ===')
        try:
            s = sim_v19(data, macro, scorer, top_k=7, sector_cap=3)
            m_res = m(s)
            if m_res is None: continue
            total, cagr, sh, dd = m_res
            print(f'  total={total*100:>8.1f}%  CAGR={cagr*100:>5.1f}%  Sh={sh:.2f}  DD={dd*100:.1f}%')
            wf = wf_alpha(s, s_bh)
            n_pos = (wf['alpha_pp'] > 0).sum()
            print(f'  WF alpha: {n_pos}/{len(wf)}, mean={wf["alpha_pp"].mean():+.1f}pp')
            results.append({
                'name': name, 'total': total, 'cagr': cagr, 'sharpe': sh, 'dd': dd,
                'wf_alpha': n_pos, 'wf_mean': wf["alpha_pp"].mean(),
            })
        except Exception as e:
            print(f'  ERROR: {str(e)[:100]}')

    print('\n=== Summary ===')
    df = pd.DataFrame(results)
    df['total%'] = (df['total']*100).round(1)
    df['cagr%'] = (df['cagr']*100).round(1)
    df['dd%'] = (df['dd']*100).round(1)
    df = df[['name','total%','cagr%','sharpe','dd%','wf_alpha','wf_mean']]
    print(df.to_string(index=False))

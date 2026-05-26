"""KR v31 — Strict-Panic 룰 정밀 검증.

가설:
    V25 zone framework의 약점: proxy>30 만으로 panic 판단 → 강세장 흥분도 panic으로 잘못 분류.
    Strict-panic: proxy>30 AND KS200 60d DD ≤ threshold 일 때만 진짜 panic (lev 2x).
    그 외 (신고가 + 변동성 큼) 시기에는 lev fallback (1.0/1.2/1.5).

검증 항목:
    1. Threshold grid: DD ≤ {-5/-7/-10/-12/-15/-20}%
    2. Lev fallback grid: 1.0 / 1.2 / 1.5 (strong-bull 시 lev)
    3. WF 6 windows 비교 (V25-full vs Strict-panic best)
    4. IS/OOS holdout
    5. Year-by-year 분석
    6. 2008-2014 약세장 zone-only OOS

⚠️ Look-ahead bias 절대 금지:
    - ks200['dd_60d'] = (close / 60d_high - 1).shift(1)
    - vkospi_prev 이미 shift됨
    - 모든 신호 d-1까지의 정보로만 d 시점 액션 결정
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
import FinanceDataReader as fdr

from kr_v11_enhanced_modules import (
    apply_sector_cap, load_macro, macro_gate,
    dd_multistage_lev, pit_universe,
)
from kr_v12_integrated_champion import fetch_all_extended
from kr_v19_accumulation_hypotheses import add_features_v19, wf_alpha
from kr_v27_us_enhancement_port import load_parking_data
from kr_v02_sector_panic_buy import compute_regime, KR_BORROW_DAILY, KR_TAX, CASH_DAILY


def m(s):
    if len(s) < 2: return None
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    total = s.iloc[-1] - 1
    cagr = s.iloc[-1] ** (1/yrs) - 1 if yrs > 0 else 0
    rets = s.pct_change().dropna()
    sh = rets.mean()/rets.std() * (252**0.5) if rets.std() > 0 else 0
    dd = (s/s.cummax()-1).min()
    cal = cagr/abs(dd) if dd < 0 else 0
    return total, cagr, sh, dd, cal


def sim_strict_panic(data, macro,
                       strict_dd_thr=-0.10,     # proxy>30 일 때 DD ≤ this 만 lev 2x
                       lev_panic_real=2.0,       # 진짜 panic 일 때 lev
                       lev_panic_fallback=1.5,   # proxy>30 but DD>thr 일 때 lev
                       include_52w_low_rebound=True,
                       low_rebound_w=0.2,
                       normal_w=0.7, elevated_w=0.5, panic_w=0.3,
                       sq_thr=0.8,
                       top_k=7, sector_cap=3, rebal_days=21):
    """V25-full + Strict-panic zone rule (look-ahead bias 제거)."""
    ks200 = compute_regime(data['KS200'])
    ks200['close_60d_high'] = ks200['close'].rolling(60, min_periods=20).max()
    # CRITICAL: shift(1) for lag (d 시점에 d-1까지의 정보만)
    ks200['dd_60d'] = (ks200['close'] / ks200['close_60d_high'] - 1).shift(1)

    feat_data = {c: add_features_v19(data[c]) for c in data if c != 'KS200'}
    all_dates = sorted(ks200.index)
    val = 1.0; vals = []; dates_out = []
    holdings = {}; last_rebal = None; peak = 1.0; pit_cache = {}

    for i, d in enumerate(all_dates):
        if i < 252:
            vals.append(1.0); dates_out.append(d); continue
        v = ks200['vkospi_prev'].get(d, None)
        # Zone with strict-panic
        if pd.isna(v):
            base_lev = 1.0
        elif v < 15:
            base_lev = 0
        elif v < 22.5:
            base_lev = 1.0
        elif v < 30:
            base_lev = 1.5
        else:
            # proxy > 30 (potential panic)
            dd = ks200['dd_60d'].get(d, None)
            if dd is not None and not pd.isna(dd) and dd <= strict_dd_thr:
                base_lev = lev_panic_real      # real panic
            else:
                base_lev = lev_panic_fallback  # strong bull, fallback

        if d in macro.index:
            g = macro_gate(macro.loc[d], ks_row=None)
            if g == 'crisis': base_lev = 0
            elif g == 'caution': base_lev = min(base_lev, 1.0)

        cur_dd = val/peak - 1
        lev = dd_multistage_lev(base_lev, cur_dd)

        # Zone label for picking weight
        if pd.isna(v): zone_label = 'normal'
        elif v < 15: zone_label = 'cash'
        elif v < 22.5: zone_label = 'normal'
        elif v < 30: zone_label = 'elevated'
        else: zone_label = 'panic'

        # PnL
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

        # Rebal
        if last_rebal is None or (d - last_rebal).days >= rebal_days:
            if lev > 0:
                qk = (d.year, (d.month-1)//3)
                if qk not in pit_cache:
                    pit_cache[qk] = pit_universe(data, d, n=50, lookback_days=60)
                universe = pit_cache[qk]
                if zone_label == 'normal': w = normal_w
                elif zone_label == 'elevated': w = elevated_w
                elif zone_label == 'panic': w = panic_w
                else: w = normal_w
                scored = []
                for c in universe:
                    if c in feat_data and d in feat_data[c].index:
                        row = feat_data[c].loc[d]
                        m_val = row.get('ret_120d', None)
                        if m_val is None or pd.isna(m_val): continue
                        score = m_val * 100
                        sq = row.get('bb_squeeze_ratio', 1) or 1
                        breakout = row.get('bb_breakout', False)
                        if not pd.isna(sq) and sq <= sq_thr:
                            bonus = (1 - sq) * 100
                            if breakout: bonus += 30
                            score += bonus * w
                        if include_52w_low_rebound:
                            dl = row.get('dist_from_low', 0) or 0
                            r5 = row.get('ret_5d', 0) or 0
                            if not pd.isna(dl) and dl > 0.30:
                                score += min(dl, 1.0) * 100 * low_rebound_w * 0.3
                                score += r5 * 50 * low_rebound_w
                        scored.append((c, score))
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


def slice_metrics(s, start, end):
    sub = s.loc[start:end]
    if len(sub) < 5: return None
    return m(sub / sub.iloc[0])


def year_returns(s, s_bh):
    out = []
    for y in range(2015, 2027):
        ys = pd.to_datetime(f'{y}-01-01')
        ye = pd.to_datetime(f'{y+1}-01-01')
        sub = s.loc[ys:ye]
        sub_bh = s_bh.loc[ys:ye]
        if len(sub) < 5 or len(sub_bh) < 5: continue
        r = sub.iloc[-1]/sub.iloc[0] - 1
        r_bh = sub_bh.iloc[-1]/sub_bh.iloc[0] - 1
        out.append({'year': y, 'bh': r_bh, 'strat': r, 'alpha': r - r_bh})
    return pd.DataFrame(out)


if __name__ == '__main__':
    print('=' * 100)
    print('KR v31 — Strict-Panic 정밀 검증 (lag enforced)')
    print('=' * 100)
    print('Loading...')
    data = fetch_all_extended('2014-03-04')
    macro = load_macro('2014-01-01')
    parking_data = load_parking_data('2014-01-01')
    ks = data['KS200']
    s_bh = ks['close'] / ks['close'].iloc[0]

    # V25-full baseline
    from kr_v29_honest_revalidation import sim_v29
    s_v25 = sim_v29(data, macro, parking_data, lag_market_signals=True,
                     include_52w_low_rebound=True, low_rebound_w=0.2)
    res = m(s_v25)
    print(f'\n=== V25-full baseline ===')
    print(f'  total={res[0]*100:>+9.0f}% Sh={res[2]:.2f} DD={res[3]*100:+.1f}% Cal={res[4]:.2f}')

    # ============================================================
    # Phase 1: Threshold grid (lev_fallback=1.5 고정)
    # ============================================================
    print('\n=== Phase 1: DD threshold grid (lev_fallback=1.5) ===')
    print(f'{"thr":<8} {"total%":<10} {"CAGR%":<7} {"Sh":<5} {"DD%":<7} {"Cal":<5}')
    for thr in [-0.03, -0.05, -0.07, -0.10, -0.12, -0.15, -0.20, -0.25]:
        s = sim_strict_panic(data, macro, strict_dd_thr=thr, lev_panic_fallback=1.5)
        res = m(s)
        print(f'DD≤{thr*100:>+4.0f}%  {res[0]*100:>+9.0f}%  {res[1]*100:>+5.1f}%  {res[2]:.2f}  {res[3]*100:>+6.1f}%  {res[4]:.2f}')

    # ============================================================
    # Phase 2: Lev fallback grid (DD≤-10% 고정)
    # ============================================================
    print('\n=== Phase 2: Lev fallback grid (DD≤-10%) ===')
    print(f'{"fallback":<10} {"total%":<10} {"Sh":<5} {"DD%":<7} {"Cal":<5}')
    for fb in [1.0, 1.2, 1.3, 1.5, 1.7, 1.8]:
        s = sim_strict_panic(data, macro, strict_dd_thr=-0.10, lev_panic_fallback=fb)
        res = m(s)
        print(f'fb={fb:<8.1f} {res[0]*100:>+9.0f}%  {res[2]:.2f}  {res[3]*100:>+6.1f}%  {res[4]:.2f}')

    # ============================================================
    # Phase 3: 2-stage strict-panic (multi-DD threshold)
    # ============================================================
    print('\n=== Phase 3: Multi-stage panic ===')
    # 일반 idea: proxy>30 + DD≤-10% = lev 2.0x; proxy>30 + DD≤-5% = lev 1.7x; proxy>30 + DD>-5% = lev 1.5x
    # 더 정밀한 단계화

    def sim_multi_stage(data, macro, thresholds, levs, fallback=1.5):
        """thresholds = [-0.05, -0.10], levs = [1.7, 2.0], fallback=1.5"""
        ks200 = compute_regime(data['KS200'])
        ks200['close_60d_high'] = ks200['close'].rolling(60, min_periods=20).max()
        ks200['dd_60d'] = (ks200['close'] / ks200['close_60d_high'] - 1).shift(1)
        feat_data = {c: add_features_v19(data[c]) for c in data if c != 'KS200'}
        all_dates = sorted(ks200.index)
        val = 1.0; vals = []; holdings = {}; last_rebal = None; peak = 1.0; pit_cache = {}
        for i, d in enumerate(all_dates):
            if i < 252:
                vals.append(1.0); continue
            v = ks200['vkospi_prev'].get(d, None)
            if pd.isna(v): base_lev = 1.0
            elif v < 15: base_lev = 0
            elif v < 22.5: base_lev = 1.0
            elif v < 30: base_lev = 1.5
            else:
                dd = ks200['dd_60d'].get(d, None)
                if dd is not None and not pd.isna(dd):
                    lev = fallback
                    for t, l in zip(thresholds, levs):
                        if dd <= t: lev = l
                    base_lev = lev
                else:
                    base_lev = fallback
            if d in macro.index:
                g = macro_gate(macro.loc[d], ks_row=None)
                if g == 'crisis': base_lev = 0
                elif g == 'caution': base_lev = min(base_lev, 1.0)
            cur_dd = val/peak - 1
            lev = dd_multistage_lev(base_lev, cur_dd)
            if pd.isna(v): zone_label = 'normal'
            elif v < 15: zone_label = 'cash'
            elif v < 22.5: zone_label = 'normal'
            elif v < 30: zone_label = 'elevated'
            else: zone_label = 'panic'
            if lev > 0 and holdings:
                pr = sum(w * (feat_data[c]['ret'].get(d, 0) or 0) for c, w in holdings.items() if c in feat_data)
                cost = max(0, lev-1) * KR_BORROW_DAILY
                net = lev * pr - cost
            else: net = CASH_DAILY if lev == 0 else 0
            val *= (1 + net); val = max(val, 0.01); peak = max(peak, val)
            vals.append(val)
            if last_rebal is None or (d - last_rebal).days >= 21:
                if lev > 0:
                    qk = (d.year, (d.month-1)//3)
                    if qk not in pit_cache:
                        pit_cache[qk] = pit_universe(data, d, n=50, lookback_days=60)
                    universe = pit_cache[qk]
                    if zone_label == 'normal': w = 0.7
                    elif zone_label == 'elevated': w = 0.5
                    elif zone_label == 'panic': w = 0.3
                    else: w = 0.7
                    scored = []
                    for c in universe:
                        if c in feat_data and d in feat_data[c].index:
                            row = feat_data[c].loc[d]
                            m_val = row.get('ret_120d', None)
                            if m_val is None or pd.isna(m_val): continue
                            score = m_val * 100
                            sq = row.get('bb_squeeze_ratio', 1) or 1
                            breakout = row.get('bb_breakout', False)
                            if not pd.isna(sq) and sq <= 0.8:
                                bonus = (1 - sq) * 100
                                if breakout: bonus += 30
                                score += bonus * w
                            dl = row.get('dist_from_low', 0) or 0
                            r5 = row.get('ret_5d', 0) or 0
                            if not pd.isna(dl) and dl > 0.30:
                                score += min(dl, 1.0) * 100 * 0.2 * 0.3
                                score += r5 * 50 * 0.2
                            scored.append((c, score))
                    scored.sort(key=lambda x: -x[1])
                    scored = apply_sector_cap(scored, max_per_sector=3)
                    top = scored[:7]
                    new_h = {c: 1.0/len(top) for c, _ in top} if top else {}
                    changed = sum(abs(new_h.get(c, 0) - holdings.get(c, 0))
                                  for c in set(list(holdings.keys()) + list(new_h.keys())))
                    val *= (1 - KR_TAX * changed/2)
                    holdings = new_h; last_rebal = d
        return pd.Series(vals, index=all_dates)

    variants = [
        ('A: fb=1.5, -10%→2.0', [-0.10], [2.0], 1.5),
        ('B: fb=1.5, -5%→1.8, -10%→2.0', [-0.05, -0.10], [1.8, 2.0], 1.5),
        ('C: fb=1.5, -5%→1.7, -10%→1.9, -15%→2.0', [-0.05, -0.10, -0.15], [1.7, 1.9, 2.0], 1.5),
        ('D: fb=1.3, -5%→1.6, -10%→1.9', [-0.05, -0.10], [1.6, 1.9], 1.3),
        ('E: fb=1.5, -3%→1.7, -8%→2.0', [-0.03, -0.08], [1.7, 2.0], 1.5),
    ]
    for name, thrs, levs, fb in variants:
        s = sim_multi_stage(data, macro, thrs, levs, fb)
        res = m(s)
        print(f'{name:<55} total={res[0]*100:>+9.0f}%  Sh={res[2]:.2f}  DD={res[3]*100:>+6.1f}%  Cal={res[4]:.2f}')

    # ============================================================
    # Phase 4: WF 6 windows 비교 (V25-full vs Strict-panic best)
    # ============================================================
    print('\n=== Phase 4: WF 6 windows (V25-full vs Strict-panic DD≤-10% fb=1.5) ===')
    s_strict = sim_strict_panic(data, macro, strict_dd_thr=-0.10, lev_panic_fallback=1.5)
    wf_v25 = wf_alpha(s_v25, s_bh, n=6)
    wf_strict = wf_alpha(s_strict, s_bh, n=6)
    cmp = pd.DataFrame({
        'win': wf_v25['win'],
        'bh%': wf_v25['bh%'],
        'v25%': wf_v25['strat%'],
        'strict%': wf_strict['strat%'],
        'v25_alpha': wf_v25['alpha_pp'],
        'strict_alpha': wf_strict['alpha_pp'],
        'Δ': wf_strict['alpha_pp'] - wf_v25['alpha_pp'],
    })
    print(cmp.to_string(index=False))
    print(f'V25-full: alpha {(wf_v25["alpha_pp"]>0).sum()}/{len(wf_v25)}, mean={wf_v25["alpha_pp"].mean():+.1f}pp')
    print(f'Strict:   alpha {(wf_strict["alpha_pp"]>0).sum()}/{len(wf_strict)}, mean={wf_strict["alpha_pp"].mean():+.1f}pp')

    # ============================================================
    # Phase 5: IS/OOS Holdout
    # ============================================================
    print('\n=== Phase 5: IS/OOS Holdout ===')
    is_end = pd.to_datetime('2022-04-20')
    oos_start = pd.to_datetime('2022-04-21')
    for label, s in [('V25-full baseline', s_v25), ('Strict-panic DD≤-10%', s_strict)]:
        is_m = slice_metrics(s, '2015-03-16', is_end)
        oos_m = slice_metrics(s, oos_start, '2026-12-31')
        if is_m and oos_m:
            print(f'  {label:30s} IS Sh={is_m[2]:.2f} (total={is_m[0]*100:+.0f}%) → OOS Sh={oos_m[2]:.2f} (total={oos_m[0]*100:+.0f}%, DD={oos_m[3]*100:+.1f}%)')

    # ============================================================
    # Phase 6: Year-by-year
    # ============================================================
    print('\n=== Phase 6: Year-by-year ===')
    yr_v25 = year_returns(s_v25, s_bh)
    yr_strict = year_returns(s_strict, s_bh)
    cmp_yr = pd.DataFrame({
        'year': yr_v25['year'],
        'BH%': yr_v25['bh']*100,
        'V25%': yr_v25['strat']*100,
        'Strict%': yr_strict['strat']*100,
        'Δ (strict-v25)': (yr_strict['strat'] - yr_v25['strat'])*100,
    })
    print(cmp_yr.round(1).to_string(index=False))

    # ============================================================
    # Phase 7: 2008-2014 약세장 OOS (zone-only, GFC 포함)
    # ============================================================
    print('\n=== Phase 7: 2008-2014 약세장 OOS (zone-only, GFC + EU 위기 포함) ===')
    ks_long = fdr.DataReader('KS200', '2003-01-01', '2014-03-04')
    ks_long.columns = [c.lower() for c in ks_long.columns]
    ks_long['ret'] = ks_long['close'].pct_change()
    ks_long['ewma_vol'] = ks_long['ret'].ewm(alpha=0.06).std() * (252**0.5) * 100
    ks_long['vkospi_proxy'] = ks_long['ewma_vol'] * 1.25
    ks_long['vkospi_prev'] = ks_long['vkospi_proxy'].shift(1)
    ks_long['close_60d_high'] = ks_long['close'].rolling(60, min_periods=20).max()
    ks_long['dd_60d'] = (ks_long['close'] / ks_long['close_60d_high'] - 1).shift(1)
    macro_long = load_macro('2002-01-01')

    def zone_only_sim(strict=False, strict_thr=-0.10):
        val = 1.0; vals = []; peak = 1.0
        for i, d in enumerate(sorted(ks_long.index)):
            if i < 252:
                vals.append(1.0); continue
            v = ks_long['vkospi_prev'].get(d, None)
            if pd.isna(v): base_lev = 1.0
            elif v < 15: base_lev = 0
            elif v < 22.5: base_lev = 1.0
            elif v < 30: base_lev = 1.5
            else:
                if strict:
                    dd = ks_long['dd_60d'].get(d, None)
                    if dd is not None and not pd.isna(dd) and dd <= strict_thr:
                        base_lev = 2.0
                    else:
                        base_lev = 1.5
                else:
                    base_lev = 2.0
            if d in macro_long.index:
                g = macro_gate(macro_long.loc[d], ks_row=None)
                if g == 'crisis': base_lev = 0
                elif g == 'caution': base_lev = min(base_lev, 1.0)
            cur_dd = val/peak - 1
            lev = dd_multistage_lev(base_lev, cur_dd)
            r = ks_long['ret'].get(d, 0) or 0
            if lev == 0:
                net = CASH_DAILY
            else:
                cost = max(0, lev-1) * KR_BORROW_DAILY
                net = lev * r - cost
            val *= (1 + net); val = max(val, 0.01); peak = max(peak, val)
            vals.append(val)
        return pd.Series(vals, index=sorted(ks_long.index))

    s_bh_long = ks_long['close'] / ks_long['close'].iloc[0]
    s_v25_long = zone_only_sim(strict=False)
    s_strict_long = zone_only_sim(strict=True, strict_thr=-0.10)
    print(f'  BH KS200 2003-2014:           total={(s_bh_long.iloc[-1]-1)*100:+.1f}% Sh={(s_bh_long.pct_change().mean()/s_bh_long.pct_change().std()*(252**0.5)):.2f} DD={(s_bh_long/s_bh_long.cummax()-1).min()*100:.1f}%')
    res = m(s_v25_long)
    print(f'  V25-full zone (no strict):    total={res[0]*100:+.1f}% Sh={res[2]:.2f} DD={res[3]*100:.1f}%')
    res = m(s_strict_long)
    print(f'  Strict-panic DD≤-10%:         total={res[0]*100:+.1f}% Sh={res[2]:.2f} DD={res[3]*100:.1f}%')

    # GFC stress
    print('\n  GFC stress (2008-08 ~ 2009-06):')
    for name, s in [('BH', s_bh_long), ('V25 zone', s_v25_long), ('Strict-panic', s_strict_long)]:
        sub = s.loc['2008-08-01':'2009-06-30']
        if len(sub) > 1:
            r = sub.iloc[-1]/sub.iloc[0] - 1
            dd_v = (sub/sub.cummax()-1).min()
            print(f'    {name:18s} total={r*100:+.1f}%  DD={dd_v*100:+.1f}%')

"""KR v32 — Rebal Frequency 검증 + V31 Phase 3 fix.

질문 1: 월간(21d) holding이 정답인가?
질문 2: 종목을 다양한 조건에 따라 바꾸는 게 더 좋은가?

검증:
    A. Fixed rebal frequency grid (3/5/7/10/15/21/30/45/60일)
    B. Event-triggered rebal:
       - Zone 변경 시 즉시 rebal
       - Macro gate 변경 시 즉시 rebal
       - 두 트리거 결합
    C. Hybrid (event + 최소 N일 lock):
       - Event 발생해도 마지막 rebal 후 7일/10일 lock
    D. Multi-stage panic Phase 3 fix 재검증

⚠️ Look-ahead bias 절대 금지:
    - 모든 market signal shift(1)
    - Picking은 d 시점 종가까지 정보 사용 (장 마감 후 결정 가정)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

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


def sim_rebal_flex(data, macro,
                    rebal_days=21,
                    # Event triggers
                    event_zone_change=False,
                    event_macro_change=False,
                    event_min_lock_days=0,
                    # Strict-panic (V31 winner)
                    use_strict_panic=False,
                    strict_dd_thr=-0.10,
                    lev_panic_fallback=1.5,
                    # V25-full picking
                    include_52w_low_rebound=True,
                    low_rebound_w=0.2,
                    normal_w=0.7, elevated_w=0.5, panic_w=0.3,
                    sq_thr=0.8,
                    top_k=7, sector_cap=3):
    """Flexible rebal simulator (frequency + event trigger)."""
    ks200 = compute_regime(data['KS200'])
    ks200['close_60d_high'] = ks200['close'].rolling(60, min_periods=20).max()
    ks200['dd_60d'] = (ks200['close'] / ks200['close_60d_high'] - 1).shift(1)
    feat_data = {c: add_features_v19(data[c]) for c in data if c != 'KS200'}
    all_dates = sorted(ks200.index)

    val = 1.0; vals = []; dates_out = []
    holdings = {}; last_rebal = None; peak = 1.0; pit_cache = {}
    prev_zone_label = None
    prev_macro_gate = None

    for i, d in enumerate(all_dates):
        if i < 252:
            vals.append(1.0); dates_out.append(d); continue
        v = ks200['vkospi_prev'].get(d, None)

        # Zone lev (with optional strict-panic)
        if pd.isna(v): base_lev = 1.0
        elif v < 15: base_lev = 0
        elif v < 22.5: base_lev = 1.0
        elif v < 30: base_lev = 1.5
        else:
            if use_strict_panic:
                dd = ks200['dd_60d'].get(d, None)
                if dd is not None and not pd.isna(dd) and dd <= strict_dd_thr:
                    base_lev = 2.0
                else:
                    base_lev = lev_panic_fallback
            else:
                base_lev = 2.0

        gate = 'normal'
        if d in macro.index:
            gate = macro_gate(macro.loc[d], ks_row=None)
            if gate == 'crisis': base_lev = 0
            elif gate == 'caution': base_lev = min(base_lev, 1.0)
        cur_dd = val/peak - 1
        lev = dd_multistage_lev(base_lev, cur_dd)

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

        # Determine rebal trigger
        rebal_trigger = False
        if last_rebal is None:
            rebal_trigger = True
        else:
            days_since = (d - last_rebal).days
            # Fixed-period trigger
            if days_since >= rebal_days:
                rebal_trigger = True
            # Event trigger (with min lock days)
            if days_since >= event_min_lock_days:
                if event_zone_change and prev_zone_label is not None and zone_label != prev_zone_label:
                    rebal_trigger = True
                if event_macro_change and prev_macro_gate is not None and gate != prev_macro_gate:
                    rebal_trigger = True

        if rebal_trigger:
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
                holdings = new_h
                last_rebal = d
            elif lev == 0 and holdings:
                # FIX: lev==0 시 holdings 정리 (V31 Phase 3 bug fix)
                changed = sum(abs(w) for w in holdings.values())
                val *= (1 - KR_TAX * changed/2)
                holdings = {}
                last_rebal = d

        prev_zone_label = zone_label
        prev_macro_gate = gate

    return pd.Series(vals, index=dates_out)


def slice_metrics(s, start, end):
    sub = s.loc[start:end]
    if len(sub) < 5: return None
    return m(sub / sub.iloc[0])


if __name__ == '__main__':
    print('=' * 100)
    print('KR v32 — Rebal Frequency + Event Trigger 정직 검증')
    print('=' * 100)
    print('Loading...')
    data = fetch_all_extended('2014-03-04')
    macro = load_macro('2014-01-01')
    parking_data = load_parking_data('2014-01-01')
    ks = data['KS200']
    s_bh = ks['close'] / ks['close'].iloc[0]

    # V25-full baseline (21d, no event)
    print('\n=== Baselines ===')
    s_v25 = sim_rebal_flex(data, macro, rebal_days=21)
    res = m(s_v25)
    print(f'  V25-full baseline (21d):                  total={res[0]*100:>+9.0f}% Sh={res[2]:.2f} DD={res[3]*100:+.1f}% Cal={res[4]:.2f}')

    # Strict-Panic V25 baseline
    s_v25s = sim_rebal_flex(data, macro, rebal_days=21, use_strict_panic=True)
    res = m(s_v25s)
    print(f'  V25 + Strict-Panic (21d):                 total={res[0]*100:>+9.0f}% Sh={res[2]:.2f} DD={res[3]*100:+.1f}% Cal={res[4]:.2f}')

    # ============================================================
    # Phase A: Fixed rebal frequency grid
    # ============================================================
    print('\n=== Phase A: Fixed rebal frequency (V25-full base) ===')
    print(f'{"days":<6} {"total%":<10} {"CAGR%":<6} {"Sh":<5} {"DD%":<7} {"Cal":<5}')
    for rd in [3, 5, 7, 10, 15, 21, 25, 30, 45, 60, 90]:
        s = sim_rebal_flex(data, macro, rebal_days=rd)
        res = m(s)
        marker = ' ←' if rd == 21 else ''
        print(f'{rd:<4}d  {res[0]*100:>+9.0f}%  {res[1]*100:>+5.1f}%  {res[2]:.2f}  {res[3]*100:>+6.1f}%  {res[4]:.2f}{marker}')

    print('\n=== Phase A2: Fixed rebal frequency (Strict-Panic base) ===')
    print(f'{"days":<6} {"total%":<10} {"CAGR%":<6} {"Sh":<5} {"DD%":<7} {"Cal":<5}')
    for rd in [5, 10, 15, 21, 30, 45]:
        s = sim_rebal_flex(data, macro, rebal_days=rd, use_strict_panic=True)
        res = m(s)
        marker = ' ←' if rd == 21 else ''
        print(f'{rd:<4}d  {res[0]*100:>+9.0f}%  {res[1]*100:>+5.1f}%  {res[2]:.2f}  {res[3]*100:>+6.1f}%  {res[4]:.2f}{marker}')

    # ============================================================
    # Phase B: Event-triggered rebal
    # ============================================================
    print('\n=== Phase B: Event-triggered (V25-full base) ===')
    variants = [
        ('B1: 21d fixed only (baseline)', dict(rebal_days=21)),
        ('B2: Zone change + 21d max', dict(rebal_days=21, event_zone_change=True)),
        ('B3: Macro gate change + 21d max', dict(rebal_days=21, event_macro_change=True)),
        ('B4: Both events + 21d max', dict(rebal_days=21, event_zone_change=True, event_macro_change=True)),
        ('B5: Both events + 10d lock + 21d max', dict(rebal_days=21, event_zone_change=True, event_macro_change=True, event_min_lock_days=10)),
        ('B6: Both events + 5d lock + 21d max', dict(rebal_days=21, event_zone_change=True, event_macro_change=True, event_min_lock_days=5)),
        ('B7: Both events + 0 lock + 60d max (event-driven)', dict(rebal_days=60, event_zone_change=True, event_macro_change=True)),
        ('B8: Both events + 0 lock + 90d max', dict(rebal_days=90, event_zone_change=True, event_macro_change=True)),
    ]
    print(f'{"variant":<55} {"total%":<10} {"Sh":<5} {"DD%":<7} {"Cal":<5}')
    for name, kw in variants:
        s = sim_rebal_flex(data, macro, **kw)
        res = m(s)
        print(f'{name:<55} {res[0]*100:>+9.0f}%  {res[2]:.2f}  {res[3]*100:>+6.1f}%  {res[4]:.2f}')

    print('\n=== Phase B2: Event-triggered (Strict-Panic base) ===')
    print(f'{"variant":<55} {"total%":<10} {"Sh":<5} {"DD%":<7} {"Cal":<5}')
    for name, kw in [
        ('Strict-Panic 21d only', dict(rebal_days=21, use_strict_panic=True)),
        ('Strict + zone change + 21d', dict(rebal_days=21, use_strict_panic=True, event_zone_change=True)),
        ('Strict + both events + 21d', dict(rebal_days=21, use_strict_panic=True, event_zone_change=True, event_macro_change=True)),
        ('Strict + both events + 10d lock + 21d', dict(rebal_days=21, use_strict_panic=True, event_zone_change=True, event_macro_change=True, event_min_lock_days=10)),
        ('Strict + both events + 5d lock + 21d', dict(rebal_days=21, use_strict_panic=True, event_zone_change=True, event_macro_change=True, event_min_lock_days=5)),
        ('Strict + both events + 60d max (event-driven)', dict(rebal_days=60, use_strict_panic=True, event_zone_change=True, event_macro_change=True)),
    ]:
        s = sim_rebal_flex(data, macro, **kw)
        res = m(s)
        print(f'{name:<55} {res[0]*100:>+9.0f}%  {res[2]:.2f}  {res[3]*100:>+6.1f}%  {res[4]:.2f}')

    # ============================================================
    # Phase C: WF 6 windows for top candidates
    # ============================================================
    print('\n=== Phase C: WF 6 windows for top candidates ===')
    candidates = {
        'V25-full 21d': dict(rebal_days=21),
        'V25-full 10d': dict(rebal_days=10),
        'Strict 21d': dict(rebal_days=21, use_strict_panic=True),
        'V25 + zone events + 21d': dict(rebal_days=21, event_zone_change=True),
        'Strict + zone events + 21d': dict(rebal_days=21, use_strict_panic=True, event_zone_change=True),
        'Strict + both + 10d lock + 21d': dict(rebal_days=21, use_strict_panic=True, event_zone_change=True, event_macro_change=True, event_min_lock_days=10),
    }
    for name, kw in candidates.items():
        s = sim_rebal_flex(data, macro, **kw)
        wf = wf_alpha(s, s_bh, n=6)
        n_pos = (wf['alpha_pp'] > 0).sum()
        print(f'  {name:<45} alpha {n_pos}/{len(wf)}, mean={wf["alpha_pp"].mean():+.1f}pp')

    # ============================================================
    # Phase D: IS/OOS Holdout
    # ============================================================
    print('\n=== Phase D: IS/OOS Holdout ===')
    is_end = pd.to_datetime('2022-04-20')
    oos_start = pd.to_datetime('2022-04-21')
    print(f'{"variant":<50} {"IS Sh":<7} {"IS total%":<11} {"OOS Sh":<7} {"OOS total%":<11} {"OOS DD%":<10}')
    for name, kw in candidates.items():
        s = sim_rebal_flex(data, macro, **kw)
        is_m = slice_metrics(s, '2015-03-16', is_end)
        oos_m = slice_metrics(s, oos_start, '2026-12-31')
        if is_m and oos_m:
            print(f'  {name:<48} {is_m[2]:<7.2f} {is_m[0]*100:<+11.0f} {oos_m[2]:<7.2f} {oos_m[0]*100:<+11.0f} {oos_m[3]*100:<+10.1f}')

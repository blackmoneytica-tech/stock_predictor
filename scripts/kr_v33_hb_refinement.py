"""KR v33 — H-B Peak Exit 정밀 재검증.

사용자 지적: V25 + H-B 20/3/34가 +21,606% / Sh 1.41 / Cal 1.68로 가장 강력
또 H-B 트리거 시 어떻게 매도? 두 가지 옵션 명확화:
  - Stock-level: 트리거된 그 종목만 매도
  - Portfolio-level: 전체 포지션 1/3 일괄 감노출 (V30 구현, 현재 검증 결과)

⚠️ Look-ahead bias 절대 금지 (V30 H-B는 전일 ret/dv 사용)
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
from kr_v19_accumulation_hypotheses import wf_alpha
from kr_v27_us_enhancement_port import load_parking_data
from kr_v02_sector_panic_buy import compute_regime, KR_BORROW_DAILY, KR_TAX, CASH_DAILY
from kr_v30_market_leader_strategy import add_features_v30


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


def sim_hb_variant(data, macro,
                    # V25-full base
                    include_52w_low_rebound=True,
                    low_rebound_w=0.2,
                    normal_w=0.7, elevated_w=0.5, panic_w=0.3, sq_thr=0.8,
                    # H-B
                    peak_exit_enabled=False,
                    peak_ret_thr=0.10,
                    peak_dv_spike_thr=3.0,
                    peak_exit_pct=0.34,
                    peak_cooldown_days=5,
                    peak_mode='portfolio',     # 'portfolio' (전체 1/3) or 'stock' (트리거 종목만)
                    # Strict-Panic (optional)
                    use_strict_panic=False,
                    strict_dd_thr=-0.10,
                    lev_panic_fallback=1.5,
                    top_k=7, sector_cap=3, rebal_days=21):
    """H-B variant with stock-level vs portfolio-level option."""
    ks200 = compute_regime(data['KS200'])
    ks200['close_60d_high'] = ks200['close'].rolling(60, min_periods=20).max()
    ks200['dd_60d'] = (ks200['close'] / ks200['close_60d_high'] - 1).shift(1)
    feat_data = {c: add_features_v30(data[c]) for c in data if c != 'KS200'}
    all_dates = sorted(ks200.index)

    val = 1.0; vals = []; dates_out = []
    holdings = {}; last_rebal = None; peak = 1.0; pit_cache = {}
    last_peak_exit_idx = -999
    deployed_pct = 1.0  # portfolio-level 추적

    for i, d in enumerate(all_dates):
        if i < 252:
            vals.append(1.0); dates_out.append(d); continue
        v = ks200['vkospi_prev'].get(d, None)

        # Zone lev (with optional Strict-Panic)
        if pd.isna(v): base_lev = 1.0
        elif v < 15: base_lev = 0
        elif v < 22.5: base_lev = 1.0
        elif v < 30: base_lev = 1.5
        else:
            if use_strict_panic:
                dd_v = ks200['dd_60d'].get(d, None)
                if dd_v is not None and not pd.isna(dd_v) and dd_v <= strict_dd_thr:
                    base_lev = 2.0
                else:
                    base_lev = lev_panic_fallback
            else:
                base_lev = 2.0
        if d in macro.index:
            g = macro_gate(macro.loc[d], ks_row=None)
            if g == 'crisis': base_lev = 0
            elif g == 'caution': base_lev = min(base_lev, 1.0)
        cur_dd_cap = val/peak - 1
        lev = dd_multistage_lev(base_lev, cur_dd_cap)

        if pd.isna(v): zone_label = 'normal'
        elif v < 15: zone_label = 'cash'
        elif v < 22.5: zone_label = 'normal'
        elif v < 30: zone_label = 'elevated'
        else: zone_label = 'panic'

        # ──── H-B: Peak Exit (전일 ret + dv spike 기반) ────
        if peak_exit_enabled and holdings and (i - last_peak_exit_idx) > peak_cooldown_days:
            prev_idx = i - 1
            triggered_stocks = []
            for c in list(holdings.keys()):
                if c not in feat_data: continue
                df = feat_data[c]
                prev_date = all_dates[prev_idx]
                if prev_date not in df.index: continue
                row = df.loc[prev_date]
                r_prev = row.get('ret', 0) or 0
                dv_spike = row.get('dv_spike', 1.0) or 1.0
                if not pd.isna(r_prev) and r_prev > peak_ret_thr and \
                   not pd.isna(dv_spike) and dv_spike > peak_dv_spike_thr:
                    triggered_stocks.append(c)

            if triggered_stocks:
                if peak_mode == 'portfolio':
                    # Portfolio-level: 전체 deployed -= exit_pct (V30 동작)
                    deployed_pct = max(0.0, deployed_pct - peak_exit_pct)
                    val *= (1 - KR_TAX * peak_exit_pct / 2)
                elif peak_mode == 'stock':
                    # Stock-level: 트리거 종목 weight × exit_pct 매도
                    for c in triggered_stocks:
                        sell_w = holdings[c] * peak_exit_pct
                        holdings[c] -= sell_w
                        val *= (1 - KR_TAX * sell_w / 2)
                last_peak_exit_idx = i

        # PnL
        if lev > 0 and holdings:
            pr = sum(w * (feat_data[c]['ret'].get(d, 0) or 0)
                     for c, w in holdings.items() if c in feat_data)
            cost = max(0, lev-1) * KR_BORROW_DAILY
            if peak_mode == 'portfolio':
                invested_ret = (lev * pr - cost) * deployed_pct
                cash_ret = CASH_DAILY * (1 - deployed_pct)
                net = invested_ret + cash_ret
            else:
                # stock-level: holdings 합이 ≤1.0 (트리거 시 일부 감소)
                holdings_sum = sum(holdings.values())
                invested_ret = lev * pr - cost
                cash_ret = CASH_DAILY * (1 - holdings_sum)
                net = invested_ret + cash_ret
        else:
            net = CASH_DAILY if lev == 0 else 0
        val *= (1 + net); val = max(val, 0.01)
        peak = max(peak, val)
        vals.append(val); dates_out.append(d)

        # Rebal (monthly 21d)
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
                deployed_pct = 1.0
            elif lev == 0 and holdings:
                changed = sum(abs(w) for w in holdings.values())
                val *= (1 - KR_TAX * changed/2)
                holdings = {}; last_rebal = d

    return pd.Series(vals, index=dates_out)


def slice_metrics(s, start, end):
    sub = s.loc[start:end]
    if len(sub) < 5: return None
    return m(sub / sub.iloc[0])


if __name__ == '__main__':
    print('=' * 100)
    print('KR v33 — H-B Peak Exit 정밀 재검증 (portfolio vs stock mode)')
    print('=' * 100)
    print('Loading...')
    data = fetch_all_extended('2014-03-04')
    macro = load_macro('2014-01-01')
    ks = data['KS200']
    s_bh = ks['close'] / ks['close'].iloc[0]

    # Baseline V25-full
    print('\n=== Baseline ===')
    s_v25 = sim_hb_variant(data, macro, peak_exit_enabled=False)
    res = m(s_v25)
    print(f'  V25-full (no H-B):  total={res[0]*100:>+9.0f}% Sh={res[2]:.2f} DD={res[3]*100:+.1f}% Cal={res[4]:.2f}')

    # Strict-Panic baseline
    s_strict = sim_hb_variant(data, macro, use_strict_panic=True, peak_exit_enabled=False)
    res = m(s_strict)
    print(f'  V25 + Strict-Panic: total={res[0]*100:>+9.0f}% Sh={res[2]:.2f} DD={res[3]*100:+.1f}% Cal={res[4]:.2f}')

    # ============================================================
    # Phase A: Portfolio-level H-B grid (V30 style)
    # ============================================================
    print('\n=== Phase A: Portfolio-level H-B (V25-full + H-B 다양 threshold) ===')
    print(f'{"variant":<35} {"total%":<10} {"CAGR%":<6} {"Sh":<5} {"DD%":<7} {"Cal":<5}')
    hb_variants = [
        (0.05, 2.0, 0.34, 'ret5%/dv2x/exit34%'),
        (0.05, 3.0, 0.34, 'ret5%/dv3x/exit34%'),
        (0.10, 2.0, 0.34, 'ret10%/dv2x/exit34%'),
        (0.10, 3.0, 0.34, 'ret10%/dv3x/exit34%'),
        (0.10, 3.0, 0.50, 'ret10%/dv3x/exit50%'),
        (0.15, 3.0, 0.34, 'ret15%/dv3x/exit34%'),
        (0.20, 3.0, 0.34, 'ret20%/dv3x/exit34%'),
        (0.20, 3.0, 0.50, 'ret20%/dv3x/exit50%'),
        (0.20, 5.0, 0.34, 'ret20%/dv5x/exit34%'),
        (0.25, 3.0, 0.34, 'ret25%/dv3x/exit34%'),
        (0.30, 3.0, 0.34, 'ret30%/dv3x/exit34%'),
    ]
    for r_t, dv_t, ex_p, name in hb_variants:
        s = sim_hb_variant(data, macro, peak_exit_enabled=True,
                            peak_ret_thr=r_t, peak_dv_spike_thr=dv_t, peak_exit_pct=ex_p,
                            peak_mode='portfolio')
        res = m(s)
        print(f'{name:<35} {res[0]*100:>+9.0f}%  {res[1]*100:>+5.1f}%  {res[2]:.2f}  {res[3]*100:>+6.1f}%  {res[4]:.2f}')

    # ============================================================
    # Phase B: Stock-level H-B grid
    # ============================================================
    print('\n=== Phase B: Stock-level H-B (트리거 종목만 매도) ===')
    print(f'{"variant":<35} {"total%":<10} {"Sh":<5} {"DD%":<7} {"Cal":<5}')
    for r_t, dv_t, ex_p, name in [
        (0.10, 3.0, 0.34, 'ret10%/dv3x/exit34%'),
        (0.10, 3.0, 0.50, 'ret10%/dv3x/exit50%'),
        (0.10, 3.0, 1.00, 'ret10%/dv3x/exit100% (전량)'),
        (0.20, 3.0, 0.34, 'ret20%/dv3x/exit34%'),
        (0.20, 3.0, 0.50, 'ret20%/dv3x/exit50%'),
        (0.20, 3.0, 1.00, 'ret20%/dv3x/exit100% (전량)'),
        (0.15, 3.0, 0.50, 'ret15%/dv3x/exit50%'),
    ]:
        s = sim_hb_variant(data, macro, peak_exit_enabled=True,
                            peak_ret_thr=r_t, peak_dv_spike_thr=dv_t, peak_exit_pct=ex_p,
                            peak_mode='stock')
        res = m(s)
        print(f'{name:<35} {res[0]*100:>+9.0f}%  {res[2]:.2f}  {res[3]*100:>+6.1f}%  {res[4]:.2f}')

    # ============================================================
    # Phase C: H-B + Strict-Panic combo
    # ============================================================
    print('\n=== Phase C: V25 + H-B + Strict-Panic combo ===')
    print(f'{"variant":<55} {"total%":<10} {"Sh":<5} {"DD%":<7} {"Cal":<5}')
    for name, kw in [
        ('V25-full baseline', dict()),
        ('V25 + Strict-Panic', dict(use_strict_panic=True)),
        ('V25 + H-B 10/3/34 portfolio', dict(peak_exit_enabled=True, peak_ret_thr=0.10, peak_dv_spike_thr=3.0, peak_exit_pct=0.34, peak_mode='portfolio')),
        ('V25 + H-B 20/3/34 portfolio', dict(peak_exit_enabled=True, peak_ret_thr=0.20, peak_dv_spike_thr=3.0, peak_exit_pct=0.34, peak_mode='portfolio')),
        ('V25 + H-B 20/3/34 + Strict-Panic', dict(peak_exit_enabled=True, peak_ret_thr=0.20, peak_dv_spike_thr=3.0, peak_exit_pct=0.34, peak_mode='portfolio', use_strict_panic=True)),
        ('V25 + H-B 20/3/50 stock', dict(peak_exit_enabled=True, peak_ret_thr=0.20, peak_dv_spike_thr=3.0, peak_exit_pct=0.50, peak_mode='stock')),
        ('V25 + H-B 20/3/100 stock + Strict-Panic', dict(peak_exit_enabled=True, peak_ret_thr=0.20, peak_dv_spike_thr=3.0, peak_exit_pct=1.00, peak_mode='stock', use_strict_panic=True)),
    ]:
        s = sim_hb_variant(data, macro, **kw)
        res = m(s)
        print(f'{name:<55} {res[0]*100:>+9.0f}%  {res[2]:.2f}  {res[3]*100:>+6.1f}%  {res[4]:.2f}')

    # ============================================================
    # Phase D: WF + IS/OOS for top candidates
    # ============================================================
    print('\n=== Phase D: WF + IS/OOS for top candidates ===')
    is_end = pd.to_datetime('2022-04-20')
    oos_start = pd.to_datetime('2022-04-21')
    candidates = {
        'V25-full': dict(),
        'V25 + Strict-Panic': dict(use_strict_panic=True),
        'V25 + H-B 10/3/34 portf': dict(peak_exit_enabled=True, peak_ret_thr=0.10, peak_dv_spike_thr=3.0, peak_exit_pct=0.34, peak_mode='portfolio'),
        'V25 + H-B 20/3/34 portf': dict(peak_exit_enabled=True, peak_ret_thr=0.20, peak_dv_spike_thr=3.0, peak_exit_pct=0.34, peak_mode='portfolio'),
        'V25 + H-B 20/3/34 stock': dict(peak_exit_enabled=True, peak_ret_thr=0.20, peak_dv_spike_thr=3.0, peak_exit_pct=0.34, peak_mode='stock'),
        'V25 + H-B 20/3/34 portf + Strict': dict(peak_exit_enabled=True, peak_ret_thr=0.20, peak_dv_spike_thr=3.0, peak_exit_pct=0.34, peak_mode='portfolio', use_strict_panic=True),
    }
    print(f'{"variant":<45} {"WF α":<8} {"IS Sh":<7} {"OOS Sh":<7} {"OOS Total":<11} {"OOS DD":<7}')
    for label, kw in candidates.items():
        s = sim_hb_variant(data, macro, **kw)
        wf = wf_alpha(s, s_bh, n=6)
        is_m = slice_metrics(s, '2015-03-16', is_end)
        oos_m = slice_metrics(s, oos_start, '2026-12-31')
        if is_m and oos_m:
            print(f'  {label:<43} {wf["alpha_pp"].mean():>+6.0f}pp  {is_m[2]:<7.2f} {oos_m[2]:<7.2f} {oos_m[0]*100:<+11.0f} {oos_m[3]*100:<+7.1f}')

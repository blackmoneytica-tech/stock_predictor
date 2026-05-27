"""KR v41 — 이미지 방법론(이평선·상대강도·거래량) 현 전략 강화 검증.

현 Champion (V25-full + H-B 20/3/34) 유지 + 3개 신호 추가:
    #2 이평선 추세: close vs EMA50/EMA200 (하락추세 회피 → DD↓ 기대)
    #5 상대강도 RS: 종목 ret_120d - 시장(KS200) ret_120d (약세장 차별화)
    #3 거래량 품질: 상승 동반 거래량 (OBV 증가 + 상승) 가산

목표: 수익률↑ AND/OR MDD↓ (현 전략 대비 robust 개선만 채택)
검증: Full 12.2년 + 최근 2년, 둘 다 개선돼야 진짜 alpha.

⚠️ Look-ahead 금지 — picking은 rebal day d 정보만 (그 시점까지 확정).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np

from kr_v11_enhanced_modules import apply_sector_cap, load_macro, macro_gate, dd_multistage_lev, pit_universe
from kr_v12_integrated_champion import fetch_all_extended
from kr_v02_sector_panic_buy import compute_regime, KR_BORROW_DAILY, KR_TAX, CASH_DAILY
from kr_v30_market_leader_strategy import add_features_v30
from kr_v34_topk_weight_exit import m


def sim_flex_enh(data, macro, top_k=7, sector_cap=3,
                  exit_ret_thr=0.20, exit_dv_thr=3.0, exit_pct=0.34, exit_cooldown=5,
                  include_52w_low_rebound=True, low_rebound_w=0.2,
                  normal_w=0.7, elevated_w=0.5, panic_w=0.3, sq_thr=0.8, rebal_days=21,
                  # === 신규 신호 ===
                  trend_filter='none',   # 'none'/'ema50_pen'/'ema200_pen'/'ema50_hard'/'ema200_hard'
                  trend_penalty=50,
                  rs_w=0.0,              # 상대강도 가산 weight
                  rs_regime='always',    # 'always'/'bear'/'bull'/'panic'/'high_dd10'/'high_dd15'
                  vol_w=0.0,             # 거래량 품질 가산 weight
                  universe_n=50,         # 선정 풀 크기 (거래대금 top N)
                  mom_lb=120,            # 모멘텀 lookback (ret_120d 기본)
                  track_counts=False):   # True면 (series, count_log) 반환
    """V25-full + H-B + 이미지 신호 (이평선/RS/거래량). H-B portfolio exit 고정."""
    ks200 = compute_regime(data['KS200'])
    feat_data = {}
    for c in data:
        if c == 'KS200': continue
        df = add_features_v30(data[c]).copy()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
        feat_data[c] = df
    ks_feat = add_features_v30(data['KS200']).copy()
    ks_feat['ema200'] = ks_feat['close'].ewm(span=200, adjust=False).mean()
    ks_high60 = ks_feat['close'].rolling(60, min_periods=20).max()
    ks_feat['dd60'] = (ks_feat['close'] / ks_high60 - 1) * 100
    all_dates = sorted(ks200.index)

    val = 1.0; vals = []; dates_out = []
    holdings = {}
    last_rebal = None
    peak = 1.0
    pit_cache = {}
    last_exit_idx = -999
    deployed_pct = 1.0
    count_log = []   # (date, zone, n_holdings, n_candidates, max_weight)

    for i, d in enumerate(all_dates):
        if i < 252:
            vals.append(1.0); dates_out.append(d); continue
        v = ks200['vkospi_prev'].get(d, None)
        if pd.isna(v): base_lev = 1.0
        elif v < 15: base_lev = 0
        elif v < 22.5: base_lev = 1.0
        elif v < 30: base_lev = 1.5
        else: base_lev = 2.0
        if d in macro.index:
            g = macro_gate(macro.loc[d], ks_row=None)
            if g == 'crisis': base_lev = 0
            elif g == 'caution': base_lev = min(base_lev, 1.0)
        cur_dd = val/peak - 1
        lev = dd_multistage_lev(base_lev, cur_dd)

        # H-B exit (lag enforced)
        if holdings and (i - last_exit_idx) > exit_cooldown:
            prev_date = all_dates[i-1]
            triggered = []
            for c in list(holdings.keys()):
                if c not in feat_data: continue
                df = feat_data[c]
                if prev_date not in df.index: continue
                row = df.loc[prev_date]
                r_prev = row.get('ret', 0) or 0
                dv_spike = row.get('dv_spike', 1.0) or 1.0
                if not pd.isna(r_prev) and r_prev > exit_ret_thr and \
                   not pd.isna(dv_spike) and dv_spike > exit_dv_thr:
                    triggered.append(c)
            if triggered:
                deployed_pct = max(0.0, deployed_pct - exit_pct)
                val *= (1 - KR_TAX * exit_pct / 2)
                last_exit_idx = i

        # PnL
        if lev > 0 and holdings:
            pr = 0
            for c, w in holdings.items():
                if c in feat_data:
                    r = feat_data[c]['ret'].get(d, 0) or 0
                    pr += w * (r if pd.notna(r) else 0)
            cost = max(0, lev-1) * KR_BORROW_DAILY
            invested_ret = (lev * pr - cost) * deployed_pct
            cash_ret = CASH_DAILY * (1 - deployed_pct)
            net = invested_ret + cash_ret
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
                    pit_cache[qk] = pit_universe(data, d, n=universe_n, lookback_days=60)
                universe = pit_cache[qk]
                if zone_label_for(v) == 'normal': w_sq = normal_w
                elif zone_label_for(v) == 'elevated': w_sq = elevated_w
                elif zone_label_for(v) == 'panic': w_sq = panic_w
                else: w_sq = normal_w

                # 시장(KS200) ret_120d for RS + regime 판단 (rebal day d 기준)
                ks_mom = ks_feat['ret_120d'].get(d, 0) or 0
                ks_close = ks_feat['close'].get(d, None)
                ks_ema200 = ks_feat['ema200'].get(d, None)
                ks_dd60 = ks_feat['dd60'].get(d, 0) or 0
                ks_below_ema = (ks_close is not None and ks_ema200 is not None
                                 and not pd.isna(ks_ema200) and ks_close < ks_ema200)
                zlab = zone_label_for(v)
                # RS 적용 여부 (regime별)
                rs_apply = False
                if rs_w > 0:
                    if rs_regime == 'always': rs_apply = True
                    elif rs_regime == 'bear': rs_apply = ks_below_ema
                    elif rs_regime == 'bull': rs_apply = not ks_below_ema
                    elif rs_regime == 'panic': rs_apply = (zlab == 'panic')
                    elif rs_regime == 'high_dd10': rs_apply = ks_dd60 <= -10
                    elif rs_regime == 'high_dd15': rs_apply = ks_dd60 <= -15

                scored = []
                mom_col = f'ret_{mom_lb}d'
                for c in universe:
                    if c in feat_data and d in feat_data[c].index:
                        row = feat_data[c].loc[d]
                        m_val = row.get(mom_col, None)
                        if m_val is None or pd.isna(m_val): continue
                        score = m_val * 100
                        sq = row.get('bb_squeeze_ratio', 1) or 1
                        breakout = row.get('bb_breakout', False)
                        if not pd.isna(sq) and sq <= sq_thr:
                            bonus = (1 - sq) * 100
                            if breakout: bonus += 30
                            score += bonus * w_sq
                        if include_52w_low_rebound:
                            dl = row.get('dist_from_low', 0) or 0
                            r5 = row.get('ret_5d', 0) or 0
                            if not pd.isna(dl) and dl > 0.30:
                                score += min(dl, 1.0) * 100 * low_rebound_w * 0.3
                                score += r5 * 50 * low_rebound_w

                        # === #5 상대강도 RS (시장 대비, regime-gated) ===
                        if rs_apply:
                            rs = (m_val - ks_mom) * 100
                            score += rs * rs_w

                        # === #3 거래량 품질 (상승 + OBV 증가) ===
                        if vol_w > 0:
                            obv_chg = row.get('obv_30d_chg', 0) or 0
                            price_chg = row.get('price_30d_chg', 0) or 0
                            # 상승 동반 거래량 (OBV와 가격 동반 상승)
                            if obv_chg > 0 and price_chg > 0:
                                score += min(obv_chg, 1.0) * 100 * vol_w

                        # === #2 이평선 추세 필터 ===
                        close = row.get('close')
                        ema50 = row.get('ema50'); ema200 = row.get('ema200')
                        skip = False
                        if trend_filter == 'ema200_hard' and not pd.isna(ema200) and close < ema200:
                            skip = True
                        elif trend_filter == 'ema50_hard' and not pd.isna(ema50) and close < ema50:
                            skip = True
                        elif trend_filter == 'ema200_pen' and not pd.isna(ema200) and close < ema200:
                            score -= trend_penalty
                        elif trend_filter == 'ema50_pen' and not pd.isna(ema50) and close < ema50:
                            score -= trend_penalty
                        if skip: continue

                        scored.append((c, score))
                scored.sort(key=lambda x: -x[1])
                scored = apply_sector_cap(scored, max_per_sector=sector_cap)
                top = scored[:top_k]

                new_h = {}
                if top:
                    w_each = 1.0 / len(top)
                    new_h = {c: w_each for c, _ in top}
                changed = sum(abs(new_h.get(c, 0) - holdings.get(c, 0))
                              for c in set(list(holdings.keys()) + list(new_h.keys())))
                val *= (1 - KR_TAX * changed/2)
                holdings = new_h
                last_rebal = d
                deployed_pct = 1.0
                if track_counts:
                    mw = max(new_h.values()) if new_h else 0
                    count_log.append((d, zone_label_for(v), len(new_h), len(scored), mw))
            elif lev == 0 and holdings:
                changed = sum(abs(w) for w in holdings.values())
                val *= (1 - KR_TAX * changed/2)
                holdings = {}
                last_rebal = d
                deployed_pct = 1.0

    if track_counts:
        return pd.Series(vals, index=dates_out), count_log
    return pd.Series(vals, index=dates_out)


def zone_label_for(v):
    if pd.isna(v): return 'normal'
    if v < 15: return 'cash'
    if v < 22.5: return 'normal'
    if v < 30: return 'elevated'
    return 'panic'


if __name__ == '__main__':
    import time
    t0 = time.time()
    print('=' * 92)
    print('KR v41 — 이미지 방법론 (이평선/RS/거래량) 현 전략 강화 검증')
    print('=' * 92)
    print('Loading...')
    data = fetch_all_extended('2014-03-04')
    macro = load_macro('2014-01-01')

    variants = [
        ('★ Baseline (현 Champion)', dict()),
        ('+ EMA200 hard (지난 검증 winner)', dict(trend_filter='ema200_hard')),
        # RS regime별 (w=0.5)
        ('RS always (w0.5)', dict(rs_w=0.5, rs_regime='always')),
        ('RS bear-only (KS<EMA200)', dict(rs_w=0.5, rs_regime='bear')),
        ('RS bull-only (KS>EMA200)', dict(rs_w=0.5, rs_regime='bull')),
        ('RS panic-only', dict(rs_w=0.5, rs_regime='panic')),
        ('RS high_dd10 (KS DD<-10%)', dict(rs_w=0.5, rs_regime='high_dd10')),
        ('RS high_dd15 (KS DD<-15%)', dict(rs_w=0.5, rs_regime='high_dd15')),
        # RS regime별 (w=1.0 더 강하게)
        ('RS bear-only (w1.0)', dict(rs_w=1.0, rs_regime='bear')),
        ('RS high_dd10 (w1.0)', dict(rs_w=1.0, rs_regime='high_dd10')),
        # best RS regime + EMA200 조합
        ('RS bear(0.5) + EMA200 hard', dict(rs_w=0.5, rs_regime='bear', trend_filter='ema200_hard')),
        ('RS high_dd10(0.5) + EMA200 hard', dict(rs_w=0.5, rs_regime='high_dd10', trend_filter='ema200_hard')),
    ]

    sims = {}
    for name, kw in variants:
        sims[name] = sim_flex_enh(data, macro, top_k=7, **kw)

    end = pd.to_datetime('2026-05-26')
    full_start = pd.to_datetime('2015-03-16')
    y2_start = pd.to_datetime('2024-05-27')

    def met(s, start):
        sub = s.loc[start:end]
        if len(sub) < 5: return None
        return m(sub / sub.iloc[0])

    def met_period(s, sd, ed):
        sub = s.loc[sd:ed]
        if len(sub) < 5: return None
        return m(sub / sub.iloc[0])
    bear22 = (pd.to_datetime('2022-01-01'), pd.to_datetime('2022-12-31'))
    bear18 = (pd.to_datetime('2018-01-01'), pd.to_datetime('2018-12-31'))

    print(f'\n{"전략":<34} {"Full Tot":<10} {"FullSh":<7} {"FullMDD":<9} | {"2022약세":<9} {"2018약세":<9} | {"2Y Sh":<6}')
    print('-' * 100)
    base_full = met(sims['★ Baseline (현 Champion)'], full_start)
    base_2y = met(sims['★ Baseline (현 Champion)'], y2_start)
    b22 = met_period(sims['★ Baseline (현 Champion)'], *bear22)
    b18 = met_period(sims['★ Baseline (현 Champion)'], *bear18)
    for name, _ in variants:
        rf = met(sims[name], full_start)
        r2 = met(sims[name], y2_start)
        r22 = met_period(sims[name], *bear22)
        r18 = met_period(sims[name], *bear18)
        if rf is None or r2 is None:
            print(f'{name:<34} (부족)'); continue
        mark = ''
        if not name.startswith('★'):
            full_ok = rf[0] >= base_full[0] * 0.97   # Full 수익 거의 유지(-3% 이내)
            bear_better = r22[0] > b22[0] and r18[0] > b18[0]   # 약세장 둘 다 개선
            sh_better = rf[2] > base_full[2] and r2[2] > base_2y[2]
            if full_ok and bear_better: mark = ' ⭐⭐ (약세장↑ + Full유지)'
            elif bear_better: mark = ' 🛡 (약세장 둘다↑)'
            elif sh_better: mark = ' ⭐ (Sharpe robust)'
        print(f'{name:<34} {rf[0]*100:>+8.0f}% {rf[2]:>6.2f} {rf[3]*100:>+7.1f}% | '
              f'{r22[0]*100:>+7.1f}% {r18[0]*100:>+7.1f}% | {r2[2]:>5.2f}{mark}')

    print(f'\nElapsed: {time.time()-t0:.0f}s')

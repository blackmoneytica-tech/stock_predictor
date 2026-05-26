"""KR v27 — US G5_22 sc_v48 enhancement 4종을 한국 V25에 검증.

미국 winner = F2 (A3 + C4 + GLD parking):
    A3. SPY 60d DD<-10% → cash
    C4. SPY DD scale + VIX scale
    GLD parking

한국 V25 baseline:
    mom120 + zone-dep squeeze + DD multi-stage + macro gate

한국에서 검증할 4 enhancement:
    A. KS200 Early Exit (60d DD<-10% / EMA200 이탈)
    B. Cash Parking 옵션 (KOFR baseline / 132030 GLD / 148070 TLT / BondMix / USDKRW)
    C. Dynamic Leverage Scale (KS200 DD / proxy zone 추가 scaling)
    D. Tranche Entry (분할 매수)

한국 특수성:
    - VKOSPI proxy >30 = panic zone = MAX BUY (반대로 미국은 cut)
    - 한국 mean-reversion 특성 → Early Exit이 alpha killer 가능성
    - 그래도 정직하게 검증
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import time

from kr_v11_enhanced_modules import (
    apply_sector_cap, load_macro, macro_gate,
    dd_multistage_lev, pit_universe,
)
from kr_v12_integrated_champion import zone_base_lev, fetch_all_extended
from kr_v19_accumulation_hypotheses import add_features_v19, m, wf_alpha
from kr_v02_sector_panic_buy import compute_regime, KR_BORROW_DAILY, KR_TAX, CASH_DAILY


# Korean parking ETFs
PARKING_ETFS = {
    'GLD': '132030',    # KODEX 골드선물
    'TLT': '152380',    # KODEX 국고채30년액티브
    'TBILL': '148070',  # KOSEF 국고채10년 (중기채로 단기 proxy)
    'CASH': None,        # KOFR 4% annualized
}


def load_parking_data(start='2014-03-04'):
    """Parking ETF 데이터 로드."""
    out = {}
    for label, code in PARKING_ETFS.items():
        if code is None:
            continue
        try:
            df = fdr.DataReader(code, start)
            df.columns = [c.lower() for c in df.columns]
            df['ret'] = df['close'].pct_change()
            out[label] = df
            print(f'  {label} ({code}): {df.index[0].date()}~{df.index[-1].date()} ({len(df)} days)')
        except Exception as e:
            print(f'  {label}: FAIL {e}')
    return out


def get_parking_ret(d, d_prev, parking, parking_data):
    """Parking 일일 수익률."""
    KR_CASH_DAILY = 0.03 / 252  # KOFR 3%/yr
    if parking == 'CASH' or parking is None:
        return KR_CASH_DAILY
    if parking == 'BONDMIX':
        # 50 cash + 50 TLT
        r1 = KR_CASH_DAILY * 0.5
        tlt = parking_data.get('TLT')
        if tlt is not None and d in tlt.index and d_prev in tlt.index:
            r2 = (tlt['close'].loc[d] / tlt['close'].loc[d_prev] - 1) * 0.5
        else:
            r2 = KR_CASH_DAILY * 0.5
        return r1 + r2
    df = parking_data.get(parking)
    if df is None or d not in df.index or d_prev not in df.index:
        return KR_CASH_DAILY
    return df['close'].loc[d] / df['close'].loc[d_prev] - 1


def sim_v27(data, macro, parking_data,
             # A. Early Exit
             exit_ks200_dd60=None,      # 예: -0.10
             exit_ks200_ema200=False,    # KS200 EMA200 이탈 + grace
             exit_grace_days=3,
             # B. Cash Parking
             parking='CASH',             # CASH/GLD/TLT/TBILL/BONDMIX
             # C. Dynamic Leverage Scale (V25 zone-lev에 추가 곱)
             lev_scale_ks_dd=None,       # 예: {-0.05: 0.85, -0.10: 0.5}
             lev_scale_proxy=None,       # 예: {30: 0.5}
             # D. Tranche Entry
             tranche_pcts=None,
             tranche_triggers=None,
             # V25 base
             normal_w=0.7, elevated_w=0.5, panic_w=0.3, sq_thr=0.8,
             top_k=7, sector_cap=3, rebal_days=21):
    """V27 enhanced simulator."""
    ks200 = compute_regime(data['KS200'])
    ks200['close_60d_high'] = ks200['close'].rolling(60, min_periods=20).max()
    ks200['dd_60d'] = ks200['close'] / ks200['close_60d_high'] - 1
    ks200['ema200'] = ks200['close'].ewm(span=200).mean()
    ks200['above_ema200'] = ks200['close'] > ks200['ema200']

    feat_data = {c: add_features_v19(data[c]) for c in data if c != 'KS200'}
    all_dates = sorted(ks200.index)
    val = 1.0; vals = []; dates_out = []
    holdings = {}; last_rebal = None; peak = 1.0; pit_cache = {}

    # Exit state
    in_exit = False
    exit_entry_idx = -1
    EXIT_REENTRY_DAYS = 20

    # Tranche state
    if tranche_pcts is None:
        tranche_pcts = [1.0]
        tranche_triggers = [(0, 'now')]
    deployed_pct = 0.0
    next_tranche = 0

    prev_d = all_dates[0]
    for i, d in enumerate(all_dates):
        if i < 252:
            vals.append(1.0); dates_out.append(d); prev_d = d; continue
        v_proxy = ks200['vkospi_prev'].get(d, None)
        base_lev = zone_base_lev(v_proxy)
        if d in macro.index:
            g = macro_gate(macro.loc[d], ks_row=None)
            if g == 'crisis': base_lev = 0
            elif g == 'caution': base_lev = min(base_lev, 1.0)
        cur_dd = val/peak - 1
        lev = dd_multistage_lev(base_lev, cur_dd)

        # Zone label
        if pd.isna(v_proxy): zone_label = 'normal'
        elif v_proxy < 15: zone_label = 'cash'
        elif v_proxy < 22.5: zone_label = 'normal'
        elif v_proxy < 30: zone_label = 'elevated'
        else: zone_label = 'panic'

        # ──── A. Early Exit signals ────
        force_exit = False
        if exit_ks200_dd60 is not None and d in ks200.index:
            mdd = ks200.loc[d, 'dd_60d']
            if not pd.isna(mdd) and mdd < exit_ks200_dd60:
                force_exit = True
        if exit_ks200_ema200 and d in ks200.index:
            ab = ks200.loc[d, 'above_ema200']
            if ab == False:
                # consecutive 이탈 체크
                consec = 0
                for j in range(i, max(0, i-exit_grace_days), -1):
                    dj = all_dates[j]
                    if dj in ks200.index and ks200.loc[dj, 'above_ema200'] == False:
                        consec += 1
                if consec >= exit_grace_days:
                    force_exit = True

        # Exit state machine
        if force_exit and not in_exit:
            in_exit = True; exit_entry_idx = i
        if in_exit:
            ds_e = i - exit_entry_idx
            recovered = (d in ks200.index and ks200.loc[d, 'above_ema200'] == True)
            if ds_e >= EXIT_REENTRY_DAYS and recovered:
                in_exit = False

        # ──── C. Dynamic Leverage Scale ────
        if lev_scale_ks_dd and d in ks200.index:
            mdd = ks200.loc[d, 'dd_60d']
            if not pd.isna(mdd):
                for thr, mult in sorted(lev_scale_ks_dd.items()):
                    if mdd <= thr:
                        lev *= mult; break
        if lev_scale_proxy and not pd.isna(v_proxy):
            for thr, mult in sorted(lev_scale_proxy.items(), reverse=True):
                if v_proxy >= thr:
                    lev *= mult; break

        # ──── D. Tranche progression ────
        if next_tranche < len(tranche_pcts):
            target_pct, trigger = tranche_triggers[next_tranche]
            triggered = False
            if trigger == 'now' and next_tranche == 0 and i >= 252:
                triggered = True
            elif trigger == 'ks_dd' and d in ks200.index:
                mdd = ks200.loc[d, 'dd_60d']
                if not pd.isna(mdd) and mdd <= target_pct:
                    triggered = True
            elif trigger == 'proxy' and not pd.isna(v_proxy):
                if v_proxy >= target_pct:
                    triggered = True
            if triggered:
                deployed_pct += tranche_pcts[next_tranche]
                next_tranche += 1

        # ──── PnL ────
        if in_exit or lev == 0:
            # 강제 cash/parking
            net = get_parking_ret(d, prev_d, parking, parking_data)
        elif holdings:
            pr = sum(w * (feat_data[c]['ret'].get(d, 0) or 0)
                     for c, w in holdings.items() if c in feat_data)
            cost = max(0, lev-1) * KR_BORROW_DAILY
            # Tranche-adjusted: deployed_pct만 strategy 노출
            invested_ret = (lev * pr - cost) * deployed_pct
            parking_ret = get_parking_ret(d, prev_d, parking, parking_data) * (1 - deployed_pct)
            net = invested_ret + parking_ret
        else:
            net = get_parking_ret(d, prev_d, parking, parking_data)
        val *= (1 + net); val = max(val, 0.01)
        peak = max(peak, val)
        vals.append(val); dates_out.append(d)

        # ──── Rebal (V25 logic) ────
        if last_rebal is None or (d - last_rebal).days >= rebal_days:
            if lev > 0 and not in_exit:
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
                        scored.append((c, score))
                scored.sort(key=lambda x: -x[1])
                scored = apply_sector_cap(scored, max_per_sector=sector_cap)
                top = scored[:top_k]
                new_h = {c: 1.0/len(top) for c, _ in top} if top else {}
                changed = sum(abs(new_h.get(c, 0) - holdings.get(c, 0))
                              for c in set(list(holdings.keys()) + list(new_h.keys())))
                val *= (1 - KR_TAX * changed/2)
                holdings = new_h; last_rebal = d
            elif (lev == 0 or in_exit) and holdings:
                changed = sum(abs(w) for w in holdings.values())
                val *= (1 - KR_TAX * changed/2)
                holdings = {}; last_rebal = d
        prev_d = d

    return pd.Series(vals, index=dates_out)


def metrics(s, label=''):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    total = s.iloc[-1] - 1
    cagr = s.iloc[-1] ** (1/yrs) - 1 if yrs > 0 else 0
    rets = s.pct_change().dropna()
    sh = rets.mean()/rets.std() * (252**0.5) if rets.std() > 0 else 0
    dd = (s/s.cummax()-1).min()
    calmar = cagr / abs(dd) if dd < 0 else 0
    if label:
        print(f'  {label:55s} total={total*100:>+9.1f}%  CAGR={cagr*100:>+5.1f}%  '
              f'Sh={sh:.2f}  DD={dd*100:>+5.1f}%  Calmar={calmar:.2f}')
    return {'total':total,'cagr':cagr,'sharpe':sh,'dd':dd,'calmar':calmar}


if __name__ == '__main__':
    t0 = time.time()
    print('=' * 100)
    print('KR v27 — US G5_22 sc_v48 enhancement 4종 한국 적용 검증')
    print('=' * 100)
    print('\nLoading data...')
    data = fetch_all_extended('2014-03-04')
    macro = load_macro('2014-01-01')
    print('Loading parking data...')
    parking_data = load_parking_data('2014-01-01')
    ks = data['KS200']
    s_bh = ks['close'] / ks['close'].iloc[0]

    # Baseline V25
    print('\n=== Baseline V25 ===')
    s_v25 = sim_v27(data, macro, parking_data)
    m_v25 = metrics(s_v25, 'V25 baseline')

    # Phase A: Early Exit (한국 mean-reversion이라 likely worse 예상)
    print('\n[Phase A] Early Exit Signals')
    print('-' * 100)
    sA = [('V25 baseline', s_v25)]
    print(f'  {"variant":<55} {"total%":<10} {"CAGR%":<8} {"Sh":<5} {"DD%":<7} {"Calmar":<7}')
    metrics(s_v25, 'A0: V25 baseline (no exit)')
    s = sim_v27(data, macro, parking_data, exit_ks200_dd60=-0.10)
    metrics(s, 'A1: KS200 60d DD<-10% exit')
    s = sim_v27(data, macro, parking_data, exit_ks200_dd60=-0.15)
    metrics(s, 'A2: KS200 60d DD<-15% exit')
    s = sim_v27(data, macro, parking_data, exit_ks200_dd60=-0.20)
    metrics(s, 'A3: KS200 60d DD<-20% exit')
    s = sim_v27(data, macro, parking_data, exit_ks200_ema200=True, exit_grace_days=3)
    metrics(s, 'A4: KS200 EMA200 이탈 (3d grace)')
    s = sim_v27(data, macro, parking_data, exit_ks200_ema200=True, exit_grace_days=10)
    metrics(s, 'A5: KS200 EMA200 이탈 (10d grace)')

    # Phase B: Cash Parking
    print('\n[Phase B] Cash Parking')
    print('-' * 100)
    metrics(s_v25, 'B0: V25 baseline (KOFR cash)')
    for park in ['GLD', 'TLT', 'TBILL', 'BONDMIX']:
        s = sim_v27(data, macro, parking_data, parking=park)
        metrics(s, f'B: parking={park}')

    # Phase C: Dynamic Leverage Scale
    print('\n[Phase C] Dynamic Leverage Scale')
    print('-' * 100)
    metrics(s_v25, 'C0: V25 baseline')
    s = sim_v27(data, macro, parking_data, lev_scale_ks_dd={-0.05: 0.7})
    metrics(s, 'C1: KS200 DD<-5% lev×0.7')
    s = sim_v27(data, macro, parking_data, lev_scale_ks_dd={-0.05: 0.85, -0.10: 0.5})
    metrics(s, 'C2: DD<-5% ×0.85, <-10% ×0.5')
    s = sim_v27(data, macro, parking_data, lev_scale_proxy={30: 0.5})
    metrics(s, 'C3: proxy>30 lev×0.5')
    s = sim_v27(data, macro, parking_data,
                  lev_scale_ks_dd={-0.05: 0.85, -0.10: 0.5},
                  lev_scale_proxy={30: 0.5})
    metrics(s, 'C4: DD + proxy combined')

    # Phase D: Tranche Entry
    print('\n[Phase D] Tranche Entry (분할 매수)')
    print('-' * 100)
    metrics(s_v25, 'D0: V25 baseline (full immediate)')
    s = sim_v27(data, macro, parking_data,
                  tranche_pcts=[0.4, 0.3, 0.3],
                  tranche_triggers=[(0, 'now'), (-0.03, 'ks_dd'), (-0.07, 'ks_dd')])
    metrics(s, 'D1: 40/30/30 KS DD -3/-7')
    s = sim_v27(data, macro, parking_data,
                  tranche_pcts=[0.5, 0.25, 0.25],
                  tranche_triggers=[(0, 'now'), (-0.05, 'ks_dd'), (-0.10, 'ks_dd')])
    metrics(s, 'D2: 50/25/25 KS DD -5/-10')
    s = sim_v27(data, macro, parking_data,
                  tranche_pcts=[0.33, 0.33, 0.34],
                  tranche_triggers=[(0, 'now'), (22.5, 'proxy'), (30, 'proxy')])
    metrics(s, 'D3: 33/33/34 proxy 22.5/30')

    print(f'\n실행 시간: {time.time()-t0:.1f}s')

"""KR v12 — 모든 개선 통합한 최종 Champion.

통합 요소:
    [기존 v09 Champion]
    - Zone framework (CASH<15 / 1x / 1.5x / 2x>30)
    - Top-7 picking
    - Monthly rebal
    - DD throttle (단계화 — v12에서 다단계로 업그레이드)

    [v11 신규]
    1. Mom ensemble (mom90+120+150 평균)
    2. Sector concentration cap (동일 sector 최대 3종)
    3. Macro gate (USDKRW + ^VIX → normal/caution/crisis)
    4. DD multi-stage (-30/-45/-55/-65)
    5. Point-in-time universe (분기별 거래량 top 50)

비교:
    - V9 Champion (baseline)
    - V12 Full (모든 개선)
    - V12 Ablation: 개선 1개씩 제거해서 기여도 측정
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import time

from kr_v02_sector_panic_buy import (
    compute_regime, KR_BORROW_DAILY, KR_TAX, CASH_DAILY,
)
from kr_v07_universe_comparison import INDIVIDUAL_STOCKS
from kr_v11_enhanced_modules import (
    SECTOR_MAP, apply_sector_cap, load_macro, macro_gate,
    dd_multistage_lev, mom_ensemble_score, pit_universe,
)


# ============================================================
# Features (mom 90, 120, 150 추가)
# ============================================================
def add_features_v12(df):
    df = df.copy()
    df['ret'] = df['close'].pct_change()
    df['ret_60d'] = df['close'].pct_change(60)
    df['ret_90d'] = df['close'].pct_change(90)
    df['ret_120d'] = df['close'].pct_change(120)
    df['ret_150d'] = df['close'].pct_change(150)
    df['ema50'] = df['close'].ewm(span=50).mean()
    df['ema200'] = df['close'].ewm(span=200).mean()
    df['above_50'] = df['close'] > df['ema50']
    df['above_200'] = df['close'] > df['ema200']
    df['below_ema200'] = df['close'] < df['ema200']
    return df


# ============================================================
# Zone (기존)
# ============================================================
def zone_base_lev(vkospi_proxy):
    if pd.isna(vkospi_proxy): return 1.0
    if vkospi_proxy < 15: return 0
    if vkospi_proxy < 22.5: return 1.0
    if vkospi_proxy < 30: return 1.5
    return 2.0


# ============================================================
# 데이터 로드 (universe 확장: 100종까지)
# ============================================================
EXTENDED_UNIVERSE = INDIVIDUAL_STOCKS + [
    # 추가 시총 50-100 후보 (PIT universe 확장)
    '402340', '267260', '034730', '004020', '008770', '003490', '011780',
    '030000', '034220', '180640', '241560', '161390', '267250', '028050',
    '139480', '003550', '003410', '000810', '329180', '047040', '009830',
    '004990', '010620', '267290', '375500', '079550', '108860', '042700',
    '294870', '008560', '012510', '267260', '023530', '001040', '028670',
    '005250', '267260', '001230', '015760', '267260',  # duplicates ok
]
EXTENDED_UNIVERSE = list(set(EXTENDED_UNIVERSE))


def fetch_all_extended(start='2014-03-04'):
    data = {}
    print(f'Loading {len(EXTENDED_UNIVERSE)} stocks...')
    for code in EXTENDED_UNIVERSE:
        try:
            df = fdr.DataReader(code, start)
            df.columns = [c.lower() for c in df.columns]
            if len(df) > 60:
                data[code] = df
        except: pass
        time.sleep(0.03)
    ks = fdr.DataReader('KS200', start)
    ks.columns = [c.lower() for c in ks.columns]
    data['KS200'] = ks
    print(f'Loaded: {len(data)-1} stocks + KS200')
    return data


# ============================================================
# V12 Full simulator (모든 개선 통합)
# ============================================================
def simulate_v12(data, macro,
                  top_k=7,
                  use_pit_universe=True,
                  use_mom_ensemble=True,
                  use_sector_cap=True,
                  use_macro_gate=True,
                  use_dd_multistage=True,
                  pit_n=50,
                  sector_cap_max=3,
                  rebal_days=21):
    """완전 통합 시뮬레이터.

    개별 옵션 toggle로 ablation 분석 가능.
    """
    ks200 = compute_regime(data['KS200'])
    ks200 = add_features_v12(ks200)
    # 모든 종목 feature 미리 컴퓨팅
    feat_data = {c: add_features_v12(data[c])
                  for c in data if c not in ('KS200',)}
    all_dates = sorted(ks200.index)

    val = 1.0; vals = []; dates_out = []
    holdings = {}; last_rebal = None
    peak = 1.0
    pit_cache = {}  # quarter → universe

    for i, d in enumerate(all_dates):
        if i < 252:
            vals.append(1.0); dates_out.append(d); continue

        # Volatility zone
        v_proxy = ks200['vkospi_prev'].get(d, None)
        base_lev = zone_base_lev(v_proxy)

        # Macro gate
        if use_macro_gate and d in macro.index:
            ks_row = ks200.loc[d] if d in ks200.index else None
            gate = macro_gate(macro.loc[d], ks_row)
            if gate == 'crisis':
                lev = 0
            elif gate == 'caution':
                lev = min(base_lev, 1.0)  # max 1x
            else:
                lev = base_lev
        else:
            lev = base_lev

        # DD multi-stage
        cur_dd = val / peak - 1
        if use_dd_multistage:
            lev = dd_multistage_lev(lev, cur_dd)
        else:
            # 기존 단일 throttle
            if lev > 1 and cur_dd < -0.50:
                lev = max(1.0, lev * 0.5)

        # PnL
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

        # Rebal monthly
        if last_rebal is None or (d - last_rebal).days >= rebal_days:
            if lev > 0:
                # Universe selection
                if use_pit_universe:
                    quarter_key = (d.year, (d.month - 1) // 3)
                    if quarter_key not in pit_cache:
                        pit_cache[quarter_key] = pit_universe(data, d, n=pit_n, lookback_days=60)
                    universe = pit_cache[quarter_key]
                else:
                    universe = INDIVIDUAL_STOCKS

                # Scoring
                scored = []
                for c in universe:
                    if c in feat_data and d in feat_data[c].index:
                        row = feat_data[c].loc[d]
                        if use_mom_ensemble:
                            sc = mom_ensemble_score(row, lookbacks=(90, 120, 150))
                        else:
                            sc = row.get('ret_120d', None)
                        if sc is not None and not pd.isna(sc):
                            scored.append((c, sc))
                scored.sort(key=lambda x: -x[1])

                # Sector cap
                if use_sector_cap:
                    scored = apply_sector_cap(scored, max_per_sector=sector_cap_max)

                top = scored[:top_k]
                new_holdings = {c: 1.0/len(top) for c, _ in top} if top else {}
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


def metrics(s, label=''):
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    total = s.iloc[-1] - 1
    cagr = s.iloc[-1] ** (1/yrs) - 1 if yrs > 0 else 0
    rets = s.pct_change().dropna()
    sharpe = rets.mean()/rets.std() * (252**0.5) if rets.std() > 0 else 0
    dd = (s/s.cummax()-1).min()
    sortino = rets.mean() / rets[rets<0].std() * (252**0.5) if (rets<0).any() and rets[rets<0].std() > 0 else 0
    calmar = cagr / abs(dd) if dd < 0 else 0
    if label:
        print(f'  {label:60s}  total={total*100:>9.1f}%  CAGR={cagr*100:>5.1f}%  '
              f'Sh={sharpe:.2f}  Sortino={sortino:.2f}  DD={dd*100:>5.1f}%  Calmar={calmar:.2f}')
    return {'total':total,'cagr':cagr,'sharpe':sharpe,'dd':dd,'sortino':sortino,'calmar':calmar}


def window_alpha(s_strategy, s_baseline, n=6):
    all_dates = sorted(s_strategy.index)
    win_len = len(all_dates) // n
    rows = []
    for k in range(n):
        s_i, e_i = k*win_len, min((k+1)*win_len, len(all_dates))
        wd = all_dates[s_i:e_i]
        if len(wd) < 100: continue
        a = s_strategy.loc[wd[0]]; b = s_strategy.loc[wd[-1]]
        c = s_baseline.loc[wd[0]] if wd[0] in s_baseline.index else None
        d_ = s_baseline.loc[wd[-1]] if wd[-1] in s_baseline.index else None
        if c is None or d_ is None: continue
        strat = b/a - 1
        bh = d_/c - 1
        sub = s_strategy.loc[wd[0]:wd[-1]].pct_change().dropna()
        sh = sub.mean()/sub.std() * (252**0.5) if sub.std() > 0 else 0
        dd = (s_strategy.loc[wd[0]:wd[-1]] / s_strategy.loc[wd[0]:wd[-1]].cummax() - 1).min()
        rows.append({
            'win': k+1, 'period': f'{wd[0].date()}~{wd[-1].date()}',
            'bh%': round(bh*100,1), 'strat%': round(strat*100,1),
            'alpha_pp': round((strat-bh)*100,1),
            'strat_sh': round(sh,2), 'dd%': round(dd*100,1),
        })
    return pd.DataFrame(rows)


if __name__ == '__main__':
    print('=' * 90)
    print('KR v12 — 통합 Champion (5개 개선 모두 적용)')
    print('=' * 90)

    print('\nLoading...')
    data = fetch_all_extended('2014-03-04')
    macro = load_macro('2014-01-01')

    ks = data['KS200']
    s_bh = ks['close'] / ks['close'].iloc[0]

    # ---- V9 Champion (baseline 재현) ----
    print('\n=== V9 Champion (baseline, no improvements) ===')
    s_v9 = simulate_v12(data, macro,
                          use_pit_universe=False, use_mom_ensemble=False,
                          use_sector_cap=False, use_macro_gate=False,
                          use_dd_multistage=False)
    metrics(s_v9, 'V9 Champion (top-7 mom120, fixed universe)')

    # ---- V12 Full (all improvements) ----
    print('\n=== V12 Full (모든 개선) ===')
    s_v12 = simulate_v12(data, macro,
                          use_pit_universe=True, use_mom_ensemble=True,
                          use_sector_cap=True, use_macro_gate=True,
                          use_dd_multistage=True)
    metrics(s_v12, 'V12 Full (PIT+ensemble+sectorCap+macroGate+ddMulti)')

    # ---- Ablation (각 개선 1개씩 제거) ----
    print('\n=== V12 Ablation (개선 1개씩 제거) ===')
    for off_name, off_kwargs in [
        ('w/o PIT universe',   {'use_pit_universe': False}),
        ('w/o mom ensemble',   {'use_mom_ensemble': False}),
        ('w/o sector cap',     {'use_sector_cap': False}),
        ('w/o macro gate',     {'use_macro_gate': False}),
        ('w/o DD multi-stage', {'use_dd_multistage': False}),
    ]:
        full_kwargs = {'use_pit_universe': True, 'use_mom_ensemble': True,
                        'use_sector_cap': True, 'use_macro_gate': True,
                        'use_dd_multistage': True}
        full_kwargs.update(off_kwargs)
        s = simulate_v12(data, macro, **full_kwargs)
        metrics(s, f'V12 {off_name}')

    # ---- Walk-forward ----
    print('\n=== Walk-forward 6 windows: V12 Full vs V9 vs BH ===')
    print('\nBH KS200:')
    print(window_alpha(s_bh, s_bh, n=6).to_string(index=False))
    print('\nV9 Champion:')
    wf_v9 = window_alpha(s_v9, s_bh, n=6)
    print(wf_v9.to_string(index=False))
    print(f'V9 alpha {(wf_v9["alpha_pp"]>0).sum()}/{len(wf_v9)}, mean={wf_v9["alpha_pp"].mean():.1f}pp')
    print('\nV12 Full:')
    wf_v12 = window_alpha(s_v12, s_bh, n=6)
    print(wf_v12.to_string(index=False))
    print(f'V12 alpha {(wf_v12["alpha_pp"]>0).sum()}/{len(wf_v12)}, mean={wf_v12["alpha_pp"].mean():.1f}pp')

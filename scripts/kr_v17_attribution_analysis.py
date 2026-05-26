"""KR v17 — 종목별 alpha attribution + DD 원인 분석.

질문:
    Q1. 12.2년간 어떤 종목이 가장 많이 picking 됐나?
    Q2. 어떤 종목이 가장 큰 PnL contribution 했나?
    Q3. 가장 큰 DD 5개 시점, 그 때 어떤 종목이 손해 만들었나?
    Q4. Sector별 평균 hold ratio + contribution?
    Q5. Win/Loss month 비율, monthly 분포?
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import numpy as np
from collections import Counter
from kr_v02_sector_panic_buy import compute_regime, KR_BORROW_DAILY, KR_TAX, CASH_DAILY
from kr_v11_enhanced_modules import (
    SECTOR_MAP, apply_sector_cap, load_macro, macro_gate,
    dd_multistage_lev, mom_ensemble_score, pit_universe,
)
from kr_v12_integrated_champion import (
    add_features_v12, zone_base_lev, fetch_all_extended,
)


# 종목명 매핑 (50종)
STOCK_NAMES = {
    '005930': '삼성전자', '000660': 'SK하이닉스', '373220': 'LG에너지솔루션',
    '207940': '삼성바이오로직스', '005380': '현대차', '012450': '한화에어로스페이스',
    '329180': 'HD현대중공업', '000270': '기아', '068270': '셀트리온',
    '035420': 'NAVER', '005490': 'POSCO홀딩스', '051910': 'LG화학',
    '028260': '삼성물산', '006400': '삼성SDI', '035720': '카카오',
    '105560': 'KB금융', '055550': '신한지주', '086790': '하나금융지주',
    '096770': 'SK이노베이션', '017670': 'SK텔레콤', '033780': 'KT&G',
    '015760': '한국전력', '009540': 'HD한국조선해양', '010130': '고려아연',
    '316140': '우리금융지주', '024110': '기업은행', '011200': 'HMM',
    '042660': '한화오션', '047810': '한국항공우주', '010140': '삼성중공업',
    '259960': '크래프톤', '011170': '롯데케미칼', '010950': 'S-Oil',
    '018260': '삼성에스디에스', '030200': 'KT', '051900': 'LG생활건강',
    '032830': '삼성생명', '086280': '현대글로비스', '003670': '포스코퓨처엠',
    '011070': 'LG이노텍', '009150': '삼성전기', '012330': '현대모비스',
    '035250': '강원랜드', '066570': 'LG전자', '071050': '한국금융지주',
    '097950': 'CJ제일제당', '006800': '미래에셋증권', '128940': '한미약품',
    '017800': '현대엘리베이', '251270': '넷마블',
}


def simulate_with_attribution(data, macro, top_k=7, sector_cap=3, rebal_days=21):
    """V12 시뮬레이션 + 종목별 detailed log."""
    ks200 = compute_regime(data['KS200'])
    ks200 = add_features_v12(ks200)
    feat_data = {c: add_features_v12(data[c])
                  for c in data if c != 'KS200'}
    all_dates = sorted(ks200.index)

    val = 1.0; vals = []; dates_out = []
    holdings = {}; last_rebal = None; peak = 1.0; pit_cache = {}

    # Detailed logs
    rebal_log = []           # 매 rebal picking
    daily_attribution = []   # 매일 종목별 contribution
    dd_events = []           # DD 깊은 시점

    for i, d in enumerate(all_dates):
        if i < 252:
            vals.append(1.0); dates_out.append(d); continue
        v = ks200['vkospi_prev'].get(d, None)
        base_lev = zone_base_lev(v)
        if d in macro.index:
            ks_row = ks200.loc[d] if d in ks200.index else None
            g = macro_gate(macro.loc[d], ks_row)
            if g == 'crisis': base_lev = 0
            elif g == 'caution': base_lev = min(base_lev, 1.0)
        cur_dd = val/peak - 1
        lev = dd_multistage_lev(base_lev, cur_dd)

        # Daily PnL + per-stock attribution
        stock_pnls = {}
        if lev > 0 and holdings:
            pr = 0
            for c, w in holdings.items():
                if c in feat_data:
                    r = feat_data[c]['ret'].get(d, 0) or 0
                    contribution = w * r
                    stock_pnls[c] = contribution
                    pr += contribution
            cost = max(0, lev-1) * KR_BORROW_DAILY
            net = lev * pr - cost
        else:
            net = CASH_DAILY if lev == 0 else 0
        val *= (1 + net); val = max(val, 0.01)
        peak = max(peak, val)
        vals.append(val); dates_out.append(d)
        daily_attribution.append({'d': d, 'val': val, 'lev': lev, 'pnls': stock_pnls})

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
                        sc = mom_ensemble_score(row, lookbacks=(90,120,150))
                        if sc is not None and not pd.isna(sc):
                            scored.append((c, sc))
                scored.sort(key=lambda x: -x[1])
                scored = apply_sector_cap(scored, max_per_sector=sector_cap)
                top = scored[:top_k]
                new_h = {c: 1.0/len(top) for c, _ in top} if top else {}
                changed = sum(abs(new_h.get(c, 0) - holdings.get(c, 0))
                              for c in set(list(holdings.keys()) + list(new_h.keys())))
                val *= (1 - KR_TAX * changed/2)
                rebal_log.append({'d': d, 'picks': [c for c,_ in top]})
                holdings = new_h; last_rebal = d
            elif lev == 0 and holdings:
                changed = sum(abs(w) for w in holdings.values())
                val *= (1 - KR_TAX * changed/2)
                holdings = {}; last_rebal = d

    s = pd.Series(vals, index=dates_out)
    return s, rebal_log, daily_attribution


if __name__ == '__main__':
    print('=' * 90)
    print('KR v17 — 종목별 Alpha Attribution + DD 분석')
    print('=' * 90)
    print('\nLoading...')
    data = fetch_all_extended('2014-03-04')
    macro = load_macro('2014-01-01')

    s, rebal_log, daily_att = simulate_with_attribution(data, macro)
    print(f'\nTotal value: {s.iloc[-1]:.1f}x ({(s.iloc[-1]-1)*100:.0f}%)')

    # ============================================================
    # Q1. 종목별 picking 빈도
    # ============================================================
    print('\n=== Q1. 종목별 picking 빈도 (총 rebal 횟수 대비) ===')
    n_rebal = len(rebal_log)
    pick_counts = Counter()
    for r in rebal_log:
        for c in r['picks']:
            pick_counts[c] += 1
    sorted_picks = sorted(pick_counts.items(), key=lambda x: -x[1])
    print(f'Total rebals: {n_rebal}')
    print(f'\n{"Code":<8} {"Name":<25} {"Sector":<12} {"Picks":<7} {"%":<6}')
    for c, n in sorted_picks[:20]:
        name = STOCK_NAMES.get(c, '?')
        sec = SECTOR_MAP.get(c, '?')
        print(f'{c:<8} {name:<25} {sec:<12} {n:>5}  {n/n_rebal*100:>4.1f}%')

    # ============================================================
    # Q2. 종목별 PnL contribution
    # ============================================================
    print('\n=== Q2. 종목별 누적 PnL contribution ===')
    stock_contribution = Counter()
    for entry in daily_att:
        for c, pnl in entry['pnls'].items():
            stock_contribution[c] += entry['lev'] * pnl  # lev-adjusted
    sorted_contrib = sorted(stock_contribution.items(), key=lambda x: -x[1])
    print(f'\n{"Code":<8} {"Name":<25} {"Sector":<12} {"Contrib %":<12}')
    print('Top 15 (양수):')
    for c, contrib in sorted_contrib[:15]:
        name = STOCK_NAMES.get(c, '?')
        sec = SECTOR_MAP.get(c, '?')
        print(f'{c:<8} {name:<25} {sec:<12} {contrib*100:>+10.1f}%')
    print('\nBottom 10 (손실):')
    for c, contrib in sorted_contrib[-10:]:
        name = STOCK_NAMES.get(c, '?')
        sec = SECTOR_MAP.get(c, '?')
        print(f'{c:<8} {name:<25} {sec:<12} {contrib*100:>+10.1f}%')

    # ============================================================
    # Q3. DD 깊은 시점 attribution
    # ============================================================
    print('\n=== Q3. Top 5 DD periods + 손실 만든 종목 ===')
    rolling_peak = s.cummax()
    rolling_dd = s / rolling_peak - 1
    # Find DD bottoms (local minima)
    dd_threshold = -0.20
    dd_events = []
    for i in range(1, len(rolling_dd)-1):
        if rolling_dd.iloc[i] < dd_threshold:
            # is local min if smaller than ±5 days
            window = rolling_dd.iloc[max(0,i-5):min(len(rolling_dd),i+5)]
            if rolling_dd.iloc[i] <= window.min():
                dd_events.append((rolling_dd.index[i], rolling_dd.iloc[i]))
    # Deduplicate within 30 days
    dedup = []
    last_d = None
    for d, dd_val in sorted(dd_events, key=lambda x: x[1]):
        if last_d is None or abs((d - last_d).days) > 30:
            dedup.append((d, dd_val))
            last_d = d
    dedup = dedup[:5]
    print(f'\nTop 5 DD bottoms:')
    for d, dd_val in dedup:
        # 30일 prior - 30일 후 attribution
        cutoff_start = d - pd.Timedelta(days=30)
        period_contrib = Counter()
        for entry in daily_att:
            if cutoff_start <= entry['d'] <= d:
                for c, pnl in entry['pnls'].items():
                    period_contrib[c] += entry['lev'] * pnl
        worst = sorted(period_contrib.items(), key=lambda x: x[1])[:5]
        print(f'\n  DD bottom {d.date()}: dd={dd_val*100:.1f}%')
        print(f'    30일간 손실 만든 종목:')
        for c, contrib in worst:
            name = STOCK_NAMES.get(c, '?')
            sec = SECTOR_MAP.get(c, '?')
            print(f'      {name:<20} ({sec}): {contrib*100:>+6.2f}%')

    # ============================================================
    # Q4. Sector별 contribution
    # ============================================================
    print('\n=== Q4. Sector별 누적 contribution ===')
    sector_contrib = Counter()
    for c, contrib in stock_contribution.items():
        sec = SECTOR_MAP.get(c, '?')
        sector_contrib[sec] += contrib
    sorted_sec = sorted(sector_contrib.items(), key=lambda x: -x[1])
    print(f'\n{"Sector":<12} {"Contrib%":<10}')
    for sec, contrib in sorted_sec:
        print(f'{sec:<12} {contrib*100:>+10.1f}%')

    # ============================================================
    # Q5. Monthly 분포
    # ============================================================
    print('\n=== Q5. 월간 수익률 분포 ===')
    monthly = s.resample('M').last().pct_change().dropna()
    print(f'  Total months: {len(monthly)}')
    print(f'  Win months: {(monthly > 0).sum()} ({(monthly > 0).mean()*100:.1f}%)')
    print(f'  Mean: {monthly.mean()*100:+.2f}%/mo')
    print(f'  Median: {monthly.median()*100:+.2f}%/mo')
    print(f'  Std: {monthly.std()*100:.2f}%')
    print(f'  Best month: {monthly.max()*100:+.1f}% ({monthly.idxmax().date()})')
    print(f'  Worst month: {monthly.min()*100:+.1f}% ({monthly.idxmin().date()})')
    # Percentile
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f'  p{p}: {monthly.quantile(p/100)*100:+.2f}%/mo')

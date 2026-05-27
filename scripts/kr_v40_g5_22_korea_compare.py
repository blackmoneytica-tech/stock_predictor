"""KR v40 — 현 한국 전략 (V25-full + H-B) vs G5_22 한국 적용판 비교 (최근 2년).

V25-full (현 Champion):
    KOSPI200 top 50 universe → mom120 + zone-dep squeeze + 52w_low → Top-7 종목 직접
    H-B 20/3/34 portfolio exit

G5_22 한국 적용판:
    섹터 그룹 (SECTOR_MAP) = ETF 대용 (equal-weight 섹터 지수)
    섹터 SC+L2 score → Top 3 섹터 rotation (min_hold 5d, rank>=5 트리거)
    각 섹터 M2 최고 leader 종목 직접 매수
    → 3 종목 (섹터당 1 leader)

공정 비교: 레버리지/매크로/DD throttle은 둘 다 동일 (한국 compute_zone + macro_gate + dd_multistage).
           picking 방식만 차이.

⚠️ Look-ahead 금지 — 모든 신호 전일 정보. zone은 vkospi_prev (이미 shift).
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
from kr_v18_operational_system import SECTOR_MAP
from kr_v34_topk_weight_exit import sim_flex, m


# G5_22 한국 적용: 섹터 그룹 (종목 2개 이상인 섹터만 — leader 선택 의미)
def build_sector_groups(data, min_stocks=2):
    groups = {}
    for c in data:
        if c == 'KS200': continue
        s = SECTOR_MAP.get(c, 'Other')
        groups.setdefault(s, []).append(c)
    # min_stocks 이상 + Other 제외 (섹터 불명확)
    return {s: cs for s, cs in groups.items() if len(cs) >= min_stocks and s != 'Other'}


def add_g5_indicators(df):
    """G5_22 신호용 추가 indicator: ema50/200, ddPct(60d), v_5d_ratio, volRatio."""
    df = df.copy()
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()
    df['ab50'] = df['close'] > df['ema50']
    df['ab200'] = df['close'] > df['ema200']
    high60 = df['close'].rolling(60, min_periods=20).max()
    df['ddPct'] = (df['close'] / high60 - 1) * 100
    # v_5d_ratio: 5일 평균 거래대금 / 60일 평균
    df['v_5d_ratio'] = df['dv_5d_avg'] / df['dv_60d_avg'].replace(0, np.nan)
    df['volRatio'] = df['dv_20d_ratio'] if 'dv_20d_ratio' in df.columns else 1.0
    return df


def build_sector_index(sector_codes, feat):
    """섹터 ETF 대용: equal-weight 섹터 종목 일별 수익률 → 지수 가격 시계열 + 신호."""
    rets = []
    for c in sector_codes:
        if c in feat:
            rets.append(feat[c]['ret'])
    if not rets:
        return None
    ret_df = pd.concat(rets, axis=1)
    sec_ret = ret_df.mean(axis=1)   # equal-weight 섹터 일일 수익률
    sec_price = (1 + sec_ret.fillna(0)).cumprod()
    # 섹터 지수에 G5 신호 계산
    idx = pd.DataFrame({'close': sec_price})
    idx['chg5d'] = idx['close'].pct_change(5) * 100
    idx['chg20d'] = idx['close'].pct_change(20) * 100
    idx['chg60d'] = idx['close'].pct_change(60) * 100
    high60 = idx['close'].rolling(60, min_periods=20).max()
    idx['ddPct'] = (idx['close'] / high60 - 1) * 100
    # 거래량 ratio: 섹터 종목 dv 합산
    dvs = [feat[c]['dv'] for c in sector_codes if c in feat]
    dv_sum = pd.concat(dvs, axis=1).sum(axis=1)
    idx['v_5d_ratio'] = dv_sum.rolling(5).mean() / dv_sum.rolling(60).mean().replace(0, np.nan)
    idx['volRatio'] = dv_sum / dv_sum.rolling(20).mean().replace(0, np.nan)
    idx['ab50'] = idx['close'] > idx['close'].ewm(span=50, adjust=False).mean()
    return idx


def sc_base_kr(row):
    """G5_22 scBase 포팅 (섹터 지수 row 기준)."""
    if pd.isna(row.get('chg20d')) or pd.isna(row.get('chg60d')): return None
    sa = row['chg20d'] + row['chg60d'] / 2 + ((row.get('volRatio', 1) or 1) - 1) * 10
    v5r = row.get('v_5d_ratio', 1) or 1
    c5 = row.get('chg5d', 0) or 0
    sb = 0
    if v5r > 1.3: sb += 3
    elif v5r > 1.1: sb += 1
    if 0 < c5 <= 8: sb += 2
    if row.get('ab50'): sb += 1
    return sa + sb * 2


def is_l2_kr(row):
    dd = row.get('ddPct')
    v5r = row.get('v_5d_ratio')
    if pd.isna(dd) or pd.isna(v5r): return False
    return dd <= -15 and v5r >= 1.1


def score_sc_l2_kr(row, boost=20):
    base = sc_base_kr(row)
    if base is None: return None
    return base + boost if is_l2_kr(row) else base


def score_m2_kr(row, sector_dd, ks_c5):
    """leader picking M2 (개별 종목 feat row)."""
    c5 = row.get('ret_5d')
    if pd.isna(c5): return -99
    c5 *= 100
    s = 0
    if c5 > 5: s += 2
    elif c5 > 0: s += 1
    elif c5 < -2: s -= 1
    dd = row.get('ddPct')
    if not pd.isna(dd) and dd > sector_dd + 5: s += 3
    if row.get('ab50'): s += 2
    if row.get('ab200'): s += 1
    if c5 - ks_c5 > 0: s += 2
    vr = row.get('volRatio')
    if not pd.isna(vr) and vr > 1.2: s += 1
    return s


def sim_g5_korea(data, macro, top_sectors=3, min_hold=5, rotate_rank=5,
                  leaders_per_sector=1,
                  exit_mode='none', exit_ret_thr=0.20, exit_dv_thr=3.0,
                  exit_pct=0.34, exit_cooldown=5):
    """G5_22 한국 적용판 시뮬레이터.

    섹터 SC+L2 rotation → 섹터 leader (M2 최고) 직접 매수.
    레버리지/매크로/DD throttle은 V25-full과 동일 (compute_zone 기반).
    exit_mode='hb_portfolio': 보유 leader 중 전일 ret>thr AND dv_spike>thr → portfolio exit_pct 매도.
    leaders_per_sector: 섹터당 leader 종목 수 (1=M2 최고만, 2=상위 2개).
    """
    ks200 = compute_regime(data['KS200'])
    feat = {c: add_g5_indicators(add_features_v30(data[c])) for c in data if c != 'KS200'}
    sector_groups = build_sector_groups(data)
    sector_idx = {s: build_sector_index(cs, feat) for s, cs in sector_groups.items()}
    sector_idx = {s: idx for s, idx in sector_idx.items() if idx is not None}
    all_dates = sorted(ks200.index)

    ks_feat = add_features_v30(data['KS200'])

    val = 1.0; vals = []; dates_out = []
    holdings = {}            # {leader_code: weight}
    held_sectors = []        # 현재 보유 섹터
    sector_entry_idx = {}    # {sector: entry index}
    last_rebal_idx = -999
    last_exit_idx = -999
    deployed_pct = 1.0
    peak = 1.0

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

        # H-B exit (보유 leader 전일 ret>thr AND dv_spike>thr → portfolio 매도)
        if exit_mode == 'hb_portfolio' and holdings and (i - last_exit_idx) > exit_cooldown:
            prev_date = all_dates[i-1]
            triggered = False
            for c in list(holdings.keys()):
                if c not in feat or prev_date not in feat[c].index: continue
                row = feat[c].loc[prev_date]
                r_prev = row.get('ret', 0) or 0
                dv_spike = row.get('dv_spike', 1.0) or 1.0
                if not pd.isna(r_prev) and r_prev > exit_ret_thr and \
                   not pd.isna(dv_spike) and dv_spike > exit_dv_thr:
                    triggered = True
                    break
            if triggered:
                deployed_pct = max(0.0, deployed_pct - exit_pct)
                val *= (1 - KR_TAX * exit_pct / 2)
                last_exit_idx = i

        # PnL (보유 leader 종목)
        if lev > 0 and holdings:
            pr = 0
            for c, w in holdings.items():
                if c in feat:
                    r = feat[c]['ret'].get(d, 0) or 0
                    pr += w * (r if pd.notna(r) else 0)
            cost = max(0, lev-1) * KR_BORROW_DAILY
            if exit_mode == 'hb_portfolio':
                net = (lev * pr - cost) * deployed_pct + CASH_DAILY * (1 - deployed_pct)
            else:
                net = lev * pr - cost
        else:
            net = CASH_DAILY if lev == 0 else 0
        val *= (1 + net); val = max(val, 0.01)
        peak = max(peak, val)
        vals.append(val); dates_out.append(d)

        # Rebal check (매일 — rotation 룰로 실제 교체 제어)
        if lev == 0:
            if holdings:
                changed = sum(abs(w) for w in holdings.values())
                val *= (1 - KR_TAX * changed/2)
                holdings = {}; held_sectors = []; sector_entry_idx = {}
                deployed_pct = 1.0
            continue

        # 섹터 score 계산 (전일 d 기준 — 섹터 지수는 당일 close 포함, lag 위해 prev 사용)
        prev_d = all_dates[i-1]
        boost = 30 if base_lev >= 1.5 else 20   # panic/elevated zone = Super 대용
        sec_scored = []
        for s, idx in sector_idx.items():
            if prev_d not in idx.index: continue
            row = idx.loc[prev_d]
            sc = score_sc_l2_kr(row, boost)
            if sc is None: continue
            sec_scored.append((s, sc, row['ddPct'] if not pd.isna(row['ddPct']) else 0))
        if not sec_scored:
            continue
        sec_scored.sort(key=lambda x: -x[1])
        ranking = {s: rank for rank, (s, _, _) in enumerate(sec_scored)}

        # Rotation 룰
        do_rotate = False
        if not held_sectors:
            do_rotate = True
        else:
            for s in held_sectors:
                rk = ranking.get(s, 99)
                days = i - sector_entry_idx.get(s, i)
                if rk >= rotate_rank and days >= min_hold:
                    do_rotate = True
                    break

        if do_rotate:
            new_sectors = [s for s, _, _ in sec_scored[:top_sectors]]
            # 각 섹터 leader (M2 상위 leaders_per_sector개)
            ks_c5 = (ks_feat['ret_5d'].get(prev_d, 0) or 0) * 100
            leader_codes = []
            for s in new_sectors:
                sec_dd = next((dd for ss, _, dd in sec_scored if ss == s), 0)
                cand = []
                for c in sector_groups[s]:
                    if c not in feat or prev_d not in feat[c].index: continue
                    m2 = score_m2_kr(feat[c].loc[prev_d], sec_dd, ks_c5)
                    cand.append((c, m2))
                cand.sort(key=lambda x: -x[1])
                for c, _ in cand[:leaders_per_sector]:
                    leader_codes.append(c)
            if leader_codes:
                w_each = 1.0 / len(leader_codes)
                new_holdings = {c: w_each for c in leader_codes}
                changed = sum(abs(new_holdings.get(c, 0) - holdings.get(c, 0))
                              for c in set(list(holdings.keys()) + list(new_holdings.keys())))
                val *= (1 - KR_TAX * changed/2)
                holdings = new_holdings
                held_sectors = new_sectors
                sector_entry_idx = {s: i for s in new_sectors}
                last_rebal_idx = i
                deployed_pct = 1.0   # rotation 시 H-B reset

    return pd.Series(vals, index=dates_out)


if __name__ == '__main__':
    import time
    t0 = time.time()
    print('=' * 90)
    print('KR v40 — 현 한국전략 (V25-full+H-B) vs G5_22 한국적용판 (최근 2년)')
    print('=' * 90)
    print('Loading...')
    data = fetch_all_extended('2014-03-04')
    macro = load_macro('2014-01-01')
    ks = data['KS200']

    sector_groups = build_sector_groups(data)
    print(f'섹터 그룹 ({len(sector_groups)}개): ' + ', '.join(f'{s}({len(cs)})' for s, cs in sector_groups.items()))

    s_bh = ks['close'] / ks['close'].iloc[0]
    end = pd.to_datetime('2026-05-26')
    start = pd.to_datetime('2024-05-27')

    def metrics_2y(s):
        sub = s.loc[start:end]
        if len(sub) < 5: return None
        return m(sub / sub.iloc[0])

    # 현 전략 (벤치마크)
    print('\n현 전략 (V25-full + H-B top-7) 시뮬...')
    s_v25 = sim_flex(data, macro, top_k=7, weighting='equal', exit_mode='hb_portfolio',
                      exit_ret_thr=0.20, exit_dv_thr=3.0, exit_pct=0.34)

    # G5_22 한국판 튜닝 grid
    print('G5_22 한국판 튜닝 grid 시뮬...')
    g5_variants = [
        ('G5 baseline (top3, no exit)', dict(top_sectors=3, leaders_per_sector=1, exit_mode='none')),
        ('G5 top3 + H-B', dict(top_sectors=3, leaders_per_sector=1, exit_mode='hb_portfolio')),
        ('G5 top4 + H-B', dict(top_sectors=4, leaders_per_sector=1, exit_mode='hb_portfolio')),
        ('G5 top5 + H-B', dict(top_sectors=5, leaders_per_sector=1, exit_mode='hb_portfolio')),
        ('G5 top2 + H-B', dict(top_sectors=2, leaders_per_sector=1, exit_mode='hb_portfolio')),
        ('G5 top3 × 2leaders + H-B', dict(top_sectors=3, leaders_per_sector=2, exit_mode='hb_portfolio')),
        ('G5 top4 × 2leaders + H-B', dict(top_sectors=4, leaders_per_sector=2, exit_mode='hb_portfolio')),
        ('G5 top5 × 2leaders + H-B', dict(top_sectors=5, leaders_per_sector=2, exit_mode='hb_portfolio')),
    ]
    g5_sims = {}
    for name, kw in g5_variants:
        g5_sims[name] = sim_g5_korea(data, macro, min_hold=5, rotate_rank=5, **kw)

    print('\n' + '=' * 96)
    print(f'최근 2년 ({start.date()} ~ {end.date()}) 재대결 — G5_22 한국판 튜닝')
    print('=' * 96)
    print(f'{"전략":<34} {"Total":<11} {"CAGR":<9} {"Sharpe":<8} {"MDD":<9} {"Calmar":<8} {"종목수"}')
    print('-' * 96)
    r = metrics_2y(s_bh)
    print(f'{"BH KS200":<34} {r[0]*100:>+9.1f}% {r[1]*100:>+7.1f}% {r[2]:>7.2f} {r[3]*100:>+7.1f}% {r[4]:>7.2f}   -')
    r = metrics_2y(s_v25)
    print(f'{"★ 현 전략 (V25-full+H-B)":<34} {r[0]*100:>+9.1f}% {r[1]*100:>+7.1f}% {r[2]:>7.2f} {r[3]*100:>+7.1f}% {r[4]:>7.2f}   7')
    print('-' * 96)
    for name, kw in g5_variants:
        r = metrics_2y(g5_sims[name])
        n_stk = kw['top_sectors'] * kw['leaders_per_sector']
        if r is None:
            print(f'{name:<34} (데이터 부족)'); continue
        print(f'{name:<34} {r[0]*100:>+9.1f}% {r[1]*100:>+7.1f}% {r[2]:>7.2f} {r[3]*100:>+7.1f}% {r[4]:>7.2f}   {n_stk}')

    # 연도별 분해 (best G5 variant vs 현 전략)
    print('\n--- 연도별 Total Return (현 전략 vs G5 best 후보들) ---')
    best_g5 = max(g5_variants, key=lambda x: (metrics_2y(g5_sims[x[0]]) or [-9])[2])  # Sharpe 기준
    print(f'(G5 best by Sharpe: {best_g5[0]})')
    print(f'{"기간":<22} {"BH":<11} {"현 전략":<12} {"G5 best":<12}')
    for yname, ys, ye in [('2024-05~2025-05 (약세)', '2024-05-27', '2025-05-26'),
                           ('2025-05~2026-05 (강세)', '2025-05-27', '2026-05-26')]:
        ysd, yed = pd.to_datetime(ys), pd.to_datetime(ye)
        row = f'{yname:<22}'
        for s in [s_bh, s_v25, g5_sims[best_g5[0]]]:
            sub = s.loc[ysd:yed]
            ret = sub.iloc[-1]/sub.iloc[0] - 1 if len(sub) >= 5 else None
            row += f'{ret*100:>+9.1f}%  ' if ret is not None else f'{"?":<11}'
        print(row)

    print(f'\nElapsed: {time.time()-t0:.0f}s')

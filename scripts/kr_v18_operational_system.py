"""KR v18 — 운영 시스템 골격 (Daily zone + Monthly picks).

매일 KST 16:00 (장 마감 후) 자동 실행:
    1. FDR로 KS200 + 50종목 최신 종가 다운로드
    2. EWMA*1.25 proxy 계산 → zone 결정 → 권장 lev
    3. Macro gate 평가 (USDKRW + ^VIX)
    4. DD throttle 체크 (필요 시 lev 자동 조정)
    5. Telegram 출력: 오늘의 zone, lev, gate 상태

매월 21일째 거래일 추가 출력:
    6. 50종 mom120 / mom_ensemble score 계산
    7. Sector cap 적용 + top-7 선정
    8. 현 보유 종목과 비교 → 매도/매수 리스트
    9. Telegram 출력: 매도 N종, 매수 N종, 최종 holdings

State: holdings + last_rebal_date를 D1 (or file)에 저장
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import json
import os
import datetime
import requests
import pandas as pd
import numpy as np
from kr_v07_universe_comparison import INDIVIDUAL_STOCKS
from kr_v11_enhanced_modules import (
    SECTOR_MAP, apply_sector_cap, load_macro, macro_gate,
    dd_multistage_lev,
)
from kr_v17_attribution_analysis import STOCK_NAMES


STATE_FILE = Path(__file__).parent.parent / 'data' / 'kr_state.json'
STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

TELEGRAM_BOT_TOKEN = os.environ.get('TG_BOT_TOKEN', '')
TELEGRAM_CHAT_ID = os.environ.get('TG_CHAT_ID', '')


# ============================================================
# Zone (V12 그대로)
# ============================================================
def compute_zone(ks_close_series):
    """KS200 close → EWMA*1.25 → zone + lev."""
    rets = ks_close_series.pct_change()
    ewma_vol = rets.ewm(alpha=0.06).std() * (252**0.5) * 100
    proxy = ewma_vol * 1.25
    today_proxy = proxy.iloc[-1]
    if pd.isna(today_proxy): return 'unknown', 1.0, None
    if today_proxy < 15: return 'CALM (CASH)', 0, today_proxy
    if today_proxy < 22.5: return 'NORMAL', 1.0, today_proxy
    if today_proxy < 30: return 'ELEVATED', 1.5, today_proxy
    return 'PANIC (MAX BUY)', 2.0, today_proxy


def mom_score(close_series, lookback=120):
    """단순 mom lookback."""
    if len(close_series) <= lookback: return None
    return close_series.iloc[-1] / close_series.iloc[-lookback-1] - 1


def mom_ensemble(close_series, lookbacks=(90, 120, 150)):
    scores = []
    for lb in lookbacks:
        s = mom_score(close_series, lb)
        if s is not None: scores.append(s)
    return sum(scores)/len(scores) if scores else None


def v20_champion_score(close_series, mom_lb=120, sq_thr=0.8, sq_w=0.3, breakout_bonus=30):
    """V20 Champion score: mom120 + squeeze<0.8 bonus (w=0.3) + breakout."""
    if len(close_series) <= mom_lb:
        return None
    m = close_series.iloc[-1] / close_series.iloc[-mom_lb-1] - 1
    score = m * 100

    # BB squeeze ratio (today)
    if len(close_series) >= 80:
        ret = close_series.pct_change()
        ema20 = close_series.ewm(span=20).mean()
        bb_std = close_series.rolling(20).std()
        bb_width = (bb_std * 2) / ema20
        bb_width_avg60 = bb_width.rolling(60).mean()
        sq_ratio = bb_width.iloc[-1] / bb_width_avg60.iloc[-1]

        # Breakout: 어제 종가가 어제 upper band 돌파
        upper_yesterday = ema20.iloc[-2] + 2 * bb_std.iloc[-2]
        breakout = close_series.iloc[-1] > upper_yesterday

        if not pd.isna(sq_ratio) and sq_ratio <= sq_thr:
            bonus = (1 - sq_ratio) * 100  # squeeze strength
            if breakout:
                bonus += breakout_bonus
            score += bonus * sq_w
    return score


def v25_full_score(close_series, current_proxy=None,
                    mom_lb=120, sq_thr=0.8, breakout_bonus=30,
                    normal_w=0.7, elevated_w=0.5, panic_w=0.3,
                    low_rebound_w=0.2):
    """V25-FULL Champion: V20 + zone-dependent squeeze weight + 52w_low_rebound.

    V29 honest revalidation 결과:
      - V18 운영판 V25 (no 52w_low_rebound): +15,586%/Sh 1.33
      - V25-full (with 52w_low_rebound w=0.2): +19,796%/Sh 1.38 ⭐

    V27 KS200 60d DD scale은 look-ahead bias 결과로 retract됨.
    """
    if len(close_series) <= 252:
        return None
    m = close_series.iloc[-1] / close_series.iloc[-mom_lb-1] - 1


def v25_champion_score(close_series, current_proxy=None,
                        mom_lb=120, sq_thr=0.8, breakout_bonus=30,
                        normal_w=0.7, elevated_w=0.5, panic_w=0.3):
    """V25 Champion (deprecated: V18 운영판은 52w_low_rebound 누락 — V25-full로 교체 예정).

    current_proxy: KS200 VKOSPI proxy 오늘 값
        - < 22.5 → normal zone → w=0.7
        - 22.5-30 → elevated → w=0.5
        - ≥30 → panic → w=0.3
    """
    if len(close_series) <= mom_lb:
        return None
    m = close_series.iloc[-1] / close_series.iloc[-mom_lb-1] - 1
    score = m * 100

    if len(close_series) >= 80:
        ema20 = close_series.ewm(span=20).mean()
        bb_std = close_series.rolling(20).std()
        bb_width = (bb_std * 2) / ema20
        bb_width_avg60 = bb_width.rolling(60).mean()
        sq_ratio = bb_width.iloc[-1] / bb_width_avg60.iloc[-1]
        upper_yesterday = ema20.iloc[-2] + 2 * bb_std.iloc[-2]
        breakout = close_series.iloc[-1] > upper_yesterday

        if not pd.isna(sq_ratio) and sq_ratio <= sq_thr:
            bonus = (1 - sq_ratio) * 100
            if breakout: bonus += breakout_bonus
            # Zone-dep weight
            if current_proxy is None or pd.isna(current_proxy):
                w = normal_w
            elif current_proxy < 22.5:
                w = normal_w
            elif current_proxy < 30:
                w = elevated_w
            else:
                w = panic_w
            score += bonus * w
    return score


def v25_full_score_v2(close_series, current_proxy=None,
                       mom_lb=120, sq_thr=0.8, breakout_bonus=30,
                       normal_w=0.7, elevated_w=0.5, panic_w=0.3,
                       low_rebound_w=0.2):
    """V25-FULL: V25 + 52w_low_rebound w=0.2.

    V29 honest result: +19,796% / Sh 1.38 (V18 운영판 +15,586%/Sh 1.33 대비 +27%).
    52w 저점에서 30%+ 회복 + 5d 양수 종목에 가산점.
    """
    if len(close_series) <= 252:
        return None
    base = v25_champion_score(close_series, current_proxy, mom_lb, sq_thr,
                                breakout_bonus, normal_w, elevated_w, panic_w)
    if base is None: return None
    score = base

    # 52w_low_rebound bonus
    low_252 = close_series.iloc[-252:].min()
    dist_from_low = close_series.iloc[-1] / low_252 - 1
    if not pd.isna(dist_from_low) and dist_from_low > 0.30:
        score += min(dist_from_low, 1.0) * 100 * low_rebound_w * 0.3
        # 5d return
        if len(close_series) >= 6:
            r5 = close_series.iloc[-1] / close_series.iloc[-6] - 1
            score += r5 * 50 * low_rebound_w
    return score


# ============================================================
# State persistence
# ============================================================
def load_state():
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {
        'holdings': {},               # {code: weight}
        'last_rebal': None,            # YYYY-MM-DD
        'peak_value': 1.0,             # 정점 자본 (DD throttle 용)
        'current_value': 1.0,          # 현 자본
        'capital_won': 100_000_000,    # 초기 자본 1억
        'history': [],                  # daily log
    }


def save_state(state):
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, ensure_ascii=False, default=str)


# ============================================================
# Daily report
# ============================================================
def daily_report(send_telegram=False):
    """매일 장 마감 후 실행."""
    import FinanceDataReader as fdr
    import yfinance as yf

    state = load_state()

    # KS200 (2y history for stable EWMA proxy)
    ks = fdr.DataReader('KS200', '2023-01-01')
    ks.columns = [c.lower() for c in ks.columns]
    today = ks.index[-1]

    zone, base_lev, proxy = compute_zone(ks['close'])

    # V27: KS200 60d DD scale
    ks_60d_high = ks['close'].rolling(60, min_periods=20).max()
    ks_dd_60d = (ks['close'].iloc[-1] / ks_60d_high.iloc[-1] - 1)

    # Macro (1y history for chg20 calculation)
    macro = load_macro('2023-01-01')
    macro_row = macro.iloc[-1] if len(macro) > 0 else None
    if macro_row is not None:
        gate = macro_gate(macro_row, ks_row=None)
    else:
        gate = 'normal'

    # Apply gate
    if gate == 'crisis': lev = 0
    elif gate == 'caution': lev = min(base_lev, 1.0)
    else: lev = base_lev

    # V27 NEW: KS200 60d DD scale — RETRACTED (look-ahead bias in v27 simulator)
    # v29 honest revalidation 결과: lag 정정 후 alpha 사라짐, DD 감소만 효과.
    # 정보용 display 유지하되 lev에 적용 안 함.
    market_dd_throttle_active = False
    market_dd_info = None
    if not pd.isna(ks_dd_60d):
        market_dd_info = ks_dd_60d
        # NOTE: 만약 다음 거래일 액션으로 만 적용한다면 의미 있을 수 있음 (보수도 도구)
        # 하지만 alpha source 아님. 사용자 선택.

    # DD throttle (자본 DD)
    cur_dd = state['current_value'] / state['peak_value'] - 1
    lev_throttled = dd_multistage_lev(lev, cur_dd)
    throttle_active = lev_throttled < lev

    # Build report
    msg = f'🇰🇷 KR Champion V25-full Daily Report\n'
    msg += f'📅 {today.date()}\n'
    msg += f'\n📊 Market Zone\n'
    msg += f'  KS200 close: {ks["close"].iloc[-1]:,.2f}\n'
    msg += f'  VKOSPI proxy: {proxy:.1f}\n'
    msg += f'  KS200 60d DD: {ks_dd_60d*100:+.1f}%\n'
    msg += f'  Zone: **{zone}**\n'
    msg += f'  Base lev: {base_lev}x\n'
    msg += f'\n🌐 Macro Gate\n'
    if macro_row is not None:
        msg += f'  USDKRW: {macro_row["close"]:.0f} '
        msg += f'(20d {macro_row.get("krw_chg20",0)*100:+.1f}%)\n'
        msg += f'  ^VIX: {macro_row.get("us_vix",0):.1f}\n'
        msg += f'  Gate: **{gate.upper()}**\n'
    msg += f'  Final lev: **{lev}x**\n'

    # V27 retraction note: market DD throttle은 look-ahead 결과로 retract됨.
    # 정보용으로만 표시 (lev에 적용 안 함).
    if market_dd_info is not None and market_dd_info <= -0.05:
        msg += f'\nℹ️ KS200 60d DD: {market_dd_info*100:.1f}% (≤ -5%)\n'
        msg += f'  (참고: V27 DD throttle retract — lev에 적용 안 함)\n'

    if throttle_active:
        msg += f'\n⚠️ Capital DD Throttle Active\n'
        msg += f'  Current DD: {cur_dd*100:.1f}%\n'
        msg += f'  Adjusted lev: {lev_throttled}x\n'
        lev = lev_throttled

    # Recommended action
    msg += f'\n💡 Recommended Action\n'
    if lev == 0:
        msg += f'  → CASH (KODEX MMF / 예금)\n'
    elif lev == 1.0:
        msg += f'  → 정상 운영 (lev 1x, 종목 7개 균등)\n'
    elif lev == 1.5:
        msg += f'  → Elevated (lev 1.5x, 신용 50%)\n'
    elif lev >= 2.0:
        msg += f'  → 🔥 PANIC BUY (lev 2x, max 신용)\n'

    # Next rebal date
    if state['last_rebal']:
        last_rebal = pd.to_datetime(state['last_rebal'])
        days_since = (today - last_rebal).days
        msg += f'\n📅 Rebal\n'
        msg += f'  Last: {state["last_rebal"]} ({days_since}d ago)\n'
        if days_since >= 21:
            msg += f'  → **Rebal due today!**\n'
        else:
            msg += f'  → Next in {21 - days_since}d\n'

    # Save state
    state['current_value'] *= 1  # 자본 평가는 별도 (실거래 PnL 입력 필요)
    state['history'].append({
        'date': str(today.date()),
        'zone': zone,
        'lev': lev,
        'gate': gate,
        'proxy': float(proxy) if proxy is not None else None,
    })
    state['history'] = state['history'][-365:]  # 1년치만 보관
    save_state(state)

    print(msg)
    if send_telegram and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        send_tg(msg)

    return {'today': today, 'zone': zone, 'lev': lev, 'gate': gate, 'proxy': proxy}


def monthly_rebal_report(send_telegram=False):
    """매월 21일째 거래일에 실행 (or 사용자 manual trigger)."""
    import FinanceDataReader as fdr
    import time

    state = load_state()

    # KS200 fetch for zone proxy
    ks = fdr.DataReader('KS200', '2023-01-01')
    ks.columns = [c.lower() for c in ks.columns]
    _, _, current_proxy = compute_zone(ks['close'])

    # 50종목 + KS200
    print('Fetching prices...')
    closes = {}
    for code in INDIVIDUAL_STOCKS:
        try:
            df = fdr.DataReader(code, '2023-06-01')
            df.columns = [c.lower() for c in df.columns]
            if len(df) > 150:
                closes[code] = df['close']
            time.sleep(0.05)
        except: pass

    today = max(s.index[-1] for s in closes.values())

    # Compute scores (V25 Champion: mom120 + zone-dep squeeze bonus)
    scored = []
    for code, c in closes.items():
        sc = v25_full_score_v2(c, current_proxy=current_proxy)
        if sc is not None:
            mom_raw = mom_score(c, 120)
            scored.append((code, sc, mom_raw))
    scored.sort(key=lambda x: -x[1])

    # Sector cap + top-7 (apply_sector_cap expects 2-tuples; strip extra)
    scored_for_cap = [(c, sc) for c, sc, _ in scored]
    scored_capped = apply_sector_cap(scored_for_cap, max_per_sector=3)
    cap_codes = [c for c, _ in scored_capped[:7]]
    # Restore mom_raw mapping
    mom_map = {c: m_raw for c, _, m_raw in scored}
    new_picks = [(c, mom_map.get(c)) for c in cap_codes]
    new_codes = cap_codes

    # Compare with current holdings
    current = list(state['holdings'].keys())
    to_sell = [c for c in current if c not in new_codes]
    to_buy = [c for c in new_codes if c not in current]
    hold = [c for c in new_codes if c in current]

    proxy_str = f'{current_proxy:.1f}' if current_proxy is not None else '?'
    msg = f'📊 KR Champion Monthly Rebal\n'
    msg += f'📅 {today.date()}  (VKOSPI proxy={proxy_str})\n\n'
    zone_label = 'normal' if current_proxy is None or current_proxy < 22.5 else ('elevated' if current_proxy < 30 else 'panic')
    sq_w = {'normal': 0.7, 'elevated': 0.5, 'panic': 0.3}[zone_label]
    msg += f'🎯 Zone: {zone_label.upper()} → squeeze_w={sq_w}\n'
    msg += f'🆕 Top-7 Picks (V25-full: mom120 + zone-dep squeeze + 52w_low_rebound + sector cap 3):\n'
    for i, (code, mom_raw) in enumerate(new_picks, 1):
        name = STOCK_NAMES.get(code, '?')
        sec = SECTOR_MAP.get(code, '?')
        action = '🔄 HOLD' if code in hold else '🟢 BUY'
        mom_pct = mom_raw * 100 if mom_raw is not None else 0
        msg += f'  {i}. {code} {name} ({sec})  mom120={mom_pct:+.1f}%  {action}\n'
    if to_sell:
        msg += f'\n🔴 매도:\n'
        for c in to_sell:
            name = STOCK_NAMES.get(c, '?')
            msg += f'  - {c} {name}\n'
    msg += f'\n💼 Allocation: 14.3% per stock (1/7)\n'

    # Update state
    state['holdings'] = {c: 1.0/7 for c in new_codes}
    state['last_rebal'] = str(today.date())
    save_state(state)

    print(msg)
    if send_telegram and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
        send_tg(msg)

    return new_picks


def send_tg(msg):
    """Send via Telegram Bot API."""
    try:
        r = requests.post(
            f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage',
            json={'chat_id': TELEGRAM_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown'},
            timeout=10,
        )
        return r.status_code == 200
    except: return False


def update_pnl(current_value_won):
    """실거래 PnL 수동 업데이트 (사용자 입력)."""
    state = load_state()
    state['current_value'] = current_value_won / state['capital_won']
    state['peak_value'] = max(state['peak_value'], state['current_value'])
    save_state(state)
    print(f'PnL updated: {current_value_won/1e8:.2f}억원 '
          f'({(state["current_value"]-1)*100:+.1f}%, peak {(state["peak_value"]-1)*100:+.1f}%)')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='KR Champion 운영 시스템')
    parser.add_argument('--mode', default='daily',
                          choices=['daily', 'monthly', 'rebal', 'pnl', 'state'],
                          help='daily=오늘 zone, monthly=월간 picking, pnl=자본 업데이트, state=현 상태')
    parser.add_argument('--tg', action='store_true', help='Telegram 전송')
    parser.add_argument('--capital', type=float, default=None,
                          help='--mode pnl 시 현재 자본 (원)')
    args = parser.parse_args()

    if args.mode == 'daily':
        daily_report(send_telegram=args.tg)
    elif args.mode in ('monthly', 'rebal'):
        monthly_rebal_report(send_telegram=args.tg)
    elif args.mode == 'pnl':
        if args.capital is None:
            print('--capital 필요')
        else:
            update_pnl(args.capital)
    elif args.mode == 'state':
        state = load_state()
        print(json.dumps(state, indent=2, ensure_ascii=False, default=str)[:2000])

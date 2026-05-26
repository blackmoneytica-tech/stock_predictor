"""KR Champion JSON Publisher.

V18 운영 시스템 결과 → kr_dashboard/data/*.json 갱신 + Telegram 알림.

usage:
    python kr_publish_json.py --mode daily   # 매일 zone + lev 계산
    python kr_publish_json.py --mode rebal   # 월간 picking
    python kr_publish_json.py --mode both    # 둘 다 (월간 rebal day)

ENV:
    TG_BOT_TOKEN: Telegram bot token
    TG_CHAT_ID: Telegram chat ID
"""
import os
import sys
import json
import datetime
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import requests
import pandas as pd
import FinanceDataReader as fdr

from kr_v18_operational_system import (
    compute_zone, v25_full_score_v2, mom_score,
    SECTOR_MAP, apply_sector_cap, load_macro, macro_gate,
    dd_multistage_lev,
)
from kr_v17_attribution_analysis import STOCK_NAMES
from kr_v07_universe_comparison import INDIVIDUAL_STOCKS


DATA_DIR = Path(__file__).parent.parent / 'kr_dashboard' / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)
HISTORY_PATH = DATA_DIR / 'history.json'
STATE_PATH = DATA_DIR / 'state.json'   # H-B + holdings 추적


def load_state():
    if STATE_PATH.exists():
        try:
            with open(STATE_PATH, encoding='utf-8') as f:
                return json.load(f)
        except: pass
    return {
        'holdings': {},                # {code: weight}
        'last_rebal_date': None,
        'last_hb_exit_date': None,
        'deployed_pct': 1.0,
    }


def save_state(s):
    with open(STATE_PATH, 'w', encoding='utf-8') as f:
        json.dump(s, f, ensure_ascii=False, indent=2)

TG_BOT_TOKEN = os.environ.get('TG_BOT_TOKEN', '').strip()
TG_CHAT_ID = os.environ.get('TG_CHAT_ID', '').strip()


def send_telegram(msg):
    if not TG_BOT_TOKEN or not TG_CHAT_ID:
        print('[tg] skip: TG_BOT_TOKEN/TG_CHAT_ID not set')
        return False
    try:
        r = requests.post(
            f'https://api.telegram.org/bot{TG_BOT_TOKEN}/sendMessage',
            json={'chat_id': TG_CHAT_ID, 'text': msg, 'parse_mode': 'Markdown',
                  'disable_web_page_preview': True},
            timeout=15,
        )
        if r.status_code == 200:
            print('[tg] sent ok')
            return True
        print(f'[tg] fail: {r.status_code} {r.text[:200]}')
        return False
    except Exception as e:
        print(f'[tg] error: {e}')
        return False


def now_kst():
    return datetime.datetime.utcnow() + datetime.timedelta(hours=9)


def fetch_macro_minimal():
    """USDKRW + ^VIX 최신값 (light)."""
    out = {'usdkrw': None, 'us_vix': None, 'krw_chg20': None, 'gate': 'normal'}
    try:
        krw = fdr.DataReader('USD/KRW', '2025-01-01')
        krw.columns = [c.lower() for c in krw.columns]
        out['usdkrw'] = float(krw['close'].iloc[-1])
        if len(krw) > 20:
            out['krw_chg20'] = float(krw['close'].iloc[-1] / krw['close'].iloc[-21] - 1)
    except Exception as e:
        print(f'[macro] krw fail: {e}')
    try:
        import yfinance as yf
        vix = yf.download('^VIX', start='2025-01-01', progress=False, auto_adjust=False)
        if isinstance(vix.columns, pd.MultiIndex):
            vix.columns = vix.columns.get_level_values(0)
        if len(vix) > 0:
            out['us_vix'] = float(vix['Close'].iloc[-1])
    except Exception as e:
        print(f'[macro] vix fail: {e}')
    # Gate
    krw20 = out.get('krw_chg20')
    vix_v = out.get('us_vix')
    if vix_v is not None and vix_v > 40:
        out['gate'] = 'crisis'
    elif krw20 is not None and krw20 > 0.08 and vix_v is not None and vix_v > 30:
        out['gate'] = 'crisis'
    elif (krw20 is not None and krw20 > 0.05) or (vix_v is not None and vix_v > 30):
        out['gate'] = 'caution'
    return out


def load_history():
    if HISTORY_PATH.exists():
        try:
            with open(HISTORY_PATH, encoding='utf-8') as f:
                return json.load(f)
        except: pass
    return {'history': []}


def save_history(hist):
    with open(HISTORY_PATH, 'w', encoding='utf-8') as f:
        json.dump(hist, f, ensure_ascii=False, indent=2)


def check_hb_peak_exit(holdings_codes, ret_thr=0.20, dv_spike_thr=3.0):
    """H-B Peak Exit signal: 보유 종목 중 전일 ret > 20% AND dv_spike > 3x 발생?

    Returns: dict {triggered: bool, triggered_stocks: [code...], ret_dv_details: [...]}
    """
    import time
    triggered = []
    details = []
    for code in holdings_codes:
        try:
            df = fdr.DataReader(code, '2025-06-01')
            df.columns = [c.lower() for c in df.columns]
            if len(df) < 65: continue
            # 전일 (어제 close 까지의 정보)
            ret_yesterday = float(df['close'].iloc[-1] / df['close'].iloc[-2] - 1)
            # dv_spike: 어제 거래대금 / 최근 5일 평균 (어제 제외)
            dv = df['close'] * df['volume']
            dv_yesterday = float(dv.iloc[-1])
            dv_5d_avg = float(dv.iloc[-6:-1].mean())
            dv_spike = dv_yesterday / dv_5d_avg if dv_5d_avg > 0 else 1.0
            details.append({
                'code': code,
                'ret_yesterday': ret_yesterday,
                'dv_spike': dv_spike,
            })
            if ret_yesterday > ret_thr and dv_spike > dv_spike_thr:
                triggered.append(code)
            time.sleep(0.03)
        except Exception as e:
            print(f'  [hb] {code} fail: {e}')
    return {
        'triggered': len(triggered) > 0,
        'triggered_stocks': triggered,
        'details': details,
    }


def refresh_pick_prices():
    """monthly.json picks의 close 가격을 최신 데이터로 갱신.

    매일 호출되어 picks의 가격이 fresh하도록 유지 (action plan 정확도 위해).
    """
    import time
    monthly_path = DATA_DIR / 'monthly.json'
    if not monthly_path.exists():
        return
    try:
        with open(monthly_path, encoding='utf-8') as f:
            data = json.load(f)
        for pick in data.get('picks', []):
            try:
                df = fdr.DataReader(pick['code'], '2026-01-01')
                df.columns = [c.lower() for c in df.columns]
                if len(df) > 0:
                    pick['close'] = float(df['close'].iloc[-1])
                time.sleep(0.05)
            except Exception as e:
                print(f'  [price] {pick["code"]} fail: {e}')
        with open(monthly_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f'[price] refreshed {len(data.get("picks", []))} picks')
    except Exception as e:
        print(f'[price] refresh fail: {e}')


def do_daily():
    """매일 zone + lev 계산 → daily.json + history append + Telegram.

    V25-full + H-B 20/3/34 portfolio (최종 Champion):
      - Picking: mom120 + zone-dep squeeze + 52w_low_rebound
      - Zone leverage: 0/1/1.5/2x (proxy 15/22.5/30 thresholds)
      - Macro gate: USDKRW + ^VIX
      - DD throttle: -30/-45/-55/-65% multi-stage
      - H-B Peak Exit: 보유 중 전일 ret>20% AND dv_spike>3x → portfolio 1/3 매도 + 5d cooldown

    Note: Strict-Panic은 채택 안 함 (사용자 결정 — Total/Cal/OOS Total 균형 위해 V25 + H-B 채택)
    """
    print('[daily] start')
    # KS200 fetch (2y history for EWMA proxy)
    ks = fdr.DataReader('KS200', '2023-01-01')
    ks.columns = [c.lower() for c in ks.columns]
    today = ks.index[-1]

    zone, base_lev, proxy = compute_zone(ks['close'])

    # KS200 60d DD (정보용 — lev에 적용 안 함, V25-full + H-B 채택)
    ks_60d_high = ks['close'].rolling(60, min_periods=20).max()
    ks_dd_60d_same = float(ks['close'].iloc[-1] / ks_60d_high.iloc[-1] - 1)
    if len(ks) >= 62:
        ks_60d_high_lagged = ks_60d_high.iloc[-2]
        ks_dd_60d_lagged = float(ks['close'].iloc[-2] / ks_60d_high_lagged - 1)
    else:
        ks_dd_60d_lagged = None

    # Macro gate
    macro = fetch_macro_minimal()
    gate = macro['gate']
    lev = base_lev
    if gate == 'crisis': lev = 0
    elif gate == 'caution': lev = min(lev, 1.0)

    # H-B Peak Exit 체크 (state의 holdings + last_peak_exit_idx 활용)
    state = load_state()
    hb_signal = {'triggered': False, 'triggered_stocks': [], 'cooldown_active': False}
    H_B_COOLDOWN_DAYS = 5
    last_hb_exit_date = state.get('last_hb_exit_date')
    if last_hb_exit_date:
        try:
            last_d = pd.to_datetime(last_hb_exit_date).date()
            days_since = (today.date() - last_d).days
            if days_since < H_B_COOLDOWN_DAYS:
                hb_signal['cooldown_active'] = True
                hb_signal['cooldown_days_left'] = H_B_COOLDOWN_DAYS - days_since
        except: pass

    holdings = state.get('holdings', {})
    if holdings and not hb_signal['cooldown_active']:
        hb_check = check_hb_peak_exit(list(holdings.keys()), ret_thr=0.20, dv_spike_thr=3.0)
        if hb_check['triggered']:
            hb_signal['triggered'] = True
            hb_signal['triggered_stocks'] = hb_check['triggered_stocks']
            hb_signal['details'] = hb_check['details']
            # State 갱신: portfolio 1/3 매도 권고
            state['last_hb_exit_date'] = str(today.date())
            state['deployed_pct'] = max(0.0, state.get('deployed_pct', 1.0) - 0.34)
            save_state(state)

    # Apply deployed_pct to effective lev display
    deployed_pct = state.get('deployed_pct', 1.0)
    effective_lev = lev * deployed_pct

    # Build payload
    payload = {
        'timestamp': now_kst().strftime('%Y-%m-%d %H:%M:%S'),
        'market_date': str(today.date()),
        'ks200_close': float(ks['close'].iloc[-1]),
        'vkospi_proxy': float(proxy) if proxy is not None else None,
        'zone': zone,
        'base_lev': float(base_lev),
        'final_lev': float(lev),
        'effective_lev': float(effective_lev),  # H-B 적용 후 실효 lev
        'deployed_pct': float(deployed_pct),     # H-B 1/3 매도 누적 후 deployed
        'macro_gate': gate,
        'usdkrw': macro['usdkrw'],
        'us_vix': macro['us_vix'],
        'krw_chg20': macro['krw_chg20'],
        'ks200_dd_60d_today': ks_dd_60d_same,
        'ks200_dd_60d_lagged': ks_dd_60d_lagged,
        'ks200_dd_60d': ks_dd_60d_lagged,
        # H-B
        'hb_triggered': hb_signal['triggered'],
        'hb_triggered_stocks': hb_signal['triggered_stocks'],
        'hb_cooldown_active': hb_signal.get('cooldown_active', False),
        'hb_cooldown_days_left': hb_signal.get('cooldown_days_left', 0),
    }
    daily_path = DATA_DIR / 'daily.json'
    with open(daily_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'[daily] wrote {daily_path}')

    # History append
    hist = load_history()
    zone_label_simple = (zone.split()[0] if zone else 'normal').lower()
    hist_entry = {
        'date': str(today.date()),
        'zone': zone,
        'zone_label': zone_label_simple,
        'lev': float(lev),
        'proxy': float(proxy) if proxy is not None else None,
        'gate': gate,
    }
    # 같은 날짜 dedup
    hist['history'] = [h for h in hist['history'] if h['date'] != hist_entry['date']]
    hist['history'].append(hist_entry)
    hist['history'] = hist['history'][-365:]   # 1년치 보관
    save_history(hist)

    # Telegram message
    proxy_str = f'{proxy:.1f}' if proxy is not None else '?'
    action = ''
    if lev == 0:
        action = '💤 CASH (KODEX MMF / 예금)'
    elif lev <= 1.0:
        action = '✅ 정상 운영 (lev 1x)'
    elif lev <= 1.5:
        action = '⚡ Elevated (lev 1.5x, 신용 50%)'
    else:
        action = '🔥 PANIC BUY (lev 2x, max 신용)'

    dd_lagged_str = f'{ks_dd_60d_lagged*100:+.1f}%' if ks_dd_60d_lagged is not None else '?'
    msg = f'🇰🇷 *KR V25-full + H-B Daily*\n'
    msg += f'📅 {today.date()}\n\n'
    msg += f'📊 Market Zone\n'
    msg += f'  KS200: `{payload["ks200_close"]:,.2f}`\n'
    msg += f'  VKOSPI proxy: `{proxy_str}`\n'
    msg += f'  60d DD (lagged): `{dd_lagged_str}`\n\n'
    msg += f'🎯 Zone: *{zone}* → Base lev: *{base_lev}x*\n'
    msg += f'🌐 Macro Gate: *{gate.upper()}*\n'
    if macro['usdkrw']:
        krw20_str = f'{macro["krw_chg20"]*100:+.1f}%' if macro['krw_chg20'] is not None else '?'
        msg += f'  USDKRW: {macro["usdkrw"]:.0f} ({krw20_str})\n'
    if macro['us_vix']:
        msg += f'  ^VIX: {macro["us_vix"]:.1f}\n'

    # H-B 상태 표시
    if hb_signal['triggered']:
        msg += f'\n🚨 *H-B PEAK EXIT 발동!*\n'
        msg += f'  트리거 종목: {", ".join(hb_signal["triggered_stocks"])}\n'
        msg += f'  → 다음 거래일 portfolio 1/3 매도\n'
        msg += f'  → deployed_pct: {deployed_pct*100:.0f}% → {(deployed_pct-0.34)*100:.0f}%\n'
        msg += f'  → 5일 cooldown 시작\n'
    elif hb_signal.get('cooldown_active'):
        msg += f'\n⏸ H-B 쿨다운 중 ({hb_signal["cooldown_days_left"]}일 남음)\n'
    elif holdings:
        msg += f'\n👀 H-B 감시 중 (ret>20% AND dv>3x 시 1/3 익절)\n'

    msg += f'\n💡 *{action}*\n'
    if deployed_pct < 1.0:
        msg += f'  현재 deployed: *{deployed_pct*100:.0f}%* (H-B 1/3 매도 후)\n'
        msg += f'  Effective lev: *{effective_lev:.2f}x*\n'
    msg += f'\nFinal lev: *{lev}x*'

    # picks의 close 가격 refresh (action plan용)
    refresh_pick_prices()

    send_telegram(msg)
    return payload


def do_rebal():
    """월간 picking → monthly.json + Telegram."""
    print('[rebal] start')
    import time

    # KS200 zone for current proxy (squeeze weight 결정)
    ks = fdr.DataReader('KS200', '2023-01-01')
    ks.columns = [c.lower() for c in ks.columns]
    _, _, current_proxy = compute_zone(ks['close'])

    # 50종목 fetch
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

    # Score
    scored = []
    for code, c in closes.items():
        sc = v25_full_score_v2(c, current_proxy=current_proxy)
        if sc is not None:
            mom_raw = mom_score(c, 120)
            scored.append((code, sc, mom_raw))
    scored.sort(key=lambda x: -x[1])

    # Sector cap + top-7
    scored_for_cap = [(c, sc) for c, sc, _ in scored]
    scored_capped = apply_sector_cap(scored_for_cap, max_per_sector=3)
    cap_codes = [c for c, _ in scored_capped[:7]]
    mom_map = {c: m_raw for c, _, m_raw in scored}
    new_codes = cap_codes

    # Load previous picks for HOLD/SELL/BUY status
    prev_path = DATA_DIR / 'monthly.json'
    prev_holdings = []
    if prev_path.exists():
        try:
            with open(prev_path, encoding='utf-8') as f:
                prev_data = json.load(f)
                prev_holdings = [p['code'] for p in prev_data.get('picks', [])]
        except: pass

    # Build picks
    if current_proxy is None or current_proxy < 22.5:
        zone_label, sq_w = 'NORMAL', 0.7
    elif current_proxy < 30:
        zone_label, sq_w = 'ELEVATED', 0.5
    else:
        zone_label, sq_w = 'PANIC', 0.3

    picks = []
    for code in new_codes:
        mom_raw = mom_map.get(code)
        status = 'HOLD' if code in prev_holdings else 'BUY'
        # 현재가 (rebal 시점 close)
        close_kr = float(closes[code].iloc[-1]) if code in closes else None
        picks.append({
            'code': code,
            'name': STOCK_NAMES.get(code, '?'),
            'sector': SECTOR_MAP.get(code, 'Other'),
            'mom120': mom_raw,
            'status': status,
            'close': close_kr,
        })

    # Compute sells (previous holdings not in new picks)
    sells = []
    for code in prev_holdings:
        if code not in new_codes:
            sells.append({
                'code': code,
                'name': STOCK_NAMES.get(code, '?'),
                'sector': SECTOR_MAP.get(code, 'Other'),
            })

    # Rebal 시 state 갱신: holdings + deployed_pct reset (1.0)
    state = load_state()
    state['holdings'] = {c: 1.0/7 for c in new_codes}
    state['last_rebal_date'] = str(today.date())
    state['deployed_pct'] = 1.0   # H-B reset
    state['last_hb_exit_date'] = None
    save_state(state)
    print(f'[rebal] state updated: holdings={len(new_codes)}, deployed=1.0')

    # next_rebal_date: 21 영업일 후 추정 (calendar로는 약 30일)
    next_rebal_date = (today + pd.Timedelta(days=30)).date()

    payload = {
        'timestamp': now_kst().strftime('%Y-%m-%d %H:%M:%S'),
        'last_rebal': str(today.date()),
        'next_rebal': '21일 후',
        'next_rebal_date': str(next_rebal_date),
        'zone': zone_label,
        'squeeze_weight': sq_w,
        'vkospi_proxy': float(current_proxy) if current_proxy is not None else None,
        'picks': picks,
        'sells': sells,
    }
    with open(prev_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'[rebal] wrote {prev_path}')

    # Telegram
    proxy_str = f'{current_proxy:.1f}' if current_proxy is not None else '?'
    msg = f'📊 *KR V25-full Monthly Rebal*\n'
    msg += f'📅 {today.date()} (proxy={proxy_str})\n\n'
    msg += f'🎯 Zone: *{zone_label}* → squeeze\\_w={sq_w}\n\n'
    msg += f'🆕 *Top-7 Picks:*\n'
    for i, p in enumerate(picks, 1):
        sector_kr = {'Semi':'반도체','Tech':'Tech','Game':'게임','Auto':'자동차',
                      'Battery':'2차전지','Chem':'화학','Oil':'정유','DefShip':'방산조선',
                      'Finance':'금융','Bio':'바이오','Consumer':'소비재','Util':'인프라',
                      'Logistics':'물류','Construct':'건설','Leisure':'레저'}.get(p['sector'], p['sector'])
        mom_str = f'+{p["mom120"]*100:.0f}%' if p['mom120'] is not None else '?'
        emoji = '🟢' if p['status'] == 'BUY' else '🔄'
        msg += f'  {i}. `{p["code"]}` {p["name"]} ({sector_kr})\n     mom120 {mom_str} · {emoji} {p["status"]}\n'
    if sells:
        msg += f'\n🔴 *매도:*\n'
        for s in sells:
            msg += f'  - {s["code"]} {s["name"]}\n'
    msg += f'\n💼 균등 배분: 14.3% per stock (1/7)'
    send_telegram(msg)
    return payload


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='daily', choices=['daily', 'rebal', 'both'])
    args = p.parse_args()
    if args.mode in ('daily', 'both'):
        do_daily()
    if args.mode in ('rebal', 'both'):
        do_rebal()

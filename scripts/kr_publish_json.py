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


def do_daily():
    """매일 zone + lev 계산 → daily.json + history append + Telegram."""
    print('[daily] start')
    # KS200 fetch (2y history for EWMA proxy)
    ks = fdr.DataReader('KS200', '2023-01-01')
    ks.columns = [c.lower() for c in ks.columns]
    today = ks.index[-1]

    zone, base_lev, proxy = compute_zone(ks['close'])

    # KS200 60d DD (information only, not applied to lev — V27 retract)
    ks_60d_high = ks['close'].rolling(60, min_periods=20).max()
    ks_dd_60d = float(ks['close'].iloc[-1] / ks_60d_high.iloc[-1] - 1)

    # Macro gate
    macro = fetch_macro_minimal()
    gate = macro['gate']
    lev = base_lev
    if gate == 'crisis': lev = 0
    elif gate == 'caution': lev = min(lev, 1.0)

    # Build payload
    payload = {
        'timestamp': now_kst().strftime('%Y-%m-%d %H:%M:%S'),
        'market_date': str(today.date()),
        'ks200_close': float(ks['close'].iloc[-1]),
        'vkospi_proxy': float(proxy) if proxy is not None else None,
        'zone': zone,
        'base_lev': float(base_lev),
        'final_lev': float(lev),
        'macro_gate': gate,
        'usdkrw': macro['usdkrw'],
        'us_vix': macro['us_vix'],
        'krw_chg20': macro['krw_chg20'],
        'ks200_dd_60d': ks_dd_60d,
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

    msg = f'🇰🇷 *KR V25-full Daily*\n'
    msg += f'📅 {today.date()}\n\n'
    msg += f'📊 Market Zone\n'
    msg += f'  KS200: `{payload["ks200_close"]:,.2f}`\n'
    msg += f'  VKOSPI proxy: `{proxy_str}`\n'
    msg += f'  60d DD: `{ks_dd_60d*100:+.1f}%`\n\n'
    msg += f'🎯 Zone: *{zone}*\n'
    msg += f'  Base lev: {base_lev}x\n\n'
    msg += f'🌐 Macro Gate: *{gate.upper()}*\n'
    if macro['usdkrw']:
        krw20_str = f'{macro["krw_chg20"]*100:+.1f}%' if macro['krw_chg20'] is not None else '?'
        msg += f'  USDKRW: {macro["usdkrw"]:.0f} ({krw20_str})\n'
    if macro['us_vix']:
        msg += f'  ^VIX: {macro["us_vix"]:.1f}\n'
    msg += f'\n💡 *{action}*\n'
    msg += f'\nFinal lev: *{lev}x*'

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
        picks.append({
            'code': code,
            'name': STOCK_NAMES.get(code, '?'),
            'sector': SECTOR_MAP.get(code, 'Other'),
            'mom120': mom_raw,
            'status': status,
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

    payload = {
        'timestamp': now_kst().strftime('%Y-%m-%d %H:%M:%S'),
        'last_rebal': str(today.date()),
        'next_rebal': '21일 후',
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

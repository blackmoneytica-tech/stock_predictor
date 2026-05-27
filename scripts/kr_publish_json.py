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

try:
    import kis_flow
    KIS_AVAILABLE = True
except Exception:
    KIS_AVAILABLE = False


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
        'pending_hb_exit': None,       # {date, stocks} — 신호 후 매도 실행 대기
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
            # 최신 거래일(iloc[-1]) 종가 기준 일일 ret (백테스트 ret 컬럼과 동일)
            ret_today = float(df['close'].iloc[-1] / df['close'].iloc[-2] - 1)
            # dv_spike: 당일 거래대금 / 당일 포함 최근 5일 평균
            #   백테스트 add_features_v30: dv_spike = dv / dv.rolling(5).mean() (당일 포함)와 일치.
            #   (이전 버그: iloc[-6:-1]로 당일 제외 → 분모↓ → 라이브가 더 자주 발동)
            dv = df['close'] * df['volume']
            dv_today = float(dv.iloc[-1])
            dv_5d_avg = float(dv.iloc[-5:].mean())   # 당일 포함 5일 (rolling(5) 동일)
            dv_spike = dv_today / dv_5d_avg if dv_5d_avg > 0 else 1.0
            details.append({
                'code': code,
                'ret_today': ret_today,
                'dv_spike': dv_spike,
            })
            if ret_today > ret_thr and dv_spike > dv_spike_thr:
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


def update_flow_signals():
    """KIS 수급 데이터 수집 → forward-collecting 누적 + monthly.json picks 신호 갱신.

    A: monthly.json picks에 frgn/inst/combo 5d 신호 + universe rank (Bottom 여부)
    B: data/flow_history.csv 누적 (forward-collecting, 미래 정량 검증용)
    """
    if not KIS_AVAILABLE:
        print('[flow] kis_flow 모듈 없음 — skip')
        return
    try:
        # universe 전체 수급 fetch (rank 계산 위해)
        print(f'[flow] fetching {len(INDIVIDUAL_STOCKS)} stocks...')
        flows = kis_flow.get_recent_flow(INDIVIDUAL_STOCKS)
        print(f'[flow] got {len(flows)} stocks')

        # B: forward-collecting 누적
        total = kis_flow.append_to_history(flows)
        print(f'[flow] history accumulated: {total} total rows')

        # universe 전체 신호 계산 → rank
        sigs = {}
        for code, df in flows.items():
            sig = kis_flow.compute_flow_signals(df).dropna(subset=['combo_5d_pct'])
            if len(sig) == 0: continue
            last = sig.iloc[-1]
            sigs[code] = {
                'date': str(last['date'].date()),
                'frgn_5d_pct': float(last['frgn_5d_pct']) if pd.notna(last['frgn_5d_pct']) else None,
                'inst_5d_pct': float(last['inst_5d_pct']) if pd.notna(last['inst_5d_pct']) else None,
                'combo_5d_pct': float(last['combo_5d_pct']) if pd.notna(last['combo_5d_pct']) else None,
            }
        # combo_5d_pct rank (낮을수록 매도 압력 → Bottom)
        valid = [(c, s['combo_5d_pct']) for c, s in sigs.items() if s['combo_5d_pct'] is not None]
        valid.sort(key=lambda x: x[1])
        n = len(valid)
        rank_map = {c: i for i, (c, _) in enumerate(valid)}   # 0 = 가장 강한 매도

        # A: monthly.json picks 갱신
        monthly_path = DATA_DIR / 'monthly.json'
        if monthly_path.exists():
            with open(monthly_path, encoding='utf-8') as f:
                mdata = json.load(f)
            bottom_threshold = max(1, int(n * 0.20))   # Bottom 20%
            for pick in mdata.get('picks', []):
                code = pick['code']
                s = sigs.get(code)
                if s:
                    pick['frgn_5d_pct'] = s['frgn_5d_pct']
                    pick['inst_5d_pct'] = s['inst_5d_pct']
                    pick['combo_5d_pct'] = s['combo_5d_pct']
                    rk = rank_map.get(code)
                    pick['flow_rank'] = rk
                    pick['flow_universe_n'] = n
                    pick['flow_bottom'] = (rk is not None and rk < bottom_threshold)
                else:
                    pick['combo_5d_pct'] = None
                    pick['flow_bottom'] = False
            mdata['flow_updated'] = now_kst().strftime('%Y-%m-%d %H:%M:%S')
            with open(monthly_path, 'w', encoding='utf-8') as f:
                json.dump(mdata, f, ensure_ascii=False, indent=2)
            print(f'[flow] monthly.json picks 갱신 완료')
    except Exception as e:
        print(f'[flow] fail: {type(e).__name__}: {e}')


def do_daily():
    """매일 zone + lev 계산 → daily.json + history append + Telegram.

    V25-full + H-B 20/3/34 portfolio + EMA200 penalty (최종 Champion):
      - Picking(월간 rebal): mom120 + zone-dep squeeze + 52w_low_rebound + EMA200 penalty 30
      - Zone leverage: 0/1/1.5/2x (proxy 15/22.5/30 thresholds). proxy=EWMA 변동성×1.25 (방향무관)
      - Macro gate: USDKRW + ^VIX → crisis=0, caution=min(lev,1.0)
      - H-B Peak Exit: 보유 중 당일 ret>20% AND dv_spike>3x(당일포함 5일) → 다음 거래일 1/3 매도
        + 거래일 5일 cooldown. pending→executed 2단계 (신호일 pending, 매도일 deployed 확정).

    Note:
      - Strict-Panic 미채택 (V25+H-B가 Total/Cal/OOS 균형 우수). proxy는 급등도 PANIC 분류하나
        검증상 급등형도 사후 +6.4% alpha (kr_v55) → lev 2.0 유지. 대시보드 라벨은 "고변동".
      - DD throttle(백테스트 dd_multistage_lev)은 라이브 미적용: 계좌 equity/peak 미추적.
        약세장 방어는 EMA200 penalty(picking) + macro gate로 대체. 자본 -30%+ 시 수동 lev 축소 권장.
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

    # H-B Peak Exit (백테스트 일치: 거래일 기준 cooldown + pending→executed 2단계)
    #   - 신호 detect(당일 종가) → pending 표시 (deployed 유지, "내일 1/3 매도")
    #   - 다음 거래일 cron → 매도 실행 가정하고 deployed 확정 + cooldown 시작
    #   백테스트 매핑: i-1 신호 → i 실행 = 라이브 T 신호 → T+1 실행.
    state = load_state()
    hb_signal = {'triggered': False, 'triggered_stocks': [], 'cooldown_active': False,
                 'pending': False, 'executed': False}
    H_B_COOLDOWN_TRADING_DAYS = 5
    H_B_EXIT_PCT = 0.34

    def trading_days_since(date_str):
        """date_str(거래일) 이후 ~ today까지 경과한 거래일 수 (KS200 인덱스 기준)."""
        try:
            d0 = pd.to_datetime(date_str)
            return int((ks.index > d0).sum())
        except Exception:
            return 999

    # 0. pending 확정: 전일 신호 → 오늘(다음 거래일) 매도 실행 가정 → deployed 감소 + cooldown 시작
    pending = state.get('pending_hb_exit')
    if pending and trading_days_since(pending.get('date')) >= 1:
        state['deployed_pct'] = max(0.0, state.get('deployed_pct', 1.0) - H_B_EXIT_PCT)
        state['last_hb_exit_date'] = str(today.date())   # cooldown은 실행일부터
        state['pending_hb_exit'] = None
        save_state(state)
        hb_signal['executed'] = True
        hb_signal['executed_stocks'] = pending.get('stocks', [])

    # 1. cooldown 체크 (거래일 기준)
    last_hb_exit_date = state.get('last_hb_exit_date')
    if last_hb_exit_date:
        td = trading_days_since(last_hb_exit_date)
        if td < H_B_COOLDOWN_TRADING_DAYS:
            hb_signal['cooldown_active'] = True
            hb_signal['cooldown_days_left'] = H_B_COOLDOWN_TRADING_DAYS - td

    # 2. 신규 신호 detect (cooldown 아니고 pending 없을 때만)
    holdings = state.get('holdings', {})
    still_pending = state.get('pending_hb_exit')
    if holdings and not hb_signal['cooldown_active'] and not still_pending:
        hb_check = check_hb_peak_exit(list(holdings.keys()), ret_thr=0.20, dv_spike_thr=3.0)
        if hb_check['triggered']:
            hb_signal['triggered'] = True
            hb_signal['pending'] = True
            hb_signal['triggered_stocks'] = hb_check['triggered_stocks']
            hb_signal['details'] = hb_check['details']
            # pending만 설정 — deployed는 다음 거래일(매도 실행 후) 확정
            state['pending_hb_exit'] = {'date': str(today.date()),
                                         'stocks': hb_check['triggered_stocks']}
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
        # H-B (pending=신호·매도대기, executed=오늘 매도확정)
        'hb_triggered': hb_signal['triggered'],
        'hb_triggered_stocks': hb_signal['triggered_stocks'],
        'hb_pending': hb_signal.get('pending', False),
        'hb_executed': hb_signal.get('executed', False),
        'hb_executed_stocks': hb_signal.get('executed_stocks', []),
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

    # H-B 상태 표시 (pending=신호·매도대기 / executed=오늘 매도확정)
    if hb_signal['triggered']:
        msg += f'\n🚨 *H-B 신호 발동! (다음 거래일 매도)*\n'
        msg += f'  트리거 종목: {", ".join(hb_signal["triggered_stocks"])}\n'
        msg += f'  → 다음 거래일 portfolio 1/3 매도 실행\n'
        msg += f'  → deployed {deployed_pct*100:.0f}% (매도 실행 후 확정)\n'
    elif hb_signal.get('executed'):
        msg += f'\n✅ *H-B 매도 실행 확정*\n'
        msg += f'  종목: {", ".join(hb_signal.get("executed_stocks", []))}\n'
        msg += f'  → deployed {deployed_pct*100:.0f}% / 거래일 5일 cooldown 시작\n'
    elif hb_signal.get('cooldown_active'):
        msg += f'\n⏸ H-B 쿨다운 중 (거래일 {hb_signal["cooldown_days_left"]}일 남음)\n'
    elif holdings:
        msg += f'\n👀 H-B 감시 중 (ret>20% AND dv>3x 시 1/3 익절)\n'

    msg += f'\n💡 *{action}*\n'
    if deployed_pct < 1.0:
        msg += f'  현재 deployed: *{deployed_pct*100:.0f}%* (H-B 1/3 매도 후)\n'
        msg += f'  Effective lev: *{effective_lev:.2f}x*\n'
    msg += f'\nFinal lev: *{lev}x*'

    # picks의 close 가격 refresh (action plan용)
    refresh_pick_prices()

    # KIS 수급 신호 갱신 + forward-collecting
    update_flow_signals()

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
    # EMA200 추세 penalty (kr_v41~v43 검증): 종가 < EMA200 (하락추세) 종목 -30점.
    #   - 약세장 방어 (2018 +2.6→+10.1%, 2022 -12.6→-7.8%), 강세장 무해
    #   - Full 12.2년: Total +21,604→+22,947%, MDD -32.8→-31.1%, Calmar 1.88→2.01
    #   - walk-forward IS/OOS 둘 다 통과 (IS Sh 1.54 유지 + OOS 1.59→1.61) — robust
    #   - penalty(30)는 hard filter보다 우수: 종목 7개 항상 유지(hard는 6개로 떨어짐), 총수익↑
    EMA200_PENALTY = 30
    scored = []
    ema200_below = {}
    for code, c in closes.items():
        sc = v25_full_score_v2(c, current_proxy=current_proxy)
        if sc is not None:
            mom_raw = mom_score(c, 120)
            ema200 = c.ewm(span=200, adjust=False).mean().iloc[-1]
            below = bool(c.iloc[-1] < ema200)
            ema200_below[code] = below
            if below:
                sc -= EMA200_PENALTY
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
            'below_ema200': ema200_below.get(code, False),   # 하락추세 여부 (penalty 적용됨)
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
    state['pending_hb_exit'] = None   # rebal 시 미실행 pending도 reset
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


def is_last_friday(d):
    """d가 그 달의 마지막 금요일인가 (d가 금요일 전제). d+7일이 다음 달이면 마지막."""
    return (d + datetime.timedelta(days=7)).month != d.month


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mode', default='daily', choices=['daily', 'rebal', 'both'])
    args = p.parse_args()
    # 수동 실행(workflow_dispatch) 또는 mode=rebal은 강제 rebal
    force_rebal = (os.environ.get('KR_FORCE_REBAL') == '1') or (args.mode == 'rebal')

    if args.mode in ('daily', 'both'):
        do_daily()
    if args.mode == 'rebal':
        do_rebal()
    elif args.mode == 'both':
        # cron은 22-31일 금요일마다 trigger → 이번달 마지막 금요일에만 rebal (중복 방지)
        today_kst = now_kst().date()
        if force_rebal or is_last_friday(today_kst):
            do_rebal()
        else:
            print(f'[monthly] {today_kst}는 이번달 마지막 금요일 아님 — rebal skip (daily만 실행)')

"""KIS OpenAPI 수급 모듈 — 최근 30일 외국인/기관 매매주체별 데이터.

운영용:
    - get_recent_flow(codes): 종목별 최근 30일 외국인/기관 순매매
    - compute_flow_signals(df): 5d/20d 누적 + 거래대금 normalize
    - append_to_history(df): forward-collecting (data/flow_history.csv 누적 dedup)

⚠️ KIS inquire-investor (FHKST01010900)는 최근 ~30거래일만 제공 (과거 불가).
    → 매일 cron으로 호출하여 점진적으로 historical 축적.
"""
import os
import time
from pathlib import Path

import requests
import pandas as pd

_ENV_LOADED = False


def _load_env():
    global _ENV_LOADED
    if _ENV_LOADED: return
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if '=' in line and not line.startswith('#'):
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())
    _ENV_LOADED = True


BASE_URL = 'https://openapi.koreainvestment.com:9443'
HISTORY_PATH = Path(__file__).parent.parent / 'data' / 'flow_history.csv'
TOKEN_PATH = Path(__file__).parent.parent / 'data' / '.kis_token.json'

_token_cache = {'token': None, 'ts': 0}


def get_token():
    """KIS access token (24h 유효). 파일 캐시 — KIS는 토큰 재발급 rate limit 엄격."""
    import json
    _load_env()
    # 메모리 캐시
    if _token_cache['token'] and (time.time() - _token_cache['ts'] < 12 * 3600):
        return _token_cache['token']
    # 파일 캐시 (프로세스 재시작 후에도 재사용)
    if TOKEN_PATH.exists():
        try:
            cached = json.loads(TOKEN_PATH.read_text())
            if time.time() - cached['ts'] < 18 * 3600:   # 18h 안이면 재사용 (24h 만료 안전 마진)
                _token_cache['token'] = cached['token']
                _token_cache['ts'] = cached['ts']
                return cached['token']
        except Exception:
            pass
    # 신규 발급
    app_key = os.environ.get('KIS_APP_KEY')
    app_secret = os.environ.get('KIS_APP_SECRET')
    if not app_key or not app_secret:
        raise RuntimeError('KIS_APP_KEY, KIS_APP_SECRET 환경변수 필요')
    r = requests.post(
        f'{BASE_URL}/oauth2/tokenP',
        json={'grant_type': 'client_credentials',
              'appkey': app_key, 'appsecret': app_secret},
        timeout=20,
    )
    r.raise_for_status()
    token = r.json()['access_token']
    _token_cache['token'] = token
    _token_cache['ts'] = time.time()
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps({'token': token, 'ts': time.time()}))
    return token


def fetch_stock_flow(code, token=None, max_retries=3):
    """단일 종목 최근 30거래일 외국인/기관 순매매.

    Returns: DataFrame [date, close, prsn_qty, frgn_qty, orgn_qty,
                        prsn_val, frgn_val, orgn_val] or None
    """
    _load_env()
    if token is None:
        token = get_token()
    app_key = os.environ.get('KIS_APP_KEY')
    app_secret = os.environ.get('KIS_APP_SECRET')
    headers = {
        'authorization': f'Bearer {token}',
        'appkey': app_key, 'appsecret': app_secret,
        'tr_id': 'FHKST01010900',
        'content-type': 'application/json',
    }
    params = {
        'FID_COND_MRKT_DIV_CODE': 'J',
        'FID_INPUT_ISCD': code,
        'FID_INPUT_DATE_1': '',
        'FID_INPUT_DATE_2': '',
        'FID_PERIOD_DIV_CODE': 'D',
    }
    for attempt in range(max_retries):
        try:
            r = requests.get(f'{BASE_URL}/uapi/domestic-stock/v1/quotations/inquire-investor',
                              headers=headers, params=params, timeout=30)
            if r.status_code != 200:
                time.sleep(2 ** attempt); continue
            data = r.json()
            if data.get('rt_cd') != '0':
                time.sleep(1.5); continue
            out = data.get('output', [])
            if not out: return None
            df = pd.DataFrame(out)
            rename = {
                'stck_bsop_date': 'date', 'stck_clpr': 'close',
                'prsn_ntby_qty': 'prsn_qty', 'frgn_ntby_qty': 'frgn_qty', 'orgn_ntby_qty': 'orgn_qty',
                'prsn_ntby_tr_pbmn': 'prsn_val', 'frgn_ntby_tr_pbmn': 'frgn_val', 'orgn_ntby_tr_pbmn': 'orgn_val',
            }
            df = df.rename(columns=rename)
            keep = [c for c in rename.values() if c in df.columns]
            df = df[keep].copy()
            for col in keep:
                if col != 'date':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='coerce')
            df = df.dropna(subset=['date']).sort_values('date')
            df['code'] = code
            return df
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            if attempt < max_retries - 1:
                time.sleep(3 + 2 ** attempt); continue
            return None
        except Exception:
            return None
    return None


def get_recent_flow(codes, sleep=0.3):
    """여러 종목 최근 30일 수급 데이터 dict."""
    token = get_token()
    out = {}
    for code in codes:
        df = fetch_stock_flow(code, token=token)
        if df is not None and len(df) > 0:
            out[code] = df
        time.sleep(sleep)
    return out


def compute_flow_signals(df):
    """5d/20d 누적 + 거래대금 normalize 신호 계산 (단일 종목 df).

    frgn_5d_val: 외국인 5일 누적 순매수 거래대금 (원)
    combo_5d_pct: (외국인+기관) 5일 순매수액 / 5일 총 거래대금 추정
    """
    df = df.sort_values('date').copy()
    # 5d/20d 누적 순매수 거래대금
    for win in [5, 20]:
        df[f'frgn_{win}d_val'] = df['frgn_val'].rolling(win).sum()
        df[f'orgn_{win}d_val'] = df['orgn_val'].rolling(win).sum()
        df[f'combo_{win}d_val'] = df[f'frgn_{win}d_val'] + df[f'orgn_{win}d_val']
    # 총 거래대금 추정 (개인+외국인+기관 순매수 절대값 합 ≈ 활동량 proxy)
    df['turnover_5d'] = (df['prsn_val'].abs() + df['frgn_val'].abs() + df['orgn_val'].abs()).rolling(5).sum()
    df['combo_5d_pct'] = df['combo_5d_val'] / df['turnover_5d'].replace(0, pd.NA)
    df['frgn_5d_pct'] = df['frgn_5d_val'] / df['turnover_5d'].replace(0, pd.NA)
    df['inst_5d_pct'] = df['orgn_5d_val'] / df['turnover_5d'].replace(0, pd.NA)
    return df


def latest_signals(codes):
    """각 종목의 최신 시점 수급 신호 dict {code: {frgn_5d_pct, inst_5d_pct, combo_5d_pct, ...}}."""
    flows = get_recent_flow(codes)
    result = {}
    for code, df in flows.items():
        sig = compute_flow_signals(df)
        # 당일(장중) 매매주체 미확정 NaN 제외 — 유효한 마지막 행 사용
        sig = sig.dropna(subset=['combo_5d_pct'])
        if len(sig) == 0: continue
        last = sig.iloc[-1]
        result[code] = {
            'date': str(last['date'].date()),
            'frgn_5d_pct': float(last['frgn_5d_pct']) if pd.notna(last['frgn_5d_pct']) else None,
            'inst_5d_pct': float(last['inst_5d_pct']) if pd.notna(last['inst_5d_pct']) else None,
            'combo_5d_pct': float(last['combo_5d_pct']) if pd.notna(last['combo_5d_pct']) else None,
            'frgn_5d_val': float(last['frgn_5d_val']) if pd.notna(last['frgn_5d_val']) else None,
            'orgn_5d_val': float(last['orgn_5d_val']) if pd.notna(last['orgn_5d_val']) else None,
        }
    return result


def append_to_history(flows):
    """forward-collecting: 수집한 raw 데이터를 flow_history.csv에 누적 (날짜+code dedup)."""
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_rows = []
    for code, df in flows.items():
        cols = ['date', 'code', 'close', 'prsn_qty', 'frgn_qty', 'orgn_qty',
                'prsn_val', 'frgn_val', 'orgn_val']
        new_rows.append(df[[c for c in cols if c in df.columns]])
    if not new_rows:
        return 0
    new_df = pd.concat(new_rows, ignore_index=True)

    if HISTORY_PATH.exists():
        old = pd.read_csv(HISTORY_PATH, encoding='utf-8-sig', parse_dates=['date'])
        old['code'] = old['code'].astype(str).str.zfill(6)
        new_df['code'] = new_df['code'].astype(str).str.zfill(6)
        combined = pd.concat([old, new_df], ignore_index=True)
    else:
        combined = new_df
        combined['code'] = combined['code'].astype(str).str.zfill(6)

    combined = combined.drop_duplicates(subset=['date', 'code'], keep='last')
    combined = combined.sort_values(['code', 'date'])
    combined.to_csv(HISTORY_PATH, index=False, encoding='utf-8-sig')
    return len(combined)


if __name__ == '__main__':
    # Test
    codes = ['009150', '011070', '000660', '005380']
    print('Testing latest_signals...')
    sigs = latest_signals(codes)
    for code, s in sigs.items():
        print(f'  {code}: combo_5d={s["combo_5d_pct"]*100:+.1f}% '
              f'(frgn {s["frgn_5d_pct"]*100:+.1f}%, inst {s["inst_5d_pct"]*100:+.1f}%) @ {s["date"]}')

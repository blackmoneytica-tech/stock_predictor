"""KR v11 — Enhanced modules (개선 P1-P3 모두 통합).

5개 개선 모듈:
    1. SECTOR_MAP — 50종목 sector 분류 (manual)
    2. macro_gate() — USDKRW + ^VIX spillover
    3. apply_sector_cap() — 동일 sector 최대 3종 제한
    4. dd_multistage() — DD throttle 다단계
    5. mom_ensemble() — mom90+120+150 평균 score
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import numpy as np
import time


# ============================================================
# 1. Sector mapping (manual, 50종목)
# ============================================================
SECTOR_MAP = {
    # 반도체
    '005930': 'Semi',       # 삼성전자
    '000660': 'Semi',       # SK하이닉스
    '011070': 'Semi',       # LG이노텍
    '009150': 'Semi',       # 삼성전기
    # IT/Tech
    '035420': 'Tech',       # NAVER
    '035720': 'Tech',       # 카카오
    '018260': 'Tech',       # 삼성에스디에스
    '030200': 'Tech',       # KT
    '017670': 'Tech',       # SK텔레콤
    '066570': 'Tech',       # LG전자 (가전+가전IT 혼합)
    # 게임
    '251270': 'Game',       # 넷마블
    '259960': 'Game',       # 크래프톤
    # 자동차/모빌리티
    '005380': 'Auto',       # 현대차
    '000270': 'Auto',       # 기아
    '012330': 'Auto',       # 현대모비스
    '086280': 'Auto',       # 현대글로비스
    # 2차전지
    '373220': 'Battery',    # LG에너지솔루션
    '006400': 'Battery',    # 삼성SDI
    '003670': 'Battery',    # 포스코퓨처엠
    # 화학/소재
    '051910': 'Chem',       # LG화학
    '011170': 'Chem',       # 롯데케미칼
    '005490': 'Chem',       # POSCO홀딩스 (철강+소재 혼합)
    '010130': 'Chem',       # 고려아연
    # 정유
    '096770': 'Oil',        # SK이노베이션
    '010950': 'Oil',        # S-Oil
    # 조선/방산
    '042660': 'DefShip',    # 한화오션
    '012450': 'DefShip',    # 한화에어로
    '329180': 'DefShip',    # HD현대중공업
    '009540': 'DefShip',    # HD한국조선해양
    '010140': 'DefShip',    # 삼성중공업
    '047810': 'DefShip',    # 한국항공우주
    # 금융
    '105560': 'Finance',    # KB금융
    '055550': 'Finance',    # 신한지주
    '086790': 'Finance',    # 하나금융지주
    '316140': 'Finance',    # 우리금융지주
    '024110': 'Finance',    # 기업은행
    '032830': 'Finance',    # 삼성생명
    '071050': 'Finance',    # 한국금융지주
    '006800': 'Finance',    # 미래에셋증권
    # 바이오/제약
    '068270': 'Bio',        # 셀트리온
    '207940': 'Bio',        # 삼성바이오로직스
    '128940': 'Bio',        # 한미약품
    # 소비재
    '051900': 'Consumer',   # LG생활건강
    '097950': 'Consumer',   # CJ제일제당
    '033780': 'Consumer',   # KT&G
    # 인프라/기타
    '015760': 'Util',       # 한국전력
    '028260': 'Util',       # 삼성물산
    '011200': 'Logistics',  # HMM
    '017800': 'Construct',  # 현대엘리베이
    '035250': 'Leisure',    # 강원랜드
}


def apply_sector_cap(picks_with_scores, max_per_sector=3):
    """동일 sector 최대 N개 cap.

    picks_with_scores: [(code, score), ...] sorted by score desc
    return: filtered list (same order, but sector cap applied)
    """
    out = []
    sector_count = {}
    for code, sc in picks_with_scores:
        sec = SECTOR_MAP.get(code, 'Other')
        if sector_count.get(sec, 0) < max_per_sector:
            out.append((code, sc))
            sector_count[sec] = sector_count.get(sec, 0) + 1
    return out


# ============================================================
# 2. Macro gate (USDKRW + ^VIX spillover)
# ============================================================
def load_macro(start='2008-01-01'):
    """USDKRW + 미국 ^VIX 데이터 로드."""
    krw = fdr.DataReader('USD/KRW', start)
    krw.columns = [c.lower() for c in krw.columns]
    krw = krw[['close']].copy()
    krw['krw_ema50'] = krw['close'].ewm(span=50).mean()
    krw['krw_chg5'] = krw['close'].pct_change(5)
    krw['krw_chg20'] = krw['close'].pct_change(20)
    krw.index = pd.to_datetime(krw.index)

    vix = yf.download('^VIX', start=start, end='2026-12-31',
                       progress=False, auto_adjust=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix.columns = vix.columns.get_level_values(0)
    vix = vix[['Close']].rename(columns={'Close': 'us_vix'})
    vix.index = pd.to_datetime(vix.index)

    macro = pd.concat([krw, vix], axis=1).ffill()
    return macro


def macro_gate(macro_row, ks_row=None):
    """Macro 게이트.

    Return:
        'normal' — full strategy 적용
        'caution' — lev 1x cap (lev 2x → 1x)
        'crisis' — CASH 강제

    트리거:
        1. USDKRW 20일 변화 > +5% → caution (원화 약세 → 외국인 이탈)
        2. USDKRW 20일 변화 > +8% AND ^VIX > 30 → crisis (panic)
        3. ^VIX > 40 → crisis (US contagion)
        4. KS200 EMA200 하단 → caution (장기 trend 깨짐)
    """
    krw_chg20 = macro_row.get('krw_chg20', 0)
    if pd.isna(krw_chg20): krw_chg20 = 0
    us_vix = macro_row.get('us_vix', 20)
    if pd.isna(us_vix): us_vix = 20

    # crisis 조건
    if us_vix > 40:
        return 'crisis'
    if krw_chg20 > 0.08 and us_vix > 30:
        return 'crisis'

    # caution 조건
    if krw_chg20 > 0.05:
        return 'caution'
    if us_vix > 30:
        return 'caution'
    if ks_row is not None and ks_row.get('below_ema200') == True:
        return 'caution'

    return 'normal'


# ============================================================
# 3. DD throttle 다단계
# ============================================================
def dd_multistage_lev(base_lev, current_dd):
    """현재 DD에 따라 lev 다단계 throttle.

    -30% → lev × 0.8
    -45% → lev × 0.6
    -55% → lev × 0.4
    -65% → CASH (lev = 0)
    """
    if current_dd <= -0.65:
        return 0
    if current_dd <= -0.55:
        return base_lev * 0.4
    if current_dd <= -0.45:
        return base_lev * 0.6
    if current_dd <= -0.30:
        return base_lev * 0.8
    return base_lev


# ============================================================
# 4. Mom 인접 lookback ensemble
# ============================================================
def mom_ensemble_score(row, lookbacks=(90, 120, 150)):
    """3개 lookback 평균 momentum.

    Note: features 컴퓨팅 시 ret_90d, ret_120d, ret_150d 모두 미리 계산 필요.
    """
    scores = []
    for lb in lookbacks:
        v = row.get(f'ret_{lb}d', None)
        if v is not None and not pd.isna(v):
            scores.append(v)
    if not scores:
        return None
    return sum(scores) / len(scores)


# ============================================================
# 5. Point-in-time universe (분기별 거래량 top N)
# ============================================================
def pit_universe(data, target_date, n=50, lookback_days=60):
    """target_date 시점 기준 lookback 60일 평균 거래대금 top N 종목.

    Survivorship bias 보정 (지금 시점 top 50 fix와 다름)
    Note: 전체 universe candidate를 미리 load해야 함.
    """
    scores = []
    for code, df in data.items():
        if code in ('KS200', '122630'): continue
        if target_date not in df.index: continue
        try:
            idx = df.index.get_loc(target_date)
            if idx < lookback_days: continue
            sub = df.iloc[idx-lookback_days:idx]
            dv = (sub['close'] * sub['volume']).mean() if 'volume' in sub.columns else 0
            if dv > 0:
                scores.append((code, dv))
        except: pass
    scores.sort(key=lambda x: -x[1])
    return [c for c, _ in scores[:n]]


# ============================================================
# 6. 약세장 OOS 검증 helper (2003-2014 zone-only)
# ============================================================
def long_history_zone_test(start='2003-01-02', end='2014-03-04'):
    """2003-2014 (12y, 약세장 + GFC 포함) zone-only KS200 백테스트.
    Sector ETF 없음 — KS200 자체에 virtual leverage 적용.
    """
    ks = fdr.DataReader('KS200', start, end)
    ks.columns = [c.lower() for c in ks.columns]
    ks['ret'] = ks['close'].pct_change()
    ks['ewma_vol'] = ks['ret'].ewm(alpha=0.06).std() * (252**0.5) * 100
    ks['vkospi_proxy'] = ks['ewma_vol'] * 1.25
    ks['vkospi_prev'] = ks['vkospi_proxy'].shift(1)
    return ks


if __name__ == '__main__':
    print('Modules created. Sector distribution:')
    from collections import Counter
    secs = Counter(SECTOR_MAP.values())
    for s, n in sorted(secs.items(), key=lambda x: -x[1]):
        print(f'  {s:15s}: {n}')
    print(f'Total: {sum(secs.values())}')

    # Test macro
    print('\nLoading macro...')
    macro = load_macro('2014-01-01')
    print(f'Macro shape: {macro.shape}')
    print(macro.tail(3))

    # Test gate on recent date
    print('\nMacro gate samples:')
    for d in pd.to_datetime(['2014-03-04', '2020-03-15', '2022-09-30', '2024-08-05', '2026-05-22']):
        if d in macro.index:
            g = macro_gate(macro.loc[d])
            print(f'  {d.date()}: {g} '
                  f'(krw_chg20={macro.loc[d, "krw_chg20"]:.2%}, '
                  f'vix={macro.loc[d, "us_vix"]:.1f})')

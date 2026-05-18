"""Sector breadth + VIX term structure + credit spread.

alert/screener_macro.pine 의 매크로 시스템을 stock_predictor에 통합.
사용자가 이미 검증한 시그널들:
- 11 섹터 ETF (XLK/XLF/XLE/XLV/XLI/XLY/XLP/XLU/XLRE/XLB/XLC)
- VIX/VIX9D ratio — term structure (< 0.95 contango = 정상, > 1.0 backwardation = 위험)
- HYG/LQD ratio — 신용 스프레드 (위기 시 ↓)
- SPY/QQQ/IWM breadth — 시장 폭

yfinance로 historical fetch 가능 → backtest에서도 lookahead 없이 활용.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ._common import cache_load, cache_store
from .price_feed import get_daily_ohlcv


# 종목 → 섹터 매핑 (alert/worker.js의 SECTOR_MAP 일부)
SECTOR_MAP = {
    # Tech
    'AAPL':'XLK','MSFT':'XLK','GOOGL':'XLC','META':'XLC','NVDA':'XLK','AVGO':'XLK',
    'AMZN':'XLY','TSLA':'XLY','ORCL':'XLK','CRM':'XLK','ADBE':'XLK','INTC':'XLK',
    'AMD':'XLK','QCOM':'XLK','TXN':'XLK','MU':'XLK','AMAT':'XLK','LRCX':'XLK',
    'KLAC':'XLK','ASML':'XLK','PLTR':'XLK','SMCI':'XLK','SNOW':'XLK','DDOG':'XLK',
    # Finance / Fintech
    'JPM':'XLF','BAC':'XLF','GS':'XLF','V':'XLF','MA':'XLF','BX':'XLF',
    'WFC':'XLF','MS':'XLF','BLK':'XLF','C':'XLF',
    'CRCL':'XLF','HOOD':'XLF','COIN':'XLF','MSTR':'XLF','SOFI':'XLF','UPST':'XLF',
    # Healthcare
    'UNH':'XLV','LLY':'XLV','JNJ':'XLV','PFE':'XLV','MRK':'XLV','ABBV':'XLV',
    'TMO':'XLV','ABT':'XLV','GILD':'XLV','AMGN':'XLV','VRTX':'XLV',
    # Consumer
    'COST':'XLP','WMT':'XLP','NKE':'XLY','SBUX':'XLY','MCD':'XLY','HD':'XLY',
    'TGT':'XLP','LULU':'XLY','DIS':'XLC','KO':'XLP','PEP':'XLP',
    # Industrial / Energy
    'CAT':'XLI','BA':'XLI','HON':'XLI','UNP':'XLI','UPS':'XLI','DE':'XLI',
    'XOM':'XLE','CVX':'XLE','COP':'XLE','OXY':'XLE',
    # Comm / Auto
    'NFLX':'XLC','T':'XLC','VZ':'XLC',
    'F':'XLY','GM':'XLY','RIVN':'XLY',
}


def get_sector_for(ticker: str) -> str:
    """종목 → 섹터 ETF (기본 XLK)."""
    return SECTOR_MAP.get(ticker.upper(), 'XLK')


# ── Sector breadth + 매크로 시그널 ──────────────────────────
def compute_macro_breadth_at(as_of: date) -> Dict:
    """as_of 시점의 시장 매크로 breadth + 신호.

    Returns:
        {
            sector_avg: 11 섹터 평균 (그날 intraday %)
            sector_green: 양수 섹터 수
            sector_red: 음수 섹터 수
            spy_chg, qqq_chg, iwm_chg: 주요 지수 변화
            vix, vix_chg, vix9d, vix_term: VIX term structure
            hyg_lqd: 신용 스프레드 ratio
            risk_off_score: 종합 위험 점수 (0~5)
            mode: STRONG_BULL / BULL / CHOPPY / BEAR / STRONG_BEAR
        }
    """
    cache_key = f"macro_breadth:{as_of.isoformat()}"
    cached = cache_load("macro_breadth", cache_key, ttl_seconds=86400)
    if cached is not None and not cached.empty:
        try:
            return cached.iloc[0].to_dict()
        except Exception:
            pass

    sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY',
                   'XLP', 'XLU', 'XLRE', 'XLB', 'XLC']
    sector_chgs = {}
    for etf in sector_etfs:
        chg = _intraday_chg_at(etf, as_of)
        if chg is not None:
            sector_chgs[etf] = chg

    sector_avg = float(np.mean(list(sector_chgs.values()))) if sector_chgs else 0.0
    sector_green = sum(1 for v in sector_chgs.values() if v > 0.1)
    sector_red = sum(1 for v in sector_chgs.values() if v < -0.1)

    spy_chg = _intraday_chg_at('SPY', as_of) or 0.0
    qqq_chg = _intraday_chg_at('QQQ', as_of) or 0.0
    iwm_chg = _intraday_chg_at('IWM', as_of) or 0.0

    vix_level = _close_at('^VIX', as_of)
    vix_chg = _intraday_chg_at('^VIX', as_of) or 0.0
    vix9d = _close_at('^VIX9D', as_of)
    vix_term = (vix9d / vix_level) if (vix_level and vix9d and vix_level > 0) else None

    hyg = _close_at('HYG', as_of)
    lqd = _close_at('LQD', as_of)
    hyg_lqd = (hyg / lqd) if (hyg and lqd and lqd > 0) else None

    # Cross-asset signals (BTC/Gold/DXY) — 사용자 alert macro에 영감
    btc_chg = _intraday_chg_at('BTC-USD', as_of) or 0.0
    gold_chg = _intraday_chg_at('GLD', as_of) or 0.0
    dxy_chg = _intraday_chg_at('DX-Y.NYB', as_of) or _intraday_chg_at('UUP', as_of) or 0.0
    tlt_chg = _intraday_chg_at('TLT', as_of) or 0.0  # 20Y bond — yield 반대 방향

    # 시장 모드 판정 (alert/worker.js classifyMarketModeV2 단순화)
    mode = _classify_mode(
        sector_green, sector_red, sector_avg,
        spy_chg, qqq_chg, vix_level, vix_chg,
    )

    # risk_off score (0~5)
    risk_off = 0
    if vix_level and vix_level > 25: risk_off += 1
    if vix_level and vix_level > 30: risk_off += 1
    if vix_chg > 10: risk_off += 1
    if vix_term and vix_term > 1.0: risk_off += 1  # backwardation
    if hyg_lqd is not None:
        # 신용 스프레드 위기 — HYG/LQD가 최근 30일 평균보다 1.5%+ 낮음
        # 단순화: 절대 ratio < 0.96 = 위기 (역사적 베이스)
        if hyg_lqd < 0.96: risk_off += 1

    result = {
        'as_of': as_of.isoformat(),
        'sector_avg': round(sector_avg, 3),
        'sector_green': int(sector_green),
        'sector_red': int(sector_red),
        'spy_chg': round(spy_chg, 3),
        'qqq_chg': round(qqq_chg, 3),
        'iwm_chg': round(iwm_chg, 3),
        'vix': round(vix_level, 2) if vix_level else None,
        'vix_chg': round(vix_chg, 2),
        'vix9d': round(vix9d, 2) if vix9d else None,
        'vix_term': round(vix_term, 3) if vix_term else None,
        'hyg_lqd': round(hyg_lqd, 4) if hyg_lqd else None,
        'risk_off_score': int(risk_off),
        'mode': mode,
        'btc_chg': round(btc_chg, 3),
        'gold_chg': round(gold_chg, 3),
        'dxy_chg': round(dxy_chg, 3),
        'tlt_chg': round(tlt_chg, 3),
        # 섹터 raw (cache용)
        **{f'sec_{k}': round(v, 3) for k, v in sector_chgs.items()},
    }
    cache_store("macro_breadth", cache_key, pd.DataFrame([result]))
    return result


def _close_at(ticker: str, as_of: date) -> Optional[float]:
    """as_of 영업일의 close."""
    try:
        df = get_daily_ohlcv(ticker, as_of - timedelta(days=10), as_of + timedelta(days=2))
        df = df[df.index <= pd.Timestamp(as_of)]
        if df.empty:
            return None
        return float(df['close'].iloc[-1])
    except Exception:
        return None


def _intraday_chg_at(ticker: str, as_of: date) -> Optional[float]:
    """as_of 영업일의 intraday change pct (open → close)."""
    try:
        df = get_daily_ohlcv(ticker, as_of - timedelta(days=5), as_of + timedelta(days=2))
        df = df[df.index <= pd.Timestamp(as_of)]
        if df.empty:
            return None
        row = df.iloc[-1]
        if row['open'] <= 0:
            return None
        return float((row['close'] - row['open']) / row['open'] * 100)
    except Exception:
        return None


def _classify_mode(
    sector_green: int, sector_red: int, sector_avg: float,
    spy_chg: float, qqq_chg: float,
    vix_level: Optional[float], vix_chg: float,
) -> str:
    """시장 모드 (alert/worker.js classifyMarketModeV2 단순화)."""
    # Tier 1: index + VIX 강제
    if qqq_chg >= 1.5 and vix_level and vix_level < 18 and vix_chg <= 0 and spy_chg > 0:
        return 'STRONG_BULL'
    if spy_chg > 0.3 and qqq_chg > 0.5 and vix_level and vix_level < 20 and vix_chg <= 0:
        return 'BULL'
    if spy_chg < -0.5 and qqq_chg < -0.5 and vix_level and vix_level > 22 and vix_chg > 5:
        return 'STRONG_BEAR'

    # Tier 2: breadth
    if sector_green >= 8 and sector_avg >= 0.5 and spy_chg > 0.3:
        return 'STRONG_BULL'
    if sector_green >= 6 and sector_avg > 0 and spy_chg > 0:
        return 'BULL'
    if sector_red >= 8 and sector_avg <= -0.5 and spy_chg < 0:
        return 'STRONG_BEAR'
    if sector_red >= 6 and sector_avg < 0 and spy_chg < 0 and qqq_chg < 0:
        return 'BEAR'

    return 'CHOPPY'


# ── Macro 시그널을 score로 변환 ─────────────────────────────
def macro_breadth_score(ticker: str, breadth: Dict, betas: Optional[Dict] = None) -> float:
    """종목의 sector + 시장 매크로 + cross-asset → -10 ~ +10 score.

    - 종목 섹터 변화 (가장 큰 가중)
    - 시장 mode
    - risk_off penalty
    - VIX term backwardation 추가 penalty
    - **NEW**: BTC/Gold/DXY cross-asset (BTC-correlated 종목용)
    """
    if not breadth:
        return 0.0
    betas = betas or {}

    sector = get_sector_for(ticker)
    sec_chg = breadth.get(f'sec_{sector}', 0.0)
    score = 0.0

    # 섹터 변화 (-2 ~ +2)
    score += np.clip(sec_chg * 1.5, -3, 3)

    # 시장 모드 (-3 ~ +3)
    mode_map = {
        'STRONG_BULL': 3.0, 'BULL': 1.5, 'CHOPPY': 0.0,
        'BEAR': -2.0, 'STRONG_BEAR': -4.0,
    }
    score += mode_map.get(breadth.get('mode', 'CHOPPY'), 0)

    # Risk off penalty
    risk_off = breadth.get('risk_off_score', 0)
    score -= risk_off * 0.8

    # VIX term backwardation penalty
    vix_term = breadth.get('vix_term')
    if vix_term is not None and vix_term > 1.0:
        score -= 2.0

    # Cross-asset (BTC/Gold/DXY) — beta 기반
    btc_chg = breadth.get('btc_chg', 0.0)
    gold_chg = breadth.get('gold_chg', 0.0)
    dxy_chg = breadth.get('dxy_chg', 0.0)
    tlt_chg = breadth.get('tlt_chg', 0.0)

    btc_beta = betas.get('btc', 0.0)  # crypto-related stocks (CRCL/COIN/MSTR 등)
    gold_beta = betas.get('gold', 0.0)
    # DXY 음의 영향 (DXY 상승 = USD 강세 = 미국주식 부담)
    dxy_beta = betas.get('dxy', -0.3)

    score += btc_chg * btc_beta * 1.5
    score += gold_chg * gold_beta * 1.0
    score += dxy_chg * dxy_beta * 0.8

    # TLT (장기 금리 반비례) — 채권 ↑ = yield ↓ = risk-on
    score += tlt_chg * 0.3

    return float(np.clip(score, -10, 10))

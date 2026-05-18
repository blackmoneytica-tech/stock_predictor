"""Trend-following 시그널 — CRCL 4주 backtest의 한계 보완.

배경:
  v1~v3 backtest에서 directional accuracy 33%만 달성.
  원인: 시스템이 모멘텀/추세를 catch 못 함 (다른 모듈들은 mean-reversion 성향)
  CRCL이 +30% rally를 했는데 시스템은 neutral/bear 출력.

해결:
  Multi-timeframe trend confirmation (1D, 1W, 1M):
  - 종가 vs SMA(20/50/200)
  - SMA cross 상태 (golden / death)
  - 모멘텀 가속 (Rate of Change)
  - ADX trend strength

스코어:
  +10: 강한 상승 추세 (모든 timeframe align + 모멘텀 가속)
  +5: 정상 상승 추세
  0: 혼조 / sideways
  -5: 하락 추세
  -10: 강한 하락 추세 + 가속
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..types import ModuleOutput
from .base import AnalysisModule


class TrendFollowingModule(AnalysisModule):
    def __init__(self, weight: float = 0.10):
        super().__init__("trend", weight=weight)

    def analyze(self, data: Dict) -> ModuleOutput:
        ohlcv = data["ohlcv"]
        current_price = data["current_price"]

        if len(ohlcv) < 30:
            return ModuleOutput(
                module_name=self.name, score=0.0,
                direction=self.score_to_direction(0),
                confidence=0.3,
                details={"reason": "insufficient_data"},
            )

        closes = ohlcv["close"]

        # MA — IPO 직후 종목용 짧은 MA + 표준 MA
        sma_10 = float(closes.rolling(10).mean().iloc[-1])
        sma_20 = float(closes.rolling(20).mean().iloc[-1])
        sma_50 = float(closes.rolling(min(50, len(closes))).mean().iloc[-1])
        sma_200 = (
            float(closes.rolling(200).mean().iloc[-1])
            if len(closes) >= 200 else None
        )

        # MA stack score (정렬 상태)
        stack_score = _compute_ma_stack(current_price, sma_10, sma_20, sma_50, sma_200)

        # Momentum acceleration — short ROC vs long ROC
        roc_5 = _safe_pct(closes, 5)
        roc_10 = _safe_pct(closes, 10)
        roc_20 = _safe_pct(closes, 20)
        momentum_score = _compute_momentum_score(roc_5, roc_10, roc_20)

        # ADX (trend strength)
        adx = _compute_adx(ohlcv, period=14)
        adx_factor = _adx_factor(adx)

        # Multi-timeframe consistency (1D vs 1W trend)
        mtf_score = _multi_timeframe_consistency(closes)

        # 종합 (-10 ~ +10)
        # MA stack: 가장 큰 weight
        composite = (
            stack_score * 0.4
            + momentum_score * 0.3
            + mtf_score * 0.3
        ) * adx_factor

        composite = float(np.clip(composite, -10, 10))

        return ModuleOutput(
            module_name=self.name,
            score=composite,
            direction=self.score_to_direction(composite),
            confidence=0.65,
            details={
                "sma_10": round(sma_10, 2),
                "sma_20": round(sma_20, 2),
                "sma_50": round(sma_50, 2),
                "sma_200": round(sma_200, 2) if sma_200 else None,
                "above_sma_50": current_price > sma_50,
                "above_sma_200": (sma_200 is not None and current_price > sma_200),
                "ma_stack_score": round(stack_score, 2),
                "roc_5d_pct": round(roc_5, 2),
                "roc_10d_pct": round(roc_10, 2),
                "roc_20d_pct": round(roc_20, 2),
                "momentum_score": round(momentum_score, 2),
                "adx": round(adx, 2),
                "adx_factor": round(adx_factor, 2),
                "mtf_score": round(mtf_score, 2),
            },
        )


def _safe_pct(closes: pd.Series, n: int) -> float:
    if len(closes) <= n:
        return 0.0
    base = closes.iloc[-n - 1]
    if base <= 0:
        return 0.0
    return float((closes.iloc[-1] - base) / base * 100)


def _compute_ma_stack(
    price: float,
    sma_10: float,
    sma_20: float,
    sma_50: float,
    sma_200: Optional[float],
) -> float:
    """MA 정렬 상태 → -10~+10 점수.

    강한 상승: price > sma_10 > sma_20 > sma_50 > sma_200
    강한 하락: 반대
    """
    score = 0
    # 가격 vs 각 MA
    if price > sma_10: score += 1
    else: score -= 1
    if price > sma_20: score += 1.5
    else: score -= 1.5
    if price > sma_50: score += 2
    else: score -= 2
    if sma_200 is not None:
        if price > sma_200: score += 2.5
        else: score -= 2.5

    # MA 정렬 (cross 상태)
    if sma_10 > sma_20: score += 1
    if sma_20 > sma_50: score += 1
    if sma_200 is not None and sma_50 > sma_200: score += 1

    return float(np.clip(score, -10, 10))


def _compute_momentum_score(
    roc_5: float,
    roc_10: float,
    roc_20: float,
) -> float:
    """모멘텀 가속/감속 — short > long이면 가속."""
    # 가속: 단기 변화율 > 장기 변화율 / 분자 평탄화
    acceleration = (roc_5 / 5) - (roc_20 / 20)  # 일별 평균 비교

    # 절대 ROC도 가중
    abs_score = np.tanh(roc_10 / 5) * 5  # ±5 범위로 압축

    return float(np.clip(abs_score + acceleration, -10, 10))


def _compute_adx(ohlcv: pd.DataFrame, period: int = 14) -> float:
    """ADX — trend strength (0~100)."""
    if len(ohlcv) < period * 2:
        return 0.0
    high = ohlcv["high"]
    low = ohlcv["low"]
    close = ohlcv["close"]

    plus_dm = (high.diff()).where((high.diff() > -low.diff()) & (high.diff() > 0), 0)
    minus_dm = (-low.diff()).where((-low.diff() > high.diff()) & (low.diff() < 0), 0)

    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(period).mean()

    last = adx.iloc[-1]
    return float(last) if pd.notna(last) else 0.0


def _adx_factor(adx: float) -> float:
    """ADX 25+ = trending → factor 1.0~1.5
    ADX < 20 = sideways → factor 0.5 (시그널 약화)
    """
    if adx >= 40:
        return 1.5
    if adx >= 25:
        return 1.0 + (adx - 25) / 30  # 1.0~1.5
    if adx >= 20:
        return 0.8
    return 0.5


def _multi_timeframe_consistency(closes: pd.Series) -> float:
    """1D vs 1W vs 1M 추세 일관성.

    각 timeframe의 마지막 N개의 평균 vs 그 이전 N개 평균 비교.
    """
    score = 0.0

    # 1D: 5일 vs 그 직전 5일
    if len(closes) >= 10:
        recent_5 = closes.iloc[-5:].mean()
        prev_5 = closes.iloc[-10:-5].mean()
        if prev_5 > 0:
            ch = (recent_5 - prev_5) / prev_5
            score += np.tanh(ch * 20) * 3

    # 1W: 5일 vs 30일 (weekly trend)
    if len(closes) >= 30:
        recent = closes.iloc[-5:].mean()
        long = closes.iloc[-30:].mean()
        if long > 0:
            ch = (recent - long) / long
            score += np.tanh(ch * 10) * 3

    # 1M: 20일 vs 60일
    if len(closes) >= 60:
        recent = closes.iloc[-20:].mean()
        long = closes.iloc[-60:].mean()
        if long > 0:
            ch = (recent - long) / long
            score += np.tanh(ch * 5) * 4

    return float(np.clip(score, -10, 10))

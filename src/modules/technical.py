"""Module 1: Technical Analysis.

핵심 통찰:
- 옵션 strike가 지지/저항 ($5 단위)
- 어림수 가격 사용 금지
- 지지/저항은 OI 큰 strikes에서 형성
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..types import ModuleOutput
from .base import AnalysisModule


def _wma(series, span):
    """Weighted moving average (선형 가중 — 최근일수록 큰 가중치)."""
    if len(series) < span:
        return float('nan')
    weights = np.arange(1, span + 1)
    # rolling apply는 느리니까 마지막 window만 계산
    window = series.iloc[-span:].values
    return float((window * weights).sum() / weights.sum())


def _trend_score_from_ma(price, ma_short, ma_mid, ma_long):
    """MA family에 무관한 trend score (-5~+5).

    공식 (기존 SMA score와 동일):
      현재가 > 200ma → +3 (큰 추세)
      현재가 > 50ma → +2
      현재가 > 20ma → +1
      20ma > 50ma → +2 (단기 추세)
      50ma > 200ma → +2 (장기 추세)
    """
    score = 0
    if price > ma_long:
        score += 3
    if price > ma_mid:
        score += 2
    if price > ma_short:
        score += 1
    if ma_short > ma_mid:
        score += 2
    if ma_mid > ma_long:
        score += 2
    return min(score, 10) - 5


class TechnicalAnalysisModule(AnalysisModule):
    def __init__(self, ma_variant: str = "sma"):
        """ma_variant: 'sma' | 'ema' | 'wma' | 'ensemble' — trend_score 공식 선택.

        backtest에서 EMA/WMA 효과 비교 가능. 기본은 기존 SMA 동작.
        """
        super().__init__("technical", weight=0.15)
        self.ma_variant = ma_variant

    def analyze(self, data: Dict) -> ModuleOutput:
        prices = data['ohlcv']
        option_strikes = data.get('option_strikes', [])
        current_price = prices['close'].iloc[-1]
        close = prices['close']

        sma_20 = close.rolling(20).mean().iloc[-1]
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1]
        ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        ema_50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        ema_200 = close.ewm(span=200, adjust=False).mean().iloc[-1]
        wma_20 = _wma(close, 20)
        wma_50 = _wma(close, 50)
        wma_200 = _wma(close, 200)

        # backtest variant override
        v = data.get('_ma_variant') or self.ma_variant

        rsi = self._compute_rsi(prices['close'], 14)
        macd, signal = self._compute_macd(prices['close'])
        bb_position = self._compute_bb_position(prices['close'])

        support_levels = sorted(
            [s for s in option_strikes if s < current_price],
            reverse=True,
        )[:3]
        resistance_levels = sorted([s for s in option_strikes if s > current_price])[:3]

        trend_score = self._compute_trend_score_variant(
            current_price, v,
            sma_20, sma_50, sma_200,
            ema_20, ema_50, ema_200,
            wma_20, wma_50, wma_200,
        )
        momentum_score = self._compute_momentum_score(rsi, macd, signal)

        composite_score = (trend_score + momentum_score) / 2

        return ModuleOutput(
            module_name=self.name,
            score=composite_score,
            direction=self.score_to_direction(composite_score),
            confidence=0.7,
            details={
                'sma_20': sma_20,
                'sma_50': sma_50,
                'sma_200': sma_200,
                'ema_20': ema_20,
                'ema_50': ema_50,
                'ema_200': ema_200,
                'wma_20': wma_20,
                'wma_50': wma_50,
                'wma_200': wma_200,
                'ma_variant': v,
                'rsi': rsi,
                'macd': macd,
                'bb_position': bb_position,
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'trend_score': trend_score,
                'momentum_score': momentum_score,
            },
        )

    def _compute_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return (100 - (100 / (1 + rs))).iloc[-1]

    def _compute_macd(self, prices, fast=12, slow=26, signal=9):
        exp1 = prices.ewm(span=fast, adjust=False).mean()
        exp2 = prices.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd.iloc[-1], signal_line.iloc[-1]

    def _compute_bb_position(self, prices, period=20, std=2):
        sma = prices.rolling(period).mean()
        std_dev = prices.rolling(period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return (prices.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1])

    def _compute_trend_score(self, price, sma_20, sma_50, sma_200):
        """Legacy SMA-only trend score. Kept for backwards-compat callers."""
        return _trend_score_from_ma(price, sma_20, sma_50, sma_200)

    def _compute_trend_score_variant(
        self, price, variant,
        sma_20, sma_50, sma_200,
        ema_20, ema_50, ema_200,
        wma_20, wma_50, wma_200,
    ):
        """variant: 'sma' | 'ema' | 'wma' | 'ensemble'.

        ensemble = (SMA + EMA + WMA) 평균 — 같은 가중치.
        """
        if variant == 'ema':
            return _trend_score_from_ma(price, ema_20, ema_50, ema_200)
        if variant == 'wma':
            return _trend_score_from_ma(price, wma_20, wma_50, wma_200)
        if variant == 'ensemble':
            s = _trend_score_from_ma(price, sma_20, sma_50, sma_200)
            e = _trend_score_from_ma(price, ema_20, ema_50, ema_200)
            w = _trend_score_from_ma(price, wma_20, wma_50, wma_200)
            return (s + e + w) / 3.0
        # default: 기존 SMA
        return _trend_score_from_ma(price, sma_20, sma_50, sma_200)

    def _compute_momentum_score(self, rsi, macd, signal):
        score = 0
        if rsi < 30:
            score += 3
        elif rsi > 70:
            score -= 3
        elif rsi < 45:
            score += 1
        elif rsi > 60:
            score -= 1

        if macd > signal:
            score += 2
        else:
            score -= 2

        return max(-5, min(5, score))

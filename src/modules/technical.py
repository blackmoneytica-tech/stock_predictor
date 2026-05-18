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


class TechnicalAnalysisModule(AnalysisModule):
    def __init__(self):
        super().__init__("technical", weight=0.15)

    def analyze(self, data: Dict) -> ModuleOutput:
        prices = data['ohlcv']
        option_strikes = data.get('option_strikes', [])
        current_price = prices['close'].iloc[-1]

        sma_20 = prices['close'].rolling(20).mean().iloc[-1]
        sma_50 = prices['close'].rolling(50).mean().iloc[-1]
        sma_200 = prices['close'].rolling(200).mean().iloc[-1]
        ema_20 = prices['close'].ewm(span=20, adjust=False).mean().iloc[-1]
        ema_50 = prices['close'].ewm(span=50, adjust=False).mean().iloc[-1]

        rsi = self._compute_rsi(prices['close'], 14)
        macd, signal = self._compute_macd(prices['close'])
        bb_position = self._compute_bb_position(prices['close'])

        support_levels = sorted(
            [s for s in option_strikes if s < current_price],
            reverse=True,
        )[:3]
        resistance_levels = sorted([s for s in option_strikes if s > current_price])[:3]

        trend_score = self._compute_trend_score(current_price, sma_20, sma_50, sma_200)
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
        score = 0
        if price > sma_200:
            score += 3
        if price > sma_50:
            score += 2
        if price > sma_20:
            score += 1
        if sma_20 > sma_50:
            score += 2
        if sma_50 > sma_200:
            score += 2
        return min(score, 10) - 5  # -5 ~ +5

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

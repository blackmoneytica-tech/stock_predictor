"""Module 7: Mean Reversion Statistics.

핵심 통계 (대화에서 검증):
- -9% 하락 후 다음 거래일 평균 반등 +0.8%, 확률 55%
- Parabolic +30% 후 -10~15% retracement 평균
- Max Pain 미달 후 다음 월요일 평균 +4.5% 반등
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..types import ModuleOutput
from .base import AnalysisModule


class MeanReversionModule(AnalysisModule):
    def __init__(self):
        super().__init__("mean_reversion", weight=0.10)

    def analyze(self, data: Dict) -> ModuleOutput:
        prices = data['ohlcv']['close']

        sma_20 = prices.rolling(20).mean().iloc[-1]
        std_20 = prices.rolling(20).std().iloc[-1]
        current = prices.iloc[-1]
        z_score = (current - sma_20) / std_20 if std_20 > 0 else 0

        recent_return = (current - prices.iloc[-2]) / prices.iloc[-2]
        return_1m = (
            (current - prices.iloc[-21]) / prices.iloc[-21]
            if len(prices) >= 21 else 0
        )

        max_pain_miss = data.get('last_friday_max_pain_missed', False)

        score = 0

        if z_score < -2:
            score += 5  # Oversold
        elif z_score > 2:
            score -= 5  # Overbought
        else:
            score += -z_score * 2

        # -9% 이상 하락 후 반등 베팅
        if recent_return < -0.09:
            score += 3
        elif recent_return > 0.09:
            score -= 3

        # Parabolic +30% 후 차익실현 우려
        if return_1m > 0.30:
            score -= 3
        elif return_1m < -0.30:
            score += 2

        # Max Pain miss 후 월요일 반등
        if max_pain_miss:
            score += 2

        score = np.clip(score, -10, 10)

        return ModuleOutput(
            module_name=self.name,
            score=score,
            direction=self.score_to_direction(score),
            confidence=0.6,
            details={
                'z_score': z_score,
                'recent_return_pct': recent_return * 100,
                'return_1m_pct': return_1m * 100,
                'reversion_probability': self._reversion_prob(z_score, recent_return),
                'max_pain_miss_boost': max_pain_miss,
                'parabolic_flag': return_1m > 0.30,
            },
        )

    def _reversion_prob(self, z_score, recent_return):
        if recent_return < -0.09 and z_score < -1.5:
            return 0.55  # 백테스트 기반
        if recent_return > 0.09 and z_score > 1.5:
            return 0.50
        return 0.4

    def get_dynamic_weight(self, context):
        if abs(context.get('recent_drop_pct', 0)) > 0.08:
            return self.weight * 2.0
        return self.weight

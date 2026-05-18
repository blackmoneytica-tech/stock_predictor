"""Module 8: Short Squeeze Potential."""
from __future__ import annotations

from typing import Dict

from ..types import ModuleOutput
from .base import AnalysisModule


class ShortSqueezeModule(AnalysisModule):
    def __init__(self):
        super().__init__("short_squeeze", weight=0.05)

    def analyze(self, data: Dict) -> ModuleOutput:
        short_pct = data.get('short_interest_pct', 0)
        days_to_cover = data.get('days_to_cover', 0)
        borrow_rate = data.get('borrow_rate', 0)
        si_momentum = data.get('short_interest_30d_change', 0)

        squeeze_score = (
            min(short_pct / 0.20, 1) * 4
            + min(days_to_cover / 5, 1) * 3
            + min(borrow_rate / 0.20, 1) * 3
        )

        if si_momentum > 0.5:
            squeeze_score += 2

        score = squeeze_score - 5  # -5 ~ +5

        return ModuleOutput(
            module_name=self.name,
            score=score,
            direction=self.score_to_direction(score),
            confidence=0.5,
            details={
                'squeeze_score': squeeze_score,
                'short_pct': short_pct,
                'days_to_cover': days_to_cover,
                'borrow_rate': borrow_rate,
                'squeeze_potential': self._squeeze_potential(squeeze_score),
            },
        )

    def _squeeze_potential(self, score):
        if score >= 8:
            return 'high'
        if score >= 6:
            return 'moderate'
        if score >= 4:
            return 'low'
        return 'minimal'

    def get_dynamic_weight(self, context):
        if context.get('short_interest_pct', 0) > 0.15:
            return self.weight * 3.0
        return self.weight

"""Module 4: Macro Correlation — 매크로 데이터 lag 1~3일 반영.

핵심 통찰: 매크로 lag effect (PPI/CPI/NFP)
  T+0: 30% / T+1: 40% / T+2: 25% / T+3: 5%
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict

import numpy as np

from ..types import ModuleOutput
from .base import AnalysisModule


# SPEC §0 §4 — lag effect 가중치
LAG_WEIGHTS = {0: 0.30, 1: 0.40, 2: 0.25, 3: 0.05}


class MacroCorrelationModule(AnalysisModule):
    def __init__(self):
        super().__init__("macro", weight=0.20)

    def analyze(self, data: Dict) -> ModuleOutput:
        fed_dovish = data.get('fed_dovish_score', 0)  # -5 ~ +5
        yield_score = data.get('yield_score', 0)
        risk_on = data.get('risk_on_score', 0)
        macro_betas = data.get('macro_betas', {})

        base_score = (
            fed_dovish * macro_betas.get('fed', 0.5) * 0.4
            + yield_score * macro_betas.get('yield', -0.5) * 0.3
            + risk_on * macro_betas.get('risk', 0.5) * 0.3
        )

        lag_effects = self._compute_lag_effects(data.get('recent_macro_releases', []))

        # 2026-05-18 신규: Sector breadth + VIX TS + HYG/LQD score 통합
        # alert/screener_macro.pine의 11 섹터 + VIX TS + credit spread 시그널
        breadth_score = 0.0
        breadth = data.get('macro_breadth', {})
        if breadth:
            try:
                from ..data.sector_macro import macro_breadth_score
                ticker = data.get('ticker', '')
                breadth_score = macro_breadth_score(
                    ticker, breadth, betas=macro_betas,
                )
            except Exception:
                pass

        # 60% FRED + 40% sector breadth (sector가 더 즉시적)
        score = (base_score + lag_effects) * 0.6 + breadth_score * 0.4
        score = np.clip(score, -10, 10)

        return ModuleOutput(
            module_name=self.name,
            score=score,
            direction=self.score_to_direction(score),
            confidence=0.7,  # breadth 추가로 confidence 0.65 → 0.70
            details={
                'base_score': base_score,
                'lag_effects': lag_effects,
                'breadth_score': breadth_score,
                'sector_mode': breadth.get('mode', 'UNKNOWN'),
                'sector_avg_pct': breadth.get('sector_avg', 0),
                'risk_off_score': breadth.get('risk_off_score', 0),
                'vix_term': breadth.get('vix_term'),
                'hyg_lqd': breadth.get('hyg_lqd'),
            },
        )

    def _compute_lag_effects(self, recent_releases):
        """매크로 발표 1~3일 lag effect 누적."""
        total_lag = 0
        for release in recent_releases:
            days_ago = (datetime.now().date() - release['date']).days
            if days_ago in LAG_WEIGHTS:
                total_lag += release['surprise'] * LAG_WEIGHTS[days_ago]
        return total_lag

    def get_dynamic_weight(self, context):
        if context.get('fomc_within_days', 999) <= 1:
            return self.weight * 1.5
        return self.weight

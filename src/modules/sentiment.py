"""Module 3: Sentiment Analysis — VIX / P/C / Analyst PT 기반."""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..types import ModuleOutput
from .base import AnalysisModule


class SentimentModule(AnalysisModule):
    def __init__(self):
        super().__init__("sentiment", weight=0.10)

    def analyze(self, data: Dict) -> ModuleOutput:
        vix = data.get('vix', 18)
        vix_30d_avg = data.get('vix_30d_avg', 18)
        vix_30d_std = data.get('vix_30d_std', 3)
        pc_ratio = data.get('put_call_ratio', 0.85)
        pt_30d = data.get('analyst_pt_30d_avg', 100)
        pt_60d = data.get('analyst_pt_60d_avg', 100)

        fear_score = (vix - vix_30d_avg) / vix_30d_std if vix_30d_std > 0 else 0
        pt_momentum = (pt_30d - pt_60d) / pt_60d if pt_60d > 0 else 0
        pc_zscore = (pc_ratio - 0.85) / 0.15

        base = -fear_score * 3 + -pc_zscore * 2 + pt_momentum * 50

        # 2026-05-18 신규: News sentiment (Finnhub /company-news)
        news_score = data.get('news_sentiment_score', 0.0)
        news_n = data.get('news_sentiment_n', 0)

        # 2026-05-18 신규: Options unusual activity flow score
        unusual_score = data.get('unusual_options_score', 0.0)

        # 가중 결합 — base 40%, news 30% (이벤트 driven), unusual 30%
        score = base * 0.4 + news_score * 0.3 + unusual_score * 0.3
        score = float(np.clip(score, -10, 10))

        return ModuleOutput(
            module_name=self.name,
            score=score,
            direction=self.score_to_direction(score),
            confidence=0.65,  # news + unusual 추가로 conf 0.6 → 0.65
            details={
                'vix': vix,
                'fear_score': fear_score,
                'pt_momentum_pct': pt_momentum * 100,
                'pc_ratio': pc_ratio,
                'news_score': news_score,
                'news_n_items': news_n,
                'unusual_score': unusual_score,
                'unusual_direction': data.get('unusual_options_direction', 'neutral'),
            },
        )

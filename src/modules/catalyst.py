"""Module 5: Catalyst Calendar — Sell-the-news 80% base rate 보정.

핵심 통찰: +30% 사전 랠리 후 카탈리스트 통과 = sell-news 80% 확률.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict

import numpy as np

from ..types import ModuleOutput
from .base import AnalysisModule


class CatalystCalendarModule(AnalysisModule):
    def __init__(self):
        super().__init__("catalyst", weight=0.15)

    def analyze(self, data: Dict) -> ModuleOutput:
        events = data.get('upcoming_events', [])
        horizon_days = data.get('horizon_days', 5)
        pre_rally_pct = data.get('pre_event_rally_pct', 0)

        score = 0
        relevant_events = []

        for event in events:
            days_until = (event['date'] - datetime.now().date()).days
            if 0 <= days_until <= horizon_days:
                decay = 1 / (1 + days_until * 0.2)

                # Sell-the-news 보정: +30% 사전 랠리 후 = bear 80% base rate
                if pre_rally_pct > 0.30:
                    expected_direction = -event.get('expected_impact', 0) * 0.8
                else:
                    expected_direction = (
                        event.get('expected_direction', 0)
                        * event.get('expected_impact', 0)
                    )

                score += expected_direction * decay
                relevant_events.append({
                    'date': (
                        event['date'].isoformat()
                        if hasattr(event['date'], 'isoformat')
                        else str(event['date'])
                    ),
                    'type': event['type'],
                    'days_until': days_until,
                    'expected_impact': event.get('expected_impact', 0),
                })

        score = np.clip(score, -10, 10)

        return ModuleOutput(
            module_name=self.name,
            score=score,
            direction=self.score_to_direction(score),
            confidence=0.55,
            details={
                'events_count': len(relevant_events),
                'events': relevant_events,
                'pre_rally_pct': pre_rally_pct,
                'sell_news_risk': pre_rally_pct > 0.30,
            },
        )

    def get_dynamic_weight(self, context):
        if context.get('event_within_days', 999) <= 2:
            return self.weight * 1.5
        return self.weight

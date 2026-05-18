"""Module 6: Insider & Smart Money.

핵심 통찰:
- 인사이더 ceiling 가격 = 매도 집중 = 강한 저항
- 매도 0 + 매수 폭증 = 강한 강세
- 매수 0 + 매도 폭증 = 강한 약세 (예: CRCL 6개월 매도 112건 / 매수 0)
"""
from __future__ import annotations

from typing import Dict

from ..types import ModuleOutput
from .base import AnalysisModule


class InsiderSmartMoneyModule(AnalysisModule):
    def __init__(self):
        super().__init__("insider", weight=0.10)

    def analyze(self, data: Dict) -> ModuleOutput:
        insider_buys = data.get('insider_buys_30d', 0)
        insider_sells = data.get('insider_sells_30d', 0)
        insider_buys_6m = data.get('insider_buys_6m', 0)
        insider_sells_6m = data.get('insider_sells_6m', 0)
        recent_sells_prices = data.get('recent_sells_prices', [])
        recent_buys_prices = data.get('recent_buys_prices', [])

        # 점수 산출
        # 2026-05-16 backtest fix #2 — CRCL 백테스트 4주에서 -8 고정이 다른 모듈 묻음
        # 6개월 매수 0 + 매도 50+ = 강한 약세지만 -8은 압도적 → -4로 완화
        # (가중치 0.10 × -4 = -0.4 영향. 다른 시그널 살아남음)
        if insider_buys_6m == 0 and insider_sells_6m > 50:
            score = -4  # 강한 약세 (완화)
        elif insider_buys + insider_sells > 0:
            net_ratio = (insider_buys - insider_sells) / (insider_buys + insider_sells)
            if net_ratio < -0.5:
                score = -5
            elif net_ratio > 0.5:
                score = 5
            else:
                score = net_ratio * 4
        else:
            score = 0

        insider_ceiling = max(recent_sells_prices) if recent_sells_prices else None
        insider_floor = min(recent_buys_prices) if recent_buys_prices else None

        ceiling_volume = sum(
            1 for p in recent_sells_prices
            if insider_ceiling and abs(p - insider_ceiling) < 2
        )
        ceiling_strength = (
            ceiling_volume / len(recent_sells_prices) if recent_sells_prices else 0
        )

        return ModuleOutput(
            module_name=self.name,
            score=score,
            direction=self.score_to_direction(score),
            confidence=0.75,
            details={
                'insider_ceiling': insider_ceiling,
                'insider_floor': insider_floor,
                'ceiling_strength': ceiling_strength,
                'buys_6m': insider_buys_6m,
                'sells_6m': insider_sells_6m,
                'net_direction': (
                    'selling' if insider_sells_6m > insider_buys_6m else 'buying'
                ),
            },
        )

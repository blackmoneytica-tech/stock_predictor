"""Module 2: Options Flow Analysis — 가장 중요한 모듈.

핵심 통찰:
- HV > IV → 옵션 underpriced → Protective Put 매수 적기
- IV Rank 낮음 → 변동성 보호 비용 저렴
- Max Pain 자석 효과 적중률 35~60%
- Implied Move = 시장 예측 변동성 = 가장 정량적 예측
- Max Pain 미달 → 다음 월요일 +4.5% 반등 (백테스트)
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict

import numpy as np

from ..types import ModuleOutput
from .base import AnalysisModule


class OptionsFlowModule(AnalysisModule):
    def __init__(self):
        super().__init__("options", weight=0.20)

    def analyze(self, data: Dict) -> ModuleOutput:
        options_chain = data['options_chain']
        current_price = data['current_price']
        target_expiration = data['target_expiration']
        # backtest 호환 — as_of_date 주입 시 그 시점 기준 days-to-exp 계산
        reference = data.get('as_of_date')

        max_pain = self.calculate_max_pain(options_chain, target_expiration)
        max_pain_distance = (max_pain - current_price) / current_price

        iv = self._get_atm_iv(options_chain, current_price, target_expiration)
        days_to_exp = self._days_to_expiration(target_expiration, reference)
        # implied_move ≥ 0 보장 (음수 days_to_exp 방어)
        implied_move = current_price * iv * np.sqrt(max(0, days_to_exp) / 252)

        pc_ratio = self._compute_pc_ratio(options_chain, target_expiration)

        iv_rank = data.get('iv_rank', 0.5)
        iv_percentile = data.get('iv_percentile', 0.5)
        hv = data.get('historic_volatility', iv)
        hv_iv_ratio = hv / iv if iv > 0 else 1.0

        skew = self._compute_skew(options_chain, current_price, target_expiration)

        score = self._compute_options_score(
            max_pain_distance, pc_ratio, iv_rank, skew, hv_iv_ratio,
        )

        return ModuleOutput(
            module_name=self.name,
            score=score,
            direction=self.score_to_direction(score),
            confidence=0.75,
            details={
                'max_pain': max_pain,
                'max_pain_distance_pct': max_pain_distance * 100,
                'implied_move': implied_move,
                'implied_move_pct': implied_move / current_price * 100,
                'put_call_ratio': pc_ratio,
                'iv': iv,
                'iv_rank': iv_rank,
                'iv_percentile': iv_percentile,
                'hv': hv,
                'hv_iv_ratio': hv_iv_ratio,
                'skew': skew,
                'days_to_expiration': days_to_exp,
                'strikes': sorted(options_chain[target_expiration].keys()),
            },
        )

    def calculate_max_pain(self, options_chain, expiration) -> float:
        """Max Pain — 옵션 매도자에게 가장 유리한 가격 (call + put pain 최소)."""
        strikes = sorted(options_chain[expiration].keys())

        min_pain = float('inf')
        max_pain_strike = strikes[0]

        for strike in strikes:
            pain = 0
            for K in strikes:
                if strike > K:
                    pain += (strike - K) * options_chain[expiration][K].get('call_oi', 0) * 100
                if strike < K:
                    pain += (K - strike) * options_chain[expiration][K].get('put_oi', 0) * 100

            if pain < min_pain:
                min_pain = pain
                max_pain_strike = strike

        return max_pain_strike

    def _get_atm_iv(self, options_chain, current_price, expiration):
        strikes = list(options_chain[expiration].keys())
        atm_strike = min(strikes, key=lambda s: abs(s - current_price))
        return options_chain[expiration][atm_strike].get('iv', 0.5)

    def _days_to_expiration(self, expiration, reference=None):
        """as_of_date 기준 (백테스트) 또는 datetime.now() 기준 (실시간)."""
        if isinstance(expiration, str):
            expiration = datetime.fromisoformat(expiration)
        if reference is None:
            reference = datetime.now()
        elif hasattr(reference, "year") and not isinstance(reference, datetime):
            reference = datetime(reference.year, reference.month, reference.day)
        return (expiration - reference).days

    def _compute_pc_ratio(self, options_chain, expiration):
        call_oi = sum(opt.get('call_oi', 0) for opt in options_chain[expiration].values())
        put_oi = sum(opt.get('put_oi', 0) for opt in options_chain[expiration].values())
        return put_oi / call_oi if call_oi > 0 else 1.0

    def _compute_skew(self, options_chain, current_price, expiration):
        """OTM Put IV − OTM Call IV (가팔라질수록 콜시장 fear)."""
        otm_put_strike = current_price * 0.9
        otm_call_strike = current_price * 1.1

        chain = options_chain[expiration]
        put_iv = self._closest_strike_iv(chain, otm_put_strike, 'put_iv')
        call_iv = self._closest_strike_iv(chain, otm_call_strike, 'call_iv')

        return put_iv - call_iv

    def _closest_strike_iv(self, chain, target, iv_key):
        closest = min(chain.keys(), key=lambda s: abs(s - target))
        return chain[closest].get(iv_key, 0)

    def _compute_options_score(self, max_pain_dist, pc_ratio, iv_rank, skew, hv_iv_ratio):
        score = 0

        # Max Pain pulls price
        score += np.clip(max_pain_dist * 50, -3, 3)

        # P/C ratio: 낮으면 bullish
        score += np.clip((0.85 - pc_ratio) * 5, -2, 2)

        # IV Rank 낮으면 보호 비용 저렴 (보유자에겐 +)
        score += np.clip((0.3 - iv_rank) * 3, -1, 2)

        # Skew 가파르면 콜시장 fear
        score += np.clip(-skew * 20, -3, 0)

        # HV > IV (underpriced) → 헷지 적기
        if hv_iv_ratio > 1.2:
            score += 1

        return np.clip(score, -10, 10)

    def get_dynamic_weight(self, context):
        # 옵션 만기 직전 또는 이벤트 직전
        if context.get('event_within_days', 999) <= 2:
            return self.weight * 1.5
        return self.weight

"""Signal Aggregator — 8개 모듈 신호 Bayesian 결합.

핵심 원칙:
- 인지 편향 교정 (호재 자동 70% 가중치 금지)
- 동적 가중치 (이벤트 상황별)
- 75% 신뢰도 hard cap
- Sell-the-news + Parabolic + Max Pain miss 시나리오 보정
"""
from __future__ import annotations

from typing import Dict

import numpy as np

from ..types import ModuleOutput, Scenario


class SignalAggregator:
    # 2026-05-18 재분배 — 100 예측 backtest 결과 trend 11% bull 편향 강함
    # → trend 0.11 → 0.07, mean_reversion 0.06 → 0.08 (반등 catch 강화)
    # 검증된 매물대(76.7% bounce) + catalyst(46.2%) 가중 ↑
    BASE_WEIGHTS = {
        'technical': 0.10,
        'options': 0.17,          # 핵심 (Max Pain + IM)
        'sentiment': 0.05,
        'macro': 0.15,            # FRED + sector breadth
        'catalyst': 0.12,         # catalyst-active 46.2% 검증
        'insider': 0.06,
        'mean_reversion': 0.08,   # 반등 catch (0.06 → 0.08)
        'short_squeeze': 0.03,
        'demand_supply': 0.11,    # 매물대 76.7% bounce 검증 (0.10 → 0.11)
        'order_block': 0.06,      # ICT
        'trend': 0.07,            # bull 편향 완화 (0.11 → 0.07)
    }

    CONFIDENCE_HARD_CAP = 0.75

    def aggregate(self, modules: Dict[str, ModuleOutput], context: Dict):
        weights = self._get_dynamic_weights(modules, context)

        composite_score = sum(
            weights[name] * output.score
            for name, output in modules.items()
        )

        # 2026-05-18 재교정: composite_score 절대값을 confidence에 반영
        # (이전 confidence 0.6+ accuracy ≤ 0.5와 동일 → 신뢰도 분별력 없음)
        confidence = self._calculate_confidence(
            modules, composite_score, context.get('ticker', '') or '',
        )
        scenarios = self._generate_scenarios(modules, composite_score, context)

        return {
            'composite_score': composite_score,
            'directional_bias': self._directional_bias(composite_score),
            'confidence': confidence,
            'scenarios': scenarios,
            'weights_used': weights,
        }

    def _get_dynamic_weights(self, modules, context) -> Dict[str, float]:
        weights = self.BASE_WEIGHTS.copy()

        if context.get('event_within_days', 999) <= 2:
            weights['options'] *= 1.5
            weights['catalyst'] *= 1.5

        if abs(context.get('recent_drop_pct', 0)) > 0.08:
            weights['mean_reversion'] *= 2.0

        if context.get('fomc_within_days', 999) <= 1:
            weights['macro'] *= 1.5

        if context.get('short_interest_pct', 0) > 0.15:
            weights['short_squeeze'] *= 3.0

        # 2026-05-18 신규: Macro regime별 가중치 (v7 backtest 검증)
        # CHOPPY 56.7% (best) / STRONG_BULL 38.7% (worst)
        regime = context.get('macro_breadth_mode', 'CHOPPY')
        try:
            from .calibration import regime_weights
            mods = regime_weights.get_multipliers(regime)
            for k, mult in mods.items():
                if k in weights:
                    weights[k] *= mult
        except Exception:
            pass

        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _calculate_confidence(
        self,
        modules: Dict,
        composite_score: float = 0.0,
        ticker: str = "",
    ) -> float:
        """Confidence 재교정 v3 (2026-05-18 발견: 평균 50% 너무 낮음).

        backtest 데이터 분석:
          - conf 0.5-0.6 → acc 55% (best 구간)
          - conf 0.4-0.5 → acc 23%
          - conf 0.6-0.75 → acc 46%
        → conf 50-65% 가 의미있는 시그널 영역. 시스템은 normal case에서
        이 영역을 더 자주 출력하도록.

        v3 변경:
          - score_strength: |score|/1.5 (이전 /3.0) — score 1.5에서 max
          - 모든 모듈 |score|>2 agreement → |score|>1.5 (조금 완화)
          - base 0.05 추가 (default가 30% → 35%)
        """
        scores = [m.score for m in modules.values()]
        total = len(scores)
        if total == 0:
            return 0.4

        # Loose agreement — |score| > 1.5
        strong_pos = sum(1 for s in scores if s > 1.5)
        strong_neg = sum(1 for s in scores if s < -1.5)
        strong_total = strong_pos + strong_neg
        agreement = (
            max(strong_pos, strong_neg) / strong_total if strong_total > 0 else 0
        )

        # Score strength — composite_score 절대값 (완화: /1.5)
        score_strength = min(1.0, abs(composite_score) / 1.5)

        avg_conf = float(np.mean([m.confidence for m in modules.values()]))
        data_quality = sum(1 for m in modules.values() if m.confidence > 0.5) / total

        # base 0.05 + 4 components
        confidence = (
            0.05
            + agreement * 0.30
            + score_strength * 0.30
            + avg_conf * 0.20
            + data_quality * 0.15
        )

        # 2026-05-18 신규: Ticker fitness modifier (검증된 적합도)
        # MSFT/GOOGL 같은 0% 적합도 종목 → conf × 0.5
        if ticker:
            try:
                from .calibration import fitness_db
                confidence *= fitness_db.confidence_modifier(ticker)
            except Exception:
                pass

        return float(min(confidence, self.CONFIDENCE_HARD_CAP))

    def _directional_bias(self, score):
        if score > 3:
            return "strong_bull"
        if score > 1:
            return "bull"
        if score > -1:
            return "neutral"
        if score > -3:
            return "bear"
        return "strong_bear"

    def _generate_scenarios(self, modules, composite_score, context):
        """5-시나리오 생성 + sell-news / parabolic / max-pain-miss 보정."""
        # 2026-05-16 backtest 결과 fix #1 — base 분포 대칭화
        # 기존: mega_bull 5% / bull 20% / base 45% / bear 20% / crisis 10% (bias -1.7%)
        # 신규: 10/20/40/20/10 (대칭) → EV bias 0
        base_dist = {
            'mega_bull': 0.10,
            'bull': 0.20,
            'base': 0.40,
            'bear': 0.20,
            'crisis': 0.10,
        }

        if composite_score > 5:
            base_dist['mega_bull'] += 0.05
            base_dist['bull'] += 0.10
            base_dist['bear'] -= 0.07
            base_dist['crisis'] -= 0.08
        elif composite_score < -5:
            base_dist['crisis'] += 0.10
            base_dist['bear'] += 0.10
            base_dist['bull'] -= 0.10
            base_dist['mega_bull'] -= 0.10

        # Sell-the-news 자동 보정 (대화 검증 + 2026-05-18 CRCL 5/11~15 case 보정)
        pc_days = context.get('post_catalyst_within_days', 999)
        rally = context.get('pre_catalyst_rally_pct', 0)
        recent_drop = context.get('recent_drop_pct', 0)  # T-1 일봉 변화
        if pc_days <= 5 and rally > 0.15:
            decay = (
                1.0 if pc_days <= 1 else
                0.7 if pc_days <= 3 else
                0.4
            )
            # Cooldown: 직전일이 이미 큰 음수면 sell-news 1차 완료 → 추가 보정 약화
            if recent_drop <= -0.04:
                decay *= 0.4
            base_dist['bear'] += 0.15 * decay
            base_dist['bull'] -= 0.10 * decay
            base_dist['base'] -= 0.05 * decay

        # 2026-05-18 신규: 반등 시그널 (Mean Reversion bounce)
        # 직전일 -4% 이상 폭락 + 시장 양호 (macro mode != BEAR) → bull 시나리오 +5%, bear -5%
        # CRCL 5/12 -6.16% 후 5/13 +2.36% 같은 case
        if recent_drop <= -0.04:
            breadth = context.get('macro_breadth_mode', 'CHOPPY')
            # 시장이 BEAR/STRONG_BEAR가 아니면 반등 expected
            if breadth not in ('BEAR', 'STRONG_BEAR'):
                # 폭락 강도에 비례 (max 8%)
                bounce_strength = min(0.10, abs(recent_drop) * 1.0)
                base_dist['bull'] += bounce_strength * 0.5
                base_dist['mega_bull'] += bounce_strength * 0.2
                base_dist['bear'] -= bounce_strength * 0.5
                base_dist['crisis'] -= bounce_strength * 0.2

        # Earnings 발표 직전 (Finnhub beat_probability proxy 활용)
        # 다음 발표 ≤3일 + beat_probability ≥ 0.65 → bull tilt
        beat_proxy = context.get('beat_probability_proxy', 0.5)
        days_to_earn = context.get('days_to_earnings', 999)
        if days_to_earn <= 3 and beat_proxy >= 0.65:
            tilt = (beat_proxy - 0.5) * 0.3  # max 0.15
            base_dist['mega_bull'] += tilt * 0.5
            base_dist['bull'] += tilt * 0.5
            base_dist['bear'] -= tilt * 0.7
            base_dist['crisis'] -= tilt * 0.3
        elif days_to_earn <= 3 and beat_proxy <= 0.35:
            # miss 우려
            tilt = (0.5 - beat_proxy) * 0.3
            base_dist['bear'] += tilt * 0.7
            base_dist['crisis'] += tilt * 0.3
            base_dist['mega_bull'] -= tilt * 0.5
            base_dist['bull'] -= tilt * 0.5

        # Parabolic 보정
        if context.get('return_1m', 0) > 0.30:
            base_dist['bear'] += 0.10
            base_dist['mega_bull'] -= 0.05
            base_dist['bull'] -= 0.05

        # Max Pain miss 후 월요일 반등 보정
        if context.get('last_friday_max_pain_missed'):
            base_dist['bull'] += 0.10
            base_dist['bear'] -= 0.05
            base_dist['base'] -= 0.05

        # 음수 방지 + normalize
        base_dist = {k: max(0, v) for k, v in base_dist.items()}
        total = sum(base_dist.values())
        base_dist = {k: v / total for k, v in base_dist.items()}

        current = context['current_price']
        implied_move = context.get('implied_move_5d', current * 0.1)

        return [
            Scenario(
                'mega_bull', base_dist['mega_bull'],
                price_range=(current + implied_move * 1.5, current + implied_move * 2.5),
                expected_value=current + implied_move * 2,
                triggers=['모든 호재 동시 발현'],
            ),
            Scenario(
                'bull', base_dist['bull'],
                price_range=(current + implied_move * 0.5, current + implied_move * 1.5),
                expected_value=current + implied_move,
                triggers=['주요 호재 1개 발현'],
            ),
            Scenario(
                'base', base_dist['base'],
                price_range=(current - implied_move * 0.5, current + implied_move * 0.5),
                expected_value=current,
                triggers=['예상대로 진행'],
            ),
            Scenario(
                'bear', base_dist['bear'],
                price_range=(current - implied_move * 1.5, current - implied_move * 0.5),
                expected_value=current - implied_move,
                triggers=['약세 시그널 발현'],
            ),
            Scenario(
                'crisis', base_dist['crisis'],
                price_range=(current - implied_move * 2.5, current - implied_move * 1.5),
                expected_value=current - implied_move * 2,
                triggers=['Black Swan'],
            ),
        ]

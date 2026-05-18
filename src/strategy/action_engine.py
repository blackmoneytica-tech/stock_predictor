"""Action Engine — 매도 트리거 / 손절 / 헷지 추천 생성.

2026-05-18 재설계: Confluence-based (백테스트 3372 levels 검증)
- 단일 source 가격 ($1 단위 옵션 strike) 대신 **여러 source confluence**
- 검증된 weight: POC/vol_profile 100% bounce → 최강
- 매도: top 3 confluence supply zone
- 손절: top 2 confluence demand zone break
"""
from __future__ import annotations

from typing import Dict, List

from ..types import HedgeRecommendation, SellTrigger


class ActionEngine:
    def generate_actions(
        self,
        prediction: Dict,
        modules: Dict,
        current_price: float,
        data: Dict = None,
    ) -> Dict:
        # 2026-05-18 신규: Confluence-based (백테스트 검증)
        confluence_zones = self._compute_confluence_zones(
            modules, current_price, data or {},
        )

        # 매도/손절 = confluence 기반만 (의미없는 fallback 제거)
        sell_triggers = self._sell_from_confluence(confluence_zones, current_price)
        stop_loss = self._stops_from_confluence(confluence_zones, current_price)
        # NOTE: confluence 없으면 빈 출력. fallback ($1 단위 옵션 strike)은
        # 무의미하므로 제거. UI에서 "가까운 강한 시그널 없음" 표시.

        hedge = self._recommend_hedges(current_price, modules, prediction)

        return {
            'sell_triggers': sell_triggers,
            'stop_loss': stop_loss,
            'hedge_recommendations': hedge,
            'confluence_zones': confluence_zones,
        }

    def _compute_confluence_zones(
        self,
        modules: Dict,
        current_price: float,
        data: Dict,
    ) -> Dict:
        """모든 source에서 가격대 후보 추출 → cluster → top 권고."""
        from .confluence import (
            extract_all_levels, cluster_levels, rank_top_clusters,
        )
        levels = extract_all_levels(
            ohlcv=data.get('ohlcv'),
            current_price=current_price,
            options_chain=data.get('options_chain'),
            target_expiration=data.get('target_expiration'),
            insider_data={
                "recent_sells_prices": modules['insider'].details.get('recent_sells_prices', [])
                if hasattr(modules['insider'].details.get('recent_sells_prices', []), '__iter__')
                else [],
                "recent_buys_prices": modules['insider'].details.get('recent_buys_prices', [])
                if hasattr(modules['insider'].details.get('recent_buys_prices', []), '__iter__')
                else [],
            },
        )
        # Max Pain 추가
        max_pain = modules['options'].details.get('max_pain')
        if max_pain:
            levels.append({
                "source": "max_pain",
                "low": max_pain, "high": max_pain, "price": max_pain,
                "strength": 5.0,
                "side": "magnet",
            })
        # Insider ceiling
        ic = modules['insider'].details.get('insider_ceiling')
        if ic and ic > 0:
            levels.append({
                "source": "insider_ceiling",
                "low": ic, "high": ic, "price": ic,
                "strength": 5.0,
                "side": "above" if ic > current_price else "below",
            })

        clusters = cluster_levels(levels, current_price, tolerance_pct=3.0)
        return {
            'all_clusters': clusters,
            'supply': rank_top_clusters(
                clusters, current_price, side="above", top_k=5,
                max_dist_pct=25.0,
            ),
            'demand': rank_top_clusters(
                clusters, current_price, side="below", top_k=5,
                max_dist_pct=25.0,
            ),
        }

    def _sell_from_confluence(
        self,
        cz: Dict,
        current_price: float,
    ) -> List[SellTrigger]:
        """Top supply confluence zones → 매도 트리거."""
        out = []
        # 분할 매도 가중치
        weights = [0.30, 0.35, 0.35]
        for i, c in enumerate(cz.get('supply', [])[:3]):
            n_src = c['n_sources']
            src_list = ", ".join(sorted(set(c['sources'])))
            label = f"{n_src} source confluence: {src_list}"
            # cluster low를 매도가로 (가장 보수적인 시점)
            out.append(SellTrigger(
                price=round(c['low'], 2),
                action=f"sell_{int(weights[i]*100)}pct",
                reason=label,
            ))
        return out

    def _stops_from_confluence(
        self,
        cz: Dict,
        current_price: float,
    ) -> List[SellTrigger]:
        """Demand confluence break 시 손절 트리거 (zone low 아래 1%)."""
        out = []
        weights = [0.50, 0.50]
        for i, c in enumerate(cz.get('demand', [])[:2]):
            n_src = c['n_sources']
            src_list = ", ".join(sorted(set(c['sources'])))
            # Zone low보다 1% 아래 = break 확인 시점
            stop_price = round(c['low'] * 0.99, 2)
            label = f"{n_src} source confluence break: {src_list}"
            out.append(SellTrigger(
                price=stop_price,
                action=f"sell_{int(weights[i]*100)}pct",
                reason=label,
            ))
        return out

    def _compute_sell_triggers(
        self,
        current: float,
        strikes: List[float],
        insider_ceiling: float | None,
    ) -> List[SellTrigger]:
        """$5 strike 단위 분할 매도 가격."""
        above_strikes = sorted([s for s in strikes if s > current])[:4]

        triggers = []
        weights = [0.20, 0.25, 0.25, 0.15]

        for i, strike in enumerate(above_strikes):
            reason = "option_strike"
            if insider_ceiling and abs(strike - insider_ceiling) <= 2:
                reason = f"insider_ceiling_${insider_ceiling:.0f}"

            triggers.append(SellTrigger(
                price=strike,
                action=f'sell_{int(weights[i] * 100)}pct',
                reason=reason,
            ))

        return triggers

    def _compute_stop_loss(
        self,
        current: float,
        support_levels: List[float],
    ) -> List[SellTrigger]:
        """기술적 지지선 기반 손절."""
        below_supports = sorted(
            [s for s in support_levels if s < current],
            reverse=True,
        )[:4]

        stops = []
        weights = [0.30, 0.50, 0.70, 0.90]

        for i, support in enumerate(below_supports):
            stops.append(SellTrigger(
                price=support,
                action=f'sell_{int(weights[i] * 100)}pct',
                reason=f'support_break_${support:.0f}',
            ))

        return stops

    def _recommend_hedges(
        self,
        current: float,
        modules: Dict,
        prediction: Dict,
    ) -> List[HedgeRecommendation]:
        hedges = []

        iv_rank = modules['options'].details.get('iv_rank', 0.5)
        hv_iv_ratio = modules['options'].details.get('hv_iv_ratio', 1.0)

        # IV Rank 낮음 + HV>IV → Protective Put
        if iv_rank < 0.30 and hv_iv_ratio > 1.0:
            put_strike = round(current * 0.92 / 5) * 5
            hedges.append(HedgeRecommendation(
                type='protective_put',
                strike=put_strike,
                expiration_days=90,
                rationale=(
                    f'IV Rank {iv_rank:.0%} (낮음) + HV>IV = 옵션 underpriced, '
                    f'헷지 비용 저렴'
                ),
            ))

        # 약세 시그널 → Collar
        if prediction.get('directional_bias') in ['bear', 'strong_bear']:
            call_strike = round(current * 1.10 / 5) * 5
            put_strike = round(current * 0.92 / 5) * 5
            hedges.append(HedgeRecommendation(
                type='collar',
                call_strike=call_strike,
                put_strike=put_strike,
                expiration_days=90,
                rationale=f'${put_strike}~${call_strike} 박스 lock, 거의 무비용',
            ))

        return hedges

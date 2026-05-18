"""기본 타입 정의 — 모든 모듈이 공유하는 데이터 구조.

명세서: STOCK_PREDICTION_SYSTEM_SPEC.md §3 (모듈 출력 schema), §6 (Output schema)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple


class Direction(Enum):
    STRONG_BULL = 2
    BULL = 1
    NEUTRAL = 0
    BEAR = -1
    STRONG_BEAR = -2


@dataclass
class ModuleOutput:
    """각 분석 모듈의 표준 출력."""
    module_name: str
    score: float          # -10 ~ +10
    direction: Direction
    confidence: float     # 0 ~ 1
    details: Dict = field(default_factory=dict)


@dataclass
class Scenario:
    """단일 시나리오 (mega_bull / bull / base / bear / crisis)."""
    name: str
    probability: float
    price_range: Tuple[float, float]
    expected_value: float
    triggers: List[str] = field(default_factory=list)


@dataclass
class SellTrigger:
    price: float
    action: str           # 'sell_15pct', 'sell_20pct' ...
    reason: str


@dataclass
class HedgeRecommendation:
    type: str             # 'protective_put' / 'collar'
    strike: Optional[float] = None
    call_strike: Optional[float] = None
    put_strike: Optional[float] = None
    expiration_days: int = 90
    rationale: str = ""


@dataclass
class PredictionResult:
    """최종 예측 결과 (SPEC §6 Output schema)."""
    ticker: str
    timestamp: datetime
    current_price: float
    horizon_days: int

    # 핵심 예측
    expected_value: float
    composite_score: float
    confidence: float
    directional_bias: str

    # 신뢰구간
    ci_50: Tuple[float, float]
    ci_80: Tuple[float, float]
    ci_95: Tuple[float, float]

    # 시나리오
    scenarios: List[Scenario]

    # 모듈별 결과
    modules: Dict[str, ModuleOutput]

    # 액션 추천
    sell_triggers: List[SellTrigger]
    stop_loss: List[SellTrigger]
    hedge_recommendations: List[HedgeRecommendation]

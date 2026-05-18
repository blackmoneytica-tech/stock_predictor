"""분석 모듈 베이스 클래스."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from ..types import Direction, ModuleOutput


class AnalysisModule(ABC):
    """모든 분석 모듈의 추상 베이스."""

    def __init__(self, name: str, weight: float):
        self.name = name
        self.weight = weight

    @abstractmethod
    def analyze(self, data: Dict) -> ModuleOutput:
        """모듈별 분석 로직 — 각 서브클래스가 구현."""

    def get_dynamic_weight(self, context: Dict) -> float:
        """이벤트 상황별 동적 가중치 (기본은 정적 weight)."""
        return self.weight

    @staticmethod
    def score_to_direction(score: float) -> Direction:
        if score >= 4:
            return Direction.STRONG_BULL
        if score >= 1:
            return Direction.BULL
        if score >= -1:
            return Direction.NEUTRAL
        if score >= -4:
            return Direction.BEAR
        return Direction.STRONG_BEAR

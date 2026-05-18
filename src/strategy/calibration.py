"""Calibration + ticker fitness + macro regime — 백테스트 기반 정확도 보정.

3가지 보정:
1. Confidence calibration — backtest 매핑으로 출력 conf → actual accuracy
2. Ticker fitness — 시스템 적합도 측정 (ticker별 hit rate)
3. Macro regime weight modifier — STRONG_BULL/CHOPPY/BEAR에 따라 모듈 가중치 조정
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "results"


# ────────────────────────────────────────────────────────────
# A. Confidence calibration (isotonic regression)
# ────────────────────────────────────────────────────────────
class ConfidenceCalibrator:
    """v7 backtest 데이터 기반 confidence 보정.

    백테스트 측정:
      conf 0.40-0.50 → acc 33.3%
      conf 0.50-0.60 → acc 24.0%  (예상보다 낮음)
      conf 0.60-0.65 → acc ~50%
      conf 0.65-0.70 → acc ~55%
      conf 0.70-0.75 → acc ~55%
    """

    # backtest 측정값 (v7 데이터)
    _BINS = [
        (0.0, 0.45, 0.33),    # very low
        (0.45, 0.55, 0.35),   # low
        (0.55, 0.62, 0.42),   # mid
        (0.62, 0.68, 0.55),   # higher mid (sweet spot)
        (0.68, 0.75, 0.62),   # high
    ]

    def calibrate(self, raw_conf: float) -> float:
        """raw conf → calibrated actual accuracy probability."""
        for lo, hi, acc in self._BINS:
            if lo <= raw_conf < hi:
                return acc
        return 0.5

    def label(self, raw_conf: float) -> str:
        cal = self.calibrate(raw_conf)
        if cal >= 0.60:
            return "🟢 강한 시그널"
        if cal >= 0.50:
            return "🟡 의미있는 시그널"
        if cal >= 0.40:
            return "🟠 약한 시그널"
        return "🔴 신뢰도 낮음 — 관망"


# ────────────────────────────────────────────────────────────
# B. Ticker fitness (시스템 적합도)
# ────────────────────────────────────────────────────────────
class TickerFitness:
    """v6+v7 backtest로 검증된 ticker별 hit rate."""

    # 측정값 (변동성 universe + 메가캡 backtest)
    _HIT_RATES: Dict[str, float] = {
        # Top 80% (검증된 high-fit)
        'CRCL': 0.80, 'MSTR': 0.80, 'NVDA': 0.80, 'KLIC': 0.80, 'WULF': 0.80,
        # 60% high-fit
        'ARM': 0.60, 'IONQ': 0.60, 'BBAI': 0.60, 'SOUN': 0.60,
        'AAOI': 0.60, 'CRDO': 0.60, 'IREN': 0.60, 'MARA': 0.60,
        'RIOT': 0.60, 'AMD': 0.60, 'MU': 0.60, 'AMAT': 0.60,
        'PLTR': 0.60, 'AAPL': 0.60, 'ORCL': 0.80, 'PLUG': 0.60,
        # 40% mid
        'TSLA': 0.40, 'GS': 0.40, 'SMCI': 0.40,
        # 0-20% low-fit (안 맞는 종목)
        'MSFT': 0.0, 'SNOW': 0.0, 'V': 0.0, 'RIVN': 0.0, 'COIN': 0.0, 'HOOD': 0.0,
        'AFRM': 0.20, 'AMZN': 0.20, 'AVGO': 0.20, 'BAC': 0.20, 'CRM': 0.20,
        'GOOGL': 0.20, 'JPM': 0.20, 'META': 0.20,
    }

    def fitness(self, ticker: str) -> float:
        """검증된 hit rate. 미측정 종목은 0.5 (중립)."""
        return self._HIT_RATES.get(ticker.upper(), 0.5)

    def label(self, ticker: str) -> str:
        f = self.fitness(ticker)
        if f >= 0.75:
            return "⭐⭐⭐ 시스템 최적합 (80% 검증)"
        if f >= 0.55:
            return "⭐⭐ 시스템 적합 (60% 검증)"
        if f >= 0.45:
            return "⭐ 평균 (미측정 또는 40%)"
        if f >= 0.25:
            return "⚠️ 약함 (20%)"
        return "❌ 시스템 부적합 (0%) — 다른 도구 권장"

    def confidence_modifier(self, ticker: str) -> float:
        """fitness가 낮으면 conf 출력 약화 (× 0.5~1.2)."""
        f = self.fitness(ticker)
        # 0% → 0.5×, 50% → 1.0×, 80%+ → 1.2×
        return float(0.5 + f * 0.875)


# ────────────────────────────────────────────────────────────
# C. Macro regime adaptive weights
# ────────────────────────────────────────────────────────────
class RegimeWeights:
    """v7 backtest 발견:
      CHOPPY:      56.7% (best) — mean reversion / matter zones 효과적
      BULL:        45.6%
      STRONG_BULL: 38.7% (worst) — 극강세 short-term noise
      BEAR:        측정 부족
    """

    # 각 regime의 가중치 multiplier (1.0이 기본)
    _MODIFIERS: Dict[str, Dict[str, float]] = {
        'CHOPPY': {
            'mean_reversion': 1.4,
            'demand_supply': 1.3,
            'options': 1.1,
            'trend': 0.7,         # 추세 약화
            'sentiment': 0.9,
        },
        'BULL': {
            'trend': 1.2,
            'demand_supply': 1.1,
            'options': 1.1,
            'mean_reversion': 0.9,
        },
        'STRONG_BULL': {
            # 1-day noise 큰 시장 — defensive
            'trend': 0.8,
            'demand_supply': 1.1,
            'mean_reversion': 1.3,  # 단기 반전 가능성 ↑
            'short_squeeze': 0.7,
        },
        'BEAR': {
            'mean_reversion': 1.4,
            'demand_supply': 1.3,
            'options': 1.1,
            'trend': 0.8,
        },
        'STRONG_BEAR': {
            'mean_reversion': 1.5,
            'options': 1.2,
            'trend': 0.5,
            'short_squeeze': 0.5,
        },
        'CHOPPY_DEFAULT': {},  # CHOPPY 다시 한 번 별칭
    }

    def get_multipliers(self, regime: str) -> Dict[str, float]:
        return self._MODIFIERS.get(regime, {})


# Singletons
calibrator = ConfidenceCalibrator()
fitness_db = TickerFitness()
regime_weights = RegimeWeights()

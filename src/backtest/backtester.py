"""Walk-forward Backtester — lookahead bias 방지.

검증 항목:
- Directional accuracy (목표 60%+)
- CI coverage (50%, 80%)
- Calibration (60% 신뢰도 = 60% 적중?)
- 시그널별 contribution
"""
from __future__ import annotations

from typing import List

import pandas as pd

from ..system import StockPredictionSystem


class Backtester:
    def __init__(self, system: StockPredictionSystem):
        self.system = system
        self.history: List[dict] = []

    def run(self, ticker: str, start_date, end_date, horizon_days: int = 5):
        """Walk-forward backtest — Phase 5에서 구현."""
        # NOTE: historical data fetcher (Phase 2) 필요
        raise NotImplementedError(
            "Walk-forward 백테스트 구현 필요. SPEC §7.2 참조"
        )

    def compute_metrics(self):
        df = pd.DataFrame(self.history)

        df['actual_direction'] = (df['actual_price'] > df['predicted_ev']).map(
            {True: 'bull', False: 'bear'},
        )
        directional_accuracy = (
            df['predicted_direction'] == df['actual_direction']
        ).mean()

        ci_50_hit = df.apply(
            lambda r: r['ci_50'][0] <= r['actual_price'] <= r['ci_50'][1],
            axis=1,
        ).mean()

        ci_80_hit = df.apply(
            lambda r: r['ci_80'][0] <= r['actual_price'] <= r['ci_80'][1],
            axis=1,
        ).mean()

        mae = abs(df['actual_price'] - df['predicted_ev']).mean()

        return {
            'n_predictions': len(df),
            'directional_accuracy': directional_accuracy,
            'ci_50_coverage': ci_50_hit,
            'ci_80_coverage': ci_80_hit,
            'mae': mae,
            'history': df,
        }

    def calibration_check(self):
        """신뢰도와 실제 적중률 일치 검증.

        예: 60% 신뢰도 예측 중 60%가 실제로 적중해야 함.
        """
        df = pd.DataFrame(self.history)
        bins = [(0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 0.75)]

        results = []
        for low, high in bins:
            subset = df[(df['confidence'] >= low) & (df['confidence'] < high)]
            if subset.empty:
                continue

            expected = (low + high) / 2
            actual = (subset['predicted_direction'] == subset['actual_direction']).mean()
            deviation = abs(actual - expected)

            results.append({
                'confidence_range': f"{low:.2f}-{high:.2f}",
                'count': len(subset),
                'expected_accuracy': expected,
                'actual_accuracy': actual,
                'deviation': deviation,
                'status': 'OK' if deviation < 0.10 else 'WARNING',
            })

        return results

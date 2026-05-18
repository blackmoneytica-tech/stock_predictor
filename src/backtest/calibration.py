"""Calibration utilities — Backtester.calibration_check를 더 정밀하게 검사.

Phase 5에서 확장. 현재는 placeholder.
"""
from __future__ import annotations

from typing import List

import pandas as pd


def reliability_diagram(history: List[dict]) -> pd.DataFrame:
    """Reliability diagram용 데이터 생성.

    bin 별 (예측 신뢰도, 실제 적중률) → 시각화에 사용.
    """
    df = pd.DataFrame(history)
    bins = pd.cut(df['confidence'], bins=[0.4, 0.5, 0.6, 0.7, 0.75])

    grouped = df.groupby(bins, observed=False).apply(
        lambda g: pd.Series({
            'count': len(g),
            'mean_confidence': g['confidence'].mean(),
            'actual_accuracy': (
                g['predicted_direction'] == g['actual_direction']
            ).mean(),
        }),
    )
    return grouped.reset_index()


def calibration_error(history: List[dict]) -> float:
    """ECE (Expected Calibration Error) 계산.

    bin별 |expected − actual| × 비중 가중합. < 0.10이면 well-calibrated.
    """
    df = pd.DataFrame(history)
    if df.empty:
        return float('nan')

    bins = pd.cut(df['confidence'], bins=[0.4, 0.5, 0.6, 0.7, 0.75])
    total = len(df)

    ece = 0.0
    for _, group in df.groupby(bins, observed=False):
        if len(group) == 0:
            continue
        expected = group['confidence'].mean()
        actual = (group['predicted_direction'] == group['actual_direction']).mean()
        ece += (len(group) / total) * abs(actual - expected)

    return ece

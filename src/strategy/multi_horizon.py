"""Multi-horizon ensemble — 1d + 3d + 5d 예측 합의 → confidence boost.

가설:
  3개 horizon이 같은 방향 → 강한 시그널 (conf × 1.2)
  엇갈리면 → 약한 시그널 (conf × 0.7)
"""
from __future__ import annotations

from typing import Dict, List


def ensemble_predictions(predictions: List[Dict]) -> Dict:
    """1d/3d/5d 예측 결과 합치기.

    Args:
        predictions: [{horizon, composite_score, ev_pct, conf, directional_bias}, ...]

    Returns:
        {
            agreement: 'all_bull' / 'all_bear' / 'mixed' / 'all_neutral',
            n_bull: 양수 score 갯수,
            n_bear: 음수 score 갯수,
            avg_score: 평균 composite score,
            avg_ev_pct: 평균 expected return,
            ensemble_conf: 보정된 confidence,
            boost_factor: 합의 정도 × multiplier,
        }
    """
    if not predictions:
        return _empty()

    scores = [p["composite_score"] for p in predictions]
    confs = [p["conf"] for p in predictions]
    evs = [p["ev_pct"] for p in predictions]

    n_bull = sum(1 for s in scores if s > 0.5)
    n_bear = sum(1 for s in scores if s < -0.5)
    n_neutral = len(scores) - n_bull - n_bear

    if n_bull == len(scores):
        agreement = "all_bull"
        boost = 1.25
    elif n_bear == len(scores):
        agreement = "all_bear"
        boost = 1.25
    elif n_bull >= 2 and n_bear == 0:
        agreement = "mostly_bull"
        boost = 1.10
    elif n_bear >= 2 and n_bull == 0:
        agreement = "mostly_bear"
        boost = 1.10
    elif n_bull > 0 and n_bear > 0:
        agreement = "mixed_conflict"
        boost = 0.70  # 다른 방향 → 약화
    else:
        agreement = "all_neutral"
        boost = 0.85

    avg_score = sum(scores) / len(scores)
    avg_conf = sum(confs) / len(confs)
    avg_ev = sum(evs) / len(evs)
    ensemble_conf = min(0.75, avg_conf * boost)

    return {
        "agreement": agreement,
        "n_bull": n_bull,
        "n_bear": n_bear,
        "n_neutral": n_neutral,
        "avg_score": avg_score,
        "avg_ev_pct": avg_ev,
        "boost_factor": boost,
        "ensemble_conf": ensemble_conf,
        "n_predictions": len(predictions),
    }


def _empty() -> Dict:
    return {
        "agreement": "none", "n_bull": 0, "n_bear": 0, "n_neutral": 0,
        "avg_score": 0.0, "avg_ev_pct": 0.0,
        "boost_factor": 1.0, "ensemble_conf": 0.5,
        "n_predictions": 0,
    }


def label_agreement(agreement: str) -> str:
    return {
        "all_bull": "🚀 3개 horizon 모두 강세 (강한 매수)",
        "all_bear": "💀 3개 horizon 모두 약세 (강한 매도)",
        "mostly_bull": "📈 다수 강세",
        "mostly_bear": "📉 다수 약세",
        "mixed_conflict": "⚠️ horizon 충돌 (관망)",
        "all_neutral": "➖ 모두 중립",
    }.get(agreement, agreement)

"""모듈 합의 카운팅 → tier 판정 (2026-05-20 backtest 검증).

1499-trade 분석:
- n_bull 1~3: noise (47% win = random)
- n_bull >= 5: 강한 alpha (65.7% win, +4.89%/trade, Sharpe 4.33)
- n_bear == 0: 만장일치 매수 = -0.96% 손실 (overhyped, priced in)
- n_bear >= 6 + STRONG_BEAR: contrarian 반등 (62.5% win, +2.66%)
- n_bear >= 6 + BULL: 절대 매수 X (20% win, -1.80%)
"""
from __future__ import annotations

from typing import Dict


def count_consensus(modules: Dict, threshold: float = 1.0) -> Dict:
    """11개 모듈 score → (n_bull, n_bear, n_neutral).

    Args:
        modules: {name: ModuleOutput} (각각 .score 보유)
        threshold: |score| > threshold 이면 bull/bear 카운트 (기본 1.0)
    """
    n_bull = sum(1 for m in modules.values() if m.score > threshold)
    n_bear = sum(1 for m in modules.values() if m.score < -threshold)
    n_neutral = len(modules) - n_bull - n_bear
    return {"n_bull": n_bull, "n_bear": n_bear, "n_neutral": n_neutral}


def classify_consensus_tier(
    n_bull: int, n_bear: int, macro_mode: str = "",
) -> Dict:
    """5-tier 판정 + 한국어 라이너 + 백테스트 출처.

    Returns: {
      tier: 'strong_consensus_buy' / 'contrarian_rebound' /
            'overhyped_warning' / 'strong_bear_trap' / 'noise',
      label: 한국어 짧은 라벨,
      tagline: 1줄 설명,
      backtest: 백테스트 출처,
      tone: 'bull' / 'bear' / 'warn' / 'neutral',
      alpha: True/False (운용 가치 있나)
    }
    """
    macro = (macro_mode or "").upper()

    # Tier 1: n_bull ≥ 5 강한 합의 alpha (Sharpe 4.33)
    if n_bull >= 5:
        return {
            "tier": "strong_consensus_buy",
            "label": "⭐ 강한 합의 (Sharpe 4.33)",
            "tagline": "5+ 모듈 매수 동의 — 단독으로 가장 강한 alpha",
            "backtest": "n=35, 65.7% win, +4.89%/trade (5d 검증)",
            "tone": "bull",
            "alpha": True,
        }

    # Tier 2: n_bear ≥ 6 + STRONG_BEAR contrarian
    if n_bear >= 6 and macro == "STRONG_BEAR":
        return {
            "tier": "contrarian_rebound",
            "label": "🔄 Contrarian 반등 기회",
            "tagline": "모듈 다 비관 + 극단 약세장 → 반등 적기",
            "backtest": "n=24, 62.5% win, +2.66%/trade (Sharpe 2.62)",
            "tone": "bull",
            "alpha": True,
        }

    # Tier 3: n_bear ≥ 6 + BULL/STRONG_BULL trap
    if n_bear >= 6 and macro in ("BULL", "STRONG_BULL"):
        return {
            "tier": "strong_bear_trap",
            "label": "🛑 강한 매도 신호 (BULL macro)",
            "tagline": "강세장에서 모듈 다 매도 동의 — 진짜 매도 신호, 절대 매수 X",
            "backtest": "n=5, 20% win, -1.80% (5d 검증)",
            "tone": "bear",
            "alpha": False,  # 회피 = 알파 (안 사면 손실 회피)
        }

    # Tier 4: n_bear == 0 (만장일치 매수) = overhyped
    if n_bear == 0 and n_bull <= 3:
        return {
            "tier": "overhyped_warning",
            "label": "⚠️ 만장일치 매수 (위험)",
            "tagline": "모듈 다 중립/매수 — 이미 priced in, 추가 상승 어려움",
            "backtest": "n=45, 40% win, -0.96%/trade (가장 낮음)",
            "tone": "warn",
            "alpha": False,
        }

    # Tier 5: noise (1~3 우세는 random level)
    return {
        "tier": "noise",
        "label": "신호 약함 (noise level)",
        "tagline": f"n_bull={n_bull}, n_bear={n_bear} — 1~3 카운팅은 random 수준",
        "backtest": "n_bull 1~3: 47% win (random)",
        "tone": "neutral",
        "alpha": False,
    }


def evaluate_module_consensus(modules: Dict, macro_mode: str = "") -> Dict:
    """count + tier를 한 번에. PredictionResult.module_consensus 필드용.

    Returns: {n_bull, n_bear, n_neutral, tier, label, tagline, backtest, tone, alpha}
    """
    counts = count_consensus(modules)
    tier_info = classify_consensus_tier(
        counts["n_bull"], counts["n_bear"], macro_mode,
    )
    return {**counts, **tier_info}

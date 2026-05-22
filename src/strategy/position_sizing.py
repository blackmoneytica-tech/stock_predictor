"""Position sizing — Sweet Spot 중심 (2026-05-22 약세장 검증 후 재설계).

⚠️ 중요 — 룰 robustness 등급:

✅ TIER 1 (강세장+약세장 둘 다 검증, 진짜 alpha):
  - Sweet Spot contrarian (RS weak + 시스템 score<0 + DD buy_cat)
    강세장 10d 95.8% win / 약세장 10d 66.1% win (-29.7%p but 여전히 64%+)
    → 적중 시 1.5x size, horizon 5d/10d 우선
  - Short 영구 금지 (강세장 -3~-7% / 약세장 33% win 일관)

⚠️ TIER 2 (강세장 한정 = bull-only):
  - Verified Rules + Sizing (강세장 Sharpe +2.18 → 약세장 -1.20)
  - 모듈 합의 n_bull≥5 (강세장 65.7% win → 약세장 28.6%)
  - 1d × BEAR + 시스템 신호 (강세장 64% / 약세장 sample 부족)

🎯 운영 모드:
  - Sweet Spot 적중: 1.5x 진입 (Tier 1, 둘 다 검증)
  - 매크로 BULL/STRONG_BULL + 다른 강세장 룰: 보조 (1.0x baseline, warning 표시)
  - 매크로 BEAR/STRONG_BEAR + Sweet Spot 미적중: cash (Tier 2 룰 음의 alpha)

검증 출처:
  강세장: 1499 trades, 2025-12 ~ 2026-05 (Sharpe 2.51 — bull only)
  약세장: 720 trades, 2022-01 ~ 2022-11 (Sharpe -1.20 — bull rules fail)
"""
from __future__ import annotations

from typing import Tuple


def is_contrarian_sweet_spot(
    macro_mode: str, cat: str, rs_grade: str,
    composite_score: float, confidence: float, ev_pct: float, horizon: int,
) -> bool:
    """2026-05-19 grid search 검증 sweet spot (14개 robust)."""
    return evaluate_sweet_spot(
        macro_mode, cat, rs_grade, composite_score, confidence, ev_pct, horizon,
    )["active"]


def evaluate_sweet_spot(
    macro_mode: str, cat: str, rs_grade: str,
    composite_score: float, confidence: float, ev_pct: float, horizon: int,
    *,
    dd_pct: float = None, rel_chg20: float = None,
) -> dict:
    """Sweet spot 적중 여부 + 적중 조건 체크리스트.

    UI에서 보라 배경 큰 박스 + 체크리스트 렌더용.
    Returns: {active, tier, conditions: [{label, met}], backtest}
    """
    macro = (macro_mode or "?").upper()
    cat_s = (cat or "").lower()
    rs = (rs_grade or "").lower()
    rs_weak = rs in ("weak", "very_weak")

    # 가독성 라벨
    cat_label = {
        "strong_buy": "Drawdown ≤ -20% (strong_buy)",
        "deep": "Drawdown -20%~-15% (deep)",
        "buy_zone": "Drawdown -10%~-3% (buy_zone)",
        "trap": "Drawdown -15%~-10% (trap ⚠️)",
        "safe": "Drawdown > -3% (safe)",
    }.get(cat_s, f"Drawdown cat = {cat_s or '?'}")
    if dd_pct is not None:
        cat_label = f"Drawdown {dd_pct:+.1f}% ({cat_s})"

    rs_label = (f"RS {rs} (시장 underperform)" if rs_weak
                else f"RS {rs} (시장 outperform — sweet spot 미충족)")
    if rel_chg20 is not None:
        rs_label = f"RS {rs} (20일 SPY 대비 {rel_chg20:+.1f}%p)"

    score_label = f"시스템 score {composite_score:+.1f}"
    conf_label = f"확신도 {confidence:.0%}"
    ev_label = f"EV {ev_pct:+.2f}%"
    macro_label = f"매크로 = {macro}"

    if horizon <= 1:
        conditions = [
            {"label": macro_label + " (BEAR/STRONG_BEAR)",
             "met": macro in ("BEAR", "STRONG_BEAR")},
            {"label": cat_label + " (strong_buy 필요)",
             "met": cat_s == "strong_buy"},
            {"label": rs_label + " (weak/very_weak 필요)",
             "met": rs_weak},
            {"label": score_label + " (< -1 필요)",
             "met": composite_score < -1},
            {"label": conf_label + " (> 60% 필요)",
             "met": confidence > 0.6},
        ]
        all_met = all(c["met"] for c in conditions)
        return {
            "active": all_met,
            "tier": "1d_contrarian" if all_met else None,
            "conditions": conditions,
            "backtest": "in 73% / out 64% win, +0.86%/trade (n=22)",
            "tagline": "약세장 바닥 + 시스템도 비관 → 반등 진입 (contrarian)",
        }

    # 5d contrarian (2022 약세장 검증: BULL은 -2.01% 음의 alpha → 제외)
    conditions = [
        {"label": macro_label + " (STRONG_BULL / CHOPPY)",
         "met": macro in ("STRONG_BULL", "CHOPPY")},
        {"label": cat_label + " (buy_zone / deep / strong_buy)",
         "met": cat_s in ("strong_buy", "deep", "buy_zone")},
        {"label": rs_label + " (weak/very_weak 필요)",
         "met": rs_weak},
        {"label": score_label + " (< 0 필요)",
         "met": composite_score < 0},
        {"label": f"{conf_label} > 70% 또는 {ev_label} > +0.3%",
         "met": confidence > 0.7 or ev_pct > 0.3},
    ]
    all_met = all(c["met"] for c in conditions)
    return {
        "active": all_met,
        "tier": "5d_contrarian" if all_met else None,
        "conditions": conditions,
        "backtest": (
            "강세장 5d 66.7%/+5.36%, 약세장 10d 66.1%/+3.70% "
            "(STRONG_BULL 81.8%/+6.36% · CHOPPY 60%/+4.38%)"
        ),
        "tagline": "DD 깊은 종목 + 시장보다 더 떨어진 + 시스템도 비관 → 반등",
    }


def trade_direction(macro_mode: str, ev_pct: float, horizon: int) -> int:
    """+1 long / 0 cash. Short은 walk-forward 검증으로 영구 금지."""
    macro = (macro_mode or "?").upper()
    if horizon == 1 and macro in ("BEAR", "CHOPPY"):
        # 시스템 신호 sweet spot (백테스트 BEAR 64% win / CHOPPY 52% win)
        if abs(ev_pct) < 0.3:
            return 0
        return 1 if ev_pct > 0 else 0  # short 금지
    return 1  # default long


def sizing_factor(macro_mode: str, ev_pct: float, confidence: float, horizon: int) -> float:
    """검증된 sizing matrix. 0.0 ~ 1.5x."""
    macro = (macro_mode or "?").upper()
    sig_strong = abs(ev_pct) > 0.5 and confidence >= 0.5

    if horizon == 1:
        if macro == "BEAR":
            return 1.5 if sig_strong else 0.5
        if macro == "CHOPPY":
            return 1.2 if sig_strong else 0.4
        if macro in ("BULL", "STRONG_BULL", "STRONG_BEAR"):
            return 0.8
        return 0.4

    # 3d / 5d horizon — macro-aligned baseline
    if macro in ("BULL", "STRONG_BULL"):
        return 1.0
    if macro == "STRONG_BEAR":
        return 0.8
    if macro == "BEAR":
        return 0.0
    return 0.4  # CHOPPY


def compute_recommendation(
    macro_mode: str, ev_pct: float, confidence: float, horizon: int,
    *,
    cat: str = "", rs_grade: str = "", composite_score: float = 0.0,
) -> Tuple[int, float, str]:
    """direction + size + 한국어 설명 (2026-05-22 약세장 검증 후 재설계).

    P1 우선순위: Sweet Spot (TIER 1 — 강세장+약세장 둘 다 검증된 진짜 alpha)
    P2 보조: 강세장 룰 (TIER 2 — bull-only, BEAR 시 음의 alpha)
    """
    macro = (macro_mode or "?").upper()

    # ────── P1: Sweet Spot (Tier 1, 강세장+약세장 robust) ──────
    if cat and rs_grade and is_contrarian_sweet_spot(
        macro_mode, cat, rs_grade, composite_score, confidence, ev_pct, horizon,
    ):
        rationale = (
            "🟢 매수 1.5× — ⭐ Sweet Spot Contrarian (Tier 1 robust alpha). "
            "강세장 10d 95.8% win / 약세장 10d 66.1% win — 시장 환경 독립 검증"
        )
        return 1, 1.5, rationale

    # ────── P2 보조: 약세장에선 cash 우선 (bull-only 룰 음의 alpha) ──────
    if macro in ("BEAR", "STRONG_BEAR"):
        # 강세장 룰 적용 X (2022 검증: Verified+Sizing Sharpe -1.20, 모듈 합의 28.6% win)
        rationale = (
            "🛑 cash 0× — 약세장에선 Sweet Spot 미적중 시 진입 금지. "
            "2022 검증: bull 룰 약세장에서 음의 alpha (Sharpe -1.20)"
        )
        return 0, 0.0, rationale

    # ────── P3: 강세장에서만 기존 sizing 적용 (bull-only warning) ──────
    direction = trade_direction(macro_mode, ev_pct, horizon)
    size = sizing_factor(macro_mode, ev_pct, confidence, horizon)

    if direction == 0 or size == 0:
        rationale = _rationale_cash(macro, horizon)
    else:
        rationale = _rationale_long(macro, ev_pct, confidence, horizon, size)
        # bull-only warning 추가 (검증 사용자 인식)
        rationale += " · ⚠️ Bull-only (약세장에선 검증 X)"
    return direction, size, rationale


def _rationale_long(macro, ev_pct, conf, horizon, size):
    macro_ko = {
        "BULL": "강세장", "STRONG_BULL": "매우 강한 강세장",
        "BEAR": "약세장", "STRONG_BEAR": "매우 강한 약세장",
        "CHOPPY": "횡보",
    }.get(macro, macro)
    sig_strong = abs(ev_pct) > 0.5 and conf >= 0.5
    sig_word = "강한 신호" if sig_strong else "약한 신호"

    if horizon == 1 and macro == "BEAR":
        return (f"🟢 매수 {size:.1f}× — 1d × BEAR + {sig_word}. "
                f"백테스트 64% win, Sharpe 2.51 검증")
    if horizon == 1 and macro == "CHOPPY":
        return (f"🟢 매수 {size:.1f}× — 1d × CHOPPY + {sig_word}. "
                f"백테스트 52% win (+13%p vs baseline)")
    if horizon == 1:
        return f"🟢 매수 {size:.1f}× — 1d × {macro_ko} baseline long (시스템 raw 신호 무시)"
    if horizon >= 3 and macro in ("BULL", "STRONG_BULL"):
        return f"🟢 매수 {size:.1f}× — {horizon}d × {macro_ko} baseline (50~56% win)"
    if horizon >= 3 and macro == "STRONG_BEAR":
        return f"🟠 cautious 매수 {size:.1f}× — {horizon}d × oversold 반등 (51.8% win)"
    return f"🟡 small 매수 {size:.1f}× — {horizon}d × {macro_ko}"


def _rationale_cash(macro, horizon):
    if horizon >= 3 and macro == "BEAR":
        return "🛑 cash 0× — 5d × BEAR. 백테스트 baseline 41% / 시스템 1.3% (모두 무의미)"
    return "🟡 cash 0× — 신호 약함 (|EV|<0.3%)"

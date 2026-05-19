"""Position sizing — walk-forward 검증 룰 (2026-05-19 backtest).

1499-trade simulation에서 baseline 대비:
  1d: Sharpe 0.33 → 2.51 (+2.18), Total +120% → +555% (4.6×), MDD -329% → -85% (3.9× 안전)
  5d: Sharpe 0.45 → 0.73, Total +781% → +924%

Direction:
  - Short 영구 비활성 (walk-forward에서 모든 short signal -3~-7%/trade)
  - 1d × BEAR/CHOPPY: 시스템 EV 신호 따라감
  - 그 외: long bias only

Sizing (0.0~1.5x):
  1d × BEAR: 강한 신호 1.5x, 약함 0.5x
  1d × CHOPPY: 강한 신호 1.2x, 약함 0.4x
  1d × BULL/STRONG_*: 0.8x (baseline)
  5d × BULL/STRONG_BULL: 1.0x
  5d × STRONG_BEAR: 0.8x (oversold rebound)
  5d × BEAR: 0.0x (cash)
  5d × CHOPPY: 0.4x (small long)
"""
from __future__ import annotations

from typing import Tuple


def is_contrarian_sweet_spot(
    macro_mode: str, cat: str, rs_grade: str,
    composite_score: float, confidence: float, ev_pct: float, horizon: int,
) -> bool:
    """2026-05-19 grid search 검증 sweet spot (14개 robust).

    공통 패턴: RS weak + 시스템 score<0 + drawdown buy cat = contrarian mean reversion.

    1d out-sample: win 63.6%, avg +0.86%/trade
    5d out-sample: win 66.7%, avg +5.36%/trade
    """
    macro = (macro_mode or "?").upper()
    cat_s = (cat or "").lower()
    rs = (rs_grade or "").lower()
    rs_weak = rs in ("weak", "very_weak")

    if horizon <= 1:
        # 1d contrarian: BEAR macro + strong_buy + RS weak + score<-1 + conf>0.6
        return (
            macro in ("BEAR", "STRONG_BEAR")
            and cat_s == "strong_buy"
            and rs_weak
            and composite_score < -1
            and confidence > 0.6
        )
    # 5d contrarian: non_BEAR + buy_cat + RS weak + score<0 + (conf>0.7 or ev>0.3)
    return (
        macro in ("CHOPPY", "BULL", "STRONG_BULL")
        and cat_s in ("strong_buy", "deep", "buy_zone")
        and rs_weak
        and composite_score < 0
        and (confidence > 0.7 or ev_pct > 0.3)
    )


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
) -> Tuple[int, float, str]:
    """direction + size + 한국어 설명. PredictionResult에 채울 3-tuple."""
    direction = trade_direction(macro_mode, ev_pct, horizon)
    size = sizing_factor(macro_mode, ev_pct, confidence, horizon)
    macro = (macro_mode or "?").upper()

    if direction == 0 or size == 0:
        rationale = _rationale_cash(macro, horizon)
    else:
        rationale = _rationale_long(macro, ev_pct, confidence, horizon, size)
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

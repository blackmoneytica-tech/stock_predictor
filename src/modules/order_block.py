"""Order Block / Fair Value Gap (ICT/SMC) — 기관 매물 흔적.

핵심 통찰 (ICT — Inner Circle Trader 방법론):
- Bullish Order Block: 큰 음봉 직전의 마지막 양봉 (down move 시작 전 last up bar)
  → 가격이 다시 그 zone에 도달하면 반등 가능성 (기관 매수 흔적)
- Bearish Order Block: 큰 양봉 직전의 마지막 음봉
  → 저항으로 작동
- Fair Value Gap (FVG): 3봉 패턴 [c1.high < c3.low (bullish gap)] — 미충족 imbalance
  → 가격이 다시 gap에 들어오면 채워질 가능성

조건:
- "큰" 봉 = ATR × 1.5 이상 (또는 거래량 평균 × 2 이상)
- "마지막" = 반전 직전 (다음 봉이 강한 반대 방향)
- valid 기간 = 30일 (이후 약화)

DemandSupply 모듈과의 차이:
- DemandSupply: 거래량 누적 분포 기반 (정적 매물대)
- OrderBlock: 가격 패턴 + 거래량 + 반전 시퀀스 기반 (이벤트 매물대)
- 합쳐서 사용 시 더 정확한 zone 식별
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..types import ModuleOutput
from .base import AnalysisModule


class OrderBlockModule(AnalysisModule):
    def __init__(self, weight: float = 0.07):
        super().__init__("order_block", weight=weight)

    def analyze(self, data: Dict) -> ModuleOutput:
        ohlcv = data["ohlcv"]
        current_price = data["current_price"]

        if len(ohlcv) < 30:
            return ModuleOutput(
                module_name=self.name, score=0.0,
                direction=self.score_to_direction(0),
                confidence=0.3,
                details={"reason": "insufficient_data"},
            )

        blocks = extract_order_blocks(ohlcv, lookback_days=60)
        fvgs = extract_fair_value_gaps(ohlcv, lookback_days=30)

        # 현재가 근처 (1.5% 이내) 블록 / FVG 찾기
        nearby_bullish_ob = _nearest_unfilled(
            [b for b in blocks if b["type"] == "bullish"],
            current_price,
            side="below",
        )
        nearby_bearish_ob = _nearest_unfilled(
            [b for b in blocks if b["type"] == "bearish"],
            current_price,
            side="above",
        )

        score = _compute_ob_score(current_price, nearby_bullish_ob, nearby_bearish_ob, fvgs)

        return ModuleOutput(
            module_name=self.name,
            score=score,
            direction=self.score_to_direction(score),
            confidence=0.60,
            details={
                "nearest_bullish_ob": _ob_summary(nearby_bullish_ob),
                "nearest_bearish_ob": _ob_summary(nearby_bearish_ob),
                "n_blocks": len(blocks),
                "n_fvgs": len(fvgs),
                "recent_fvgs": [_fvg_summary(f) for f in fvgs[:3]],
            },
        )


# ── Order Block 추출 ────────────────────────────────────────
def extract_order_blocks(
    ohlcv: pd.DataFrame,
    lookback_days: int = 60,
    body_mult_atr: float = 1.5,
) -> List[Dict]:
    """Order Block 추출 — 큰 봉 직전의 last opposite candle.

    Returns:
        [{type, low, high, mid, date, body_atr_ratio, filled, strength}]
    """
    df = ohlcv.tail(lookback_days).copy()
    if len(df) < 10:
        return []

    # ATR 계산 (14일)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    blocks = []
    df = df.assign(atr=atr)
    rows = df.dropna(subset=["atr"]).to_dict("records")
    idx_list = df.dropna(subset=["atr"]).index.tolist()

    for i in range(1, len(rows) - 1):
        prev = rows[i - 1]
        curr = rows[i]
        body = abs(curr["close"] - curr["open"])
        if curr["atr"] <= 0 or body < body_mult_atr * curr["atr"]:
            continue

        # 큰 양봉 = 직전 음봉이 bearish OB (저항)
        # 큰 음봉 = 직전 양봉이 bullish OB (지지)
        if curr["close"] > curr["open"]:
            # 큰 양봉 — 직전 음봉이 bearish OB
            if prev["close"] < prev["open"]:
                blocks.append({
                    "type": "bearish",
                    "low": float(min(prev["open"], prev["close"])),
                    "high": float(max(prev["open"], prev["close"])),
                    "mid": float((prev["open"] + prev["close"]) / 2),
                    "date": idx_list[i - 1],
                    "body_atr_ratio": float(body / curr["atr"]),
                    "filled": False,
                    "strength": 0.0,
                })
        else:
            # 큰 음봉 — 직전 양봉이 bullish OB
            if prev["close"] > prev["open"]:
                blocks.append({
                    "type": "bullish",
                    "low": float(min(prev["open"], prev["close"])),
                    "high": float(max(prev["open"], prev["close"])),
                    "mid": float((prev["open"] + prev["close"]) / 2),
                    "date": idx_list[i - 1],
                    "body_atr_ratio": float(body / curr["atr"]),
                    "filled": False,
                    "strength": 0.0,
                })

    # filled 체크 (이후 가격이 OB를 채웠는지) + 강도 점수
    for blk in blocks:
        # 이 OB 이후 데이터
        after_idx = [j for j, idx in enumerate(idx_list) if idx > blk["date"]]
        if not after_idx:
            continue
        after = df.iloc[after_idx[0]:]
        # bullish OB는 low까지 하락 시 filled; bearish OB는 high까지 상승 시 filled
        if blk["type"] == "bullish":
            blk["filled"] = bool((after["low"] <= blk["low"]).any())
        else:
            blk["filled"] = bool((after["high"] >= blk["high"]).any())
        # 강도: body × (1 - age_decay)
        age_days = (df.index[-1] - blk["date"]).days
        age_decay = min(age_days / 30, 1.0)  # 30일 지나면 0 가중
        blk["strength"] = blk["body_atr_ratio"] * (1.0 - age_decay * 0.6)

    # filled 안 된 것만 active
    return [b for b in blocks if not b["filled"]]


# ── Fair Value Gap (3봉 패턴 imbalance) ──────────────────────
def extract_fair_value_gaps(
    ohlcv: pd.DataFrame,
    lookback_days: int = 30,
) -> List[Dict]:
    """Fair Value Gap — 3봉 패턴의 가격 imbalance.

    Bullish FVG: candle[i+2].low > candle[i].high (위쪽 갭, 채워질 가능성)
    Bearish FVG: candle[i+2].high < candle[i].low (아래쪽 갭)
    """
    df = ohlcv.tail(lookback_days).copy()
    if len(df) < 5:
        return []

    rows = df.to_dict("records")
    idx_list = df.index.tolist()
    gaps = []

    for i in range(len(rows) - 2):
        c1, c2, c3 = rows[i], rows[i + 1], rows[i + 2]
        # Bullish FVG
        if c3["low"] > c1["high"]:
            gaps.append({
                "type": "bullish",
                "low": float(c1["high"]),
                "high": float(c3["low"]),
                "mid": float((c1["high"] + c3["low"]) / 2),
                "date": idx_list[i + 1],
                "size_pct": float((c3["low"] - c1["high"]) / c1["high"] * 100),
            })
        # Bearish FVG
        elif c3["high"] < c1["low"]:
            gaps.append({
                "type": "bearish",
                "low": float(c3["high"]),
                "high": float(c1["low"]),
                "mid": float((c3["high"] + c1["low"]) / 2),
                "date": idx_list[i + 1],
                "size_pct": float((c1["low"] - c3["high"]) / c3["high"] * 100),
            })

    return gaps


def _nearest_unfilled(
    blocks: List[Dict],
    current_price: float,
    side: str,
    max_dist_pct: float = 5.0,
) -> Optional[Dict]:
    """side='below' (bullish OB, 지지) 또는 'above' (bearish OB, 저항)."""
    if side == "below":
        cand = [b for b in blocks if b["high"] <= current_price * 1.005]
        if not cand:
            return None
        nearest = max(cand, key=lambda b: b["high"])
        if (current_price - nearest["high"]) / current_price * 100 > max_dist_pct:
            return None
        return nearest
    else:
        cand = [b for b in blocks if b["low"] >= current_price * 0.995]
        if not cand:
            return None
        nearest = min(cand, key=lambda b: b["low"])
        if (nearest["low"] - current_price) / current_price * 100 > max_dist_pct:
            return None
        return nearest


def _compute_ob_score(
    current_price: float,
    bullish_ob: Optional[Dict],
    bearish_ob: Optional[Dict],
    fvgs: List[Dict],
) -> float:
    """현재가 근처 OB/FVG 영향."""
    score = 0.0

    if bullish_ob:
        dist_pct = (current_price - bullish_ob["high"]) / current_price * 100
        if dist_pct < 2.0:
            # 강도(body_atr_ratio 기준) × 가중
            score += min(5, bullish_ob["strength"] * 2.5)

    if bearish_ob:
        dist_pct = (bearish_ob["low"] - current_price) / current_price * 100
        if dist_pct < 2.0:
            score -= min(5, bearish_ob["strength"] * 2.5)

    # FVG: 가까운 미충족 gap이 있으면 그 방향으로 약한 끌림
    for fvg in fvgs[:3]:
        if fvg["low"] <= current_price <= fvg["high"]:
            # 현재가가 FVG 안 (채워지는 중)
            continue
        dist = min(abs(current_price - fvg["low"]), abs(current_price - fvg["high"]))
        if dist / current_price * 100 < 3.0:
            # bullish FVG = 가격 위쪽 = +
            if fvg["type"] == "bullish" and current_price < fvg["mid"]:
                score += min(1.5, fvg["size_pct"] * 0.3)
            elif fvg["type"] == "bearish" and current_price > fvg["mid"]:
                score -= min(1.5, fvg["size_pct"] * 0.3)

    return float(np.clip(score, -10, 10))


def _ob_summary(ob: Optional[Dict]) -> Optional[Dict]:
    if not ob:
        return None
    return {
        "type": ob["type"],
        "low": round(ob["low"], 2),
        "high": round(ob["high"], 2),
        "body_atr_ratio": round(ob["body_atr_ratio"], 2),
        "strength": round(ob["strength"], 2),
        "date": str(ob["date"].date()) if hasattr(ob["date"], "date") else str(ob["date"]),
    }


def _fvg_summary(fvg: Dict) -> Dict:
    return {
        "type": fvg["type"],
        "low": round(fvg["low"], 2),
        "high": round(fvg["high"], 2),
        "size_pct": round(fvg["size_pct"], 2),
    }

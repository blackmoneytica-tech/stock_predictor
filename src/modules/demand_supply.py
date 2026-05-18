"""Module 9: DEMAND/SUPPLY zones — 매물대 (Volume Profile + Order Block + Strength).

핵심:
- HVN (High Volume Node): 거래량 누적 큰 가격대 = 강한 매물대
- LVN (Low Volume Node): 거래량 적은 가격대 = 진공 영역 (가격 빠르게 통과)
- POC (Point of Control): 단일 최대 거래량 가격
- Value Area High/Low (70% 거래량 구간) = 시장 합의 가격대
- Order Block: 큰 양/음봉 후 반전 zone (ICT)
- Strength score: 매물대별 강도 (volume + 최근성 + 옵션 OI 가중)

매물대 강도 정의 (백테스트로 검증할 가설):
  S(zone) = volume_pct × recency_factor × option_oi_factor
    volume_pct:     이 zone에서 거래된 총 거래량 / 전체 거래량
    recency_factor: 최근 거래 ~1.0, 오래된 거래 ~0.3 (지수 감쇠)
    option_oi_factor: 이 zone에 있는 옵션 OI 큰 strike와 일치 시 1.0~1.5

신호 출력 (-10 ~ +10):
- 현재가가 강한 demand zone (HVN, 가격 위 가까이): +5~+8 (지지 가까움)
- 현재가가 강한 supply zone 근처 아래: -5~-8 (저항 가까움)
- LVN (진공) 안에 있음: ±1 (방향성 가속)
- POC 위/아래: weak signal
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..types import ModuleOutput
from .base import AnalysisModule


class DemandSupplyModule(AnalysisModule):
    """매물대 분석 모듈."""

    def __init__(self, weight: float = 0.10):
        super().__init__("demand_supply", weight=weight)

    def analyze(self, data: Dict) -> ModuleOutput:
        ohlcv = data["ohlcv"]
        current_price = data["current_price"]
        option_strikes_oi = data.get("option_oi_by_strike", {})

        if len(ohlcv) < 30:
            return ModuleOutput(
                module_name=self.name, score=0.0,
                direction=self.score_to_direction(0),
                confidence=0.3,
                details={"reason": "insufficient_data"},
            )

        # 1) Volume Profile 계산 (최근 90일 기본)
        profile = compute_volume_profile(ohlcv, lookback_days=90, num_bins=50)

        # 2) Zone 식별 + 강도 점수
        zones = extract_zones(profile, ohlcv, current_price, option_strikes_oi)

        # 3) 현재가 근처 지지/저항 찾기
        demand = nearest_zone(zones, current_price, side="below")
        supply = nearest_zone(zones, current_price, side="above")

        # 4) Score 계산
        score = compute_zone_score(current_price, demand, supply, profile)

        return ModuleOutput(
            module_name=self.name,
            score=score,
            direction=self.score_to_direction(score),
            confidence=0.65,
            details={
                "poc": profile["poc"],
                "value_area_high": profile["vah"],
                "value_area_low": profile["val"],
                "nearest_demand": _zone_summary(demand),
                "nearest_supply": _zone_summary(supply),
                "all_zones": [_zone_summary(z) for z in zones[:8]],
                "in_value_area": profile["val"] <= current_price <= profile["vah"],
            },
        )


# ── Volume Profile ───────────────────────────────────────────
def compute_volume_profile(
    ohlcv: pd.DataFrame,
    lookback_days: int = 90,
    num_bins: int = 50,
) -> Dict:
    """Volume Profile — 가격을 bin으로 나누고 각 bin의 거래량 누적.

    각 봉의 거래량을 그 봉의 high~low 범위에 균등 배분 (단순 모델).
    POC = 거래량 최대 bin
    Value Area = 70% 거래량 포함 연속 구간 (POC 중심 양쪽 확장)
    """
    df = ohlcv.tail(lookback_days).copy()
    if df.empty:
        return _empty_profile()

    price_lo = float(df["low"].min())
    price_hi = float(df["high"].max())
    if price_hi <= price_lo:
        return _empty_profile()

    edges = np.linspace(price_lo, price_hi, num_bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    vol_per_bin = np.zeros(num_bins)

    # 각 봉의 거래량을 high~low에 균등 배분
    for _, row in df.iterrows():
        lo, hi, vol = row["low"], row["high"], row["volume"]
        if hi <= lo or vol <= 0:
            continue
        # bin index range
        i_lo = max(0, int((lo - price_lo) / (price_hi - price_lo) * num_bins))
        i_hi = min(num_bins - 1, int((hi - price_lo) / (price_hi - price_lo) * num_bins))
        if i_hi < i_lo:
            continue
        per_bin = vol / (i_hi - i_lo + 1)
        vol_per_bin[i_lo:i_hi + 1] += per_bin

    # POC
    poc_idx = int(np.argmax(vol_per_bin))
    poc_price = float(centers[poc_idx])

    # Value Area (70% 거래량) — POC에서 양쪽 확장
    total_vol = vol_per_bin.sum()
    if total_vol > 0:
        target = total_vol * 0.70
        acc = vol_per_bin[poc_idx]
        lo_i, hi_i = poc_idx, poc_idx
        while acc < target and (lo_i > 0 or hi_i < num_bins - 1):
            left_vol = vol_per_bin[lo_i - 1] if lo_i > 0 else 0
            right_vol = vol_per_bin[hi_i + 1] if hi_i < num_bins - 1 else 0
            if left_vol >= right_vol and lo_i > 0:
                lo_i -= 1
                acc += left_vol
            elif hi_i < num_bins - 1:
                hi_i += 1
                acc += right_vol
            else:
                break
        val = float(centers[lo_i])
        vah = float(centers[hi_i])
    else:
        val, vah = poc_price, poc_price

    return {
        "edges": edges,
        "centers": centers,
        "volumes": vol_per_bin,
        "poc": poc_price,
        "poc_idx": poc_idx,
        "vah": vah,
        "val": val,
        "total_volume": float(total_vol),
        "price_lo": price_lo,
        "price_hi": price_hi,
    }


def _empty_profile() -> Dict:
    return {
        "edges": np.array([]), "centers": np.array([]), "volumes": np.array([]),
        "poc": 0.0, "poc_idx": 0, "vah": 0.0, "val": 0.0,
        "total_volume": 0.0, "price_lo": 0.0, "price_hi": 0.0,
    }


# ── Zone Extraction + Strength ───────────────────────────────
def extract_zones(
    profile: Dict,
    ohlcv: pd.DataFrame,
    current_price: float,
    option_strikes_oi: Optional[Dict[float, int]] = None,
) -> List[Dict]:
    """매물대 zones 식별 + 강도 점수.

    Returns:
        [{center, low, high, volume, volume_pct, recency, option_boost,
          strength, side: 'demand'/'supply'}]
    """
    if not profile.get("volumes", np.array([])).size:
        return []

    centers = profile["centers"]
    volumes = profile["volumes"]
    total = profile["total_volume"]
    edges = profile["edges"]

    # 1) 상위 거래량 bin들을 zone 후보로 (top 20%)
    threshold = np.percentile(volumes[volumes > 0], 80) if (volumes > 0).any() else 0
    candidate_idx = np.where(volumes >= threshold)[0]

    # 2) 인접 bin들을 합쳐 zone (cluster)
    zones = []
    if len(candidate_idx) > 0:
        clusters = _cluster_adjacent(candidate_idx)
        for cluster in clusters:
            i_lo, i_hi = cluster[0], cluster[-1]
            z_low = float(edges[i_lo])
            z_high = float(edges[i_hi + 1])
            z_volume = float(volumes[i_lo:i_hi + 1].sum())
            z_center = (z_low + z_high) / 2

            # 강도 컴포넌트
            vol_pct = z_volume / total if total > 0 else 0
            recency = _recency_factor(ohlcv, z_low, z_high)
            opt_boost = _option_oi_factor(z_low, z_high, option_strikes_oi or {})

            # 종합 강도 (0~10 scale)
            strength = min(10, vol_pct * 100 * recency * opt_boost)

            side = "demand" if z_center < current_price else "supply"

            zones.append({
                "center": z_center,
                "low": z_low,
                "high": z_high,
                "volume": z_volume,
                "volume_pct": vol_pct,
                "recency": recency,
                "option_boost": opt_boost,
                "strength": strength,
                "side": side,
            })

    # 강도 desc 정렬
    zones.sort(key=lambda z: -z["strength"])
    return zones


def _cluster_adjacent(idx_array: np.ndarray) -> List[List[int]]:
    """인접 bin 인덱스 cluster (gap >= 2면 분리)."""
    if len(idx_array) == 0:
        return []
    clusters = [[int(idx_array[0])]]
    for i in idx_array[1:]:
        if i - clusters[-1][-1] <= 1:
            clusters[-1].append(int(i))
        else:
            clusters.append([int(i)])
    return clusters


def _recency_factor(ohlcv: pd.DataFrame, z_low: float, z_high: float) -> float:
    """이 zone에서 최근 거래된 비율 (지수 감쇠).

    최근 30일 거래량 / 전체 90일 거래량 비율. 0.3~1.5 범위.
    """
    if ohlcv.empty:
        return 1.0
    in_zone = (ohlcv["low"] <= z_high) & (ohlcv["high"] >= z_low)
    in_zone_df = ohlcv[in_zone]
    if in_zone_df.empty:
        return 0.5

    total = in_zone_df["volume"].sum()
    if total <= 0:
        return 0.5

    recent_cutoff = in_zone_df.index.max() - pd.Timedelta(days=30)
    recent_vol = in_zone_df[in_zone_df.index >= recent_cutoff]["volume"].sum()
    ratio = recent_vol / total

    # 0.3 (오래된) ~ 1.5 (최근 집중) 매핑
    return float(0.3 + ratio * 1.2)


def _option_oi_factor(
    z_low: float,
    z_high: float,
    option_oi: Dict[float, int],
) -> float:
    """이 zone에 옵션 OI 큰 strike이 있으면 강도 boost (1.0 ~ 1.5)."""
    if not option_oi:
        return 1.0

    strikes_in_zone = [s for s in option_oi if z_low <= s <= z_high]
    if not strikes_in_zone:
        return 1.0

    total_oi = sum(option_oi.values())
    zone_oi = sum(option_oi[s] for s in strikes_in_zone)
    if total_oi <= 0:
        return 1.0

    ratio = zone_oi / total_oi
    # 0 → 1.0, 1.0 → 1.5
    return float(1.0 + ratio * 0.5)


# ── 현재가 근처 zone 검색 ────────────────────────────────────
def nearest_zone(zones: List[Dict], current_price: float, side: str) -> Optional[Dict]:
    """side='below' (demand) 또는 'above' (supply)."""
    if side == "below":
        candidates = [z for z in zones if z["high"] <= current_price * 1.005]
        if not candidates:
            return None
        # 가장 가까운 (가장 높은 high)
        return max(candidates, key=lambda z: z["high"])
    else:  # above
        candidates = [z for z in zones if z["low"] >= current_price * 0.995]
        if not candidates:
            return None
        return min(candidates, key=lambda z: z["low"])


# ── Score ────────────────────────────────────────────────────
def compute_zone_score(
    current_price: float,
    demand: Optional[Dict],
    supply: Optional[Dict],
    profile: Dict,
) -> float:
    """현재가 위치 + 매물대 강도 → 점수.

    2026-05-17 큰 표본 backtest (n=541 touched, 50 종목 × 12 시점 = 1510 zones):
    - 전체 bounce rate 76.7% (매물대 = 강한 시그널)
    - **거리가 진짜 predictor**:
        <1%:    54.6% bounce (이미 흔들리는 영역, weak)
        1-2%:   72.5% (평균)
        2-3%:   84.4% (강함)
        3-5%:   89.2% (최강)
        5-10%:  87.5%
        10%+:   100%
    - side: supply <1% 깨질 가능 57% (skip 권장), demand 2-10% 90%+ bounce
    - **rollback**: strength quartile 가중 X (Q1=82% Q4=75% noise)
    - **rollback**: option_boost X (효과 1.4%p 미미)
    - **rollback**: supply 1.25배 X (큰 표본에서 demand > supply)

    새 모델: distance-weighted (단순+robust):
      base = 7 (만남=강한 신호 76.7% 반영)
      distance multiplier:
        <1%:    0.3  (이미 흔들리는 영역)
        1-2%:   0.6
        2-3%:   0.9
        3-10%:  1.0  (최강 — backtest 89%)
        10%+:   0.9  (희귀 케이스)
    """
    score = 0.0

    if demand:
        dist_pct = (current_price - demand["high"]) / current_price * 100
        mult = _distance_multiplier(dist_pct)
        # base 7 × multiplier — 그러나 supply <1%는 fragmenting (skip)
        if dist_pct < 1.0:
            # 가까운 demand도 약함 (66% bounce, ~random)
            score += 7 * 0.3
        else:
            score += 7 * mult

    if supply:
        dist_pct = (supply["low"] - current_price) / current_price * 100
        if dist_pct < 1.0:
            # supply <1% = 43% bounce = 57% break = 매수 비추천 약한 -1 (저항 깨짐)
            score -= 1.5
        else:
            mult = _distance_multiplier(dist_pct)
            score -= 7 * mult

    # Value Area 안 = 합의 영역 (POC 자석) — 약화
    if profile["val"] <= current_price <= profile["vah"]:
        score *= 0.7

    return float(np.clip(score, -10, 10))


def _distance_multiplier(dist_pct: float) -> float:
    """Backtest n=541로 검증된 거리 multiplier.

    <1%:    0.3  (54.6% bounce)
    1-2%:   0.6  (72.5%)
    2-3%:   0.9  (84.4%)
    3-10%:  1.0  (89% — 최강)
    10%+:   0.9  (100% but rare)
    """
    if dist_pct < 1.0:
        return 0.3
    if dist_pct < 2.0:
        return 0.6
    if dist_pct < 3.0:
        return 0.9
    if dist_pct < 10.0:
        return 1.0
    return 0.9


def _zone_summary(zone: Optional[Dict]) -> Optional[Dict]:
    """log/output용 zone 간단 요약."""
    if not zone:
        return None
    return {
        "low": round(zone["low"], 2),
        "high": round(zone["high"], 2),
        "center": round(zone["center"], 2),
        "strength": round(zone["strength"], 2),
        "volume_pct": round(zone["volume_pct"], 4),
        "side": zone["side"],
    }

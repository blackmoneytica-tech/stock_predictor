"""Options Unusual Activity — 큰 거래량 / OI spike 감지.

가설: 기관 큰 베팅 = 다음 가격 방향 단서.
- volume / OI > 1.0 = 그날 OI보다 많은 거래 = unusual
- volume * 가격 = $ flow
- call vs put 방향성

Marketdata.app chain에서 추출. yfinance 데이터로도 부분 가능 (volume만).
"""
from __future__ import annotations

from typing import Dict, List, Optional


def detect_unusual(
    options_chain: Dict[str, Dict[float, Dict]],
    expiration: str,
    current_price: float,
    vol_oi_ratio_min: float = 0.5,
    min_volume: int = 500,
) -> Dict:
    """단일 만기 chain에서 unusual activity 추출.

    Returns:
        {
            unusual_calls: [(strike, vol, oi, ratio), ...] desc by vol
            unusual_puts:  [...]
            net_call_flow_usd: 콜 큰 거래 $
            net_put_flow_usd: 풋 큰 거래 $
            score: -10~+10 (call flow 크면 +, put 크면 -)
            direction_bias: 'bullish' / 'bearish' / 'neutral'
        }
    """
    if expiration not in options_chain or not options_chain[expiration]:
        return _empty()

    chain = options_chain[expiration]
    unusual_calls = []
    unusual_puts = []
    call_flow = 0.0
    put_flow = 0.0

    for strike, slot in chain.items():
        c_vol = slot.get("call_volume", 0)
        p_vol = slot.get("put_volume", 0)
        c_oi = slot.get("call_oi", 0)
        p_oi = slot.get("put_oi", 0)

        # call 큰 거래
        if c_vol >= min_volume:
            ratio = c_vol / max(c_oi, 1)
            if ratio >= vol_oi_ratio_min or c_oi == 0:
                unusual_calls.append({
                    "strike": float(strike),
                    "volume": int(c_vol),
                    "oi": int(c_oi),
                    "ratio": round(ratio, 2),
                    "dist_pct": round((strike - current_price) / current_price * 100, 1),
                })
                # $ flow proxy = vol × |strike-current| (premium 추정)
                # OTM = 가격 작음, ATM = 큼. 단순화: vol × strike × 0.05
                call_flow += c_vol * float(strike) * 0.03

        if p_vol >= min_volume:
            ratio = p_vol / max(p_oi, 1)
            if ratio >= vol_oi_ratio_min or p_oi == 0:
                unusual_puts.append({
                    "strike": float(strike),
                    "volume": int(p_vol),
                    "oi": int(p_oi),
                    "ratio": round(ratio, 2),
                    "dist_pct": round((strike - current_price) / current_price * 100, 1),
                })
                put_flow += p_vol * float(strike) * 0.03

    unusual_calls.sort(key=lambda x: -x["volume"])
    unusual_puts.sort(key=lambda x: -x["volume"])

    # Score: call flow - put flow, 정규화
    net_flow = call_flow - put_flow
    total_flow = call_flow + put_flow
    if total_flow > 0:
        flow_ratio = net_flow / total_flow  # -1 ~ +1
        score = float(max(-10, min(10, flow_ratio * 8)))  # 정규화
    else:
        score = 0.0

    direction = "bullish" if score > 1 else "bearish" if score < -1 else "neutral"

    return {
        "unusual_calls": unusual_calls[:5],
        "unusual_puts": unusual_puts[:5],
        "net_call_flow_usd": round(call_flow, 0),
        "net_put_flow_usd": round(put_flow, 0),
        "score": score,
        "direction_bias": direction,
        "n_call_unusual": len(unusual_calls),
        "n_put_unusual": len(unusual_puts),
    }


def _empty() -> Dict:
    return {
        "unusual_calls": [], "unusual_puts": [],
        "net_call_flow_usd": 0, "net_put_flow_usd": 0,
        "score": 0.0, "direction_bias": "neutral",
        "n_call_unusual": 0, "n_put_unusual": 0,
    }

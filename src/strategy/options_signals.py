"""옵션 신호 종합 평가 (2026-05-20 backtest 검증).

Tier 1 (최강 alpha):
  - is_options_sweet_spot: put_wall_dist ∈ [-5,0] + news ≥ +1 + iv_rank < 0.5
    → 10d 95.8% win, +10.81%/trade (n=24)
  - is_iv_underpriced: iv_rank < 0.3
    → 10d 93% win, +17.72%/trade (n=29)

❌ 회피:
  - iv_rank > 0.7 (catalyst 위험)
  - call_wall_dist > 10% (약한 종목)
  - news_score ≤ -2 (단독 부정 뉴스)
"""
from __future__ import annotations

from typing import Dict


def extract_walls(option_oi_by_strike: Dict, options_chain: Dict, target_exp: str) -> Dict:
    """call/put wall + vol/OI ratio 추출.

    Args:
        option_oi_by_strike: {strike: total_oi} (system._fetch_data가 이미 계산)
        options_chain: full chain dict
        target_exp: target expiration key
    """
    if not options_chain or target_exp not in options_chain:
        return {}
    strikes_data = options_chain[target_exp]

    call_oi_map = {s: (d.get('call_oi', 0) or 0) for s, d in strikes_data.items()}
    put_oi_map = {s: (d.get('put_oi', 0) or 0) for s, d in strikes_data.items()}
    if not call_oi_map:
        return {}

    call_wall = max(call_oi_map, key=call_oi_map.get)
    put_wall = max(put_oi_map, key=put_oi_map.get) if put_oi_map else call_wall

    total_vol = sum((d.get('call_volume', 0) or 0) + (d.get('put_volume', 0) or 0)
                    for d in strikes_data.values())
    total_oi = sum(call_oi_map.values()) + sum(put_oi_map.values())
    vol_oi_ratio = total_vol / max(total_oi, 1)

    return {
        "call_wall": float(call_wall),
        "put_wall": float(put_wall),
        "call_wall_oi": int(call_oi_map.get(call_wall, 0)),
        "put_wall_oi": int(put_oi_map.get(put_wall, 0)),
        "vol_oi_ratio": round(vol_oi_ratio, 3),
        "total_oi": int(total_oi),
    }


def is_options_sweet_spot(
    put_wall_dist_pct: float, news_score: float, iv_rank: float,
) -> bool:
    """Tier 1: 10d 95.8% win 검증 룰."""
    return (
        -5 <= put_wall_dist_pct <= 0
        and news_score >= 1
        and iv_rank < 0.5
    )


def is_iv_underpriced(iv_rank: float) -> bool:
    """Tier 2: 10d 93% win, +17.72% (옵션 저평가)."""
    return iv_rank < 0.3


def evaluate_options_signals(
    current_price: float,
    options_details: Dict,
    walls: Dict,
    news_score: float = 0.0,
    news_n: int = 0,
) -> Dict:
    """options 신호 종합 tier + UI용 dict.

    Returns: {
      tier: 'options_sweet_spot' / 'iv_underpriced' / 'iv_overpriced' /
            'call_wall_warning' / 'news_positive' / 'normal',
      call_wall, put_wall, call_wall_dist_pct, put_wall_dist_pct,
      vol_oi_ratio, iv_rank, iv_rank_label, news_score,
      backtest_win_pct, tagline, tone
    }
    """
    iv_rank = options_details.get('iv_rank', 0.5) or 0.5
    if iv_rank > 1:
        iv_rank = iv_rank / 100  # if % stored as 70 instead of 0.7
    call_wall = walls.get('call_wall', 0)
    put_wall = walls.get('put_wall', 0)
    vol_oi_ratio = walls.get('vol_oi_ratio', 0)

    call_wall_dist = ((call_wall - current_price) / current_price * 100
                      if call_wall and current_price else 0)
    put_wall_dist = ((put_wall - current_price) / current_price * 100
                     if put_wall and current_price else 0)

    # IV rank label
    if iv_rank < 0.3:
        iv_label = f"저평가 ({iv_rank*100:.0f}% — 옵션 쌈)"
    elif iv_rank > 0.7:
        iv_label = f"고평가 ({iv_rank*100:.0f}% — catalyst 위험)"
    else:
        iv_label = f"적정 ({iv_rank*100:.0f}%)"

    # tier 판정
    if is_options_sweet_spot(put_wall_dist, news_score, iv_rank):
        tier = "options_sweet_spot"
        tagline = "🔥 Put wall 근접 + 뉴스 긍정 + IV 적정 — 검증된 최강 신호"
        backtest = "n=24, 10d 95.8% win, +10.81%/trade"
        tone = "bull"
    elif is_iv_underpriced(iv_rank):
        tier = "iv_underpriced"
        tagline = "🔥 옵션 저평가 — 변동성 catalyst 대기 = 상승 잠재력"
        backtest = "n=29, 10d 93% win, +17.72%/trade"
        tone = "bull"
    elif iv_rank > 0.7:
        tier = "iv_overpriced"
        tagline = "⚠️ 옵션 비쌈 — 큰 catalyst 임박 (양방향 위험)"
        backtest = "n=104, 10d 63.5% win (baseline 79% 미달)"
        tone = "warn"
    elif call_wall_dist > 10:
        tier = "call_wall_warning"
        tagline = "⚠️ Call wall 멀리 (>10%) — 상방 모멘텀 약함"
        backtest = "n=5, 10d 20% win, -1.04%"
        tone = "warn"
    elif news_score >= 2:
        tier = "news_positive"
        tagline = "🟢 뉴스 매우 긍정 — baseline 상회"
        backtest = "n=153, 10d 83% win, +10.3%"
        tone = "bull"
    else:
        tier = "normal"
        tagline = ""
        backtest = ""
        tone = "neutral"

    return {
        "tier": tier,
        "call_wall": round(call_wall, 2) if call_wall else None,
        "put_wall": round(put_wall, 2) if put_wall else None,
        "call_wall_dist_pct": round(call_wall_dist, 2),
        "put_wall_dist_pct": round(put_wall_dist, 2),
        "vol_oi_ratio": vol_oi_ratio,
        "iv_rank": round(iv_rank, 3),
        "iv_rank_label": iv_label,
        "news_score": round(news_score, 2),
        "news_n": news_n,
        "backtest": backtest,
        "tagline": tagline,
        "tone": tone,
        "alpha": tier in ("options_sweet_spot", "iv_underpriced"),
    }

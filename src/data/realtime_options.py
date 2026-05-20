"""실시간 옵션 chain 통합 fetcher.

우선순위 (2026-05-20 갱신):
  1) **CBOE** — 무료, 가입X, 한국 IP OK, 옵션 거래소 본가, 15분 지연 ⭐ primary
  2) Marketdata.app — 무료 100/day, IP 1개 제한 → 차단 자주 발생
  3) yfinance — 무료, 무제한, OI=0/IV=placeholder 자주
  4) Alpha Vantage — 25 req/day, EOD only

설계:
- 동일 schema 반환: {expiration: {strike: {call_oi, put_oi, call_iv, put_iv, iv, ...}}}
- get_realtime_chain(ticker, horizon_days) 단일 진입점
- 자동 fallback chain
- 모든 source 실패 시 RuntimeError
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional

from ._common import env
from .options_chain import get_options_chain as yf_get_chain
from .options_chain import list_expirations as yf_list_expirations

try:
    from .cboe_options import (
        get_chain as cboe_get_chain,
        pick_horizon_expiration as cboe_pick_exp,
    )
    _HAS_CBOE = True
except ImportError:
    _HAS_CBOE = False

try:
    from .alpha_vantage_options import get_realtime_chain as av_get_chain
    _HAS_AV = True
except ImportError:
    _HAS_AV = False

try:
    from .marketdata_options import (
        get_chain as md_get_chain,
        pick_horizon_expiration as md_pick_exp,
    )
    _HAS_MD = True
except ImportError:
    _HAS_MD = False

log = logging.getLogger(__name__)


def pick_expiration_horizon(expirations: list, horizon_days: int) -> Optional[str]:
    """horizon_days에 가장 가까운 (≥) 만기 선택."""
    if not expirations:
        return None
    target = (datetime.utcnow() + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
    future = [e for e in expirations if e >= target]
    return future[0] if future else expirations[-1]


def get_realtime_chain(
    ticker: str,
    horizon_days: int = 5,
    prefer: str = "cboe",
) -> Dict:
    """실시간 옵션 chain — 우선순위:
    1. **CBOE** (무료, 가입X, 한국 IP OK, 옵션 거래소 본가) ⭐
    2. Marketdata.app (무료 100/day, IP 1개 제한 → 차단 가능)
    3. yfinance (무제한, 단 OI=0/IV placeholder 자주)
    4. Alpha Vantage (25 req/day)

    OI 신뢰도 게이트: chain의 총 OI 합이 너무 작으면 (1000 미만) 다음 source 시도.

    Args:
        prefer: 'cboe' | 'marketdata' | 'yfinance' | 'alphavantage'
    """
    sources_tried = []
    last_err: Optional[Exception] = None

    if prefer == "yfinance":
        order = ["yfinance", "cboe", "marketdata", "alphavantage"]
    elif prefer == "marketdata":
        order = ["marketdata", "cboe", "yfinance", "alphavantage"]
    elif prefer == "alphavantage":
        order = ["alphavantage", "cboe", "marketdata", "yfinance"]
    else:
        # default: cboe primary (가입X, 한국 IP 안정), 그 후 marketdata → yfinance
        order = ["cboe", "marketdata", "yfinance", "alphavantage"]

    for src in order:
        try:
            if src == "cboe":
                if not _HAS_CBOE:
                    sources_tried.append(f"{src} (skipped: no module)")
                    continue
                exp = cboe_pick_exp(ticker, horizon_days=horizon_days)
                if not exp:
                    sources_tried.append(f"{src} (no exp)")
                    continue
                chain = cboe_get_chain(ticker, exp)
            elif src == "marketdata":
                if not _HAS_MD or not env("MARKETDATA_KEY"):
                    sources_tried.append(f"{src} (skipped: no key)")
                    continue
                exp = md_pick_exp(ticker, horizon_days=horizon_days)
                if not exp:
                    sources_tried.append(f"{src} (no exp)")
                    continue
                chain = md_get_chain(ticker, exp)
            elif src == "yfinance":
                chain = _fetch_yfinance(ticker, horizon_days)
            elif src == "alphavantage":
                if not _HAS_AV or not env("ALPHA_VANTAGE_KEY"):
                    sources_tried.append(f"{src} (skipped: no key)")
                    continue
                chain = av_get_chain(ticker, horizon_days=horizon_days)
            else:
                continue

            if chain:
                exp = next(iter(chain))
                if chain[exp]:
                    # OI 신뢰도 체크
                    total_oi = sum(
                        (slot.get("call_oi", 0) or 0) + (slot.get("put_oi", 0) or 0)
                        for slot in chain[exp].values()
                    )
                    if total_oi < 1000:
                        log.warning(
                            "realtime options [%s] %s exp=%s low OI (%d) — try next",
                            src, ticker, exp, total_oi,
                        )
                        sources_tried.append(f"{src} (low_oi={total_oi})")
                        continue
                    log.info("realtime options [%s] %s exp=%s strikes=%d OI=%d",
                             src, ticker, exp, len(chain[exp]), total_oi)
                    return chain
            sources_tried.append(f"{src} (empty)")
        except Exception as e:
            sources_tried.append(f"{src} ({type(e).__name__})")
            last_err = e

    raise RuntimeError(
        f"실시간 옵션 fetch 실패 [{ticker}]: tried={sources_tried}, "
        f"last={last_err}"
    )


def _fetch_yfinance(ticker: str, horizon_days: int) -> Dict:
    """yfinance — 무료 무제한, 15분 지연."""
    exps = yf_list_expirations(ticker)
    if not exps:
        raise RuntimeError(f"yfinance: no options for {ticker}")
    chosen = pick_expiration_horizon(exps, horizon_days)
    if not chosen:
        raise RuntimeError(f"yfinance: no expiration for horizon {horizon_days}d")
    return yf_get_chain(ticker, chosen)


# ── ATM IV 추출 ─────────────────────────────────────────────
def get_atm_iv_realtime(ticker: str, current_price: float, horizon_days: int = 5) -> float:
    """ATM strike의 현재 IV (15분 지연)."""
    chain = get_realtime_chain(ticker, horizon_days=horizon_days)
    exp = next(iter(chain))
    strikes = list(chain[exp].keys())
    if not strikes:
        return float("nan")
    atm = min(strikes, key=lambda s: abs(s - current_price))
    iv = chain[exp][atm].get("iv", float("nan"))
    return float(iv) if iv == iv else float("nan")  # nan check

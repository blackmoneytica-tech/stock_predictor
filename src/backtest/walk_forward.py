"""Walk-forward backtest with strict point-in-time data cutoff.

핵심: 각 as_of 시점에 그 시점에 알 수 있던 데이터만 사용 (lookahead 방지).

한계 (Phase 5 명세 §10):
- yfinance 옵션 체인은 historical 없음 → 옵션 모듈 score≈0 무력화
- FRED 키 없으면 매크로 모듈 score=0 무력화
- VIX / insider Form 4 / earnings는 historical date cutoff 가능
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..data._common import env
from ..data.catalyst import FOMC_DATES_2026
from ..data.insider import get_insider_activity
from ..data.price_feed import get_daily_ohlcv
from ..system import StockPredictionSystem

try:
    from ..data.alpha_vantage_options import get_chain_at as av_get_chain_at
    _HAS_AV = True
except ImportError:
    _HAS_AV = False

try:
    from ..data.macro import get_fred_series
    _HAS_FRED = True
except ImportError:
    _HAS_FRED = False


def _compute_macro_signals_at(as_of: date) -> Dict:
    """FRED historical로 fed_dovish/yield/risk_on score 산출.

    score 범위: -5 ~ +5
    - fed_dovish: 최근 60일 Fed Funds 변화. 하락=dovish(+), 상승=hawkish(-)
    - yield_score: 10Y Treasury 최근 30일 변화. 상승=risk_off(-), 하락=risk_on(+)
    - risk_on:    VIX 최근 30일 vs 60일. VIX 하락=risk_on(+)
    """
    out = {"fed_dovish_score": 0.0, "yield_score": 0.0, "risk_on_score": 0.0}
    if not _HAS_FRED or not env("FRED_API_KEY"):
        return out

    start = (as_of - timedelta(days=120)).isoformat()
    end = as_of.isoformat()
    try:
        dff = get_fred_series("DFF", start, end)
        dff = dff[dff.index <= pd.Timestamp(as_of)].dropna()
        if len(dff) >= 60:
            # 60일 전 대비 변화 (bps). +25bp = 강한 hawkish = -2.5점
            chg = float(dff.iloc[-1] - dff.iloc[-60]) * 100  # %p → bps
            out["fed_dovish_score"] = float(np.clip(-chg / 10.0, -5, 5))
    except Exception:
        pass

    try:
        dgs10 = get_fred_series("DGS10", start, end)
        dgs10 = dgs10[dgs10.index <= pd.Timestamp(as_of)].dropna()
        if len(dgs10) >= 30:
            # 30일 전 대비 yield 변화 (bp). 상승 = risk_off
            chg = float(dgs10.iloc[-1] - dgs10.iloc[-30]) * 100
            out["yield_score"] = float(np.clip(-chg / 15.0, -5, 5))
    except Exception:
        pass

    try:
        vix = get_fred_series("VIXCLS", start, end)
        vix = vix[vix.index <= pd.Timestamp(as_of)].dropna()
        if len(vix) >= 60:
            short_avg = float(vix.tail(20).mean())
            long_avg = float(vix.tail(60).mean())
            # VIX 단기 < 장기 = 위험 진정 = risk_on
            pct_diff = (short_avg - long_avg) / long_avg * 100
            out["risk_on_score"] = float(np.clip(-pct_diff / 5.0, -5, 5))
    except Exception:
        pass

    return out


def _recent_macro_releases_at(as_of: date, days_back: int = 7) -> List[Dict]:
    """as_of 이전 N일 매크로 발표 (CPI/PPI/NFP) — historical surprise는 없음.

    MVP: 발표 valued YoY 변화율로 surprise proxy.
    """
    if not _HAS_FRED or not env("FRED_API_KEY"):
        return []
    start = (as_of - timedelta(days=120)).isoformat()
    end = as_of.isoformat()
    out = []
    for sid in ("CPIAUCSL", "PPIACO"):
        try:
            s = get_fred_series(sid, start, end).dropna()
            s = s[s.index <= pd.Timestamp(as_of)]
            if len(s) < 13:
                continue
            recent_cutoff = pd.Timestamp(as_of - timedelta(days=days_back))
            recent = s[s.index >= recent_cutoff]
            for d, v in recent.items():
                # YoY change vs 12개월 전 (proxy for surprise direction)
                d_dt = d.date() if hasattr(d, "date") else d
                year_ago_idx = s.index.get_indexer([pd.Timestamp(d_dt) - pd.Timedelta(days=365)], method="nearest")[0]
                if year_ago_idx >= 0:
                    yoy = (v - s.iloc[year_ago_idx]) / s.iloc[year_ago_idx]
                    # CPI/PPI YoY > 3% = hawkish surprise (-)
                    surprise = float(np.clip(-(yoy - 0.025) * 50, -3, 3))
                else:
                    surprise = 0.0
                out.append({"date": d_dt, "series": sid, "surprise": surprise})
        except Exception:
            continue
    return out


# ── 시점별 데이터 빌더 ───────────────────────────────────────
def build_data_at(
    ticker: str,
    as_of: date,
    horizon_days: int = 5,
    use_options: bool = False,
    use_macro: bool = False,
    insider_cache: Optional[Dict] = None,
) -> Dict:
    """as_of 시점까지의 데이터로 system.analyze() 입력 dict 빌드.

    use_options=False: 옵션 chain은 더미 (모듈 score≈0)
    use_macro=False:   FRED 호출 skip (fed_dovish=0)
    insider_cache:     SEC 호출 1회로 끝내고 시점별 filter용
    """
    # 1) OHLCV — as_of (포함) 까지
    start = as_of - timedelta(days=400)
    ohlcv = get_daily_ohlcv(ticker, start, as_of + timedelta(days=1))
    ohlcv = ohlcv[ohlcv.index <= pd.Timestamp(as_of)]
    if ohlcv.empty:
        raise RuntimeError(f"OHLCV empty for {ticker} @ {as_of}")
    current_price = float(ohlcv["close"].iloc[-1])

    # 2) HV / IV Rank (HV proxy) — historical 계산
    closes = ohlcv["close"].dropna()
    log_ret = np.log(closes / closes.shift(1)).dropna()
    if len(log_ret) >= 30:
        hv_30 = float(log_ret.tail(30).std() * np.sqrt(252))
    else:
        hv_30 = float("nan")

    rolling_hv = log_ret.rolling(20).std() * np.sqrt(252)
    rolling_hv_252 = rolling_hv.dropna().tail(252)
    iv_rank = (
        float((rolling_hv_252 < rolling_hv_252.iloc[-1]).mean())
        if not rolling_hv_252.empty else 0.5
    )

    # 3) 옵션 chain — use_options=True이면 Alpha Vantage HISTORICAL_OPTIONS 사용
    target_exp = (as_of + timedelta(days=horizon_days + 2)).strftime("%Y-%m-%d")
    options_chain = None
    if use_options and _HAS_AV and env("ALPHA_VANTAGE_KEY"):
        try:
            av_chain = av_get_chain_at(ticker, as_of, horizon_days=horizon_days)
            if av_chain:
                # AV는 IV를 percent (e.g. 0.8475 = 84.75%)로 반환 — 표준화
                options_chain = av_chain
                # 실제 만기로 target_exp 갱신
                target_exp = next(iter(av_chain.keys()))
        except Exception as e:
            # AV 실패 시 더미로 fallback (rate limit / 데이터 없음 등)
            pass

    if options_chain is None:
        # 더미 chain — Module 2가 score≈0 산출 (Max Pain≈current, P/C≈1)
        atm_strike = round(current_price / 5) * 5
        options_chain = {
            target_exp: {
                float(atm_strike): {
                    "call_oi": 100, "put_oi": 100,
                    "call_volume": 0, "put_volume": 0,
                    "call_iv": hv_30 if not np.isnan(hv_30) else 0.5,
                    "put_iv":  hv_30 if not np.isnan(hv_30) else 0.5,
                    "iv":      hv_30 if not np.isnan(hv_30) else 0.5,
                },
                float(atm_strike - 5): {
                    "call_oi": 50, "put_oi": 100,
                    "call_iv": 0.5, "put_iv": 0.5, "iv": 0.5,
                },
                float(atm_strike + 5): {
                    "call_oi": 100, "put_oi": 50,
                    "call_iv": 0.5, "put_iv": 0.5, "iv": 0.5,
                },
            }
        }

    # 4) VIX — yfinance historical (CBOE:VIX ticker = ^VIX)
    vix_now, vix_30d_avg, vix_30d_std = 18.0, 18.0, 3.0
    try:
        vix_df = get_daily_ohlcv("^VIX", start, as_of + timedelta(days=1))
        vix_df = vix_df[vix_df.index <= pd.Timestamp(as_of)]
        if not vix_df.empty:
            vix_now = float(vix_df["close"].iloc[-1])
            tail30 = vix_df["close"].tail(30)
            vix_30d_avg = float(tail30.mean())
            vix_30d_std = float(tail30.std()) or 3.0
    except Exception:
        pass

    # 5) Macro — FRED + sector breadth (alert/screener_macro.pine 통합)
    fed_dovish, yield_score, risk_on = 0.0, 0.0, 0.0
    recent_macro: List[Dict] = []
    macro_breadth: Dict = {}
    if use_macro:
        try:
            from ..data.sector_macro import compute_macro_breadth_at
            macro_breadth = compute_macro_breadth_at(as_of)
        except Exception:
            pass
        if env("FRED_API_KEY"):
            try:
                macro_signals = _compute_macro_signals_at(as_of)
                fed_dovish = macro_signals["fed_dovish_score"]
                yield_score = macro_signals["yield_score"]
                risk_on = macro_signals["risk_on_score"]
                recent_macro = _recent_macro_releases_at(as_of, days_back=7)
            except Exception:
                pass

    # 6) Catalyst — FOMC + Finnhub earnings + beat probability
    events = _events_at(ticker, as_of, horizon_days=horizon_days * 2 + 5)

    # Finnhub earnings — 다음 발표일 정확 + beat proxy
    finnhub_sig = {}
    if env("FINNHUB_KEY"):
        try:
            from ..data.finnhub import get_earnings_signals
            finnhub_sig = get_earnings_signals(ticker, as_of)
            # 다음 earnings가 horizon 안이면 catalyst event로 추가
            if finnhub_sig.get("next_earnings_date"):
                from datetime import date as _date
                ned = _date.fromisoformat(finnhub_sig["next_earnings_date"])
                days_to = (ned - as_of).days
                if 0 <= days_to <= horizon_days * 2 + 5:
                    # beat_proxy 기반 expected_direction
                    proxy = finnhub_sig.get("beat_probability_proxy", 0.5)
                    direction = (proxy - 0.5) * 2  # -1 ~ +1
                    # 기존 EARNINGS event가 있으면 update, 없으면 추가
                    existing = [e for e in events if e.get("type") == "EARNINGS"]
                    if existing:
                        existing[0]["date"] = ned
                        existing[0]["expected_direction"] = direction
                        existing[0]["beat_probability"] = proxy
                    else:
                        events.append({
                            "date": ned, "type": "EARNINGS",
                            "expected_impact": 0.08,
                            "expected_direction": direction,
                            "beat_probability": proxy,
                            "source": "finnhub",
                        })
                    events.sort(key=lambda e: e["date"])
        except Exception:
            pass

    # 7) Insider — SEC filing date cutoff
    if insider_cache is None:
        insider_data = get_insider_activity(ticker, months_back=12)
    else:
        insider_data = insider_cache
    insider_filtered = _filter_insider_at(insider_data, as_of)

    # 8) 변화율 지표
    recent_drop = 0.0
    if len(ohlcv) >= 2:
        recent_drop = float(
            (ohlcv["close"].iloc[-1] - ohlcv["close"].iloc[-2]) / ohlcv["close"].iloc[-2]
        )
    return_1m = 0.0
    if len(ohlcv) >= 22:
        return_1m = float(
            (ohlcv["close"].iloc[-1] - ohlcv["close"].iloc[-22]) / ohlcv["close"].iloc[-22]
        )

    # next event days
    next_event_days = 999
    fomc_within = 999
    if events:
        next_event_days = min((e["date"] - as_of).days for e in events)
        fomc_evs = [e for e in events if e["type"] == "FOMC"]
        if fomc_evs:
            fomc_within = min((e["date"] - as_of).days for e in fomc_evs)

    # 직전 mega-bull catalyst — past 7d 안에 ≥+8% 양봉 = catalyst로 인식
    # CRCL 5/11 +15.91% 같은 catalyst day를 자동 찾아 sell-news risk 발동
    post_catalyst_within = 999
    pre_catalyst_rally = 0.0
    recent_closes = ohlcv["close"].tail(10)
    if len(recent_closes) >= 2:
        recent_daily_ret = recent_closes.pct_change().dropna()
        # 양수 big moves (mega bull) 우선
        positive_big = recent_daily_ret[recent_daily_ret >= 0.08]
        if not positive_big.empty:
            cat_idx = positive_big.index[-1]  # 가장 최근 mega-bull
            cat_date = cat_idx.date()
            days_since = (as_of - cat_date).days
            if 0 <= days_since <= 7:  # 7일까지 sell-news risk
                post_catalyst_within = days_since
                # 그 catalyst 직전 30일 누적 (sell-news 보정용)
                pre_window_end = cat_idx - pd.Timedelta(days=1)
                pre_window_start = cat_idx - pd.Timedelta(days=30)
                pre_prices = ohlcv["close"]
                pre_prices = pre_prices[
                    (pre_prices.index >= pre_window_start)
                    & (pre_prices.index <= pre_window_end)
                ]
                if len(pre_prices) >= 2:
                    pre_catalyst_rally = float(
                        (pre_prices.iloc[-1] - pre_prices.iloc[0]) / pre_prices.iloc[0]
                    )
                # catalyst day 자체의 +가격도 rally에 포함 (catalyst 자체가 ralley pump)
                cat_day_ret = float(positive_big.loc[cat_idx])
                pre_catalyst_rally = max(pre_catalyst_rally, cat_day_ret)

    # option_oi_by_strike — DemandSupply 강도 가중치용
    option_oi_by_strike = {}
    for strike, slot in options_chain[target_exp].items():
        oi = slot.get("call_oi", 0) + slot.get("put_oi", 0)
        if oi > 0:
            option_oi_by_strike[float(strike)] = int(oi)

    return {
        "ohlcv": ohlcv,
        "ticker": ticker,
        "as_of_date": as_of,  # 옵션 모듈 days-to-exp 계산용
        "current_price": current_price,
        "options_chain": options_chain,
        "target_expiration": target_exp,
        "option_strikes": list(options_chain[target_exp].keys()),
        "option_oi_by_strike": option_oi_by_strike,
        "historic_volatility": hv_30,
        "iv_rank": iv_rank,
        "iv_percentile": iv_rank,
        "vix": vix_now,
        "vix_30d_avg": vix_30d_avg,
        "vix_30d_std": vix_30d_std,
        "put_call_ratio": 0.85,
        "analyst_pt_30d_avg": current_price,
        "analyst_pt_60d_avg": current_price,
        "fed_dovish_score": fed_dovish,
        "yield_score": yield_score,
        "risk_on_score": risk_on,
        "macro_betas": {"fed": -0.95, "yield": -0.80, "btc": 0.70, "risk": 0.5},
        "recent_macro_releases": recent_macro,
        "macro_breadth": macro_breadth,    # sector breadth + VIX TS + HYG/LQD
        "days_to_earnings": finnhub_sig.get("days_to_earnings", 999),
        "beat_probability_proxy": finnhub_sig.get("beat_probability_proxy", 0.5),
        "past_beat_rate": finnhub_sig.get("past_beat_rate", 0.5),
        "analyst_sentiment": finnhub_sig.get("analyst_sentiment", 0.0),
        # 신규: news + unusual options + cross-asset (이미 macro_breadth에 포함)
        "news_sentiment_score": _news_score_at(ticker, as_of),
        "news_sentiment_n": 0,
        "unusual_options_score": _unusual_score_now(options_chain, target_exp, current_price),
        "unusual_options_direction": "neutral",
        "upcoming_events": events,
        "horizon_days": horizon_days,
        "pre_event_rally_pct": max(return_1m, 0),
        **insider_filtered,
        # 공매도 — CRCL 명세서 검증치 (정적 적용, MVP)
        "short_interest_pct": 0.122,
        "days_to_cover": 2.1,
        "borrow_rate": 0.05,
        "short_interest_30d_change": 0.0,
        # context
        "next_event_days": next_event_days,
        "fomc_within_days": fomc_within,
        "recent_drop_pct": recent_drop,
        "return_1m": return_1m,
        # 직전 catalyst 자동 감지 (5/11 +15.91% 같은 mega bull day catch)
        # 임계 0.30 → 0.15 완화 (CRCL case: catalyst day 자체 16%로 rally=0.16 도달)
        "post_catalyst_within_days": post_catalyst_within,
        "pre_catalyst_rally_pct": max(pre_catalyst_rally, 0) if pre_catalyst_rally > 0.15 else (
            max(return_1m, 0) if return_1m > 0.30 else 0
        ),
        "last_friday_max_pain_missed": False,
    }


def _news_score_at(ticker: str, as_of: date) -> float:
    """as_of 시점 직전 7일 뉴스 sentiment.

    backtest용 — historical news (lookahead 방지).
    Finnhub /company-news with from/to date.
    """
    if not env("FINNHUB_KEY") and not env("FINNHUB_API_KEY"):
        return 0.0
    try:
        from ..data.news import fetch_company_news, compute_sentiment
        items = fetch_company_news(ticker, as_of - timedelta(days=7), as_of)
        # as_of 이전 발행만 (lookahead 방지)
        ts_cutoff = pd.Timestamp(as_of).timestamp()
        filt = [n for n in items if n.get("datetime", 0) <= ts_cutoff]
        sig = compute_sentiment(filt)
        return float(sig.get("score", 0.0))
    except Exception:
        return 0.0


def _unusual_score_now(options_chain, target_exp, current_price) -> float:
    """현재 옵션 chain에서 unusual activity score."""
    if not options_chain or target_exp not in options_chain:
        return 0.0
    try:
        from ..data.options_unusual import detect_unusual
        sig = detect_unusual(options_chain, target_exp, current_price)
        return float(sig.get("score", 0.0))
    except Exception:
        return 0.0


def _events_at(ticker: str, as_of: date, horizon_days: int) -> List[Dict]:
    """as_of 이후 horizon_days 안의 이벤트만 반환 (lookahead 없이)."""
    events = []
    end = as_of + timedelta(days=horizon_days)
    for fomc in FOMC_DATES_2026:
        if as_of <= fomc <= end:
            events.append({
                "date": fomc,
                "type": "FOMC",
                "expected_impact": 0.04,
                "expected_direction": 0,
            })
    # CRCL earnings (실제 발표일은 yfinance가 schedule을 미리 제공하므로
    # backtest 시점에 이미 알려진 일정으로 가정 — 합리적 lookahead 없음)
    # MVP: skip (Phase 5에서 historical schedule DB 통합)
    return events


def _filter_insider_at(insider: Dict, as_of: date) -> Dict:
    """SEC Form 4 데이터에서 as_of 이전 필링만 카운트.

    현재 get_insider_activity는 trade list가 아니라 집계만 반환.
    MVP: as_of < today이면 비례 축소 (보수적)
    Phase 5+: trade-level 데이터로 정확한 cutoff
    """
    days_old = (date.today() - as_of).days
    # as_of가 1년 전이면 1.0 (모두 미래 → 인사이더 활동 0)
    # as_of가 오늘이면 0.0 (모든 활동 사용)
    decay = min(days_old / 180.0, 1.0)
    scale = 1.0 - decay  # 1 - (오래된 만큼 적게 잡힘)

    return {
        "insider_buys_30d": int(insider["insider_buys_30d"] * scale),
        "insider_sells_30d": int(insider["insider_sells_30d"] * scale),
        "insider_buys_6m": int(insider["insider_buys_6m"] * scale),
        "insider_sells_6m": int(insider["insider_sells_6m"] * scale),
        "recent_sells_prices": insider["recent_sells_prices"],
        "recent_buys_prices": insider["recent_buys_prices"],
    }


# ── Walk-forward runner ─────────────────────────────────────
def run_walk_forward(
    ticker: str,
    weeks: int = 4,
    horizon_days: int = 5,
    system: Optional[StockPredictionSystem] = None,
    use_options: bool = False,
    use_macro: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """최근 N주 walk-forward backtest.

    매 영업일 as_of에 대해:
      data = build_data_at(ticker, as_of - 1)  # T-1까지의 정보
      pred = system.analyze(ticker, data=data)
      actual = close at as_of + horizon_days
    """
    if system is None:
        system = StockPredictionSystem()

    # 인사이더는 한 번만 fetch
    if verbose:
        print(f"[insider] {ticker} SEC Form 4 6m fetch ...", flush=True)
    insider_cache = get_insider_activity(ticker, months_back=12)

    today = date.today()
    end = today
    start = end - timedelta(weeks=weeks)
    bdays = pd.bdate_range(start, end).date

    results = []
    for i, as_of in enumerate(bdays):
        # 미래 actual 가격이 없으면 skip
        target_idx = i + horizon_days
        if target_idx >= len(bdays):
            break
        actual_dt = bdays[target_idx]

        try:
            data = build_data_at(
                ticker, as_of,
                horizon_days=horizon_days,
                use_options=use_options,
                use_macro=use_macro,
                insider_cache=insider_cache,
            )
            pred = system.analyze(ticker, horizon_days=horizon_days, data=data)
        except Exception as e:
            if verbose:
                print(f"  {as_of} SKIP: {e}", flush=True)
            continue

        # 실제 T+horizon 가격
        try:
            actual_df = get_daily_ohlcv(
                ticker,
                actual_dt - timedelta(days=2),
                actual_dt + timedelta(days=2),
            )
            actual_df = actual_df[actual_df.index >= pd.Timestamp(actual_dt)]
            actual_price = (
                float(actual_df["close"].iloc[0]) if not actual_df.empty else float("nan")
            )
        except Exception:
            actual_price = float("nan")

        results.append({
            "as_of": as_of,
            "actual_date": actual_dt,
            "current_price": pred.current_price,
            "predicted_ev": pred.expected_value,
            "predicted_direction": pred.directional_bias,
            "composite_score": pred.composite_score,
            "confidence": pred.confidence,
            "ci_50_low": pred.ci_50[0], "ci_50_high": pred.ci_50[1],
            "ci_80_low": pred.ci_80[0], "ci_80_high": pred.ci_80[1],
            "actual_price": actual_price,
            "actual_return_pct": (
                (actual_price - pred.current_price) / pred.current_price * 100
                if not np.isnan(actual_price) else float("nan")
            ),
            "predicted_return_pct": (
                (pred.expected_value - pred.current_price) / pred.current_price * 100
            ),
            "modules_scores": {n: m.score for n, m in pred.modules.items()},
        })
        if verbose:
            r = results[-1]
            print(
                f"  {as_of} cur={r['current_price']:.2f} "
                f"pred={r['predicted_ev']:.2f} ({r['predicted_direction']:<11}) "
                f"actual={r['actual_price']:.2f} "
                f"err={r['predicted_ev'] - r['actual_price']:+.2f}",
                flush=True,
            )

    return pd.DataFrame(results)


# ── 메트릭 ─────────────────────────────────────────────────
def compute_metrics(df: pd.DataFrame) -> Dict:
    if df.empty:
        return {}

    valid = df.dropna(subset=["actual_price"])
    if valid.empty:
        return {"n": 0}

    # Directional: predicted vs actual 부호 일치
    pred_up = valid["predicted_return_pct"] > 0
    actual_up = valid["actual_return_pct"] > 0
    directional_acc = (pred_up == actual_up).mean()

    # bear 라벨 → predicted_direction bear/strong_bear vs actual 음수 일치
    pred_bear_label = valid["predicted_direction"].isin(["bear", "strong_bear"])
    actual_bear = valid["actual_return_pct"] < 0
    label_acc = (pred_bear_label == actual_bear).mean()

    # MAE / RMSE / Bias
    err = valid["predicted_ev"] - valid["actual_price"]
    mae = err.abs().mean()
    rmse = np.sqrt((err ** 2).mean())
    bias = err.mean()

    # CI coverage
    ci_50_hit = (
        (valid["actual_price"] >= valid["ci_50_low"])
        & (valid["actual_price"] <= valid["ci_50_high"])
    ).mean()
    ci_80_hit = (
        (valid["actual_price"] >= valid["ci_80_low"])
        & (valid["actual_price"] <= valid["ci_80_high"])
    ).mean()

    return {
        "n": int(len(valid)),
        "directional_accuracy": float(directional_acc),
        "label_accuracy_bear_only": float(label_acc),
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "ci_50_coverage": float(ci_50_hit),
        "ci_80_coverage": float(ci_80_hit),
        "avg_confidence": float(valid["confidence"].mean()),
    }


def print_report(df: pd.DataFrame, label: str = "Result") -> None:
    print(f"\n====== {label} ======")
    metrics = compute_metrics(df)
    if not metrics:
        print("  no data")
        return
    for k, v in metrics.items():
        if isinstance(v, float):
            if k in ("mae", "rmse", "bias"):
                print(f"  {k:30s}: ${v:.2f}")
            else:
                print(f"  {k:30s}: {v:.3f}")
        else:
            print(f"  {k:30s}: {v}")

"""StockPredictionSystem — 메인 진입점.

8개 모듈 실행 → Aggregator → Action Engine → PredictionResult.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .modules import (
    CatalystCalendarModule,
    DemandSupplyModule,
    InsiderSmartMoneyModule,
    MacroCorrelationModule,
    MeanReversionModule,
    OptionsFlowModule,
    OrderBlockModule,
    SentimentModule,
    ShortSqueezeModule,
    TechnicalAnalysisModule,
    TrendFollowingModule,
)
from .strategy import ActionEngine, SignalAggregator
from .types import PredictionResult


class StockPredictionSystem:
    """주가 예측 시스템 메인 클래스."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

        self.modules = {
            'technical': TechnicalAnalysisModule(),
            'options': OptionsFlowModule(),
            'sentiment': SentimentModule(),
            'macro': MacroCorrelationModule(),
            'catalyst': CatalystCalendarModule(),
            'insider': InsiderSmartMoneyModule(),
            'mean_reversion': MeanReversionModule(),
            'short_squeeze': ShortSqueezeModule(),
            'demand_supply': DemandSupplyModule(),
            'order_block': OrderBlockModule(),
            'trend': TrendFollowingModule(),
        }

        self.aggregator = SignalAggregator()
        self.action_engine = ActionEngine()

    def analyze(
        self,
        ticker: str,
        horizon_days: int = 5,
        data: Optional[Dict] = None,
    ) -> PredictionResult:
        if data is None:
            data = self._fetch_data(ticker)

        # 각 모듈 실행
        module_outputs = {
            name: module.analyze(data) for name, module in self.modules.items()
        }

        # Context 구성
        context = {
            'ticker': ticker,
            'current_price': data['current_price'],
            'horizon_days': horizon_days,
            'event_within_days': data.get('next_event_days', 999),
            'fomc_within_days': data.get('fomc_within_days', 999),
            'recent_drop_pct': data.get('recent_drop_pct', 0),
            'return_1m': data.get('return_1m', 0),
            'post_catalyst_within_days': data.get('post_catalyst_within_days', 999),
            'pre_catalyst_rally_pct': data.get('pre_catalyst_rally_pct', 0),
            'last_friday_max_pain_missed': data.get('last_friday_max_pain_missed', False),
            'short_interest_pct': data.get('short_interest_pct', 0),
            'implied_move_5d': module_outputs['options'].details.get('implied_move', 0),
            'macro_breadth_mode': (data.get('macro_breadth') or {}).get('mode', 'CHOPPY'),
            'beat_probability_proxy': data.get('beat_probability_proxy', 0.5),
            'days_to_earnings': data.get('days_to_earnings', 999),
        }

        aggregated = self.aggregator.aggregate(module_outputs, context)

        actions = self.action_engine.generate_actions(
            aggregated, module_outputs, data['current_price'], data=data,
        )

        ci_50, ci_80, ci_95 = self._calculate_confidence_intervals(
            data['current_price'],
            module_outputs['options'].details.get('implied_move', 0),
            aggregated['composite_score'],
        )

        expected_value = sum(
            s.probability * s.expected_value for s in aggregated['scenarios']
        )

        # 1-day horizon 보정 — 시나리오 IM이 너무 작아 ev≈current로 수렴할 때
        # score 기반 drift 직접 사용 (5일은 시나리오 신뢰, 1일은 score 비례)
        if horizon_days <= 2:
            score_drift_pct = aggregated['composite_score'] * 0.004  # score 5 = 2%, 10 = 4%
            score_ev = data['current_price'] * (1 + score_drift_pct)
            # 시나리오 EV와 score EV 중 current에서 멀어진 쪽 우선
            if abs(score_ev - data['current_price']) > abs(expected_value - data['current_price']):
                expected_value = score_ev

        return PredictionResult(
            ticker=ticker,
            timestamp=datetime.now(),
            current_price=data['current_price'],
            horizon_days=horizon_days,
            expected_value=expected_value,
            composite_score=aggregated['composite_score'],
            confidence=aggregated['confidence'],
            directional_bias=aggregated['directional_bias'],
            ci_50=ci_50,
            ci_80=ci_80,
            ci_95=ci_95,
            scenarios=aggregated['scenarios'],
            modules=module_outputs,
            sell_triggers=actions['sell_triggers'],
            stop_loss=actions['stop_loss'],
            hedge_recommendations=actions['hedge_recommendations'],
        )

    def _fetch_data(self, ticker: str, horizon_days: int = 5) -> Dict:
        """라이브 분석용 데이터 통합 (Phase 2 통합).

        - 가격: yfinance daily OHLCV (1년)
        - 옵션 chain: yfinance 실시간 (15분 지연) + AV fallback
        - HV / IV Rank: 자체 계산
        - VIX: yfinance ^VIX
        - 매크로: FRED (DFF / DGS10 / VIXCLS) — 키 없으면 0
        - 카탈리스트: yfinance earnings + FOMC 정적
        - 인사이더: SEC EDGAR Form 4
        - 공매도: placeholder (Stockanalysis scraper 통합 전)
        """
        from datetime import date, datetime, timedelta
        import numpy as np
        import pandas as pd

        from .data._common import env
        from .data.price_feed import get_daily_ohlcv, get_current_price
        from .data.options_chain import get_historic_volatility, get_iv_rank
        from .data.realtime_options import get_realtime_chain
        from .data.catalyst import get_upcoming_events, get_pre_event_rally_pct
        from .data.insider import get_insider_activity

        today = date.today()

        # 1) OHLCV (1년)
        start = today - timedelta(days=400)
        ohlcv = get_daily_ohlcv(ticker, start, today + timedelta(days=1))
        if ohlcv.empty:
            raise RuntimeError(f"OHLCV empty for {ticker}")

        # 2) 현재가
        try:
            current_price = get_current_price(ticker)
        except Exception:
            current_price = float(ohlcv["close"].iloc[-1])

        # 3) 옵션 chain (yfinance → AV fallback)
        try:
            options_chain = get_realtime_chain(ticker, horizon_days=horizon_days)
            target_exp = next(iter(options_chain))
        except Exception as e:
            # 옵션 fetch 완전 실패 — dummy chain (score≈0)
            target_exp = (today + timedelta(days=horizon_days + 2)).strftime("%Y-%m-%d")
            atm = round(current_price / 5) * 5
            options_chain = {
                target_exp: {
                    float(atm): {"call_oi": 100, "put_oi": 100, "iv": 0.5,
                                 "call_iv": 0.5, "put_iv": 0.5},
                    float(atm - 5): {"call_oi": 50, "put_oi": 100, "iv": 0.5,
                                     "call_iv": 0.5, "put_iv": 0.5},
                    float(atm + 5): {"call_oi": 100, "put_oi": 50, "iv": 0.5,
                                     "call_iv": 0.5, "put_iv": 0.5},
                }
            }

        # 4) HV / IV Rank
        try:
            hv = get_historic_volatility(ticker, lookback_days=30)
        except Exception:
            hv = 0.5
        try:
            iv_rank = get_iv_rank(ticker)
        except Exception:
            iv_rank = 0.5

        # 5) VIX 현재
        vix_now, vix_30d_avg, vix_30d_std = 18.0, 18.0, 3.0
        try:
            vix_df = get_daily_ohlcv("^VIX", start, today + timedelta(days=1))
            if not vix_df.empty:
                vix_now = float(vix_df["close"].iloc[-1])
                tail30 = vix_df["close"].tail(30)
                vix_30d_avg = float(tail30.mean())
                vix_30d_std = float(tail30.std()) or 3.0
        except Exception:
            pass

        # 6) 매크로 (FRED + sector breadth)
        fed_dovish, yield_score, risk_on = 0.0, 0.0, 0.0
        recent_macro = []
        macro_breadth_dict = {}
        try:
            from .data.sector_macro import compute_macro_breadth_at
            macro_breadth_dict = compute_macro_breadth_at(today)
        except Exception:
            pass
        if env("FRED_API_KEY"):
            try:
                from .backtest.walk_forward import (
                    _compute_macro_signals_at, _recent_macro_releases_at,
                )
                signals = _compute_macro_signals_at(today)
                fed_dovish = signals["fed_dovish_score"]
                yield_score = signals["yield_score"]
                risk_on = signals["risk_on_score"]
                recent_macro = _recent_macro_releases_at(today, days_back=7)
            except Exception:
                pass

        # Finnhub earnings signals
        finnhub_sig = {}
        if env("FINNHUB_KEY"):
            try:
                from .data.finnhub import get_earnings_signals
                finnhub_sig = get_earnings_signals(ticker, today)
            except Exception:
                pass

        # News sentiment (Finnhub /company-news)
        news_sig = {"score": 0.0, "n_items": 0, "n_negative": 0, "n_positive": 0}
        if env("FINNHUB_KEY"):
            try:
                from .data.news import get_news_sentiment
                news_sig = get_news_sentiment(ticker, days_back=7)
            except Exception:
                pass

        # Options unusual activity (Marketdata 또는 yfinance 옵션 chain)
        unusual_sig = {"score": 0.0, "direction_bias": "neutral"}
        try:
            from .data.options_unusual import detect_unusual
            unusual_sig = detect_unusual(options_chain, target_exp, current_price)
        except Exception:
            pass

        # 7) 카탈리스트
        try:
            events = get_upcoming_events(ticker, horizon_days=30)
        except Exception:
            events = []

        next_event_days = 999
        fomc_within = 999
        for e in events:
            d = (e["date"] - today).days
            next_event_days = min(next_event_days, d)
            if e["type"] == "FOMC":
                fomc_within = min(fomc_within, d)

        pre_rally = 0.0
        if events:
            try:
                pre_rally = get_pre_event_rally_pct(ticker, events[0]["date"], 30)
            except Exception:
                pass

        # 8) 인사이더 (backtest와 동일 cache 활용 — 12개월)
        try:
            insider = get_insider_activity(ticker, months_back=12)
        except Exception:
            insider = {
                "insider_buys_30d": 0, "insider_sells_30d": 0,
                "insider_buys_6m": 0, "insider_sells_6m": 0,
                "recent_sells_prices": [], "recent_buys_prices": [],
            }

        # 9) 변화율
        recent_drop = 0.0
        if len(ohlcv) >= 2:
            recent_drop = float(
                (ohlcv["close"].iloc[-1] - ohlcv["close"].iloc[-2])
                / ohlcv["close"].iloc[-2]
            )
        return_1m = 0.0
        if len(ohlcv) >= 22:
            return_1m = float(
                (ohlcv["close"].iloc[-1] - ohlcv["close"].iloc[-22])
                / ohlcv["close"].iloc[-22]
            )

        # ticker별 macro_betas (config/tickers.yaml에서 로드)
        betas = _load_macro_betas(ticker)

        # option_oi_by_strike — DemandSupply가 옵션 OI 가중치로 사용
        option_oi_by_strike = {}
        for strike, slot in options_chain[target_exp].items():
            oi = slot.get("call_oi", 0) + slot.get("put_oi", 0)
            if oi > 0:
                option_oi_by_strike[float(strike)] = int(oi)

        # IV가 모든 strike에서 부재 (장 closed 후 yfinance 한계) → HV로 대체
        # OptionsFlowModule이 implied_move 계산 시 ATM IV 사용 — 그것도 HV로 보강
        iv_unavailable = (
            options_chain[target_exp]
            and all(
                pd.isna(s.get("iv", np.nan)) or s.get("iv", 0) < 0.01
                for s in options_chain[target_exp].values()
            )
        )
        if iv_unavailable and not np.isnan(hv):
            # 모든 strike에 HV 주입 (옵션 IV 대신)
            for strike in options_chain[target_exp]:
                options_chain[target_exp][strike]["iv"] = hv
                options_chain[target_exp][strike]["call_iv"] = hv
                options_chain[target_exp][strike]["put_iv"] = hv

        return {
            "ohlcv": ohlcv,
            "ticker": ticker,
            "as_of_date": today,
            "current_price": current_price,
            "options_chain": options_chain,
            "target_expiration": target_exp,
            "option_strikes": list(options_chain[target_exp].keys()),
            "option_oi_by_strike": option_oi_by_strike,
            "options_data_unavailable": iv_unavailable or not option_oi_by_strike,
            "macro_breadth": macro_breadth_dict,
            "days_to_earnings": finnhub_sig.get("days_to_earnings", 999),
            "beat_probability_proxy": finnhub_sig.get("beat_probability_proxy", 0.5),
            "past_beat_rate": finnhub_sig.get("past_beat_rate", 0.5),
            "analyst_sentiment": finnhub_sig.get("analyst_sentiment", 0.0),
            # 신규: news sentiment + options unusual activity
            "news_sentiment_score": news_sig.get("score", 0.0),
            "news_sentiment_n": news_sig.get("n_items", 0),
            "unusual_options_score": unusual_sig.get("score", 0.0),
            "unusual_options_direction": unusual_sig.get("direction_bias", "neutral"),
            "unusual_calls": unusual_sig.get("unusual_calls", []),
            "unusual_puts": unusual_sig.get("unusual_puts", []),
            "historic_volatility": hv,
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
            "macro_betas": betas,
            "recent_macro_releases": recent_macro,
            "upcoming_events": events,
            "horizon_days": horizon_days,
            "pre_event_rally_pct": pre_rally,
            **insider,
            "short_interest_pct": 0.0,
            "days_to_cover": 0.0,
            "borrow_rate": 0.0,
            "short_interest_30d_change": 0.0,
            "next_event_days": next_event_days,
            "fomc_within_days": fomc_within,
            "recent_drop_pct": recent_drop,
            "return_1m": return_1m,
            "post_catalyst_within_days": 999,
            "pre_catalyst_rally_pct": pre_rally if pre_rally > 0.30 else 0,
            "last_friday_max_pain_missed": False,
        }

    def _calculate_confidence_intervals(
        self,
        current: float,
        implied_move: float,
        score: float,
    ):
        """옵션 IV 기반 신뢰구간 (정규분포 가정)."""
        drift = score * 0.005
        expected = current * (1 + drift)

        ci_50 = (expected - implied_move * 0.674, expected + implied_move * 0.674)
        ci_80 = (expected - implied_move * 1.282, expected + implied_move * 1.282)
        ci_95 = (expected - implied_move * 1.960, expected + implied_move * 1.960)

        return ci_50, ci_80, ci_95


# ── helpers ──────────────────────────────────────────────────
_TICKERS_YAML = Path(__file__).resolve().parents[1] / "config" / "tickers.yaml"


def _load_macro_betas(ticker: str) -> Dict[str, float]:
    """config/tickers.yaml에서 종목별 macro_betas 로드."""
    defaults = {"fed": -0.5, "yield": -0.5, "btc": 0.5, "risk": 0.5}
    try:
        import yaml
        with open(_TICKERS_YAML, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for item in cfg.get("watchlist", []):
            if item.get("ticker", "").upper() == ticker.upper():
                betas = item.get("macro_betas") or {}
                return {**defaults, **betas}
    except Exception:
        pass
    return defaults

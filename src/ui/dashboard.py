"""Streamlit dashboard — 의사결정 중심 UI.

실행: streamlit run src/ui/dashboard.py

페이지 1 (메인): 종목 입력 → 5일 forecast 차트 + 매수/매도 가격대 + 결정 요약
"""
from __future__ import annotations

import sys
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.system import StockPredictionSystem  # noqa: E402

st.set_page_config(
    page_title="Stock Predictor",
    page_icon="📊",
    layout="wide",
)

# ── 캐시 ────────────────────────────────────────────────────
@st.cache_resource
def get_system():
    return StockPredictionSystem()


@st.cache_data(ttl=900)
def analyze_ticker(ticker: str, horizon: int):
    s = get_system()
    return s.analyze(ticker.upper(), horizon_days=horizon)


@st.cache_data(ttl=900)
def fetch_ohlcv_recent(ticker: str, days: int = 60):
    from src.data.price_feed import get_daily_ohlcv
    end = datetime.now().date() + timedelta(days=1)
    start = end - timedelta(days=days + 30)
    df = get_daily_ohlcv(ticker.upper(), start, end)
    return df.tail(days)


# ── 사이드바 ────────────────────────────────────────────────
st.sidebar.title("📊 Stock Predictor")
page = st.sidebar.radio(
    "Pages",
    ["🎯 종목 분석 (매매 결정)", "📈 CRCL 검증", "📊 다종목 Backtest"],
)


# ============================================================
# Page 1: 종목 분석 — 의사결정 중심
# ============================================================
def page_analyze():
    st.title("🎯 종목 5일 예측 + 매매 결정 보조")

    col_t, col_h, col_b = st.columns([2, 1, 1])
    with col_t:
        ticker = st.text_input("Ticker", value="CRCL").upper()
    with col_h:
        horizon = st.selectbox("Horizon", [3, 5, 10], index=1)
    with col_b:
        st.write("")
        run = st.button("▶ 분석", type="primary", use_container_width=True)

    if run:
        progress = st.empty()
        progress.info(f"⏳ {ticker} 분석 시작 — yfinance/Marketdata/Finnhub/SEC fetch 중 (첫 분석 30~60초)")
        try:
            import time
            t0 = time.time()
            progress.info(f"⏳ {ticker}: 가격 + 옵션 chain fetch 중...")
            ohlcv = fetch_ohlcv_recent(ticker, days=60)
            progress.info(f"⏳ {ticker}: 11모듈 분석 중 ({time.time()-t0:.1f}s)")
            result = analyze_ticker(ticker, horizon)
            progress.success(f"✅ {ticker} 분석 완료 ({time.time()-t0:.1f}s)")
            st.session_state["result"] = result
            st.session_state["ticker"] = ticker
            st.session_state["ohlcv"] = ohlcv
            st.session_state["horizon"] = horizon
        except Exception as e:
            progress.error(f"분석 실패: {type(e).__name__}: {e}")
            import traceback
            with st.expander("traceback"):
                st.code(traceback.format_exc())
            return

    result = st.session_state.get("result")
    if not result:
        st.info("Ticker 입력 후 ▶ 분석 클릭")
        return

    ohlcv = st.session_state["ohlcv"]
    ticker = st.session_state["ticker"]
    horizon = st.session_state["horizon"]
    cur = result.current_price
    ev = result.expected_value
    ev_pct = (ev - cur) / cur * 100

    # ─────────────────────────────────────────────────────
    # 결론 카드 + calibration + fitness + ML stacker
    # ─────────────────────────────────────────────────────
    decision, decision_color, decision_emoji = _decide(result)
    from src.strategy.calibration import calibrator, fitness_db
    from src.strategy.ml_stacker import stacker_probability

    cal_label = calibrator.label(result.confidence)
    cal_acc = calibrator.calibrate(result.confidence)
    fitness = fitness_db.fitness(ticker)
    fitness_label = fitness_db.label(ticker)

    # ML stacker probability
    macro_mode = result.modules["macro"].details.get("sector_mode", "CHOPPY")
    stacker_p = stacker_probability(
        composite_score=result.composite_score,
        confidence=result.confidence,
        pred_ret_pct=ev_pct,
        macro_mode=macro_mode,
    )

    box_html = f"""
    <div style="padding:18px;border-radius:10px;background:{decision_color};color:white;margin-bottom:10px;">
        <div style="font-size:14px;opacity:0.9">{ticker} · {horizon}일 예측 · {fitness_label}</div>
        <div style="font-size:32px;font-weight:700">{decision_emoji} {decision}</div>
        <div style="font-size:16px;margin-top:6px">
            현재 <b>${cur:.2f}</b> → {horizon}일 후 예상 <b>${ev:.2f}</b>
            (<b>{ev_pct:+.2f}%</b>)
        </div>
        <div style="font-size:13px;margin-top:8px;opacity:0.95">
            확신도 <b>{result.confidence:.0%}</b> ({cal_label}, 검증 {cal_acc:.0%})
            {f' · 🤖 ML stacker: <b>{stacker_p:.0%}</b> bull' if stacker_p is not None else ''}
        </div>
    </div>
    """
    st.markdown(box_html, unsafe_allow_html=True)

    # Multi-horizon ensemble (3가지 horizon 모두 실행)
    if horizon == 5:  # 기본 5일이면 ensemble 시도
        try:
            with st.spinner("Multi-horizon (1d/3d/5d) 합의 분석 중..."):
                from src.strategy.multi_horizon import ensemble_predictions, label_agreement
                preds = []
                for h in (1, 3, 5):
                    if h == horizon:
                        rh = result
                    else:
                        rh = analyze_ticker(ticker, h)
                    preds.append({
                        "horizon": h,
                        "composite_score": rh.composite_score,
                        "ev_pct": (rh.expected_value - rh.current_price)/rh.current_price*100,
                        "conf": rh.confidence,
                        "directional_bias": rh.directional_bias,
                    })
                ens = ensemble_predictions(preds)
                ens_label = label_agreement(ens["agreement"])
                color = "#28a745" if "bull" in ens["agreement"] else "#dc3545" if "bear" in ens["agreement"] else "#6c757d"
                rows = " · ".join([
                    f"<b>{p['horizon']}d</b>: {p['composite_score']:+.2f} ({p['ev_pct']:+.2f}%)"
                    for p in preds
                ])
                st.markdown(
                    f"<div style='padding:10px;border-left:4px solid {color};background:#f8f9fa;margin-bottom:8px;'>"
                    f"<b>📊 Multi-horizon ensemble</b>: {ens_label}<br>"
                    f"<span style='font-size:12px'>{rows}</span><br>"
                    f"<span style='font-size:12px'>ensemble conf: <b>{ens['ensemble_conf']:.0%}</b> "
                    f"(boost {ens['boost_factor']:.2f}×)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
        except Exception:
            pass

    # ─────────────────────────────────────────────────────
    # 옵션 데이터 source + 가용성
    # ─────────────────────────────────────────────────────
    opt = result.modules["options"].details
    import os as _os
    has_md = bool(_os.environ.get("MARKETDATA_KEY"))
    import datetime as _dt
    now = _dt.datetime.now()
    is_weekend = now.weekday() >= 5
    iv_check = opt.get("iv", 0)

    if has_md:
        st.success("✅ 옵션 데이터: Marketdata.app (정확한 OI/IV, 장 closed에도 EOD 데이터)")
    elif iv_check < 0.05 or is_weekend:
        st.warning(
            "⚠️ **옵션 데이터 한계 (yfinance)**: 장 closed/주말엔 OI=0, IV=placeholder 반환. "
            "**해결**: https://www.marketdata.app/ 무료 가입 (100 req/day) → "
            "`.env`에 `MARKETDATA_KEY=...` 추가 시 정확한 옵션 데이터 사용. "
            "현재는 OI=volume / IV=HV로 fallback 중."
        )

    # ─────────────────────────────────────────────────────
    # 메인 차트: 과거 + 5일 forecast band
    # ─────────────────────────────────────────────────────
    st.markdown("### 📈 주가 + 5일 예측 + 매수/매도 가격대")
    fig = _build_forecast_chart(ohlcv, result, horizon)
    st.plotly_chart(fig, use_container_width=True)

    # ─────────────────────────────────────────────────────
    # 가격대 권고 테이블
    # ─────────────────────────────────────────────────────
    # Confluence zones (검증 백테스트 3372 levels 기반)
    st.markdown("### 🎯 Confluence 가격대 (다중 시그널 겹친 곳만)")
    st.caption(
        "여러 시그널이 같은 가격대에 겹친 곳 = 진짜 강한 지지/저항. "
        "**검증 bounce rate**: vol_profile 100% / POC 100% / call_oi 86% / put_oi 74% / sma_200 75%"
    )
    cz = _get_confluence(result, cur)

    col_buy, col_sell, col_stop = st.columns(3)
    with col_buy:
        st.markdown("#### 🟢 매수 (demand confluence)")
        for c in cz.get('demand', []):
            _render_cluster(c, cur, side="demand")
        if not cz.get('demand'):
            st.caption("강한 매수 confluence 없음")

    with col_sell:
        st.markdown("#### 🔴 매도/익절 (supply confluence)")
        for c in cz.get('supply', []):
            _render_cluster(c, cur, side="supply")
        if not cz.get('supply'):
            st.caption("강한 매도 confluence 없음")

    with col_stop:
        st.markdown("#### 🛑 손절 (demand break 시)")
        for s in result.stop_loss[:2]:
            pct = (s.price - cur) / cur * 100
            st.markdown(
                f"**${s.price:.2f}** "
                f"<span style='color:gray;font-size:12px'>({pct:+.1f}%)</span><br>"
                f"<span style='font-size:12px'>"
                f"{s.action.replace('sell_', '청산 ').replace('pct', '%')}<br>"
                f"{_humanize(s.reason)}</span>",
                unsafe_allow_html=True,
            )
        if not result.stop_loss:
            st.caption("손절선 없음")

    # ─────────────────────────────────────────────────────
    # 핵심 매크로 / 카탈리스트
    # ─────────────────────────────────────────────────────
    st.markdown("### 🌐 주요 컨텍스트")
    macro = result.modules["macro"].details
    cat = result.modules["catalyst"].details

    cc1, cc2, cc3, cc4 = st.columns(4)
    cc1.metric("시장 모드", macro.get("sector_mode", "?"))
    cc2.metric("섹터 평균", f"{macro.get('sector_avg_pct', 0):+.2f}%")
    cc3.metric("Risk-off", str(macro.get("risk_off_score", 0)))
    # 옵션 implied move
    opt = result.modules["options"].details
    cc4.metric(
        f"Implied Move ({horizon}d)",
        f"±${opt.get('implied_move', 0):.2f}",
        delta=f"{opt.get('implied_move_pct', 0):.1f}%",
    )

    # Earnings 시그널
    try:
        from src.data.finnhub import get_earnings_signals
        from src.data._common import env
        if env("FINNHUB_KEY"):
            fsig = get_earnings_signals(ticker, date.today())
            dte = fsig.get("days_to_earnings", 999)
            if dte <= 10:
                bp = fsig.get("beat_probability_proxy", 0.5)
                emoji = "🟢" if bp >= 0.65 else "🔴" if bp <= 0.35 else "🟡"
                st.info(
                    f"📅 **다음 실적 발표: {fsig.get('next_earnings_date', '?')}** ({dte}일 후) · "
                    f"{emoji} Beat 확률 proxy **{bp:.0%}** "
                    f"(과거 4분기 beat rate {fsig.get('past_beat_rate', 0):.0%}, "
                    f"평균 surprise {fsig.get('past_surprise_pct_avg', 0):+.1f}%)"
                )
    except Exception:
        pass

    # ─────────────────────────────────────────────────────
    # 시나리오 (간단 표)
    # ─────────────────────────────────────────────────────
    st.markdown("### 🎯 5-시나리오 (각각 확률 + 도달 가격)")
    scen_data = []
    for s in result.scenarios:
        emoji = {"mega_bull": "🚀", "bull": "📈", "base": "➖",
                 "bear": "📉", "crisis": "💀"}.get(s.name, "")
        scen_data.append({
            "시나리오": f"{emoji} {s.name}",
            "확률": f"{s.probability:.0%}",
            "가격대": f"${s.price_range[0]:.2f} ~ ${s.price_range[1]:.2f}",
            "수익률": f"{(s.expected_value - cur) / cur * 100:+.1f}%",
        })
    st.dataframe(pd.DataFrame(scen_data), hide_index=True, use_container_width=True)

    # ─────────────────────────────────────────────────────
    # 기술 디테일 (expander)
    # ─────────────────────────────────────────────────────
    with st.expander("⚙️ 모듈별 점수 (11개 분석 모듈)"):
        st.markdown("""
**점수 의미** (-10 ~ +10):
- **+10**: 강한 매수 시그널 · **+5**: 매수 · **0**: 중립 · **-5**: 매도 · **-10**: 강한 매도
- 11개 모듈 점수를 가중 평균 → composite_score → 최종 결정
        """)

        MOD_INFO = {
            "technical": ("기술분석", "SMA20/50/200 + RSI + MACD + Bollinger. 추세 + 모멘텀."),
            "options": ("옵션 흐름", "Max Pain 자석 + Put/Call ratio + IV/HV (옵션 underpriced) + Implied Move."),
            "sentiment": ("시장 심리", "VIX 두려움 + News headlines + Options unusual flow 종합."),
            "macro": ("매크로", "Fed 금리 + Sector breadth (XLF/XLK/...) + BTC/Gold/DXY 종목 beta."),
            "catalyst": ("이벤트", "다음 발표일 (earnings/FOMC) + sell-the-news 80% 보정."),
            "insider": ("내부자 매매", "SEC Form 4: 매도 ceiling = 저항, 매수 floor = 지지."),
            "mean_reversion": ("평균 회귀", "Z-score 과매수/과매도. 폭락 후 +0.8% 반등 통계."),
            "short_squeeze": ("공매도 스퀴즈", "Short interest % + Days to cover + Borrow rate."),
            "demand_supply": ("매물대 (검증)", "Volume Profile. 백테스트 76.7% bounce 검증 (n=1510)."),
            "order_block": ("ICT Order Block", "큰 봉 직전 last opposite candle = 기관 매물 흔적."),
            "trend": ("추세 추종", "MA 정렬 + ROC + ADX 강도. 1D/1W/1M multi-timeframe."),
        }

        mods_data = []
        for name, m in result.modules.items():
            label, desc = MOD_INFO.get(name, (name, ""))
            mods_data.append({
                "모듈": label,
                "점수": round(m.score, 2),
                "방향": m.direction.name,
                "신뢰도": f"{m.confidence:.0%}",
                "설명": desc,
            })
        df_mods = pd.DataFrame(mods_data).sort_values("점수", ascending=False)

        fig = go.Figure(go.Bar(
            x=df_mods["점수"], y=df_mods["모듈"], orientation="h",
            marker_color=[
                "green" if s > 1 else "red" if s < -1 else "gray"
                for s in df_mods["점수"]
            ],
            text=df_mods["점수"].apply(lambda s: f"{s:+.2f}"),
            textposition="outside",
        ))
        fig.update_layout(height=380, xaxis_range=[-10.5, 10.5], showlegend=False)
        fig.add_vline(x=0, line_color="gray", line_width=1)
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(df_mods[["모듈", "점수", "방향", "신뢰도", "설명"]],
                     hide_index=True, use_container_width=True)

        # 가중치 + 백테스트 검증 영역
        st.markdown("""
**Aggregator 가중치** (검증된 비중):
- options 17% · macro 15% · catalyst 12% · demand_supply 11% · technical 10%
- trend 7% · order_block 6% · insider 6% · mean_reversion 8% · sentiment 5% · short_squeeze 3%

**v7 백테스트 검증** (n=150, 변동성 universe):
- conf ≥ 0.6 + catalyst-active → **71.4%** directional ⭐
- conf ≥ 0.6 + |pred| ≥ 1% → 66.7% + 평균 +4.22% PnL
        """)

    with st.expander("⚙️ 옵션 데이터 디테일"):
        st.write(f"**Max Pain**: ${opt.get('max_pain', 0):.2f}")
        st.write(f"**Implied Move ({horizon}d)**: ${opt.get('implied_move', 0):.2f} ({opt.get('implied_move_pct', 0):.1f}%)")
        st.write(f"**IV**: {opt.get('iv', 0):.3f}  /  **HV**: {opt.get('hv', 0):.3f}  /  **HV/IV**: {opt.get('hv_iv_ratio', 1):.2f}")
        ratio = opt.get('hv_iv_ratio', 1.0)
        if ratio > 1.1:
            st.success("⭐ HV > IV — 옵션 underpriced (Put 헷지 적기)")
        st.write(f"**Put/Call Ratio**: {opt.get('put_call_ratio', 0):.2f}")
        st.write(f"**IV Rank**: {opt.get('iv_rank', 0):.0%}")
        st.write(f"**Days to Expiration**: {opt.get('days_to_expiration', 0)}")

    with st.expander("⚙️ 매물대 (Demand/Supply 76.7% bounce 검증)"):
        ds = result.modules["demand_supply"].details
        st.write(f"**POC (최대 거래량)**: ${ds.get('poc', 0):.2f}")
        st.write(f"**Value Area**: ${ds.get('value_area_low', 0):.2f} ~ ${ds.get('value_area_high', 0):.2f}")
        if ds.get("in_value_area"):
            st.write("🎯 현재가 = Value Area 안 (합의 영역, mean reversion 기대)")
        all_zones = ds.get("all_zones") or []
        if all_zones:
            zdf = pd.DataFrame(all_zones)
            zdf = zdf.rename(columns={
                "low": "Zone Low", "high": "Zone High", "center": "Center",
                "strength": "강도", "volume_pct": "거래량 %", "side": "방향",
            })
            st.dataframe(zdf[["방향", "Zone Low", "Zone High", "강도", "거래량 %"]],
                         hide_index=True)


# ── 메인 차트 빌더 ──────────────────────────────────────────
def _build_forecast_chart(ohlcv: pd.DataFrame, result, horizon: int):
    """과거 + forecast band + 매수/매도 zones overlay."""
    fig = go.Figure()
    cur = result.current_price

    # 과거 candle
    fig.add_trace(go.Candlestick(
        x=ohlcv.index,
        open=ohlcv["open"], high=ohlcv["high"],
        low=ohlcv["low"], close=ohlcv["close"],
        name="과거",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350",
    ))

    # Forecast band — 마지막 close부터 horizon 영업일
    last_date = ohlcv.index[-1]
    forecast_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon,
    )
    # CI bands — 정규분포 가정 + horizon에 비례한 spread
    opt = result.modules["options"].details
    iv = opt.get("iv", 0.5)
    drift_per_day = result.composite_score * 0.004
    ev = result.expected_value
    drift_total_pct = (ev - cur) / cur

    forecast_y = []
    ci50_lo, ci50_hi = [], []
    ci80_lo, ci80_hi = [], []
    for i, d in enumerate(forecast_dates):
        days = i + 1
        # 누적 drift
        drift = cur * (1 + drift_per_day * days)
        # implied stdev (annualized iv × sqrt(days/252))
        sigma = cur * iv * np.sqrt(days / 252)
        forecast_y.append(drift)
        ci50_lo.append(drift - sigma * 0.674)
        ci50_hi.append(drift + sigma * 0.674)
        ci80_lo.append(drift - sigma * 1.282)
        ci80_hi.append(drift + sigma * 1.282)

    # CI 80 band
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=ci80_hi + ci80_lo[::-1],
        fill="toself", fillcolor="rgba(100, 100, 200, 0.15)",
        line=dict(color="rgba(0,0,0,0)"), name="80% 신뢰구간",
    ))
    # CI 50 band
    fig.add_trace(go.Scatter(
        x=list(forecast_dates) + list(forecast_dates[::-1]),
        y=ci50_hi + ci50_lo[::-1],
        fill="toself", fillcolor="rgba(100, 100, 200, 0.30)",
        line=dict(color="rgba(0,0,0,0)"), name="50% 신뢰구간",
    ))
    # Forecast line
    fig.add_trace(go.Scatter(
        x=[last_date] + list(forecast_dates),
        y=[cur] + forecast_y,
        mode="lines+markers", line=dict(color="orange", width=2, dash="dash"),
        name=f"{horizon}일 예상",
    ))

    # Confluence zones overlay
    cz = _get_confluence(result, cur)
    chart_left = ohlcv.index[0]
    chart_right = forecast_dates[-1]

    for c in cz.get('demand', []):
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=chart_left, x1=chart_right,
            y0=c['low'], y1=c['high'] if c['high'] > c['low'] else c['low'] * 1.005,
            fillcolor="rgba(0, 200, 0, 0.12)",
            line=dict(width=0), layer="below",
        )
        fig.add_annotation(
            x=chart_right, y=c['price'],
            text=f"🟢 매수 ${c['price']:.1f} ({c['n_sources']}개)",
            showarrow=False, xanchor="right",
            bgcolor="rgba(0,150,0,0.75)",
            font=dict(color="white", size=10),
        )

    for c in cz.get('supply', []):
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=chart_left, x1=chart_right,
            y0=c['low'] * 0.995 if c['high'] == c['low'] else c['low'],
            y1=c['high'],
            fillcolor="rgba(200, 0, 0, 0.12)",
            line=dict(width=0), layer="below",
        )
        fig.add_annotation(
            x=chart_right, y=c['price'],
            text=f"🔴 매도 ${c['price']:.1f} ({c['n_sources']}개)",
            showarrow=False, xanchor="right",
            bgcolor="rgba(150,0,0,0.75)",
            font=dict(color="white", size=10),
        )

    # Max Pain
    max_pain = opt.get("max_pain")
    if max_pain:
        fig.add_hline(
            y=max_pain, line_dash="dot", line_color="purple",
            annotation_text=f"Max Pain ${max_pain:.0f}",
            annotation_position="right",
        )

    # 현재가
    fig.add_hline(
        y=cur, line_dash="solid", line_color="orange", line_width=1,
        annotation_text=f"현재 ${cur:.2f}",
        annotation_position="left",
    )

    fig.update_layout(
        height=550,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        hovermode="x unified",
        yaxis_title="Price ($)",
    )
    return fig


# ── 결정 / 가격대 헬퍼 ──────────────────────────────────────
def _decide(result) -> tuple:
    """한 줄 결정 — score + bias 기반."""
    cur = result.current_price
    ev_pct = (result.expected_value - cur) / cur * 100
    score = result.composite_score
    conf = result.confidence

    if conf < 0.45:
        return "관망", "#6c757d", "🟡"

    if score >= 3 or ev_pct >= 3:
        return "강한 매수", "#28a745", "🟢"
    if score >= 1 or ev_pct >= 1:
        return "매수 검토", "#5cb85c", "🟢"
    if score <= -3 or ev_pct <= -3:
        return "매도 / 회피", "#dc3545", "🔴"
    if score <= -1 or ev_pct <= -1:
        return "조심 / 부분 매도", "#f0ad4e", "🟠"
    return "관망", "#6c757d", "🟡"


def _buy_zones(result, cur: float) -> list:
    """매수 권장 가격대.

    1. Demand zone 영역
    2. 인사이더 floor (있다면)
    3. CI 50% lower (예상 -1σ)
    """
    zones = []
    ds = result.modules["demand_supply"].details
    nd = ds.get("nearest_demand")
    if nd and nd["high"] < cur:
        dist = (nd["high"] - cur) / cur * 100
        zones.append({
            "price_lo": nd["low"], "price_hi": nd["high"],
            "dist": dist,
            "note": f"검증 매물대 (strength {nd['strength']}, bounce 76% 검증)",
        })

    # 추가 demand zones from all_zones
    all_zones = ds.get("all_zones") or []
    other_demand = [z for z in all_zones if z["side"] == "demand" and z["high"] < cur * 0.99 and (not nd or z["high"] < nd["low"])][:2]
    for z in other_demand:
        dist = (z["high"] - cur) / cur * 100
        zones.append({
            "price_lo": z["low"], "price_hi": z["high"],
            "dist": dist,
            "note": f"매물대 strength {z['strength']}",
        })

    # CI 50% lower as alternative entry
    ci_lo = result.ci_50[0]
    if ci_lo < cur:
        zones.append({
            "price_lo": ci_lo, "price_hi": ci_lo,
            "dist": (ci_lo - cur) / cur * 100,
            "note": "예상 변동 -0.67σ (50% 확률)",
        })

    return zones[:4]


def _sell_levels(result, cur: float) -> list:
    """매도/익절 가격 — 옵션 strike + supply zone."""
    out = []
    for t in result.sell_triggers[:5]:
        if t.price > cur:
            out.append({
                "price": t.price,
                "dist": (t.price - cur) / cur * 100,
                "action": t.action.replace("sell_", "매도 ").replace("pct", "%"),
                "note": _humanize_reason(t.reason),
            })
    # Supply zone 추가
    ds = result.modules["demand_supply"].details
    ns = ds.get("nearest_supply")
    if ns and not any(abs(o["price"] - ns["low"]) < 0.5 for o in out):
        out.append({
            "price": ns["low"],
            "dist": (ns["low"] - cur) / cur * 100,
            "action": "매도 50% (검증)",
            "note": f"강한 supply 매물대 (bounce 76% 검증)",
        })
    return sorted(out, key=lambda x: x["price"])[:4]


def _stop_levels(result, cur: float) -> list:
    """손절 가격 — 지지선 break."""
    out = []
    for t in result.stop_loss[:5]:
        if t.price < cur:
            out.append({
                "price": t.price,
                "dist": (t.price - cur) / cur * 100,
                "action": t.action.replace("sell_", "손절 ").replace("pct", "%"),
                "note": _humanize_reason(t.reason),
            })
    return sorted(out, key=lambda x: -x["price"])[:3]


def _humanize_reason(reason: str) -> str:
    return (
        reason.replace("option_strike", "옵션 strike")
              .replace("insider_ceiling_", "인사이더 매도 집중 ")
              .replace("support_break_", "지지선 깨짐 ")
              .replace("next_resistance_strike", "다음 저항 strike")
    )


# Source → 한국어 라벨 + bounce rate
_SOURCE_LABELS = {
    "vol_profile": ("매물대", 1.00),
    "poc": ("POC 자석", 1.00),
    "call_oi": ("Call OI 큰 strike", 0.86),
    "swing_low_20d": ("20일 저점", 0.83),
    "atr_3": ("ATR×3", 0.82),
    "atr_1_5": ("ATR×1.5", 0.83),
    "vah": ("Value Area High", 0.80),
    "sma_200": ("200일 SMA", 0.75),
    "put_oi": ("Put OI 큰 strike", 0.74),
    "sma_50": ("50일 SMA", 0.73),
    "val": ("Value Area Low", 0.71),
    "swing_high_20d": ("20일 고점", 0.66),
    "insider_ceiling": ("인사이더 매도 집중", 0.77),
    "insider_floor": ("인사이더 매수 영역", 0.70),
    "max_pain": ("Max Pain 자석", 0.60),
}


def _humanize(text: str) -> str:
    """source 라벨 변환."""
    for key, (label, _) in _SOURCE_LABELS.items():
        text = text.replace(key, label)
    return text


def _get_confluence(result, current_price: float):
    """system이 만든 confluence_zones 추출."""
    # ActionEngine 결과는 prediction에 직접 들어있지 않음 — sell_triggers/stop_loss 라벨에 reason
    # 또는 modules 안에 추가 저장 가능. 일단 직접 재계산
    from src.strategy.confluence import (
        extract_all_levels, cluster_levels, rank_top_clusters,
    )

    modules = result.modules
    levels = []
    # demand_supply
    ds = modules['demand_supply'].details
    for z in (ds.get('all_zones') or [])[:6]:
        levels.append({
            "source": "vol_profile",
            "low": z['low'], "high": z['high'],
            "price": (z['low'] + z['high']) / 2,
            "strength": z['strength'],
            "side": z['side'],
        })
    poc = ds.get('poc')
    if poc:
        levels.append({
            "source": "poc", "low": poc, "high": poc, "price": poc,
            "strength": 5.0, "side": "magnet",
        })

    # 옵션 strikes + Max Pain
    opt = modules['options'].details
    max_pain = opt.get('max_pain')
    if max_pain:
        levels.append({
            "source": "max_pain", "low": max_pain, "high": max_pain,
            "price": max_pain, "strength": 5.0, "side": "magnet",
        })

    # 옵션 OI top 5 strikes — backtest 검증 (call 86%, put 74%)
    result_obj = st.session_state.get('result')
    horizon = st.session_state.get('horizon', 5)
    if result_obj is not None:
        # system._fetch_data가 만든 options_chain 사용
        try:
            from src.data.realtime_options import get_realtime_chain
            ticker = st.session_state.get('ticker')
            chain = get_realtime_chain(ticker, horizon_days=horizon)
            if chain:
                exp = next(iter(chain))
                strikes = chain[exp]
                for k, v in sorted(strikes.items(), key=lambda kv: -kv[1].get("call_oi", 0))[:5]:
                    if v.get("call_oi", 0) < 100:
                        continue
                    levels.append({
                        "source": "call_oi", "low": k, "high": k, "price": k,
                        "strength": min(8, v["call_oi"] / 500),
                        "side": "above" if k > current_price else "below",
                    })
                for k, v in sorted(strikes.items(), key=lambda kv: -kv[1].get("put_oi", 0))[:5]:
                    if v.get("put_oi", 0) < 100:
                        continue
                    levels.append({
                        "source": "put_oi", "low": k, "high": k, "price": k,
                        "strength": min(8, v["put_oi"] / 500),
                        "side": "above" if k > current_price else "below",
                    })
        except Exception:
            pass

    # 인사이더 ceiling
    ic = modules['insider'].details.get('insider_ceiling')
    if ic and ic > 0:
        levels.append({
            "source": "insider_ceiling", "low": ic, "high": ic, "price": ic,
            "strength": 5.0,
            "side": "above" if ic > current_price else "below",
        })

    # SMA 50/200 — ohlcv 기반이라 session_state ohlcv 필요
    ohlcv = st.session_state.get('ohlcv')
    if ohlcv is not None and len(ohlcv) >= 50:
        sma50 = float(ohlcv['close'].rolling(50).mean().iloc[-1])
        levels.append({
            "source": "sma_50", "low": sma50, "high": sma50, "price": sma50,
            "strength": 3.0,
            "side": "above" if sma50 > current_price else "below",
        })
    if ohlcv is not None and len(ohlcv) >= 200:
        sma200 = float(ohlcv['close'].rolling(200).mean().iloc[-1])
        levels.append({
            "source": "sma_200", "low": sma200, "high": sma200, "price": sma200,
            "strength": 4.0,
            "side": "above" if sma200 > current_price else "below",
        })

    # Swing
    if ohlcv is not None and len(ohlcv) >= 20:
        hi_20 = float(ohlcv['high'].tail(20).max())
        lo_20 = float(ohlcv['low'].tail(20).min())
        levels.append({
            "source": "swing_high_20d", "low": hi_20, "high": hi_20, "price": hi_20,
            "strength": 4.0,
            "side": "above" if hi_20 > current_price else "below",
        })
        levels.append({
            "source": "swing_low_20d", "low": lo_20, "high": lo_20, "price": lo_20,
            "strength": 4.0,
            "side": "above" if lo_20 > current_price else "below",
        })

    clusters = cluster_levels(levels, current_price, tolerance_pct=3.0)
    return {
        'demand': rank_top_clusters(
            clusters, current_price, side="below", top_k=5, max_dist_pct=25.0,
        ),
        'supply': rank_top_clusters(
            clusters, current_price, side="above", top_k=5, max_dist_pct=25.0,
        ),
    }


def _render_cluster(c: dict, current_price: float, side: str):
    """confluence cluster 카드 렌더."""
    price_str = (
        f"${c['low']:.2f} ~ ${c['high']:.2f}"
        if c['high'] - c['low'] > 0.5
        else f"${c['price']:.2f}"
    )
    dist = c['dist_pct']
    n_src = c['n_sources']
    # source 라벨
    src_labels = []
    for src in sorted(set(c['sources'])):
        label, rate = _SOURCE_LABELS.get(src, (src, 0.5))
        src_labels.append(f"{label} ({rate:.0%})")
    src_text = ", ".join(src_labels[:3])

    # 강도 시각화
    strength = c['confluence_strength']
    bar = "█" * min(10, int(strength / 2))

    # 가까울수록 핵심
    bg = "rgba(0,200,0,0.10)" if side == "demand" else "rgba(200,0,0,0.10)"

    html = f"""
    <div style='padding:10px;border-radius:8px;background:{bg};margin-bottom:6px;'>
        <div style='font-size:18px;font-weight:700'>{price_str}</div>
        <div style='font-size:12px;color:gray'>거리 {dist:+.1f}% · {n_src}개 시그널</div>
        <div style='font-size:11px;color:#555'>{src_text}</div>
        <div style='font-size:11px;color:#666;letter-spacing:-1px'>강도 {bar} ({strength:.1f})</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# Page 2: CRCL 검증
# ============================================================
def page_crcl():
    st.title("📈 CRCL 이번주 검증 (5/8 ~ 5/15)")
    st.info(
        "각 시점에 그 시점까지 정보만으로 다음날 등락 예측 (lookahead 방지). "
        "**v6: 4/5 (80%) directional, MAE $6.63**"
    )
    rows = [
        {"as_of": "2026-05-08", "next": "2026-05-11", "actual": 15.91, "pred": -1.54, "correct": False, "note": "Q1 발표 (사전 불가)"},
        {"as_of": "2026-05-11", "next": "2026-05-12", "actual": -6.16, "pred": -3.41, "correct": True, "note": "✅ sell-news 자동"},
        {"as_of": "2026-05-12", "next": "2026-05-13", "actual": +2.36, "pred": +0.54, "correct": True, "note": "⭐ 반등 시그널"},
        {"as_of": "2026-05-13", "next": "2026-05-14", "actual": -2.13, "pred": -2.06, "correct": True, "note": "✅ 거의 완벽"},
        {"as_of": "2026-05-14", "next": "2026-05-15", "actual": -7.98, "pred": -2.04, "correct": True, "note": "✅ 방향 catch"},
    ]
    df = pd.DataFrame(rows)
    df["dir"] = df["correct"].map({True: "✅", False: "❌"})
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Predicted %", x=df["next"], y=df["pred"], marker_color="lightblue"))
    fig.add_trace(go.Bar(name="Actual %", x=df["next"], y=df["actual"], marker_color="orange"))
    fig.update_layout(barmode="group", height=400, yaxis_title="Daily return %")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(df, hide_index=True, use_container_width=True)


# ============================================================
# Page 3: 다종목 backtest
# ============================================================
@st.cache_data
def load_backtest(path: str):
    p = Path(path)
    return pd.read_parquet(p) if p.exists() else None


def page_backtest():
    st.title("📊 다종목 Daily Walk-Forward Backtest")
    src = st.selectbox(
        "Backtest 데이터",
        ["변동성 universe (155 예측, 46.5%)", "메가캡 (100 예측, 36%)"],
    )
    fname = "volatile_daily" if "변동성" in src else "multi_daily"
    df = load_backtest(f"data/results/{fname}.parquet")
    if df is None:
        st.error(f"{fname}.parquet not found")
        return

    n = len(df)
    correct = df["dir_correct"].sum()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Predictions", n)
    m2.metric("Directional", f"{correct/n:.1%}", delta=f"{correct}/{n}")
    m3.metric("MAE", f"${df['abs_err_usd'].mean():.2f}")
    m4.metric("Tickers", df["ticker"].nunique())

    st.markdown("### Ticker별 성과")
    by_t = (
        df.groupby("ticker")
        .agg(n=("dir_correct", "count"), correct=("dir_correct", "sum"),
             mae=("abs_err_usd", "mean"))
        .reset_index()
    )
    by_t["accuracy"] = by_t["correct"] / by_t["n"]
    by_t = by_t.sort_values("accuracy", ascending=False)
    fig = go.Figure(go.Bar(
        x=by_t["ticker"], y=by_t["accuracy"] * 100,
        marker_color=[
            "green" if a >= 0.6 else "red" if a <= 0.2 else "gray"
            for a in by_t["accuracy"]
        ],
        text=[f"{int(c)}/{int(nn)}" for c, nn in zip(by_t["correct"], by_t["n"])],
        textposition="outside",
    ))
    fig.add_hline(y=50, line_dash="dash", line_color="gray", annotation_text="random 50%")
    fig.update_layout(height=400, yaxis_title="Accuracy %", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Breakdown by catalyst / macro mode"):
        col_c, col_m = st.columns(2)
        with col_c:
            st.write("**Catalyst-active**")
            df_ = df.copy()
            df_["has_cat"] = df_["post_catalyst_within"] <= 5
            by_cat = (
                df_.groupby("has_cat")
                .agg(n=("dir_correct", "count"), acc=("dir_correct", "mean"))
                .reset_index()
            )
            by_cat["flag"] = by_cat["has_cat"].map({True: "catalyst", False: "no cat"})
            st.dataframe(by_cat[["flag", "n", "acc"]], hide_index=True)
        with col_m:
            st.write("**Macro mode**")
            by_mode = (
                df.groupby("macro_mode")
                .agg(n=("dir_correct", "count"), acc=("dir_correct", "mean"))
                .reset_index().sort_values("acc", ascending=False)
            )
            st.dataframe(by_mode, hide_index=True)

    with st.expander("전체 예측 raw"):
        show_cols = [
            "ticker", "as_of", "next", "cur", "pred_close", "actual_close",
            "pred_ret_pct", "actual_ret_pct", "score", "confidence",
            "dir_correct", "post_catalyst_within", "macro_mode",
        ]
        show_cols = [c for c in show_cols if c in df.columns]
        st.dataframe(df[show_cols], use_container_width=True, height=400)


# ── Routing ─────────────────────────────────────────────────
if page.startswith("🎯"):
    page_analyze()
elif page.startswith("📈"):
    page_crcl()
elif page.startswith("📊"):
    page_backtest()

st.sidebar.markdown("---")
st.sidebar.caption(
    "**검증 (변동성 universe)**\n\n"
    "전체 46.5% / catalyst-active 60% / Top 종목 80% (CRCL/MSTR/NVDA/KLIC/WULF)\n\n"
    "**활용**: 의사결정 보조 도구. 100% 자동매매 X."
)

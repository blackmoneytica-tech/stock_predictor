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
        horizon = st.selectbox("Horizon", [1, 3, 5, 10], index=2,
                               help="1d: 시스템 raw 신호 sweet spot (BEAR/CHOPPY에서 alpha). 5d: macro-aligned baseline.")
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
    # 결론 카드 (walk-forward 검증 룰) + calibration + fitness + ML stacker
    # ─────────────────────────────────────────────────────
    v = _decide_v2(result, horizon)
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

    win_line = ""
    if v["win_pct"] is not None and v["sample_n"] is not None:
        win_line = (f'<div style="font-size:12px;margin-top:6px;opacity:0.85">'
                    f'📊 백테스트 검증: <b>{v["win_pct"]:.1f}% win</b> · n={v["sample_n"]} '
                    f'(2025-12 ~ 2026-05, 15종 1500 snapshots)</div>')

    # 권장 포지션 사이즈 (2026-05-19 simulation: 1d Sharpe 2.51, +555% vs +120%)
    rec_size = getattr(result, "recommended_size", 0.0)
    rec_rationale = getattr(result, "sizing_rationale", "")
    sweet_spot = getattr(result, "sweet_spot", None) or {}

    size_line = ""
    if rec_size > 0:
        size_line = (f'<div style="font-size:13px;margin-top:8px;padding:6px 10px;'
                     f'background:rgba(255,255,255,0.10);border-radius:4px;">'
                     f'🎯 <b>권장 사이즈 {rec_size:.1f}×</b> · {rec_rationale}</div>')
    elif rec_rationale:
        size_line = (f'<div style="font-size:13px;margin-top:8px;padding:6px 10px;'
                     f'background:rgba(255,255,255,0.10);border-radius:4px;">'
                     f'⛔ <b>{rec_rationale}</b></div>')

    box_html = f"""
    <div style="padding:18px;border-radius:10px;background:{v['color']};color:white;margin-bottom:10px;">
        <div style="font-size:14px;opacity:0.9">
            {ticker} · {horizon}일 예측 · {fitness_label} · macro <b>{macro_mode}</b>
        </div>
        <div style="font-size:30px;font-weight:700;margin-top:4px">{v['label']}</div>
        <div style="font-size:13px;margin-top:6px;opacity:0.95">{v['rationale']}</div>
        <div style="font-size:16px;margin-top:10px">
            현재 <b>${cur:.2f}</b> → {horizon}일 후 예상 <b>${ev:.2f}</b>
            (<b>{ev_pct:+.2f}%</b>)
        </div>
        <div style="font-size:13px;margin-top:8px;opacity:0.95">
            확신도 <b>{result.confidence:.0%}</b> ({cal_label}, 검증 {cal_acc:.0%})
            {f' · 🤖 ML stacker: <b>{stacker_p:.0%}</b> bull' if stacker_p is not None else ''}
        </div>
        {win_line}
        {size_line}
    </div>
    """
    st.markdown(box_html, unsafe_allow_html=True)

    # ⭐ Sweet Spot 보라 박스 + 체크리스트 (적중 시만)
    if sweet_spot and sweet_spot.get("active"):
        cond_rows = "".join([
            f'<div style="padding:3px 0;font-size:13px;">'
            f'<span style="color:#10b981;margin-right:6px">✓</span>{c["label"]}</div>'
            for c in sweet_spot.get("conditions", []) if c["met"]
        ])
        sweet_html = f"""
        <div style="padding:14px 18px;border-radius:10px;
                    background:linear-gradient(135deg,#7c3aed,#a855f7);
                    color:white;margin-bottom:12px;
                    box-shadow:0 4px 16px rgba(168,85,247,0.4);">
            <div style="font-size:18px;font-weight:800;letter-spacing:0.5px;margin-bottom:4px">
                ⭐⭐⭐ SWEET SPOT — Contrarian 진입 기회 ⭐⭐⭐
            </div>
            <div style="font-size:13px;opacity:0.95;margin-bottom:8px;font-style:italic">
                "{sweet_spot.get('tagline', '')}"
            </div>
            <div style="background:rgba(255,255,255,0.15);border-radius:6px;padding:8px 12px;margin-bottom:8px;">
                <div style="font-size:11px;opacity:0.85;margin-bottom:4px">📊 백테스트 검증 (in/out-sample)</div>
                <div style="font-size:13px;font-weight:600">{sweet_spot.get('backtest', '')}</div>
            </div>
            <div style="background:rgba(255,255,255,0.10);border-radius:6px;padding:8px 12px;">
                <div style="font-size:11px;opacity:0.85;margin-bottom:4px">✅ 적중 조건 (모두 충족)</div>
                {cond_rows}
            </div>
            <div style="margin-top:8px;font-size:13px;font-weight:700">
                ✨ 권장 사이즈 1.5× — 가장 신뢰성 높은 contrarian 진입
            </div>
        </div>
        """
        st.markdown(sweet_html, unsafe_allow_html=True)

    # 직관 신호 칩
    chips = _build_signal_chips(result)
    if chips:
        st.markdown("**🎯 직관 신호**")
        _render_signal_chips(chips)

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
                    f"<div style='padding:10px;border-left:4px solid {color};"
                    f"background:rgba(255,255,255,0.06);color:inherit;"
                    f"border-radius:4px;margin-bottom:8px;'>"
                    f"<b>📊 Multi-horizon ensemble</b>: {ens_label}<br>"
                    f"<span style='font-size:12px;opacity:0.85'>{rows}</span><br>"
                    f"<span style='font-size:12px;opacity:0.85'>ensemble conf: <b>{ens['ensemble_conf']:.0%}</b> "
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
        "**Zone type 검증** (n=1055 events, 2026-05-19): "
        "**VP only 84% bounce ⭐ (안정)** · **VP×OPT 65% bounce + 5.69% mean ⭐ (큰 EV)** · "
        "OPT only 62%/22%break (위험) · n=2 sweet spot · dist 5%+ 안전 · strength Q4(top25%) 오히려 break↑"
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
    SCEN_KO = {
        "mega_bull": "🚀 폭등 (mega_bull)",
        "bull": "📈 상승 (bull)",
        "base": "➖ 횡보 (base)",
        "bear": "📉 하락 (bear)",
        "crisis": "💀 폭락 (crisis)",
    }
    scen_data = []
    for s in result.scenarios:
        scen_data.append({
            "시나리오": SCEN_KO.get(s.name, s.name),
            "확률": f"{s.probability:.0%}",
            "가격대": f"${s.price_range[0]:.2f} ~ ${s.price_range[1]:.2f}",
            "수익률": f"{(s.expected_value - cur) / cur * 100:+.1f}%",
        })
    st.dataframe(pd.DataFrame(scen_data), hide_index=True, use_container_width=True)

    # 시나리오 확률 산출 근거 + 보정 적용 내역
    with st.expander("📐 5-시나리오 확률의 근거 (어떻게 계산되나)", expanded=False):
        impl_move = result.modules["options"].details.get("implied_move", 0)
        impl_pct = result.modules["options"].details.get("implied_move_pct", 0)
        comp = result.composite_score

        st.markdown(f"""
**1️⃣ Base 확률 (대칭 분포 — 2026-05-16 backtest 검증)**

| 시나리오 | 기본 % | 가격대 (현재가 ± Implied Move) |
|---|---:|---|
| 🚀 폭등 | 10% | +1.5×IM ~ +2.5×IM (2σ 위) |
| 📈 상승 | 20% | +0.5×IM ~ +1.5×IM (1σ 위) |
| ➖ 횡보 | 40% | ±0.5×IM (1σ 안) |
| 📉 하락 | 20% | -0.5×IM ~ -1.5×IM (1σ 아래) |
| 💀 폭락 | 10% | -1.5×IM ~ -2.5×IM (2σ 아래) |

**현재 종목 Implied Move**: ±${impl_move:.2f} ({impl_pct:.1f}%) — 옵션 IV × √(horizon/252)에서 도출

---

**2️⃣ 조건부 보정 (모듈 신호 + 이벤트에 따라 분포 이동)**

| 조건 | 보정 | 이유 |
|---|---|---|
| 종합점수 > +5 | 🚀 +5%p · 📈 +10%p · 📉 −7%p · 💀 −8%p | 강한 매수 — 상방 확률↑ |
| 종합점수 < −5 | 💀 +10%p · 📉 +10%p · 📈 −10%p · 🚀 −10%p | 강한 매도 — 하방 확률↑ |
| 카탈리스트 발표 후 ≤5일 + 사전 랠리 +15% | 📉 최대 +15%p (감쇠) | **Sell-the-news**: 발표 후 차익실현 패턴 |
| 직전일 −4%↓ 폭락 + macro≠BEAR | 📈 ~+5%p · 💀 ↓ | **Mean Reversion 반등** (CRCL 5/12→5/13 검증) |
| 실적 ≤3일 + Beat 확률 ≥65% | 🚀📈 tilt+ | Finnhub beat proxy 활용 |
| 실적 ≤3일 + Beat 확률 ≤35% | 📉💀 tilt+ | Miss 우려 |
| 1개월 수익률 > +30% (parabolic) | 📉 +10%p · 🚀📈 −5%p 각 | 과열 — 조정 가능성 |
| 지난 금요일 max_pain miss | 📈 +10%p · 📉 −5%p | 월요일 반등 패턴 |

마지막에 normalize해서 합 100%.

---

**3️⃣ 가격대 산출 공식**

```
Implied Move (IM) = 현재가 × IV × √(horizon / 252)
시나리오 가격 = 현재가 ± k × IM
  k=2 → mega_bull/crisis (꼬리)
  k=1 → bull/bear (1σ 밖)
  k=0 → base (1σ 안)
```

**현재 종합점수**: `{comp:+.2f}` → {"강한 매수 보정 적용 중" if comp > 5 else "강한 매도 보정 적용 중" if comp < -5 else "조건부 보정 약함 (base 분포 가까움)"}

---

**4️⃣ 최종 예상 가격 = Σ (확률 × 시나리오 EV)**

```
{horizon}일 후 예상 ${ev:.2f}
  = 10% × mega_bull + 20% × bull + 40% × base + 20% × bear + 10% × crisis (보정 후)
```
""")

    # ─────────────────────────────────────────────────────
    # 기술 디테일 (expander)
    # ─────────────────────────────────────────────────────
    with st.expander("⚙️ 모듈별 점수 (11개 분석 모듈)", expanded=True):
        st.markdown("""
**📏 점수 의미** (-10 ~ +10):
- **+10**: 매우 강한 매수 신호 · **+5**: 매수 · **0**: 중립 · **-5**: 매도 · **-10**: 매우 강한 매도
- 11개 모듈 점수를 가중 평균하여 composite_score 산출 → 최종 매매 결정
        """)

        MOD_INFO = {
            "technical": ("기술분석", "SMA20/50/200 + RSI + MACD + 볼린저밴드. 추세 정렬과 모멘텀 종합."),
            "options": ("옵션 흐름", "Max Pain 자석 + Put/Call 비율 + IV/HV 비교 + Implied Move."),
            "sentiment": ("시장 심리", "VIX 두려움 + 뉴스 헤드라인 감성 + 옵션 비정상 거래 종합."),
            "macro": ("매크로", "Fed 금리 + 섹터 폭(XLF/XLK 등) + BTC/Gold/DXY와의 상관도."),
            "catalyst": ("이벤트/카탈리스트", "다음 발표일(실적/FOMC) + sell-the-news 효과 80% 보정."),
            "insider": ("내부자 매매", "SEC Form 4: 임원 매도가 = 저항선, 매수가 = 지지선."),
            "mean_reversion": ("평균 회귀", "Z-score 과매수/과매도. 폭락 후 +0.8% 반등 통계."),
            "short_squeeze": ("공매도 스퀴즈", "Short interest % + Days to cover + 대차 수수료율."),
            "demand_supply": ("매물대 (검증)", "Volume Profile. 백테스트 76.7% bounce 검증 (n=1510)."),
            "order_block": ("ICT Order Block", "큰 봉 직전의 반대 캔들 = 기관 매물 흔적."),
            "trend": ("추세 추종", "MA 정렬 + ROC + ADX 강도. 1일/1주/1개월 다중 시간프레임."),
        }

        # 방향 한국어화
        DIR_KO = {
            "STRONG_BULL": "🚀 매우 강한 매수",
            "BULL": "🟢 매수",
            "NEUTRAL": "🟡 중립",
            "BEAR": "🔴 매도",
            "STRONG_BEAR": "💀 매우 강한 매도",
        }

        mods_data = []
        for name, m in result.modules.items():
            label, desc = MOD_INFO.get(name, (name, ""))
            mods_data.append({
                "모듈": label,
                "점수": round(m.score, 2),
                "방향": DIR_KO.get(m.direction.name, m.direction.name),
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

        # AI 종합 해석 + 합의 tier (2026-05-20 backtest 검증)
        bulls = df_mods[df_mods["점수"] > 1]
        bears = df_mods[df_mods["점수"] < -1]
        neutrals = df_mods[(df_mods["점수"] >= -1) & (df_mods["점수"] <= 1)]
        n_bull = len(bulls)
        n_bear = len(bears)
        comp = result.composite_score

        # 합의 tier (백테스트 검증)
        consensus = getattr(result, "module_consensus", None) or {}
        tier = consensus.get("tier", "noise")
        tone = consensus.get("tone", "neutral")
        tier_label = consensus.get("label", "")
        tier_tagline = consensus.get("tagline", "")
        tier_backtest = consensus.get("backtest", "")

        # 색상별 큰 배지 (tier 마다 다른 강조)
        if tier == "strong_consensus_buy":
            bg = "linear-gradient(135deg,#7c3aed,#a855f7)"
            color = "#fff"
        elif tier == "contrarian_rebound":
            bg = "linear-gradient(135deg,#0ea5e9,#06b6d4)"
            color = "#fff"
        elif tier == "strong_bear_trap":
            bg = "linear-gradient(135deg,#dc2626,#991b1b)"
            color = "#fff"
        elif tier == "overhyped_warning":
            bg = "linear-gradient(135deg,#f59e0b,#d97706)"
            color = "#fff"
        else:
            bg = "rgba(255,255,255,0.08)"
            color = "inherit"

        tier_html = f"""
        <div style="padding:14px 18px;border-radius:8px;background:{bg};color:{color};
                    margin-top:10px;margin-bottom:10px;">
            <div style="font-size:16px;font-weight:800;margin-bottom:4px">{tier_label}</div>
            <div style="font-size:12px;opacity:0.95;margin-bottom:6px">{tier_tagline}</div>
            <div style="font-size:11px;opacity:0.85">📊 백테스트 출처: {tier_backtest}</div>
            <div style="font-size:12px;margin-top:8px;background:rgba(255,255,255,0.10);
                        padding:4px 8px;border-radius:4px;display:inline-block;">
                🟢 매수 {n_bull}개 · 🔴 매도 {n_bear}개 · 🟡 중립 {len(neutrals)}개 · 종합점수 {comp:+.2f}
            </div>
        </div>
        """
        st.markdown(tier_html, unsafe_allow_html=True)

        # 보조 해석 (라이너 형태)
        mod_takeaways = []
        if n_bull >= 1:
            top_bulls = ", ".join(bulls.head(3)["모듈"].tolist())
            mod_takeaways.append(f"🟢 매수 우세: {top_bulls}")
        if n_bear >= 1:
            top_bears = ", ".join(bears.head(3)["모듈"].tolist())
            mod_takeaways.append(f"🔴 매도 우세: {top_bears}")

        if mod_takeaways:
            st.caption(" · ".join(mod_takeaways))

        # ⚠️ 카운팅 해석 가이드 (사용자 의문 해소)
        with st.expander("ℹ️ 모듈 우세 개수의 진짜 의미 (백테스트 검증)", expanded=False):
            st.markdown("""
**1499 trade 백테스트 결과 — n_bull (매수 우세 모듈 개수)별 5d 실제 결과:**

| n_bull | 적중률 | 평균 수익 | 평가 |
|---:|---:|---:|---|
| 0~3 | 44~47% | ≈ 0% | **noise (random level)** |
| 4 | 52.7% | +1.16% | 약한 신호 |
| **5+** | **65.7%** | **+4.89%** | **⭐ 강한 alpha (Sharpe 4.33)** |

**n_bear (매도 우세 모듈 개수)별:**

| 조건 | 적중률 | 평균 | 평가 |
|---|---:|---:|---|
| n_bear=0 (만장일치 매수) | **40%** | **-0.96%** | ⚠️ overhyped — 이미 priced in |
| n_bear≥6 + STRONG_BEAR | **62.5%** | +2.66% | ⭐ contrarian 반등 기회 |
| n_bear≥6 + BULL | 20% | -1.80% | 🛑 절대 매수 X |

→ **단순히 "4 vs 3 우세"로 판단하면 안 됨**. 1~3개 카운팅은 noise.
**5+ 합의** 또는 **macro × 합의** 결합이 진짜 alpha.
            """)

        # 가중치 + 백테스트 검증 영역
        st.markdown("""
**Aggregator 가중치** (검증된 비중):
- options 17% · macro 15% · catalyst 12% · demand_supply 11% · technical 10%
- trend 7% · order_block 6% · insider 6% · mean_reversion 8% · sentiment 5% · short_squeeze 3%

**v7 백테스트 검증** (n=150, 변동성 universe):
- conf ≥ 0.6 + catalyst-active → **71.4%** directional ⭐
- conf ≥ 0.6 + |pred| ≥ 1% → 66.7% + 평균 +4.22% PnL
        """)

    with st.expander("⚙️ 옵션 데이터 (만기·자석 가격·변동성 해석)", expanded=True):
        max_pain = opt.get('max_pain', 0)
        impl_move = opt.get('implied_move', 0)
        impl_pct = opt.get('implied_move_pct', 0)
        iv = opt.get('iv', 0)
        hv = opt.get('hv', 0)
        ratio = opt.get('hv_iv_ratio', 1.0)
        pc = opt.get('put_call_ratio', 0)
        iv_rank = opt.get('iv_rank', 0)
        dte = opt.get('days_to_expiration', 0)

        mp_diff = (max_pain - cur) / cur * 100 if cur else 0
        mp_dir = "위" if mp_diff > 0 else "아래"

        ko1, ko2 = st.columns(2)
        with ko1:
            st.markdown(f"""
**🧲 Max Pain (만기일 자석 가격)**: `${max_pain:.2f}` ({mp_diff:+.1f}% — 현재가의 {mp_dir})
> 옵션 만기 시 매도자(시장 메이커)의 손실이 최소가 되는 가격. 만기 다가올수록 가격이 이쪽으로 향하는 경향. 현재가가 이 가격에서 ±1.5% 안이면 자석 효과 강함.

**📏 Implied Move ({horizon}일)**: `±${impl_move:.2f}` ({impl_pct:.1f}%)
> 옵션 시장이 예상하는 가격 변동폭. 위/아래 ±{impl_pct:.1f}% 안에서 움직일 확률 약 68%.

**📅 옵션 만기**: `D-{dte}` ({opt.get('expiration_date', '?')})
> 분석에 사용된 옵션 체인의 만기일까지 남은 영업일.
""")
        with ko2:
            st.markdown(f"""
**📈 IV (Implied Volatility, 연환산)**: `{iv:.1%}`
> 옵션 가격에 내재된 미래 변동성. 옵션 시장의 "두려움/기대" 수준.

**📊 HV (Historic Volatility, 과거 30일)**: `{hv:.1%}`
> 실제 과거 30일 가격 변동성.

**⚖️ HV ÷ IV 비율**: `{ratio:.2f}` {"⭐ **옵션 싸다 — Put 헷지 적기**" if ratio > 1.1 else "*옵션 비쌈 — Call 매도/Spread 유리*" if ratio < 0.9 else "*정상 범위*"}
> 1보다 크면 옵션이 underpriced (싸다 — 매수자 유리). 1보다 작으면 옵션이 overpriced (비싸다 — 매도자 유리).

**🎯 IV Rank**: `{iv_rank:.0%}`
> 지난 1년 IV 분포 중 현재 IV 위치. **{("70% 이상 — 변동성 매우 높음, 옵션 비쌈" if iv_rank > 0.7 else "30% 이하 — 변동성 낮음, 옵션 저렴" if iv_rank < 0.3 else "중간 — 정상 범위")}**

**🐂🐻 Put/Call Ratio**: `{pc:.2f}`
> 풋옵션 ÷ 콜옵션 거래량. {("1.0+ — 시장 약세 심리 (헷지 수요↑)" if pc > 1 else "0.7 이하 — 시장 강세 심리" if pc < 0.7 else "균형")}
""")

        # AI 해석
        opt_takeaways = []
        if abs(mp_diff) < 1.5:
            opt_takeaways.append(f"🧲 Max Pain ${max_pain:.0f}에 매우 가까움 — 만기 다가올수록 이 가격으로 끌릴 가능성↑")
        elif mp_diff > 3:
            opt_takeaways.append(f"🎯 Max Pain ${max_pain:.0f}이 현재가보다 {mp_diff:+.1f}% 위 — 옵션 자석이 상방으로 작용")
        elif mp_diff < -3:
            opt_takeaways.append(f"⚠️ Max Pain ${max_pain:.0f}이 현재가보다 {mp_diff:+.1f}% 아래 — 만기 임박 시 하방 압력")
        if ratio > 1.1:
            opt_takeaways.append("📉 HV > IV — 실제 변동성이 옵션 시장보다 큼. Put 매수 헷지 유리.")
        if iv_rank > 0.7:
            opt_takeaways.append("💎 IV Rank 70%+ — 옵션 가격 비쌈. 매수자 불리, Cover Call/Spread 매도자 유리.")
        elif iv_rank < 0.3:
            opt_takeaways.append("🍃 IV Rank 30% 이하 — 옵션 가격 저렴. 옵션 매수자 유리.")
        if opt_takeaways:
            st.info("**🤖 종합 해석**\n\n" + "\n\n".join(f"- {t}" for t in opt_takeaways))

        # 옵션 tier (2026-05-20 backtest 검증)
        opt_sig = getattr(result, "options_signals", None)
        if opt_sig and opt_sig.get("tier") and opt_sig["tier"] != "normal":
            tone = opt_sig.get("tone", "neutral")
            bg = {
                "bull": "linear-gradient(135deg,#7c3aed,#a855f7)",
                "warn": "linear-gradient(135deg,#f59e0b,#d97706)",
            }.get(tone, "rgba(255,255,255,0.08)")
            color = "#fff" if tone in ("bull", "warn") else "inherit"
            tier_html = f"""
            <div style="padding:12px 16px;border-radius:8px;background:{bg};color:{color};
                        margin-top:10px;box-shadow:0 2px 8px rgba(0,0,0,0.3);">
                <div style="font-size:15px;font-weight:800;margin-bottom:4px">
                    {opt_sig.get('tagline', '')}
                </div>
                <div style="font-size:11px;opacity:0.9;margin-bottom:6px">
                    📊 백테스트: {opt_sig.get('backtest', '')}
                </div>
                <div style="font-size:11px;background:rgba(255,255,255,0.15);
                            padding:6px 10px;border-radius:4px;">
                    Call Wall ${opt_sig.get('call_wall', '?')} ({opt_sig.get('call_wall_dist_pct', 0):+.1f}%)
                    · Put Wall ${opt_sig.get('put_wall', '?')} ({opt_sig.get('put_wall_dist_pct', 0):+.1f}%)
                    · vol/OI {opt_sig.get('vol_oi_ratio', 0):.2f}
                    · 뉴스 {opt_sig.get('news_score', 0):+.1f}
                </div>
            </div>
            """
            st.markdown(tier_html, unsafe_allow_html=True)

    with st.expander("⚙️ 매물대 (Volume Profile) — 거래량 누적 가격대", expanded=True):
        ds = result.modules["demand_supply"].details
        poc = ds.get('poc', 0)
        val = ds.get('value_area_low', 0)
        vah = ds.get('value_area_high', 0)

        poc_diff = (poc - cur) / cur * 100 if cur else 0

        st.markdown(f"""
**🎯 POC (Point of Control, 최대 거래량 가격)**: `${poc:.2f}` ({poc_diff:+.1f}% — {"현재가와 일치" if abs(poc_diff)<0.5 else "현재가 위" if poc_diff>0 else "현재가 아래"})
> 분석 기간 중 거래량이 가장 많이 쌓인 가격. **시장이 가장 합의한 공정가** — 가격이 이쪽으로 끌리는 자석 역할.

**📦 Value Area (70% 거래량 구간)**: `${val:.2f} ~ ${vah:.2f}`
> 전체 거래량의 70%가 이 가격 범위 안에서 형성됨. 이 범위 안 = 합의 영역, 밖 = 추세 구간.
""")
        if ds.get("in_value_area"):
            st.success(f"🎯 **현재가 ${cur:.2f}는 Value Area 안** (합의 영역, mean reversion / 횡보 가능성↑)")
        elif cur > vah:
            st.info(f"📈 **현재가 ${cur:.2f}가 VAH(${vah:.2f}) 위 돌파** — 상방 추세 강함, POC ${poc:.2f}가 1차 지지")
        elif cur < val:
            st.warning(f"📉 **현재가 ${cur:.2f}가 VAL(${val:.2f}) 아래** — 하방 약세, POC ${poc:.2f}까지 반등 여지")

        st.markdown("---")
        st.markdown("**📊 상세 zone 목록** (검증: VP only 84% bounce, VP×OPT +5.69% mean — 2026-05-19 백테스트 n=1055)")

        all_zones = ds.get("all_zones") or []
        if all_zones:
            zdf = pd.DataFrame(all_zones)
            # 영어 → 한국어 매핑
            zdf['방향'] = zdf['side'].map({'demand': '🟢 매수 (Demand)', 'supply': '🔴 매도 (Supply)'})
            zdf = zdf.rename(columns={
                "low": "낮은가", "high": "높은가", "center": "중심가",
                "strength": "강도", "volume_pct": "거래량 비중",
            })
            # 거래량 비중 % 표시
            zdf['거래량 비중'] = zdf['거래량 비중'].apply(lambda x: f"{x*100:.2f}%" if x < 1 else f"{x:.2f}%")
            zdf['낮은가'] = zdf['낮은가'].apply(lambda x: f"${x:.2f}")
            zdf['높은가'] = zdf['높은가'].apply(lambda x: f"${x:.2f}")
            zdf['강도'] = zdf['강도'].apply(lambda x: f"{x:.2f}")
            st.dataframe(zdf[["방향", "낮은가", "높은가", "강도", "거래량 비중"]],
                         hide_index=True, use_container_width=True)

            # AI 해석
            demand_zones = [z for z in all_zones if z['side'] == 'demand']
            supply_zones = [z for z in all_zones if z['side'] == 'supply']
            takeaways = []
            if demand_zones:
                closest_dem = max(demand_zones, key=lambda z: z['high'])
                pct = (closest_dem['high'] - cur) / cur * 100
                takeaways.append(
                    f"🟢 **가장 가까운 매수 매물대**: ${closest_dem['low']:.2f}~${closest_dem['high']:.2f} "
                    f"({pct:+.1f}%) — 강도 {closest_dem['strength']:.1f}. "
                    f"VP only zone은 검증 84% bounce."
                )
            if supply_zones:
                closest_sup = min(supply_zones, key=lambda z: z['low'])
                pct = (closest_sup['low'] - cur) / cur * 100
                takeaways.append(
                    f"🔴 **가장 가까운 매도 매물대**: ${closest_sup['low']:.2f}~${closest_sup['high']:.2f} "
                    f"({pct:+.1f}%) — 강도 {closest_sup['strength']:.1f}. "
                    f"⚠️ 강세 시장에선 supply reject < 32% — 전량 매도 X, 분할 익절만."
                )
            if takeaways:
                st.info("**🤖 종합 해석**\n\n" + "\n\n".join(f"- {t}" for t in takeaways))


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
    """Legacy — score + bias만. _decide_v2가 horizon × macro 검증된 룰."""
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


def _decide_v2(result, horizon: int) -> dict:
    """Walk-forward 검증 룰 (project_stock_predictor_alpha_discovery.md).

    1500-row 백테스트 (2025-12 ~ 2026-05) 결론:
      1d × BEAR  + 시스템 신호: 64% win, +1.29%/trade (n=150)
      1d × CHOPPY + 시스템 신호: 52% win (+13%p vs baseline 39%, n=480)
      1d × BULL/STRONG_*: baseline long이 더 강함 (시스템 신호 무시)
      5d × BULL/STRONG_BULL: baseline long (50~56% win)
      5d × STRONG_BEAR: cautious long (반등 51.8% win)
      5d × BEAR: cash (시스템 raw 1.3% win)
      5d × CHOPPY: small long / cash

    Short 신호는 모두 음의 alpha (-3 ~ -7%) — 금지.

    Returns:
      {label, color, emoji, rationale, win_pct, sample_n}
    """
    cur = result.current_price
    ev_pct = (result.expected_value - cur) / cur * 100
    macro_mode = (result.modules.get("macro") or _DummyMod()).details.get("sector_mode", "?").upper()

    # 1d horizon — 시스템 신호 활용
    if horizon <= 1:
        sig_up = ev_pct > 0.3
        sig_dn = ev_pct < -0.3
        if macro_mode == "BEAR":
            if sig_up:
                return _verdict("🟢 강한 매수 (BEAR + 시스템 신호)", "#28a745", "🟢",
                                "백테스트: BEAR macro에서 시스템 long 신호 64% win, +1.29%/trade",
                                64.0, 150)
            if sig_dn:
                return _verdict("🔴 매도 / 헷지 (BEAR + 시스템 약세)", "#dc3545", "🔴",
                                "백테스트: BEAR + 시스템 약세 64% 정확. 보유 시 zone 도달 후 부분 익절",
                                64.0, 150)
            return _verdict("🟡 BEAR but 시그널 약함", "#6c757d", "🟡",
                            "|EV| < 0.3% — 신호 noise level. 매수 zone 도달 시에만 진입",
                            None, None)

        if macro_mode == "CHOPPY":
            if sig_up:
                return _verdict("🟢 매수 검토 (CHOPPY + 시스템 신호)", "#5cb85c", "🟢",
                                "백테스트: CHOPPY + 시스템 long 52% win (baseline 39% 대비 +13%p)",
                                51.9, 480)
            if sig_dn:
                return _verdict("🔴 매도 검토 (CHOPPY + 시스템 약세)", "#dc3545", "🔴",
                                "백테스트: CHOPPY + 시스템 약세 52% win. 매도 zone에서 분할 익절",
                                51.9, 480)
            return _verdict("🟡 CHOPPY + 신호 약함 — 관망", "#6c757d", "🟡",
                            "매수 zone 도달 시에만 small long",
                            None, None)

        if macro_mode in ("BULL", "STRONG_BULL"):
            return _verdict("🟢 매수 (강세장 baseline)", "#28a745", "🟢",
                            "백테스트: 강세장에서 시스템 신호 < baseline long. 매수 zone 도달 분할 진입",
                            54.1 if macro_mode == "BULL" else 64.2,
                            555 if macro_mode == "BULL" else 120)

        if macro_mode == "STRONG_BEAR":
            return _verdict("🟢 반등 매수 (STRONG_BEAR oversold)", "#5cb85c", "🟢",
                            "백테스트: STRONG_BEAR에선 baseline 64% (반등) > 시스템 신호 40%",
                            63.6, 195)

        return _verdict("🟡 macro 불확실", "#6c757d", "🟡",
                        f"sector_mode={macro_mode}. 가까운 매수 zone에서만 진입",
                        None, None)

    # 5d / 10d horizon — macro-aligned baseline (raw 신호 무의미)
    if macro_mode in ("BULL", "STRONG_BULL"):
        return _verdict("🟢 매수 (강세장 5d baseline)", "#28a745", "🟢",
                        "백테스트: 5d × BULL → baseline long 50~56% win. 시스템 raw 신호는 5d에서 무의미",
                        55.8 if macro_mode == "STRONG_BULL" else 50.3,
                        120 if macro_mode == "STRONG_BULL" else 555)

    if macro_mode == "STRONG_BEAR":
        return _verdict("🟠 반등 노린 cautious long", "#f0ad4e", "🟠",
                        "백테스트: 5d × STRONG_BEAR baseline 51.8% (oversold rebound). 매수 zone 명확히 도달 시만",
                        51.8, 195)

    if macro_mode == "BEAR":
        return _verdict("🛑 약세장 — Cash 권고", "#dc3545", "🔴",
                        "백테스트: 5d × BEAR baseline 41% (음의 EV). 시스템 신호 1.3%로 의미 없음",
                        41.3, 150)

    return _verdict("🟡 CHOPPY — small long 또는 cash", "#6c757d", "🟡",
                    "백테스트: 5d × CHOPPY baseline 44.4%. 매수 zone 도달 시 small long",
                    44.4, 480)


def _verdict(label, color, emoji, rationale, win_pct, sample_n):
    return {
        "label": label, "color": color, "emoji": emoji,
        "rationale": rationale, "win_pct": win_pct, "sample_n": sample_n,
    }


class _DummyMod:
    details = {}


def _build_signal_chips(result) -> list:
    """직관 신호 카드 (정배열, RSI, Value Area, Max Pain, IV 등).

    각 카드: {label, tone, detail}.
    tone: 'bull' / 'bear' / 'neutral' / 'warn'
    """
    cur = result.current_price
    tech = result.modules.get("technical")
    ds = result.modules.get("demand_supply")
    opt = result.modules.get("options")
    if not (tech and ds and opt):
        return []
    t = tech.details
    d = ds.details
    o = opt.details

    chips = []

    # 1) MA 정배열
    ema20, ema50, sma200 = t.get("ema_20"), t.get("ema_50"), t.get("sma_200")
    if ema20 and ema50 and sma200 and cur:
        if cur > ema20 > ema50 > sma200:
            chips.append({"label": "정배열 ✓", "tone": "bull",
                          "detail": f"현재가 > EMA20 ${ema20:.2f} > EMA50 ${ema50:.2f} > SMA200 ${sma200:.2f}"})
        elif cur < ema20 < ema50 < sma200:
            chips.append({"label": "역배열 ✗", "tone": "bear",
                          "detail": f"현재가 < EMA20 < EMA50 < SMA200"})
        else:
            above = sum(1 for x in (ema20, ema50, sma200) if cur > x)
            chips.append({"label": f"MA {above}/3 위", "tone": "bull" if above >= 2 else "bear",
                          "detail": f"EMA20 ${ema20:.2f} · EMA50 ${ema50:.2f} · SMA200 ${sma200:.2f}"})

    # 2) RSI
    rsi = t.get("rsi")
    if rsi is not None and rsi == rsi:  # not NaN
        if rsi >= 70:
            chips.append({"label": f"RSI 과열 {rsi:.0f}", "tone": "warn",
                          "detail": "70 이상 — 단기 조정 가능"})
        elif rsi <= 30:
            chips.append({"label": f"RSI 과매도 {rsi:.0f}", "tone": "bull",
                          "detail": "30 이하 — 반등 가능 구간"})
        else:
            tone = "bull" if rsi > 55 else "bear" if rsi < 45 else "neutral"
            chips.append({"label": f"RSI {rsi:.0f}", "tone": tone, "detail": "중립 구간"})

    # 3) Value Area
    poc = d.get("poc")
    vah = d.get("value_area_high")
    val = d.get("value_area_low")
    if poc and vah and val and cur:
        if cur > vah:
            chips.append({"label": "VA 위 (추세 강)", "tone": "bull",
                          "detail": f"VAH ${vah:.2f} 돌파, POC ${poc:.2f} 자석"})
        elif cur < val:
            chips.append({"label": "VA 아래 (약세)", "tone": "bear",
                          "detail": f"VAL ${val:.2f} 이탈, POC ${poc:.2f}까지 반등 여지"})
        else:
            dist = (cur - poc) / cur * 100
            chips.append({"label": "VA 안 (횡보)", "tone": "neutral",
                          "detail": f"POC ${poc:.2f} ({dist:+.1f}%)"})

    # 4) Max Pain
    mp = o.get("max_pain")
    if mp and cur:
        diff = (mp - cur) / cur * 100
        if abs(diff) < 1.5:
            chips.append({"label": f"Max Pain ${mp:.0f} 근접", "tone": "neutral",
                          "detail": f"옵션 만기 시 자석 가격 ({diff:+.1f}%)"})
        elif diff > 0:
            chips.append({"label": f"Max Pain ${mp:.0f} 위 ({diff:+.1f}%)", "tone": "bull",
                          "detail": "옵션 자석이 현재가 위 — 상방 압력"})
        else:
            chips.append({"label": f"Max Pain ${mp:.0f} 아래 ({diff:+.1f}%)", "tone": "bear",
                          "detail": "옵션 자석이 현재가 아래 — 하방 압력"})

    # 5) IV Rank
    iv_rank = o.get("iv_rank")
    if iv_rank is not None:
        rank_pct = iv_rank * 100 if iv_rank <= 1 else iv_rank
        if rank_pct >= 70:
            chips.append({"label": f"IV 높음 {rank_pct:.0f}%", "tone": "warn",
                          "detail": "옵션 비쌈 — Put 헷지 비효율, Call 매도 유리"})
        elif rank_pct <= 30:
            chips.append({"label": f"IV 낮음 {rank_pct:.0f}%", "tone": "bull",
                          "detail": "옵션 쌈 — Put 헷지 유리"})

    return chips


def _render_signal_chips(chips: list):
    """Streamlit에 chip 형태로 렌더링."""
    if not chips:
        return
    parts = []
    for c in chips:
        tone = c.get("tone", "neutral")
        color = {"bull": "#28a745", "bear": "#dc3545", "warn": "#f0ad4e", "neutral": "#6c757d"}[tone]
        bg = {"bull": "rgba(40,167,69,0.15)", "bear": "rgba(220,53,69,0.15)",
              "warn": "rgba(240,173,78,0.15)", "neutral": "rgba(108,117,125,0.15)"}[tone]
        parts.append(
            f'<span title="{c.get("detail", "")}" '
            f'style="display:inline-block;padding:4px 10px;margin:2px 4px 2px 0;'
            f'border-radius:14px;background:{bg};color:{color};'
            f'border:1px solid {color}40;font-size:12px;font-weight:600;">'
            f'{c["label"]}</span>'
        )
    st.markdown("".join(parts), unsafe_allow_html=True)


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


def _zone_quality(c: dict, side: str) -> dict:
    """zone type 분류 + 검증된 bounce/reject rate (2026-05-19 백테스트 n=1055).

    Returns: {type, bounce_pct, badge, badge_color, comment}
    """
    sources = set(c.get('sources') or [])
    vp = sources & {"vol_profile", "poc", "value_area_high", "value_area_low"}
    opt = sources & {"call_oi", "put_oi", "max_pain"}
    ma = sources & {"sma_20", "sma_50", "sma_200", "ema_20", "ema_50"}
    n_sources = c.get('n_sources', 0)

    if side == "demand":
        # 검증된 bounce rate (백테스트 n=456 touched)
        if vp and not opt:
            return {"type": "VP only", "bounce_pct": 84, "badge": "★ 안정 매수 (84% 반등)",
                    "badge_color": "#28a745", "comment": "단순 매물대 — 가장 안정 진입 zone"}
        if vp and opt:
            return {"type": "VP×OPT", "bounce_pct": 65, "badge": "⭐ 큰 EV 매수 (+5.69%/trade)",
                    "badge_color": "#0d6efd", "comment": "매물대 + 옵션 — 65% bounce, 한 번 작동시 큰 PnL"}
        if opt and ma and not vp:
            return {"type": "OPT×MA", "bounce_pct": 86, "badge": "옵션+MA (86% bounce)",
                    "badge_color": "#5cb85c", "comment": "옵션 strike + MA 지지"}
        if opt and not vp and not ma:
            return {"type": "OPT only", "bounce_pct": 62, "badge": "⚠️ 옵션만 (22% break 위험)",
                    "badge_color": "#f0ad4e", "comment": "단독 옵션 strike — 사이즈 줄이거나 회피"}
        if ma and not vp and not opt:
            return {"type": "MA only", "bounce_pct": 70, "badge": "MA 지지 (sample 작음)",
                    "badge_color": "#6c757d", "comment": "이동평균만 — sample 부족"}
        if vp and opt and ma:
            return {"type": "VP×OPT×MA", "bounce_pct": 56, "badge": "multi (효과 감소)",
                    "badge_color": "#6c757d", "comment": "너무 많은 source — 효과 줄어듦"}
        return {"type": "other", "bounce_pct": 0, "badge": "etc",
                "badge_color": "#6c757d", "comment": "분류 불명"}
    else:  # supply
        # supply는 reject rate 모두 < 32% — 전량 매도 금지
        if vp and opt:
            return {"type": "VP×OPT", "bounce_pct": 32,
                    "badge": "1차 익절 (32% reject)", "badge_color": "#dc3545",
                    "comment": "매물대 + 옵션 — supply 중 가장 잘 막음. 분할 익절"}
        if vp and not opt:
            return {"type": "VP only", "bounce_pct": 22,
                    "badge": "분할 익절 (22% reject)", "badge_color": "#dc3545",
                    "comment": "강세 시장 38% 돌파 — 일부만 익절"}
        if opt and not vp:
            return {"type": "OPT", "bounce_pct": 23,
                    "badge": "약한 익절 (47% 돌파)", "badge_color": "#f0ad4e",
                    "comment": "단독 옵션 strike — 거의 돌파됨"}
        return {"type": "multi", "bounce_pct": 12,
                "badge": "거의 돌파 (~65~88%)", "badge_color": "#dc3545",
                "comment": "multi-confluence supply — 강세 시장에서 거의 돌파"}


def _render_cluster(c: dict, current_price: float, side: str):
    """confluence cluster 카드 렌더 (검증된 bounce rate 강조)."""
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
        src_labels.append(f"{label}")
    src_text = ", ".join(src_labels[:4])

    # 강도 시각화
    strength = c['confluence_strength']
    bar = "█" * min(10, int(strength / 2))

    # zone quality (검증된 bounce rate)
    q = _zone_quality(c, side)

    # n_sources 2 sweet spot 표시
    n_warn = ""
    if n_src == 1:
        n_warn = " <span style='color:#f0ad4e;font-size:10px'>· n=1 약함</span>"
    elif n_src == 2:
        n_warn = " <span style='color:#28a745;font-size:10px'>· n=2 sweet spot</span>"
    elif n_src >= 4:
        n_warn = " <span style='color:#6c757d;font-size:10px'>· n=4+ 효과↓</span>"

    bg = "rgba(0,200,0,0.10)" if side == "demand" else "rgba(200,0,0,0.10)"

    html = f"""
    <div style='padding:10px;border-radius:8px;background:{bg};margin-bottom:6px;
                border-left:4px solid {q["badge_color"]};'>
        <div style='display:flex;justify-content:space-between;align-items:baseline'>
            <div style='font-size:18px;font-weight:700'>{price_str}</div>
            <span style='font-size:11px;padding:2px 8px;border-radius:10px;
                background:{q["badge_color"]};color:white;font-weight:600'>
                {q["badge"]}
            </span>
        </div>
        <div style='font-size:12px;color:gray'>거리 {dist:+.1f}% · {n_src}개 시그널{n_warn}</div>
        <div style='font-size:11px;opacity:0.85'>{src_text}</div>
        <div style='font-size:10px;opacity:0.7;letter-spacing:-1px'>강도 {bar} ({strength:.1f})</div>
        <div style='font-size:10px;opacity:0.7;margin-top:3px;font-style:italic'>
            💡 {q["comment"]}
        </div>
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

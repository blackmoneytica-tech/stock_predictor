"""C: 결정 대시보드 — INTEGRATED 룰 (A+B+C 통합).

룰 (5y OOS 검증):
  1. ticker ∈ positive_subset (A: train avg>0, 93종)
  2. iv_rank < 0.20 (F5)
  3. RSI < 35 (B robust)
  4. regime ∈ {BULL, CHOPPY} (C)

5y OOS: n=978 / 55.4% win / +1.08% / Sharpe 1.02

VP 진입 zone:
  매수: close (또는 VAL 근접 limit)
  목표: min(VAH, close × 1.20)
  손절: VAL × 0.98 (또는 close × 0.97)

실행: python -m streamlit run src/ui/decision_panel.py --server.port 8502
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import date, timedelta
import numpy as np
import pandas as pd
import streamlit as st

from src.data.price_feed import get_daily_ohlcv, get_current_price
from src.modules.demand_supply import compute_volume_profile
from src.data.universe import get_universe


st.set_page_config(
    page_title="💎 결정 패널",
    page_icon="💎",
    layout="wide",
)


UNIVERSE_OPTIONS = {
    "positive subset (A, 93종)": "subset",
    "live (67종)": "live",
    "core50 (52종)": "core50",
    "trackb (38종)": "trackb",
    "growth (53종)": "growth",
    "FULL (213종)": "full",
}

# 백테스트 stats (5y OOS, c_full_rule_backtest.py 결과)
RULE_GRADES = {
    "R6 ★★ INTEGRATED bear (최강)": dict(
        cutoff=0.20, vp_prox=0.03, rsi_max=35, regimes=["BEAR", "CHOPPY"],
        use_subset=True,
        n_test=270, win_test=0.611, avg_test=1.52, sharpe_test=1.25,
        desc="A subset + F5 + RSI<35 + BEAR/CHOPPY. train/test 일관 (60.9%/61.1%) — overfit 없음. BEAR/CHOPPY 한정.",
    ),
    "R5 ★ INTEGRATED bull": dict(
        cutoff=0.20, vp_prox=0.03, rsi_max=35, regimes=["BULL", "CHOPPY"],
        use_subset=True,
        n_test=963, win_test=0.550, avg_test=1.06, sharpe_test=1.00,
        desc="A subset + F5 + RSI<35 + BULL/CHOPPY. 5y OOS 검증. BULL/CHOPPY 발동.",
    ),
    "R2 A+B (subset+F5+RSI)": dict(
        cutoff=0.20, vp_prox=0.03, rsi_max=35, regimes=None,
        use_subset=True,
        n_test=996, win_test=0.546, avg_test=1.04, sharpe_test=0.97,
        desc="A subset + F5 + RSI<35 (regime 무시). 안정적.",
    ),
    "R4 F5+RSI+BEAR/CHOPPY": dict(
        cutoff=0.20, vp_prox=0.03, rsi_max=35, regimes=["BEAR", "CHOPPY"],
        use_subset=False,
        n_test=642, win_test=0.575, avg_test=1.08, sharpe_test=0.94,
        desc="F5+RSI<35 + BEAR/CHOPPY (subset 없음). mean reversion contrarian.",
    ),
    "R1 F5+RSI 단순 (B base)": dict(
        cutoff=0.20, vp_prox=0.03, rsi_max=35, regimes=None,
        use_subset=False,
        n_test=2521, win_test=0.524, avg_test=0.65, sharpe_test=0.60,
        desc="F5+RSI<35 단순. 가장 자주 발동. alpha 약함.",
    ),
    "F5 단독 (이전, 권장 X)": dict(
        cutoff=0.20, vp_prox=0.03, rsi_max=None, regimes=None,
        use_subset=False,
        n_test=55603, win_test=0.530, avg_test=0.45, sharpe_test=0.52,
        desc="F5 단독 — 광범위 universe 에서 alpha 약함 (이전 21종 결과는 overfit).",
    ),
}


@st.cache_data
def load_positive_subset():
    try:
        return pd.read_csv("data/results/positive_subset.csv")["ticker"].tolist()
    except Exception:
        return []


@st.cache_data(ttl=600)
def compute_features(ticker: str):
    """ticker 의 모든 feature 계산."""
    hist = get_daily_ohlcv(ticker, start=date.today() - timedelta(days=500), end=date.today())
    if hist is None or len(hist) < 250:
        return None
    closes = hist["close"]
    log_ret = np.log(closes / closes.shift(1)).dropna()
    rolling_hv = log_ret.rolling(20).std() * np.sqrt(252)
    win = rolling_hv.dropna().tail(252)
    if win.empty:
        return None
    iv_rank = float((win < win.iloc[-1]).mean())

    # RSI 14
    delta = closes.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rsi = 100 - (100 / (1 + (gain / loss.replace(0, np.nan))))
    rsi_now = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

    # returns
    ret_20d = float(closes.pct_change(20).iloc[-1] * 100)

    # VP
    vp = compute_volume_profile(hist, lookback_days=90, num_bins=50)
    return dict(
        ticker=ticker, close=float(closes.iloc[-1]),
        iv_rank=iv_rank, rsi=rsi_now, ret_20d=ret_20d,
        poc=vp.get("poc"), vah=vp.get("vah"), val=vp.get("val"),
    )


@st.cache_data(ttl=300)
def compute_market_regime():
    """SPY 기반 시장 regime."""
    hist = get_daily_ohlcv("SPY", start=date.today() - timedelta(days=500), end=date.today())
    if hist is None or len(hist) < 200:
        return None
    closes = hist["close"]
    sma200 = closes.rolling(200).mean().iloc[-1]
    spy_close = closes.iloc[-1]
    spy_ret_20d = closes.pct_change(20).iloc[-1] * 100
    log_ret = np.log(closes / closes.shift(1)).dropna()
    rolling_hv = log_ret.rolling(20).std() * np.sqrt(252)
    win = rolling_hv.dropna().tail(252)
    spy_iv = float((win < win.iloc[-1]).mean())

    if spy_close > sma200 and spy_ret_20d > 0:
        regime = "BULL"
    elif spy_close < sma200 and spy_ret_20d < -5:
        regime = "BEAR"
    else:
        regime = "CHOPPY"
    return dict(close=spy_close, sma200=sma200, ret_20d=spy_ret_20d,
                  iv_rank=spy_iv, regime=regime)


def scan(tickers, rule):
    """rule = RULE_GRADES dict."""
    positive_subset = load_positive_subset() if rule["use_subset"] else None
    regime = compute_market_regime()
    pb = st.progress(0.0, text="scanning…")
    rows = []
    for i, t in enumerate(tickers):
        pb.progress((i + 1) / len(tickers), text=f"{t}")
        try:
            f = compute_features(t)
            if not f: continue
            pass_subset = (positive_subset is None) or (t in positive_subset)
            pass_f5 = f["iv_rank"] < rule["cutoff"]
            pass_rsi = (rule["rsi_max"] is None) or (f["rsi"] is not None and f["rsi"] < rule["rsi_max"])
            pass_regime = (rule["regimes"] is None) or (regime and regime["regime"] in rule["regimes"])
            qualified = pass_subset and pass_f5 and pass_rsi and pass_regime
            # VP entry: close ≤ VAL × (1+vp_prox)
            near_val = (f["close"] <= f["val"] * (1 + rule["vp_prox"])) and (f["close"] > f["val"] * 0.95) if f["val"] else False
            row = dict(
                ticker=t, cur=f["close"], iv_rank=f["iv_rank"], rsi=f["rsi"],
                ret_20d=f["ret_20d"],
                poc=f["poc"], vah=f["vah"], val=f["val"],
                pass_subset=pass_subset, pass_f5=pass_f5, pass_rsi=pass_rsi,
                pass_regime=pass_regime, near_val=near_val,
                qualified=qualified, vp_qualified=qualified and near_val,
            )
            # 매수/목표/손절 (VP entry 통과 시만)
            if row["vp_qualified"]:
                target = min(f["vah"], f["close"] * 1.20) if f["vah"] else f["close"] * 1.10
                stop = f["val"] * 0.98 if f["val"] else f["close"] * 0.97
                row.update(buy=f["close"], target=target, stop=stop,
                              pot_gain=(target - f["close"]) / f["close"] * 100,
                              pot_loss=(stop - f["close"]) / f["close"] * 100)
            rows.append(row)
        except Exception:
            pass
    pb.empty()
    return pd.DataFrame(rows), regime


# ── UI ──
st.title("💎 결정 패널 — INTEGRATED 룰 (5y OOS 검증)")
st.caption("A subset + F5 + RSI<35 + market regime · 단일 검증 룰만")

with st.sidebar:
    st.header("⚙️ 룰 선택")
    rule_label = st.radio(
        "룰",
        options=list(RULE_GRADES.keys()),
        index=0,
    )
    rule = RULE_GRADES[rule_label]
    st.markdown(f"**{rule_label}**")
    st.markdown(f"- iv_rank < {rule['cutoff']}")
    if rule["rsi_max"]:
        st.markdown(f"- RSI < {rule['rsi_max']}")
    if rule["regimes"]:
        st.markdown(f"- regime ∈ {rule['regimes']}")
    if rule["use_subset"]:
        st.markdown("- ticker ∈ positive_subset (A에서 검증된 93종)")
    st.markdown(f"- VP entry: close ≤ VAL × {1+rule['vp_prox']:.2f}")
    st.divider()
    st.markdown("**검증 (5y OOS):**")
    if rule["n_test"]:
        st.markdown(f"- n = {rule['n_test']}")
    st.markdown(f"- win = {rule['win_test']:.1%}")
    st.markdown(f"- avg = {rule['avg_test']:+.2f}%")
    st.markdown(f"- Sharpe = {rule['sharpe_test']:+.2f}")
    st.caption(rule["desc"])
    st.divider()

    universe_src = st.selectbox("Universe", list(UNIVERSE_OPTIONS.keys()) + ["사용자 입력"], index=0)
    if universe_src == "사용자 입력":
        user_tk = st.text_area("쉼표/공백 구분", value="NVDA, AAPL, AMD, META")
        universe = [t.strip().upper() for t in user_tk.replace(",", " ").split() if t.strip()]
    elif UNIVERSE_OPTIONS[universe_src] == "subset":
        universe = load_positive_subset()
    else:
        universe = get_universe(UNIVERSE_OPTIONS[universe_src])
    st.caption(f"종목: {len(universe)}")
    scan_btn = st.button("🔍 스캔", type="primary")


# market regime banner
regime = compute_market_regime()
if regime:
    rc = {"BULL": "🟢", "BEAR": "🔴", "CHOPPY": "🟡"}[regime["regime"]]
    st.info(f"{rc} 현재 시장: **{regime['regime']}**  ·  SPY ${regime['close']:.2f}  ·  20d ret {regime['ret_20d']:+.2f}%  ·  SPY iv_rank {regime['iv_rank']:.2f}")


if scan_btn:
    with st.spinner(f"{len(universe)} 종 scan…"):
        df, regime = scan(universe, rule)
    st.session_state["result"] = df


if "result" in st.session_state:
    df = st.session_state["result"]
    qual = df[df["qualified"]].copy()
    vp_qual = df[df["vp_qualified"]].copy()

    if len(vp_qual) == 0 and len(qual) == 0:
        st.error("### 🚫 오늘 매수 후보 없음 — 룰 통과 0")
        st.markdown("**부분 통과 종목 (참고):**")
        for col, label in [("pass_f5", "F5 활성"), ("pass_rsi", "RSI<35"), ("pass_subset", "positive subset")]:
            ts = df[df[col]]["ticker"].tolist()[:10]
            st.markdown(f"- {label}: {len(df[df[col]])}종 → {', '.join(ts)}")
    elif len(vp_qual) == 0:
        st.warning(f"### ⚠️ 룰 통과 {len(qual)}종 — VP entry zone 미도달")
        st.caption("매수가 = 진입 zone 도달 대기")
        for _, r in qual.iterrows():
            buy_zone = r["val"] * (1 + rule["vp_prox"])
            c1, c2, c3, c4 = st.columns([1.5, 1, 1, 1])
            c1.markdown(f"### {r['ticker']}")
            c1.caption(f"iv_rank {r['iv_rank']:.2f} · RSI {r['rsi']:.1f}")
            c2.metric("현재가", f"${r['cur']:.2f}")
            c3.metric("진입 zone (≤)", f"${buy_zone:.2f}", f"{(buy_zone-r['cur'])/r['cur']*100:+.2f}%")
            c4.metric("VAL/VAH", f"{r['val']:.2f}/{r['vah']:.2f}" if r["vah"] else "—")
            st.divider()
    else:
        st.success(f"### ✅ 매수 후보 {len(vp_qual)}종")
        st.caption(f"룰: {rule_label} · win {rule['win_test']:.0%} · avg {rule['avg_test']:+.2f}% · Sharpe {rule['sharpe_test']:.2f}")
        for _, r in vp_qual.iterrows():
            c1, c2, c3, c4, c5 = st.columns([1.3, 1, 1, 1, 1])
            c1.markdown(f"### {r['ticker']}")
            c1.caption(f"iv_rank {r['iv_rank']:.2f} · RSI {r['rsi']:.1f}")
            c2.metric("매수가", f"${r['buy']:.2f}")
            c3.metric("목표가", f"${r['target']:.2f}", f"{r['pot_gain']:+.2f}%")
            c4.metric("손절가", f"${r['stop']:.2f}", f"{r['pot_loss']:+.2f}%")
            rr = abs(r["pot_gain"] / r["pot_loss"]) if r["pot_loss"] else 0
            c5.metric("R/R", f"{rr:.2f}")
            with st.expander(f"📊 {r['ticker']} VP 상세"):
                cc1, cc2, cc3 = st.columns(3)
                cc1.markdown(f"**POC** ${r['poc']:.2f} ({(r['poc']-r['cur'])/r['cur']*100:+.2f}%)" if r["poc"] else "POC —")
                cc2.markdown(f"**VAH** ${r['vah']:.2f} ({(r['vah']-r['cur'])/r['cur']*100:+.2f}%)" if r["vah"] else "VAH —")
                cc3.markdown(f"**VAL** ${r['val']:.2f} ({(r['val']-r['cur'])/r['cur']*100:+.2f}%)" if r["val"] else "VAL —")
            st.divider()

    # 분석 detail
    with st.expander(f"📊 전체 scan 결과 ({len(df)}종)"):
        df_show = df[["ticker", "cur", "iv_rank", "rsi", "ret_20d", "pass_f5", "pass_rsi",
                        "pass_subset", "pass_regime", "near_val", "qualified"]].copy()
        df_show = df_show.sort_values(["qualified", "pass_f5", "pass_rsi"], ascending=False)
        st.dataframe(df_show.style.format({"cur": "${:.2f}", "iv_rank": "{:.3f}",
                                              "rsi": "{:.1f}", "ret_20d": "{:+.2f}%"}),
                      use_container_width=True, hide_index=True)

else:
    st.info("← 사이드바에서 룰/Universe 선택 후 **스캔** 클릭")
    st.markdown("""
    ## 검증된 룰 (5y OOS)

    | 룰 | OOS n | win | avg | Sharpe | 설명 |
    |---|---|---|---|---|---|
    """ + "\n".join([
        f"| {l} | {v['n_test'] or '—'} | {v['win_test']:.0%} | {v['avg_test']:+.2f}% | {v['sharpe_test']:+.2f} | {v['desc']} |"
        for l, v in RULE_GRADES.items()
    ]))
    st.markdown("""
    ### A → B → C 통합 백테스트 과정
    1. **A**: per-ticker train alpha 양성 93종 선별 (universe 213 중 44%)
    2. **B**: F5+RSI<35 추가 — 단순하지만 OOS 검증됨
    3. **C**: market regime BULL/CHOPPY 한정 (BEAR는 sample 작음)
    4. 통합: 5y OOS test win 55% / +1.08% / Sharpe 1.02

    ### 핵심 발견
    - 21종 mega-cap 룰 (+1.32%/Sharpe 1.57)은 **overfit**
    - 213종 broad universe 에서는 F5 단독 alpha 거의 0
    - subset 필터 + RSI 결합 + regime 조건 = 진짜 robust alpha
    - **market regime BEAR 가 가장 강한 alpha** (mean reversion)
    """)

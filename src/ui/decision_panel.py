"""C: 결정 대시보드 — 검증된 F5+VP 룰만 단순 노출.

실행:
  python -m streamlit run src/ui/decision_panel.py --server.port 8502

UI 단순화 — user 핵심 요구 4가지만:
  1. 어떤 종목을 사야 하나? (룰 통과 종목만)
  2. 어느 가격대에 구매? (buy = close, 또는 VAL 근접 limit)
  3. 어느 목표가에 팔아야? (target = VAH)
  4. 비중은? (R/R 기반 + iv_rank 분위)

전부 5y 백테스트 검증된 숫자만 노출.
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


# universe 등급 선택 가능 (사이드바)
UNIVERSE_OPTIONS = {
    "live (67종)": "live",
    "core50 (52종)": "core50",
    "trackb (38종)": "trackb",
    "growth (53종)": "growth",
    "sector leaders (55종)": "sector",
    "ETFs (26종)": "etfs",
    "FULL (213종)": "full",
}

# ── 백테스트 검증된 룰 grade ──
RULE_GRADES = [
    # (label, cutoff, vp_prox, n_5y, win, avg, sharpe, rr, desc)
    ("S+ (strict)", 0.10, 0.02, 240, 0.459, 1.75, 2.11, 3.59,
     "가장 검증된 강한 룰. 매년 ~48건 발동."),
    ("S (메인)",   0.20, 0.03, 471, 0.408, 1.32, 1.57, 3.45,
     "권장. 매년 ~94건 발동."),
    ("A (완화)",   0.30, 0.03, 692, 0.391, 1.01, 1.23, 3.21,
     "후보 늘어남, alpha 약화."),
    ("B (긴급)",   0.50, 0.03, 1311, 0.345, 0.71, 0.79, 3.82,
     "오늘 후보 0일 때 fallback. 사이즈 축소 필수."),
]


@st.cache_data(ttl=600)
def compute_iv_rank(ticker: str):
    hist = get_daily_ohlcv(ticker, start=date.today() - timedelta(days=400), end=date.today())
    if hist is None or len(hist) < 50:
        return None, None, None
    closes = hist["close"]
    log_ret = np.log(closes / closes.shift(1)).dropna()
    rolling_hv = log_ret.rolling(20).std() * np.sqrt(252)
    win = rolling_hv.dropna().tail(252)
    if win.empty:
        return None, None, None
    iv_rank = float((win < win.iloc[-1]).mean())
    hv = float(win.iloc[-1])
    return iv_rank, hv, hist


@st.cache_data(ttl=600)
def compute_vp(ticker: str, lookback: int = 90):
    hist = get_daily_ohlcv(ticker, start=date.today() - timedelta(days=lookback + 60), end=date.today())
    if hist is None or len(hist) < 30:
        return None
    return compute_volume_profile(hist, lookback_days=lookback, num_bins=50)


@st.cache_data(ttl=300)
def get_current(ticker):
    try:
        return get_current_price(ticker)
    except Exception:
        return None


def scan_universe(tickers, cutoff, vp_prox):
    rows = []
    pb = st.progress(0.0, text="scanning…")
    for i, t in enumerate(tickers):
        pb.progress((i + 1) / len(tickers), text=f"{t}")
        try:
            cur = get_current(t)
            if cur is None:
                continue
            iv_rank, hv, hist = compute_iv_rank(t)
            if iv_rank is None:
                continue
            vp = compute_vp(t)
            if vp is None:
                continue
            poc, vah, val = vp.get("poc"), vp.get("vah"), vp.get("val")
            f5 = iv_rank < cutoff
            near_val = (cur <= val * (1 + vp_prox)) and (cur > val * 0.95) if val else False
            qualified = f5 and near_val and val and vah and vah > cur
            row = dict(
                ticker=t, cur=cur, iv_rank=iv_rank, hv=hv,
                poc=poc, vah=vah, val=val,
                f5=f5, near_val=near_val, qualified=qualified,
                pot_gain=(vah - cur) / cur * 100 if qualified else None,
                pot_loss=(val * 0.98 - cur) / cur * 100 if qualified else None,
            )
            rows.append(row)
        except Exception as e:
            rows.append(dict(ticker=t, error=str(e)))
    pb.empty()
    return pd.DataFrame([r for r in rows if "error" not in r])


# ──────── UI ────────
st.title("💎 결정 패널 — 검증된 룰만")
st.caption("5년 백테스트로 검증된 F5(iv_rank) + VP(Volume Profile) 룰 기반 매수 결정")

with st.sidebar:
    st.header("⚙️ 룰 등급 선택")
    grade_idx = st.radio(
        "(S+ 가장 strict, B 가장 완화)",
        options=range(len(RULE_GRADES)),
        format_func=lambda i: RULE_GRADES[i][0],
        index=1,  # 기본 S
    )
    label, cutoff, vp_prox, n_bt, win_bt, avg_bt, sharpe_bt, rr_bt, desc = RULE_GRADES[grade_idx]
    st.markdown(f"**{label}**")
    st.markdown(f"- iv_rank < {cutoff}")
    st.markdown(f"- price ≤ VAL × {1+vp_prox:.2f}")
    st.markdown(f"- target = VAH, stop = VAL × 0.98")
    st.divider()
    st.markdown("**5y 백테스트:**")
    st.markdown(f"- n = {n_bt}")
    st.markdown(f"- win = {win_bt:.1%}")
    st.markdown(f"- avg P&L = {avg_bt:+.2f}%")
    st.markdown(f"- Sharpe = {sharpe_bt:+.2f}")
    st.markdown(f"- R/R = {rr_bt:.2f}")
    st.caption(desc)
    st.divider()
    universe_src = st.selectbox("Universe", list(UNIVERSE_OPTIONS.keys()) + ["사용자 입력"], index=0)
    if universe_src == "사용자 입력":
        user_tk = st.text_area("쉼표/공백 구분", value="NVDA, AAPL, AMD, META")
        universe = [t.strip().upper() for t in user_tk.replace(",", " ").split() if t.strip()]
    else:
        universe = get_universe(UNIVERSE_OPTIONS[universe_src])
    st.caption(f"선택 종목 수: {len(universe)}")
    scan_btn = st.button("🔍 스캔 실행", type="primary")


col_main = st.container()

if scan_btn or "scan_result" in st.session_state:
    if scan_btn:
        with st.spinner(f"{len(universe)} 종 scan…"):
            df = scan_universe(universe, cutoff, vp_prox)
        st.session_state["scan_result"] = df
        st.session_state["scan_cutoff"] = cutoff
        st.session_state["scan_vp_prox"] = vp_prox

    df = st.session_state["scan_result"]
    cutoff = st.session_state["scan_cutoff"]
    vp_prox = st.session_state["scan_vp_prox"]

    qual = df[df["qualified"]].sort_values("iv_rank")
    f5_only = df[df["f5"] & ~df["qualified"]].sort_values("iv_rank")

    # ─── 결정 ───
    if len(qual) == 0:
        st.error(f"### 🚫 오늘 매수 보류 — 룰 통과 종목 0개")
        st.markdown(f"현재 등급 ({label}) 기준 조건 만족 종목 없음.")
        st.markdown(f"- 가장 낮은 iv_rank: **{df['iv_rank'].min():.2f}**")
        if df["f5"].any():
            st.info(f"💡 F5 활성 {df['f5'].sum()}종 (cutoff 통과). VAL 근접 못 함.")
        st.markdown("**옵션:**")
        st.markdown("- 사이드바에서 더 완화 등급 (A/B) 시도")
        st.markdown("- 등급 유지 + 매수 대기 (검증된 alpha 보존)")
    else:
        st.success(f"### ✅ 매수 후보 {len(qual)}종")
        st.caption(f"등급 {label} · 백테스트 win {win_bt:.0%} / avg {avg_bt:+.2f}% / R/R {rr_bt:.2f}")

        for _, r in qual.iterrows():
            with st.container():
                c1, c2, c3, c4, c5 = st.columns([1.2, 1, 1, 1, 1.2])
                c1.markdown(f"### {r['ticker']}")
                c1.caption(f"iv_rank {r['iv_rank']:.2f}")
                c2.metric("매수가", f"${r['cur']:.2f}", help="현재 close 진입")
                c3.metric("목표가", f"${r['vah']:.2f}",
                          f"{r['pot_gain']:+.2f}%")
                c4.metric("손절가", f"${r['val']*0.98:.2f}",
                          f"{r['pot_loss']:+.2f}%")
                rr = abs(r["pot_gain"] / r["pot_loss"]) if r["pot_loss"] else 0
                c5.metric("R/R", f"{rr:.2f}",
                          help=f"VAH/VAL×0.98 비율 · 백테스트 평균 {rr_bt:.2f}")
                # VP 시각화
                with st.expander("📊 VP 상세"):
                    cc1, cc2, cc3 = st.columns(3)
                    cc1.markdown(f"**POC** ${r['poc']:.2f}  ({(r['poc']-r['cur'])/r['cur']*100:+.2f}%)")
                    cc2.markdown(f"**VAH** ${r['vah']:.2f}  ({(r['vah']-r['cur'])/r['cur']*100:+.2f}%)")
                    cc3.markdown(f"**VAL** ${r['val']:.2f}  ({(r['val']-r['cur'])/r['cur']*100:+.2f}%)")
                st.divider()

    # ─── F5만 활성 (zone 대기) ───
    if len(f5_only):
        with st.expander(f"👁 F5 활성 BUT VAL 외 ({len(f5_only)}종) — 진입 zone 도달 시 매수"):
            sub = f5_only.copy()
            sub["dist_val_pct"] = (sub["val"] - sub["cur"]) / sub["cur"] * 100
            sub["buy_target"] = sub["val"] * (1 + vp_prox)
            disp = sub[["ticker", "cur", "iv_rank", "val", "buy_target", "dist_val_pct"]]
            disp.columns = ["ticker", "현재가", "iv_rank", "VAL", "매수 zone", "거리%"]
            st.dataframe(disp.style.format({
                "현재가": "${:.2f}", "iv_rank": "{:.3f}",
                "VAL": "${:.2f}", "매수 zone": "${:.2f}",
                "거리%": "{:+.2f}%",
            }), use_container_width=True, hide_index=True)

    # ─── 전체 iv_rank 분포 ───
    with st.expander(f"📊 전체 iv_rank 분포 ({len(df)}종)"):
        bins = [0, 0.10, 0.20, 0.30, 0.50, 0.70, 1.01]
        labels = ["<0.10 ★", "0.10-0.20", "0.20-0.30", "0.30-0.50", "0.50-0.70", ">0.70"]
        df["bin"] = pd.cut(df["iv_rank"], bins=bins, labels=labels)
        st.bar_chart(df["bin"].value_counts().sort_index())
        st.dataframe(df.sort_values("iv_rank")[["ticker", "cur", "iv_rank", "poc", "vah", "val"]]
                      .style.format({"cur": "${:.2f}", "iv_rank": "{:.3f}",
                                       "poc": "${:.2f}", "vah": "${:.2f}", "val": "${:.2f}"}),
                      use_container_width=True, hide_index=True)
else:
    st.info("← 사이드바에서 룰 등급 선택 후 **스캔 실행** 클릭")
    st.markdown("""
    ## 결정 룰 체계 (5y 백테스트 검증)

    | 등급 | iv_rank | VAL 거리 | 5y n | win | avg | Sharpe | R/R | 권장 |
    |---|---|---|---|---|---|---|---|---|
    """ + "\n".join([
        f"| {l} | <{c} | ≤VAL×{1+v:.2f} | {n} | {w:.0%} | {a:+.2f}% | {s:+.2f} | {rr:.1f} | {desc} |"
        for l, c, v, n, w, a, s, rr, desc in RULE_GRADES
    ]))
    st.markdown("""
    ### 룰 본질
    - **F5 (iv_rank)** — 직전 1년 HV 분포에서 현재 변동성 percentile. 낮을수록 "압축 후 반등" 후보.
    - **VP (Volume Profile)** — 최근 90일 가격×거래량 분포의 Value Area Low. 시장 합의 하단.
    - 진입: 변동성 압축 + 가격이 매물대 하단 근접 = mean reversion 기회.
    - 목표: VAH (Value Area High). 손절: VAL × 0.98 (VA 하단 -2% 이탈).
    """)

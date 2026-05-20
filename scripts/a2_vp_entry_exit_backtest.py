"""A2: VP (Volume Profile) 기반 진입/목표/손절 룰 백테스트.

가설:
  F5 (iv_rank < 0.20) 활성 + 현재가가 VAL 근처 (mean reversion 진입) →
  목표 = VAH 또는 POC, 손절 = VAL - 2%.

  매수 룰: F5 ON AND price ≤ VAL * (1 + 0.02)
  exit (다음 10일 first hit):
    A. target1 = VAH  → +수익
    B. target2 = POC  → 보수 수익 (POC < VAH 인 경우)
    C. stop    = VAL × 0.98  → 손절

검증:
  1. hit rate (target1 vs target2 vs stop vs timeout)
  2. avg P&L per trade (위험 보정)
  3. risk/reward 비율
  4. 연도별 일관성
  5. ticker별 일관성
  6. F5 cutoff 변경 효과 (0.20 vs 0.30)

데이터: f5_panel_5y.parquet + rolling VP per (ticker, date)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date, timedelta
import numpy as np
import pandas as pd
from scipy import stats

from src.data.price_feed import get_daily_ohlcv
from src.modules.demand_supply import compute_volume_profile


def section(t): print(f"\n{'='*65}\n{t}\n{'='*65}")


def _build_rolling_vp(ohlcv, lookback=90):
    """각 date 에 대해 직전 lookback 일 VP (POC/VAH/VAL) rolling 계산."""
    rows = []
    closes = ohlcv["close"].values
    idx = ohlcv.index
    for i in range(len(ohlcv)):
        if i < 30:  # 최소 30일 필요
            rows.append((idx[i], np.nan, np.nan, np.nan))
            continue
        start = max(0, i - lookback)
        sub = ohlcv.iloc[start:i + 1]
        try:
            vp = compute_volume_profile(sub, lookback_days=lookback, num_bins=50)
            rows.append((idx[i], vp.get("poc"), vp.get("vah"), vp.get("val")))
        except Exception:
            rows.append((idx[i], np.nan, np.nan, np.nan))
    df = pd.DataFrame(rows, columns=["date", "poc", "vah", "val"]).set_index("date")
    return df


def build_panel():
    section("1. VP rolling 계산 + 매수 후보 추출")
    panel = pd.read_parquet("data/results/f5_panel_5y.parquet")
    print(f"  loaded {len(panel)} F5 panel rows")

    # 각 ticker 별 OHLCV 다시 fetch (forward path 위해)
    all_trades = []
    for t in sorted(panel["ticker"].unique()):
        sub = panel[panel["ticker"] == t].copy()
        sub["date"] = pd.to_datetime(sub["date"])
        if len(sub) < 100:
            continue
        try:
            start_d = (sub["date"].min() - timedelta(days=120)).date()
            end_d = (sub["date"].max() + timedelta(days=20)).date()
            hist = get_daily_ohlcv(t, start=start_d, end=end_d)
            if hist is None or len(hist) < 100:
                continue
            vp = _build_rolling_vp(hist, lookback=90)
            merged = sub.merge(vp, left_on="date", right_index=True, how="left")
            merged["ticker"] = t
            # forward path = 다음 10 봉의 high/low (target/stop hit 측정)
            hist["dh"] = hist["high"]; hist["dl"] = hist["low"]
            for h in range(1, 11):
                merged[f"fhigh_{h}"] = merged["date"].map(
                    lambda d: hist["high"].shift(-h).get(d, np.nan)
                )
                merged[f"flow_{h}"] = merged["date"].map(
                    lambda d: hist["low"].shift(-h).get(d, np.nan)
                )
            all_trades.append(merged)
            print(f"  [{t}] {len(merged)} rows, VP coverage {merged['poc'].notna().mean():.1%}")
        except Exception as e:
            print(f"  [{t}] 실패: {type(e).__name__}: {e}")
    p = pd.concat(all_trades, ignore_index=True)
    p = p.dropna(subset=["iv_rank", "poc", "vah", "val", "close"])
    print(f"\n  총 sample (VP 모두 있는 trades): {len(p)}")
    return p


def simulate_trade(row, target_choice="vah", stop_pct=0.02, max_days=10):
    """단일 trade 시뮬레이션.

    매수가 = close (entry day).
    target = VAH (또는 POC).
    stop  = VAL × (1 - stop_pct).

    Returns dict with: outcome ('target'/'stop'/'timeout'), days_to_exit, pnl_pct, exit_price.
    """
    entry = row["close"]
    target = row["vah"] if target_choice == "vah" else row["poc"]
    stop = row["val"] * (1 - stop_pct)
    if pd.isna(target) or pd.isna(stop) or target <= entry or stop >= entry:
        return None

    for h in range(1, max_days + 1):
        hi = row.get(f"fhigh_{h}")
        lo = row.get(f"flow_{h}")
        if pd.isna(hi) or pd.isna(lo):
            return None
        hit_target = hi >= target
        hit_stop = lo <= stop
        if hit_target and hit_stop:
            # 가정: stop 먼저 hit (보수적)
            return dict(outcome="stop", days=h, pnl=(stop - entry) / entry * 100)
        if hit_target:
            return dict(outcome="target", days=h, pnl=(target - entry) / entry * 100)
        if hit_stop:
            return dict(outcome="stop", days=h, pnl=(stop - entry) / entry * 100)
    # timeout — 마지막 close
    last_close = row.get(f"flow_{max_days}")
    if pd.isna(last_close):
        return None
    return dict(outcome="timeout", days=max_days, pnl=(last_close - entry) / entry * 100)


def run_rule(panel, f5_cutoff=0.20, val_proximity=0.03, target_choice="vah",
              stop_pct=0.02, label=""):
    """매수 룰 = F5 ON + price ≤ VAL × (1 + val_proximity)."""
    section(f"매수 룰 — {label}  (F5<{f5_cutoff}, price≤VAL×{1+val_proximity:.2f}, target={target_choice}, stop=VAL×{1-stop_pct:.2f})")
    mask = (
        (panel["iv_rank"] < f5_cutoff)
        & (panel["close"] <= panel["val"] * (1 + val_proximity))
        & (panel["close"] > panel["val"] * 0.95)  # 너무 멀리 떨어진 건 제외
    )
    candidates = panel[mask].copy()
    print(f"  매수 후보: {len(candidates)} (전체 panel 중 {len(candidates)/len(panel)*100:.2f}%)")
    if len(candidates) < 20:
        print(f"  sample 부족, skip")
        return None

    results = candidates.apply(lambda r: simulate_trade(r, target_choice, stop_pct), axis=1)
    res = pd.DataFrame([r for r in results if r is not None])
    if not len(res):
        print("  trade 시뮬 실패")
        return None
    out = res["outcome"].value_counts()
    print(f"\n  outcome 분포 (n={len(res)}):")
    for k in ["target", "stop", "timeout"]:
        if k in out.index:
            pct = out[k] / len(res) * 100
            print(f"    {k:<10} : {out[k]:>4} ({pct:.1f}%)")
    avg = res["pnl"].mean()
    win = (res["pnl"] > 0).mean()
    med = res["pnl"].median()
    sharpe = (avg / res["pnl"].std() * np.sqrt(252 / 5)) if res["pnl"].std() > 0 else 0
    avg_target = res[res["outcome"] == "target"]["pnl"].mean()
    avg_stop = res[res["outcome"] == "stop"]["pnl"].mean()
    avg_days = res["days"].mean()
    rr = abs(avg_target / avg_stop) if avg_stop and avg_stop != 0 else 0
    print(f"\n  P&L: avg={avg:+.2f}%  win={win:.1%}  med={med:+.2f}%  Sharpe={sharpe:+.2f}")
    print(f"  target avg={avg_target:+.2f}%  stop avg={avg_stop:+.2f}%  R/R={rr:.2f}")
    print(f"  avg days to exit: {avg_days:.1f}")
    return dict(n=len(res), avg=avg, win=win, sharpe=sharpe, rr=rr,
                target=avg_target, stop=avg_stop, days=avg_days, raw=res, cand=candidates)


def yearly_breakdown(candidates, res):
    section("연도별 P&L (룰 일관성)")
    res = res.reset_index(drop=True)
    cand = candidates.reset_index(drop=True)
    res["date"] = cand["date"]
    res["year"] = pd.to_datetime(res["date"]).dt.year
    res["ticker"] = cand["ticker"]
    print(f"  {'year':<6} {'n':<6} {'win%':<7} {'avg%':<8} {'target%':<8} {'stop%':<7}")
    for y in sorted(res["year"].unique()):
        sub = res[res["year"] == y]
        if not len(sub): continue
        n = len(sub); win = (sub["pnl"] > 0).mean()
        avg = sub["pnl"].mean()
        t = sub[sub["outcome"] == "target"]["pnl"].mean()
        s = sub[sub["outcome"] == "stop"]["pnl"].mean()
        print(f"  {y:<6} {n:<6} {win:.1%}   {avg:+.2f}%   {t:+.2f}%   {s:+.2f}%")


def per_ticker_breakdown(candidates, res):
    section("ticker별 P&L (룰 일관성)")
    res = res.reset_index(drop=True)
    cand = candidates.reset_index(drop=True)
    res["ticker"] = cand["ticker"]
    rows = []
    for t in res["ticker"].unique():
        sub = res[res["ticker"] == t]
        if len(sub) < 5: continue
        n = len(sub)
        win = (sub["pnl"] > 0).mean()
        avg = sub["pnl"].mean()
        rows.append(dict(ticker=t, n=n, win=f"{win:.1%}", avg=f"{avg:+.2f}", _sort=avg))
    df = pd.DataFrame(rows).sort_values("_sort", ascending=False).drop(columns="_sort")
    print(df.to_string(index=False))


def baseline_no_vp(panel, f5_cutoff=0.20):
    """비교군: F5 만 (VP 무시, 단순 5d hold)."""
    section(f"비교군 — F5<{f5_cutoff} 만 (VP 무시, 5d hold)")
    active = panel[panel["iv_rank"] < f5_cutoff]
    rets = active["fwd_5d_pct"].dropna()
    if not len(rets): return
    print(f"  n={len(rets)}  win={ (rets>0).mean():.1%}  avg={rets.mean():+.2f}%  med={rets.median():+.2f}%")


def main():
    p = build_panel()

    # 메인 룰 — F5<0.20 + price near VAL + target=VAH + stop=VAL×0.98
    main_result = run_rule(p, f5_cutoff=0.20, val_proximity=0.03,
                              target_choice="vah", stop_pct=0.02, label="primary")
    if main_result:
        yearly_breakdown(main_result["cand"], main_result["raw"])
        per_ticker_breakdown(main_result["cand"], main_result["raw"])

    # 변형 1 — target = POC (더 보수적)
    run_rule(p, f5_cutoff=0.20, val_proximity=0.03,
              target_choice="poc", stop_pct=0.02, label="target=POC")

    # 변형 2 — F5 cutoff 0.30
    run_rule(p, f5_cutoff=0.30, val_proximity=0.03,
              target_choice="vah", stop_pct=0.02, label="F5<0.30 완화")

    # 변형 3 — VAL proximity 더 가깝게 (1%)
    run_rule(p, f5_cutoff=0.20, val_proximity=0.01,
              target_choice="vah", stop_pct=0.02, label="VAL 1% 이내 strict")

    # 비교군
    baseline_no_vp(p, f5_cutoff=0.20)


if __name__ == "__main__":
    main()

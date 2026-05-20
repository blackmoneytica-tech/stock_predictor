"""B2: F5 cutoff 완화 효과 — alpha 유지 vs 후보 발견 trade-off.

문제: 0.20 strict 으로 오늘 후보 0. 0.30 / 0.40 / 0.50 / 0.60 어디까지 완화 가능?

검증 측면:
  - 5y backtest 에서 cutoff 0.20 vs 0.30 vs 0.40 vs 0.50 의 alpha
  - 완화 시 sample 늘어남 + alpha 약화 trade-off
  - 오늘 watchlist 38종 에 각 cutoff 적용 → 후보 수

추가:
  - VP entry 룰과 결합 (price ≤ VAL × 1.03) 도 같이 sweep
  - 권고: 적정 cutoff 선택 — sample 의미 + alpha 양호 균형
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date, timedelta
import numpy as np
import pandas as pd
from scipy import stats

from src.data.price_feed import get_daily_ohlcv, get_current_price
from src.modules.demand_supply import compute_volume_profile


def section(t): print(f"\n{'='*65}\n{t}\n{'='*65}")


def simulate_trade(row, target_choice="vah", stop_pct=0.02, max_days=10):
    entry = row["close"]
    target = row["vah"] if target_choice == "vah" else row["poc"]
    stop = row["val"] * (1 - stop_pct)
    if pd.isna(target) or pd.isna(stop) or target <= entry or stop >= entry:
        return None
    for h in range(1, max_days + 1):
        hi = row.get(f"fhigh_{h}")
        lo = row.get(f"flow_{h}")
        if pd.isna(hi) or pd.isna(lo): return None
        ht = hi >= target; hs = lo <= stop
        if ht and hs: return dict(outcome="stop", pnl=(stop-entry)/entry*100)
        if ht: return dict(outcome="target", pnl=(target-entry)/entry*100)
        if hs: return dict(outcome="stop", pnl=(stop-entry)/entry*100)
    last = row.get(f"flow_{max_days}")
    if pd.isna(last): return None
    return dict(outcome="timeout", pnl=(last-entry)/entry*100)


def main():
    # A2 와 동일 panel 재로드 (VP 이미 계산된 거)
    panel = pd.read_parquet("data/results/f5_panel_5y.parquet")

    # VP panel 디스크에서 load (a2 가 미리 저장)
    p = pd.read_parquet("data/results/a2_vp_panel.parquet")
    print(f"\nVP panel loaded: {len(p)} rows")

    # === 1. cutoff sweep — pure F5 (VP 없이) 5d hold ===
    section("1. F5 cutoff sweep — 단순 5d hold (VP 무시)")
    print(f"  {'cutoff':<10} {'n':<8} {'활성률':<8} {'win%':<8} {'avg%':<8} {'Sharpe':<7}")
    rows1 = []
    for c in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60]:
        active = p[p["iv_rank"] < c]["fwd_5d_pct"].dropna()
        n = len(active); rate = n / len(p.dropna(subset=["fwd_5d_pct"]))
        if n < 30: continue
        win = (active > 0).mean()
        avg = active.mean()
        std = active.std()
        sharpe = (avg / std * np.sqrt(252 / 5)) if std > 0 else 0
        print(f"  <{c:.2f}     {n:<8} {rate:.1%}    {win:.1%}    {avg:+.2f}%   {sharpe:+.2f}")
        rows1.append(dict(cutoff=c, n=n, win=win, avg=avg, sharpe=sharpe))

    # === 2. cutoff sweep — F5 + VP 룰 (primary) ===
    section("2. F5 cutoff sweep — VP 룰 결합 (target=VAH, stop=VAL×0.98)")
    print(f"  {'cutoff':<10} {'후보':<8} {'활성률':<8} {'win%':<8} {'avg%':<8} {'Sharpe':<7} {'R/R':<6}")
    rows2 = []
    for c in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        mask = (
            (p["iv_rank"] < c)
            & (p["close"] <= p["val"] * 1.03)
            & (p["close"] > p["val"] * 0.95)
        )
        cands = p[mask].copy()
        n_cands = len(cands)
        if n_cands < 30: continue
        rate = n_cands / len(p)
        sim = cands.apply(lambda r: simulate_trade(r), axis=1)
        res = pd.DataFrame([s for s in sim if s is not None])
        if not len(res): continue
        win = (res["pnl"] > 0).mean()
        avg = res["pnl"].mean()
        std = res["pnl"].std()
        sharpe = (avg / std * np.sqrt(252 / 5)) if std > 0 else 0
        tgt = res[res["outcome"] == "target"]["pnl"].mean()
        stp = res[res["outcome"] == "stop"]["pnl"].mean()
        rr = abs(tgt / stp) if stp else 0
        print(f"  <{c:.2f}     {n_cands:<8} {rate:.2%}    {win:.1%}    {avg:+.2f}%   {sharpe:+.2f}   {rr:.2f}")
        rows2.append(dict(cutoff=c, n=n_cands, win=win, avg=avg, sharpe=sharpe, rr=rr))

    # === 3. 오늘 watchlist 적용 — cutoff 별 후보 수 ===
    section("3. 오늘 watchlist 적용 (cutoff 별 후보 수)")
    today_df = pd.read_parquet("data/results/b1_live_scan.parquet")
    print(f"  watchlist {len(today_df)} 종 검증")
    print(f"  {'cutoff':<10} {'F5 활성':<10} {'F5+VP qualified':<18} {'후보 종목':<40}")
    for c in [0.20, 0.30, 0.40, 0.50, 0.60]:
        f5_active = today_df[today_df["iv_rank"] < c]
        n_f5 = len(f5_active)
        vp_qual = f5_active[
            (f5_active["cur"] <= f5_active["val"] * 1.03)
            & (f5_active["cur"] > f5_active["val"] * 0.95)
        ]
        n_qual = len(vp_qual)
        tickers = ", ".join(vp_qual["ticker"].tolist()[:6])
        print(f"  <{c:.2f}     {n_f5:<10} {n_qual:<18} {tickers}")

    # === 4. 권고 — cutoff 선택 가이드 ===
    section("4. 권고 — cutoff 선택")
    print("  trade-off: cutoff 완화 → 후보 늘어남 + alpha 약화")
    print()
    df2 = pd.DataFrame(rows2)
    if len(df2):
        best_sharpe = df2.iloc[df2["sharpe"].idxmax()]
        print(f"  최고 Sharpe: cutoff <{best_sharpe['cutoff']:.2f}  Sharpe={best_sharpe['sharpe']:.2f}  avg={best_sharpe['avg']:+.2f}%  n={best_sharpe['n']}")
        # find cutoff with reasonable sample AND alpha
        valid = df2[df2["n"] >= 100]
        if len(valid):
            best_pratical = valid.iloc[valid["sharpe"].idxmax()]
            print(f"  실용 추천 (n≥100): cutoff <{best_pratical['cutoff']:.2f}  Sharpe={best_pratical['sharpe']:.2f}  avg={best_pratical['avg']:+.2f}%  n={best_pratical['n']}")

    # === 5. 추가 룰 — VAL proximity 완화 ===
    section("5. VAL proximity sweep (cutoff 고정 0.30)")
    print(f"  {'proximity':<14} {'후보':<8} {'win%':<8} {'avg%':<8} {'Sharpe':<7}")
    for vp_prox in [0.01, 0.02, 0.03, 0.05, 0.08]:
        mask = (
            (p["iv_rank"] < 0.30)
            & (p["close"] <= p["val"] * (1 + vp_prox))
            & (p["close"] > p["val"] * 0.95)
        )
        cands = p[mask].copy()
        if len(cands) < 30: continue
        sim = cands.apply(lambda r: simulate_trade(r), axis=1)
        res = pd.DataFrame([s for s in sim if s is not None])
        if not len(res): continue
        win = (res["pnl"] > 0).mean()
        avg = res["pnl"].mean()
        std = res["pnl"].std()
        sharpe = (avg / std * np.sqrt(252 / 5)) if std > 0 else 0
        print(f"  VAL × {1+vp_prox:.2f}    {len(cands):<8} {win:.1%}    {avg:+.2f}%   {sharpe:+.2f}")


if __name__ == "__main__":
    main()

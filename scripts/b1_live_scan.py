"""B1: 워치 종목 live scan — F5 + VP 룰 적용해서 오늘 매수 후보 추출.

검증된 룰 (A1+A2):
  매수 조건: iv_rank < 0.20 AND 현재가 ≤ VAL × 1.03
  매수가:    현재 close
  목표가:    VAH
  손절가:    VAL × 0.98
  보유:     최대 10일

5y backtest:
  win 40.8% / avg +1.32% / Sharpe 1.57 / R/R 3.45

Universe: 메모리 기반 워치 종목 + 백테스트 universe.
환경변수 WATCHLIST_URL + AUTH_TOKEN 설정 시 D1 sync 가능.
"""
import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date, timedelta
import json
import urllib.request
import numpy as np
import pandas as pd

from src.data.price_feed import get_daily_ohlcv, get_current_price
from src.modules.demand_supply import compute_volume_profile


# ── 워치 종목 (메모리 + 백테스트 + 최근 push) ──
WATCHLIST = [
    # Track B HIGH conviction (2026-05-18 push)
    "BWXT", "CRDO", "ALAB", "STRL",
    # Track B Top 12 (Phase 2-3)
    "ONTO", "RKLB", "MYRG", "COHR", "AEIS", "KTOS", "MOD", "VCEL",
    "AMKR", "MKSI", "STVN",
    # Sweet spot mentions
    "PKE", "GLBE", "KLIC", "AAOI",
    # Recent Q1 validation
    "HBM", "GPOR", "RELY",
    # Yesterday's re-analysis
    "LQDA", "SMR", "ASTS", "CRCL", "MSTR",
    # Mega-caps + F5 backtest universe
    "NVDA", "AAPL", "AMD", "META", "MSFT", "GOOGL",
    "AMZN", "TSLA", "AVGO", "NFLX", "PLTR",
]


def fetch_live_watchlist():
    """trade-journal D1 에서 워치 sync (AUTH_TOKEN 필요)."""
    url = os.environ.get("WATCHLIST_URL", "https://trade-journal-1dv.pages.dev/api/kv/tj_w1")
    token = os.environ.get("AUTH_TOKEN") or os.environ.get("TJ_AUTH_TOKEN")
    if not token:
        return None
    try:
        req = urllib.request.Request(url, headers={"X-Auth-Token": token})
        with urllib.request.urlopen(req, timeout=15) as resp:
            j = json.loads(resp.read())
        items = json.loads(j.get("value", "[]"))
        tickers = [w["ticker"] for w in items if (w.get("status") or "active") == "active"]
        print(f"  ✓ D1 워치 sync: {len(tickers)} tickers")
        return tickers
    except Exception as e:
        print(f"  ✗ D1 sync 실패 ({type(e).__name__}: {e}) → 메모리 워치 사용")
        return None


def compute_iv_rank(hist):
    closes = hist["close"]
    log_ret = np.log(closes / closes.shift(1)).dropna()
    if len(log_ret) < 50:
        return None
    rolling_hv = log_ret.rolling(20).std() * np.sqrt(252)
    win = rolling_hv.dropna().tail(252)
    if win.empty:
        return None
    return float((win < win.iloc[-1]).mean())


def scan_ticker(t):
    """단일 종목 진단."""
    try:
        cur = get_current_price(t)
    except Exception as e:
        return dict(ticker=t, error=f"price: {e}")

    try:
        hist = get_daily_ohlcv(t, start=date.today() - timedelta(days=400), end=date.today())
        if hist is None or len(hist) < 50:
            return dict(ticker=t, cur=cur, error="hist insufficient")
    except Exception as e:
        return dict(ticker=t, cur=cur, error=f"hist: {e}")

    iv_rank = compute_iv_rank(hist)
    if iv_rank is None:
        return dict(ticker=t, cur=cur, error="iv_rank fail")

    try:
        vp = compute_volume_profile(hist, lookback_days=90, num_bins=50)
        poc, vah, val = vp.get("poc"), vp.get("vah"), vp.get("val")
    except Exception as e:
        return dict(ticker=t, cur=cur, iv_rank=iv_rank, error=f"vp: {e}")

    # 매수 룰 평가
    f5 = iv_rank < 0.20
    near_val = (cur <= val * 1.03) and (cur > val * 0.95) if val else False

    qualified = f5 and near_val

    return dict(
        ticker=t, cur=cur, iv_rank=iv_rank, poc=poc, vah=vah, val=val,
        f5=f5, near_val=near_val, qualified=qualified,
        dist_vah=(vah - cur) / cur * 100 if vah else None,
        dist_val=(val - cur) / cur * 100 if val else None,
        dist_poc=(poc - cur) / cur * 100 if poc else None,
        # 매수가/목표가/손절가
        buy=cur if qualified else None,
        target=vah if qualified else None,
        stop=val * 0.98 if qualified else None,
        # 예상 R/R
        pot_gain=(vah - cur) / cur * 100 if qualified and vah else None,
        pot_loss=(val * 0.98 - cur) / cur * 100 if qualified and val else None,
    )


def main():
    print(f"B1: live scan — {date.today()}")
    print(f"룰 (5y 검증): iv_rank<0.20 + price≤VAL×1.03 → buy=close target=VAH stop=VAL×0.98")
    print(f"백테스트: win 40.8% / avg +1.32% / Sharpe 1.57 / R/R 3.45\n")

    live = fetch_live_watchlist()
    universe = live if live else WATCHLIST
    print(f"universe: {len(universe)} tickers ({'D1 sync' if live else '메모리 fallback'})\n")

    rows = []
    for i, t in enumerate(universe):
        print(f"[{i+1}/{len(universe)}] {t}...", end=" ", flush=True)
        r = scan_ticker(t)
        rows.append(r)
        if "error" in r:
            print(f"ERR {r['error']}")
        else:
            mark = "★ 매수" if r["qualified"] else ("F5" if r["f5"] else "")
            print(f"${r['cur']:.2f}  iv_rank={r['iv_rank']:.2f}  {mark}")

    df = pd.DataFrame([r for r in rows if "error" not in r])
    df = df.sort_values(["qualified", "iv_rank"], ascending=[False, True]).reset_index(drop=True)

    # 결과 요약
    print(f"\n{'='*70}")
    print(f"매수 후보 (qualified): {df['qualified'].sum()}/{len(df)}")
    print(f"F5 활성: {df['f5'].sum()}/{len(df)}")
    print(f"{'='*70}")

    # 매수 후보 상세
    qual = df[df["qualified"]]
    if len(qual):
        print(f"\n💎 매수 후보 ({len(qual)}개):")
        print(f"{'ticker':<8} {'cur':<9} {'iv_rank':<9} {'매수가':<9} {'목표가':<10} {'손절가':<9} {'R/R':<7} {'pot+':<7} {'pot-':<7}")
        for _, r in qual.iterrows():
            rr = abs(r["pot_gain"] / r["pot_loss"]) if r["pot_loss"] else 0
            print(f"{r['ticker']:<8} ${r['cur']:<8.2f} {r['iv_rank']:<9.3f} "
                  f"${r['buy']:<8.2f} ${r['target']:<9.2f} ${r['stop']:<8.2f} "
                  f"{rr:<7.2f} {r['pot_gain']:+.2f}%  {r['pot_loss']:+.2f}%")
    else:
        print("\n매수 후보 없음 — 오늘 매수 보류 (검증 룰 통과 종목 0)")

    # F5 활성이지만 VAL 멀리 (관심 대상)
    f5_only = df[df["f5"] & ~df["qualified"]]
    if len(f5_only):
        print(f"\n👁 F5 활성 but VAL 외 ({len(f5_only)}개 — 진입 zone 도달 시 매수):")
        print(f"{'ticker':<8} {'cur':<9} {'iv_rank':<9} {'VAL':<9} {'dist_val':<10}")
        for _, r in f5_only.head(10).iterrows():
            print(f"{r['ticker']:<8} ${r['cur']:<8.2f} {r['iv_rank']:<9.3f} ${r['val']:<8.2f} {r['dist_val']:+.2f}%")

    # 전체 iv_rank 분포 (관망 종목)
    print(f"\n📊 iv_rank 분포 (전체):")
    bins = [0, 0.20, 0.30, 0.40, 0.50, 0.70, 1.0]
    labels = ["<0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50", "0.50-0.70", "0.70-1.00"]
    df["iv_bin"] = pd.cut(df["iv_rank"], bins=bins, labels=labels)
    print(df["iv_bin"].value_counts().sort_index().to_string())

    # save
    df.to_parquet("data/results/b1_live_scan.parquet")
    print(f"\n  saved → data/results/b1_live_scan.parquet")


if __name__ == "__main__":
    main()

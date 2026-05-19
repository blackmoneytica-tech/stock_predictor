"""매물대 × 옵션 strike confluence 검증.

가설: 매물대(Volume Profile/POC) + 옵션 strike(Call OI/Put OI/Max Pain)이
같은 가격대에 겹친 zone에 가격이 도달했을 때, 단독 source zone보다
훨씬 높은 bounce/reject rate를 가질 것이다.

설계:
- 각 (ticker, snapshot)마다 system.analyze → confluence_zones 추출
- 각 zone에 이후 5/10 영업일 동안:
  * touch 여부 (가격이 zone low~high 사이 진입)
  * bounce (demand zone에서 회복: close > zone_high)
  * reject (supply zone에서 매도: close < zone_low)
  * break (zone 반대편 ±2% 돌파)
- zone source 분류:
  * VP only: vol_profile, poc, value_area_*
  * OPT only: call_oi, put_oi, max_pain
  * VP+OPT: 둘 다 (진짜 confluence)
  * VP+OPT+MA: 매물대 × 옵션 × 이동평균 (multi-confluence)
- strength 분위수별 bounce rate
- dist_pct (현재가에서 거리) 분위수별
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd

from ..data.insider import get_insider_activity
from ..data.price_feed import get_daily_ohlcv
from ..system import StockPredictionSystem
from .walk_forward import build_data_at


VP_SOURCES = {"vol_profile", "poc", "value_area_high", "value_area_low"}
OPT_SOURCES = {"call_oi", "put_oi", "max_pain"}
MA_SOURCES = {"sma_20", "sma_50", "sma_200", "ema_20", "ema_50"}
SWING_SOURCES = {"swing_high_20d", "swing_low_20d", "swing_high_60d", "swing_low_60d"}


def classify_zone(sources: List[str]) -> dict:
    """zone source list → flags."""
    s = set(sources)
    return {
        "has_vp": bool(s & VP_SOURCES),
        "has_opt": bool(s & OPT_SOURCES),
        "has_ma": bool(s & MA_SOURCES),
        "has_swing": bool(s & SWING_SOURCES),
    }


def zone_type_label(flags: dict) -> str:
    """4가지 type 분류."""
    has_vp = flags["has_vp"]
    has_opt = flags["has_opt"]
    has_ma = flags["has_ma"]
    if has_vp and has_opt and has_ma:
        return "VP×OPT×MA (multi)"
    if has_vp and has_opt:
        return "VP×OPT"
    if has_vp:
        return "VP only"
    if has_opt and has_ma:
        return "OPT×MA"
    if has_opt:
        return "OPT only"
    if has_ma:
        return "MA only"
    return "other"


def collect_zone_events(
    tickers: List[str],
    snapshot_dates: List[date],
    horizon_lookahead: int = 5,
    verbose: bool = True,
) -> pd.DataFrame:
    """각 (ticker, snap)마다 zone 추출 → 이후 N일 outcome."""
    system = StockPredictionSystem()
    rows = []

    if verbose:
        print("[prefetch] ETFs...", flush=True)
    earliest_etf = min(snapshot_dates) - timedelta(days=400)
    latest_etf = max(snapshot_dates) + timedelta(days=horizon_lookahead + 10)
    for etf in ('XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY',
                'XLP', 'XLU', 'XLRE', 'XLB', 'XLC',
                'SPY', 'QQQ', 'IWM', '^VIX', 'HYG', 'LQD'):
        try:
            get_daily_ohlcv(etf, earliest_etf, latest_etf)
        except Exception:
            pass
    try:
        from ..data.sector_macro import compute_macro_breadth_at
        for snap in snapshot_dates:
            try:
                compute_macro_breadth_at(snap)
            except Exception:
                pass
    except ImportError:
        pass
    if verbose:
        print("[prefetch] done", flush=True)

    for ticker in tickers:
        if verbose:
            print(f"[{ticker}]", flush=True)
        try:
            earliest = min(snapshot_dates) - timedelta(days=400)
            latest = max(snapshot_dates) + timedelta(days=horizon_lookahead + 10)
            full = get_daily_ohlcv(ticker, earliest, latest)
            if full.empty:
                continue
            insider_cache = get_insider_activity(ticker, months_back=12)
        except Exception as e:
            if verbose:
                print(f"  SKIP fetch: {e}", flush=True)
            continue

        for snap in snapshot_dates:
            ts_snap = pd.Timestamp(snap)
            at_or_before = full[full.index <= ts_snap]
            after = full[full.index > ts_snap]
            if at_or_before.empty or len(after) < horizon_lookahead:
                continue

            try:
                data = build_data_at(
                    ticker, snap,
                    horizon_days=horizon_lookahead,
                    use_macro=True,
                    insider_cache=insider_cache,
                )
                result = system.analyze(
                    ticker, horizon_days=horizon_lookahead, data=data,
                )
            except Exception as e:
                if verbose:
                    print(f"  {snap} fail: {e}", flush=True)
                continue

            cur = result.current_price
            conf_zones = result.confluence_zones or {}
            all_clusters = (
                list(conf_zones.get("demand") or [])
                + list(conf_zones.get("supply") or [])
            )
            if not all_clusters:
                continue

            future = after.head(horizon_lookahead)
            macro_mode = (data.get("macro_breadth") or {}).get("mode", "?")

            for c in all_clusters:
                low = float(c.get("low", 0))
                high = float(c.get("high", 0))
                if low <= 0 or high <= 0:
                    continue
                if high < low:
                    low, high = high, low
                center = (low + high) / 2
                dist_pct = (center - cur) / cur * 100
                side = "demand" if center < cur else "supply"
                sources = list(c.get("sources") or [])
                n_sources = int(c.get("n_sources") or 0)
                strength = float(c.get("confluence_strength") or 0)
                flags = classify_zone(sources)
                ztype = zone_type_label(flags)

                # touch: 미래 5d에 zone과 가격대 겹침
                touched = bool(
                    ((future["low"] <= high) & (future["high"] >= low)).any()
                )

                # zone 내부 진입 깊이 (close 기준 가장 가까이 도달)
                last_close = float(future["close"].iloc[-1])
                max_high = float(future["high"].max())
                min_low = float(future["low"].min())

                # bounce / reject / break 로직
                if side == "demand":
                    # demand zone (현재가 아래): touch 후 zone_high 위로 회복 = bounce
                    bounce = touched and last_close > high
                    # 더 떨어져서 zone_low * 0.98 아래 = break (down break)
                    broke = min_low < low * 0.98 and last_close < low
                    outcome = "bounce" if bounce else "broke" if broke else "neutral"
                else:
                    # supply zone (현재가 위): touch 후 zone_low 아래로 = reject (매도세)
                    reject = touched and last_close < low
                    # 더 올라가서 zone_high * 1.02 위로 = break (up break)
                    broke = max_high > high * 1.02 and last_close > high
                    outcome = "reject" if reject else "broke" if broke else "neutral"

                rows.append({
                    "ticker": ticker,
                    "as_of": snap.isoformat(),
                    "side": side,
                    "zone_low": round(low, 2),
                    "zone_high": round(high, 2),
                    "zone_center": round(center, 2),
                    "dist_pct": round(dist_pct, 2),
                    "n_sources": n_sources,
                    "strength": round(strength, 2),
                    "zone_type": ztype,
                    "has_vp": flags["has_vp"],
                    "has_opt": flags["has_opt"],
                    "has_ma": flags["has_ma"],
                    "has_swing": flags["has_swing"],
                    "touched_5d": touched,
                    "outcome": outcome,
                    "last_close": round(last_close, 2),
                    "last_ret_pct": round((last_close - cur) / cur * 100, 2),
                    "macro_mode": macro_mode,
                    "n_zone_sources_list": ",".join(sorted(set(sources))),
                })

    return pd.DataFrame(rows)


def analyze_zone_results(df: pd.DataFrame):
    """zone type × side × strength × dist_pct별 outcome 분포."""
    if df.empty:
        print("결과 없음")
        return

    print(f"\n=== 전체: {len(df)} zone events ({df['ticker'].nunique()} tickers) ===")
    print(f"side 분포: {df['side'].value_counts().to_dict()}")
    print(f"touched_5d rate: {df['touched_5d'].mean():.1%}")
    print()

    # 분석은 touched zones에 한정 (실제로 가격이 가까이 갔던 것만)
    t = df[df['touched_5d']].copy()
    if t.empty:
        print("touched zone 없음")
        return
    print(f"=== touched zones만 (n={len(t)}) ===\n")

    # --- 1) zone type별 outcome (demand) ---
    print("--- 1) DEMAND zone × type별 outcome ---")
    d = t[t['side'] == 'demand']
    g = d.groupby('zone_type').agg(
        n=('outcome', 'size'),
        bounce_rate=('outcome', lambda x: (x == 'bounce').mean()),
        break_rate=('outcome', lambda x: (x == 'broke').mean()),
        neutral_rate=('outcome', lambda x: (x == 'neutral').mean()),
        mean_ret_5d=('last_ret_pct', 'mean'),
    ).sort_values('n', ascending=False)
    for c in ['bounce_rate', 'break_rate', 'neutral_rate']:
        g[c] = g[c].apply(lambda x: f"{x:.1%}")
    g['mean_ret_5d'] = g['mean_ret_5d'].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())
    print()

    # --- 2) zone type별 outcome (supply) ---
    print("--- 2) SUPPLY zone × type별 outcome ---")
    s = t[t['side'] == 'supply']
    g = s.groupby('zone_type').agg(
        n=('outcome', 'size'),
        reject_rate=('outcome', lambda x: (x == 'reject').mean()),
        break_rate=('outcome', lambda x: (x == 'broke').mean()),
        neutral_rate=('outcome', lambda x: (x == 'neutral').mean()),
        mean_ret_5d=('last_ret_pct', 'mean'),
    ).sort_values('n', ascending=False)
    for c in ['reject_rate', 'break_rate', 'neutral_rate']:
        g[c] = g[c].apply(lambda x: f"{x:.1%}")
    g['mean_ret_5d'] = g['mean_ret_5d'].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())
    print()

    # --- 3) confluence 직접 비교 (VP only vs VP×OPT vs VP×OPT×MA) ---
    print("--- 3) Confluence 효과: VP only vs VP×OPT (demand만) ---")
    for ztype in ["VP only", "OPT only", "VP×OPT", "VP×OPT×MA (multi)", "MA only", "OPT×MA"]:
        sub = d[d['zone_type'] == ztype]
        if sub.empty:
            continue
        win = (sub['outcome'] == 'bounce').mean()
        lose = (sub['outcome'] == 'broke').mean()
        n = len(sub)
        print(f"  {ztype:<25s} n={n:>4d} bounce={win:.1%} break={lose:.1%} mean_ret={sub['last_ret_pct'].mean():+.2f}%")
    print()

    # --- 4) n_sources별 demand bounce rate ---
    print("--- 4) n_sources별 demand bounce rate ---")
    d['n_src_bucket'] = pd.cut(d['n_sources'], bins=[0, 1, 2, 3, 99], labels=["1", "2", "3", "4+"])
    g = d.groupby('n_src_bucket', observed=False).agg(
        n=('outcome', 'size'),
        bounce_rate=('outcome', lambda x: (x == 'bounce').mean()),
        break_rate=('outcome', lambda x: (x == 'broke').mean()),
        mean_ret_5d=('last_ret_pct', 'mean'),
    )
    for c in ['bounce_rate', 'break_rate']:
        g[c] = g[c].apply(lambda x: f"{x:.1%}")
    g['mean_ret_5d'] = g['mean_ret_5d'].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())
    print()

    # --- 5) strength 분위수별 demand bounce rate ---
    print("--- 5) strength 분위수별 demand bounce rate ---")
    if len(d) >= 10:
        d['strength_bucket'] = pd.qcut(d['strength'], q=4, duplicates='drop',
                                       labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
        g = d.groupby('strength_bucket', observed=False).agg(
            n=('outcome', 'size'),
            bounce_rate=('outcome', lambda x: (x == 'bounce').mean()),
            break_rate=('outcome', lambda x: (x == 'broke').mean()),
            mean_ret_5d=('last_ret_pct', 'mean'),
            mean_strength=('strength', 'mean'),
        )
        for c in ['bounce_rate', 'break_rate']:
            g[c] = g[c].apply(lambda x: f"{x:.1%}")
        g['mean_ret_5d'] = g['mean_ret_5d'].apply(lambda x: f"{x:+.2f}%")
        print(g.to_string())
    print()

    # --- 6) dist_pct × demand bounce rate (zone이 현재가에서 얼마나 떨어졌나) ---
    print("--- 6) demand zone dist_pct별 bounce rate ---")
    d['dist_bucket'] = pd.cut(
        d['dist_pct'].abs(),
        bins=[0, 2, 5, 10, 100],
        labels=["≤2%", "2~5%", "5~10%", ">10%"],
    )
    g = d.groupby('dist_bucket', observed=False).agg(
        n=('outcome', 'size'),
        bounce_rate=('outcome', lambda x: (x == 'bounce').mean()),
        break_rate=('outcome', lambda x: (x == 'broke').mean()),
        mean_ret_5d=('last_ret_pct', 'mean'),
    )
    for c in ['bounce_rate', 'break_rate']:
        g[c] = g[c].apply(lambda x: f"{x:.1%}")
    g['mean_ret_5d'] = g['mean_ret_5d'].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())
    print()

    # --- 7) 종합 효과: VP만 vs VP×OPT (직접 비교) demand & supply ---
    print("--- 7) Pairwise comparison (VP only vs VP×OPT) ---")
    for side_name, subset in [("DEMAND", d), ("SUPPLY", s)]:
        vp = subset[subset['zone_type'] == 'VP only']
        vpopt = subset[subset['zone_type'].isin(['VP×OPT', 'VP×OPT×MA (multi)'])]
        if len(vp) == 0 or len(vpopt) == 0:
            continue
        target = 'bounce' if side_name == 'DEMAND' else 'reject'
        vp_win = (vp['outcome'] == target).mean()
        vpopt_win = (vpopt['outcome'] == target).mean()
        delta = vpopt_win - vp_win
        print(f"  {side_name}: VP only {target}={vp_win:.1%} (n={len(vp)}) vs "
              f"VP×OPT {target}={vpopt_win:.1%} (n={len(vpopt)}) → Δ {delta:+.1%}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers",
        default="NVDA,AMD,TSLA,AAPL,MSFT,GOOGL,META,AMZN,CRCL,MSTR,COIN,HOOD,NFLX,PLTR,SMCI",
    )
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--end", default=None)
    parser.add_argument("--lookahead", type=int, default=5)
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    if args.end:
        end_date = date.fromisoformat(args.end)
    else:
        end_date = date.today() - timedelta(days=args.lookahead + 5)

    snaps = []
    d = end_date
    while len(snaps) < args.days:
        if d.weekday() < 5:
            snaps.append(d)
        d -= timedelta(days=1)
    snaps.sort()

    print(f"tickers ({len(tickers)}): {tickers}", flush=True)
    print(f"snapshots: {snaps[0]} ~ {snaps[-1]} ({len(snaps)} BD)", flush=True)
    print(f"lookahead: {args.lookahead}d", flush=True)

    df = collect_zone_events(tickers, snaps, horizon_lookahead=args.lookahead)

    if df.empty:
        print("결과 없음")
        return

    from pathlib import Path
    out = Path(__file__).resolve().parents[2] / "data" / "results" / "zone_confluence.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n[saved] {out} ({len(df)} zone events)\n", flush=True)

    analyze_zone_results(df)


if __name__ == "__main__":
    main()

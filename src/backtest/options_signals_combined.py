"""Options 신호 종합 백테스트 — Max Pain + Call/Put wall + vol/OI + IV + 뉴스.

가설: 단독 max_pain은 약하지만, 다음 조합이 강한 신호일 수 있다.
  - Max Pain + Call Wall 근접 (위) + vol/OI 정상 → 상방 막힘 (short)
  - Max Pain + Put Wall 근접 (아래) + vol/OI 정상 → 하방 지지 (long)
  - Max Pain 자석 + IV 안정 (catalyst 없음) → 진짜 수렴
  - Max Pain 무력 + IV 급증 OR 뉴스 strong → catalyst가 자석 깸

수집 차원 (시스템 _fetch_data로 이미 fetch됨):
  - max_pain, options.implied_move, iv_rank, hv_iv_ratio
  - call_wall_strike = max(call_oi 기준) → 거리 %
  - put_wall_strike = max(put_oi 기준) → 거리 %
  - vol_oi_ratio = sum(call_volume + put_volume) / sum(oi)
  - news_sentiment_score (-10 ~ +10)
  - unusual_options_score / direction
  - 그리고 actual N일 return → win/loss

3, 5, 10일 horizon 모두 측정.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def extract_walls(options_chain: dict, target_exp: str) -> dict:
    """Call/Put wall + vol/OI ratio."""
    strikes_data = options_chain.get(target_exp) or {}
    if not strikes_data:
        return {}

    # 최대 call_oi strike
    call_oi_map = {s: d.get('call_oi', 0) or 0 for s, d in strikes_data.items()}
    put_oi_map = {s: d.get('put_oi', 0) or 0 for s, d in strikes_data.items()}
    call_wall = max(call_oi_map, key=call_oi_map.get) if call_oi_map else 0
    put_wall = max(put_oi_map, key=put_oi_map.get) if put_oi_map else 0
    call_wall_oi = call_oi_map.get(call_wall, 0)
    put_wall_oi = put_oi_map.get(put_wall, 0)

    # 총 volume / total OI
    total_vol = sum((d.get('call_volume', 0) or 0) + (d.get('put_volume', 0) or 0)
                    for d in strikes_data.values())
    total_oi = sum(call_oi_map.values()) + sum(put_oi_map.values())
    vol_oi_ratio = total_vol / max(total_oi, 1)

    # ATM IV
    return {
        "call_wall": float(call_wall) if call_wall else 0.0,
        "put_wall": float(put_wall) if put_wall else 0.0,
        "call_wall_oi": int(call_wall_oi),
        "put_wall_oi": int(put_wall_oi),
        "vol_oi_ratio": round(vol_oi_ratio, 3),
        "total_oi": int(total_oi),
    }


def collect_events(
    tickers: List[str], snapshot_dates: List[date],
    horizons_lookahead: List[int] = [3, 5, 10],
    verbose: bool = True,
) -> pd.DataFrame:
    from ..data.insider import get_insider_activity
    from ..data.price_feed import get_daily_ohlcv
    from ..system import StockPredictionSystem
    from .walk_forward import build_data_at

    system = StockPredictionSystem()
    rows = []

    if verbose:
        print("[prefetch] ETFs...", flush=True)
    earliest = min(snapshot_dates) - timedelta(days=400)
    latest = max(snapshot_dates) + timedelta(days=max(horizons_lookahead) + 10)
    for etf in ('SPY','QQQ','IWM','^VIX'):
        try:
            get_daily_ohlcv(etf, earliest, latest)
        except Exception:
            pass

    for ticker in tickers:
        if verbose:
            print(f"[{ticker}]", flush=True)
        try:
            full = get_daily_ohlcv(ticker, earliest, latest)
            if full.empty:
                continue
            insider = get_insider_activity(ticker, months_back=12)
        except Exception:
            continue

        for snap in snapshot_dates:
            ts_snap = pd.Timestamp(snap)
            at_or_before = full[full.index <= ts_snap]
            after = full[full.index > ts_snap]
            if at_or_before.empty or len(after) < max(horizons_lookahead):
                continue
            cur = float(full.loc[at_or_before.index[-1], "close"])

            try:
                data = build_data_at(
                    ticker, snap, horizon_days=5, use_macro=False,
                    insider_cache=insider,
                )
                result = system.analyze(ticker, horizon_days=5, data=data)
            except Exception:
                continue

            opt = result.modules['options'].details
            max_pain = opt.get('max_pain') or 0
            implied_move = opt.get('implied_move') or 0
            iv = opt.get('iv') or 0
            iv_rank = opt.get('iv_rank') or 0
            hv_iv = opt.get('hv_iv_ratio') or 1
            pc_ratio = opt.get('put_call_ratio') or 0
            dte = opt.get('days_to_expiration') or 0
            if not max_pain or not dte:
                continue

            # walls + vol/OI
            walls = extract_walls(
                data.get('options_chain', {}),
                data.get('target_expiration', ''),
            )

            news_score = data.get('news_sentiment_score', 0)
            news_n = data.get('news_sentiment_n', 0)
            unusual_score = data.get('unusual_options_score', 0)
            unusual_dir = data.get('unusual_options_direction', 'neutral')

            row = {
                "ticker": ticker,
                "as_of": snap.isoformat(),
                "cur": round(cur, 2),
                "max_pain": round(max_pain, 2),
                "mp_dist_pct": round((cur - max_pain) / cur * 100, 2),
                "dte": int(dte),
                "implied_move_pct": round(implied_move / cur * 100, 2) if cur else 0,
                "iv": round(iv, 4),
                "iv_rank": round(iv_rank, 3),
                "hv_iv": round(hv_iv, 3),
                "pc_ratio": round(pc_ratio, 3),
                "call_wall": walls.get("call_wall", 0),
                "put_wall": walls.get("put_wall", 0),
                "call_wall_dist_pct": round((walls.get("call_wall", 0) - cur) / cur * 100, 2) if walls.get("call_wall") else 0,
                "put_wall_dist_pct": round((walls.get("put_wall", 0) - cur) / cur * 100, 2) if walls.get("put_wall") else 0,
                "vol_oi_ratio": walls.get("vol_oi_ratio", 0),
                "total_oi": walls.get("total_oi", 0),
                "news_score": round(news_score, 2),
                "news_n": int(news_n),
                "unusual_score": round(unusual_score, 3),
                "unusual_dir": unusual_dir,
                "composite_score": round(result.composite_score, 3),
                "ev_pct_5d": round(
                    (result.expected_value - cur) / cur * 100, 3
                ),
            }
            # actual returns
            for h in horizons_lookahead:
                future = full[full.index > ts_snap].head(h)
                if len(future) >= h:
                    fp = float(future["close"].iloc[h - 1])
                    row[f"actual_ret_{h}d"] = round((fp - cur) / cur * 100, 3)
                else:
                    row[f"actual_ret_{h}d"] = None

            rows.append(row)

    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame, horizon: int = 5):
    actual_col = f"actual_ret_{horizon}d"
    s = df.dropna(subset=[actual_col]).copy()
    if s.empty:
        print("no data")
        return
    n_total = len(s)
    base_win = (s[actual_col] > 0).mean()
    base_avg = s[actual_col].mean()
    print(f"\n=== Horizon {horizon}d — Baseline (n={n_total}) ===")
    print(f"  Baseline win: {base_win:.1%}, avg: {base_avg:+.2f}%\n")

    # ─── 1. 단독 max_pain (control) ───
    print("--- 1. 단독 Max Pain 거리 (control) ---")
    s['mp_bucket'] = pd.cut(
        s['mp_dist_pct'],
        bins=[-100, -5, -2, 2, 5, 100],
        labels=["<-5%", "-5~-2%", "±2%", "+2~+5%", ">+5%"],
    )
    g = s.groupby('mp_bucket', observed=False).agg(
        n=(actual_col, 'size'),
        win=(actual_col, lambda x: (x > 0).mean()),
        avg=(actual_col, 'mean'),
    )
    for c in ['win']:
        g[c] = g[c].apply(lambda x: f"{x:.1%}")
    g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())
    print()

    # ─── 2. Put Wall 근접 (지지) — long 적기? ───
    print("--- 2. Put Wall (현재가 아래 큰 put OI) 근접 → 지지 효과? ---")
    s['pw_dist'] = s['put_wall_dist_pct']
    s['pw_bucket'] = pd.cut(
        s['pw_dist'],
        bins=[-100, -10, -5, -2, 0, 100],
        labels=["<-10%", "-10~-5%", "-5~-2%", "-2~0%", ">0%(위)"],
    )
    g = s.groupby('pw_bucket', observed=False).agg(
        n=(actual_col, 'size'),
        win=(actual_col, lambda x: (x > 0).mean()),
        avg=(actual_col, 'mean'),
    )
    g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
    g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())
    print()

    # ─── 3. Call Wall 근접 (저항) → 상승 막힘? ───
    print("--- 3. Call Wall (현재가 위 큰 call OI) 근접 → 저항 효과? ---")
    s['cw_dist'] = s['call_wall_dist_pct']
    s['cw_bucket'] = pd.cut(
        s['cw_dist'],
        bins=[-100, 0, 2, 5, 10, 100],
        labels=["<0%(아래)", "0~2%", "2~5%", "5~10%", ">10%"],
    )
    g = s.groupby('cw_bucket', observed=False).agg(
        n=(actual_col, 'size'),
        win=(actual_col, lambda x: (x > 0).mean()),
        avg=(actual_col, 'mean'),
    )
    g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
    g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())
    print()

    # ─── 4. vol/OI 비정상 (catalyst 신호) ───
    print("--- 4. vol/OI ratio (비정상 거래 = catalyst 임박?) ---")
    if s['vol_oi_ratio'].std() > 0:
        s['voi_bucket'] = pd.qcut(s['vol_oi_ratio'], q=4, duplicates='drop',
                                  labels=["Q1 (낮음)", "Q2", "Q3", "Q4 (높음)"])
        g = s.groupby('voi_bucket', observed=False).agg(
            n=(actual_col, 'size'),
            win=(actual_col, lambda x: (x > 0).mean()),
            avg=(actual_col, 'mean'),
        )
        g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
        g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
        print(g.to_string())
    print()

    # ─── 5. IV rank bucket ───
    print("--- 5. IV Rank bucket ---")
    s['iv_bucket'] = pd.cut(
        s['iv_rank'],
        bins=[-0.01, 0.3, 0.5, 0.7, 1.01],
        labels=["IV<30%", "30~50%", "50~70%", ">70%"],
    )
    g = s.groupby('iv_bucket', observed=False).agg(
        n=(actual_col, 'size'),
        win=(actual_col, lambda x: (x > 0).mean()),
        avg=(actual_col, 'mean'),
    )
    g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
    g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())
    print()

    # ─── 6. 뉴스 sentiment bucket ───
    print("--- 6. News Sentiment bucket ---")
    s['news_bucket'] = pd.cut(
        s['news_score'],
        bins=[-15, -2, -0.5, 0.5, 2, 15],
        labels=["매우 부정", "부정", "중립", "긍정", "매우 긍정"],
    )
    g = s.groupby('news_bucket', observed=False).agg(
        n=(actual_col, 'size'),
        win=(actual_col, lambda x: (x > 0).mean()),
        avg=(actual_col, 'mean'),
    )
    g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
    g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())
    print()

    # ─── 7. Multi-factor 결합 (in-sample 가설 검증) ───
    print(f"--- 7. Multi-factor 결합 (n_min=10 필터링) ---")
    print(f"  {'filter':<60s} {'n':>4} {'win':>7} {'avg':>9}")
    print("-" * 90)
    base_str = f"baseline (always long)"
    print(f"  {base_str:<60s} {n_total:>4d} {base_win:>6.1%} {base_avg:>+8.2f}%")

    filters = [
        # Put wall 지지
        ("put_wall ±2% 안 (강한 floor)",
         s['put_wall_dist_pct'].between(-2, 0)),
        ("put_wall -5~-2% (적당한 지지)",
         s['put_wall_dist_pct'].between(-5, -2)),
        # 뉴스
        ("뉴스 매우 긍정 (≥+2)", s['news_score'] >= 2),
        ("뉴스 매우 부정 (≤-2)", s['news_score'] <= -2),
        # 뉴스 + put_wall
        ("뉴스 긍정 + put_wall 근접 (-5~0%)",
         (s['news_score'] >= 1) & s['put_wall_dist_pct'].between(-5, 0)),
        # IV
        ("IV<30% (옵션 쌈)", s['iv_rank'] < 0.3),
        ("IV>70% (옵션 비쌈, catalyst 임박)", s['iv_rank'] > 0.7),
        # Call wall 저항 + 뉴스 부정 (short 적기?)
        ("call_wall 2% 안 + 뉴스 부정 (저항 + 약세)",
         s['call_wall_dist_pct'].between(0, 2) & (s['news_score'] <= -1)),
        # 비정상 거래
        ("vol/OI > 0.5 (비정상 거래)",
         s['vol_oi_ratio'] > 0.5),
        # 종합 (multi-confluence)
        ("put_wall 근접 + 뉴스 긍정 + IV<50%",
         s['put_wall_dist_pct'].between(-5, 0) & (s['news_score'] >= 1) & (s['iv_rank'] < 0.5)),
        ("max_pain 위 + call_wall 근접 (상방 막힘)",
         (s['mp_dist_pct'] > 0) & s['call_wall_dist_pct'].between(0, 3)),
        # Unusual
        ("unusual bullish flow + put_wall 근접",
         (s['unusual_dir'] == 'bullish') & s['put_wall_dist_pct'].between(-5, 0)),
    ]
    for name, mask in filters:
        sub = s[mask]
        if len(sub) < 5:
            continue
        win = (sub[actual_col] > 0).mean()
        avg = sub[actual_col].mean()
        marker = "⭐" if win > 0.6 and avg > 1.0 else (
            "🔥" if avg > 2.0 else
            "❌" if win < 0.35 else ""
        )
        print(f"  {name:<60s} {len(sub):>4d} {win:>6.1%} {avg:>+8.2f}% {marker}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers",
        default="NVDA,AMD,TSLA,AAPL,MSFT,META,AMZN,CRCL,MSTR,COIN,HOOD,SMR",
    )
    parser.add_argument("--days", type=int, default=30)
    args = parser.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",")]
    end_date = date.today() - timedelta(days=12)
    snaps = []
    d = end_date
    while len(snaps) < args.days:
        if d.weekday() < 5:
            snaps.append(d)
        d -= timedelta(days=1)
    snaps.sort()

    print(f"tickers ({len(tickers)}): {tickers}")
    print(f"snapshots: {snaps[0]} ~ {snaps[-1]} ({len(snaps)} BD)")

    df = collect_events(tickers, snaps)
    if df.empty:
        print("no events")
        return
    out = Path(__file__).resolve().parents[2] / "data" / "results" / "options_signals.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n[saved] {out} ({len(df)} events)")

    for h in [3, 5, 10]:
        analyze(df, horizon=h)


if __name__ == "__main__":
    main()

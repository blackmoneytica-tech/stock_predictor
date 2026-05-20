"""옵션 다중 신호 결합 백테스트 — 진짜 alpha 있는 조합 찾기.

시점별 옵션 chain에서 추출:
  - max_pain
  - call_wall, put_wall (가장 큰 OI strike)
  - vol_oi_ratio (당일 옵션 거래량 / OI — 비정상 거래)
  - ATM IV, HV/IV ratio
  - 뉴스 sentiment_score
  - unusual_options_score

각 (ticker, snap, expiration) 단위로 5d/실제 만기 outcome 측정:
  - 가격이 어디로 갔나
  - max_pain 자석 vs call/put wall 자석 어느 게 강한가
  - vol/OI 비정상 = catalyst 임박 신호인지
  - 뉴스 + 옵션 결합 시 win rate

목표: "단일 옵션 신호로는 부정확하지만 결합하면 진짜 예측력 있는 조합"이 있는지.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def extract_walls_and_vol_oi(option_oi_by_strike: Dict, options_chain_exp: Dict) -> Dict:
    """OI 벽 + 비정상 거래 시그널 추출."""
    if not option_oi_by_strike:
        return {}

    # Call wall / Put wall — 가장 큰 OI strike (전체 또는 call/put 분리)
    # call_oi와 put_oi 따로 max 계산
    max_call_oi = 0
    call_wall = 0
    max_put_oi = 0
    put_wall = 0
    total_vol = 0
    total_oi = 0
    for strike, slot in options_chain_exp.items():
        c_oi = slot.get('call_oi', 0) or 0
        p_oi = slot.get('put_oi', 0) or 0
        c_vol = slot.get('call_volume', 0) or 0
        p_vol = slot.get('put_volume', 0) or 0
        if c_oi > max_call_oi:
            max_call_oi = c_oi
            call_wall = float(strike)
        if p_oi > max_put_oi:
            max_put_oi = p_oi
            put_wall = float(strike)
        total_vol += c_vol + p_vol
        total_oi += c_oi + p_oi

    vol_oi_ratio = total_vol / max(total_oi, 1)
    return {
        "call_wall": call_wall,
        "call_wall_oi": int(max_call_oi),
        "put_wall": put_wall,
        "put_wall_oi": int(max_put_oi),
        "total_vol_oi_ratio": round(vol_oi_ratio, 3),
    }


def collect_enhanced_events(
    tickers: List[str], snapshot_dates: List[date], verbose: bool = True,
) -> pd.DataFrame:
    from ..data.insider import get_insider_activity
    from ..data.price_feed import get_daily_ohlcv
    from ..system import StockPredictionSystem
    from .walk_forward import build_data_at

    system = StockPredictionSystem()
    rows = []

    if verbose:
        print("[prefetch] ETFs + macro...", flush=True)
    earliest = min(snapshot_dates) - timedelta(days=400)
    latest = max(snapshot_dates) + timedelta(days=60)
    for etf in ('XLK','XLF','XLE','XLV','XLI','XLY','XLP','XLU','XLRE','XLB','XLC',
                'SPY','QQQ','IWM','^VIX','HYG','LQD'):
        try: get_daily_ohlcv(etf, earliest, latest)
        except Exception: pass
    try:
        from ..data.sector_macro import compute_macro_breadth_at
        for snap in snapshot_dates:
            try: compute_macro_breadth_at(snap)
            except Exception: pass
    except ImportError: pass

    for ticker in tickers:
        if verbose:
            print(f"[{ticker}]", flush=True)
        try:
            full = get_daily_ohlcv(ticker, earliest, latest)
            if full.empty: continue
            insider = get_insider_activity(ticker, months_back=12)
        except Exception:
            continue

        for snap in snapshot_dates:
            ts_snap = pd.Timestamp(snap)
            at_or_before = full[full.index <= ts_snap]
            after = full[full.index > ts_snap]
            if at_or_before.empty or len(after) < 6:
                continue
            actual_today = float(full.loc[at_or_before.index[-1], "close"])
            # 5d / 10d future
            future_5 = after.head(5)
            future_10 = after.head(10)
            if len(future_5) < 5:
                continue
            close_5d = float(future_5['close'].iloc[-1])
            close_10d = float(future_10['close'].iloc[-1]) if len(future_10) >= 10 else None
            ret_5d = (close_5d - actual_today) / actual_today * 100
            ret_10d = (close_10d - actual_today) / actual_today * 100 if close_10d else None

            try:
                data = build_data_at(
                    ticker, snap, horizon_days=5, use_macro=True,
                    insider_cache=insider,
                )
                result = system.analyze(ticker, horizon_days=5, data=data)
            except Exception:
                continue

            opt = result.modules['options'].details
            max_pain = opt.get('max_pain')
            iv = opt.get('iv')
            hv = opt.get('hv')
            hv_iv = opt.get('hv_iv_ratio')
            iv_rank = opt.get('iv_rank')
            pc_ratio = opt.get('put_call_ratio')
            dte = opt.get('days_to_expiration')
            exp_date = opt.get('expiration_date')

            # Call/Put wall + vol/OI 추출
            option_oi = data.get('option_oi_by_strike') or {}
            options_chain = data.get('options_chain') or {}
            target_exp = data.get('target_expiration')
            walls = extract_walls_and_vol_oi(
                option_oi, options_chain.get(target_exp, {}),
            ) if target_exp else {}

            # 뉴스 sentiment + unusual
            news_score = data.get('news_sentiment_score', 0)
            news_n = data.get('news_sentiment_n', 0)
            unusual_score = data.get('unusual_options_score', 0)
            unusual_dir = data.get('unusual_options_direction', 'neutral')

            # max_pain / call_wall / put_wall vs cur 거리
            mp_dist = (actual_today - max_pain) / actual_today * 100 if max_pain else 0
            cw_dist = (actual_today - walls.get('call_wall', actual_today)) / actual_today * 100
            pw_dist = (actual_today - walls.get('put_wall', actual_today)) / actual_today * 100

            rows.append({
                "ticker": ticker,
                "as_of": snap.isoformat(),
                "cur": round(actual_today, 2),
                "close_5d": round(close_5d, 2),
                "ret_5d": round(ret_5d, 2),
                "ret_10d": round(ret_10d, 2) if ret_10d else None,
                "max_pain": round(max_pain, 2) if max_pain else None,
                "mp_dist_pct": round(mp_dist, 2),
                "call_wall": walls.get("call_wall"),
                "cw_dist_pct": round(cw_dist, 2),
                "call_wall_oi": walls.get("call_wall_oi"),
                "put_wall": walls.get("put_wall"),
                "pw_dist_pct": round(pw_dist, 2),
                "put_wall_oi": walls.get("put_wall_oi"),
                "vol_oi_ratio": walls.get("total_vol_oi_ratio"),
                "iv": round(iv, 3) if iv else None,
                "iv_rank": round(iv_rank, 3) if iv_rank else None,
                "hv_iv": round(hv_iv, 3) if hv_iv else None,
                "pc_ratio": round(pc_ratio, 3) if pc_ratio else None,
                "dte": int(dte) if dte else None,
                "news_score": round(news_score, 2),
                "news_n": int(news_n),
                "unusual_score": round(unusual_score, 2),
                "unusual_dir": unusual_dir,
                "macro_mode": (data.get("macro_breadth") or {}).get("mode", "?"),
                "composite_score": round(result.composite_score, 2),
                "confidence": round(result.confidence, 3),
            })

    return pd.DataFrame(rows)


def analyze(df: pd.DataFrame):
    if df.empty:
        print("결과 없음")
        return

    print(f"\n=== {len(df)} events, {df['ticker'].nunique()} tickers ===")
    print(f"date: {df['as_of'].min()} ~ {df['as_of'].max()}")
    print(f"baseline 5d: win {(df['ret_5d']>0).mean():.1%}, avg {df['ret_5d'].mean():+.2f}%")
    print()

    # === 1. Max Pain distance 분석 ===
    print("--- 1. Max Pain 거리 bucket × 5d outcome ---")
    df['mp_bkt'] = pd.cut(
        df['mp_dist_pct'],
        bins=[-100, -10, -5, -2, 2, 5, 10, 100],
        labels=["<-10%", "-10~-5%", "-5~-2%", "±2%", "+2~+5%", "+5~+10%", ">+10%"],
    )
    g = df.groupby('mp_bkt', observed=False).agg(
        n=('ret_5d', 'size'),
        win=('ret_5d', lambda x: (x>0).mean()),
        avg=('ret_5d', 'mean'),
    )
    g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
    g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())
    print()

    # === 2. Call wall 거리 ===
    print("--- 2. Call wall 거리(현재가 - call_wall) × 5d ---")
    df['cw_bkt'] = pd.cut(
        df['cw_dist_pct'],
        bins=[-100, -10, -5, -2, 2, 5, 10, 100],
        labels=["<-10%", "-10~-5%", "-5~-2%", "±2%", "+2~+5%", "+5~+10%", ">+10%"],
    )
    g = df.groupby('cw_bkt', observed=False).agg(
        n=('ret_5d', 'size'),
        win=('ret_5d', lambda x: (x>0).mean()),
        avg=('ret_5d', 'mean'),
    )
    g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
    g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())
    print()

    # === 3. Put wall ===
    print("--- 3. Put wall 거리(현재가 - put_wall) × 5d ---")
    df['pw_bkt'] = pd.cut(
        df['pw_dist_pct'],
        bins=[-100, -10, -5, -2, 2, 5, 10, 100],
        labels=["<-10%", "-10~-5%", "-5~-2%", "±2%", "+2~+5%", "+5~+10%", ">+10%"],
    )
    g = df.groupby('pw_bkt', observed=False).agg(
        n=('ret_5d', 'size'),
        win=('ret_5d', lambda x: (x>0).mean()),
        avg=('ret_5d', 'mean'),
    )
    g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
    g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())
    print()

    # === 4. Volume/OI ratio (비정상 거래) ===
    print("--- 4. vol/OI ratio (비정상 거래 강도) × 5d ---")
    df['vol_bkt'] = pd.qcut(df['vol_oi_ratio'].fillna(0), q=4, duplicates='drop',
                            labels=["Q1 (낮음)","Q2","Q3","Q4 (높음)"])
    g = df.groupby('vol_bkt', observed=False).agg(
        n=('ret_5d', 'size'),
        win=('ret_5d', lambda x: (x>0).mean()),
        avg=('ret_5d', 'mean'),
        mean_ratio=('vol_oi_ratio', 'mean'),
    )
    g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
    g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
    g['mean_ratio'] = g['mean_ratio'].apply(lambda x: f"{x:.2f}")
    print(g.to_string())
    print()

    # === 5. IV Rank ===
    if df['iv_rank'].notna().sum() > 10:
        print("--- 5. IV Rank × 5d ---")
        df['iv_bkt'] = pd.cut(
            df['iv_rank'],
            bins=[0, 0.3, 0.5, 0.7, 1.0],
            labels=["저 IV (<30%)","중저","중고","고 IV (>70%)"],
        )
        g = df.groupby('iv_bkt', observed=False).agg(
            n=('ret_5d', 'size'),
            win=('ret_5d', lambda x: (x>0).mean()),
            avg=('ret_5d', 'mean'),
        )
        g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
        g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
        print(g.to_string())
        print()

    # === 6. 뉴스 sentiment ===
    print("--- 6. 뉴스 sentiment × 5d ---")
    df['news_bkt'] = pd.cut(
        df['news_score'],
        bins=[-10, -2, -0.5, 0.5, 2, 10],
        labels=["매우 부정","부정","중립","긍정","매우 긍정"],
    )
    g = df.groupby('news_bkt', observed=False).agg(
        n=('ret_5d', 'size'),
        win=('ret_5d', lambda x: (x>0).mean()),
        avg=('ret_5d', 'mean'),
    )
    g['win'] = g['win'].apply(lambda x: f"{x:.1%}")
    g['avg'] = g['avg'].apply(lambda x: f"{x:+.2f}%")
    print(g.to_string())
    print()

    # === 7. Multi-factor combo (가장 강한 신호 찾기) ===
    print("--- 7. Multi-factor combo: 진짜 alpha 있는 조합 ---")
    base_win = (df['ret_5d']>0).mean()
    base_avg = df['ret_5d'].mean()
    print(f"  baseline: win {base_win:.1%}, avg {base_avg:+.2f}%")

    combos = [
        ("Call wall 위 (현재가 > call_wall)",
         df['cw_dist_pct'] > 1),
        ("Put wall 아래 (현재가 < put_wall)",
         df['pw_dist_pct'] < -1),
        ("Call wall 근접 ±2%",
         df['cw_dist_pct'].abs() < 2),
        ("Put wall 근접 ±2%",
         df['pw_dist_pct'].abs() < 2),
        ("vol/OI Q4 (비정상 거래 강함)",
         df['vol_oi_ratio'] >= df['vol_oi_ratio'].quantile(0.75)),
        ("IV Rank 낮음 (<30%)",
         df['iv_rank'] < 0.3),
        ("IV Rank 높음 (>70%)",
         df['iv_rank'] > 0.7),
        ("뉴스 매우 긍정 (>+2)",
         df['news_score'] > 2),
        ("뉴스 매우 부정 (<-2)",
         df['news_score'] < -2),
        ("Unusual 강함 (>|2|)",
         df['unusual_score'].abs() > 2),
        ("ATM near max_pain (±1%) + vol/OI Q4",
         (df['mp_dist_pct'].abs() < 1) & (df['vol_oi_ratio'] >= df['vol_oi_ratio'].quantile(0.75))),
        ("Put wall 위 + 뉴스 긍정",
         (df['pw_dist_pct'] > 0) & (df['news_score'] > 0.5)),
        ("Call wall 아래 + 뉴스 부정",
         (df['cw_dist_pct'] < 0) & (df['news_score'] < -0.5)),
        ("call_wall > current > put_wall (sandwich)",
         (df['cw_dist_pct'] > 0) & (df['pw_dist_pct'] < 0)),
        ("Put wall break (cur < put_wall) — 약세 신호",
         df['pw_dist_pct'] < -1),
        ("Call wall break (cur > call_wall) — 강세 신호",
         df['cw_dist_pct'] > 1),
    ]
    print(f"  {'combo':<55s} {'n':<5s} {'win':<7s} {'avg':<8s} {'vs baseline':<12s}")
    print("  " + "-"*90)
    for name, mask in combos:
        sub = df[mask.fillna(False)]
        if len(sub) < 5:
            continue
        w = (sub['ret_5d']>0).mean()
        a = sub['ret_5d'].mean()
        d_win = w - base_win
        d_avg = a - base_avg
        marker = "⭐" if w > base_win + 0.05 and a > base_avg + 0.5 else "  "
        print(f"  {marker}{name:<53s} {len(sub):<5d} {w:.1%}   {a:+.2f}%  ({d_win:+.1%}p, {d_avg:+.2f}%p)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tickers",
        default="NVDA,AMD,TSLA,AAPL,MSFT,GOOGL,META,AMZN,CRCL,MSTR,COIN,HOOD",
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

    df = collect_enhanced_events(tickers, snaps)
    if df.empty:
        print("no events")
        return

    out = Path("data/results/options_signal_events.parquet")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n[saved] {out} ({len(df)} events)\n")
    analyze(df)


if __name__ == "__main__":
    main()

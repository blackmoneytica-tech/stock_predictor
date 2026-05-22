"""D: 전략 비교 — R6 vs Momentum vs Hybrid vs Buy-Hold benchmark.

가설: R6 (mean reversion) 의 기대수익률이 momentum picker / buy-hold 보다 큰가?
       hybrid 가 단독 전략 보다 우월한가?

전략 4개:
  M1 R6 (mean reversion, 검증된 룰)
     subset + iv_rank<0.20 + RSI<35 + (any regime)
     hold 최대 10일, target VAH, stop VAL×0.98

  M2 Momentum (52w 신고가 breakout)
     close >= 52w_high × 0.97  AND  vol > vol_20 × 1.5  AND  MA50 > MA200
     hold 최대 20일, trailing stop -8%, target +15%

  M3 Buy-Hold QQQ / TQQQ / SPY (benchmark)
     매일 1주, 5y rebalance 안 함

  M4 Hybrid (R6 + M2)
     같은 capital pool 에서 R6 신호 OR M2 신호 trigger 시 진입

측정:
  - 총 수익률 (5y)
  - CAGR
  - Sharpe (per-trade-annualized)
  - max drawdown
  - 발동 빈도 (trades / year)
  - capacity (max concurrent positions)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date, timedelta
import numpy as np
import pandas as pd
from scipy import stats

from src.data.price_feed import get_daily_ohlcv


def section(t): print(f"\n{'='*70}\n{t}\n{'='*70}")


# ── Momentum signals ──
def add_momentum_signals(hist):
    df = hist.copy()
    c = df["close"]; v = df["volume"]; h = df["high"]
    df["high_52w"] = h.rolling(252).max()
    df["near_52w_high"] = c >= df["high_52w"] * 0.97
    df["vol_20"] = v.rolling(20).mean()
    df["vol_surge"] = v > df["vol_20"] * 1.5
    df["ma50"] = c.rolling(50).mean()
    df["ma200"] = c.rolling(200).mean()
    df["uptrend"] = df["ma50"] > df["ma200"]
    df["mom_5d"] = c.pct_change(5)
    df["mom_20d"] = c.pct_change(20)
    return df


# ── 전략별 trade list 생성 ──
def gen_r6_trades(panel, subset, h_days=10):
    """R6 mean reversion trades."""
    p = panel.copy()
    p["F2"] = p["iv_rank"] < 0.20
    p["F3"] = p["rsi"] < 35
    p["F1"] = p["ticker"].isin(subset)
    # R6 도 regime 무관 simple 버전 (가장 많은 sample)
    mask = p["F2"] & p["F3"] & p["F1"]
    cands = p[mask].copy()
    # 시뮬레이션: 진입 close, target VAH or +20%, stop VAL×0.98, hold 10d
    trades = []
    for _, r in cands.iterrows():
        entry = r["close"]
        target = min(r["vah"], entry * 1.20) if pd.notna(r["vah"]) and r["vah"] > entry else entry * 1.10
        stop = r["val"] * 0.98 if pd.notna(r["val"]) and r["val"] < entry else entry * 0.97
        outcome = "timeout"; pnl = 0; days = h_days
        for hh in range(1, h_days + 1):
            hi = r.get(f"fhigh_{hh}"); lo = r.get(f"flow_{hh}")
            if pd.isna(hi) or pd.isna(lo): break
            if lo <= stop:
                outcome, pnl, days = "stop", (stop - entry) / entry, hh; break
            if hi >= target:
                outcome, pnl, days = "target", (target - entry) / entry, hh; break
        else:
            last = r.get(f"flow_{h_days}")
            if pd.notna(last):
                pnl = (last - entry) / entry
        trades.append(dict(
            ticker=r["ticker"], date=r["date"], strategy="R6",
            entry=entry, pnl=pnl, outcome=outcome, days=days,
        ))
    return pd.DataFrame(trades)


def gen_momentum_trades(hist_by_ticker, h_days=20, target_pct=0.15, stop_pct=0.08):
    """52w 고점 breakout + vol + uptrend trades."""
    trades = []
    for t, hist in hist_by_ticker.items():
        if hist is None or len(hist) < 300: continue
        f = add_momentum_signals(hist)
        mask = f["near_52w_high"] & f["vol_surge"] & f["uptrend"]
        signals = f[mask]
        for i, (idx, r) in enumerate(signals.iterrows()):
            entry = r["close"]
            target = entry * (1 + target_pct)
            stop = entry * (1 - stop_pct)
            # forward
            pos = f.index.get_loc(idx)
            outcome = "timeout"; pnl = 0; days = h_days
            for hh in range(1, h_days + 1):
                if pos + hh >= len(f): break
                fwd = f.iloc[pos + hh]
                if fwd["low"] <= stop:
                    outcome, pnl, days = "stop", (stop - entry) / entry, hh; break
                if fwd["high"] >= target:
                    outcome, pnl, days = "target", (target - entry) / entry, hh; break
            else:
                if pos + h_days < len(f):
                    pnl = (f.iloc[pos + h_days]["close"] - entry) / entry
            trades.append(dict(
                ticker=t, date=idx, strategy="Momentum",
                entry=entry, pnl=pnl, outcome=outcome, days=days,
            ))
    return pd.DataFrame(trades)


def buy_hold_return(ticker, start_date, end_date):
    hist = get_daily_ohlcv(ticker, start=start_date, end=end_date)
    if hist is None or len(hist) < 10:
        return None
    first = hist["close"].iloc[0]
    last = hist["close"].iloc[-1]
    return dict(
        ticker=ticker, start=hist.index[0], end=hist.index[-1],
        first=float(first), last=float(last),
        total_ret=(last - first) / first,
        years=(hist.index[-1] - hist.index[0]).days / 365,
    )


# ── Equity curve simulation (Time-weighted) ──
def simulate_equity(trades_df, max_positions=5):
    """trade 단위 P&L 을 capital 1.0 시작점에서 capital 단위로 동시 보유 (max_positions) 가정.

    각 trade 가 capital/max_positions 비중 사용. 동시 진행 trade 가 max 까지.
    """
    if not len(trades_df):
        return None
    df = trades_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["exit_date"] = df["date"] + pd.to_timedelta(df["days"], unit="D")

    capital = 1.0
    open_positions = []  # (exit_date, exit_capital_delta)
    daily_equity = []
    cur_date = df["date"].min()
    end_date = df["exit_date"].max()
    while cur_date <= end_date:
        # close finished positions
        kept = []
        for pos_exit, delta in open_positions:
            if pos_exit <= cur_date:
                capital *= (1 + delta)
            else:
                kept.append((pos_exit, delta))
        open_positions = kept
        # open new positions
        new_today = df[df["date"] == cur_date]
        for _, t in new_today.iterrows():
            if len(open_positions) >= max_positions: break
            size = 1.0 / max_positions
            open_positions.append((t["exit_date"], t["pnl"] * size))
        daily_equity.append((cur_date, capital))
        cur_date += pd.Timedelta(days=1)
    eq = pd.DataFrame(daily_equity, columns=["date", "equity"])
    return eq


def metrics(eq, label):
    if eq is None or len(eq) < 10:
        return dict(label=label, cagr=None, sharpe=None, max_dd=None, total_ret=None)
    eq = eq.copy()
    days = (eq["date"].iloc[-1] - eq["date"].iloc[0]).days
    years = days / 365.25
    total_ret = eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1
    cagr = (eq["equity"].iloc[-1] / eq["equity"].iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    daily_ret = eq["equity"].pct_change().dropna()
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else 0
    peak = eq["equity"].cummax()
    dd = eq["equity"] / peak - 1
    max_dd = dd.min()
    return dict(label=label, total_ret=total_ret, cagr=cagr, sharpe=sharpe, max_dd=max_dd, years=years)


def fmt_metrics(m):
    if m["cagr"] is None: return f"{m['label']}: 데이터 부족"
    return (f"{m['label']:<32} "
            f"total={m['total_ret']*100:+7.1f}% "
            f"CAGR={m['cagr']*100:+6.1f}% "
            f"Sharpe={m['sharpe']:+5.2f} "
            f"maxDD={m['max_dd']*100:+6.1f}% "
            f"years={m['years']:.1f}")


def main():
    section("0. 데이터 로드 (R6 panel + RSI)")
    panel = pd.read_parquet("data/results/a2_vp_panel.parquet")
    panel["date"] = pd.to_datetime(panel["date"])
    # RSI 부착
    tickers = panel["ticker"].unique()
    print(f"  panel {len(panel)} rows, {len(tickers)} tickers")

    rsi_rows = []
    for t in tickers:
        hist = get_daily_ohlcv(t, start=date.today() - timedelta(days=int(5 * 365) + 60),
                                  end=date.today())
        if hist is None or len(hist) < 100: continue
        delta = hist["close"].diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - 100 / (1 + rs)
        rsi_rows.append(pd.DataFrame({"ticker": t, "date": hist.index, "rsi": rsi.values}))
    rsi_df = pd.concat(rsi_rows, ignore_index=True)
    rsi_df["date"] = pd.to_datetime(rsi_df["date"])
    panel = panel.merge(rsi_df, on=["ticker", "date"], how="left")
    print(f"  RSI merged")

    subset = pd.read_csv("data/results/positive_subset.csv")["ticker"].tolist()
    print(f"  positive_subset: {len(subset)}")

    section("1. R6 trades 생성")
    r6_trades = gen_r6_trades(panel, subset)
    print(f"  R6 trades: {len(r6_trades)}")
    if len(r6_trades):
        print(f"  R6 outcome:")
        print(r6_trades["outcome"].value_counts().to_string())
        print(f"  R6 avg pnl: {r6_trades['pnl'].mean() * 100:+.2f}%, win {(r6_trades['pnl'] > 0).mean():.1%}")

    section("2. Momentum trades 생성 (52w 고점 + vol + uptrend)")
    print(f"  ticker별 hist fetch...")
    hist_dict = {}
    for t in tickers:
        try:
            h = get_daily_ohlcv(t, start=date.today() - timedelta(days=int(5 * 365) + 60),
                                  end=date.today())
            if h is not None and len(h) >= 300:
                hist_dict[t] = h
        except Exception:
            pass
    print(f"  {len(hist_dict)}/{len(tickers)} tickers")
    mom_trades = gen_momentum_trades(hist_dict)
    print(f"  Momentum trades: {len(mom_trades)}")
    if len(mom_trades):
        print(f"  Momentum outcome:")
        print(mom_trades["outcome"].value_counts().to_string())
        print(f"  Momentum avg pnl: {mom_trades['pnl'].mean() * 100:+.2f}%, win {(mom_trades['pnl'] > 0).mean():.1%}")

    section("3. Buy-Hold benchmark")
    start_d = date.today() - timedelta(days=int(5 * 365))
    end_d = date.today()
    bh_results = {}
    for sym in ["SPY", "QQQ", "TQQQ", "SOXL"]:
        r = buy_hold_return(sym, start_d, end_d)
        if r:
            cagr = (1 + r["total_ret"]) ** (1 / r["years"]) - 1
            bh_results[sym] = dict(label=f"Buy-Hold {sym}", total_ret=r["total_ret"],
                                     cagr=cagr, years=r["years"])
            print(f"  {sym}: total {r['total_ret']*100:+.1f}% / CAGR {cagr*100:+.1f}% / years {r['years']:.1f}")

    section("4. Equity curve simulation (max 5 concurrent positions)")
    eq_r6 = simulate_equity(r6_trades, max_positions=5)
    eq_mom = simulate_equity(mom_trades, max_positions=5)

    # hybrid: 두 trade pool 결합
    hybrid_trades = pd.concat([r6_trades, mom_trades], ignore_index=True).sort_values("date")
    eq_hybrid = simulate_equity(hybrid_trades, max_positions=5)

    m_r6 = metrics(eq_r6, "R6 (mean reversion)")
    m_mom = metrics(eq_mom, "Momentum (52w breakout)")
    m_hyb = metrics(eq_hybrid, "Hybrid (R6 + Momentum)")

    print()
    print(fmt_metrics(m_r6))
    print(fmt_metrics(m_mom))
    print(fmt_metrics(m_hyb))
    print()
    for sym, b in bh_results.items():
        print(f"{'Buy-Hold ' + sym:<32} "
              f"total={b['total_ret']*100:+7.1f}% "
              f"CAGR={b['cagr']*100:+6.1f}% "
              f"Sharpe=  N/A maxDD=  N/A "
              f"years={b['years']:.1f}")

    section("5. 종합 — 가장 큰 기대수익률 + 위험조정")
    rows = [
        dict(strategy="R6 mean reversion", **m_r6),
        dict(strategy="Momentum (52w breakout)", **m_mom),
        dict(strategy="Hybrid (R6+Momentum)", **m_hyb),
    ]
    for sym, b in bh_results.items():
        rows.append(dict(strategy=f"Buy-Hold {sym}", total_ret=b["total_ret"],
                          cagr=b["cagr"], sharpe=None, max_dd=None, years=b["years"]))
    df = pd.DataFrame(rows)
    df = df.sort_values("cagr", ascending=False).reset_index(drop=True)
    print(df.to_string(index=False))

    # save trades for reuse
    r6_trades.to_parquet("data/results/r6_trades.parquet")
    mom_trades.to_parquet("data/results/momentum_trades.parquet")
    print(f"\n  saved trades → data/results/")


if __name__ == "__main__":
    main()

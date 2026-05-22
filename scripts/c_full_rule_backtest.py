"""C: A+B+C 통합 룰 — final 백테스트.

A: positive subset 필터 (train avg>0인 93종만)
B: F5 + RSI<35 (T1) — 검증된 multi-factor
C: 시장 regime filter (SPY 기반) — BEAR/CHOPPY/BULL

테스트할 룰:
  R1: F5 + RSI<35  (B 기본)
  R2: F5 + RSI<35 + subset (A+B)
  R3: F5 + RSI<35 + regime ∈ {BULL, CHOPPY}  (B+C bull-aligned)
  R4: F5 + RSI<35 + regime ∈ {BEAR, CHOPPY}  (B+C bear-mean-reversion)
  R5: subset + F5 + RSI<35 + BULL/CHOPPY (A+B+C INTEGRATED)
  R6: subset + F5 + RSI<35 + BEAR/CHOPPY (A+B+C contrarian)

검증:
  - 5y 전체 + train/test 분할
  - 각 룰의 alpha + 안정성
  - 오늘 watchlist 통과 종목
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date, timedelta
import numpy as np
import pandas as pd
from scipy import stats

from src.data.price_feed import get_daily_ohlcv
from src.data.universe import get_universe


def section(t): print(f"\n{'='*70}\n{t}\n{'='*70}")


def block_stats(rets_pct, h_days=5):
    """input rets_pct: 이미 percent 단위 (e.g., 1.0 = 1%)."""
    r = pd.Series(rets_pct).dropna()
    if len(r) < 5: return None
    wins = int((r > 0).sum())
    avg = r.mean()
    std = r.std()
    sharpe = (avg / std * np.sqrt(252 / h_days)) if std > 0 else 0
    p = stats.binomtest(wins, len(r), 0.5, alternative="greater").pvalue
    return dict(n=len(r), win=wins / len(r), avg=avg, sharpe=sharpe, p=p)


def compute_spy_regime():
    """SPY 일별 regime (BULL/CHOPPY/BEAR)."""
    spy = get_daily_ohlcv("SPY", start=date.today() - timedelta(days=int(5 * 365) + 60),
                            end=date.today())
    if spy is None or len(spy) < 250:
        return pd.DataFrame()
    c = spy["close"]
    sma200 = c.rolling(200).mean()
    ret_20d = c.pct_change(20) * 100
    regime = pd.Series(index=c.index, dtype=object)
    regime[(c > sma200) & (ret_20d > 0)] = "BULL"
    regime[(c < sma200) & (ret_20d < -5)] = "BEAR"
    regime = regime.fillna("CHOPPY")
    out = pd.DataFrame({"date": c.index, "regime": regime.values})
    out["date"] = pd.to_datetime(out["date"])
    return out


def add_rsi(p_base, universe):
    """기존 panel 에 RSI14 추가."""
    rows = []
    for t in universe:
        try:
            hist = get_daily_ohlcv(t, start=date.today() - timedelta(days=int(5 * 365) + 60),
                                     end=date.today())
            if hist is None or len(hist) < 100: continue
            c = hist["close"]
            delta = c.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - 100 / (1 + rs)
            rows.append(pd.DataFrame({
                "ticker": t, "date": hist.index, "rsi": rsi.values,
            }))
        except Exception:
            continue
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main():
    section("0. data load — panel + RSI + SPY regime")
    p = pd.read_parquet("data/results/f5_panel_5y.parquet")
    p["date"] = pd.to_datetime(p["date"])
    print(f"  panel: {len(p)} rows, {p['ticker'].nunique()} tickers")

    rsi_df = add_rsi(p, p["ticker"].unique())
    rsi_df["date"] = pd.to_datetime(rsi_df["date"])
    p = p.merge(rsi_df, on=["ticker", "date"], how="left")
    print(f"  RSI merged: coverage {p['rsi'].notna().mean():.1%}")

    regime_df = compute_spy_regime()
    p = p.merge(regime_df, on="date", how="left")
    print(f"  regime distribution:")
    print(p["regime"].value_counts().to_string())

    # A: subset
    try:
        subset = pd.read_csv("data/results/positive_subset.csv")["ticker"].tolist()
        print(f"\n  positive_subset (A): {len(subset)} tickers")
    except Exception:
        subset = list(p["ticker"].unique())
        print(f"\n  ⚠️ subset 없음, all tickers 사용")

    # ── 룰 정의 ──
    p["F5"] = p["iv_rank"] < 0.20
    p["RSI_low"] = p["rsi"] < 35
    p["in_subset"] = p["ticker"].isin(subset)
    p["regime_bull_choppy"] = p["regime"].isin(["BULL", "CHOPPY"])
    p["regime_bear_choppy"] = p["regime"].isin(["BEAR", "CHOPPY"])

    rules = {
        "R1 F5+RSI (B base)":             p["F5"] & p["RSI_low"],
        "R2 F5+RSI+subset (A+B)":         p["F5"] & p["RSI_low"] & p["in_subset"],
        "R3 F5+RSI+BULL/CHOPPY (B+C)":    p["F5"] & p["RSI_low"] & p["regime_bull_choppy"],
        "R4 F5+RSI+BEAR/CHOPPY (B+C)":    p["F5"] & p["RSI_low"] & p["regime_bear_choppy"],
        "R5 INTEGRATED (A+B+C bull)":     p["F5"] & p["RSI_low"] & p["in_subset"] & p["regime_bull_choppy"],
        "R6 INTEGRATED (A+B+C bear)":     p["F5"] & p["RSI_low"] & p["in_subset"] & p["regime_bear_choppy"],
        "baseline (all)":                  pd.Series(True, index=p.index),
    }

    # ── 전체 5y ──
    section("1. 5y 전체 alpha (5d horizon)")
    print(f"  {'rule':<38} {'n':<7} {'win%':<8} {'avg%':<8} {'Sharpe':<7} {'p<.5':<7}")
    for label, mask in rules.items():
        sub = p[mask]["fwd_5d_pct"].dropna()  # already %
        st = block_stats(sub)
        if st:
            print(f"  {label:<38} {st['n']:<7} {st['win']:.1%}   {st['avg']:+.2f}%   {st['sharpe']:+.2f}   {st['p']:.4f}")

    # ── 5y train/test 분할 ──
    section("2. 시간순 train/test 60/40 분할")
    p_s = p.sort_values("date").reset_index(drop=True)
    cut = p_s["date"].quantile(0.60)
    train = p_s[p_s["date"] <= cut]
    test = p_s[p_s["date"] > cut]
    print(f"  train: {train['date'].min().date()} ~ {train['date'].max().date()}  n={len(train)}")
    print(f"  test : {test['date'].min().date()} ~ {test['date'].max().date()}  n={len(test)}")

    print(f"\n  {'rule':<38} {'TRAIN n':<8} {'tr win':<8} {'tr avg':<8} | {'TEST n':<8} {'te win':<8} {'te avg':<8} {'Sharpe':<7}")
    print(f"  {'-'*38} {'-'*8} {'-'*8} {'-'*8}-+-{'-'*8} {'-'*8} {'-'*8} {'-'*7}")
    for label, _ in rules.items():
        tr_mask = (
            (train["F5"] if "F5" in label or "INTEG" in label or "base" in label else pd.Series(True, index=train.index))
        )
        # rebuild mask per train/test
        tr_mask_list = {
            "R1 F5+RSI (B base)":             train["F5"] & train["RSI_low"],
            "R2 F5+RSI+subset (A+B)":         train["F5"] & train["RSI_low"] & train["ticker"].isin(subset),
            "R3 F5+RSI+BULL/CHOPPY (B+C)":    train["F5"] & train["RSI_low"] & train["regime"].isin(["BULL", "CHOPPY"]),
            "R4 F5+RSI+BEAR/CHOPPY (B+C)":    train["F5"] & train["RSI_low"] & train["regime"].isin(["BEAR", "CHOPPY"]),
            "R5 INTEGRATED (A+B+C bull)":     train["F5"] & train["RSI_low"] & train["ticker"].isin(subset) & train["regime"].isin(["BULL", "CHOPPY"]),
            "R6 INTEGRATED (A+B+C bear)":     train["F5"] & train["RSI_low"] & train["ticker"].isin(subset) & train["regime"].isin(["BEAR", "CHOPPY"]),
            "baseline (all)":                  pd.Series(True, index=train.index),
        }
        te_mask_list = {
            "R1 F5+RSI (B base)":             test["F5"] & test["RSI_low"],
            "R2 F5+RSI+subset (A+B)":         test["F5"] & test["RSI_low"] & test["ticker"].isin(subset),
            "R3 F5+RSI+BULL/CHOPPY (B+C)":    test["F5"] & test["RSI_low"] & test["regime"].isin(["BULL", "CHOPPY"]),
            "R4 F5+RSI+BEAR/CHOPPY (B+C)":    test["F5"] & test["RSI_low"] & test["regime"].isin(["BEAR", "CHOPPY"]),
            "R5 INTEGRATED (A+B+C bull)":     test["F5"] & test["RSI_low"] & test["ticker"].isin(subset) & test["regime"].isin(["BULL", "CHOPPY"]),
            "R6 INTEGRATED (A+B+C bear)":     test["F5"] & test["RSI_low"] & test["ticker"].isin(subset) & test["regime"].isin(["BEAR", "CHOPPY"]),
            "baseline (all)":                  pd.Series(True, index=test.index),
        }
        tr_m = tr_mask_list[label]
        te_m = te_mask_list[label]
        tr_ret = train[tr_m]["fwd_5d_pct"].dropna()
        te_ret = test[te_m]["fwd_5d_pct"].dropna()
        st_tr = block_stats(tr_ret)
        st_te = block_stats(te_ret)
        if st_tr and st_te:
            print(f"  {label:<38} {st_tr['n']:<8} {st_tr['win']:.1%}   {st_tr['avg']:+.2f}%  | "
                  f"{st_te['n']:<8} {st_te['win']:.1%}   {st_te['avg']:+.2f}%   {st_te['sharpe']:+.2f}")

    # ── 오늘 적용 ──
    section("3. 오늘 watchlist 룰 통과 종목 (각 룰)")
    print(f"  현재 SPY regime: {regime_df.iloc[-1]['regime']}")

    today_rows = []
    universe = get_universe("full")
    for t in universe:
        try:
            hist = get_daily_ohlcv(t, start=date.today() - timedelta(days=400), end=date.today())
            if hist is None or len(hist) < 100: continue
            c = hist["close"]
            log_ret = np.log(c / c.shift(1)).dropna()
            rolling_hv = log_ret.rolling(20).std() * np.sqrt(252)
            win = rolling_hv.dropna().tail(252)
            iv_rank = float((win < win.iloc[-1]).mean()) if not win.empty else None
            delta = c.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - 100 / (1 + rs)
            rsi_now = float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None
            today_rows.append(dict(
                ticker=t, cur=float(c.iloc[-1]),
                iv_rank=iv_rank, rsi=rsi_now,
                in_subset=t in subset,
                F5=iv_rank < 0.20 if iv_rank else False,
                RSI_low=rsi_now < 35 if rsi_now else False,
            ))
        except Exception:
            continue
    td = pd.DataFrame(today_rows)
    current_regime = regime_df.iloc[-1]["regime"]

    print(f"\n  {'rule':<38} {'후보 수':<10} {'종목':<60}")
    rule_today_masks = {
        "R1 F5+RSI (B base)":             td["F5"] & td["RSI_low"],
        "R2 F5+RSI+subset (A+B)":         td["F5"] & td["RSI_low"] & td["in_subset"],
        "R3 F5+RSI+BULL/CHOPPY (B+C)":    td["F5"] & td["RSI_low"] if current_regime in ["BULL", "CHOPPY"] else pd.Series(False, index=td.index),
        "R4 F5+RSI+BEAR/CHOPPY (B+C)":    td["F5"] & td["RSI_low"] if current_regime in ["BEAR", "CHOPPY"] else pd.Series(False, index=td.index),
        "R5 INTEGRATED (A+B+C bull)":     td["F5"] & td["RSI_low"] & td["in_subset"] if current_regime in ["BULL", "CHOPPY"] else pd.Series(False, index=td.index),
        "R6 INTEGRATED (A+B+C bear)":     td["F5"] & td["RSI_low"] & td["in_subset"] if current_regime in ["BEAR", "CHOPPY"] else pd.Series(False, index=td.index),
    }
    for label, m in rule_today_masks.items():
        cand = td[m]
        tickers_str = ", ".join(cand["ticker"].tolist()) if len(cand) else "(없음)"
        print(f"  {label:<38} {len(cand):<10} {tickers_str:<60}")


if __name__ == "__main__":
    main()

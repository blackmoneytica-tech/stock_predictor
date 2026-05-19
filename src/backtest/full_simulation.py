"""통합 시뮬레이션 — Trade PnL + Aggregator 재학습 + Sizing 룰.

기존 improvement_data.parquet (1500 rows × 1d/3d/5d) 활용.
또한 11개 모듈 raw score를 새로 수집 (시스템 다시 호출).

Output:
1. Strategy별 PnL / Sharpe / MDD / Win rate
2. Logistic regression으로 학습된 새 aggregator weight vs 현재
3. Position sizing matrix (conviction × macro × horizon)
4. 통합 시스템 (새 weight + sizing) PnL vs baseline
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════
# Part A: 11개 모듈 score per snapshot 수집
# ════════════════════════════════════════════════════════════
def collect_module_scores(tickers: List[str], snapshot_dates, horizon_lookahead: int = 5):
    """기존 walk-forward와 동일하지만 11개 모듈 score 모두 저장."""
    from ..data.insider import get_insider_activity
    from ..data.price_feed import get_daily_ohlcv
    from ..system import StockPredictionSystem
    from .walk_forward import build_data_at

    system = StockPredictionSystem()
    rows = []

    print("[prefetch] ETFs + macro...", flush=True)
    earliest = min(snapshot_dates) - timedelta(days=400)
    latest = max(snapshot_dates) + timedelta(days=horizon_lookahead + 10)
    for etf in ('XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLRE',
                'XLB', 'XLC', 'SPY', 'QQQ', 'IWM', '^VIX', 'HYG', 'LQD'):
        try:
            get_daily_ohlcv(etf, earliest, latest)
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
    print("[prefetch] done", flush=True)

    for ticker in tickers:
        print(f"[{ticker}]", flush=True)
        try:
            full = get_daily_ohlcv(ticker, earliest, latest)
            if full.empty:
                continue
            insider = get_insider_activity(ticker, months_back=12)
        except Exception as e:
            print(f"  SKIP fetch: {e}", flush=True)
            continue

        for snap in snapshot_dates:
            ts_snap = pd.Timestamp(snap)
            after = full[full.index > ts_snap]
            at_or_before = full[full.index <= ts_snap]
            if at_or_before.empty or len(after) < horizon_lookahead:
                continue
            as_of_idx = at_or_before.index[-1]
            actual_today = float(full.loc[as_of_idx, "close"])
            actuals = {}
            for h in (1, 3, 5):
                future = full[full.index > ts_snap]
                if len(future) < h:
                    actuals[h] = None
                else:
                    actuals[h] = float(future["close"].iloc[h - 1])
            if any(v is None for v in actuals.values()):
                continue

            try:
                data = build_data_at(
                    ticker, snap, horizon_days=5, use_macro=True,
                    insider_cache=insider,
                )
                pred = system.analyze(ticker, horizon_days=5, data=data)
            except Exception as e:
                print(f"  {snap} fail: {e}", flush=True)
                continue

            row = {
                "ticker": ticker,
                "as_of": snap.isoformat(),
                "cur": round(actual_today, 2),
                "composite_score": round(pred.composite_score, 3),
                "confidence": round(pred.confidence, 3),
                "ev_pct_5d": round(
                    (pred.expected_value - pred.current_price) / pred.current_price * 100, 3
                ),
                "macro_mode": (data.get("macro_breadth") or {}).get("mode", "?"),
            }
            for name, m in pred.modules.items():
                row[f"mod_{name}"] = round(m.score, 3)

            for h in (1, 3, 5):
                ret = (actuals[h] - actual_today) / actual_today * 100
                row[f"actual_ret_{h}d"] = round(ret, 3)
                row[f"actual_up_{h}d"] = int(ret > 0)

            rows.append(row)

    return pd.DataFrame(rows)


# ════════════════════════════════════════════════════════════
# Part B: Logistic regression으로 새 aggregator weight 학습
# ════════════════════════════════════════════════════════════
def learn_aggregator_weights(df: pd.DataFrame, horizon: int = 1) -> Dict:
    """단순 logistic regression (numpy) — module scores → P(actual up).

    in-sample/out-of-sample 분리해서 generalize 검증.
    """
    mod_cols = [c for c in df.columns if c.startswith("mod_")]
    y_col = f"actual_up_{horizon}d"

    # in-sample (앞 50%) / out-of-sample (뒤 50%)
    s = df.copy().sort_values('as_of').reset_index(drop=True)
    cut = len(s) // 2
    train, test = s.iloc[:cut], s.iloc[cut:]

    X_tr = train[mod_cols].fillna(0).values
    y_tr = train[y_col].values.astype(float)
    X_te = test[mod_cols].fillna(0).values
    y_te = test[y_col].values.astype(float)

    # 단순 logistic regression — gradient descent
    w = np.zeros(X_tr.shape[1])
    b = 0.0
    lr = 0.01
    for _ in range(2000):
        z = X_tr @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -50, 50)))
        grad_w = X_tr.T @ (p - y_tr) / len(y_tr) + 0.01 * w  # L2 0.01
        grad_b = (p - y_tr).mean()
        w -= lr * grad_w
        b -= lr * grad_b

    def acc(X, y):
        z = X @ w + b
        p = 1 / (1 + np.exp(-np.clip(z, -50, 50)))
        return float(((p > 0.5).astype(int) == y).mean())

    train_acc = acc(X_tr, y_tr)
    test_acc = acc(X_te, y_te)

    # 모듈명 → weight (normalize abs sum to 1 for comparison)
    abs_sum = np.sum(np.abs(w)) or 1.0
    weights_normalized = {
        mod_cols[i].replace("mod_", ""): float(w[i] / abs_sum)
        for i in range(len(w))
    }
    weights_raw = {
        mod_cols[i].replace("mod_", ""): float(w[i])
        for i in range(len(w))
    }

    return {
        "horizon": horizon,
        "n_train": len(train),
        "n_test": len(test),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "weights_normalized": weights_normalized,
        "weights_raw": weights_raw,
        "bias": float(b),
    }


# ════════════════════════════════════════════════════════════
# Part C: Position sizing matrix + 통합 strategy 시뮬레이션
# ════════════════════════════════════════════════════════════
def sizing_factor(macro_mode: str, ev_pct: float, conf: float, horizon: int) -> float:
    """walk-forward 검증된 alpha 기반 sizing factor (0.0 ~ 1.5).

    1d × BEAR + 시스템 신호: 64% win → 큰 사이즈 (1.5x)
    1d × CHOPPY + 시스템 신호: 52% win (+13%p) → 1.2x
    1d × BULL: baseline > 시스템 → 시스템 무시, baseline 0.8x
    5d × BULL/STRONG_BULL: baseline long 1.0x
    5d × STRONG_BEAR: oversold long 0.8x (반등)
    5d × BEAR: 0 (cash)
    5d × CHOPPY: 0.4x (small long)
    """
    macro = (macro_mode or "?").upper()
    sig_strong = abs(ev_pct) > 0.5 and conf >= 0.5

    if horizon == 1:
        if macro == "BEAR":
            return 1.5 if sig_strong else 0.5
        if macro == "CHOPPY":
            return 1.2 if sig_strong else 0.4
        if macro in ("BULL", "STRONG_BULL", "STRONG_BEAR"):
            return 0.8  # baseline long
        return 0.4

    # 3d / 5d horizon — macro-aligned baseline
    if macro in ("BULL", "STRONG_BULL"):
        return 1.0
    if macro == "STRONG_BEAR":
        return 0.8  # rebound
    if macro == "BEAR":
        return 0.0  # cash
    return 0.4  # CHOPPY small long


def trade_direction(macro_mode: str, ev_pct: float, horizon: int) -> int:
    """+1 long / -1 short / 0 cash. Short은 walk-forward 검증으로 모두 -EV → 항상 0.

    1d × BEAR/CHOPPY: 시스템 신호 따라감
    그 외: long bias only
    """
    macro = (macro_mode or "?").upper()
    if horizon == 1 and macro in ("BEAR", "CHOPPY"):
        # 시스템 신호 sweet spot — 신호 약함이면 0
        if abs(ev_pct) < 0.3:
            return 0
        # 단 short은 영구 금지 (walk-forward에서 -3~-7%)
        return 1 if ev_pct > 0 else 0
    return 1  # default long


def simulate_strategy(df: pd.DataFrame, name: str, direction_fn, sizing_fn, horizon: int):
    """direction × size × actual return → PnL series."""
    s = df.copy().sort_values('as_of').reset_index(drop=True)
    actual_col = f"actual_ret_{horizon}d"
    s = s.dropna(subset=[actual_col])

    s['direction'] = s.apply(
        lambda r: direction_fn(r['macro_mode'], r.get(f'ev_pct_{horizon}d', r.get('ev_pct_5d', 0)), horizon),
        axis=1,
    )
    s['size'] = s.apply(
        lambda r: sizing_fn(r['macro_mode'], r.get(f'ev_pct_{horizon}d', r.get('ev_pct_5d', 0)), r['confidence'], horizon),
        axis=1,
    )
    s['pnl'] = s['direction'] * s['size'] * s[actual_col]

    n_trades = (s['size'] > 0).sum()
    pnl = s['pnl']
    win_pnl = pnl[pnl > 0]
    loss_pnl = pnl[pnl < 0]
    cum_pnl = pnl.cumsum()

    if n_trades == 0:
        return {"name": name, "n_trades": 0}

    sharpe = (pnl.mean() / pnl.std() * np.sqrt(252 / max(1, horizon))) if pnl.std() > 0 else 0
    mdd = float((cum_pnl - cum_pnl.cummax()).min())

    return {
        "name": name,
        "horizon": horizon,
        "n_trades": int(n_trades),
        "total_pnl": round(float(pnl.sum()), 1),
        "avg_pnl": round(float(pnl.mean()), 3),
        "win_rate": round(float((pnl > 0).mean()), 3),
        "profit_factor": round(
            float(win_pnl.sum() / abs(loss_pnl.sum())) if len(loss_pnl) > 0 else float('inf'),
            2,
        ),
        "sharpe": round(float(sharpe), 2),
        "mdd": round(mdd, 1),
        "best_trade": round(float(pnl.max()), 2),
        "worst_trade": round(float(pnl.min()), 2),
    }


def main():
    """기존 improvement_data.parquet에 11 module scores 추가 후 시뮬레이션."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--collect", action="store_true",
        help="11모듈 score 다시 수집 (필요 시 30분~)"
    )
    parser.add_argument(
        "--tickers",
        default="NVDA,AMD,TSLA,AAPL,MSFT,GOOGL,META,AMZN,CRCL,MSTR,COIN,HOOD,NFLX,PLTR,SMCI",
    )
    parser.add_argument("--days", type=int, default=100)
    args = parser.parse_args()

    data_path = Path("data/results/module_scores.parquet")

    if args.collect or not data_path.exists():
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        end_date = date.today() - timedelta(days=8)
        snaps = []
        d = end_date
        while len(snaps) < args.days:
            if d.weekday() < 5:
                snaps.append(d)
            d -= timedelta(days=1)
        snaps.sort()
        print(f"collecting: {len(tickers)} tickers × {len(snaps)} BD ({snaps[0]} ~ {snaps[-1]})")
        df = collect_module_scores(tickers, snaps, horizon_lookahead=5)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_path, index=False)
        print(f"[saved] {data_path} ({len(df)} rows)")
    else:
        df = pd.read_parquet(data_path)
        print(f"[loaded] {data_path} ({len(df)} rows)")

    # ────────────────────────────────────────────
    # Part B: Aggregator weight 재학습
    # ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Part B: Aggregator weight 재학습 (logistic regression)")
    print('='*70)

    CURRENT_WEIGHTS = {
        "options": 0.17, "macro": 0.15, "catalyst": 0.12,
        "demand_supply": 0.11, "technical": 0.10, "mean_reversion": 0.08,
        "trend": 0.07, "order_block": 0.06, "insider": 0.06,
        "sentiment": 0.05, "short_squeeze": 0.03,
    }

    for h in [1, 3, 5]:
        result = learn_aggregator_weights(df, horizon=h)
        print(f"\n--- Horizon {h}d ---")
        print(f"in-sample acc: {result['train_acc']:.1%} (n={result['n_train']})")
        print(f"out-sample acc: {result['test_acc']:.1%} (n={result['n_test']})")
        # 모듈별 weight 비교 (현재 vs 학습)
        learned = result['weights_normalized']
        comparison = pd.DataFrame({
            'module': list(CURRENT_WEIGHTS.keys()),
            'current_w': [CURRENT_WEIGHTS.get(k, 0) for k in CURRENT_WEIGHTS],
            'learned_w': [learned.get(k, 0) for k in CURRENT_WEIGHTS],
        })
        comparison['delta'] = comparison['learned_w'] - comparison['current_w']
        comparison = comparison.sort_values('learned_w', key=abs, ascending=False)
        comparison['current_w'] = comparison['current_w'].apply(lambda x: f"{x:+.3f}")
        comparison['learned_w'] = comparison['learned_w'].apply(lambda x: f"{x:+.3f}")
        comparison['delta'] = comparison['delta'].apply(lambda x: f"{x:+.3f}")
        print(comparison.to_string(index=False))

    # ────────────────────────────────────────────
    # Part C: Strategy 시뮬레이션 (PnL/Sharpe/MDD)
    # ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("Part C: Strategy 시뮬레이션 (sizing × direction)")
    print('='*70)

    # ev_pct_*d 컬럼이 1d/3d/5d별로 따로 있어야 sizing이 정확. 우선 5d만으로.
    df['ev_pct_1d'] = df['ev_pct_5d']  # 시스템은 5d로 분석했지만 1d 동일 score 사용 (근사)
    df['ev_pct_3d'] = df['ev_pct_5d']

    results = []

    # baseline (always long, full size)
    for h in [1, 3, 5]:
        results.append(simulate_strategy(
            df, f"baseline_long_{h}d",
            direction_fn=lambda m, e, h: 1,
            sizing_fn=lambda m, e, c, h: 1.0,
            horizon=h,
        ))

    # raw system signal (sign of ev_pct, full size)
    for h in [1, 3, 5]:
        results.append(simulate_strategy(
            df, f"raw_signal_{h}d",
            direction_fn=lambda m, e, h: 1 if e > 0 else 0,  # short 금지
            sizing_fn=lambda m, e, c, h: 1.0,
            horizon=h,
        ))

    # NEW: walk-forward 검증 룰 + sizing
    for h in [1, 3, 5]:
        results.append(simulate_strategy(
            df, f"verified_rules_{h}d",
            direction_fn=trade_direction,
            sizing_fn=sizing_factor,
            horizon=h,
        ))

    out = pd.DataFrame(results)
    print(out.to_string(index=False))

    # 비교 — baseline vs verified rules
    print(f"\n{'='*70}")
    print("핵심 비교: baseline vs verified rules")
    print('='*70)
    for h in [1, 3, 5]:
        base = out[(out['name'] == f'baseline_long_{h}d')].iloc[0]
        ver = out[(out['name'] == f'verified_rules_{h}d')].iloc[0]
        delta_pnl = ver['total_pnl'] - base['total_pnl']
        delta_sharpe = ver['sharpe'] - base['sharpe']
        print(f"  {h}d: verified vs baseline → "
              f"total {delta_pnl:+.1f}%, Sharpe {delta_sharpe:+.2f}, "
              f"win {ver['win_rate']:.1%} vs {base['win_rate']:.1%}, "
              f"PF {ver['profit_factor']} vs {base['profit_factor']}")


if __name__ == "__main__":
    main()

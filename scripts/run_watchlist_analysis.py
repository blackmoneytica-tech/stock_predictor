"""GitHub Actions cron + manual trigger entry point.

1. trade-journal /api/watchlist 호출 → "내 워치" 종목 fetch
2. 각 ticker에 대해 1d/3d/5d 분석 (multi-horizon ensemble)
3. /api/predictions/save POST

환경 변수 (Github Secrets):
    TRADE_JOURNAL_URL=https://trade-journal-1dv.pages.dev
    AUTH_TOKEN=<trade-journal AUTH_TOKEN>
    FRED_API_KEY, MARKETDATA_KEY, FINNHUB_KEY, ALPHA_VANTAGE_KEY,
    SEC_USER_AGENT=stock_predictor <email>

실행:
    python scripts/run_watchlist_analysis.py
    python scripts/run_watchlist_analysis.py --tickers NVDA,CRCL  # 강제 ticker
    python scripts/run_watchlist_analysis.py --horizons 1,3,5     # 분석 horizon
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import requests

# src 경로 import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.system import StockPredictionSystem  # noqa: E402
from src.strategy.calibration import calibrator, fitness_db  # noqa: E402
from src.strategy.multi_horizon import ensemble_predictions, label_agreement  # noqa: E402


def _env(key: str, default: str = "") -> str:
    return (os.environ.get(key) or default).strip()


def fetch_watchlist(base_url: str, auth_token: str, source: str = "") -> List[str]:
    """trade-journal /api/watchlist (source 없으면 전체 active)."""
    qs = f"?source={source}" if source else ""
    r = requests.get(
        f"{base_url}/api/watchlist{qs}",
        headers={"X-Auth-Token": auth_token},
        timeout=15,
    )
    r.raise_for_status()
    return r.json().get("tickers", [])


def analyze_one(system: StockPredictionSystem, ticker: str, horizons: List[int]) -> List[Dict]:
    """단일 ticker × N horizons → predictions list (DB row 1:N)."""
    now_iso = datetime.now(timezone.utc).isoformat()
    snap_at = now_iso[:19]  # ISO 초까지

    rows = []
    horizon_preds = []  # ensemble용
    for h in horizons:
        try:
            r = system.analyze(ticker, horizon_days=h)
        except Exception as e:
            print(f"  {ticker} h={h} FAIL: {e}", flush=True)
            continue

        cur = r.current_price
        ev = r.expected_value
        ev_pct = (ev - cur) / cur * 100 if cur else 0
        cal_acc = calibrator.calibrate(r.confidence)
        fitness = fitness_db.fitness(ticker)
        decision_label = _decision_label(r.composite_score, ev_pct, r.confidence)

        # 모듈 score + 시나리오 + 매도/손절 zones
        payload = {
            "modules": {
                name: {
                    "score": round(m.score, 3),
                    "direction": m.direction.name,
                    "confidence": round(m.confidence, 3),
                }
                for name, m in r.modules.items()
            },
            "scenarios": [
                {
                    "name": s.name,
                    "probability": round(s.probability, 4),
                    "low": round(s.price_range[0], 2),
                    "high": round(s.price_range[1], 2),
                    "ev": round(s.expected_value, 2),
                }
                for s in r.scenarios
            ],
            "ci_50": [round(r.ci_50[0], 2), round(r.ci_50[1], 2)],
            "ci_80": [round(r.ci_80[0], 2), round(r.ci_80[1], 2)],
            "sell_triggers": [
                {"price": round(t.price, 2), "action": t.action, "reason": t.reason}
                for t in r.sell_triggers[:5]
            ],
            "stop_loss": [
                {"price": round(t.price, 2), "action": t.action, "reason": t.reason}
                for t in r.stop_loss[:3]
            ],
            "hedges": [
                {
                    "type": h.type,
                    "strike": h.strike or h.put_strike,
                    "expiration_days": h.expiration_days,
                    "rationale": h.rationale,
                }
                for h in r.hedge_recommendations[:3]
            ],
            "options": {
                "max_pain": r.modules["options"].details.get("max_pain"),
                "implied_move": r.modules["options"].details.get("implied_move"),
                "iv": r.modules["options"].details.get("iv"),
                "hv": r.modules["options"].details.get("hv"),
                "hv_iv_ratio": r.modules["options"].details.get("hv_iv_ratio"),
                "iv_rank": r.modules["options"].details.get("iv_rank"),
            },
            "macro_breadth_mode": r.modules["macro"].details.get("sector_mode"),
        }

        horizon_preds.append({
            "horizon": h,
            "composite_score": r.composite_score,
            "ev_pct": ev_pct,
            "conf": r.confidence,
            "directional_bias": r.directional_bias,
        })

        rows.append({
            "ticker": ticker.upper(),
            "snapshot_at": snap_at,
            "horizon_days": h,
            "current_price": round(cur, 2),
            "expected_value": round(ev, 2),
            "ev_pct": round(ev_pct, 3),
            "composite_score": round(r.composite_score, 3),
            "confidence": round(r.confidence, 4),
            "cal_acc": round(cal_acc, 4),
            "fitness": round(fitness, 3),
            "directional_bias": r.directional_bias,
            "decision_label": decision_label,
            "agreement": None,  # ensemble 후 채움
            "payload": payload,
        })

    # Multi-horizon ensemble → 모든 row에 agreement 저장
    if len(horizon_preds) >= 2:
        ens = ensemble_predictions(horizon_preds)
        agree = label_agreement(ens["agreement"])
        for row in rows:
            row["agreement"] = agree

    return rows


def _decision_label(score: float, ev_pct: float, conf: float) -> str:
    if conf < 0.45:
        return "관망"
    if score >= 3 or ev_pct >= 3:
        return "강한 매수"
    if score >= 1 or ev_pct >= 1:
        return "매수 검토"
    if score <= -3 or ev_pct <= -3:
        return "매도/회피"
    if score <= -1 or ev_pct <= -1:
        return "조심/부분 매도"
    return "관망"


def _json_default(o):
    """numpy/pandas types → Python native."""
    if hasattr(o, "item"):
        try:
            return o.item()
        except Exception:
            return str(o)
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return str(o)


def post_predictions(base_url: str, auth_token: str, predictions: List[Dict]) -> Dict:
    """Bulk POST (numpy-safe JSON encoding)."""
    body = json.dumps(
        {"predictions": predictions},
        default=_json_default,
        allow_nan=False,
    )
    r = requests.post(
        f"{base_url}/api/predictions/save",
        headers={"X-Auth-Token": auth_token, "Content-Type": "application/json"},
        data=body,
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", help="comma-separated override")
    parser.add_argument("--horizons", default="1,3,5", help="comma-separated days")
    parser.add_argument("--dry-run", action="store_true", help="POST 하지 않음")
    args = parser.parse_args()

    base_url = _env("TRADE_JOURNAL_URL", "https://trade-journal-1dv.pages.dev")
    auth = _env("AUTH_TOKEN")
    if not auth and not args.dry_run:
        print("ERROR: AUTH_TOKEN 미설정", file=sys.stderr)
        sys.exit(1)

    horizons = [int(h) for h in args.horizons.split(",")]

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
        print(f"[manual] tickers: {tickers}", flush=True)
    else:
        print("[fetch] watchlist from trade-journal", flush=True)
        tickers = fetch_watchlist(base_url, auth)
        print(f"[watchlist] {len(tickers)} tickers: {tickers}", flush=True)

    if not tickers:
        print("[done] empty watchlist, nothing to do", flush=True)
        return

    system = StockPredictionSystem()
    t0 = time.time()
    all_rows = []
    for i, ticker in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] analyzing {ticker}...", flush=True)
        rows = analyze_one(system, ticker, horizons)
        all_rows.extend(rows)
        print(f"  → {len(rows)} predictions ({time.time()-t0:.0f}s elapsed)", flush=True)

    print(f"\n[total] {len(all_rows)} predictions in {time.time()-t0:.1f}s", flush=True)

    if args.dry_run:
        print("[dry-run] not posting")
        print(json.dumps(all_rows[:2], indent=2, default=str))
        return

    print("[post] /api/predictions/save", flush=True)
    resp = post_predictions(base_url, auth, all_rows)
    print(f"[done] saved {resp.get('saved')} / errors {len(resp.get('errors', []))}", flush=True)


if __name__ == "__main__":
    main()

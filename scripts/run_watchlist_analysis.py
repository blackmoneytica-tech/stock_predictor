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


def fetch_settings(base_url: str, auth_token: str) -> Dict:
    """예측 분석 주기 설정 (D1 kv → /api/predictions/settings)."""
    try:
        r = requests.get(
            f"{base_url}/api/predictions/settings",
            headers={"X-Auth-Token": auth_token},
            timeout=10,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[settings] fetch fail (default 사용): {e}", flush=True)
        return {"enabled": True, "interval_hours": 1, "cooldown_remaining_min": 0}


def mark_run_now(base_url: str, auth_token: str) -> None:
    """분석 완료 후 last_run_at 갱신."""
    try:
        requests.post(
            f"{base_url}/api/predictions/settings",
            headers={"X-Auth-Token": auth_token, "Content-Type": "application/json"},
            json={"mark_run": True},
            timeout=10,
        )
    except Exception as e:
        print(f"[settings] mark_run fail: {e}", flush=True)


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

        # 모듈 score + 시나리오 + 매도/손절 zones + EMA/POC/confluence
        tech = r.modules["technical"].details
        ds = r.modules["demand_supply"].details
        opt = r.modules["options"].details

        # 직관 신호 카드 (UI에서 그대로 노출)
        signals = _build_signal_cards(cur, tech, ds, opt, r)

        payload = {
            "decision": {
                "label": decision_label,
                "directional_bias": r.directional_bias,
                "confidence": round(r.confidence, 3),
                "ev_pct": round(ev_pct, 2),
                "rr_ratio": _safe_rr(cur, r.sell_triggers, r.stop_loss),
            },
            "signals": signals,
            "levels": {
                "current": round(cur, 2),
                "ema_20": _r2(tech.get("ema_20")),
                "ema_50": _r2(tech.get("ema_50")),
                "sma_20": _r2(tech.get("sma_20")),
                "sma_50": _r2(tech.get("sma_50")),
                "sma_200": _r2(tech.get("sma_200")),
                "poc": _r2(ds.get("poc")),
                "value_area_high": _r2(ds.get("value_area_high")),
                "value_area_low": _r2(ds.get("value_area_low")),
                "max_pain": _r2(opt.get("max_pain")),
                "expected_5d": _r2(ev),
                "ci_50_low": round(r.ci_50[0], 2),
                "ci_50_high": round(r.ci_50[1], 2),
                "ci_80_low": round(r.ci_80[0], 2),
                "ci_80_high": round(r.ci_80[1], 2),
            },
            "indicators": {
                "rsi": _r2(tech.get("rsi")),
                "macd": _r2(tech.get("macd")),
                "bb_position": _r2(tech.get("bb_position")),
            },
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
            "confluence_zones": _serialize_zones(r.confluence_zones, cur),
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
                "max_pain": opt.get("max_pain"),
                "implied_move": opt.get("implied_move"),
                "iv": opt.get("iv"),
                "hv": opt.get("hv"),
                "hv_iv_ratio": opt.get("hv_iv_ratio"),
                "iv_rank": opt.get("iv_rank"),
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


def _r2(v):
    """None-safe round to 2 decimals."""
    if v is None:
        return None
    try:
        f = float(v)
        if f != f:  # NaN
            return None
        return round(f, 2)
    except (TypeError, ValueError):
        return None


def _safe_rr(cur: float, sells, stops) -> float:
    """1차 매도 vs 1차 손절의 R/R 비율."""
    if not sells or not stops or not cur:
        return 0.0
    try:
        upside = sells[0].price - cur
        downside = cur - stops[0].price
        if downside <= 0:
            return 0.0
        return round(upside / downside, 2)
    except Exception:
        return 0.0


def _build_signal_cards(cur: float, tech: Dict, ds: Dict, opt: Dict, r) -> List[Dict]:
    """직관 신호 카드 (UI 배지용). 각 카드: {key, label, tone, detail}.

    tone: 'bull' / 'bear' / 'neutral' / 'warn'
    """
    out = []

    # 1) MA 정배열 / 역배열
    ema20, ema50, sma200 = tech.get("ema_20"), tech.get("ema_50"), tech.get("sma_200")
    if ema20 and ema50 and sma200 and cur:
        if cur > ema20 > ema50 > sma200:
            out.append({"key": "ma_trend", "label": "정배열", "tone": "bull",
                        "detail": f"현재가 > EMA20 > EMA50 > SMA200"})
        elif cur < ema20 < ema50 < sma200:
            out.append({"key": "ma_trend", "label": "역배열", "tone": "bear",
                        "detail": f"현재가 < EMA20 < EMA50 < SMA200"})
        else:
            above = sum(1 for x in (ema20, ema50, sma200) if cur > x)
            out.append({"key": "ma_trend",
                        "label": f"MA {above}/3 위" if above >= 2 else f"MA {above}/3 위",
                        "tone": "bull" if above >= 2 else "bear",
                        "detail": f"EMA20 ${ema20:.2f} · EMA50 ${ema50:.2f} · SMA200 ${sma200:.2f}"})

    # 2) RSI
    rsi = tech.get("rsi")
    if rsi is not None and not (rsi != rsi):
        if rsi >= 70:
            out.append({"key": "rsi", "label": f"RSI 과열 {rsi:.0f}", "tone": "warn",
                        "detail": "70 이상 — 단기 조정 가능"})
        elif rsi <= 30:
            out.append({"key": "rsi", "label": f"RSI 과매도 {rsi:.0f}", "tone": "bull",
                        "detail": "30 이하 — 반등 가능 구간"})
        else:
            tone = "bull" if rsi > 55 else "bear" if rsi < 45 else "neutral"
            out.append({"key": "rsi", "label": f"RSI {rsi:.0f}", "tone": tone,
                        "detail": "중립 구간"})

    # 3) 매물대 위치 (POC / VAH / VAL)
    poc = ds.get("poc")
    vah = ds.get("value_area_high")
    val = ds.get("value_area_low")
    if poc and vah and val and cur:
        if cur > vah:
            out.append({"key": "vp", "label": "Value Area 위", "tone": "bull",
                        "detail": f"VAH ${vah:.2f} 돌파 — 추세 강함, POC ${poc:.2f} 자석"})
        elif cur < val:
            out.append({"key": "vp", "label": "Value Area 아래", "tone": "bear",
                        "detail": f"VAL ${val:.2f} 이탈 — 약세, POC ${poc:.2f}까지 반등 여지"})
        else:
            dist_to_poc = (cur - poc) / cur * 100
            out.append({"key": "vp",
                        "label": "Value Area 안",
                        "tone": "neutral",
                        "detail": f"POC ${poc:.2f} ({dist_to_poc:+.1f}%) · 횡보 가능"})

    # 4) 옵션 Max Pain
    mp = opt.get("max_pain")
    if mp and cur:
        diff = (mp - cur) / cur * 100
        if abs(diff) < 1.5:
            tone, label = "neutral", f"Max Pain ${mp:.2f} 근접"
        elif diff > 0:
            tone, label = "bull", f"Max Pain ${mp:.2f} 위 ({diff:+.1f}%)"
        else:
            tone, label = "bear", f"Max Pain ${mp:.2f} 아래 ({diff:+.1f}%)"
        out.append({"key": "max_pain", "label": label, "tone": tone,
                    "detail": "옵션 만기 시 가격이 향하는 자석 가격"})

    # 5) IV / IV Rank
    iv = opt.get("iv")
    iv_rank = opt.get("iv_rank")
    if iv_rank is not None:
        rank_pct = iv_rank * 100 if iv_rank <= 1 else iv_rank
        if rank_pct >= 70:
            out.append({"key": "iv", "label": f"IV 높음 {rank_pct:.0f}%",
                        "tone": "warn",
                        "detail": "옵션 비싼 구간 — 매도자 유리"})
        elif rank_pct <= 30:
            out.append({"key": "iv", "label": f"IV 낮음 {rank_pct:.0f}%",
                        "tone": "bull",
                        "detail": "옵션 싼 구간 — 매수자 유리"})

    # 6) Multi-horizon agreement (set by analyze_one 후처리에서)
    return out


def _serialize_zones(conf, cur: float) -> Dict:
    """ActionEngine confluence_zones → JSON-safe dict (UI용).

    Input: {'supply': [...], 'demand': [...], 'all_clusters': [...]}
    Output: {'buy': [demand zones], 'sell': [supply zones]}
    """
    if not conf:
        return {}
    mapping = {"buy": "demand", "sell": "supply"}
    out = {}
    for ui_side, conf_side in mapping.items():
        zones = conf.get(conf_side) or []
        out[ui_side] = []
        for z in zones[:6]:
            low = z.get("low") or z.get("price")
            high = z.get("high") or z.get("price")
            if low is None:
                continue
            try:
                low_f = float(low)
                high_f = float(high) if high is not None else low_f
                price = (low_f + high_f) / 2
                dist_pct = (price - cur) / cur * 100 if cur else 0
            except (TypeError, ValueError):
                continue
            out[ui_side].append({
                "low": round(low_f, 2),
                "high": round(high_f, 2),
                "price": round(price, 2),
                "dist_pct": round(dist_pct, 2),
                "n_sources": int(z.get("n_sources") or 0),
                "sources": list(z.get("sources") or [])[:6],
                "strength": round(float(z.get("confluence_strength") or 0), 2),
            })
    return out


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
    parser.add_argument("--force", action="store_true", help="cooldown/enabled 무시 (수동 trigger)")
    args = parser.parse_args()

    base_url = _env("TRADE_JOURNAL_URL", "https://trade-journal-1dv.pages.dev")
    auth = _env("AUTH_TOKEN")
    if not auth and not args.dry_run:
        print("ERROR: AUTH_TOKEN 미설정", file=sys.stderr)
        sys.exit(1)

    # 사용자 설정 체크 (cron 매시간 → 사용자 interval 적용)
    # --tickers 수동 override 또는 --force는 cooldown 무시
    if not args.tickers and not args.force and not args.dry_run:
        settings = fetch_settings(base_url, auth)
        if not settings.get("enabled", True):
            print(f"[settings] disabled — skip (interval_hours={settings.get('interval_hours')})", flush=True)
            return
        cooldown = settings.get("cooldown_remaining_min", 0)
        if cooldown > 0:
            print(
                f"[settings] cooldown {cooldown} min remaining "
                f"(interval_hours={settings.get('interval_hours')}) — skip",
                flush=True,
            )
            return
        print(
            f"[settings] enabled, interval={settings.get('interval_hours')}h, "
            f"last_run={settings.get('last_run_at')}",
            flush=True,
        )

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

    # last_run_at 갱신 (cooldown 시작점)
    if not args.tickers:
        mark_run_now(base_url, auth)
        print("[settings] last_run_at updated", flush=True)


if __name__ == "__main__":
    main()

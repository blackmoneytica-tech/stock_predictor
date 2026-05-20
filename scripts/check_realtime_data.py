"""실시간 옵션/매물대 데이터 fetch 정확성 점검 — 실제 시스템 함수와 동일.

핵심 발견 (이전 잘못):
  F5 의 "iv_rank" 는 실제로는 HV (historical volatility) percentile rank.
  즉 옵션 IV 가 아니라 yfinance OHLCV 로 계산 가능 → 데이터 한계 없음.

점검:
  1. 옵션 chain (Marketdata.app) — 만기/strike/OI/IV 신선도
  2. F5 핵심 지표: HV-based iv_rank — walk_forward.py 와 동일 공식
  3. F6 핵심 지표: Volume Profile (POC/VAH/VAL) — demand_supply.py compute_volume_profile
  4. Max Pain / call/put walls — 자석/저항선
  5. 신선도 — 마지막 봉 timestamp, cache stale 여부
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd

from src.data.realtime_options import get_realtime_chain
from src.data.price_feed import get_daily_ohlcv, get_current_price
from src.modules.demand_supply import compute_volume_profile


TEST_TICKERS = ["NVDA", "AAPL", "AMD", "META", "CRCL", "MSTR"]


def _f(x, prec=3):
    if x is None: return "None"
    try:
        if x != x: return "NaN"
        return f"{float(x):.{prec}f}"
    except Exception:
        return repr(x)


def section(t): print(f"\n{'='*65}\n{t}\n{'='*65}")


def fetch_hist(ticker, days=400):
    start = date.today() - timedelta(days=days)
    return get_daily_ohlcv(ticker, start=start, end=date.today())


def compute_hv_iv_rank(hist):
    """walk_forward.py 와 동일 — HV percentile rank (0~1)."""
    closes = hist["close"]
    log_ret = np.log(closes / closes.shift(1)).dropna()
    if len(log_ret) < 50:
        return None, None
    hv_30 = float(log_ret.tail(30).std() * np.sqrt(252))
    rolling_hv = log_ret.rolling(20).std() * np.sqrt(252)
    rolling_hv_252 = rolling_hv.dropna().tail(252)
    if rolling_hv_252.empty:
        return hv_30, None
    iv_rank = float((rolling_hv_252 < rolling_hv_252.iloc[-1]).mean())
    return hv_30, iv_rank


def check_ticker(t):
    section(f"[{t}]")
    # 1) 현재가
    try:
        cur = get_current_price(t)
        print(f"  현재가: ${cur:.2f}")
    except Exception as e:
        print(f"  현재가 실패: {e}"); return

    # 2) Historical (HV based F5)
    try:
        hist = fetch_hist(t, days=400)
        if hist is None or len(hist) < 50:
            print(f"  historical 부족 ({len(hist) if hist is not None else 0} bars)")
            return
        last_bar_date = pd.to_datetime(hist.index[-1]).date()
        days_behind = (date.today() - last_bar_date).days
        print(f"  ── F5 (HV iv_rank — 백테스트 룰의 진짜 핵심) ──")
        print(f"  OHLCV bars: {len(hist)}, 마지막 봉 = {last_bar_date} ({days_behind}일 전)")
        hv30, iv_rank = compute_hv_iv_rank(hist)
        print(f"    HV30 (현재 변동성): {_f(hv30, 4)}")
        print(f"    iv_rank (1y HV percentile): {_f(iv_rank, 3)}")
        f5_active = iv_rank is not None and iv_rank < 0.30
        print(f"    F5 (iv_rank < 0.30): {'★ 활성' if f5_active else '비활성'}")
    except Exception as e:
        print(f"  historical 실패: {type(e).__name__}: {e}")
        return

    # 3) Volume Profile (F6 핵심)
    try:
        print(f"\n  ── F6 (Volume Profile — 매물대 핵심) ──")
        for lb in [30, 90]:
            vp = compute_volume_profile(hist, lookback_days=lb, num_bins=50)
            poc, vah, val = vp.get("poc"), vp.get("vah"), vp.get("val")
            print(f"    lookback {lb}d: POC=${_f(poc, 2)} VAH=${_f(vah, 2)} VAL=${_f(val, 2)}")
            if poc and cur:
                dist_poc = (poc - cur) / cur * 100
                in_va = (val or 0) <= cur <= (vah or 0)
                print(f"      POC dist {dist_poc:+.2f}%  VAH dist {(vah-cur)/cur*100:+.2f}%  VAL dist {(val-cur)/cur*100:+.2f}%")
                print(f"      현재가 in value area: {in_va}")
    except Exception as e:
        print(f"  VP 실패: {type(e).__name__}: {e}")

    # 4) Options chain (Max Pain / Walls — F6 stack 의 보조)
    try:
        print(f"\n  ── 옵션 chain (Max Pain / Walls) ──")
        t0 = time.time()
        chain = get_realtime_chain(t, horizon_days=5)
        dt = time.time() - t0
        exp = next(iter(chain))
        strikes = sorted(chain[exp].keys())
        print(f"    fetch: {dt:.2f}s, exp={exp}, strikes={len(strikes)}")

        # IV null rate (Marketdata.app 데이터 품질)
        n_iv_null = sum(1 for s in chain[exp].values()
                          if s.get("iv") is None or (s.get("iv") != s.get("iv")))
        n_oi_zero = sum(1 for s in chain[exp].values()
                          if (s.get("call_oi", 0) + s.get("put_oi", 0)) == 0)
        print(f"    데이터 품질: IV null = {n_iv_null}/{len(strikes)} ({n_iv_null/max(len(strikes),1)*100:.0f}%), "
              f"OI=0 strikes = {n_oi_zero}/{len(strikes)} ({n_oi_zero/max(len(strikes),1)*100:.0f}%)")

        # Max pain
        min_pain, mp_strike = float("inf"), strikes[0]
        for s in strikes:
            pain = 0
            for K in strikes:
                if s > K:
                    pain += (s - K) * chain[exp][K].get("call_oi", 0) * 100
                elif s < K:
                    pain += (K - s) * chain[exp][K].get("put_oi", 0) * 100
            if pain < min_pain:
                min_pain, mp_strike = pain, s
        print(f"    Max Pain: ${mp_strike} (dist {(mp_strike-cur)/cur*100:+.2f}%)")

        # Walls (현재가 ±15% 내에서만)
        valid_strikes = [s for s in strikes if abs(s - cur)/cur < 0.15]
        if valid_strikes:
            cw = max(valid_strikes, key=lambda s: chain[exp][s].get("call_oi", 0))
            pw = max(valid_strikes, key=lambda s: chain[exp][s].get("put_oi", 0))
            cw_oi = chain[exp][cw].get("call_oi", 0)
            pw_oi = chain[exp][pw].get("put_oi", 0)
            print(f"    Call wall (저항, ±15% 제한): ${cw} OI {cw_oi:,} (dist {(cw-cur)/cur*100:+.2f}%)")
            print(f"    Put wall  (지지, ±15% 제한): ${pw} OI {pw_oi:,} (dist {(pw-cur)/cur*100:+.2f}%)")
        else:
            print(f"    ⚠️ 현재가 ±15% 내 strikes 없음 — 데이터 비정상")

        # ATM IV
        atm = min(strikes, key=lambda s: abs(s - cur))
        atm_iv = chain[exp][atm].get("iv")
        print(f"    ATM strike ${atm}: iv={_f(atm_iv, 3)}, call_oi={chain[exp][atm].get('call_oi')}, put_oi={chain[exp][atm].get('put_oi')}")

    except Exception as e:
        print(f"    옵션 chain 실패: {type(e).__name__}: {e}")


def main():
    print(f"실시간 데이터 점검 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"테스트 종목: {TEST_TICKERS}")
    for t in TEST_TICKERS:
        try:
            check_ticker(t)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"  {t} fatal: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

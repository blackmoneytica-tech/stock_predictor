"""실시간 옵션/매물대 데이터 fetch 정확성 점검.

확인 사항:
  1. options chain — Marketdata.app 1순위, yfinance fallback
     - OI/IV/strike/expiration 정상?
     - 가장 최근 만기 / strike 분포 합리적?
  2. IV rank — historic IV 대비 현재 IV 위치 (F5의 핵심)
  3. Max Pain / call/put walls — 자석/저항선
  4. 매물대 (volume profile) — POC / VAH / VAL / 핵심 가격대
  5. 데이터 신선도 — 마지막 fetch 시각 / API 지연
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import time
from datetime import datetime
import pandas as pd

# 모듈
from src.data.realtime_options import get_realtime_chain, get_atm_iv_realtime
from src.data.options_chain import get_options_chain as yf_get_chain, list_expirations as yf_exps
from src.data.price_feed import get_daily_ohlcv, get_current_price

def get_recent_history(ticker, days=60):
    from datetime import date, timedelta
    return get_daily_ohlcv(ticker, start=date.today() - timedelta(days=days * 2), end=date.today())


# 테스트할 종목 — F5 백테스트에 등장한 종목들 + 현재 워치 일부
TEST_TICKERS = ["NVDA", "AAPL", "AMD", "META", "CRCL", "MSTR"]


def section(t): print(f"\n{'='*60}\n{t}\n{'='*60}")


def check_options_for(ticker):
    section(f"[{ticker}] 옵션 chain 점검")
    cur_price = None
    try:
        cur_price = get_current_price(ticker)
        print(f"  현재가: ${cur_price:.2f}")
    except Exception as e:
        print(f"  현재가 fetch 실패: {e}")

    # 1) 만기 리스트
    try:
        exps = yf_exps(ticker)
        print(f"  yfinance 만기 리스트: {len(exps)}개")
        if exps:
            print(f"    첫 3개: {exps[:3]}, 마지막: {exps[-1]}")
    except Exception as e:
        print(f"  yfinance exps 실패: {e}")
        exps = []

    # 2) realtime chain (Marketdata 1순위)
    t0 = time.time()
    try:
        chain = get_realtime_chain(ticker, horizon_days=5)
        dt = time.time() - t0
        exp = next(iter(chain))
        strikes = sorted(chain[exp].keys())
        print(f"  realtime chain fetch: {dt:.2f}s, exp={exp}, strikes={len(strikes)}")
        if cur_price and strikes:
            in_range = [s for s in strikes if abs(s - cur_price) / cur_price < 0.20]
            print(f"    현재가 ±20% 내 strikes: {len(in_range)}개")

        # 샘플 strikes
        if cur_price and strikes:
            atm = min(strikes, key=lambda s: abs(s - cur_price))
            slot = chain[exp][atm]
            print(f"  ATM strike ${atm}:")
            print(f"    call_oi: {slot.get('call_oi')}, put_oi: {slot.get('put_oi')}")
            ci = slot.get('call_iv'); pi = slot.get('put_iv'); iv = slot.get('iv')
            def _f(x):
                if x is None: return "None"
                try:
                    if x != x: return "NaN"  # nan check
                    return f"{x:.3f}"
                except Exception:
                    return repr(x)
            print(f"    call_iv: {_f(ci)}, put_iv: {_f(pi)}, iv(평균): {_f(iv)}")
            # IV missing rate across all strikes
            n_iv_null = sum(1 for s in chain[exp].values() if s.get("iv") is None or (s.get("iv") != s.get("iv")))
            print(f"    IV missing rate: {n_iv_null}/{len(chain[exp])} strikes ({n_iv_null/len(chain[exp])*100:.0f}%)")

        # OI 분포 sanity
        total_call_oi = sum(slot.get("call_oi", 0) for slot in chain[exp].values())
        total_put_oi = sum(slot.get("put_oi", 0) for slot in chain[exp].values())
        print(f"  총 call OI: {total_call_oi:,}, 총 put OI: {total_put_oi:,}")
        print(f"  PC ratio: {(total_put_oi/total_call_oi if total_call_oi else 0):.3f}")
    except Exception as e:
        print(f"  realtime chain 실패: {type(e).__name__}: {e}")


def check_iv_rank(ticker):
    """IV rank = 현재 IV가 지난 52주 IV 분포의 percentile."""
    print(f"\n  IV rank 계산 (F5 핵심):")
    try:
        hist = get_recent_history(ticker, days=260)  # ~1년
        if hist is None or len(hist) < 30:
            print(f"    가격 history 부족 ({len(hist) if hist is not None else 0} bars)")
            return
        # HV 계산
        rets = hist["close"].pct_change().dropna()
        hv_60 = rets.rolling(60).std() * (252 ** 0.5)
        if hv_60.notna().sum() < 50:
            print(f"    HV60 sample 부족")
            return
        hv_min = hv_60.min()
        hv_max = hv_60.max()
        hv_cur = hv_60.iloc[-1]
        rank = (hv_cur - hv_min) / (hv_max - hv_min) * 100 if hv_max > hv_min else 50
        print(f"    52w HV 분포: min={hv_min:.3f}, max={hv_max:.3f}, 현재={hv_cur:.3f}")
        print(f"    HV rank (proxy for IV rank): {rank:.1f}")
        # IV 실측
        cur_price = get_current_price(ticker)
        iv_now = get_atm_iv_realtime(ticker, cur_price, horizon_days=30)
        print(f"    ATM IV (30d) 실측: {iv_now:.3f}")
        # 합리적 정도 — IV vs HV
        if iv_now > 0:
            ratio = iv_now / hv_cur if hv_cur > 0 else 1
            print(f"    IV/HV ratio: {ratio:.2f}  ({'IV 비쌈' if ratio > 1.2 else 'IV 저렴' if ratio < 0.8 else '비슷'})")
    except Exception as e:
        print(f"    IV rank 실패: {type(e).__name__}: {e}")


def check_max_pain(ticker):
    """Max pain — 현재가 대비 자석 효과 거리."""
    print(f"\n  Max Pain 계산:")
    try:
        chain = get_realtime_chain(ticker, horizon_days=5)
        exp = next(iter(chain))
        strikes = sorted(chain[exp].keys())
        if not strikes:
            print("    strikes 없음")
            return

        min_pain = float("inf")
        max_pain_strike = strikes[0]
        for strike in strikes:
            pain = 0
            for K in strikes:
                if strike > K:
                    pain += (strike - K) * chain[exp][K].get("call_oi", 0) * 100
                elif strike < K:
                    pain += (K - strike) * chain[exp][K].get("put_oi", 0) * 100
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = strike
        cur_price = get_current_price(ticker)
        dist_pct = (max_pain_strike - cur_price) / cur_price * 100
        print(f"    Max pain strike: ${max_pain_strike} (현재가 ${cur_price:.2f}, dist {dist_pct:+.2f}%)")

        # Call/Put walls — 최대 OI strike (저항/지지)
        call_wall_strike = max(strikes, key=lambda s: chain[exp][s].get("call_oi", 0))
        put_wall_strike = max(strikes, key=lambda s: chain[exp][s].get("put_oi", 0))
        call_wall_oi = chain[exp][call_wall_strike].get("call_oi", 0)
        put_wall_oi = chain[exp][put_wall_strike].get("put_oi", 0)
        print(f"    Call wall (저항): ${call_wall_strike} (OI {call_wall_oi:,}, dist {(call_wall_strike-cur_price)/cur_price*100:+.2f}%)")
        print(f"    Put wall (지지): ${put_wall_strike} (OI {put_wall_oi:,}, dist {(put_wall_strike-cur_price)/cur_price*100:+.2f}%)")
    except Exception as e:
        print(f"    Max pain 실패: {type(e).__name__}: {e}")


def check_volume_profile(ticker):
    """매물대 — 가격 × 거래량 분포. POC / VAH / VAL 계산."""
    print(f"\n  매물대 (Volume Profile) 점검:")
    try:
        hist = get_recent_history(ticker, days=60)
        if hist is None or len(hist) < 20:
            print(f"    가격 history 부족")
            return

        # 가격 × 거래량 분포 (typical price 사용)
        import numpy as np
        tp = (hist["high"] + hist["low"] + hist["close"]) / 3
        vol = hist["volume"]
        # 100 bin
        bins = np.linspace(tp.min(), tp.max(), 51)
        vp = pd.cut(tp, bins=bins).value_counts(sort=False)
        # 거래량 가중
        vp_weighted = pd.Series(0.0, index=vp.index)
        for i, (t_p, v) in enumerate(zip(tp, vol)):
            for j, interval in enumerate(vp.index):
                if t_p in interval:
                    vp_weighted.iloc[j] += v
                    break

        # POC — 가장 거래량 많은 bin
        poc_idx = vp_weighted.idxmax()
        poc_price = (poc_idx.left + poc_idx.right) / 2

        # Value Area (70% 거래량 포함)
        total_vol = vp_weighted.sum()
        sorted_idx = vp_weighted.sort_values(ascending=False).index
        cum_vol = 0
        va_bins = []
        for idx in sorted_idx:
            va_bins.append(idx)
            cum_vol += vp_weighted[idx]
            if cum_vol >= 0.70 * total_vol:
                break
        vah = max(va_bins, key=lambda x: x.right).right
        val = min(va_bins, key=lambda x: x.left).left

        cur_price = get_current_price(ticker)
        print(f"    POC (자석): ${poc_price:.2f} (현재 ${cur_price:.2f}, dist {(poc_price-cur_price)/cur_price*100:+.2f}%)")
        print(f"    VAH (저항): ${vah:.2f} (dist {(vah-cur_price)/cur_price*100:+.2f}%)")
        print(f"    VAL (지지): ${val:.2f} (dist {(val-cur_price)/cur_price*100:+.2f}%)")
        print(f"    60일 가격 range: ${tp.min():.2f} ~ ${tp.max():.2f}")
        print(f"    Value area 폭: ${vah-val:.2f} ({(vah-val)/cur_price*100:.1f}% of price)")
    except Exception as e:
        print(f"    매물대 계산 실패: {type(e).__name__}: {e}")


def main():
    print(f"실시간 데이터 점검 — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"테스트 종목: {TEST_TICKERS}")
    for t in TEST_TICKERS:
        try:
            check_options_for(t)
            check_iv_rank(t)
            check_max_pain(t)
            check_volume_profile(t)
        except KeyboardInterrupt:
            print("\n중단됨")
            break
        except Exception as e:
            print(f"  {t} fatal: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

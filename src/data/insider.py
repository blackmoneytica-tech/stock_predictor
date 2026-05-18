"""Insider data — SEC EDGAR Form 4 직접 fetch.

SEC EDGAR fair use:
- User-Agent 헤더 필수 (이메일 포함)
- 초당 10회 제한 (실제로는 더 보수적으로)

핵심: insider_ceiling 추출 (매도 집중 가격대 = 강한 저항).
검증 케이스: CRCL 5/12 Ostling director $132.06 매도 → $132 ceiling 확인.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from xml.etree import ElementTree as ET

import pandas as pd
import requests

from ._common import SEC_LIMITER, cache_load, cache_store, env, retry
from ._time import utcnow_naive


SEC_BASE = "https://www.sec.gov"
SEC_DATA_BASE = "https://data.sec.gov"

# SEC fair use: User-Agent에 실제 연락처 포함 권고
DEFAULT_UA = "stock_predictor research@local.dev"


def _ua() -> str:
    return env("SEC_USER_AGENT") or DEFAULT_UA


def _headers() -> Dict[str, str]:
    return {
        "User-Agent": _ua(),
        "Accept": "application/json",
    }


# ── 티커 → CIK 매핑 ───────────────────────────────────────────
_cik_cache: Dict[str, str] = {}


@retry(max_attempts=3, base_delay=1.0)
def _lookup_cik(ticker: str) -> Optional[str]:
    """티커 → 10자리 CIK (zero-padded)."""
    ticker = ticker.upper()
    if ticker in _cik_cache:
        return _cik_cache[ticker]

    SEC_LIMITER.wait()
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=_headers(), timeout=15)
    r.raise_for_status()
    data = r.json()

    # data: {idx: {cik_str, ticker, title}, ...}
    for _, row in data.items():
        if row.get("ticker", "").upper() == ticker:
            cik = str(row["cik_str"]).zfill(10)
            _cik_cache[ticker] = cik
            return cik
    return None


# ── Form 4 (개별 필링 파싱은 느리므로 인덱스만 사용) ─────────
@retry(max_attempts=3, base_delay=1.0)
def _list_form4_filings(cik: str, months_back: int = 6) -> List[Dict]:
    """CIK의 최근 Form 4 필링 목록 (메타데이터만).

    Returns:
        [{filing_date, accession, primary_doc_url}, ...]
    """
    SEC_LIMITER.wait()
    url = f"{SEC_DATA_BASE}/submissions/CIK{cik}.json"
    r = requests.get(url, headers=_headers(), timeout=15)
    r.raise_for_status()
    data = r.json()

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    cutoff = utcnow_naive() - timedelta(days=months_back * 31)
    out = []
    for i, form in enumerate(forms):
        if form != "4":
            continue
        try:
            d = datetime.strptime(dates[i], "%Y-%m-%d")
        except (ValueError, IndexError):
            continue
        if d < cutoff:
            continue
        acc = accessions[i].replace("-", "")
        primary = primary_docs[i] if i < len(primary_docs) else None
        out.append({
            "filing_date": dates[i],
            "accession": accessions[i],
            "primary_doc_url": (
                f"{SEC_BASE}/Archives/edgar/data/{int(cik)}/{acc}/{primary}"
                if primary else None
            ),
        })
    return out


# ── Form 4 XML 파싱 ──────────────────────────────────────────
@retry(max_attempts=2, base_delay=1.0)
def _parse_form4(primary_doc_url: str) -> List[Dict]:
    """단일 Form 4 XML에서 거래 내역 추출.

    Returns:
        [{date, code, shares, price, transaction_type}, ...]
        code 'A': acquired (매수), 'D': disposed (매도)
    """
    if not primary_doc_url:
        return []
    SEC_LIMITER.wait()
    r = requests.get(
        primary_doc_url,
        headers={"User-Agent": _ua(), "Accept": "*/*"},
        timeout=15,
    )
    if r.status_code != 200:
        return []

    txt = r.text
    # primary_doc는 HTML wrapper일 수도, XML일 수도 있음
    # Form 4 XML 본문 패턴
    try:
        # Strip HTML wrapper if present
        xml_start = txt.find("<?xml")
        if xml_start == -1:
            xml_start = txt.find("<ownershipDocument")
        if xml_start == -1:
            return []
        root = ET.fromstring(txt[xml_start:])
    except ET.ParseError:
        return []

    trades = []
    for tx in root.iter("nonDerivativeTransaction"):
        try:
            tx_date = tx.find(".//transactionDate/value")
            tx_code = tx.find(".//transactionCoding/transactionCode")
            shares = tx.find(".//transactionShares/value")
            price = tx.find(".//transactionPricePerShare/value")
            acq = tx.find(".//transactionAcquiredDisposedCode/value")
            trades.append({
                "date": tx_date.text if tx_date is not None else None,
                "code": acq.text if acq is not None else None,  # 'A' or 'D'
                "tx_code": tx_code.text if tx_code is not None else None,
                "shares": float(shares.text) if shares is not None else 0.0,
                "price": float(price.text) if (price is not None and price.text) else None,
            })
        except (AttributeError, ValueError):
            continue
    return trades


# ── 집계 함수 (모듈 6에서 호출) ──────────────────────────────
def get_insider_activity(ticker: str, months_back: int = 6) -> Dict:
    """6개월 인사이더 매수/매도 집계 + ceiling/floor 가격.

    Returns:
        {
            insider_buys_30d, insider_sells_30d,
            insider_buys_6m, insider_sells_6m,
            recent_sells_prices: [...],
            recent_buys_prices: [...],
        }
    """
    cache_key = f"insider:{ticker}:{months_back}"
    cached = cache_load("insider_activity", cache_key, ttl_seconds=86400)
    if cached is not None and not cached.empty:
        try:
            row = cached.iloc[0].to_dict()
            # numpy array `or []`는 ambiguity 발생 → 명시적 변환
            sells = row.get("recent_sells_prices")
            buys = row.get("recent_buys_prices")
            row["recent_sells_prices"] = list(sells) if sells is not None else []
            row["recent_buys_prices"] = list(buys) if buys is not None else []
            return row
        except Exception:
            pass

    cik = _lookup_cik(ticker)
    if cik is None:
        return _empty_insider()

    filings = _list_form4_filings(cik, months_back)
    all_trades: List[Dict] = []
    # 효율: 최근 30개 필링만 파싱 (성능 보호)
    for f in filings[:30]:
        all_trades.extend(_parse_form4(f["primary_doc_url"]))

    now = utcnow_naive()
    cutoff_30d = now - timedelta(days=30)

    buys_6m = sum(1 for t in all_trades if t["code"] == "A" and (t.get("price") or 0) > 0)
    sells_6m = sum(1 for t in all_trades if t["code"] == "D" and (t.get("price") or 0) > 0)

    def _within_30d(t):
        try:
            return datetime.strptime(t["date"], "%Y-%m-%d") >= cutoff_30d
        except (TypeError, ValueError):
            return False

    buys_30d = sum(1 for t in all_trades if t["code"] == "A" and _within_30d(t) and (t.get("price") or 0) > 0)
    sells_30d = sum(1 for t in all_trades if t["code"] == "D" and _within_30d(t) and (t.get("price") or 0) > 0)

    sell_prices = [t["price"] for t in all_trades if t["code"] == "D" and t.get("price")]
    buy_prices = [t["price"] for t in all_trades if t["code"] == "A" and t.get("price")]

    result = {
        "insider_buys_30d": buys_30d,
        "insider_sells_30d": sells_30d,
        "insider_buys_6m": buys_6m,
        "insider_sells_6m": sells_6m,
        "recent_sells_prices": sell_prices,
        "recent_buys_prices": buy_prices,
    }

    # 캐시 — list는 parquet object dtype으로 OK
    df = pd.DataFrame([result])
    cache_store("insider_activity", cache_key, df)
    return result


def _empty_insider() -> Dict:
    return {
        "insider_buys_30d": 0,
        "insider_sells_30d": 0,
        "insider_buys_6m": 0,
        "insider_sells_6m": 0,
        "recent_sells_prices": [],
        "recent_buys_prices": [],
    }


# ── 공매도 지표 (별도 source 필요 — placeholder) ─────────────
def get_short_interest(ticker: str) -> Dict:
    """공매도 지표.

    SEC가 매월 발표하지만 적시성 X. MVP는 placeholder + 향후
    Stockanalysis.com / Fintel scraper 통합 예정.
    """
    return {
        "short_interest_pct": 0.0,
        "days_to_cover": 0.0,
        "borrow_rate": 0.0,
        "short_interest_30d_change": 0.0,
    }

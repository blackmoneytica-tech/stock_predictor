"""통합 universe — 모든 backtest / live scan / UI 가 공유하는 종목 리스트.

세트:
  CORE_50        : 메가캡 50개 (가장 유동적, 옵션 chain 풍부)
  SECTOR_LEADERS : 11 섹터 대장주 ~40개
  GROWTH_MOMENTUM: 최근 모멘텀/그로스 ~40개
  TRACK_B        : 사용자 워치 + Track B push picks ~50개
  ETFS           : 인덱스/섹터 ETF ~20개

WATCH_FULL = 위 5개 합집합 (~200 unique).
WATCH_LIVE = 워치 + Track B 만 (~80, live scan 용 빠른 set).
"""
from __future__ import annotations

# ── 메가캡 / 가장 유동성 높은 50종 ──
CORE_50 = [
    # FAANG+
    "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "TSLA", "NFLX",
    # 반도체 large
    "AMD", "AVGO", "TSM", "INTC", "QCOM", "MU", "ARM", "TXN", "AMAT", "LRCX", "KLAC",
    # 빅테크 mid
    "ORCL", "CRM", "ADBE", "NOW", "PLTR", "SNOW", "DDOG", "MDB", "PANW", "CRWD",
    # 금융
    "JPM", "BAC", "GS", "MS", "WFC", "BLK", "V", "MA", "AXP",
    # 컨슈머
    "HD", "MCD", "SBUX", "NKE", "WMT", "COST", "DIS",
    # 헬스 / 산업
    "LLY", "UNH", "JNJ", "PFE", "MRK", "ABBV",
]

# ── 11 섹터 대표 ──
SECTOR_LEADERS = [
    # Energy
    "XOM", "CVX", "COP", "EOG", "MPC", "PSX",
    # Industrials
    "CAT", "BA", "DE", "GE", "HON", "RTX", "LMT", "NOC", "GD", "ETN",
    # Materials
    "LIN", "APD", "ECL", "SHW", "DOW",
    # Utilities
    "NEE", "DUK", "SO", "AEP", "EXC",
    # Real Estate
    "PLD", "AMT", "EQIX", "PSA", "O",
    # Comms
    "T", "VZ", "TMUS", "DIS",
    # Staples
    "PG", "KO", "PEP", "WMT", "COST", "PM",
    # Discretionary mid
    "MCD", "LOW", "TJX", "BKNG",
    # Healthcare extra
    "TMO", "DHR", "ABT", "ISRG", "ELV", "CVS",
    # Tech extra
    "CSCO", "IBM", "ACN", "INTU",
]

# ── 그로스/모멘텀 (실적/뉴스 catalyst 잘 일어남) ──
GROWTH_MOMENTUM = [
    # AI/cloud growth
    "ANET", "CDNS", "SNPS", "FTNT", "NET", "ZS", "WDAY",
    # Crypto-related
    "COIN", "MSTR", "MARA", "RIOT", "CLSK",
    # 핀테크 / payment
    "HOOD", "SOFI", "AFRM", "PYPL", "SQ", "ADYEY",
    # EV / clean energy
    "RIVN", "LCID", "NIO", "ENPH", "FSLR", "PLUG", "CHPT",
    # 스페이스 / 우주
    "RKLB", "ASTS", "LUNR", "BWXT", "SMR", "CEG", "VST",
    # 방산 small/mid
    "KTOS", "AVAV", "AIRJ",
    # 메디테크 / 바이오
    "VRTX", "REGN", "GILD", "MRNA", "BNTX", "RBLX",
    # 그로스 mid
    "APP", "DASH", "ABNB", "U", "PINS", "SPOT",
    # 신규 IPO / 모멘텀
    "CRCL", "CRWV", "RDDT", "TEMP", "TEM", "CHWY",
]

# ── 사용자 워치 + Track B push picks ──
TRACK_B = [
    # Track B HIGH conviction (2026-05-18 push)
    "BWXT", "CRDO", "ALAB", "STRL",
    # Track B Top 12 / mentioned in memory
    "ONTO", "RKLB", "MYRG", "COHR", "AEIS", "KTOS", "MOD", "VCEL",
    "AMKR", "MKSI", "STVN", "PKE", "GLBE", "KLIC", "AAOI",
    "HBM", "GPOR", "RELY",
    # Yesterday's re-analysis
    "LQDA", "ASTS",
    # 추가 small/mid cap from memory mentions
    "POWL", "AAOI", "AEIS", "RKLB", "WBI", "MOD",
    # Sweet spot 7종
    "CHYM", "BFLY", "GILT", "AIP", "PDFS",
    # 추가 Track B 백테스트 picks
    "DUOL", "HIMS", "ANET", "LSCC", "LRCX",
    "BKV",
    # Recent winners
    "AAOI", "POWL", "CRWV",
]

# ── ETF ──
ETFS = [
    # Index
    "SPY", "QQQ", "IWM", "DIA", "VTI",
    # Sector
    "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB", "XLC",
    # Theme
    "SMH", "SOXL", "XBI", "CIBR", "SKYY", "ITA", "PAVE", "ARKK", "GDX", "TLT",
]


def _unique(seq):
    """순서 유지 중복 제거."""
    seen = set()
    out = []
    for x in seq:
        x = x.strip().upper()
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out


WATCH_LIVE = _unique(TRACK_B + CORE_50[:30])
WATCH_FULL = _unique(CORE_50 + SECTOR_LEADERS + GROWTH_MOMENTUM + TRACK_B + ETFS)


def get_universe(name: str = "full") -> list[str]:
    """name in {core50, sector, growth, trackb, etfs, live, full}"""
    table = {
        "core50": CORE_50,
        "sector": SECTOR_LEADERS,
        "growth": GROWTH_MOMENTUM,
        "trackb": TRACK_B,
        "etfs": ETFS,
        "live": WATCH_LIVE,
        "full": WATCH_FULL,
    }
    if name not in table:
        raise ValueError(f"unknown universe '{name}', choose from {list(table)}")
    return _unique(table[name])


if __name__ == "__main__":
    for k in ["core50", "sector", "growth", "trackb", "etfs", "live", "full"]:
        u = get_universe(k)
        print(f"{k:<8}: {len(u):>4} tickers")
    print(f"\nfull unique: {len(get_universe('full'))}")

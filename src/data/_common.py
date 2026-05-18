"""Data layer 공통 유틸 — 캐시 / rate limit / retry / 환경 변수.

캐시:
  data/cache/{namespace}_{key_hash}.parquet  (1h~24h TTL)
Rate limit:
  분당 60건 (sliding window) — yfinance / Yahoo 비공식 API 보호
Retry:
  Exponential backoff (1s → 2s → 4s, max 3회)
"""
from __future__ import annotations

import functools
import hashlib
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


# ── 경로 ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "data" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def env(key: str, default: Optional[str] = None) -> Optional[str]:
    """환경 변수 (.env 또는 OS env)."""
    return os.environ.get(key, default)


# ── 캐시 (Parquet, TTL 기반) ─────────────────────────────────
def _cache_path(namespace: str, key: str) -> Path:
    h = hashlib.md5(key.encode()).hexdigest()[:16]
    return CACHE_DIR / f"{namespace}_{h}.parquet"


def cache_load(namespace: str, key: str, ttl_seconds: int) -> Optional[pd.DataFrame]:
    """캐시에서 DataFrame 로드. TTL 만료 시 None."""
    path = _cache_path(namespace, key)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > ttl_seconds:
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def cache_store(namespace: str, key: str, df: pd.DataFrame) -> None:
    """DataFrame을 parquet으로 저장 (실패 silent)."""
    if df is None or df.empty:
        return
    path = _cache_path(namespace, key)
    try:
        df.to_parquet(path, compression="snappy")
    except Exception:
        # 캐시 실패는 비치명적
        pass


# ── Rate Limiter (sliding window) ─────────────────────────────
class RateLimiter:
    """분당 N건 제한. thread-safe."""

    def __init__(self, max_per_minute: int = 60):
        self.max_per_minute = max_per_minute
        self.window = deque()
        self.lock = threading.Lock()

    def wait(self) -> None:
        now = time.time()
        with self.lock:
            while self.window and self.window[0] < now - 60:
                self.window.popleft()
            if len(self.window) >= self.max_per_minute:
                sleep_s = 60 - (now - self.window[0]) + 0.05
                if sleep_s > 0:
                    time.sleep(sleep_s)
                now = time.time()
                while self.window and self.window[0] < now - 60:
                    self.window.popleft()
            self.window.append(now)


YFINANCE_LIMITER = RateLimiter(max_per_minute=60)
FRED_LIMITER = RateLimiter(max_per_minute=120)
POLYGON_FREE_LIMITER = RateLimiter(max_per_minute=5)
SEC_LIMITER = RateLimiter(max_per_minute=10)  # SEC fair use: 10/sec but 보수적으로


# ── Retry (exponential backoff) ───────────────────────────────
def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    exceptions: tuple = (Exception,),
) -> Callable:
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_attempts - 1:
                        time.sleep(base_delay * (2 ** attempt))
            raise last_exc
        return wrapper
    return decorator


# ── 날짜 정규화 ────────────────────────────────────────────────
def normalize_date(d: Any) -> str:
    """date | datetime | 'YYYY-MM-DD' → 'YYYY-MM-DD' 문자열."""
    if isinstance(d, str):
        return d[:10]
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    return d.isoformat()[:10]

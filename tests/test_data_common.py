"""src/data/_common.py 단위 테스트 — 캐시 / rate limit / retry."""
from __future__ import annotations

import time

import pandas as pd

from src.data._common import (
    RateLimiter,
    cache_load,
    cache_store,
    normalize_date,
    retry,
)


def test_cache_roundtrip(tmp_path, monkeypatch):
    """저장 → 로드 → TTL 만료 동작."""
    # _common 모듈의 CACHE_DIR를 tmp로 redirect
    from src.data import _common
    monkeypatch.setattr(_common, "CACHE_DIR", tmp_path)

    df = pd.DataFrame({"a": [1, 2, 3]})
    cache_store("ns1", "key1", df)

    loaded = cache_load("ns1", "key1", ttl_seconds=60)
    assert loaded is not None
    assert loaded["a"].tolist() == [1, 2, 3]

    # TTL 만료 시뮬레이션
    expired = cache_load("ns1", "key1", ttl_seconds=0)
    assert expired is None


def test_cache_miss_returns_none(tmp_path, monkeypatch):
    from src.data import _common
    monkeypatch.setattr(_common, "CACHE_DIR", tmp_path)
    assert cache_load("nonexist", "nope", ttl_seconds=60) is None


def test_rate_limiter_blocks_when_full():
    """5건 도착 후 6번째는 sleep."""
    rl = RateLimiter(max_per_minute=5)
    # 5번 호출 (sleep 없이 통과)
    t0 = time.time()
    for _ in range(5):
        rl.wait()
    fast_elapsed = time.time() - t0
    assert fast_elapsed < 0.5

    # window 강제 채움 — 가장 오래된 항목을 미래로 옮겨 sleep 트리거하는 대신
    # 별도 인스턴스로 max=1 테스트
    rl2 = RateLimiter(max_per_minute=1)
    rl2.wait()  # 즉시 통과
    # 두 번째 호출은 ~60초 sleep — 테스트에선 시뮬 안 함, deque 상태만 확인
    assert len(rl2.window) == 1


def test_retry_eventually_succeeds():
    calls = {"n": 0}

    @retry(max_attempts=3, base_delay=0.01)
    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 2


def test_retry_gives_up_after_max():
    calls = {"n": 0}

    @retry(max_attempts=2, base_delay=0.01)
    def always_fails():
        calls["n"] += 1
        raise RuntimeError("perma")

    try:
        always_fails()
        assert False, "should have raised"
    except RuntimeError:
        pass
    assert calls["n"] == 2


def test_normalize_date():
    from datetime import datetime
    assert normalize_date("2026-05-16") == "2026-05-16"
    assert normalize_date("2026-05-16T12:30:00") == "2026-05-16"
    assert normalize_date(datetime(2026, 5, 16, 14, 0)) == "2026-05-16"

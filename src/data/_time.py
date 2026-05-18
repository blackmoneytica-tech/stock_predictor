"""시간 헬퍼 — Python 3.12+ utcnow() deprecation 회피."""
from __future__ import annotations

from datetime import datetime, timezone


def utcnow_naive() -> datetime:
    """tz-naive UTC datetime (기존 datetime.utcnow() 대체).

    pandas/parquet 호환을 위해 naive 유지.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)

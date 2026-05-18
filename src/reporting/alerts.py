"""Alerts — 이메일 + Telegram. Phase 6에서 구현."""
from __future__ import annotations


def send_email(subject: str, body: str) -> bool:
    raise NotImplementedError("Phase 6 — smtplib 통합")


def send_telegram(message: str) -> bool:
    raise NotImplementedError("Phase 6 — Telegram Bot API 통합")

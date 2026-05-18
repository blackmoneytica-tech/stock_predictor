"""CLI dashboard — analyze + backtest 명령어.

Usage:
    python -m src.reporting.cli_dashboard analyze CRCL
    python -m src.reporting.cli_dashboard backtest CRCL --start 2025-06-01 --end 2026-05-15
"""
from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stock Prediction System v2.0 CLI",
    )
    subparsers = parser.add_subparsers(dest='command', required=True)

    analyze = subparsers.add_parser('analyze', help='단일 종목 분석')
    analyze.add_argument('ticker', type=str)
    analyze.add_argument('--horizon', type=int, default=5)

    backtest = subparsers.add_parser('backtest', help='walk-forward 백테스트')
    backtest.add_argument('ticker', type=str)
    backtest.add_argument('--start', required=True, help='YYYY-MM-DD')
    backtest.add_argument('--end', required=True, help='YYYY-MM-DD')
    backtest.add_argument('--horizon', type=int, default=5)

    args = parser.parse_args()

    if args.command == 'analyze':
        print(f"[TODO Phase 6] analyze {args.ticker} h={args.horizon}d")
    elif args.command == 'backtest':
        print(f"[TODO Phase 6] backtest {args.ticker} {args.start} ~ {args.end}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())

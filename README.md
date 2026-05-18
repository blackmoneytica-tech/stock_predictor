# Stock Prediction System v2.0

다중 시그널 통합 주가 예측 시스템 — 8개 분석 모듈 + Bayesian 결합 + 5-시나리오 + 75% 신뢰도 hard cap.

## 핵심 원칙

1. **인지 편향 사전 교정** — 호재 자동 +70% 가중치 금지
2. **매크로 데이터 lag 1~3일** 반영
3. **카탈리스트 직후 sell-news 80% base rate**
4. **Parabolic +30% 후 mean reversion**
5. **75% 신뢰도 hard cap** — 그 이상은 data leakage 의심

## 디렉토리

```
stock_predictor/
├── pyproject.toml        # 패키지 + dependencies
├── README.md
├── .env.example          # API 키 템플릿
├── config/
│   ├── default.yaml      # 시스템 기본 설정
│   └── tickers.yaml      # watchlist
├── src/
│   ├── types.py          # Direction / ModuleOutput / Scenario / PredictionResult 등
│   ├── system.py         # StockPredictionSystem 메인
│   ├── data/             # Phase 2: yfinance + Polygon + FRED + SEC ingestion
│   ├── modules/          # 8개 분석 모듈 (technical/options/sentiment/macro/...)
│   ├── strategy/         # aggregator + action_engine
│   ├── backtest/         # walk-forward + calibration
│   └── reporting/        # CLI dashboard + alerts
├── tests/                # pytest
├── notebooks/            # 탐색 + 백테스트 분석
└── data/
    ├── cache/            # parquet 캐시 (1h TTL)
    └── results/          # daily report + predictions.csv
```

## 빠른 시작

```bash
# 의존성 설치 (editable mode)
python -m pip install -e ".[dev]"

# 환경 변수 설정
cp .env.example .env
# .env 편집 후 FRED_API_KEY 등 채우기

# 단일 종목 분석 (Phase 8 완료 후 사용 가능)
python -m src.reporting.cli_dashboard analyze CRCL

# 백테스트
python -m src.reporting.cli_dashboard backtest CRCL --start 2025-06-01 --end 2026-05-15
```

## 진행 단계

- [x] **Phase 1-2**: 프로젝트 구조 + 데이터 수집 (yfinance + FRED + SEC + AV + Finnhub)
- [x] **11 모듈**: 8 기본 + DemandSupply + OrderBlock + Trend
- [x] **Sector breadth + Finnhub estimates** 통합
- [x] **CRCL 이번주 walk-forward**: 4/5 = 80% directional
- [x] **변동성 universe backtest**: 155 예측 / 46.5% (메가캡 36% 대비 +10.5%p)
- [x] **Streamlit UI**: `streamlit run src/ui/dashboard.py` (또는 run_ui.cmd 더블클릭)
- [ ] **Phase 5 정식**: 옵션 historical + 6m-1y walk-forward + calibration
- [ ] **웹서비스 통합**: FastAPI + trade-journal CF Pages

## UI 실행

```bash
# Windows
run_ui.cmd

# 또는 직접
python -m streamlit run src/ui/dashboard.py
```

브라우저 → http://localhost:8501

**페이지**:
- 🔍 종목 분석: 라이브 11모듈 score / 시나리오 / 매물대 / 옵션 / 액션
- 📈 CRCL Walk-Forward: 이번주 일별 예측 vs 실제
- 📊 다종목 Backtest: 변동성/메가캡 universe 결과 + ticker별 acc

상세 가이드: `CLAUDE_CODE_USAGE_GUIDE.md`, 명세서: `STOCK_PREDICTION_SYSTEM_SPEC.md`.

## 주의

- 신뢰도 75% 이상 출력 시 lookahead bias / overfitting 의심
- 시스템 = 의사결정 보조 도구 (100% 정답 아님)
- 분할 매도 + 명확한 손절 + 옵션 헷지 (Protective Put / Collar) 병행 필수

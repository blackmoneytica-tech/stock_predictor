"""Smoke test — Phase 1 패키지 import 검증.

실제 모듈 테스트는 Phase 3에서 mock data로 작성.
"""


def test_imports():
    from src import types
    from src.modules import (
        AnalysisModule,
        CatalystCalendarModule,
        InsiderSmartMoneyModule,
        MacroCorrelationModule,
        MeanReversionModule,
        OptionsFlowModule,
        SentimentModule,
        ShortSqueezeModule,
        TechnicalAnalysisModule,
    )
    from src.strategy import ActionEngine, SignalAggregator
    from src.system import StockPredictionSystem

    assert types.Direction.BULL.value == 1
    assert SignalAggregator.CONFIDENCE_HARD_CAP == 0.75


def test_system_constructor():
    """11개 모듈 (8 + demand_supply + order_block + trend) 등록."""
    from src.system import StockPredictionSystem

    system = StockPredictionSystem()
    assert len(system.modules) == 11
    for required in ('demand_supply', 'order_block', 'trend'):
        assert required in system.modules

    weights = {name: m.weight for name, m in system.modules.items()}
    assert weights['options'] >= weights['technical']
    assert weights['macro'] >= weights['technical']
    assert weights['demand_supply'] == 0.10
    assert weights['trend'] == 0.10


def test_aggregator_normalizes_weights():
    """Dynamic weights 결과는 합 = 1.0이어야 함."""
    from src.strategy import SignalAggregator

    agg = SignalAggregator()
    context = {'current_price': 100.0}
    dynamic = agg._get_dynamic_weights({}, context)
    assert abs(sum(dynamic.values()) - 1.0) < 1e-9

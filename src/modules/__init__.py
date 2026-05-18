"""8개 분석 모듈 — 각 모듈은 AnalysisModule 베이스를 상속."""

from .base import AnalysisModule
from .technical import TechnicalAnalysisModule
from .options_flow import OptionsFlowModule
from .sentiment import SentimentModule
from .macro_corr import MacroCorrelationModule
from .catalyst import CatalystCalendarModule
from .insider import InsiderSmartMoneyModule
from .mean_reversion import MeanReversionModule
from .short_squeeze import ShortSqueezeModule
from .demand_supply import DemandSupplyModule
from .order_block import OrderBlockModule
from .trend import TrendFollowingModule

__all__ = [
    "AnalysisModule",
    "TechnicalAnalysisModule",
    "OptionsFlowModule",
    "SentimentModule",
    "MacroCorrelationModule",
    "CatalystCalendarModule",
    "InsiderSmartMoneyModule",
    "MeanReversionModule",
    "ShortSqueezeModule",
    "DemandSupplyModule",
    "OrderBlockModule",
    "TrendFollowingModule",
]

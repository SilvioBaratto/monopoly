"""Strategy implementations for Monopoly game decisions."""

from monopoly.strategies.base import Strategy
from monopoly.strategies.buy_everything import BuyEverything
from monopoly.strategies.buy_nothing import BuyNothing
from monopoly.strategies.color_targeted import ColorTargeted
from monopoly.strategies.jail_camper import JailCamper
from monopoly.strategies.three_houses_rush import ThreeHousesRush
from monopoly.strategies.trader import Trader
from monopoly.strategies.types import BuildOrder, JailDecision, SellOrder, TradeOffer

__all__ = [
    # Base types
    "Strategy",
    "JailDecision",
    "BuildOrder",
    "SellOrder",
    "TradeOffer",
    # Concrete strategies
    "BuyEverything",
    "BuyNothing",
    "ColorTargeted",
    "JailCamper",
    "ThreeHousesRush",
    "Trader",
]

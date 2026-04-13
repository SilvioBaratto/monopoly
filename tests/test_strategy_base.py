"""Tests for the Strategy ABC, JailDecision, BuildOrder, and TradeOffer.

TDD: these tests were written before the split of base.py → types.py.
They drive the structural requirement that types live in strategies/types.py
and are re-exported from strategies/base.py and strategies/__init__.py.
"""

from __future__ import annotations

import pytest
from dataclasses import fields as dc_fields


# ---------------------------------------------------------------------------
# JailDecision
# ---------------------------------------------------------------------------


class TestJailDecision:
    def test_importable_from_types(self):
        from monopoly.strategies.types import JailDecision

        assert JailDecision is not None

    def test_importable_from_base(self):
        from monopoly.strategies.base import JailDecision

        assert JailDecision is not None

    def test_importable_from_package(self):
        from monopoly.strategies import JailDecision

        assert JailDecision is not None

    def test_has_exactly_three_members(self):
        from monopoly.strategies.types import JailDecision

        assert len(JailDecision) == 3

    def test_members_are_roll_doubles_pay_fine_use_goojf(self):
        from monopoly.strategies.types import JailDecision

        names = {m.name for m in JailDecision}
        assert names == {"ROLL_DOUBLES", "PAY_FINE", "USE_GOOJF"}

    def test_is_string_enum(self):
        from monopoly.strategies.types import JailDecision

        assert JailDecision.PAY_FINE == "pay_fine"
        assert JailDecision.USE_GOOJF == "use_goojf"
        assert JailDecision.ROLL_DOUBLES == "roll_doubles"


# ---------------------------------------------------------------------------
# BuildOrder
# ---------------------------------------------------------------------------


class TestBuildOrder:
    def test_importable_from_types(self):
        from monopoly.strategies.types import BuildOrder

        assert BuildOrder is not None

    def test_importable_from_base(self):
        from monopoly.strategies.base import BuildOrder

        assert BuildOrder is not None

    def test_importable_from_package(self):
        from monopoly.strategies import BuildOrder

        assert BuildOrder is not None

    def test_field_names(self):
        from monopoly.strategies.types import BuildOrder

        names = {f.name for f in dc_fields(BuildOrder)}
        assert "position" in names
        assert "count" in names

    def test_field_types(self):
        from monopoly.strategies.types import BuildOrder

        # With `from __future__ import annotations`, types are stored as strings
        type_map = {f.name: f.type for f in dc_fields(BuildOrder)}
        assert type_map["position"] in (int, "int")
        assert type_map["count"] in (int, "int")

    def test_construction(self):
        from monopoly.strategies.types import BuildOrder

        order = BuildOrder(position=6, count=2)
        assert order.position == 6
        assert order.count == 2


# ---------------------------------------------------------------------------
# SellOrder
# ---------------------------------------------------------------------------


class TestSellOrder:
    def test_importable_from_types(self):
        from monopoly.strategies.types import SellOrder

        assert SellOrder is not None

    def test_field_names(self):
        from monopoly.strategies.types import SellOrder

        names = {f.name for f in dc_fields(SellOrder)}
        assert "position" in names
        assert "count" in names

    def test_construction(self):
        from monopoly.strategies.types import SellOrder

        order = SellOrder(position=11, count=3)
        assert order.position == 11
        assert order.count == 3


# ---------------------------------------------------------------------------
# TradeOffer
# ---------------------------------------------------------------------------


class TestTradeOffer:
    def test_importable_from_types(self):
        from monopoly.strategies.types import TradeOffer

        assert TradeOffer is not None

    def test_importable_from_base(self):
        from monopoly.strategies.base import TradeOffer

        assert TradeOffer is not None

    def test_importable_from_package(self):
        from monopoly.strategies import TradeOffer

        assert TradeOffer is not None

    def test_field_names(self):
        from monopoly.strategies.types import TradeOffer

        names = {f.name for f in dc_fields(TradeOffer)}
        assert "offered_positions" in names
        assert "requested_positions" in names
        assert "cash_offered" in names
        assert "cash_requested" in names

    def test_cash_fields_default_to_zero(self):
        from monopoly.strategies.types import TradeOffer

        offer = TradeOffer(offered_positions=[1], requested_positions=[3])
        assert offer.cash_offered == 0
        assert offer.cash_requested == 0

    def test_construction_with_all_fields(self):
        from monopoly.strategies.types import TradeOffer

        offer = TradeOffer(
            offered_positions=[1, 3],
            requested_positions=[6, 8, 9],
            cash_offered=100,
            cash_requested=0,
        )
        assert offer.offered_positions == [1, 3]
        assert offer.requested_positions == [6, 8, 9]
        assert offer.cash_offered == 100


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------


class TestStrategyABC:
    def test_cannot_instantiate_strategy_directly(self):
        from monopoly.strategies.base import Strategy

        with pytest.raises(TypeError):
            Strategy()  # type: ignore[abstract]

    def test_importable_from_package(self):
        from monopoly.strategies import Strategy

        assert Strategy is not None

    def test_concrete_subclass_must_implement_all_methods(self):
        """A class that omits even one abstract method cannot be instantiated."""
        from monopoly.strategies.base import Strategy

        class Partial(Strategy):
            def should_buy_property(self, player, square, game_state):
                return False

            # Missing all other abstract methods

        with pytest.raises(TypeError):
            Partial()  # type: ignore[abstract]

    def test_concrete_subclass_works_when_all_methods_implemented(self):
        """A fully-implemented subclass can be instantiated and called."""
        from monopoly.strategies.base import (
            Strategy,
            BuildOrder,
            JailDecision,
            TradeOffer,
        )

        class AlwaysBuy(Strategy):
            def should_buy_property(self, player, square, game_state) -> bool:
                return True

            def choose_properties_to_build(
                self, player, game_state
            ) -> list[BuildOrder]:
                return []

            def get_jail_decision(self, player, game_state) -> JailDecision:
                return JailDecision.PAY_FINE

            def choose_properties_to_mortgage(
                self, player, amount_needed, game_state
            ) -> list[int]:
                return []

            def should_accept_trade(self, player, trade_offer, game_state) -> bool:
                return False

            def propose_trade(self, player, game_state) -> TradeOffer | None:
                return None

        strategy = AlwaysBuy()
        assert isinstance(strategy, Strategy)

    def test_concrete_subclass_is_substitutable_for_strategy(self):
        """LSP: a concrete strategy IS-A Strategy."""
        from monopoly.strategies.base import Strategy
        from monopoly.strategies.buy_nothing import BuyNothing
        from monopoly.strategies.buy_everything import BuyEverything

        assert isinstance(BuyNothing(), Strategy)
        assert isinstance(BuyEverything(), Strategy)

    def test_strategy_has_six_abstract_methods(self):
        """The ABC exposes exactly the 6 decision methods the engine needs."""
        from monopoly.strategies.base import Strategy

        expected = {
            "should_buy_property",
            "choose_properties_to_build",
            "get_jail_decision",
            "choose_properties_to_mortgage",
            "should_accept_trade",
            "propose_trade",
        }
        assert Strategy.__abstractmethods__ == expected

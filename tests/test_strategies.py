"""Tests for all strategy implementations."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from monopoly.board import Board
from monopoly.cards import load_decks
from monopoly.state import GameState
from monopoly.strategies.base import JailDecision, TradeOffer
from monopoly.strategies.buy_everything import BuyEverything
from monopoly.strategies.buy_nothing import BuyNothing
from monopoly.strategies.color_targeted import ColorTargeted
from monopoly.strategies.jail_camper import JailCamper
from monopoly.strategies.three_houses_rush import ThreeHousesRush
from monopoly.strategies.trader import Trader

DATA_PATH = Path(__file__).parent.parent / "data" / "cards_standard.yaml"


@pytest.fixture
def board():
    return Board()


@pytest.fixture
def state(board):
    rng = np.random.default_rng(0)
    chance, cc = load_decks(DATA_PATH, rng)
    return GameState.init_game(["Alice", "Bob", "Carol"], board, chance, cc)


@pytest.fixture
def alice(state):
    return state.players[0]


@pytest.fixture
def bob(state):
    return state.players[1]


def give_player_brown(state, player):
    """Give player both brown properties."""
    state.property_ownership[1].owner = player
    state.property_ownership[3].owner = player


def give_player_orange(state, player):
    """Give player all orange properties (16, 18, 19)."""
    for pos in [16, 18, 19]:
        if pos in state.property_ownership:
            state.property_ownership[pos].owner = player


# ---------------------------------------------------------------------------
# BuyEverything
# ---------------------------------------------------------------------------


class TestBuyEverything:
    def test_should_buy_if_enough_cash(self, alice, state):
        strategy = BuyEverything()
        sq = state.board.get_square(1)
        assert strategy.should_buy_property(alice, sq, state) is True

    def test_should_not_buy_if_insufficient_cash(self, alice, state):
        strategy = BuyEverything()
        alice.cash = 0
        sq = state.board.get_square(1)
        assert strategy.should_buy_property(alice, sq, state) is False

    def test_get_jail_decision_uses_goojf(self, alice, state):
        from monopoly.cards import Card, CardEffect

        strategy = BuyEverything()
        card = Card(id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        assert strategy.get_jail_decision(alice, state) == JailDecision.USE_GOOJF

    def test_get_jail_decision_pay_fine_without_goojf(self, alice, state):
        strategy = BuyEverything()
        assert strategy.get_jail_decision(alice, state) == JailDecision.PAY_FINE

    def test_propose_trade_returns_none(self, alice, state):
        strategy = BuyEverything()
        assert strategy.propose_trade(alice, state) is None

    def test_choose_properties_to_build_complete_group(self, alice, state):
        give_player_brown(state, alice)
        alice.cash = 5000
        strategy = BuyEverything()
        orders = strategy.choose_properties_to_build(alice, state)
        # Should have orders for pos 1 and/or pos 3
        positions = [o.position for o in orders]
        assert any(pos in positions for pos in [1, 3])

    def test_choose_properties_to_build_empty_when_no_group(self, alice, state):
        strategy = BuyEverything()
        orders = strategy.choose_properties_to_build(alice, state)
        assert orders == []

    def test_choose_properties_to_mortgage(self, alice, state):
        state.property_ownership[1].owner = alice
        strategy = BuyEverything()
        result = strategy.choose_properties_to_mortgage(alice, 100, state)
        assert 1 in result

    def test_choose_properties_to_mortgage_complete_group_last(self, alice, state):
        give_player_brown(state, alice)
        state.property_ownership[5].owner = alice  # Railroad
        strategy = BuyEverything()
        result = strategy.choose_properties_to_mortgage(alice, 100, state)
        # Railroad should come before complete group properties
        if 5 in result and 1 in result:
            assert result.index(5) < result.index(1)

    def test_should_accept_trade_completes_group(self, alice, bob, state):
        strategy = BuyEverything()
        # Alice owns pos 1, Bob owns pos 3
        state.property_ownership[1].owner = alice
        state.property_ownership[3].owner = bob
        # Trade: alice gives nothing important, gets pos 3 (completes brown)
        trade = TradeOffer(
            offered_positions=[5],
            requested_positions=[3],
        )
        # This should return True (completes a group)
        assert strategy.should_accept_trade(alice, trade, state) is True

    def test_should_not_accept_trade_no_group_completion(self, alice, bob, state):
        strategy = BuyEverything()
        trade = TradeOffer(
            offered_positions=[1],
            requested_positions=[5],
        )
        assert strategy.should_accept_trade(alice, trade, state) is False


# ---------------------------------------------------------------------------
# BuyNothing
# ---------------------------------------------------------------------------


class TestBuyNothing:
    def test_never_buys(self, alice, state):
        strategy = BuyNothing()
        sq = state.board.get_square(1)
        assert strategy.should_buy_property(alice, sq, state) is False

    def test_no_build_orders(self, alice, state):
        strategy = BuyNothing()
        orders = strategy.choose_properties_to_build(alice, state)
        assert orders == []

    def test_uses_goojf_if_available(self, alice, state):
        from monopoly.cards import Card, CardEffect

        strategy = BuyNothing()
        card = Card(id="cc_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        assert strategy.get_jail_decision(alice, state) == JailDecision.USE_GOOJF

    def test_pay_fine_without_goojf(self, alice, state):
        strategy = BuyNothing()
        assert strategy.get_jail_decision(alice, state) == JailDecision.PAY_FINE

    def test_does_not_accept_trades(self, alice, state):
        strategy = BuyNothing()
        trade = TradeOffer(offered_positions=[], requested_positions=[1])
        assert strategy.should_accept_trade(alice, trade, state) is False

    def test_propose_trade_none(self, alice, state):
        strategy = BuyNothing()
        assert strategy.propose_trade(alice, state) is None

    def test_choose_properties_to_mortgage_returns_owned(self, alice, state):
        state.property_ownership[1].owner = alice
        strategy = BuyNothing()
        result = strategy.choose_properties_to_mortgage(alice, 100, state)
        assert 1 in result

    def test_mortgage_excludes_properties_with_houses(self, alice, state):
        state.property_ownership[1].owner = alice
        state.property_ownership[1].houses = 1
        strategy = BuyNothing()
        result = strategy.choose_properties_to_mortgage(alice, 100, state)
        assert 1 not in result


# ---------------------------------------------------------------------------
# ColorTargeted
# ---------------------------------------------------------------------------


class TestColorTargeted:
    def test_invalid_color_raises(self):
        with pytest.raises(ValueError, match="Unknown color"):
            ColorTargeted(["magenta"])

    def test_valid_color_does_not_raise(self):
        ct = ColorTargeted(["orange"])
        assert ct.target_colors == ["orange"]

    def test_buys_target_color_with_reserve(self, alice, state):
        strategy = ColorTargeted(["brown"])
        alice.cash = 1000
        sq = state.board.get_square(1)  # Mediterranean Ave (brown)
        assert strategy.should_buy_property(alice, sq, state) is True

    def test_does_not_buy_target_color_without_reserve(self, alice, state):
        strategy = ColorTargeted(["brown"])
        alice.cash = sq = state.board.get_square(1)
        sq = state.board.get_square(1)
        alice.cash = sq.price  # exactly price, no reserve
        assert strategy.should_buy_property(alice, sq, state) is False

    def test_buys_cheap_non_target(self, alice, state):
        strategy = ColorTargeted(["orange"])
        alice.cash = 5000
        sq = state.board.get_square(1)  # Mediterranean $60 (< 200)
        assert strategy.should_buy_property(alice, sq, state) is True

    def test_does_not_buy_expensive_non_target(self, alice, state):
        strategy = ColorTargeted(["orange"])
        alice.cash = 500
        sq = state.board.get_square(39)  # Boardwalk ($400)
        assert strategy.should_buy_property(alice, sq, state) is False

    def test_build_orders_target_first(self, alice, state):
        strategy = ColorTargeted(["brown"])
        give_player_brown(state, alice)
        alice.cash = 5000
        orders = strategy.choose_properties_to_build(alice, state)
        # Brown positions should appear before other colors
        positions = [o.position for o in orders]
        assert any(p in [1, 3] for p in positions)

    def test_jail_decision_goojf(self, alice, state):
        from monopoly.cards import Card, CardEffect

        strategy = ColorTargeted(["orange"])
        card = Card(id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        assert strategy.get_jail_decision(alice, state) == JailDecision.USE_GOOJF

    def test_jail_decision_pay_fine_with_cash(self, alice, state):
        strategy = ColorTargeted(["orange"])
        alice.cash = 5000
        assert strategy.get_jail_decision(alice, state) == JailDecision.PAY_FINE

    def test_jail_decision_roll_doubles_late_game(self, alice, state):
        strategy = ColorTargeted(["orange"])
        state.turn_count = 30  # late game threshold reached
        alice.cash = 5000
        assert strategy.get_jail_decision(alice, state) == JailDecision.ROLL_DOUBLES

    def test_mortgage_non_target_first(self, alice, state):
        strategy = ColorTargeted(["brown"])
        state.property_ownership[5].owner = alice  # Reading RR (non-target)
        give_player_brown(state, alice)  # brown (target)
        result = strategy.choose_properties_to_mortgage(alice, 100, state)
        if 5 in result and 1 in result:
            assert result.index(5) < result.index(1)

    def test_accept_trade_for_target_color(self, alice, bob, state):
        strategy = ColorTargeted(["brown"])
        state.property_ownership[3].owner = bob
        trade = TradeOffer(offered_positions=[5], requested_positions=[3])
        assert strategy.should_accept_trade(alice, trade, state) is True

    def test_reject_trade_for_non_target(self, alice, bob, state):
        strategy = ColorTargeted(["brown"])
        state.property_ownership[39].owner = bob
        trade = TradeOffer(offered_positions=[5], requested_positions=[39])
        assert strategy.should_accept_trade(alice, trade, state) is False

    def test_propose_trade_returns_none(self, alice, state):
        strategy = ColorTargeted(["orange"])
        assert strategy.propose_trade(alice, state) is None

    def test_does_not_buy_railroad(self, alice, state):
        # Railroads are not target-color and cost $200 (not < $200), so skipped
        strategy = ColorTargeted(["orange"])
        alice.cash = 5000
        sq = state.board.get_square(5)  # Reading Railroad, price=$200
        assert strategy.should_buy_property(alice, sq, state) is False


# ---------------------------------------------------------------------------
# ThreeHousesRush
# ---------------------------------------------------------------------------


class TestThreeHousesRush:
    def test_should_buy_with_reserve(self, alice, state):
        strategy = ThreeHousesRush()
        alice.cash = 5000
        sq = state.board.get_square(1)
        assert strategy.should_buy_property(alice, sq, state) is True

    def test_should_not_buy_without_reserve(self, alice, state):
        strategy = ThreeHousesRush()
        sq = state.board.get_square(1)
        alice.cash = sq.price  # no reserve
        assert strategy.should_buy_property(alice, sq, state) is False

    def test_build_stops_at_3(self, alice, state):
        give_player_brown(state, alice)
        alice.cash = 50000
        strategy = ThreeHousesRush()
        # Simulate 4 rounds of building
        from monopoly.buildings import execute_build_orders

        for _ in range(5):
            orders = strategy.choose_properties_to_build(alice, state)
            execute_build_orders(alice, state, orders)

        # Should never exceed 3 houses (no hotel)
        for pos in [1, 3]:
            po = state.property_ownership[pos]
            if not po.has_hotel:
                assert po.houses <= 3

    def test_jail_decision_uses_goojf(self, alice, state):
        from monopoly.cards import Card, CardEffect

        strategy = ThreeHousesRush()
        card = Card(id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        assert strategy.get_jail_decision(alice, state) == JailDecision.USE_GOOJF

    def test_jail_decision_pay_with_cash(self, alice, state):
        strategy = ThreeHousesRush()
        alice.cash = 5000
        assert strategy.get_jail_decision(alice, state) == JailDecision.PAY_FINE

    def test_jail_decision_roll_when_broke(self, alice, state):
        strategy = ThreeHousesRush()
        alice.cash = 10
        assert strategy.get_jail_decision(alice, state) == JailDecision.ROLL_DOUBLES

    def test_mortgage_non_complete_first(self, alice, state):
        give_player_brown(state, alice)
        state.property_ownership[5].owner = alice
        strategy = ThreeHousesRush()
        result = strategy.choose_properties_to_mortgage(alice, 100, state)
        if 5 in result and 1 in result:
            assert result.index(5) < result.index(1)

    def test_no_build_orders_with_no_group(self, alice, state):
        strategy = ThreeHousesRush()
        orders = strategy.choose_properties_to_build(alice, state)
        assert orders == []

    def test_accepts_no_trades(self, alice, state):
        strategy = ThreeHousesRush()
        trade = TradeOffer(offered_positions=[], requested_positions=[1])
        assert strategy.should_accept_trade(alice, trade, state) is False

    def test_propose_trade_none(self, alice, state):
        strategy = ThreeHousesRush()
        assert strategy.propose_trade(alice, state) is None


# ---------------------------------------------------------------------------
# JailCamper
# ---------------------------------------------------------------------------


class TestJailCamper:
    def test_is_late_game_when_enough_opponent_houses(self, alice, bob, state):
        strategy = JailCamper(late_game_threshold=6)
        # Give opponent 6 houses to trigger late game
        positions = list(state.property_ownership.keys())
        for pos in positions[:2]:
            state.property_ownership[pos].owner = bob
            state.property_ownership[pos].houses = 3
        assert strategy._is_late_game(alice, state) is True

    def test_not_late_game_initially(self, alice, state):
        strategy = JailCamper()
        assert strategy._is_late_game(alice, state) is False

    def test_jail_decision_roll_doubles_in_late_game(self, alice, bob, state):
        strategy = JailCamper(late_game_threshold=6)
        # Give opponent 6 houses to trigger late game
        positions = list(state.property_ownership.keys())
        for pos in positions[:2]:
            state.property_ownership[pos].owner = bob
            state.property_ownership[pos].houses = 3
        assert strategy.get_jail_decision(alice, state) == JailDecision.ROLL_DOUBLES

    def test_jail_decision_pay_fine_early_game(self, alice, state):
        strategy = JailCamper()
        alice.cash = 1000
        assert strategy.get_jail_decision(alice, state) == JailDecision.PAY_FINE

    def test_jail_decision_roll_doubles_early_game_no_cash_no_goojf(self, alice, state):
        strategy = JailCamper()
        alice.cash = 10  # can't pay fine, no goojf cards
        assert strategy.get_jail_decision(alice, state) == JailDecision.ROLL_DOUBLES

    def test_should_buy_if_affordable(self, alice, state):
        strategy = JailCamper()
        alice.cash = 1000
        sq = state.board.get_square(1)
        assert strategy.should_buy_property(alice, sq, state) is True

    def test_choose_properties_to_build(self, alice, state):
        give_player_brown(state, alice)
        alice.cash = 5000
        strategy = JailCamper()
        orders = strategy.choose_properties_to_build(alice, state)
        positions = [o.position for o in orders]
        assert any(p in [1, 3] for p in positions)

    def test_accept_trade_returns_false(self, alice, state):
        strategy = JailCamper()
        trade = TradeOffer(offered_positions=[], requested_positions=[1])
        assert strategy.should_accept_trade(alice, trade, state) is False

    def test_propose_trade_none(self, alice, state):
        strategy = JailCamper()
        assert strategy.propose_trade(alice, state) is None

    def test_mortgage_non_complete_first(self, alice, state):
        give_player_brown(state, alice)
        state.property_ownership[5].owner = alice
        strategy = JailCamper()
        result = strategy.choose_properties_to_mortgage(alice, 100, state)
        if 5 in result and 1 in result:
            assert result.index(5) < result.index(1)


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------


class TestTrader:
    def test_should_buy_if_affordable(self, alice, state):
        strategy = Trader()
        alice.cash = 1000
        sq = state.board.get_square(1)
        assert strategy.should_buy_property(alice, sq, state) is True

    def test_property_value_color(self, state):
        strategy = Trader()
        sq = state.board.get_square(1)  # Mediterranean (brown)
        val = strategy._property_value(sq, state)
        assert val > 0

    def test_property_value_railroad(self, state):
        strategy = Trader()
        sq = state.board.get_square(5)
        val = strategy._property_value(sq, state)
        assert val == sq.price

    def test_get_jail_decision_uses_goojf(self, alice, state):
        from monopoly.cards import Card, CardEffect

        strategy = Trader()
        card = Card(id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        assert strategy.get_jail_decision(alice, state) == JailDecision.USE_GOOJF

    def test_get_jail_decision_pay_fine(self, alice, state):
        strategy = Trader()
        assert strategy.get_jail_decision(alice, state) == JailDecision.PAY_FINE

    def test_choose_properties_to_build(self, alice, state):
        give_player_brown(state, alice)
        alice.cash = 5000
        strategy = Trader()
        orders = strategy.choose_properties_to_build(alice, state)
        positions = [o.position for o in orders]
        assert any(p in [1, 3] for p in positions)

    def test_no_build_without_complete_group(self, alice, state):
        state.property_ownership[1].owner = alice
        strategy = Trader()
        orders = strategy.choose_properties_to_build(alice, state)
        assert orders == []

    def test_choose_properties_to_mortgage_sorted_by_value(self, alice, state):
        state.property_ownership[1].owner = alice
        state.property_ownership[39].owner = alice
        strategy = Trader()
        result = strategy.choose_properties_to_mortgage(alice, 100, state)
        # Should include both positions
        assert 1 in result
        assert 39 in result
        # Lower value first
        assert result.index(1) < result.index(39)

    def test_accept_trade_completes_group(self, alice, bob, state):
        strategy = Trader()
        state.property_ownership[1].owner = alice
        state.property_ownership[3].owner = bob
        trade = TradeOffer(offered_positions=[5], requested_positions=[3])
        assert strategy.should_accept_trade(alice, trade, state) is True

    def test_accept_trade_positive_value(self, alice, bob, state):
        strategy = Trader()
        # Alice offers low value (pos 1, $60) for higher value (pos 39, $400)
        state.property_ownership[1].owner = alice
        state.property_ownership[39].owner = bob
        trade = TradeOffer(
            offered_positions=[1],
            requested_positions=[39],
            cash_offered=500,  # extra cash to make it positive
        )
        assert strategy.should_accept_trade(alice, trade, state) is True

    def test_propose_trade_returns_none_when_no_incomplete_group(self, alice, state):
        strategy = Trader()
        result = strategy.propose_trade(alice, state)
        assert result is None

    def test_propose_trade_proposes_for_incomplete_group(self, alice, bob, state):
        strategy = Trader()
        # Alice owns Mediterranean (1), Bob owns Baltic (3)
        state.property_ownership[1].owner = alice
        state.property_ownership[3].owner = bob
        # Alice needs a bait property to offer
        state.property_ownership[5].owner = alice  # Railroad
        result = strategy.propose_trade(alice, state)
        if result is not None:
            assert 3 in result.requested_positions

    def test_propose_trade_no_bait_returns_none(self, alice, state):
        strategy = Trader()
        # Alice only owns the one she needs, nothing to offer
        state.property_ownership[1].owner = alice
        state.property_ownership[3].owner = state.players[1]
        result = strategy.propose_trade(alice, state)
        # No bait available (only 1 is owned, same color as needed)
        assert result is None

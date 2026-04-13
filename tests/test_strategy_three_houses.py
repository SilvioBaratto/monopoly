"""TDD tests for ThreeHousesRush strategy.

Acceptance criteria verified here:
- should_buy_property: always True for completing a group (bypass reserve)
- choose_properties_to_build: groups closest to 3 houses built first
- choose_properties_to_build: cash reserve of $150 respected
- choose_properties_to_build: never targets hotel (count <= TARGET_HOUSES)
- get_jail_decision: PAY_FINE/USE_GOOJF in early game (< 50% sold)
- get_jail_decision: ROLL_DOUBLES in late game (>= 50% sold)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from monopoly.board import Board
from monopoly.buildings import execute_build_orders
from monopoly.cards import Card, CardEffect, load_decks
from monopoly.state import GameState, Player
from monopoly.strategies.base import JailDecision
from monopoly.strategies.three_houses_rush import ThreeHousesRush

DATA_PATH = Path(__file__).parent.parent / "data" / "cards_standard.yaml"

_CASH_RESERVE = 150
_JAIL_FINE = 50


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def board():
    return Board()


@pytest.fixture
def state(board):
    rng = np.random.default_rng(0)
    chance, cc = load_decks(DATA_PATH, rng)
    return GameState.init_game(["Alice", "Bob"], board, chance, cc)


@pytest.fixture
def alice(state):
    return state.players[0]


@pytest.fixture
def bob(state):
    return state.players[1]


def give_complete_group(state: GameState, player: Player, color: str) -> list[int]:
    """Assign all properties of a color group to player. Returns positions."""
    group = state.board.get_group(color)
    for sq in group:
        state.property_ownership[sq.position].owner = player
    return [sq.position for sq in group]


def set_houses(state: GameState, positions: list[int], count: int) -> None:
    """Set house count on given positions."""
    for pos in positions:
        state.property_ownership[pos].houses = count


def sell_all_properties(state: GameState, player: Player) -> None:
    """Mark all buyable squares as owned (saturate market for late-game test)."""
    for po in state.property_ownership.values():
        if po.owner is None:
            po.owner = player


# ---------------------------------------------------------------------------
# should_buy_property — group-completion bypass
# ---------------------------------------------------------------------------


class TestShouldBuyPropertyGroupCompletion:
    def test_buys_completing_property_even_without_reserve(self, alice, state):
        """Completing a color group is always worth buying regardless of cash reserve."""
        strategy = ThreeHousesRush()
        # Alice owns Mediterranean (1), needs Baltic (3) to complete brown
        state.property_ownership[1].owner = alice
        sq = state.board.get_square(3)  # Baltic Ave (brown)
        # Exactly the price — no reserve remaining
        alice.cash = sq.price
        assert strategy.should_buy_property(alice, sq, state) is True

    def test_does_not_bypass_reserve_for_non_completing_property(self, alice, state):
        """Without group completion, cash reserve still required."""
        strategy = ThreeHousesRush()
        sq = state.board.get_square(1)  # Mediterranean alone doesn't complete brown
        alice.cash = sq.price  # no reserve
        assert strategy.should_buy_property(alice, sq, state) is False

    def test_buys_completing_property_when_broke_within_price(self, alice, state):
        """Edge case: can afford completing property even with minimal cash."""
        strategy = ThreeHousesRush()
        # Bob owns Mediterranean (1), Alice owns nothing but wants Baltic (3)
        # Wait — Alice must own Mediterranean to complete by getting Baltic
        state.property_ownership[1].owner = alice
        sq = state.board.get_square(3)  # Baltic Ave $60
        alice.cash = sq.price  # exactly $60, no reserve
        assert strategy.should_buy_property(alice, sq, state) is True

    def test_normal_purchase_still_requires_reserve(self, alice, state):
        """Non-completing purchase requires cash after purchase >= reserve."""
        strategy = ThreeHousesRush()
        sq = state.board.get_square(1)  # Mediterranean $60
        alice.cash = sq.price + _CASH_RESERVE  # exactly enough
        assert strategy.should_buy_property(alice, sq, state) is True


# ---------------------------------------------------------------------------
# choose_properties_to_build — group priority (closest to 3 first)
# ---------------------------------------------------------------------------


class TestBuildPriorityClosestToThreeFirst:
    def test_prioritises_group_with_higher_min_houses(self, alice, state):
        """Group with minimum 2 houses is built before group with minimum 0 houses."""
        strategy = ThreeHousesRush()
        alice.cash = 50_000

        # Give alice orange (3 props: 16, 18, 19) with 2 houses each
        orange_positions = give_complete_group(state, alice, "orange")
        set_houses(state, orange_positions, 2)

        # Give alice light_blue (3 props: 6, 8, 9) with 0 houses
        light_blue_positions = give_complete_group(state, alice, "light_blue")
        set_houses(state, light_blue_positions, 0)

        orders = strategy.choose_properties_to_build(alice, state)
        order_positions = [o.position for o in orders]

        # Orange positions must appear before light_blue positions
        first_orange = next(
            (i for i, p in enumerate(order_positions) if p in orange_positions),
            None,
        )
        first_light_blue = next(
            (i for i, p in enumerate(order_positions) if p in light_blue_positions),
            None,
        )
        assert first_orange is not None, "Orange group not in orders"
        assert first_light_blue is not None, "Light blue group not in orders"
        assert first_orange < first_light_blue, (
            "Orange (min=2) should come before light_blue (min=0)"
        )

    def test_prioritises_group_with_min_1_before_min_0(self, alice, state):
        """Group with minimum 1 house comes before group with minimum 0."""
        strategy = ThreeHousesRush()
        alice.cash = 50_000

        brown_positions = give_complete_group(state, alice, "brown")
        set_houses(state, brown_positions, 1)

        light_blue_positions = give_complete_group(state, alice, "light_blue")
        set_houses(state, light_blue_positions, 0)

        orders = strategy.choose_properties_to_build(alice, state)
        order_positions = [o.position for o in orders]

        first_brown = next(
            (i for i, p in enumerate(order_positions) if p in brown_positions), None
        )
        first_light_blue = next(
            (i for i, p in enumerate(order_positions) if p in light_blue_positions),
            None,
        )
        assert first_brown is not None
        assert first_light_blue is not None
        assert first_brown < first_light_blue

    def test_groups_at_same_level_both_included(self, alice, state):
        """Groups at the same development level all appear in orders."""
        strategy = ThreeHousesRush()
        alice.cash = 50_000

        brown_positions = give_complete_group(state, alice, "brown")
        set_houses(state, brown_positions, 1)

        light_blue_positions = give_complete_group(state, alice, "light_blue")
        set_houses(state, light_blue_positions, 1)

        orders = strategy.choose_properties_to_build(alice, state)
        order_positions = [o.position for o in orders]

        assert any(p in order_positions for p in brown_positions)
        assert any(p in order_positions for p in light_blue_positions)


# ---------------------------------------------------------------------------
# choose_properties_to_build — cash reserve $150
# ---------------------------------------------------------------------------


class TestBuildCashReserve:
    def test_does_not_build_when_only_reserve_remains(self, alice, state):
        """No build orders when cash equals only the reserve."""
        strategy = ThreeHousesRush()
        give_complete_group(state, alice, "brown")
        alice.cash = _CASH_RESERVE  # nothing above reserve
        orders = strategy.choose_properties_to_build(alice, state)
        assert orders == []

    def test_caps_build_count_to_available_cash(self, alice, state):
        """Build count is limited so that player retains at least $150."""
        strategy = ThreeHousesRush()
        brown_positions = give_complete_group(state, alice, "brown")

        sq = state.board.get_square(brown_positions[0])
        house_cost = sq.house_cost  # $50 for brown

        # Enough for exactly 1 house on 1 property while keeping $150 reserve
        alice.cash = house_cost + _CASH_RESERVE  # $200

        orders = strategy.choose_properties_to_build(alice, state)
        total_houses = sum(o.count for o in orders)
        assert total_houses <= 1, (
            f"Expected at most 1 house with ${alice.cash} cash, got {total_houses}"
        )

    def test_builds_all_3_when_fully_funded(self, alice, state):
        """Builds up to 3 houses when cash is plentiful."""
        strategy = ThreeHousesRush()
        give_complete_group(state, alice, "brown")
        alice.cash = 50_000
        orders = strategy.choose_properties_to_build(alice, state)
        assert len(orders) > 0


# ---------------------------------------------------------------------------
# choose_properties_to_build — never builds hotel
# ---------------------------------------------------------------------------


class TestNeverBuildsHotel:
    def test_never_adds_build_order_beyond_3_houses(self, alice, state):
        """All BuildOrders request at most (3 - current_houses) count."""
        strategy = ThreeHousesRush()
        give_complete_group(state, alice, "brown")
        alice.cash = 50_000

        for _ in range(10):
            orders = strategy.choose_properties_to_build(alice, state)
            execute_build_orders(alice, state, orders)

        for pos in [1, 3]:
            po = state.property_ownership[pos]
            assert not po.has_hotel, f"Hotel found at position {pos}"
            assert po.houses <= 3, f"More than 3 houses at position {pos}"

    def test_returns_no_orders_when_all_at_3(self, alice, state):
        """No orders returned when every property in every group is at 3 houses."""
        strategy = ThreeHousesRush()
        brown_positions = give_complete_group(state, alice, "brown")
        set_houses(state, brown_positions, 3)
        alice.cash = 50_000

        orders = strategy.choose_properties_to_build(alice, state)
        assert orders == []


# ---------------------------------------------------------------------------
# get_jail_decision — game phase
# ---------------------------------------------------------------------------


class TestGetJailDecisionGamePhase:
    def test_early_game_pays_fine_with_cash(self, alice, state):
        """Early game (< 50% sold): pays fine when affordable."""
        strategy = ThreeHousesRush()
        alice.cash = 5_000
        # No properties owned → early game
        assert strategy.get_jail_decision(alice, state) == JailDecision.PAY_FINE

    def test_early_game_uses_goojf_when_available(self, alice, state):
        """Early game: uses GOOJF card when available."""
        strategy = ThreeHousesRush()
        alice.cash = 5_000
        card = Card(id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        assert strategy.get_jail_decision(alice, state) == JailDecision.USE_GOOJF

    def test_late_game_rolls_doubles_even_with_cash(self, alice, bob, state):
        """Late game (>= 50% sold): ROLL_DOUBLES even with plenty of cash."""
        strategy = ThreeHousesRush()
        alice.cash = 5_000
        sell_all_properties(state, bob)  # saturate market
        assert strategy.get_jail_decision(alice, state) == JailDecision.ROLL_DOUBLES

    def test_late_game_rolls_doubles_even_with_goojf(self, alice, bob, state):
        """Late game: ROLL_DOUBLES even when GOOJF card is available."""
        strategy = ThreeHousesRush()
        alice.cash = 5_000
        card = Card(id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        sell_all_properties(state, bob)
        assert strategy.get_jail_decision(alice, state) == JailDecision.ROLL_DOUBLES

    def test_phase_boundary_exactly_half_sold_is_late_game(self, alice, bob, state):
        """Exactly 50% properties sold triggers late-game ROLL_DOUBLES."""
        strategy = ThreeHousesRush()
        alice.cash = 5_000
        total = len(state.board.buyable_squares)
        half = total // 2
        positions = list(state.property_ownership.keys())
        for pos in positions[:half]:
            state.property_ownership[pos].owner = bob
        assert strategy.get_jail_decision(alice, state) == JailDecision.ROLL_DOUBLES

    def test_one_below_half_sold_is_early_game(self, alice, bob, state):
        """One property below 50% threshold remains early game."""
        strategy = ThreeHousesRush()
        alice.cash = 5_000
        total = len(state.board.buyable_squares)
        below_half = total // 2 - 1
        positions = list(state.property_ownership.keys())
        for pos in positions[:below_half]:
            state.property_ownership[pos].owner = bob
        assert strategy.get_jail_decision(alice, state) == JailDecision.PAY_FINE


# ---------------------------------------------------------------------------
# Export from strategies/__init__.py
# ---------------------------------------------------------------------------


class TestExport:
    def test_three_houses_rush_importable_from_strategies(self):
        """ThreeHousesRush must be importable from the strategies package."""
        from monopoly.strategies import ThreeHousesRush as THR  # noqa: F401

        assert THR is ThreeHousesRush

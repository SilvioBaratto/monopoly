"""TDD tests for ColorTargeted strategy.

Acceptance criteria verified here:
- Constructor raises ValueError for unknown color groups
- should_buy_property: buys target-color with reserve, skips when insufficient
- should_buy_property: buys cheap non-target (< $200), skips expensive (>= $200)
- choose_properties_to_build: only target monopolies, cap at 3 houses, cash reserve respected
- get_jail_decision: PAY_FINE/USE_GOOJF in early game (turn < 30), ROLL_DOUBLES in late game
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from monopoly.board import Board
from monopoly.cards import Card, CardEffect, load_decks
from monopoly.state import GameState, Player
from monopoly.strategies.base import JailDecision
from monopoly.strategies.color_targeted import (
    CASH_RESERVE,
    EARLY_GAME_THRESHOLD,
    ColorTargeted,
)

DATA_PATH = Path(__file__).parent.parent / "data" / "cards_standard.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def board() -> Board:
    return Board()


@pytest.fixture
def state(board: Board) -> GameState:
    rng = np.random.default_rng(0)
    chance, cc = load_decks(DATA_PATH, rng)
    return GameState.init_game(["Alice", "Bob"], board, chance, cc)


@pytest.fixture
def alice(state: GameState) -> Player:
    return state.players[0]


@pytest.fixture
def bob(state: GameState) -> Player:
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


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


class TestConstructorValidation:
    def test_raises_for_unknown_color(self) -> None:
        """Unknown color group raises ValueError."""
        with pytest.raises(ValueError, match="Unknown color"):
            ColorTargeted(["magenta"])

    def test_raises_for_one_invalid_among_valid(self) -> None:
        """Single invalid color in a mixed list still raises ValueError."""
        with pytest.raises(ValueError, match="Unknown color"):
            ColorTargeted(["orange", "not_a_color"])

    def test_accepts_all_valid_colors(self) -> None:
        """All known board colors are accepted without error."""
        for color in [
            "brown",
            "light_blue",
            "pink",
            "orange",
            "red",
            "yellow",
            "green",
            "dark_blue",
        ]:
            strategy = ColorTargeted([color])
            assert strategy.target_colors == [color]

    def test_stores_multiple_target_colors(self) -> None:
        """Multiple valid colors are stored correctly."""
        strategy = ColorTargeted(["orange", "light_blue"])
        assert strategy.target_colors == ["orange", "light_blue"]


# ---------------------------------------------------------------------------
# Class-level constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_cash_reserve_is_200(self) -> None:
        """CASH_RESERVE class constant equals 200."""
        assert ColorTargeted.CASH_RESERVE == 200

    def test_early_game_threshold_is_30(self) -> None:
        """EARLY_GAME_THRESHOLD class constant equals 30."""
        assert ColorTargeted.EARLY_GAME_THRESHOLD == 30

    def test_constants_importable_at_module_level(self) -> None:
        """Constants CASH_RESERVE and EARLY_GAME_THRESHOLD are importable."""
        assert CASH_RESERVE == 200
        assert EARLY_GAME_THRESHOLD == 30


# ---------------------------------------------------------------------------
# should_buy_property — target color
# ---------------------------------------------------------------------------


class TestShouldBuyTargetColor:
    def test_buys_target_color_when_cash_sufficient(
        self, alice: Player, state: GameState
    ) -> None:
        """Buys target-color property when cash >= price + CASH_RESERVE."""
        strategy = ColorTargeted(["brown"])
        sq = state.board.get_square(1)  # Mediterranean Ave (brown), price=$60
        alice.cash = sq.price + CASH_RESERVE  # exactly enough
        assert strategy.should_buy_property(alice, sq, state) is True

    def test_skips_target_color_when_cash_insufficient(
        self, alice: Player, state: GameState
    ) -> None:
        """Skips target-color property when buying would breach cash reserve."""
        strategy = ColorTargeted(["brown"])
        sq = state.board.get_square(1)  # Mediterranean Ave (brown), price=$60
        alice.cash = sq.price  # exactly price, no reserve remaining
        assert strategy.should_buy_property(alice, sq, state) is False

    def test_buys_target_color_with_surplus_cash(
        self, alice: Player, state: GameState
    ) -> None:
        """Buys when cash well exceeds price + reserve."""
        strategy = ColorTargeted(["orange"])
        sq = state.board.get_square(16)  # St. James Place (orange)
        alice.cash = 5000
        assert strategy.should_buy_property(alice, sq, state) is True


# ---------------------------------------------------------------------------
# should_buy_property — non-target color
# ---------------------------------------------------------------------------


class TestShouldBuyNonTargetColor:
    def test_buys_cheap_non_target_when_affordable(
        self, alice: Player, state: GameState
    ) -> None:
        """Buys non-target property priced under $200 when cash sufficient."""
        strategy = ColorTargeted(["orange"])
        sq = state.board.get_square(1)  # Mediterranean Ave (brown), price=$60 < $200
        alice.cash = 5000
        assert strategy.should_buy_property(alice, sq, state) is True

    def test_skips_cheap_non_target_when_cash_insufficient(
        self, alice: Player, state: GameState
    ) -> None:
        """Skips cheap non-target when buying would breach reserve."""
        strategy = ColorTargeted(["orange"])
        sq = state.board.get_square(1)  # Mediterranean Ave, price=$60
        alice.cash = sq.price  # no reserve
        assert strategy.should_buy_property(alice, sq, state) is False

    def test_skips_expensive_non_target_at_200(
        self, alice: Player, state: GameState
    ) -> None:
        """Skips non-target property priced at exactly $200 (boundary: under means < 200)."""
        strategy = ColorTargeted(["orange"])
        sq = state.board.get_square(5)  # Reading Railroad, price=$200
        alice.cash = 5000
        assert strategy.should_buy_property(alice, sq, state) is False

    def test_skips_expensive_non_target_above_200(
        self, alice: Player, state: GameState
    ) -> None:
        """Skips non-target property priced above $200."""
        strategy = ColorTargeted(["orange"])
        sq = state.board.get_square(39)  # Boardwalk, price=$400
        alice.cash = 5000
        assert strategy.should_buy_property(alice, sq, state) is False


# ---------------------------------------------------------------------------
# choose_properties_to_build — target monopoly only
# ---------------------------------------------------------------------------


class TestBuildTargetMonopolyOnly:
    def test_builds_on_target_monopoly(self, alice: Player, state: GameState) -> None:
        """Returns build orders for completed target color group."""
        strategy = ColorTargeted(["brown"])
        give_complete_group(state, alice, "brown")
        alice.cash = 5000
        orders = strategy.choose_properties_to_build(alice, state)
        positions = [o.position for o in orders]
        assert any(p in [1, 3] for p in positions)

    def test_does_not_build_on_non_target_group(
        self, alice: Player, state: GameState
    ) -> None:
        """Does not build on non-target complete group."""
        strategy = ColorTargeted(["orange"])
        give_complete_group(state, alice, "brown")  # non-target
        alice.cash = 5000
        orders = strategy.choose_properties_to_build(alice, state)
        assert orders == []

    def test_no_orders_without_complete_target_monopoly(
        self, alice: Player, state: GameState
    ) -> None:
        """Returns empty list when target monopoly is incomplete."""
        strategy = ColorTargeted(["brown"])
        state.property_ownership[1].owner = alice  # only one of two brown properties
        alice.cash = 5000
        orders = strategy.choose_properties_to_build(alice, state)
        assert orders == []


# ---------------------------------------------------------------------------
# choose_properties_to_build — 3-house cap
# ---------------------------------------------------------------------------


class TestBuildThreeHouseCap:
    def test_builds_to_3_houses(self, alice: Player, state: GameState) -> None:
        """Build orders cap at 3 houses per property."""
        from monopoly.buildings import execute_build_orders

        strategy = ColorTargeted(["brown"])
        give_complete_group(state, alice, "brown")
        alice.cash = 50_000

        for _ in range(10):
            orders = strategy.choose_properties_to_build(alice, state)
            execute_build_orders(alice, state, orders)

        for pos in [1, 3]:
            po = state.property_ownership[pos]
            assert not po.has_hotel
            assert po.houses <= 3

    def test_returns_no_orders_when_all_at_3(
        self, alice: Player, state: GameState
    ) -> None:
        """No orders returned when every target property is already at 3 houses."""
        strategy = ColorTargeted(["brown"])
        positions = give_complete_group(state, alice, "brown")
        set_houses(state, positions, 3)
        alice.cash = 50_000

        orders = strategy.choose_properties_to_build(alice, state)
        assert orders == []


# ---------------------------------------------------------------------------
# choose_properties_to_build — cash reserve
# ---------------------------------------------------------------------------


class TestBuildCashReserve:
    def test_does_not_build_when_only_reserve_remains(
        self, alice: Player, state: GameState
    ) -> None:
        """No build orders when cash equals only the reserve."""
        strategy = ColorTargeted(["brown"])
        give_complete_group(state, alice, "brown")
        alice.cash = CASH_RESERVE  # nothing above reserve
        orders = strategy.choose_properties_to_build(alice, state)
        assert orders == []

    def test_does_not_build_when_cash_below_reserve(
        self, alice: Player, state: GameState
    ) -> None:
        """No build orders when cash is below reserve."""
        strategy = ColorTargeted(["brown"])
        give_complete_group(state, alice, "brown")
        alice.cash = CASH_RESERVE - 1
        orders = strategy.choose_properties_to_build(alice, state)
        assert orders == []

    def test_caps_build_count_to_keep_reserve(
        self, alice: Player, state: GameState
    ) -> None:
        """Build count limited so player retains at least CASH_RESERVE."""
        strategy = ColorTargeted(["brown"])
        give_complete_group(state, alice, "brown")

        sq = state.board.get_square(1)
        house_cost = sq.house_cost  # $50 for brown

        # Exactly enough for 1 house while keeping $200 reserve
        alice.cash = house_cost + CASH_RESERVE  # $250

        orders = strategy.choose_properties_to_build(alice, state)
        total_houses = sum(o.count for o in orders)
        assert total_houses <= 1


# ---------------------------------------------------------------------------
# get_jail_decision — turn-based
# ---------------------------------------------------------------------------


class TestGetJailDecisionEarlyGame:
    def test_pays_fine_in_early_game_without_goojf(
        self, alice: Player, state: GameState
    ) -> None:
        """Early game (turn < 30): returns PAY_FINE when no GOOJF card."""
        strategy = ColorTargeted(["orange"])
        state.turn_count = 0
        alice.cash = 5000
        assert strategy.get_jail_decision(alice, state) == JailDecision.PAY_FINE

    def test_uses_goojf_in_early_game_when_available(
        self, alice: Player, state: GameState
    ) -> None:
        """Early game (turn < 30): returns USE_GOOJF when card is held."""
        strategy = ColorTargeted(["orange"])
        state.turn_count = 0
        card = Card(id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        assert strategy.get_jail_decision(alice, state) == JailDecision.USE_GOOJF

    def test_pays_fine_at_turn_29(self, alice: Player, state: GameState) -> None:
        """Turn 29 is still early game: PAY_FINE."""
        strategy = ColorTargeted(["orange"])
        state.turn_count = EARLY_GAME_THRESHOLD - 1
        alice.cash = 5000
        assert strategy.get_jail_decision(alice, state) == JailDecision.PAY_FINE


class TestGetJailDecisionLateGame:
    def test_rolls_doubles_in_late_game(self, alice: Player, state: GameState) -> None:
        """Late game (turn >= 30): returns ROLL_DOUBLES."""
        strategy = ColorTargeted(["orange"])
        state.turn_count = EARLY_GAME_THRESHOLD
        alice.cash = 5000
        assert strategy.get_jail_decision(alice, state) == JailDecision.ROLL_DOUBLES

    def test_rolls_doubles_with_goojf_in_late_game(
        self, alice: Player, state: GameState
    ) -> None:
        """Late game: ROLL_DOUBLES even with GOOJF card (stay in jail)."""
        strategy = ColorTargeted(["orange"])
        state.turn_count = EARLY_GAME_THRESHOLD
        card = Card(id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        assert strategy.get_jail_decision(alice, state) == JailDecision.ROLL_DOUBLES

    def test_rolls_doubles_at_turn_31(self, alice: Player, state: GameState) -> None:
        """Turn 31 is late game: ROLL_DOUBLES."""
        strategy = ColorTargeted(["orange"])
        state.turn_count = EARLY_GAME_THRESHOLD + 1
        alice.cash = 5000
        assert strategy.get_jail_decision(alice, state) == JailDecision.ROLL_DOUBLES

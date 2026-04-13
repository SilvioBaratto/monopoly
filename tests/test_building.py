"""Tests for build_houses and sell_houses functions (Issue #18).

Covers:
- BuildResult and SellResult dataclasses
- build_houses: success, shortage, even-building, mortgaged, incomplete group, cash
- sell_houses: success, even-selling, hotel downgrade, supply invariant
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from monopoly.board import Board
from monopoly.buildings import build_houses, sell_houses
from monopoly.cards import load_decks
from monopoly.state import GameState, Player
from monopoly.strategies.base import BuildOrder, SellOrder

DATA_PATH = Path(__file__).parent.parent / "data" / "cards_standard.yaml"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def board() -> Board:
    return Board()


@pytest.fixture
def state(board: Board) -> GameState:
    rng = np.random.default_rng(42)
    chance, cc = load_decks(DATA_PATH, rng)
    return GameState.init_game(["Alice", "Bob"], board, chance, cc)


@pytest.fixture
def alice(state: GameState) -> Player:
    return state.players[0]


def give_alice_brown(state: GameState, alice: Player) -> None:
    """Give Alice both brown properties: Mediterranean=1, Baltic=3."""
    state.property_ownership[1].owner = alice
    state.property_ownership[3].owner = alice


def give_alice_light_blue(state: GameState, alice: Player) -> None:
    """Give Alice all three light-blue properties: 6, 8, 9."""
    state.property_ownership[6].owner = alice
    state.property_ownership[8].owner = alice
    state.property_ownership[9].owner = alice


# ---------------------------------------------------------------------------
# Tests: build_houses — success paths
# ---------------------------------------------------------------------------


class TestBuildHousesSuccess:
    def test_build_one_house_success(self, alice: Player, state: GameState) -> None:
        give_alice_brown(state, alice)
        alice.cash = 5000

        results = build_houses(alice, [BuildOrder(position=1, count=1)], state)

        assert len(results) == 1
        assert results[0].position == 1
        assert results[0].success is True
        assert state.property_ownership[1].houses == 1

    def test_build_four_houses_success(self, alice: Player, state: GameState) -> None:
        give_alice_brown(state, alice)
        alice.cash = 50000
        # Build evenly: alternate between pos 1 and 3, 4 houses each
        orders = []
        for _ in range(4):
            orders.append(BuildOrder(position=1, count=1))
            orders.append(BuildOrder(position=3, count=1))
        results = build_houses(alice, orders, state)

        assert all(r.success for r in results)
        assert state.property_ownership[1].houses == 4
        assert state.property_ownership[3].houses == 4

    def test_build_hotel_success(self, alice: Player, state: GameState) -> None:
        give_alice_brown(state, alice)
        alice.cash = 50000
        # Bring both props to 4 houses first
        for _ in range(4):
            build_houses(
                alice,
                [
                    BuildOrder(position=1, count=1),
                    BuildOrder(position=3, count=1),
                ],
                state,
            )
        houses_before = state.houses_available

        results = build_houses(alice, [BuildOrder(position=1, count=1)], state)

        assert results[0].success is True
        assert state.property_ownership[1].has_hotel is True
        assert state.property_ownership[1].houses == 0
        # Hotel build returns 4 houses to supply
        assert state.houses_available == houses_before + 4

    def test_build_hotel_reduces_hotels_available(
        self, alice: Player, state: GameState
    ) -> None:
        give_alice_brown(state, alice)
        alice.cash = 50000
        for _ in range(4):
            build_houses(
                alice,
                [
                    BuildOrder(position=1, count=1),
                    BuildOrder(position=3, count=1),
                ],
                state,
            )
        hotels_before = state.hotels_available

        build_houses(alice, [BuildOrder(position=1, count=1)], state)

        assert state.hotels_available == hotels_before - 1

    def test_build_reduces_houses_available(
        self, alice: Player, state: GameState
    ) -> None:
        give_alice_brown(state, alice)
        alice.cash = 5000
        houses_before = state.houses_available

        build_houses(alice, [BuildOrder(position=1, count=1)], state)

        assert state.houses_available == houses_before - 1

    def test_build_deducts_cash(self, alice: Player, state: GameState) -> None:
        give_alice_brown(state, alice)
        alice.cash = 5000
        sq = state.board.get_square(1)

        build_houses(alice, [BuildOrder(position=1, count=1)], state)

        assert alice.cash == 5000 - sq.house_cost


# ---------------------------------------------------------------------------
# Tests: build_houses — failure paths
# ---------------------------------------------------------------------------


class TestBuildHousesFailures:
    def test_housing_shortage_fails(self, alice: Player, state: GameState) -> None:
        give_alice_brown(state, alice)
        alice.cash = 5000
        state.houses_available = 0

        results = build_houses(alice, [BuildOrder(position=1, count=1)], state)

        assert results[0].success is False
        assert results[0].reason == "housing shortage"

    def test_hotel_shortage_fails(self, alice: Player, state: GameState) -> None:
        give_alice_brown(state, alice)
        alice.cash = 50000
        # Bring both to 4 houses
        for _ in range(4):
            build_houses(
                alice,
                [
                    BuildOrder(position=1, count=1),
                    BuildOrder(position=3, count=1),
                ],
                state,
            )
        state.hotels_available = 0

        results = build_houses(alice, [BuildOrder(position=1, count=1)], state)

        assert results[0].success is False
        assert results[0].reason == "hotel shortage"

    def test_even_building_violation(self, alice: Player, state: GameState) -> None:
        give_alice_brown(state, alice)
        alice.cash = 5000
        # Build first house on pos 1 (valid — group is [0, 0], going to [1, 0])
        build_houses(alice, [BuildOrder(position=1, count=1)], state)

        # Now try to build second house on pos 1 before pos 3 has any house
        results = build_houses(alice, [BuildOrder(position=1, count=1)], state)

        assert results[0].success is False
        assert results[0].reason == "even building rule"

    def test_cannot_build_on_mortgaged(self, alice: Player, state: GameState) -> None:
        give_alice_brown(state, alice)
        state.property_ownership[1].is_mortgaged = True
        alice.cash = 5000

        results = build_houses(alice, [BuildOrder(position=1, count=1)], state)

        assert results[0].success is False
        assert results[0].reason == "mortgaged"

    def test_cannot_build_incomplete_group(
        self, alice: Player, state: GameState
    ) -> None:
        # Only own Mediterranean, not Baltic
        state.property_ownership[1].owner = alice
        alice.cash = 5000

        results = build_houses(alice, [BuildOrder(position=1, count=1)], state)

        assert results[0].success is False
        assert results[0].reason == "incomplete group"

    def test_insufficient_cash_fails(self, alice: Player, state: GameState) -> None:
        give_alice_brown(state, alice)
        alice.cash = 0

        results = build_houses(alice, [BuildOrder(position=1, count=1)], state)

        assert results[0].success is False
        assert results[0].reason == "insufficient cash"

    def test_already_hotel_fails(self, alice: Player, state: GameState) -> None:
        give_alice_brown(state, alice)
        alice.cash = 50000
        for _ in range(4):
            build_houses(
                alice,
                [
                    BuildOrder(position=1, count=1),
                    BuildOrder(position=3, count=1),
                ],
                state,
            )
        build_houses(alice, [BuildOrder(position=1, count=1)], state)
        assert state.property_ownership[1].has_hotel is True

        # Attempt to build again on a hotel
        results = build_houses(alice, [BuildOrder(position=1, count=1)], state)

        assert results[0].success is False
        assert results[0].reason == "already hotel"

    def test_not_owner_fails(self, alice: Player, state: GameState) -> None:
        # Position not owned by alice
        results = build_houses(alice, [BuildOrder(position=1, count=1)], state)

        assert results[0].success is False
        assert results[0].reason == "not owner or invalid"

    def test_returns_one_result_per_order(
        self, alice: Player, state: GameState
    ) -> None:
        give_alice_brown(state, alice)
        alice.cash = 5000

        orders = [
            BuildOrder(position=1, count=1),
            BuildOrder(position=3, count=1),
        ]
        results = build_houses(alice, orders, state)

        assert len(results) == 2


# ---------------------------------------------------------------------------
# Tests: sell_houses — success paths
# ---------------------------------------------------------------------------


class TestSellHousesSuccess:
    def test_sell_house_success(self, alice: Player, state: GameState) -> None:
        give_alice_brown(state, alice)
        alice.cash = 5000
        build_houses(
            alice,
            [
                BuildOrder(position=1, count=1),
                BuildOrder(position=3, count=1),
            ],
            state,
        )
        sq = state.board.get_square(1)
        expected_cash = sq.house_cost // 2

        results = sell_houses(alice, [SellOrder(position=1, count=1)], state)

        assert results[0].success is True
        assert results[0].cash_received == expected_cash
        assert alice.cash == 5000 - sq.house_cost * 2 + expected_cash

    def test_sell_house_returns_to_supply(
        self, alice: Player, state: GameState
    ) -> None:
        give_alice_brown(state, alice)
        alice.cash = 5000
        build_houses(
            alice,
            [
                BuildOrder(position=1, count=1),
                BuildOrder(position=3, count=1),
            ],
            state,
        )
        supply_after_build = state.houses_available

        sell_houses(alice, [SellOrder(position=1, count=1)], state)

        assert state.houses_available == supply_after_build + 1

    def test_sell_hotel_success(self, alice: Player, state: GameState) -> None:
        give_alice_brown(state, alice)
        alice.cash = 50000
        for _ in range(4):
            build_houses(
                alice,
                [
                    BuildOrder(position=1, count=1),
                    BuildOrder(position=3, count=1),
                ],
                state,
            )
        build_houses(alice, [BuildOrder(position=1, count=1)], state)
        assert state.property_ownership[1].has_hotel is True

        sq = state.board.get_square(1)
        expected_cash = sq.house_cost // 2 * 5
        cash_before = alice.cash
        hotels_before = state.hotels_available

        results = sell_houses(alice, [SellOrder(position=1, count=5)], state)

        assert results[0].success is True
        assert results[0].cash_received == expected_cash
        assert alice.cash == cash_before + expected_cash
        assert state.property_ownership[1].has_hotel is False
        assert state.hotels_available == hotels_before + 1

    def test_sell_hotel_adds_four_houses_to_supply(
        self, alice: Player, state: GameState
    ) -> None:
        give_alice_brown(state, alice)
        alice.cash = 50000
        for _ in range(4):
            build_houses(
                alice,
                [
                    BuildOrder(position=1, count=1),
                    BuildOrder(position=3, count=1),
                ],
                state,
            )
        build_houses(alice, [BuildOrder(position=1, count=1)], state)
        houses_before = state.houses_available

        sell_houses(alice, [SellOrder(position=1, count=5)], state)

        assert state.houses_available == houses_before + 4

    def test_sell_returns_one_result_per_order(
        self, alice: Player, state: GameState
    ) -> None:
        give_alice_brown(state, alice)
        alice.cash = 5000
        build_houses(
            alice,
            [
                BuildOrder(position=1, count=1),
                BuildOrder(position=3, count=1),
            ],
            state,
        )

        results = sell_houses(
            alice,
            [
                SellOrder(position=1, count=1),
                SellOrder(position=3, count=1),
            ],
            state,
        )

        assert len(results) == 2


# ---------------------------------------------------------------------------
# Tests: sell_houses — failure paths
# ---------------------------------------------------------------------------


class TestSellHousesFailures:
    def test_even_selling_rule_enforced(self, alice: Player, state: GameState) -> None:
        give_alice_brown(state, alice)
        alice.cash = 50000
        # Build: pos 1 gets 3 houses, pos 3 gets 2 houses
        for _ in range(2):
            build_houses(
                alice,
                [
                    BuildOrder(position=1, count=1),
                    BuildOrder(position=3, count=1),
                ],
                state,
            )
        build_houses(alice, [BuildOrder(position=1, count=1)], state)

        assert state.property_ownership[1].houses == 3
        assert state.property_ownership[3].houses == 2

        # Selling from pos 3 (2 houses, the minimum) violates even-selling
        results = sell_houses(alice, [SellOrder(position=3, count=1)], state)

        assert results[0].success is False
        assert results[0].reason == "even selling rule"

    def test_can_sell_from_max_when_uneven(
        self, alice: Player, state: GameState
    ) -> None:
        give_alice_brown(state, alice)
        alice.cash = 50000
        for _ in range(2):
            build_houses(
                alice,
                [
                    BuildOrder(position=1, count=1),
                    BuildOrder(position=3, count=1),
                ],
                state,
            )
        build_houses(alice, [BuildOrder(position=1, count=1)], state)

        # pos 1 has 3, pos 3 has 2 — selling from pos 1 is allowed
        results = sell_houses(alice, [SellOrder(position=1, count=1)], state)

        assert results[0].success is True

    def test_hotel_downgrade_blocked_insufficient_houses(
        self, alice: Player, state: GameState
    ) -> None:
        give_alice_brown(state, alice)
        alice.cash = 50000
        for _ in range(4):
            build_houses(
                alice,
                [
                    BuildOrder(position=1, count=1),
                    BuildOrder(position=3, count=1),
                ],
                state,
            )
        build_houses(alice, [BuildOrder(position=1, count=1)], state)
        assert state.property_ownership[1].has_hotel is True

        state.houses_available = 0

        results = sell_houses(alice, [SellOrder(position=1, count=5)], state)

        assert results[0].success is False
        assert results[0].reason == "insufficient houses in supply"

    def test_sell_no_buildings_fails(self, alice: Player, state: GameState) -> None:
        give_alice_brown(state, alice)

        results = sell_houses(alice, [SellOrder(position=1, count=1)], state)

        assert results[0].success is False
        assert results[0].reason == "no buildings to sell"

    def test_sell_not_owner_fails(self, alice: Player, state: GameState) -> None:
        results = sell_houses(alice, [SellOrder(position=1, count=1)], state)

        assert results[0].success is False
        assert results[0].reason == "not owner or invalid"


# ---------------------------------------------------------------------------
# Tests: supply invariant
# ---------------------------------------------------------------------------


class TestSupplyInvariant:
    def test_supply_invariant_after_build_and_sell(
        self, alice: Player, state: GameState
    ) -> None:
        """Total houses_available + hotels_available * 4 never exceeds 32 + 12 * 4."""
        give_alice_brown(state, alice)
        alice.cash = 100000
        initial_house_supply = 32
        initial_hotel_supply = 12
        initial_total = initial_house_supply + initial_hotel_supply * 4  # 80

        # Build 4 houses on each, then hotel on pos 1
        for _ in range(4):
            build_houses(
                alice,
                [
                    BuildOrder(position=1, count=1),
                    BuildOrder(position=3, count=1),
                ],
                state,
            )
        build_houses(alice, [BuildOrder(position=1, count=1)], state)

        total = state.houses_available + state.hotels_available * 4
        assert total <= initial_total

        # Sell hotel on pos 1
        sell_houses(alice, [SellOrder(position=1, count=5)], state)

        total_after = state.houses_available + state.hotels_available * 4
        assert total_after <= initial_total

    def test_supply_never_exceeds_initial(
        self, alice: Player, state: GameState
    ) -> None:
        give_alice_brown(state, alice)
        alice.cash = 100000

        initial_houses = state.houses_available
        initial_hotels = state.hotels_available

        # Build and sell many operations
        for _ in range(4):
            build_houses(
                alice,
                [
                    BuildOrder(position=1, count=1),
                    BuildOrder(position=3, count=1),
                ],
                state,
            )
        sell_houses(
            alice,
            [
                SellOrder(position=1, count=1),
                SellOrder(position=3, count=1),
            ],
            state,
        )
        sell_houses(
            alice,
            [
                SellOrder(position=1, count=1),
                SellOrder(position=3, count=1),
            ],
            state,
        )

        assert state.houses_available <= initial_houses
        assert state.hotels_available <= initial_hotels

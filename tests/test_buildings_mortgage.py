"""Tests for buildings.py and mortgage.py."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from monopoly.board import Board
from monopoly.buildings import execute_build_orders, sell_buildings
from monopoly.cards import load_decks
from monopoly.mortgage import mortgage_property, unmortgage_property
from monopoly.state import GameState
from monopoly.strategies.base import BuildOrder

DATA_PATH = Path(__file__).parent.parent / "data" / "cards_standard.yaml"


@pytest.fixture
def board():
    return Board()


@pytest.fixture
def state(board):
    rng = np.random.default_rng(0)
    chance, cc = load_decks(DATA_PATH, rng)
    return GameState.init_game(["Alice", "Bob"], board, chance, cc)


@pytest.fixture
def bob(state):
    return state.players[1]


@pytest.fixture
def alice(state):
    return state.players[0]


def give_alice_brown(state, alice):
    """Give Alice both brown properties (Mediterranean=1, Baltic=3)."""
    state.property_ownership[1].owner = alice
    state.property_ownership[3].owner = alice


class TestMortgageProperty:
    def test_mortgage_gives_value(self, alice, state):
        state.property_ownership[1].owner = alice
        sq = state.board.get_square(1)
        initial_cash = alice.cash
        result = mortgage_property(alice, 1, state)
        assert result is True
        assert alice.cash == initial_cash + sq.mortgage

    def test_mortgage_marks_mortgaged(self, alice, state):
        state.property_ownership[1].owner = alice
        mortgage_property(alice, 1, state)
        assert state.property_ownership[1].is_mortgaged is True

    def test_cannot_mortgage_with_houses(self, alice, state):
        state.property_ownership[1].owner = alice
        state.property_ownership[1].houses = 1
        result = mortgage_property(alice, 1, state)
        assert result is False

    def test_cannot_mortgage_unowned(self, alice, state):
        result = mortgage_property(alice, 1, state)
        assert result is False

    def test_cannot_mortgage_already_mortgaged(self, alice, state):
        state.property_ownership[1].owner = alice
        mortgage_property(alice, 1, state)
        result = mortgage_property(alice, 1, state)
        assert result is False


class TestUnmortgageProperty:
    def test_unmortgage_costs_mortgage_plus_10_percent(self, alice, state):
        state.property_ownership[1].owner = alice
        sq = state.board.get_square(1)
        mortgage_property(alice, 1, state)
        cash_after_mortgage = alice.cash
        unmortgage_property(alice, 1, state)
        expected_cost = sq.mortgage + round(sq.mortgage * 0.1)
        assert alice.cash == cash_after_mortgage - expected_cost

    def test_unmortgage_clears_flag(self, alice, state):
        state.property_ownership[1].owner = alice
        mortgage_property(alice, 1, state)
        unmortgage_property(alice, 1, state)
        assert state.property_ownership[1].is_mortgaged is False

    def test_cannot_unmortgage_if_not_mortgaged(self, alice, state):
        state.property_ownership[1].owner = alice
        result = unmortgage_property(alice, 1, state)
        assert result is False

    def test_cannot_unmortgage_with_insufficient_cash(self, alice, state):
        state.property_ownership[1].owner = alice
        mortgage_property(alice, 1, state)
        alice.cash = 0  # Can't afford to unmortgage
        result = unmortgage_property(alice, 1, state)
        assert result is False


class TestBuildOrders:
    def test_build_requires_full_group(self, alice, state):
        # Own only Mediterranean, not Baltic
        state.property_ownership[1].owner = alice
        orders = [BuildOrder(position=1, count=1)]
        initial_houses = state.houses_available
        execute_build_orders(alice, state, orders)
        assert state.property_ownership[1].houses == 0
        assert state.houses_available == initial_houses

    def test_build_on_complete_group(self, alice, state):
        give_alice_brown(state, alice)
        alice.cash = 5000
        orders = [BuildOrder(position=1, count=1)]
        execute_build_orders(alice, state, orders)
        assert state.property_ownership[1].houses == 1

    def test_build_reduces_supply(self, alice, state):
        give_alice_brown(state, alice)
        alice.cash = 5000
        initial = state.houses_available
        orders = [BuildOrder(position=1, count=1)]
        execute_build_orders(alice, state, orders)
        assert state.houses_available == initial - 1

    def test_build_deducts_cash(self, alice, state):
        give_alice_brown(state, alice)
        alice.cash = 5000
        sq = state.board.get_square(1)
        orders = [BuildOrder(position=1, count=1)]
        execute_build_orders(alice, state, orders)
        assert alice.cash == 5000 - sq.house_cost

    def test_even_build_rule_enforced(self, alice, state):
        give_alice_brown(state, alice)
        alice.cash = 5000
        # Build on pos 1 only - even build requires Baltic also gets one
        orders = [BuildOrder(position=1, count=2)]
        execute_build_orders(alice, state, orders)
        # Should stop at 1 due to even building
        assert state.property_ownership[1].houses == 1

    def test_build_to_hotel(self, alice, state):
        give_alice_brown(state, alice)
        alice.cash = 50000
        # Build 4 houses on both, then hotel on pos 1
        for _ in range(4):
            execute_build_orders(
                alice,
                state,
                [
                    BuildOrder(position=1, count=1),
                    BuildOrder(position=3, count=1),
                ],
            )
        # Now build hotel on pos 1
        execute_build_orders(alice, state, [BuildOrder(position=1, count=1)])
        assert state.property_ownership[1].has_hotel is True

    def test_cannot_build_on_mortgaged(self, alice, state):
        give_alice_brown(state, alice)
        state.property_ownership[1].is_mortgaged = True
        alice.cash = 5000
        orders = [BuildOrder(position=1, count=1)]
        execute_build_orders(alice, state, orders)
        assert state.property_ownership[1].houses == 0

    def test_cannot_build_without_cash(self, alice, state):
        give_alice_brown(state, alice)
        alice.cash = 0
        orders = [BuildOrder(position=1, count=1)]
        execute_build_orders(alice, state, orders)
        assert state.property_ownership[1].houses == 0


class TestMortgageEdgeCases:
    def test_mortgage_invalid_position(self, alice, state):
        result = mortgage_property(alice, 999, state)
        assert result is False

    def test_unmortgage_invalid_position(self, alice, state):
        result = unmortgage_property(alice, 999, state)
        assert result is False

    def test_unmortgage_not_owned_by_player(self, alice, bob, state):
        state.property_ownership[1].owner = bob
        mortgage_property(bob, 1, state)
        result = unmortgage_property(alice, 1, state)
        assert result is False


class TestSellBuildings:
    def test_sell_house_returns_half_cost(self, alice, state):
        give_alice_brown(state, alice)
        alice.cash = 5000
        execute_build_orders(alice, state, [BuildOrder(position=1, count=1)])
        sq = state.board.get_square(1)
        initial_cash = alice.cash
        sell_buildings(alice, state, 1, 1)
        assert alice.cash == initial_cash + sq.house_cost // 2

    def test_sell_reduces_houses(self, alice, state):
        give_alice_brown(state, alice)
        alice.cash = 5000
        execute_build_orders(alice, state, [BuildOrder(position=1, count=1)])
        sell_buildings(alice, state, 1, 1)
        assert state.property_ownership[1].houses == 0

    def test_sell_returns_house_to_supply(self, alice, state):
        give_alice_brown(state, alice)
        alice.cash = 5000
        execute_build_orders(alice, state, [BuildOrder(position=1, count=1)])
        supply_after_build = state.houses_available
        sell_buildings(alice, state, 1, 1)
        assert state.houses_available == supply_after_build + 1

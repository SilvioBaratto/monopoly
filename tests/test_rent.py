"""Tests for rent.py — calculate_rent()."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from monopoly.board import Board
from monopoly.cards import load_decks
from monopoly.rent import calculate_rent
from monopoly.state import GameState

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
def alice(state):
    return state.players[0]


@pytest.fixture
def bob(state):
    return state.players[1]


class TestUnownedProperty:
    def test_unowned_returns_zero(self, alice, state):
        # Mediterranean Ave at position 1
        rent = calculate_rent(alice, 1, state)
        assert rent == 0


class TestOwnedBySelf:
    def test_own_property_returns_zero(self, alice, bob, state):
        state.property_ownership[1].owner = alice
        rent = calculate_rent(alice, 1, state, dice_total=7)
        assert rent == 0


class TestMortgagedProperty:
    def test_mortgaged_returns_zero(self, alice, bob, state):
        state.property_ownership[1].owner = bob
        state.property_ownership[1].is_mortgaged = True
        rent = calculate_rent(alice, 1, state)
        assert rent == 0


class TestColorPropertyRent:
    def test_base_rent_no_monopoly(self, alice, bob, state):
        # Mediterranean Ave at pos 1, Baltic at pos 3 (brown group)
        state.property_ownership[1].owner = bob
        # Only one owned, no monopoly
        rent = calculate_rent(alice, 1, state)
        sq = state.board.get_square(1)
        assert rent == sq.rents[0]

    def test_monopoly_doubles_base_rent(self, alice, bob, state):
        # Bob owns both brown properties
        state.property_ownership[1].owner = bob
        state.property_ownership[3].owner = bob
        rent = calculate_rent(alice, 1, state)
        sq = state.board.get_square(1)
        assert rent == sq.rents[0] * 2

    def test_monopoly_broken_if_one_mortgaged(self, alice, bob, state):
        state.property_ownership[1].owner = bob
        state.property_ownership[3].owner = bob
        state.property_ownership[3].is_mortgaged = True
        rent = calculate_rent(alice, 1, state)
        sq = state.board.get_square(1)
        assert rent == sq.rents[0]  # no monopoly bonus

    def test_one_house_rent(self, alice, bob, state):
        state.property_ownership[1].owner = bob
        state.property_ownership[1].houses = 1
        sq = state.board.get_square(1)
        rent = calculate_rent(alice, 1, state)
        assert rent == sq.rents[1]

    def test_two_houses_rent(self, alice, bob, state):
        state.property_ownership[1].owner = bob
        state.property_ownership[1].houses = 2
        sq = state.board.get_square(1)
        rent = calculate_rent(alice, 1, state)
        assert rent == sq.rents[2]

    def test_three_houses_rent(self, alice, bob, state):
        state.property_ownership[1].owner = bob
        state.property_ownership[1].houses = 3
        sq = state.board.get_square(1)
        rent = calculate_rent(alice, 1, state)
        assert rent == sq.rents[3]

    def test_four_houses_rent(self, alice, bob, state):
        state.property_ownership[1].owner = bob
        state.property_ownership[1].houses = 4
        sq = state.board.get_square(1)
        rent = calculate_rent(alice, 1, state)
        assert rent == sq.rents[4]

    def test_hotel_rent(self, alice, bob, state):
        state.property_ownership[1].owner = bob
        state.property_ownership[1].has_hotel = True
        sq = state.board.get_square(1)
        rent = calculate_rent(alice, 1, state)
        assert rent == sq.rents[5]

    def test_double_rent_flag(self, alice, bob, state):
        state.property_ownership[1].owner = bob
        base = calculate_rent(alice, 1, state)
        doubled = calculate_rent(alice, 1, state, double_rent=True)
        assert doubled == base * 2


class TestRailroadRent:
    def test_one_railroad_owned(self, alice, bob, state):
        state.property_ownership[5].owner = bob  # Reading RR
        sq = state.board.get_square(5)
        rent = calculate_rent(alice, 5, state)
        assert rent == sq.rents[0]

    def test_two_railroads_owned(self, alice, bob, state):
        state.property_ownership[5].owner = bob
        state.property_ownership[15].owner = bob
        sq = state.board.get_square(5)
        rent = calculate_rent(alice, 5, state)
        assert rent == sq.rents[1]

    def test_four_railroads_owned(self, alice, bob, state):
        for pos in [5, 15, 25, 35]:
            state.property_ownership[pos].owner = bob
        sq = state.board.get_square(5)
        rent = calculate_rent(alice, 5, state)
        assert rent == sq.rents[3]

    def test_mortgaged_railroad_not_counted(self, alice, bob, state):
        state.property_ownership[5].owner = bob
        state.property_ownership[15].owner = bob
        state.property_ownership[15].is_mortgaged = True
        sq = state.board.get_square(5)
        rent = calculate_rent(alice, 5, state)
        assert rent == sq.rents[0]  # only 1 counts


class TestUtilityRent:
    def test_one_utility_4x_dice(self, alice, bob, state):
        state.property_ownership[12].owner = bob  # Electric Co
        rent = calculate_rent(alice, 12, state, dice_total=8)
        sq = state.board.get_square(12)
        assert rent == sq.rents[0] * 8

    def test_two_utilities_10x_dice(self, alice, bob, state):
        state.property_ownership[12].owner = bob
        state.property_ownership[28].owner = bob
        sq = state.board.get_square(12)
        rent = calculate_rent(alice, 12, state, dice_total=6)
        assert rent == sq.rents[1] * 6

    def test_force_dice_multiplier(self, alice, bob, state):
        state.property_ownership[12].owner = bob
        rent = calculate_rent(alice, 12, state, dice_total=5, force_dice_multiplier=10)
        assert rent == 50

    def test_utility_raises_value_error_when_dice_roll_is_none(self, alice, bob, state):
        state.property_ownership[12].owner = bob
        with pytest.raises(ValueError, match="dice_roll must be provided"):
            calculate_rent(alice, 12, state, dice_total=None)


class TestNonBuyableSquare:
    def test_non_buyable_square_returns_zero(self, alice, state):
        # Position 0 is Go — non-buyable, no ownership entry
        rent = calculate_rent(alice, 0, state)
        assert rent == 0

    def test_tax_square_returns_zero(self, alice, state):
        # Position 4 is Income Tax — non-buyable
        rent = calculate_rent(alice, 4, state)
        assert rent == 0

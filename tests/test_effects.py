"""Tests for effects.py — execute_card_effect() and EffectResult."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from monopoly.board import Board
from monopoly.cards import Card, CardEffect, load_decks
from monopoly.effects import execute_card_effect
from monopoly.state import GameState

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


def make_card(effect: CardEffect, params: dict = None, card_id: str = "test") -> Card:
    return Card(id=card_id, text="test", effect=effect, params=params or {})


class TestMoveAbsolute:
    def test_moves_to_target(self, alice, state):
        alice.position = 5
        card = make_card(
            CardEffect.move_absolute, {"target": 24, "pass_go_collects": 200}
        )
        result = execute_card_effect(card, alice, state)
        assert result.new_position == 24

    def test_passes_go_when_wrapping(self, alice, state):
        alice.position = 30
        card = make_card(
            CardEffect.move_absolute, {"target": 5, "pass_go_collects": 200}
        )
        result = execute_card_effect(card, alice, state)
        assert result.passed_go is True
        assert result.cash_delta == 200

    def test_no_go_bonus_when_not_wrapping(self, alice, state):
        alice.position = 5
        card = make_card(
            CardEffect.move_absolute, {"target": 24, "pass_go_collects": 200}
        )
        result = execute_card_effect(card, alice, state)
        assert result.passed_go is False

    def test_advance_to_go_collects(self, alice, state):
        alice.position = 10
        card = make_card(CardEffect.move_absolute, {"target": 0})
        result = execute_card_effect(card, alice, state)
        assert result.new_position == 0


class TestMoveRelative:
    def test_back_3_spaces(self, alice, state):
        alice.position = 7
        card = make_card(CardEffect.move_relative, {"steps": -3})
        result = execute_card_effect(card, alice, state)
        assert result.new_position == 4

    def test_no_go_bonus(self, alice, state):
        alice.position = 1
        card = make_card(CardEffect.move_relative, {"steps": -3})
        result = execute_card_effect(card, alice, state)
        assert result.passed_go is False
        assert result.cash_delta == 0

    def test_wraps_around_board(self, alice, state):
        alice.position = 1
        card = make_card(CardEffect.move_relative, {"steps": -3})
        result = execute_card_effect(card, alice, state)
        assert result.new_position == 38  # (1 - 3) % 40


class TestMoveToNearest:
    def test_nearest_railroad_from_7(self, alice, state):
        alice.position = 7
        card = make_card(CardEffect.move_to_nearest, {"target": "railroad"})
        result = execute_card_effect(card, alice, state)
        assert result.new_position == 15  # Pennsylvania RR

    def test_nearest_railroad_from_36(self, alice, state):
        alice.position = 36
        card = make_card(CardEffect.move_to_nearest, {"target": "railroad"})
        result = execute_card_effect(card, alice, state)
        assert result.new_position == 5  # Wraps to Reading RR

    def test_nearest_utility_from_7(self, alice, state):
        alice.position = 7
        card = make_card(CardEffect.move_to_nearest, {"target": "utility"})
        result = execute_card_effect(card, alice, state)
        assert result.new_position == 12

    def test_double_rent_flag_set(self, alice, state):
        alice.position = 7
        card = make_card(
            CardEffect.move_to_nearest,
            {"target": "railroad", "double_rent": True},
        )
        result = execute_card_effect(card, alice, state)
        assert result.double_rent is True

    def test_passes_go_when_wrapping(self, alice, state):
        alice.position = 36
        card = make_card(CardEffect.move_to_nearest, {"target": "railroad"})
        result = execute_card_effect(card, alice, state)
        assert result.passed_go is True


class TestPayReceiveBank:
    def test_pay_bank_deducts_cash(self, alice, state):
        alice.cash = 500
        card = make_card(CardEffect.pay_bank, {"amount": 50})
        result = execute_card_effect(card, alice, state)
        assert alice.cash == 450
        assert result.cash_delta == -50

    def test_receive_bank_adds_cash(self, alice, state):
        alice.cash = 500
        card = make_card(CardEffect.receive_bank, {"amount": 150})
        result = execute_card_effect(card, alice, state)
        assert alice.cash == 650
        assert result.cash_delta == 150


class TestPayEachPlayer:
    def test_pay_each_player(self, alice, state):
        alice.cash = 1000
        bob = state.players[1]
        carol = state.players[2]
        bob_initial = bob.cash
        carol_initial = carol.cash
        card = make_card(CardEffect.pay_each_player, {"amount": 50})
        execute_card_effect(card, alice, state)
        assert alice.cash == 900  # paid 50 to each of 2 players
        assert bob.cash == bob_initial + 50
        assert carol.cash == carol_initial + 50

    def test_receive_each_player(self, alice, state):
        bob = state.players[1]
        carol = state.players[2]
        alice_initial = alice.cash
        card = make_card(CardEffect.receive_each_player, {"amount": 10})
        execute_card_effect(card, alice, state)
        assert alice.cash == alice_initial + 20  # received from 2 players
        assert bob.cash == 1500 - 10
        assert carol.cash == 1500 - 10


class TestRepairs:
    def test_repairs_charges_per_house(self, alice, state):
        alice.cash = 1000
        # Give alice some houses
        pos = state.board.buyable_squares[0].position
        state.property_ownership[pos].owner = alice
        state.property_ownership[pos].houses = 2
        card = make_card(CardEffect.repairs, {"house_cost": 25, "hotel_cost": 100})
        result = execute_card_effect(card, alice, state)
        assert result.cash_delta == -50

    def test_repairs_charges_per_hotel(self, alice, state):
        alice.cash = 1000
        pos = state.board.buyable_squares[0].position
        state.property_ownership[pos].owner = alice
        state.property_ownership[pos].has_hotel = True
        card = make_card(CardEffect.repairs, {"house_cost": 25, "hotel_cost": 100})
        result = execute_card_effect(card, alice, state)
        assert result.cash_delta == -100


class TestGoToJail:
    def test_returns_go_to_jail_true(self, alice, state):
        card = make_card(CardEffect.go_to_jail)
        result = execute_card_effect(card, alice, state)
        assert result.go_to_jail is True


class TestGetOutOfJail:
    def test_stores_card_in_player(self, alice, state):
        card = make_card(CardEffect.get_out_of_jail)
        result = execute_card_effect(card, alice, state)
        assert result.goojf_card is card
        assert card in alice.goojf_cards

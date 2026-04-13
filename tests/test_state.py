"""Tests for state.py — GameState, Player, PropertyOwnership."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from monopoly.board import Board
from monopoly.cards import load_decks
from monopoly.state import GameState, Player, PropertyOwnership

DATA_PATH = Path(__file__).parent.parent / "data" / "cards_standard.yaml"


@pytest.fixture
def board():
    return Board()


@pytest.fixture
def decks(board):
    rng = np.random.default_rng(0)
    return load_decks(DATA_PATH, rng)


@pytest.fixture
def game_state(board, decks):
    chance, cc = decks
    return GameState.init_game(["Alice", "Bob"], board, chance, cc)


class TestPlayer:
    def test_default_cash_is_1500(self):
        p = Player(name="Alice")
        assert p.cash == 1500

    def test_default_position_is_go(self):
        p = Player(name="Alice")
        assert p.position == 0

    def test_not_in_jail_by_default(self):
        p = Player(name="Alice")
        assert p.in_jail is False

    def test_not_bankrupt_by_default(self):
        p = Player(name="Alice")
        assert p.bankrupt is False

    def test_no_goojf_cards_by_default(self):
        p = Player(name="Alice")
        assert p.goojf_cards == []


class TestPropertyOwnership:
    def test_default_unowned(self):
        po = PropertyOwnership()
        assert po.owner is None

    def test_default_no_buildings(self):
        po = PropertyOwnership()
        assert po.houses == 0
        assert po.has_hotel is False

    def test_default_not_mortgaged(self):
        po = PropertyOwnership()
        assert po.is_mortgaged is False


class TestGameStateInitGame:
    def test_creates_correct_player_count(self, game_state):
        assert len(game_state.players) == 2

    def test_all_players_start_at_go(self, game_state):
        for p in game_state.players:
            assert p.position == 0

    def test_all_players_start_with_1500(self, game_state):
        for p in game_state.players:
            assert p.cash == 1500

    def test_all_buyable_squares_have_ownership_entry(self, board, game_state):
        for sq in board.buyable_squares:
            assert sq.position in game_state.property_ownership

    def test_all_properties_start_unowned(self, game_state):
        for po in game_state.property_ownership.values():
            assert po.owner is None

    def test_initial_houses_available(self, game_state):
        assert game_state.houses_available == 32

    def test_initial_hotels_available(self, game_state):
        assert game_state.hotels_available == 12

    def test_current_player_index_starts_at_zero(self, game_state):
        assert game_state.current_player_index == 0

    def test_turn_count_starts_at_zero(self, game_state):
        assert game_state.turn_count == 0


class TestGameStateProperties:
    def test_active_players_excludes_bankrupt(self, game_state):
        game_state.players[0].bankrupt = True
        active = game_state.active_players
        assert len(active) == 1
        assert active[0].name == "Bob"

    def test_current_player_returns_correct_player(self, game_state):
        assert game_state.current_player.name == "Alice"
        game_state.current_player_index = 1
        assert game_state.current_player.name == "Bob"

    def test_active_players_all_when_none_bankrupt(self, game_state):
        assert len(game_state.active_players) == 2

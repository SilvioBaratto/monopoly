"""Tests for jail.py — send_to_jail(), resolve_jail_turn(), JailResult."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import numpy as np
import pytest
from pathlib import Path

from monopoly.board import Board
from monopoly.cards import load_decks
from monopoly.dice import DiceRoll
from monopoly.jail import resolve_jail_turn, send_to_jail
from monopoly.state import GameState
from monopoly.strategies.base import JailDecision

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


class TestSendToJail:
    def test_sets_position_to_10(self, alice, state):
        alice.position = 22
        send_to_jail(alice, state)
        assert alice.position == 10

    def test_sets_in_jail_true(self, alice, state):
        send_to_jail(alice, state)
        assert alice.in_jail is True

    def test_resets_consecutive_doubles(self, alice, state):
        alice.consecutive_doubles = 2
        send_to_jail(alice, state)
        assert alice.consecutive_doubles == 0

    def test_resets_jail_turns(self, alice, state):
        alice.jail_turns = 1
        send_to_jail(alice, state)
        assert alice.jail_turns == 0


class TestResolveJailTurnPayFine:
    def test_pay_fine_leaves_jail(self, alice, state):
        alice.in_jail = True
        strategy = MagicMock()
        strategy.get_jail_decision.return_value = JailDecision.PAY_FINE
        rng = np.random.default_rng(0)
        result = resolve_jail_turn(alice, state, strategy, rng)
        assert result.left_jail is True
        assert result.paid_fine is True
        assert alice.in_jail is False

    def test_pay_fine_deducts_50(self, alice, state):
        alice.in_jail = True
        alice.cash = 500
        strategy = MagicMock()
        strategy.get_jail_decision.return_value = JailDecision.PAY_FINE
        rng = np.random.default_rng(0)
        resolve_jail_turn(alice, state, strategy, rng)
        assert alice.cash == 450


class TestResolveJailTurnUseGOOJF:
    def test_use_goojf_leaves_jail(self, alice, state):
        from monopoly.cards import Card, CardEffect

        alice.in_jail = True
        card = Card(id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        strategy = MagicMock()
        strategy.get_jail_decision.return_value = JailDecision.USE_GOOJF
        rng = np.random.default_rng(0)
        result = resolve_jail_turn(alice, state, strategy, rng)
        assert result.left_jail is True
        assert result.paid_fine is False
        assert alice.in_jail is False
        assert len(alice.goojf_cards) == 0

    def test_no_goojf_cards_falls_back(self, alice, state):
        """If USE_GOOJF chosen but no cards, should roll doubles instead."""
        alice.in_jail = True
        strategy = MagicMock()
        strategy.get_jail_decision.return_value = JailDecision.USE_GOOJF
        rng = np.random.default_rng(42)
        result = resolve_jail_turn(alice, state, strategy, rng)
        # No card available — should have rolled dice
        assert result.dice_roll is not None


class TestResolveJailTurnRollDoubles:
    def test_doubles_leaves_jail(self, alice, state):
        alice.in_jail = True
        strategy = MagicMock()
        strategy.get_jail_decision.return_value = JailDecision.ROLL_DOUBLES
        # Use a known seed that will produce doubles
        with patch("monopoly.jail.roll") as mock_roll:
            mock_roll.return_value = DiceRoll(die1=3, die2=3, total=6, is_doubles=True)
            rng = np.random.default_rng(0)
            result = resolve_jail_turn(alice, state, strategy, rng)
        assert result.left_jail is True
        assert alice.in_jail is False

    def test_no_doubles_stays_in_jail(self, alice, state):
        alice.in_jail = True
        strategy = MagicMock()
        strategy.get_jail_decision.return_value = JailDecision.ROLL_DOUBLES
        with patch("monopoly.jail.roll") as mock_roll:
            mock_roll.return_value = DiceRoll(die1=3, die2=4, total=7, is_doubles=False)
            rng = np.random.default_rng(0)
            result = resolve_jail_turn(alice, state, strategy, rng)
        assert result.left_jail is False
        assert alice.in_jail is True
        assert alice.jail_turns == 1

    def test_third_turn_forced_pay(self, alice, state):
        alice.in_jail = True
        alice.jail_turns = 2
        alice.cash = 500
        strategy = MagicMock()
        strategy.get_jail_decision.return_value = JailDecision.ROLL_DOUBLES
        rng = np.random.default_rng(0)
        result = resolve_jail_turn(alice, state, strategy, rng)
        assert result.left_jail is True
        assert result.paid_fine is True
        assert alice.cash == 450


class TestResolveJailTurnForcedPayOnThird:
    def test_jail_turns_2_forces_pay(self, alice, state):
        """jail_turns >= 2 forces pay regardless of strategy."""
        alice.in_jail = True
        alice.jail_turns = 2
        alice.cash = 500
        strategy = MagicMock()
        strategy.get_jail_decision.return_value = JailDecision.ROLL_DOUBLES
        rng = np.random.default_rng(0)
        result = resolve_jail_turn(alice, state, strategy, rng)
        assert result.left_jail is True
        assert result.paid_fine is True

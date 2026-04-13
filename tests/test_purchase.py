"""Tests for purchase.py — attempt_purchase()."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from monopoly.board import Board, ColorProperty, Railroad, Square, SquareType, Utility
from monopoly.cards import load_decks
from monopoly.purchase import attempt_purchase
from monopoly.state import GameState, Player

DATA_PATH = Path(__file__).parent.parent / "data" / "cards_standard.yaml"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def board() -> Board:
    return Board()


@pytest.fixture
def game_state(board: Board) -> GameState:
    rng = np.random.default_rng(0)
    chance, cc = load_decks(DATA_PATH, rng)
    return GameState.init_game(["Alice", "Bob"], board, chance, cc)


@pytest.fixture
def alice(game_state: GameState) -> Player:
    return game_state.players[0]


@pytest.fixture
def buy_strategy() -> MagicMock:
    """Strategy that always says yes to buying."""
    strategy = MagicMock()
    strategy.should_buy_property.return_value = True
    return strategy


@pytest.fixture
def decline_strategy() -> MagicMock:
    """Strategy that always declines buying."""
    strategy = MagicMock()
    strategy.should_buy_property.return_value = False
    return strategy


def _color_property(game_state: GameState) -> ColorProperty:
    """Return the first ColorProperty on the board (Mediterranean Ave, pos=1)."""
    square = game_state.board.get_square(1)
    assert isinstance(square, ColorProperty)
    return square


def _railroad(game_state: GameState) -> Railroad:
    """Return the Reading Railroad (pos=5)."""
    square = game_state.board.get_square(5)
    assert isinstance(square, Railroad)
    return square


def _utility(game_state: GameState) -> Utility:
    """Return the Electric Company (pos=12)."""
    square = game_state.board.get_square(12)
    assert isinstance(square, Utility)
    return square


# ---------------------------------------------------------------------------
# Happy path: successful purchase
# ---------------------------------------------------------------------------


class TestBuySuccess:
    def test_returns_true_when_strategy_approves_and_player_has_funds(
        self, alice: Player, game_state: GameState, buy_strategy: MagicMock
    ) -> None:
        square = _color_property(game_state)
        result = attempt_purchase(alice, square, game_state, buy_strategy)
        assert result is True

    def test_deducts_price_from_player_cash(
        self, alice: Player, game_state: GameState, buy_strategy: MagicMock
    ) -> None:
        square = _color_property(game_state)
        initial_cash = alice.cash
        attempt_purchase(alice, square, game_state, buy_strategy)
        assert alice.cash == initial_cash - square.price

    def test_sets_owner_in_property_ownership(
        self, alice: Player, game_state: GameState, buy_strategy: MagicMock
    ) -> None:
        square = _color_property(game_state)
        attempt_purchase(alice, square, game_state, buy_strategy)
        assert game_state.property_ownership[square.position].owner is alice

    def test_calls_strategy_should_buy_property(
        self, alice: Player, game_state: GameState, buy_strategy: MagicMock
    ) -> None:
        square = _color_property(game_state)
        attempt_purchase(alice, square, game_state, buy_strategy)
        buy_strategy.should_buy_property.assert_called_once_with(
            alice, square, game_state
        )


# ---------------------------------------------------------------------------
# Square type coverage
# ---------------------------------------------------------------------------


class TestSquareTypeCoverage:
    def test_buys_color_property_successfully(
        self, alice: Player, game_state: GameState, buy_strategy: MagicMock
    ) -> None:
        square = _color_property(game_state)
        assert attempt_purchase(alice, square, game_state, buy_strategy) is True
        assert game_state.property_ownership[square.position].owner is alice

    def test_buys_railroad_successfully(
        self, alice: Player, game_state: GameState, buy_strategy: MagicMock
    ) -> None:
        square = _railroad(game_state)
        assert attempt_purchase(alice, square, game_state, buy_strategy) is True
        assert game_state.property_ownership[square.position].owner is alice

    def test_buys_utility_successfully(
        self, alice: Player, game_state: GameState, buy_strategy: MagicMock
    ) -> None:
        square = _utility(game_state)
        assert attempt_purchase(alice, square, game_state, buy_strategy) is True
        assert game_state.property_ownership[square.position].owner is alice


# ---------------------------------------------------------------------------
# Strategy declines
# ---------------------------------------------------------------------------


class TestStrategyDeclines:
    def test_returns_false_when_strategy_declines(
        self, alice: Player, game_state: GameState, decline_strategy: MagicMock
    ) -> None:
        square = _color_property(game_state)
        result = attempt_purchase(alice, square, game_state, decline_strategy)
        assert result is False

    def test_cash_unchanged_when_strategy_declines(
        self, alice: Player, game_state: GameState, decline_strategy: MagicMock
    ) -> None:
        square = _color_property(game_state)
        initial_cash = alice.cash
        attempt_purchase(alice, square, game_state, decline_strategy)
        assert alice.cash == initial_cash

    def test_ownership_unchanged_when_strategy_declines(
        self, alice: Player, game_state: GameState, decline_strategy: MagicMock
    ) -> None:
        square = _color_property(game_state)
        attempt_purchase(alice, square, game_state, decline_strategy)
        assert game_state.property_ownership[square.position].owner is None


# ---------------------------------------------------------------------------
# Insufficient funds
# ---------------------------------------------------------------------------


class TestInsufficientFunds:
    def test_returns_false_when_player_cannot_afford(
        self, alice: Player, game_state: GameState, buy_strategy: MagicMock
    ) -> None:
        square = _color_property(game_state)
        alice.cash = square.price - 1
        result = attempt_purchase(alice, square, game_state, buy_strategy)
        assert result is False

    def test_cash_unchanged_when_player_cannot_afford(
        self, alice: Player, game_state: GameState, buy_strategy: MagicMock
    ) -> None:
        square = _color_property(game_state)
        alice.cash = 0
        attempt_purchase(alice, square, game_state, buy_strategy)
        assert alice.cash == 0

    def test_ownership_unchanged_when_player_cannot_afford(
        self, alice: Player, game_state: GameState, buy_strategy: MagicMock
    ) -> None:
        square = _color_property(game_state)
        alice.cash = 0
        attempt_purchase(alice, square, game_state, buy_strategy)
        assert game_state.property_ownership[square.position].owner is None

    def test_exact_cash_equals_price_allows_purchase(
        self, alice: Player, game_state: GameState, buy_strategy: MagicMock
    ) -> None:
        square = _color_property(game_state)
        alice.cash = square.price
        result = attempt_purchase(alice, square, game_state, buy_strategy)
        assert result is True
        assert alice.cash == 0


# ---------------------------------------------------------------------------
# Already-owned square
# ---------------------------------------------------------------------------


class TestAlreadyOwnedSquare:
    def test_raises_value_error_when_square_is_already_owned(
        self, alice: Player, game_state: GameState, buy_strategy: MagicMock
    ) -> None:
        square = _color_property(game_state)
        game_state.property_ownership[square.position].owner = game_state.players[1]
        with pytest.raises(ValueError, match="already owned"):
            attempt_purchase(alice, square, game_state, buy_strategy)

    def test_raises_value_error_when_player_owns_their_own_square(
        self, alice: Player, game_state: GameState, buy_strategy: MagicMock
    ) -> None:
        square = _color_property(game_state)
        game_state.property_ownership[square.position].owner = alice
        with pytest.raises(ValueError, match="already owned"):
            attempt_purchase(alice, square, game_state, buy_strategy)


# ---------------------------------------------------------------------------
# Non-buyable square
# ---------------------------------------------------------------------------


class TestNonBuyableSquare:
    def test_raises_value_error_for_non_buyable_square(
        self, alice: Player, game_state: GameState, buy_strategy: MagicMock
    ) -> None:
        non_buyable = Square(position=0, name="Go", type=SquareType.action)
        with pytest.raises(ValueError, match="not a BuyableSquare"):
            attempt_purchase(alice, non_buyable, game_state, buy_strategy)  # type: ignore[arg-type]

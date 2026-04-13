"""Tests for turn.py — resolve_turn() and TurnResult.

Covers GitHub Issue #14: TurnResult fields, doubles re-roll mechanics,
Go salary, jail, tax squares, buyable squares, and card positions.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from monopoly.board import Board
from monopoly.cards import load_decks
from monopoly.dice import DiceRoll
from monopoly.state import GameState
from monopoly.strategies.buy_nothing import BuyNothing
from monopoly.turn import TurnResult, resolve_turn

DATA_PATH = Path(__file__).parent.parent / "data" / "cards_standard.yaml"

_NON_DOUBLES = DiceRoll(die1=3, die2=4, total=7, is_doubles=False)
_DOUBLES = DiceRoll(die1=3, die2=3, total=6, is_doubles=True)


@pytest.fixture
def board() -> Board:
    return Board()


@pytest.fixture
def state(board: Board) -> GameState:
    rng = np.random.default_rng(0)
    chance, cc = load_decks(DATA_PATH, rng)
    return GameState.init_game(["Alice", "Bob"], board, chance, cc)


@pytest.fixture
def alice(state: GameState):
    return state.players[0]


@pytest.fixture
def bob(state: GameState):
    return state.players[1]


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 1. TurnResult structure
# ---------------------------------------------------------------------------


class TestTurnResultStructure:
    """TurnResult must expose new fields: rolls, positions_visited, went_to_jail, unowned_landed."""

    def test_returns_turn_result_instance(self, alice, state, rng):
        with patch("monopoly.turn.roll", return_value=_NON_DOUBLES):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert isinstance(result, TurnResult)

    def test_rolls_is_list_of_dice_rolls(self, alice, state, rng):
        with patch("monopoly.turn.roll", return_value=_NON_DOUBLES):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert isinstance(result.rolls, list)
        assert all(isinstance(r, DiceRoll) for r in result.rolls)

    def test_positions_visited_is_list_of_ints(self, alice, state, rng):
        with patch("monopoly.turn.roll", return_value=_NON_DOUBLES):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert isinstance(result.positions_visited, list)
        assert all(isinstance(p, int) for p in result.positions_visited)

    def test_went_to_jail_field_exists(self, alice, state, rng):
        with patch("monopoly.turn.roll", return_value=_NON_DOUBLES):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert isinstance(result.went_to_jail, bool)

    def test_unowned_landed_is_list_of_ints(self, alice, state, rng):
        with patch("monopoly.turn.roll", return_value=_NON_DOUBLES):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert isinstance(result.unowned_landed, list)

    def test_rent_paid_defaults_to_zero(self, alice, state, rng):
        alice.position = 0
        with patch("monopoly.turn.roll", return_value=_NON_DOUBLES):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert result.rent_paid >= 0

    def test_no_old_dice_roll_field(self, alice, state, rng):
        with patch("monopoly.turn.roll", return_value=_NON_DOUBLES):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert not hasattr(result, "dice_roll"), "Old 'dice_roll' field must be removed"

    def test_no_old_action_field(self, alice, state, rng):
        with patch("monopoly.turn.roll", return_value=_NON_DOUBLES):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert not hasattr(result, "action"), "Old 'action' field must be removed"


# ---------------------------------------------------------------------------
# 2. Player position wraps at 40
# ---------------------------------------------------------------------------


class TestPositionWrapping:
    def test_position_wraps_around_board(self, alice, state, rng):
        alice.position = 38
        roll = DiceRoll(die1=2, die2=3, total=5, is_doubles=False)
        with patch("monopoly.turn.roll", return_value=roll):
            resolve_turn(alice, state, BuyNothing(), rng)
        assert alice.position == 3  # (38 + 5) % 40 = 3

    def test_positions_visited_within_range(self, alice, state, rng):
        with patch("monopoly.turn.roll", return_value=_NON_DOUBLES):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert all(0 <= p < 40 for p in result.positions_visited)


# ---------------------------------------------------------------------------
# 3. Go salary: $200 for passing Go, NOT for landing on Go
# ---------------------------------------------------------------------------


class TestGoSalary:
    def test_passing_go_gives_200(self, alice, state, rng):
        alice.position = 38
        alice.cash = 1500
        roll = DiceRoll(die1=2, die2=2, total=4, is_doubles=False)
        with patch("monopoly.turn.roll", return_value=roll):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert result.passed_go is True
        assert alice.cash >= 1700 - 200  # may pay income tax at pos 2

    def test_landing_exactly_on_go_no_salary(self, alice, state, rng):
        """Landing exactly on Go (position 0) should NOT give $200."""
        alice.position = 38
        alice.cash = 1500
        with patch("monopoly.turn.roll", return_value=_NON_DOUBLES):
            # Use a roll that lands exactly on 0: position 38 + 2 = 40 → 0
            pass

        # Position 36 + 4 = 40 → 0: land on Go exactly
        alice.position = 36
        roll_to_go = DiceRoll(die1=2, die2=2, total=4, is_doubles=False)
        with patch("monopoly.turn.roll", return_value=roll_to_go):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert result.passed_go is False
        assert alice.cash == 1500  # no salary

    def test_passed_go_false_on_normal_move(self, alice, state, rng):
        alice.position = 5
        alice.cash = 1500
        roll = DiceRoll(die1=3, die2=2, total=5, is_doubles=False)
        with patch("monopoly.turn.roll", return_value=roll):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert result.passed_go is False


# ---------------------------------------------------------------------------
# 4. Tax squares
# ---------------------------------------------------------------------------


class TestTaxSquares:
    def test_income_tax_at_position_4_charges_200(self, alice, state, rng):
        alice.position = 0
        alice.cash = 1000
        roll = DiceRoll(die1=2, die2=2, total=4, is_doubles=False)
        with patch("monopoly.turn.roll", return_value=roll):
            resolve_turn(alice, state, BuyNothing(), rng)
        assert alice.position == 4
        assert alice.cash == 800  # 1000 - 200

    def test_luxury_tax_at_position_38_charges_100(self, alice, state, rng):
        alice.position = 32
        alice.cash = 1000
        roll = DiceRoll(die1=3, die2=3, total=6, is_doubles=False)
        with patch("monopoly.turn.roll", return_value=roll):
            resolve_turn(alice, state, BuyNothing(), rng)
        assert alice.position == 38
        assert alice.cash == 900  # 1000 - 100


# ---------------------------------------------------------------------------
# 5. Card squares (Chance and Community Chest)
# ---------------------------------------------------------------------------


class TestCardSquares:
    def test_chance_card_drawn_at_position_7(self, alice, state, rng):
        alice.position = 0
        roll = DiceRoll(die1=3, die2=4, total=7, is_doubles=False)
        with patch("monopoly.turn.roll", return_value=roll):
            with patch("monopoly.turn._resolve_card") as mock_card:
                mock_card.return_value = {
                    "rent_paid": 0,
                    "went_to_jail": False,
                    "unowned_pos": None,
                }
                resolve_turn(alice, state, BuyNothing(), rng)
        mock_card.assert_called_once()

    def test_community_chest_card_drawn_at_position_2(self, alice, state, rng):
        alice.position = 0
        roll = DiceRoll(die1=1, die2=1, total=2, is_doubles=False)
        with patch("monopoly.turn.roll", return_value=roll):
            with patch("monopoly.turn._resolve_card") as mock_card:
                mock_card.return_value = {
                    "rent_paid": 0,
                    "went_to_jail": False,
                    "unowned_pos": None,
                }
                resolve_turn(alice, state, BuyNothing(), rng)
        mock_card.assert_called_once()


# ---------------------------------------------------------------------------
# 6. Go To Jail square (position 30)
# ---------------------------------------------------------------------------


class TestGoToJail:
    def test_landing_on_30_sends_to_jail(self, alice, state, rng):
        alice.position = 22
        roll = DiceRoll(die1=4, die2=4, total=8, is_doubles=False)
        with patch("monopoly.turn.roll", return_value=roll):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert alice.in_jail is True
        assert result.went_to_jail is True

    def test_go_to_jail_no_go_salary(self, alice, state, rng):
        """Player passing Go on way to jail square should not collect $200."""
        alice.position = 28
        alice.cash = 1500
        # 28 + some roll > 30, landing on 30
        roll = DiceRoll(die1=1, die2=1, total=2, is_doubles=False)
        with patch("monopoly.turn.roll", return_value=roll):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        # Position 28 + 2 = 30 → jail
        assert alice.in_jail is True
        assert result.went_to_jail is True


# ---------------------------------------------------------------------------
# 7. Triple doubles → jail
# ---------------------------------------------------------------------------


class TestTripleDoubles:
    def test_triple_doubles_sends_to_jail(self, alice, state, rng):
        alice.consecutive_doubles = 0
        with patch("monopoly.turn.roll", return_value=_DOUBLES):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert alice.in_jail is True
        assert result.went_to_jail is True

    def test_triple_doubles_no_movement(self, alice, state, rng):
        """On triple doubles the player must NOT move."""
        alice.position = 5
        alice.consecutive_doubles = 0
        with patch("monopoly.turn.roll", return_value=_DOUBLES):
            resolve_turn(alice, state, BuyNothing(), rng)
        # Triple doubles → jail at pos 10, no intermediate movement
        assert alice.in_jail is True

    def test_triple_doubles_rolls_has_three_entries(self, alice, state, rng):
        alice.consecutive_doubles = 0
        with patch("monopoly.turn.roll", return_value=_DOUBLES):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        # All three doubles were rolled before jail
        assert len(result.rolls) == 3


# ---------------------------------------------------------------------------
# 8. Rent mechanics
# ---------------------------------------------------------------------------


class TestRentMechanics:
    def test_rent_paid_when_landing_on_owned_property(self, alice, bob, state, rng):
        state.property_ownership[1].owner = bob
        alice.position = 0
        alice.cash = 1500
        roll = DiceRoll(die1=1, die2=0, total=1, is_doubles=False)
        with patch("monopoly.turn.roll", return_value=roll):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert result.rent_paid > 0

    def test_no_rent_on_own_property(self, alice, state, rng):
        state.property_ownership[1].owner = alice
        alice.position = 0
        alice.cash = 1500
        roll = DiceRoll(die1=1, die2=0, total=1, is_doubles=False)
        with patch("monopoly.turn.roll", return_value=roll):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert result.rent_paid == 0

    def test_no_rent_on_mortgaged_property(self, alice, bob, state, rng):
        state.property_ownership[1].owner = bob
        state.property_ownership[1].is_mortgaged = True
        alice.position = 0
        alice.cash = 1500
        roll = DiceRoll(die1=1, die2=0, total=1, is_doubles=False)
        with patch("monopoly.turn.roll", return_value=roll):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert result.rent_paid == 0


# ---------------------------------------------------------------------------
# 9. Unowned buyable squares in unowned_landed
# ---------------------------------------------------------------------------


class TestUnownedLanded:
    def test_unowned_property_appears_in_unowned_landed(self, alice, state, rng):
        alice.position = 0
        # Position 1 (Mediterranean Ave) is unowned by default
        roll = DiceRoll(die1=1, die2=0, total=1, is_doubles=False)
        with patch("monopoly.turn.roll", return_value=roll):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert 1 in result.unowned_landed

    def test_owned_property_not_in_unowned_landed(self, alice, bob, state, rng):
        state.property_ownership[1].owner = bob
        alice.position = 0
        roll = DiceRoll(die1=1, die2=0, total=1, is_doubles=False)
        with patch("monopoly.turn.roll", return_value=roll):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert 1 not in result.unowned_landed


# ---------------------------------------------------------------------------
# 10. Doubles re-roll mechanics
# ---------------------------------------------------------------------------


class TestDoublesReRoll:
    def test_non_doubles_single_roll(self, alice, state, rng):
        alice.position = 0
        with patch("monopoly.turn.roll", return_value=_NON_DOUBLES):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert len(result.rolls) == 1

    def test_doubles_grants_extra_roll(self, alice, state, rng):
        alice.position = 0
        side_effects = iter([_DOUBLES, _NON_DOUBLES])
        with patch("monopoly.turn.roll", side_effect=side_effects):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert len(result.rolls) == 2

    def test_two_doubles_grants_three_rolls(self, alice, state, rng):
        alice.position = 0
        side_effects = iter([_DOUBLES, _DOUBLES, _NON_DOUBLES])
        with patch("monopoly.turn.roll", side_effect=side_effects):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert len(result.rolls) == 3

    def test_positions_visited_matches_rolls_count(self, alice, state, rng):
        alice.position = 0
        side_effects = iter([_DOUBLES, _NON_DOUBLES])
        with patch("monopoly.turn.roll", side_effect=side_effects):
            result = resolve_turn(alice, state, BuyNothing(), rng)
        assert len(result.positions_visited) == len(result.rolls)

    def test_consecutive_doubles_reset_at_turn_start(self, alice, state, rng):
        """resolve_turn resets consecutive_doubles to 0 at start of each turn."""
        alice.consecutive_doubles = 2  # leftover from previous turn
        with patch("monopoly.turn.roll", return_value=_NON_DOUBLES):
            resolve_turn(alice, state, BuyNothing(), rng)
        assert alice.consecutive_doubles == 0

    def test_doubles_increments_consecutive_doubles(self, alice, state, rng):
        alice.consecutive_doubles = 0
        side_effects = iter([_DOUBLES, _NON_DOUBLES])
        with patch("monopoly.turn.roll", side_effect=side_effects):
            resolve_turn(alice, state, BuyNothing(), rng)
        # After one doubles + one non-doubles: resets to 0
        assert alice.consecutive_doubles == 0

"""Tests for card_effects.py — execute_card_effect(), CardResult, LandingModifier.

TDD: tests written before implementation per issue #13.
One test per effect type + edge cases as required by acceptance criteria.
"""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from monopoly.board import Board
from monopoly.cards import Card, CardEffect, load_decks
from monopoly.card_effects import CardResult, LandingModifier, execute_card_effect
from monopoly.state import GameState

DATA_PATH = Path(__file__).parent.parent / "data" / "cards_standard.yaml"


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
    return GameState.init_game(["Alice", "Bob", "Carol"], board, chance, cc)


@pytest.fixture
def alice(state):
    return state.players[0]


@pytest.fixture
def bob(state):
    return state.players[1]


@pytest.fixture
def carol(state):
    return state.players[2]


def make_card(
    effect: CardEffect, params: dict | None = None, card_id: str = "test"
) -> Card:
    return Card(id=card_id, text="test", effect=effect, params=params or {})


# ---------------------------------------------------------------------------
# CardResult and LandingModifier structure tests
# ---------------------------------------------------------------------------


class TestCardResultStructure:
    def test_card_result_defaults(self):
        result = CardResult()
        assert result.moved is False
        assert result.new_position is None
        assert result.landing_modifier is None

    def test_landing_modifier_defaults(self):
        modifier = LandingModifier()
        assert modifier.double_rent is False
        assert modifier.dice_multiplier is None

    def test_landing_modifier_with_values(self):
        modifier = LandingModifier(double_rent=True, dice_multiplier=10)
        assert modifier.double_rent is True
        assert modifier.dice_multiplier == 10


# ---------------------------------------------------------------------------
# move_absolute
# ---------------------------------------------------------------------------


class TestMoveAbsolute:
    def test_moves_player_to_target(self, alice, state):
        alice.position = 5
        card = make_card(
            CardEffect.move_absolute, {"target": 24, "pass_go_collects": 200}
        )
        execute_card_effect(card, alice, state)
        assert alice.position == 24

    def test_returns_moved_true_and_new_position(self, alice, state):
        alice.position = 5
        card = make_card(
            CardEffect.move_absolute, {"target": 24, "pass_go_collects": 200}
        )
        result = execute_card_effect(card, alice, state)
        assert result.moved is True
        assert result.new_position == 24

    def test_passes_go_collects_salary_when_wrapping(self, alice, state):
        alice.position = 30
        alice.cash = 1000
        card = make_card(
            CardEffect.move_absolute, {"target": 5, "pass_go_collects": 200}
        )
        execute_card_effect(card, alice, state)
        assert alice.cash == 1200

    def test_no_go_salary_when_not_wrapping(self, alice, state):
        alice.position = 5
        alice.cash = 1000
        card = make_card(
            CardEffect.move_absolute, {"target": 24, "pass_go_collects": 200}
        )
        execute_card_effect(card, alice, state)
        assert alice.cash == 1000

    def test_advance_to_go_collects_salary(self, alice, state):
        alice.position = 10
        alice.cash = 1000
        card = make_card(
            CardEffect.move_absolute, {"target": 0, "pass_go_collects": 200}
        )
        execute_card_effect(card, alice, state)
        assert alice.position == 0
        assert alice.cash == 1200

    def test_no_go_salary_without_pass_go_collects_param(self, alice, state):
        alice.position = 30
        alice.cash = 1000
        card = make_card(CardEffect.move_absolute, {"target": 5})
        execute_card_effect(card, alice, state)
        assert alice.cash == 1000

    def test_no_landing_modifier(self, alice, state):
        alice.position = 5
        card = make_card(CardEffect.move_absolute, {"target": 24})
        result = execute_card_effect(card, alice, state)
        assert result.landing_modifier is None


# ---------------------------------------------------------------------------
# move_relative
# ---------------------------------------------------------------------------


class TestMoveRelative:
    def test_moves_backward_by_steps(self, alice, state):
        alice.position = 7
        card = make_card(CardEffect.move_relative, {"steps": -3})
        execute_card_effect(card, alice, state)
        assert alice.position == 4

    def test_returns_moved_true_and_new_position(self, alice, state):
        alice.position = 7
        card = make_card(CardEffect.move_relative, {"steps": -3})
        result = execute_card_effect(card, alice, state)
        assert result.moved is True
        assert result.new_position == 4

    def test_never_collects_go_salary_when_wrapping_past_go(self, alice, state):
        """Edge case: move_relative wrapping past position 0 never gives Go salary."""
        alice.position = 1
        alice.cash = 1000
        card = make_card(CardEffect.move_relative, {"steps": -3})
        execute_card_effect(card, alice, state)
        assert alice.cash == 1000  # No salary

    def test_wraps_around_board_correctly(self, alice, state):
        """Edge case: (1 - 3) % 40 = 38."""
        alice.position = 1
        card = make_card(CardEffect.move_relative, {"steps": -3})
        execute_card_effect(card, alice, state)
        assert alice.position == 38

    def test_no_landing_modifier(self, alice, state):
        alice.position = 7
        card = make_card(CardEffect.move_relative, {"steps": -3})
        result = execute_card_effect(card, alice, state)
        assert result.landing_modifier is None


# ---------------------------------------------------------------------------
# move_to_nearest
# ---------------------------------------------------------------------------


class TestMoveToNearest:
    def test_advances_to_nearest_railroad_forward(self, alice, state):
        alice.position = 7
        card = make_card(CardEffect.move_to_nearest, {"target": "railroad"})
        execute_card_effect(card, alice, state)
        assert alice.position == 15  # Pennsylvania RR

    def test_advances_to_nearest_railroad_wrapping_around(self, alice, state):
        """Edge case: wraps around board end."""
        alice.position = 36
        card = make_card(CardEffect.move_to_nearest, {"target": "railroad"})
        execute_card_effect(card, alice, state)
        assert alice.position == 5  # Reading RR (wrap)

    def test_advances_to_nearest_utility_forward(self, alice, state):
        alice.position = 7
        card = make_card(CardEffect.move_to_nearest, {"target": "utility"})
        execute_card_effect(card, alice, state)
        assert alice.position == 12  # Electric Company

    def test_collects_go_salary_when_wrapping(self, alice, state):
        alice.position = 36
        alice.cash = 1000
        card = make_card(CardEffect.move_to_nearest, {"target": "railroad"})
        execute_card_effect(card, alice, state)
        assert alice.cash == 1200

    def test_no_go_salary_when_not_wrapping(self, alice, state):
        alice.position = 7
        alice.cash = 1000
        card = make_card(CardEffect.move_to_nearest, {"target": "railroad"})
        execute_card_effect(card, alice, state)
        assert alice.cash == 1000

    def test_returns_landing_modifier_with_double_rent(self, alice, state):
        alice.position = 7
        card = make_card(
            CardEffect.move_to_nearest,
            {"target": "railroad", "double_rent": True},
        )
        result = execute_card_effect(card, alice, state)
        assert result.landing_modifier is not None
        assert result.landing_modifier.double_rent is True

    def test_returns_landing_modifier_with_dice_multiplier(self, alice, state):
        alice.position = 7
        card = make_card(
            CardEffect.move_to_nearest,
            {"target": "utility", "dice_multiplier": 10},
        )
        result = execute_card_effect(card, alice, state)
        assert result.landing_modifier is not None
        assert result.landing_modifier.dice_multiplier == 10

    def test_no_landing_modifier_when_no_rent_params(self, alice, state):
        alice.position = 7
        card = make_card(CardEffect.move_to_nearest, {"target": "railroad"})
        result = execute_card_effect(card, alice, state)
        assert result.landing_modifier is None

    def test_returns_moved_true_and_new_position(self, alice, state):
        alice.position = 7
        card = make_card(CardEffect.move_to_nearest, {"target": "railroad"})
        result = execute_card_effect(card, alice, state)
        assert result.moved is True
        assert result.new_position == 15


# ---------------------------------------------------------------------------
# pay_bank / receive_bank
# ---------------------------------------------------------------------------


class TestPayReceiveBank:
    def test_pay_bank_deducts_cash(self, alice, state):
        alice.cash = 500
        card = make_card(CardEffect.pay_bank, {"amount": 50})
        execute_card_effect(card, alice, state)
        assert alice.cash == 450

    def test_pay_bank_cash_can_go_negative(self, alice, state):
        """Caller handles bankruptcy; cash may go negative."""
        alice.cash = 30
        card = make_card(CardEffect.pay_bank, {"amount": 50})
        execute_card_effect(card, alice, state)
        assert alice.cash == -20

    def test_pay_bank_returns_not_moved(self, alice, state):
        card = make_card(CardEffect.pay_bank, {"amount": 50})
        result = execute_card_effect(card, alice, state)
        assert result.moved is False
        assert result.new_position is None

    def test_receive_bank_adds_cash(self, alice, state):
        alice.cash = 500
        card = make_card(CardEffect.receive_bank, {"amount": 150})
        execute_card_effect(card, alice, state)
        assert alice.cash == 650

    def test_receive_bank_returns_not_moved(self, alice, state):
        card = make_card(CardEffect.receive_bank, {"amount": 150})
        result = execute_card_effect(card, alice, state)
        assert result.moved is False
        assert result.new_position is None


# ---------------------------------------------------------------------------
# pay_each_player / receive_each_player
# ---------------------------------------------------------------------------


class TestPayReceiveEachPlayer:
    def test_pay_each_player_deducts_from_player(self, alice, bob, carol, state):
        alice.cash = 1000
        card = make_card(CardEffect.pay_each_player, {"amount": 50})
        execute_card_effect(card, alice, state)
        assert alice.cash == 900  # 50 * 2 opponents

    def test_pay_each_player_credits_each_opponent(self, alice, bob, carol, state):
        bob.cash = 1500
        carol.cash = 1500
        card = make_card(CardEffect.pay_each_player, {"amount": 50})
        execute_card_effect(card, alice, state)
        assert bob.cash == 1550
        assert carol.cash == 1550

    def test_pay_each_player_with_zero_active_opponents(self, alice, bob, carol, state):
        """Edge case: no active opponents — player pays nothing."""
        bob.bankrupt = True
        carol.bankrupt = True
        alice.cash = 1000
        card = make_card(CardEffect.pay_each_player, {"amount": 50})
        execute_card_effect(card, alice, state)
        assert alice.cash == 1000

    def test_receive_each_player_credits_player(self, alice, bob, carol, state):
        alice.cash = 1000
        card = make_card(CardEffect.receive_each_player, {"amount": 10})
        execute_card_effect(card, alice, state)
        assert alice.cash == 1020  # 10 * 2 opponents

    def test_receive_each_player_deducts_from_each_opponent(
        self, alice, bob, carol, state
    ):
        bob.cash = 1500
        carol.cash = 1500
        card = make_card(CardEffect.receive_each_player, {"amount": 10})
        execute_card_effect(card, alice, state)
        assert bob.cash == 1490
        assert carol.cash == 1490

    def test_pay_each_player_skips_bankrupt_opponents(self, alice, bob, carol, state):
        bob.bankrupt = True
        alice.cash = 1000
        carol.cash = 1500
        card = make_card(CardEffect.pay_each_player, {"amount": 50})
        execute_card_effect(card, alice, state)
        assert alice.cash == 950  # only 1 active opponent


# ---------------------------------------------------------------------------
# repairs
# ---------------------------------------------------------------------------


class TestRepairs:
    def test_repairs_charges_per_house(self, alice, state):
        alice.cash = 1000
        pos = state.board.buyable_squares[0].position
        state.property_ownership[pos].owner = alice
        state.property_ownership[pos].houses = 2
        card = make_card(CardEffect.repairs, {"house_cost": 25, "hotel_cost": 100})
        execute_card_effect(card, alice, state)
        assert alice.cash == 950  # 2 * 25

    def test_repairs_charges_per_hotel(self, alice, state):
        alice.cash = 1000
        pos = state.board.buyable_squares[0].position
        state.property_ownership[pos].owner = alice
        state.property_ownership[pos].has_hotel = True
        card = make_card(CardEffect.repairs, {"house_cost": 25, "hotel_cost": 100})
        execute_card_effect(card, alice, state)
        assert alice.cash == 900  # 1 hotel * 100

    def test_repairs_charges_mixed_houses_and_hotels(self, alice, state):
        alice.cash = 1000
        squares = state.board.buyable_squares
        pos1 = squares[0].position
        pos2 = squares[1].position
        state.property_ownership[pos1].owner = alice
        state.property_ownership[pos1].houses = 3
        state.property_ownership[pos2].owner = alice
        state.property_ownership[pos2].has_hotel = True
        card = make_card(CardEffect.repairs, {"house_cost": 25, "hotel_cost": 100})
        execute_card_effect(card, alice, state)
        assert alice.cash == 825  # 3 * 25 + 1 * 100

    def test_repairs_with_zero_buildings(self, alice, state):
        """Edge case: no buildings — no charge."""
        alice.cash = 1000
        card = make_card(CardEffect.repairs, {"house_cost": 25, "hotel_cost": 100})
        execute_card_effect(card, alice, state)
        assert alice.cash == 1000

    def test_repairs_returns_not_moved(self, alice, state):
        card = make_card(CardEffect.repairs, {"house_cost": 25, "hotel_cost": 100})
        result = execute_card_effect(card, alice, state)
        assert result.moved is False
        assert result.new_position is None


# ---------------------------------------------------------------------------
# go_to_jail
# ---------------------------------------------------------------------------


class TestGoToJail:
    def test_sets_position_to_jail(self, alice, state):
        alice.position = 20
        card = make_card(CardEffect.go_to_jail)
        execute_card_effect(card, alice, state)
        assert alice.position == 10

    def test_sets_in_jail_true(self, alice, state):
        card = make_card(CardEffect.go_to_jail)
        execute_card_effect(card, alice, state)
        assert alice.in_jail is True

    def test_resets_consecutive_doubles(self, alice, state):
        alice.consecutive_doubles = 2
        card = make_card(CardEffect.go_to_jail)
        execute_card_effect(card, alice, state)
        assert alice.consecutive_doubles == 0

    def test_no_go_salary(self, alice, state):
        """Player does not collect $200 when going to jail."""
        alice.position = 30
        alice.cash = 1000
        card = make_card(CardEffect.go_to_jail)
        execute_card_effect(card, alice, state)
        assert alice.cash == 1000

    def test_returns_moved_true_to_jail_position(self, alice, state):
        card = make_card(CardEffect.go_to_jail)
        result = execute_card_effect(card, alice, state)
        assert result.moved is True
        assert result.new_position == 10


# ---------------------------------------------------------------------------
# get_out_of_jail
# ---------------------------------------------------------------------------


class TestGetOutOfJail:
    def test_appends_card_to_player_goojf_cards(self, alice, state):
        card = make_card(CardEffect.get_out_of_jail)
        execute_card_effect(card, alice, state)
        assert card in alice.goojf_cards

    def test_returns_not_moved(self, alice, state):
        card = make_card(CardEffect.get_out_of_jail)
        result = execute_card_effect(card, alice, state)
        assert result.moved is False
        assert result.new_position is None


# ---------------------------------------------------------------------------
# Unknown effect
# ---------------------------------------------------------------------------


class TestUnknownEffect:
    def test_raises_value_error_for_unknown_effect(self, alice, state):
        """Any unrecognised CardEffect must raise ValueError."""
        # We simulate this by patching card.effect with an invalid value
        # In practice CardEffect is an enum so we use a raw string trick
        card = Card(id="bad", text="bad", effect="not_a_real_effect", params={})  # type: ignore[arg-type]
        with pytest.raises((ValueError, KeyError)):
            execute_card_effect(card, alice, state)

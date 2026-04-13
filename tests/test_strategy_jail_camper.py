"""TDD tests for JailCamper strategy.

Acceptance criteria:
- Constructor: JailCamper(late_game_threshold: int = 6)
- Early game: PAY_FINE if affordable, USE_GOOJF if available, else ROLL_DOUBLES
- Late game (opponents >= threshold houses+hotels): always ROLL_DOUBLES
- Threshold boundary: exactly at threshold → late game; one below → early game
- Buying/building delegates to BuyEverything (no duplicated logic)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from monopoly.board import Board
from monopoly.cards import Card, CardEffect, load_decks
from monopoly.state import GameState, Player
from monopoly.strategies.base import JailDecision, TradeOffer
from monopoly.strategies.buy_everything import BuyEverything
from monopoly.strategies.jail_camper import JailCamper

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


@pytest.fixture
def carol(state):
    return state.players[2]


def give_opponent_houses(state: GameState, opponent: Player, count: int) -> None:
    """Assign `count` houses across opponent-owned properties."""
    positions = list(state.property_ownership.keys())
    remaining = count
    for pos in positions:
        if remaining <= 0:
            break
        po = state.property_ownership[pos]
        po.owner = opponent
        houses = min(remaining, 4)
        po.houses = houses
        remaining -= houses


def give_opponent_hotels(state: GameState, opponent: Player, count: int) -> None:
    """Assign `count` hotels across opponent-owned properties."""
    positions = list(state.property_ownership.keys())
    for pos in positions[:count]:
        po = state.property_ownership[pos]
        po.owner = opponent
        po.has_hotel = True


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------


class TestJailCamperConstructor:
    def test_default_threshold_is_6(self):
        strategy = JailCamper()
        assert strategy.late_game_threshold == 6

    def test_custom_threshold_is_stored(self):
        strategy = JailCamper(late_game_threshold=10)
        assert strategy.late_game_threshold == 10


# ---------------------------------------------------------------------------
# _is_late_game — early game
# ---------------------------------------------------------------------------


class TestIsLateGameEarlyGame:
    def test_returns_false_when_no_opponent_houses(self, alice, state):
        strategy = JailCamper()
        assert strategy._is_late_game(alice, state) is False

    def test_returns_false_when_opponent_houses_below_threshold(
        self, alice, bob, state
    ):
        strategy = JailCamper(late_game_threshold=6)
        give_opponent_houses(state, bob, 5)  # one below threshold
        assert strategy._is_late_game(alice, state) is False

    def test_own_houses_are_not_counted(self, alice, bob, state):
        """Houses owned by the current player should not trigger late game."""
        strategy = JailCamper(late_game_threshold=6)
        give_opponent_houses(state, alice, 10)  # alice is the current player
        assert strategy._is_late_game(alice, state) is False


# ---------------------------------------------------------------------------
# _is_late_game — late game
# ---------------------------------------------------------------------------


class TestIsLateGameLateGame:
    def test_returns_true_when_opponent_houses_at_threshold(self, alice, bob, state):
        strategy = JailCamper(late_game_threshold=6)
        give_opponent_houses(state, bob, 6)  # exactly at threshold
        assert strategy._is_late_game(alice, state) is True

    def test_returns_true_when_opponent_houses_exceed_threshold(
        self, alice, bob, state
    ):
        strategy = JailCamper(late_game_threshold=6)
        give_opponent_houses(state, bob, 8)
        assert strategy._is_late_game(alice, state) is True

    def test_hotels_count_toward_threshold(self, alice, bob, state):
        strategy = JailCamper(late_game_threshold=6)
        give_opponent_hotels(state, bob, 6)  # 6 hotels = 6 units
        assert strategy._is_late_game(alice, state) is True

    def test_combined_houses_and_hotels_trigger_late_game(
        self, alice, bob, carol, state
    ):
        strategy = JailCamper(late_game_threshold=6)
        give_opponent_houses(state, bob, 3)
        # Place 3 hotels on carol's properties (different positions)
        positions = list(state.property_ownership.keys())
        carol_positions = [
            p for p in positions if state.property_ownership[p].owner is None
        ]
        for pos in carol_positions[:3]:
            state.property_ownership[pos].owner = carol
            state.property_ownership[pos].has_hotel = True
        assert strategy._is_late_game(alice, state) is True

    def test_threshold_boundary_one_below_is_early_game(self, alice, bob, state):
        strategy = JailCamper(late_game_threshold=6)
        give_opponent_houses(state, bob, 5)
        assert strategy._is_late_game(alice, state) is False

    def test_threshold_boundary_exactly_at_threshold_is_late_game(
        self, alice, bob, state
    ):
        strategy = JailCamper(late_game_threshold=6)
        give_opponent_houses(state, bob, 6)
        assert strategy._is_late_game(alice, state) is True

    def test_custom_threshold_respected(self, alice, bob, state):
        strategy = JailCamper(late_game_threshold=3)
        give_opponent_houses(state, bob, 3)
        assert strategy._is_late_game(alice, state) is True


# ---------------------------------------------------------------------------
# get_jail_decision — early game
# ---------------------------------------------------------------------------


class TestGetJailDecisionEarlyGame:
    def test_pays_fine_when_affordable_and_early_game(self, alice, state):
        strategy = JailCamper()
        alice.cash = 1500
        assert strategy.get_jail_decision(alice, state) == JailDecision.PAY_FINE

    def test_uses_goojf_when_cannot_afford_fine(self, alice, state):
        strategy = JailCamper()
        alice.cash = 10  # below $50 fine
        card = Card(id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        assert strategy.get_jail_decision(alice, state) == JailDecision.USE_GOOJF

    def test_rolls_doubles_when_no_fine_and_no_goojf(self, alice, state):
        strategy = JailCamper()
        alice.cash = 10
        assert strategy.get_jail_decision(alice, state) == JailDecision.ROLL_DOUBLES

    def test_pays_fine_preferred_over_goojf_early_game(self, alice, state):
        """When affordable, PAY_FINE takes priority over USE_GOOJF."""
        strategy = JailCamper()
        alice.cash = 1500
        card = Card(id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        assert strategy.get_jail_decision(alice, state) == JailDecision.PAY_FINE

    def test_exactly_50_can_afford_fine(self, alice, state):
        strategy = JailCamper()
        alice.cash = 50
        assert strategy.get_jail_decision(alice, state) == JailDecision.PAY_FINE

    def test_49_cannot_afford_fine_uses_goojf(self, alice, state):
        strategy = JailCamper()
        alice.cash = 49
        card = Card(id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        assert strategy.get_jail_decision(alice, state) == JailDecision.USE_GOOJF


# ---------------------------------------------------------------------------
# get_jail_decision — late game
# ---------------------------------------------------------------------------


class TestGetJailDecisionLateGame:
    def test_always_rolls_doubles_in_late_game(self, alice, bob, state):
        strategy = JailCamper()
        give_opponent_houses(state, bob, 6)
        alice.cash = 1500  # can afford fine, but still stays
        assert strategy.get_jail_decision(alice, state) == JailDecision.ROLL_DOUBLES

    def test_rolls_doubles_even_with_goojf_in_late_game(self, alice, bob, state):
        strategy = JailCamper()
        give_opponent_houses(state, bob, 6)
        alice.cash = 1500
        card = Card(id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        assert strategy.get_jail_decision(alice, state) == JailDecision.ROLL_DOUBLES

    def test_rolls_doubles_even_when_broke_in_late_game(self, alice, bob, state):
        strategy = JailCamper()
        give_opponent_houses(state, bob, 6)
        alice.cash = 0
        assert strategy.get_jail_decision(alice, state) == JailDecision.ROLL_DOUBLES


# ---------------------------------------------------------------------------
# Delegation — buying/building must not duplicate BuyEverything logic
# ---------------------------------------------------------------------------


class TestBuyingDelegation:
    def test_should_buy_delegates_to_buy_everything(self, alice, state):
        """JailCamper buying behaviour matches BuyEverything exactly."""
        jail_camper = JailCamper()
        buy_everything = BuyEverything()
        sq = state.board.get_square(1)

        for cash in [0, 59, 60, 1500]:
            alice.cash = cash
            assert jail_camper.should_buy_property(
                alice, sq, state
            ) == buy_everything.should_buy_property(alice, sq, state)

    def test_choose_properties_to_build_delegates_to_buy_everything(self, alice, state):
        """Build orders from JailCamper match BuyEverything for same state."""
        jail_camper = JailCamper()
        buy_everything = BuyEverything()

        # Give alice complete brown group
        state.property_ownership[1].owner = alice
        state.property_ownership[3].owner = alice
        alice.cash = 5000

        jc_orders = jail_camper.choose_properties_to_build(alice, state)
        be_orders = buy_everything.choose_properties_to_build(alice, state)
        assert jc_orders == be_orders

    def test_mortgage_logic_delegates_to_buy_everything(self, alice, state):
        """Mortgage ordering matches BuyEverything."""
        jail_camper = JailCamper()
        buy_everything = BuyEverything()

        state.property_ownership[1].owner = alice
        state.property_ownership[5].owner = alice

        jc_result = jail_camper.choose_properties_to_mortgage(alice, 100, state)
        be_result = buy_everything.choose_properties_to_mortgage(alice, 100, state)
        assert jc_result == be_result


# ---------------------------------------------------------------------------
# Trade decisions
# ---------------------------------------------------------------------------


class TestTradeDecisions:
    def test_never_accepts_trade(self, alice, state):
        strategy = JailCamper()
        trade = TradeOffer(offered_positions=[], requested_positions=[1])
        assert strategy.should_accept_trade(alice, trade, state) is False

    def test_never_proposes_trade(self, alice, state):
        strategy = JailCamper()
        assert strategy.propose_trade(alice, state) is None

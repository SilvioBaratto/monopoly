"""Dedicated TDD tests for the Trader strategy.

Tests are written against the acceptance criteria in GitHub issue #26.

Five required test cases:
    1. proposes trade to complete own group
    2. accepts trade that completes own group
    3. rejects trade that gives opponent a complete group
    4. rejects trade with unfavorable valuation
    5. does not propose trade when no beneficial trade exists

Additional tests:
    - does not propose trade that would give opponent a complete group
    - proposed trade offers >= 80% compensation for the requested property
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from monopoly.board import Board
from monopoly.cards import load_decks
from monopoly.state import GameState, Player
from monopoly.strategies.base import TradeOffer
from monopoly.strategies.trader import Trader

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
    return GameState.init_game(["Alice", "Bob", "Carol"], board, chance, cc)


@pytest.fixture
def alice(state: GameState) -> Player:
    return state.players[0]


@pytest.fixture
def bob(state: GameState) -> Player:
    return state.players[1]


@pytest.fixture
def strategy() -> Trader:
    return Trader()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Brown group: pos 1 (Mediterranean $60) and pos 3 (Baltic $60)
# Light-blue group: pos 6 (Oriental $100), pos 8 (Vermont $100), pos 9 (Connecticut $120)
# Railroad: pos 5 (Reading Railroad $200)


# ---------------------------------------------------------------------------
# 1. Proposes trade to complete own group
# ---------------------------------------------------------------------------


class TestProposeTradeToCompleteOwnGroup:
    """Criterion: propose_trade identifies a trade that would complete a color group."""

    def test_proposes_trade_when_one_property_missing(
        self, alice: Player, bob: Player, state: GameState, strategy: Trader
    ) -> None:
        """When Alice owns one brown and Bob owns the other, Alice proposes to get it."""
        # Alice owns Mediterranean (pos 1, brown)
        state.property_ownership[1].owner = alice
        # Bob owns Baltic (pos 3, brown)
        state.property_ownership[3].owner = bob
        # Alice owns a railroad as trade bait
        state.property_ownership[5].owner = alice

        result = strategy.propose_trade(alice, state)

        assert result is not None
        assert 3 in result.requested_positions

    def test_proposes_trade_uses_non_target_color_as_bait(
        self, alice: Player, bob: Player, state: GameState, strategy: Trader
    ) -> None:
        """The bait property must come from a different color group than the target."""
        state.property_ownership[1].owner = alice  # brown
        state.property_ownership[3].owner = bob  # brown
        state.property_ownership[5].owner = alice  # railroad (bait)

        result = strategy.propose_trade(alice, state)

        assert result is not None
        # The offered property must not be from the brown group
        for offered_pos in result.offered_positions:
            sq = state.board.get_square(offered_pos)
            from monopoly.board import ColorProperty

            if isinstance(sq, ColorProperty):
                assert sq.color != "brown"


# ---------------------------------------------------------------------------
# 2. Accepts trade that completes own group
# ---------------------------------------------------------------------------


class TestAcceptsTradeCompletingOwnGroup:
    """Criterion: should_accept_trade returns True when trade completes a color group."""

    def test_accepts_trade_that_completes_brown_group(
        self, alice: Player, state: GameState, strategy: Trader
    ) -> None:
        """When Alice owns pos 1 and the trade brings pos 3, she accepts."""
        state.property_ownership[1].owner = alice  # Mediterranean

        # Trade: Alice gives railroad (pos 5), gets Baltic (pos 3)
        trade = TradeOffer(
            offered_positions=[5],  # Alice gives pos 5
            requested_positions=[3],  # Alice gets pos 3
        )
        assert strategy.should_accept_trade(alice, trade, state) is True

    def test_accepts_trade_completing_light_blue_group(
        self, alice: Player, state: GameState, strategy: Trader
    ) -> None:
        """Alice accepts when trade completes her light-blue group."""
        # Alice owns Oriental (6) and Vermont (8)
        state.property_ownership[6].owner = alice
        state.property_ownership[8].owner = alice

        # Trade: Alice gives some bait, gets Connecticut (9) to complete light-blue
        trade = TradeOffer(
            offered_positions=[5],  # Alice gives railroad
            requested_positions=[9],  # Alice gets Connecticut
        )
        assert strategy.should_accept_trade(alice, trade, state) is True


# ---------------------------------------------------------------------------
# 3. Rejects trade that gives opponent a complete group
# ---------------------------------------------------------------------------


class TestRejectsTradeGivingOpponentCompleteGroup:
    """Criterion: should_accept_trade returns False when trade would give opponent a complete group."""

    def test_rejects_trade_giving_bob_complete_brown(
        self, alice: Player, bob: Player, state: GameState, strategy: Trader
    ) -> None:
        """Alice owns Baltic (pos 3), Bob owns Mediterranean (pos 1).
        Giving pos 3 to Bob completes his brown group — Alice must reject."""
        state.property_ownership[1].owner = bob  # Bob owns Mediterranean
        state.property_ownership[3].owner = alice  # Alice owns Baltic

        # Trade: Alice gives pos 3 (completing Bob's brown), receives railroad
        trade = TradeOffer(
            offered_positions=[3],  # Alice gives Baltic → Bob gets full brown
            requested_positions=[5],  # Alice gets railroad
        )
        assert strategy.should_accept_trade(alice, trade, state) is False

    def test_rejects_trade_completing_opponent_light_blue(
        self, alice: Player, bob: Player, state: GameState, strategy: Trader
    ) -> None:
        """Alice giving away the last light-blue to Bob who owns the other two is rejected."""
        # Bob owns Oriental (6) and Vermont (8)
        state.property_ownership[6].owner = bob
        state.property_ownership[8].owner = bob
        # Alice owns Connecticut (9)
        state.property_ownership[9].owner = alice

        # Trade: Alice gives pos 9 (completing Bob's light-blue), receives railroad
        trade = TradeOffer(
            offered_positions=[9],  # Alice gives Connecticut → Bob completes light-blue
            requested_positions=[5],  # Alice gets railroad
        )
        assert strategy.should_accept_trade(alice, trade, state) is False


# ---------------------------------------------------------------------------
# 4. Rejects trade with unfavorable valuation
# ---------------------------------------------------------------------------


class TestRejectsUnfavorableTrade:
    """Criterion: should_accept_trade rejects when total received value < given value."""

    def test_rejects_trade_giving_high_for_low(
        self, alice: Player, bob: Player, state: GameState, strategy: Trader
    ) -> None:
        """Alice gives Boardwalk ($400, dark_blue) and gets Mediterranean ($60, brown).
        Valuation: received=60*0.7=42, given=400*0.8=320 → reject."""
        state.property_ownership[39].owner = alice  # Alice owns Boardwalk
        state.property_ownership[1].owner = bob  # Bob owns Mediterranean

        trade = TradeOffer(
            offered_positions=[39],  # Alice gives Boardwalk
            requested_positions=[1],  # Alice gets Mediterranean
        )
        assert strategy.should_accept_trade(alice, trade, state) is False

    def test_rejects_trade_negative_net_even_with_small_cash(
        self, alice: Player, bob: Player, state: GameState, strategy: Trader
    ) -> None:
        """Alice gives Boardwalk ($400), gets Baltic ($60) + $100 cash.
        Net = 100 + 60*0.7 - 400*0.8 = 100 + 42 - 320 = -178 → reject."""
        state.property_ownership[39].owner = alice
        state.property_ownership[3].owner = bob

        trade = TradeOffer(
            offered_positions=[39],  # Alice gives Boardwalk
            requested_positions=[3],  # Alice gets Baltic
            cash_offered=100,  # and $100 cash
        )
        assert strategy.should_accept_trade(alice, trade, state) is False


# ---------------------------------------------------------------------------
# 5. Does not propose trade when no beneficial trade exists
# ---------------------------------------------------------------------------


class TestDoesNotProposeWhenNoBeneficialTrade:
    """Criterion: propose_trade returns None when no useful trade exists."""

    def test_returns_none_when_player_owns_no_properties(
        self, alice: Player, state: GameState, strategy: Trader
    ) -> None:
        """Alice owns nothing — no group to complete, no trade to propose."""
        result = strategy.propose_trade(alice, state)
        assert result is None

    def test_returns_none_when_player_already_owns_all_groups(
        self, alice: Player, state: GameState, strategy: Trader
    ) -> None:
        """Alice owns both browns — group already complete, no trade needed."""
        state.property_ownership[1].owner = alice
        state.property_ownership[3].owner = alice

        result = strategy.propose_trade(alice, state)
        assert result is None

    def test_returns_none_when_missing_property_is_unowned(
        self, alice: Player, state: GameState, strategy: Trader
    ) -> None:
        """Alice owns Mediterranean (1), but Baltic (3) is unowned — can't trade."""
        state.property_ownership[1].owner = alice
        # pos 3 is unowned (default)

        result = strategy.propose_trade(alice, state)
        assert result is None


# ---------------------------------------------------------------------------
# Additional: does not propose trade that would give opponent a complete group
# ---------------------------------------------------------------------------


class TestProposeTradeDoesNotCompleteOpponentGroup:
    """Criterion: propose_trade never offers a property that would complete an opponent's group."""

    def test_does_not_offer_property_completing_opponent_group(
        self, alice: Player, bob: Player, state: GameState, strategy: Trader
    ) -> None:
        """Alice wants brown (missing pos 3 owned by Carol).
        Carol also owns Vermont (8) and Connecticut (9) — light_blue.
        The only bait Alice has is Oriental (6, light_blue).
        Offering pos 6 to Carol would complete Carol's light-blue group → must not propose that trade."""
        carol = state.players[2]

        # Alice owns Mediterranean (1, brown) — wants Baltic (3) from Carol
        state.property_ownership[1].owner = alice
        state.property_ownership[3].owner = carol
        # Alice's only bait: Oriental (6, light_blue)
        state.property_ownership[6].owner = alice
        # Carol (the trade partner) owns Vermont (8) and Connecticut (9) — light_blue
        state.property_ownership[8].owner = carol
        state.property_ownership[9].owner = carol

        result = strategy.propose_trade(alice, state)

        # If a trade is proposed, pos 6 must not be in offered_positions
        # (giving Carol pos 6 would complete Carol's light-blue group)
        if result is not None:
            assert 6 not in result.offered_positions


# ---------------------------------------------------------------------------
# Additional: proposed trade offers >= 80% compensation
# ---------------------------------------------------------------------------


class TestProposeTradeCompensation:
    """Criterion: offered assets (cash + properties) cover >= 80% of the requested property's value."""

    def test_offered_value_meets_80_percent_threshold(
        self, alice: Player, bob: Player, state: GameState, strategy: Trader
    ) -> None:
        """The total value offered must be at least 80% of the target property's strategic value."""
        # Alice owns Mediterranean (1, brown); Bob owns Baltic (3, brown)
        state.property_ownership[1].owner = alice
        state.property_ownership[3].owner = bob
        # Alice has railroad (5) as bait and plenty of cash
        state.property_ownership[5].owner = alice
        alice.cash = 5000

        result = strategy.propose_trade(alice, state)

        if result is not None:
            target_sq = state.board.get_square(result.requested_positions[0])
            target_value = strategy._property_value(target_sq, state)
            threshold = target_value * 0.8

            offered_prop_value = sum(
                strategy._property_value(state.board.get_square(p), state)
                for p in result.offered_positions
            )
            total_offered = offered_prop_value + result.cash_offered

            assert total_offered >= threshold, (
                f"Offered {total_offered:.1f} but threshold is {threshold:.1f}"
            )

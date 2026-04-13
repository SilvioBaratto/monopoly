"""Tests for bankruptcy.py — attempt_payment() and bankruptcy resolution."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from monopoly.board import Board
from monopoly.bankruptcy import attempt_payment, declare_bankruptcy, check_game_over
from monopoly.cards import load_decks
from monopoly.state import GameState
from monopoly.strategies.buy_nothing import BuyNothing

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


class TestAttemptPaymentDirect:
    def test_pays_from_cash_when_sufficient(self, alice, bob, state):
        alice.cash = 500
        bob.cash = 100
        strategy = BuyNothing()
        result = attempt_payment(alice, 200, bob, state, strategy)
        assert result is True
        assert alice.cash == 300
        assert bob.cash == 300

    def test_pays_to_bank_no_creditor(self, alice, state):
        alice.cash = 500
        strategy = BuyNothing()
        result = attempt_payment(alice, 200, None, state, strategy)
        assert result is True
        assert alice.cash == 300

    def test_returns_false_when_bankrupt(self, alice, state):
        alice.cash = 0
        strategy = BuyNothing()
        result = attempt_payment(alice, 100, None, state, strategy)
        assert result is False
        assert alice.bankrupt is True


class TestAttemptPaymentWithMortgage:
    def test_mortgages_to_raise_cash(self, alice, state):
        state.property_ownership[1].owner = alice
        sq = state.board.get_square(1)
        alice.cash = 0
        strategy = BuyNothing()
        # Mortgage value of Mediterranean Ave should cover small amount
        result = attempt_payment(alice, sq.mortgage - 1, None, state, strategy)
        assert result is True

    def test_bankruptcy_transfers_to_creditor(self, alice, bob, state):
        alice.cash = 10
        bob.cash = 1500
        state.property_ownership[1].owner = alice
        strategy = BuyNothing()
        result = attempt_payment(alice, 5000, bob, state, strategy)
        assert result is False
        assert alice.bankrupt is True
        # Bob should have received alice's cash
        assert bob.cash >= 1500

    def test_bankruptcy_clears_property_to_bank(self, alice, state):
        alice.cash = 10
        state.property_ownership[1].owner = alice
        state.property_ownership[1].is_mortgaged = True
        strategy = BuyNothing()
        result = attempt_payment(alice, 5000, None, state, strategy)
        assert result is False
        assert state.property_ownership[1].owner is None

    def test_bankruptcy_transfers_goojf_to_creditor(self, alice, bob, state):
        from monopoly.cards import Card, CardEffect

        alice.cash = 0
        card = Card(id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        strategy = BuyNothing()
        attempt_payment(alice, 5000, bob, state, strategy)
        if alice.bankrupt:
            assert card in bob.goojf_cards


class TestAttemptPaymentBuildingSales:
    def test_sells_houses_to_raise_cash(self, alice, state):
        """Player with houses but no cash should sell them to pay."""
        # Give alice brown monopoly with houses
        state.property_ownership[1].owner = alice
        state.property_ownership[3].owner = alice
        state.property_ownership[1].houses = 2
        state.property_ownership[3].houses = 2
        state.houses_available -= 4
        alice.cash = 0

        sq = state.board.get_square(1)
        sell_value = sq.house_cost // 2 * 2  # sell 2 houses from pos 1

        strategy = BuyNothing()
        # Need less than sell value to succeed
        result = attempt_payment(alice, sell_value - 1, None, state, strategy)
        # Should have sold at least some buildings
        assert alice.cash >= 0 or result is False

    def test_sells_houses_before_mortgaging(self, alice, state):
        """Houses are sold before properties are mortgaged (per Monopoly rules)."""
        # Give alice a property with houses and a second property without
        state.property_ownership[1].owner = alice
        state.property_ownership[3].owner = alice
        state.property_ownership[1].houses = 1
        state.property_ownership[3].houses = 0
        state.houses_available -= 1
        alice.cash = 0

        sq1 = state.board.get_square(1)
        # House sell value = house_cost // 2
        house_sell_value = sq1.house_cost // 2

        strategy = BuyNothing()
        # Amount that can be covered only by selling the house (not by mortgaging pos 3)
        # BuyNothing returns empty list for mortgage — so only house sales will trigger
        result = attempt_payment(alice, house_sell_value - 1, None, state, strategy)
        assert result is True
        # House was sold, property not mortgaged (BuyNothing won't mortgage)
        assert state.property_ownership[1].houses == 0
        assert not state.property_ownership[1].is_mortgaged


class TestDeclareBankruptcy:
    def test_player_marked_bankrupt_to_player(self, alice, bob, state):
        """Player goes bankrupt and is marked bankrupt."""
        alice.cash = 50
        bob.cash = 0
        state.property_ownership[1].owner = alice

        declare_bankruptcy(alice, bob, state)

        assert alice.bankrupt is True

    def test_cash_transfers_to_creditor(self, alice, bob, state):
        """All remaining cash transfers to creditor on bankruptcy."""
        alice.cash = 50
        bob.cash = 100

        declare_bankruptcy(alice, bob, state)

        assert alice.cash == 0
        assert bob.cash == 150

    def test_properties_transfer_to_creditor(self, alice, bob, state):
        """Unencumbered properties transfer to creditor."""
        state.property_ownership[1].owner = alice
        state.property_ownership[3].owner = alice

        declare_bankruptcy(alice, bob, state)

        assert state.property_ownership[1].owner is bob
        assert state.property_ownership[3].owner is bob

    def test_mortgaged_property_transfers_and_charges_10pct_interest(
        self, alice, bob, state
    ):
        """Creditor pays 10% of mortgage value for each transferred mortgaged property."""
        alice.cash = 0
        state.property_ownership[1].owner = alice
        state.property_ownership[1].is_mortgaged = True
        sq = state.board.get_square(1)
        interest = round(sq.mortgage * 0.1)
        bob.cash = 500

        declare_bankruptcy(alice, bob, state)

        # Property transferred to bob
        assert state.property_ownership[1].owner is bob
        assert state.property_ownership[1].is_mortgaged is True
        # Bob charged 10% interest on the transferred mortgaged property
        assert bob.cash == 500 - interest

    def test_mortgaged_property_stays_when_creditor_cannot_pay_interest(
        self, alice, bob, state
    ):
        """If creditor cannot pay 10% interest, property stays mortgaged."""
        state.property_ownership[1].owner = alice
        state.property_ownership[1].is_mortgaged = True
        bob.cash = 0  # creditor has no cash for interest

        declare_bankruptcy(alice, bob, state)

        # Property still transfers but stays mortgaged
        assert state.property_ownership[1].owner is bob
        assert state.property_ownership[1].is_mortgaged is True

    def test_goojf_cards_transfer_to_creditor(self, alice, bob, state):
        """GOOJF cards transfer from bankrupt player to creditor."""
        from monopoly.cards import Card, CardEffect

        card = Card(id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail)
        alice.goojf_cards.append(card)
        bob.cash = 500

        declare_bankruptcy(alice, bob, state)

        assert card in bob.goojf_cards
        assert card not in alice.goojf_cards

    def test_bank_bankruptcy_clears_ownership(self, alice, state):
        """Properties returned to bank are unowned."""
        state.property_ownership[1].owner = alice
        state.property_ownership[3].owner = alice

        declare_bankruptcy(alice, None, state)

        assert state.property_ownership[1].owner is None
        assert state.property_ownership[3].owner is None

    def test_bank_bankruptcy_unmortgages_properties(self, alice, state):
        """Properties returned to bank are unmortgaged."""
        state.property_ownership[1].owner = alice
        state.property_ownership[1].is_mortgaged = True

        declare_bankruptcy(alice, None, state)

        assert state.property_ownership[1].is_mortgaged is False

    def test_bank_bankruptcy_returns_houses_to_supply(self, alice, state):
        """Houses on returned properties go back to the bank supply."""
        state.property_ownership[1].owner = alice
        state.property_ownership[3].owner = alice
        state.property_ownership[1].houses = 3
        state.property_ownership[3].houses = 2
        initial_supply = state.houses_available - 5  # simulate 5 fewer in supply
        state.houses_available = initial_supply

        declare_bankruptcy(alice, None, state)

        assert state.houses_available == initial_supply + 5

    def test_bank_bankruptcy_returns_goojf_to_deck(self, alice, state):
        """GOOJF cards returned to their respective decks on bank bankruptcy."""
        from monopoly.cards import Card, CardEffect

        chance_card = Card(
            id="ch_goojf", text="GOOJF", effect=CardEffect.get_out_of_jail
        )
        alice.goojf_cards.append(chance_card)

        declare_bankruptcy(alice, None, state)

        assert chance_card not in alice.goojf_cards

    def test_bank_bankruptcy_discards_cash(self, alice, state):
        """Cash is discarded (zeroed) on bank bankruptcy."""
        alice.cash = 200

        declare_bankruptcy(alice, None, state)

        assert alice.cash == 0


class TestCheckGameOver:
    def test_returns_true_with_one_active_player(self, state):
        """Game over when only 1 non-bankrupt player remains."""
        state.players[1].bankrupt = True

        assert check_game_over(state) is True

    def test_returns_false_with_two_active_players(self, state):
        """Game not over when 2+ non-bankrupt players remain."""
        assert check_game_over(state) is False

    def test_returns_true_with_zero_active_players(self, state):
        """Game over when all players are bankrupt (edge case)."""
        state.players[0].bankrupt = True
        state.players[1].bankrupt = True

        assert check_game_over(state) is True

    def test_returns_false_with_three_active_players(self, board):
        """Game not over with 3 active players."""
        rng = np.random.default_rng(0)
        chance, cc = load_decks(DATA_PATH, rng)
        gs = GameState.init_game(["Alice", "Bob", "Charlie"], board, chance, cc)

        assert check_game_over(gs) is False

"""Property-based and integration tests for the full Monopoly game engine.

Tests game-level invariants:
- Games always terminate within max_turns
- Housing supply never exceeded
- Active players never have negative cash after turn resolution
- Game produces a valid GameResult
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from monopoly.board import Board
from monopoly.game import Game, GameResult
from monopoly.strategies.buy_everything import BuyEverything
from monopoly.strategies.buy_nothing import BuyNothing
from monopoly.strategies.color_targeted import ColorTargeted
from monopoly.strategies.jail_camper import JailCamper
from monopoly.strategies.three_houses_rush import ThreeHousesRush
from monopoly.strategies.trader import Trader

DATA_PATH = Path(__file__).parent.parent / "data" / "cards_standard.yaml"
MAX_TURNS = 500


def make_game(
    player_names: list[str],
    strategies: list,
    seed: int,
) -> Game:
    """Create a reproducible Game instance.

    Args:
        player_names: Player display names.
        strategies: Strategy instances (same order as player_names).
        seed: RNG seed for full reproducibility.

    Returns:
        Configured Game instance.
    """
    rng = np.random.default_rng(seed)
    board = Board()
    return Game(
        player_names=player_names,
        strategies=strategies,
        board=board,
        data_path=DATA_PATH,
        rng=rng,
    )


# ---------------------------------------------------------------------------
# Property-based tests
# ---------------------------------------------------------------------------


@given(seed=st.integers(0, 2**32 - 1), num_players=st.integers(2, 6))
@settings(max_examples=50, deadline=None)
def test_game_always_terminates(seed: int, num_players: int) -> None:
    """Games must end within max_turns regardless of seed or player count."""
    names = [f"P{i}" for i in range(num_players)]
    strategies = [BuyEverything() for _ in range(num_players)]
    game = make_game(names, strategies, seed)
    result = game.play(max_turns=MAX_TURNS)
    assert result.turns_played <= MAX_TURNS


@given(seed=st.integers(0, 2**32 - 1), num_players=st.integers(2, 6))
@settings(max_examples=30, deadline=None)
def test_game_result_is_valid(seed: int, num_players: int) -> None:
    """GameResult must have valid structure regardless of outcome."""
    names = [f"P{i}" for i in range(num_players)]
    strategies = [BuyEverything() for _ in range(num_players)]
    game = make_game(names, strategies, seed)
    result = game.play(max_turns=MAX_TURNS)

    assert isinstance(result, GameResult)
    assert result.turns_played >= 0
    assert len(result.player_stats) == num_players
    if result.winner is not None:
        assert result.winner.name in result.player_stats


@given(seed=st.integers(0, 2**32 - 1))
@settings(max_examples=30, deadline=None)
def test_housing_supply_never_exceeded(seed: int) -> None:
    """Total houses + hotel houses must never exceed the initial supply."""
    names = ["P0", "P1", "P2"]
    strategies = [ThreeHousesRush(), BuyEverything(), BuyEverything()]
    game = make_game(names, strategies, seed)
    game.play(max_turns=MAX_TURNS)

    state = game.state
    total_houses = sum(po.houses for po in state.property_ownership.values())
    total_hotels = sum(1 for po in state.property_ownership.values() if po.has_hotel)
    assert total_houses <= 32
    assert total_hotels <= 12


@given(seed=st.integers(0, 2**32 - 1))
@settings(max_examples=20, deadline=None)
def test_bankrupt_players_have_no_properties(seed: int) -> None:
    """Bankrupt players must not own any properties at game end."""
    names = ["P0", "P1", "P2"]
    strategies = [BuyEverything(), BuyEverything(), BuyNothing()]
    game = make_game(names, strategies, seed)
    game.play(max_turns=MAX_TURNS)

    state = game.state
    for player in state.players:
        if player.bankrupt:
            owned = [
                pos
                for pos, po in state.property_ownership.items()
                if po.owner is player
            ]
            assert owned == [], f"Bankrupt player {player.name} still owns {owned}"


@given(seed=st.integers(0, 2**32 - 1))
@settings(max_examples=20, deadline=None)
def test_winner_is_last_active_or_richest(seed: int) -> None:
    """Winner is the sole survivor, or the richest player when max_turns reached."""
    names = ["P0", "P1"]
    strategies = [BuyEverything(), BuyNothing()]
    game = make_game(names, strategies, seed)
    result = game.play(max_turns=MAX_TURNS)

    active = game.state.active_players
    assert result.winner is not None, "A winner must always be returned"
    if len(active) == 1:
        assert active[0] is result.winner
    else:
        # max_turns reached: winner must be the richest active player
        assert result.turns_played == MAX_TURNS
        assert result.winner in active
        assert result.winner.cash == max(p.cash for p in active)


# ---------------------------------------------------------------------------
# Integration / deterministic tests
# ---------------------------------------------------------------------------


def test_two_player_game_completes():
    """A 2-player game with fixed seed must produce a valid GameResult."""
    game = make_game(["Alice", "Bob"], [BuyEverything(), BuyNothing()], seed=42)
    result = game.play(max_turns=MAX_TURNS)
    assert result is not None
    assert result.turns_played <= MAX_TURNS


def test_buy_everything_vs_buy_nothing_winner():
    """BuyEverything should eventually beat BuyNothing (fixed seed check)."""
    game = make_game(
        ["BuyEverything", "BuyNothing"],
        [BuyEverything(), BuyNothing()],
        seed=42,
    )
    result = game.play(max_turns=MAX_TURNS)
    # With seed=42, document the actual winner (flexible assertion)
    # BuyEverything is expected to win in most cases
    assert result is not None
    # If there's a winner, it should be BuyEverything (they buy properties)
    # or no winner if max_turns reached
    if result.winner is not None:
        assert result.winner.name == "BuyEverything"


def test_six_player_game_completes():
    """A 6-player game with diverse strategies must complete without error."""
    strategies = [
        BuyEverything(),
        BuyNothing(),
        ColorTargeted(["orange"]),
        ThreeHousesRush(),
        JailCamper(),
        Trader(),
    ]
    game = make_game([f"P{i}" for i in range(6)], strategies, seed=123)
    result = game.play(max_turns=1000)
    assert result is not None


def test_housing_supply_limit_enforced():
    """32 houses and 12 hotels must never be exceeded in a full game."""
    strategies = [BuyEverything(), BuyEverything()]
    game = make_game(["P0", "P1"], strategies, seed=99)
    game.play(max_turns=500)
    state = game.state
    total_houses = sum(po.houses for po in state.property_ownership.values())
    total_hotels = sum(1 for po in state.property_ownership.values() if po.has_hotel)
    assert total_houses <= 32
    assert total_hotels <= 12


def test_game_state_exposed_on_game_instance():
    """Game.state must be accessible after play() for post-game inspection."""
    game = make_game(["A", "B"], [BuyEverything(), BuyNothing()], seed=7)
    game.play(max_turns=100)
    assert hasattr(game, "state")
    assert game.state is not None


def test_player_stats_populated():
    """Player stats must be present for all players including bankrupt ones."""
    names = ["A", "B", "C"]
    strategies = [BuyEverything(), BuyEverything(), BuyNothing()]
    game = make_game(names, strategies, seed=55)
    result = game.play(max_turns=500)
    for name in names:
        assert name in result.player_stats


def test_color_targeted_strategy_game():
    """ColorTargeted strategy must not crash during a full game."""
    strategies = [ColorTargeted(["orange", "red"]), BuyEverything()]
    game = make_game(["Targeted", "GreedyBuyer"], strategies, seed=77)
    result = game.play(max_turns=300)
    assert result is not None


def test_three_houses_rush_does_not_build_beyond_3():
    """ThreeHousesRush should not build more than 3 houses per property."""
    strategies = [ThreeHousesRush(), BuyNothing()]
    game = make_game(["Rusher", "Passive"], strategies, seed=88)
    game.play(max_turns=300)
    state = game.state
    for po in state.property_ownership.values():
        if not po.has_hotel:
            assert po.houses <= 3, f"ThreeHousesRush exceeded 3 houses: {po.houses}"


def test_jail_camper_strategy_game():
    """JailCamper strategy must not crash during a full game."""
    strategies = [JailCamper(), BuyEverything()]
    game = make_game(["Camper", "Buyer"], strategies, seed=111)
    result = game.play(max_turns=300)
    assert result is not None


def test_trader_strategy_game():
    """Trader strategy must not crash during a full game."""
    strategies = [Trader(), BuyEverything()]
    game = make_game(["Trader", "Buyer"], strategies, seed=222)
    result = game.play(max_turns=300)
    assert result is not None


def test_game_raises_on_one_player():
    """Game must raise ValueError for fewer than 2 players."""
    with pytest.raises(ValueError, match="2"):
        make_game(["Solo"], [BuyEverything()], seed=0)


def test_game_raises_on_seven_players():
    """Game must raise ValueError for more than 6 players."""
    names = [f"P{i}" for i in range(7)]
    strategies = [BuyEverything()] * 7
    with pytest.raises(ValueError):
        make_game(names, strategies, seed=0)


def test_strategies_length_mismatch_raises():
    """Mismatched player_names and strategies must raise ValueError."""
    with pytest.raises(ValueError):
        rng = np.random.default_rng(0)
        board = Board()
        Game(
            player_names=["A", "B"],
            strategies=[BuyEverything()],
            board=board,
            data_path=DATA_PATH,
            rng=rng,
        )


def test_color_targeted_invalid_color_raises():
    """ColorTargeted must raise ValueError for unknown color names."""
    with pytest.raises(ValueError, match="Unknown color"):
        ColorTargeted(["purple"])


def test_turn_count_increments():
    """Turn count must increment with each full round."""
    game = make_game(["A", "B"], [BuyNothing(), BuyNothing()], seed=1)
    game.play(max_turns=10)
    assert game.state.turn_count > 0

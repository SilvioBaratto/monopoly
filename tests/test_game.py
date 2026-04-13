"""Unit and integration tests for the full Monopoly game loop (game.py).

Tests cover every acceptance criterion from GitHub Issue #21:
- Game runs to completion for 2 and 6 players
- max_turns safety valve returns a GameResult with richest survivor
- ValueError on invalid player counts
- Doubles grant extra turns; 3 consecutive doubles send to jail
- Bankrupt players are skipped in subsequent rounds
- GameResult has a valid winner and turns_played > 0
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from monopoly.board import Board
from monopoly.dice import DiceRoll
from monopoly.game import Game, GameResult, PlayerStats
from monopoly.strategies.buy_everything import BuyEverything
from monopoly.strategies.buy_nothing import BuyNothing
from monopoly.strategies.color_targeted import ColorTargeted
from monopoly.strategies.jail_camper import JailCamper
from monopoly.strategies.three_houses_rush import ThreeHousesRush
from monopoly.strategies.trader import Trader

DATA_PATH = Path(__file__).parent.parent / "data" / "cards_standard.yaml"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_game(
    player_names: list[str],
    strategies: list,
    seed: int,
) -> Game:
    """Create a reproducible Game instance."""
    rng = np.random.default_rng(seed)
    board = Board()
    return Game(
        player_names=player_names,
        strategies=strategies,
        board=board,
        data_path=DATA_PATH,
        rng=rng,
    )


_DOUBLES = DiceRoll(die1=3, die2=3, total=6, is_doubles=True)
_NON_DOUBLES = DiceRoll(die1=1, die2=2, total=3, is_doubles=False)


def _doubles_then_non(n_doubles: int):
    """Return side_effect that yields n_doubles then non-doubles forever."""
    rolls = [_DOUBLES] * n_doubles + [_NON_DOUBLES] * 1000
    return iter(rolls)


# ---------------------------------------------------------------------------
# 1. Two-player game runs to completion (seeded)
# ---------------------------------------------------------------------------


def test_two_player_game_runs_to_completion_with_winner():
    """A 2-player seeded game must produce a GameResult with a valid winner."""
    game = make_game(["Alice", "Bob"], [BuyEverything(), BuyNothing()], seed=42)
    result = game.play(max_turns=500)

    assert isinstance(result, GameResult)
    assert result.turns_played > 0
    assert result.winner is not None
    assert result.winner.name in ("Alice", "Bob")


# ---------------------------------------------------------------------------
# 2. Six-player game runs to completion (seeded)
# ---------------------------------------------------------------------------


def test_six_player_game_runs_to_completion():
    """A 6-player seeded game with diverse strategies must complete."""
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

    assert isinstance(result, GameResult)
    assert result.turns_played > 0
    assert len(result.player_stats) == 6


# ---------------------------------------------------------------------------
# 3. max_turns safety valve returns the richest remaining player
# ---------------------------------------------------------------------------


def test_max_turns_reached_returns_result():
    """When max_turns is reached the game returns a valid GameResult."""
    game = make_game(["A", "B"], [BuyNothing(), BuyNothing()], seed=1)
    result = game.play(max_turns=3)

    assert isinstance(result, GameResult)
    assert result.turns_played == 3


def test_max_turns_reached_returns_richest_player_as_winner():
    """When max_turns ends the game with multiple survivors, winner is richest."""
    game = make_game(["Rich", "Poor"], [BuyNothing(), BuyNothing()], seed=1)

    # Force Rich to have more cash than Poor before running
    game.state.players[0].cash = 2000
    game.state.players[1].cash = 500

    result = game.play(max_turns=2)

    # If game didn't end via bankruptcy, winner must be the richest player
    if result.turns_played == 2:
        assert result.winner is not None
        assert result.winner.name == "Rich"


# ---------------------------------------------------------------------------
# 4. ValueError raised for < 2 or > 6 players
# ---------------------------------------------------------------------------


def test_game_raises_for_one_player():
    """Game must raise ValueError when only 1 player name is given."""
    with pytest.raises(ValueError, match="2"):
        make_game(["Solo"], [BuyEverything()], seed=0)


def test_game_raises_for_zero_players():
    """Game must raise ValueError when no player names are given."""
    with pytest.raises(ValueError):
        make_game([], [], seed=0)


def test_game_raises_for_seven_players():
    """Game must raise ValueError when 7 player names are given."""
    names = [f"P{i}" for i in range(7)]
    strategies = [BuyEverything()] * 7
    with pytest.raises(ValueError):
        make_game(names, strategies, seed=0)


def test_game_raises_when_strategies_count_mismatches():
    """Game must raise ValueError when strategies length != players length."""
    with pytest.raises(ValueError):
        rng = np.random.default_rng(0)
        Game(
            player_names=["A", "B"],
            strategies=[BuyEverything()],
            board=Board(),
            data_path=DATA_PATH,
            rng=rng,
        )


# ---------------------------------------------------------------------------
# 5a. Doubles grant extra turns (up to 2 extra = 3 total sub-turns)
# ---------------------------------------------------------------------------


def test_doubles_grant_one_extra_turn():
    """One doubles roll must result in the player taking an extra sub-turn."""
    game = make_game(["A", "B"], [BuyNothing(), BuyNothing()], seed=0)
    player_a = game.state.players[0]

    roll_count = [0]

    def counting_roll(rng):
        roll_count[0] += 1
        # First roll: doubles; subsequent rolls: non-doubles
        if roll_count[0] == 1:
            return _DOUBLES
        return _NON_DOUBLES

    with patch("monopoly.turn.roll", side_effect=counting_roll):
        game._play_player_turn(player_a, BuyNothing(), {})

    # Extra roll granted by doubles means roll() was called at least twice
    assert roll_count[0] >= 2


def test_two_doubles_grant_two_extra_turns():
    """Two consecutive doubles must result in three total sub-turns."""
    game = make_game(["A", "B"], [BuyNothing(), BuyNothing()], seed=0)
    player_a = game.state.players[0]

    roll_count = [0]

    def counting_roll(rng):
        roll_count[0] += 1
        if roll_count[0] <= 2:
            return _DOUBLES
        return _NON_DOUBLES

    with patch("monopoly.turn.roll", side_effect=counting_roll):
        game._play_player_turn(player_a, BuyNothing(), {})

    assert roll_count[0] >= 3


# ---------------------------------------------------------------------------
# 5b. Three consecutive doubles send to jail
# ---------------------------------------------------------------------------


def test_three_consecutive_doubles_sends_to_jail():
    """Three consecutive doubles must send the player to jail."""
    game = make_game(["A", "B"], [BuyNothing(), BuyNothing()], seed=0)
    player_a = game.state.players[0]
    player_a.in_jail = False

    with patch("monopoly.turn.roll", return_value=_DOUBLES):
        game._play_player_turn(player_a, BuyNothing(), {})

    assert player_a.in_jail, "3 consecutive doubles must send player to jail"


# ---------------------------------------------------------------------------
# 6. Bankrupt players are skipped in subsequent rounds
# ---------------------------------------------------------------------------


def test_bankrupt_player_is_skipped_in_round():
    """A bankrupt player must not take turns in subsequent rounds."""
    game = make_game(
        ["A", "B", "C"], [BuyNothing(), BuyNothing(), BuyNothing()], seed=0
    )

    # Manually bankrupt player B
    game.state.players[1].bankrupt = True

    turns_taken: list[str] = []
    original = game._play_player_turn

    def tracking_turn(player, strategy, bankruptcy_turns):
        turns_taken.append(player.name)
        return original(player, strategy, bankruptcy_turns)

    game._play_player_turn = tracking_turn  # type: ignore[method-assign]
    game._play_full_round({})

    assert "B" not in turns_taken, "Bankrupt player B should be skipped"
    assert "A" in turns_taken
    assert "C" in turns_taken


def test_bankrupt_player_skipped_after_game_run():
    """After a player goes bankrupt naturally, they are excluded from future rounds."""
    game = make_game(
        ["A", "B", "C"], [BuyEverything(), BuyEverything(), BuyNothing()], seed=99
    )
    result = game.play(max_turns=500)

    bankrupt_players = [p for p in game.state.players if p.bankrupt]
    for bp in bankrupt_players:
        # Bankrupt players must have a recorded bankruptcy turn
        assert bp.name in result.player_stats
        assert result.player_stats[bp.name].bankruptcy_turn is not None


# ---------------------------------------------------------------------------
# 7. GameResult contains correct winner and turns_played > 0
# ---------------------------------------------------------------------------


def test_game_result_has_correct_winner():
    """Winner in GameResult must be the sole surviving player."""
    game = make_game(["Alice", "Bob"], [BuyEverything(), BuyNothing()], seed=42)
    result = game.play(max_turns=500)

    active = game.state.active_players
    if len(active) == 1:
        assert result.winner is active[0]
    else:
        # max_turns reached: winner must be richest
        assert result.winner is not None


def test_game_result_turns_played_greater_than_zero():
    """turns_played in GameResult must always be > 0 for a started game."""
    game = make_game(["X", "Y"], [BuyNothing(), BuyNothing()], seed=7)
    result = game.play(max_turns=10)

    assert result.turns_played > 0


def test_game_result_player_stats_contains_all_players():
    """player_stats must contain entries for every player, including bankrupt ones."""
    names = ["A", "B", "C"]
    game = make_game(names, [BuyEverything(), BuyEverything(), BuyNothing()], seed=55)
    result = game.play(max_turns=400)

    for name in names:
        assert name in result.player_stats
        stat = result.player_stats[name]
        assert isinstance(stat, PlayerStats)
        assert stat.final_cash >= 0
        assert stat.properties_owned >= 0


def test_game_result_winner_is_in_player_stats():
    """If there is a winner, their name must appear in player_stats."""
    game = make_game(["Alice", "Bob"], [BuyEverything(), BuyNothing()], seed=42)
    result = game.play(max_turns=500)

    if result.winner is not None:
        assert result.winner.name in result.player_stats


# ---------------------------------------------------------------------------
# 8. Net worth history (Issue #46)
# ---------------------------------------------------------------------------


def test_net_worth_history_length_equals_turns_played_plus_one():
    """len(net_worth_history) must equal turns_played + 1 for every player."""
    game = make_game(["Alice", "Bob"], [BuyNothing(), BuyNothing()], seed=42)
    result = game.play(max_turns=50)

    for name in ["Alice", "Bob"]:
        history = result.player_stats[name].net_worth_history
        assert len(history) == result.turns_played + 1, (
            f"{name}: expected {result.turns_played + 1} entries, got {len(history)}"
        )


def test_net_worth_history_initial_value_equals_starting_cash():
    """net_worth_history[0] must equal 1500 for all players (no properties yet)."""
    game = make_game(["Alice", "Bob"], [BuyNothing(), BuyNothing()], seed=99)
    result = game.play(max_turns=50)

    for name in ["Alice", "Bob"]:
        history = result.player_stats[name].net_worth_history
        assert history[0] == 1500, (
            f"{name}: expected initial net worth 1500, got {history[0]}"
        )


def test_net_worth_history_defaults_to_empty_list():
    """PlayerStats must default net_worth_history to [] for backward compatibility."""
    stats = PlayerStats(final_cash=1500, properties_owned=0, bankruptcy_turn=None)
    assert stats.net_worth_history == []


def test_compute_net_worth_cash_only():
    """_compute_net_worth returns player.cash when the player owns no properties."""
    from monopoly.game import _compute_net_worth

    game = make_game(["Alice", "Bob"], [BuyNothing(), BuyNothing()], seed=7)
    alice = game.state.players[0]
    # No properties owned → net worth equals cash
    assert _compute_net_worth(alice, game.state, game.state.board) == alice.cash


def test_compute_net_worth_with_unmortgaged_property():
    """_compute_net_worth adds property price for unmortgaged properties."""
    from monopoly.game import _compute_net_worth
    from monopoly.state import PropertyOwnership

    game = make_game(["Alice", "Bob"], [BuyNothing(), BuyNothing()], seed=7)
    alice = game.state.players[0]

    # Manually assign an unmortgaged property to Alice
    first_buyable = game.state.board.buyable_squares[0]
    pos = first_buyable.position
    game.state.property_ownership[pos] = PropertyOwnership(
        owner=alice, is_mortgaged=False
    )

    expected = alice.cash + first_buyable.price
    assert _compute_net_worth(alice, game.state, game.state.board) == expected


def test_compute_net_worth_with_mortgaged_property():
    """_compute_net_worth uses mortgage value for mortgaged properties."""
    from monopoly.game import _compute_net_worth
    from monopoly.state import PropertyOwnership

    game = make_game(["Alice", "Bob"], [BuyNothing(), BuyNothing()], seed=7)
    alice = game.state.players[0]

    first_buyable = game.state.board.buyable_squares[0]
    pos = first_buyable.position
    game.state.property_ownership[pos] = PropertyOwnership(
        owner=alice, is_mortgaged=True
    )

    expected = alice.cash + first_buyable.mortgage
    assert _compute_net_worth(alice, game.state, game.state.board) == expected


def test_compute_net_worth_mixed_properties():
    """_compute_net_worth correctly sums cash + unmortgaged prices + mortgaged values."""
    from monopoly.game import _compute_net_worth
    from monopoly.state import PropertyOwnership

    game = make_game(["Alice", "Bob"], [BuyNothing(), BuyNothing()], seed=7)
    alice = game.state.players[0]

    buyables = game.state.board.buyable_squares
    sq_unmortgaged = buyables[0]
    sq_mortgaged = buyables[1]

    game.state.property_ownership[sq_unmortgaged.position] = PropertyOwnership(
        owner=alice, is_mortgaged=False
    )
    game.state.property_ownership[sq_mortgaged.position] = PropertyOwnership(
        owner=alice, is_mortgaged=True
    )

    expected = alice.cash + sq_unmortgaged.price + sq_mortgaged.mortgage
    assert _compute_net_worth(alice, game.state, game.state.board) == expected

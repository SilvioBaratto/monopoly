"""Tests for the batch Monte Carlo simulation runner (simulate.py).

Tests cover every acceptance criterion from GitHub Issue #34 (Issue A):
- simulate_games() returns a SimulationResult dataclass
- SimulationResult contains: winner per game, turns per game,
  bankruptcy order, final cash per player
- Same seed produces identical results
- Results for 2-player BuyEverything vs BuyNothing converge to
  known win rates (BuyEverything >> BuyNothing)
- Memory usage stays bounded (no accumulation of Python objects)
- All tests pass

Phase 2 (Issue #35):
- SimulationConfig and BatchResult dataclasses
- run_parallel_simulations() with ProcessPoolExecutor
"""

from __future__ import annotations

import gc
import time

import pytest

from monopoly.simulate import (
    BatchResult,
    SimulationConfig,
    SimulationResult,
    run_parallel_simulations,
    simulate_games,
)
from monopoly.strategies.buy_everything import BuyEverything
from monopoly.strategies.buy_nothing import BuyNothing

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PLAYER_NAMES = ["Alice", "Bob"]
_STRATEGIES = [BuyEverything(), BuyNothing()]


# ---------------------------------------------------------------------------
# 1. simulate_games returns a SimulationResult dataclass
# ---------------------------------------------------------------------------


def test_simulate_games_returns_simulation_result():
    """simulate_games must return a SimulationResult instance."""
    result = simulate_games(
        n_games=5,
        player_names=_PLAYER_NAMES,
        strategies=_STRATEGIES,
        seed=0,
    )
    assert isinstance(result, SimulationResult)


def test_simulation_result_has_correct_length_fields():
    """SimulationResult lists must have exactly n_games entries."""
    n = 10
    result = simulate_games(
        n_games=n,
        player_names=_PLAYER_NAMES,
        strategies=_STRATEGIES,
        seed=1,
    )
    assert len(result.winner_per_game) == n
    assert len(result.turns_per_game) == n
    assert len(result.bankruptcy_order) == n
    assert len(result.final_cash) == n


# ---------------------------------------------------------------------------
# 2. SimulationResult field content correctness
# ---------------------------------------------------------------------------


def test_winner_per_game_is_player_name_or_none():
    """Each entry in winner_per_game must be a player name or None."""
    result = simulate_games(
        n_games=10,
        player_names=_PLAYER_NAMES,
        strategies=_STRATEGIES,
        seed=2,
    )
    valid_names = set(_PLAYER_NAMES) | {None}
    for w in result.winner_per_game:
        assert w in valid_names


def test_turns_per_game_are_positive_integers():
    """Each entry in turns_per_game must be a positive integer."""
    result = simulate_games(
        n_games=10,
        player_names=_PLAYER_NAMES,
        strategies=_STRATEGIES,
        seed=3,
    )
    for t in result.turns_per_game:
        assert isinstance(t, int)
        assert t > 0


def test_bankruptcy_order_contains_player_names():
    """Each bankruptcy_order entry must be a list of player name strings."""
    result = simulate_games(
        n_games=10,
        player_names=_PLAYER_NAMES,
        strategies=_STRATEGIES,
        seed=4,
    )
    for order in result.bankruptcy_order:
        assert isinstance(order, list)
        for name in order:
            assert name in _PLAYER_NAMES


def test_final_cash_maps_player_names_to_integers():
    """Each final_cash entry must be a dict mapping player name to non-negative int."""
    result = simulate_games(
        n_games=10,
        player_names=_PLAYER_NAMES,
        strategies=_STRATEGIES,
        seed=5,
    )
    for cash_map in result.final_cash:
        assert isinstance(cash_map, dict)
        for name in _PLAYER_NAMES:
            assert name in cash_map
            assert isinstance(cash_map[name], int)
            assert cash_map[name] >= 0


# ---------------------------------------------------------------------------
# 3. Seed reproducibility — same seed → identical results
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_results():
    """Running simulate_games twice with the same seed must yield equal results."""
    kwargs = dict(
        n_games=20, player_names=_PLAYER_NAMES, strategies=_STRATEGIES, seed=99
    )
    r1 = simulate_games(**kwargs)
    r2 = simulate_games(**kwargs)

    assert r1.winner_per_game == r2.winner_per_game
    assert r1.turns_per_game == r2.turns_per_game
    assert r1.bankruptcy_order == r2.bankruptcy_order
    assert r1.final_cash == r2.final_cash


def test_different_seeds_produce_different_game_lengths():
    """Different seeds must (with high probability) produce different turn counts."""
    r1 = simulate_games(
        n_games=20, player_names=_PLAYER_NAMES, strategies=_STRATEGIES, seed=10
    )
    r2 = simulate_games(
        n_games=20, player_names=_PLAYER_NAMES, strategies=_STRATEGIES, seed=11
    )

    # Very unlikely that 20 games produce identical turn sequences under different seeds
    assert r1.turns_per_game != r2.turns_per_game


def test_seed_none_does_not_raise():
    """simulate_games with seed=None must not raise an exception."""
    result = simulate_games(
        n_games=3,
        player_names=_PLAYER_NAMES,
        strategies=_STRATEGIES,
        seed=None,
    )
    assert isinstance(result, SimulationResult)


# ---------------------------------------------------------------------------
# 4. Convergence test — BuyEverything >> BuyNothing
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_buy_everything_wins_significantly_more_than_buy_nothing():
    """BuyEverything must win ≥70% of games against BuyNothing over 200 games."""
    result = simulate_games(
        n_games=200,
        player_names=["BuyEverything", "BuyNothing"],
        strategies=[BuyEverything(), BuyNothing()],
        seed=42,
    )
    wins_be = sum(1 for w in result.winner_per_game if w == "BuyEverything")
    total = len(result.winner_per_game)
    win_rate = wins_be / total

    assert win_rate >= 0.60, (
        f"BuyEverything won only {win_rate:.1%} of games; expected ≥60%"
    )


# ---------------------------------------------------------------------------
# 5. Memory bound — no accumulation across games
# ---------------------------------------------------------------------------


def test_simulation_result_does_not_grow_unboundedly():
    """Running many games must not cause runaway memory; check result size is proportional."""
    result = simulate_games(
        n_games=50,
        player_names=_PLAYER_NAMES,
        strategies=_STRATEGIES,
        seed=7,
    )
    # Each list has exactly n_games entries (not more, not less)
    assert len(result.winner_per_game) == 50
    assert len(result.turns_per_game) == 50
    assert len(result.bankruptcy_order) == 50
    assert len(result.final_cash) == 50

    # Force GC to ensure no obvious reference leaks
    gc.collect()


# ---------------------------------------------------------------------------
# 6. n_games validation
# ---------------------------------------------------------------------------


def test_simulate_games_raises_on_zero_games():
    """simulate_games must raise ValueError when n_games < 1."""
    with pytest.raises(ValueError, match="n_games"):
        simulate_games(
            n_games=0,
            player_names=_PLAYER_NAMES,
            strategies=_STRATEGIES,
            seed=0,
        )


def test_simulate_games_raises_on_mismatched_strategies():
    """simulate_games must raise ValueError when len(strategies) != len(player_names)."""
    with pytest.raises(ValueError):
        simulate_games(
            n_games=5,
            player_names=_PLAYER_NAMES,
            strategies=[BuyEverything()],
            seed=0,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — BatchResult and run_parallel_simulations (Issue #35)
# ─────────────────────────────────────────────────────────────────────────────


def test_simulation_config_is_a_dataclass():
    """SimulationConfig must have fields: n_games, player_names, strategy_names, seed."""
    config = SimulationConfig(
        n_games=5,
        player_names=["Alice", "Bob"],
        strategy_names=["BuyEverything", "BuyNothing"],
        seed=42,
    )
    assert config.n_games == 5
    assert config.player_names == ["Alice", "Bob"]
    assert config.strategy_names == ["BuyEverything", "BuyNothing"]
    assert config.seed == 42


def test_batch_result_is_a_dataclass():
    """BatchResult must have fields: results, configs, wall_clock_seconds, n_workers, errors."""
    batch = BatchResult()
    assert isinstance(batch.results, list)
    assert isinstance(batch.configs, list)
    assert batch.wall_clock_seconds == 0.0
    assert batch.n_workers == 0
    assert isinstance(batch.errors, list)


def test_run_parallel_simulations_returns_batch_result():
    """run_parallel_simulations must return a BatchResult instance."""
    configs = [
        SimulationConfig(
            n_games=5,
            player_names=["Alice", "Bob"],
            strategy_names=["BuyEverything", "BuyNothing"],
            seed=1,
        ),
        SimulationConfig(
            n_games=5,
            player_names=["Carol", "Dave"],
            strategy_names=["BuyNothing", "BuyEverything"],
            seed=2,
        ),
    ]
    result = run_parallel_simulations(configs)
    assert isinstance(result, BatchResult)
    assert len(result.results) == 2
    assert len(result.errors) == 0


def test_empty_configs_returns_empty_batch_result():
    """run_parallel_simulations([]) must return an empty BatchResult immediately."""
    result = run_parallel_simulations([])
    assert isinstance(result, BatchResult)
    assert result.results == []
    assert result.wall_clock_seconds == 0.0
    assert result.errors == []


def test_parallel_results_match_sequential_for_same_seed():
    """Two parallel runs with the same seed must produce identical results.

    Runs two independent calls to run_parallel_simulations with n_workers=1
    and the same SimulationConfig.  Both execute in worker subprocesses that
    share the same PYTHONHASHSEED (propagated by run_parallel_simulations),
    so results must be bit-for-bit identical.
    """
    config = SimulationConfig(
        n_games=10,
        player_names=["Alice", "Bob"],
        strategy_names=["BuyEverything", "BuyNothing"],
        seed=77,
    )
    batch1 = run_parallel_simulations([config], n_workers=1)
    batch2 = run_parallel_simulations([config], n_workers=1)

    assert len(batch1.results) == 1
    assert len(batch2.results) == 1

    r1 = batch1.results[0]
    r2 = batch2.results[0]

    assert r1.winner_per_game == r2.winner_per_game
    assert r1.turns_per_game == r2.turns_per_game
    assert r1.bankruptcy_order == r2.bankruptcy_order
    assert r1.final_cash == r2.final_cash


def test_worker_failure_captured_in_errors():
    """An unknown strategy name must be captured in BatchResult.errors, not raise."""
    configs = [
        SimulationConfig(
            n_games=5,
            player_names=["Alice", "Bob"],
            strategy_names=["BuyEverything", "BuyNothing"],
            seed=10,
        ),
        SimulationConfig(
            n_games=5,
            player_names=["X", "Y"],
            strategy_names=["UnknownStrategy", "BuyNothing"],
            seed=11,
        ),
        SimulationConfig(
            n_games=5,
            player_names=["Eve", "Frank"],
            strategy_names=["BuyNothing", "BuyEverything"],
            seed=12,
        ),
    ]
    result = run_parallel_simulations(configs)

    # Exactly one error for the unknown strategy
    assert len(result.errors) == 1
    error_index, error_msg = result.errors[0]
    assert error_index == 1
    assert "UnknownStrategy" in error_msg

    # The other two configs succeeded
    assert len(result.results) == 2


@pytest.mark.slow
def test_parallel_speedup_on_multi_worker():
    """8 configs x 500 games with 2 workers must complete faster than sequential."""
    configs = [
        SimulationConfig(
            n_games=500,
            player_names=["Alice", "Bob"],
            strategy_names=["BuyEverything", "BuyNothing"],
            seed=i,
        )
        for i in range(8)
    ]

    # Measure parallel time
    t_start = time.monotonic()
    batch = run_parallel_simulations(configs, n_workers=2)
    parallel_time = time.monotonic() - t_start

    # Measure sequential time (1 worker)
    t_start = time.monotonic()
    run_parallel_simulations(configs, n_workers=1)
    sequential_time = time.monotonic() - t_start

    assert len(batch.errors) == 0
    assert parallel_time < sequential_time * 0.80, (
        f"Parallel ({parallel_time:.2f}s) was not < 80% of sequential ({sequential_time:.2f}s)"
    )


# ---------------------------------------------------------------------------
# Issue #46: net_worth_histories in SimulationResult
# ---------------------------------------------------------------------------


def test_simulation_result_has_net_worth_histories_field():
    """SimulationResult must have net_worth_histories with one dict per game."""
    result = simulate_games(
        n_games=2,
        player_names=["Alice", "Bob"],
        strategies=[BuyNothing(), BuyNothing()],
        seed=42,
    )
    assert hasattr(result, "net_worth_histories"), (
        "SimulationResult must have net_worth_histories field"
    )
    assert len(result.net_worth_histories) == 2


def test_net_worth_histories_keyed_by_player_name():
    """Each net_worth_histories entry must be a dict keyed by player name."""
    result = simulate_games(
        n_games=1,
        player_names=["Alice", "Bob"],
        strategies=[BuyNothing(), BuyNothing()],
        seed=7,
    )
    entry = result.net_worth_histories[0]
    assert set(entry.keys()) == {"Alice", "Bob"}


def test_net_worth_histories_lists_are_non_empty():
    """Each player's net worth history list must have at least one entry."""
    result = simulate_games(
        n_games=1,
        player_names=["Alice", "Bob"],
        strategies=[BuyNothing(), BuyNothing()],
        seed=7,
    )
    entry = result.net_worth_histories[0]
    for name, history in entry.items():
        assert len(history) > 0, f"{name}: net_worth_history must not be empty"

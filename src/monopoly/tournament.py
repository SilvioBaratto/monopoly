"""Round-robin tournament system for Monopoly strategy comparison.

Responsibilities (SRP):
- Orchestrate all 2-player head-to-head matchups between strategies
- Aggregate per-matchup simulation results into a TournamentResult
- Delegate game execution to simulate.run_parallel_simulations (Issue #35)

No game rule logic — delegates entirely to simulate.py.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

from monopoly.simulate import SimulationConfig, run_parallel_simulations
from monopoly.strategies.base import Strategy

# Separator used when serialising tuple keys to plain dict strings.
# Double-pipe is safe: Python class names never contain it.
_KEY_SEP = "||"


@dataclass
class TournamentResult:
    """Results from a complete round-robin tournament.

    Attributes:
        win_rates: Matchup matrix.  Keys are ``(name_a, name_b)`` tuples
            where ``name_a`` is the first strategy and ``name_b`` the second.
            The value is ``name_a``'s win rate in that head-to-head matchup.
        aggregate_stats: Per-strategy summary.  Keys are strategy class names.
            Each value is a dict with keys:
            ``overall_win_rate``, ``avg_turns``, ``avg_final_cash``.
        n_games_per_matchup: Number of games played in each head-to-head matchup.
            Required for statistically rigorous ranking (BT/Elo). Defaults to 0
            (meaning "unset"); ranking functions treat 0 as 1.
    """

    win_rates: dict[tuple[str, str], float] = field(default_factory=dict)
    aggregate_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    n_games_per_matchup: int = 0

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a plain dict (JSON-safe keys, primitive values)."""
        return {
            "win_rates": {
                f"{a}{_KEY_SEP}{b}": rate for (a, b), rate in self.win_rates.items()
            },
            "aggregate_stats": {
                name: dict(stats) for name, stats in self.aggregate_stats.items()
            },
            "n_games_per_matchup": self.n_games_per_matchup,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TournamentResult:
        """Restore a TournamentResult from a plain dict produced by ``to_dict``."""
        win_rates: dict[tuple[str, str], float] = {
            tuple(key.split(_KEY_SEP, 1)): float(rate)  # type: ignore[misc]
            for key, rate in data["win_rates"].items()
        }
        aggregate_stats: dict[str, dict[str, float]] = {
            name: dict(stats) for name, stats in data["aggregate_stats"].items()
        }
        return cls(
            win_rates=win_rates,
            aggregate_stats=aggregate_stats,
            n_games_per_matchup=int(data.get("n_games_per_matchup", 0)),
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_tournament(
    strategies: list[Strategy],
    n_games_per_matchup: int,
    seed: int | None = None,
) -> TournamentResult:
    """Run a round-robin tournament between every pair of strategies.

    Every pair plays ``n_games_per_matchup`` head-to-head games using the
    parallel simulation engine from Issue #35.  Results are fully deterministic
    when ``seed`` is provided.

    Args:
        strategies: Two or more strategy instances to pit against each other.
            Strategy names are derived from their class names (must be unique
            within the list).
        n_games_per_matchup: Number of 2-player games per matchup (≥ 1).
        seed: Master RNG seed for reproducibility.  ``None`` uses OS entropy.

    Returns:
        TournamentResult with the complete matchup win-rate matrix and
        per-strategy aggregate statistics.

    Raises:
        ValueError: If fewer than 2 strategies are provided.
        ValueError: If ``n_games_per_matchup`` < 1.
        RuntimeError: If any matchup simulation fails.
    """
    _validate_inputs(strategies, n_games_per_matchup)

    strategy_names = [type(s).__name__ for s in strategies]
    matchup_pairs = list(itertools.combinations(range(len(strategies)), 2))
    configs = _build_configs(matchup_pairs, strategy_names, n_games_per_matchup, seed)

    batch = run_parallel_simulations(configs)
    _assert_no_errors(batch)

    result = _build_result(matchup_pairs, strategy_names, batch.results)
    result.n_games_per_matchup = n_games_per_matchup
    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_inputs(strategies: list[Strategy], n_games_per_matchup: int) -> None:
    if len(strategies) < 2:
        raise ValueError(
            f"strategies must contain at least 2 strategies, got {len(strategies)}"
        )
    if n_games_per_matchup < 1:
        raise ValueError(f"n_games_per_matchup must be ≥ 1, got {n_games_per_matchup}")


def _build_configs(
    matchup_pairs: list[tuple[int, int]],
    strategy_names: list[str],
    n_games: int,
    seed: int | None,
) -> list[SimulationConfig]:
    configs = []
    for matchup_index, (i, j) in enumerate(matchup_pairs):
        matchup_seed = _derive_matchup_seed(seed, matchup_index)
        configs.append(
            SimulationConfig(
                n_games=n_games,
                player_names=[strategy_names[i], strategy_names[j]],
                strategy_names=[strategy_names[i], strategy_names[j]],
                seed=matchup_seed,
            )
        )
    return configs


def _derive_matchup_seed(master_seed: int | None, matchup_index: int) -> int | None:
    """Deterministic per-matchup seed derived from the master seed.

    Uses a different mixing constant than simulate._derive_sub_seed to ensure
    no collision between the two seed namespaces.

    Args:
        master_seed: Top-level seed from ``run_tournament``.
        matchup_index: Zero-based position of this matchup.

    Returns:
        An integer sub-seed, or None for non-reproducible runs.
    """
    if master_seed is None:
        return None
    return (master_seed * 2862933555777941757 + matchup_index) & 0xFFFF_FFFF_FFFF_FFFF


def _assert_no_errors(batch) -> None:
    if batch.errors:
        failed = ", ".join(f"config {i}: {msg}" for i, msg in batch.errors)
        raise RuntimeError(f"Tournament matchup(s) failed: {failed}")


def _build_result(
    matchup_pairs: list[tuple[int, int]],
    strategy_names: list[str],
    sim_results,
) -> TournamentResult:
    win_rates: dict[tuple[str, str], float] = {}
    wins: dict[str, int] = {name: 0 for name in strategy_names}
    total_games: dict[str, int] = {name: 0 for name in strategy_names}
    all_turns: dict[str, list[int]] = {name: [] for name in strategy_names}
    all_cash: dict[str, list[float]] = {name: [] for name in strategy_names}

    for matchup_index, (i, j) in enumerate(matchup_pairs):
        name_a, name_b = strategy_names[i], strategy_names[j]
        sim = sim_results[matchup_index]
        _accumulate_matchup(
            name_a, name_b, sim, win_rates, wins, total_games, all_turns, all_cash
        )

    aggregate_stats = _compute_aggregate_stats(
        strategy_names, wins, total_games, all_turns, all_cash
    )
    return TournamentResult(win_rates=win_rates, aggregate_stats=aggregate_stats)


def _accumulate_matchup(
    name_a: str,
    name_b: str,
    sim,
    win_rates: dict,
    wins: dict,
    total_games: dict,
    all_turns: dict,
    all_cash: dict,
) -> None:
    n_games = len(sim.winner_per_game)
    wins_a = sum(1 for w in sim.winner_per_game if w == name_a)
    wins_b = sum(1 for w in sim.winner_per_game if w == name_b)

    win_rates[(name_a, name_b)] = wins_a / n_games if n_games > 0 else 0.0

    wins[name_a] += wins_a
    wins[name_b] += wins_b
    total_games[name_a] += n_games
    total_games[name_b] += n_games

    all_turns[name_a].extend(sim.turns_per_game)
    all_turns[name_b].extend(sim.turns_per_game)

    for cash_dict in sim.final_cash:
        if name_a in cash_dict:
            all_cash[name_a].append(cash_dict[name_a])
        if name_b in cash_dict:
            all_cash[name_b].append(cash_dict[name_b])


def _compute_aggregate_stats(
    strategy_names: list[str],
    wins: dict[str, int],
    total_games: dict[str, int],
    all_turns: dict[str, list[int]],
    all_cash: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    aggregate_stats = {}
    for name in strategy_names:
        n = total_games[name]
        aggregate_stats[name] = {
            "overall_win_rate": wins[name] / n if n > 0 else 0.0,
            "avg_turns": sum(all_turns[name]) / len(all_turns[name])
            if all_turns[name]
            else 0.0,
            "avg_final_cash": sum(all_cash[name]) / len(all_cash[name])
            if all_cash[name]
            else 0.0,
        }
    return aggregate_stats

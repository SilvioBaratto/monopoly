"""Batch Monte Carlo simulation runner.

Responsibilities (SRP):
- Run N independent Monopoly games using the existing Game engine
- Collect per-game outcomes into a SimulationResult dataclass
- Manage deterministic sub-seeds for full reproducibility
- Provide simulate_landing_frequencies() for Markov chain cross-validation

No game rule logic — delegates entirely to Game.play() or the Monopoly
movement model encoded in the private landing-frequency helpers.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from monopoly.board import Board
from monopoly.game import Game
from monopoly.strategies.base import Strategy
from monopoly.strategies.buy_everything import BuyEverything
from monopoly.strategies.buy_nothing import BuyNothing
from monopoly.strategies.color_targeted import ColorTargeted
from monopoly.strategies.jail_camper import JailCamper
from monopoly.strategies.three_houses_rush import ThreeHousesRush
from monopoly.strategies.trader import Trader

_DATA_PATH = Path(__file__).parent.parent.parent / "data" / "cards_standard.yaml"
_MAX_TURNS = 1000

# ---------------------------------------------------------------------------
# Constants for the landing-frequency simulation (mirrors markov.py values)
# ---------------------------------------------------------------------------

_FREQ_BOARD_SIZE = 40
_FREQ_JAIL_POSITION = 10
_FREQ_GO_TO_JAIL = 30
_FREQ_CHANCE_POSITIONS = frozenset({7, 22, 36})
_FREQ_CC_POSITIONS = frozenset({2, 17, 33})
_FREQ_RAILROAD_POSITIONS = (5, 15, 25, 35)
_FREQ_UTILITY_POSITIONS = (12, 28)
_FREQ_JAIL_SENTINEL = -1  # sentinel: "send this player to jail"


# ---------------------------------------------------------------------------
# Public API — landing frequency simulation
# ---------------------------------------------------------------------------


def simulate_landing_frequencies(
    n_rolls: int,
    seed: int | None = None,
) -> np.ndarray:
    """Count normalised per-square landing frequencies over n_rolls dice rolls.

    Implements the **same one-roll Markov model** as
    ``markov.build_transition_matrix()``:

    * One Markov step = one dice roll.
    * Jail: ROLL_DOUBLES strategy — 3-turn cycle, then forced release.
    * Chance / Community Chest: same card fractions used in markov.py
      (6/16 non-movement for Chance, 14/16 for CC).
    * Triple doubles → jail via the same ``1/36`` correction the matrix uses
      (not explicit consecutive-doubles tracking).
    * Go To Jail square (pos 30) → always redirects to jail.

    Position 30 (Go To Jail) is **never** a resting state and always has
    frequency 0.  Position 10 accumulates both "Just Visiting" and all three
    "In Jail" sub-states, matching ``get_square_probabilities()`` semantics.

    Minimum N for ±0.1 pp tolerance (3σ):
      Most variable square is Jail (~9.6 % combined).
      SE = sqrt(0.096 × 0.904 / N) < 0.033 pp  ⟹  N > 250 000.
      Recommended for cross-validation: 5 000 000 rolls → SE < 0.005 pp.

    Args:
        n_rolls: Total dice rolls to simulate (each = one Markov chain step).
        seed: RNG seed for full reproducibility.  ``None`` uses OS entropy.

    Returns:
        ``numpy.ndarray`` of shape ``(40,)`` with normalised landing
        frequencies that sum to 1.  Entry at index 30 is always 0.
    """
    counts = _simulate_landing_counts(n_rolls, seed)
    return counts / counts.sum()


# ---------------------------------------------------------------------------
# Private helpers — landing-frequency simulation
# ---------------------------------------------------------------------------


def _simulate_landing_counts(
    n_rolls: int,
    seed: int | None,
) -> np.ndarray:
    """Return raw integer landing counts per square over n_rolls dice rolls.

    Implements the **same one-roll Markov model** as ``markov.py``:

    * Each loop iteration = one dice roll = one Markov step.
    * Triple-doubles are handled via the same uniform 1/36 correction the
      transition matrix uses: for any doubles roll, ``1/36`` probability of
      jail rather than the normal destination.  This matches:
        ``result[i, JAIL] += (1/36) * (1/36)`` for each doubles outcome
      in ``_apply_triple_doubles_correction``.
    * No explicit consecutive-doubles tracking — the 43-state Markov chain
      does not include consecutive-doubles as a state dimension.
    """
    rng = np.random.default_rng(seed)
    counts = np.zeros(_FREQ_BOARD_SIZE, dtype=np.int64)

    position: int = 0
    in_jail: bool = False
    jail_turns: int = 0  # failed escape attempts: 0, 1, 2 → forced release

    for _ in range(n_rolls):
        d1 = int(rng.integers(1, 7))
        d2 = int(rng.integers(1, 7))
        is_doubles = d1 == d2
        total = d1 + d2

        if in_jail:
            position, in_jail, jail_turns = _step_from_jail(
                position, total, is_doubles, jail_turns, rng
            )
        else:
            position, in_jail, jail_turns = _step_from_board(
                position, total, is_doubles, rng
            )

        counts[position] += 1

    return counts


def _step_from_jail(
    position: int,
    total: int,
    is_doubles: bool,
    jail_turns: int,
    rng: np.random.Generator,
) -> tuple[int, bool, int]:
    """Resolve one dice roll when the player is in jail.

    Args:
        position: Current position (always 10 while in jail).
        total: Sum of both dice.
        is_doubles: Whether both dice show the same face.
        jail_turns: Failed escape attempts so far (0–2).
        rng: Random number generator (for card draws on escape).

    Returns:
        (new_position, in_jail, new_jail_turns)
    """
    if is_doubles or jail_turns >= 2:
        # Escape jail: doubles roll or forced release (3rd attempt)
        raw = (_FREQ_JAIL_POSITION + total) % _FREQ_BOARD_SIZE
        dest = _apply_landing(raw, rng)
        if dest == _FREQ_JAIL_SENTINEL:
            return _FREQ_JAIL_POSITION, True, 0
        return dest, False, 0

    # Stay in jail — advance jail sub-state counter
    return _FREQ_JAIL_POSITION, True, jail_turns + 1


def _step_from_board(
    position: int,
    total: int,
    is_doubles: bool,
    rng: np.random.Generator,
) -> tuple[int, bool, int]:
    """Resolve one dice roll when the player is not in jail.

    Applies the **same triple-doubles correction** as
    ``markov._apply_triple_doubles_correction``: for each doubles roll,
    ``1/36`` probability of redirecting to jail instead of the normal
    destination.  This keeps the simulation faithful to the 43-state Markov
    chain that does **not** track consecutive doubles as a state.

    Args:
        position: Current board position (0–39).
        total: Sum of both dice.
        is_doubles: Whether both dice show the same face.
        rng: Random number generator (for card draws and triple-doubles check).

    Returns:
        (new_position, in_jail, jail_turns)
    """
    if is_doubles and int(rng.integers(0, 36)) == 0:
        # Triple-doubles correction: 1/36 probability of jail on any doubles
        return _FREQ_JAIL_POSITION, True, 0

    raw = (position + total) % _FREQ_BOARD_SIZE
    dest = _apply_landing(raw, rng)
    if dest == _FREQ_JAIL_SENTINEL:
        return _FREQ_JAIL_POSITION, True, 0

    return dest, False, 0


def _apply_landing(position: int, rng: np.random.Generator) -> int:
    """Apply the board-square effect when landing on ``position``.

    Returns the final resting position, or ``_FREQ_JAIL_SENTINEL`` when the
    player must go to jail.
    """
    if position == _FREQ_GO_TO_JAIL:
        return _FREQ_JAIL_SENTINEL
    if position in _FREQ_CHANCE_POSITIONS:
        return _draw_chance_card(position, rng)
    if position in _FREQ_CC_POSITIONS:
        return _draw_cc_card(position, rng)
    return position


def _draw_chance_card(position: int, rng: np.random.Generator) -> int:
    """Sample one Chance card and return the destination position.

    Card distribution (16 cards total, matching markov.py constants):
      - 6 non-movement cards → stay at ``position``
      - 1 → Boardwalk (39)
      - 1 → Advance to Go (0)
      - 1 → Illinois Ave (24)
      - 1 → St. Charles Place (11)
      - 1 → Reading Railroad (5)
      - 2 → Advance to nearest Railroad
      - 1 → Advance to nearest Utility
      - 1 → Go Back 3 Spaces
      - 1 → Go to Jail

    Returns ``_FREQ_JAIL_SENTINEL`` for the Go to Jail card.
    Back-3-spaces landing on Community Chest (pos 33 from pos 36) applies
    that CC card immediately, matching ``markov._resolve_landing`` semantics.
    """
    card = int(rng.integers(0, 16))
    if card < 6:  # 0–5: non-movement
        return position
    card -= 6  # 0–9: movement cards
    if card == 0:
        return 39  # Boardwalk
    if card == 1:
        return 0  # Advance to Go
    if card == 2:
        return 24  # Illinois Ave
    if card == 3:
        return 11  # St. Charles Place
    if card == 4:
        return 5  # Reading Railroad
    if card in (5, 6):  # two "Advance to nearest RR" cards
        return _nearest_forward(position, _FREQ_RAILROAD_POSITIONS)
    if card == 7:  # Advance to nearest Utility
        return _nearest_forward(position, _FREQ_UTILITY_POSITIONS)
    if card == 8:  # Go Back 3 Spaces
        back3 = (position - 3) % _FREQ_BOARD_SIZE
        if back3 in _FREQ_CC_POSITIONS:
            return _draw_cc_card(back3, rng)
        return back3
    return _FREQ_JAIL_SENTINEL  # card == 9: Go to Jail


def _draw_cc_card(position: int, rng: np.random.Generator) -> int:
    """Sample one Community Chest card and return the destination position.

    Card distribution (16 cards total, matching markov.py constants):
      - 14 non-movement cards → stay at ``position``
      - 1 → Advance to Go (0)
      - 1 → Go to Jail

    Returns ``_FREQ_JAIL_SENTINEL`` for the Go to Jail card.
    """
    card = int(rng.integers(0, 16))
    if card < 14:  # 0–13: non-movement
        return position
    if card == 14:  # Advance to Go
        return 0
    return _FREQ_JAIL_SENTINEL  # card == 15: Go to Jail


def _nearest_forward(current: int, positions: tuple[int, ...]) -> int:
    """Return the nearest position clockwise (forward) from ``current``.

    Mirrors ``markov._nearest()`` exactly.

    Args:
        current: Current board position.
        positions: Sorted or unsorted tuple of candidate positions.

    Returns:
        The smallest position in ``positions`` that is strictly greater than
        ``current``, or the smallest position overall if none qualifies
        (wrap-around the board).
    """
    for pos in sorted(positions):
        if pos > current:
            return pos
    return sorted(positions)[0]


@dataclass
class SimulationResult:
    """Outcomes collected across a batch of simulated Monopoly games.

    Args:
        winner_per_game: Name of the winning player per game, or None
            when the game ended by max_turns with a tie.
        turns_per_game: Number of full rounds played per game.
        bankruptcy_order: For each game, the ordered list of player names
            that went bankrupt (earliest first). Survivors are excluded.
        final_cash: For each game, a mapping of player name → cash held
            at game end.
    """

    winner_per_game: list[str | None] = field(default_factory=list)
    turns_per_game: list[int] = field(default_factory=list)
    bankruptcy_order: list[list[str]] = field(default_factory=list)
    final_cash: list[dict[str, int]] = field(default_factory=list)


def simulate_games(
    n_games: int,
    player_names: list[str],
    strategies: list[Strategy],
    seed: int | None = None,
) -> SimulationResult:
    """Run ``n_games`` independent Monopoly games and return aggregated results.

    Each game uses a deterministic sub-seed derived from ``seed`` so that
    calling this function twice with the same seed produces identical output.

    Args:
        n_games: Number of games to simulate. Must be ≥ 1.
        player_names: Display names for each player (2–6 required).
        strategies: One Strategy per player, same order as ``player_names``.
        seed: Master RNG seed for reproducibility. ``None`` uses a random seed.

    Returns:
        SimulationResult with per-game winner, turns, bankruptcy order, and
        final cash for every player.

    Raises:
        ValueError: If ``n_games`` < 1.
        ValueError: If ``len(strategies)`` != ``len(player_names)``.
    """
    if n_games < 1:
        raise ValueError(f"n_games must be ≥ 1, got {n_games}")
    if len(strategies) != len(player_names):
        raise ValueError(
            f"strategies length ({len(strategies)}) must match "
            f"player_names length ({len(player_names)})"
        )

    board = Board()
    result = SimulationResult()

    for game_index in range(n_games):
        sub_seed = _derive_sub_seed(seed, game_index)
        rng = np.random.default_rng(sub_seed)

        game = Game(
            player_names=player_names,
            strategies=strategies,
            board=board,
            data_path=_DATA_PATH,
            rng=rng,
        )
        game_result = game.play(max_turns=_MAX_TURNS)

        result.winner_per_game.append(
            game_result.winner.name if game_result.winner is not None else None
        )
        result.turns_per_game.append(int(game_result.turns_played))
        result.bankruptcy_order.append(_bankruptcy_order(game_result, player_names))
        result.final_cash.append(
            {
                name: int(game_result.player_stats[name].final_cash)
                for name in player_names
            }
        )

    return result


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _derive_sub_seed(master_seed: int | None, game_index: int) -> int | None:
    """Produce a deterministic sub-seed for one game in the batch.

    When ``master_seed`` is None the result is None, which instructs NumPy to
    seed from OS entropy (non-reproducible).

    Args:
        master_seed: The top-level seed passed to ``simulate_games``.
        game_index: Zero-based position of this game within the batch.

    Returns:
        An integer sub-seed, or None for non-reproducible runs.
    """
    if master_seed is None:
        return None
    # Mix master seed and game index to avoid correlated streams while
    # remaining fully deterministic.
    return (master_seed * 6364136223846793005 + game_index) & 0xFFFF_FFFF_FFFF_FFFF


def _bankruptcy_order(game_result, player_names: list[str]) -> list[str]:
    """Return players sorted by bankruptcy turn (earliest first).

    Only includes players who actually went bankrupt; survivors are excluded.

    Args:
        game_result: The GameResult from a completed game.
        player_names: All player names in the game.

    Returns:
        Ordered list of bankrupt player names.
    """
    bankrupt_entries: list[tuple[int, str]] = []
    for name in player_names:
        stats = game_result.player_stats.get(name)
        if stats is not None and stats.bankruptcy_turn is not None:
            bankrupt_entries.append((stats.bankruptcy_turn, name))

    bankrupt_entries.sort(key=lambda x: x[0])
    return [name for _, name in bankrupt_entries]


# ---------------------------------------------------------------------------
# Strategy registry (for safe pickling in multiprocessing workers)
# ---------------------------------------------------------------------------

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "BuyEverything": BuyEverything,
    "BuyNothing": BuyNothing,
    "ColorTargeted": ColorTargeted,
    "JailCamper": JailCamper,
    "ThreeHousesRush": ThreeHousesRush,
    "Trader": Trader,
}


@dataclass
class SimulationConfig:
    """Configuration for a single simulation run.

    Uses strategy names (not instances) for safe pickling across processes.

    Args:
        n_games: Number of games to simulate.
        player_names: Display names for each player.
        strategy_names: Strategy class names (keys in STRATEGY_REGISTRY), one per player.
        seed: Master RNG seed for reproducibility. None uses OS entropy.
    """

    n_games: int
    player_names: list[str]
    strategy_names: list[str]
    seed: int | None = None


@dataclass
class BatchResult:
    """Aggregated results from a parallel simulation sweep.

    Args:
        results: Successful SimulationResult objects (one per successful config).
        configs: The original list of SimulationConfig objects submitted.
        wall_clock_seconds: Elapsed wall-clock time for the entire batch.
        n_workers: Number of worker processes used.
        errors: List of (config_index, error_message) tuples for failed configs.
    """

    results: list[SimulationResult] = field(default_factory=list)
    configs: list[SimulationConfig] = field(default_factory=list)
    wall_clock_seconds: float = 0.0
    n_workers: int = 0
    errors: list[tuple[int, str]] = field(default_factory=list)


def _run_config_worker(
    args: tuple[int, SimulationConfig],
) -> tuple[int, SimulationResult]:
    """Module-level worker: runs one SimulationConfig, returns (index, result).

    Must be module-level for multiprocessing pickling compatibility.

    Args:
        args: Tuple of (config_index, config) to run.

    Returns:
        Tuple of (config_index, SimulationResult).

    Raises:
        KeyError: If a strategy name is not found in STRATEGY_REGISTRY.
    """
    config_index, config = args
    strategies = [STRATEGY_REGISTRY[name]() for name in config.strategy_names]
    result = simulate_games(
        n_games=config.n_games,
        player_names=config.player_names,
        strategies=strategies,
        seed=config.seed,
    )
    return config_index, result


def run_parallel_simulations(
    configs: list[SimulationConfig],
    n_workers: int | None = None,
) -> BatchResult:
    """Run multiple SimulationConfigs concurrently using ProcessPoolExecutor.

    Args:
        configs: List of simulation configurations to run.
        n_workers: Number of worker processes. Defaults to
            ``min(len(configs), max(1, (os.cpu_count() or 2) - 1))``.

    Returns:
        BatchResult with all successful results, error records for
        failures, wall-clock time, and worker count.
    """
    if not configs:
        return BatchResult()

    effective_workers = (
        n_workers
        if n_workers is not None
        else min(len(configs), max(1, (os.cpu_count() or 2) - 1))
    )

    result_slots: list[SimulationResult | None] = [None] * len(configs)
    errors: list[tuple[int, str]] = []

    # Propagate a stable PYTHONHASHSEED to worker processes.
    # Python randomises the hash seed per-process by default (PEP 456);
    # spawned workers would otherwise start with a *different* seed, making
    # string-hash-dependent dict/set ordering diverge from the parent.
    # Setting a fixed value before the executor is created ensures all
    # workers inherit the same deterministic seed via os.environ.
    _original_hashseed = os.environ.get("PYTHONHASHSEED")
    if _original_hashseed is None:
        os.environ["PYTHONHASHSEED"] = "0"

    start = time.monotonic()

    with ProcessPoolExecutor(max_workers=effective_workers) as executor:
        futures = {
            executor.submit(_run_config_worker, (i, config)): i
            for i, config in enumerate(configs)
        }
        for future in as_completed(futures):
            config_index = futures[future]
            try:
                _, sim_result = future.result()
                result_slots[config_index] = sim_result
            except Exception as exc:
                logging.warning("Config %d failed: %s", config_index, exc)
                errors.append((config_index, str(exc)))

    wall_clock = time.monotonic() - start

    # Restore PYTHONHASHSEED to its original state after workers are done.
    if _original_hashseed is None:
        del os.environ["PYTHONHASHSEED"]

    successful_results = [r for r in result_slots if r is not None]

    return BatchResult(
        results=successful_results,
        configs=configs,
        wall_clock_seconds=wall_clock,
        n_workers=effective_workers,
        errors=errors,
    )

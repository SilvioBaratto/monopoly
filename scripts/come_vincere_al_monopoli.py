"""Script narrativo: Come vincere al Monopoly — analisi matematica completa.

Ripercorre il video passo dopo passo e rigenera tutte le figure con:
    python scripts/come_vincere_al_monopoli.py

Struttura:
  section_01_intro()       — introduzione e tabellone
  section_02_dice()        — analisi probabilità dadi
  section_03_markov()      — catena di Markov e heatmap
  section_04_roi()         — ROI per gruppo colore
  section_05_monte_carlo() — simulazione Monte Carlo
  section_06_sample_game() — partita campione
  section_07_tournament()  — torneo tra strategie
  section_08_conclusion()  — conclusioni

Configura la dimensione della simulazione:
    MONOPOLY_N_GAMES=500 python scripts/come_vincere_al_monopoli.py
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy

# ---------------------------------------------------------------------------
# Reproducibility seeds — set once, before any library import
# ---------------------------------------------------------------------------
random.seed(42)
numpy.random.seed(42)

# ---------------------------------------------------------------------------
# Project root on sys.path (allows running from any working directory)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).parent.parent
_SCRIPTS_DIR = Path(__file__).parent
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# ---------------------------------------------------------------------------
# Monopoly library imports
# ---------------------------------------------------------------------------
from monopoly.board import Board  # noqa: E402
from monopoly.dice import DISTRIBUTION as DICE_DISTRIBUTION  # noqa: E402
from monopoly.game import Game, GameHistory  # noqa: E402
from monopoly.markov import build_transition_matrix, compute_stationary_distribution  # noqa: E402
from monopoly.metrics import roi_ranking_table, win_probability_table  # noqa: E402
from monopoly.plots import (  # noqa: E402
    animate_sample_game,
    plot_board_heatmap,
    plot_net_worth,
    plot_roi_bars,
    plot_win_rate_curves,
)
from monopoly.ranking import bradley_terry_ranking  # noqa: E402
from monopoly.simulate import SimulationResult, simulate_games  # noqa: E402
from monopoly.strategies import (  # noqa: E402
    BuyEverything,
    BuyNothing,
    JailCamper,
    ThreeHousesRush,
)
from monopoly.tournament import run_tournament  # noqa: E402

from _narrative_text import TEXTS  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_GAMES: int = int(os.environ.get("MONOPOLY_N_GAMES", "10000"))
FIGURES_DIR: Path = _PROJECT_ROOT / "figures"

_DATA_PATH = _PROJECT_ROOT / "data" / "cards_standard.yaml"

_STRATEGIES = [
    BuyEverything(),
    ThreeHousesRush(),
    JailCamper(),
    BuyNothing(),
]
_PLAYER_COUNTS = [2, 3, 4]
_TOURNAMENT_N_GAMES = max(N_GAMES // 100, 10)
_MC_N_GAMES = max(N_GAMES // 10, 20)


# ---------------------------------------------------------------------------
# Section functions — each delegates computation to library modules
# ---------------------------------------------------------------------------


def section_01_intro() -> None:
    """Print Italian introduction and board overview."""
    print(TEXTS["header_01"])
    print(TEXTS["intro"])


def section_02_dice() -> None:
    """Print the 2d6 probability distribution in Italian."""
    print(TEXTS["header_02"])
    print(TEXTS["dice_intro"])
    _print_dice_table()


def section_03_markov() -> None:
    """Compute stationary distribution and save board heatmap."""
    print(TEXTS["header_03"])
    print(TEXTS["markov_intro"])
    board = Board()
    matrix = build_transition_matrix()
    dist43 = compute_stationary_distribution(matrix)
    distribution = _collapse_to_40(dist43)
    plot_board_heatmap(distribution, board, FIGURES_DIR / "heatmap.png")
    print(TEXTS["markov_done"])


def section_04_roi() -> None:
    """Compute ROI table and save bar chart."""
    print(TEXTS["header_04"])
    print(TEXTS["roi_intro"])
    board = Board()
    matrix = build_transition_matrix()
    dist43 = compute_stationary_distribution(matrix)
    distribution = _collapse_to_40(dist43)
    roi_table = roi_ranking_table(board, distribution)
    plot_roi_bars(roi_table, FIGURES_DIR / "roi_bars.png")
    print(TEXTS["roi_done"])


def section_05_monte_carlo() -> None:
    """Run multi-strategy Monte Carlo simulations and save win rate curves."""
    print(TEXTS["header_05"])
    print(TEXTS["monte_carlo_intro"])
    sim_results = _build_win_prob_results()
    strategy_names = [type(s).__name__ for s in _STRATEGIES]
    df = win_probability_table(sim_results, strategy_names, _PLAYER_COUNTS)
    plot_win_rate_curves(df, FIGURES_DIR / "win_rate_curves.png")
    print(TEXTS["monte_carlo_done"])


def section_06_sample_game() -> None:
    """Run a sample game and save net worth chart and animation."""
    print(TEXTS["header_06"])
    print(TEXTS["sample_game_intro"])
    history = _run_sample_game()
    nw_by_player = _extract_per_player_net_worth(history)
    plot_net_worth(nw_by_player, history.player_names, FIGURES_DIR / "net_worth.png")
    animate_sample_game(history, FIGURES_DIR / "game_animation.mp4")
    print(TEXTS["sample_game_done"])


def section_07_tournament() -> None:
    """Run round-robin tournament and print Bradley-Terry ranking."""
    print(TEXTS["header_07"])
    print(TEXTS["tournament_intro"])
    result = run_tournament(_STRATEGIES, _TOURNAMENT_N_GAMES, seed=42)
    ranking = bradley_terry_ranking(result)
    print(ranking.to_string(index=False))
    print(TEXTS["tournament_done"])


def section_08_conclusion() -> None:
    """Print Italian summary of winning strategies."""
    print(TEXTS["header_08"])
    print(TEXTS["conclusion"])


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _collapse_to_40(dist43: numpy.ndarray) -> numpy.ndarray:
    """Collapse the 43-state Markov distribution to 40 board positions.

    Jail sub-states (40, 41, 42) are folded into position 10 (physical jail square).
    """
    dist40 = dist43[:40].copy()
    dist40[10] += dist43[40] + dist43[41] + dist43[42]
    return dist40


def _print_dice_table() -> None:
    """Print 2d6 totals and probabilities as a console table."""
    print(f"  {'Totale':>7}  {'Probabilità':>12}  {'Barre':}")
    print(f"  {'-' * 7}  {'-' * 12}  {'-' * 20}")
    for total in range(2, 13):
        prob = DICE_DISTRIBUTION[total]
        bar = "█" * int(prob * 100)
        print(f"  {total:>7}  {prob:>11.1%}  {bar}")


def _build_win_prob_results() -> dict[tuple[str, int], SimulationResult]:
    """Run simulations for each (strategy, n_players) pair."""
    results: dict[tuple[str, int], SimulationResult] = {}
    for strat in _STRATEGIES:
        strat_name = type(strat).__name__
        for n in _PLAYER_COUNTS:
            results[(strat_name, n)] = _simulate_one_strategy(strat, strat_name, n)
    return results


def _simulate_one_strategy(
    strat: object,
    strat_name: str,
    n_players: int,
) -> SimulationResult:
    """Run MC simulation for one strategy vs BuyEverything opponents."""
    opponents = [BuyEverything() for _ in range(n_players - 1)]
    opp_names = [f"Avversario{i + 1}" for i in range(n_players - 1)]
    return simulate_games(
        _MC_N_GAMES,
        [strat_name] + opp_names,
        [strat] + opponents,  # type: ignore[list-item]
        seed=42,
    )


def _run_sample_game() -> GameHistory:
    """Run a 4-player sample game and return per-turn history."""
    board = Board()
    rng = numpy.random.default_rng(42)
    player_names = ["Alice", "Bob", "Carla", "Diego"]
    strategies = [BuyEverything(), ThreeHousesRush(), JailCamper(), BuyNothing()]
    game = Game(player_names, strategies, board, _DATA_PATH, rng)
    return game.play_with_history()


def _extract_per_player_net_worth(
    history: GameHistory,
) -> dict[str, list[int]]:
    """Convert per-turn net worth snapshots to per-player time series."""
    return {
        name: [turn.get(name, 0) for turn in history.net_worth_history]
        for name in history.player_names
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run all 8 sections in order, saving figures to FIGURES_DIR."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    section_01_intro()
    section_02_dice()
    section_03_markov()
    section_04_roi()
    section_05_monte_carlo()
    section_06_sample_game()
    section_07_tournament()
    section_08_conclusion()


if __name__ == "__main__":
    main()

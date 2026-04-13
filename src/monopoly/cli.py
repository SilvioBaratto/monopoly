"""Typer CLI for the Monopoly simulation toolkit.

Exposes five commands:
  monopoly simulate     — run Monte Carlo simulations
  monopoly markov       — compute Markov stationary distribution
  monopoly tournament   — run round-robin strategy tournament
  monopoly plot         — generate static PNG figures
  monopoly export-video — generate sample game animation

Design (SRP):
  - Command functions are thin orchestrators (parse args → call API → print).
  - Strategy resolution, player-name generation, and table building are each
    delegated to focused private helpers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from monopoly.board import Board
from monopoly.markov import (
    build_transition_matrix,
    compute_stationary_distribution,
    get_square_probabilities,
)
from monopoly.ranking import elo_ranking
from monopoly.simulate import STRATEGY_REGISTRY, simulate_games
from monopoly.strategies.base import Strategy
from monopoly.tournament import run_tournament

# Default constructor kwargs for strategies that require arguments.
# ColorTargeted needs a target_colors list; "orange" is a robust default.
_STRATEGY_KWARGS: dict[str, dict] = {
    "ColorTargeted": {"target_colors": ["orange"]},
}

# Strategies that can be instantiated with no arguments — safe for tournament.
# Strategies needing constructor args (e.g. ColorTargeted) are excluded because
# run_parallel_simulations re-instantiates them via STRATEGY_REGISTRY[name]().
_TOURNAMENT_STRATEGIES = [n for n in STRATEGY_REGISTRY if n not in _STRATEGY_KWARGS]

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = typer.Typer(
    name="monopoly",
    rich_markup_mode="rich",
    help="[bold]Monopoly[/bold] simulation toolkit.",
)

_console = Console()

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

_DEFAULT_STRATEGY = "BuyEverything"
_DEFAULT_N_GAMES = 1000
_DEFAULT_PLAYERS = 4
_DEFAULT_MARKOV_TOP = 10
_DEFAULT_TOURNAMENT_GAMES = 100


@app.command()
def simulate(
    n_games: int = typer.Option(
        _DEFAULT_N_GAMES, "--n-games", help="Games to simulate."
    ),
    players: int = typer.Option(
        _DEFAULT_PLAYERS, "--players", help="Number of players (2–6)."
    ),
    strategy: Optional[list[str]] = typer.Option(
        None, "--strategy", help="Strategy name (repeatable; round-robin assigned)."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="RNG seed for reproducibility."
    ),
) -> None:
    """Run Monte Carlo simulations and display summary statistics."""
    strategy_names = _resolve_strategy_names(strategy, players)
    _validate_strategies(strategy_names)
    player_names = _build_player_names(players)
    strategies = [_instantiate_strategy(name) for name in strategy_names]
    result = simulate_games(n_games, player_names, strategies, seed=seed)
    _print_simulate_table(result, player_names, strategy_names)


@app.command()
def markov(
    top: int = typer.Option(
        _DEFAULT_MARKOV_TOP, "--top", help="Number of squares to show."
    ),
) -> None:
    """Compute the Markov stationary distribution and display top squares."""
    matrix = build_transition_matrix()
    distribution = compute_stationary_distribution(matrix)
    board = Board()
    square_probs = get_square_probabilities(board, distribution)
    _print_markov_table(square_probs, top)


@app.command()
def tournament(
    n_games: int = typer.Option(
        _DEFAULT_TOURNAMENT_GAMES, "--n-games", help="Games per matchup."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="RNG seed for reproducibility."
    ),
) -> None:
    """Run a round-robin tournament across all registered strategies."""
    strategies = [STRATEGY_REGISTRY[name]() for name in _TOURNAMENT_STRATEGIES]
    result = run_tournament(strategies, n_games_per_matchup=n_games, seed=seed)
    ranking = elo_ranking(result)
    _print_tournament_table(ranking)


# ---------------------------------------------------------------------------
# Plot command constants
# ---------------------------------------------------------------------------

_VALID_FIGURES = ("heatmap", "roi", "win-rate", "net-worth")
_DEFAULT_FIGURE_OUTPUT = Path("figures")
_DEFAULT_N_GAMES_PLOT = 1000
_DEFAULT_FPS = 5
_DEFAULT_EXPORT_OUTPUT = Path("figures/game.gif")

# ---------------------------------------------------------------------------
# Commands — plot and export-video
# ---------------------------------------------------------------------------


@app.command()
def plot(
    figure: str = typer.Argument(
        ..., help=f"Figure to generate: {', '.join(_VALID_FIGURES)}"
    ),
    output: Path = typer.Option(
        _DEFAULT_FIGURE_OUTPUT, "--output", help="Destination PNG path."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="RNG seed for reproducibility."
    ),
    n_games: int = typer.Option(
        _DEFAULT_N_GAMES_PLOT, "--n-games", help="Games for simulation-based figures."
    ),
) -> None:
    """Generate a static PNG figure (heatmap, roi, win-rate, net-worth)."""
    _validate_figure_name(figure)
    output.parent.mkdir(parents=True, exist_ok=True)
    _dispatch_plot(figure, output, seed, n_games)
    _console.print(f"[green]Saved[/green] {output}")


@app.command(name="export-video")
def export_video(
    output: Path = typer.Option(
        _DEFAULT_EXPORT_OUTPUT, "--output", help="Destination MP4 or GIF path."
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", help="RNG seed for reproducibility."
    ),
    fps: int = typer.Option(_DEFAULT_FPS, "--fps", help="Frames per second."),
) -> None:
    """Generate a sample game animation (MP4 or GIF)."""
    output.parent.mkdir(parents=True, exist_ok=True)
    history = _run_sample_game(seed)
    from monopoly.plots import animate_sample_game

    animate_sample_game(history, output, fps=fps)
    _console.print(f"[green]Saved[/green] {output}")


# ---------------------------------------------------------------------------
# Private helpers — plot dispatch
# ---------------------------------------------------------------------------


def _validate_figure_name(figure: str) -> None:
    """Raise a user-friendly error if figure is not in _VALID_FIGURES."""
    if figure not in _VALID_FIGURES:
        valid = ", ".join(_VALID_FIGURES)
        typer.echo(
            f"Error: '{figure}' is not a valid figure. Valid figures are: {valid}",
            err=True,
        )
        raise typer.Exit(code=1)


def _dispatch_plot(
    figure: str, output: Path, seed: Optional[int], n_games: int
) -> None:
    """Route to the correct plot function based on figure name."""
    handlers = {
        "heatmap": _plot_heatmap,
        "roi": _plot_roi,
        "win-rate": _plot_win_rate,
        "net-worth": _plot_net_worth,
    }
    handlers[figure](output, seed, n_games)


def _plot_heatmap(output: Path, seed: Optional[int], n_games: int) -> None:
    """Compute Markov stationary distribution and save board heatmap."""
    from monopoly.plots import plot_board_heatmap

    matrix = build_transition_matrix()
    distribution = compute_stationary_distribution(matrix)
    board = Board()
    dist_40 = _collapse_to_40_states(distribution, board)
    plot_board_heatmap(dist_40, board, output)


def _plot_roi(output: Path, seed: Optional[int], n_games: int) -> None:
    """Compute ROI table and save ROI bar chart."""
    from monopoly.metrics import roi_ranking_table
    from monopoly.plots import plot_roi_bars

    matrix = build_transition_matrix()
    distribution = compute_stationary_distribution(matrix)
    board = Board()
    dist_40 = _collapse_to_40_states(distribution, board)
    roi_table = roi_ranking_table(board, dist_40)
    plot_roi_bars(roi_table, output)


def _collapse_to_40_states(distribution: np.ndarray, board: Board) -> np.ndarray:
    """Collapse 43-state distribution to 40-element array for the heatmap.

    Jail states 40, 41, 42 are summed into position 10.
    """
    result = np.zeros(40)
    for sq in board.squares:
        pos = sq.position
        if pos == 10:
            result[pos] = (
                distribution[10]
                + distribution[40]
                + distribution[41]
                + distribution[42]
            )
        else:
            result[pos] = float(distribution[pos])
    return result


def _plot_win_rate(output: Path, seed: Optional[int], n_games: int) -> None:
    """Run simulations for multiple player counts and save win rate curves."""
    from monopoly.metrics import win_probability_table
    from monopoly.plots import plot_win_rate_curves

    strategies = _TOURNAMENT_STRATEGIES
    player_counts = [2, 3, 4, 5, 6]
    sim_results = _simulate_win_rate_data(strategies, player_counts, n_games, seed)
    win_prob_df = win_probability_table(sim_results, strategies, player_counts)
    plot_win_rate_curves(win_prob_df, output)


def _plot_net_worth(output: Path, seed: Optional[int], n_games: int) -> None:
    """Run a single game and save net worth over time chart."""
    from monopoly.plots import plot_net_worth

    history = _run_sample_game(seed)
    net_worth_history = {
        name: [turn[name] for turn in history.net_worth_history]
        for name in history.player_names
    }
    plot_net_worth(net_worth_history, history.player_names, output)


def _simulate_win_rate_data(
    strategies: list[str],
    player_counts: list[int],
    n_games: int,
    seed: Optional[int],
) -> dict:
    """Run simulations for each (strategy, player_count) pair."""
    results = {}
    for n_players in player_counts:
        player_names = [strategies[i % len(strategies)] for i in range(n_players)]
        strategy_instances = [_instantiate_strategy(name) for name in player_names]
        sim = simulate_games(n_games, player_names, strategy_instances, seed=seed)
        for strategy in strategies:
            results[(strategy, n_players)] = sim
    return results


def _run_sample_game(seed: Optional[int]):
    """Run a single game with the given seed and return GameHistory."""
    from monopoly.game import Game
    from monopoly.simulate import _DATA_PATH

    board = Board()
    rng = np.random.default_rng(seed)
    strategy = _instantiate_strategy("BuyEverything")
    game = Game(
        player_names=["Player_1", "Player_2", "Player_3", "Player_4"],
        strategies=[strategy, strategy, strategy, strategy],
        board=board,
        data_path=_DATA_PATH,
        rng=rng,
    )
    return game.play_with_history()


# ---------------------------------------------------------------------------
# Private helpers — strategy resolution
# ---------------------------------------------------------------------------


def _resolve_strategy_names(
    requested: Optional[list[str]],
    n_players: int,
) -> list[str]:
    """Return one strategy name per player using round-robin from the request.

    When no strategies are requested, defaults to BuyEverything for all.
    """
    names = requested if requested else [_DEFAULT_STRATEGY]
    return [names[i % len(names)] for i in range(n_players)]


def _validate_strategies(names: list[str]) -> None:
    """Raise a user-friendly error if any strategy name is not in the registry."""
    for name in names:
        if name not in STRATEGY_REGISTRY:
            valid = ", ".join(sorted(STRATEGY_REGISTRY.keys()))
            typer.echo(
                f"Error: '{name}' is not a valid strategy. "
                f"Valid strategies are: {valid}",
                err=True,
            )
            raise typer.Exit(code=1)


def _instantiate_strategy(name: str) -> Strategy:
    """Instantiate a strategy by registry name, applying default kwargs where needed."""
    kwargs = _STRATEGY_KWARGS.get(name, {})
    return STRATEGY_REGISTRY[name](**kwargs)


def _build_player_names(n_players: int) -> list[str]:
    """Generate auto-numbered player names: Player_1, Player_2, …"""
    return [f"Player_{i + 1}" for i in range(n_players)]


# ---------------------------------------------------------------------------
# Private helpers — output tables
# ---------------------------------------------------------------------------


def _print_simulate_table(
    result, player_names: list[str], strategy_names: list[str]
) -> None:
    """Print a Rich table summarising simulation results per player."""
    n_games = len(result.winner_per_game)
    table = _build_simulate_table(result, player_names, strategy_names, n_games)
    _console.print(f"\n[bold]Simulation Summary[/bold] ({n_games} games)\n")
    _console.print(table)


def _build_simulate_table(result, player_names, strategy_names, n_games) -> Table:
    """Build a Rich Table with win rate, avg turns, avg cash per player."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Player")
    table.add_column("Strategy")
    table.add_column("Win Rate", justify="right")
    table.add_column("Avg Turns", justify="right")
    table.add_column("Avg Cash", justify="right")

    for i, name in enumerate(player_names):
        wins = sum(1 for w in result.winner_per_game if w == name)
        win_rate = wins / n_games if n_games > 0 else 0.0
        avg_cash = _avg_cash_for_player(result.final_cash, name)
        table.add_row(
            name,
            strategy_names[i],
            f"{win_rate:.1%}",
            f"{sum(result.turns_per_game) / max(n_games, 1):.1f}",
            f"${avg_cash:,.0f}",
        )
    return table


def _avg_cash_for_player(final_cash_list: list[dict], name: str) -> float:
    """Compute average final cash across all games for one player."""
    values = [game[name] for game in final_cash_list if name in game]
    return sum(values) / len(values) if values else 0.0


def _print_markov_table(
    square_probs: dict[int, tuple[str, float]],
    top: int,
) -> None:
    """Print a Rich table of the top-N most-visited Monopoly squares."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Rank", justify="right")
    table.add_column("Position", justify="right")
    table.add_column("Square")
    table.add_column("Probability", justify="right")

    for rank, (pos, (name, prob)) in enumerate(
        list(square_probs.items())[:top], start=1
    ):
        table.add_row(str(rank), str(pos), name, f"{prob:.2%}")

    _console.print("\n[bold]Markov Stationary Distribution[/bold] (top squares)\n")
    _console.print(table)


def _print_tournament_table(ranking) -> None:
    """Print a Rich table of tournament Elo ranking."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Rank", justify="right")
    table.add_column("Strategy")
    table.add_column("Elo", justify="right")
    table.add_column("Games", justify="right")
    table.add_column("95% CI", justify="right")

    for rank, row in enumerate(ranking.itertuples(index=False), start=1):
        ci = f"[{row.ci_lower:.0f}, {row.ci_upper:.0f}]"
        table.add_row(str(rank), row.strategy, f"{row.elo:.1f}", str(row.n_games), ci)

    _console.print("\n[bold]Tournament Ranking[/bold] (Elo)\n")
    _console.print(table)

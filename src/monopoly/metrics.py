"""Investment metrics and win probability analysis for Monopoly simulation.

Responsibilities (SRP):
- Compute expected rent per roll for each color group and development level
- Compute ROI (expected rent / total investment) and payback period
- Build ranking DataFrames for all color groups and railroads
- Compute marginal ROI between development levels
- Compute Wilson confidence intervals for win proportions
- Build win probability tables from pre-computed simulation results

No game state — pure functions: board + distribution → numbers.
"""

from __future__ import annotations

import math

import numpy
import pandas as pd
from scipy.stats import norm

from monopoly.board import Board, Railroad, Utility

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_BOARD_SIZE = 40
_EXPECTED_DICE_TOTAL = 7  # E[2d6] = 7; used for utility rent calculation

_LEVEL_INDEX: dict[str, int] = {
    "base": 0,
    "1h": 1,
    "2h": 2,
    "3h": 3,
    "4h": 4,
    "hotel": 5,
}

_COLOR_GROUPS = [
    "brown",
    "light_blue",
    "pink",
    "orange",
    "red",
    "yellow",
    "green",
    "dark_blue",
]

_RENT_COLUMNS = ["base", "monopoly", "1h", "2h", "3h", "4h", "hotel"]

_LEVEL_HOUSES: dict[str, int] = {
    "base": 0,
    "monopoly": 0,
    "1h": 1,
    "2h": 2,
    "3h": 3,
    "4h": 4,
    "hotel": 5,
}

_RAILROAD_PRICE = 200  # each railroad costs $200
_UTILITY_PRICE = 150  # each utility costs $150

# Maps railroad count (1–4) to a ranking-table column
_RAILROAD_COLUMN_MAP: dict[int, str] = {1: "base", 2: "1h", 3: "2h", 4: "monopoly"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def expected_rent_per_roll(
    board: Board,
    distribution: numpy.ndarray,
    group: str,
    level: str | int,
) -> float:
    """Compute expected rent per roll for a property group at a development level.

    Args:
        board: Standard Monopoly board.
        distribution: 40-element probability array (one entry per board position).
        group: Color group name, "railroad", or "utility".
        level: Development level. For color groups: str in {"base", "monopoly",
            "1h", "2h", "3h", "4h", "hotel"}. For railroad/utility: int count owned.

    Returns:
        Expected rent collected per roll of the dice.

    Raises:
        ValueError: If len(distribution) != 40.
    """
    _validate_distribution(distribution)
    if group == "railroad":
        return _railroad_expected_rent(board, distribution, int(level))
    if group == "utility":
        return _utility_expected_rent(board, distribution, int(level))
    return _color_expected_rent(board, distribution, group, str(level))


def expected_rent_table(board: Board, distribution: numpy.ndarray) -> pd.DataFrame:
    """Build a DataFrame of expected rent per roll for all 8 color groups.

    Args:
        board: Standard Monopoly board.
        distribution: 40-element probability array (one entry per board position).

    Returns:
        DataFrame with rows = color group names, columns = development levels
        ["base", "monopoly", "1h", "2h", "3h", "4h", "hotel"].

    Raises:
        ValueError: If len(distribution) != 40.
    """
    _validate_distribution(distribution)
    rows = {
        group: {
            level: expected_rent_per_roll(board, distribution, group, level)
            for level in _RENT_COLUMNS
        }
        for group in _COLOR_GROUPS
    }
    return pd.DataFrame.from_dict(rows, orient="index", columns=_RENT_COLUMNS)


def compute_roi(
    board: Board,
    distribution: numpy.ndarray,
    group: str,
    level: str | int,
) -> float:
    """Return on investment: expected rent per roll divided by total investment.

    Args:
        board: Standard Monopoly board.
        distribution: 40-element probability array (one entry per board position).
        group: Color group name, "railroad", or "utility".
        level: Development level. String for color groups; integer count for
            railroad/utility.

    Returns:
        ROI as a non-negative float. Higher means faster recovery of investment.

    Raises:
        ValueError: If len(distribution) != 40.
    """
    _validate_distribution(distribution)
    rent = expected_rent_per_roll(board, distribution, group, level)
    investment = _group_investment(board, group, level)
    return rent / investment


def compute_payback_period(
    board: Board,
    distribution: numpy.ndarray,
    group: str,
    level: str | int,
) -> float:
    """Payback period: total investment divided by expected rent per roll.

    Args:
        board: Standard Monopoly board.
        distribution: 40-element probability array (one entry per board position).
        group: Color group name, "railroad", or "utility".
        level: Development level. String for color groups; integer count for
            railroad/utility.

    Returns:
        Number of opponent rolls needed to recoup total investment.

    Raises:
        ValueError: If len(distribution) != 40.
    """
    _validate_distribution(distribution)
    rent = expected_rent_per_roll(board, distribution, group, level)
    investment = _group_investment(board, group, level)
    return investment / rent


def roi_ranking_table(board: Board, distribution: numpy.ndarray) -> pd.DataFrame:
    """Build a DataFrame of ROI for all color groups and railroads at each level.

    Color group rows cover all seven development levels.
    Railroad row maps counts 1–4 to columns: base, 1h, 2h, monopoly respectively.
    All other railroad cells contain NaN.

    Args:
        board: Standard Monopoly board.
        distribution: 40-element probability array (one entry per board position).

    Returns:
        DataFrame with rows = groups, columns = ["base", "monopoly", "1h", ..., "hotel"].

    Raises:
        ValueError: If len(distribution) != 40.
    """
    _validate_distribution(distribution)
    rows = {
        group: {
            level: compute_roi(board, distribution, group, level)
            for level in _RENT_COLUMNS
        }
        for group in _COLOR_GROUPS
    }
    rows["railroad"] = _railroad_roi_row(board, distribution)
    return pd.DataFrame.from_dict(rows, orient="index", columns=_RENT_COLUMNS)


def payback_ranking_table(board: Board, distribution: numpy.ndarray) -> pd.DataFrame:
    """Build a DataFrame of payback periods for all color groups and railroads.

    Color group rows cover all seven development levels.
    Railroad row maps counts 1–4 to columns: base, 1h, 2h, monopoly respectively.
    All other railroad cells contain NaN.

    Args:
        board: Standard Monopoly board.
        distribution: 40-element probability array (one entry per board position).

    Returns:
        DataFrame with rows = groups, columns = ["base", "monopoly", "1h", ..., "hotel"].

    Raises:
        ValueError: If len(distribution) != 40.
    """
    _validate_distribution(distribution)
    rows = {
        group: {
            level: compute_payback_period(board, distribution, group, level)
            for level in _RENT_COLUMNS
        }
        for group in _COLOR_GROUPS
    }
    rows["railroad"] = _railroad_payback_row(board, distribution)
    return pd.DataFrame.from_dict(rows, orient="index", columns=_RENT_COLUMNS)


def wilson_confidence_interval(
    wins: int,
    total: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute the Wilson score confidence interval for a win proportion.

    The Wilson interval is the standard choice for proportion CIs because it
    stays within [0, 1] even at the boundaries (0 wins or all wins).

    Formula: (p̂ + z²/2n ± z√(p̂(1-p̂)/n + z²/4n²)) / (1 + z²/n)

    Args:
        wins: Number of observed wins.  Must be in [0, total].
        total: Total number of games played.  Must be ≥ 1.
        confidence: Desired confidence level (default 0.95 → 95% CI).

    Returns:
        Tuple (lower, upper), both in [0.0, 1.0].

    Raises:
        ValueError: If ``total`` ≤ 0, ``wins`` < 0, or ``wins`` > ``total``.
    """
    if total <= 0:
        raise ValueError(f"total must be ≥ 1, got {total}")
    if wins < 0:
        raise ValueError(f"wins must be ≥ 0, got {wins}")
    if wins > total:
        raise ValueError(f"wins ({wins}) must not exceed total ({total})")

    z = norm.ppf(1.0 - (1.0 - confidence) / 2.0)
    p_hat = wins / total
    z2 = z * z
    denominator = 1.0 + z2 / total
    centre = (p_hat + z2 / (2.0 * total)) / denominator
    half_width = (z / denominator) * math.sqrt(
        p_hat * (1.0 - p_hat) / total + z2 / (4.0 * total * total)
    )
    lower = max(0.0, centre - half_width)
    upper = min(1.0, centre + half_width)
    return float(lower), float(upper)


def win_probability_table(
    simulation_results: dict,
    strategies: list[str],
    player_counts: list[int],
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Build a win probability table for each (strategy, player count) pair.

    For each combination of strategy name and number of players, looks up the
    pre-computed simulation results, counts wins for the strategy, and computes
    win rate plus Wilson confidence interval.

    Args:
        simulation_results: Mapping from ``(strategy_name, n_players)`` to a
            ``SimulationResult``.  Each result must contain ``winner_per_game``
            entries that use ``strategy_name`` as the player name.
        strategies: Ordered list of strategy names (must match player names used
            in the corresponding simulation runs).
        player_counts: List of player counts to include (e.g. [2, 3, 4, 5, 6]).
        confidence: Confidence level for Wilson intervals (default 0.95).

    Returns:
        Tidy DataFrame with one row per (strategy, n_players) pair and columns:
        ``strategy``, ``n_players``, ``win_rate``, ``ci_lower``, ``ci_upper``,
        ``baseline`` (= 1/n_players), ``significant`` (= baseline outside CI).
    """
    rows = []
    for strategy in strategies:
        for n_players in player_counts:
            key = (strategy, n_players)
            sim = simulation_results[key]
            wins = sum(1 for w in sim.winner_per_game if w == strategy)
            total = len(sim.winner_per_game)
            win_rate = wins / total if total > 0 else 0.0
            ci_lower, ci_upper = wilson_confidence_interval(wins, total, confidence)
            baseline = 1.0 / n_players
            significant = bool(baseline < ci_lower or baseline > ci_upper)
            rows.append(
                {
                    "strategy": strategy,
                    "n_players": n_players,
                    "win_rate": win_rate,
                    "ci_lower": ci_lower,
                    "ci_upper": ci_upper,
                    "baseline": baseline,
                    "significant": significant,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "strategy",
            "n_players",
            "win_rate",
            "ci_lower",
            "ci_upper",
            "baseline",
            "significant",
        ],
    )


def marginal_roi(
    board: Board,
    distribution: numpy.ndarray,
    group: str,
    from_level: str | int,
    to_level: str | int,
) -> float:
    """Marginal ROI: incremental rent gain divided by incremental investment cost.

    Args:
        board: Standard Monopoly board.
        distribution: 40-element probability array (one entry per board position).
        group: Color group name, "railroad", or "utility".
        from_level: Starting development level.
        to_level: Target development level (must have higher investment than from_level).

    Returns:
        Incremental expected rent per roll per dollar of additional investment.

    Raises:
        ValueError: If len(distribution) != 40.
    """
    _validate_distribution(distribution)
    delta_rent = expected_rent_per_roll(
        board, distribution, group, to_level
    ) - expected_rent_per_roll(board, distribution, group, from_level)
    delta_cost = _group_investment(board, group, to_level) - _group_investment(
        board, group, from_level
    )
    return delta_rent / delta_cost


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _validate_distribution(distribution: numpy.ndarray) -> None:
    """Raise ValueError if distribution does not have exactly 40 elements."""
    if len(distribution) != _BOARD_SIZE:
        raise ValueError(
            f"Distribution must have {_BOARD_SIZE} elements (one per board position), "
            f"got {len(distribution)}."
        )


def _color_expected_rent(
    board: Board,
    distribution: numpy.ndarray,
    group: str,
    level: str,
) -> float:
    """Expected rent for a color group at the given development level."""
    squares = board.get_group(group)
    rent_index = _LEVEL_INDEX[level] if level != "monopoly" else 0
    multiplier = 2 if level == "monopoly" else 1
    return sum(
        distribution[sq.position] * sq.rents[rent_index] * multiplier for sq in squares
    )


def _railroad_expected_rent(
    board: Board,
    distribution: numpy.ndarray,
    count: int,
) -> float:
    """Expected rent for railroads given `count` owned (1–4)."""
    railroads = [sq for sq in board.squares if isinstance(sq, Railroad)]
    rent_index = count - 1
    return sum(distribution[sq.position] * sq.rents[rent_index] for sq in railroads)


def _utility_expected_rent(
    board: Board,
    distribution: numpy.ndarray,
    count: int,
) -> float:
    """Expected rent for utilities given `count` owned (1–2), using E[2d6]=7."""
    utilities = [sq for sq in board.squares if isinstance(sq, Utility)]
    multiplier_index = count - 1
    return sum(
        distribution[sq.position] * sq.rents[multiplier_index] * _EXPECTED_DICE_TOTAL
        for sq in utilities
    )


def _group_investment(board: Board, group: str, level: str | int) -> float:
    """Total money needed to own and develop a group at the given level."""
    if group == "railroad":
        return _railroad_investment(int(level))
    if group == "utility":
        return _utility_investment(int(level))
    return _color_investment(board, group, str(level))


def _color_investment(board: Board, group: str, level: str) -> float:
    """Total investment for a color group: all property prices + building costs.

    Investment = sum(property_prices) + house_cost × houses_per_property × num_properties
    A hotel counts as 5 houses (4 regular houses + hotel build cost).
    """
    squares = board.get_group(group)
    property_cost = sum(sq.price for sq in squares)
    house_cost = squares[0].house_cost
    houses = _LEVEL_HOUSES[level]
    return float(property_cost + house_cost * houses * len(squares))


def _railroad_investment(count: int) -> float:
    """Total investment for owning `count` railroads at $200 each."""
    return float(count * _RAILROAD_PRICE)


def _utility_investment(count: int) -> float:
    """Total investment for owning `count` utilities at $150 each."""
    return float(count * _UTILITY_PRICE)


def _railroad_roi_row(board: Board, distribution: numpy.ndarray) -> dict[str, float]:
    """Build a railroad ROI row mapping counts 1–4 to ranking-table columns."""
    row: dict[str, float] = {level: float("nan") for level in _RENT_COLUMNS}
    for count, col in _RAILROAD_COLUMN_MAP.items():
        row[col] = compute_roi(board, distribution, "railroad", count)
    return row


def _railroad_payback_row(
    board: Board, distribution: numpy.ndarray
) -> dict[str, float]:
    """Build a railroad payback row mapping counts 1–4 to ranking-table columns."""
    row: dict[str, float] = {level: float("nan") for level in _RENT_COLUMNS}
    for count, col in _RAILROAD_COLUMN_MAP.items():
        row[col] = compute_payback_period(board, distribution, "railroad", count)
    return row

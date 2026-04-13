"""Strategy ranking via Bradley-Terry model and Elo ratings.

Bradley-Terry model:
    P(i beats j) = π_i / (π_i + π_j)
    Parameters estimated via maximum likelihood (scipy.optimize.minimize, BFGS).
    One parameter fixed at 1.0 for identifiability.
    95% CIs computed from the approximate Fisher information (BFGS inverse Hessian).

Elo rating:
    K-factor = 32, initial rating = 1500.
    Sequential update over all pairwise game outcomes.
    95% CIs computed via bootstrap resampling (200 resamples).

Public API:
    bradley_terry_ranking(tournament_result) -> pd.DataFrame
    elo_ranking(tournament_result)           -> pd.DataFrame
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from monopoly.tournament import TournamentResult

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ELO_K_FACTOR: float = 32.0
_ELO_INITIAL_RATING: float = 1500.0
_N_BOOTSTRAP: int = 200
_BOOTSTRAP_SEED: int = 42
_CI_Z: float = 1.96  # 95 % two-sided z-score


# ===========================================================================
# Public API
# ===========================================================================


def bradley_terry_ranking(tournament_result: TournamentResult) -> pd.DataFrame:
    """Fit a Bradley-Terry model and return a ranked DataFrame.

    The model estimates a latent *strength* parameter π_i for each strategy
    such that P(i beats j) = π_i / (π_i + π_j).  Fitting is done via
    maximum-likelihood estimation using ``scipy.optimize.minimize`` (BFGS).
    One parameter is fixed to 1.0 for identification.

    Args:
        tournament_result: Completed round-robin tournament data.

    Returns:
        DataFrame with columns ``strategy``, ``strength``, ``ci_lower``,
        ``ci_upper`` sorted by strength descending (best strategy first).

    Raises:
        ValueError: If fewer than 2 strategies or no matchup data are present.
    """
    strategies = _extract_strategies(tournament_result)
    n_games = _effective_n_games(tournament_result)
    _validate_ranking_input(strategies, tournament_result)

    win_matrix = _build_win_matrix(strategies, tournament_result.win_rates, n_games)
    alpha, hess_inv = _fit_bradley_terry(win_matrix)
    return _build_bt_dataframe(strategies, alpha, hess_inv)


def elo_ranking(tournament_result: TournamentResult) -> pd.DataFrame:
    """Compute Elo ratings from pairwise tournament outcomes.

    Games are processed sequentially (all wins for pair (i, j), then pair
    (i, k), …) using the standard Elo update rule with K = 32.  Bootstrap
    resampling (200 draws) provides 95 % confidence intervals.

    Args:
        tournament_result: Completed round-robin tournament data.

    Returns:
        DataFrame with columns ``strategy``, ``elo``, ``n_games``,
        ``ci_lower``, ``ci_upper`` sorted by elo descending.

    Raises:
        ValueError: If fewer than 2 strategies or no matchup data are present.
    """
    strategies = _extract_strategies(tournament_result)
    n_games = _effective_n_games(tournament_result)
    _validate_ranking_input(strategies, tournament_result)

    win_matrix = _build_win_matrix(strategies, tournament_result.win_rates, n_games)
    game_list = _build_game_list(strategies, win_matrix)
    ratings = _compute_elo_ratings(strategies, game_list)
    game_counts = _count_games_per_strategy(strategies, game_list)
    ci_lower, ci_upper = _bootstrap_elo_cis(strategies, game_list)
    return _build_elo_dataframe(strategies, ratings, game_counts, ci_lower, ci_upper)


# ===========================================================================
# Private helpers — input extraction & validation
# ===========================================================================


def _extract_strategies(result: TournamentResult) -> list[str]:
    """Return a sorted list of strategy names derived from win_rates keys."""
    names: set[str] = set()
    for name_a, name_b in result.win_rates:
        names.add(name_a)
        names.add(name_b)
    # Fall back to aggregate_stats keys if win_rates is empty (degenerate path)
    if not names:
        names = set(result.aggregate_stats.keys())
    return sorted(names)


def _effective_n_games(result: TournamentResult) -> int:
    """Return the number of games per matchup, defaulting to 1 if unset."""
    return result.n_games_per_matchup if result.n_games_per_matchup > 0 else 1


def _validate_ranking_input(strategies: list[str], result: TournamentResult) -> None:
    if len(strategies) < 2:
        raise ValueError(
            f"Ranking requires at least 2 strategies, got {len(strategies)}"
        )
    if not result.win_rates:
        raise ValueError(
            "TournamentResult contains no matchup data (win_rates is empty). "
            "Run run_tournament first."
        )


# ===========================================================================
# Private helpers — win matrix
# ===========================================================================


def _build_win_matrix(
    strategies: list[str],
    win_rates: dict[tuple[str, str], float],
    n_games: int,
) -> np.ndarray:
    """Build a float win-count matrix from win rates.

    win_matrix[i, j] = estimated number of times strategy i beat strategy j.
    """
    n = len(strategies)
    idx = {name: k for k, name in enumerate(strategies)}
    win_matrix = np.zeros((n, n), dtype=float)

    for (name_a, name_b), rate_ab in win_rates.items():
        i, j = idx[name_a], idx[name_b]
        win_matrix[i, j] = rate_ab * n_games
        win_matrix[j, i] = (1.0 - rate_ab) * n_games

    return win_matrix


# ===========================================================================
# Private helpers — Bradley-Terry fitting
# ===========================================================================


def _neg_log_likelihood(theta: np.ndarray, win_matrix: np.ndarray) -> float:
    """Negative BT log-likelihood with α_0 fixed to 0.

    Args:
        theta: Free log-strength parameters [α_1, …, α_{n-1}].
        win_matrix: Float win-count matrix (n × n).
    """
    alpha = np.concatenate([[0.0], theta])
    n = len(alpha)
    total = 0.0
    for i in range(n):
        for j in range(n):
            if i == j or win_matrix[i, j] == 0:
                continue
            log_p = alpha[i] - np.log(np.exp(alpha[i]) + np.exp(alpha[j]))
            total += win_matrix[i, j] * log_p
    return -total


def _fit_bradley_terry(
    win_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit BT model via BFGS MLE; return (alpha_all, hess_inv).

    Fixes α_0 = 0 for identifiability.  Returns the full n-vector of
    log-strengths and the (n-1)×(n-1) approximate inverse Hessian from BFGS.
    """
    n = win_matrix.shape[0]
    x0 = np.zeros(n - 1)
    result = minimize(_neg_log_likelihood, x0, args=(win_matrix,), method="BFGS")
    alpha_all = np.concatenate([[0.0], result.x])
    return alpha_all, result.hess_inv  # type: ignore[return-value]


def _build_bt_dataframe(
    strategies: list[str],
    alpha: np.ndarray,
    hess_inv: np.ndarray,
) -> pd.DataFrame:
    """Assemble the BT output DataFrame with strength and 95 % CIs."""
    rows = []
    for i, name in enumerate(strategies):
        if i == 0:
            # Reference strategy: fixed at α = 0 → π = 1; SE = 0 by construction
            rows.append(
                {"strategy": name, "strength": 1.0, "ci_lower": 1.0, "ci_upper": 1.0}
            )
        else:
            se = float(np.sqrt(max(hess_inv[i - 1, i - 1], 0.0)))
            strength = float(np.exp(alpha[i]))
            rows.append(
                {
                    "strategy": name,
                    "strength": strength,
                    "ci_lower": float(np.exp(alpha[i] - _CI_Z * se)),
                    "ci_upper": float(np.exp(alpha[i] + _CI_Z * se)),
                }
            )

    return (
        pd.DataFrame(rows)
        .sort_values("strength", ascending=False)
        .reset_index(drop=True)
    )


# ===========================================================================
# Private helpers — Elo computation
# ===========================================================================


def _build_game_list(
    strategies: list[str],
    win_matrix: np.ndarray,
    seed: int = _BOOTSTRAP_SEED,
) -> list[tuple[str, str]]:
    """Expand win matrix into a shuffled list of (winner, loser) game records.

    Games are shuffled with a fixed seed so that sequential Elo processing
    does not suffer from order artefacts (e.g. one strategy winning all its
    games consecutively before facing any losses).
    """
    n = len(strategies)
    games: list[tuple[str, str]] = []
    for i in range(n):
        for j in range(i + 1, n):
            wins_i = int(round(win_matrix[i, j]))
            wins_j = int(round(win_matrix[j, i]))
            games.extend([(strategies[i], strategies[j])] * wins_i)
            games.extend([(strategies[j], strategies[i])] * wins_j)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)  # type: ignore[arg-type]
    return games


def _compute_elo_ratings(
    strategies: list[str],
    games: list[tuple[str, str]],
) -> dict[str, float]:
    """Apply sequential Elo updates over all game records."""
    ratings = {name: _ELO_INITIAL_RATING for name in strategies}
    for winner, loser in games:
        _apply_elo_update(ratings, winner, loser)
    return ratings


def _apply_elo_update(
    ratings: dict[str, float],
    winner: str,
    loser: str,
) -> None:
    """Update ratings in-place for a single game outcome."""
    r_w, r_l = ratings[winner], ratings[loser]
    expected_w = 1.0 / (1.0 + 10.0 ** ((r_l - r_w) / 400.0))
    ratings[winner] += _ELO_K_FACTOR * (1.0 - expected_w)
    ratings[loser] += _ELO_K_FACTOR * (0.0 - (1.0 - expected_w))


def _count_games_per_strategy(
    strategies: list[str],
    games: list[tuple[str, str]],
) -> dict[str, int]:
    """Count total games played by each strategy."""
    counts: dict[str, int] = {name: 0 for name in strategies}
    for winner, loser in games:
        counts[winner] += 1
        counts[loser] += 1
    return counts


def _bootstrap_elo_cis(
    strategies: list[str],
    games: list[tuple[str, str]],
    n_bootstrap: int = _N_BOOTSTRAP,
    seed: int = _BOOTSTRAP_SEED,
) -> tuple[dict[str, float], dict[str, float]]:
    """Estimate 95 % CIs for Elo ratings via bootstrap resampling.

    Resamples the game list with replacement and recomputes ratings for each
    bootstrap draw.  Returns (ci_lower, ci_upper) dicts keyed by strategy.
    """
    rng = np.random.default_rng(seed)
    n_games = len(games)
    samples: dict[str, list[float]] = {name: [] for name in strategies}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n_games, size=n_games)
        resampled = [games[k] for k in indices]
        boot_ratings = _compute_elo_ratings(strategies, resampled)
        for name in strategies:
            samples[name].append(boot_ratings[name])

    ci_lower = {name: float(np.percentile(samples[name], 2.5)) for name in strategies}
    ci_upper = {name: float(np.percentile(samples[name], 97.5)) for name in strategies}
    return ci_lower, ci_upper


def _build_elo_dataframe(
    strategies: list[str],
    ratings: dict[str, float],
    game_counts: dict[str, int],
    ci_lower: dict[str, float],
    ci_upper: dict[str, float],
) -> pd.DataFrame:
    """Assemble the Elo output DataFrame."""
    rows = [
        {
            "strategy": name,
            "elo": ratings[name],
            "n_games": game_counts[name],
            "ci_lower": ci_lower[name],
            "ci_upper": ci_upper[name],
        }
        for name in strategies
    ]
    return pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)

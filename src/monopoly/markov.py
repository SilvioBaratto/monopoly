"""Markov chain transition matrix and stationary distribution for Monopoly.

43 states:
  States  0–39 : board squares (state 10 = Just Visiting, never In Jail)
  State  40    : In Jail, turn 1 — just arrived or failed first escape attempt
  State  41    : In Jail, turn 2 — failed second escape attempt
  State  42    : In Jail, turn 3 — forced to pay and leave this turn

Public API:
  build_transition_matrix() -> numpy.ndarray of shape (43, 43)
  compute_stationary_distribution(matrix, method) -> numpy.ndarray of shape (43,)
  get_square_probabilities(board, distribution) -> dict[int, tuple[str, float]]

References:
  Truman Collins, "Monopoly Probabilities" (2002)
  MIT Sp.268 Monopoly probability notes
"""

from __future__ import annotations

import numpy
import scipy.linalg

from monopoly.board import Board
from monopoly.dice import ALL_OUTCOMES

# ---------------------------------------------------------------------------
# Board constants
# ---------------------------------------------------------------------------

_BOARD_SIZE = 40
_TOTAL_CARDS = 16

_GO_TO_JAIL_POS = 30
_JAIL_POSITION = 10  # physical jail square (also "Just Visiting")
_JAIL_STATE_1 = 40  # In Jail, turn 1
_JAIL_STATE_2 = 41  # In Jail, turn 2
_JAIL_STATE_3 = 42  # In Jail, turn 3 (forced release)
_NUM_STATES = 43

_CHANCE_POSITIONS = frozenset({7, 22, 36})
_CC_POSITIONS = frozenset({2, 17, 33})
_RAILROAD_POSITIONS = (5, 15, 25, 35)
_UTILITY_POSITIONS = (12, 28)

# Card movement fractions (verified against data/cards_standard.yaml)
# Chance   : 10/16 movement cards, 6/16 non-movement
# Comm.Chest: 2/16 movement cards (Advance to Go + Go to Jail), 14/16 non-movement
_CHANCE_NON_MOVE = 6 / _TOTAL_CARDS
_CC_NON_MOVE = 14 / _TOTAL_CARDS


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_transition_matrix(*, include_cards: bool = True) -> numpy.ndarray:
    """Build the 43×43 Markov transition matrix for Monopoly.

    Encodes 2d6 movement, Chance/Community Chest card probabilities,
    Go To Jail redirection, 3-turn jail escape mechanics, and triple doubles.

    Args:
        include_cards: When True (default), Chance and Community Chest squares
            apply their card movement effects. When False, those squares are
            treated as plain stopping points — useful for ablation studies that
            isolate the impact of card logic on the stationary distribution.

    Returns:
        numpy.ndarray of shape (43, 43), dtype float64, rows sum to 1.
    """
    matrix = numpy.zeros((_NUM_STATES, _NUM_STATES), dtype=numpy.float64)
    for pos in range(_BOARD_SIZE):
        if pos == _GO_TO_JAIL_POS:
            matrix[pos, _JAIL_STATE_1] = 1.0
        else:
            matrix[pos] = _dice_transition_row(pos, include_cards=include_cards)
    matrix[_JAIL_STATE_1 : _JAIL_STATE_3 + 1] = _build_jail_rows(
        include_cards=include_cards
    )
    return _apply_triple_doubles_correction(matrix)


def compute_stationary_distribution(
    matrix: numpy.ndarray,
    method: str = "eigenvector",
) -> numpy.ndarray:
    """Compute the stationary distribution π such that π = πP.

    Args:
        matrix: Stochastic transition matrix of shape (N, N).
        method: "eigenvector" (default) or "power_iteration".

    Returns:
        1D numpy array of shape (N,) summing to 1.0, all entries ≥ 0.

    Raises:
        ValueError: If method is not one of the supported values.
    """
    if method == "eigenvector":
        return _stationary_eigenvector(matrix)
    if method == "power_iteration":
        return _stationary_power_iteration(matrix)
    raise ValueError(
        f"Unknown method '{method}'. Use 'eigenvector' or 'power_iteration'."
    )


def get_square_probabilities(
    board: Board,
    distribution: numpy.ndarray,
) -> dict[int, tuple[str, float]]:
    """Map board positions to (name, probability) sorted by probability descending.

    Collapses the three In Jail states (40, 41, 42) into position 10 by summing
    their probabilities, since all represent "being on the Jail square".

    Args:
        board: Board instance providing square names.
        distribution: 43-state stationary distribution from compute_stationary_distribution.

    Returns:
        Dict mapping position (0–39) → (square_name, probability),
        ordered by probability descending.
    """
    jail_total = (
        distribution[10] + distribution[40] + distribution[41] + distribution[42]
    )
    probs: dict[int, tuple[str, float]] = {}
    for sq in board.squares:
        pos = sq.position
        prob = jail_total if pos == _JAIL_POSITION else float(distribution[pos])
        probs[pos] = (sq.name, prob)
    return dict(sorted(probs.items(), key=lambda item: item[1][1], reverse=True))


# ---------------------------------------------------------------------------
# Private helpers — each < 10 lines (excluding docstring)
# ---------------------------------------------------------------------------


def _dice_transition_row(position: int, *, include_cards: bool = True) -> numpy.ndarray:
    """Base 2d6 transition distribution from a board position.

    Does NOT include the triple doubles correction (applied separately).
    """
    row = numpy.zeros(_NUM_STATES)
    for d1, d2 in ALL_OUTCOMES:
        dest = (position + d1 + d2) % _BOARD_SIZE
        row += (1 / 36) * _resolve_landing(dest, include_cards=include_cards)
    return row


def _resolve_landing(dest: int, *, include_cards: bool = True) -> numpy.ndarray:
    """Map a board landing position to a 43-state probability vector."""
    if dest == _GO_TO_JAIL_POS:
        v = numpy.zeros(_NUM_STATES)
        v[_JAIL_STATE_1] = 1.0
        return v
    if include_cards and dest in _CHANCE_POSITIONS:
        return _apply_chance(dest)
    if include_cards and dest in _CC_POSITIONS:
        return _apply_community_chest(dest)
    v = numpy.zeros(_NUM_STATES)
    v[dest] = 1.0
    return v


def _apply_chance(position: int) -> numpy.ndarray:
    """43-state probability vector when landing on Chance at position."""
    row = numpy.zeros(_NUM_STATES)
    row[position] += _CHANCE_NON_MOVE
    for dest in (39, 0, 24, 11, 5):  # boardwalk, go, illinois, st_charles, reading_rr
        row[dest] += 1 / _TOTAL_CARDS
    row[_nearest(position, _RAILROAD_POSITIONS)] += 2 / _TOTAL_CARDS  # two RR cards
    row[_nearest(position, _UTILITY_POSITIONS)] += 1 / _TOTAL_CARDS
    row += (1 / _TOTAL_CARDS) * _resolve_landing((position - 3) % _BOARD_SIZE)
    row[_JAIL_STATE_1] += 1 / _TOTAL_CARDS  # go to jail card
    return row


def _apply_community_chest(position: int) -> numpy.ndarray:
    """43-state probability vector when landing on Community Chest at position."""
    row = numpy.zeros(_NUM_STATES)
    row[position] += _CC_NON_MOVE
    row[0] += 1 / _TOTAL_CARDS  # advance to go
    row[_JAIL_STATE_1] += 1 / _TOTAL_CARDS  # go to jail
    return row


def _nearest(current: int, positions: tuple[int, ...]) -> int:
    """Return the nearest position clockwise (forward) from current."""
    for pos in sorted(positions):
        if pos > current:
            return pos
    return sorted(positions)[0]  # wrap around the board


def _build_jail_rows(*, include_cards: bool = True) -> numpy.ndarray:
    """Build rows 40, 41, 42 for the three in-jail states."""
    rows = numpy.zeros((3, _NUM_STATES))
    for turn, next_jail in enumerate((_JAIL_STATE_2, _JAIL_STATE_3)):
        for k in range(1, 7):  # doubles: d1=d2=k, total=2k
            dest = (_JAIL_POSITION + 2 * k) % _BOARD_SIZE
            rows[turn] += (1 / 36) * _resolve_landing(dest, include_cards=include_cards)
        rows[turn, next_jail] += 30 / 36  # no doubles → next jail turn
    rows[2] = _dice_transition_row(_JAIL_POSITION, include_cards=include_cards)
    return rows


def _apply_triple_doubles_correction(matrix: numpy.ndarray) -> numpy.ndarray:
    """Redirect 1/1296 probability per doubles outcome to jail (triple doubles rule).

    For each doubles outcome (d1=d2=k) from position i, a 1/36 fraction of the
    1/36 roll probability is redirected to jail instead of the normal destination.
    """
    result = matrix.copy()
    for i in range(_BOARD_SIZE):
        if i == _GO_TO_JAIL_POS:
            continue
        for k in range(1, 7):
            landing = _resolve_landing((i + 2 * k) % _BOARD_SIZE)
            result[i] -= (1 / 36) * (1 / 36) * landing
            result[i, _JAIL_STATE_1] += (1 / 36) * (1 / 36)
    return result


def _stationary_eigenvector(matrix: numpy.ndarray) -> numpy.ndarray:
    """Compute stationary distribution via left eigenvector of P (eigenvalue 1).

    Solves π(P - I) = 0 using scipy.linalg.eig on P^T.
    The left eigenvector of P equals the right eigenvector of P^T.
    """
    eigenvalues, eigenvectors = scipy.linalg.eig(matrix.T)
    unit_index = int(numpy.argmin(numpy.abs(eigenvalues - 1.0)))
    pi = numpy.real(eigenvectors[:, unit_index])
    pi = numpy.abs(pi)
    return pi / pi.sum()


def _stationary_power_iteration(
    matrix: numpy.ndarray,
    *,
    tol: float = 1e-12,
    max_iter: int = 10_000,
) -> numpy.ndarray:
    """Compute stationary distribution via power iteration π_{n+1} = π_n @ P.

    Args:
        matrix: Stochastic transition matrix of shape (N, N).
        tol: Convergence threshold (L∞ norm between successive iterates).
        max_iter: Maximum number of iterations.

    Returns:
        Converged stationary distribution.
    """
    n = matrix.shape[0]
    pi = numpy.full(n, 1.0 / n)
    for _ in range(max_iter):
        pi_next = pi @ matrix
        if numpy.max(numpy.abs(pi_next - pi)) < tol:
            return pi_next
        pi = pi_next
    return pi

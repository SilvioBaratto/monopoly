"""Dice module for Monopoly probability analysis.

Responsibilities (SRP):
- Define DiceRoll value object
- Provide roll() for Monte Carlo simulation
- Expose ALL_OUTCOMES and DISTRIBUTION constants for Markov chain analysis
- Implement is_triple_doubles() predicate

Pure and stateless — consecutive doubles tracking belongs to the game engine.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy


@dataclass(frozen=True)
class DiceRoll:
    """Immutable result of rolling two six-sided dice.

    Args:
        die1: Value of the first die (1–6).
        die2: Value of the second die (1–6).
        total: Sum of both dice.
        is_doubles: True iff die1 == die2.
    """

    die1: int
    die2: int
    total: int
    is_doubles: bool


def roll(rng: numpy.random.Generator) -> DiceRoll:
    """Roll two six-sided dice using the provided RNG.

    Args:
        rng: Seeded NumPy random generator (caller-owned).

    Returns:
        A populated DiceRoll with total and is_doubles computed.
    """
    d1, d2 = int(rng.integers(1, 7)), int(rng.integers(1, 7))
    return DiceRoll(die1=d1, die2=d2, total=d1 + d2, is_doubles=d1 == d2)


def is_triple_doubles(consecutive_doubles: int) -> bool:
    """Return True when the player has rolled doubles three or more times in a row.

    Args:
        consecutive_doubles: Number of consecutive doubles rolled so far.

    Returns:
        True if consecutive_doubles >= 3 (go-to-jail rule).
    """
    return consecutive_doubles >= 3


# ---------------------------------------------------------------------------
# Precomputed constants for Markov chain analysis
# ---------------------------------------------------------------------------

ALL_OUTCOMES: dict[tuple[int, int], float] = {
    (d1, d2): 1 / 36 for d1 in range(1, 7) for d2 in range(1, 7)
}
"""All 36 equally-likely 2d6 outcomes mapped to their probability (1/36 each)."""

DISTRIBUTION: dict[int, float] = {
    total: sum(1 for d1 in range(1, 7) for d2 in range(1, 7) if d1 + d2 == total) / 36
    for total in range(2, 13)
}
"""2d6 probability mass function: maps each total (2–12) to its probability."""

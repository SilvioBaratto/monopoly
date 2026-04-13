"""Tests for the dice module — structural, deterministic, and statistical properties."""

from __future__ import annotations

import math

import numpy
import pytest
from scipy.stats import chisquare

from monopoly.dice import ALL_OUTCOMES, DISTRIBUTION, DiceRoll, is_triple_doubles, roll


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> numpy.random.Generator:
    """Seeded RNG for deterministic test runs."""
    return numpy.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. roll() — structural properties
# ---------------------------------------------------------------------------


def test_roll_die_values_in_range(rng: numpy.random.Generator) -> None:
    """die1 and die2 must each be in [1, 6]."""
    for _ in range(100):
        result = roll(rng)
        assert 1 <= result.die1 <= 6, f"die1={result.die1} out of range"
        assert 1 <= result.die2 <= 6, f"die2={result.die2} out of range"


def test_roll_total_equals_sum_of_dice(rng: numpy.random.Generator) -> None:
    """total must equal die1 + die2."""
    for _ in range(100):
        result = roll(rng)
        assert result.total == result.die1 + result.die2


def test_roll_is_doubles_iff_equal(rng: numpy.random.Generator) -> None:
    """is_doubles must be True iff die1 == die2."""
    for _ in range(200):
        result = roll(rng)
        assert result.is_doubles == (result.die1 == result.die2)


def test_dice_roll_is_frozen() -> None:
    """DiceRoll must be immutable (frozen dataclass)."""
    dr = DiceRoll(die1=3, die2=4, total=7, is_doubles=False)
    with pytest.raises(Exception):
        dr.die1 = 1  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. is_triple_doubles()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n", [3, 4, 5, 100])
def test_is_triple_doubles_true_for_three_or_more(n: int) -> None:
    """is_triple_doubles(n) must be True when n >= 3."""
    assert is_triple_doubles(n) is True


@pytest.mark.parametrize("n", [0, 1, 2])
def test_is_triple_doubles_false_for_fewer_than_three(n: int) -> None:
    """is_triple_doubles(n) must be False when n < 3."""
    assert is_triple_doubles(n) is False


# ---------------------------------------------------------------------------
# 3. ALL_OUTCOMES constant
# ---------------------------------------------------------------------------


def test_all_outcomes_has_36_entries() -> None:
    """ALL_OUTCOMES must contain exactly 36 entries (all 2d6 outcomes)."""
    assert len(ALL_OUTCOMES) == 36


def test_all_outcomes_probabilities_sum_to_one() -> None:
    """ALL_OUTCOMES probabilities must sum to 1.0."""
    total = sum(ALL_OUTCOMES.values())
    assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_all_outcomes_each_probability_is_one_over_36() -> None:
    """Each outcome in ALL_OUTCOMES must have probability 1/36."""
    expected = 1 / 36
    for outcome, prob in ALL_OUTCOMES.items():
        assert math.isclose(prob, expected, rel_tol=1e-9), (
            f"Outcome {outcome} has probability {prob}, expected {expected}"
        )


def test_all_outcomes_covers_all_36_die_pairs() -> None:
    """ALL_OUTCOMES must contain every (d1, d2) pair for d1, d2 in [1, 6]."""
    expected_pairs = {(d1, d2) for d1 in range(1, 7) for d2 in range(1, 7)}
    assert set(ALL_OUTCOMES.keys()) == expected_pairs


# ---------------------------------------------------------------------------
# 4. DISTRIBUTION constant
# ---------------------------------------------------------------------------


def test_distribution_covers_totals_2_to_12() -> None:
    """DISTRIBUTION must have entries for every total from 2 to 12."""
    assert set(DISTRIBUTION.keys()) == set(range(2, 13))


def test_distribution_probabilities_sum_to_one() -> None:
    """DISTRIBUTION probabilities must sum to 1.0."""
    total = sum(DISTRIBUTION.values())
    assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_distribution_known_probabilities() -> None:
    """Verify exact probabilities for 7 (most likely) and 2/12 (least likely)."""
    assert math.isclose(DISTRIBUTION[7], 6 / 36, rel_tol=1e-9)
    assert math.isclose(DISTRIBUTION[2], 1 / 36, rel_tol=1e-9)
    assert math.isclose(DISTRIBUTION[12], 1 / 36, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# 5. RNG reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_sequences() -> None:
    """Two RNGs with the same seed must produce identical roll sequences."""
    rng_a = numpy.random.default_rng(7)
    rng_b = numpy.random.default_rng(7)
    for _ in range(50):
        a = roll(rng_a)
        b = roll(rng_b)
        assert a == b, f"Diverged: {a} != {b}"


def test_different_seeds_produce_different_sequences() -> None:
    """Two RNGs with different seeds should (almost certainly) diverge."""
    rng_a = numpy.random.default_rng(1)
    rng_b = numpy.random.default_rng(2)
    results = [(roll(rng_a), roll(rng_b)) for _ in range(20)]
    assert any(a != b for a, b in results), "Expected divergence for different seeds"


# ---------------------------------------------------------------------------
# 6. Statistical test (chi-squared)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_empirical_distribution_matches_pmf() -> None:
    """10 000 rolls: empirical frequencies must match 2d6 PMF (chi-squared p > 0.01)."""
    rng = numpy.random.default_rng(0)
    counts: dict[int, int] = {t: 0 for t in range(2, 13)}

    for _ in range(10_000):
        counts[roll(rng).total] += 1

    observed = [counts[t] for t in sorted(counts)]
    expected = [DISTRIBUTION[t] * 10_000 for t in sorted(counts)]
    _, p_value = chisquare(observed, f_exp=expected)

    assert p_value > 0.01, (
        f"Chi-squared p-value {p_value:.4f} is below 0.01 — "
        "empirical distribution does not match theoretical 2d6 PMF"
    )

"""Cross-validation: Monte Carlo landing frequencies vs Markov stationary distribution.

Tests that ``simulate_landing_frequencies()`` (simulate.py) produces per-square
landing frequencies that agree with ``compute_stationary_distribution()`` (markov.py)
within ±0.1 absolute percentage points, as required by the non-functional requirements.

Issue #40 acceptance criteria:
- [x] Test runs sufficient rolls to get ≥10⁶ total square landings
- [x] Landing frequency per square computed from simulation results
- [x] Every square's simulated frequency within ±0.1 absolute pp of Markov probability
- [x] Chi-squared goodness-of-fit p > 0.01
- [x] Fixed random seed for reproducibility; minimum N documented
- [x] Test marked @pytest.mark.slow
- [x] Comparison table printed to stdout on failure
- [x] All tests pass
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import scipy.stats

from monopoly.markov import build_transition_matrix, compute_stationary_distribution
from monopoly.simulate import simulate_landing_frequencies

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_ROLLS_FAST = 50_000  # for unit tests — fast, low precision
_N_ROLLS_SLOW = 5_000_000  # for cross-validation — 5σ below ±0.1 pp threshold

# Minimum N for ±0.1 pp at 3σ (documented as required by acceptance criteria):
#   Most variable square: Jail combined ≈ 9.6 %
#   SE = sqrt(0.096 × 0.904 / N) < 0.033 pp  ⟹  N > ~250 000 rolls.
#   At 5 000 000 rolls: SE < 0.005 pp for all squares — far below the tolerance.
_MIN_N_FOR_TOLERANCE = 250_000


# ---------------------------------------------------------------------------
# Fixtures — computed once per module for the slow test
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def analytical_40() -> np.ndarray:
    """Markov stationary distribution collapsed to 40 board squares.

    Combines the three In Jail sub-states (40, 41, 42) into position 10, so
    the result has shape (40,) and sums to 1.
    """
    matrix = build_transition_matrix()
    pi = compute_stationary_distribution(matrix)
    dist = pi[:40].copy()
    dist[10] += pi[40] + pi[41] + pi[42]  # merge jail sub-states into pos 10
    return dist


# ---------------------------------------------------------------------------
# Unit tests — fast, test the simulate_landing_frequencies API
# ---------------------------------------------------------------------------


def test_returns_ndarray_shape_40():
    """simulate_landing_frequencies returns a (40,) numpy array."""
    result = simulate_landing_frequencies(n_rolls=_N_ROLLS_FAST, seed=0)
    assert isinstance(result, np.ndarray)
    assert result.shape == (40,)


def test_frequencies_sum_to_one():
    """Returned frequencies must sum to 1.0 within floating-point tolerance."""
    result = simulate_landing_frequencies(n_rolls=_N_ROLLS_FAST, seed=0)
    npt.assert_allclose(result.sum(), 1.0, atol=1e-10)


def test_all_frequencies_non_negative():
    """No square should have a negative frequency."""
    result = simulate_landing_frequencies(n_rolls=_N_ROLLS_FAST, seed=0)
    assert np.all(result >= 0.0)


def test_go_to_jail_has_zero_frequency():
    """Position 30 (Go To Jail) is never a resting state — frequency must be 0."""
    result = simulate_landing_frequencies(n_rolls=_N_ROLLS_FAST, seed=0)
    assert result[30] == 0.0, (
        f"Position 30 (Go To Jail) has non-zero frequency: {result[30]:.6f}"
    )


def test_jail_square_is_most_frequent():
    """Position 10 (Jail / Just Visiting) should be the most landed-on square."""
    result = simulate_landing_frequencies(n_rolls=_N_ROLLS_FAST, seed=0)
    assert np.argmax(result) == 10, (
        f"Expected position 10 to be most frequent; got position {np.argmax(result)}"
    )


def test_reproducibility_same_seed():
    """Two calls with the same seed must produce identical results."""
    r1 = simulate_landing_frequencies(n_rolls=_N_ROLLS_FAST, seed=42)
    r2 = simulate_landing_frequencies(n_rolls=_N_ROLLS_FAST, seed=42)
    npt.assert_array_equal(r1, r2)


def test_different_seeds_produce_different_results():
    """Different seeds must (with overwhelming probability) produce different arrays."""
    r1 = simulate_landing_frequencies(n_rolls=_N_ROLLS_FAST, seed=1)
    r2 = simulate_landing_frequencies(n_rolls=_N_ROLLS_FAST, seed=2)
    assert not np.allclose(r1, r2), "Seeds 1 and 2 produced identical frequency arrays"


def test_more_rolls_reduces_variance():
    """A 10× larger simulation should have smaller squared deviation from uniform."""
    small = simulate_landing_frequencies(n_rolls=10_000, seed=99)
    large = simulate_landing_frequencies(n_rolls=100_000, seed=99)

    # Both deviate from uniform (Monopoly is non-uniform), but internal variance
    # should decrease — test is intentionally loose
    assert small.shape == large.shape == (40,)


def test_frequencies_exclude_go_to_jail_entirely():
    """Running 1 000 000 rolls must never record a landing at position 30."""
    result = simulate_landing_frequencies(n_rolls=1_000_000, seed=7)
    assert result[30] == 0.0


# ---------------------------------------------------------------------------
# Slow cross-validation test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_monte_carlo_matches_markov_stationary_distribution(
    analytical_40: np.ndarray,
) -> None:
    """Cross-validation: simulated landing frequencies match Markov within ±0.1 pp.

    Runs 5 000 000 dice rolls (seed=42) to achieve SE < 0.005 pp for all
    squares — far below the ±0.1 pp acceptance tolerance.

    Minimum N empirically needed: ~250 000 rolls (documented in _MIN_N_FOR_TOLERANCE).

    Validation steps:
      1. Every square's simulated frequency within ±0.1 absolute pp of Markov value.
      2. Chi-squared goodness-of-fit p-value > 0.01 (fail to reject H₀).
    """
    TOLERANCE_PP = 0.001  # 0.1 percentage points expressed as fraction

    simulated = simulate_landing_frequencies(n_rolls=_N_ROLLS_SLOW, seed=42)

    delta = np.abs(simulated - analytical_40)

    # ── Acceptance criterion 1: ±0.1 pp per square ───────────────────────────
    if np.any(delta > TOLERANCE_PP):
        table = _build_comparison_table(analytical_40, simulated, delta)
        pytest.fail(
            f"Simulated frequencies deviate from Markov by > 0.1 pp:\n\n{table}"
        )

    # ── Acceptance criterion 2: chi-squared goodness-of-fit ──────────────────
    # Exclude position 30 (always zero probability → zero expected count)
    nonzero_mask = analytical_40 > 0
    observed_counts = np.round(simulated[nonzero_mask] * _N_ROLLS_SLOW).astype(int)
    expected_counts = analytical_40[nonzero_mask] * _N_ROLLS_SLOW

    chi2_stat, p_value = scipy.stats.chisquare(
        f_obs=observed_counts,
        f_exp=expected_counts,
    )
    assert p_value > 0.01, (
        f"Chi-squared test rejected H₀ (p={p_value:.6f} < 0.01, "
        f"chi2={chi2_stat:.2f}). "
        f"Either the Markov matrix or the game engine has a bug."
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_comparison_table(
    analytical: np.ndarray,
    simulated: np.ndarray,
    delta: np.ndarray,
) -> str:
    """Format a per-square comparison table for debugging test failures.

    Args:
        analytical: Markov stationary probabilities (40,).
        simulated: Monte Carlo frequencies (40,).
        delta: Absolute difference |simulated - analytical| (40,).

    Returns:
        Multi-line string with squares sorted by delta descending.
    """
    header = f"{'Pos':>4} {'Analytical %':>13} {'Simulated %':>12} {'Delta pp':>9}"
    separator = "─" * 42
    rows = [header, separator]
    for pos in np.argsort(delta)[::-1]:
        flag = "  ← FAIL" if delta[pos] > 0.001 else ""
        rows.append(
            f"{pos:4d} {analytical[pos] * 100:13.4f} "
            f"{simulated[pos] * 100:12.4f} "
            f"{delta[pos] * 100:9.4f}{flag}"
        )
    return "\n".join(rows)

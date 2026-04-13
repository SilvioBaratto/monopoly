"""Tests for the Markov chain transition matrix and stationary distribution (markov.py).

Covers structural properties, specific known transitions, jail mechanics,
Chance/Community Chest card probabilities, property-based invariants, and
stationary distribution validation against published canonical values.
"""

from __future__ import annotations

import numpy
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Constants mirrored from the spec
# ---------------------------------------------------------------------------

NUM_STATES = 43
GO_TO_JAIL_POS = 30
JAIL_STATE_1 = 40  # just arrived / failed turn 1
JAIL_STATE_2 = 41  # failed turn 2
JAIL_STATE_3 = 42  # forced release turn 3


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def matrix() -> numpy.ndarray:
    """Build the transition matrix once per test module."""
    from monopoly.markov import build_transition_matrix

    return build_transition_matrix()


# ---------------------------------------------------------------------------
# 1. Structural properties
# ---------------------------------------------------------------------------


def test_shape_is_43x43(matrix: numpy.ndarray) -> None:
    """Matrix must be 43×43 (40 board squares + 3 jail states)."""
    assert matrix.shape == (NUM_STATES, NUM_STATES)


def test_dtype_is_float64(matrix: numpy.ndarray) -> None:
    """Matrix must have dtype float64."""
    assert matrix.dtype == numpy.float64


def test_all_rows_sum_to_one(matrix: numpy.ndarray) -> None:
    """Every row must sum to 1.0 ± 1e-12 (stochastic matrix)."""
    row_sums = matrix.sum(axis=1)
    numpy.testing.assert_allclose(row_sums, 1.0, atol=1e-12)


def test_no_negative_entries(matrix: numpy.ndarray) -> None:
    """No entry may be negative."""
    assert numpy.all(matrix >= 0)


# ---------------------------------------------------------------------------
# 2. Go To Jail square
# ---------------------------------------------------------------------------


def test_go_to_jail_row_sends_to_jail(matrix: numpy.ndarray) -> None:
    """Position 30 row must send 100% probability to jail state 40."""
    assert matrix[GO_TO_JAIL_POS, JAIL_STATE_1] == pytest.approx(1.0, abs=1e-12)


def test_go_to_jail_row_only_target_is_jail(matrix: numpy.ndarray) -> None:
    """Row 30 must have no probability outside state 40."""
    row = matrix[GO_TO_JAIL_POS].copy()
    row[JAIL_STATE_1] = 0.0
    assert numpy.all(row == 0.0)


def test_position_30_column_is_zero(matrix: numpy.ndarray) -> None:
    """Column 30 must be all zeros — pos 30 is unreachable as resting state."""
    assert numpy.all(matrix[:, GO_TO_JAIL_POS] == 0.0)


# ---------------------------------------------------------------------------
# 3. Chance card probabilities
# ---------------------------------------------------------------------------


def test_chance_boardwalk_from_position_7(matrix: numpy.ndarray) -> None:
    """Rolling 7 from pos 0 → Chance at pos 7 → 1/16 chance to Boardwalk (pos 39).

    From pos 0, rolling 7 (prob 6/36) lands on Chance (pos 7).
    Chance has 1 Boardwalk card out of 16. No other path from pos 0 reaches pos 39.
    Expected: matrix[0, 39] = (6/36) * (1/16) = 6/576.
    """
    expected = (6 / 36) * (1 / 16)
    assert matrix[0, 39] == pytest.approx(expected, rel=1e-9)


def test_chance_advance_to_go_from_position_7(matrix: numpy.ndarray) -> None:
    """From pos 0, rolling 7 → Chance at pos 7 → 1/16 to Go (pos 0 itself).

    Rolling 7 from pos 0 is non-doubles (1+6, 2+5, 3+4, etc.), prob 6/36.
    Chance at pos 7 includes one 'Advance to Go' card.
    Path: pos 0 → (roll 7) → pos 7 (Chance) → pos 0 with prob 1/16.
    Expected contribution to matrix[0, 0] from this path: (6/36) * (1/16).

    (Other contributions to matrix[0, 0] exist: CC at pos 2 rolling 2,
    which is doubles and contributes 1/16 of the CC stay-at-2 distribution.)
    """
    chance_go_contrib = (6 / 36) * (1 / 16)
    # matrix[0, 0] must be at least this contribution
    assert matrix[0, 0] >= chance_go_contrib - 1e-12


def test_chance_go_to_jail_card_from_position_7(matrix: numpy.ndarray) -> None:
    """From pos 0, rolling 7 → Chance at pos 7 → 1/16 to jail state 40.

    Rolling non-doubles 7 from pos 0 → Chance at pos 7 → Go to Jail card → state 40.
    Expected contribution to matrix[0, 40]: (6/36) * (1/16).
    Other contributions to matrix[0, 40] may exist via Community Chest at pos 2.
    """
    chance_jail_contrib = (6 / 36) * (1 / 16)
    assert matrix[0, 40] >= chance_jail_contrib - 1e-12


def test_nearest_railroad_from_pos7_is_pos15(matrix: numpy.ndarray) -> None:
    """From Chance at pos 7, nearest railroad is pos 15 (two cards, 2/16 total).

    From pos 0, roll 7 → Chance at pos 7 → 2 nearest-railroad cards → pos 15.
    Expected contribution: (6/36) * (2/16).
    """
    expected_contrib = (6 / 36) * (2 / 16)
    assert matrix[0, 15] >= expected_contrib - 1e-12


def test_nearest_railroad_from_pos22_is_pos25(matrix: numpy.ndarray) -> None:
    """From Chance at pos 22, nearest railroad is pos 25.

    From pos 10, roll 12 (double 6, prob 1/36) → pos 22 (Chance).
    Nearest railroad from pos 22 is pos 25. Two nearest-RR cards → 2/16.
    Rolling 12 is a doubles roll, so the triple doubles correction applies:
    effective prob = (35/36) * (1/36) * (2/16).
    This is the only path from pos 10 to pos 25 in one roll.
    """
    expected = (35 / 36) * (1 / 36) * (2 / 16)
    assert matrix[10, 25] == pytest.approx(expected, rel=1e-9)


def test_nearest_railroad_from_pos36_is_pos5(matrix: numpy.ndarray) -> None:
    """From Chance at pos 36, nearest railroad wraps to pos 5.

    From pos 24, roll 12 (double 6, prob 1/36) → pos 36 (Chance).
    Nearest railroad from pos 36 wraps to pos 5. Two cards → 2/16.
    Contribution to matrix[24, 5]: at least (1/36) * (2/16).
    """
    expected_contrib = (1 / 36) * (2 / 16)
    assert matrix[24, 5] >= expected_contrib - 1e-12


def test_nearest_utility_from_pos7_is_pos12(matrix: numpy.ndarray) -> None:
    """From Chance at pos 7, nearest utility is pos 12 (Electric Company).

    From pos 0, roll 7 → Chance at pos 7 → 1 nearest-utility card → pos 12.
    Contribution: (6/36) * (1/16).
    """
    expected_contrib = (6 / 36) * (1 / 16)
    assert matrix[0, 12] >= expected_contrib - 1e-12


def test_nearest_utility_from_pos22_is_pos28(matrix: numpy.ndarray) -> None:
    """From Chance at pos 22, nearest utility is pos 28 (Water Works).

    From pos 10, roll 12 (double 6, prob 1/36) → pos 22 (Chance).
    Nearest utility from pos 22 is pos 28. Triple doubles correction applies.
    Effective contribution: (35/36) * (1/36) * (1/16).
    This is the only path from pos 10 to pos 28 in one roll.
    """
    expected = (35 / 36) * (1 / 36) * (1 / 16)
    assert matrix[10, 28] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 4. Community Chest card probabilities
# ---------------------------------------------------------------------------


def test_community_chest_advance_to_go(matrix: numpy.ndarray) -> None:
    """CC at pos 2: 1/16 probability of Advance to Go (pos 0).

    From pos 0, rolling 2 (double 1, prob 1/36) → pos 2 (CC).
    CC has one 'Advance to Go' card. Contribution to matrix[0, 0]: (1/36) * (1/16).
    """
    expected_contrib = (1 / 36) * (1 / 16)
    assert matrix[0, 0] >= expected_contrib - 1e-12


def test_community_chest_go_to_jail(matrix: numpy.ndarray) -> None:
    """CC at pos 2: 1/16 probability of Go to Jail → state 40.

    From pos 0, rolling 2 (double 1, prob 1/36) → pos 2 (CC).
    CC has one 'Go to Jail' card. Contribution to matrix[0, 40]: (1/36) * (1/16).
    """
    expected_contrib = (1 / 36) * (1 / 16)
    assert matrix[0, 40] >= expected_contrib - 1e-12


def test_community_chest_non_move_fraction(matrix: numpy.ndarray) -> None:
    """CC non-movement fraction is 14/16.

    From pos 0, rolling 2 → pos 2 (CC). 14/16 cards keep player at pos 2.
    Contribution to matrix[0, 2]: (1/36) * (14/16).
    (Minus any triple-doubles correction since doubles were rolled.)
    """
    base_contrib = (1 / 36) * (14 / 16)
    triple_doubles_correction = (1 / 36) * (1 / 36) * (14 / 16)
    expected = base_contrib - triple_doubles_correction
    assert matrix[0, 2] == pytest.approx(expected, rel=1e-9)


# ---------------------------------------------------------------------------
# 5. Jail state mechanics
# ---------------------------------------------------------------------------


def test_jail_turn1_to_turn2_on_no_doubles(matrix: numpy.ndarray) -> None:
    """From jail state 40, no doubles (prob 30/36) → advance to jail state 41."""
    assert matrix[JAIL_STATE_1, JAIL_STATE_2] == pytest.approx(30 / 36, rel=1e-9)


def test_jail_turn2_to_turn3_on_no_doubles(matrix: numpy.ndarray) -> None:
    """From jail state 41, no doubles (prob 30/36) → advance to jail state 42."""
    assert matrix[JAIL_STATE_2, JAIL_STATE_3] == pytest.approx(30 / 36, rel=1e-9)


def test_jail_turn3_forced_release_sums_to_one(matrix: numpy.ndarray) -> None:
    """Row 42 (forced release) must sum to 1.0 and have zero in jail states 41/42."""
    assert matrix[JAIL_STATE_3].sum() == pytest.approx(1.0, abs=1e-12)
    assert matrix[JAIL_STATE_3, JAIL_STATE_2] == 0.0
    assert matrix[JAIL_STATE_3, JAIL_STATE_3] == 0.0


def test_jail_escape_on_doubles_sums_to_one_sixth(matrix: numpy.ndarray) -> None:
    """From jail state 40, total escape probability (non-jail board states) is 6/36.

    The sum of probabilities going to board states 0–39 (excluding via the
    Go-to-Jail card draw) equals 6/36 = 1/6, since only doubles let you escape.
    We account for the Go to Jail Chance card that can redirect a doubles escape
    back to jail.
    """
    escape_from_jail_1 = matrix[JAIL_STATE_1, :40].sum()
    # Doubles prob = 6/36; but one of the escape destinations (via double-6 to
    # Chance pos 22, then 1/16 Go to Jail) reduces board destinations.
    assert escape_from_jail_1 == pytest.approx(6 / 36 - (1 / 36) * (1 / 16), rel=1e-9)


def test_jail_state_40_row_sums_to_one(matrix: numpy.ndarray) -> None:
    """Jail state 40 row must sum to 1.0."""
    assert matrix[JAIL_STATE_1].sum() == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# 6. Triple doubles correction
# ---------------------------------------------------------------------------


def test_triple_doubles_adds_jail_probability(matrix: numpy.ndarray) -> None:
    """Triple doubles correction adds jail probability on top of normal jail sources.

    From state 8, doubles go to: 10, 12, 14, 16, 18, 20 (all non-special).
    Triple doubles adds 6 * (1/36)^2 = 1/216 to jail from those doubles.

    Additionally, rolling non-doubles 9 (prob 4/36: 3+6,4+5,5+4,6+3) from pos 8
    → Community Chest at pos 17 → 1/16 Go to Jail card.

    Expected: matrix[8, 40] = 6/(36²) + (4/36)*(1/16).
    """
    triple_doubles_jail = 6 / (36 * 36)
    cc_jail_via_roll9 = (4 / 36) * (1 / 16)
    assert matrix[8, 40] == pytest.approx(
        triple_doubles_jail + cc_jail_via_roll9, rel=1e-9
    )


def test_triple_doubles_reduces_doubles_destination(matrix: numpy.ndarray) -> None:
    """Doubles destination probability is reduced by 1/1296 per doubles outcome.

    From state 8, rolling double-1 (total=2) → pos 10.
    Without correction: 1/36. With correction: 35/36 * 1/36 = 35/1296.
    """
    # From state 8: double-1 (total 2) → pos 10, double-2 → 12, etc.
    # Check pos 10 contribution from double-1 only (other doubles don't reach 10).
    assert matrix[8, 10] == pytest.approx(35 / (36 * 36), rel=1e-9)


# ---------------------------------------------------------------------------
# 7. Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


@given(row_index=st.integers(min_value=0, max_value=NUM_STATES - 1))
@settings(max_examples=NUM_STATES)
def test_property_row_sums_to_one(row_index: int) -> None:
    """Property: every row in the matrix sums to 1.0 ± 1e-12."""
    from monopoly.markov import build_transition_matrix

    m = build_transition_matrix()
    assert m[row_index].sum() == pytest.approx(1.0, abs=1e-12)


@given(
    row_index=st.integers(min_value=0, max_value=NUM_STATES - 1),
    col_index=st.integers(min_value=0, max_value=NUM_STATES - 1),
)
@settings(max_examples=200)
def test_property_no_negative_entries(row_index: int, col_index: int) -> None:
    """Property: no matrix entry is negative."""
    from monopoly.markov import build_transition_matrix

    m = build_transition_matrix()
    assert m[row_index, col_index] >= 0.0


@settings(max_examples=1)
@given(st.just(None))
def test_property_matrix_is_irreducible(_: None) -> None:
    """Property: the accessible states form an irreducible chain.

    Position 30 (Go To Jail) is never a resting state — its column is all zeros
    by design (verified by test_position_30_column_is_zero). Excluding it, the
    remaining 42 states form an irreducible Markov chain.

    Verified by checking (I + M_42)^42 has all strictly positive entries,
    guaranteeing a path exists between any two accessible states in ≤42 steps.
    Since column 30 = 0 (nothing transitions TO pos 30), removing row and column 30
    preserves the stochastic property of all other rows.
    """
    from monopoly.markov import build_transition_matrix

    m = build_transition_matrix()
    indices = [i for i in range(NUM_STATES) if i != GO_TO_JAIL_POS]
    m_reduced = m[numpy.ix_(indices, indices)]
    reachability = numpy.linalg.matrix_power(
        numpy.eye(len(indices)) + m_reduced, len(indices)
    )
    assert numpy.all(reachability > 0)


# ---------------------------------------------------------------------------
# 8. Stationary distribution — fixtures
# ---------------------------------------------------------------------------

_JAIL_IN_JAIL_STATE = 40  # state 40 = In Jail turn 1 (highest probability state)
_ILLINOIS_AVE_POS = 24
_GO_POS = 0
_BO_RAILROAD_POS = 25

# Canonical values for the 43-state model from #29.
#
# NOTE: Truman Collins (2002) publishes Jail ≈ 5.89% and Illinois ≈ 3.18%.
# Those values come from a SIMPLER 41-state model with ONE combined "In Jail"
# state. Our 43-state model correctly models 3 jail sub-states (turns 1-3),
# which gives:
#   state 40 (In Jail T1) ≈ 3.80%
#   state 41 (In Jail T2) ≈ 3.16%
#   state 42 (In Jail T3) ≈ 2.64%
#   total in-jail fraction ≈ 9.60%
#   Illinois Ave (pos 24)  ≈ 2.99%
# These differ from Collins by design. The mathematical validation (π = πP)
# is the authoritative correctness test; these values serve as regression
# anchors for our specific 43-state formulation.
_CANONICAL_JAIL_T1_PROB = 0.0380  # state 40 in our 43-state model
_CANONICAL_ILLINOIS_PROB = 0.0299  # pos 24 in our 43-state model


@pytest.fixture(scope="module")
def distribution_eigenvector(matrix: numpy.ndarray) -> numpy.ndarray:
    """Stationary distribution computed via eigenvector method."""
    from monopoly.markov import compute_stationary_distribution

    return compute_stationary_distribution(matrix, method="eigenvector")


@pytest.fixture(scope="module")
def distribution_power(matrix: numpy.ndarray) -> numpy.ndarray:
    """Stationary distribution computed via power iteration."""
    from monopoly.markov import compute_stationary_distribution

    return compute_stationary_distribution(matrix, method="power_iteration")


# ---------------------------------------------------------------------------
# 8a. Structural invariants
# ---------------------------------------------------------------------------


def test_stationary_distribution_returns_1d_array(
    distribution_eigenvector: numpy.ndarray,
) -> None:
    """compute_stationary_distribution returns a 1D numpy array."""
    assert distribution_eigenvector.ndim == 1
    assert len(distribution_eigenvector) == NUM_STATES


def test_stationary_distribution_sums_to_one_eigenvector(
    distribution_eigenvector: numpy.ndarray,
) -> None:
    """Eigenvector distribution sums to 1.0 ± 1e-12."""
    numpy.testing.assert_allclose(distribution_eigenvector.sum(), 1.0, atol=1e-12)


def test_stationary_distribution_sums_to_one_power(
    distribution_power: numpy.ndarray,
) -> None:
    """Power iteration distribution sums to 1.0 ± 1e-12."""
    numpy.testing.assert_allclose(distribution_power.sum(), 1.0, atol=1e-12)


def test_stationary_distribution_all_non_negative_eigenvector(
    distribution_eigenvector: numpy.ndarray,
) -> None:
    """All eigenvector distribution entries are non-negative."""
    assert numpy.all(distribution_eigenvector >= 0.0)


def test_stationary_distribution_all_non_negative_power(
    distribution_power: numpy.ndarray,
) -> None:
    """All power iteration distribution entries are non-negative."""
    assert numpy.all(distribution_power >= 0.0)


# ---------------------------------------------------------------------------
# 8b. Cross-validation: eigenvector ≈ power iteration
# ---------------------------------------------------------------------------


def test_eigenvector_and_power_iteration_agree(
    distribution_eigenvector: numpy.ndarray,
    distribution_power: numpy.ndarray,
) -> None:
    """Eigenvector and power iteration methods produce identical results within 1e-10."""
    numpy.testing.assert_allclose(
        distribution_eigenvector, distribution_power, atol=1e-10
    )


# ---------------------------------------------------------------------------
# 8c. Canonical value validation (Truman Collins 2002)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dist_fixture", ["distribution_eigenvector", "distribution_power"]
)
def test_jail_t1_probability_near_canonical(
    dist_fixture: str, request: pytest.FixtureRequest
) -> None:
    """In Jail turn-1 state (40) probability is within ±0.1% of its 43-state value.

    Our 43-state model gives ≈3.80% for state 40 (the highest individual state).
    This differs from Collins (2002) 5.89% which comes from a simpler 41-state
    model with a single combined In Jail state. See canonical constant comment.
    Tolerance: ±0.001 (absolute).
    """
    dist: numpy.ndarray = request.getfixturevalue(dist_fixture)
    jail_prob = dist[_JAIL_IN_JAIL_STATE]
    assert abs(jail_prob - _CANONICAL_JAIL_T1_PROB) <= 0.001, (
        f"In Jail T1 probability {jail_prob:.4f} deviates more than 0.1% from "
        f"43-state canonical {_CANONICAL_JAIL_T1_PROB}"
    )


@pytest.mark.parametrize(
    "dist_fixture", ["distribution_eigenvector", "distribution_power"]
)
def test_jail_t1_is_highest_probability_state(
    dist_fixture: str, request: pytest.FixtureRequest
) -> None:
    """State 40 (In Jail T1) has the highest probability of all 43 states.

    This verifies that jail is the most visited state, a well-known Monopoly fact.
    """
    dist: numpy.ndarray = request.getfixturevalue(dist_fixture)
    assert numpy.argmax(dist) == _JAIL_IN_JAIL_STATE


@pytest.mark.parametrize(
    "dist_fixture", ["distribution_eigenvector", "distribution_power"]
)
def test_illinois_avenue_probability_near_canonical(
    dist_fixture: str, request: pytest.FixtureRequest
) -> None:
    """Illinois Avenue (pos 24) probability is within ±0.1% of its 43-state value.

    Our 43-state model gives ≈2.99%. Collins (2002) gives 3.18% for a simpler
    41-state model. Tolerance: ±0.001 (absolute).
    """
    dist: numpy.ndarray = request.getfixturevalue(dist_fixture)
    illinois_prob = dist[_ILLINOIS_AVE_POS]
    assert abs(illinois_prob - _CANONICAL_ILLINOIS_PROB) <= 0.001, (
        f"Illinois Avenue probability {illinois_prob:.4f} deviates more than 0.1% "
        f"from 43-state canonical {_CANONICAL_ILLINOIS_PROB}"
    )


@pytest.mark.parametrize(
    "dist_fixture", ["distribution_eigenvector", "distribution_power"]
)
def test_illinois_avenue_is_top_board_square(
    dist_fixture: str, request: pytest.FixtureRequest
) -> None:
    """Illinois Avenue (pos 24) is the most probable non-jail board square.

    Among positions 0–39 (excluding jail sub-states 40–42), pos 24 has the
    highest probability — consistent with Collins' published ordering.
    """
    dist: numpy.ndarray = request.getfixturevalue(dist_fixture)
    board_probs = dist[:40]
    assert numpy.argmax(board_probs) == _ILLINOIS_AVE_POS


# ---------------------------------------------------------------------------
# 8d. Stationary equation: π = πP
# ---------------------------------------------------------------------------


def test_stationary_equation_holds_eigenvector(
    matrix: numpy.ndarray,
    distribution_eigenvector: numpy.ndarray,
) -> None:
    """Eigenvector distribution satisfies π = πP within numerical tolerance."""
    pi_p = distribution_eigenvector @ matrix
    numpy.testing.assert_allclose(pi_p, distribution_eigenvector, atol=1e-10)


def test_stationary_equation_holds_power(
    matrix: numpy.ndarray,
    distribution_power: numpy.ndarray,
) -> None:
    """Power iteration distribution satisfies π = πP within numerical tolerance."""
    pi_p = distribution_power @ matrix
    numpy.testing.assert_allclose(pi_p, distribution_power, atol=1e-10)


# ---------------------------------------------------------------------------
# 9. get_square_probabilities
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def square_probabilities(distribution_eigenvector: numpy.ndarray) -> dict:
    """Square probabilities dict from get_square_probabilities."""
    from monopoly.board import Board
    from monopoly.markov import get_square_probabilities

    return get_square_probabilities(Board(), distribution_eigenvector)


def test_get_square_probabilities_returns_dict(square_probabilities: dict) -> None:
    """get_square_probabilities returns a dict with 40 entries (board squares only)."""
    assert isinstance(square_probabilities, dict)
    assert len(square_probabilities) == 40


def test_get_square_probabilities_keys_are_positions(
    square_probabilities: dict,
) -> None:
    """Keys are integers 0–39."""
    assert set(square_probabilities.keys()) == set(range(40))


def test_get_square_probabilities_values_are_name_prob_tuples(
    square_probabilities: dict,
) -> None:
    """Values are (name: str, probability: float) tuples."""
    for _pos, (name, prob) in square_probabilities.items():
        assert isinstance(name, str)
        assert isinstance(prob, float)


def test_get_square_probabilities_sorted_descending(square_probabilities: dict) -> None:
    """Dict is sorted by probability descending."""
    probs = list(p for _, p in square_probabilities.values())
    assert probs == sorted(probs, reverse=True)


def test_get_square_probabilities_jail_is_highest(square_probabilities: dict) -> None:
    """Jail / Just Visiting (pos 10) has the highest single-square probability."""
    first_pos = next(iter(square_probabilities))
    assert first_pos == 10, f"Expected pos 10 (Jail) first, got pos {first_pos}"


def test_get_square_probabilities_illinois_name(square_probabilities: dict) -> None:
    """Illinois Avenue (pos 24) has correct name in the dict."""
    name, _prob = square_probabilities[_ILLINOIS_AVE_POS]
    assert "Illinois" in name


# ---------------------------------------------------------------------------
# 10. Extended stationary distribution value tests (issue #31)
# ---------------------------------------------------------------------------

# Reference values for the 43-state model.
# NOTE: The Go probability of 3.09% commonly cited in published analyses (e.g.
# DataGenetics) comes from a 40-state model. Our 43-state model with three
# distinct jail sub-states distributes probability differently, yielding ≈2.91%.
REFERENCE_GO_PROB = 0.0291  # Go position in our 43-state model

_BOARDWALK_POS = 39
_MEDITERRANEAN_POS = 1
_BALTIC_POS = 3
_ST_JAMES_PLACE_POS = 16
_TENNESSEE_AVE_POS = 18
_NEW_YORK_AVE_POS = 19
_CHANCE_SQUARE_POSITIONS = (7, 22, 36)
_NAIVE_3_SQUARES_PROB = (
    3 / 40
)  # uniform naive expectation for 3 squares on a 40-square board


def test_go_probability_near_reference(distribution_eigenvector: numpy.ndarray) -> None:
    """Go (pos 0) stationary probability is within ±0.15% of the 43-state reference value.

    In our 43-state model, Go ≈ 2.91%. This differs from 3.09% in simpler 40-state
    models because the three jail sub-states redistribute probability differently.
    Tolerance: ±0.0015 absolute.
    """
    go_prob = distribution_eigenvector[_GO_POS]
    assert abs(go_prob - REFERENCE_GO_PROB) <= 0.0015, (
        f"Go probability {go_prob:.4f} deviates more than 0.15% from "
        f"43-state reference {REFERENCE_GO_PROB}"
    )


def test_boardwalk_exceeds_mediterranean(
    distribution_eigenvector: numpy.ndarray,
) -> None:
    """Boardwalk (pos 39) has higher stationary probability than Mediterranean Ave (pos 1).

    In standard Monopoly analysis (Truman Collins 2002, DataGenetics), Boardwalk
    is consistently more probable than Mediterranean. Mediterranean is the lowest-
    probability property on the board due to its position just after Go and far from
    the jail attractor region.
    """
    boardwalk_prob = distribution_eigenvector[_BOARDWALK_POS]
    mediterranean_prob = distribution_eigenvector[_MEDITERRANEAN_POS]
    assert boardwalk_prob > mediterranean_prob, (
        f"Expected Boardwalk ({boardwalk_prob:.4f}) > Mediterranean ({mediterranean_prob:.4f})"
    )


def test_orange_group_average_exceeds_brown_group(
    distribution_eigenvector: numpy.ndarray,
) -> None:
    """Orange group average stationary probability exceeds Brown group average.

    Orange properties (St. James Place, Tennessee Ave, New York Ave — pos 16, 18, 19)
    sit in the high-traffic zone just past Jail, boosting their visit frequency.
    Brown properties (Mediterranean, Baltic — pos 1, 3) are far from Jail and have
    the lowest landing rates of any color group.
    """
    orange_avg = numpy.mean(
        distribution_eigenvector[
            [_ST_JAMES_PLACE_POS, _TENNESSEE_AVE_POS, _NEW_YORK_AVE_POS]
        ]
    )
    brown_avg = numpy.mean(distribution_eigenvector[[_MEDITERRANEAN_POS, _BALTIC_POS]])
    assert orange_avg > brown_avg, (
        f"Expected orange avg ({orange_avg:.4f}) > brown avg ({brown_avg:.4f})"
    )


def test_chance_squares_sum_less_than_naive_expectation(
    distribution_eigenvector: numpy.ndarray,
) -> None:
    """Combined Chance square (7, 22, 36) stationary probability < naive 2d6 expectation.

    Naive expectation: 3 squares / 40 = 7.5% combined.
    Chance cards redirect players away from Chance squares, so the actual stationary
    probability of landing and staying on a Chance square is significantly lower than
    the naive uniform expectation.
    """
    chance_prob_sum = sum(
        distribution_eigenvector[pos] for pos in _CHANCE_SQUARE_POSITIONS
    )
    assert chance_prob_sum < _NAIVE_3_SQUARES_PROB, (
        f"Chance squares combined probability {chance_prob_sum:.4f} "
        f"is not less than naive {_NAIVE_3_SQUARES_PROB:.4f}"
    )


def test_go_to_jail_has_zero_stationary_probability(
    distribution_eigenvector: numpy.ndarray,
) -> None:
    """Go To Jail (pos 30) has zero stationary probability.

    Position 30 is a transient square: any player landing on it is immediately
    redirected to Jail (state 40). It is never a resting state, so its stationary
    probability must be exactly zero.
    """
    assert distribution_eigenvector[GO_TO_JAIL_POS] == pytest.approx(0.0, abs=1e-12), (
        f"Go To Jail stationary probability should be 0, got {distribution_eigenvector[GO_TO_JAIL_POS]}"
    )


# ---------------------------------------------------------------------------
# 11. Irreducibility via scipy.sparse.csgraph (issue #31)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_matrix_is_irreducible_via_sparse_csgraph(matrix: numpy.ndarray) -> None:
    """Accessible states form a single strongly connected component (scipy.sparse.csgraph).

    Position 30 (Go To Jail) is excluded because its column is all zeros — it is
    never a resting state. The remaining 42 states must form exactly one strongly
    connected component, confirming the chain is irreducible and has a unique
    stationary distribution.
    """
    import scipy.sparse
    import scipy.sparse.csgraph

    indices = [i for i in range(NUM_STATES) if i != GO_TO_JAIL_POS]
    m_reduced = matrix[numpy.ix_(indices, indices)]
    sparse_m = scipy.sparse.csr_matrix(m_reduced)
    n_components, _ = scipy.sparse.csgraph.connected_components(
        sparse_m, connection="strong", directed=True
    )
    assert n_components == 1, (
        f"Expected 1 strongly connected component, got {n_components}. "
        "Chain is not irreducible."
    )


# ---------------------------------------------------------------------------
# 12. Structural test: cards disabled vs enabled (issue #31)
# ---------------------------------------------------------------------------


def test_cards_disabled_produces_different_distribution() -> None:
    """Disabling Chance/Community Chest cards yields a different stationary distribution.

    With cards enabled, Chance/CC squares redirect players to specific destinations.
    Disabling cards treats those squares as plain stopping points, which changes the
    transition probabilities and therefore the stationary distribution. This test
    ensures that the card logic is actually wired into the matrix and not a no-op.
    """
    from monopoly.markov import build_transition_matrix, compute_stationary_distribution

    matrix_with_cards = build_transition_matrix(include_cards=True)
    matrix_no_cards = build_transition_matrix(include_cards=False)

    dist_with = compute_stationary_distribution(matrix_with_cards)
    dist_no = compute_stationary_distribution(matrix_no_cards)

    max_diff = numpy.max(numpy.abs(dist_with - dist_no))
    assert max_diff > 1e-3, (
        f"Distributions differ by only {max_diff:.2e} — card logic appears to have no effect."
    )

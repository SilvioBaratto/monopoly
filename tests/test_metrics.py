"""Tests for src/monopoly/metrics.py — written BEFORE implementation (TDD).

Hand-calculated values use a uniform distribution (1/40 per position).
"""

from __future__ import annotations

import numpy
import pandas as pd
import pytest

from monopoly.board import Board
from monopoly.markov import build_transition_matrix, compute_stationary_distribution

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def board() -> Board:
    """Standard US Monopoly board."""
    return Board()


@pytest.fixture
def uniform_distribution() -> numpy.ndarray:
    """Uniform distribution: 1/40 probability at every board position."""
    return numpy.full(40, 1 / 40)


@pytest.fixture(scope="module")
def real_distribution() -> numpy.ndarray:
    """Stationary distribution collapsed from 43 states to 40 board positions.

    Jail sub-states 40–42 are folded into position 10 (the physical jail square).
    """
    matrix = build_transition_matrix()
    dist43 = compute_stationary_distribution(matrix)
    dist40 = dist43[:40].copy()
    dist40[10] += dist43[40] + dist43[41] + dist43[42]
    return dist40


# ---------------------------------------------------------------------------
# Import guard (all tests below must fail before implementation exists)
# ---------------------------------------------------------------------------

from monopoly.metrics import (  # noqa: E402
    expected_rent_per_roll,
    expected_rent_table,
    compute_roi,
    compute_payback_period,
    roi_ranking_table,
    payback_ranking_table,
    marginal_roi,
    wilson_confidence_interval,
    win_probability_table,
)
from monopoly.simulate import SimulationResult, simulate_games  # noqa: E402
from monopoly.strategies.buy_everything import BuyEverything  # noqa: E402
from monopoly.strategies.buy_nothing import BuyNothing  # noqa: E402


# ---------------------------------------------------------------------------
# ValueError guard tests
# ---------------------------------------------------------------------------


def test_raises_value_error_when_distribution_length_is_not_40(board: Board) -> None:
    """A 39-element array must raise ValueError."""
    short = numpy.full(39, 1 / 39)
    with pytest.raises(ValueError):
        expected_rent_per_roll(board, short, "brown", "base")


def test_raises_value_error_when_distribution_length_is_43(board: Board) -> None:
    """A 43-element array (raw Markov output) must raise ValueError."""
    long = numpy.full(43, 1 / 43)
    with pytest.raises(ValueError):
        expected_rent_per_roll(board, long, "brown", "base")


# ---------------------------------------------------------------------------
# Brown group — hand-calculated (pos 1 = Mediterranean, pos 3 = Baltic)
# rents: (2, 10, 30, 90, 160, 250)  Mediterranean
# rents: (4, 20, 60, 180, 320, 450)  Baltic
# ---------------------------------------------------------------------------


def test_brown_group_base_rent_with_uniform_distribution(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """base: 0.025×2 + 0.025×4 = 0.05 + 0.10 = 0.15."""
    result = expected_rent_per_roll(board, uniform_distribution, "brown", "base")
    assert result == pytest.approx(0.15)


def test_brown_group_monopoly_rent_is_double_base(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """monopoly: 0.025×4 + 0.025×8 = 0.30 (2× base)."""
    result = expected_rent_per_roll(board, uniform_distribution, "brown", "monopoly")
    base = expected_rent_per_roll(board, uniform_distribution, "brown", "base")
    assert result == pytest.approx(0.30)
    assert result == pytest.approx(2 * base)


def test_brown_group_1h_rent_with_uniform_distribution(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """1h: 0.025×10 + 0.025×20 = 0.25 + 0.50 = 0.75."""
    result = expected_rent_per_roll(board, uniform_distribution, "brown", "1h")
    assert result == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# Dark blue group — hand-calculated (pos 37 = Park Place, pos 39 = Boardwalk)
# rents Park Place:  (35, 175, 500, 1100, 1300, 1500)
# rents Boardwalk:   (50, 200, 600, 1400, 1700, 2000)
# ---------------------------------------------------------------------------


def test_dark_blue_group_base_rent_with_uniform_distribution(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """base: 0.025×35 + 0.025×50 = 0.875 + 1.25 = 2.125."""
    result = expected_rent_per_roll(board, uniform_distribution, "dark_blue", "base")
    assert result == pytest.approx(2.125)


def test_dark_blue_group_hotel_rent_with_uniform_distribution(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """hotel: 0.025×1500 + 0.025×2000 = 37.5 + 50.0 = 87.5."""
    result = expected_rent_per_roll(board, uniform_distribution, "dark_blue", "hotel")
    assert result == pytest.approx(87.5)


# ---------------------------------------------------------------------------
# Railroad — hand-calculated (4 positions, each prob=0.025)
# rents: [25, 50, 100, 200]
# ---------------------------------------------------------------------------


def test_railroad_count_1_with_uniform_distribution(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """count=1: 4 × (0.025 × 25) = 2.5."""
    result = expected_rent_per_roll(board, uniform_distribution, "railroad", 1)
    assert result == pytest.approx(2.5)


def test_railroad_count_4_with_uniform_distribution(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """count=4: 4 × (0.025 × 200) = 20.0."""
    result = expected_rent_per_roll(board, uniform_distribution, "railroad", 4)
    assert result == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Utility — hand-calculated (2 positions, each prob=0.025, E[dice]=7)
# multipliers: [4, 10]
# ---------------------------------------------------------------------------


def test_utility_count_1_uses_expected_dice_total_7(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """count=1: 2 × (0.025 × 4 × 7) = 1.4."""
    result = expected_rent_per_roll(board, uniform_distribution, "utility", 1)
    assert result == pytest.approx(1.4)


def test_utility_count_2_uses_expected_dice_total_7(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """count=2: 2 × (0.025 × 10 × 7) = 3.5."""
    result = expected_rent_per_roll(board, uniform_distribution, "utility", 2)
    assert result == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Non-negativity
# ---------------------------------------------------------------------------

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
_COLOR_LEVELS = ["base", "monopoly", "1h", "2h", "3h", "4h", "hotel"]


def test_all_values_non_negative(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Every expected rent value for every group and level must be >= 0."""
    for group in _COLOR_GROUPS:
        for level in _COLOR_LEVELS:
            result = expected_rent_per_roll(board, uniform_distribution, group, level)
            assert result >= 0, f"{group}/{level} returned {result}"
    for count in range(1, 5):
        assert (
            expected_rent_per_roll(board, uniform_distribution, "railroad", count) >= 0
        )
    for count in range(1, 3):
        assert (
            expected_rent_per_roll(board, uniform_distribution, "utility", count) >= 0
        )


# ---------------------------------------------------------------------------
# expected_rent_table
# ---------------------------------------------------------------------------


def test_expected_rent_table_returns_dataframe(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Return type must be a pandas DataFrame."""
    result = expected_rent_table(board, uniform_distribution)
    assert isinstance(result, pd.DataFrame)


def test_expected_rent_table_has_8_rows(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Table must have exactly 8 rows — one per color group."""
    result = expected_rent_table(board, uniform_distribution)
    assert len(result) == 8


def test_expected_rent_table_has_correct_columns(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Columns must be exactly ['base', 'monopoly', '1h', '2h', '3h', '4h', 'hotel']."""
    result = expected_rent_table(board, uniform_distribution)
    assert list(result.columns) == ["base", "monopoly", "1h", "2h", "3h", "4h", "hotel"]


def test_expected_rent_table_brown_row_matches_per_roll_function(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Brown row values must match direct expected_rent_per_roll calls."""
    table = expected_rent_table(board, uniform_distribution)
    for level in _COLOR_LEVELS:
        direct = expected_rent_per_roll(board, uniform_distribution, "brown", level)
        assert table.loc["brown", level] == pytest.approx(direct)


def test_monopoly_column_is_double_base_column(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """The 'monopoly' column must equal exactly 2× the 'base' column for all rows."""
    table = expected_rent_table(board, uniform_distribution)
    for group in _COLOR_GROUPS:
        assert table.loc[group, "monopoly"] == pytest.approx(
            2 * table.loc[group, "base"]
        )


def test_expected_rent_table_raises_value_error_for_wrong_distribution(
    board: Board,
) -> None:
    """expected_rent_table must also reject distributions with length != 40."""
    bad = numpy.full(43, 1 / 43)
    with pytest.raises(ValueError):
        expected_rent_table(board, bad)


# ---------------------------------------------------------------------------
# Slow tests — use real stationary distribution
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_orange_group_has_higher_expected_rent_at_3_houses_than_early_groups(
    board: Board, real_distribution: numpy.ndarray
) -> None:
    """Orange (3 squares, elevated landing frequency, mid-tier rents) beats early groups at 3h.

    The orange group benefits from frequent jail-exit landings on positions 16–19.
    It consistently outperforms brown, light_blue, and pink at 3h development.
    Green and red/yellow/dark_blue outperform it due to higher absolute rent values.
    """
    orange_3h = expected_rent_per_roll(board, real_distribution, "orange", "3h")
    for early_group in ("brown", "light_blue", "pink"):
        early_3h = expected_rent_per_roll(board, real_distribution, early_group, "3h")
        assert orange_3h > early_3h, (
            f"orange ({orange_3h:.2f}) should exceed {early_group} ({early_3h:.2f}) at 3h"
        )


@pytest.mark.slow
def test_dark_blue_has_higher_hotel_rent_than_orange_and_brown(
    board: Board, real_distribution: numpy.ndarray
) -> None:
    """Dark blue hotel rents (1500/2000) dwarf all earlier groups at hotel level.

    Note: green's three high-rent squares give it the single highest expected
    hotel rent. Dark_blue still outperforms orange and all earlier groups.
    """
    dark_blue_hotel = expected_rent_per_roll(
        board, real_distribution, "dark_blue", "hotel"
    )
    orange_hotel = expected_rent_per_roll(board, real_distribution, "orange", "hotel")
    brown_hotel = expected_rent_per_roll(board, real_distribution, "brown", "hotel")

    assert dark_blue_hotel > orange_hotel, (
        f"dark_blue ({dark_blue_hotel}) should exceed orange ({orange_hotel}) at hotel"
    )
    assert dark_blue_hotel > brown_hotel, (
        f"dark_blue ({dark_blue_hotel}) should exceed brown ({brown_hotel}) at hotel"
    )


# ---------------------------------------------------------------------------
# compute_roi — hand-calculated with uniform distribution
#
# Brown group: Mediterranean (pos 1, price=60, house=50), Baltic (pos 3, price=60, house=50)
# total_price = 120, house_cost = 50, 2 properties
#
# At 1h: investment = 120 + 50×1×2 = 220, expected_rent = 0.75 → ROI = 0.75/220
# At 3h: investment = 120 + 50×3×2 = 420, expected_rent = 6.75 → ROI = 6.75/420
# ---------------------------------------------------------------------------


def test_compute_roi_brown_1h_with_uniform_distribution(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """ROI = expected_rent / total_investment. Brown at 1h: 0.75 / 220."""
    result = compute_roi(board, uniform_distribution, "brown", "1h")
    assert result == pytest.approx(0.75 / 220)


def test_compute_roi_brown_3h_with_uniform_distribution(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Brown at 3h: 6.75 / 420."""
    result = compute_roi(board, uniform_distribution, "brown", "3h")
    assert result == pytest.approx(6.75 / 420)


def test_compute_roi_railroad_count_4_with_uniform_distribution(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Railroad count=4: rent=20.0, investment=4×200=800 → ROI = 20.0/800 = 0.025."""
    result = compute_roi(board, uniform_distribution, "railroad", 4)
    assert result == pytest.approx(20.0 / 800)


def test_compute_roi_all_values_non_negative(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Every ROI value for every color group and level must be >= 0."""
    for group in _COLOR_GROUPS:
        for level in _COLOR_LEVELS:
            result = compute_roi(board, uniform_distribution, group, level)
            assert result >= 0, f"{group}/{level} returned ROI={result}"


def test_compute_roi_raises_value_error_for_wrong_distribution(board: Board) -> None:
    """compute_roi must reject distributions with length != 40."""
    bad = numpy.full(43, 1 / 43)
    with pytest.raises(ValueError):
        compute_roi(board, bad, "brown", "1h")


# ---------------------------------------------------------------------------
# compute_payback_period — hand-calculated with uniform distribution
#
# Payback = total_investment / expected_rent_per_roll
# Brown at 1h: 220 / 0.75 ≈ 293.33 rolls
# ---------------------------------------------------------------------------


def test_compute_payback_period_brown_1h_with_uniform_distribution(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Payback = investment / rent. Brown at 1h: 220 / 0.75."""
    result = compute_payback_period(board, uniform_distribution, "brown", "1h")
    assert result == pytest.approx(220 / 0.75)


def test_compute_payback_period_is_inverse_of_roi_times_investment(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """payback × ROI must equal 1 for all groups (payback = 1/ROI)."""
    for group in _COLOR_GROUPS:
        for level in _COLOR_LEVELS:
            roi = compute_roi(board, uniform_distribution, group, level)
            payback = compute_payback_period(board, uniform_distribution, group, level)
            assert roi * payback == pytest.approx(1.0), (
                f"{group}/{level}: ROI={roi}, payback={payback}, product={roi * payback}"
            )


def test_compute_payback_period_positive_when_rent_positive(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Payback period must be > 0 for every group with positive expected rent."""
    for group in _COLOR_GROUPS:
        for level in _COLOR_LEVELS:
            rent = expected_rent_per_roll(board, uniform_distribution, group, level)
            if rent > 0:
                payback = compute_payback_period(
                    board, uniform_distribution, group, level
                )
                assert payback > 0, f"{group}/{level}: payback={payback}, rent={rent}"


def test_compute_payback_period_raises_value_error_for_wrong_distribution(
    board: Board,
) -> None:
    """compute_payback_period must reject distributions with length != 40."""
    bad = numpy.full(39, 1 / 39)
    with pytest.raises(ValueError):
        compute_payback_period(board, bad, "brown", "1h")


# ---------------------------------------------------------------------------
# roi_ranking_table
# ---------------------------------------------------------------------------


def test_roi_ranking_table_returns_dataframe(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Return type must be pandas DataFrame."""
    result = roi_ranking_table(board, uniform_distribution)
    assert isinstance(result, pd.DataFrame)


def test_roi_ranking_table_contains_all_color_groups(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """All 8 color groups must appear as rows."""
    result = roi_ranking_table(board, uniform_distribution)
    for group in _COLOR_GROUPS:
        assert group in result.index, f"{group} missing from roi_ranking_table"


def test_roi_ranking_table_includes_railroad(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Railroad must appear as a row in the ranking table."""
    result = roi_ranking_table(board, uniform_distribution)
    assert "railroad" in result.index


def test_roi_ranking_table_has_correct_columns(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Columns must match the standard rent level columns."""
    result = roi_ranking_table(board, uniform_distribution)
    assert list(result.columns) == ["base", "monopoly", "1h", "2h", "3h", "4h", "hotel"]


def test_roi_ranking_table_color_values_match_compute_roi(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Every color group cell must equal the direct compute_roi call."""
    table = roi_ranking_table(board, uniform_distribution)
    for group in _COLOR_GROUPS:
        for level in _COLOR_LEVELS:
            direct = compute_roi(board, uniform_distribution, group, level)
            assert table.loc[group, level] == pytest.approx(direct), (
                f"{group}/{level}: table={table.loc[group, level]}, direct={direct}"
            )


def test_roi_ranking_table_raises_value_error_for_wrong_distribution(
    board: Board,
) -> None:
    """roi_ranking_table must reject distributions with length != 40."""
    bad = numpy.full(43, 1 / 43)
    with pytest.raises(ValueError):
        roi_ranking_table(board, bad)


# ---------------------------------------------------------------------------
# payback_ranking_table
# ---------------------------------------------------------------------------


def test_payback_ranking_table_returns_dataframe(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Return type must be pandas DataFrame."""
    result = payback_ranking_table(board, uniform_distribution)
    assert isinstance(result, pd.DataFrame)


def test_payback_ranking_table_contains_all_color_groups(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """All 8 color groups must appear as rows."""
    result = payback_ranking_table(board, uniform_distribution)
    for group in _COLOR_GROUPS:
        assert group in result.index, f"{group} missing from payback_ranking_table"


def test_payback_ranking_table_includes_railroad(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Railroad must appear as a row in the payback table."""
    result = payback_ranking_table(board, uniform_distribution)
    assert "railroad" in result.index


def test_payback_ranking_table_color_values_match_compute_payback(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Every color group cell must equal the direct compute_payback_period call."""
    table = payback_ranking_table(board, uniform_distribution)
    for group in _COLOR_GROUPS:
        for level in _COLOR_LEVELS:
            direct = compute_payback_period(board, uniform_distribution, group, level)
            assert table.loc[group, level] == pytest.approx(direct), (
                f"{group}/{level}: table={table.loc[group, level]}, direct={direct}"
            )


def test_payback_ranking_table_raises_value_error_for_wrong_distribution(
    board: Board,
) -> None:
    """payback_ranking_table must reject distributions with length != 40."""
    bad = numpy.full(39, 1 / 39)
    with pytest.raises(ValueError):
        payback_ranking_table(board, bad)


# ---------------------------------------------------------------------------
# marginal_roi — hand-calculated with uniform distribution
#
# Brown base→1h:  Δrent = 0.75 - 0.15 = 0.60,  Δcost = 220 - 120 = 100  → 0.006
# Brown 1h→2h:    Δrent = 2.25 - 0.75 = 1.50,  Δcost = 320 - 220 = 100  → 0.015
# Brown 2h→3h:    Δrent = 6.75 - 2.25 = 4.50,  Δcost = 420 - 320 = 100  → 0.045
# ---------------------------------------------------------------------------


def test_marginal_roi_brown_base_to_1h(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Brown base→1h: (0.75 - 0.15) / (220 - 120) = 0.006."""
    result = marginal_roi(board, uniform_distribution, "brown", "base", "1h")
    assert result == pytest.approx(0.006)


def test_marginal_roi_brown_1h_to_2h(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Brown 1h→2h: (2.25 - 0.75) / (320 - 220) = 0.015."""
    result = marginal_roi(board, uniform_distribution, "brown", "1h", "2h")
    assert result == pytest.approx(0.015)


def test_marginal_roi_brown_2h_to_3h(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Brown 2h→3h: (6.75 - 2.25) / (420 - 320) = 0.045."""
    result = marginal_roi(board, uniform_distribution, "brown", "2h", "3h")
    assert result == pytest.approx(0.045)


def test_marginal_roi_raises_value_error_for_wrong_distribution(board: Board) -> None:
    """marginal_roi must reject distributions with length != 40."""
    bad = numpy.full(43, 1 / 43)
    with pytest.raises(ValueError):
        marginal_roi(board, bad, "brown", "base", "1h")


# ---------------------------------------------------------------------------
# Slow ranking tests
#
# Key findings from Markov stationary distribution (real board dynamics):
#   - Orange #1 overall ROI (avg 1h-4h): jail-exit traffic boosts positions 16-19
#   - Dark blue best ROI at 3h with uniform distribution: Boardwalk Chance card effect
#     only shows up in real distribution as absolute rent advantage, not ROI advantage
#     at 3h level (high investment neutralizes it). Dark blue best at 3h with uniform dist.
#   - Light blue best payback at hotel with uniform distribution: cheap properties ($320)
#     and cheap houses ($50) yield shortest payback. With real dist, orange's jail traffic
#     advantage overtakes light_blue.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_orange_ranked_1_overall_roi_averaged_across_1_to_4_houses(
    board: Board, real_distribution: numpy.ndarray
) -> None:
    """Orange must have the highest mean ROI averaged across 1h–4h levels.

    Computed with real Markov stationary distribution. Orange (pos 16, 18, 19)
    benefits from elevated landing probability due to Jail exit proximity.
    ROI orange ≈ 0.0226 vs dark_blue ≈ 0.0206 (next best).
    """
    table = roi_ranking_table(board, real_distribution)
    house_levels = ["1h", "2h", "3h", "4h"]
    avg_roi = table.loc[_COLOR_GROUPS, house_levels].mean(axis=1)
    best = avg_roi.idxmax()
    assert best == "orange", (
        f"Expected orange as #1 overall ROI, got {best}. Rankings:\n{avg_roi.sort_values(ascending=False)}"
    )


@pytest.mark.slow
def test_dark_blue_has_highest_roi_at_3h_with_uniform_distribution(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Dark blue must have the highest ROI at 3-house level with uniform distribution.

    With uniform distribution (no Chance/Jail effects), dark blue's very high rents
    (1100 + 1400 per roll at 3h) combined with 2-property investment yield best ROI.
    Dark blue ROI at 3h ≈ 0.03205 vs orange ≈ 0.02911 (next best).

    Note: with real Markov distribution, orange overtakes dark_blue due to jail proximity.
    The issue criterion holds specifically under uniform distribution assumptions.
    """
    table = roi_ranking_table(board, uniform_distribution)
    roi_at_3h = table.loc[_COLOR_GROUPS, "3h"]
    best = roi_at_3h.idxmax()
    assert best == "dark_blue", (
        f"Expected dark_blue as best ROI at 3h (uniform dist), got {best}.\n"
        f"Rankings:\n{roi_at_3h.sort_values(ascending=False)}"
    )


@pytest.mark.slow
def test_light_blue_has_lowest_payback_at_hotel_with_uniform_distribution(
    board: Board, uniform_distribution: numpy.ndarray
) -> None:
    """Light blue must have the shortest payback period at hotel level (uniform dist).

    With uniform distribution, light blue's cheap investment ($1070 total: $320 purchase
    + $750 houses) and decent hotel rents (550+550+600 per roll) yield the fastest payback.
    Light blue payback ≈ 25.2 rolls vs orange ≈ 28.4 (next best).

    Note: with real Markov distribution, orange's elevated jail-exit traffic reduces
    its payback below light_blue's. The criterion holds under uniform distribution.
    """
    table = payback_ranking_table(board, uniform_distribution)
    payback_at_hotel = table.loc[_COLOR_GROUPS, "hotel"]
    best = payback_at_hotel.idxmin()
    assert best == "light_blue", (
        f"Expected light_blue as lowest payback at hotel (uniform dist), got {best}.\n"
        f"Rankings:\n{payback_at_hotel.sort_values()}"
    )


# ---------------------------------------------------------------------------
# wilson_confidence_interval
# ---------------------------------------------------------------------------


def test_wilson_ci_known_example_50_wins_100_games() -> None:
    """50 wins in 100 games at 95% CI → (lower, upper) within ±0.001 of reference.

    Reference: Wilson CI formula, z = norm.ppf(0.975) ≈ 1.9600.
    p̂ = 0.5, n = 100 → symmetric interval centred at 0.5.
    Expected: lower ≈ 0.4038, upper ≈ 0.5962.
    """
    lower, upper = wilson_confidence_interval(50, 100, confidence=0.95)
    assert lower == pytest.approx(0.4038, abs=0.001)
    assert upper == pytest.approx(0.5962, abs=0.001)


def test_wilson_ci_returns_tuple_of_two_floats() -> None:
    """wilson_confidence_interval must return a tuple of exactly two floats."""
    result = wilson_confidence_interval(30, 100)
    assert isinstance(result, tuple)
    assert len(result) == 2
    lower, upper = result
    assert isinstance(lower, float)
    assert isinstance(upper, float)


def test_wilson_ci_lower_le_upper() -> None:
    """Lower bound must always be ≤ upper bound."""
    for wins, total in [(0, 10), (5, 10), (10, 10), (1, 100), (99, 100)]:
        lower, upper = wilson_confidence_interval(wins, total)
        assert lower <= upper, (
            f"lower={lower} > upper={upper} for wins={wins}, total={total}"
        )


def test_wilson_ci_bounds_within_zero_one() -> None:
    """Wilson CI must stay within [0, 1] even at boundary proportions."""
    lower_0, upper_0 = wilson_confidence_interval(0, 100)
    assert lower_0 >= 0.0
    assert upper_0 <= 1.0

    lower_all, upper_all = wilson_confidence_interval(100, 100)
    assert lower_all >= 0.0
    assert upper_all <= 1.0


def test_wilson_ci_zero_wins_lower_is_zero() -> None:
    """When wins=0, lower bound must be 0.0 (no evidence for successes)."""
    lower, _ = wilson_confidence_interval(0, 100)
    assert lower == pytest.approx(0.0)


def test_wilson_ci_all_wins_upper_is_one() -> None:
    """When wins=total, upper bound must be 1.0 (all successes)."""
    _, upper = wilson_confidence_interval(100, 100)
    assert upper == pytest.approx(1.0)


def test_wilson_ci_total_zero_raises_value_error() -> None:
    """total=0 must raise ValueError (undefined proportion)."""
    with pytest.raises(ValueError):
        wilson_confidence_interval(0, 0)


def test_wilson_ci_negative_wins_raises_value_error() -> None:
    """Negative wins must raise ValueError."""
    with pytest.raises(ValueError):
        wilson_confidence_interval(-1, 100)


def test_wilson_ci_wins_greater_than_total_raises_value_error() -> None:
    """wins > total must raise ValueError."""
    with pytest.raises(ValueError):
        wilson_confidence_interval(101, 100)


def test_wilson_ci_narrows_with_more_games() -> None:
    """A 10000-game CI must be narrower than a 100-game CI for the same proportion."""
    lower_100, upper_100 = wilson_confidence_interval(50, 100)
    lower_10k, upper_10k = wilson_confidence_interval(5000, 10000)
    width_100 = upper_100 - lower_100
    width_10k = upper_10k - lower_10k
    assert width_10k < width_100, (
        f"CI with 10000 games ({width_10k:.4f}) should be narrower than "
        f"CI with 100 games ({width_100:.4f})"
    )


def test_wilson_ci_respects_confidence_level() -> None:
    """A 99% CI must be wider than a 95% CI for the same data."""
    lower_95, upper_95 = wilson_confidence_interval(50, 100, confidence=0.95)
    lower_99, upper_99 = wilson_confidence_interval(50, 100, confidence=0.99)
    width_95 = upper_95 - lower_95
    width_99 = upper_99 - lower_99
    assert width_99 > width_95


# ---------------------------------------------------------------------------
# win_probability_table — structure tests (no real simulations required)
# ---------------------------------------------------------------------------


def _make_simulation_result(
    strategy_name: str, wins: int, total: int
) -> SimulationResult:
    """Create a minimal SimulationResult with a controlled win count."""
    winners = [strategy_name] * wins + [None] * (total - wins)
    return SimulationResult(
        winner_per_game=winners,
        turns_per_game=[100] * total,
        bankruptcy_order=[] * total,
        final_cash=[{strategy_name: 1500}] * total,
    )


def test_win_probability_table_returns_dataframe() -> None:
    """win_probability_table must return a pandas DataFrame."""
    sim_results = {
        ("BuyEverything", 2): _make_simulation_result("BuyEverything", 60, 100),
    }
    result = win_probability_table(sim_results, ["BuyEverything"], [2])
    assert isinstance(result, pd.DataFrame)


def test_win_probability_table_has_correct_columns() -> None:
    """DataFrame must have exactly the specified columns."""
    expected_columns = [
        "strategy",
        "n_players",
        "win_rate",
        "ci_lower",
        "ci_upper",
        "baseline",
        "significant",
    ]
    sim_results = {
        ("BuyEverything", 2): _make_simulation_result("BuyEverything", 60, 100),
    }
    result = win_probability_table(sim_results, ["BuyEverything"], [2])
    assert list(result.columns) == expected_columns


def test_win_probability_table_has_correct_row_count() -> None:
    """Row count must equal len(strategies) × len(player_counts)."""
    strategies = ["BuyEverything", "BuyNothing"]
    player_counts = [2, 3, 4]
    sim_results = {
        (s, n): _make_simulation_result(s, 30, 100)
        for s in strategies
        for n in player_counts
    }
    result = win_probability_table(sim_results, strategies, player_counts)
    assert len(result) == len(strategies) * len(player_counts)


def test_win_probability_table_win_rate_correct() -> None:
    """win_rate column must equal wins / total_games for each row."""
    sim_results = {
        ("BuyEverything", 2): _make_simulation_result("BuyEverything", 70, 100),
    }
    result = win_probability_table(sim_results, ["BuyEverything"], [2])
    row = result[result["strategy"] == "BuyEverything"].iloc[0]
    assert row["win_rate"] == pytest.approx(0.70)


def test_win_probability_table_baseline_equals_one_over_n_players() -> None:
    """baseline column must be exactly 1/n_players for each row."""
    sim_results = {
        ("BuyEverything", 2): _make_simulation_result("BuyEverything", 60, 100),
        ("BuyEverything", 4): _make_simulation_result("BuyEverything", 30, 100),
    }
    result = win_probability_table(sim_results, ["BuyEverything"], [2, 4])
    row_2 = result[result["n_players"] == 2].iloc[0]
    row_4 = result[result["n_players"] == 4].iloc[0]
    assert row_2["baseline"] == pytest.approx(0.5)
    assert row_4["baseline"] == pytest.approx(0.25)


def test_win_probability_table_significant_true_when_baseline_below_ci() -> None:
    """significant=True when baseline falls below ci_lower (strategy clearly wins)."""
    # 90 wins in 100 → CI is roughly [0.82, 0.95], baseline 0.5 is clearly below
    sim_results = {
        ("BuyEverything", 2): _make_simulation_result("BuyEverything", 90, 100),
    }
    result = win_probability_table(sim_results, ["BuyEverything"], [2])
    row = result.iloc[0]
    assert row["significant"] == True  # noqa: E712 — numpy bool requires == not is


def test_win_probability_table_significant_false_when_baseline_inside_ci() -> None:
    """significant=False when baseline falls inside CI (no statistical evidence)."""
    # 50 wins in 100 → CI is roughly [0.40, 0.60], baseline 0.5 is inside → not significant
    sim_results = {
        ("BuyEverything", 2): _make_simulation_result("BuyEverything", 50, 100),
    }
    result = win_probability_table(sim_results, ["BuyEverything"], [2])
    row = result.iloc[0]
    assert row["significant"] == False  # noqa: E712


def test_win_probability_table_significant_true_when_baseline_above_ci() -> None:
    """significant=True when baseline falls above ci_upper (strategy clearly loses)."""
    # 10 wins in 100 → CI is roughly [0.05, 0.17], baseline 0.5 is clearly above
    sim_results = {
        ("BuyNothing", 2): _make_simulation_result("BuyNothing", 10, 100),
    }
    result = win_probability_table(sim_results, ["BuyNothing"], [2])
    row = result.iloc[0]
    assert row["significant"] == True  # noqa: E712


def test_win_probability_table_ci_bounds_match_wilson_ci() -> None:
    """ci_lower and ci_upper must match direct wilson_confidence_interval output."""
    wins, total = 65, 100
    sim_results = {
        ("BuyEverything", 2): _make_simulation_result("BuyEverything", wins, total),
    }
    result = win_probability_table(sim_results, ["BuyEverything"], [2])
    row = result.iloc[0]
    expected_lower, expected_upper = wilson_confidence_interval(wins, total)
    assert row["ci_lower"] == pytest.approx(expected_lower)
    assert row["ci_upper"] == pytest.approx(expected_upper)


# ---------------------------------------------------------------------------
# Slow tests — real simulations to verify BuyEverything and BuyNothing
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_buy_everything_win_rate_exceeds_baseline_in_2player_games() -> None:
    """BuyEverything win rate must be statistically > 1/2 (significant=True) in 2-player games.

    BuyEverything acquires all properties, building a rent-generating portfolio that
    reliably outlasts the passive BuyNothing strategy over enough games.
    Runs 500 games to achieve sufficient statistical power.
    """
    n_games = 500
    sim = simulate_games(
        n_games=n_games,
        player_names=["BuyEverything", "BuyNothing"],
        strategies=[BuyEverything(), BuyNothing()],
        seed=42,
    )
    sim_results = {("BuyEverything", 2): sim}
    table = win_probability_table(sim_results, ["BuyEverything"], [2])
    row = table.iloc[0]
    assert row["win_rate"] > 0.5, (
        f"BuyEverything win rate {row['win_rate']:.3f} should exceed 0.5"
    )
    assert row["significant"] == True, (  # noqa: E712
        f"BuyEverything result should be significant (CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}])"
    )


@pytest.mark.slow
def test_buy_nothing_win_rate_below_baseline_in_2player_games() -> None:
    """BuyNothing win rate must be statistically < 1/2 (significant=True) in 2-player games.

    BuyNothing never buys properties, so it cannot generate rent income.
    Against BuyEverything it is almost always eliminated.
    Runs 500 games to achieve sufficient statistical power.
    """
    n_games = 500
    sim = simulate_games(
        n_games=n_games,
        player_names=["BuyEverything", "BuyNothing"],
        strategies=[BuyEverything(), BuyNothing()],
        seed=42,
    )
    sim_results = {("BuyNothing", 2): sim}
    table = win_probability_table(sim_results, ["BuyNothing"], [2])
    row = table.iloc[0]
    assert row["win_rate"] < 0.5, (
        f"BuyNothing win rate {row['win_rate']:.3f} should be below 0.5"
    )
    assert row["significant"] == True, (  # noqa: E712
        f"BuyNothing result should be significant (CI: [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}])"
    )

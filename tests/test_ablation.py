"""Tests for the ablation study module (ablation.py).

Covers all acceptance criteria from GitHub Issue #38:
- AblationResult dataclass structure
- AblationResult.to_dataframe() output shape and columns
- Wilson CI correctness (bounds in [0,1], lower <= win_rate <= upper)
- building_threshold ablation with ThreeHousesRush
- cash_reserve ablation with ThreeHousesRush
- jail_threshold ablation with JailCamper
- color_targeting ablation with ColorTargeted
- All 8 color groups ablation
"""

from __future__ import annotations

import pytest

from monopoly.ablation import AblationResult, run_ablation_study
from monopoly.strategies.buy_everything import BuyEverything
from monopoly.strategies.color_targeted import ColorTargeted
from monopoly.strategies.jail_camper import JailCamper
from monopoly.strategies.three_houses_rush import ThreeHousesRush

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_N_GAMES = 5
_SEED = 42
_OPPONENT = BuyEverything()

_ALL_COLORS = [
    "brown",
    "light_blue",
    "pink",
    "orange",
    "red",
    "yellow",
    "green",
    "dark_blue",
]


# ---------------------------------------------------------------------------
# 1. AblationResult structure
# ---------------------------------------------------------------------------


def test_ablation_result_has_correct_structure() -> None:
    """run_ablation_study with 2 values returns AblationResult with exactly 2 entries."""
    result = run_ablation_study(
        base_strategy=ThreeHousesRush(),
        parameter="building_threshold",
        values=[2, 3],
        opponent_strategies=[_OPPONENT],
        n_games=_N_GAMES,
        seed=_SEED,
    )

    assert isinstance(result, AblationResult)
    assert result.parameter_name == "building_threshold"
    assert len(result.values) == 2
    assert len(result.win_rates) == 2
    assert len(result.confidence_intervals) == 2


# ---------------------------------------------------------------------------
# 2. to_dataframe() columns and row count
# ---------------------------------------------------------------------------


def test_ablation_to_dataframe_columns() -> None:
    """AblationResult.to_dataframe() returns DataFrame with the correct columns and rows."""
    result = run_ablation_study(
        base_strategy=ThreeHousesRush(),
        parameter="building_threshold",
        values=[1, 2, 3],
        opponent_strategies=[_OPPONENT],
        n_games=_N_GAMES,
        seed=_SEED,
    )

    df = result.to_dataframe()
    assert list(df.columns) == ["parameter_value", "win_rate", "ci_lower", "ci_upper"]
    assert len(df) == 3


# ---------------------------------------------------------------------------
# 3. Wilson CI correctness
# ---------------------------------------------------------------------------


def test_wilson_ci_bounds() -> None:
    """Wilson CI bounds must be in [0, 1] and lower <= win_rate <= upper."""
    result = run_ablation_study(
        base_strategy=ThreeHousesRush(),
        parameter="cash_reserve",
        values=[0, 100, 200],
        opponent_strategies=[_OPPONENT],
        n_games=_N_GAMES,
        seed=_SEED,
    )

    for win_rate, (ci_lower, ci_upper) in zip(
        result.win_rates, result.confidence_intervals
    ):
        assert 0.0 <= ci_lower <= 1.0, f"ci_lower={ci_lower} out of [0, 1]"
        assert 0.0 <= ci_upper <= 1.0, f"ci_upper={ci_upper} out of [0, 1]"
        assert ci_lower <= win_rate <= ci_upper, (
            f"win_rate={win_rate} not in [{ci_lower}, {ci_upper}]"
        )


# ---------------------------------------------------------------------------
# 4. building_threshold ablation
# ---------------------------------------------------------------------------


def test_building_threshold_ablation() -> None:
    """Ablation over building_threshold [1, 2, 3] with ThreeHousesRush produces 3 entries."""
    result = run_ablation_study(
        base_strategy=ThreeHousesRush(),
        parameter="building_threshold",
        values=[1, 2, 3],
        opponent_strategies=[_OPPONENT],
        n_games=_N_GAMES,
        seed=_SEED,
    )

    assert len(result.values) == 3
    assert len(result.win_rates) == 3
    assert len(result.confidence_intervals) == 3


# ---------------------------------------------------------------------------
# 5. cash_reserve ablation
# ---------------------------------------------------------------------------


def test_cash_reserve_ablation() -> None:
    """Ablation over cash_reserve [0, 100, 200] with ThreeHousesRush produces 3 entries."""
    result = run_ablation_study(
        base_strategy=ThreeHousesRush(),
        parameter="cash_reserve",
        values=[0, 100, 200],
        opponent_strategies=[_OPPONENT],
        n_games=_N_GAMES,
        seed=_SEED,
    )

    assert len(result.values) == 3
    assert len(result.win_rates) == 3
    assert len(result.confidence_intervals) == 3


# ---------------------------------------------------------------------------
# 6. jail_threshold ablation
# ---------------------------------------------------------------------------


def test_jail_threshold_ablation() -> None:
    """Ablation over late_game_threshold [3, 6, 9] with JailCamper produces 3 entries."""
    result = run_ablation_study(
        base_strategy=JailCamper(),
        parameter="jail_threshold",
        values=[3, 6, 9],
        opponent_strategies=[_OPPONENT],
        n_games=_N_GAMES,
        seed=_SEED,
    )

    assert len(result.values) == 3
    assert len(result.win_rates) == 3
    assert len(result.confidence_intervals) == 3


# ---------------------------------------------------------------------------
# 7. color_targeting ablation
# ---------------------------------------------------------------------------


def test_color_targeting_ablation() -> None:
    """Ablation over ["brown", "orange"] with ColorTargeted produces 2 entries."""
    result = run_ablation_study(
        base_strategy=ColorTargeted(target_colors=["orange"]),
        parameter="color_targeting",
        values=["brown", "orange"],
        opponent_strategies=[_OPPONENT],
        n_games=_N_GAMES,
        seed=_SEED,
    )

    assert len(result.values) == 2
    assert len(result.win_rates) == 2
    assert len(result.confidence_intervals) == 2


# ---------------------------------------------------------------------------
# 8. All 8 color groups ablation
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_all_8_color_groups_ablation() -> None:
    """Ablation over all 8 Monopoly color groups produces 8 result entries."""
    result = run_ablation_study(
        base_strategy=ColorTargeted(target_colors=["orange"]),
        parameter="color_targeting",
        values=_ALL_COLORS,
        opponent_strategies=[_OPPONENT],
        n_games=_N_GAMES,
        seed=_SEED,
    )

    assert len(result.values) == 8
    assert len(result.win_rates) == 8
    assert len(result.confidence_intervals) == 8

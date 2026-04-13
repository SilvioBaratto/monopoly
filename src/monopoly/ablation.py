"""Ablation study runner for Monopoly strategy hyperparameters.

Responsibilities (SRP):
- AblationResult dataclass: stores parameter sweep outcomes
- run_ablation_study: sweeps a single parameter over a list of values,
  running simulate_games for each and collecting win rates + Wilson CIs
- Wilson CI delegated to metrics.wilson_confidence_interval

No game rule logic — delegates entirely to simulate_games().
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from monopoly.metrics import wilson_confidence_interval
from monopoly.simulate import simulate_games
from monopoly.strategies.base import Strategy
from monopoly.strategies.color_targeted import ColorTargeted
from monopoly.strategies.jail_camper import JailCamper
from monopoly.strategies.three_houses_rush import ThreeHousesRush


@dataclass
class AblationResult:
    """Outcomes of a single-parameter ablation study.

    Args:
        parameter_name: The name of the swept parameter (e.g. "building_threshold").
        values: Ordered list of parameter values that were tested.
        win_rates: Win rate of the variant strategy for each value in ``values``.
        confidence_intervals: Wilson 95% CI as (lower, upper) for each value.
    """

    parameter_name: str
    values: list = field(default_factory=list)
    win_rates: list[float] = field(default_factory=list)
    confidence_intervals: list[tuple[float, float]] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Return results as a tidy DataFrame.

        Returns:
            DataFrame with columns: parameter_value, win_rate, ci_lower, ci_upper.
            Row count equals the number of swept values.
        """
        rows = [
            {
                "parameter_value": value,
                "win_rate": win_rate,
                "ci_lower": ci[0],
                "ci_upper": ci[1],
            }
            for value, win_rate, ci in zip(
                self.values, self.win_rates, self.confidence_intervals
            )
        ]
        return pd.DataFrame(
            rows, columns=["parameter_value", "win_rate", "ci_lower", "ci_upper"]
        )


def run_ablation_study(
    base_strategy: Strategy,
    parameter: str,
    values: list,
    opponent_strategies: list[Strategy],
    n_games: int = 100,
    seed: int | None = None,
) -> AblationResult:
    """Sweep a single hyperparameter across a list of values and collect win rates.

    For each value in ``values``:
    1. Instantiate a variant strategy of the same type as ``base_strategy``,
       with ``parameter`` set to ``value``.
    2. Run ``n_games`` games: variant vs. ``opponent_strategies``.
    3. Compute the variant's win rate (wins / n_games).
    4. Compute the Wilson 95% CI via scipy.stats.proportion_confint.

    Player names: variant is "Variant"; opponents are "Opponent_0", "Opponent_1", …

    Seeds are deterministically derived from ``seed`` and the value index using the
    same mixing function as ``simulate_games._derive_sub_seed``:
    ``(seed * 6364136223846793005 + value_index) & 0xFFFF_FFFF_FFFF_FFFF``

    Args:
        base_strategy: Strategy instance whose type and parameter are swept.
        parameter: Name of the hyperparameter to vary.
        values: Ordered list of values to test.
        opponent_strategies: Strategies for the opponent players (1+).
        n_games: Number of games per value. Defaults to 100.
        seed: Master RNG seed for reproducibility. None for non-reproducible runs.

    Returns:
        AblationResult with per-value win rates and confidence intervals.

    Raises:
        ValueError: If ``parameter`` is not supported for ``type(base_strategy)``.
    """
    player_names = ["Variant"] + [
        f"Opponent_{i}" for i in range(len(opponent_strategies))
    ]

    win_rates: list[float] = []
    confidence_intervals: list[tuple[float, float]] = []

    for value_index, value in enumerate(values):
        per_value_seed = _derive_value_seed(seed, value_index)
        variant = _create_variant(base_strategy, parameter, value)
        strategies = [variant, *opponent_strategies]

        sim_result = simulate_games(
            n_games=n_games,
            player_names=player_names,
            strategies=strategies,
            seed=per_value_seed,
        )

        wins = sum(1 for w in sim_result.winner_per_game if w == "Variant")
        win_rate = wins / n_games

        ci_lower, ci_upper = wilson_confidence_interval(wins, n_games, confidence=0.95)

        win_rates.append(win_rate)
        confidence_intervals.append((float(ci_lower), float(ci_upper)))

    return AblationResult(
        parameter_name=parameter,
        values=list(values),
        win_rates=win_rates,
        confidence_intervals=confidence_intervals,
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _derive_value_seed(master_seed: int | None, value_index: int) -> int | None:
    """Produce a deterministic seed for one value in the ablation sweep.

    Args:
        master_seed: The top-level seed passed to ``run_ablation_study``.
        value_index: Zero-based position of the value within the sweep.

    Returns:
        An integer seed, or None for non-reproducible runs.
    """
    if master_seed is None:
        return None
    return (master_seed * 6364136223846793005 + value_index) & 0xFFFF_FFFF_FFFF_FFFF


def _create_variant(base_strategy: Strategy, parameter: str, value) -> Strategy:
    """Instantiate a variant of ``base_strategy`` with ``parameter`` set to ``value``.

    Supported (type, parameter) combinations:
    - (ThreeHousesRush, "building_threshold") → ThreeHousesRush(target_houses=value)
    - (ThreeHousesRush, "cash_reserve")       → ThreeHousesRush(cash_reserve=value)
    - (JailCamper, "jail_threshold")          → JailCamper(late_game_threshold=value)
    - (ColorTargeted, "color_targeting")      → ColorTargeted(target_colors=[value])
                                                  or ColorTargeted(target_colors=value)

    Args:
        base_strategy: The base strategy instance (used for type dispatch).
        parameter: Name of the hyperparameter to set.
        value: The value to assign to the hyperparameter.

    Returns:
        A new strategy instance with the specified parameter value.

    Raises:
        ValueError: If the (type, parameter) combination is not supported.
    """
    strategy_type = type(base_strategy)

    if strategy_type is ThreeHousesRush:
        return _create_three_houses_rush_variant(base_strategy, parameter, value)  # type: ignore[arg-type]
    if strategy_type is JailCamper:
        return _create_jail_camper_variant(base_strategy, parameter, value)  # type: ignore[arg-type]
    if strategy_type is ColorTargeted:
        return _create_color_targeted_variant(base_strategy, parameter, value)  # type: ignore[arg-type]

    raise ValueError(
        f"Unsupported strategy type for ablation: {strategy_type.__name__!r}"
    )


def _create_three_houses_rush_variant(
    base: ThreeHousesRush, parameter: str, value
) -> ThreeHousesRush:
    """Create a ThreeHousesRush variant with one parameter changed.

    Args:
        base: The base ThreeHousesRush instance (for default values).
        parameter: One of "building_threshold" or "cash_reserve".
        value: The new value for the parameter.

    Returns:
        A new ThreeHousesRush with the specified parameter set.

    Raises:
        ValueError: If ``parameter`` is not recognised.
    """
    if parameter == "building_threshold":
        return ThreeHousesRush(
            target_houses=value,
            cash_reserve=base.cash_reserve,
        )
    if parameter == "cash_reserve":
        return ThreeHousesRush(
            target_houses=base.target_houses,
            cash_reserve=value,
        )
    raise ValueError(
        f"Unsupported parameter for ThreeHousesRush: {parameter!r}. "
        "Expected 'building_threshold' or 'cash_reserve'."
    )


def _create_jail_camper_variant(base: JailCamper, parameter: str, value) -> JailCamper:
    """Create a JailCamper variant with one parameter changed.

    Args:
        base: The base JailCamper instance.
        parameter: Must be "jail_threshold".
        value: The new late_game_threshold value.

    Returns:
        A new JailCamper with late_game_threshold set to ``value``.

    Raises:
        ValueError: If ``parameter`` is not "jail_threshold".
    """
    if parameter == "jail_threshold":
        return JailCamper(late_game_threshold=value)
    raise ValueError(
        f"Unsupported parameter for JailCamper: {parameter!r}. "
        "Expected 'jail_threshold'."
    )


def _create_color_targeted_variant(
    base: ColorTargeted, parameter: str, value
) -> ColorTargeted:
    """Create a ColorTargeted variant with one parameter changed.

    Args:
        base: The base ColorTargeted instance.
        parameter: Must be "color_targeting".
        value: A single color string or a list of color strings.

    Returns:
        A new ColorTargeted with target_colors derived from ``value``.

    Raises:
        ValueError: If ``parameter`` is not "color_targeting".
    """
    if parameter == "color_targeting":
        target_colors = [value] if isinstance(value, str) else list(value)
        return ColorTargeted(target_colors=target_colors)
    raise ValueError(
        f"Unsupported parameter for ColorTargeted: {parameter!r}. "
        "Expected 'color_targeting'."
    )

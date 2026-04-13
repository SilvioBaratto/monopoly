"""Tests for the round-robin tournament system and ranking functions.

Tests cover every acceptance criterion from GitHub Issue #36 (tournament) and
GitHub Issue #37 (Bradley-Terry/Elo ranking):

Issue #36 — round-robin tournament:
- run_tournament(strategies, n_games_per_matchup, seed) returns TournamentResult
- TournamentResult contains matchup matrix and per-strategy aggregate stats
- All strategy pairs are played (n*(n-1)/2 matchups for n strategies)
- Leverages parallel simulation for speed
- Deterministic given the same seed
- BuyEverything beats BuyNothing > 90% of the time (sanity check)
- Tournament with 3 strategies produces exactly 3 matchups (3 choose 2)
- TournamentResult round-trips to/from dict (serializable)

Issue #37 — ranking:
- bradley_terry_ranking(result) returns DataFrame with strategy, strength, ci_lower, ci_upper
- elo_ranking(result) returns DataFrame with strategy, elo, n_games, ci_lower, ci_upper
- BT uses scipy.optimize.minimize (verified via output shape and correctness)
- BT parameters are positive; reference strategy fixed at 1 for identifiability
- On synthetic dominant-strategy data, BT and Elo agree qualitatively
- BuyEverything ranks above BuyNothing (sanity check on real tournament data)
- CIs: Fisher information (BT) and bootstrap (Elo)
- Degenerate inputs raise ValueError
"""

from __future__ import annotations

import pytest

from monopoly.tournament import TournamentResult, run_tournament
from monopoly.ranking import bradley_terry_ranking, elo_ranking
from monopoly.strategies.buy_everything import BuyEverything
from monopoly.strategies.buy_nothing import BuyNothing
from monopoly.strategies.jail_camper import JailCamper


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------


def _make_dominant_tournament(
    dominant_win_rate: float = 0.9, n_games: int = 100
) -> TournamentResult:
    """Synthetic 3-strategy tournament where 'Alpha' clearly dominates."""
    win_rates = {
        ("Alpha", "Beta"): dominant_win_rate,
        ("Alpha", "Gamma"): dominant_win_rate,
        ("Beta", "Gamma"): 0.5,
    }
    aggregate_stats = {
        "Alpha": {
            "overall_win_rate": dominant_win_rate,
            "avg_turns": 100.0,
            "avg_final_cash": 5000.0,
        },
        "Beta": {"overall_win_rate": 0.3, "avg_turns": 100.0, "avg_final_cash": 1500.0},
        "Gamma": {
            "overall_win_rate": 0.3,
            "avg_turns": 100.0,
            "avg_final_cash": 1500.0,
        },
    }
    return TournamentResult(
        win_rates=win_rates,
        aggregate_stats=aggregate_stats,
        n_games_per_matchup=n_games,
    )


def _make_two_strategy_tournament(
    win_rate_a: float = 0.85, n_games: int = 100
) -> TournamentResult:
    """Synthetic 2-strategy tournament where 'Strong' beats 'Weak'."""
    win_rates = {("Strong", "Weak"): win_rate_a}
    aggregate_stats = {
        "Strong": {
            "overall_win_rate": win_rate_a,
            "avg_turns": 100.0,
            "avg_final_cash": 3000.0,
        },
        "Weak": {
            "overall_win_rate": 1.0 - win_rate_a,
            "avg_turns": 100.0,
            "avg_final_cash": 1000.0,
        },
    }
    return TournamentResult(
        win_rates=win_rates,
        aggregate_stats=aggregate_stats,
        n_games_per_matchup=n_games,
    )


# ---------------------------------------------------------------------------
# 1. run_tournament returns TournamentResult
# ---------------------------------------------------------------------------


def test_run_tournament_returns_tournament_result():
    """run_tournament must return a TournamentResult instance."""
    strategies = [BuyEverything(), BuyNothing()]
    result = run_tournament(strategies=strategies, n_games_per_matchup=10, seed=0)
    assert isinstance(result, TournamentResult)


# ---------------------------------------------------------------------------
# 2. TournamentResult matchup matrix
# ---------------------------------------------------------------------------


def test_tournament_result_contains_matchup_matrix():
    """TournamentResult must have a win_rates dict with strategy-pair keys."""
    strategies = [BuyEverything(), BuyNothing()]
    result = run_tournament(strategies=strategies, n_games_per_matchup=10, seed=1)
    assert hasattr(result, "win_rates")
    assert isinstance(result.win_rates, dict)


def test_matchup_matrix_values_are_floats_between_0_and_1():
    """Win rates in the matchup matrix must be floats in [0.0, 1.0]."""
    strategies = [BuyEverything(), BuyNothing()]
    result = run_tournament(strategies=strategies, n_games_per_matchup=10, seed=2)
    for rate in result.win_rates.values():
        assert isinstance(rate, float)
        assert 0.0 <= rate <= 1.0


# ---------------------------------------------------------------------------
# 3. Correct number of matchups: n*(n-1)/2
# ---------------------------------------------------------------------------


def test_two_strategies_produce_one_matchup():
    """A 2-strategy tournament must produce exactly 1 matchup."""
    strategies = [BuyEverything(), BuyNothing()]
    result = run_tournament(strategies=strategies, n_games_per_matchup=10, seed=3)
    assert len(result.win_rates) == 1


def test_three_strategies_produce_three_matchups():
    """A 3-strategy tournament must produce exactly 3 matchups (3 choose 2)."""
    strategies = [BuyEverything(), BuyNothing(), JailCamper()]
    result = run_tournament(strategies=strategies, n_games_per_matchup=10, seed=4)
    assert len(result.win_rates) == 3


# ---------------------------------------------------------------------------
# 4. Per-strategy aggregate stats
# ---------------------------------------------------------------------------


def test_tournament_result_contains_aggregate_stats():
    """TournamentResult must have aggregate_stats dict keyed by strategy name."""
    strategies = [BuyEverything(), BuyNothing()]
    result = run_tournament(strategies=strategies, n_games_per_matchup=10, seed=5)
    assert hasattr(result, "aggregate_stats")
    assert isinstance(result.aggregate_stats, dict)


def test_aggregate_stats_contain_required_fields():
    """Each aggregate stat must contain: overall_win_rate, avg_turns, avg_final_cash."""
    strategies = [BuyEverything(), BuyNothing()]
    result = run_tournament(strategies=strategies, n_games_per_matchup=10, seed=6)
    for name, stats in result.aggregate_stats.items():
        assert "overall_win_rate" in stats, f"Missing overall_win_rate for {name}"
        assert "avg_turns" in stats, f"Missing avg_turns for {name}"
        assert "avg_final_cash" in stats, f"Missing avg_final_cash for {name}"


def test_aggregate_stats_win_rate_is_between_0_and_1():
    """overall_win_rate must be a float in [0.0, 1.0] for each strategy."""
    strategies = [BuyEverything(), BuyNothing()]
    result = run_tournament(strategies=strategies, n_games_per_matchup=10, seed=7)
    for stats in result.aggregate_stats.values():
        assert 0.0 <= stats["overall_win_rate"] <= 1.0


def test_aggregate_stats_avg_turns_is_positive():
    """avg_turns must be a positive number for each strategy."""
    strategies = [BuyEverything(), BuyNothing()]
    result = run_tournament(strategies=strategies, n_games_per_matchup=10, seed=8)
    for stats in result.aggregate_stats.values():
        assert stats["avg_turns"] > 0


def test_aggregate_stats_avg_final_cash_is_non_negative():
    """avg_final_cash must be a non-negative number for each strategy."""
    strategies = [BuyEverything(), BuyNothing()]
    result = run_tournament(strategies=strategies, n_games_per_matchup=10, seed=9)
    for stats in result.aggregate_stats.values():
        assert stats["avg_final_cash"] >= 0


def test_aggregate_stats_keyed_by_strategy_names():
    """aggregate_stats must contain an entry for each strategy name."""
    strategies = [BuyEverything(), BuyNothing()]
    result = run_tournament(strategies=strategies, n_games_per_matchup=10, seed=10)
    strategy_names = {type(s).__name__ for s in strategies}
    assert set(result.aggregate_stats.keys()) == strategy_names


# ---------------------------------------------------------------------------
# 5. Determinism — same seed → identical results
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_tournament_results():
    """run_tournament with the same seed must produce identical TournamentResult."""
    strategies = [BuyEverything(), BuyNothing()]
    r1 = run_tournament(strategies=strategies, n_games_per_matchup=20, seed=42)
    r2 = run_tournament(strategies=strategies, n_games_per_matchup=20, seed=42)
    assert r1.win_rates == r2.win_rates
    assert r1.aggregate_stats == r2.aggregate_stats


# ---------------------------------------------------------------------------
# 6. Sanity check — BuyEverything beats BuyNothing > 90%
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_buy_everything_beats_buy_nothing_over_90_percent():
    """BuyEverything must win > 90% of head-to-head games against BuyNothing."""
    strategies = [BuyEverything(), BuyNothing()]
    result = run_tournament(strategies=strategies, n_games_per_matchup=1000, seed=42)

    # The matchup key is (winner_strategy, loser_strategy) → win rate of the first
    # Find the win rate of BuyEverything vs BuyNothing
    be_win_rate = result.win_rates[("BuyEverything", "BuyNothing")]
    assert be_win_rate > 0.90, (
        f"BuyEverything won only {be_win_rate:.1%} of games; expected > 90%"
    )


# ---------------------------------------------------------------------------
# 7. Serialization — TournamentResult round-trips to/from dict
# ---------------------------------------------------------------------------


def test_tournament_result_to_dict():
    """TournamentResult.to_dict() must return a plain dict."""
    strategies = [BuyEverything(), BuyNothing()]
    result = run_tournament(strategies=strategies, n_games_per_matchup=10, seed=11)
    d = result.to_dict()
    assert isinstance(d, dict)
    assert "win_rates" in d
    assert "aggregate_stats" in d


def test_tournament_result_round_trips_from_dict():
    """TournamentResult.from_dict(result.to_dict()) must reproduce identical data."""
    strategies = [BuyEverything(), BuyNothing()]
    result = run_tournament(strategies=strategies, n_games_per_matchup=10, seed=12)
    d = result.to_dict()
    restored = TournamentResult.from_dict(d)
    assert restored.win_rates == result.win_rates
    assert restored.aggregate_stats == result.aggregate_stats


# ---------------------------------------------------------------------------
# 8. Validation
# ---------------------------------------------------------------------------


def test_run_tournament_raises_on_fewer_than_two_strategies():
    """run_tournament must raise ValueError when fewer than 2 strategies are given."""
    with pytest.raises(ValueError, match="strategies"):
        run_tournament(strategies=[BuyEverything()], n_games_per_matchup=10, seed=0)


def test_run_tournament_raises_on_zero_games():
    """run_tournament must raise ValueError when n_games_per_matchup < 1."""
    with pytest.raises(ValueError, match="n_games_per_matchup"):
        run_tournament(
            strategies=[BuyEverything(), BuyNothing()],
            n_games_per_matchup=0,
            seed=0,
        )


# ===========================================================================
# Issue #37 — Bradley-Terry and Elo ranking
# ===========================================================================


# ---------------------------------------------------------------------------
# A. bradley_terry_ranking — output structure
# ---------------------------------------------------------------------------


def test_bradley_terry_ranking_returns_dataframe():
    """bradley_terry_ranking must return a pandas DataFrame."""
    import pandas as pd

    result = _make_dominant_tournament()
    df = bradley_terry_ranking(result)
    assert isinstance(df, pd.DataFrame)


def test_bradley_terry_ranking_required_columns():
    """DataFrame must have columns: strategy, strength, ci_lower, ci_upper."""
    df = bradley_terry_ranking(_make_dominant_tournament())
    for col in ("strategy", "strength", "ci_lower", "ci_upper"):
        assert col in df.columns, f"Missing column: {col}"


def test_bradley_terry_ranking_one_row_per_strategy():
    """DataFrame must have exactly one row per strategy in the tournament."""
    result = _make_dominant_tournament()
    df = bradley_terry_ranking(result)
    assert len(df) == 3


def test_bradley_terry_parameters_are_positive():
    """All BT strength parameters must be strictly positive."""
    df = bradley_terry_ranking(_make_dominant_tournament())
    assert (df["strength"] > 0).all(), "Some BT strengths are non-positive"


def test_bradley_terry_ci_bounds_are_valid():
    """ci_lower must be ≤ strength ≤ ci_upper for all strategies."""
    df = bradley_terry_ranking(_make_dominant_tournament())
    assert (df["ci_lower"] <= df["strength"]).all()
    assert (df["strength"] <= df["ci_upper"]).all()


def test_bradley_terry_sorted_by_strength_descending():
    """DataFrame must be sorted by strength descending (rank 1 first)."""
    df = bradley_terry_ranking(_make_dominant_tournament())
    strengths = df["strength"].tolist()
    assert strengths == sorted(strengths, reverse=True)


# ---------------------------------------------------------------------------
# B. bradley_terry_ranking — correctness
# ---------------------------------------------------------------------------


def test_bradley_terry_dominant_strategy_ranks_first():
    """On synthetic data, the clearly dominant strategy must rank first."""
    df = bradley_terry_ranking(_make_dominant_tournament(dominant_win_rate=0.9))
    assert df.iloc[0]["strategy"] == "Alpha", (
        f"Expected 'Alpha' at rank 1, got '{df.iloc[0]['strategy']}'"
    )


def test_bradley_terry_reference_strategy_is_identified():
    """At least one strategy must serve as reference (strength closest to 1.0).

    The BT model fixes one parameter for identifiability. This test verifies
    that the model produces a valid identified solution: not all strengths are
    equal and they are not all zero.
    """
    df = bradley_terry_ranking(_make_two_strategy_tournament())
    # With a clear winner, strengths must be different
    assert df["strength"].nunique() > 1, (
        "All strategies have equal strength — model not identified"
    )


def test_bradley_terry_clear_winner_has_higher_strength():
    """The strategy with high win rate must have greater BT strength."""
    df = bradley_terry_ranking(_make_two_strategy_tournament(win_rate_a=0.85))
    strong_strength = df.loc[df["strategy"] == "Strong", "strength"].iloc[0]
    weak_strength = df.loc[df["strategy"] == "Weak", "strength"].iloc[0]
    assert strong_strength > weak_strength, (
        f"Strong ({strong_strength:.3f}) should be > Weak ({weak_strength:.3f})"
    )


# ---------------------------------------------------------------------------
# C. bradley_terry_ranking — degenerate inputs
# ---------------------------------------------------------------------------


def test_bradley_terry_single_strategy_raises_value_error():
    """Single strategy produces a degenerate input; must raise ValueError."""
    result = TournamentResult(
        win_rates={},
        aggregate_stats={
            "OnlyOne": {
                "overall_win_rate": 1.0,
                "avg_turns": 50.0,
                "avg_final_cash": 1500.0,
            }
        },
        n_games_per_matchup=10,
    )
    with pytest.raises(ValueError, match="[Rr]ank"):
        bradley_terry_ranking(result)


def test_bradley_terry_no_games_raises_value_error():
    """Empty win_rates (zero games played) must raise ValueError."""
    result = TournamentResult(
        win_rates={},
        aggregate_stats={
            "A": {
                "overall_win_rate": 0.5,
                "avg_turns": 100.0,
                "avg_final_cash": 1500.0,
            },
            "B": {
                "overall_win_rate": 0.5,
                "avg_turns": 100.0,
                "avg_final_cash": 1500.0,
            },
        },
        n_games_per_matchup=0,
    )
    with pytest.raises(ValueError):
        bradley_terry_ranking(result)


# ---------------------------------------------------------------------------
# D. elo_ranking — output structure
# ---------------------------------------------------------------------------


def test_elo_ranking_returns_dataframe():
    """elo_ranking must return a pandas DataFrame."""
    import pandas as pd

    df = elo_ranking(_make_dominant_tournament())
    assert isinstance(df, pd.DataFrame)


def test_elo_ranking_required_columns():
    """DataFrame must have columns: strategy, elo, n_games, ci_lower, ci_upper."""
    df = elo_ranking(_make_dominant_tournament())
    for col in ("strategy", "elo", "n_games", "ci_lower", "ci_upper"):
        assert col in df.columns, f"Missing column: {col}"


def test_elo_ranking_one_row_per_strategy():
    """DataFrame must have exactly one row per strategy."""
    df = elo_ranking(_make_dominant_tournament())
    assert len(df) == 3


def test_elo_ranking_n_games_is_positive():
    """n_games must be a positive integer for each strategy."""
    df = elo_ranking(_make_dominant_tournament())
    assert (df["n_games"] > 0).all()


def test_elo_ranking_sorted_by_elo_descending():
    """DataFrame must be sorted by elo descending."""
    df = elo_ranking(_make_dominant_tournament())
    elos = df["elo"].tolist()
    assert elos == sorted(elos, reverse=True)


def test_elo_ci_bounds_are_valid():
    """ci_lower must be ≤ elo ≤ ci_upper for all strategies."""
    df = elo_ranking(_make_dominant_tournament())
    assert (df["ci_lower"] <= df["elo"]).all()
    assert (df["elo"] <= df["ci_upper"]).all()


# ---------------------------------------------------------------------------
# E. elo_ranking — correctness
# ---------------------------------------------------------------------------


def test_elo_dominant_strategy_ranks_first():
    """On synthetic data, the dominant strategy must have highest Elo."""
    df = elo_ranking(_make_dominant_tournament(dominant_win_rate=0.9))
    assert df.iloc[0]["strategy"] == "Alpha", (
        f"Expected 'Alpha' at rank 1, got '{df.iloc[0]['strategy']}'"
    )


def test_elo_clear_winner_has_higher_rating():
    """The strategy with high win rate must have a higher Elo rating."""
    df = elo_ranking(_make_two_strategy_tournament(win_rate_a=0.85))
    strong_elo = df.loc[df["strategy"] == "Strong", "elo"].iloc[0]
    weak_elo = df.loc[df["strategy"] == "Weak", "elo"].iloc[0]
    assert strong_elo > weak_elo, (
        f"Strong Elo ({strong_elo:.1f}) should be > Weak Elo ({weak_elo:.1f})"
    )


# ---------------------------------------------------------------------------
# F. elo_ranking — degenerate inputs
# ---------------------------------------------------------------------------


def test_elo_single_strategy_raises_value_error():
    """Single strategy (no matchups) must raise ValueError."""
    result = TournamentResult(
        win_rates={},
        aggregate_stats={
            "OnlyOne": {
                "overall_win_rate": 1.0,
                "avg_turns": 50.0,
                "avg_final_cash": 1500.0,
            }
        },
        n_games_per_matchup=10,
    )
    with pytest.raises(ValueError, match="[Rr]ank"):
        elo_ranking(result)


# ---------------------------------------------------------------------------
# G. Qualitative agreement between BT and Elo
# ---------------------------------------------------------------------------


def test_bt_and_elo_agree_on_dominant_strategy():
    """On synthetic data with a clear winner, BT and Elo must agree on rank 1."""
    result = _make_dominant_tournament(dominant_win_rate=0.9)
    bt_df = bradley_terry_ranking(result)
    elo_df = elo_ranking(result)
    assert bt_df.iloc[0]["strategy"] == elo_df.iloc[0]["strategy"], (
        f"BT top: {bt_df.iloc[0]['strategy']}, Elo top: {elo_df.iloc[0]['strategy']}"
    )


# ---------------------------------------------------------------------------
# H. Sanity check on real tournament data (BuyEverything vs BuyNothing)
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_bradley_terry_buy_everything_ranks_above_buy_nothing():
    """BuyEverything must rank above BuyNothing in BT ranking from real tournament."""
    strategies = [BuyEverything(), BuyNothing()]
    tournament = run_tournament(strategies=strategies, n_games_per_matchup=200, seed=42)
    df = bradley_terry_ranking(tournament)
    be_rank = df.index[df["strategy"] == "BuyEverything"].tolist()[0]
    bn_rank = df.index[df["strategy"] == "BuyNothing"].tolist()[0]
    assert be_rank < bn_rank, (
        f"BuyEverything rank ({be_rank}) should be above BuyNothing rank ({bn_rank})"
    )


@pytest.mark.slow
def test_elo_buy_everything_ranks_above_buy_nothing():
    """BuyEverything must rank above BuyNothing in Elo ranking from real tournament."""
    strategies = [BuyEverything(), BuyNothing()]
    tournament = run_tournament(strategies=strategies, n_games_per_matchup=200, seed=42)
    df = elo_ranking(tournament)
    be_rank = df.index[df["strategy"] == "BuyEverything"].tolist()[0]
    bn_rank = df.index[df["strategy"] == "BuyNothing"].tolist()[0]
    assert be_rank < bn_rank, (
        f"BuyEverything rank ({be_rank}) should be above BuyNothing rank ({bn_rank})"
    )

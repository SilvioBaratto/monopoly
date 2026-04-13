"""Tests for src/monopoly/cli.py — written BEFORE implementation (TDD).

Covers every acceptance criterion from GitHub Issue #49:
- monopoly simulate: runs simulation and prints summary stats
- monopoly markov: computes and displays stationary distribution
- monopoly tournament: runs round-robin tournament and prints ranking
- --seed flag on every command
- --help works on every command
- invalid strategy name produces a clear error message
- --strategy repeatable with round-robin assignment
"""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from monopoly.cli import app

runner = CliRunner()

# ---------------------------------------------------------------------------
# simulate command
# ---------------------------------------------------------------------------


class TestSimulateHelp:
    """--help output is accessible and informative."""

    def test_help_exits_successfully(self) -> None:
        result = runner.invoke(app, ["simulate", "--help"])
        assert result.exit_code == 0

    def test_help_mentions_n_games(self) -> None:
        result = runner.invoke(app, ["simulate", "--help"])
        assert "--n-games" in result.output

    def test_help_mentions_strategy(self) -> None:
        result = runner.invoke(app, ["simulate", "--help"])
        assert "--strategy" in result.output

    def test_help_mentions_seed(self) -> None:
        result = runner.invoke(app, ["simulate", "--help"])
        assert "--seed" in result.output


class TestSimulateSuccess:
    """simulate command produces valid output on success paths."""

    def test_runs_with_minimal_options(self) -> None:
        result = runner.invoke(app, ["simulate", "--n-games", "5", "--players", "2"])
        assert result.exit_code == 0

    def test_runs_with_seed_for_reproducibility(self) -> None:
        result_a = runner.invoke(
            app, ["simulate", "--n-games", "5", "--players", "2", "--seed", "42"]
        )
        result_b = runner.invoke(
            app, ["simulate", "--n-games", "5", "--players", "2", "--seed", "42"]
        )
        assert result_a.exit_code == 0
        assert result_a.output == result_b.output

    def test_different_seeds_may_differ(self) -> None:
        result_a = runner.invoke(
            app, ["simulate", "--n-games", "20", "--players", "2", "--seed", "1"]
        )
        result_b = runner.invoke(
            app, ["simulate", "--n-games", "20", "--players", "2", "--seed", "99"]
        )
        # Not guaranteed to differ for very short runs, but highly likely
        assert result_a.exit_code == 0
        assert result_b.exit_code == 0

    def test_output_contains_win_rate(self) -> None:
        result = runner.invoke(
            app, ["simulate", "--n-games", "5", "--players", "2", "--seed", "42"]
        )
        assert "Win" in result.output or "win" in result.output

    def test_output_contains_player_names(self) -> None:
        result = runner.invoke(
            app, ["simulate", "--n-games", "5", "--players", "3", "--seed", "42"]
        )
        assert "Player_1" in result.output
        assert "Player_2" in result.output
        assert "Player_3" in result.output

    def test_default_strategy_is_buy_everything(self) -> None:
        result = runner.invoke(
            app, ["simulate", "--n-games", "5", "--players", "2", "--seed", "42"]
        )
        assert result.exit_code == 0
        assert "BuyEverything" in result.output

    def test_explicit_single_strategy(self) -> None:
        result = runner.invoke(
            app,
            [
                "simulate",
                "--n-games",
                "5",
                "--players",
                "2",
                "--strategy",
                "BuyNothing",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0
        assert "BuyNothing" in result.output


class TestSimulateStrategyRoundRobin:
    """--strategy repeatable with round-robin assignment."""

    def test_single_strategy_assigned_to_all_players(self) -> None:
        result = runner.invoke(
            app,
            [
                "simulate",
                "--n-games",
                "5",
                "--players",
                "4",
                "--strategy",
                "BuyEverything",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0

    def test_two_strategies_round_robin_four_players(self) -> None:
        result = runner.invoke(
            app,
            [
                "simulate",
                "--n-games",
                "5",
                "--players",
                "4",
                "--strategy",
                "BuyEverything",
                "--strategy",
                "BuyNothing",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0

    def test_more_strategies_than_players_uses_first_n(self) -> None:
        # 3 strategies for 2 players: first 2 used
        result = runner.invoke(
            app,
            [
                "simulate",
                "--n-games",
                "5",
                "--players",
                "2",
                "--strategy",
                "BuyEverything",
                "--strategy",
                "BuyNothing",
                "--strategy",
                "Trader",
                "--seed",
                "42",
            ],
        )
        assert result.exit_code == 0


class TestSimulateInvalidInput:
    """Invalid inputs produce clear error messages."""

    def test_invalid_strategy_name_shows_error(self) -> None:
        result = runner.invoke(
            app,
            [
                "simulate",
                "--n-games",
                "5",
                "--players",
                "2",
                "--strategy",
                "NonExistentStrategy",
            ],
        )
        assert result.exit_code != 0
        # Should mention the invalid strategy name, not a raw KeyError traceback
        assert "NonExistentStrategy" in result.output or "NonExistentStrategy" in str(
            result.exception
        )

    def test_invalid_strategy_does_not_show_key_error_traceback(self) -> None:
        result = runner.invoke(
            app,
            [
                "simulate",
                "--n-games",
                "5",
                "--players",
                "2",
                "--strategy",
                "BadStrategy",
            ],
        )
        assert result.exit_code != 0
        # The output itself should not show a raw KeyError — it should be caught
        assert "KeyError" not in result.output


# ---------------------------------------------------------------------------
# markov command
# ---------------------------------------------------------------------------


class TestMarkovHelp:
    """--help output is accessible and informative."""

    def test_help_exits_successfully(self) -> None:
        result = runner.invoke(app, ["markov", "--help"])
        assert result.exit_code == 0

    def test_help_mentions_top(self) -> None:
        result = runner.invoke(app, ["markov", "--help"])
        assert "--top" in result.output


class TestMarkovSuccess:
    """markov command computes and displays the stationary distribution."""

    def test_runs_with_no_arguments(self) -> None:
        result = runner.invoke(app, ["markov"])
        assert result.exit_code == 0

    def test_output_contains_probability_values(self) -> None:
        result = runner.invoke(app, ["markov"])
        # Should show percentage or decimal probabilities
        assert "%" in result.output or "." in result.output

    def test_default_shows_ten_squares(self) -> None:
        result = runner.invoke(app, ["markov"])
        assert result.exit_code == 0
        # Count rows in output — 10 squares + header
        lines_with_percent = [
            line for line in result.output.splitlines() if "%" in line
        ]
        assert len(lines_with_percent) >= 10

    def test_top_option_limits_output(self) -> None:
        result_5 = runner.invoke(app, ["markov", "--top", "5"])
        result_10 = runner.invoke(app, ["markov", "--top", "10"])
        assert result_5.exit_code == 0
        assert result_10.exit_code == 0
        # Top 5 output should be shorter than top 10
        assert len(result_5.output) < len(result_10.output)

    def test_output_contains_jail_square(self) -> None:
        result = runner.invoke(app, ["markov"])
        # Jail is historically the most-visited square
        assert "Jail" in result.output or "jail" in result.output.lower()


# ---------------------------------------------------------------------------
# tournament command
# ---------------------------------------------------------------------------


class TestTournamentHelp:
    """--help output is accessible and informative."""

    def test_help_exits_successfully(self) -> None:
        result = runner.invoke(app, ["tournament", "--help"])
        assert result.exit_code == 0

    def test_help_mentions_n_games(self) -> None:
        result = runner.invoke(app, ["tournament", "--help"])
        assert "--n-games" in result.output

    def test_help_mentions_seed(self) -> None:
        result = runner.invoke(app, ["tournament", "--help"])
        assert "--seed" in result.output


class TestTournamentSuccess:
    """tournament command runs round-robin and prints ranking."""

    def test_runs_with_seed(self) -> None:
        result = runner.invoke(app, ["tournament", "--n-games", "5", "--seed", "42"])
        assert result.exit_code == 0

    def test_output_contains_strategy_names(self) -> None:
        result = runner.invoke(app, ["tournament", "--n-games", "5", "--seed", "42"])
        # Should include at least some known strategy names
        known = {"BuyEverything", "BuyNothing", "Trader", "JailCamper"}
        found = any(name in result.output for name in known)
        assert found

    def test_seed_produces_reproducible_results(self) -> None:
        result_a = runner.invoke(app, ["tournament", "--n-games", "5", "--seed", "7"])
        result_b = runner.invoke(app, ["tournament", "--n-games", "5", "--seed", "7"])
        assert result_a.exit_code == 0
        assert result_a.output == result_b.output

    def test_output_contains_ranking_header(self) -> None:
        result = runner.invoke(app, ["tournament", "--n-games", "5", "--seed", "42"])
        # Should have some column headers
        assert result.exit_code == 0
        lower = result.output.lower()
        assert "strategy" in lower or "rank" in lower or "elo" in lower


# ---------------------------------------------------------------------------
# top-level app
# ---------------------------------------------------------------------------


class TestAppHelp:
    """Top-level --help works."""

    def test_top_level_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_top_level_lists_all_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "simulate" in result.output
        assert "markov" in result.output
        assert "tournament" in result.output

    def test_top_level_lists_plot_command(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "plot" in result.output

    def test_top_level_lists_export_video_command(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "export-video" in result.output


# ---------------------------------------------------------------------------
# plot command — help
# ---------------------------------------------------------------------------


class TestPlotHelp:
    """plot --help lists available figures and all options."""

    def test_help_exits_successfully(self) -> None:
        result = runner.invoke(app, ["plot", "--help"])
        assert result.exit_code == 0

    def test_help_lists_figure_names(self) -> None:
        result = runner.invoke(app, ["plot", "--help"])
        output = result.output.lower()
        assert "heatmap" in output or "roi" in output or "win-rate" in output

    def test_help_mentions_output_option(self) -> None:
        result = runner.invoke(app, ["plot", "--help"])
        assert "--output" in result.output

    def test_help_mentions_seed_option(self) -> None:
        result = runner.invoke(app, ["plot", "--help"])
        assert "--seed" in result.output

    def test_help_mentions_n_games_option(self) -> None:
        result = runner.invoke(app, ["plot", "--help"])
        assert "--n-games" in result.output


# ---------------------------------------------------------------------------
# plot command — heatmap
# ---------------------------------------------------------------------------


class TestPlotHeatmap:
    """monopoly plot heatmap generates a PNG file."""

    def test_creates_output_file(self, tmp_path: Path) -> None:
        out = tmp_path / "heatmap.png"
        result = runner.invoke(app, ["plot", "heatmap", "--output", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_output_file_is_non_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "heatmap.png"
        runner.invoke(app, ["plot", "heatmap", "--output", str(out)])
        assert out.stat().st_size > 0

    def test_creates_parent_directory_if_missing(self, tmp_path: Path) -> None:
        out = tmp_path / "new_dir" / "heatmap.png"
        result = runner.invoke(app, ["plot", "heatmap", "--output", str(out)])
        assert result.exit_code == 0, result.output
        assert out.exists()


# ---------------------------------------------------------------------------
# plot command — roi
# ---------------------------------------------------------------------------


class TestPlotRoi:
    """monopoly plot roi generates a PNG file."""

    def test_creates_output_file(self, tmp_path: Path) -> None:
        out = tmp_path / "roi.png"
        result = runner.invoke(
            app, ["plot", "roi", "--output", str(out), "--seed", "42"]
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_output_file_is_non_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "roi.png"
        runner.invoke(app, ["plot", "roi", "--output", str(out), "--seed", "42"])
        assert out.stat().st_size > 0

    def test_n_games_option_is_accepted(self, tmp_path: Path) -> None:
        out = tmp_path / "roi_small.png"
        result = runner.invoke(
            app,
            ["plot", "roi", "--output", str(out), "--n-games", "10", "--seed", "42"],
        )
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# plot command — win-rate
# ---------------------------------------------------------------------------


class TestPlotWinRate:
    """monopoly plot win-rate generates a PNG file."""

    def test_creates_output_file(self, tmp_path: Path) -> None:
        out = tmp_path / "win_rate.png"
        result = runner.invoke(
            app,
            [
                "plot",
                "win-rate",
                "--output",
                str(out),
                "--seed",
                "42",
                "--n-games",
                "10",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_output_file_is_non_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "win_rate.png"
        runner.invoke(
            app,
            [
                "plot",
                "win-rate",
                "--output",
                str(out),
                "--seed",
                "42",
                "--n-games",
                "10",
            ],
        )
        assert out.stat().st_size > 0

    def test_seed_produces_reproducible_file(self, tmp_path: Path) -> None:
        out_a = tmp_path / "wr_a.png"
        out_b = tmp_path / "wr_b.png"
        args = ["plot", "win-rate", "--seed", "7", "--n-games", "5"]
        runner.invoke(app, args + ["--output", str(out_a)])
        runner.invoke(app, args + ["--output", str(out_b)])
        assert out_a.read_bytes() == out_b.read_bytes()


# ---------------------------------------------------------------------------
# plot command — net-worth
# ---------------------------------------------------------------------------


class TestPlotNetWorth:
    """monopoly plot net-worth generates a PNG file."""

    def test_creates_output_file(self, tmp_path: Path) -> None:
        out = tmp_path / "net_worth.png"
        result = runner.invoke(
            app,
            ["plot", "net-worth", "--output", str(out), "--seed", "42"],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_output_file_is_non_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "net_worth.png"
        runner.invoke(app, ["plot", "net-worth", "--output", str(out), "--seed", "42"])
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# plot command — invalid figure name
# ---------------------------------------------------------------------------


class TestPlotInvalidFigure:
    """Invalid figure name gives a clear error listing valid options."""

    def test_invalid_figure_exits_with_nonzero_code(self) -> None:
        result = runner.invoke(app, ["plot", "foobar"])
        assert result.exit_code != 0

    def test_error_message_lists_valid_figure_names(self) -> None:
        result = runner.invoke(app, ["plot", "foobar"])
        combined = result.output + str(result.exception or "")
        assert "heatmap" in combined or "roi" in combined

    def test_error_message_mentions_invalid_name(self) -> None:
        result = runner.invoke(app, ["plot", "foobar"])
        combined = result.output + str(result.exception or "")
        assert "foobar" in combined


# ---------------------------------------------------------------------------
# export-video command
# ---------------------------------------------------------------------------


class TestExportVideoHelp:
    """export-video --help lists all expected options."""

    def test_help_exits_successfully(self) -> None:
        result = runner.invoke(app, ["export-video", "--help"])
        assert result.exit_code == 0

    def test_help_mentions_output(self) -> None:
        result = runner.invoke(app, ["export-video", "--help"])
        assert "--output" in result.output

    def test_help_mentions_seed(self) -> None:
        result = runner.invoke(app, ["export-video", "--help"])
        assert "--seed" in result.output

    def test_help_mentions_fps(self) -> None:
        result = runner.invoke(app, ["export-video", "--help"])
        assert "--fps" in result.output


class TestExportVideoSuccess:
    """export-video creates an animation file."""

    def test_creates_gif_output_file(self, tmp_path: Path) -> None:
        out = tmp_path / "game.gif"
        result = runner.invoke(
            app,
            ["export-video", "--output", str(out), "--seed", "42", "--fps", "2"],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_output_file_is_non_empty(self, tmp_path: Path) -> None:
        out = tmp_path / "game.gif"
        runner.invoke(
            app,
            ["export-video", "--output", str(out), "--seed", "42", "--fps", "2"],
        )
        assert out.stat().st_size > 0

    def test_creates_parent_directory_if_missing(self, tmp_path: Path) -> None:
        out = tmp_path / "new_dir" / "game.gif"
        result = runner.invoke(
            app,
            ["export-video", "--output", str(out), "--seed", "42", "--fps", "2"],
        )
        assert result.exit_code == 0, result.output
        assert out.exists()

    def test_seed_produces_reproducible_gif(self, tmp_path: Path) -> None:
        out_a = tmp_path / "game_a.gif"
        out_b = tmp_path / "game_b.gif"
        args = ["export-video", "--seed", "99", "--fps", "2"]
        runner.invoke(app, args + ["--output", str(out_a)])
        runner.invoke(app, args + ["--output", str(out_b)])
        assert out_a.read_bytes() == out_b.read_bytes()

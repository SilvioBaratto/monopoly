"""Tests for src/monopoly/plots.py — written BEFORE implementation (TDD).

Covers: figure dimensions, palette completeness, Italian label coverage,
factory function signatures, save helper behaviour, and board heatmap.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
import numpy
import pytest

matplotlib.use("Agg")  # headless — no display required

import matplotlib.pyplot as plt

from monopoly.board import Board
import pandas as pd

from monopoly.game import GameHistory
from monopoly.plots import (
    COLOR_GROUPS,
    DPI,
    FIGURE_SIZE_PX,
    ITALIAN_LABELS,
    PALETTE,
    animate_sample_game,
    create_figure,
    plot_board_heatmap,
    plot_net_worth,
    plot_roi_bars,
    plot_win_rate_curves,
    save_figure,
    _sample_animation_frames,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_COLOR_GROUPS = {
    "brown",
    "light_blue",
    "pink",
    "orange",
    "red",
    "yellow",
    "green",
    "dark_blue",
}


class TestConstants:
    """FIGURE_SIZE_PX and DPI match the spec."""

    def test_figure_size_is_1920_by_1080(self) -> None:
        assert FIGURE_SIZE_PX == (1920, 1080)

    def test_dpi_is_150(self) -> None:
        assert DPI == 150

    def test_color_groups_contains_all_eight_standard_groups(self) -> None:
        assert EXPECTED_COLOR_GROUPS.issubset(set(COLOR_GROUPS))

    def test_color_groups_includes_railroad_and_utility(self) -> None:
        assert "railroad" in COLOR_GROUPS
        assert "utility" in COLOR_GROUPS


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------


class TestPalette:
    """Colorblind-safe palette covers all required groups."""

    def test_palette_has_at_least_ten_distinct_entries(self) -> None:
        assert len(PALETTE) >= 10

    def test_all_palette_values_are_valid_hex_colors(self) -> None:
        import re

        hex_pattern = re.compile(r"^#[0-9A-Fa-f]{6}$")
        for group, color in PALETTE.items():
            assert hex_pattern.match(color), (
                f"Group '{group}' has invalid hex color '{color}'"
            )

    def test_palette_covers_all_color_groups(self) -> None:
        for group in COLOR_GROUPS:
            assert group in PALETTE, f"Missing palette entry for '{group}'"

    def test_palette_entries_are_distinct(self) -> None:
        values = list(PALETTE.values())
        assert len(values) == len(set(v.upper() for v in values)), (
            "Duplicate hex colors found in PALETTE"
        )


# ---------------------------------------------------------------------------
# Italian labels
# ---------------------------------------------------------------------------


class TestItalianLabels:
    """ITALIAN_LABELS covers all groups and board categories."""

    def test_italian_labels_covers_all_color_groups(self) -> None:
        for group in EXPECTED_COLOR_GROUPS:
            assert group in ITALIAN_LABELS, (
                f"Italian label missing for color group '{group}'"
            )

    def test_italian_labels_covers_railroad_and_utility(self) -> None:
        assert "railroad" in ITALIAN_LABELS
        assert "utility" in ITALIAN_LABELS

    def test_italian_label_for_brown_is_marrone(self) -> None:
        assert ITALIAN_LABELS["brown"] == "Marrone"

    def test_italian_label_for_light_blue_is_azzurro(self) -> None:
        assert ITALIAN_LABELS["light_blue"] == "Azzurro"

    def test_italian_label_for_pink_is_rosa(self) -> None:
        assert ITALIAN_LABELS["pink"] == "Rosa"

    def test_italian_label_for_orange_is_arancione(self) -> None:
        assert ITALIAN_LABELS["orange"] == "Arancione"

    def test_italian_label_for_red_is_rosso(self) -> None:
        assert ITALIAN_LABELS["red"] == "Rosso"

    def test_italian_label_for_yellow_is_giallo(self) -> None:
        assert ITALIAN_LABELS["yellow"] == "Giallo"

    def test_italian_label_for_green_is_verde(self) -> None:
        assert ITALIAN_LABELS["green"] == "Verde"

    def test_italian_label_for_dark_blue_is_blu_scuro(self) -> None:
        assert ITALIAN_LABELS["dark_blue"] == "Blu scuro"

    def test_italian_label_for_railroad_is_ferrovie(self) -> None:
        assert ITALIAN_LABELS["railroad"] == "Ferrovie"

    def test_italian_label_for_utility_is_societa(self) -> None:
        assert ITALIAN_LABELS["utility"] == "Società"

    def test_italian_labels_covers_board_categories(self) -> None:
        expected_categories = {"go", "jail", "free_parking", "go_to_jail", "tax"}
        for cat in expected_categories:
            assert cat in ITALIAN_LABELS, (
                f"Italian label missing for board category '{cat}'"
            )


# ---------------------------------------------------------------------------
# create_figure
# ---------------------------------------------------------------------------


class TestCreateFigure:
    """create_figure() returns a correctly sized (fig, ax) pair."""

    def setup_method(self) -> None:
        """Close any open figures before each test."""
        plt.close("all")

    def teardown_method(self) -> None:
        """Close figures after each test to avoid resource leaks."""
        plt.close("all")

    def test_returns_tuple_of_figure_and_axes(self) -> None:
        fig, ax = create_figure()
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)

    def test_figure_width_matches_spec_at_given_dpi(self) -> None:
        fig, _ = create_figure()
        expected_width = FIGURE_SIZE_PX[0] / DPI
        assert abs(fig.get_figwidth() - expected_width) < 0.01

    def test_figure_height_matches_spec_at_given_dpi(self) -> None:
        fig, _ = create_figure()
        expected_height = FIGURE_SIZE_PX[1] / DPI
        assert abs(fig.get_figheight() - expected_height) < 0.01

    def test_figure_dpi_matches_constant(self) -> None:
        fig, _ = create_figure()
        assert fig.get_dpi() == DPI

    def test_each_call_returns_a_new_independent_figure(self) -> None:
        fig1, _ = create_figure()
        fig2, _ = create_figure()
        assert fig1 is not fig2


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------


class TestSaveFigure:
    """save_figure() writes a PNG at the correct resolution."""

    def setup_method(self) -> None:
        plt.close("all")

    def teardown_method(self) -> None:
        plt.close("all")

    def test_creates_png_file_at_given_path(self, tmp_path: Path) -> None:
        fig, _ = create_figure()
        out = tmp_path / "test_output.png"
        save_figure(fig, out)
        assert out.exists()

    def test_saved_png_has_correct_pixel_dimensions(self, tmp_path: Path) -> None:
        from PIL import Image

        fig, _ = create_figure()
        out = tmp_path / "dims.png"
        save_figure(fig, out)
        with Image.open(out) as img:
            assert img.size == FIGURE_SIZE_PX

    def test_save_accepts_string_path(self, tmp_path: Path) -> None:
        fig, _ = create_figure()
        out = str(tmp_path / "str_path.png")
        save_figure(fig, out)
        assert os.path.exists(out)


# ---------------------------------------------------------------------------
# Fixtures shared by heatmap tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def italian_board() -> Board:
    """Board with Italian square names."""
    return Board(locale="it")


@pytest.fixture()
def uniform_distribution() -> numpy.ndarray:
    """Flat 40-element probability distribution (each square = 1/40)."""
    return numpy.full(40, 1.0 / 40)


@pytest.fixture()
def realistic_distribution() -> numpy.ndarray:
    """Approximate stationary distribution from the Markov chain."""
    from monopoly.markov import (
        build_transition_matrix,
        compute_stationary_distribution,
        get_square_probabilities,
    )

    matrix = build_transition_matrix()
    dist43 = compute_stationary_distribution(matrix)
    probs = get_square_probabilities(Board(), dist43)
    arr = numpy.zeros(40)
    for pos, (_name, prob) in probs.items():
        arr[pos] = prob
    return arr


# ---------------------------------------------------------------------------
# plot_board_heatmap
# ---------------------------------------------------------------------------


class TestPlotBoardHeatmap:
    """plot_board_heatmap() produces a valid 1920×1080 PNG board visualization."""

    def setup_method(self) -> None:
        plt.close("all")

    def teardown_method(self) -> None:
        plt.close("all")

    # ------------------------------------------------------------------
    # Happy path
    # ------------------------------------------------------------------

    def test_produces_a_file_at_the_given_path(
        self,
        tmp_path: Path,
        uniform_distribution: numpy.ndarray,
        italian_board: Board,
    ) -> None:
        """Output PNG file is created at the specified path."""
        out = tmp_path / "heatmap.png"
        plot_board_heatmap(uniform_distribution, italian_board, out)
        assert out.exists()

    def test_output_file_is_non_empty(
        self,
        tmp_path: Path,
        uniform_distribution: numpy.ndarray,
        italian_board: Board,
    ) -> None:
        """Output file is not empty."""
        out = tmp_path / "heatmap.png"
        plot_board_heatmap(uniform_distribution, italian_board, out)
        assert out.stat().st_size > 0

    def test_output_is_readable_by_pil(
        self,
        tmp_path: Path,
        uniform_distribution: numpy.ndarray,
        italian_board: Board,
    ) -> None:
        """PIL can open the output file without errors."""
        from PIL import Image

        out = tmp_path / "heatmap.png"
        plot_board_heatmap(uniform_distribution, italian_board, out)
        with Image.open(out) as img:
            img.verify()

    def test_output_dimensions_are_1920_by_1080(
        self,
        tmp_path: Path,
        uniform_distribution: numpy.ndarray,
        italian_board: Board,
    ) -> None:
        """Output PNG has exactly 1920×1080 pixels."""
        from PIL import Image

        out = tmp_path / "dims.png"
        plot_board_heatmap(uniform_distribution, italian_board, out)
        with Image.open(out) as img:
            assert img.size == (1920, 1080)

    def test_accepts_string_output_path(
        self,
        tmp_path: Path,
        uniform_distribution: numpy.ndarray,
        italian_board: Board,
    ) -> None:
        """Output path may be given as a plain string."""
        out = str(tmp_path / "heatmap_str.png")
        plot_board_heatmap(uniform_distribution, italian_board, out)
        assert os.path.exists(out)

    def test_works_with_realistic_markov_distribution(
        self,
        tmp_path: Path,
        realistic_distribution: numpy.ndarray,
        italian_board: Board,
    ) -> None:
        """Function handles a realistic (non-uniform) stationary distribution."""
        out = tmp_path / "realistic.png"
        plot_board_heatmap(realistic_distribution, italian_board, out)
        assert out.exists()

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_raises_value_error_for_distribution_shorter_than_40(
        self,
        italian_board: Board,
    ) -> None:
        """ValueError when distribution has fewer than 40 elements."""
        short = numpy.full(39, 1.0 / 39)
        with pytest.raises(ValueError, match="40"):
            plot_board_heatmap(short, italian_board, "/tmp/should_not_exist.png")

    def test_raises_value_error_for_distribution_longer_than_40(
        self,
        italian_board: Board,
    ) -> None:
        """ValueError when distribution has more than 40 elements."""
        long = numpy.full(43, 1.0 / 43)
        with pytest.raises(ValueError, match="40"):
            plot_board_heatmap(long, italian_board, "/tmp/should_not_exist.png")

    def test_raises_value_error_for_negative_probabilities(
        self,
        italian_board: Board,
    ) -> None:
        """ValueError when distribution contains a negative value."""
        bad = numpy.full(40, 0.025)
        bad[5] = -0.01
        with pytest.raises(ValueError, match="negative"):
            plot_board_heatmap(bad, italian_board, "/tmp/should_not_exist.png")


# ---------------------------------------------------------------------------
# Fixtures shared by ROI bar chart tests
# ---------------------------------------------------------------------------

_ROI_COLUMNS = ["base", "monopoly", "1h", "2h", "3h", "4h", "hotel"]
_COLOR_GROUP_ROWS = [
    "brown",
    "light_blue",
    "pink",
    "orange",
    "red",
    "yellow",
    "green",
    "dark_blue",
]


@pytest.fixture()
def roi_fixture_table() -> pd.DataFrame:
    """Fixture ROI table matching the shape of metrics.roi_ranking_table().

    8 color group rows have all 7 development levels populated.
    Railroad row has NaN for '3h', '4h', and 'hotel' (no houses on railroads).
    """
    data: dict[str, dict[str, float]] = {}
    for i, group in enumerate(_COLOR_GROUP_ROWS):
        data[group] = {
            col: 0.1 * (i + 1) * (j + 1) for j, col in enumerate(_ROI_COLUMNS)
        }
    data["railroad"] = {
        "base": 0.05,
        "monopoly": 0.20,
        "1h": 0.30,
        "2h": 0.45,
        "3h": float("nan"),
        "4h": float("nan"),
        "hotel": float("nan"),
    }
    return pd.DataFrame.from_dict(data, orient="index", columns=_ROI_COLUMNS)


# ---------------------------------------------------------------------------
# plot_roi_bars
# ---------------------------------------------------------------------------


class TestPlotRoiBars:
    """plot_roi_bars() produces a valid grouped bar chart PNG."""

    def setup_method(self) -> None:
        plt.close("all")

    def teardown_method(self) -> None:
        plt.close("all")

    def test_produces_a_png_at_given_path(
        self, tmp_path: Path, roi_fixture_table: pd.DataFrame
    ) -> None:
        """Output PNG file is created at the specified path."""
        out = tmp_path / "roi_bars.png"
        plot_roi_bars(roi_fixture_table, out)
        assert out.exists()

    def test_output_file_is_non_empty(
        self, tmp_path: Path, roi_fixture_table: pd.DataFrame
    ) -> None:
        """Output file is not empty."""
        out = tmp_path / "roi_bars.png"
        plot_roi_bars(roi_fixture_table, out)
        assert out.stat().st_size > 0

    def test_figure_has_seven_bar_containers(
        self, tmp_path: Path, roi_fixture_table: pd.DataFrame
    ) -> None:
        """Figure contains exactly 7 BarContainer objects — one per development level."""
        out = tmp_path / "roi_bars_containers.png"
        # Capture the figure before it is closed by inspecting the last active figure
        plot_roi_bars(roi_fixture_table, out)
        # Re-generate capturing the containers via a patched save
        import matplotlib.pyplot as pyplot

        pyplot.close("all")

        from monopoly import plots as plots_mod

        containers: list = []
        original_save = plots_mod.save_figure

        def capturing_save(fig: plt.Figure, path: object) -> None:
            containers.extend(fig.axes[0].containers)
            original_save(fig, path)

        plots_mod.save_figure = capturing_save  # type: ignore[assignment]
        try:
            plot_roi_bars(roi_fixture_table, out)
        finally:
            plots_mod.save_figure = original_save

        assert len(containers) == 7, f"Expected 7 BarContainers, got {len(containers)}"

    def test_accepts_string_output_path(
        self, tmp_path: Path, roi_fixture_table: pd.DataFrame
    ) -> None:
        """Output path may be given as a plain string."""
        out = str(tmp_path / "roi_bars_str.png")
        plot_roi_bars(roi_fixture_table, out)
        assert os.path.exists(out)

    def test_nan_railroad_bars_do_not_raise(
        self, tmp_path: Path, roi_fixture_table: pd.DataFrame
    ) -> None:
        """NaN railroad values for house levels are silently skipped (no crash)."""
        out = tmp_path / "roi_railroad_nan.png"
        # Should complete without any exception
        plot_roi_bars(roi_fixture_table, out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Fixtures shared by win rate curve tests
# ---------------------------------------------------------------------------

_WIN_RATE_STRATEGIES = [
    "BuyEverything",
    "BuyNothing",
    "ColorTargeted",
    "ThreeHousesRush",
    "JailCamper",
    "Trader",
]
_WIN_RATE_PLAYER_COUNTS = [2, 3, 4, 5, 6]


@pytest.fixture()
def win_prob_fixture_df() -> pd.DataFrame:
    """Fixture win probability DataFrame matching metrics.win_probability_table() output.

    One row per (strategy, n_players) combination; all six strategies across
    five player counts. Significant flag alternates to cover both code paths.
    """
    rows = []
    for i, strategy in enumerate(_WIN_RATE_STRATEGIES):
        for j, n_players in enumerate(_WIN_RATE_PLAYER_COUNTS):
            baseline = 1.0 / n_players
            win_rate = baseline + 0.02 * (i % 3 - 1)
            rows.append(
                {
                    "strategy": strategy,
                    "n_players": n_players,
                    "win_rate": win_rate,
                    "ci_lower": win_rate - 0.03,
                    "ci_upper": win_rate + 0.03,
                    "baseline": baseline,
                    "significant": bool((i + j) % 2 == 0),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# plot_win_rate_curves
# ---------------------------------------------------------------------------


class TestPlotWinRateCurves:
    """plot_win_rate_curves() produces a valid win rate line chart PNG."""

    def setup_method(self) -> None:
        plt.close("all")

    def teardown_method(self) -> None:
        plt.close("all")

    def test_produces_a_png_at_given_path(
        self, tmp_path: Path, win_prob_fixture_df: pd.DataFrame
    ) -> None:
        """Output PNG file is created at the specified path."""
        out = tmp_path / "win_rate_curves.png"
        plot_win_rate_curves(win_prob_fixture_df, out)
        assert out.exists()

    def test_output_file_is_non_empty(
        self, tmp_path: Path, win_prob_fixture_df: pd.DataFrame
    ) -> None:
        """Output file is not empty."""
        out = tmp_path / "win_rate_curves.png"
        plot_win_rate_curves(win_prob_fixture_df, out)
        assert out.stat().st_size > 0

    def test_axes_contains_one_line_per_strategy_plus_baseline(
        self, tmp_path: Path, win_prob_fixture_df: pd.DataFrame
    ) -> None:
        """Axes has exactly len(strategies) + 1 lines (one per strategy + baseline)."""
        out = tmp_path / "win_rate_lines.png"
        captured_lines: list = []

        from monopoly import plots as plots_mod

        original_save = plots_mod.save_figure

        def capturing_save(fig: plt.Figure, path: object) -> None:
            captured_lines.extend(fig.axes[0].get_lines())
            original_save(fig, path)

        plots_mod.save_figure = capturing_save  # type: ignore[assignment]
        try:
            plot_win_rate_curves(win_prob_fixture_df, out)
        finally:
            plots_mod.save_figure = original_save

        n_strategies = win_prob_fixture_df["strategy"].nunique()
        expected = n_strategies + 1  # strategies + 1 baseline
        assert len(captured_lines) == expected, (
            f"Expected {expected} lines, got {len(captured_lines)}"
        )

    def test_legend_contains_all_strategy_names(
        self, tmp_path: Path, win_prob_fixture_df: pd.DataFrame
    ) -> None:
        """Legend texts include every strategy name present in the DataFrame."""
        out = tmp_path / "win_rate_legend.png"
        captured_legend_texts: list[str] = []

        from monopoly import plots as plots_mod

        original_save = plots_mod.save_figure

        def capturing_save(fig: plt.Figure, path: object) -> None:
            legend = fig.axes[0].get_legend()
            if legend is not None:
                captured_legend_texts.extend(t.get_text() for t in legend.get_texts())
            original_save(fig, path)

        plots_mod.save_figure = capturing_save  # type: ignore[assignment]
        try:
            plot_win_rate_curves(win_prob_fixture_df, out)
        finally:
            plots_mod.save_figure = original_save

        strategies = set(win_prob_fixture_df["strategy"].unique())
        for strategy in strategies:
            assert strategy in captured_legend_texts, (
                f"Strategy '{strategy}' missing from legend: {captured_legend_texts}"
            )

    def test_raises_value_error_on_empty_dataframe(self, tmp_path: Path) -> None:
        """ValueError with descriptive message when input DataFrame is empty."""
        empty_df = pd.DataFrame(
            columns=[
                "strategy",
                "n_players",
                "win_rate",
                "ci_lower",
                "ci_upper",
                "baseline",
                "significant",
            ]
        )
        out = tmp_path / "should_not_exist.png"
        with pytest.raises(ValueError, match="empty"):
            plot_win_rate_curves(empty_df, out)


# ---------------------------------------------------------------------------
# plot_net_worth
# ---------------------------------------------------------------------------


@pytest.fixture
def net_worth_fixture() -> dict[str, list[int]]:
    """Two players: one survives all turns, one goes bankrupt mid-game."""
    return {
        "Alice": [1500, 1600, 1750, 1400, 1800, 1900],
        "Bob": [1500, 1300, 1100, 0, 0, 0],  # bankrupt at turn 3
    }


class TestPlotNetWorth:
    """plot_net_worth() produces a valid net-worth line chart PNG."""

    def setup_method(self) -> None:
        plt.close("all")

    def teardown_method(self) -> None:
        plt.close("all")

    # -- basic output --------------------------------------------------------

    def test_output_file_exists(
        self, tmp_path: Path, net_worth_fixture: dict[str, list[int]]
    ) -> None:
        """Output PNG file is created at the specified path."""
        out = tmp_path / "net_worth.png"
        plot_net_worth(net_worth_fixture, list(net_worth_fixture.keys()), out)
        assert out.exists()

    def test_output_file_is_non_empty(
        self, tmp_path: Path, net_worth_fixture: dict[str, list[int]]
    ) -> None:
        """Output file has non-zero size."""
        out = tmp_path / "net_worth.png"
        plot_net_worth(net_worth_fixture, list(net_worth_fixture.keys()), out)
        assert out.stat().st_size > 0

    def test_output_has_valid_png_magic_bytes(
        self, tmp_path: Path, net_worth_fixture: dict[str, list[int]]
    ) -> None:
        """Output file starts with the PNG magic bytes (\\x89PNG)."""
        out = tmp_path / "net_worth.png"
        plot_net_worth(net_worth_fixture, list(net_worth_fixture.keys()), out)
        with out.open("rb") as fh:
            header = fh.read(4)
        assert header == b"\x89PNG"

    def test_output_dimensions_are_1920_by_1080(
        self, tmp_path: Path, net_worth_fixture: dict[str, list[int]]
    ) -> None:
        """Saved PNG has exactly 1920 × 1080 pixels."""
        out = tmp_path / "net_worth.png"
        plot_net_worth(net_worth_fixture, list(net_worth_fixture.keys()), out)
        from PIL import Image

        with Image.open(out) as img:
            assert img.size == (1920, 1080)

    # -- chart content -------------------------------------------------------

    def test_axes_contains_one_line_per_player_plus_zero_reference(
        self, tmp_path: Path, net_worth_fixture: dict[str, list[int]]
    ) -> None:
        """Axes has one line per player plus the horizontal $0 reference line."""
        out = tmp_path / "net_worth_lines.png"
        captured_lines: list = []

        from monopoly import plots as plots_mod

        original_save = plots_mod.save_figure

        def capturing_save(fig: plt.Figure, path: object) -> None:
            captured_lines.extend(fig.axes[0].get_lines())
            original_save(fig, path)

        plots_mod.save_figure = capturing_save  # type: ignore[assignment]
        try:
            plot_net_worth(net_worth_fixture, list(net_worth_fixture.keys()), out)
        finally:
            plots_mod.save_figure = original_save

        n_players = len(net_worth_fixture)
        expected = n_players + 1  # player lines + zero reference
        assert len(captured_lines) == expected, (
            f"Expected {expected} lines, got {len(captured_lines)}"
        )

    def test_bankrupt_player_line_ends_at_last_nonzero_turn(
        self, tmp_path: Path, net_worth_fixture: dict[str, list[int]]
    ) -> None:
        """Bankrupt player's line ends at their last non-zero net worth turn."""
        out = tmp_path / "net_worth_bankrupt.png"
        captured_x_data: dict[str, list] = {}

        from monopoly import plots as plots_mod

        original_save = plots_mod.save_figure

        def capturing_save(fig: plt.Figure, path: object) -> None:
            for line in fig.axes[0].get_lines():
                label = line.get_label()
                if label.startswith("_"):
                    continue
                captured_x_data[label] = list(line.get_xdata())
            original_save(fig, path)

        plots_mod.save_figure = capturing_save  # type: ignore[assignment]
        try:
            player_names = list(net_worth_fixture.keys())
            plot_net_worth(net_worth_fixture, player_names, out)
        finally:
            plots_mod.save_figure = original_save

        # Bob goes bankrupt at index 3 (first 0) → last non-zero is index 2
        assert "Bob" in captured_x_data
        assert max(captured_x_data["Bob"]) == 2, (
            f"Expected Bob's line to end at turn 2, got x={captured_x_data['Bob']}"
        )

    def test_legend_contains_all_player_names(
        self, tmp_path: Path, net_worth_fixture: dict[str, list[int]]
    ) -> None:
        """Legend includes every player name passed in player_names."""
        out = tmp_path / "net_worth_legend.png"
        captured_legend_texts: list[str] = []

        from monopoly import plots as plots_mod

        original_save = plots_mod.save_figure

        def capturing_save(fig: plt.Figure, path: object) -> None:
            legend = fig.axes[0].get_legend()
            if legend is not None:
                captured_legend_texts.extend(t.get_text() for t in legend.get_texts())
            original_save(fig, path)

        plots_mod.save_figure = capturing_save  # type: ignore[assignment]
        try:
            player_names = list(net_worth_fixture.keys())
            plot_net_worth(net_worth_fixture, player_names, out)
        finally:
            plots_mod.save_figure = original_save

        for name in net_worth_fixture:
            assert name in captured_legend_texts, (
                f"Player '{name}' missing from legend: {captured_legend_texts}"
            )

    def test_axes_has_italian_labels(
        self, tmp_path: Path, net_worth_fixture: dict[str, list[int]]
    ) -> None:
        """X-axis label is 'Turno' and y-axis label starts with 'Patrimonio'."""
        out = tmp_path / "net_worth_axes.png"
        captured: dict[str, str] = {}

        from monopoly import plots as plots_mod

        original_save = plots_mod.save_figure

        def capturing_save(fig: plt.Figure, path: object) -> None:
            ax = fig.axes[0]
            captured["xlabel"] = ax.get_xlabel()
            captured["ylabel"] = ax.get_ylabel()
            original_save(fig, path)

        plots_mod.save_figure = capturing_save  # type: ignore[assignment]
        try:
            plot_net_worth(net_worth_fixture, list(net_worth_fixture.keys()), out)
        finally:
            plots_mod.save_figure = original_save

        assert captured.get("xlabel") == "Turno", f"xlabel={captured.get('xlabel')}"
        assert captured.get("ylabel", "").startswith("Patrimonio"), (
            f"ylabel={captured.get('ylabel')}"
        )

    # -- edge cases ----------------------------------------------------------

    def test_handles_empty_history_gracefully(self, tmp_path: Path) -> None:
        """Empty net_worth_history dict produces a file without raising."""
        out = tmp_path / "net_worth_empty.png"
        plot_net_worth({}, [], out)  # must not raise
        assert out.exists()

    def test_handles_single_player(self, tmp_path: Path) -> None:
        """Single player produces a valid PNG without error."""
        out = tmp_path / "net_worth_single.png"
        history = {"Solo": [1500, 1600, 1700]}
        plot_net_worth(history, ["Solo"], out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_handles_zero_length_list(self, tmp_path: Path) -> None:
        """Player with an empty net worth list is skipped without error."""
        out = tmp_path / "net_worth_zero_len.png"
        history = {"Alice": [1500, 1600], "Bob": []}
        plot_net_worth(history, ["Alice", "Bob"], out)
        assert out.exists()


# ---------------------------------------------------------------------------
# Fixtures shared by animate_sample_game tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_game_history() -> GameHistory:
    """5-turn GameHistory for 2 players — minimal valid animation input."""
    names = ["Alice", "Bob"]
    return GameHistory(
        player_names=names,
        position_history=[
            {"Alice": (i * 3) % 40, "Bob": (i * 5) % 40} for i in range(5)
        ],
        net_worth_history=[
            {"Alice": 1500 + i * 100, "Bob": 1500 - i * 50} for i in range(5)
        ],
        ownership_history=[{1: "Alice", 3: "Bob"} for _ in range(5)],
    )


# ---------------------------------------------------------------------------
# animate_sample_game
# ---------------------------------------------------------------------------


class TestAnimateSampleGame:
    """animate_sample_game() produces a valid animation file."""

    def setup_method(self) -> None:
        plt.close("all")

    def teardown_method(self) -> None:
        plt.close("all")

    def test_gif_output_file_is_created(
        self, tmp_path: Path, mock_game_history: GameHistory
    ) -> None:
        """GIF file is created at the specified path."""
        out = tmp_path / "game.gif"
        animate_sample_game(mock_game_history, out)
        assert out.exists()

    def test_gif_output_is_non_zero_size(
        self, tmp_path: Path, mock_game_history: GameHistory
    ) -> None:
        """GIF file has non-zero size."""
        out = tmp_path / "game.gif"
        animate_sample_game(mock_game_history, out)
        assert out.stat().st_size > 0

    def test_gif_has_correct_extension(
        self, tmp_path: Path, mock_game_history: GameHistory
    ) -> None:
        """Return value has .gif extension when given a .gif path."""
        out = tmp_path / "game.gif"
        result = animate_sample_game(mock_game_history, out)
        assert result.suffix == ".gif"

    def test_returns_output_path_as_path_object(
        self, tmp_path: Path, mock_game_history: GameHistory
    ) -> None:
        """Function returns the output_path as a Path instance."""
        out = tmp_path / "game.gif"
        result = animate_sample_game(mock_game_history, out)
        assert isinstance(result, Path)
        assert result == out

    def test_accepts_string_output_path(
        self, tmp_path: Path, mock_game_history: GameHistory
    ) -> None:
        """Output path may be given as a plain string."""
        out = str(tmp_path / "game_str.gif")
        animate_sample_game(mock_game_history, out)
        assert Path(out).exists()

    def test_mp4_output_when_ffmpeg_available(
        self, tmp_path: Path, mock_game_history: GameHistory
    ) -> None:
        """MP4 file is created when ffmpeg writer is available."""
        import matplotlib.animation as anim_mod

        if "ffmpeg" not in anim_mod.writers.list():
            pytest.skip("ffmpeg writer not available")
        out = tmp_path / "game.mp4"
        animate_sample_game(mock_game_history, out)
        assert out.exists()
        assert out.stat().st_size > 0


# ---------------------------------------------------------------------------
# _sample_animation_frames
# ---------------------------------------------------------------------------


class TestSampleAnimationFrames:
    """_sample_animation_frames() correctly samples long games."""

    def test_short_game_returns_all_frame_indices(self) -> None:
        """Games with ≤ 200 turns return every frame index."""
        frames = _sample_animation_frames(100)
        assert frames == list(range(100))

    def test_200_turn_game_returns_all_200_frames(self) -> None:
        """Games with exactly 200 turns return all 200 frame indices."""
        frames = _sample_animation_frames(200)
        assert len(frames) == 200

    def test_long_game_is_sampled_to_at_most_200_frames(self) -> None:
        """Games longer than 200 turns produce at most 200 frames."""
        frames = _sample_animation_frames(500)
        assert len(frames) <= 200

    def test_very_long_game_is_sampled_to_at_most_200_frames(self) -> None:
        """1000-turn game is sampled down to at most 200 frames."""
        frames = _sample_animation_frames(1000)
        assert len(frames) <= 200

    def test_first_sampled_frame_is_turn_zero(self) -> None:
        """First sampled frame index is always 0."""
        frames = _sample_animation_frames(500)
        assert frames[0] == 0

    def test_sampled_frames_are_monotonically_increasing(self) -> None:
        """Sampled frame indices are in strictly ascending order."""
        frames = _sample_animation_frames(300)
        assert all(frames[i] < frames[i + 1] for i in range(len(frames) - 1))

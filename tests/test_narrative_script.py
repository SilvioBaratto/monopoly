"""Tests for scripts/come_vincere_al_monopoli.py — written BEFORE implementation (TDD).

Acceptance criteria (GitHub Issue #51):
  (a) Each section function is callable and returns without error on a minimal fixture.
  (b) The ``figures/`` directory is created when main() runs.
  (c) Importing the module does not trigger side effects.

All heavy computation (plots, simulations, animation) is patched so the tests
run in milliseconds.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers / shared fixtures
# ---------------------------------------------------------------------------

_SCRIPT_DIR = Path(__file__).parent.parent / "scripts"
_SCRIPT_PATH = _SCRIPT_DIR / "come_vincere_al_monopoli.py"


def _import_script() -> Any:
    """Import the narrative script as a module and return it."""
    spec = importlib.util.spec_from_file_location(
        "come_vincere_al_monopoli", _SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


@pytest.fixture(scope="module")
def script():
    """Return the imported narrative script module (loaded once per test session)."""
    return _import_script()


@pytest.fixture()
def figures_dir(tmp_path: Path) -> Path:
    """Provide a temporary directory to capture figure output."""
    return tmp_path / "figures"


# ---------------------------------------------------------------------------
# (c) No side effects on import
# ---------------------------------------------------------------------------


class TestNoImportSideEffects:
    """Importing the script must not execute any simulation or plotting code."""

    def test_script_file_exists(self) -> None:
        """The script file must be present at the expected path."""
        assert _SCRIPT_PATH.exists(), f"Script not found: {_SCRIPT_PATH}"

    def test_import_does_not_raise(self) -> None:
        """Importing the module must not raise any exception."""
        module = _import_script()
        assert module is not None

    def test_import_does_not_call_main(self) -> None:
        """Importing the module must not call main() automatically."""
        with patch("sys.stdout"):
            module = _import_script()
        # If main() had run, it would have tried to create figures and run
        # simulations — any AttributeError or FileNotFoundError would bubble up.
        # Reaching this line means the guard worked.
        assert hasattr(module, "main")


# ---------------------------------------------------------------------------
# (a) Section functions are callable without error
# ---------------------------------------------------------------------------


class TestSectionFunctions:
    """Each section_XX function must be callable and return None without error."""

    def _patch_all_io(self):
        """Context manager that patches plots, simulations, and print."""
        return patch.multiple(
            "come_vincere_al_monopoli",
            plot_board_heatmap=MagicMock(),
            plot_roi_bars=MagicMock(),
            plot_win_rate_curves=MagicMock(),
            plot_net_worth=MagicMock(),
            animate_sample_game=MagicMock(),
        )

    def test_section_01_intro_is_callable(self, script, capsys) -> None:
        """section_01_intro() must print Italian header and return None."""
        result = script.section_01_intro()
        assert result is None
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_section_02_dice_is_callable(self, script, capsys) -> None:
        """section_02_dice() must print dice distribution and return None."""
        result = script.section_02_dice()
        assert result is None
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_section_03_markov_is_callable(self, script, tmp_path) -> None:
        """section_03_markov() must call plot_board_heatmap and return None."""
        figures = tmp_path / "figures"
        figures.mkdir()
        with (
            patch.object(script, "plot_board_heatmap") as mock_plot,
            patch.object(script, "FIGURES_DIR", figures),
        ):
            result = script.section_03_markov()
        assert result is None
        mock_plot.assert_called_once()

    def test_section_04_roi_is_callable(self, script, tmp_path) -> None:
        """section_04_roi() must call plot_roi_bars and return None."""
        figures = tmp_path / "figures"
        figures.mkdir()
        with (
            patch.object(script, "plot_roi_bars") as mock_plot,
            patch.object(script, "FIGURES_DIR", figures),
        ):
            result = script.section_04_roi()
        assert result is None
        mock_plot.assert_called_once()

    def test_section_05_monte_carlo_is_callable(self, script, tmp_path) -> None:
        """section_05_monte_carlo() must call plot_win_rate_curves and return None."""
        figures = tmp_path / "figures"
        figures.mkdir()
        fake_df = pd.DataFrame(
            {
                "strategy": ["BuyEverything"],
                "n_players": [2],
                "win_rate": [0.5],
                "ci_lower": [0.4],
                "ci_upper": [0.6],
                "baseline": [0.5],
                "significant": [False],
            }
        )
        with (
            patch.object(script, "plot_win_rate_curves") as mock_plot,
            patch.object(script, "_build_win_prob_results", return_value={}),
            patch.object(script, "win_probability_table", return_value=fake_df),
            patch.object(script, "FIGURES_DIR", figures),
        ):
            result = script.section_05_monte_carlo()
        assert result is None
        mock_plot.assert_called_once()

    def test_section_06_sample_game_is_callable(self, script, tmp_path) -> None:
        """section_06_sample_game() must call plot_net_worth + animate_sample_game."""
        figures = tmp_path / "figures"
        figures.mkdir()
        fake_history = MagicMock()
        fake_history.player_names = ["Alice", "Bob"]
        fake_history.net_worth_history = [{"Alice": 1500, "Bob": 1500}]
        fake_history.position_history = [{"Alice": 0, "Bob": 0}]
        fake_history.ownership_history = [{}]
        with (
            patch.object(script, "plot_net_worth") as mock_nw,
            patch.object(script, "animate_sample_game") as mock_anim,
            patch.object(script, "_run_sample_game", return_value=fake_history),
            patch.object(script, "FIGURES_DIR", figures),
        ):
            result = script.section_06_sample_game()
        assert result is None
        mock_nw.assert_called_once()
        mock_anim.assert_called_once()

    def test_section_07_tournament_is_callable(self, script, capsys) -> None:
        """section_07_tournament() must print ranking table and return None."""
        fake_result = MagicMock()
        fake_df = pd.DataFrame(
            {
                "strategy": ["BuyEverything", "BuyNothing"],
                "strength": [1.5, 0.8],
                "ci_lower": [1.2, 0.6],
                "ci_upper": [1.8, 1.0],
            }
        )
        with (
            patch.object(script, "run_tournament", return_value=fake_result),
            patch.object(script, "bradley_terry_ranking", return_value=fake_df),
        ):
            result = script.section_07_tournament()
        assert result is None
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_section_08_conclusion_is_callable(self, script, capsys) -> None:
        """section_08_conclusion() must print Italian summary and return None."""
        result = script.section_08_conclusion()
        assert result is None
        captured = capsys.readouterr()
        assert len(captured.out) > 0


# ---------------------------------------------------------------------------
# (b) figures/ directory is created by main()
# ---------------------------------------------------------------------------


class TestFiguresDirectoryCreation:
    """main() must create the figures/ directory if it does not exist."""

    def test_main_creates_figures_dir(self, script, tmp_path) -> None:
        """main() must create a figures/ directory."""
        figures = tmp_path / "figures"
        assert not figures.exists()

        fake_df = pd.DataFrame(
            {
                "strategy": ["BuyEverything"],
                "n_players": [2],
                "win_rate": [0.5],
                "ci_lower": [0.4],
                "ci_upper": [0.6],
                "baseline": [0.5],
                "significant": [False],
            }
        )
        fake_history = MagicMock()
        fake_history.player_names = ["Alice", "Bob"]
        fake_history.net_worth_history = [{"Alice": 1500, "Bob": 1500}]
        fake_ranking = pd.DataFrame(
            {
                "strategy": ["BuyEverything"],
                "strength": [1.0],
                "ci_lower": [0.8],
                "ci_upper": [1.2],
            }
        )

        with (
            patch.object(script, "FIGURES_DIR", figures),
            patch.object(script, "plot_board_heatmap"),
            patch.object(script, "plot_roi_bars"),
            patch.object(script, "plot_win_rate_curves"),
            patch.object(script, "plot_net_worth"),
            patch.object(script, "animate_sample_game"),
            patch.object(script, "_build_win_prob_results", return_value={}),
            patch.object(script, "win_probability_table", return_value=fake_df),
            patch.object(script, "_run_sample_game", return_value=fake_history),
            patch.object(script, "run_tournament", return_value=MagicMock()),
            patch.object(script, "bradley_terry_ranking", return_value=fake_ranking),
        ):
            script.main()

        assert figures.exists()
        assert figures.is_dir()

    def test_main_does_not_fail_if_figures_dir_already_exists(
        self, script, tmp_path
    ) -> None:
        """main() must not raise if figures/ directory already exists."""
        figures = tmp_path / "figures"
        figures.mkdir()

        fake_df = pd.DataFrame(
            {
                "strategy": ["BuyEverything"],
                "n_players": [2],
                "win_rate": [0.5],
                "ci_lower": [0.4],
                "ci_upper": [0.6],
                "baseline": [0.5],
                "significant": [False],
            }
        )
        fake_history = MagicMock()
        fake_history.player_names = ["Alice", "Bob"]
        fake_history.net_worth_history = [{"Alice": 1500, "Bob": 1500}]
        fake_ranking = pd.DataFrame(
            {
                "strategy": ["BuyEverything"],
                "strength": [1.0],
                "ci_lower": [0.8],
                "ci_upper": [1.2],
            }
        )

        with (
            patch.object(script, "FIGURES_DIR", figures),
            patch.object(script, "plot_board_heatmap"),
            patch.object(script, "plot_roi_bars"),
            patch.object(script, "plot_win_rate_curves"),
            patch.object(script, "plot_net_worth"),
            patch.object(script, "animate_sample_game"),
            patch.object(script, "_build_win_prob_results", return_value={}),
            patch.object(script, "win_probability_table", return_value=fake_df),
            patch.object(script, "_run_sample_game", return_value=fake_history),
            patch.object(script, "run_tournament", return_value=MagicMock()),
            patch.object(script, "bradley_terry_ranking", return_value=fake_ranking),
        ):
            script.main()  # Must not raise FileExistsError


# ---------------------------------------------------------------------------
# Section function naming convention
# ---------------------------------------------------------------------------


class TestSectionFunctionNames:
    """Script must export section functions with the specified naming convention."""

    @pytest.mark.parametrize(
        "name",
        [
            "section_01_intro",
            "section_02_dice",
            "section_03_markov",
            "section_04_roi",
            "section_05_monte_carlo",
            "section_06_sample_game",
            "section_07_tournament",
            "section_08_conclusion",
        ],
    )
    def test_section_function_exists(self, script, name: str) -> None:
        """Each section_XX function must be defined in the script."""
        assert hasattr(script, name), f"Missing function: {name}"
        assert callable(getattr(script, name)), f"Not callable: {name}"


# ---------------------------------------------------------------------------
# Reproducibility constants
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Script must expose N_GAMES constant and use seeds for reproducibility."""

    def test_n_games_constant_exists(self, script) -> None:
        """Script must expose N_GAMES as a module-level integer constant."""
        assert hasattr(script, "N_GAMES")
        assert isinstance(script.N_GAMES, int)
        assert script.N_GAMES >= 1

    def test_figures_dir_constant_exists(self, script) -> None:
        """Script must expose FIGURES_DIR as a module-level Path."""
        assert hasattr(script, "FIGURES_DIR")
        assert isinstance(script.FIGURES_DIR, Path)

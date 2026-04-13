"""Shared plot configuration and visualization functions for Monopoly.

Single Responsibility: this module owns the shared visual constants
(figure size, DPI, colorblind-safe palette, Italian labels), the two
thin factory/helper functions used by all callers, and the board
heatmap visualization.

Nothing unrelated to plotting belongs here.
See metrics.py, markov.py, etc. for simulation data and statistics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Union

import matplotlib

if TYPE_CHECKING:
    from monopoly.game import GameHistory
import pandas as pd

matplotlib.use("Agg")  # headless — safe for both tests and CLI

import matplotlib.cm as mpl_cm
import matplotlib.colors as mpl_colors
import matplotlib.patches as mpl_patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy
import yaml

# ---------------------------------------------------------------------------
# Figure geometry
# ---------------------------------------------------------------------------

FIGURE_SIZE_PX: tuple[int, int] = (1920, 1080)
"""Output resolution in pixels (width × height)."""

DPI: int = 150
"""Dots per inch used for all figures."""

# ---------------------------------------------------------------------------
# Colorblind-safe palette
# ---------------------------------------------------------------------------

# Eight colors from Wong (2011) — perceptually distinct under the three most
# common forms of colour-vision deficiency (deuteranopia, protanopia,
# tritanopia).
# Two additional colours from Tol's "bright" scheme extend the palette to
# cover all ten Monopoly asset groups (8 colour groups + railroads + utilities).

PALETTE: dict[str, str] = {
    # 8 standard colour groups ------------------------------------------------
    "brown": "#E69F00",  # warm amber         — Wong
    "light_blue": "#56B4E9",  # sky blue           — Wong
    "pink": "#CC79A7",  # reddish purple     — Wong
    "orange": "#D55E00",  # vermillion         — Wong
    "red": "#EE6677",  # rose               — Tol bright
    "yellow": "#F0E442",  # yellow             — Wong
    "green": "#009E73",  # bluish green       — Wong
    "dark_blue": "#0072B2",  # blue               — Wong
    # Non-colour asset groups -------------------------------------------------
    "railroad": "#000000",  # black              — Wong
    "utility": "#BBBBBB",  # light grey         — Tol bright
}
"""Colorblind-safe hex colours keyed by Monopoly group name."""

# ---------------------------------------------------------------------------
# Group registry
# ---------------------------------------------------------------------------

COLOR_GROUPS: list[str] = [
    "brown",
    "light_blue",
    "pink",
    "orange",
    "red",
    "yellow",
    "green",
    "dark_blue",
    "railroad",
    "utility",
]
"""Canonical order of all asset groups (matches PALETTE keys)."""

# ---------------------------------------------------------------------------
# Italian label mapping
# ---------------------------------------------------------------------------

ITALIAN_LABELS: dict[str, str] = {
    # 8 colour groups
    "brown": "Marrone",
    "light_blue": "Azzurro",
    "pink": "Rosa",
    "orange": "Arancione",
    "red": "Rosso",
    "yellow": "Giallo",
    "green": "Verde",
    "dark_blue": "Blu scuro",
    # Asset groups
    "railroad": "Ferrovie",
    "utility": "Società",
    # Board square categories
    "go": "Via!",
    "jail": "Prigione",
    "free_parking": "Parcheggio Libero",
    "go_to_jail": "Vai in Prigione",
    "tax": "Tassa",
    "chance": "Probabilità",
    "community": "Imprevisti",
}
"""Italian names for every Monopoly group and board-square category."""

# ---------------------------------------------------------------------------
# Heatmap constants
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent.parent / "data"
_HEATMAP_COLORMAP = "YlOrRd"
_GRID_MAX = 10  # board ring spans a (GRID_MAX+1) × (GRID_MAX+1) grid

# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_figure() -> tuple[plt.Figure, plt.Axes]:
    """Return a new (fig, ax) pair sized for 1920 × 1080 output.

    Figure width and height are derived from FIGURE_SIZE_PX and DPI so
    that saving via save_figure() produces exactly the target pixel count.
    """
    width_in = FIGURE_SIZE_PX[0] / DPI
    height_in = FIGURE_SIZE_PX[1] / DPI
    fig, ax = plt.subplots(figsize=(width_in, height_in), dpi=DPI)
    fig.tight_layout()
    return fig, ax


def save_figure(fig: plt.Figure, path: Union[str, Path]) -> None:
    """Save *fig* as a PNG at the canonical 1920 × 1080 resolution.

    Args:
        fig:  The matplotlib Figure to save.
        path: Destination file path (str or Path).  Parent directory must
              exist; the file extension should be ``.png``.
    """
    fig.savefig(path, dpi=DPI)


# ---------------------------------------------------------------------------
# Board heatmap
# ---------------------------------------------------------------------------


def plot_board_heatmap(
    distribution: numpy.ndarray,
    board: object,
    output_path: Union[str, Path],
) -> None:
    """Render a board heatmap colored by landing probability.

    Each of the 40 squares is drawn in a ring layout and colored by its
    stationary landing probability.  Italian square names label each cell.

    Args:
        distribution: 40-element array of square landing probabilities.
        board: Board instance (provides square structure; currently unused
               directly — Italian names are loaded from board_italia.yaml).
        output_path: Destination PNG path.

    Raises:
        ValueError: If ``distribution`` length ≠ 40 or contains negative values.
    """
    _validate_heatmap_distribution(distribution)
    italian_names = _load_italian_square_names()
    fig, ax = create_figure()
    cmap, norm = _build_probability_colormap(distribution)
    _draw_board_ring(ax, distribution, italian_names, cmap, norm)
    _add_probability_colorbar(fig, ax, cmap, norm)
    _configure_board_axes(ax)
    save_figure(fig, output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Heatmap helpers — each function has a single responsibility, < 10 lines
# ---------------------------------------------------------------------------


def _validate_heatmap_distribution(distribution: numpy.ndarray) -> None:
    """Raise ValueError if distribution is not a valid 40-element non-negative array."""
    if len(distribution) != 40:
        raise ValueError(
            f"distribution must have exactly 40 elements, got {len(distribution)}"
        )
    if numpy.any(distribution < 0):
        raise ValueError("distribution must not contain negative values")


def _load_italian_square_names() -> dict[int, str]:
    """Return a mapping of board position → Italian square name."""
    path = _DATA_DIR / "board_italia.yaml"
    entries: list[dict] = yaml.safe_load(path.read_text())
    return {entry["position"]: entry["name_it"] for entry in entries}


def _build_probability_colormap(
    distribution: numpy.ndarray,
) -> tuple[mpl_colors.Colormap, mpl_colors.Normalize]:
    """Return a YlOrRd colormap and a normalizer spanning the distribution range."""
    cmap = matplotlib.colormaps[_HEATMAP_COLORMAP]
    norm = mpl_colors.Normalize(
        vmin=float(distribution.min()), vmax=float(distribution.max())
    )
    return cmap, norm


def _draw_board_ring(
    ax: plt.Axes,
    distribution: numpy.ndarray,
    italian_names: dict[int, str],
    cmap: mpl_colors.Colormap,
    norm: mpl_colors.Normalize,
) -> None:
    """Draw all 40 probability-colored squares on the axes."""
    for pos in range(40):
        col, row = _square_grid_position(pos)
        _draw_square(
            ax, col, row, float(distribution[pos]), italian_names[pos], cmap, norm
        )


def _square_grid_position(position: int) -> tuple[int, int]:
    """Map board position 0–39 to (col, row) on an 11×11 ring grid.

    Layout (corners at 0, 10, 20, 30):
      bottom (left→right) : 0–10 at row 0
      right  (bottom→top) : 10–20 at col 10
      top    (right→left) : 20–30 at row 10
      left   (top→bottom) : 30–39 at col 0
    """
    if position <= 10:
        return (position, 0)
    if position <= 20:
        return (_GRID_MAX, position - _GRID_MAX)
    if position <= 30:
        return (30 - position, _GRID_MAX)
    return (0, 40 - position)


def _draw_square(
    ax: plt.Axes,
    col: int,
    row: int,
    probability: float,
    name: str,
    cmap: mpl_colors.Colormap,
    norm: mpl_colors.Normalize,
) -> None:
    """Add one colored rectangle and its text labels to the axes."""
    color = cmap(norm(probability))
    rect = mpl_patches.FancyBboxPatch(
        (col + 0.04, row + 0.04),
        0.92,
        0.92,
        boxstyle="round,pad=0.02",
        facecolor=color,
        edgecolor="white",
        linewidth=0.5,
    )
    ax.add_patch(rect)
    _draw_square_text(ax, col, row, name, probability)


def _draw_square_text(
    ax: plt.Axes,
    col: int,
    row: int,
    name: str,
    probability: float,
) -> None:
    """Render the Italian name and formatted probability inside a square."""
    cx, cy = col + 0.5, row + 0.5
    ax.text(
        cx,
        cy + 0.17,
        _wrap_name(name),
        ha="center",
        va="center",
        fontsize=3.8,
        linespacing=1.1,
    )
    ax.text(
        cx,
        cy - 0.18,
        f"{probability:.1%}",
        ha="center",
        va="center",
        fontsize=4.5,
        fontweight="bold",
    )


def _wrap_name(name: str, line_len: int = 9) -> str:
    """Split a long name into two lines for compact display."""
    if len(name) <= line_len:
        return name
    words = name.split()
    if len(words) == 1:
        return name[: line_len - 1] + "…"
    mid = len(words) // 2
    return " ".join(words[:mid]) + "\n" + " ".join(words[mid:])


def _add_probability_colorbar(
    fig: plt.Figure,
    ax: plt.Axes,
    cmap: mpl_colors.Colormap,
    norm: mpl_colors.Normalize,
) -> None:
    """Attach a colorbar with percentage labels to the right of the axes."""
    sm = mpl_cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.02)
    cbar.set_label("Probabilità di atterraggio", fontsize=9)
    cbar.ax.yaxis.set_major_formatter(mpl_ticker.FuncFormatter(lambda x, _: f"{x:.1%}"))


def _configure_board_axes(ax: plt.Axes) -> None:
    """Set axis limits, equal aspect, no borders, and a title."""
    ax.set_xlim(-0.2, 11.2)
    ax.set_ylim(-0.2, 11.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Heatmap delle probabilità di atterraggio",
        fontsize=13,
        pad=8,
    )


# ---------------------------------------------------------------------------
# ROI bar chart
# ---------------------------------------------------------------------------

_ROI_RENT_COLUMNS: list[str] = ["base", "monopoly", "1h", "2h", "3h", "4h", "hotel"]
"""Development level columns, matching metrics._RENT_COLUMNS order."""

_LEVEL_ITALIAN_LABELS: dict[str, str] = {
    "base": "Base",
    "monopoly": "Monopolio",
    "1h": "1 Casa",
    "2h": "2 Case",
    "3h": "3 Case",
    "4h": "4 Case",
    "hotel": "Hotel",
}
"""Italian names for each development level (used in the legend)."""

# Seven colorblind-safe colors — first 7 entries of PALETTE.
_LEVEL_COLORS: list[str] = list(PALETTE.values())[:7]


def plot_roi_bars(roi_table: pd.DataFrame, output_path: Union[str, Path]) -> None:
    """Render a grouped bar chart of ROI per color group at each development level.

    Args:
        roi_table:   DataFrame from ``metrics.roi_ranking_table()`` — rows are
                     color groups / railroads, columns are development levels.
                     NaN values (e.g. railroads at house levels) are skipped.
        output_path: Destination PNG path (1920 × 1080).
    """
    fig, ax = create_figure()
    _draw_roi_grouped_bars(ax, roi_table)
    _configure_roi_axes(ax, roi_table)
    save_figure(fig, output_path)
    plt.close(fig)


def _draw_roi_grouped_bars(ax: plt.Axes, roi_table: pd.DataFrame) -> None:
    """Draw one BarContainer per development level, skipping NaN positions."""
    n_groups = len(roi_table)
    n_levels = len(_ROI_RENT_COLUMNS)
    bar_width = 0.8 / n_levels
    x = numpy.arange(n_groups)

    for i, (level, color) in enumerate(zip(_ROI_RENT_COLUMNS, _LEVEL_COLORS)):
        offsets = x + (i - n_levels / 2 + 0.5) * bar_width
        values = roi_table[level].to_numpy(dtype=float)
        mask = ~numpy.isnan(values)
        ax.bar(
            offsets[mask],
            values[mask],
            width=bar_width,
            color=color,
            label=_LEVEL_ITALIAN_LABELS[level],
        )


def _configure_roi_axes(ax: plt.Axes, roi_table: pd.DataFrame) -> None:
    """Set x-ticks with Italian group names, y-axis label, legend, and title."""
    groups = list(roi_table.index)
    x = numpy.arange(len(groups))
    x_labels = [ITALIAN_LABELS.get(g, g) for g in groups]
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("ROI (rendita attesa / investimento)", fontsize=10)
    ax.set_title("ROI per gruppo colore e livello di sviluppo", fontsize=13, pad=8)
    ax.legend(title="Livello di sviluppo", loc="upper left", fontsize=8)


# ---------------------------------------------------------------------------
# Win rate curves
# ---------------------------------------------------------------------------

# Six distinct colors from the colorblind-safe palette — one per strategy.
_STRATEGY_COLORS: list[str] = list(PALETTE.values())[:6]


def plot_win_rate_curves(
    win_prob_df: pd.DataFrame, output_path: Union[str, Path]
) -> None:
    """Render win rate curves per strategy with Wilson CI bands and 1/N baseline.

    Args:
        win_prob_df:  DataFrame from ``metrics.win_probability_table()`` with columns:
                      strategy, n_players, win_rate, ci_lower, ci_upper, baseline, significant.
        output_path:  Destination PNG path (1920 × 1080).

    Raises:
        ValueError: If ``win_prob_df`` is empty.
    """
    _validate_win_prob_df(win_prob_df)
    fig, ax = create_figure()
    strategies = list(win_prob_df["strategy"].unique())
    _draw_strategy_lines(ax, win_prob_df, strategies)
    _draw_baseline_line(ax, win_prob_df)
    _configure_win_rate_axes(ax)
    save_figure(fig, output_path)
    plt.close(fig)


def _validate_win_prob_df(win_prob_df: pd.DataFrame) -> None:
    """Raise ValueError if the input DataFrame is empty."""
    if win_prob_df.empty:
        raise ValueError("win_prob_df is empty — cannot render win rate curves")


def _draw_strategy_lines(
    ax: plt.Axes,
    win_prob_df: pd.DataFrame,
    strategies: list[str],
) -> None:
    """Plot one line + CI band per strategy, bold for significant ones."""
    for i, strategy in enumerate(strategies):
        color = _STRATEGY_COLORS[i % len(_STRATEGY_COLORS)]
        subset = win_prob_df[win_prob_df["strategy"] == strategy].sort_values(
            "n_players"
        )
        is_significant = bool(subset["significant"].any())
        linewidth = 2.5 if is_significant else 1.2
        ax.plot(
            subset["n_players"],
            subset["win_rate"],
            color=color,
            linewidth=linewidth,
            label=strategy,
        )
        ax.fill_between(
            subset["n_players"],
            subset["ci_lower"],
            subset["ci_upper"],
            color=color,
            alpha=0.2,
        )


def _draw_baseline_line(ax: plt.Axes, win_prob_df: pd.DataFrame) -> None:
    """Draw the dashed 1/N fair-chance baseline across all player counts."""
    player_counts = sorted(win_prob_df["n_players"].unique())
    baselines = [1.0 / n for n in player_counts]
    ax.plot(
        player_counts,
        baselines,
        color="black",
        linewidth=1.0,
        linestyle="--",
        label="Baseline (1/N)",
    )


def _configure_win_rate_axes(ax: plt.Axes) -> None:
    """Set axis labels, x-ticks at integer player counts, legend, and title."""
    ax.set_xlabel("Numero di giocatori", fontsize=11)
    ax.set_ylabel("Tasso di vittoria", fontsize=11)
    ax.set_title("Tasso di vittoria per strategia", fontsize=13, pad=8)
    ax.legend(title="Strategia", loc="upper right", fontsize=8)


# ---------------------------------------------------------------------------
# Net worth over time
# ---------------------------------------------------------------------------

# Colors for players — cycle through palette values.
_PLAYER_COLORS: list[str] = list(PALETTE.values())


def plot_net_worth(
    net_worth_history: dict[str, list[int]],
    player_names: list[str],
    output_path: Union[str, Path],
) -> None:
    """Render per-player net worth evolution as a line chart.

    Args:
        net_worth_history:  Mapping of player name → per-turn net worth list.
        player_names:       Ordered list of player names to plot (keys of the dict).
        output_path:        Destination PNG path (1920 × 1080).
    """
    fig, ax = create_figure()
    _draw_player_lines(ax, net_worth_history, player_names)
    _draw_zero_reference(ax)
    _configure_net_worth_axes(ax)
    save_figure(fig, output_path)
    plt.close(fig)


def _draw_player_lines(
    ax: plt.Axes,
    net_worth_history: dict[str, list[int]],
    player_names: list[str],
) -> None:
    """Plot one line per player, stopping at their bankruptcy turn."""
    for i, name in enumerate(player_names):
        history = net_worth_history.get(name, [])
        turns, values = _trim_to_last_nonzero(history)
        if not turns:
            continue
        color = _PLAYER_COLORS[i % len(_PLAYER_COLORS)]
        ax.plot(turns, values, color=color, linewidth=2.0, label=name)


def _trim_to_last_nonzero(history: list[int]) -> tuple[list[int], list[int]]:
    """Return (turn indices, values) up to and including the last nonzero entry."""
    if not history:
        return [], []
    last_nonzero = len(history) - 1
    for idx in range(len(history) - 1, -1, -1):
        if history[idx] != 0:
            last_nonzero = idx
            break
    else:
        return [], []
    turns = list(range(last_nonzero + 1))
    values = history[: last_nonzero + 1]
    return turns, values


def _draw_zero_reference(ax: plt.Axes) -> None:
    """Draw a horizontal dashed reference line at net worth = 0."""
    ax.axhline(y=0, color="black", linewidth=1.0, linestyle="--", label="_zero")


def _configure_net_worth_axes(ax: plt.Axes) -> None:
    """Set Italian axis labels, legend, and title."""
    ax.set_xlabel("Turno", fontsize=11)
    ax.set_ylabel("Patrimonio netto ($)", fontsize=11)
    ax.set_title("Andamento del patrimonio netto per giocatore", fontsize=13, pad=8)
    ax.legend(title="Giocatore", loc="upper left", fontsize=9)


# ---------------------------------------------------------------------------
# Sample game animation
# ---------------------------------------------------------------------------

_ANIM_MAX_FRAMES: int = 200
"""Maximum number of frames in the exported animation."""

_ANIM_BOARD_WIDTH_RATIO: int = 3
"""Board panel is this many times wider than the sidebar."""

_ANIM_TOKEN_RADIUS: float = 0.22
"""Radius of the circular player tokens drawn on the board."""


def animate_sample_game(
    game_history: "GameHistory",
    output_path: Union[str, Path],
    fps: int = 5,
) -> Path:
    """Animate a sample Monopoly game, showing tokens moving turn by turn.

    Each frame shows the board state after one turn: player token positions,
    property ownership coloring, and a sidebar with net worth bars.
    For games longer than 200 turns, frames are sampled evenly.

    Args:
        game_history: Per-turn history with positions, net worth, and ownership.
        output_path:  Destination file path (.mp4 uses ffmpeg, .gif uses pillow).
        fps:          Frames per second for the output animation.

    Returns:
        The output_path as a resolved Path object.
    """
    from matplotlib.animation import FuncAnimation

    output_path = Path(output_path)
    frames = _sample_animation_frames(len(game_history.position_history))
    player_colors = _make_player_color_map(game_history.player_names)
    fig, board_ax, sidebar_ax = _create_animation_figure()

    def update(frame_idx: int) -> None:
        board_ax.cla()
        sidebar_ax.cla()
        _draw_board_frame(board_ax, game_history, frame_idx, player_colors)
        _update_sidebar(sidebar_ax, game_history, frame_idx, player_colors)

    anim = FuncAnimation(fig, update, frames=frames, blit=False)  # type: ignore[arg-type]
    writer = _select_animation_writer(output_path)
    anim.save(str(output_path), writer=writer, fps=fps, dpi=DPI)
    plt.close(fig)
    return output_path


def _sample_animation_frames(
    n_turns: int, max_frames: int = _ANIM_MAX_FRAMES
) -> list[int]:
    """Return frame indices, sampling every N turns for games > max_frames."""
    if n_turns <= max_frames:
        return list(range(n_turns))
    step = math.ceil(n_turns / max_frames)
    return list(range(0, n_turns, step))


def _make_player_color_map(player_names: list[str]) -> dict[str, str]:
    """Assign a colorblind-safe palette color to each player by index."""
    colors = list(PALETTE.values())
    return {name: colors[i % len(colors)] for i, name in enumerate(player_names)}


def _create_animation_figure() -> tuple[plt.Figure, plt.Axes, plt.Axes]:
    """Return (fig, board_ax, sidebar_ax) at 1920×1080 resolution."""
    width_in = FIGURE_SIZE_PX[0] / DPI
    height_in = FIGURE_SIZE_PX[1] / DPI
    fig, (board_ax, sidebar_ax) = plt.subplots(
        1,
        2,
        figsize=(width_in, height_in),
        dpi=DPI,
        gridspec_kw={"width_ratios": [_ANIM_BOARD_WIDTH_RATIO, 1]},
    )
    return fig, board_ax, sidebar_ax


def _select_animation_writer(output_path: Path) -> str:
    """Return 'ffmpeg' for .mp4 output, 'pillow' for all others (.gif)."""
    return "ffmpeg" if output_path.suffix.lower() == ".mp4" else "pillow"


def _draw_board_frame(
    ax: plt.Axes,
    history: "GameHistory",
    frame_idx: int,
    player_colors: dict[str, str],
) -> None:
    """Render board squares, ownership coloring, and tokens for one frame."""
    _draw_ownership(ax, history.ownership_history[frame_idx], player_colors)
    _draw_tokens(ax, history.position_history[frame_idx], player_colors)
    ax.set_xlim(-0.2, 11.2)
    ax.set_ylim(-0.2, 11.2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Tabellone di gioco", fontsize=10)


def _draw_ownership(
    ax: plt.Axes,
    ownership: dict[int, str],
    player_colors: dict[str, str],
) -> None:
    """Draw all 40 squares with ownership-colored borders."""
    for pos in range(40):
        col, row = _square_grid_position(pos)
        owner = ownership.get(pos)
        edge_color = player_colors.get(owner, "lightgray") if owner else "lightgray"
        lw = 2.0 if owner else 0.5
        rect = mpl_patches.FancyBboxPatch(
            (col + 0.04, row + 0.04),
            0.92,
            0.92,
            boxstyle="round,pad=0.02",
            facecolor="whitesmoke",
            edgecolor=edge_color,
            linewidth=lw,
        )
        ax.add_patch(rect)


def _draw_tokens(
    ax: plt.Axes,
    positions: dict[str, int],
    player_colors: dict[str, str],
) -> None:
    """Draw player tokens, offsetting multiple tokens sharing a square."""
    pos_players: dict[int, list[str]] = {}
    for name, pos in positions.items():
        pos_players.setdefault(pos, []).append(name)
    for pos, names in pos_players.items():
        _draw_tokens_at_square(ax, pos, names, player_colors)


def _draw_tokens_at_square(
    ax: plt.Axes,
    pos: int,
    names: list[str],
    player_colors: dict[str, str],
) -> None:
    """Draw one circular token per player at the given square, spread horizontally."""
    col, row = _square_grid_position(pos)
    cx, cy = col + 0.5, row + 0.5
    for j, name in enumerate(names):
        x_off = (j - len(names) / 2 + 0.5) * 0.3
        circle = mpl_patches.Circle(
            (cx + x_off, cy),
            _ANIM_TOKEN_RADIUS,
            color=player_colors.get(name, "gray"),
            zorder=5,
        )
        ax.add_patch(circle)


def _update_sidebar(
    ax: plt.Axes,
    history: "GameHistory",
    frame_idx: int,
    player_colors: dict[str, str],
) -> None:
    """Draw a horizontal net worth bar chart for the current frame."""
    names = history.player_names
    values = [history.net_worth_history[frame_idx].get(n, 0) for n in names]
    colors = [player_colors.get(n, "gray") for n in names]
    ax.barh(names, values, color=colors)
    ax.set_xlabel("Patrimonio netto ($)", fontsize=8)
    ax.set_title(f"Turno {frame_idx + 1}", fontsize=9)
    ax.tick_params(labelsize=7)

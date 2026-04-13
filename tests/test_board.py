"""Tests for Board model correctness — structure, data integrity, and locale support.

TDD Red phase: all tests are written before implementation exists.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from hypothesis import given, settings
from hypothesis import strategies as st

from monopoly.board import Board, ColorProperty, SquareType

# ---------------------------------------------------------------------------
# Data directory helper
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / "data"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OFFICIAL_GROUP_SIZES: dict[str, int] = {
    "brown": 2,
    "light_blue": 3,
    "pink": 3,
    "orange": 3,
    "red": 3,
    "yellow": 3,
    "green": 3,
    "dark_blue": 2,
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def board_us() -> Board:
    """Standard US Monopoly board."""
    return Board(locale="us")


@pytest.fixture(scope="module")
def board_it() -> Board:
    """Italian locale Monopoly board."""
    return Board(locale="it")


# ---------------------------------------------------------------------------
# 1. Board has exactly 40 squares at positions 0–39, no gaps, no duplicates
# ---------------------------------------------------------------------------


def test_board_has_exactly_40_squares(board_us: Board) -> None:
    """Board must contain exactly 40 squares."""
    assert len(board_us.squares) == 40


def test_board_positions_are_0_to_39(board_us: Board) -> None:
    """Positions must be exactly 0 through 39 with no gaps or duplicates."""
    positions = sorted(sq.position for sq in board_us.squares)
    assert positions == list(range(40))


# ---------------------------------------------------------------------------
# 2. mortgage == price // 2 for every buyable square (property-based)
# ---------------------------------------------------------------------------


def test_mortgage_equals_half_price_for_all_buyable(board_us: Board) -> None:
    """mortgage must equal price // 2 for every buyable square."""
    buyable = board_us.buyable_squares
    assert len(buyable) > 0
    for sq in buyable:
        assert sq.mortgage == sq.price // 2, (
            f"{sq.name}: mortgage {sq.mortgage} != price {sq.price} // 2"
        )


@given(seed=st.integers(min_value=0, max_value=2**16 - 1))
@settings(max_examples=10)
def test_mortgage_hypothesis(seed: int) -> None:
    """Property-based: mortgage == price // 2 holds for all buyable squares."""
    board = Board(locale="us")
    for sq in board.buyable_squares:
        assert sq.mortgage == sq.price // 2


# ---------------------------------------------------------------------------
# 3. Every color property has a rents list of length 6
# ---------------------------------------------------------------------------


def test_color_property_rents_length_6(board_us: Board) -> None:
    """Each color property must have exactly 6 rent values."""
    color_props = [sq for sq in board_us.squares if sq.type == SquareType.property]
    assert len(color_props) > 0
    for sq in color_props:
        assert isinstance(sq, ColorProperty)
        assert len(sq.rents) == 6, f"{sq.name}: expected 6 rents, got {len(sq.rents)}"


# ---------------------------------------------------------------------------
# 4. Every railroad has a rents list of length 4
# ---------------------------------------------------------------------------


def test_railroad_rents_length_4(board_us: Board) -> None:
    """Each railroad must have exactly 4 rent values."""
    railroads = [sq for sq in board_us.squares if sq.type == SquareType.railroad]
    assert len(railroads) == 4
    for sq in railroads:
        assert len(sq.rents) == 4, f"{sq.name}: expected 4 rents, got {len(sq.rents)}"


# ---------------------------------------------------------------------------
# 5. Color group sizes match official Monopoly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("color,expected_size", OFFICIAL_GROUP_SIZES.items())
def test_color_group_sizes(board_us: Board, color: str, expected_size: int) -> None:
    """Each color group must have the correct number of properties."""
    group = board_us.get_group(color)
    assert len(group) == expected_size, (
        f"Color '{color}': expected {expected_size}, got {len(group)}"
    )


# ---------------------------------------------------------------------------
# 6. Exactly the right number of non-property squares
# ---------------------------------------------------------------------------


def test_action_and_tax_square_count(board_us: Board) -> None:
    """Board must have exactly 10 action squares and 2 tax squares."""
    action_count = sum(1 for sq in board_us.squares if sq.type == SquareType.action)
    tax_count = sum(1 for sq in board_us.squares if sq.type == SquareType.tax)
    assert action_count == 10
    assert tax_count == 2


# ---------------------------------------------------------------------------
# 7. Square type counts: 4 railroads, 2 utilities, 22 color properties, 12 non-buyable
# ---------------------------------------------------------------------------


def test_square_type_counts(board_us: Board) -> None:
    """Verify exact counts for each square type."""
    railroad_count = sum(1 for sq in board_us.squares if sq.type == SquareType.railroad)
    utility_count = sum(1 for sq in board_us.squares if sq.type == SquareType.utility)
    property_count = sum(1 for sq in board_us.squares if sq.type == SquareType.property)
    action_count = sum(1 for sq in board_us.squares if sq.type == SquareType.action)
    tax_count = sum(1 for sq in board_us.squares if sq.type == SquareType.tax)

    assert railroad_count == 4
    assert utility_count == 2
    assert property_count == 22
    assert action_count == 10
    assert tax_count == 2
    assert (
        railroad_count + utility_count + property_count + action_count + tax_count == 40
    )


# ---------------------------------------------------------------------------
# 8. Board(locale="it") has same positions, prices, rents but different names
# ---------------------------------------------------------------------------


def test_italian_board_same_positions(board_us: Board, board_it: Board) -> None:
    """Italian board must have identical positions to US board."""
    us_positions = sorted(sq.position for sq in board_us.squares)
    it_positions = sorted(sq.position for sq in board_it.squares)
    assert us_positions == it_positions


def test_italian_board_same_prices(board_us: Board, board_it: Board) -> None:
    """Italian board must have identical prices to US board."""
    us_buyable = {sq.position: sq.price for sq in board_us.buyable_squares}
    it_buyable = {sq.position: sq.price for sq in board_it.buyable_squares}
    assert us_buyable == it_buyable


def test_italian_board_same_rents(board_us: Board, board_it: Board) -> None:
    """Italian board must have identical rents to US board."""
    us_props = {
        sq.position: sq.rents
        for sq in board_us.squares
        if sq.type == SquareType.property
    }
    it_props = {
        sq.position: sq.rents
        for sq in board_it.squares
        if sq.type == SquareType.property
    }
    assert us_props == it_props


def test_italian_board_different_names(board_us: Board, board_it: Board) -> None:
    """Italian board must have different names for localised squares."""
    # Position 1: Mediterranean Ave vs Vicolo Corto
    us_sq = board_us.get_square(1)
    it_sq = board_it.get_square(1)
    assert us_sq.name != it_sq.name
    assert it_sq.name == "Vicolo Corto"


# ---------------------------------------------------------------------------
# 9. House costs are identical within each color group
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("color", OFFICIAL_GROUP_SIZES.keys())
def test_house_costs_uniform_within_group(board_us: Board, color: str) -> None:
    """All properties in a color group must share the same house cost."""
    group = board_us.get_group(color)
    house_costs = {sq.house_cost for sq in group}
    assert len(house_costs) == 1, (
        f"Color '{color}' has inconsistent house costs: {house_costs}"
    )


# ---------------------------------------------------------------------------
# 10. All rents are non-negative integers and strictly increasing
# ---------------------------------------------------------------------------


def test_color_property_rents_non_negative_and_increasing(board_us: Board) -> None:
    """All color property rents must be non-negative and strictly increasing."""
    for sq in board_us.squares:
        if sq.type != SquareType.property:
            continue
        assert isinstance(sq, ColorProperty)
        for rent in sq.rents:
            assert rent >= 0, f"{sq.name}: negative rent {rent}"
        for i in range(len(sq.rents) - 1):
            assert sq.rents[i] < sq.rents[i + 1], (
                f"{sq.name}: rents not strictly increasing at index {i}"
            )


def test_railroad_rents_non_negative_and_increasing(board_us: Board) -> None:
    """All railroad rents must be non-negative and strictly increasing."""
    for sq in board_us.squares:
        if sq.type != SquareType.railroad:
            continue
        for rent in sq.rents:
            assert rent >= 0
        for i in range(len(sq.rents) - 1):
            assert sq.rents[i] < sq.rents[i + 1]


# ---------------------------------------------------------------------------
# 11. Board.get_square(pos) returns correct square for key positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "position,expected_name",
    [
        (0, "Go"),
        (10, "Jail / Just Visiting"),
        (20, "Free Parking"),
        (30, "Go To Jail"),
        (39, "Boardwalk"),
    ],
)
def test_get_square_returns_correct_square(
    board_us: Board, position: int, expected_name: str
) -> None:
    """get_square(pos) must return the square at that position."""
    sq = board_us.get_square(position)
    assert sq.position == position
    assert sq.name == expected_name


def test_get_square_raises_for_invalid_position(board_us: Board) -> None:
    """get_square with an out-of-range position must raise ValueError."""
    with pytest.raises(ValueError):
        board_us.get_square(40)
    with pytest.raises(ValueError):
        board_us.get_square(-1)


# ---------------------------------------------------------------------------
# 12. Board.get_group(color) returns correct count for each color group
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("color,expected_count", OFFICIAL_GROUP_SIZES.items())
def test_get_group_returns_correct_count(
    board_us: Board, color: str, expected_count: int
) -> None:
    """get_group(color) must return all properties of that color."""
    group = board_us.get_group(color)
    assert len(group) == expected_count
    for sq in group:
        assert isinstance(sq, ColorProperty)
        assert sq.color == color


def test_get_group_returns_empty_for_unknown_color(board_us: Board) -> None:
    """get_group with an unknown color must return an empty list."""
    group = board_us.get_group("purple")
    assert group == []


# ---------------------------------------------------------------------------
# 13. board_italia.yaml data file structure (Issue #3 acceptance criteria)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def italia_raw() -> list[dict]:
    """Load board_italia.yaml directly as raw YAML."""
    path = _DATA_DIR / "board_italia.yaml"
    data = yaml.safe_load(path.read_text())
    assert isinstance(data, list), "board_italia.yaml must be a list at the top level"
    return data


def test_italia_yaml_is_list(italia_raw: list[dict]) -> None:
    """board_italia.yaml must be a list of mappings, not a dict."""
    assert isinstance(italia_raw, list)


def test_italia_yaml_has_exactly_40_entries(italia_raw: list[dict]) -> None:
    """board_italia.yaml must have exactly 40 entries."""
    assert len(italia_raw) == 40


def test_italia_yaml_positions_0_to_39_contiguous(italia_raw: list[dict]) -> None:
    """Positions in board_italia.yaml must be exactly 0–39 with no gaps or duplicates."""
    positions = sorted(entry["position"] for entry in italia_raw)
    assert positions == list(range(40))


def test_italia_yaml_entries_have_only_position_and_name_it(
    italia_raw: list[dict],
) -> None:
    """Each entry must have exactly 'position' and 'name_it' keys — no numeric data."""
    for entry in italia_raw:
        assert set(entry.keys()) == {"position", "name_it"}, (
            f"Entry at pos {entry.get('position')} has unexpected keys: {set(entry.keys())}"
        )


def test_italia_yaml_no_numeric_values(italia_raw: list[dict]) -> None:
    """name_it values must be strings — no numeric economic data allowed."""
    for entry in italia_raw:
        assert isinstance(entry["name_it"], str), (
            f"Position {entry['position']}: name_it must be a string"
        )


@pytest.mark.parametrize(
    "position,expected_name_it",
    [
        # Non-property squares
        (0, "Via!"),
        (2, "Probabilità"),
        (4, "Tassa Patrimoniale"),
        (7, "Imprevisti"),
        (10, "Prigione"),
        (17, "Probabilità"),
        (20, "Parcheggio Gratuito"),
        (22, "Imprevisti"),
        (30, "In Prigione!"),
        (33, "Probabilità"),
        (36, "Imprevisti"),
        (38, "Tassa di Lusso"),
        # Brown
        (1, "Vicolo Corto"),
        (3, "Vicolo Stretto"),
        # Light blue
        (6, "Bastioni Gran Sasso"),
        (8, "Viale Monte Rosa"),
        (9, "Viale Vesuvio"),
        # Pink
        (11, "Via Accademia"),
        (13, "Corso Ateneo"),
        (14, "Piazza Università"),
        # Orange
        (16, "Via Verdi"),
        (18, "Corso Raffaello"),
        (19, "Piazza Dante"),
        # Red
        (21, "Via Marco Polo"),
        (23, "Corso Magellano"),
        (24, "Largo Colombo"),
        # Yellow
        (26, "Viale Costantino"),
        (27, "Viale Traiano"),
        (29, "Piazza Giulio Cesare"),
        # Green
        (31, "Via Roma"),
        (32, "Corso Impero"),
        (34, "Largo Augusto"),
        # Dark blue
        (37, "Viale dei Giardini"),
        (39, "Parco della Vittoria"),
        # Railroads
        (5, "Stazione Sud"),
        (15, "Stazione Ovest"),
        (25, "Stazione Nord"),
        (35, "Stazione Est"),
        # Utilities
        (12, "Società Elettrica"),
        (28, "Società Acqua Potabile"),
    ],
)
def test_italia_yaml_specific_names(
    italia_raw: list[dict], position: int, expected_name_it: str
) -> None:
    """Each Italian name must match the specification in requirements.md."""
    entry = next(e for e in italia_raw if e["position"] == position)
    assert entry["name_it"] == expected_name_it, (
        f"Position {position}: expected '{expected_name_it}', got '{entry['name_it']}'"
    )


def test_italian_board_model_uses_all_italian_names(board_it: Board) -> None:
    """Board(locale='it') must apply Italian names from board_italia.yaml for all squares."""
    italia_raw = yaml.safe_load((_DATA_DIR / "board_italia.yaml").read_text())
    name_map = {entry["position"]: entry["name_it"] for entry in italia_raw}
    for sq in board_it.squares:
        assert sq.name == name_map[sq.position], (
            f"Position {sq.position}: Board has '{sq.name}', expected '{name_map[sq.position]}'"
        )

"""Validation tests for data/board_standard.yaml raw structure.

These tests validate the YAML data file directly using PyYAML only.
They do NOT import or depend on the Board model (src/monopoly/board.py).

TDD: written before verifying every acceptance criterion is met.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
import yaml

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent / "data"


@pytest.fixture(scope="module")
def squares() -> list[dict]:
    """Load board_standard.yaml and return the list of square dicts."""
    path = _DATA_DIR / "board_standard.yaml"
    raw = yaml.safe_load(path.read_text())
    return raw["squares"]


# ---------------------------------------------------------------------------
# 1. YAML loads successfully and contains exactly 40 entries
# ---------------------------------------------------------------------------


def test_yaml_loads_successfully_and_has_40_entries(squares: list[dict]) -> None:
    """board_standard.yaml must load without errors and contain exactly 40 squares."""
    assert isinstance(squares, list)
    assert len(squares) == 40


# ---------------------------------------------------------------------------
# 2. Positions are exactly 0–39 with no gaps or duplicates
# ---------------------------------------------------------------------------


def test_positions_are_0_to_39_no_gaps_or_duplicates(squares: list[dict]) -> None:
    """Positions must be exactly 0 through 39 — contiguous, no duplicates."""
    positions = sorted(entry["position"] for entry in squares)
    assert positions == list(range(40))


# ---------------------------------------------------------------------------
# 3. Every entry has at minimum 'name', 'position', and 'type' fields
# ---------------------------------------------------------------------------


def test_every_entry_has_required_base_fields(squares: list[dict]) -> None:
    """Each square must declare name, position, and type."""
    for entry in squares:
        pos = entry.get("position", "?")
        assert "name" in entry, f"Position {pos}: missing 'name'"
        assert "position" in entry, f"Position {pos}: missing 'position'"
        assert "type" in entry, f"Position {pos}: missing 'type'"
        assert isinstance(entry["name"], str), (
            f"Position {pos}: 'name' must be a string"
        )
        assert isinstance(entry["position"], int), (
            f"Position {pos}: 'position' must be int"
        )
        assert isinstance(entry["type"], str), (
            f"Position {pos}: 'type' must be a string"
        )


# ---------------------------------------------------------------------------
# 4. mortgage == price // 2 for every entry that has a 'price' field
# ---------------------------------------------------------------------------


def test_mortgage_equals_half_price_for_all_buyable_entries(
    squares: list[dict],
) -> None:
    """For every square with a price, mortgage must equal price // 2."""
    buyable = [entry for entry in squares if "price" in entry]
    assert len(buyable) > 0, "No buyable squares found"
    for entry in buyable:
        pos = entry["position"]
        price = entry["price"]
        mortgage = entry["mortgage"]
        assert mortgage == price // 2, (
            f"Position {pos} ({entry['name']}): "
            f"mortgage {mortgage} != price {price} // 2 = {price // 2}"
        )


# ---------------------------------------------------------------------------
# 5. Rent array lengths: 6 for color properties, 4 for railroads
# ---------------------------------------------------------------------------


def test_color_property_rents_have_length_6(squares: list[dict]) -> None:
    """Each color property must have a 'rents' array of exactly 6 values."""
    properties = [entry for entry in squares if entry["type"] == "property"]
    assert len(properties) > 0
    for entry in properties:
        pos = entry["position"]
        assert "rents" in entry, f"Position {pos}: property missing 'rents'"
        rents = entry["rents"]
        assert isinstance(rents, list), f"Position {pos}: 'rents' must be a list"
        assert len(rents) == 6, (
            f"Position {pos} ({entry['name']}): expected 6 rents, got {len(rents)}"
        )


def test_railroad_rents_have_length_4(squares: list[dict]) -> None:
    """Each railroad must have a 'rents' array of exactly 4 values."""
    railroads = [entry for entry in squares if entry["type"] == "railroad"]
    assert len(railroads) == 4
    for entry in railroads:
        pos = entry["position"]
        assert "rents" in entry, f"Position {pos}: railroad missing 'rents'"
        rents = entry["rents"]
        assert isinstance(rents, list), f"Position {pos}: 'rents' must be a list"
        assert len(rents) == 4, (
            f"Position {pos} ({entry['name']}): expected 4 rents, got {len(rents)}"
        )


# ---------------------------------------------------------------------------
# 6. Type counts: 22 property + 4 railroad + 2 utility + 2 tax + 10 action = 40
# ---------------------------------------------------------------------------


def test_type_counts_match_standard_monopoly_board(squares: list[dict]) -> None:
    """Type counts must match the official Monopoly board composition."""
    counts = Counter(entry["type"] for entry in squares)
    assert counts["property"] == 22, f"Expected 22 properties, got {counts['property']}"
    assert counts["railroad"] == 4, f"Expected 4 railroads, got {counts['railroad']}"
    assert counts["utility"] == 2, f"Expected 2 utilities, got {counts['utility']}"
    assert counts["tax"] == 2, f"Expected 2 tax squares, got {counts['tax']}"
    assert counts["action"] == 10, f"Expected 10 action squares, got {counts['action']}"
    total = sum(counts.values())
    assert total == 40, f"Expected 40 total squares, got {total}"


# ---------------------------------------------------------------------------
# 7. House costs are consistent within each color group
# ---------------------------------------------------------------------------


def test_house_costs_are_uniform_within_each_color_group(squares: list[dict]) -> None:
    """All properties in the same color group must share the same house_cost."""
    properties = [entry for entry in squares if entry["type"] == "property"]
    group_costs: dict[str, set[int]] = {}
    for entry in properties:
        pos = entry["position"]
        assert "color" in entry, f"Position {pos}: property missing 'color'"
        assert "house_cost" in entry, f"Position {pos}: property missing 'house_cost'"
        color = entry["color"]
        cost = entry["house_cost"]
        group_costs.setdefault(color, set()).add(cost)
    for color, costs in group_costs.items():
        assert len(costs) == 1, (
            f"Color group '{color}' has inconsistent house costs: {costs}"
        )


# ---------------------------------------------------------------------------
# 8. Color properties have required fields
# ---------------------------------------------------------------------------


def test_color_properties_have_all_required_fields(squares: list[dict]) -> None:
    """Each color property must declare: name, position, type, price, mortgage,
    rents, house_cost, and color."""
    required = {
        "name",
        "position",
        "type",
        "price",
        "mortgage",
        "rents",
        "house_cost",
        "color",
    }
    properties = [entry for entry in squares if entry["type"] == "property"]
    for entry in properties:
        pos = entry["position"]
        missing = required - set(entry.keys())
        assert not missing, (
            f"Position {pos} ({entry.get('name', '?')}): missing fields {missing}"
        )


# ---------------------------------------------------------------------------
# 9. Railroad prices and mortgage values
# ---------------------------------------------------------------------------


def test_all_railroads_have_price_200_and_mortgage_100(squares: list[dict]) -> None:
    """All four railroads must have price=200 and mortgage=100."""
    railroads = [entry for entry in squares if entry["type"] == "railroad"]
    for entry in railroads:
        pos = entry["position"]
        assert entry.get("price") == 200, (
            f"Position {pos} ({entry['name']}): railroad price must be 200"
        )
        assert entry.get("mortgage") == 100, (
            f"Position {pos} ({entry['name']}): railroad mortgage must be 100"
        )


# ---------------------------------------------------------------------------
# 10. Utilities price and mortgage values
# ---------------------------------------------------------------------------


def test_all_utilities_have_price_150_and_mortgage_75(squares: list[dict]) -> None:
    """Both utilities must have price=150 and mortgage=75."""
    utilities = [entry for entry in squares if entry["type"] == "utility"]
    assert len(utilities) == 2
    for entry in utilities:
        pos = entry["position"]
        assert entry.get("price") == 150, (
            f"Position {pos} ({entry['name']}): utility price must be 150"
        )
        assert entry.get("mortgage") == 75, (
            f"Position {pos} ({entry['name']}): utility mortgage must be 75"
        )


# ---------------------------------------------------------------------------
# 11. Tax squares have correct amounts
# ---------------------------------------------------------------------------


def test_tax_squares_have_correct_amounts(squares: list[dict]) -> None:
    """Income Tax must be $200 and Luxury Tax must be $100."""
    tax_squares = [entry for entry in squares if entry["type"] == "tax"]
    assert len(tax_squares) == 2
    tax_by_pos = {entry["position"]: entry for entry in tax_squares}
    # Income Tax at position 4
    assert 4 in tax_by_pos, "Income Tax (position 4) not found"
    assert tax_by_pos[4]["amount"] == 200, (
        f"Income Tax amount: expected 200, got {tax_by_pos[4].get('amount')}"
    )
    # Luxury Tax at position 38
    assert 38 in tax_by_pos, "Luxury Tax (position 38) not found"
    assert tax_by_pos[38]["amount"] == 100, (
        f"Luxury Tax amount: expected 100, got {tax_by_pos[38].get('amount')}"
    )


# ---------------------------------------------------------------------------
# 12. Known square names at key positions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "position,expected_name",
    [
        (0, "Go"),
        (10, "Jail / Just Visiting"),
        (20, "Free Parking"),
        (30, "Go To Jail"),
        (1, "Mediterranean Avenue"),
        (39, "Boardwalk"),
        (5, "Reading Railroad"),
        (12, "Electric Company"),
        (28, "Water Works"),
    ],
)
def test_known_square_names_at_key_positions(
    squares: list[dict], position: int, expected_name: str
) -> None:
    """Specific positions must have the correct canonical US square names."""
    entry = next((e for e in squares if e["position"] == position), None)
    assert entry is not None, f"No entry found at position {position}"
    assert entry["name"] == expected_name, (
        f"Position {position}: expected '{expected_name}', got '{entry['name']}'"
    )


# ---------------------------------------------------------------------------
# 13. All eight color groups are present
# ---------------------------------------------------------------------------


def test_all_eight_color_groups_present(squares: list[dict]) -> None:
    """All 8 official color groups must appear in the data."""
    expected_colors = {
        "brown",
        "light_blue",
        "pink",
        "orange",
        "red",
        "yellow",
        "green",
        "dark_blue",
    }
    actual_colors = {entry["color"] for entry in squares if entry["type"] == "property"}
    assert actual_colors == expected_colors, (
        f"Missing colors: {expected_colors - actual_colors}, "
        f"extra colors: {actual_colors - expected_colors}"
    )

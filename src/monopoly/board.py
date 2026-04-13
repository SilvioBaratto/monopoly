"""Board model module for Monopoly probability analysis.

Responsibilities (SRP):
- Define immutable square data structures (Square hierarchy)
- Enumerate square types (SquareType)
- Load board from YAML with locale support (Board)
- Provide query methods: get_square(), get_group()

No game logic — this module manages board structure only.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Data directory
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent.parent.parent / "data"


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SquareType(str, Enum):
    """All valid Monopoly square types."""

    property = "property"
    railroad = "railroad"
    utility = "utility"
    action = "action"
    tax = "tax"


# ---------------------------------------------------------------------------
# Square hierarchy (frozen dataclasses — immutable value objects)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Square:
    """Immutable base square with position, name, and type.

    Args:
        position: Board position (0–39).
        name: Display name of the square.
        type: Square classification.
    """

    position: int
    name: str
    type: SquareType


@dataclass(frozen=True)
class BuyableSquare(Square):
    """A square that can be purchased.

    Args:
        price: Purchase price in dollars.
        mortgage: Mortgage value in dollars.
    """

    price: int
    mortgage: int


@dataclass(frozen=True)
class ColorProperty(BuyableSquare):
    """A color-group property with house building.

    Args:
        color: Color group identifier.
        house_cost: Cost per house/hotel.
        rents: Tuple of 6 rents: base, 1h, 2h, 3h, 4h, hotel.
    """

    color: str
    house_cost: int
    rents: tuple[int, ...]


@dataclass(frozen=True)
class Railroad(BuyableSquare):
    """A railroad square.

    Args:
        rents: Tuple of 4 rents based on railroads owned (1–4).
    """

    rents: tuple[int, ...]


@dataclass(frozen=True)
class Utility(BuyableSquare):
    """A utility square (Electric Company / Water Works).

    Args:
        rents: Tuple of 2 multipliers: [1-utility-owned, 2-utilities-owned].
    """

    rents: tuple[int, ...]


@dataclass(frozen=True)
class TaxSquare(Square):
    """A tax square that charges a fixed amount.

    Args:
        amount: Tax amount in dollars.
    """

    amount: int


# ---------------------------------------------------------------------------
# Square factory
# ---------------------------------------------------------------------------


def _build_square(raw: dict[str, Any], name_override: str | None) -> Square:
    """Construct the correct Square subclass from a raw YAML entry.

    Args:
        raw: Dict parsed from YAML for one square.
        name_override: Locale-specific name override, or None.

    Returns:
        The appropriate frozen Square subclass instance.
    """
    name = name_override if name_override is not None else raw["name"]
    sq_type = SquareType(raw["type"])
    position = raw["position"]

    if sq_type == SquareType.property:
        return ColorProperty(
            position=position,
            name=name,
            type=sq_type,
            price=raw["price"],
            mortgage=raw["mortgage"],
            color=raw["color"],
            house_cost=raw["house_cost"],
            rents=tuple(raw["rents"]),
        )
    if sq_type == SquareType.railroad:
        return Railroad(
            position=position,
            name=name,
            type=sq_type,
            price=raw["price"],
            mortgage=raw["mortgage"],
            rents=tuple(raw["rents"]),
        )
    if sq_type == SquareType.utility:
        return Utility(
            position=position,
            name=name,
            type=sq_type,
            price=raw["price"],
            mortgage=raw["mortgage"],
            rents=tuple(raw["rents"]),
        )
    if sq_type == SquareType.tax:
        return TaxSquare(
            position=position,
            name=name,
            type=sq_type,
            amount=raw["amount"],
        )
    return Square(position=position, name=name, type=sq_type)


# ---------------------------------------------------------------------------
# Board
# ---------------------------------------------------------------------------


class Board:
    """Monopoly board: 40 immutable squares with locale support.

    Args:
        locale: "us" for standard English names, "it" for Italian names.
    """

    def __init__(self, locale: str = "us") -> None:
        standard_path = _DATA_DIR / "board_standard.yaml"
        raw_board: dict[str, Any] = yaml.safe_load(standard_path.read_text())
        name_overrides = self._load_locale_overrides(locale)
        self._squares: list[Square] = [
            _build_square(entry, name_overrides.get(entry["position"]))
            for entry in raw_board["squares"]
        ]
        self._by_position: dict[int, Square] = {sq.position: sq for sq in self._squares}

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def squares(self) -> list[Square]:
        """All 40 squares in board order."""
        return list(self._squares)

    @property
    def buyable_squares(self) -> list[BuyableSquare]:
        """All squares that can be purchased (properties, railroads, utilities)."""
        return [sq for sq in self._squares if isinstance(sq, BuyableSquare)]

    def get_square(self, position: int) -> Square:
        """Return the square at the given position.

        Args:
            position: Board position (0–39).

        Returns:
            The Square at that position.

        Raises:
            ValueError: If position is outside 0–39.
        """
        if position not in self._by_position:
            raise ValueError(f"Invalid position: {position}. Must be 0–39.")
        return self._by_position[position]

    def get_group(self, color: str) -> list[ColorProperty]:
        """Return all ColorProperty squares in the given color group.

        Args:
            color: Color group name (e.g. "brown", "dark_blue").

        Returns:
            List of ColorProperty in that group, empty list if unknown color.
        """
        return [
            sq
            for sq in self._squares
            if isinstance(sq, ColorProperty) and sq.color == color
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_locale_overrides(locale: str) -> dict[int, str]:
        """Load name overrides for the given locale.

        Args:
            locale: Locale code ("us" or "it").

        Returns:
            Dict mapping position to locale-specific name. Empty for "us".
        """
        if locale == "us":
            return {}
        _locale_filenames: dict[str, str] = {
            "it": "board_italia.yaml",
        }
        filename = _locale_filenames.get(locale, f"board_{locale}.yaml")
        locale_path = _DATA_DIR / filename
        if not locale_path.exists():
            return {}
        entries: list[dict[str, Any]] = yaml.safe_load(locale_path.read_text())
        return {entry["position"]: entry["name_it"] for entry in entries}

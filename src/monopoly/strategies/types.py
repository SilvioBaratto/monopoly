"""Value objects and enums shared by Strategy and the game engine.

Responsibilities (SRP):
- JailDecision: enum of decisions available to a jailed player
- BuildOrder: instruction to add houses/hotel to a single property
- SellOrder: instruction to remove houses/hotel from a single property
- TradeOffer: proposed property exchange between two players

This module is owned by the strategies package. The jail module imports
JailDecision from here — not the other way around — to avoid circular deps.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class JailDecision(str, Enum):
    """Available decisions when a player is in jail at turn start."""

    PAY_FINE = "pay_fine"
    USE_GOOJF = "use_goojf"
    ROLL_DOUBLES = "roll_doubles"


@dataclass
class BuildOrder:
    """Instructions to build houses/hotel on a single property.

    Args:
        position: Board position of the property.
        count: Number of houses to add (1–5; 5 means hotel).
    """

    position: int
    count: int


@dataclass
class SellOrder:
    """Instructions to sell houses/hotel on a single property.

    Args:
        position: Board position of the property.
        count: Number of houses to sell (1–4 for houses; 5 means sell hotel).
    """

    position: int
    count: int


@dataclass
class TradeOffer:
    """A proposed property trade between two players.

    Args:
        offered_positions: Board positions the proposer will give.
        requested_positions: Board positions the proposer wants in return.
        cash_offered: Cash the proposer will pay.
        cash_requested: Cash the proposer wants in return.
    """

    offered_positions: list[int]
    requested_positions: list[int]
    cash_offered: int = 0
    cash_requested: int = 0

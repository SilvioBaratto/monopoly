"""Mortgage and unmortgage operations for Monopoly.

Responsibilities (SRP):
- Mortgage a property: mark as mortgaged, give player mortgage value
- Unmortgage a property: pay mortgage value + 10% interest to unencumber

No building or bankruptcy logic — this module handles mortgage state only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from monopoly.board import BuyableSquare

if TYPE_CHECKING:
    from monopoly.state import GameState, Player

_UNMORTGAGE_INTEREST_RATE = 0.1


def mortgage_property(
    player: Player,
    position: int,
    game_state: GameState,
) -> bool:
    """Mortgage a property, giving the player its mortgage value.

    Cannot mortgage if the property has houses or a hotel.

    Args:
        player: The player who owns the property.
        position: Board position of the property.
        game_state: Current game state.

    Returns:
        True if successfully mortgaged, False otherwise.
    """
    if position not in game_state.property_ownership:
        return False

    ownership = game_state.property_ownership[position]

    if ownership.owner is not player:
        return False
    if ownership.is_mortgaged:
        return False
    if ownership.houses > 0 or ownership.has_hotel:
        return False

    square = game_state.board.get_square(position)
    if not isinstance(square, BuyableSquare):
        return False

    ownership.is_mortgaged = True
    player.cash += square.mortgage
    return True


def unmortgage_property(
    player: Player,
    position: int,
    game_state: GameState,
) -> bool:
    """Unmortgage a property by paying mortgage value + 10% interest.

    Args:
        player: The player who owns the property.
        position: Board position of the property.
        game_state: Current game state.

    Returns:
        True if successfully unmortgaged, False otherwise.
    """
    if position not in game_state.property_ownership:
        return False

    ownership = game_state.property_ownership[position]

    if ownership.owner is not player:
        return False
    if not ownership.is_mortgaged:
        return False

    square = game_state.board.get_square(position)
    if not isinstance(square, BuyableSquare):
        return False

    interest = round(square.mortgage * _UNMORTGAGE_INTEREST_RATE)
    total_cost = square.mortgage + interest

    if player.cash < total_cost:
        return False

    player.cash -= total_cost
    ownership.is_mortgaged = False
    return True

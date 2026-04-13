"""Pure rent calculation for Monopoly.

Responsibilities (SRP):
- Compute rent owed when a player lands on a property
- Handle all property types: ColorProperty, Railroad, Utility
- Apply monopoly (color group complete) and house/hotel multipliers

No state mutation — this module is a pure calculator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from monopoly.board import ColorProperty, Railroad, Utility

if TYPE_CHECKING:
    from monopoly.state import GameState, Player


def calculate_rent(
    landing_player: Player,
    position: int,
    game_state: GameState,
    dice_total: int | None = None,
    double_rent: bool = False,
    force_dice_multiplier: int | None = None,
) -> int:
    """Compute rent owed when landing_player lands on the given position.

    Returns 0 when:
    - The property is unowned
    - The property is owned by landing_player
    - The property is mortgaged

    Args:
        landing_player: The player who just landed on the square.
        position: Board position of the square landed on.
        game_state: Current game state.
        dice_total: Sum of the dice roll (required for utilities).
        double_rent: If True, double the computed rent (for Chance cards).
        force_dice_multiplier: Override dice multiplier for utilities (Chance card).

    Returns:
        Rent amount in dollars (0 if no rent is due).
    """
    if position not in game_state.property_ownership:
        return 0

    ownership = game_state.property_ownership[position]

    if ownership.owner is None:
        return 0
    if ownership.owner is landing_player:
        return 0
    if ownership.is_mortgaged:
        return 0

    square = game_state.board.get_square(position)

    if isinstance(square, ColorProperty):
        rent = _color_property_rent(square, ownership, game_state)
    elif isinstance(square, Railroad):
        rent = _railroad_rent(square, ownership, game_state)
    elif isinstance(square, Utility):
        rent = _utility_rent(
            square, ownership, game_state, dice_total, force_dice_multiplier
        )
    else:
        return 0

    if double_rent:
        rent *= 2

    return rent


def _color_property_rent(
    square: ColorProperty,
    ownership: object,
    game_state: GameState,
) -> int:
    """Compute rent for a color property.

    Args:
        square: The color property square.
        ownership: PropertyOwnership instance.
        game_state: Current game state.

    Returns:
        Rent amount in dollars.
    """
    from monopoly.state import PropertyOwnership

    assert isinstance(ownership, PropertyOwnership)

    if ownership.has_hotel:
        return square.rents[5]

    if ownership.houses > 0:
        return square.rents[ownership.houses]

    # No houses: check if owner has monopoly (all group owned and unmortgaged)
    owner = ownership.owner
    group = game_state.board.get_group(square.color)
    group_positions = {sq.position for sq in group}

    all_owned_unmortgaged = all(
        game_state.property_ownership[pos].owner is owner
        and not game_state.property_ownership[pos].is_mortgaged
        for pos in group_positions
        if pos in game_state.property_ownership
    )

    if all_owned_unmortgaged and len(group) > 0:
        return square.rents[0] * 2

    return square.rents[0]


def _railroad_rent(
    square: Railroad,
    ownership: object,
    game_state: GameState,
) -> int:
    """Compute rent for a railroad based on how many railroads the owner has.

    Args:
        square: The railroad square.
        ownership: PropertyOwnership instance.
        game_state: Current game state.

    Returns:
        Rent amount in dollars.
    """
    from monopoly.state import PropertyOwnership

    assert isinstance(ownership, PropertyOwnership)

    owner = ownership.owner
    railroad_positions = [5, 15, 25, 35]

    count_owned = sum(
        1
        for pos in railroad_positions
        if pos in game_state.property_ownership
        and game_state.property_ownership[pos].owner is owner
        and not game_state.property_ownership[pos].is_mortgaged
    )

    if count_owned == 0:
        return 0

    return square.rents[count_owned - 1]


def _utility_rent(
    square: Utility,
    ownership: object,
    game_state: GameState,
    dice_total: int | None,
    force_dice_multiplier: int | None,
) -> int:
    """Compute rent for a utility.

    Args:
        square: The utility square.
        ownership: PropertyOwnership instance.
        game_state: Current game state.
        dice_total: The dice total rolled when landing here.
        force_dice_multiplier: Overrides the multiplier (from Chance cards).

    Returns:
        Rent amount in dollars.
    """
    from monopoly.state import PropertyOwnership

    assert isinstance(ownership, PropertyOwnership)

    if dice_total is None:
        raise ValueError(
            "dice_roll must be provided (non-None) when landing on a Utility"
        )

    owner = ownership.owner
    utility_positions = [12, 28]

    count_owned = sum(
        1
        for pos in utility_positions
        if pos in game_state.property_ownership
        and game_state.property_ownership[pos].owner is owner
        and not game_state.property_ownership[pos].is_mortgaged
    )

    if force_dice_multiplier is not None:
        multiplier = force_dice_multiplier
    else:
        multiplier = square.rents[min(count_owned, 2) - 1]

    return multiplier * dice_total

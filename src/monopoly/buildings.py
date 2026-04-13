"""House and hotel building management for Monopoly.

Responsibilities (SRP):
- Execute build orders with even-building rule enforcement
- Sell houses/hotels and return cash to player
- Manage global house/hotel supply limits

No purchase or bankruptcy logic — this module handles building state only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from monopoly.board import ColorProperty

if TYPE_CHECKING:
    from monopoly.state import GameState, Player
    from monopoly.strategies.base import BuildOrder, SellOrder

_MAX_HOUSES = 4
_HOUSES_PER_HOTEL = 4


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BuildResult:
    """Result of a single build order.

    Args:
        position: Board position of the property.
        success: Whether the build succeeded.
        reason: Failure reason if success is False.
    """

    position: int
    success: bool
    reason: str = ""


@dataclass
class SellResult:
    """Result of a single sell order.

    Args:
        position: Board position of the property.
        success: Whether the sale succeeded.
        reason: Failure reason if success is False.
        cash_received: Cash received from the sale.
    """

    position: int
    success: bool
    reason: str = ""
    cash_received: int = 0


def execute_build_orders(
    player: Player,
    game_state: GameState,
    build_orders: list[BuildOrder],
) -> None:
    """Execute a list of build orders for a player, discarding per-order results.

    Thin wrapper around :func:`build_houses` — all validation and supply
    management is delegated there.

    Args:
        player: The building player.
        game_state: Current game state.
        build_orders: Ordered list of BuildOrder instructions.
    """
    build_houses(player, build_orders, game_state)


def _owns_full_group(
    player: Player,
    color: str,
    game_state: GameState,
) -> bool:
    """Check if a player owns all properties in a color group, none mortgaged.

    Args:
        player: The player to check.
        color: Color group identifier.
        game_state: Current game state.

    Returns:
        True if player owns all unmortgaged properties in the group.
    """
    group = game_state.board.get_group(color)
    return all(
        game_state.property_ownership[sq.position].owner is player
        and not game_state.property_ownership[sq.position].is_mortgaged
        for sq in group
        if sq.position in game_state.property_ownership
    )


def _even_build_allows(
    color: str,
    ownership: object,
    game_state: GameState,
    to_hotel: bool,
) -> bool:
    """Check even-building rule: max 1 house difference within group.

    Args:
        color: Color group.
        ownership: The PropertyOwnership being built on.
        game_state: Current game state.
        to_hotel: Whether we're attempting to build a hotel.

    Returns:
        True if even-building rule allows this build.
    """
    from monopoly.state import PropertyOwnership

    assert isinstance(ownership, PropertyOwnership)

    group = game_state.board.get_group(color)
    levels = []
    for sq in group:
        po = game_state.property_ownership.get(sq.position)
        if po is None:
            continue
        if po.has_hotel:
            levels.append(5)  # hotel counts as level 5
        else:
            levels.append(po.houses)

    if not levels:
        return True

    current_level = 5 if ownership.has_hotel else ownership.houses
    target_level = current_level + 1

    # All others must be at least target_level - 1
    other_levels = [lvl for lvl in levels if lvl != current_level]
    # Handle group of size 1
    if not other_levels:
        return True

    min_other = min(other_levels)
    return min_other >= target_level - 1


def sell_buildings(
    player: Player,
    game_state: GameState,
    position: int,
    count: int,
) -> int:
    """Sell houses or a hotel at a position, returning cash to player.

    Thin wrapper around :func:`sell_houses` — all validation and supply
    management is delegated there.

    Args:
        player: The property owner.
        game_state: Current game state.
        position: Board position of the property.
        count: Number of houses to sell (1 or more; hotel is sold whenever
            ``has_hotel`` is True regardless of count).

    Returns:
        Cash received from the sale, or 0 if the sale was rejected.
    """
    from monopoly.strategies.types import SellOrder

    results = sell_houses(player, [SellOrder(position=position, count=count)], game_state)
    return results[0].cash_received if results else 0


# ---------------------------------------------------------------------------
# build_houses — returns per-order BuildResult list
# ---------------------------------------------------------------------------


def build_houses(
    player: Player,
    build_orders: list[BuildOrder],
    game_state: GameState,
) -> list[BuildResult]:
    """Build houses/hotels for each order, returning per-order results.

    Each BuildResult has success=True if built, or success=False with a reason:
    - "not owner or invalid"
    - "mortgaged"
    - "incomplete group"
    - "even building rule"
    - "housing shortage"
    - "hotel shortage"
    - "already hotel"
    - "insufficient cash"

    Args:
        player: The building player.
        build_orders: Ordered list of BuildOrder instructions.
        game_state: Current game state.

    Returns:
        One BuildResult per BuildOrder.
    """
    return [_build_one_order(player, order, game_state) for order in build_orders]


def _build_one_order(
    player: Player,
    order: BuildOrder,
    game_state: GameState,
) -> BuildResult:
    """Attempt to fulfil one BuildOrder, returning a BuildResult.

    Args:
        player: The building player.
        order: The build instruction.
        game_state: Current game state.

    Returns:
        BuildResult indicating success or the first failure reason.
    """
    from monopoly.state import PropertyOwnership

    position = order.position
    failure = _validate_build_preconditions(player, position, game_state)
    if failure:
        return BuildResult(position=position, success=False, reason=failure)

    ownership = game_state.property_ownership[position]
    assert isinstance(ownership, PropertyOwnership)
    square = game_state.board.get_square(position)
    assert isinstance(square, ColorProperty)

    for _ in range(order.count):
        reason = _try_build_one(player, square, ownership, game_state)
        if reason:
            return BuildResult(position=position, success=False, reason=reason)

    return BuildResult(position=position, success=True)


def _validate_build_preconditions(
    player: Player,
    position: int,
    game_state: GameState,
) -> str:
    """Return a failure reason string, or empty string if all checks pass.

    Args:
        player: The building player.
        position: Board position.
        game_state: Current game state.

    Returns:
        Failure reason or "" if valid.
    """
    if position not in game_state.property_ownership:
        return "not owner or invalid"

    ownership = game_state.property_ownership[position]
    if ownership.owner is not player:
        return "not owner or invalid"

    square = game_state.board.get_square(position)
    if not isinstance(square, ColorProperty):
        return "not owner or invalid"

    if ownership.is_mortgaged:
        return "mortgaged"

    if not _owns_full_group(player, square.color, game_state):
        return "incomplete group"

    return ""


def _try_build_one(
    player: Player,
    square: ColorProperty,
    ownership: object,
    game_state: GameState,
) -> str:
    """Attempt to build exactly one house/hotel, returning a failure reason or "".

    Args:
        player: The building player.
        square: The color property square.
        ownership: PropertyOwnership for the square.
        game_state: Current game state.

    Returns:
        Failure reason string, or "" on success.
    """
    from monopoly.state import PropertyOwnership

    assert isinstance(ownership, PropertyOwnership)

    if ownership.has_hotel:
        return "already hotel"

    if ownership.houses >= _MAX_HOUSES:
        return _try_build_hotel(player, square, ownership, game_state)

    return _try_build_house(player, square, ownership, game_state)


def _try_build_house(
    player: Player,
    square: ColorProperty,
    ownership: object,
    game_state: GameState,
) -> str:
    """Build one house, returning a failure reason or "".

    Args:
        player: The building player.
        square: The color property.
        ownership: PropertyOwnership instance.
        game_state: Current game state.

    Returns:
        Failure reason or "" on success.
    """
    from monopoly.state import PropertyOwnership

    assert isinstance(ownership, PropertyOwnership)

    if game_state.houses_available < 1:
        return "housing shortage"
    if not _even_build_allows(square.color, ownership, game_state, to_hotel=False):
        return "even building rule"
    if player.cash < square.house_cost:
        return "insufficient cash"

    game_state.houses_available -= 1
    ownership.houses += 1
    player.cash -= square.house_cost
    return ""


def _try_build_hotel(
    player: Player,
    square: ColorProperty,
    ownership: object,
    game_state: GameState,
) -> str:
    """Build one hotel (upgrading from 4 houses), returning a failure reason or "".

    Args:
        player: The building player.
        square: The color property.
        ownership: PropertyOwnership instance.
        game_state: Current game state.

    Returns:
        Failure reason or "" on success.
    """
    from monopoly.state import PropertyOwnership

    assert isinstance(ownership, PropertyOwnership)

    if game_state.hotels_available < 1:
        return "hotel shortage"
    if not _even_build_allows(square.color, ownership, game_state, to_hotel=True):
        return "even building rule"
    if player.cash < square.house_cost:
        return "insufficient cash"

    game_state.houses_available += _HOUSES_PER_HOTEL
    game_state.hotels_available -= 1
    ownership.houses = 0
    ownership.has_hotel = True
    player.cash -= square.house_cost
    return ""


# ---------------------------------------------------------------------------
# sell_houses — returns per-order SellResult list
# ---------------------------------------------------------------------------


def sell_houses(
    player: Player,
    sell_orders: list[SellOrder],
    game_state: GameState,
) -> list[SellResult]:
    """Sell houses/hotels for each order, returning per-order results.

    Each SellResult has success=True with cash_received, or success=False with reason:
    - "not owner or invalid"
    - "no buildings to sell"
    - "even selling rule"
    - "insufficient houses in supply"

    Args:
        player: The property owner.
        sell_orders: Ordered list of SellOrder instructions.
        game_state: Current game state.

    Returns:
        One SellResult per SellOrder.
    """
    return [_sell_one_order(player, order, game_state) for order in sell_orders]


def _sell_one_order(
    player: Player,
    order: SellOrder,
    game_state: GameState,
) -> SellResult:
    """Attempt to fulfil one SellOrder, returning a SellResult.

    Args:
        player: The property owner.
        order: The sell instruction.
        game_state: Current game state.

    Returns:
        SellResult indicating success or the failure reason.
    """
    from monopoly.state import PropertyOwnership

    position = order.position
    failure = _validate_sell_preconditions(player, position, game_state)
    if failure:
        return SellResult(position=position, success=False, reason=failure)

    ownership = game_state.property_ownership[position]
    assert isinstance(ownership, PropertyOwnership)
    square = game_state.board.get_square(position)
    assert isinstance(square, ColorProperty)

    if ownership.has_hotel:
        return _sell_hotel(player, square, ownership, game_state, position)

    return _sell_houses_from_property(
        player, square, ownership, game_state, position, order.count
    )


def _validate_sell_preconditions(
    player: Player,
    position: int,
    game_state: GameState,
) -> str:
    """Return a failure reason for sell, or "" if valid.

    Args:
        player: The property owner.
        position: Board position.
        game_state: Current game state.

    Returns:
        Failure reason or "".
    """
    if position not in game_state.property_ownership:
        return "not owner or invalid"

    ownership = game_state.property_ownership[position]
    if ownership.owner is not player:
        return "not owner or invalid"

    square = game_state.board.get_square(position)
    if not isinstance(square, ColorProperty):
        return "not owner or invalid"

    if not ownership.has_hotel and ownership.houses == 0:
        return "no buildings to sell"

    return ""


def _sell_hotel(
    player: Player,
    square: ColorProperty,
    ownership: object,
    game_state: GameState,
    position: int,
) -> SellResult:
    """Sell a hotel, requiring >= 4 houses in supply.

    Args:
        player: The property owner.
        square: The color property.
        ownership: PropertyOwnership instance.
        game_state: Current game state.
        position: Board position (for result).

    Returns:
        SellResult for the hotel sale.
    """
    from monopoly.state import PropertyOwnership

    assert isinstance(ownership, PropertyOwnership)

    if game_state.houses_available < _HOUSES_PER_HOTEL:
        return SellResult(
            position=position,
            success=False,
            reason="insufficient houses in supply",
        )

    if not _even_sell_allows(square.color, ownership, game_state):
        return SellResult(position=position, success=False, reason="even selling rule")

    sell_price = square.house_cost // 2 * 5
    ownership.has_hotel = False
    ownership.houses = 0
    game_state.hotels_available += 1
    game_state.houses_available += _HOUSES_PER_HOTEL
    player.cash += sell_price
    return SellResult(position=position, success=True, cash_received=sell_price)


def _sell_houses_from_property(
    player: Player,
    square: ColorProperty,
    ownership: object,
    game_state: GameState,
    position: int,
    count: int,
) -> SellResult:
    """Sell houses from a non-hotel property.

    Args:
        player: The property owner.
        square: The color property.
        ownership: PropertyOwnership instance.
        game_state: Current game state.
        position: Board position (for result).
        count: Number of houses to sell.

    Returns:
        SellResult for the house sale.
    """
    from monopoly.state import PropertyOwnership

    assert isinstance(ownership, PropertyOwnership)

    if not _even_sell_allows(square.color, ownership, game_state):
        return SellResult(position=position, success=False, reason="even selling rule")

    houses_to_sell = min(count, ownership.houses)
    sell_price = square.house_cost // 2 * houses_to_sell
    ownership.houses -= houses_to_sell
    game_state.houses_available += houses_to_sell
    player.cash += sell_price
    return SellResult(position=position, success=True, cash_received=sell_price)


def _even_sell_allows(
    color: str,
    ownership: object,
    game_state: GameState,
) -> bool:
    """Check even-selling rule: can only sell from the property with the most houses.

    A property P can be sold from if no other property in the group has FEWER
    buildings than P (i.e., P has the maximum or is tied for maximum).

    Args:
        color: Color group.
        ownership: The PropertyOwnership being sold from.
        game_state: Current game state.

    Returns:
        True if the even-selling rule allows this sale.
    """
    from monopoly.state import PropertyOwnership

    assert isinstance(ownership, PropertyOwnership)

    group = game_state.board.get_group(color)
    current_level = 5 if ownership.has_hotel else ownership.houses

    for sq in group:
        po = game_state.property_ownership.get(sq.position)
        if po is None or po is ownership:
            continue
        other_level = 5 if po.has_hotel else po.houses
        if other_level < current_level:
            return True  # current is max — allowed
        if other_level > current_level:
            return False  # another property has more — must sell from that one first

    return True  # all equal or sole property

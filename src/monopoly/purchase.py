"""Property purchase logic for Monopoly.

Responsibilities (SRP):
- Decide whether to attempt a purchase (delegate to strategy)
- Execute the purchase: deduct cash, set ownership

No rent or building logic — this module handles initial purchase only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from monopoly.board import BuyableSquare

if TYPE_CHECKING:
    from monopoly.board import Square
    from monopoly.state import GameState, Player
    from monopoly.strategies.base import Strategy


def attempt_purchase(
    player: Player,
    square: Square,
    game_state: GameState,
    strategy: Strategy,
) -> bool:
    """Attempt to purchase an unowned property.

    Asks the strategy whether to buy, then executes if cash is sufficient.

    Args:
        player: The player who landed on the square.
        square: The buyable square (must be a BuyableSquare and unowned).
        game_state: Current game state.
        strategy: The player's decision strategy.

    Returns:
        True if the purchase was completed, False otherwise.

    Raises:
        ValueError: If square is not a BuyableSquare or is already owned.
    """
    if not isinstance(square, BuyableSquare):
        raise ValueError(
            f"Square at position {square.position} is not a BuyableSquare."
        )

    ownership = game_state.property_ownership.get(square.position)
    if ownership is not None and ownership.owner is not None:
        raise ValueError(
            f"Square '{square.name}' at position {square.position} is already owned."
        )

    if not strategy.should_buy_property(player, square, game_state):
        return False

    if player.cash < square.price:
        return False

    player.cash -= square.price
    game_state.property_ownership[square.position].owner = player
    return True

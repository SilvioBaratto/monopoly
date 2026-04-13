"""Bankruptcy and payment handling for Monopoly.

Responsibilities (SRP):
- Attempt to pay an amount, raising cash via building sales/mortgages if needed
- Transfer assets to creditor or bank on bankruptcy
- Mark player bankrupt and detect game-over condition

No turn or rent logic — this module handles payment resolution only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from monopoly.board import BuyableSquare, ColorProperty
from monopoly.buildings import sell_buildings
from monopoly.mortgage import mortgage_property

if TYPE_CHECKING:
    from monopoly.state import GameState, Player
    from monopoly.strategies.base import Strategy

_MORTGAGE_INTEREST_RATE = 0.1


def attempt_payment(
    player: Player,
    amount: int,
    creditor: Player | None,
    game_state: GameState,
    strategy: Strategy,
) -> bool:
    """Try to pay an amount, liquidating assets if necessary.

    Payment flow:
    1. If cash >= amount: pay directly and return True.
    2. Sell houses/hotels at half cost to raise cash.
    3. Mortgage properties (strategy decides which) to raise cash.
    4. If still can't pay: declare bankruptcy and return False.

    Args:
        player: The player making the payment.
        amount: Amount to pay.
        creditor: Player receiving the payment, or None (bank).
        game_state: Current game state.
        strategy: The player's decision strategy.

    Returns:
        True if payment succeeded, False if player went bankrupt.
    """
    if player.cash >= amount:
        _transfer_cash(player, creditor, amount)
        return True

    # Monopoly rules: sell houses first, then mortgage
    _raise_cash_via_building_sales(player, amount, game_state)

    if player.cash >= amount:
        _transfer_cash(player, creditor, amount)
        return True

    _raise_cash_via_mortgages(player, amount, game_state, strategy)

    if player.cash >= amount:
        _transfer_cash(player, creditor, amount)
        return True

    declare_bankruptcy(player, creditor, game_state)
    return False


def declare_bankruptcy(
    player: Player,
    creditor: Player | None,
    game_state: GameState,
) -> None:
    """Transfer all assets and mark the player bankrupt.

    If creditor is a player: transfer cash, properties (charging 10% interest
    on each mortgaged property), and GOOJF cards.
    If creditor is bank: clear all property ownership and return buildings to supply.

    Args:
        player: The bankrupt player.
        creditor: Receiving player, or None (bank).
        game_state: Current game state.
    """
    player.bankrupt = True

    if creditor is not None:
        _transfer_assets_to_creditor(player, creditor, game_state)
    else:
        _return_assets_to_bank(player, game_state)


def check_game_over(game_state: GameState) -> bool:
    """Return True when one or fewer active (non-bankrupt) players remain.

    Args:
        game_state: Current game state.

    Returns:
        True if the game is over, False otherwise.
    """
    return len(game_state.active_players) <= 1


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _transfer_cash(
    player: Player,
    creditor: Player | None,
    amount: int,
) -> None:
    """Transfer cash from player to creditor (or discard to bank).

    Args:
        player: The paying player.
        creditor: Receiving player, or None for the bank.
        amount: Amount to transfer.
    """
    player.cash -= amount
    if creditor is not None:
        creditor.cash += amount


def _raise_cash_via_building_sales(
    player: Player,
    amount: int,
    game_state: GameState,
) -> None:
    """Sell houses/hotels to raise cash until amount is met.

    Sells buildings from most expensive properties first.

    Args:
        player: The cash-strapped player.
        amount: Target cash amount to reach.
        game_state: Current game state.
    """
    built_positions = _built_positions_sorted_by_cost(player, game_state)

    for pos in built_positions:
        if player.cash >= amount:
            break
        po = game_state.property_ownership[pos]
        if po.has_hotel:
            sell_buildings(player, game_state, pos, 1)
        while po.houses > 0 and player.cash < amount:
            if sell_buildings(player, game_state, pos, 1) == 0:
                break  # even-selling rule blocked this position; move to the next


def _built_positions_sorted_by_cost(
    player: Player,
    game_state: GameState,
) -> list[int]:
    """Return positions with buildings owned by player, sorted by house_cost descending.

    Args:
        player: The player whose buildings to consider.
        game_state: Current game state.

    Returns:
        Sorted list of board positions with buildings.
    """
    positions = [
        pos
        for pos, po in game_state.property_ownership.items()
        if po.owner is player and (po.houses > 0 or po.has_hotel)
    ]
    positions.sort(
        key=lambda pos: _house_cost(pos, game_state),
        reverse=True,
    )
    return positions


def _house_cost(pos: int, game_state: GameState) -> int:
    """Return house_cost for a position, or 0 if not a ColorProperty."""
    sq = game_state.board.get_square(pos)
    if isinstance(sq, ColorProperty):
        return sq.house_cost
    return 0


def _raise_cash_via_mortgages(
    player: Player,
    amount: int,
    game_state: GameState,
    strategy: Strategy,
) -> None:
    """Mortgage properties to raise cash until amount is met.

    Args:
        player: The cash-strapped player.
        amount: Target cash amount to reach.
        game_state: Current game state.
        strategy: Decision strategy for choosing which to mortgage.
    """
    positions = strategy.choose_properties_to_mortgage(player, amount, game_state)
    for pos in positions:
        if player.cash >= amount:
            break
        mortgage_property(player, pos, game_state)


def _transfer_assets_to_creditor(
    player: Player,
    creditor: Player,
    game_state: GameState,
) -> None:
    """Transfer all player assets (cash, properties, GOOJF cards) to creditor.

    For each transferred mortgaged property, the creditor is charged 10%
    of the mortgage value immediately (if they can afford it).

    Args:
        player: The bankrupt player.
        creditor: The player receiving the assets.
        game_state: Current game state.
    """
    creditor.cash += player.cash
    player.cash = 0

    for po in game_state.property_ownership.values():
        if po.owner is player:
            po.owner = creditor
            if po.is_mortgaged:
                _charge_mortgage_interest(creditor, po, game_state)

    creditor.goojf_cards.extend(player.goojf_cards)
    player.goojf_cards.clear()


def _charge_mortgage_interest(
    creditor: Player,
    ownership: object,
    game_state: GameState,
) -> None:
    """Deduct 10% mortgage interest from creditor for a transferred mortgaged property.

    If creditor cannot afford the interest, property stays mortgaged (no action taken).

    Args:
        creditor: The player receiving the mortgaged property.
        ownership: The PropertyOwnership instance (already transferred to creditor).
        game_state: Current game state.
    """
    from monopoly.state import PropertyOwnership

    assert isinstance(ownership, PropertyOwnership)

    # Resolve the board position to get the mortgage value
    for pos, po in game_state.property_ownership.items():
        if po is not ownership:
            continue
        sq = game_state.board.get_square(pos)
        if not isinstance(sq, BuyableSquare):
            return
        interest = round(sq.mortgage * _MORTGAGE_INTEREST_RATE)
        if creditor.cash >= interest:
            creditor.cash -= interest
        return


def _return_assets_to_bank(
    player: Player,
    game_state: GameState,
) -> None:
    """Return all player assets to the bank.

    Properties are freed (no owner, no buildings, not mortgaged).
    Buildings return to supply. GOOJF cards return to their decks.

    Args:
        player: The bankrupt player.
        game_state: Current game state.
    """
    player.cash = 0

    for po in game_state.property_ownership.values():
        if po.owner is player:
            houses_to_return = po.houses
            po.owner = None
            po.has_hotel = False
            po.houses = 0
            po.is_mortgaged = False
            game_state.houses_available = min(
                32, game_state.houses_available + houses_to_return
            )

    for card in player.goojf_cards:
        if card.id.startswith("ch_"):
            game_state.chance_deck.return_card(card)
        else:
            game_state.community_chest_deck.return_card(card)
    player.goojf_cards.clear()

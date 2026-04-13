"""ThreeHousesRush strategy — build to exactly 3 houses on complete groups."""

from __future__ import annotations

from typing import TYPE_CHECKING

from monopoly.board import BuyableSquare, ColorProperty
from monopoly.strategies.base import BuildOrder, JailDecision, Strategy, TradeOffer
from monopoly.strategies.buy_everything import _complete_unmortgaged_colors

if TYPE_CHECKING:
    from monopoly.state import GameState, Player

_CASH_RESERVE = 150
_TARGET_HOUSES = 3


class ThreeHousesRush(Strategy):
    """Build to exactly 3 houses on all complete color groups, then stop.

    This targets the optimal rent/cost sweet spot in standard Monopoly while
    hoarding houses to starve opponents of the 32-house supply.
    """

    def should_buy_property(
        self,
        player: Player,
        square: BuyableSquare,
        game_state: GameState,
    ) -> bool:
        if _would_complete_group(player, square, game_state):
            return player.cash >= square.price
        return player.cash >= square.price + _CASH_RESERVE

    def choose_properties_to_build(
        self,
        player: Player,
        game_state: GameState,
    ) -> list[BuildOrder]:
        """Build toward 3 houses per property; never exceed 3.

        Priorities:
        1. Groups with the highest minimum house count go first.
        2. Within a group, build on least-developed properties first (even-building).
        3. Building stops when remaining cash would drop below _CASH_RESERVE.
        """
        complete = _complete_unmortgaged_colors(player, game_state)
        sorted_colors = sorted(
            complete,
            key=lambda c: _group_min_houses(c, player, game_state),
            reverse=True,
        )

        orders: list[BuildOrder] = []
        available = player.cash - _CASH_RESERVE

        for color in sorted_colors:
            group = game_state.board.get_group(color)
            sorted_group = sorted(
                group,
                key=lambda sq: _house_count(sq.position, player, game_state),
            )
            for sq in sorted_group:
                if available <= 0:
                    return orders
                order = _build_order_for(sq, player, game_state, available)
                if order is not None:
                    orders.append(order)
                    available -= order.count * sq.house_cost

        return orders

    def get_jail_decision(
        self,
        player: Player,
        game_state: GameState,
    ) -> JailDecision:
        if not _is_early_game(game_state):
            return JailDecision.ROLL_DOUBLES
        if player.goojf_cards:
            return JailDecision.USE_GOOJF
        if player.cash >= _JAIL_FINE + _CASH_RESERVE:
            return JailDecision.PAY_FINE
        return JailDecision.ROLL_DOUBLES

    def choose_properties_to_mortgage(
        self,
        player: Player,
        amount_needed: int,
        game_state: GameState,
    ) -> list[int]:
        """Mortgage non-complete groups first by lowest rent; skip properties with houses."""
        complete = _complete_unmortgaged_colors(player, game_state)
        non_complete: list[int] = []
        in_complete: list[int] = []

        for pos, po in game_state.property_ownership.items():
            if po.owner is not player or po.is_mortgaged:
                continue
            if po.houses > 0 or po.has_hotel:
                continue
            sq = game_state.board.get_square(pos)
            if not isinstance(sq, BuyableSquare):
                continue
            if isinstance(sq, ColorProperty) and sq.color in complete:
                in_complete.append(pos)
            else:
                non_complete.append(pos)

        non_complete.sort(key=lambda pos: _mortgage_value(pos, game_state))
        in_complete.sort(key=lambda pos: _mortgage_value(pos, game_state))
        return non_complete + in_complete

    def should_accept_trade(
        self,
        player: Player,
        trade_offer: TradeOffer,
        game_state: GameState,
    ) -> bool:
        return False

    def propose_trade(
        self,
        player: Player,
        game_state: GameState,
    ) -> TradeOffer | None:
        return None


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

_JAIL_FINE = 50


def _would_complete_group(
    player: Player,
    square: BuyableSquare,
    game_state: GameState,
) -> bool:
    """Return True if buying square gives player a complete color group."""
    if not isinstance(square, ColorProperty):
        return False
    group = game_state.board.get_group(square.color)
    for sq in group:
        if sq.position == square.position:
            continue
        po = game_state.property_ownership.get(sq.position)
        if po is None or po.owner is not player:
            return False
    return True


def _group_min_houses(
    color: str,
    player: Player,
    game_state: GameState,
) -> int:
    """Return the minimum house count among player-owned properties in a group."""
    group = game_state.board.get_group(color)
    counts = [
        game_state.property_ownership[sq.position].houses
        for sq in group
        if sq.position in game_state.property_ownership
        and game_state.property_ownership[sq.position].owner is player
    ]
    return min(counts, default=0)


def _house_count(pos: int, player: Player, game_state: GameState) -> int:
    """Return house count for a player-owned position (0 if not owned)."""
    po = game_state.property_ownership.get(pos)
    if po is None or po.owner is not player:
        return 0
    return po.houses


def _build_order_for(
    sq: ColorProperty,
    player: Player,
    game_state: GameState,
    available_cash: int,
) -> BuildOrder | None:
    """Return a capped BuildOrder for sq, or None if nothing to build."""
    po = game_state.property_ownership.get(sq.position)
    if po is None or po.owner is not player or po.is_mortgaged:
        return None
    if po.has_hotel or po.houses >= _TARGET_HOUSES:
        return None
    needed = _TARGET_HOUSES - po.houses
    affordable = available_cash // sq.house_cost
    count = min(needed, affordable)
    if count <= 0:
        return None
    return BuildOrder(position=sq.position, count=count)


def _is_early_game(game_state: GameState) -> bool:
    """Return True when fewer than half of all buyable properties are owned."""
    total = len(game_state.board.buyable_squares)
    owned = sum(
        1 for po in game_state.property_ownership.values() if po.owner is not None
    )
    return owned < total / 2


def _mortgage_value(pos: int, game_state: GameState) -> int:
    """Return mortgage value for sorting (lower = mortgage first)."""
    sq = game_state.board.get_square(pos)
    if isinstance(sq, BuyableSquare):
        return sq.mortgage
    return 0

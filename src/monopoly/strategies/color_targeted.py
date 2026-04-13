"""ColorTargeted strategy — focuses on acquiring and developing specific color groups."""

from __future__ import annotations

from typing import TYPE_CHECKING

from monopoly.board import BuyableSquare, ColorProperty
from monopoly.strategies.base import BuildOrder, JailDecision, Strategy, TradeOffer
from monopoly.strategies.buy_everything import _complete_unmortgaged_colors

if TYPE_CHECKING:
    from monopoly.state import GameState, Player


VALID_COLORS = frozenset(
    {
        "brown",
        "light_blue",
        "pink",
        "orange",
        "red",
        "yellow",
        "green",
        "dark_blue",
    }
)

# Module-level aliases for importability
CASH_RESERVE: int = 200
EARLY_GAME_THRESHOLD: int = 30

_CHEAP_THRESHOLD = 200
_TARGET_HOUSES = 3


class ColorTargeted(Strategy):
    """Buy target color properties aggressively; only buy non-target if priced under $200.

    Builds exclusively on completed target color monopolies, capping at 3 houses.
    In early game stays active; in late game camps in jail to avoid rent.

    Args:
        target_colors: Color groups this strategy prioritises.

    Raises:
        ValueError: If any color in target_colors is not a valid Monopoly color.
    """

    CASH_RESERVE: int = CASH_RESERVE
    EARLY_GAME_THRESHOLD: int = EARLY_GAME_THRESHOLD

    def __init__(self, target_colors: list[str]) -> None:
        for color in target_colors:
            if color not in VALID_COLORS:
                raise ValueError(f"Unknown color: {color!r}")
        self.target_colors = list(target_colors)

    def should_buy_property(
        self,
        player: Player,
        square: BuyableSquare,
        game_state: GameState,
    ) -> bool:
        if isinstance(square, ColorProperty):
            if square.color in self.target_colors:
                return player.cash >= square.price + CASH_RESERVE
            if square.price < _CHEAP_THRESHOLD:
                return player.cash >= square.price + CASH_RESERVE
        return False

    def choose_properties_to_build(
        self,
        player: Player,
        game_state: GameState,
    ) -> list[BuildOrder]:
        """Build on completed target color monopolies only, capping at 3 houses."""
        complete = _complete_unmortgaged_colors(player, game_state)
        orders: list[BuildOrder] = []
        available = player.cash - CASH_RESERVE

        for color in self.target_colors:
            if color not in complete:
                continue
            group = game_state.board.get_group(color)
            for sq in group:
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
        if game_state.turn_count >= EARLY_GAME_THRESHOLD:
            return JailDecision.ROLL_DOUBLES
        if player.goojf_cards:
            return JailDecision.USE_GOOJF
        return JailDecision.PAY_FINE

    def choose_properties_to_mortgage(
        self,
        player: Player,
        amount_needed: int,
        game_state: GameState,
    ) -> list[int]:
        """Mortgage non-target properties first, then target, by lowest mortgage value."""
        non_target: list[int] = []
        target: list[int] = []

        for pos, po in game_state.property_ownership.items():
            if po.owner is not player or po.is_mortgaged:
                continue
            if po.houses > 0 or po.has_hotel:
                continue
            sq = game_state.board.get_square(pos)
            if not isinstance(sq, BuyableSquare):
                continue

            if isinstance(sq, ColorProperty) and sq.color in self.target_colors:
                target.append(pos)
            else:
                non_target.append(pos)

        non_target.sort(key=lambda pos: _mortgage_value(pos, game_state))
        target.sort(key=lambda pos: _mortgage_value(pos, game_state))
        return non_target + target

    def should_accept_trade(
        self,
        player: Player,
        trade_offer: TradeOffer,
        game_state: GameState,
    ) -> bool:
        for pos in trade_offer.requested_positions:
            sq = game_state.board.get_square(pos)
            if isinstance(sq, ColorProperty) and sq.color in self.target_colors:
                return True
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


def _build_order_for(
    sq: ColorProperty,
    player: Player,
    game_state: GameState,
    available_cash: int,
) -> BuildOrder | None:
    """Return a BuildOrder capped at 3 houses, or None if nothing to build."""
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


def _mortgage_value(pos: int, game_state: GameState) -> int:
    """Return mortgage value for sorting (lower = mortgage first)."""
    sq = game_state.board.get_square(pos)
    if isinstance(sq, BuyableSquare):
        return sq.mortgage
    return 0

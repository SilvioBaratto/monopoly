"""BuyEverything strategy — buys every affordable property and builds aggressively."""

from __future__ import annotations

from typing import TYPE_CHECKING

from monopoly.board import BuyableSquare, ColorProperty
from monopoly.strategies.base import BuildOrder, JailDecision, Strategy, TradeOffer

if TYPE_CHECKING:
    from monopoly.state import GameState, Player


class BuyEverything(Strategy):
    """Greedy strategy: buy everything affordable, build on all complete groups.

    Mortgage priority: non-complete groups first, lowest mortgage value first.
    Jail decision: use GOOJF if available, else pay fine.
    """

    def should_buy_property(
        self,
        player: Player,
        square: BuyableSquare,
        game_state: GameState,
    ) -> bool:
        return player.cash >= square.price

    def choose_properties_to_build(
        self,
        player: Player,
        game_state: GameState,
    ) -> list[BuildOrder]:
        """Build on all complete color groups, cheapest house_cost first."""
        orders: list[BuildOrder] = []

        complete_colors = _complete_unmortgaged_colors(player, game_state)
        # Sort colors by house_cost ascending (cheapest first)
        color_costs = {}
        for color in complete_colors:
            group = game_state.board.get_group(color)
            if group:
                color_costs[color] = group[0].house_cost

        for color in sorted(complete_colors, key=lambda c: color_costs.get(c, 0)):
            group = game_state.board.get_group(color)
            for sq in group:
                po = game_state.property_ownership.get(sq.position)
                if po and po.owner is player and not po.is_mortgaged:
                    orders.append(BuildOrder(position=sq.position, count=1))

        return orders

    def get_jail_decision(
        self,
        player: Player,
        game_state: GameState,
    ) -> JailDecision:
        if player.goojf_cards:
            return JailDecision.USE_GOOJF
        return JailDecision.PAY_FINE

    def choose_properties_to_mortgage(
        self,
        player: Player,
        amount_needed: int,
        game_state: GameState,
    ) -> list[int]:
        """Mortgage non-complete groups first, lowest mortgage value first."""
        complete_colors = _complete_unmortgaged_colors(player, game_state)

        non_complete: list[int] = []
        complete_group: list[int] = []

        for pos, po in game_state.property_ownership.items():
            if po.owner is not player or po.is_mortgaged:
                continue
            if po.houses > 0 or po.has_hotel:
                continue  # Must sell buildings first
            sq = game_state.board.get_square(pos)
            if not isinstance(sq, BuyableSquare):
                continue

            if isinstance(sq, ColorProperty) and sq.color in complete_colors:
                complete_group.append(pos)
            else:
                non_complete.append(pos)

        def sort_key(pos: int) -> int:
            sq = game_state.board.get_square(pos)
            if isinstance(sq, BuyableSquare):
                return sq.mortgage
            return 0

        non_complete.sort(key=sort_key)
        complete_group.sort(key=sort_key)

        return non_complete + complete_group

    def should_accept_trade(
        self,
        player: Player,
        trade_offer: TradeOffer,
        game_state: GameState,
    ) -> bool:
        """Accept if the trade would give us a complete color group."""
        if not trade_offer.requested_positions:
            return False

        # Check if acquiring requested positions completes a color group
        for pos in trade_offer.requested_positions:
            sq = game_state.board.get_square(pos)
            if isinstance(sq, ColorProperty):
                group = game_state.board.get_group(sq.color)
                group_positions = {s.position for s in group}
                # After trade: we'd own our current props + requested - offered
                current_owned = {
                    p
                    for p, po in game_state.property_ownership.items()
                    if po.owner is player
                }
                after_trade = current_owned - set(trade_offer.offered_positions) | set(
                    trade_offer.requested_positions
                )
                if group_positions <= after_trade:
                    return True
        return False

    def propose_trade(
        self,
        player: Player,
        game_state: GameState,
    ) -> TradeOffer | None:
        return None


def _complete_unmortgaged_colors(player: Player, game_state: GameState) -> set[str]:
    """Return color groups where player owns all properties, none mortgaged."""
    from monopoly.board import ColorProperty

    owned_colors: dict[str, list[int]] = {}
    for pos, po in game_state.property_ownership.items():
        if po.owner is not player or po.is_mortgaged:
            continue
        sq = game_state.board.get_square(pos)
        if isinstance(sq, ColorProperty):
            owned_colors.setdefault(sq.color, []).append(pos)

    complete = set()
    for color, positions in owned_colors.items():
        group = game_state.board.get_group(color)
        if len(positions) == len(group):
            complete.add(color)
    return complete

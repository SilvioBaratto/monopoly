"""BuyNothing strategy — never purchases or builds anything."""

from __future__ import annotations

from typing import TYPE_CHECKING

from monopoly.strategies.base import BuildOrder, JailDecision, Strategy, TradeOffer

if TYPE_CHECKING:
    from monopoly.board import BuyableSquare
    from monopoly.state import GameState, Player


class BuyNothing(Strategy):
    """A passive strategy that never buys, builds, or trades.

    Useful as a baseline and for testing the game engine in isolation.
    """

    def should_buy_property(
        self,
        player: Player,
        square: BuyableSquare,
        game_state: GameState,
    ) -> bool:
        return False

    def choose_properties_to_build(
        self,
        player: Player,
        game_state: GameState,
    ) -> list[BuildOrder]:
        return []

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
        # Mortgage all owned properties sorted by lowest mortgage value first
        owned = [
            pos
            for pos, po in game_state.property_ownership.items()
            if po.owner is player
            and not po.is_mortgaged
            and po.houses == 0
            and not po.has_hotel
        ]
        from monopoly.board import BuyableSquare

        def mortgage_value(pos: int) -> int:
            sq = game_state.board.get_square(pos)
            if isinstance(sq, BuyableSquare):
                return sq.mortgage
            return 0

        return sorted(owned, key=mortgage_value)

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

"""JailCamper strategy — stays in jail during late game to avoid rent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from monopoly.strategies.base import BuildOrder, JailDecision, Strategy, TradeOffer
from monopoly.strategies.buy_everything import BuyEverything

if TYPE_CHECKING:
    from monopoly.board import BuyableSquare
    from monopoly.state import GameState, Player

_JAIL_FINE = 50


class JailCamper(Strategy):
    """Stay in jail during late game; delegate buying/building to BuyEverything.

    Late game is defined as: opponent-owned houses + hotels >= late_game_threshold.
    In late game, always attempt to stay in jail (ROLL_DOUBLES) to avoid landing
    on opponents' developed properties.

    Args:
        late_game_threshold: Minimum opponent houses+hotels to trigger late game.
    """

    def __init__(self, late_game_threshold: int = 6) -> None:
        self.late_game_threshold = late_game_threshold
        self._delegate = BuyEverything()

    def _is_late_game(self, player: Player, game_state: GameState) -> bool:
        """Return True when opponents have built enough houses/hotels.

        Args:
            player: The player making the decision (excluded from opponent count).
            game_state: Current game state.

        Returns:
            True if total opponent houses+hotels >= late_game_threshold.
        """
        total = sum(
            po.houses + (1 if po.has_hotel else 0)
            for po in game_state.property_ownership.values()
            if po.owner is not None and po.owner is not player
        )
        return total >= self.late_game_threshold

    def should_buy_property(
        self,
        player: Player,
        square: BuyableSquare,
        game_state: GameState,
    ) -> bool:
        return self._delegate.should_buy_property(player, square, game_state)

    def choose_properties_to_build(
        self,
        player: Player,
        game_state: GameState,
    ) -> list[BuildOrder]:
        return self._delegate.choose_properties_to_build(player, game_state)

    def get_jail_decision(
        self,
        player: Player,
        game_state: GameState,
    ) -> JailDecision:
        """Stay in jail in late game; otherwise pay fine or use GOOJF."""
        if self._is_late_game(player, game_state):
            return JailDecision.ROLL_DOUBLES

        if player.cash >= _JAIL_FINE:
            return JailDecision.PAY_FINE
        if player.goojf_cards:
            return JailDecision.USE_GOOJF
        return JailDecision.ROLL_DOUBLES

    def choose_properties_to_mortgage(
        self,
        player: Player,
        amount_needed: int,
        game_state: GameState,
    ) -> list[int]:
        return self._delegate.choose_properties_to_mortgage(
            player, amount_needed, game_state
        )

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

"""Trader strategy — values properties by strategic weight and proposes trades."""

from __future__ import annotations

from typing import TYPE_CHECKING

from monopoly.board import BuyableSquare, ColorProperty
from monopoly.strategies.base import BuildOrder, JailDecision, Strategy, TradeOffer
from monopoly.strategies.buy_everything import _complete_unmortgaged_colors

if TYPE_CHECKING:
    from monopoly.state import GameState, Player

# Strategic weight per color group (higher = more valuable to land on)
_POSITION_WEIGHT: dict[str, float] = {
    "brown": 0.7,
    "light_blue": 0.9,
    "pink": 1.0,
    "orange": 1.3,
    "red": 1.1,
    "yellow": 1.0,
    "green": 0.9,
    "dark_blue": 0.8,
}


class Trader(Strategy):
    """Aggressive buyer who also proposes trades to complete color groups.

    Trade evaluation is based on strategic position weight.
    """

    def _property_value(self, square: BuyableSquare, game_state: GameState) -> float:
        """Compute a strategic value score for a property.

        Args:
            square: The buyable square to value.
            game_state: Current game state (unused but kept for API).

        Returns:
            Strategic value as a float.
        """
        if isinstance(square, ColorProperty):
            return square.price * _POSITION_WEIGHT.get(square.color, 1.0)
        return float(square.price)

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
        """Build on complete groups, highest-weight first."""
        orders: list[BuildOrder] = []
        complete = _complete_unmortgaged_colors(player, game_state)

        def color_weight(color: str) -> float:
            return _POSITION_WEIGHT.get(color, 1.0)

        for color in sorted(complete, key=color_weight, reverse=True):
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
        """Mortgage lowest-weight properties first."""
        mortgageable = []
        for pos, po in game_state.property_ownership.items():
            if po.owner is not player or po.is_mortgaged:
                continue
            if po.houses > 0 or po.has_hotel:
                continue
            sq = game_state.board.get_square(pos)
            if not isinstance(sq, BuyableSquare):
                continue
            mortgageable.append(pos)

        def sort_key(pos: int) -> float:
            sq = game_state.board.get_square(pos)
            if isinstance(sq, BuyableSquare):
                return self._property_value(sq, game_state)
            return 0.0

        return sorted(mortgageable, key=sort_key)

    def should_accept_trade(
        self,
        player: Player,
        trade_offer: TradeOffer,
        game_state: GameState,
    ) -> bool:
        """Accept if trade completes a group or has positive net value; reject if it completes an opponent's group."""
        if self._trade_completes_opponent_group(player, trade_offer, game_state):
            return False
        if self._trade_completes_own_group(player, trade_offer, game_state):
            return True
        return self._trade_has_positive_net_value(trade_offer, game_state)

    def _trade_completes_opponent_group(
        self,
        player: Player,
        trade_offer: TradeOffer,
        game_state: GameState,
    ) -> bool:
        """Return True if giving away offered_positions would complete a color group for any opponent."""
        for pos in trade_offer.offered_positions:
            sq = game_state.board.get_square(pos)
            if not isinstance(sq, ColorProperty):
                continue
            group_positions = {s.position for s in game_state.board.get_group(sq.color)}
            for other in game_state.players:
                if other is player or other.bankrupt:
                    continue
                opponent_owned = {
                    p
                    for p, po in game_state.property_ownership.items()
                    if po.owner is other
                }
                if group_positions <= (opponent_owned | {pos}):
                    return True
        return False

    def _trade_completes_own_group(
        self,
        player: Player,
        trade_offer: TradeOffer,
        game_state: GameState,
    ) -> bool:
        """Return True if receiving requested_positions would complete a color group for player."""
        current_owned = {
            p for p, po in game_state.property_ownership.items() if po.owner is player
        }
        after_trade = current_owned - set(trade_offer.offered_positions) | set(
            trade_offer.requested_positions
        )
        for pos in trade_offer.requested_positions:
            sq = game_state.board.get_square(pos)
            if isinstance(sq, ColorProperty):
                group_positions = {
                    s.position for s in game_state.board.get_group(sq.color)
                }
                if group_positions <= after_trade:
                    return True
        return False

    def _trade_has_positive_net_value(
        self,
        trade_offer: TradeOffer,
        game_state: GameState,
    ) -> bool:
        """Return True if the net strategic value of the trade is positive for the player."""
        received_values = [
            self._property_value(
                game_state.board.get_square(pos),  # type: ignore[arg-type]
                game_state,
            )
            for pos in trade_offer.requested_positions
            if isinstance(game_state.board.get_square(pos), BuyableSquare)
        ]
        received_value: float = sum(received_values)
        given_values = [
            self._property_value(
                game_state.board.get_square(pos),  # type: ignore[arg-type]
                game_state,
            )
            for pos in trade_offer.offered_positions
            if isinstance(game_state.board.get_square(pos), BuyableSquare)
        ]
        given_value: float = sum(given_values)
        net_cash = trade_offer.cash_offered - trade_offer.cash_requested
        return (received_value + net_cash) > given_value

    def propose_trade(
        self,
        player: Player,
        game_state: GameState,
    ) -> TradeOffer | None:
        """Find a trade that would complete a color group for this player.

        Never proposes a trade that would give the opponent a complete color group.
        Ensures offered compensation meets the 80% valuation threshold.
        """
        color_owned = self._colors_partially_owned(player, game_state)

        for color, owned_positions in color_owned.items():
            group = game_state.board.get_group(color)
            missing = [
                sq.position for sq in group if sq.position not in owned_positions
            ]
            if not missing:
                continue

            for pos in missing:
                po = game_state.property_ownership.get(pos)
                if not (po and po.owner is not None and po.owner is not player):
                    continue
                opponent = po.owner
                offer = self._build_offer(player, pos, color, opponent, game_state)
                if offer is not None:
                    return offer

        return None

    def _colors_partially_owned(
        self, player: Player, game_state: GameState
    ) -> dict[str, list[int]]:
        """Return mapping of color → owned positions for colors the player partially owns."""
        result: dict[str, list[int]] = {}
        for pos, po in game_state.property_ownership.items():
            if po.owner is not player:
                continue
            sq = game_state.board.get_square(pos)
            if isinstance(sq, ColorProperty):
                result.setdefault(sq.color, []).append(pos)
        return result

    def _build_offer(
        self,
        player: Player,
        target_pos: int,
        target_color: str,
        opponent: Player,
        game_state: GameState,
    ) -> TradeOffer | None:
        """Build a TradeOffer for target_pos, or None if no safe bait exists."""
        target_sq = game_state.board.get_square(target_pos)
        if not isinstance(target_sq, BuyableSquare):
            return None
        target_value = self._property_value(target_sq, game_state)
        threshold = target_value * 0.8

        bait = self._find_trade_bait(player, game_state, target_color, opponent)
        if bait is None:
            return None

        bait_sq = game_state.board.get_square(bait)
        if not isinstance(bait_sq, BuyableSquare):
            return None
        bait_value = self._property_value(bait_sq, game_state)

        cash_offered = (
            max(0, int(threshold - bait_value) + 1) if bait_value < threshold else 0
        )
        if cash_offered > player.cash:
            return None

        return TradeOffer(
            offered_positions=[bait],
            requested_positions=[target_pos],
            cash_offered=cash_offered,
            cash_requested=0,
        )

    def _find_trade_bait(
        self,
        player: Player,
        game_state: GameState,
        exclude_color: str,
        opponent: Player | None = None,
    ) -> int | None:
        """Find a low-value property to offer that is safe to give away.

        Args:
            player: The proposing player.
            game_state: Current game state.
            exclude_color: Color group to exclude (the target color).
            opponent: The player receiving the bait; exclude positions that
                      would complete a color group for them.

        Returns:
            Board position to offer, or None.
        """
        candidates = []
        for pos, po in game_state.property_ownership.items():
            if po.owner is not player or po.is_mortgaged:
                continue
            if po.houses > 0 or po.has_hotel:
                continue
            sq = game_state.board.get_square(pos)
            if not isinstance(sq, BuyableSquare):
                continue
            if isinstance(sq, ColorProperty) and sq.color == exclude_color:
                continue
            if opponent and self._would_complete_group(pos, opponent, game_state):
                continue
            candidates.append(pos)

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda pos: self._property_value(
                game_state.board.get_square(pos),  # type: ignore[arg-type]
                game_state,
            ),
        )

    def _would_complete_group(
        self, pos: int, recipient: Player, game_state: GameState
    ) -> bool:
        """Return True if giving pos to recipient would complete a color group for them."""
        sq = game_state.board.get_square(pos)
        if not isinstance(sq, ColorProperty):
            return False
        group_positions = {s.position for s in game_state.board.get_group(sq.color)}
        recipient_owned = {
            p
            for p, po in game_state.property_ownership.items()
            if po.owner is recipient
        }
        return group_positions <= (recipient_owned | {pos})

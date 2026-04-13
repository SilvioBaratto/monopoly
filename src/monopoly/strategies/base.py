"""Abstract Strategy interface.

Responsibilities (SRP):
- Define the Strategy ABC — the contract every concrete strategy must fulfill

Value objects (JailDecision, BuildOrder, SellOrder, TradeOffer) live in
`strategies.types` and are re-exported here for backward compatibility.

No game logic — this module defines the contract only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from monopoly.strategies.types import (  # noqa: F401  (re-exported)
    BuildOrder,
    JailDecision,
    SellOrder,
    TradeOffer,
)

if TYPE_CHECKING:
    from monopoly.board import BuyableSquare
    from monopoly.state import GameState, Player


class Strategy(ABC):
    """Abstract base for all Monopoly player strategies.

    Each concrete strategy encapsulates a full decision policy:
    - When to buy properties
    - When and where to build houses/hotels
    - How to handle being in jail
    - Which properties to mortgage when cash is needed
    - Whether to accept or propose trades
    """

    @abstractmethod
    def should_buy_property(
        self,
        player: Player,
        square: BuyableSquare,
        game_state: GameState,
    ) -> bool:
        """Decide whether to purchase an unowned property.

        Args:
            player: The player considering the purchase.
            square: The unowned buyable square.
            game_state: Current game state.

        Returns:
            True if the player should buy the property.
        """
        ...

    @abstractmethod
    def choose_properties_to_build(
        self,
        player: Player,
        game_state: GameState,
    ) -> list[BuildOrder]:
        """Choose where to build houses or hotels.

        Args:
            player: The building player.
            game_state: Current game state.

        Returns:
            Ordered list of BuildOrder instructions to execute.
        """
        ...

    @abstractmethod
    def get_jail_decision(
        self,
        player: Player,
        game_state: GameState,
    ) -> JailDecision:
        """Decide what to do at the start of a jail turn.

        Args:
            player: The jailed player.
            game_state: Current game state.

        Returns:
            A JailDecision indicating how to attempt to leave jail.
        """
        ...

    @abstractmethod
    def choose_properties_to_mortgage(
        self,
        player: Player,
        amount_needed: int,
        game_state: GameState,
    ) -> list[int]:
        """Choose which properties to mortgage to raise cash.

        Args:
            player: The cash-strapped player.
            amount_needed: Minimum amount required.
            game_state: Current game state.

        Returns:
            List of board positions to mortgage (in order).
        """
        ...

    @abstractmethod
    def should_accept_trade(
        self,
        player: Player,
        trade_offer: TradeOffer,
        game_state: GameState,
    ) -> bool:
        """Decide whether to accept a trade offer.

        Args:
            player: The player receiving the offer.
            trade_offer: The proposed trade.
            game_state: Current game state.

        Returns:
            True if the player accepts the trade.
        """
        ...

    @abstractmethod
    def propose_trade(
        self,
        player: Player,
        game_state: GameState,
    ) -> TradeOffer | None:
        """Optionally propose a trade to another player.

        Args:
            player: The proposing player.
            game_state: Current game state.

        Returns:
            A TradeOffer, or None if no trade is proposed.
        """
        ...

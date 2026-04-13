"""Game state models for Monopoly engine.

Responsibilities (SRP):
- Define mutable game state data structures (Player, PropertyOwnership, GameState)
- Provide factory method GameState.init_game()
- Expose computed properties (active_players, current_player)

No game logic — this module manages state structure only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from monopoly.board import Board
from monopoly.cards import Card, Deck

if TYPE_CHECKING:
    pass


@dataclass
class Player:
    """Mutable player state for one Monopoly participant.

    Args:
        name: Display name.
        cash: Current cash balance (starts at $1500).
        position: Current board position (0–39).
        in_jail: Whether the player is currently in jail.
        jail_turns: Number of failed jail-escape attempts (0–2).
        consecutive_doubles: Consecutive doubles rolled this turn sequence.
        goojf_cards: Get Out of Jail Free cards held.
        bankrupt: Whether the player has gone bankrupt.
    """

    name: str
    cash: int = 1500
    position: int = 0
    in_jail: bool = False
    jail_turns: int = 0
    consecutive_doubles: int = 0
    goojf_cards: list[Card] = field(default_factory=list)
    bankrupt: bool = False


@dataclass
class PropertyOwnership:
    """Ownership and development state for one buyable square.

    Args:
        owner: Player who owns the property, or None if unowned.
        houses: Number of houses built (0–4).
        has_hotel: Whether a hotel has been built.
        is_mortgaged: Whether the property is currently mortgaged.
    """

    owner: Player | None = None
    houses: int = 0
    has_hotel: bool = False
    is_mortgaged: bool = False


@dataclass
class GameState:
    """Complete mutable state of a Monopoly game.

    Args:
        players: All players (including bankrupt ones).
        board: The immutable board.
        chance_deck: The Chance card deck.
        community_chest_deck: The Community Chest card deck.
        property_ownership: Maps board position → PropertyOwnership.
        houses_available: Houses remaining in the bank supply (max 32).
        hotels_available: Hotels remaining in the bank supply (max 12).
        current_player_index: Index into players list for whose turn it is.
        turn_count: Total number of full rounds completed.
    """

    players: list[Player]
    board: Board
    chance_deck: Deck
    community_chest_deck: Deck
    property_ownership: dict[int, PropertyOwnership]
    houses_available: int = 32
    hotels_available: int = 12
    current_player_index: int = 0
    turn_count: int = 0

    @classmethod
    def init_game(
        cls,
        player_names: list[str],
        board: Board,
        chance_deck: Deck,
        community_chest_deck: Deck,
    ) -> GameState:
        """Create a fresh game state from player names and board data.

        Args:
            player_names: Display names for each player.
            board: The loaded Board instance.
            chance_deck: Pre-shuffled Chance deck.
            community_chest_deck: Pre-shuffled Community Chest deck.

        Returns:
            A GameState with all players at Go ($1500 cash) and no properties owned.
        """
        players = [Player(name=n) for n in player_names]
        prop_ownership = {
            sq.position: PropertyOwnership() for sq in board.buyable_squares
        }
        return cls(
            players=players,
            board=board,
            chance_deck=chance_deck,
            community_chest_deck=community_chest_deck,
            property_ownership=prop_ownership,
        )

    @property
    def active_players(self) -> list[Player]:
        """Players who have not gone bankrupt."""
        return [p for p in self.players if not p.bankrupt]

    @property
    def current_player(self) -> Player:
        """The player whose turn it currently is."""
        return self.players[self.current_player_index]

"""Card deck module for Monopoly probability analysis.

Responsibilities (SRP):
- Define card data structures (Card, CardEffect)
- Manage deck state with FIFO cycling (Deck)
- Load decks from YAML (factory functions)

No effect execution — this module manages state only.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy
import yaml


class CardEffect(str, Enum):
    """All valid card effects in standard Monopoly."""

    move_absolute = "move_absolute"
    move_relative = "move_relative"
    move_to_nearest = "move_to_nearest"
    pay_bank = "pay_bank"
    receive_bank = "receive_bank"
    pay_each_player = "pay_each_player"
    receive_each_player = "receive_each_player"
    repairs = "repairs"
    go_to_jail = "go_to_jail"
    get_out_of_jail = "get_out_of_jail"


@dataclass(frozen=True)
class Card:
    """Immutable card with identity, display text, effect, and parameters.

    Args:
        id: Unique card identifier.
        text: Human-readable card description.
        effect: The card's game effect.
        params: Effect parameters (vary by effect type).
    """

    id: str
    text: str
    effect: CardEffect
    params: dict[str, Any] = field(default_factory=dict)


class Deck:
    """FIFO card deck with special handling for Get Out of Jail Free cards.

    Normal cards are reinserted at the bottom after drawing.
    GOOJF cards are held out until explicitly returned via return_card().
    """

    def __init__(self, cards: list[Card]) -> None:
        self._deque: collections.deque[Card] = collections.deque(cards)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def draw(self) -> Card:
        """Draw the top card.

        Normal cards cycle to the bottom automatically.
        GOOJF cards are NOT reinserted — caller must call return_card().

        Returns:
            The drawn Card.

        Raises:
            IndexError: If the deck is empty.
        """
        card = self._deque.popleft()
        if card.effect != CardEffect.get_out_of_jail:
            self._deque.append(card)
        return card

    def return_card(self, card: Card) -> None:
        """Return a GOOJF card to the bottom of the deck.

        Args:
            card: The GOOJF card being returned.
        """
        self._deque.append(card)

    @property
    def cards(self) -> list[Card]:
        """Current deck contents in draw order (top → bottom)."""
        return list(self._deque)

    def __len__(self) -> int:
        return len(self._deque)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        deck_name: str,
        rng: numpy.random.Generator,
    ) -> "Deck":
        """Load and shuffle a named deck from a YAML file.

        Args:
            path: Path to the YAML data file.
            deck_name: Key in the YAML (e.g. "chance" or "community_chest").
            rng: Seeded NumPy random generator for reproducible shuffles.

        Returns:
            A new shuffled Deck instance.
        """
        raw: dict[str, list[dict[str, Any]]] = yaml.safe_load(path.read_text())
        cards = [
            Card(
                id=entry["id"],
                text=entry["text"],
                effect=CardEffect(entry["effect"]),
                params=entry.get("params") or {},
            )
            for entry in raw[deck_name]
        ]
        indices = rng.permutation(len(cards)).tolist()
        shuffled = [cards[i] for i in indices]
        return cls(shuffled)


def load_decks(
    data_path: Path,
    rng: numpy.random.Generator,
) -> tuple[Deck, Deck]:
    """Load both standard Monopoly decks from YAML.

    Args:
        data_path: Path to cards_standard.yaml.
        rng: Seeded NumPy random generator (shared, applied sequentially).

    Returns:
        Tuple of (chance_deck, community_chest_deck).
    """
    chance = Deck.from_yaml(data_path, "chance", rng)
    community_chest = Deck.from_yaml(data_path, "community_chest", rng)
    return chance, community_chest

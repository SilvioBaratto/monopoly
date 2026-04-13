"""Turn resolution for Monopoly.

Responsibilities (SRP):
- Roll dice and move the current player (including doubles re-rolls)
- Resolve the landing square (tax, card, rent, purchase)
- Return a TurnResult summarising what happened

No jail-turn logic (handled in jail.py) or building phase (handled in game.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from monopoly.bankruptcy import attempt_payment
from monopoly.board import BuyableSquare, TaxSquare
from monopoly.dice import DiceRoll, is_triple_doubles, roll
from monopoly.effects import execute_card_effect
from monopoly.jail import send_to_jail
from monopoly.purchase import attempt_purchase
from monopoly.rent import calculate_rent

if TYPE_CHECKING:
    import numpy

    from monopoly.state import GameState, Player
    from monopoly.strategies.base import Strategy

_BOARD_SIZE = 40
_GO_SALARY = 200
_CHANCE_POSITIONS = frozenset({7, 22, 36})
_COMMUNITY_CHEST_POSITIONS = frozenset({2, 17, 33})
_GO_TO_JAIL_POSITION = 30
_MAX_ROLLS_PER_TURN = 3


@dataclass
class TurnResult:
    """Summary of one complete turn's outcome.

    Args:
        rolls: All dice rolls taken this turn (1–3 entries).
        positions_visited: Board positions reached per roll (same length as rolls).
        passed_go: True when the player crossed position 0 (not landed on it).
        went_to_jail: True when the player was sent to jail this turn.
        unowned_landed: Positions of unowned buyable squares encountered.
        rent_paid: Total rent paid this turn (default 0).
    """

    rolls: list[DiceRoll]
    positions_visited: list[int]
    passed_go: bool
    went_to_jail: bool
    unowned_landed: list[int]
    rent_paid: int = 0


def _did_pass_go(old_position: int, dice_total: int) -> bool:
    """Return True when movement crosses position 0 without landing on it.

    Args:
        old_position: Player's position before moving.
        dice_total: Sum of the dice roll.

    Returns:
        True iff the move crosses Go but does not land exactly on it.
    """
    return old_position + dice_total > _BOARD_SIZE


def resolve_turn(
    player: Player,
    game_state: GameState,
    strategy: Strategy,
    rng: numpy.random.Generator,
) -> TurnResult:
    """Resolve a complete turn for the given player, including doubles re-rolls.

    Resets consecutive_doubles to 0, then loops up to 3 rolls. Triple doubles
    send the player to jail without movement on the third roll.

    Args:
        player: The active player.
        game_state: Current game state.
        strategy: The player's decision strategy.
        rng: Random number generator.

    Returns:
        TurnResult summarising the entire turn.
    """
    player.consecutive_doubles = 0

    rolls: list[DiceRoll] = []
    positions_visited: list[int] = []
    passed_go = False
    went_to_jail = False
    total_rent_paid = 0
    unowned_landed: list[int] = []

    for _ in range(_MAX_ROLLS_PER_TURN):
        dice = roll(rng)
        rolls.append(dice)

        if dice.is_doubles:
            player.consecutive_doubles += 1

        if is_triple_doubles(player.consecutive_doubles):
            player.consecutive_doubles = 0
            send_to_jail(player, game_state)
            went_to_jail = True
            break

        old_position = player.position
        new_position = (old_position + dice.total) % _BOARD_SIZE
        player.position = new_position
        positions_visited.append(new_position)

        if _did_pass_go(old_position, dice.total):
            passed_go = True
            player.cash += _GO_SALARY

        result = _resolve_square(player, game_state, strategy, rng, dice)
        total_rent_paid += result.get("rent_paid", 0)

        if result.get("went_to_jail", False):
            went_to_jail = True
            break

        if player.bankrupt:
            break

        if result.get("unowned_pos") is not None:
            unowned_landed.append(result["unowned_pos"])

        if not dice.is_doubles:
            player.consecutive_doubles = 0
            break

    return TurnResult(
        rolls=rolls,
        positions_visited=positions_visited,
        passed_go=passed_go,
        went_to_jail=went_to_jail,
        unowned_landed=unowned_landed,
        rent_paid=total_rent_paid,
    )


def _resolve_square(
    player: Player,
    game_state: GameState,
    strategy: Strategy,
    rng: numpy.random.Generator,  # noqa: ARG001
    dice: DiceRoll,
) -> dict:
    """Resolve the effect of landing on player.position.

    Args:
        player: The player who moved.
        game_state: Current game state.
        strategy: The player's decision strategy.
        rng: Random number generator (unused here, kept for API).
        dice: The dice roll used to move.

    Returns:
        Dict with 'rent_paid', 'went_to_jail', and optional 'unowned_pos' keys.
    """
    position = player.position

    if position == _GO_TO_JAIL_POSITION:
        send_to_jail(player, game_state)
        return {"rent_paid": 0, "went_to_jail": True}

    square = game_state.board.get_square(position)

    if isinstance(square, TaxSquare):
        attempt_payment(player, square.amount, None, game_state, strategy)
        return {"rent_paid": 0, "went_to_jail": False}

    if position in _CHANCE_POSITIONS:
        return _resolve_card(player, game_state, strategy, dice, game_state.chance_deck)

    if position in _COMMUNITY_CHEST_POSITIONS:
        return _resolve_card(
            player, game_state, strategy, dice, game_state.community_chest_deck
        )

    if isinstance(square, BuyableSquare):
        return _resolve_buyable(player, square, game_state, strategy, dice)

    return {"rent_paid": 0, "went_to_jail": False}


def _resolve_card(
    player: Player,
    game_state: GameState,
    strategy: Strategy,
    dice: DiceRoll,
    deck: object,
) -> dict:
    """Draw and execute a card from the given deck.

    Args:
        player: The active player.
        game_state: Current game state.
        strategy: The player's decision strategy.
        dice: Current dice roll.
        deck: The deck to draw from.

    Returns:
        Dict with 'rent_paid', 'went_to_jail', and optional 'unowned_pos' keys.
    """
    from monopoly.cards import Deck

    assert isinstance(deck, Deck)
    card = deck.draw()
    result = execute_card_effect(card, player, game_state)

    if result.go_to_jail:
        send_to_jail(player, game_state)
        return {"rent_paid": 0, "went_to_jail": True}

    if result.new_position is not None:
        player.position = result.new_position
        new_square = game_state.board.get_square(result.new_position)
        if isinstance(new_square, BuyableSquare):
            buyable_result = _resolve_buyable(
                player,
                new_square,
                game_state,
                strategy,
                dice,
                double_rent=result.double_rent,
                force_dice_multiplier=result.force_dice_multiplier,
            )
            return {
                "rent_paid": buyable_result["rent_paid"],
                "went_to_jail": False,
                "unowned_pos": buyable_result.get("unowned_pos"),
            }

    return {"rent_paid": 0, "went_to_jail": False}


def _resolve_buyable(
    player: Player,
    square: BuyableSquare,
    game_state: GameState,
    strategy: Strategy,
    dice: DiceRoll,
    double_rent: bool = False,
    force_dice_multiplier: int | None = None,
) -> dict:
    """Resolve landing on a buyable square (buy or pay rent).

    Args:
        player: The active player.
        square: The buyable square.
        game_state: Current game state.
        strategy: The player's decision strategy.
        dice: Current dice roll.
        double_rent: Whether rent should be doubled.
        force_dice_multiplier: Utility dice multiplier override.

    Returns:
        Dict with 'rent_paid', 'went_to_jail', and optional 'unowned_pos' keys.
    """
    ownership = game_state.property_ownership.get(square.position)
    if ownership is None:
        return {"rent_paid": 0, "went_to_jail": False}

    if ownership.owner is None:
        attempt_purchase(player, square, game_state, strategy)
        return {"rent_paid": 0, "went_to_jail": False, "unowned_pos": square.position}

    if ownership.owner is player or ownership.is_mortgaged:
        return {"rent_paid": 0, "went_to_jail": False}

    owner = ownership.owner
    rent = calculate_rent(
        player,
        square.position,
        game_state,
        dice_total=dice.total,
        double_rent=double_rent,
        force_dice_multiplier=force_dice_multiplier,
    )

    if rent > 0:
        attempt_payment(player, rent, owner, game_state, strategy)

    return {"rent_paid": rent, "went_to_jail": False}

"""Jail mechanics for Monopoly.

Responsibilities (SRP):
- Send a player to jail (position 10, in_jail=True)
- Resolve one jail turn using a strategy decision
- Return a JailResult describing what happened

No rent or turn logic — this module handles jail state only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from monopoly.dice import DiceRoll, roll
from monopoly.strategies.base import JailDecision

if TYPE_CHECKING:
    import numpy

    from monopoly.state import GameState, Player
    from monopoly.strategies.base import Strategy

_JAIL_FINE = 50
_JAIL_POSITION = 10
_MAX_JAIL_TURNS = 3


@dataclass
class JailResult:
    """Result of resolving one jail turn.

    Args:
        left_jail: Whether the player has left jail after this turn.
        dice_roll: The dice roll used if doubles were attempted, else None.
        paid_fine: Whether the player paid the $50 fine.
    """

    left_jail: bool
    dice_roll: DiceRoll | None
    paid_fine: bool


def send_to_jail(player: Player, game_state: GameState) -> None:  # noqa: ARG001
    """Move a player to jail immediately.

    Sets position to 10, in_jail=True, resets consecutive_doubles.

    Args:
        player: The player to send to jail.
        game_state: Current game state (unused but kept for API consistency).
    """
    player.position = _JAIL_POSITION
    player.in_jail = True
    player.jail_turns = 0
    player.consecutive_doubles = 0


def resolve_jail_turn(
    player: Player,
    game_state: GameState,
    strategy: Strategy,
    rng: numpy.random.Generator,
) -> JailResult:
    """Handle one turn for a jailed player.

    Decision priority:
    1. If on 3rd attempt (jail_turns >= 2), must pay $50 and roll.
    2. Otherwise ask strategy for JailDecision.

    Args:
        player: The jailed player.
        game_state: Current game state.
        strategy: The player's strategy.
        rng: Random number generator for dice rolls.

    Returns:
        JailResult describing what happened.
    """
    # 3rd attempt: forced to pay fine and move regardless of decision
    if player.jail_turns >= 2:
        return _pay_fine_and_leave(player, rng)

    decision = strategy.get_jail_decision(player, game_state)

    if decision == JailDecision.PAY_FINE:
        return _pay_fine_and_leave(player, rng)

    if decision == JailDecision.USE_GOOJF and player.goojf_cards:
        return _use_goojf_card(player, game_state)

    # ROLL_DOUBLES (or USE_GOOJF fallback with no cards)
    return _attempt_roll_doubles(player, rng)


def _pay_fine_and_leave(
    player: Player,
    rng: numpy.random.Generator,
) -> JailResult:
    """Pay the $50 fine, leave jail, and prepare for movement.

    Args:
        player: The jailed player.
        rng: Random number generator (unused here but kept for signature parity).

    Returns:
        JailResult with left_jail=True, paid_fine=True, no dice roll.
    """
    player.cash -= _JAIL_FINE
    player.in_jail = False
    player.jail_turns = 0
    return JailResult(left_jail=True, dice_roll=None, paid_fine=True)


def _use_goojf_card(player: Player, game_state: GameState) -> JailResult:
    """Use a Get Out of Jail Free card to leave jail.

    Args:
        player: The jailed player.
        game_state: Current game state (used to return card to deck).

    Returns:
        JailResult with left_jail=True, paid_fine=False, no dice roll.
    """
    card = player.goojf_cards.pop(0)
    # Return to appropriate deck based on card id prefix
    if card.id.startswith("ch_"):
        game_state.chance_deck.return_card(card)
    else:
        game_state.community_chest_deck.return_card(card)

    player.in_jail = False
    player.jail_turns = 0
    return JailResult(left_jail=True, dice_roll=None, paid_fine=False)


def _attempt_roll_doubles(
    player: Player,
    rng: numpy.random.Generator,
) -> JailResult:
    """Roll dice attempting to roll doubles to leave jail.

    Args:
        player: The jailed player.
        rng: Random number generator.

    Returns:
        JailResult indicating success (doubles) or failure.
    """
    dice = roll(rng)

    if dice.is_doubles:
        player.in_jail = False
        player.jail_turns = 0
        return JailResult(left_jail=True, dice_roll=dice, paid_fine=False)

    # Failed attempt
    player.jail_turns += 1

    # After 3 failed attempts, must pay fine (jail_turns now == 3)
    if player.jail_turns >= _MAX_JAIL_TURNS:
        player.cash -= _JAIL_FINE
        player.in_jail = False
        player.jail_turns = 0
        return JailResult(left_jail=True, dice_roll=dice, paid_fine=True)

    return JailResult(left_jail=False, dice_roll=dice, paid_fine=False)

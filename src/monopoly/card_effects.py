"""Card effect executor for Monopoly.

Responsibilities (SRP):
- Execute card effects, mutating game state directly
- Return a CardResult describing movement and optional LandingModifier

No rent charging or turn flow — this module applies card effects only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from monopoly.cards import Card, CardEffect

if TYPE_CHECKING:
    from monopoly.state import GameState, Player

# Board constants
_BOARD_SIZE = 40
_GO_SALARY = 200
_JAIL_POSITION = 10
_RAILROAD_POSITIONS = (5, 15, 25, 35)
_UTILITY_POSITIONS = (12, 28)


@dataclass
class LandingModifier:
    """Rent adjustment hint for the caller after move_to_nearest.

    Args:
        double_rent: True if rent should be doubled on the landed square.
        dice_multiplier: Override dice multiplier for utility rent (e.g. 10).
    """

    double_rent: bool = False
    dice_multiplier: int | None = None


@dataclass
class CardResult:
    """Outcome of executing a card effect (movement + landing modifier only).

    Cash mutations are applied directly to player/opponents as side effects.

    Args:
        moved: True if the player's position was changed.
        new_position: The new board position (0–39), or None if no movement.
        landing_modifier: Optional rent modifier for move_to_nearest effects.
    """

    moved: bool = False
    new_position: int | None = None
    landing_modifier: LandingModifier | None = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def execute_card_effect(
    card: Card,
    player: Player,
    game_state: GameState,
    dice_roll: int | None = None,  # reserved for future use (e.g. utility rent)
) -> CardResult:
    """Apply a card's effect to the game state and return a CardResult.

    Directly mutates player and game_state (cash, position, jail flags, etc.).
    Returns movement info and optional LandingModifier for the caller.

    Args:
        card: The drawn card.
        player: The player who drew the card.
        game_state: Current full game state.
        dice_roll: Last dice roll total (reserved; not used by current effects).

    Returns:
        CardResult with movement and landing modifier info.

    Raises:
        ValueError: If the card's effect is not a recognised CardEffect value.
    """
    handlers = {
        CardEffect.move_absolute: _handle_move_absolute,
        CardEffect.move_relative: _handle_move_relative,
        CardEffect.move_to_nearest: _handle_move_to_nearest,
        CardEffect.pay_bank: _handle_pay_bank,
        CardEffect.receive_bank: _handle_receive_bank,
        CardEffect.pay_each_player: _handle_pay_each_player,
        CardEffect.receive_each_player: _handle_receive_each_player,
        CardEffect.repairs: _handle_repairs,
        CardEffect.go_to_jail: _handle_go_to_jail,
        CardEffect.get_out_of_jail: _handle_get_out_of_jail,
    }

    handler = handlers.get(card.effect)
    if handler is None:
        raise ValueError(f"Unknown card effect: {card.effect!r}")

    return handler(card, player, game_state)


# ---------------------------------------------------------------------------
# Private handlers — each mutates state and returns CardResult
# ---------------------------------------------------------------------------


def _handle_move_absolute(
    card: Card,
    player: Player,
    game_state: GameState,  # noqa: ARG001
) -> CardResult:
    target = card.params["target"]
    if _crosses_go(player.position, target) and "pass_go_collects" in card.params:
        player.cash += card.params["pass_go_collects"]
    player.position = target
    return CardResult(moved=True, new_position=target)


def _handle_move_relative(
    card: Card,
    player: Player,
    game_state: GameState,  # noqa: ARG001
) -> CardResult:
    new_position = (player.position + card.params["steps"]) % _BOARD_SIZE
    # Relative moves never collect Go salary even when wrapping past position 0
    player.position = new_position
    return CardResult(moved=True, new_position=new_position)


def _handle_move_to_nearest(
    card: Card,
    player: Player,
    game_state: GameState,  # noqa: ARG001
) -> CardResult:
    target_type = card.params["target"]
    positions = _RAILROAD_POSITIONS if target_type == "railroad" else _UTILITY_POSITIONS
    nearest = _nearest_forward_position(player.position, positions)

    if nearest < player.position:  # wrapped around — passed Go
        player.cash += _GO_SALARY

    player.position = nearest
    modifier = _build_landing_modifier(card.params)
    return CardResult(moved=True, new_position=nearest, landing_modifier=modifier)


def _handle_pay_bank(
    card: Card,
    player: Player,
    game_state: GameState,  # noqa: ARG001
) -> CardResult:
    player.cash -= card.params["amount"]
    return CardResult()


def _handle_receive_bank(
    card: Card,
    player: Player,
    game_state: GameState,  # noqa: ARG001
) -> CardResult:
    player.cash += card.params["amount"]
    return CardResult()


def _handle_pay_each_player(
    card: Card,
    player: Player,
    game_state: GameState,
) -> CardResult:
    amount = card.params["amount"]
    active_others = [p for p in game_state.active_players if p is not player]
    player.cash -= amount * len(active_others)
    for other in active_others:
        other.cash += amount
    return CardResult()


def _handle_receive_each_player(
    card: Card,
    player: Player,
    game_state: GameState,
) -> CardResult:
    amount = card.params["amount"]
    active_others = [p for p in game_state.active_players if p is not player]
    for other in active_others:
        other.cash -= amount
    player.cash += amount * len(active_others)
    return CardResult()


def _handle_repairs(
    card: Card,
    player: Player,
    game_state: GameState,
) -> CardResult:
    house_cost = card.params["house_cost"]
    hotel_cost = card.params["hotel_cost"]
    total_houses = sum(
        po.houses
        for po in game_state.property_ownership.values()
        if po.owner is player and not po.has_hotel
    )
    total_hotels = sum(
        1
        for po in game_state.property_ownership.values()
        if po.owner is player and po.has_hotel
    )
    player.cash -= total_houses * house_cost + total_hotels * hotel_cost
    return CardResult()


def _handle_go_to_jail(
    card: Card,  # noqa: ARG001
    player: Player,
    game_state: GameState,  # noqa: ARG001
) -> CardResult:
    player.position = _JAIL_POSITION
    player.in_jail = True
    player.consecutive_doubles = 0
    return CardResult(moved=True, new_position=_JAIL_POSITION)


def _handle_get_out_of_jail(
    card: Card,
    player: Player,
    game_state: GameState,  # noqa: ARG001
) -> CardResult:
    player.goojf_cards.append(card)
    return CardResult()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _crosses_go(old_position: int, new_position: int) -> bool:
    """Return True when moving from old_position to new_position crosses Go.

    Crosses Go when the target is strictly behind (or equal to) old_position,
    meaning the player wrapped around the board, OR when target is Go (0).
    """
    return new_position <= old_position and not (old_position == new_position)


def _nearest_forward_position(current: int, positions: tuple[int, ...]) -> int:
    """Return the nearest position clockwise (forward) from current.

    Args:
        current: Current board position (0–39).
        positions: Candidate positions to advance to.

    Returns:
        Nearest position strictly ahead, wrapping if needed.
    """
    for pos in sorted(positions):
        if pos > current:
            return pos
    return sorted(positions)[0]  # wrap around


def _build_landing_modifier(params: dict) -> LandingModifier | None:
    """Build a LandingModifier from card params, or None if no rent hints."""
    double_rent = bool(params.get("double_rent", False))
    dice_multiplier = params.get("dice_multiplier")
    if not double_rent and dice_multiplier is None:
        return None
    return LandingModifier(double_rent=double_rent, dice_multiplier=dice_multiplier)

"""Card effect executor for Monopoly.

Responsibilities (SRP):
- Execute card effects and return an EffectResult
- Handle all CardEffect variants: movement, cash, repairs, jail, GOOJF
- Compute Go crossing for absolute/nearest movements

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
_RAILROAD_POSITIONS = (5, 15, 25, 35)
_UTILITY_POSITIONS = (12, 28)


@dataclass
class EffectResult:
    """Outcome of executing a card effect.

    Args:
        new_position: New board position if the player moved, else None.
        passed_go: True if the player passed or landed on Go.
        cash_delta: Net cash change for the current player (positive = gain).
        double_rent: True if rent should be doubled on the new square.
        force_dice_multiplier: Override dice multiplier for utility rent.
        go_to_jail: True if the player should be sent to jail.
        goojf_card: The GOOJF card received, if any.
    """

    new_position: int | None = None
    passed_go: bool = False
    cash_delta: int = 0
    double_rent: bool = False
    force_dice_multiplier: int | None = None
    go_to_jail: bool = False
    goojf_card: Card | None = None


def execute_card_effect(
    card: Card,
    player: Player,
    game_state: GameState,
) -> EffectResult:
    """Apply a card's effect to the game state and return the result.

    Cash changes are applied directly to player and affected players.
    Movement is indicated via EffectResult.new_position (caller must update
    player.position to avoid circular responsibility).

    Args:
        card: The drawn card.
        player: The player who drew the card.
        game_state: Current game state.

    Returns:
        EffectResult describing all state changes.
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
        return EffectResult()

    return handler(card, player, game_state)


# ---------------------------------------------------------------------------
# Private handlers
# ---------------------------------------------------------------------------


def _handle_move_absolute(
    card: Card,
    player: Player,
    game_state: GameState,  # noqa: ARG001
) -> EffectResult:
    target = card.params["target"]
    passed_go = False
    cash_delta = 0

    if "pass_go_collects" in card.params and target != player.position:
        # Passes Go if target is behind current position (wrapping around)
        if target <= player.position and target != 0:
            passed_go = True
            cash_delta += card.params["pass_go_collects"]
        elif target == 0:
            # Moving to Go itself always collects
            passed_go = True
            cash_delta += card.params.get("pass_go_collects", _GO_SALARY)

    if passed_go:
        player.cash += cash_delta

    return EffectResult(
        new_position=target,
        passed_go=passed_go,
        cash_delta=cash_delta,
    )


def _handle_move_relative(
    card: Card,
    player: Player,
    game_state: GameState,  # noqa: ARG001
) -> EffectResult:
    steps = card.params["steps"]
    new_position = (player.position + steps) % _BOARD_SIZE
    # Relative moves never collect Go salary
    return EffectResult(new_position=new_position, passed_go=False, cash_delta=0)


def _handle_move_to_nearest(
    card: Card,
    player: Player,
    game_state: GameState,  # noqa: ARG001
) -> EffectResult:
    target_type = card.params["target"]
    double_rent = bool(card.params.get("double_rent", False))
    force_dice_multiplier: int | None = card.params.get("dice_multiplier")

    positions: tuple[int, ...]
    if target_type == "railroad":
        positions = _RAILROAD_POSITIONS
    else:
        positions = _UTILITY_POSITIONS

    nearest = _nearest_position(player.position, positions)
    passed_go = nearest < player.position
    cash_delta = 0

    if passed_go:
        cash_delta = _GO_SALARY
        player.cash += cash_delta

    return EffectResult(
        new_position=nearest,
        passed_go=passed_go,
        cash_delta=cash_delta,
        double_rent=double_rent,
        force_dice_multiplier=force_dice_multiplier,
    )


def _handle_pay_bank(
    card: Card,
    player: Player,
    game_state: GameState,  # noqa: ARG001
) -> EffectResult:
    amount = card.params["amount"]
    player.cash -= amount
    return EffectResult(cash_delta=-amount)


def _handle_receive_bank(
    card: Card,
    player: Player,
    game_state: GameState,  # noqa: ARG001
) -> EffectResult:
    amount = card.params["amount"]
    player.cash += amount
    return EffectResult(cash_delta=amount)


def _handle_pay_each_player(
    card: Card,
    player: Player,
    game_state: GameState,
) -> EffectResult:
    amount = card.params["amount"]
    active_others = [p for p in game_state.active_players if p is not player]
    total_paid = amount * len(active_others)
    player.cash -= total_paid
    for other in active_others:
        other.cash += amount
    return EffectResult(cash_delta=-total_paid)


def _handle_receive_each_player(
    card: Card,
    player: Player,
    game_state: GameState,
) -> EffectResult:
    amount = card.params["amount"]
    active_others = [p for p in game_state.active_players if p is not player]
    total_received = amount * len(active_others)
    for other in active_others:
        other.cash -= amount
    player.cash += total_received
    return EffectResult(cash_delta=total_received)


def _handle_repairs(
    card: Card,
    player: Player,
    game_state: GameState,
) -> EffectResult:
    house_cost = card.params["house_cost"]
    hotel_cost = card.params["hotel_cost"]

    total_houses = sum(
        po.houses
        for pos, po in game_state.property_ownership.items()
        if po.owner is player and not po.has_hotel
    )
    total_hotels = sum(
        1
        for pos, po in game_state.property_ownership.items()
        if po.owner is player and po.has_hotel
    )

    total_cost = total_houses * house_cost + total_hotels * hotel_cost
    player.cash -= total_cost
    return EffectResult(cash_delta=-total_cost)


def _handle_go_to_jail(
    card: Card,  # noqa: ARG001
    player: Player,  # noqa: ARG001
    game_state: GameState,  # noqa: ARG001
) -> EffectResult:
    return EffectResult(go_to_jail=True)


def _handle_get_out_of_jail(
    card: Card,
    player: Player,
    game_state: GameState,  # noqa: ARG001
) -> EffectResult:
    player.goojf_cards.append(card)
    return EffectResult(goojf_card=card)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nearest_position(current: int, positions: tuple[int, ...]) -> int:
    """Find the nearest board position ahead of current (wrapping around).

    Args:
        current: Current board position.
        positions: Tuple of candidate positions to advance to.

    Returns:
        The nearest position clockwise from current.
    """
    for pos in sorted(positions):
        if pos > current:
            return pos
    # Wrap around: take the first position
    return sorted(positions)[0]

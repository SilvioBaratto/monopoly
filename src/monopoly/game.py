"""Full Monopoly game loop.

Responsibilities (SRP):
- Orchestrate turn sequence for all active players
- Manage doubles re-rolls and jail resolution
- Detect game-over condition (≤1 active player or max turns)
- Return a GameResult summary

No rule-specific logic — delegates to turn.py, jail.py, buildings.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy

from monopoly.board import Board
from monopoly.buildings import execute_build_orders
from monopoly.cards import load_decks
from monopoly.jail import resolve_jail_turn
from monopoly.state import GameState
from monopoly.turn import resolve_turn

from .strategies.base import Strategy


@dataclass
class PlayerStats:
    """End-of-game statistics for one player.

    Args:
        final_cash: Cash held at game end.
        properties_owned: Number of properties owned at game end.
        bankruptcy_turn: Turn on which the player went bankrupt, or None.
    """

    final_cash: int
    properties_owned: int
    bankruptcy_turn: int | None


@dataclass
class GameResult:
    """Summary of a completed game.

    Args:
        winner: The surviving player, or None if max_turns reached.
        turns_played: Total number of full rounds completed.
        player_stats: Per-player stats keyed by player name.
    """

    winner: object  # Player | None
    turns_played: int
    player_stats: dict[str, PlayerStats]


class Game:
    """Orchestrates a full Monopoly game from start to finish.

    Args:
        player_names: Display names (2–6 players required).
        strategies: One Strategy per player (same order as player_names).
        board: Pre-loaded Board instance.
        data_path: Path to cards_standard.yaml.
        rng: Seeded NumPy random generator for full reproducibility.

    Raises:
        ValueError: If number of players is not between 2 and 6.
        ValueError: If strategies list length does not match player_names.
    """

    def __init__(
        self,
        player_names: list[str],
        strategies: list[Strategy],
        board: Board,
        data_path: Path,
        rng: numpy.random.Generator,
    ) -> None:
        if not (2 <= len(player_names) <= 6):
            raise ValueError(f"Monopoly requires 2–6 players, got {len(player_names)}")
        if len(strategies) != len(player_names):
            raise ValueError(
                "strategies list must have the same length as player_names"
            )

        self._rng = rng
        self._strategies: dict[str, Strategy] = {
            name: strat for name, strat in zip(player_names, strategies)
        }
        chance_deck, cc_deck = load_decks(data_path, rng)
        self.state = GameState.init_game(
            player_names=player_names,
            board=board,
            chance_deck=chance_deck,
            community_chest_deck=cc_deck,
        )

    def play(self, max_turns: int = 1000) -> GameResult:
        """Run the game until one player remains or max_turns is reached.

        Args:
            max_turns: Maximum full rounds before declaring no winner.

        Returns:
            GameResult with winner, turns played, and per-player stats.
        """
        bankruptcy_turns: dict[str, int] = {}

        while self.state.turn_count < max_turns:
            if len(self.state.active_players) <= 1:
                break

            self._play_full_round(bankruptcy_turns)
            self.state.turn_count += 1

        return self._build_result(bankruptcy_turns)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _play_full_round(self, bankruptcy_turns: dict[str, int]) -> None:
        """Play one full round — every active player takes their turn.

        Args:
            bankruptcy_turns: Dict to record bankruptcy turn numbers.
        """
        # Snapshot active players at round start to avoid skipping newly
        # bankrupt players mid-round
        players_this_round = list(self.state.active_players)

        for player in players_this_round:
            if player.bankrupt:
                continue
            if len(self.state.active_players) <= 1:
                break

            strategy = self._strategies[player.name]
            self._play_player_turn(player, strategy, bankruptcy_turns)

    def _play_player_turn(
        self,
        player: object,
        strategy: Strategy,
        bankruptcy_turns: dict[str, int],
    ) -> None:
        """Handle one player's complete turn, including doubles re-rolls.

        Args:
            player: The active Player.
            strategy: The player's strategy.
            bankruptcy_turns: Dict to record when bankruptcies occur.
        """
        from monopoly.state import Player as PlayerType

        assert isinstance(player, PlayerType)

        if player.in_jail:
            jail_result = resolve_jail_turn(player, self.state, strategy, self._rng)
            if not jail_result.left_jail:
                # Still in jail, turn ends
                pass
            elif jail_result.dice_roll is not None and jail_result.dice_roll.is_doubles:
                self._move_from_jail_roll(player, jail_result.dice_roll, strategy)
            else:
                # Paid fine or used GOOJF — resolve full turn (handles doubles loop)
                resolve_turn(player, self.state, strategy, self._rng)
        else:
            resolve_turn(player, self.state, strategy, self._rng)
            if player.bankrupt:
                bankruptcy_turns[player.name] = self.state.turn_count

        # Building phase after all rolls
        if not player.bankrupt:
            self._building_phase(player, strategy)

        if player.bankrupt and player.name not in bankruptcy_turns:
            bankruptcy_turns[player.name] = self.state.turn_count

    def _move_from_jail_roll(
        self,
        player: object,
        dice: object,
        strategy: Strategy,
    ) -> None:
        """Move and resolve square after leaving jail via doubles roll.

        Does NOT grant an extra roll (leaving jail via doubles is special).

        Args:
            player: The player who just left jail.
            dice: The DiceRoll used.
            strategy: The player's strategy.
        """
        from monopoly.dice import DiceRoll
        from monopoly.state import Player as PlayerType

        assert isinstance(player, PlayerType)
        assert isinstance(dice, DiceRoll)

        _BOARD_SIZE = 40
        _GO_SALARY = 200

        old_position = player.position
        new_position = (old_position + dice.total) % _BOARD_SIZE
        passed_go = (
            old_position + dice.total
        ) >= _BOARD_SIZE and new_position != old_position

        player.position = new_position
        if passed_go:
            player.cash += _GO_SALARY

        from monopoly.turn import _resolve_square

        _resolve_square(player, self.state, strategy, self._rng, dice)

    def _building_phase(self, player: object, strategy: Strategy) -> None:
        """Allow player to build houses/hotels after their turn.

        Args:
            player: The active player.
            strategy: The player's strategy.
        """
        from monopoly.state import Player as PlayerType

        assert isinstance(player, PlayerType)

        try:
            orders = strategy.choose_properties_to_build(player, self.state)
            if orders:
                execute_build_orders(player, self.state, orders)
        except Exception:
            pass  # Building phase errors should not crash the game

    def _build_result(self, bankruptcy_turns: dict[str, int]) -> GameResult:
        """Construct the GameResult from final game state.

        When exactly one player remains, they are the winner.
        When max_turns ends the game with multiple survivors, the richest
        active player is declared the winner.

        Args:
            bankruptcy_turns: Recorded bankruptcy turn numbers.

        Returns:
            Populated GameResult.
        """
        active = self.state.active_players
        if len(active) == 1:
            winner = active[0]
        elif active:
            winner = max(active, key=lambda p: p.cash)
        else:
            winner = None

        stats: dict[str, PlayerStats] = {}
        for player in self.state.players:
            owned = sum(
                1 for po in self.state.property_ownership.values() if po.owner is player
            )
            stats[player.name] = PlayerStats(
                final_cash=player.cash,
                properties_owned=owned,
                bankruptcy_turn=bankruptcy_turns.get(player.name),
            )

        return GameResult(
            winner=winner,
            turns_played=self.state.turn_count,
            player_stats=stats,
        )

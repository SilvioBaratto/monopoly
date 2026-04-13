"""Full Monopoly game loop.

Responsibilities (SRP):
- Orchestrate turn sequence for all active players
- Manage doubles re-rolls and jail resolution
- Detect game-over condition (≤1 active player or max turns)
- Return a GameResult summary

No rule-specific logic — delegates to turn.py, jail.py, buildings.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy

from monopoly.board import Board
from monopoly.buildings import execute_build_orders
from monopoly.cards import load_decks
from monopoly.jail import resolve_jail_turn
from monopoly.state import GameState, Player
from monopoly.turn import resolve_turn

from .strategies.base import Strategy


def _compute_net_worth(player: Player, state: GameState, board: Board) -> int:
    """Return total net worth for one player: cash + property values.

    Unmortgaged properties are valued at purchase price; mortgaged ones at
    mortgage value.  Properties are looked up via state.property_ownership,
    never via a player attribute.

    Args:
        player: The player whose net worth to compute.
        state: Current game state (owns property_ownership mapping).
        board: Board instance (owns square price/mortgage data).

    Returns:
        Total net worth in dollars.
    """
    property_value = sum(
        board.squares[pos].mortgage if own.is_mortgaged else board.squares[pos].price  # type: ignore[attr-defined]
        for pos, own in state.property_ownership.items()
        if own.owner is player
    )
    return player.cash + property_value


@dataclass
class GameHistory:
    """Per-turn snapshot history of a Monopoly game for visualization.

    Args:
        player_names: Ordered list of player names in the game.
        position_history: Per-turn mapping of player name → board position (0–39).
        net_worth_history: Per-turn mapping of player name → net worth in dollars.
        ownership_history: Per-turn mapping of board position → owner player name.
                           Only owned squares appear; unowned squares are absent.
    """

    player_names: list[str]
    position_history: list[dict[str, int]]
    net_worth_history: list[dict[str, int]]
    ownership_history: list[dict[int, str]]


@dataclass
class PlayerStats:
    """End-of-game statistics for one player.

    Args:
        final_cash: Cash held at game end.
        properties_owned: Number of properties owned at game end.
        bankruptcy_turn: Turn on which the player went bankrupt, or None.
        net_worth_history: Net worth snapshot per round (index 0 = start of
            game, index n = after round n). Defaults to [] for backward
            compatibility.
    """

    final_cash: int
    properties_owned: int
    bankruptcy_turn: int | None
    net_worth_history: list[int] = field(default_factory=list)


@dataclass
class GameResult:
    """Summary of a completed game.

    Args:
        winner: The surviving player, or None if max_turns reached.
        turns_played: Total number of full rounds completed.
        player_stats: Per-player stats keyed by player name.
    """

    winner: Player | None
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
        net_worth_snapshots: dict[str, list[int]] = {
            p.name: [] for p in self.state.players
        }

        self._record_net_worth_snapshot(net_worth_snapshots)

        while self.state.turn_count < max_turns:
            if len(self.state.active_players) <= 1:
                break

            self._play_full_round(bankruptcy_turns)
            self.state.turn_count += 1
            self._record_net_worth_snapshot(net_worth_snapshots)

        return self._build_result(bankruptcy_turns, net_worth_snapshots)

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

    def _record_net_worth_snapshot(self, snapshots: dict[str, list[int]]) -> None:
        """Append a net worth snapshot for every player to the history dict.

        Args:
            snapshots: Mutable dict mapping player name → history list.
        """
        board = self.state.board
        for player in self.state.players:
            snapshots[player.name].append(_compute_net_worth(player, self.state, board))

    def play_with_history(self, max_turns: int = 200) -> "GameHistory":
        """Run a game and return a per-turn GameHistory for animation.

        Records position, net worth, and property ownership after every round.

        Args:
            max_turns: Maximum full rounds before stopping.

        Returns:
            GameHistory with per-turn snapshots.
        """
        position_history: list[dict[str, int]] = []
        net_worth_history: list[dict[str, int]] = []
        ownership_history: list[dict[int, str]] = []
        bankruptcy_turns: dict[str, int] = {}
        net_worth_snapshots: dict[str, list[int]] = {
            p.name: [] for p in self.state.players
        }

        self._record_net_worth_snapshot(net_worth_snapshots)
        self._append_history_snapshots(
            position_history, net_worth_history, ownership_history
        )

        while self.state.turn_count < max_turns:
            if len(self.state.active_players) <= 1:
                break
            self._play_full_round(bankruptcy_turns)
            self.state.turn_count += 1
            self._record_net_worth_snapshot(net_worth_snapshots)
            self._append_history_snapshots(
                position_history, net_worth_history, ownership_history
            )

        return GameHistory(
            player_names=[p.name for p in self.state.players],
            position_history=position_history,
            net_worth_history=net_worth_history,
            ownership_history=ownership_history,
        )

    def _append_history_snapshots(
        self,
        position_history: list[dict[str, int]],
        net_worth_history: list[dict[str, int]],
        ownership_history: list[dict[int, str]],
    ) -> None:
        """Append current state snapshots to per-turn history lists."""
        board = self.state.board
        position_history.append({p.name: p.position for p in self.state.players})
        net_worth_history.append(
            {
                p.name: _compute_net_worth(p, self.state, board)
                for p in self.state.players
            }
        )
        ownership_history.append(
            {
                pos: own.owner.name
                for pos, own in self.state.property_ownership.items()
                if own.owner is not None
            }
        )

    def _build_result(
        self,
        bankruptcy_turns: dict[str, int],
        net_worth_snapshots: dict[str, list[int]] | None = None,
    ) -> GameResult:
        """Construct the GameResult from final game state.

        When exactly one player remains, they are the winner.
        When max_turns ends the game with multiple survivors, the richest
        active player is declared the winner.

        Args:
            bankruptcy_turns: Recorded bankruptcy turn numbers.
            net_worth_snapshots: Per-player net worth history (optional).

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

        history = net_worth_snapshots or {}
        stats: dict[str, PlayerStats] = {}
        for player in self.state.players:
            owned = sum(
                1 for po in self.state.property_ownership.values() if po.owner is player
            )
            stats[player.name] = PlayerStats(
                final_cash=player.cash,
                properties_owned=owned,
                bankruptcy_turn=bankruptcy_turns.get(player.name),
                net_worth_history=history.get(player.name, []),
            )

        return GameResult(
            winner=winner,
            turns_played=self.state.turn_count,
            player_stats=stats,
        )

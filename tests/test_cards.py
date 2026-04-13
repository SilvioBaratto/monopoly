"""Tests for card deck correctness — FIFO cycle, frequency uniformity, GOOJF behavior."""

from collections import Counter
from pathlib import Path

import numpy
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import chisquare

from monopoly.cards import CardEffect, Deck, load_decks

DATA_PATH = Path(__file__).parent.parent / "data" / "cards_standard.yaml"
DECK_NAMES = ["chance", "community_chest"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng() -> numpy.random.Generator:
    """Seeded RNG for deterministic test runs."""
    return numpy.random.default_rng(42)


@pytest.fixture
def chance_deck(rng: numpy.random.Generator) -> Deck:
    """Loaded and shuffled Chance deck."""
    chance, _ = load_decks(DATA_PATH, rng)
    return chance


@pytest.fixture
def cc_deck(rng: numpy.random.Generator) -> Deck:
    """Loaded and shuffled Community Chest deck."""
    _, cc = load_decks(DATA_PATH, rng)
    return cc


def _load_deck(deck_name: str, seed: int = 42) -> Deck:
    """Helper: load a named deck with a given seed."""
    rng = numpy.random.default_rng(seed)
    chance, cc = load_decks(DATA_PATH, rng)
    return chance if deck_name == "chance" else cc


# ---------------------------------------------------------------------------
# 1. Deck loads exactly 16 cards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deck_name", DECK_NAMES)
def test_deck_loads_exactly_16_cards(deck_name: str) -> None:
    """Each deck must contain exactly 16 cards after loading."""
    deck = _load_deck(deck_name)
    assert len(deck) == 16


# ---------------------------------------------------------------------------
# 2. All card effects are valid CardEffect enum members
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deck_name", DECK_NAMES)
def test_all_card_effects_are_valid(deck_name: str) -> None:
    """Every card's effect must be a member of CardEffect."""
    deck = _load_deck(deck_name)
    valid_effects = set(CardEffect)
    for card in deck.cards:
        assert card.effect in valid_effects, (
            f"Card '{card.id}' has unknown effect '{card.effect}'"
        )


# ---------------------------------------------------------------------------
# 3. Drawing 16 cards returns all 16 unique card IDs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deck_name", DECK_NAMES)
def test_draw_16_cards_returns_all_unique(deck_name: str) -> None:
    """Drawing all 16 cards from a fresh deck yields 16 unique card IDs."""
    deck = _load_deck(deck_name)
    drawn_ids: list[str] = []
    for _ in range(16):
        card = deck.draw()
        drawn_ids.append(card.id)
    assert len(set(drawn_ids)) == 16, "Expected 16 unique card IDs from one full pass"


# ---------------------------------------------------------------------------
# 4. FIFO cycle preserves order for non-GOOJF cards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deck_name", DECK_NAMES)
def test_fifo_cycle_preserves_order(deck_name: str) -> None:
    """After drawing all non-GOOJF cards, they reappear at the bottom in original order."""
    deck = _load_deck(deck_name)
    initial_order = [c.id for c in deck.cards if c.effect != CardEffect.get_out_of_jail]

    # Draw until GOOJF is hit (it won't cycle), collect non-GOOJF cards in draw order
    drawn_non_goojf: list[str] = []
    drawn_total = 0
    while drawn_total < 16:
        card = deck.draw()
        drawn_total += 1
        if card.effect != CardEffect.get_out_of_jail:
            drawn_non_goojf.append(card.id)

    # The order in which non-GOOJF cards were drawn must match their initial relative order
    assert drawn_non_goojf == initial_order


# ---------------------------------------------------------------------------
# 5. Chi-squared frequency uniformity over 10 000 draws
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deck_name", DECK_NAMES)
def test_chi_squared_frequency(deck_name: str) -> None:
    """Chi-squared test: each of the 16 cards should appear ~625 times in 10 000 draws."""
    deck = _load_deck(deck_name, seed=0)
    N = 10_000
    counts: Counter[str] = Counter()

    for _ in range(N):
        card = deck.draw()
        counts[card.id] += 1
        if card.effect == CardEffect.get_out_of_jail:
            deck.return_card(card)  # keep GOOJF cycling so all 16 are sampled

    observed = [counts[cid] for cid in sorted(counts)]
    expected_freq = N / 16
    _, p_value = chisquare(observed, f_exp=[expected_freq] * len(observed))
    assert p_value > 0.01, (
        f"Chi-squared p-value {p_value:.4f} is below 0.01 — distribution not uniform"
    )


# ---------------------------------------------------------------------------
# 6. GOOJF is NOT reinserted after draw — deck shrinks to 15
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deck_name", DECK_NAMES)
def test_goojf_not_reinserted_after_draw(deck_name: str) -> None:
    """After drawing GOOJF, the deck must contain exactly 15 cards."""
    deck = _load_deck(deck_name)
    goojf_drawn = False
    for _ in range(16):
        card = deck.draw()
        if card.effect == CardEffect.get_out_of_jail:
            goojf_drawn = True
            break
    assert goojf_drawn, "GOOJF card was not encountered in 16 draws"
    assert len(deck) == 15, f"Expected 15 cards after GOOJF drawn, got {len(deck)}"


# ---------------------------------------------------------------------------
# 7. return_card() reinserts GOOJF at the bottom — deck returns to 16
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deck_name", DECK_NAMES)
def test_return_card_reinserts_at_bottom(deck_name: str) -> None:
    """return_card(goojf) appends GOOJF to the bottom; deck size returns to 16."""
    deck = _load_deck(deck_name)
    goojf_card = None
    for _ in range(16):
        card = deck.draw()
        if card.effect == CardEffect.get_out_of_jail:
            goojf_card = card
            break

    assert goojf_card is not None
    deck.return_card(goojf_card)

    assert len(deck) == 16, f"Expected 16 cards after return_card, got {len(deck)}"
    assert deck.cards[-1].id == goojf_card.id, (
        "GOOJF must be at the bottom after return_card"
    )


# ---------------------------------------------------------------------------
# 8. Deck cycles correctly through the 15 remaining cards without GOOJF
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("deck_name", DECK_NAMES)
def test_deck_cycles_correctly_without_goojf(deck_name: str) -> None:
    """After GOOJF is removed, the remaining 15 cards cycle in consistent order."""
    deck = _load_deck(deck_name)

    # Draw until GOOJF is consumed (not reinserted)
    for _ in range(16):
        card = deck.draw()
        if card.effect == CardEffect.get_out_of_jail:
            break

    # Snapshot the 15-card cycle order
    cycle_one = [deck.draw().id for _ in range(15)]
    cycle_two = [deck.draw().id for _ in range(15)]

    assert cycle_one == cycle_two, "15-card cycle order must repeat exactly (FIFO)"


# ---------------------------------------------------------------------------
# 9. RNG seed reproducibility
# ---------------------------------------------------------------------------


def test_rng_seed_reproducibility() -> None:
    """Two decks loaded with the same seed must have identical card order."""
    chance_a, cc_a = load_decks(DATA_PATH, numpy.random.default_rng(99))
    chance_b, cc_b = load_decks(DATA_PATH, numpy.random.default_rng(99))

    assert [c.id for c in chance_a.cards] == [c.id for c in chance_b.cards]
    assert [c.id for c in cc_a.cards] == [c.id for c in cc_b.cards]


# ---------------------------------------------------------------------------
# 10. Property-based: any seed loads exactly 16 cards per deck
# ---------------------------------------------------------------------------


@given(seed=st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=50)
def test_any_seed_loads_16_cards(seed: int) -> None:
    """For any RNG seed, both decks must contain exactly 16 cards."""
    chance, cc = load_decks(DATA_PATH, numpy.random.default_rng(seed))
    assert len(chance) == 16
    assert len(cc) == 16

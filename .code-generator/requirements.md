# Monopoly: How to Win at Monopoly

Repo: https://github.com/SilvioBaratto/monopoly

## Description
Quantitative analysis and simulation project for the game of Monopoly,
designed as the foundation for an educational video ("How to Win at
Monopoly"). The goal is to scientifically answer two questions: **what is
the probability of winning a game of Monopoly** (as a function of the
number of players, the strategy adopted, and chance) and **what is the
optimal strategy** (which color groups to buy, when to build
houses/hotels, when to leave jail, when to accept trades). Target
audience: a creator/educator producing a YouTube video with charts, board
heatmaps, and visualizable simulations.

## Tech Stack
- **Language**: Python 3.12
- **Framework**: core library + Typer CLI (no web backend required)
- **Database**: none — simulation results serialized to local
  Parquet/SQLite
- **Deploy**: local export of PNG figures
- **Tests**: pytest + hypothesis (property-based tests on the game rules)
- **Numerics / Viz**: NumPy, SciPy, pandas, matplotlib, seaborn, networkx
  (Markov chain), numba (vectorized simulations), manim or
  matplotlib.animation for the video animations

## Features
1. **Board model** — 40 official squares (standard US "Atlantic City"
   edition; Italian variant with Vicolo Corto / Parco della Vittoria
   selectable), with rents, house/hotel costs, mortgages, color groups,
   railroads, and utilities. Data in versioned `data/board_standard.yaml`.
   Canonical values to model (sources: Wikibooks *Monopoly/Properties
   reference*, Falstad Monopoly Rents, Drexel CSP HW5):

   **Non-property squares (10):** Go (0), Jail / Just Visiting (10),
   Free Parking (20), Go To Jail (30), Chance ×3 (7, 22, 36),
   Community Chest ×3 (2, 17, 33), Income Tax (4, $200), Luxury Tax
   (38, $100).

   **Brown properties (house cost $50):**
   - 1  Mediterranean Avenue — price $60, mortgage $30, rent 2/10/30/90/160/250
   - 3  Baltic Avenue        — price $60, mortgage $30, rent 4/20/60/180/320/450

   **Light Blue (house cost $50):**
   - 6  Oriental Avenue      — $100, $50, 6/30/90/270/400/550
   - 8  Vermont Avenue       — $100, $50, 6/30/90/270/400/550
   - 9  Connecticut Avenue   — $120, $60, 8/40/100/300/450/600

   **Pink (house cost $100):**
   - 11 St. Charles Place    — $140, $70, 10/50/150/450/625/750
   - 13 States Avenue        — $140, $70, 10/50/150/450/625/750
   - 14 Virginia Avenue      — $160, $80, 12/60/180/500/700/900

   **Orange (house cost $100):**
   - 16 St. James Place      — $180, $90, 14/70/200/550/750/950
   - 18 Tennessee Avenue     — $180, $90, 14/70/200/550/750/950
   - 19 New York Avenue      — $200, $100, 16/80/220/600/800/1000

   **Red (house cost $150):**
   - 21 Kentucky Avenue      — $220, $110, 18/90/250/700/875/1050
   - 23 Indiana Avenue       — $220, $110, 18/90/250/700/875/1050
   - 24 Illinois Avenue      — $240, $120, 20/100/300/750/925/1100

   **Yellow (house cost $150):**
   - 26 Atlantic Avenue      — $260, $130, 22/110/330/800/975/1150
   - 27 Ventnor Avenue       — $260, $130, 22/110/330/800/975/1150
   - 29 Marvin Gardens       — $280, $140, 24/120/360/850/1025/1200

   **Green (house cost $200):**
   - 31 Pacific Avenue       — $300, $150, 26/130/390/900/1100/1275
   - 32 North Carolina Ave   — $300, $150, 26/130/390/900/1100/1275
   - 34 Pennsylvania Avenue  — $320, $160, 28/150/450/1000/1200/1400

   **Dark Blue (house cost $200):**
   - 37 Park Place           — $350, $175, 35/175/500/1100/1300/1500
   - 39 Boardwalk            — $400, $200, 50/200/600/1400/1700/2000

   **Railroads ($200 each, mortgage $100, no houses):**
   rent 25/50/100/200 for 1/2/3/4 railroads owned
   - 5  Reading Railroad
   - 15 Pennsylvania Railroad
   - 25 B&O Railroad
   - 35 Short Line

   **Utilities ($150 each, mortgage $75, no houses):**
   rent = 4× dice roll if 1 owned, 10× if both owned
   - 12 Electric Company
   - 28 Water Works

   **Invariant economic rules:** mortgage = price / 2; to unmortgage, pay
   mortgage + 10% interest; hotel = 4 houses + group house cost; on a
   fully owned, unmortgaged set with 0 houses the base rent is doubled;
   $200 salary for passing Go.

   **Italian variant (name mapping, identical numeric values):**
   Vicolo Corto/Vicolo Stretto (brown), Bastioni Gran Sasso/Viale Monte
   Rosa/Viale Vesuvio (light blue), Via Accademia/Corso Ateneo/Piazza
   Università (pink), Via Verdi/Corso Raffaello/Piazza Dante (orange),
   Via Marco Polo/Corso Magellano/Largo Colombo (red), Viale Costantino/
   Viale Traiano/Piazza Giulio Cesare (yellow), Via Roma/Corso Impero/
   Largo Augusto (green), Viale dei Giardini/Parco della Vittoria (dark
   blue). Railroads: Sud, Ovest, Nord, Est. Utilities: Società Elettrica,
   Società Acqua Potabile.
2. **Deterministic, tested game engine** — two dice (with doubles and
   the 3-doubles-to-jail rule), full Chance and Community Chest decks
   (see feature 3), Jail with up to 3 double-roll attempts, Luxury Tax,
   $200 salary on passing Go.
3. **Full Chance + Community Chest decks** — 16 cards per deck, loaded
   from `data/cards_standard.yaml` and managed as two FIFO queues with
   reinsertion at the bottom (except the two "Get Out of Jail Free"
   cards, which leave the deck until used). Each card is a typed object
   with `id`, `text` (original text), `effect` (enum: `move_absolute`,
   `move_relative`, `move_to_nearest`, `pay_bank`, `receive_bank`,
   `pay_each_player`, `receive_each_player`, `repairs`, `go_to_jail`,
   `get_out_of_jail`), and parameters (`target`, `amount`, `house_cost`,
   `hotel_cost`, `pass_go_collects`). Movement cards directly affect the
   Markov-chain transition matrix (feature 4), not just the Monte Carlo
   simulation. Cards to model:

   **Chance (16 cards):**
   1. Advance to Boardwalk (move to square 39; no explicit Go bonus — but collect if you pass Go)
   2. Advance to Go — collect $200
   3. Advance to Illinois Avenue — collect $200 if you pass Go
   4. Advance to St. Charles Place — collect $200 if you pass Go
   5. Advance to nearest Railroad — pay double rent if owned, otherwise may buy
   6. Advance to nearest Railroad (identical second copy)
   7. Advance to nearest Utility — pay 10× the dice roll if owned
   8. Bank pays you dividend of $50
   9. Get Out of Jail Free (kept in hand)
   10. Go Back 3 Spaces
   11. Go to Jail (directly to Jail square, do not collect Go)
   12. Make general repairs — $25/house, $100/hotel
   13. Speeding fine — pay $15
   14. Take a trip to Reading Railroad — collect $200 if you pass Go
   15. Chairman of the Board — pay $50 to every player
   16. Building loan matures — collect $150

   **Community Chest (16 cards):**
   1. Advance to Go — collect $200
   2. Bank error in your favor — collect $200
   3. Doctor's fee — pay $50
   4. From sale of stock you get $50
   5. Get Out of Jail Free (kept in hand)
   6. Go to Jail
   7. Holiday fund matures — collect $100
   8. Income tax refund — collect $20
   9. It is your birthday — collect $10 from every player
   10. Life insurance matures — collect $100
   11. Hospital fees — pay $100
   12. School fees — pay $50
   13. Receive $25 consultancy fee
   14. Street repairs — $40/house, $115/hotel
   15. You have won second prize in a beauty contest — collect $10
   16. You inherit $100

   Test: a property-based test must draw 10⁴ cards and verify that each
   of the 32 cards is drawn with a frequency consistent with 1/16 per
   deck (chi-squared), and that the deck state after a full cycle is
   identical to the initial state (except for "Get Out of Jail Free"
   cards).
4. **Closed-form Markov chain analysis** — build the 40×40 transition
   matrix (handling Chance/Community Chest movement and the "just
   visiting" vs "in jail" logic) and compute the stationary distribution
   → landing probability per square. Expected output: Jail ~5.9%,
   Illinois Ave ~3%, orange/red groups as the most profitable.
5. **Expected rent per roll and ROI per color** — ranking of color groups
   (brown, light blue, pink, orange, red, yellow, green, dark blue,
   railroads, utilities) by expected rent at 0/1/2/3/4 houses and hotel,
   and computation of the **payback period** (how many dice rolls are
   needed to recover the house investment). Expected answer: **orange =
   best overall ROI**, dark blue best ROI at 3 houses, light blue best
   ROI with hotels relative to capital invested.
6. **Monte Carlo simulation of full games** — N players (2–6) playing
   with configurable strategies until all but one go bankrupt. Target
   ≥ 10⁵ games per configuration, parallelized. Metrics: win rate per
   strategy, average duration (turns), average bankruptcy time, wealth
   over time.
7. **Pluggable strategy library** — `Strategy` interface with at least:
   `BuyEverything`, `BuyNothing`, `ColorTargeted(orange|lightblue|darkblue)`,
   `ThreeHousesRush` (build up to 3 houses as fast as possible, then
   stop — the "housing shortage" rule), `JailCamper` (stay in jail in
   the late game), `Trader` (accept/propose trades rationally via
   expected value).
8. **Answer to "what is the probability of winning?"** — table and chart
   of win rate per strategy vs number of opponents (2, 3, 4, 5, 6), with
   Wilson confidence intervals. Baseline = 1/N (fair game).
9. **Answer to "what is the best strategy?"** — round-robin tournament
   between all strategies, final ranking with a statistical test
   (Bradley-Terry or Elo), and an ablation study on the key parameters
   (building threshold, cash reserve threshold, early-vs-late jail
   policy).
10. **Video-ready visualizations** — (a) board heatmap colored by
    stationary probability, (b) bar chart of ROI per color at 0–5
    buildings, (c) win rate curve per strategy, (d) animation of a
    sample game, (e) chart of players' net worth over time. All outputs
    at 1920×1080, accessible palette, legible font.
11. **Typer CLI** — commands: `monopoly simulate`, `monopoly markov`,
    `monopoly tournament`, `monopoly plot <figure>`, `monopoly export-video`.
12. **Narrative Python script** — `scripts/come_vincere_al_monopoli.py`
    which follows the video outline step by step (rules → dice math →
    Markov → ROI → simulation → winning strategy) and regenerates all
    figures with a single `python scripts/come_vincere_al_monopoli.py`
    invocation. **No Jupyter notebooks**: the project uses only `.py`
    scripts.
13. **Video script at `docs/video_script.md`** — sections: Intro (the
    question), Essential rules, Dice are not uniform, Jail is the center
    of the board, Orange wins, The 3-houses rule, When to leave jail,
    100,000-game simulation, The optimal strategy, Conclusion. Each
    section references a figure/number generated by the code.

## Non-functional Requirements
- **Reproducibility**: every simulation accepts a `--seed`; all video
  figures must be regeneratable with a single command
  (`make video-assets`).
- **Performance**: ≥ 50,000 full games / minute on a modern laptop
  (NumPy/numba vectorization; no Python loops in the hot path).
- **Correctness**: ≥ 90% test coverage on the game engine; property
  tests verifying invariants (total money conserved excluding
  taxes/salaries, transition-matrix rows summing to 1, no unreachable
  squares).
- **Numerical verifiability**: analytically computed stationary
  probabilities must match Monte Carlo measurements within ±0.1%.
- **Visual accessibility**: colorblind-friendly palette, labels in
  Italian.
- **Licensing**: no Hasbro assets; square names and values from
  public-domain / educational fair-use sources, with a disclaimer in the
  README.

## Project Structure
```
monopoly/
├── src/monopoly/
│   ├── board.py          # load board from YAML
│   ├── dice.py           # 2d6 distribution, doubles
│   ├── cards.py          # Chance / Community Chest decks
│   ├── game.py           # game engine (state, turns, bankruptcy)
│   ├── markov.py         # transition matrix + stationary distribution
│   ├── strategies/       # one class per strategy
│   ├── simulate.py       # vectorized Monte Carlo
│   ├── tournament.py     # round-robin + ranking
│   ├── metrics.py        # win rate, ROI, payback
│   ├── plots.py          # all video figures
│   └── cli.py            # Typer entrypoint
├── data/board_standard.yaml
├── data/board_italia.yaml
├── data/cards_standard.yaml   # 16 Chance + 16 Community Chest
├── scripts/come_vincere_al_monopoli.py  # no notebooks, .py only
├── docs/video_script.md
├── tests/
└── pyproject.toml
```

## Additional Notes
- **Python environment**: use the existing `monopoly` conda/mamba
  environment (`conda activate monopoly`). All `python`, `pytest`,
  `pip install`, and Typer CLI commands must run inside this env. No
  new `venv`, no `poetry shell`. Record the env in `environment.yml` at
  the repo root for reproducibility.
- **Mathematical reference sources** to validate results:
  - Truman Collins, *Probabilities in the Game of Monopoly®*
    (http://www.tkcs-collins.com/truman/monopoly/monopoly.shtml) —
    canonical per-square stationary probabilities.
  - MIT Sp.268, *Markov and Mr. Monopoly Make Millions*
    (https://web.mit.edu/sp.268/www/probability_and_monopoly.pdf).
  - Williams College, *Markov Chains in the Game of Monopoly*
    (Hudson/Li).
  - Cornell ORIE, Shane Henderson, *monopoly.pdf* (strategy
    optimization).
  - TED Ideas, *Here's how to win at Monopoly, according to math
    experts* (3-houses rule, building when the opponent is ~7 squares
    away).
- **Expected results to validate against** (if the numbers diverge,
  there's a bug in the engine): Jail 5.89%, Illinois Ave 3.18%, orange
  group highest expected rent per roll, dark blue best ROI at 3 houses,
  light blue best relative ROI with hotels, ~30 average turns per
  player per game.

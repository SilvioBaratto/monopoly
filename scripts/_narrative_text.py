"""Italian commentary strings for the narrative script.

Single Responsibility: owns all human-readable Italian text used by
come_vincere_al_monopoli.py.  No computation lives here.
"""

from __future__ import annotations

TEXTS: dict[str, str] = {
    # -----------------------------------------------------------------------
    # Section headers
    # -----------------------------------------------------------------------
    "header_01": (
        "\n" + "=" * 70 + "\n  SEZIONE 1 — Introduzione e tabellone\n" + "=" * 70
    ),
    "header_02": (
        "\n" + "=" * 70 + "\n"
        "  SEZIONE 2 — Analisi della probabilità dei dadi\n" + "=" * 70
    ),
    "header_03": (
        "\n" + "=" * 70 + "\n"
        "  SEZIONE 3 — Catena di Markov e distribuzione stazionaria\n" + "=" * 70
    ),
    "header_04": (
        "\n" + "=" * 70 + "\n  SEZIONE 4 — ROI per gruppo colore\n" + "=" * 70
    ),
    "header_05": (
        "\n" + "=" * 70 + "\n  SEZIONE 5 — Simulazione Monte Carlo\n" + "=" * 70
    ),
    "header_06": ("\n" + "=" * 70 + "\n  SEZIONE 6 — Partita campione\n" + "=" * 70),
    "header_07": (
        "\n" + "=" * 70 + "\n  SEZIONE 7 — Torneo tra strategie\n" + "=" * 70
    ),
    "header_08": ("\n" + "=" * 70 + "\n  SEZIONE 8 — Conclusioni\n" + "=" * 70),
    # -----------------------------------------------------------------------
    # Section body text
    # -----------------------------------------------------------------------
    "intro": (
        "Benvenuti all'analisi matematica del Monopoly!\n"
        "Questo script ripercorre tutte le sezioni del video, dalla struttura\n"
        "del tabellone alle strategie ottimali, passando per la matematica\n"
        "dei dadi, le catene di Markov e le simulazioni Monte Carlo.\n"
        "\n"
        "Il tabellone standard contiene 40 caselle:\n"
        "  - 22 proprietà colorate (suddivise in 8 gruppi)\n"
        "  - 4 stazioni ferroviarie\n"
        "  - 2 società di pubblica utilità\n"
        "  - Caselle speciali: Via!, Prigione, Parcheggio Libero, Vai in Prigione\n"
        "  - Carte Probabilità e Imprevisti, tasse\n"
    ),
    "dice_intro": (
        "Con due dadi a sei facce la somma varia da 2 a 12.\n"
        "Il 7 è il totale più frequente (probabilità ≈ 16.7%).\n"
        "Ecco la distribuzione completa:\n"
    ),
    "markov_intro": (
        "Modelliamo il movimento sul tabellone come catena di Markov a 43 stati\n"
        "(40 caselle + 3 stati di Prigione).\n"
        "La distribuzione stazionaria rivela le caselle più visitate.\n"
    ),
    "markov_done": "→ Heatmap salvata in figures/heatmap.png",
    "roi_intro": (
        "Per ogni gruppo colore calcoliamo il ROI atteso come:\n"
        "  ROI = rendita media per turno / investimento totale\n"
        "L'investimento include proprietà + case/hotel.\n"
    ),
    "roi_done": "→ Grafico ROI salvato in figures/roi_bars.png",
    "monte_carlo_intro": (
        "Simuliamo migliaia di partite per stimare il tasso di vittoria\n"
        "di ciascuna strategia al variare del numero di giocatori.\n"
        "Le bande di confidenza sono calcolate con l'intervallo di Wilson (95%).\n"
    ),
    "monte_carlo_done": "→ Curve win rate salvate in figures/win_rate_curves.png",
    "sample_game_intro": (
        "Eseguiamo una singola partita di esempio e tracciamo\n"
        "l'andamento del patrimonio netto per ciascun giocatore.\n"
    ),
    "sample_game_done": (
        "→ Grafico patrimonio netto salvato in figures/net_worth.png\n"
        "→ Animazione partita salvata in figures/game_animation.mp4"
    ),
    "tournament_intro": (
        "Torneo round-robin tra tutte le strategie disponibili.\n"
        "Ogni coppia disputa N partite; il ranking finale usa Bradley-Terry.\n"
    ),
    "tournament_done": "→ Classifica torneo stampata sopra.",
    "conclusion": (
        "Riepilogo delle strategie vincenti:\n"
        "\n"
        "  1. Costruisci case sulle proprietà con ROI più alto (arancione, rosso).\n"
        "  2. Raggiungi rapidamente 3 case per massimizzare la rendita.\n"
        "  3. Rimani in Prigione nei turni finali per evitare affitti elevati.\n"
        "  4. Acquista sempre tutte le proprietà che puoi nel primo giro.\n"
        "\n"
        "Buona fortuna — e ricorda: la matematica è dalla tua parte!\n"
    ),
}

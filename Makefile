# ---------------------------------------------------------------------------
# Monopoly simulation — build targets
# ---------------------------------------------------------------------------
# Usage:
#   make                  → show this help
#   make video-assets     → regenerate all figures and game animation
#   make clean-figures    → remove the figures/ directory
#   make test             → run the test suite
#
# Reproducibility overrides (defaults shown):
#   make video-assets SEED=42 N_GAMES=10000
# ---------------------------------------------------------------------------

SEED     ?= 42
N_GAMES  ?= 10000

FIGURES_DIR := figures
SCRIPT      := scripts/come_vincere_al_monopoli.py

# All generated output files
FIGURE_FILES := \
	$(FIGURES_DIR)/heatmap.png \
	$(FIGURES_DIR)/roi_bars.png \
	$(FIGURES_DIR)/win_rate_curves.png \
	$(FIGURES_DIR)/net_worth.png \
	$(FIGURES_DIR)/game_animation.mp4

.PHONY: help video-assets clean-figures test

# Default target — print available targets
help:
	@echo ""
	@echo "Monopoly simulation — available targets"
	@echo "----------------------------------------"
	@echo "  make video-assets           Regenerate all figures in $(FIGURES_DIR)/"
	@echo "  make clean-figures          Remove the $(FIGURES_DIR)/ directory"
	@echo "  make test                   Run the test suite (pytest tests/)"
	@echo ""
	@echo "Reproducibility variables (override on the command line):"
	@echo "  SEED=$(SEED)               Random seed passed as MONOPOLY_SEED"
	@echo "  N_GAMES=$(N_GAMES)         Simulation size passed as MONOPOLY_N_GAMES"
	@echo ""
	@echo "Example:"
	@echo "  make video-assets SEED=123 N_GAMES=500"
	@echo ""

# Main target — run the narrative script and produce all figures
video-assets: $(FIGURES_DIR)
	MONOPOLY_SEED=$(SEED) MONOPOLY_N_GAMES=$(N_GAMES) python $(SCRIPT)

# Ensure the figures/ output directory exists
$(FIGURES_DIR):
	mkdir -p $(FIGURES_DIR)

# Individual figure targets (for partial regeneration)
$(FIGURES_DIR)/heatmap.png \
$(FIGURES_DIR)/roi_bars.png \
$(FIGURES_DIR)/win_rate_curves.png \
$(FIGURES_DIR)/net_worth.png \
$(FIGURES_DIR)/game_animation.mp4: video-assets

# Remove all generated figures
clean-figures:
	rm -rf $(FIGURES_DIR)

# Run the full test suite
test:
	pytest tests/

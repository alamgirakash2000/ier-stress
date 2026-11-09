# Paper assets

| Folder | Content | Built by |
|---|---|---|
| `figures/main/` | Every figure of the paper, as vector PDF and 300 dpi PNG | `python scripts/make_paper_assets.py` |
| `tables/main/` | Every table of the paper, LaTeX with booktabs | `python scripts/make_paper_assets.py` |
| `figures/`, `tables/` | Figures and tables of the analysis pipeline fixed and hashed before the main runs | `python scripts/rebuild.py` |
| `fonts/` | Latin Modern Roman for the figure text | not built |

`figures/main/fig_schematic.tex` is the TikZ source of the schematic in the paper. It is written by hand and holds no numbers.

The paper figures need `results/aggregated`, `results/replay_check` and `results/precision_check`. See [`reproduce/README.md`](../reproduce/README.md#5-paper-figures-and-tables) for the commands.

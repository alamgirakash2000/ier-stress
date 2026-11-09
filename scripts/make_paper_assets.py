"""Build every figure and table of the paper from the result files.

    python scripts/make_paper_assets.py               uses the saved simulation outputs in results/paper_cache
    python scripts/make_paper_assets.py --force-sim   renders the filmstrip again and reruns the float64 replays

Reads results/aggregated (from scripts/aggregate.py, scripts/e1_analysis.py and scripts/replay_check.py),
results/replay_check (the stored episodes of the replay check) and results/precision_check (from
scripts/precision_check.py). Writes the figures to paper_assets/figures/main (PDF and a 300 dpi PNG), the tables to
paper_assets/tables/main, the exact sign-flip table of the 38 plan contrasts to
results/aggregated/multiple_comparisons.csv, and the numbers the captions quote to
results/paper_cache/report_numbers.json.

The frozen pipeline (scripts/rebuild.py) is not touched. Its figures and tables stay in paper_assets/figures and
paper_assets/tables.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering for the filmstrip
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from iers.envcheck import require_ierstress  # noqa: E402

require_ierstress()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force-sim", action="store_true",
                        help="render the filmstrip again and rerun the float64 replays")
    args = parser.parse_args()
    start = time.time()

    from iers.plotting.paper import common
    from iers.plotting.paper import derived

    common.apply_style()
    report = {"multiple_comparisons": str(derived.exact_family().relative_to(ROOT))}

    from iers.plotting.paper import sim

    report["f64"] = sim.float64_raw(args.force_sim)["check"]
    report["filmstrip"] = sim.filmstrip(args.force_sim)

    # The per-seed scores the figures use are recomputed from eval.csv. Check them against the aggregates.
    report["check_cells"] = cells = derived.check_cells()
    if max(cells["max_abs_diff_auc_mean"], cells["max_abs_diff_final_mean"]) > 1e-9:
        raise SystemExit(f"per-seed scores do not match results/aggregated/cells.csv: {cells}")
    report["check_family_p"] = family = derived.check_family_p()
    if max(max(v.values()) for v in family.values()) > 1e-9:
        raise SystemExit("the sign-flip p-values do not match results/aggregated/multiple_comparisons.csv")
    report["check_triggers"] = derived.check_triggers().to_dict("records")
    derived.gains_with_p(force=True)

    from iers.plotting.paper import figs_appendix as fa
    from iers.plotting.paper import figs_main as fm
    from iers.plotting.paper import main_tables as mt
    from iers.plotting.paper import tables as tb

    fm.fig_reproduction()
    _, report["fig_effects_rules"] = fm.fig_effects()
    fm.fig_filmstrip()
    _, report["fig_replay"] = fm.fig_replay()
    _, report["fig_copy_rules"] = fm.fig_copy()
    _, report["fig_stress"] = fm.fig_stress()
    fm.fig_crossq()
    _, report["fig_random_replay_check"] = fm.fig_random_replay()
    for exp in fa.EXPERIMENTS:
        _, report[f"fig_curves_{exp}"] = fa.fig_curves(exp)
    _, report["fig_rliable"] = fa.fig_rliable()

    mt.build_all()
    report["tab_reproduction"] = tb.tab_reproduction()
    verdicts = tb.tab_verdicts()
    report["tab_verdicts"] = {"verdicts": verdicts["verdicts"], "footnotes": verdicts["footnotes"]}
    report["tab_all_contrasts"] = tb.tab_all_contrasts()
    report["tab_repetition"] = tb.tab_repetition()

    common.CACHE.mkdir(parents=True, exist_ok=True)
    (common.CACHE / "report_numbers.json").write_text(json.dumps(report, indent=1, default=str))
    print(f"figures in {common.FIG_OUT.relative_to(ROOT)}, tables in {common.TAB_OUT.relative_to(ROOT)}, "
          f"{time.time() - start:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

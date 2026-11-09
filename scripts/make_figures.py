"""Rebuilds every figure from results/aggregated/, and writes paper_assets/captions.md.

    python scripts/make_figures.py
    python scripts/make_figures.py --agg results/aggregated/pilot_check --out results/aggregated/pilot_check/figures

A figure without data yet is skipped and listed as such.
"""

import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iers.envcheck import require_ierstress  # noqa: E402

require_ierstress()

from iers import ROOT  # noqa: E402
from iers.plotting import FIGURES  # noqa: E402
from iers.plotting.style import AGGREGATED, apply  # noqa: E402
from iers.plotting.style import FIGURES as FIGURE_DIR  # noqa: E402
from iers.specs import atomic_write_text  # noqa: E402

CAPTIONS = ROOT / "paper_assets" / "captions.md"


def build(agg: Path, out: Path, captions: Path) -> dict[str, list[str]]:
    apply()
    made, skipped = [], []
    lines = ["# Figure captions", "",
             "Written by `scripts/make_figures.py` from `results/aggregated/`. Edit the figure modules, not this file.", ""]
    for name in FIGURES:
        module = importlib.import_module(f"iers.plotting.{name}")
        written = module.make(agg, out)
        lines += [f"## {name}", ""]
        if written:
            made.append(name)
            lines += [module.caption(agg), ""]
        else:
            skipped.append(name)
            lines += ["No data for this figure yet.", ""]
    atomic_write_text(captions, "\n".join(lines))
    return {"made": made, "skipped": skipped}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agg", type=Path, default=AGGREGATED)
    parser.add_argument("--out", type=Path, default=FIGURE_DIR)
    parser.add_argument("--captions", type=Path, default=None, help="default: paper_assets/captions.md, or next to --out")
    args = parser.parse_args()
    captions = args.captions or (CAPTIONS if args.out.resolve() == FIGURE_DIR.resolve() else args.out / "captions.md")
    result = build(args.agg, args.out, captions)
    print(f"made {len(result['made'])} figures: {', '.join(result['made']) or 'none'}")
    if result["skipped"]:
        print(f"no data yet for: {', '.join(result['skipped'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

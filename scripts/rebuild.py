"""One command that rebuilds every aggregated result, figure and table from the raw results.

    python scripts/rebuild.py

Runs scripts/aggregate.py, scripts/make_figures.py, scripts/make_tables.py and scripts/make_results.py in
that order, and stops at the first that fails.
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from iers.envcheck import require_ierstress  # noqa: E402

require_ierstress()

SCRIPTS = Path(__file__).resolve().parent
STEPS = ["aggregate.py", "make_figures.py", "make_tables.py", "make_results.py"]


def main() -> int:
    for name in STEPS:
        print(f"--- {name}", flush=True)
        code = subprocess.run([sys.executable, str(SCRIPTS / name)]).returncode
        if code:
            print(f"{name} failed with exit code {code}")
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())

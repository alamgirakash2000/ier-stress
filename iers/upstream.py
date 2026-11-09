"""Access to the unmodified IER code in third_party/ier.

The upstream code imports `pydantic` (for config classes) and `cv2` (for video frames). The real
packages are installed in the `ierstress` environment.
"""

from __future__ import annotations

import sys

from . import ROOT

IER_DIR = ROOT / "third_party" / "ier"


def setup() -> None:
    """Make the upstream modules importable. Safe to call more than once."""
    if str(IER_DIR) not in sys.path:
        sys.path.append(str(IER_DIR))


def sac_network_classes():
    """Return the upstream SAC (Actor, Critic) classes."""
    setup()
    from networks.sac_networks import Actor, Critic

    return Actor, Critic

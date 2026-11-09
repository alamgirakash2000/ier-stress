"""Statistics. An effect is called clear only when its 95 percent CI excludes zero."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_RESAMPLES = 10_000
Z_95 = 1.959963984540054


@dataclass
class PairedResult:
    n: int
    mean: float
    low: float
    high: float
    win_rate: float
    sd: float
    clear: bool


def paired_bootstrap(differences, seed: int = 0, resamples: int = N_RESAMPLES) -> PairedResult:
    """Percentile bootstrap CI of the mean per-seed difference, with the win rate.

    NaN differences are an error, because seeds are never dropped silently.
    """
    values = np.asarray(differences, dtype=np.float64)
    if values.size == 0:
        raise ValueError("No differences to bootstrap.")
    if not np.isfinite(values).all():
        raise ValueError("A per-seed difference is not finite. Report that run, do not drop it.")
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, values.size, size=(resamples, values.size))
    means = values[picks].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return PairedResult(
        n=int(values.size),
        mean=float(values.mean()),
        low=float(low),
        high=float(high),
        win_rate=float((values > 0).mean()),
        sd=float(values.std(ddof=1)) if values.size > 1 else float("nan"),
        clear=bool(low > 0 or high < 0),
    )


def seeds_for_half_width(sd: float, half_width: float) -> int:
    """Seeds needed for a normal-approximation 95 percent CI half-width below `half_width`."""
    if not np.isfinite(sd):
        return 10**9
    return max(2, int(np.ceil((Z_95 * sd / half_width) ** 2)))


def iqm_interval(scores: dict[str, np.ndarray], reps: int = N_RESAMPLES):
    """IQM with stratified bootstrap 95 percent CIs. Each value has shape (seeds, tasks)."""
    from rliable import library as rly
    from rliable import metrics

    return rly.get_interval_estimates(scores, lambda x: np.array([metrics.aggregate_iqm(x)]), reps=reps)

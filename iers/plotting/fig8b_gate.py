"""Figure 8b, appendix: the implementation check, our learners against the stable-baselines3 references."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .style import (DOUBLE, OKABE_ITO, TASK_LABEL, bootstrap_band, cli, line, panel_label, read_csv, save,
                    seed_matrix, steps_axis)

NAME = "fig8b_gate"
PAIRS = [("sac", "ours_sac", "sb3_sac", "SAC"), ("crossq", "ours_crossq", "sb3_crossq", "CrossQ")]
STYLE = {
    "ours": {"color": OKABE_ITO["black"], "linestyle": "-", "marker": "o", "label": "this study"},
    "sb3": {"color": OKABE_ITO["vermillion"], "linestyle": "--", "marker": "s", "label": "reference"},
    "debug": {"color": OKABE_ITO["sky_blue"], "linestyle": ":", "marker": "^", "label": "reference, upstream log std"},
}


def make(agg: Path, out: Path):
    data = read_csv(agg, "gate_curves.csv")
    if data is None:
        return None
    pairs = [p for p in PAIRS if {p[1], p[2]} & set(data.implementation)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(DOUBLE if len(pairs) > 1 else DOUBLE / 2, 1.9), squeeze=False)
    for ax, (_, ours, ref, label) in zip(axes[0], pairs):
        for implementation, kind in ((ours, "ours"), (ref, "sb3"), (f"{ref}_upstream_logstd", "debug")):
            part = data[data.implementation == implementation]
            if not len(part):
                continue
            matrix = seed_matrix(part)
            mean, low, high = bootstrap_band(matrix.to_numpy())
            style = STYLE[kind]
            line(ax, matrix.columns, mean, style, label=f"{style['label']} ({len(matrix)} seeds)")
            ax.fill_between(matrix.columns, low, high, color=style["color"], alpha=0.2, linewidth=0)
        task = data[data.implementation.isin([ours, ref])].task.iloc[0]
        panel_label(ax, f"{label}, {TASK_LABEL.get(task, task)}")
        steps_axis(ax)
        ax.set_xlabel("environment steps")
        ax.legend(loc="lower right")
    axes[0][0].set_ylabel("evaluation return")
    fig.tight_layout()
    return save(fig, out, NAME)


def caption(agg: Path) -> str:
    return ("Implementation check on pilot seeds: evaluation return of this study's SAC and CrossQ against "
            "stable-baselines3 SAC and sb3-contrib CrossQ with matched settings. The dotted SAC curve is "
            "stable-baselines3 SAC with the log standard deviation of the upstream IER actor, the one difference "
            "found between the two SAC implementations. Lines show the mean over seeds and bands a 95 percent "
            "percentile bootstrap interval of the mean. Pilot seeds do not enter any other result.")


if __name__ == "__main__":
    cli(make, caption)

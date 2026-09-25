"""Draw Figure 5 of the paper from the records this repository ships.

Figure 5 is a forest plot of Table III's four nine-seed reliability rates with their
exact 95% Clopper-Pearson intervals, against the flat-manifold chance level of 1/3.

Nothing plotted is typed into this file. The rates are composed from the shipped
per-checkpoint grade records by `instruments.grade_composer`, and the intervals are
computed by `instruments.statistics.clopper_pearson` -- the same two instruments that
`python -m instruments.reproduce_table3` runs. This file only draws.

    python docs/paper/figure_data/generate_fig5.py [out.pdf]

Needs matplotlib (see requirements-figures.txt); the instruments themselves need
nothing outside the standard library. PDF metadata dates are suppressed, so two runs
with the same matplotlib version write byte-identical files.
"""
import json
import sys
from pathlib import Path

# Resolve the repository root from THIS FILE, as generate_fig1_data.py does, so the
# script works from any working directory.
_REPO_ROOT = Path(__file__).resolve()
while _REPO_ROOT.parent != _REPO_ROOT and not (_REPO_ROOT / "instruments" / "grade_composer.py").is_file():
    _REPO_ROOT = _REPO_ROOT.parent
if not (_REPO_ROOT / "instruments" / "grade_composer.py").is_file():
    raise RuntimeError("could not locate the repository root (no ancestor contains instruments/grade_composer.py)")
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

from instruments import grade_composer as composer  # noqa: E402
from instruments import statistics as stats  # noqa: E402

PT = 72.0
COLUMN_WIDTH = 251.07   # the paper's column width, in points
GUARD = 2.0             # canvas 2 pt narrower than the column; placed at column width, 8-pt text renders at >= 8 pt
HEIGHT = 157.0          # canvas height, in points
CHANCE = 1.0 / 3.0      # the flat-manifold chance level
ANN = 8                 # annotation and guide-label size
INK = "#000000"         # one ink for every row: no row is emphasized
LW = 1.0                # interval bar and guide weight
MS = 4.0                # point-estimate marker size
CAP = 0.16              # half-height of the endpoint serifs, in row units

# Where each Table III row label breaks onto two lines (layout only; the words are
# grade_composer.CRITERION_LABELS).
LABEL_BREAK = {
    "task_success": ("Task success", "(joint with hazard)"),
    "core_hold": ("Selective-sensing", "hold (core)"),
    "anchor_consolidation": ("Anchor consolidation", "(load-bearing)"),
    "transfer": ("Transfer to", "validation"),
}

matplotlib.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "font.family": "serif",
    "axes.linewidth": 0.6,
    "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "figure.constrained_layout.use": False,
    "path.simplify": False,
})


def rows_from_records():
    """The four rows, in Table III's order, composed from the shipped records."""
    n9 = composer.tallies(composer.compose_all())
    rows = []
    for criterion in composer.CRITERIA:
        first, second = LABEL_BREAK[criterion]
        if f"{first} {second}" != composer.CRITERION_LABELS[criterion]:
            raise RuntimeError(f"label break for {criterion} does not match CRITERION_LABELS")
        k, n = n9[criterion]
        lo, hi = stats.clopper_pearson(k, n)
        rows.append({"label": f"{first}\n{second}", "k": k, "n": n, "lo": lo, "hi": hi})
    return rows


def draw(rows, path):
    fig, ax = plt.subplots(figsize=((COLUMN_WIDTH - GUARD) / PT, HEIGHT / PT), layout="constrained")
    ys = list(range(len(rows) - 1, -1, -1))  # Table III order, top row first
    ax.axvline(CHANCE, color="0.55", ls=":", lw=LW, zorder=0.5)
    for r, y in zip(rows, ys):
        ax.plot([r["lo"], r["hi"]], [y, y], color=INK, lw=LW, solid_capstyle="butt", zorder=2)
        for x in (r["lo"], r["hi"]):
            ax.plot([x, x], [y - CAP, y + CAP], color=INK, lw=LW, zorder=2)
        est = r["k"] / r["n"]
        ax.plot([est], [y], marker="o", ms=MS, mfc=INK, mec=INK, lw=0, zorder=3)
        ax.text(est, y + 0.28, "%d/%d" % (r["k"], r["n"]), fontsize=ANN,
                ha="center", va="bottom", color=INK, zorder=4)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["label"] for r in rows], ma="right")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.62, 4.20)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, CHANCE, 2 * CHANCE, 1.0])
    ax.set_xticklabels(["0", "1/3", "2/3", "1"])
    ax.text(CHANCE + 0.015, 4.10, "flat-manifold chance", fontsize=ANN, ha="left", va="top", color="0.3")
    ax.set_xlabel("rate over nine seeds")
    ax.tick_params(axis="x", direction="out", length=2.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.02)
    fig.savefig(path, metadata={"CreationDate": None, "Producer": None, "Creator": None})
    plt.close(fig)


def main(argv):
    out = Path(argv[1]) if len(argv) > 1 else Path("fig5_intervals.pdf")
    rows = rows_from_records()
    draw(rows, out)
    print(json.dumps([{k: v for k, v in r.items()} for r in rows], indent=1))
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

# Reference pages

These pages describe the code and records in this repository. The six pages below each answer one question. The tag `v1.0-icet2026` is the state reported in the paper; the [README](../README.md) says how the default branch differs from it.

## The pages

- [The environment and the risk fork](environment.md): what is the environment, and what is a risk fork?
- [What an agent observes: the two encodings](encodings.md): what does an agent observe, and why does the original encoding alias?
- [The control corridor](corridor.md): what does the control corridor control, and where does each Table I lever live?
- [How a run is graded](grading.md): how is a run graded?
- [Scenarios and the reserved held-out geometry](scenarios.md): which scenarios exist, and what is held out?
- [Run manifests and grade records](schemas.md): what is in a run manifest and a grade record?

## Other documentation under docs/

- [`PAPER_MAP.md`](PAPER_MAP.md): tables that map items in the paper, by PDF page, to the command that prints each value or the file that holds it.
- [`evidence/`](evidence/): the five certification and regression instruments (`*.py`) and the 17 graded per-unit run analyses (`*_analysis.json`).
- [`paper/figure_data/`](paper/figure_data/): the backing data for Fig. 1 and its generator, and a generator for Fig. 5.

Source: `README.md` (`What is here, by directory`, `The state reported in the paper is the tag`); `docs/PAPER_MAP.md` (`These tables map items in the paper`); `docs/evidence/*.py`; `docs/evidence/*_analysis.json`; `docs/paper/figure_data/fig1_aliasing.json`; `docs/paper/figure_data/generate_fig1_data.py`; `docs/paper/figure_data/generate_fig5.py`

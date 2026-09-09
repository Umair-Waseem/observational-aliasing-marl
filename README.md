# Observational Aliasing Bounds Generalization in Constrained Active-Sensing Multi-Agent Reinforcement Learning

Code, grading instruments, run provenance, per-checkpoint grade records and a
claim-to-source traceability register accompanying the IEEE ICET 2026 paper of
the same title.

**The state reported in the paper is the tag [`v1.0-icet2026`](../../releases/tag/v1.0-icet2026).**
The default branch may move; that tag will not.

---

## Reproduce the paper's numbers

Python **3.12** or newer. Nothing else — the three release instruments are
standard library only, so this needs no packages at all, not even PyTorch:

```
git clone https://github.com/Umair-Waseem/observational-aliasing-marl
cd observational-aliasing-marl
git checkout v1.0-icet2026
python -m instruments.reproduce_table3
```

That recomputes, from the records in this repository, every number in Table III
of the paper — the four nine-seed reliability rates, their exact 95 %
Clopper–Pearson intervals, the reroute-basin partition, and the arm-fire
de-confound Fisher *p* — and prints the paper's printed value beside each one.
It exits non-zero if any of them stops matching, so it is usable as a gate:

```
Criterion                                     n=9   exact 95% interval fresh-only
Task success (joint with hazard)       5/9 = 0.56       [0.212, 0.863] 3/6 = 0.50
Selective-sensing hold (core)          4/9 = 0.44       [0.137, 0.788] 2/6 = 0.33
Anchor consolidation (load-bearing)    7/9 = 0.78       [0.400, 0.972] 4/6 = 0.67
Transfer to validation                 2/9 = 0.22       [0.028, 0.600] 1/6 = 0.17

The transfer interval [0.028, 0.600] CONTAINS the chance level 1/3 = 0.333.

Reroute-basin classification (n=9):
  equivariant / overshoot / freeze  =  4/9 / 2/9 / 3/9   (paper: 4/9 / 2/9 / 3/9)

Arm-fire de-confound: 2x2 = [[1, 1], [1, 6]]   Fisher p = 0.4167  (paper: 0.417)
```

The three instruments also run individually:

```
python -m instruments.grade_composer --check    # per-seed composition + registry audit
python -m instruments.reroute_basin             # the basin partition, seed by seed
python -c "from instruments.statistics import clopper_pearson; print(clopper_pearson(2, 9))"
```

To run the environment, the training drivers, or a re-grade from the shipped
policy checkpoints, install PyTorch at the recorded version and the package
itself:

```
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The pip upgrade is not ceremony: `pyproject.toml` uses PEP 639 license metadata,
which pip below 24.2 rejects with a metadata error.

`requirements.txt` carries its own `--index-url` for the PyTorch CPU wheel, so
that command is correct as written on every platform; on Linux the default PyPI
wheel is the multi-gigabyte CUDA build and the recorded one is CPU. Install
**editable** (`-e`):
`tests/conftest.py` and two of the five `docs/evidence` instruments put
`<repo>/src` on `sys.path` themselves, and a non-editable install would put a
second copy of `raas_marl` in `site-packages` alongside it.

To run the test suite you also need pytest, which is a development dependency
rather than a runtime one:

```
python -m pip install -r requirements-dev.txt
python -m pytest tests -o addopts="" -p no:cacheprovider -q     # 1539 tests
```

Every command in this file is written to run unchanged on Linux, macOS and
Windows: `python -m` invocations, forward slashes, no shell scripts.

---

## Data availability

This section is the data-availability statement promised in the response to
reviewer comment R1-S9.

### What this repository contains

| Promise | Where it is |
|---|---|
| The full implementation — environment, encodings, corridor controller, certification harness | `src/raas_marl/` (34 files, 29 modules) and `docs/evidence/c1_selectivity_harness.py` |
| The per-run configuration records | `results/experiments/<run_id>/RUN_MANIFEST.json` — **all 55 runs** |
| The grading instruments, including the reroute-basin classifier and the grade composer | `instruments/` (three instruments plus a driver) and `docs/evidence/*.py` (five modules) |
| The recorded per-checkpoint grade records | `results/experiments/<run_id>/checkpoint_grades/grade_r{003500,003750,004000}.json` — **27 records**, the nine replication seeds × three checkpoints |
| A README data-availability statement | this section |

Beyond what was promised, the repository also ships the **27 graded policy
checkpoints** (`results/experiments/<run_id>/state_round_{003500,003750,004000}.pt`,
3.35 MB in total, ≤ 122 KB each), so a reader can re-grade from the weights in
place rather than only recompose from the records:

```
python docs/evidence/c1_selectivity_harness.py --grade-run t1_retain_s147 --fork
```

That rolls the shipped policy on all ten fork surfaces, re-runs the own-ablation
causal witness, and returns `SELECTIVE_COMPOSITE` with component counts
`success 8, senses 10, within_ceiling 10, adapt 5, causal 5, witness 4` —
identical to the recorded grade in that run's
`checkpoint_grades/grade_r004000.json`. It needs torch, and it takes a few
minutes per run.

### Which runs are included, and where each appears in the paper

All 55, by run id. Every one carries a complete configuration record; seeds and
the commit each run was launched from are in its `RUN_MANIFEST.json`.

| Group | Runs | Seeds | In the paper |
|---|---|---|---|
| **The nine replication seeds** | `t1_retain_s147`–`s149`, `t1_seedscale_s153`–`s158` | 147–149, 153–158 | Table III, Fig. 4, Fig. 5 — the reliability rates, the intervals, the basin partition |
| **Table II, two-cell-obstacle-patch arm** | `t1_retain_s147`–`s149` | 147–149 | Table II — the enriched encoding (retention 2/3) |
| **Table II, one-cell-scalar-count arm** | `t1_d15_s138`–`s140` | 138–140 | Table II — the original encoding (retention 0/3). Same mixture and configuration; the encoding is the single variable. The two arms use different seed sets, as the caption states |
| **Single-pair runs, one-cell-scalar count** | `t1_d13_s132`–`s134` | 132–134 | Sec. V-A — the anchor pair alone |
| **Single-pair runs, two-cell obstacle patch** | `anchor_rerun_s144`–`s146` | 144–146 | Sec. V-A — the same pair after the encoding change |
| **The three original-encoding curricula** | `t1_d14_s135`–`s137` (four pairs), `t1_d15_s138`–`s140` (two pairs, shared gate row), `t1_d16_s141`–`s143` (two pairs, disjoint gate rows) | 135–143 | Sec. V-B — the curriculum families that did not lift generalization |
| **The tuning campaign** | `t1_d4`–`t1_d12` (nine sets of three) | 105–131 | Table I and Sec. IV — how the corridor controller was arrived at |
| **Baselines** | `b101_stoch`, `b102_stoch`, `b103_stoch`, `b104_greedy` | 101–104 | Sec. IV — the pre-fork baseline on `risk_gate_hidden_hazard` |

Seeds are drawn in ascending order from a fixed training pool (integers 101–199).
**Seeds 150, 151 and 152 were allocated to an experiment cancelled before launch
and were never run**, which is why the nine-seed replication is 147–149 and
153–158; no run directory for them exists here, and that absence is checkable.

### What recomputes from what

The reported reliability rates recompute from the shipped per-checkpoint grade
records via the released composer; the reported intervals and *p*-values follow
from those records by exact-binomial (Clopper–Pearson) and Fisher computations
over the released counts. Stated precisely, node by node:

```
  results/experiments/*/checkpoint_grades/grade_r*.json      (27 records)
  results/held2_conjuncts.json                               (curve-derived conjuncts)
  docs/evidence/t1_seedscale_analysis.json                   (per-mirror tail hazard)
                    |
                    |   python -m instruments.grade_composer
                    v
  the four n=9 rates  5/9, 4/9, 7/9, 2/9   and the fresh-seed-only 3/6, 2/6, 4/6, 1/6
                    |
                    |   python -m instruments.statistics  (Clopper-Pearson)
                    v
  the four exact 95% intervals   [0.212,0.863] [0.137,0.788] [0.400,0.972] [0.028,0.600]

  docs/evidence/t1_seedscale_analysis.json :: per_seed[*].readiness_lower_outcome
                    |
                    |   python -m instruments.reroute_basin
                    v
  the basin partition   equivariant {147,153,157,158} / overshoot {149,154} / freeze {148,155,156}

  docs/evidence/t1_seedscale_analysis.json :: per_seed[*].mech_f84_fired  x  the transfer conjunct
                    |
                    |   python -m instruments.statistics  (Fisher exact, two-sided)
                    v
  the de-confound   2x2 = [[1,1],[1,6]]   p = 0.417
```

Two of the four criteria are conjunctions that draw on both shipped record kinds,
and the composer takes each conjunct from its own source. **Transfer** — the
paper's headline negative — recomputes from the per-checkpoint records alone.
**Task success** is a training-curve quantity and uses no checkpoint record.
**Core hold** and **anchor consolidation** each combine a curve-derived conjunct
with a two-of-three witness conjunct derived from the records. The module
docstring of `instruments/grade_composer.py` states this per criterion, and
`--check` audits the recomputation against the registered per-seed values seed by
seed rather than only in aggregate.

### The reserved held-out geometry

The paper says of the gate-row split (Sec. IV):

> a validation set of readiness geometries at gate rows {1,2} interpolates
> between trained rows 0 and 3; and a held-out set at gate rows {5,6} is
> deliberately preserved and never instantiated, read, or cited throughout the
> campaign, so that a future held-out evaluation would be meaningful.

Nothing in this repository contradicts that, and the claim is checkable here
rather than only asserted:

* **No such scenario exists.** Deriving the gate rows of all ten fork scenarios
  from `grid_environment.py` — the free rows of the wall on column 4 — gives
  `{3,4}` (trained), `{1,2}` (validation), `{0,3}`, `{4,7}` and `{0,7}`
  (curriculum). None is `{5,6}`, and no hidden hazard sits on row 5 or 6
  anywhere in the catalogue. `stage24_diagnostics.py` records why: the held-out
  fork group "is DEFERRED to the Protocol-v1 lock (I-3), so it is not named here".
* **No run touched one.** Across all 55 `RUN_MANIFEST.json` the only
  scenario-carrying field is `config.scenario_names`, and its union over the 55
  is nine names, none held-out. No manifest records a layout override, so no run
  could have built one under another name.
* **No grade record contains one.** The 27 per-checkpoint records grade exactly
  ten surfaces, grouped `training`/`curriculum`/`readiness` — there is no
  `held_out` group — and the hidden-hazard rows they grade are 0, 1, 2, 3, 4
  and 7.
* **Nothing was read, either.** The union of every scenario token across all 55
  probe logs of the whole campaign is thirteen names; none is a held-out
  surface. (The probe logs are Zenodo-class data and are not in this repository;
  the check is stated so it can be repeated against that record.)

What *is* retained, deliberately: the scenario catalogue's reservation entries
and comments, the `held_out` group label in the Stage 24-A variant table, and
the `_assert_not_held_out` guard in the certification harness. Those are the
documentation of the claim, not a use of it, and the test suite asserts the
reserved name stays absent from the graded surfaces. Gate rows are ordinary
parameters of the environment, so the code can of course express other
geometries; what the claim is about is what was run, and none was.

### The raw training logs

The raw episode logs, training curves and probe logs are large: **≈ 0.74 GB for
the nine replication seeds** (586 MB of episode logs, 125 MB of curves, 15 MB of
probe logs) and **≈ 4.1 GB across all 55 runs**. They are not in this repository.

They will be archived in a separate public data record with a persistent
identifier. **No DOI has been minted as of this release, so none is claimed
here.** Until one is, the logs are available on request from the corresponding
author (Musadaq Mansoor, `musadaq.mansoor@paf-iast.edu.pk`); when the record is
minted, its DOI will be added to `CITATION.cff` and to this section.

Independent re-grading *from the raw logs* — for the witness-gated cells in
particular — needs that record. Everything the paper reports in Table III,
however, recomputes from what is here, and the shipped policy checkpoints allow a
re-grade at the three graded rounds without it.

---

## Contents

| Path | What it is |
|---|---|
| `instruments/` | The three release instruments — the reroute-basin classifier, the grade composer, and the Clopper–Pearson/Fisher statistics — plus `reproduce_table3.py`, which runs all three. Standard library only. |
| `src/raas_marl/` | The package: environment, encodings, MAPPO-Lagrangian core, corridor controller, training drivers. |
| `docs/evidence/*.py` | The five certification and regression instruments, preserved at their source-repository coordinates. |
| `docs/evidence/*_analysis.json` | The graded per-unit run analyses, one per experimental unit. |
| `results/experiments/*/RUN_MANIFEST.json` | Per-run provenance and configuration for all 55 runs. |
| `results/experiments/*/checkpoint_grades/` | The 27 per-checkpoint grade records (9 seeds × rounds 3500/3750/4000). |
| `results/experiments/*/state_round_*.pt` | The 27 graded policy checkpoints, for the same 9 × 3. |
| `results/held2_conjuncts.json` | The curve-derived conjuncts of the task-success and core-hold criteria. |
| `results/checkpoint_grades_manifest.json`, `results/checkpoints_manifest.json` | Byte sizes and SHA-256 for every shipped record and checkpoint, with the filename each came from. |
| `CLAIMS.yaml` | The traceability register: every registered paper claim mapped to the file that substantiates it. Paths are in source-repository coordinates, which this artifact preserves exactly. |
| `docs/paper/figure_data/` | Backing data and generator for Figure 1. |
| `tests/` | The test suite, shipped whole and unmodified (1539 tests). |
| `scripts/check_provenance.py` | A gate proving each instrument loads this repository's code and not some other tree's. |

## What is not here

* **The raw training logs and the non-graded checkpoints** — see the data
  availability section above.
* **Anything touching the reserved held-out geometry** — see above.
* **A DOI for the data record** — not minted yet, and not claimed.
* **The decision and evidence dossiers** (`docs/decisions/**`, `docs/evidence/ED-*.md`,
  the countersign records). These are the campaign's internal deliberation — 31
  decision records and 24 dossiers — and are classified private by the artifact's
  shipping manifest. `CLAIMS.yaml` cites one of them, `ED-basin-lever-search.md`, as
  the source for a single introduction claim, and that entry says so in its own
  `criterion_note`: the claim "is NOT verifiable from the public artifact". Every
  other one of the 117 claim entries resolves to a file that is here.
* **`docs/paper/template/main.tex` and `docs/paper/VENUE_KIT.md`.** `CITATION.cff` and
  `pyproject.toml` name them as where their values were read from; they live in the
  source repository, and both comments now say so.

---

## Reproduction environment

Every value in this section was read from the `runtime_record` block of the
**nine graded** run manifests — `t1_retain_s147`, `s148`, `s149`,
`t1_seedscale_s153`, `s154`, `s155`, `s156`, `s157`, `s158` — which are shipped
in this repository under `results/experiments/`. Check any of them yourself.

### Software

| | Recorded value | Agreement across the nine |
|---|---|---|
| Python | `3.12.10` (`tags/v3.12.10:0cc8128, Apr 8 2025, 12:21:36`, `MSC v.1943 64 bit (AMD64)`) | identical in all nine |
| PyTorch | `2.5.1+cpu` | identical in all nine |
| Device | `cpu` | identical in all nine |
| Tensor dtype | `float32` | identical in all nine |

`requires-python` is declared as `>=3.12` in `pyproject.toml` — the minor
version that was actually exercised. The exact patch is above. `requirements.txt`
pins `torch==2.5.1`, the version that was run; `pyproject.toml` declares the
looser floor `torch>=2.5.1` for installing the package.

`torch` is the only third-party runtime dependency. An AST census over all 42
shipped Python files finds four non-standard-library top-level import names, of
which only two are external: `raas_marl` (this package importing itself) and
`c1_selectivity_harness` (a grading instrument imported as a sibling module by
two others, and shipped alongside them) are internal; `torch` (19 files) and
`pytest` (1 file, the test suite) are the external ones. `pytest` is therefore
declared only under `[project.optional-dependencies].dev`. The `instruments/`
modules add no dependency at all: CI asserts by AST that they import nothing
outside the standard library.

### Threading

| Setting | Recorded value |
|---|---|
| `OMP_NUM_THREADS` | `2` |
| `MKL_NUM_THREADS` | `2` |
| `OPENBLAS_NUM_THREADS` | not set |
| `NUMEXPR_NUM_THREADS` | not set |
| `PYTHONHASHSEED` | not set |
| `torch.get_num_threads()` | `2` |
| `torch.get_num_interop_threads()` | `8` |

The thread-settings policy is recorded as `observed_only`: the runs recorded
the environment they found, they did not set it.

### Hardware

| | Recorded value |
|---|---|
| Platform | `Windows-11-10.0.26200-SP0` |
| Machine | `AMD64` |
| Processor | `Intel64 Family 6 Model 158 Stepping 13, GenuineIntel` |

**Physical core count is not a recorded field.** The manifests record
`torch_num_interop_threads = 8`, which on this platform defaults to the core
count, but that is an inference and is not stated as a fact here.

### Wall-clock

Derived from the recorded `launch_timestamp_utc` and `finish_timestamp_utc` of
each of the nine graded runs, each of which completed 4000 rounds
(`last_completed_round = 3999`, `status = complete`; and no `resumed` key is
present in any of the nine — the driver writes that key only on the resume
branch, so its absence is what shows none of them was resumed):

**2.38 h to 3.50 h per run** (minimum `t1_seedscale_s158`, maximum
`t1_retain_s147`).

**Read that range as measured elapsed time, not as isolated single-run cost.**
Concurrency is *not* a recorded manifest field, so these are wall-clock figures
obtained under a concurrency the manifests do not state. What the timestamps
*do* show is that the nine ran as three batches of three, each batch sharing one
launch instant to the second:

| Batch | Runs | Launched (UTC) |
|---|---|---|
| 1 | `t1_retain_s147`, `s148`, `s149` | `20260712T093148Z` |
| 2 | `t1_seedscale_s153`, `s154`, `s155` | `20260712T224844Z` |
| 3 | `t1_seedscale_s156`, `s157`, `s158` | `20260713T013645Z` |

So three runs were resident simultaneously on one machine, each pinned to two
threads. A single run executing alone on the same hardware would plausibly be
faster than the range above rather than slower; conversely, reproducing the full
set of nine takes roughly three batch-durations of machine time, not one.

### Determinism

The manifests record `cross_platform_bitwise_determinism_claimed = false`.
Bitwise-identical reproduction across a different platform, CPU, or PyTorch
build is **not** claimed. The seeds are recorded per run
(`t1_retain` 147–149, `t1_seedscale` 153–158).

The `instruments/` recomputation is a different matter and *is* platform
independent: it reads shipped JSON and does exact integer and floating-point
arithmetic on it, with no BLAS reduction order and no dict-iteration dependence
for a platform to change.

### Provenance

Each manifest records the commit its run was launched from, in `git_head`:

| Runs | `git_head` |
|---|---|
| `t1_retain_s147`, `s148`, `s149` | `e6b616e7729954944d9a11d0a43c1403a3b37b4b` |
| `t1_seedscale_s153`, `s154`, `s155` | `873bfcecaacbaf532afa1b09b28b6b10f86993a1` |
| `t1_seedscale_s156`, `s157`, `s158` | `2d3ffa6781237a9874129f6252ba9c5deb79329b` |

In the research repository, `git diff <each of those> HEAD -- src/` is empty:
the code did not move between the three launches.

**Twenty-nine of the 34 shipped `src/` files are byte-identical to that code.**
The other five —

    raas_marl/environments/active_sensing/grid_environment.py
    raas_marl/environments/active_sensing/tensor_adapter.py
    raas_marl/mappo_lagrangian/buffer.py
    raas_marl/mappo_lagrangian/config.py
    raas_marl/mappo_lagrangian/losses.py

— carry **annotation-only additions made for this release**: an
`if TYPE_CHECKING:` guard so a type checker can resolve `torch` in the five
modules that import it lazily, and some return annotations. Every one of the
five declares `from __future__ import annotations`, so annotations are deferred
strings and are never evaluated at run time; with docstrings, the guards and the
annotations removed, the executable AST of all five is identical to the graded
code. Nothing was renamed, reordered or rewritten.

That is a weaker statement than "byte-identical" and it is the true one. It is
also checkable: clone the research repository at any of the three commits above
and diff, or run the AST comparison yourself.

---

## Licence

Software is MIT (`LICENSE`). Data and records — the analysis JSONs, everything
under `results/`, the Figure 1 backing data, and `CLAIMS.yaml` — are CC BY 4.0
(`LICENSES/CC-BY-4.0.txt`). `LICENSE` states the split file-set by file-set.

## Citation

See `CITATION.cff`.

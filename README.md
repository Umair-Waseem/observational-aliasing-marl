# Observational Aliasing Bounds Generalization in Constrained Active-Sensing Multi-Agent Reinforcement Learning

Code, grading instruments, run provenance, per-checkpoint grade records and a
claim-to-source traceability register accompanying the IEEE ICET 2026 paper of
the same title.

**The state reported in the paper is the tag [`v1.0-icet2026`](https://github.com/Umair-Waseem/observational-aliasing-marl/releases/tag/v1.0-icet2026).**
The default branch may move; that tag will not.

---

## Reproduce the numbers in Table III

Python **3.12** or newer. Nothing else — the three release instruments are
standard library only, so this needs no packages at all, not even PyTorch:

```
git clone https://github.com/Umair-Waseem/observational-aliasing-marl
cd observational-aliasing-marl
git checkout v1.0-icet2026
python -m instruments.reproduce_table3
```

That recomputes, from the records in this repository, the n=9 column of
Table III of the paper: the four nine-seed reliability rates, their exact 95%
Clopper–Pearson intervals, the fresh-seed-only rates, and the reroute-basin
classification. It also recomputes the Fisher *p* of the arm-fire de-confound,
reported in the notes to Table III. It does not print the n=3 column. It
prints the paper's value beside the reroute-basin classification and the
Fisher *p*. It exits non-zero if any recomputed value stops matching the paper,
so it is usable as a pass/fail check. Its output is:

```
------------------------------------------------------------------------------
TABLE III, REPRODUCED FROM THE SHIPPED RECORDS
------------------------------------------------------------------------------

  Criterion                                     n=9   exact 95% interval fresh-only
  Task success (joint with hazard)       5/9 = 0.56       [0.212, 0.863] 3/6 = 0.50
  Selective-sensing hold (core)          4/9 = 0.44       [0.137, 0.788] 2/6 = 0.33
  Anchor consolidation (load-bearing)    7/9 = 0.78       [0.400, 0.972] 4/6 = 0.67
  Transfer to validation                 2/9 = 0.22       [0.028, 0.600] 1/6 = 0.17

  The transfer interval [0.028, 0.600] CONTAINS the chance level 1/3 = 0.333.

  Reroute-basin classification (n=9):
    equivariant / overshoot / freeze  =  4/9 / 2/9 / 3/9   (paper: 4/9 / 2/9 / 3/9)
      EQUIVARIANT  {147, 153, 157, 158}
      OVERSHOOT    {149, 154}
      FREEZE       {148, 155, 156}

  Arm-fire de-confound (note to Table III):
    arm-fire seeds     ['147', '156']
    transferring seeds ['147', '158']
    2x2 = [[1, 1], [1, 6]]   Fisher exact (two-sided) p = 0.4167   (paper: 0.417)

------------------------------------------------------------------------------
All reproduced values match the camera-ready.
------------------------------------------------------------------------------
```

Table III's n=3 column covers the three anchor seeds, 147 to 149. The
`grade_composer` instrument below lists the criteria that each seed meets.
For each criterion, the number of those three seeds that meet it, out of
three, is its n=3 value. The three instruments also run individually:

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

`requirements.txt` carries its own `--index-url` for the PyTorch CPU wheel.
As of 2026-09-24 UTC, that index has a `torch` 2.5.1 wheel for Python 3.12 on
Windows x86_64, Linux x86_64, Linux aarch64, and macOS arm64, and for Python
3.13 on Linux x86_64 only. It has none for macOS x86_64, Windows ARM64, or
Python 3.14. On Linux x86_64, the default PyPI wheel of `torch` 2.5.1 is the
CUDA build. That wheel and the GPU packages it requires are about 3 GB to
download. The recorded build is CPU (`2.5.1+cpu`). Install
**editable** (`-e`):
`tests/conftest.py` and four of the five `docs/evidence` instruments put
`<repo>/src` on `sys.path` themselves, and a non-editable install would put a
second copy of `raas_marl` in `site-packages` alongside it.

To run the test suite you also need pytest, which is a development dependency
rather than a runtime one:

```
python -m pip install -r requirements-dev.txt
python -m pytest tests -o addopts="" -p no:cacheprovider -q     # 1539 tests
```

The commands in this file invoke pip and pytest only through `python -m`, use
forward slashes, and call no shell scripts. On 2026-09-24 UTC, every command
above except `git clone` ran as written in Windows PowerShell 5.1 on Windows
11, with Python 3.12.10, and exited 0. In cmd.exe, remove a line's trailing
`#` comment before running it. On Linux, the repository's CI runs
`python -m instruments.reproduce_table3` on GitHub's `ubuntu-latest` runner.
In each of its runs at the tag's commit on 2026-09-09 UTC, that command
exited 0.

---

## Data availability

This section is the data-availability statement promised in the response to
reviewer comment R1-S9.

### What this repository contains

| Promise | Where it is |
|---|---|
| The full implementation — environment, encodings, control corridor, certification harness | `src/raas_marl/` (34 files: 29 modules and 5 package `__init__.py` files) and `docs/evidence/c1_selectivity_harness.py`. The code builds only the two-cell obstacle patch, one of the two encodings the paper compares. Its `stage25_driver.py` rejects the configuration recorded for `t1_d14_s135`–`s137`. The `RUN_MANIFEST.json` files name 18 commits, none of which is in this repository |
| The per-run configuration records | `results/experiments/<run_id>/RUN_MANIFEST.json` — **all 55 runs** |
| The grading instruments, including the reroute-basin classifier and the grade composer | `instruments/` (three instruments plus a driver) and `docs/evidence/*.py` (five modules) |
| The recorded per-checkpoint grade records | `results/experiments/<run_id>/checkpoint_grades/grade_r{003500,003750,004000}.json` — **27 records**, the nine replication seeds × three checkpoints |
| A README data-availability statement | this section |

Beyond what was promised, the repository also ships the **27 graded policy
checkpoints** (`results/experiments/<run_id>/state_round_{003500,003750,004000}.pt`,
3.35 MB in total, ≤ 125 kB each), so a reader can re-grade from the weights in
place rather than only recompose from the records:

```
python docs/evidence/c1_selectivity_harness.py --grade-run t1_retain_s147 --fork
```

That rolls the shipped policy on all ten fork surfaces, re-runs the own-ablation
causal witness, and returns `SELECTIVE_COMPOSITE` with component counts
`success 8, senses 10, within_ceiling 10, adapt 5, causal 5, witness 4` —
identical to the recorded grade in that run's
`checkpoint_grades/grade_r004000.json`. It needs torch. As of 2026-09-24
UTC, this command takes about 3 s on an Intel Core i7-9700 CPU under Windows 11
with torch 2.5.1+cpu.

### Which runs are included, and their place in the paper

All 55, by run id. Every one carries a complete configuration record; seeds and
the commit each run was launched from are in its `RUN_MANIFEST.json`.

| Group | Runs | Seeds | In the paper |
|---|---|---|---|
| **The nine replication seeds** | `t1_retain_s147`–`s149`, `t1_seedscale_s153`–`s158` | 147–149, 153–158 | Table III, Fig. 4, Fig. 5 — the reliability rates, the intervals, the reroute-basin classification |
| **Table II, two-cell-obstacle-patch arm** | `t1_retain_s147`–`s149` | 147–149 | Table II — the enriched encoding (retention 2/3) |
| **Table II, one-cell-scalar-count arm** | `t1_d15_s138`–`s140` | 138–140 | Table II — the original encoding (retention 0/3). Same mixture and configuration apart from the observation encoding: a single-configuration-variable comparison (Sec. IV). The encoding is not a field of the records. The two arms use different seed sets, as the caption states |
| **Single-pair runs, one-cell scalar count** | `t1_d13_s132`–`s134` | 132–134 | Sec. V-A — the anchor pair alone |
| **Single-pair runs, two-cell obstacle patch** | `anchor_rerun_s144`–`s146` | 144–146 | Sec. V-A — the same pair after the encoding change |
| **The three original-encoding curricula** | `t1_d14_s135`–`s137` (uniform four-pair mixture), `t1_d15_s138`–`s140` (concentrated two-pair mixture, pairs sharing gate row 3), `t1_d16_s141`–`s143` (row-disjoint two-pair mixture) | 135–143 | Sec. V-B — each broke multi-geometry retention on all three seeds |
| **The runs at hazard budget 0.5** | `t1_d4`–`t1_d12` (nine sets of three; `t1_d4` and `t1_d5` trained on `risk_gate_hidden_hazard`) | 105–131 | Sec. V-A reported that the floor armed on five seeds across the corridor's tuning campaign on the single training pair at the original encoding. The analysis records `docs/evidence/t1_d11_analysis.json`, `t1_d12_analysis.json`, and `t1_d13_analysis.json` log that the floor armed on three seeds of this row (`t1_d11_s128`, `t1_d12_s130`, `s131`) and on `t1_d13_s132` and `s133` |
| **The first four runs** | `b101_stoch`, `b102_stoch`, `b103_stoch`, `b104_greedy` | 101–104 | The paper does not mention these runs. They trained on `risk_gate_hidden_hazard`, which has no two-gate wall |

Seeds are drawn in ascending order from a fixed training pool (integers 101–199).
Sec. IV of the paper states that "seeds 150 to 152 were allocated to an
experiment cancelled before launch and never run". This is why the nine-seed
replication is 147–149 and 153–158. No run directory for seeds 150 to 152 exists
here, and that absence is checkable.

### What recomputes from what

The reported reliability rates recompute from the shipped per-checkpoint grade
records and the two other shipped files listed below, via the released
composer. The reported intervals and *p*-value follow from the same shipped
inputs by exact-binomial (Clopper–Pearson) and Fisher computations over the
released counts. Stated precisely, node by node:

```
  results/experiments/*/checkpoint_grades/grade_r*.json      (27 records)
  results/held2_conjuncts.json                               (curve-derived conjuncts)
  docs/evidence/t1_seedscale_analysis.json                   (per-mirror tail hazard)
                    |
                    |   python -m instruments.grade_composer
                    v
  the four n=9 rates  5/9, 4/9, 7/9, 2/9   and the fresh-seed-only 3/6, 2/6, 4/6, 1/6
                    |
                    |   instruments.statistics.clopper_pearson(k, 9)
                    v
  the four exact 95% intervals   [0.212, 0.863] [0.137, 0.788] [0.400, 0.972] [0.028, 0.600]

  docs/evidence/t1_seedscale_analysis.json :: per_seed[*].readiness_lower_outcome
                    |
                    |   python -m instruments.reroute_basin
                    v
  the reroute-basin classification   equivariant {147,153,157,158} / overshoot {149,154} / freeze {148,155,156}

  docs/evidence/t1_seedscale_analysis.json :: per_seed[*].mech_f84_fired  x  the transfer conjunct
                    |
                    |   instruments.statistics.fisher_exact_two_sided(a, b, c, d)
                    v
  the de-confound   2x2 = [[1,1],[1,6]]   p = 0.417
```

Two of the four criteria are conjunctions that draw on both shipped record kinds,
and the composer takes each conjunct from its own source. **Transfer**
recomputes from the per-checkpoint records alone.
**Task success** is a training-curve quantity and uses no checkpoint record.
**Core hold** and **anchor consolidation** each combine a curve-derived conjunct
with a two-of-three witness conjunct derived from the records. The module
docstring of `instruments/grade_composer.py` states this per criterion, and
`--check` audits the recomputation against the registered per-seed values seed by
seed rather than only in aggregate.

### The reserved held-out geometry

The paper says of the gate-row split (Sec. IV):

> a validation set of readiness geometries at gate rows {1, 2} interpolates
> between trained rows 0 and 3; and a held-out set at gate rows {5, 6} is
> deliberately preserved and never instantiated, read, or cited throughout the
> campaign, so that a future held-out evaluation would be meaningful.

Nothing in this repository contradicts that. The points below can be checked
here, except the last:

* **No such scenario exists.** Deriving the gate rows of all ten fork scenarios
  from `grid_environment.py` (the open rows of the wall on column 4) gives
  `{3,4}` (trained), `{1,2}` (validation), `{0,3}`, `{4,7}` and `{0,7}`
  (curriculum). None is `{5,6}`, and no hidden hazard sits on row 5 or 6
  anywhere in the catalog. `stage24_diagnostics.py` records why: the held-out
  fork group "is DEFERRED to the Protocol-v1 lock (I-3), so it is not named here".
* **No run touched one.** Across all 55 `RUN_MANIFEST.json` the only
  scenario-carrying field is `config.scenario_names`, and its union over the 55
  is nine names, none held-out. No manifest records a layout override, so no run
  could have built one under another name.
* **No grade record contains one.** The 27 per-checkpoint records grade exactly
  ten surfaces, grouped `training`/`curriculum`/`readiness` — there is no
  `held_out` group — and the hidden-hazard rows they grade are 0, 1, 2, 3, 4
  and 7.
* **Nothing was read, the authors report.** By their count, the union of every
  scenario token across all 55 probe logs is thirteen names. None is a held-out
  surface. The probe logs are not in this repository, so this point cannot be
  checked here.

What *is* retained, deliberately: the scenario catalog's reservation comments,
the `held_out` group label in the Stage 24-A variant table, and the
`_assert_not_held_out` guard in `docs/evidence/c1_selectivity_harness.py`. The
comments document the claim and do not use it. The `held_out` label and the
guard concern another layout, `risk_gate_heldout_near_gate`: the
`risk_gate_hidden_hazard` map, which has no wall and so no gate rows, with
hidden hazards on rows 2 and 3. The guard raises an error if that layout
reaches the harness. `run_stage24a_variant_readiness_diagnostics` builds it
and runs scripted policies on it. The test suite asserts that no scenario name
in the catalog contains `heldout` or `held_out`. It also asserts that the
output of `run_readiness_probe` omits the layout's name. Gate rows are ordinary
parameters of the environment, so the code can of course express other
geometries; what the claim is about is what was run, and none was.

### The raw training logs

The authors state that the raw episode logs, training curves, and probe logs
are large: **≈ 0.73 GB for the nine replication seeds** (586 MB of episode
logs, 124 MB of curves, 15 MB of probe logs) and **≈ 4.0 GB across all 55
runs**. They are not in this repository.

The authors will archive them in a separate public data record with a
persistent identifier. As of 2026-09-24 UTC, that record does not
exist, so it has no DOI. As of 2026-09-24 UTC, two DOIs identify Zenodo's
archive of this repository at the tag `v1.0-icet2026`:
10.5281/zenodo.22680695 for that version and 10.5281/zenodo.22680694 for all
versions. That archive holds no logs. The authors state that, until the record
exists, the logs are available on request from the corresponding author
(Musadaq Mansoor, `musadaq.mansoor@paf-iast.edu.pk`). When the record is
minted, the authors will add its DOI to `CITATION.cff` and to this section.

Independent re-grading *from the raw logs* — for the witness-gated cells in
particular — needs that record. Every number the paper reported in Table III,
however, recomputes from what is here. No instrument here repeats the lift
search reported in the notes to Table III. Its result is recorded in prose in
`docs/evidence/t1_seedscale_analysis.json`. For seeds 153–158,
`results/held2_conjuncts.json` names `seedscale_mech_summary.json` as its
source. No file of that name is in this repository.
The shipped policy checkpoints allow a
re-grade at the three graded rounds without the record.

Three instruments in `docs/evidence/` point to the "Data availability" section
of this README for "the record's location". Each looks for a run's files in
`results/experiments/<run_id>/`. Without the record:

* `held2_bar.py` passes its `--validate` checks but reports them as degraded,
  because it skips its three real-curve regressions. Its `--grade-run` stops
  with exit status 1 for want of the run's `training_curve.jsonl`. No run's
  `training_curve.jsonl` is in this repository.
* `drh5_trigger_regression.py` replays four runs from their
  `training_curve.jsonl`: `t1_d11_s128`, `t1_d12_s130`, `t1_d12_s131`, and
  `t1_d12_s129`. It stops with exit status 1 at the first, for want of
  `results/experiments/t1_d11_s128/training_curve.jsonl`.
* `c1_selectivity_harness.py` completes its `--grade-run` (with `--fork`) on
  each of the nine replication seeds, whose checkpoints ship here. For a run
  without a checkpoint, it stops with exit status 1 and reports "no policy
  checkpoint found".

The messages of `held2_bar.py` and `drh5_trigger_regression.py` say that the
code repository ships only `results/experiments/<run>/RUN_MANIFEST.json`. For
each of the nine replication seeds, this repository also ships its three graded
policy checkpoints and their per-checkpoint grade records.

---

## Contents

| Path | What it is |
|---|---|
| `instruments/` | The three release instruments — the reroute-basin classifier, the grade composer, and the Clopper–Pearson/Fisher statistics — plus `reproduce_table3.py`, which runs all three. Standard library only. |
| `src/raas_marl/` | The package: environment, enriched encoding, MAPPO-Lagrangian core, control corridor, training drivers. |
| `docs/evidence/*.py` | The five certification and regression instruments, at their source-repository paths. |
| `docs/evidence/*_analysis.json` | The graded per-unit run analyses, one per experimental unit. |
| `results/experiments/*/RUN_MANIFEST.json` | Per-run provenance and configuration for all 55 runs. |
| `results/experiments/*/checkpoint_grades/` | The 27 per-checkpoint grade records (9 seeds × rounds 3500/3750/4000). |
| `results/experiments/*/state_round_*.pt` | The 27 graded policy checkpoints, for the same 9 × 3. |
| `results/held2_conjuncts.json` | The curve-derived conjuncts of the task-success and core-hold criteria. |
| `results/checkpoint_grades_manifest.json`, `results/checkpoints_manifest.json` | Byte sizes and SHA-256 for the 27 per-checkpoint grade records and the 27 graded checkpoints, and the source filename of each grade record. |
| `CLAIMS.yaml` | The traceability register: 117 registered paper-claim entries, 108 of which give a `source_file` that resolves to a file here. Paths are in source-repository coordinates. Each cited file that is shipped keeps its path, but 27 of the 38 file:line citations into shipped files point at different text here. |
| `docs/paper/figure_data/` | Backing data and its generator, for Fig. 1. |
| `tests/` | The test suite, shipped whole and unmodified (1539 tests). |
| `scripts/check_provenance.py` | A check that each of the five `docs/evidence/*.py` instruments loads this repository's code and not some other tree's. |

## What is not here

* **The raw training logs and the non-graded checkpoints.** The data availability
  section above covers the raw logs. Only the 27 graded policy checkpoints are here.
* **Anything touching the reserved held-out geometry** — see above.
* **A DOI for the data record.** As of 2026-09-24 UTC, none has been minted, so none is claimed.
* **The decision and evidence dossiers** (`docs/decisions/**`, `docs/evidence/ED-*.md`,
  the countersign records). These are the campaign's internal deliberation. None
  of them is in this repository. As of 2026-09-24 UTC, the source repository that
  holds them (including 31 decision records and 24 dossiers) is not public. `CLAIMS.yaml` cites one of them, `ED-basin-lever-search.md`, as
  the source for a single introduction claim, and that entry says so in its own
  `criterion_note`: the claim "is NOT verifiable from the public artifact". The
  other 116 claim entries comprise 108 whose `source_file` resolves to a file here and
  8 that give `source_file: NONE`. Those 8 include 6 whose `derived_from` field names
  a file here.
* **`docs/paper/template/main.tex` and `docs/paper/VENUE_KIT.md`.** `CITATION.cff` and
  `pyproject.toml` name them as where their values were read from; they live in the
  source repository, and both comments now say so.

---

## Reproduction environment

The recorded values in the Software, Threading, and Hardware tables below were
read from the `runtime_record` block of the
**nine graded** run manifests — `t1_retain_s147`, `s148`, `s149`,
`t1_seedscale_s153`, `s154`, `s155`, `s156`, `s157`, `s158` — which are shipped
in this repository under `results/experiments/`. Check any of them yourself.

### Software

| | Recorded value | Agreement across the nine |
|---|---|---|
| Python | `3.12.10` (`tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36`, `MSC v.1943 64 bit (AMD64)`) | identical in all nine |
| PyTorch | `2.5.1+cpu` | identical in all nine |
| Device | `cpu` | identical in all nine |
| Tensor dtype | `float32` | identical in all nine |

`requires-python` is declared as `>=3.12` in `pyproject.toml` — the minor
version that was actually exercised. The exact patch is above. `requirements.txt`
pins `torch==2.5.1`, the version that was run; `pyproject.toml` declares the
looser floor `torch>=2.5.1` for installing the package.

`torch` is the only third-party runtime dependency. An AST census over all 48
shipped Python files finds four non-standard-library top-level import names, of
which only two are external: `raas_marl` (this package importing itself) and
`c1_selectivity_harness` (a grading instrument imported as a sibling module by
two others, and shipped alongside them) are internal; `torch` (22 files) and
`pytest` (1 file, the test suite) are the external ones. `pytest` is therefore
declared only under `[project.optional-dependencies].dev`. The `instruments/`
modules add no dependency at all: CI asserts by AST that they import nothing
outside the standard library and one another.

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

So three runs ran simultaneously, each with two intra-op
threads. A single run executing alone on the same hardware would plausibly be
somewhat faster than its recorded time rather than slower; conversely, reproducing the full
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

In the research repository, `src/` is identical at the three commits:
the code did not move between the three launches.

**Twenty-nine of the 34 shipped `src/` files are byte-identical to that code.**
The other five —

    raas_marl/environments/active_sensing/grid_environment.py
    raas_marl/environments/active_sensing/tensor_adapter.py
    raas_marl/mappo_lagrangian/buffer.py
    raas_marl/mappo_lagrangian/config.py
    raas_marl/mappo_lagrangian/losses.py

— carry **additions made for this release**. `grid_environment.py` gains one
docstring. The other four import `torch` lazily. Each gains an
`if TYPE_CHECKING:` guard, with its `from typing import TYPE_CHECKING` import,
so a type checker can resolve `torch`. Two of them, `tensor_adapter.py` and
`config.py`, also gain return annotations. At run time `TYPE_CHECKING` is
`False`, so the body of each guard never runs. Every one of the
five declares `from __future__ import annotations`, so annotations are deferred
strings and are never evaluated at run time; with docstrings, the guards, their
`from typing import TYPE_CHECKING` imports, and the annotations removed, the
executable AST of all five is identical to the graded
code. Nothing was renamed, reordered or rewritten.

That is a weaker statement than "byte-identical" and it is the true one. The three
commits above are not in this repository. As of 2026-09-24 UTC, the research
repository is not public, so a reader cannot repeat these comparisons.

---

## License

The software file-sets that `LICENSE` lists are MIT. Data and records (the analysis JSONs, everything
under `results/`, the Fig. 1 backing data, and `CLAIMS.yaml`) are CC BY 4.0
(`LICENSES/CC-BY-4.0.txt`). `LICENSE` states the split file-set by file-set.
`LICENSE` lists no file-set under `instruments/` or `scripts/`.

## Citation

See `CITATION.cff`.

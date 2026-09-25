# Observational Aliasing Bounds Generalization in Constrained Active-Sensing Multi-Agent Reinforcement Learning

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22680694.svg)](https://doi.org/10.5281/zenodo.22680694) [![License: MIT and CC BY 4.0](https://img.shields.io/badge/license-MIT%20%2F%20CC%20BY%204.0-blue)](#license) [![CI](https://github.com/Umair-Waseem/observational-aliasing-marl/actions/workflows/ci.yml/badge.svg)](https://github.com/Umair-Waseem/observational-aliasing-marl/actions/workflows/ci.yml)

This repository holds the code, grading instruments, run provenance,
per-checkpoint grade records, the 27 graded policy checkpoints, and a
claim-to-source traceability register that accompany the IEEE ICET 2026 paper
of the same title. As of 2026-09-24 UTC,
the paper is accepted and not yet published. The paper reported
existence-and-boundary evidence, not a reliability claim. Its target behavior is
"a conjunction: acquire information selectively, adapt movement to what is
revealed, reduce hazard exposure, and preserve task success." Its central
finding is that the barrier to transferring the behavior across grid geometries is a
property of the observation encoding. Under the original encoding, two scenarios
present a byte-identical observation at one decision cell yet require opposite
reroutes (Fig. 1). Widening only that encoding turned a training mixture that
failed on all three seeds into one that recovered the behavior on two of three
(Table II, retention 2/3). Under the enriched encoding, the method can produce a
policy satisfying all four components of that behavior. Such a policy was
verified causally and is selectable on a validation geometry. On two seeds, the
behavior transferred to a never-trained validation geometry. The method does not
do so reliably: at replication, the core selective-sensing conjunct held on four
of nine seeds and transfer on two of nine, indistinguishable from chance. The paper made no
evaluation on the held-out geometry, which is deliberately preserved.

The state reported in the paper is the tag [`v1.0-icet2026`](https://github.com/Umair-Waseem/observational-aliasing-marl/releases/tag/v1.0-icet2026).
This branch differs from the tag only in `.github/workflows/ci.yml`, `README.md`, `CITATION.cff`, `docs/PAPER_MAP.md`, `pyproject.toml`, `CLAIMS.yaml`, `LICENSE`, `docs/paper/figure_data/generate_fig5.py`, and `requirements-figures.txt`. Apart from `CLAIMS.yaml`, no instrument, record, manifest, or checkpoint differs from the tag, and no number the paper reported differs.

[Citation](#citation) · [Reproduce the numbers in Table III](#reproduce-the-numbers-in-table-iii) · [Install and tests](#install-and-tests) · [The paper map](#the-paper-map) · [Data availability](#data-availability) · [What is here, by directory](#what-is-here-by-directory) · [What is not here and why](#what-is-not-here-and-why) · [Reproduction environment](#reproduction-environment) · [Provenance](#provenance) · [License](#license)

## Citation

To cite the paper, use this entry. Its `note` marks the paper as accepted and
not yet published. Pages and the IEEE Xplore DOI follow publication.

```bibtex
@inproceedings{waseem2026observational,
  title     = {Observational Aliasing Bounds Generalization in Constrained
               Active-Sensing Multi-Agent Reinforcement Learning},
  author    = {Waseem, Muhammad Umair and Abdullah and Khan, Shayan and Mansoor, Musadaq},
  booktitle = {2026 21st International Conference on Emerging Technologies (ICET)},
  year      = {2026},
  note      = {Accepted, not yet published}
}
```

See `CITATION.cff`. Its `preferred-citation` gives the same title, authors,
year, and proceedings title, with `status: in-press`.

As of 2026-09-24 UTC, the version DOI
[10.5281/zenodo.22680695](https://doi.org/10.5281/zenodo.22680695) identifies
Zenodo's archive of the tag `v1.0-icet2026`, the state reported in the paper.
The concept DOI [10.5281/zenodo.22680694](https://doi.org/10.5281/zenodo.22680694)
identifies all versions of that archive. `CITATION.cff` gives the version DOI in
its `doi` field. As of 2026-09-25 UTC, GitHub marks the release of `v1.0-icet2026` immutable, so the tag is locked to its commit.

## Reproduce the numbers in Table III

This needs git and Python. `pyproject.toml` declares Python 3.12 or newer.
The runs of `instruments.reproduce_table3` reported below used Python 3.12.
The three release instruments are standard library only, so this needs no packages:

```shell
git clone https://github.com/Umair-Waseem/observational-aliasing-marl
cd observational-aliasing-marl
git checkout v1.0-icet2026
python -m instruments.reproduce_table3
```

That recomputes, from the records in this repository, the n=9 column of
Table III of the paper: the four nine-seed reliability rates, their exact 95%
Clopper–Pearson intervals, the fresh-seed-only rates, and the reroute-basin
classification. It also recomputes the Fisher p of the arm-fire de-confound,
reported in the notes to Table III. It does not print the n=3 column. It
prints the paper's value beside the reroute-basin classification and the
Fisher p. It exits non-zero if any recomputed value stops matching the paper,
so it is usable as a pass/fail check. A pass is exit status 0 and the line
"All reproduced values match the camera-ready." Its output is:

```text
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

```shell
python -m instruments.grade_composer --check    # per-seed composition + registry audit
python -m instruments.reroute_basin             # the reroute-basin classification, seed by seed
python -c "from instruments.statistics import clopper_pearson; print(clopper_pearson(2, 9))"
```

On 2026-09-24 UTC, the module `instruments.reproduce_table3` took about 0.13 s
on Windows 11 with Python 3.12.10 and no packages installed, on an Intel Core
i7-9700. On Linux, the repository's CI runs the same module on GitHub's
`ubuntu-latest` runner. In each of its runs at the tag's commit on 2026-09-09
UTC, that module exited 0. This file records no run on macOS.

The CI badge above shows the result of the latest run of
`.github/workflows/ci.yml` on the default branch. Three of its steps run with
`continue-on-error`, so their failure does not fail the run:

* "Type-check (mypy)"
* "Provenance gate (every instrument loads from THIS repository)"
* "Grading instruments (self-contained modes only)"

A failure in any of its other eight steps, including "Test" and "Reproduce Table III from
the shipped records", fails the run. In the runs at the tag's commit on
2026-09-09 UTC, the type-check step exited 1 and the runs still passed.

## Install and tests

To run the environment, the training drivers, or a re-grade from the shipped
policy checkpoints, install PyTorch at the recorded version and the package
itself:

```shell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` carries its own `--index-url` for the PyTorch CPU wheel.
As of 2026-09-24 UTC, that index has a `torch` 2.5.1 wheel for each platform and
Python version marked "yes" below:

| Platform | Python 3.12 | Python 3.13 | Python 3.14 |
|---|---|---|---|
| Windows x86_64 | yes | no | no |
| Windows ARM64 | no | no | no |
| Linux x86_64 | yes | yes | no |
| Linux aarch64 | yes | no | no |
| macOS arm64 | yes | no | no |
| macOS x86_64 | no | no | no |

As of 2026-09-25 UTC, on a combination marked "no", the `requirements.txt`
install above stops with "ERROR: No matching distribution found for torch==2.5.1".

As of 2026-09-25 UTC, on Linux x86_64, the default PyPI wheel of `torch` 2.5.1 is the
CUDA build. That wheel and the GPU packages it requires are about 3 GB to
download. The recorded build is CPU (`2.5.1+cpu`). Install
editable (`-e`). `tests/conftest.py` and four of the five `docs/evidence`
instruments put `<repo>/src` on `sys.path` themselves. A non-editable install
would put a second copy of `raas_marl` in `site-packages` alongside it.

To run the test suite, install the requirements in `requirements-dev.txt`, which
includes `requirements.txt` and adds the development tools pytest, ruff, and mypy:

```shell
python -m pip install -r requirements-dev.txt
python -m pytest tests -o addopts="" -p no:cacheprovider -q     # 1539 tests
```

On 2026-09-24 UTC, on Windows 11 with Python 3.12.10 on an Intel Core i7-9700,
the four `pip` commands above took 236 s and 279 s in two fresh clones, with no
pip cache. The largest download was the `torch` 2.5.1+cpu wheel, at 205.4 MB.
The test command then reported "1539 passed, 1 warning" in about 38 s. The
warning comes from `torch`: NumPy is not installed. Neither `requirements.txt`
nor `requirements-dev.txt` installs it.

The commands in this file invoke pip and pytest only through `python -m`, use
forward slashes, and call no shell scripts. Every command that runs Python calls
it `python`. Run the pip commands with a `python` that accepts installs, such as
the one in an activated virtual environment. On 2026-09-25 UTC, every command
above except `git clone` ran as written in Windows PowerShell 5.1 on Windows
11, with Python 3.12.10, and exited 0. In cmd.exe, remove a line's trailing
`#` comment before running it.

To draw Fig. 5 from the shipped records, install matplotlib from
`requirements-figures.txt` and run the generator. It needs neither PyTorch nor
the package:

```shell
python -m pip install -r requirements-figures.txt
python docs/paper/figure_data/generate_fig5.py fig5.pdf
```

It prints the four rows it draws, then the file it wrote:

```text
[
 {
  "label": "Task success\n(joint with hazard)",
  "k": 5,
  "n": 9,
  "lo": 0.212008506778868,
  "hi": 0.8630043377348333
 },
 {
  "label": "Selective-sensing\nhold (core)",
  "k": 4,
  "n": 9,
  "lo": 0.13699566226516657,
  "hi": 0.7879914932211318
 },
 {
  "label": "Anchor consolidation\n(load-bearing)",
  "k": 7,
  "n": 9,
  "lo": 0.399906426283688,
  "hi": 0.9718550265221018
 },
 {
  "label": "Transfer to\nvalidation",
  "k": 2,
  "n": 9,
  "lo": 0.02814497347789824,
  "hi": 0.6000935737163118
 }
]
wrote fig5.pdf
```

On 2026-09-25 UTC, on Windows 11 with Python 3.12.10 and matplotlib 3.11.1, the
written `fig5.pdf` had SHA-256
`2201bd8ae85f5c0301ab2b513a6812d16bf9108022e611c3a78fc4b3076cbcb2`. The paper's
source build includes a Fig. 5 file with the same SHA-256.

## The paper map

Each row gives a value as the paper prints it, or describes the item, with its
page, and the command
that prints it or the file that holds it. The last column names the platform
on which the command was run, or says "opened" where the value was read from a
file. Dates are UTC. Where no command prints a value, the row says so, or its last column says
"opened". The rows
are grouped in this order: Table I's hazard budget, Table II, Table III and its
notes, Fig. 3, the compute line of Sec. IV, and the certification witness.
The output shown under [Reproduce the numbers in Table III](#reproduce-the-numbers-in-table-iii)
holds Table III's n=9 values. [`docs/PAPER_MAP.md`](docs/PAPER_MAP.md) has all
51 rows, in seven groups (Table I, Table II, Table III, the figures, the
headline numbers, compute, and the certification witness), with a row for each
Table III value.

| Paper item | Value in paper | Command or file | Verified (platform, date) |
|---|---|---|---|
| Hazard budget, p. 4 | 0.25 | `config.hazard_budget` is 0.25 in 24 of the 55 records, including the nine replication records. `config.stage25_update_config.hazard_budget` is 0.5 in the 27 records of `t1_d4` to `t1_d12`. The four records of `b101_stoch`, `b102_stoch`, `b103_stoch`, and `b104_greedy` have neither field. | opened, 2026-09-24 |
| one-cell scalar count, Retention, p. 5 | 0/3 | In `docs/evidence/t1_d15_analysis.json`, `RETAIN.verdict.bar_c4_retain_pass_count` is `0` for `t1_d15_s138` to `s140`. No command prints this cell. The training curves it was graded from are not in this repository. | opened, 2026-09-24 |
| two-cell obstacle patch, Retention, p. 5 | 2/3 | `python -m instruments.grade_composer`: of the rows for seeds 147, 148, and 149, the `criteria met` column lists `task_success` for 147 and 149. No command prints this tally. `docs/evidence/t1_retain_analysis.json` records `2/3` in `tallies.bar_c4_retain`. `grade_composer` takes `task_success` from the `bar_c4_pass` field of `results/held2_conjuncts.json`. The training curves `bar_c4_pass` was graded from are not in this repository. | Windows 11, Python 3.12.10, 2026-09-24; counted from the per-seed rows |
| Transfer to validation, n=3, p. 6 | 1/3 | `python -m instruments.grade_composer`: of the rows for seeds 147, 148, and 149, the `criteria met` column lists `transfer` for 147 only. No command prints the n=3 tally. `docs/evidence/t1_retain_analysis.json` records `1/3 (s147 only, 2/3 checkpoints)` in `tallies.c4_gen_2of3_seeds`. | Windows 11, Python 3.12.10, 2026-09-24; counted from the per-seed rows |
| Transfer to validation, n=9, p. 6 | 2/9 = 0.22 | `python -m instruments.reproduce_table3` prints `2/9 = 0.22` on the `Transfer to validation` line. | Windows 11, Python 3.12.10, 2026-09-24 |
| Notes to Table III, arm-fire de-confound, p. 6 | Fisher p = 0.417 | `python -m instruments.reproduce_table3` prints `2x2 = [[1, 1], [1, 6]]` and `p = 0.4167`. `python -c "from instruments.statistics import fisher_exact_two_sided; print(fisher_exact_two_sided(1, 1, 1, 6))"` prints `0.4166666666666667`. | Windows 11, Python 3.12.10, 2026-09-24 |
| Notes to Table III, lift search, p. 6 | "the strongest lift (single- and paired-feature search) does not survive multiple-comparison correction" | In `docs/evidence/t1_seedscale_analysis.json`, `step3_fork.adversarial_confirmation` records the search's result in prose. No command here repeats the search. | opened, 2026-09-24 |
| Fig. 3, applied constraint multiplier λ per round, p. 5 | the nine replication seeds over training rounds; seeds that hold the joint task-success bar solid, the rest dashed; dotted guides at the armed floor 3.0 and the ceiling 5.0; "on the holding seeds the tail multiplier rests at the armed floor of 3.0 with the dual integral at zero" | No generator is included. The solid seeds are those whose `bar_c4_pass` is `true` in `results/held2_conjuncts.json`: 147, 149, 153, 156, and 158. Their checkpoints check the quoted caption claim only in part. In the 15 checkpoints of those five seeds (`state_round_003500.pt`, `state_round_003750.pt`, and `state_round_004000.pt`), `controller.integral` is 0.0 and `arm_state.armed` is `true`. The checkpoints do not store the applied multiplier, so the claimed value 3.0 is not checked here. | opened, 2026-09-24; per-round values are in `training_curve.jsonl` files, which are not in this repository |
| Sec. IV, wall-clock time, p. 4 | "2.4 to 3.5 h of wall-clock time (median 2.8 h)" | `finish_timestamp_utc` minus `launch_timestamp_utc` in the nine replication `RUN_MANIFEST.json` files gives a minimum of 2.3758 h (`t1_seedscale_s158`), a maximum of 3.4989 h (`t1_retain_s147`), and a median of 2.7844 h. No field records how many runs ran at once. | opened, 2026-09-24 |

| Paper item | Value in paper | Command or file | Verified (platform, date) |
|---|---|---|---|
| Sec. III, the counterfactual own-ablation witness, on one seed, pp. 3, 4 | "counterfactual own-ablation witness": the policy is re-rolled "with its revealed-information channel zeroed for the whole episode", and the witness requires "hazard exposure to rise" | `python docs/evidence/c1_selectivity_harness.py --grade-run t1_retain_s147 --fork` (needs PyTorch) grades the shipped checkpoint `state_round_004000.pt` and prints the verdict `SELECTIVE_COMPOSITE`. Its result for the run equals the `grade` object in `results/experiments/t1_retain_s147/checkpoint_grades/grade_r004000.json`, except that it adds `state_file` and its `label` lacks `@r4000`. | Windows 11, Python 3.12.10, 2026-09-24 |

## Data availability

This section is the data-availability statement promised in the response to
reviewer comment R1-S9. As of 2026-09-25 UTC, that response is not public, the authors report. Text
in quotation marks in the first column below is quoted from it.

### What this repository contains

| Promise | Where it is | Status as of 2026-09-24 UTC |
|---|---|---|
| The full implementation: environment, encodings, control corridor, certification harness | `src/raas_marl/` (34 files: 29 modules and 5 package `__init__.py` files) and `docs/evidence/c1_selectivity_harness.py`. The code builds only the two-cell obstacle patch, one of the two encodings the paper compares. Its `stage25_driver.py` rejects the configuration recorded for `t1_d14_s135`–`s137`. The `RUN_MANIFEST.json` files name 18 commits, none of which is in this repository. | Partly delivered |
| The per-run configuration records | `results/experiments/<run_id>/RUN_MANIFEST.json`: all 55 runs | Delivered |
| "Every run ships a complete configuration record" | Each of the 55 `RUN_MANIFEST.json` files holds its run's configuration. No record names the observation encoding, the one variable of Table II's single-configuration-variable comparison. The next subsection names the older records that lack fields the shipped `Stage25RunConfig` defines. | Partly delivered |
| The grading instruments, including the reroute-basin classifier and the grade composer | `instruments/` (three instruments plus a driver) and `docs/evidence/*.py` (five modules) | Delivered |
| "the reroute-basin classifier, validated to reproduce the reported equivariant/overshoot/freeze partition" | `instruments/reroute_basin.py`, which compares its classification of the nine replication seeds with the reported partition. Its output ends with "MATCHES the reported partition." | Delivered |

| Promise | Where it is | Status as of 2026-09-24 UTC |
|---|---|---|
| The recorded per-checkpoint grade records | `results/experiments/<run_id>/checkpoint_grades/grade_r{003500,003750,004000}.json`: 27 records, the nine replication seeds × three checkpoints | Delivered |
| A README data-availability statement | this section | Delivered |
| "The raw training logs, which are large, will be archived in a separate public data record with a persistent identifier" | As of 2026-09-24 UTC, no such record exists. See [The raw training logs](#the-raw-training-logs). | Not delivered |
| "otherwise the repository README carries the record’s location" | This README gives no location for the record. | Not delivered |
| "The reported reliability rates recompute from the shipped per-checkpoint grade records via the released composer" | `instruments/grade_composer.py` recomputes them from those records together with `results/held2_conjuncts.json` and `docs/evidence/t1_seedscale_analysis.json`. Only transfer recomputes from the grade records alone. See [What recomputes from what](#what-recomputes-from-what). | Partly delivered |
| "the reported intervals and p-values follow from those records" | They follow by Clopper–Pearson and Fisher computations over counts that also draw on the two files named in the row above. | Partly delivered |

Beyond what was promised, the repository also ships the 27 graded policy
checkpoints (`results/experiments/<run_id>/state_round_{003500,003750,004000}.pt`,
3.35 MB in total, ≤ 125 kB each). With them, a reader can re-grade from the
weights in place rather than only recompose from the records:

```shell
python docs/evidence/c1_selectivity_harness.py --grade-run t1_retain_s147 --fork
```

That rolls the shipped policy on all ten fork surfaces, re-runs the own-ablation
causal witness, and returns `SELECTIVE_COMPOSITE`. Its result equals the
`grade` object in that run's `checkpoint_grades/grade_r004000.json`, except that
it adds `state_file` and its `label` lacks `@r4000`. The equal fields include
`component_pass_counts`: `success` 8, `senses` 10, `within_ceiling` 10,
`adapt` 5, `causal` 5, and `witness` 4. It needs PyTorch. On 2026-09-24 UTC,
this command took about 3 s on an Intel Core i7-9700 under Windows 11, with
Python 3.12.10 and `torch` 2.5.1+cpu.

### Which runs are included, and their place in the paper

All 55 runs are included, grouped below by their place in the paper. Each has a `RUN_MANIFEST.json` that holds its configuration,
its seed, and, in `git_head`, a commit hash. `config_sha256` is present in 51 of
the records and holds the hash that the launching driver computed from the
configuration. Each of the 51 hashes recomputes from its record by the rule in
the shipped `stage25_driver.py`. The records of `t1_d4`–`t1_d12` lack fields
that the shipped `Stage25RunConfig` defines. The six records marked `resumed`
(`t1_d11_s126`–`s128`, `t1_d16_s141`–`s143`) hold in `git_head` the commit
checked out at the last resume. None of these six `RUN_MANIFEST.json` files
records the launch commit.

`t1_retain_s147`–`s149` and `t1_d15_s138`–`s140` each appear in two groups
of the table below.

| Group | Runs | Seeds | In the paper |
|---|---|---|---|
| The nine replication seeds | `t1_retain_s147`–`s149`, `t1_seedscale_s153`–`s158` | 147–149, 153–158 | Table II (anchor consolidation, n=9: 7/9), Table III, Fig. 3, Fig. 4, Fig. 5: the reliability rates, the intervals, the reroute-basin classification, and the applied multiplier per round |
| Table II, two-cell-obstacle-patch arm | `t1_retain_s147`–`s149` | 147–149 | Table II: the enriched encoding (retention 2/3) |
| Table II, one-cell-scalar-count arm | `t1_d15_s138`–`s140` | 138–140 | Table II: the original encoding (retention 0/3). Same mixture and configuration apart from the observation encoding: a single-configuration-variable comparison (Sec. IV). The encoding is not a field of the records. The two arms use different seed sets, as the caption states. |
| Single-pair runs, one-cell scalar count | `t1_d13_s132`–`s134` | 132–134 | Sec. V-A: the anchor pair alone |
| Single-pair runs, two-cell obstacle patch | `anchor_rerun_s144`–`s146` | 144–146 | Sec. V-A: the same pair after the encoding change |
| The three original-encoding curricula | `t1_d14_s135`–`s137` (uniform four-pair mixture), `t1_d15_s138`–`s140` (concentrated two-pair mixture, pairs sharing gate row 3), `t1_d16_s141`–`s143` (row-disjoint two-pair mixture) | 135–143 | Sec. V-B: each broke multi-geometry retention on all three seeds |
| The runs at hazard budget 0.5 | `t1_d4`–`t1_d12` (nine sets of three; `t1_d4` and `t1_d5` trained on `risk_gate_hidden_hazard`) | 105–131 | Sec. V-A reported that the floor armed on five seeds across the corridor's tuning campaign on the single training pair at the original encoding. Three analysis files, `docs/evidence/t1_d11_analysis.json`, `t1_d12_analysis.json`, and `t1_d13_analysis.json`, log that the floor armed on three seeds of this row (`t1_d11_s128`, `t1_d12_s130`, `s131`) and on `t1_d13_s132` and `s133`. |
| The first four runs | `b101_stoch`, `b102_stoch`, `b103_stoch`, `b104_greedy` | 101–104 | The paper does not mention these runs. They trained on `risk_gate_hidden_hazard`, which has no two-gate wall. Their analysis record is `docs/evidence/baseline_t1_analysis.json`. |

Seeds are drawn in ascending order from a fixed training pool (integers 101–199).
Sec. IV of the paper states that "seeds 150 to 152 were allocated to an
experiment cancelled before launch and never run". This is why the nine-seed
replication is 147–149 and 153–158. No run directory for seeds 150 to 152 exists
here.

### What recomputes from what

The reported reliability rates recompute from the shipped per-checkpoint grade
records and the two other shipped files listed below, via the released
composer. The reported intervals and p-value follow from the same shipped
inputs by exact-binomial (Clopper–Pearson) and Fisher computations over the
released counts. The diagram below traces each input to its output:

```text
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
  the reroute basins   equivariant {147,153,157,158} / overshoot {149,154} / freeze {148,155,156}

  docs/evidence/t1_seedscale_analysis.json :: per_seed[*].mech_f84_fired  x  the transfer conjunct
                    |
                    |   instruments.statistics.fisher_exact_two_sided(a, b, c, d)
                    v
  the de-confound   2x2 = [[1,1],[1,6]]   p = 0.417
```

Two of the four criteria are conjunctions that draw on both the per-checkpoint grade records and a curve-derived file.
The composer takes each conjunct from its own source. **Transfer**
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

None of the four points below contradicts that statement. The first three can
be checked here, and the last cannot:

* No such scenario exists. Deriving the gate rows of all ten fork scenarios
  from `grid_environment.py` (the open rows of the wall on column 4) gives
  {3, 4} (the anchor pair), {1, 2} (validation), and three curriculum pairs: {0, 3},
  {4, 7}, and {0, 7}. None is {5, 6}, and no hidden hazard sits on row 5 or 6
  anywhere in the catalog. `stage24_diagnostics.py` records why: the held-out
  fork group "is DEFERRED to the Protocol-v1 lock (I-3), so it is not named here".
* No run touched one. Across all 55 `RUN_MANIFEST.json` files, the only
  scenario-carrying field is `config.scenario_names`, and its union over the 55
  is nine names, none held-out. No manifest records a layout override. The
  readiness probe in `baseline_driver.py` also builds layouts that no manifest
  records. On this branch, its code admits only the `training` and `readiness`
  variant groups.
* No grade record contains one. The 27 per-checkpoint records grade exactly
  ten surfaces, grouped `training`, `curriculum`, and `readiness`. There is no
  `held_out` group. The hidden-hazard rows they grade are 0, 1, 2, 3, 4, and 7.
* Nothing was read, the authors report. By their count, the union of every
  scenario token across all 55 probe logs is thirteen names. None is a held-out
  surface. The probe logs are not in this repository, so this point cannot be
  checked here.

This repository retains the scenario catalog's reservation comments, the
`held_out` group label in the Stage 24-A variant table, and the
`_assert_not_held_out` guard in `docs/evidence/c1_selectivity_harness.py`. The
comments restate the reservation of gate rows 5 and 6. No fork scenario in the
catalog has a gate on either row. The `held_out` label and the guard concern
another layout, `risk_gate_heldout_near_gate`. That layout places its hidden
hazards on rows 2 and 3 of the `risk_gate_hidden_hazard` map, which has no wall
and so no gate rows. The guard raises an error if that layout
reaches the harness. `run_stage24a_variant_readiness_diagnostics` builds it
and runs scripted policies on it. The test suite asserts that no scenario name
in the catalog contains `heldout` or `held_out`. It also asserts that the
output of `run_readiness_probe` omits the layout's name. Gate rows are ordinary
parameters of the environment, so the code can express other geometries. The
claim is about what was run, and no run recorded here used a geometry with gate
rows 5 and 6.

### The raw training logs

The authors state that the raw episode logs, training curves, and probe logs
are large: ≈ 0.73 GB for the nine replication seeds (586 MB of episode
logs, 124 MB of curves, 15 MB of probe logs) and ≈ 4.0 GB across all 55
runs. They are not in this repository.

The response to reviewer comment R1-S9 states that they "will be archived in
a separate public data record with a persistent identifier". As of 2026-09-24
UTC, that record does not exist, so it has no DOI. As of 2026-09-24 UTC, two
DOIs identify Zenodo's archive of this repository at the tag `v1.0-icet2026`:
10.5281/zenodo.22680695 for that version and 10.5281/zenodo.22680694 for all
versions. That archive holds no logs.

Independent re-grading from the raw logs, for the witness-gated cells in
particular, needs that record. Every number the paper reported in Table III,
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

Three self-contained checks need no record and exit with status 0:
`r2_null_rejection_check.py`, `c1_tightened_adversarial_test.py`, and
`c1_selectivity_harness.py` with `--validate --fork`.

The messages of `held2_bar.py` and `drh5_trigger_regression.py` say that the
code repository ships only `results/experiments/<run>/RUN_MANIFEST.json`. For
each of the nine replication seeds, this repository also ships its three graded
policy checkpoints and their per-checkpoint grade records.

## What is here, by directory

| Path | What it is |
|---|---|
| `instruments/` | The three release instruments (the reroute-basin classifier, the grade composer, and the Clopper–Pearson/Fisher statistics), plus `reproduce_table3.py`, which runs all three. Standard library only. |
| `src/raas_marl/` | The package: environment, enriched encoding, MAPPO-Lagrangian core, control corridor, training drivers. |
| `docs/evidence/*.py` | The five certification and regression instruments, at their source-repository paths. |
| `docs/evidence/*_analysis.json` | The graded per-unit run analyses, one per experimental unit. |
| `results/experiments/*/RUN_MANIFEST.json` | Per-run provenance and configuration for all 55 runs. |
| `results/experiments/*/checkpoint_grades/` | The 27 per-checkpoint grade records (9 seeds × rounds 3500/3750/4000). |
| `results/experiments/*/state_round_*.pt` | The 27 graded policy checkpoints, for the same 9 × 3. |
| `results/held2_conjuncts.json` | The curve-derived conjuncts of the task-success and core-hold criteria. |
| `results/checkpoint_grades_manifest.json`, `results/checkpoints_manifest.json` | Byte sizes and SHA-256 for the 27 per-checkpoint grade records and the 27 graded checkpoints, and the source filename of each grade record. |
| `CLAIMS.yaml` | The traceability register: 117 registered paper-claim entries, 109 of which give a `source_file` that resolves to a file here. Each cited file that is shipped keeps its source-repository path, apart from `instruments/reproduce_table3.py` and `instruments/statistics.py`, which the source repository does not have. Its 38 file:line citations into shipped files use this repository's line numbers. Its `main.tex` line numbers refer to `docs/paper/template/main.tex`, which is not in this repository. |
| `docs/paper/figure_data/` | Backing data and its generator, for Fig. 1, and a generator for Fig. 5. |
| `tests/` | The test suite (1539 tests). The authors state that its two files are identical to those in the source repository, which, as of 2026-09-24 UTC, is not public. |
| `scripts/check_provenance.py` | A check that each of the five `docs/evidence/*.py` instruments loads this repository's code and not some other tree's. |

## What is not here and why

* **The raw training logs and the non-graded checkpoints.** The data availability
  section above covers the raw logs. Only the 27 graded policy checkpoints are here.
* **Anything touching the reserved held-out geometry.** The paper reported that
  the held-out set is deliberately preserved and never instantiated, read, or
  cited, so that a future held-out evaluation would be meaningful (Sec. IV).
  See [The reserved held-out geometry](#the-reserved-held-out-geometry).
* **Code to re-run the original-encoding runs.** The shipped code builds only
  the two-cell obstacle patch. Asked for an observation radius of 1, its actor
  encoding raises a `ValueError`. The shipped driver accepts the configuration
  recorded for Table II's one-cell-scalar-count arm (`t1_d15_s138`–`s140`). A
  re-launch from that configuration would run at the two-cell obstacle patch
  without an error.
* **A DOI for the data record.** As of 2026-09-24 UTC, none has been minted, so none is claimed.
* **The decision and evidence dossiers** (`docs/decisions/**`, `docs/evidence/ED-*.md`,
  the countersign records). These are the campaign's internal deliberation. None
  of them is in this repository. As of 2026-09-24 UTC, the source repository that
  holds them (including, by the authors' count, 31 decision records and 24 dossiers) is not public. `CLAIMS.yaml` cites one of them, `ED-basin-lever-search.md`, as
  the source for a single introduction claim. That entry's `criterion_note` says that the second half of
  the claim "is NOT verifiable from the public artifact". The
  other 116 claim entries comprise 109 whose `source_file` resolves to a file here and
  7 that give `source_file: NONE`. Those 7 include 5 whose `derived_from` field names
  a file here.
* **`docs/paper/template/main.tex`.** A comment in `pyproject.toml` names this
  file as the source of its `authors` list and says that the file is not shipped
  here. It is in the source repository, which, as of 2026-09-24 UTC, is not
  public.

## Reproduction environment

The recorded values in the Software, Threading, and Hardware tables below were
read from the `runtime_record` block of the
nine graded run manifests (`t1_retain_s147`, `s148`, `s149`,
`t1_seedscale_s153`, `s154`, `s155`, `s156`, `s157`, `s158`).
These manifests are shipped under `results/experiments/`. Check any of them yourself.

### Software

| Component | Recorded value | Agreement across the nine |
|---|---|---|
| Python | `3.12.10` (`tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36`, `MSC v.1943 64 bit (AMD64)`) | identical in all nine |
| PyTorch | `2.5.1+cpu` | identical in all nine |
| Device | `cpu` | identical in all nine |
| Tensor dtype | `float32` | identical in all nine |

`requires-python` is declared as `>=3.12` in `pyproject.toml`: the minor
version that was exercised. The exact patch is above. `requirements.txt`
pins `torch==2.5.1`, the version that was run. `pyproject.toml` declares the
looser floor `torch>=2.5.1` for installing the package.

`torch` is the only third-party runtime dependency. An AST census over all 49
shipped Python files finds six non-standard-library top-level import names:

| Import name | Kind | Imported by |
|---|---|---|
| `raas_marl` | internal | 33 files, including the package's own modules |
| `c1_selectivity_harness` | internal | 2 other grading instruments, as a sibling module |
| `instruments` | internal | 1 file, the Fig. 5 generator |
| `torch` | external | 22 files |
| `pytest` | external | 1 file, the test suite |
| `matplotlib` | external | 1 file, the Fig. 5 generator |

`pytest` is therefore declared only as a development dependency: in `requirements-dev.txt`
and under `[project.optional-dependencies].dev` in `pyproject.toml`. `matplotlib` is declared only in
`requirements-figures.txt`. The `instruments/` modules add no
dependency: CI asserts by AST that they import nothing
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
the environment they found and did not set it.

### Hardware

| Field | Recorded value |
|---|---|
| Platform | `Windows-11-10.0.26200-SP0` |
| Machine | `AMD64` |
| Processor | `Intel64 Family 6 Model 158 Stepping 13, GenuineIntel` |

Physical core count is not a recorded field. The manifests record
`torch_num_interop_threads = 8`. On 2026-09-24 UTC, on an Intel Core i7-9700
under Windows 11 with `torch` 2.5.1+cpu, that setting defaulted to the core
count, 8. Reading the recorded machine's core count from the recorded value is
an inference, so that core count is not stated as a fact here.

### Wall-clock

The recorded `launch_timestamp_utc` and `finish_timestamp_utc` of the nine
graded runs give 2.38 h to 3.50 h per run (minimum `t1_seedscale_s158`,
maximum `t1_retain_s147`). Each of the nine completed 4000 rounds
(`last_completed_round = 3999`, `status = complete`).

No `resumed` key is present in any of the nine. The driver writes that key
only when it resumes a run, so its absence shows that none of them was
resumed.

Read that range as measured elapsed time, not as isolated single-run cost.
Concurrency is not a recorded manifest field, so these are wall-clock figures
obtained under a concurrency the manifests do not state. The timestamps
show that the nine ran as three batches of three, each batch sharing one
launch instant to the second:

| Batch | Runs | Launched (UTC) |
|---|---|---|
| 1 | `t1_retain_s147`, `s148`, `s149` | `20260712T093148Z` |
| 2 | `t1_seedscale_s153`, `s154`, `s155` | `20260712T224844Z` |
| 3 | `t1_seedscale_s156`, `s157`, `s158` | `20260713T013645Z` |

So three runs ran simultaneously, each with two intra-op
threads. None of the nine was timed alone on the same hardware.

### Determinism

The manifests record `cross_platform_bitwise_determinism_claimed = false`.
Bitwise-identical reproduction across a different platform, CPU, or PyTorch
build is not claimed. The seeds are recorded per run
(`t1_retain` 147–149, `t1_seedscale` 153–158).

The `instruments/` recomputation reads shipped JSON and does floating-point and
exact integer arithmetic on it. It involves no BLAS reduction order and no
dict-iteration dependence. Its Clopper–Pearson
bounds use `exp`, `log`, and `lgamma` from the `math` module, whose results can
differ between C math libraries. In 100 trials that shifted each call's result
at random by up to 1,048,576 units in the last place, the output of
`instruments.reproduce_table3` did not change.

## Provenance

The three commits below are not in this repository. As of 2026-09-24 UTC, the
source repository is not public, so a reader cannot repeat the comparisons in
this section. Those comparisons are the authors' statements.

Each of the nine graded manifests records, in `git_head`, the commit checked
out when its run was launched:

| Runs | `git_head` |
|---|---|
| `t1_retain_s147`, `s148`, `s149` | `e6b616e7729954944d9a11d0a43c1403a3b37b4b` |
| `t1_seedscale_s153`, `s154`, `s155` | `873bfcecaacbaf532afa1b09b28b6b10f86993a1` |
| `t1_seedscale_s156`, `s157`, `s158` | `2d3ffa6781237a9874129f6252ba9c5deb79329b` |

In the source repository, `src/` is identical at the three commits:
the code did not move between the three launches.

Twenty-nine of the 34 shipped `src/` files are byte-identical to that code.
The other five carry additions made for this release:

```text
raas_marl/environments/active_sensing/grid_environment.py
raas_marl/environments/active_sensing/tensor_adapter.py
raas_marl/mappo_lagrangian/buffer.py
raas_marl/mappo_lagrangian/config.py
raas_marl/mappo_lagrangian/losses.py
```

`grid_environment.py` gains one
docstring. The other four import `torch` lazily. Each gains an
`if TYPE_CHECKING:` guard, with its `from typing import TYPE_CHECKING` import,
so a type checker can resolve `torch`. Two of them, `tensor_adapter.py` and
`config.py`, also gain return annotations. At run time `TYPE_CHECKING` is
`False`, so the body of each guard never runs. Every one of the
five declares `from __future__ import annotations`, so annotations are deferred
strings and are never evaluated at run time. With docstrings, the guards, their
`from typing import TYPE_CHECKING` imports, and the annotations removed, the
executable AST of all five is identical to the graded
code. Nothing was renamed, reordered, or rewritten.

That is a weaker statement than "byte-identical".

The five `docs/evidence/*.py` instruments also differ from their versions in the
source repository. Comments in three of them record changes. In `c1_selectivity_harness.py`, a
hardcoded absolute-path fallback was deleted. In
`c1_tightened_adversarial_test.py` and `r2_null_rejection_check.py`, the two
`sys.path` insertions, which came from a hardcoded absolute path, are now
derived from `__file__`. `r2_null_rejection_check.py` also makes its `--json`
output path relative to the repository root.

## License

The software file-sets that `LICENSE` lists are MIT. Data and records (the analysis JSONs, everything
under `results/`, the Fig. 1 backing data, and `CLAIMS.yaml`) are CC BY 4.0
(`LICENSES/CC-BY-4.0.txt`). `LICENSE` states the split file-set by file-set.
`LICENSE` lists `instruments/` and `scripts/` among its software file-sets.

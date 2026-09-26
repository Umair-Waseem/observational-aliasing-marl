# Scenarios and the reserved held-out geometry

This page lists the scenarios the code defines, which of them the run manifests name, and what the paper held out. It also names the tests that assert the reservation, and runs them.

## Running the commands on this page

Run the commands from the repository root after the install in [Install and tests](../README.md#install-and-tests). The test command also needs `pytest`, from `requirements-dev.txt`, which that section installs.

Source: `requirements-dev.txt` (`pytest`)

## The scenario catalog

The scenario catalog, `stage23_scenario_catalog`, defines 14 scenarios, each a fixed `Stage23Scenario` on an 8 by 8 grid. `available_scenarios` returns their names sorted, and `make_scenario` returns one by name. `make_scenario` raises `ValueError` for a name that is not in the catalog.

Ten scenarios are risk forks (`risk_fork_*`). Each has a wall on column 4 with two open cells, the gates. The catalog lists the wall cells in each scenario's `obstacles`; the diagnostics read the column from `_FORK_WALL_COLUMN`. The code does not store gate rows. The table derives them as the open rows of column 4. The code groups the forks in two functions, `fork_hazard_layout_families` and `fork_curriculum_scenarios`. The last column counts, of the 55 run manifests, those whose `config.scenario_names` names the scenario.

| scenario | paper term | grouped by | gate rows | hidden hazard | manifests naming it |
|---|---|---|---|---|---|
| `standard_branching_hazard` | none | none (not a risk fork) | none | (2, 2), (3, 4), (4, 5) | 0 of 55 |
| `risk_gate_hidden_hazard` | none | none (not a risk fork) | none | (2, 2) | 10 of 55 |
| `unit_empty` | none | none (not a risk fork) | none | none | 0 of 55 |
| `unit_single_hazard` | none | none (not a risk fork) | none | (1, 1) | 0 of 55 |
| `risk_fork_train_upper` | anchor pair | `fork_hazard_layout_families`, key `training` | 3, 4 | (3, 4) | 45 of 55 |
| `risk_fork_train_lower` | anchor pair | `fork_hazard_layout_families`, key `training` | 3, 4 | (4, 4) | 45 of 55 |
| `risk_fork_readiness_upper` | validation geometry | `fork_hazard_layout_families`, key `readiness` | 1, 2 | (1, 4) | 0 of 55 |
| `risk_fork_readiness_lower` | validation geometry | `fork_hazard_layout_families`, key `readiness` | 1, 2 | (2, 4) | 0 of 55 |
| `risk_fork_curriculum_r0r3_upper` | curriculum pair | `fork_curriculum_scenarios` | 0, 3 | (0, 4) | 15 of 55 |
| `risk_fork_curriculum_r0r3_lower` | curriculum pair | `fork_curriculum_scenarios` | 0, 3 | (3, 4) | 15 of 55 |
| `risk_fork_curriculum_r4r7_upper` | curriculum pair | `fork_curriculum_scenarios` | 4, 7 | (4, 4) | 3 of 55 |
| `risk_fork_curriculum_r4r7_lower` | curriculum pair | `fork_curriculum_scenarios` | 4, 7 | (7, 4) | 3 of 55 |
| `risk_fork_curriculum_r0r7_upper` | curriculum pair | `fork_curriculum_scenarios` | 0, 7 | (0, 4) | 6 of 55 |
| `risk_fork_curriculum_r0r7_lower` | curriculum pair | `fork_curriculum_scenarios` | 0, 7 | (7, 4) | 6 of 55 |

Cells are (row, column); column 4 is the wall.

This command lists each catalog name in sorted order, its gate rows, and its hidden hazard cells. A `-` marks a scenario that is not a risk fork:

```shell
python -c "from raas_marl.environments.active_sensing.scenarios import available_scenarios, make_scenario; [print(n, sorted(r for r in range(8) if (r, 4) not in s.obstacles) if n.startswith('risk_fork') else '-', s.hidden_hazard_cells) for n in available_scenarios() for s in [make_scenario(n)]]"
```

```text
risk_fork_curriculum_r0r3_lower [0, 3] ((3, 4),)
risk_fork_curriculum_r0r3_upper [0, 3] ((0, 4),)
risk_fork_curriculum_r0r7_lower [0, 7] ((7, 4),)
risk_fork_curriculum_r0r7_upper [0, 7] ((0, 4),)
risk_fork_curriculum_r4r7_lower [4, 7] ((7, 4),)
risk_fork_curriculum_r4r7_upper [4, 7] ((4, 4),)
risk_fork_readiness_lower [1, 2] ((2, 4),)
risk_fork_readiness_upper [1, 2] ((1, 4),)
risk_fork_train_lower [3, 4] ((4, 4),)
risk_fork_train_upper [3, 4] ((3, 4),)
risk_gate_hidden_hazard - ((2, 2),)
standard_branching_hazard - ((2, 2), (3, 4), (4, 5))
unit_empty - ()
unit_single_hazard - ((1, 1),)
```

No gate and no hidden hazard of any catalog scenario lies on row 5 or 6.

Source: `src/raas_marl/environments/active_sensing/grid_environment.py` (`stage23_scenario_catalog`, `Stage23Scenario`); `src/raas_marl/environments/active_sensing/scenarios.py` (`available_scenarios`, `make_scenario`); `src/raas_marl/environments/active_sensing/stage24_diagnostics.py` (`_FORK_WALL_COLUMN`, `fork_hazard_layout_families`, `fork_curriculum_scenarios`); `results/experiments/*/RUN_MANIFEST.json` (`config.scenario_names`); command `python -c "from raas_marl.environments.active_sensing.scenarios import available_scenarios, make_scenario; [print(n, sorted(r for r in range(8) if (r, 4) not in s.obstacles) if n.startswith('risk_fork') else '-', s.hidden_hazard_cells) for n in available_scenarios() for s in [make_scenario(n)]]"`

## The four scenarios that are not risk forks

The catalog's other four scenarios have no two-gate wall. The paper does not name them.

- `risk_gate_hidden_hazard`: no obstacles, one hidden hazard at (2, 2), starts (2, 0) and (3, 0), and goal (2, 4). Its description says the public shortest path crosses the hidden hazard.
- `standard_branching_hazard`: seven obstacles and three hidden hazards. It is the default `scenario_name` of `Stage23EnvironmentConfig`.
- `unit_empty`: no obstacles and no hidden hazard.
- `unit_single_hazard`: no obstacles and one hidden hazard at (1, 1).

Of the 55 manifests, 10 name `risk_gate_hidden_hazard`, and these 10 name no other scenario. They are `b101_stoch`, `b102_stoch`, `b103_stoch`, `b104_greedy`, `t1_d4_s105` to `s107`, and `t1_d5_s108` to `s110`. The README places them in [Which runs are included, and their place in the paper](../README.md#which-runs-are-included-and-their-place-in-the-paper). No manifest names the other three.

Source: `src/raas_marl/environments/active_sensing/grid_environment.py` (`stage23_scenario_catalog`, `Stage23EnvironmentConfig`); `results/experiments/*/RUN_MANIFEST.json` (`config.scenario_names`)

## The reserved gate rows

The paper reported that gate positions partitioned the scenario family into three disjoint sets (Sec. IV). The trainable gate rows were {0, 3, 4, 7}. A validation set of readiness geometries sat at gate rows {1, 2}. A held-out set at gate rows {5, 6} was "deliberately preserved and never instantiated, read, or cited throughout the campaign". In Sec. VI, the paper reported that it kept the held-out geometry unspent.

In the code, the validation geometry is the pair `risk_fork_readiness_upper` and `risk_fork_readiness_lower`, and no manifest names either. `Stage25RunConfig` refuses either name in a training scenario set, with `ValueError`. The 27 grade records grade both. The held-out geometry is the paper's term alone: no catalog scenario and no registry in `src/` holds a fork with gates on rows 5 and 6. Comments and docstrings in `src/` restate the reservation, for example:

- above the risk forks in `stage23_scenario_catalog`: "Held-out fork scenarios (gates 5,6) are DEFERRED to the Protocol-v1 lock (I-3)."
- above `_FORK_WALL_COLUMN` in `stage24_diagnostics.py`: "The held-out fork group (gates 5,6) is DEFERRED to the Protocol-v1 lock (I-3), so it is not named here."

A fork's gates follow from its `obstacles`, an ordinary field of `Stage23Scenario`, so the code can express a fork with gates on rows 5 and 6. None exists at this commit. The README's [The reserved held-out geometry](../README.md#the-reserved-held-out-geometry) gives four points on the reservation. Three can be checked in this repository. This page checks the catalog point and counts the names in `config.scenario_names`; the README gives the rest. The fourth, about the probe logs, is the authors' report, and the probe logs are not in this repository.

Source: camera-ready, p. 4, Sec. IV; camera-ready, p. 7, Sec. VI; `src/raas_marl/environments/active_sensing/grid_environment.py` (`stage23_scenario_catalog`); `src/raas_marl/environments/active_sensing/stage24_diagnostics.py` (`_FORK_WALL_COLUMN`, `fork_hazard_layout_families`); `src/raas_marl/final_training/stage25_driver.py` (`Stage25RunConfig`); `results/experiments/*/RUN_MANIFEST.json` (`config.scenario_names`); `results/experiments/*/checkpoint_grades/grade_r*.json` (`grade.per_surface.risk_fork_readiness_upper`, `grade.per_surface.risk_fork_readiness_lower`)

## The `held_out` label concerns a different layout

The label `held_out` is a group of `stage24a_hazard_layout_variants`, three Stage 24-A hazard-layout variants that are not catalog scenarios. In `src/`, the string `held_out` names only this group. Two comments in `docs/evidence/c1_selectivity_harness.py` (`_fork_training_readiness_variants`) use it for the reserved gate rows 5 and 6. `_stage24a_variant_environment_config` builds each on the `risk_gate_hidden_hazard` map: it keeps that map's starts, goal, and obstacles, and replaces the hidden hazard cells. That map has no obstacles, so no wall and no gate rows. The layout `risk_gate_heldout_near_gate` puts hidden hazards on rows 2 and 3:

```shell
python -c "from raas_marl.environments.active_sensing.stage24_diagnostics import stage24a_hazard_layout_variants; [print(n, v.group, v.hidden_hazard_cells) for n, v in stage24a_hazard_layout_variants().items()]"
```

```text
risk_gate_train_center training ((2, 2),)
risk_gate_readiness_shifted readiness ((2, 3),)
risk_gate_heldout_near_gate held_out ((2, 2), (3, 2))
```

`Stage24AHazardLayoutVariant` accepts only the groups `training`, `readiness`, and `held_out`, and the variant table must contain all three. `run_stage24a_variant_readiness_diagnostics` builds all three layouts and runs the five scripted comparator policies on each. This command prints the layouts it lists as held out, and the policies it ran on `risk_gate_heldout_near_gate`:

```shell
python -c "from raas_marl.environments.active_sensing.stage24_diagnostics import run_stage24a_variant_readiness_diagnostics; d = run_stage24a_variant_readiness_diagnostics(); print(d['held_out_variant_names']); print(sorted(d['variants']['risk_gate_heldout_near_gate']['comparators']))"
```

```text
['risk_gate_heldout_near_gate']
['always_sense_shortest_path', 'no_sense_shortest_path', 'random_policy', 'risk_aware_oracle_or_heuristic', 'selective_sense_risk_aware']
```

Source: `docs/evidence/c1_selectivity_harness.py` (`_fork_training_readiness_variants`); `src/raas_marl/environments/active_sensing/stage24_diagnostics.py` (`stage24a_hazard_layout_variants`, `Stage24AHazardLayoutVariant`, `_stage24a_variant_environment_config`, `_validate_stage24a_variant_collection`, `run_stage24a_variant_readiness_diagnostics`); command `python -c "from raas_marl.environments.active_sensing.stage24_diagnostics import stage24a_hazard_layout_variants; [print(n, v.group, v.hidden_hazard_cells) for n, v in stage24a_hazard_layout_variants().items()]"`; command `python -c "from raas_marl.environments.active_sensing.stage24_diagnostics import run_stage24a_variant_readiness_diagnostics; d = run_stage24a_variant_readiness_diagnostics(); print(d['held_out_variant_names']); print(sorted(d['variants']['risk_gate_heldout_near_gate']['comparators']))"`

## The guards on the `held_out` layout

Two functions raise `RuntimeError` when a variant in the `held_out` group reaches them:

| guard | file | called by | message |
|---|---|---|---|
| `_assert_probe_variant_not_held_out` | `src/raas_marl/final_training/baseline_driver.py` | `run_readiness_probe` | held-out hazard-layout variant must never reach the Stage 25 probe runner; its hidden layout must remain unseen |
| `_assert_not_held_out` | `docs/evidence/c1_selectivity_harness.py` | `grade_policy` | I-3: the held-out hazard layout must never reach the C1 harness |

Both callers drop the `held_out` group before they call the guard: `run_readiness_probe` keeps only the groups in `_VARIANT_PROBE_GROUPS`, and the harness enumerates `_training_readiness_variants` (the default family) or `_fork_training_readiness_variants` (`--fork`), neither of which yields a `held_out` group. On these paths the guard is a second check. When `run_readiness_probe` is given fork scenario names, it refuses any name outside `fork_hazard_layout_families` and `fork_curriculum_scenarios`, also with `RuntimeError`.

No test in `tests/` calls `_assert_not_held_out`. This command calls it on the held-out layout. It exits with code 1; the excerpt shows the last line of its output, whose other lines name local paths:

```shell
python -c "import sys; sys.path.insert(0, 'docs/evidence'); from c1_selectivity_harness import _assert_not_held_out; from raas_marl.environments.active_sensing.stage24_diagnostics import stage24a_hazard_layout_variants; _assert_not_held_out(stage24a_hazard_layout_variants()['risk_gate_heldout_near_gate'])"
```

```text
RuntimeError: I-3: the held-out hazard layout must never reach the C1 harness
```

Source: `src/raas_marl/final_training/baseline_driver.py` (`_assert_probe_variant_not_held_out`, `run_readiness_probe`, `_VARIANT_PROBE_GROUPS`, `_run_fork_readiness_probe`); `docs/evidence/c1_selectivity_harness.py` (`_assert_not_held_out`, `grade_policy`, `_training_readiness_variants`, `_fork_training_readiness_variants`, `_select_family`); command `python -c "import sys; sys.path.insert(0, 'docs/evidence'); from c1_selectivity_harness import _assert_not_held_out; from raas_marl.environments.active_sensing.stage24_diagnostics import stage24a_hazard_layout_variants; _assert_not_held_out(stage24a_hazard_layout_variants()['risk_gate_heldout_near_gate'])"`

## Tests on the reserved gate rows and the `held_out` layout

These tests in `tests/test_clean.py` assert something about the reserved gate rows or the `held_out` layout. The second column says which.

| test | concerns | what it asserts |
|---|---|---|
| `test_grid_environment_catalog_has_fourteen_scenarios` | reserved gate rows | The catalog holds exactly the 14 names in the table above. Any added scenario, held-out or not, makes it fail. |
| `test_fork_hazard_layout_families_registry` | reserved gate rows | The fork registry has only the keys `training` and `readiness`, and no catalog name contains `heldout` or `held_out`. |
| `test_drcurr_registry_exact_and_separate_from_grade_surface` | reserved gate rows | The curriculum registry is the six curriculum names, none of them in the fork registry, and no catalog name contains `heldout` or `held_out`. |
| `test_drcurr_gate_rows_free_and_row_disjoint_from_reserved` | reserved gate rows | Each curriculum scenario has two gate rows, both in {0, 3, 4, 7} and none in {1, 2, 5, 6}. |
| `test_drcurr3_gate_rows_disjoint_and_unique_control` | reserved gate rows | The {0, 7} pair shares no gate row with the anchor pair or with {1, 2, 5, 6}, and is the only pair of rows from {0, 3, 4, 7} that shares none with the anchor pair. |
| `test_drcurr_existing_catalog_entries_byte_identical` | reserved gate rows | The eight catalog entries that existed before the curriculum, among them the anchor and validation pairs, keep their exact starts, goals, obstacles, and hidden hazards, so the two pairs' gate rows stay {3, 4} and {1, 2}. |
| `test_drcurr_readiness_exclusion_is_unconditional` | reserved gate rows | The run configuration refuses the validation pair in a training scenario set with `ValueError`, whether the armed floor is off or on, and when one of its two names is mixed into a curriculum set. |
| `test_drd1l1b_probe_repoint_fork_family_and_i3_guard` | reserved gate rows | The fork readiness probe grades exactly the training and validation scenarios it is given, and refuses `risk_gate_hidden_hazard`, a name outside the fork registries, with `RuntimeError`. |
| `test_drcurr_fork_probe_admits_curriculum_and_rejects_unknown` | reserved gate rows | The fork readiness probe admits a curriculum name and refuses `risk_gate_hidden_hazard` with `RuntimeError`. |
| `test_stage24_diagnostics_hazard_layout_variants_groups` | `held_out` layout | The Stage 24-A variants include the groups `training`, `readiness`, and `held_out`. |
| `test_s25_run_readiness_probe_structure_and_heldout_absence` | `held_out` layout | `run_readiness_probe` returns the two variants `risk_gate_train_center` and `risk_gate_readiness_shifted`, and the name `risk_gate_heldout_near_gate` appears nowhere in its output. |
| `test_s25_held_out_variant_guard_raises_runtime_error` | `held_out` layout | `_assert_probe_variant_not_held_out` raises `RuntimeError` for a hand-built variant in the `held_out` group. |
| `test_s25_completed_run_probe_log_records` | `held_out` layout | In a two-round run of `run_baseline_training`, each `probe_log.jsonl` record holds the two variants above, and none contains `risk_gate_heldout_near_gate`. |
| `test_s25u_collection_config_rejects_non_catalog_names` | `held_out` layout | The training collection's `Stage25CollectionConfig` rejects four names that are not in the catalog with `ValueError`; one of them is `risk_gate_heldout_near_gate`. |

No test reads the shipped run manifests; the manifest counts on this page come from those records. This command runs the fourteen tests. The last test has four parameters, so the run has 17 cases:

```shell
python -m pytest tests/test_clean.py -o addopts="" -p no:cacheprovider -q -rp -k "test_grid_environment_catalog_has_fourteen_scenarios or test_fork_hazard_layout_families_registry or test_drcurr_registry_exact_and_separate_from_grade_surface or test_drcurr_gate_rows_free_and_row_disjoint_from_reserved or test_drcurr3_gate_rows_disjoint_and_unique_control or test_drcurr_existing_catalog_entries_byte_identical or test_drcurr_readiness_exclusion_is_unconditional or test_drd1l1b_probe_repoint_fork_family_and_i3_guard or test_drcurr_fork_probe_admits_curriculum_and_rejects_unknown or test_stage24_diagnostics_hazard_layout_variants_groups or test_s25_run_readiness_probe_structure_and_heldout_absence or test_s25_held_out_variant_guard_raises_runtime_error or test_s25_completed_run_probe_log_records or test_s25u_collection_config_rejects_non_catalog_names"
```

```text
PASSED tests/test_clean.py::test_grid_environment_catalog_has_fourteen_scenarios
PASSED tests/test_clean.py::test_stage24_diagnostics_hazard_layout_variants_groups
PASSED tests/test_clean.py::test_s25_run_readiness_probe_structure_and_heldout_absence
PASSED tests/test_clean.py::test_s25_held_out_variant_guard_raises_runtime_error
PASSED tests/test_clean.py::test_s25_completed_run_probe_log_records
PASSED tests/test_clean.py::test_fork_hazard_layout_families_registry
PASSED tests/test_clean.py::test_s25u_collection_config_rejects_non_catalog_names[risk_gate_train_center]
PASSED tests/test_clean.py::test_s25u_collection_config_rejects_non_catalog_names[risk_gate_readiness_shifted]
PASSED tests/test_clean.py::test_s25u_collection_config_rejects_non_catalog_names[risk_gate_heldout_near_gate]
PASSED tests/test_clean.py::test_s25u_collection_config_rejects_non_catalog_names[not_a_scenario]
PASSED tests/test_clean.py::test_drd1l1b_probe_repoint_fork_family_and_i3_guard
PASSED tests/test_clean.py::test_drcurr_registry_exact_and_separate_from_grade_surface
PASSED tests/test_clean.py::test_drcurr_gate_rows_free_and_row_disjoint_from_reserved
PASSED tests/test_clean.py::test_drcurr_readiness_exclusion_is_unconditional
PASSED tests/test_clean.py::test_drcurr_fork_probe_admits_curriculum_and_rejects_unknown
PASSED tests/test_clean.py::test_drcurr_existing_catalog_entries_byte_identical
PASSED tests/test_clean.py::test_drcurr3_gate_rows_disjoint_and_unique_control
17 passed, 1522 deselected, 1 warning in 3.57s
```

The excerpt shows the result lines only. The time on the last line varies between runs. The omitted warnings summary names local paths; its one warning is the NumPy warning from `torch` that [Install and tests](../README.md#install-and-tests) describes.

Source: `tests/test_clean.py` (`test_grid_environment_catalog_has_fourteen_scenarios`, `test_fork_hazard_layout_families_registry`, `test_drcurr_registry_exact_and_separate_from_grade_surface`, `test_drcurr_gate_rows_free_and_row_disjoint_from_reserved`, `test_drcurr3_gate_rows_disjoint_and_unique_control`, `test_stage24_diagnostics_hazard_layout_variants_groups`, `test_s25_run_readiness_probe_structure_and_heldout_absence`, `test_s25_held_out_variant_guard_raises_runtime_error`, `test_s25_completed_run_probe_log_records`, `test_s25u_collection_config_rejects_non_catalog_names`, `test_drcurr_existing_catalog_entries_byte_identical`, `test_drcurr_readiness_exclusion_is_unconditional`, `test_drd1l1b_probe_repoint_fork_family_and_i3_guard`, `test_drcurr_fork_probe_admits_curriculum_and_rejects_unknown`); command `python -m pytest tests/test_clean.py -o addopts="" -p no:cacheprovider -q -rp -k "test_grid_environment_catalog_has_fourteen_scenarios or test_fork_hazard_layout_families_registry or test_drcurr_registry_exact_and_separate_from_grade_surface or test_drcurr_gate_rows_free_and_row_disjoint_from_reserved or test_drcurr3_gate_rows_disjoint_and_unique_control or test_drcurr_existing_catalog_entries_byte_identical or test_drcurr_readiness_exclusion_is_unconditional or test_drd1l1b_probe_repoint_fork_family_and_i3_guard or test_drcurr_fork_probe_admits_curriculum_and_rejects_unknown or test_stage24_diagnostics_hazard_layout_variants_groups or test_s25_run_readiness_probe_structure_and_heldout_absence or test_s25_held_out_variant_guard_raises_runtime_error or test_s25_completed_run_probe_log_records or test_s25u_collection_config_rejects_non_catalog_names"`

Checked against the code at commit `6a5e22d` on 2026-09-26 UTC.

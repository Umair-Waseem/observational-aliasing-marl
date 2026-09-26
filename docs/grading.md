# How a run is graded

This page describes how the shipped instruments grade a run against the four bars of the paper's Table III: the bars, the own-ablation witness, the two-of-three rule over the graded checkpoints, and the reroute-basin classifier. For the inputs each Table III number is recomputed from, see [What recomputes from what](../README.md#what-recomputes-from-what).

## What each command needs

| command | reads | needs | exit status here |
|---|---|---|---|
| `python -m instruments.grade_composer --check` | the grade records, `results/held2_conjuncts.json`, `docs/evidence/t1_seedscale_analysis.json` | the standard library | 0 |
| `python -m instruments.reroute_basin` | `docs/evidence/t1_seedscale_analysis.json` | the standard library | 0 |
| `python docs/evidence/c1_selectivity_harness.py --grade-run t1_retain_s147 --fork` | the shipped checkpoint `state_round_004000.pt` | PyTorch | 0 |
| `python docs/evidence/c1_selectivity_harness.py --validate --fork` | no record; scripted policies | PyTorch | 0 |
| `python docs/evidence/r2_null_rejection_check.py` | no record; synthetic grades | PyTorch | 0 |
| `python docs/evidence/c1_tightened_adversarial_test.py` | no record; scripted policies | PyTorch | 0 |
| `python docs/evidence/held2_bar.py --validate` | no record; synthetic curves, and three training curves if present | the standard library | 0, reported as degraded |
| `python docs/evidence/held2_bar.py --grade-run t1_retain_s147` | the run's `training_curve.jsonl` | the standard library and the raw training logs | 1 |
| `python docs/evidence/drh5_trigger_regression.py` | four runs' `training_curve.jsonl` | PyTorch and the raw training logs | 1 |

The exit statuses are from runs on 2026-09-26 UTC, on Windows 11 with Python 3.12.10 and `torch` 2.5.1+cpu. Run each command from the repository root: `instruments` is importable only from there. "PyTorch" means the install in [Install and tests](../README.md#install-and-tests). The `docs/evidence` instruments that import the package put the repository's `src` on `sys.path` themselves. Those that import `torch` also print a warning that NumPy is not installed. The excerpts below omit it.

As of 2026-09-26 UTC, no file in this repository matches `results/experiments/*/training_curve.jsonl`, so the two commands that need the raw training logs stop. The README states that, as of 2026-09-24 UTC, the separate data record promised for those logs does not exist. This page does not verify that. See [The raw training logs](../README.md#the-raw-training-logs).

Source: `instruments/grade_composer.py` (`compose_seed`, `load_checkpoints`); `instruments/reroute_basin.py` (`ANALYSIS_JSON`); `docs/evidence/c1_selectivity_harness.py` (`latest_state_file`, `load_model_from_state`); `docs/evidence/held2_bar.py` (`load_run_curve`); `docs/evidence/drh5_trigger_regression.py` (`load_curve`, `EXPECTED`); `results/experiments/*/state_round_*.pt`; command `python -m instruments.grade_composer --check`; command `python -m instruments.reroute_basin`; command `python docs/evidence/c1_selectivity_harness.py --grade-run t1_retain_s147 --fork`; command `python docs/evidence/c1_selectivity_harness.py --validate --fork`; command `python docs/evidence/held2_bar.py --validate`; command `python docs/evidence/held2_bar.py --grade-run t1_retain_s147`; command `python docs/evidence/drh5_trigger_regression.py`; command `python docs/evidence/r2_null_rejection_check.py`; command `python docs/evidence/c1_tightened_adversarial_test.py`

## The four bars

`compose_seed` in `instruments/grade_composer.py` combines each bar from conjuncts. [What recomputes from what](../README.md#what-recomputes-from-what) gives the file each conjunct is read from.

| paper bar | code criterion | conjuncts |
|---|---|---|
| task success (joint with hazard) | `task_success` | `bar_c4_pass` |
| selective-sensing hold (core) | `core_hold` | `held2_ab_pass`, `bar_c4_pass`, and `retain_c` at two of three checkpoints |
| anchor consolidation | `anchor_consolidation` | hazard at most 0.5 (`ANCHOR_HAZARD_CEILING`) on both anchor mirrors, and `anchor_c` at two of three checkpoints |
| transfer to validation | `transfer` | `c4_gen` at two of three checkpoints |

`held2_verdict` in `docs/evidence/held2_bar.py` computes the curve conjuncts from a run's per-round training curve:

| conjunct | rule | constants |
|---|---|---|
| `bar_c4_pass` | over the curve's last 100 rounds, mean team success at least 0.9 and mean episodic hazard at most 0.5 | `C4_TAIL`, `C4_SUCCESS`, `C4_HAZARD` |
| (a) joint occupancy | in rounds 3500 to 3999, at least 85% of rounds have team success at least 0.9 and hazard at most 0.5 | `TAIL_START`, `TAIL_END`, `OCCUPANCY_SUCCESS`, `OCCUPANCY_HAZARD`, `OCCUPANCY_MIN_FRACTION` |
| (b) relapse windows | in the same rounds, no 50-round window with mean success below 0.75 or mean hazard at least 0.75 | `RELAPSE_WINDOW`, `RELAPSE_SUCCESS_FLOOR`, `RELAPSE_HAZARD_CEIL` |

`held2_ab_pass` is (a) and (b) together. `held2_verdict` raises a `ValueError` unless the curve holds all 500 rounds from 3500 to 3999. These rules match the paper's task-success bar (Sec. III) and its definition of a run that holds (Sec. IV). The code's core hold adds two things that definition does not list: the task-success bar, and a witness on a trained mirror rather than on any surface. For anchor consolidation and transfer, the paper states no checkpoint rule (Sec. IV). The code applies two of three to both. The paper also reported that hazard is graded per mirror on all four training mirrors (Sec. IV). The composer reads the per-mirror hazard of the two anchor mirrors only (`ANCHOR_PER_MIRROR_KEYS`), and the per-mirror hazard of the {0, 3} mirrors enters no criterion.

No shipped script writes `results/held2_conjuncts.json` or the `per_mirror` blocks of `docs/evidence/t1_seedscale_analysis.json`. The composer reads them as recorded. The file names a source for each seed (`per_seed.*.source`): `docs/evidence/t1_retain_analysis.json` for seeds 147 to 149, and a grader output, `seedscale_mech_summary.json`, for seeds 153 to 158. No file of that name is in this repository, and no shipped record shows that `held2_verdict` produced these values. The composer's module docstring describes the `per_mirror` hazard as tail-100 hazard.

Source: `instruments/grade_composer.py` (`compose_seed`, `CRITERIA`, `CRITERION_LABELS`, `ANCHOR_HAZARD_CEILING`, `ANCHOR_PER_MIRROR_KEYS`, `MAJORITY`); `docs/evidence/held2_bar.py` (`held2_verdict`, `C4_TAIL`, `C4_SUCCESS`, `C4_HAZARD`, `TAIL_START`, `TAIL_END`, `OCCUPANCY_SUCCESS`, `OCCUPANCY_HAZARD`, `OCCUPANCY_MIN_FRACTION`, `RELAPSE_WINDOW`, `RELAPSE_SUCCESS_FLOOR`, `RELAPSE_HAZARD_CEIL`); `results/held2_conjuncts.json` (`per_seed.*.bar_c4_pass`, `per_seed.*.held2_ab_pass`, `per_seed.*.source`); `docs/evidence/t1_seedscale_analysis.json` (`per_seed.*.per_mirror`); camera-ready, p. 3, Sec. III; camera-ready, p. 4, Sec. IV

## The own-ablation witness

The paper certified selective sensing with a counterfactual own-ablation witness (Sec. III). The code computes it in `grade_policy`. With `--fork`, it grades the ten fork surfaces: the training, curriculum, and validation mirrors, the last labeled `readiness` in the records. Without `--fork`, the harness grades the older `risk_gate` variants instead. On each surface it runs three episodes at the probe seed 9001 (`PROBE_SEED`). The intact episode runs the trained policy greedily. The own-blind episode zeroes the revealed-information channel for the whole episode, the paper's ablation. Every observation the policy receives has `revealed_local_hazards` emptied, from the reset to the last step (`blind_reveals`), so its revealed-information input (`revealed_information_from_stage23`) is all zeros. The environment still reveals and still charges each sense. Only the policy's copy is blinded. The third episode runs the scripted `no_sense` route, for comparison only. `grade_policy` calls `_assert_not_held_out` on each surface. None of the ten carries the `held_out` label.

`own_blind_hazard_avoided` is the own-blind episode's hazard entries minus the intact episode's. A hazard entry is one agent ending one step on the hidden hazard. `tightened_composite_c1_verdict` counts a surface as a witness when four conditions hold. The intact episode reaches team success (`team_success`). The policy senses on at least one of its decisions on that surface and on at most half of them, the selectivity ceiling (`SELECTIVITY_CEILING`, 0.5). Revealed information changed a move (`adapt_fired`). And `own_blind_hazard_avoided` is above 0: hazard exposure rises under the ablation. `adapt_fired` is set when, at a step with a nonzero reveal, the policy's move differs from the move it makes from the same recurrent state with the reveal zeroed (`greedy_model_policy_fn`).

The verdict is the first that applies. The first three test the decisions of all ten surfaces pooled: `FAIL_NO_DECISIONS` (no decision), `FAIL_NEVER_SENSE` (no sense), and `FAIL_ALWAYS_SENSE` (a sense at every decision). The rest look at surfaces one by one: `SELECTIVE_COMPOSITE` (at least one witness), `CAUSAL_BUT_NOT_SELECTIVE` (the other three conditions on a surface that senses above the ceiling), `SENSES_ON_FAILURE_ONLY` (it senses only on surfaces it fails), and otherwise `SENSES_NOT_CAUSAL`. In the 27 shipped grade records, the verdict is `SELECTIVE_COMPOSITE` in 21, `SENSES_ON_FAILURE_ONLY` in 3, `SENSES_NOT_CAUSAL` in 2, and `FAIL_NEVER_SENSE` in 1. `grade.seed` is 9001 in all 27 records, and `grade.surface_count` is 10 in all 27 records.

`--grade-run` grades a run's last checkpoint by file name. The 27 shipped policy checkpoints belong to the nine replication runs, at rounds 3500, 3750, and 4000, so this grades seed 147 at round 4000 (excerpt):

```shell
python docs/evidence/c1_selectivity_harness.py --grade-run t1_retain_s147 --fork
```

```text
        "state_file": "state_round_004000.pt",
          "tightened_verdict": "SELECTIVE_COMPOSITE",
          "witness_count": 4,
    "c1_aggregate_pass_2of3": false,
```

Its `grade_run.per_run.t1_retain_s147` object equals the `grade` object in that run's `checkpoint_grades/grade_r004000.json`, except that it adds `state_file` and its `label` lacks `@r4000`. For any of the other 46 runs, `--grade-run` stops with exit status 1 and reports `no policy checkpoint found`, as the README's [The raw training logs](../README.md#the-raw-training-logs) says. `c1_aggregate_pass_2of3` counts runs, not checkpoints. It needs a `SELECTIVE_COMPOSITE` verdict on at least two of the runs graded, so it is false for one run.

Source: `docs/evidence/c1_selectivity_harness.py` (`grade_policy`, `_assert_not_held_out`, `rollout_episode`, `_blind_reveal_observations`, `_fork_training_readiness_variants`, `_select_family`, `tightened_composite_c1_verdict`, `SELECTIVITY_CEILING`, `PROBE_SEED`, `latest_state_file`, `grade_run_ids`); `src/raas_marl/environments/active_sensing/tensor_adapter.py` (`revealed_information_from_stage23`); `src/raas_marl/final_training/baseline_driver.py` (`greedy_model_policy_fn`); `results/experiments/*/checkpoint_grades/grade_r*.json` (`grade.tightened_composite.tightened_verdict`, `grade.seed`, `grade.surface_count`, `grade.per_surface.risk_fork_readiness_lower.group`); `results/experiments/*/state_round_*.pt`; `results/experiments/*/RUN_MANIFEST.json`; camera-ready, p. 3, Sec. III; camera-ready, p. 4, Sec. III; command `python docs/evidence/c1_selectivity_harness.py --grade-run t1_retain_s147 --fork`

## The two-of-three rule

Each of the nine replication seeds was graded at three checkpoints, rounds 3500, 3750, and 4000 (`CHECKPOINT_ROUNDS`). `derive_checkpoint` recomputes four conjuncts from each grade record instead of reading a stored pass or fail:

| conjunct | true at a checkpoint when |
|---|---|
| `retain_c` | the verdict is `SELECTIVE_COMPOSITE` with a witness on an anchor or {0, 3} curriculum mirror (`TRAINED_SURFACES`) |
| `anchor_c` | the verdict is `SELECTIVE_COMPOSITE` with a witness on an anchor mirror (`ANCHOR_SURFACES`) |
| `c4_gen` | both validation mirrors (`VALIDATION_SURFACES`) reach team success with zero hazard entries |
| `c1_gen` | a witness is on a validation mirror |

`compose_seed` counts the checkpoints at which each conjunct is true, and passes the conjunct when the count is at least 2 (`MAJORITY`). Each conjunct is counted on its own, so two conjuncts may pass on different checkpoints. `c1_gen` enters no criterion. The composer only prints it. Transfer therefore needs no witness, only success at zero hazard. The paper reported that each transferring seed had a witness on the validation surface at two graded checkpoints (Sec. V-C). The `c1_gen` column shows it. The command prints (excerpt; it also prints the four rates):

```shell
python -m instruments.grade_composer --check
```

```text
Per-seed composition (counts are of the three checkpoints r3500/r3750/r4000)

   seed                  run  retain_c anchor_c  c4_gen  c1_gen   criteria met
    147       t1_retain_s147       3/3      3/3     2/3     2/3   task_success, core_hold, anchor_consolidation, transfer
    148       t1_retain_s148       2/3      2/3     0/3     0/3   anchor_consolidation
    149       t1_retain_s149       3/3      3/3     0/3     0/3   task_success, core_hold, anchor_consolidation
    153    t1_seedscale_s153       3/3      3/3     0/3     0/3   task_success, core_hold, anchor_consolidation
    154    t1_seedscale_s154       1/3      1/3     0/3     0/3   -
    155    t1_seedscale_s155       3/3      3/3     0/3     1/3   anchor_consolidation
    156    t1_seedscale_s156       3/3      3/3     0/3     0/3   task_success, anchor_consolidation
    157    t1_seedscale_s157       0/3      0/3     0/3     0/3   -
    158    t1_seedscale_s158       3/3      3/3     2/3     2/3   task_success, core_hold, anchor_consolidation, transfer
  REGISTRY AUDIT: every seed agrees with the registered per-seed values.
  MATCHES Table III (both the n=9 and the fresh-seed-only columns).
```

`--check` compares each seed's four criteria with the per-seed values registered in `docs/evidence/t1_seedscale_analysis.json` (`audit`). `reproduce_table3` applies `clopper_pearson` to the four tallies, and `fisher_exact_two_sided` to a 2x2 table of the arm-fire flag (`mech_f84_fired`) against transfer (`fisher_arm_fire_deconfound`). See [Reproduce the numbers in Table III](../README.md#reproduce-the-numbers-in-table-iii).

Source: `instruments/grade_composer.py` (`CHECKPOINT_ROUNDS`, `MAJORITY`, `derive_checkpoint`, `compose_seed`, `TRAINED_SURFACES`, `ANCHOR_SURFACES`, `VALIDATION_SURFACES`, `SELECTIVE_VERDICT`, `audit`); `instruments/reproduce_table3.py` (`main`, `fisher_arm_fire_deconfound`); `docs/evidence/t1_seedscale_analysis.json` (`per_seed.*.mech_f84_fired`); `instruments/statistics.py` (`clopper_pearson`, `fisher_exact_two_sided`); camera-ready, p. 4, Sec. IV; camera-ready, p. 6, Sec. V-C; command `python -m instruments.grade_composer --check`

## The reroute-basin classifier

The reroute-basin classification of Table III comes from `classify` in `instruments/reroute_basin.py`, which labels the reroute each policy executes on the lower validation mirror, `risk_fork_readiness_lower`. `classify_shipped_seeds` reads one recorded rollout per seed from `per_seed.<seed>.readiness_lower_outcome` in `docs/evidence/t1_seedscale_analysis.json`. Four fields decide the basin: `any_row0` (whether either agent ever occupied row 0), `cross_rows` (the wall-column row at which each agent crossed, or null), `haz` (hazard entries), and `steps` (episode length). The classifier also accepts `succ`, team success, and does not test it. The first rule that matches gives the basin:

| basin | rule |
|---|---|
| `OVERSHOOT` | `any_row0` is true |
| `EQUIVARIANT` | an agent crossed at row 1 (`SAFE_GATE_ROW`), with no hazard entry |
| `FREEZE` | no agent crossed, in an episode of at least 30 steps (`FREEZE_STEP_FLOOR`) |
| `BLIND` | at least one hazard entry |
| `OTHER` | none of the above |

The paper reported the equivariant, overshoot, and freeze counts in Table III (Sec. V-D). The command prints each seed's inputs and basin, and compares the partition with the reported one (`validate`):

```shell
python -m instruments.reroute_basin
```

```text
Reroute basin on the lower validation mirror (risk_fork_readiness_lower)

   seed    row0  cross_rows             haz  steps  basin
    147   False  0:1, 1:1                 0     12  EQUIVARIANT
    148   False  0:None, 1:None           0     32  FREEZE
    149    True  0:2, 1:2                 2     17  OVERSHOOT
    153   False  0:1, 1:1                 0     14  EQUIVARIANT
    154    True  0:None, 1:1              0     32  OVERSHOOT
    155   False  0:None, 1:None           0     32  FREEZE
    156   False  0:None, 1:None           0     32  FREEZE
    157   False  0:1, 1:1                 0     32  EQUIVARIANT
    158   False  0:1, 1:1                 0     12  EQUIVARIANT

  EQUIVARIANT  4/9   {147, 153, 157, 158}
  OVERSHOOT    2/9   {149, 154}
  FREEZE       3/9   {148, 155, 156}

  Table III prints  equivariant / overshoot / freeze = 4/9 / 2/9 / 3/9
  MATCHES the reported partition.
```

The rule order decides seeds 149 and 154. Seed 149 entered the hazard twice, and is `OVERSHOOT` because it reached row 0. Seed 154 also reached row 0. Its agent 1 crossed at row 1 with no hazard entry, so without the first rule it would be `EQUIVARIANT`. Seed 157 is `EQUIVARIANT` with a 32-step episode, because success is not tested.

Source: `instruments/reroute_basin.py` (`classify`, `classify_record`, `classify_shipped_seeds`, `BASINS`, `SAFE_GATE_ROW`, `FREEZE_STEP_FLOOR`, `SHIPPED_FIELD_ALIASES`, `validate`); `docs/evidence/t1_seedscale_analysis.json` (`per_seed.*.readiness_lower_outcome`); camera-ready, p. 6, Table III; camera-ready, p. 6, Sec. V-D; command `python -m instruments.reroute_basin`

## The other checks under docs/evidence

`held2_bar.py --validate` runs 17 checks on synthetic curves, which pass. It skips its three real-curve regressions, three checks each, on the `t1_d10_s123` to `s125` curves. Its output ends (excerpt):

```shell
python docs/evidence/held2_bar.py --validate
```

```text
  *** DEGRADED: 3 of 3 real-curve regressions were SKIPPED (t1_d10_s123, t1_d10_s124, t1_d10_s125).
ALL CHECKS PASS  (DEGRADED: 3 real-curve regression(s) skipped)
```

Three checks build their policies or grades in code and need no record. Each exits 0:

```shell
python docs/evidence/c1_selectivity_harness.py --validate --fork
python docs/evidence/r2_null_rejection_check.py
python docs/evidence/c1_tightened_adversarial_test.py
```

The first grades the scripted comparators `selective_sense`, `no_sense`, and `always_sense` on the ten fork surfaces. The last field of its JSON output is `"all_passed": true`. The second checks the verdict that `tightened_composite_c1_verdict` gives synthetic per-surface grades. Its last line is `ALL PASS`. The third checks that a policy whose route avoids the hazard by reading its location directly, while sensing incidentally, is not a witness. Its last line is `ADVERSARIAL CAUSAL-SOUNDNESS: ALL PASS`.

Two commands need a run's `training_curve.jsonl`:

```shell
python docs/evidence/held2_bar.py --grade-run t1_retain_s147
python docs/evidence/drh5_trigger_regression.py
```

The first computes the curve conjuncts for one run. It stops with exit status 1. The first text line of its message is `held2_bar --grade-run: required input not found`, and the next names `results/experiments/t1_retain_s147/training_curve.jsonl`. The second checks the armed-floor and recovery triggers of the training driver, not a Table III bar. It replays four runs' curves (`EXPECTED`) and stops at the first with exit status 1. The first text line of its message is `drh5_trigger_regression: required input not found`, and the next names `results/experiments/t1_d11_s128/training_curve.jsonl`.

Both messages then say to download the data record and to see the README's "Data availability" section for its location. As of 2026-09-24 UTC, the README states that the record does not exist and gives no location. Both also say that the code repository ships only the `RUN_MANIFEST.json` files. For the nine replication seeds, this repository also ships the policy checkpoints and their grade records, as the README notes.

Source: `docs/evidence/held2_bar.py` (`_validate`, `main`, `_missing_curve_message`); `docs/evidence/c1_selectivity_harness.py` (`validate`); `docs/evidence/r2_null_rejection_check.py` (`CHECKS`, `verdict`, `main`); `docs/evidence/c1_tightened_adversarial_test.py` (`privileged_safe`, `main`); `docs/evidence/drh5_trigger_regression.py` (`EXPECTED`, `load_curve`, `main`); command `python docs/evidence/held2_bar.py --validate`; command `python docs/evidence/c1_selectivity_harness.py --validate --fork`; command `python docs/evidence/r2_null_rejection_check.py`; command `python docs/evidence/c1_tightened_adversarial_test.py`; command `python docs/evidence/held2_bar.py --grade-run t1_retain_s147`; command `python docs/evidence/drh5_trigger_regression.py`

Checked against the code at commit `6a5e22d` on 2026-09-26 UTC.

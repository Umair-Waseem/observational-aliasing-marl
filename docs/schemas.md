# Run manifests and grade records: what each key holds

This page lists every key path in the 55 run manifests and the 27 per-checkpoint grade records under `results/experiments/`. For each path it gives the JSON type, the number of records that hold it, and its meaning from the code that writes or reads it. A path such as `seeds[]` means an element of the list `seeds`. Values that identify a process, a host, a commit, or a time are described but not shown. The README's [Which runs are included](../README.md#which-runs-are-included-and-their-place-in-the-paper) section says which run belongs to which part of the paper.

## Which code writes and reads each record

| record | files | written by | read by |
|---|---|---|---|
| run manifest, `results/experiments/*/RUN_MANIFEST.json` | 55 | `build_run_manifest_payload` and `write_run_manifest`, called by `run_stage25_training` or `run_baseline_training` | no shipped instrument |
| grade record, `results/experiments/*/checkpoint_grades/grade_r*.json` | 27 | no shipped script; the value of its key `grade` is what `grade_policy` returns | `derive_checkpoint` in `instruments/grade_composer.py` |

Of the 55 manifests, 51 have `stage` `25` and come from `run_stage25_training`. The other 4 have `stage` `25-harness` and come from `run_baseline_training`. They are the baseline records `b101_stoch`, `b102_stoch`, `b103_stoch`, and `b104_greedy`. A driver writes its manifest at launch with status `running` and rewrites it every `heartbeat_every` rounds. It writes it a last time with status `complete`, `killed`, or `crashed`.

No shipped script writes the grade-record files. `c1_selectivity_harness.py --grade-run` prints the same object for a run's last checkpoint under `grade_run.per_run.<run_id>`, with `state_file` added and a `label` without the round, as the README's [Data availability](../README.md#data-availability) section says.

Source: `src/raas_marl/final_training/run_logging.py` (`build_run_manifest_payload`, `write_run_manifest`, `_VALID_STATUSES`); `src/raas_marl/final_training/stage25_driver.py` (`run_stage25_training`, `heartbeat_every`); `src/raas_marl/final_training/baseline_driver.py` (`run_baseline_training`); `docs/evidence/c1_selectivity_harness.py` (`grade_policy`, `grade_run_ids`); `instruments/grade_composer.py` (`derive_checkpoint`); record `results/experiments/*/RUN_MANIFEST.json` (`stage`, `run_id`)

## Run manifest: top-level keys

`build_run_manifest_payload` assembles the core keys and the boundary flags. Both drivers add `runtime_record` and the progress keys `last_completed_round` and `finish_timestamp_utc`. `run_stage25_training` also adds `config_sha256` and the resume keys, and replaces the builder's labels and flags with those of `stage25_boundary_flags`.

| keys (type) | present in | meaning |
|---|---|---|
| `run_id` (str) | 55 of 55 | Run name; the name of its folder, and equal to `config.run_id`. |
| `tier` (str) | 55 of 55 | Run tier. The builder accepts `T0`, `T1`, and `T2` and rejects `T3`. `T1` in all 55. |
| `seeds` (list); `seeds[]` (int) | 55 of 55 | A one-element list holding the training seed, `config.seed`. |
| `implementing_dr_ids` (list); `implementing_dr_ids[]` (str) | 55 of 55 (the list); 51 of 55 (its elements) | Identifiers of the campaign's decision records that the driver names. Empty in the baseline records. The decision records are not in this repository. |
| `config` (dict) | 55 of 55 | The run configuration. See the next two sections. |
| `config_sha256` (str) | 51 of 55 | A hash of `config`. See [The `config_sha256` rule](#the-config_sha256-rule). Absent from the baseline records. |
| `runtime_record` (dict) | 55 of 55 | Runtime metadata. See [The `runtime_record` block](#the-runtime_record-block). |
| `git_head` (str) | 55 of 55 | Output of `git rev-parse HEAD` when the process started (40 hex characters; not shown). The code writes `null` if the command fails or prints nothing; no shipped record holds `null`. |
| `pid` (int) | 55 of 55 | Operating-system process id of the run (not shown). |
| `launch_timestamp_utc`, `finish_timestamp_utc` (str) | 55 of 55 | UTC times in the form `YYYYMMDDTHHMMSSZ` (not shown). The finish time is written with status `complete`. |
| `status` (str) | 55 of 55 | One of `running`, `complete`, `killed`, `crashed`. `complete` in all 55. |
| `last_completed_round` (int) | 55 of 55 | Zero-based index of the last finished round. With status `complete` the driver writes `config.update_rounds` minus 1. |
| `manifest_version` (int) | 55 of 55 | Schema version, `1` in all 55. |
| `ablation_run`, `baseline_comparison_run`, `claim_evidence_created`, `cross_platform_bitwise_determinism_claimed`, `evaluation_run`, `final_evaluation_run`, `paper_facing_results_created`, `replay_buffer_artifact_created`, `rollout_buffer_artifact_created`, `statistical_test_run` (bool) | 55 of 55 | Development-stage boundary flags. `false` in all 55. |
| `checkpoint_created`, `optimizer_state_saved`, `serialized_model_artifact_created` (bool) | 55 of 55 | Boundary flags that `stage25_boundary_flags` sets to `true` once a training state is saved: `true` in 51, `false` in the 4 baseline records. |
| `development_only` (bool) | 55 of 55 | `true` in all 55. |
| `stage`, `training_run`, `claim_status` (str) | 55 of 55 | Labels: `stage` is `25` or `25-harness`; `training_run` names the driver's run type; `claim_status` is `not tested / not supported` in all 55. |
| `resumed` (bool); `resumed_from_round`, `resume_reconciled_records_dropped` (int); `resumed_from_file`, `resume_timestamp_utc` (str) | 6 of 55 | Resume provenance. See [The `resumed` key](#the-resumed-key). |

Two keys that the drivers can write appear in no shipped manifest. `heartbeat_timestamp_utc` goes with status `running`, and `kill_reason` with `killed` or `crashed`. Each is absent from all 55.

Source: `src/raas_marl/final_training/run_logging.py` (`build_run_manifest_payload`, `harness_boundary_flags`, `_VALID_TIERS`, `MANIFEST_VERSION`); `src/raas_marl/mappo_lagrangian/_governance.py` (`boundary_flags`, `_FALSE_BOUNDARY_KEYS`, `utc_timestamp`); `src/raas_marl/final_training/stage25_driver.py` (`stage25_boundary_flags`, `run_stage25_training`, `kill_reason`, `heartbeat_timestamp_utc`); `src/raas_marl/final_training/baseline_driver.py` (`_git_head`); record `results/experiments/*/RUN_MANIFEST.json` (`status`, `tier`, `implementing_dr_ids`)

## The `config` block

`_config_payload` in `stage25_driver.py` builds `config` from the run's `Stage25RunConfig` with `dataclasses.asdict`. It adds the run's `Stage25UpdateConfig` as `stage25_update_config` and the `torch` version as `torch_version`. The baseline driver builds it from `BaselineRunConfig` and adds `stage22_update_config` instead. The table groups the keys directly under `config` by the number of records that hold them.

| present in | keys under `config` (type) | also under `config.stage25_update_config` | absent from | sets |
|---|---|---|---|---|
| 55 of 55 | `entropy_floor`, `lambda_kill_bound` (float); `episodes_per_round`, `heartbeat_every`, `probe_every`, `probe_seed`, `seed`, `update_rounds` (int); `result_parent`, `steps_per_episode` (null); `run_id`, `scenario_names[]`, `tier`, `torch_version` (str); `sample_actions` (bool); `scenario_names` (list) | none | none | Run settings of both drivers: `seed`; the round size (`episodes_per_round`, `steps_per_episode`, null meaning the full environment horizon) and `update_rounds`; the training `scenario_names`; stochastic collection (`sample_actions`); probe and heartbeat cadences (`probe_every`, `heartbeat_every`); the seed of the periodic readiness probes (`probe_seed`), whose logs are not in this repository; the applied-multiplier kill bound; the entropy-collapse flag threshold (`entropy_floor`); `result_parent` (null: the default folder); and `torch_version`, which `_config_payload` adds. |
| 51 of 55 | `stage25_update_config` (dict); `state_save_every` (int) | none | baseline | Training-state save cadence, and the update settings (see the next table). |
| 48 of 55 | `advantage_std_guard_threshold`, `movement_entropy_floor`, `sensing_entropy_floor`, `task_progress_potential_weight` (float); `constraint_warmup_rounds` (int) | `advantage_std_guard_threshold`, `constraint_warmup_rounds`, `movement_entropy_floor`, `sensing_entropy_floor` | baseline; `t1_d4` | Potential-based shaping toward the goal, constraint warmup, the advantage-standardization guard, and the entropy floors. |
| 42 of 55 | `sensing_credit_potential_weight`, `sensing_entropy_target`, `sensing_keepalive_coefficient` (float); `sensing_keepalive_rounds` (int) | `sensing_entropy_target`, `sensing_keepalive_coefficient`, `sensing_keepalive_rounds` | baseline; `t1_d4` to `t1_d6` | Sensing keep-alive, and a sensing-credit potential. |
| 39 of 55 | `sensing_zone_potential_weight` (float) | none | baseline; `t1_d4` to `t1_d7` | The potential for occupying the public risk zone. |
| 36 of 55 | `lambda_integral_cap` (float) | `lambda_integral_cap` | baseline; `t1_d4` to `t1_d8` | Integral cap. |
| 33 of 55 | `entropy_sustain_alpha_cap`, `entropy_sustain_learning_rate`, `lambda_floor`, `movement_entropy_sustain_target`, `sensing_entropy_sustain_target` (float); `entropy_sustain_anneal_rounds`, `entropy_sustain_hold_rounds` (int) | `entropy_sustain_alpha_cap`, `entropy_sustain_anneal_rounds`, `entropy_sustain_hold_rounds`, `entropy_sustain_learning_rate`, `lambda_floor`, `movement_entropy_sustain_target`, `sensing_entropy_sustain_target` | baseline; `t1_d4` to `t1_d9` | Entropy sustain, and `lambda_floor`, a floor on the applied multiplier that holds without arming. |
| 30 of 55 | `arm_hazard_threshold`, `arm_sensing_threshold`, `arm_success_threshold`, `armed_lambda_floor`, `dearm_success_threshold`, `lambda_ceiling` (float); `arm_window_rounds`, `dearm_window_rounds` (int) | `armed_lambda_floor`, `lambda_ceiling` | baseline; `t1_d4` to `t1_d10` | Multiplier ceiling, armed floor, its arm window, and its hatch. |
| 27 of 55 | `sensing_sustain_armed_gate` (bool) | `sensing_sustain_armed_gate` | baseline; `t1_d4` to `t1_d11` | Sensing target armed-gated. |
| 24 of 55 | `hazard_budget`, `midband_recovery_containment_widen`, `midband_recovery_hazard_cap`, `midband_recovery_success_hi`, `midband_recovery_success_lo` (float); `midband_recovery_max_rounds`, `midband_recovery_window_rounds` (int); `movement_recovery_gate` (bool) | `movement_recovery_gate` | baseline; `t1_d4` to `t1_d12` | Hazard budget, and the movement-recovery gate with its `midband_recovery_*` trigger settings. |

The records of `t1_d4` to `t1_d12` lack fields that the shipped `Stage25RunConfig` defines, as the "absent from" column shows; 24 records hold every `Stage25RunConfig` field. The shipped defaults of the fields held by fewer than 51 records leave their levers off, except `hazard_budget`, which defaults to 0.5. Among the 51 records with `stage` `25`, a record that lacks a field does not record the value its run used, with one exception. `hazard_budget` is under `config.stage25_update_config` in 51 of 55 records and directly under `config` in 24 of 55: the 27 records of `t1_d4` to `t1_d12` lack `config.hazard_budget` but hold 0.5 under `config.stage25_update_config.hazard_budget`.

The table below lists the update keys that the table above does not. `cost_estimator` is `episodic_team_mean` in all 51: the undiscounted episodic mean team hazard cost that the paper defined in Sec. III. `dual_step_timing` is `pre_epochs` in all 51: one dual step precedes the policy epochs, except during the constraint warmup (`constraint_warmup_rounds`, 300 in 48 of the 51), when no dual step runs and the multiplier is held at zero.

| present in | keys under `config.stage25_update_config` (type) | sets |
|---|---|---|
| 51 of 55 | `adam_epsilon`, `algorithm.discount_factor`, `algorithm.gae_lambda`, `algorithm.ppo_clip_range`, `hazard_budget`, `initial_integral`, `integral_gain`, `kl_stop_margin`, `learning_rate`, `losses.hazard_cost_coefficient`, `losses.movement_entropy_coefficient`, `losses.reward_entropy_coefficient`, `losses.sensing_cost_coefficient`, `losses.sensing_entropy_coefficient`, `losses.value_loss_coefficient`, `max_grad_norm`, `proportional_gain`, `target_kl` (float); `algorithm`, `losses` (dict); `cost_estimator`, `dual_step_timing` (str); `kl_early_stop`, `standardize_advantages` (bool); `max_update_epochs` (int) | The update constants: PPO settings (`algorithm`, `losses`, learning rate, epochs, gradient-norm clip, KL early stop), the proportional-integral controller's K_P, K_I, and I_0 (`proportional_gain`, `integral_gain`, `initial_integral`), the hazard budget d (`hazard_budget`), and the cost estimator and dual-step timing. |
| 4 of 55 | under `config`: `stage22_update_config`, `stage22_update_config.algorithm`, `stage22_update_config.lagrange`, `stage22_update_config.losses` (dict); `stage22_update_config.algorithm.discount_factor`, `stage22_update_config.algorithm.gae_lambda`, `stage22_update_config.algorithm.ppo_clip_range`, `stage22_update_config.lagrange.hazard_budget`, `stage22_update_config.lagrange.initial_multiplier`, `stage22_update_config.lagrange.learning_rate`, `stage22_update_config.learning_rate`, `stage22_update_config.losses.hazard_cost_coefficient`, `stage22_update_config.losses.movement_entropy_coefficient`, `stage22_update_config.losses.reward_entropy_coefficient`, `stage22_update_config.losses.sensing_cost_coefficient`, `stage22_update_config.losses.sensing_entropy_coefficient`, `stage22_update_config.losses.value_loss_coefficient`, `stage22_update_config.max_grad_norm` (float); `stage22_update_config.max_update_epochs` (int); `stage22_update_config.normalize_advantages` (bool) | The baseline driver's update settings, in place of `stage25_update_config`. |

Source: `src/raas_marl/final_training/stage25_driver.py` (`_config_payload`, `Stage25RunConfig`, `hazard_budget`); `src/raas_marl/final_training/baseline_driver.py` (`_config_payload`, `BaselineRunConfig`); `src/raas_marl/final_training/stage25_update.py` (`Stage25UpdateConfig`, `from_decision_records`); `src/raas_marl/mappo_lagrangian/update.py` (`Stage22UpdateConfig`, `default_stage22_update_config`); record `results/experiments/*/RUN_MANIFEST.json` (`config.hazard_budget`, `config.stage25_update_config.hazard_budget`, `config.stage25_update_config.cost_estimator`, `config.stage25_update_config.constraint_warmup_rounds`); camera-ready, p. 3, Sec. III

## The `runtime_record` block

`stage24a_runtime_reproducibility_record` builds this block when the run starts, from the run's seed, device `cpu`, and dtype `float32`. The drivers request no thread counts, so the block only observes them. The README's [Reproduction environment](../README.md#reproduction-environment) section gives the recorded values for the nine graded runs.

| keys under `runtime_record` (type) | present in | meaning |
|---|---|---|
| `python_version`, `platform`, `platform_machine`, `platform_processor` (str) | 55 of 55 | Interpreter and host strings from `sys.version` and `platform` (not shown). |
| `torch_version` (str); `torch_num_threads`, `torch_num_interop_threads` (int) | 55 of 55 | The `torch` version and the thread counts `torch` reports. |
| `env` (dict); `env.OMP_NUM_THREADS`, `env.MKL_NUM_THREADS` (str); `env.OPENBLAS_NUM_THREADS`, `env.NUMEXPR_NUM_THREADS`, `env.PYTHONHASHSEED` (null) | 55 of 55 | Environment variables at launch; null means not set. `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are `2` in all 55. |
| `device`, `dtype`, `stage` (str); `seed_values` (list); `seed_values[]` (int) | 55 of 55 | `cpu`, `float32`, and `24-A` in all 55; the run's seed. |
| `thread_settings_requested` (dict); `thread_settings_requested.torch_num_threads`, `thread_settings_requested.torch_num_interop_threads` (null); `thread_settings_set` (bool); `thread_settings_policy`, `cpu_thread_settings_set_or_only_observed` (str); `thread_setting_errors` (list) | 55 of 55 | No thread count was requested or set: `thread_settings_set` is `false`, both policy strings are `observed_only`, and `thread_setting_errors` is empty, in all 55. |
| `cross_platform_bitwise_determinism_claimed`, `runtime_metadata_used_as_claim_evidence` (bool) | 55 of 55 | `false`, fixed in the code. |

Source: `src/raas_marl/mappo_lagrangian/stage24_collector.py` (`stage24a_runtime_reproducibility_record`); `src/raas_marl/final_training/stage25_driver.py` (`stage24a_runtime_reproducibility_record`); `src/raas_marl/final_training/baseline_driver.py` (`stage24a_runtime_reproducibility_record`); record `results/experiments/*/RUN_MANIFEST.json` (`runtime_record.thread_settings_policy`, `runtime_record.env.OMP_NUM_THREADS`)

## The `config_sha256` rule

`_config_fingerprint` computes the hash. It removes `torch_version` and `result_parent` from the `config` block, serializes the rest with `deterministic_json_string` (sorted keys, no spaces, ASCII only), and takes the SHA-256 hex digest. Its docstring gives the reasons: `torch_version` is environment provenance, and `result_parent` only locates the run folder. The driver also saves the hash in every training state. On resume it refuses a state whose hash differs from that of the resumed configuration. The baseline driver writes no hash. This command recomputes the hash of every manifest that holds one. It needs the package installed as in the README's [Install and tests](../README.md#install-and-tests) section; run it from the repository root.

```shell
python -c "import glob, hashlib, json; from raas_marl.mappo_lagrangian.artifacts import deterministic_json_string as dj; ms = [json.load(open(p, encoding='utf-8')) for p in sorted(glob.glob('results/experiments/*/RUN_MANIFEST.json'))]; hs = [m for m in ms if 'config_sha256' in m]; ok = sum(hashlib.sha256(dj({k: v for k, v in m['config'].items() if k not in ('torch_version', 'result_parent')}).encode('ascii')).hexdigest() == m['config_sha256'] for m in hs); print(len(ms), 'manifests;', len(hs), 'hold config_sha256;', ok, 'recompute by the rule')"
```

```text
55 manifests; 51 hold config_sha256; 51 recompute by the rule
```

Source: `src/raas_marl/final_training/stage25_driver.py` (`_config_fingerprint`, `_load_latest_training_state`); `src/raas_marl/mappo_lagrangian/artifacts.py` (`deterministic_json_string`); command `python -c "import glob, hashlib, json; from raas_marl.mappo_lagrangian.artifacts import deterministic_json_string as dj; ms = [json.load(open(p, encoding='utf-8')) for p in sorted(glob.glob('results/experiments/*/RUN_MANIFEST.json'))]; hs = [m for m in ms if 'config_sha256' in m]; ok = sum(hashlib.sha256(dj({k: v for k, v in m['config'].items() if k not in ('torch_version', 'result_parent')}).encode('ascii')).hexdigest() == m['config_sha256'] for m in hs); print(len(ms), 'manifests;', len(hs), 'hold config_sha256;', ok, 'recompute by the rule')"`

## The `resumed` key

`run_stage25_training` resumes a run when called with `resume=True`. It loads the latest saved training state and adds five keys to the manifest. `resumed` is always `true`, and `resumed_from_round` is the number of rounds completed in that state. `resumed_from_file` is the state's file name, and `resume_timestamp_utc` is the time of the resume. `resume_reconciled_records_dropped` counts the lines dropped from `training_curve.jsonl`, `episode_log.jsonl`, and `probe_log.jsonl`: the records of rounds the resume runs again, plus a partial last line left by the interrupted process, if any; these logs are not in this repository ([The raw training logs](../README.md#the-raw-training-logs)). A record without these keys was never resumed. The resumed process sets `git_head`, `pid`, and `launch_timestamp_utc` anew. So in the 6 records marked `resumed`, those keys describe the last resume, not the first launch. This command lists them. It needs only the standard library; run it from the repository root, or it finds no record and prints `0 of 0`:

```shell
python -c "import glob, json; ms = [json.load(open(p, encoding='utf-8')) for p in sorted(glob.glob('results/experiments/*/RUN_MANIFEST.json'))]; rs = [m for m in ms if 'resumed' in m]; print('\n'.join(' '.join(str(m[k]) for k in ('run_id', 'resumed', 'resumed_from_round', 'resumed_from_file', 'resume_reconciled_records_dropped')) for m in rs)); print(sum(m['launch_timestamp_utc'] == m['resume_timestamp_utc'] for m in rs), 'of', len(rs), 'resumed records have launch_timestamp_utc equal to resume_timestamp_utc')"
```

```text
t1_d11_s126 True 1500 state_round_001500.pt 3657
t1_d11_s127 True 1250 state_round_001250.pt 1157
t1_d11_s128 True 1750 state_round_001750.pt 119
t1_d16_s141 True 1250 state_round_001250.pt 1939
t1_d16_s142 True 1500 state_round_001500.pt 884
t1_d16_s143 True 1250 state_round_001250.pt 3271
6 of 6 resumed records have launch_timestamp_utc equal to resume_timestamp_utc
```

Source: `src/raas_marl/final_training/stage25_driver.py` (`run_stage25_training`, `_load_latest_training_state`, `_reconcile_resumed_log`, `resumed_from_file`); command `python -c "import glob, json; ms = [json.load(open(p, encoding='utf-8')) for p in sorted(glob.glob('results/experiments/*/RUN_MANIFEST.json'))]; rs = [m for m in ms if 'resumed' in m]; print('\n'.join(' '.join(str(m[k]) for k in ('run_id', 'resumed', 'resumed_from_round', 'resumed_from_file', 'resume_reconciled_records_dropped')) for m in rs)); print(sum(m['launch_timestamp_utc'] == m['resume_timestamp_utc'] for m in rs), 'of', len(rs), 'resumed records have launch_timestamp_utc equal to resume_timestamp_utc')"`

## Grade record: top level and surfaces

Each of the 27 files holds one object whose only key is `grade`. `grade_policy` builds its value from a policy and a seed. On each of the ten fork surfaces it rolls the policy once as it is. It rolls it again with the revealed-information channel zeroed for the whole episode: the ablation behind what the paper called the counterfactual own-ablation witness. It also rolls the scripted no-sense policy as a reported cross-check. `derive_checkpoint` reads `tightened_verdict`, `witness_surfaces`, and, on the two validation surfaces, `team_success` and `hazard_entry_count`. The README's [What this repository contains](../README.md#what-this-repository-contains) section compares a re-grade from a shipped checkpoint with a stored record.

| keys (type) | present in | meaning |
|---|---|---|
| `grade` (dict) | 27 of 27 | The object `grade_policy` returns. It is the only top-level key. |
| `label` (str) | 27 of 27 | Run and graded round, for example `t1_retain_s147@r4000`. |
| `seed` (int); `family` (str) | 27 of 27 | Rollout seed (`9001` in all 27) and surface family (`fork` in all 27). |
| `surface_count`, `surfaces_with_success` (int) | 27 of 27 | Surfaces graded (`10` in all 27) and surfaces the policy solved. |
| `c2_adapt_episode_fraction` (float) | 27 of 27 | Share of surfaces whose episode set the adapt flag (`adapt_fired`). |
| `per_surface`, `per_surface.<surface>` (dict) | 27 of 27 | One entry per surface. See the next table. |
| `pooled_selectivity` (dict) | 27 of 27 | `selectivity_stats` over the decisions of all ten surfaces. |
| `composite`, `tightened_composite` (dict) | 27 of 27 | The pooled verdict (`composite_c1_verdict`) and the own-ablation verdict (`tightened_composite_c1_verdict`). |

The ten keys of `per_surface` are the surface names: `risk_fork_curriculum_r0r3_lower`, `risk_fork_curriculum_r0r3_upper`, `risk_fork_curriculum_r0r7_lower`, `risk_fork_curriculum_r0r7_upper`, `risk_fork_curriculum_r4r7_lower`, `risk_fork_curriculum_r4r7_upper`, `risk_fork_readiness_lower`, `risk_fork_readiness_upper`, `risk_fork_train_lower`, `risk_fork_train_upper`.

| keys under `per_surface.<surface>` (type) | present in | meaning |
|---|---|---|
| `group` (str); `hidden_hazard_cells`, `hidden_hazard_cells[]` (list); `hidden_hazard_cells[][]` (int) | 27 of 27 | The surface's group (`training`, `curriculum`, or `readiness`) and its hidden hazard cells as `[row, column]` pairs. |
| `steps`, `hazard_entry_count` (int); `team_success`, `adapt_fired` (bool); `sensing_cost_sum` (float) | 27 of 27 | The intact episode: its length, team success, hazard entries, and sensing cost, and whether the adapt flag fired: a forward pass with the reveal zeroed chose a different movement. |
| `own_blind_hazard_entry_count`, `own_blind_hazard_avoided` (int); `own_blind_team_success` (bool) | 27 of 27 | The own-ablation re-roll: the same policy with the revealed-information channel zeroed for the whole episode. `own_blind_hazard_avoided` is its hazard entries minus the intact episode's. |
| `no_sense_hazard_entry_count`, `hazard_avoided_vs_no_sense` (int); `no_sense_team_success` (bool) | 27 of 27 | The scripted no-sense policy on the same surface, a reported cross-check. `hazard_avoided_vs_no_sense` is its hazard entries minus the intact episode's. |
| `selectivity` (dict) | 27 of 27 | `selectivity_stats` of the intact episode's decisions. See the next table. |

Source: `docs/evidence/c1_selectivity_harness.py` (`grade_policy`, `rollout_episode`, `_blind_reveal_observations`, `_fork_training_readiness_variants`, `PROBE_SEED`); `src/raas_marl/final_training/baseline_driver.py` (`greedy_model_policy_fn`); `instruments/grade_composer.py` (`derive_checkpoint`, `VALIDATION_SURFACES`); record `results/experiments/*/checkpoint_grades/grade_r*.json` (`grade.label`, `grade.per_surface`); camera-ready, p. 3, Sec. III

## Grade record: selectivity and verdicts

Both `per_surface.<surface>.selectivity` and `pooled_selectivity` hold these keys:

| keys (type) | present in | meaning |
|---|---|---|
| `n_decisions`, `total_sense` (int); `overall_sense_rate` (float) | 27 of 27 | Decisions (one per agent per step), sensing decisions, and their ratio. |
| `contrast_near_far`, `contrast_near_hazard`, `contrast_strict_in_zone` (dict) | 27 of 27 | Three contrasts. The relevant decisions are those within one step of the public risk zone (`near_far`), inside it (`strict_in_zone`), or within one step of the hidden hazard (`near_hazard`). |
| `contrast_near_far.delta`, `contrast_near_far.p_sense_nonrelevant`, `contrast_near_far.p_sense_relevant`, `contrast_near_hazard.p_sense_nonrelevant`, `contrast_strict_in_zone.p_sense_nonrelevant` (float); `contrast_near_far.n_nonrelevant`, `contrast_near_far.n_relevant`, `contrast_near_far.sense_nonrelevant`, `contrast_near_far.sense_relevant`, `contrast_near_hazard.n_nonrelevant`, `contrast_near_hazard.n_relevant`, `contrast_near_hazard.sense_nonrelevant`, `contrast_near_hazard.sense_relevant`, `contrast_strict_in_zone.n_nonrelevant`, `contrast_strict_in_zone.n_relevant`, `contrast_strict_in_zone.sense_nonrelevant`, `contrast_strict_in_zone.sense_relevant` (int); `contrast_near_hazard.delta`, `contrast_near_hazard.p_sense_relevant`, `contrast_strict_in_zone.delta`, `contrast_strict_in_zone.p_sense_relevant` (float or null per surface, float pooled) | 27 of 27 | Per contrast, the relevant and other decisions: counts, sensing counts, sensing rates, and `delta`, the first rate minus the second (null when a side is empty). |
| `point_biserial_r_zone`, `point_biserial_r_hazard`, `mean_dist_at_sense` (float or null); `perm_p_near_far`, `mean_dist_at_nonsense` (float); `far_axis_estimable` (bool) | 27 of 27 | Correlations of sensing with nearness to the zone and to the hazard (null without variance), a permutation p-value for the `near_far` delta, mean distances to the zone, and whether both near and far decisions exist. |
| `verdict` (str) | 27 of 27 | The spatial verdict of `selectivity_stats`. |

The two verdict objects, `composite` and `tightened_composite`, hold these keys:

| keys (type) | present in | meaning |
|---|---|---|
| `composite_verdict` (str); `informative` (bool); `policy_hazard_entries`, `no_sense_hazard_entries`, `total_sense`, `n_decisions`, `surfaces_with_success`, `surface_count` (int); `overall_sense_rate`, `c2_adapt_episode_fraction` (float) | 27 of 27 | `composite`: the pooled verdict, and whether the policy's pooled hazard entries are below the no-sense policy's (`informative`). |
| `tightened_verdict`, `legacy_pooled_composite`, `eval_ablation`, `causal_conjunct` (str); `selectivity_ceiling`, `overall_sense_rate` (float); `total_sense`, `n_decisions`, `surfaces_sensed_on`, `surfaces_sensed_on_with_success`, `witness_count`, `causal_surface_count` (int) | 27 of 27 | `tightened_composite`: the verdict, a copy of `composite_verdict`, the selectivity ceiling `SELECTIVITY_CEILING` (`0.5` in all 27), two fixed descriptions, and counts. |
| `witness_surfaces`, `causal_surfaces`, `witness_groups` (list) | 27 of 27 | Surfaces that complete the causal chain (`causal_surfaces`), those of them within the ceiling (`witness_surfaces`), and the witnesses' groups. Each list is empty in 6 of the 27, which then have no element paths. |
| `witness_surfaces[]`, `causal_surfaces[]` (dict); `witness_groups[]`, `witness_surfaces[].surface`, `witness_surfaces[].group`, `causal_surfaces[].surface`, `causal_surfaces[].group` (str); `witness_surfaces[].sense`, `witness_surfaces[].own_blind_hazard_avoided`, `witness_surfaces[].hazard_avoided_vs_no_sense`, `causal_surfaces[].sense`, `causal_surfaces[].own_blind_hazard_avoided`, `causal_surfaces[].hazard_avoided_vs_no_sense` (int); `witness_surfaces[].sense_rate`, `causal_surfaces[].sense_rate` (float); `witness_surfaces[].adapt_fired`, `causal_surfaces[].adapt_fired` (bool) | 21 of 27 | One entry per listed surface, built from its `per_surface` entry. |
| `per_component_report`, `per_component_report[].failed_conjuncts` (list); `per_component_report[]` (dict); `per_component_report[].surface`, `per_component_report[].group`, `per_component_report[].failed_conjuncts[]` (str); `per_component_report[].success`, `per_component_report[].senses`, `per_component_report[].within_ceiling`, `per_component_report[].adapt_fired`, `per_component_report[].causal`, `per_component_report[].is_witness` (bool); `per_component_report[].sense_rate` (float); `per_component_report[].own_blind_hazard_avoided` (int); `per_component_report[].in_zone_sense_rate` (float or null) | 27 of 27 | One entry per surface, one boolean per conjunct. `is_witness` is true when the surface succeeds, senses on more than zero and at most half of its decisions, sets the adapt flag, and has `own_blind_hazard_avoided` above zero. |
| `component_pass_counts` (dict); `component_pass_counts.success`, `component_pass_counts.senses`, `component_pass_counts.within_ceiling`, `component_pass_counts.adapt`, `component_pass_counts.causal`, `component_pass_counts.witness`, `component_pass_counts.surface_count` (int) | 27 of 27 | How many surfaces pass each conjunct. |
| `reconciliation` (dict); `reconciliation.rederived_verdict`, `reconciliation.note` (str); `reconciliation.reconciles` (bool) | 27 of 27 | A drift guard: the verdict derived again from the conjunct counts. The harness raises an error rather than write `false`, so `reconciles` is `true` in all 27; it is not an independent check. |
| `positionally_induced` (dict); `positionally_induced.suspected` (bool); `positionally_induced.max_in_zone_sense_rate_on_success`, `positionally_induced.floor` (float); `positionally_induced.causal_witness_count`, `positionally_induced.causal_surface_count` (int); `positionally_induced.reason` (str) | 27 of 27 | A diagnostic flag: `suspected` is true when, on a surface the policy solves, it senses in the zone at a rate of at least `floor`, and no surface completes the causal chain. `floor` is `0.1` in all 27, and `suspected` is `true` in 2. It never changes the verdict. |

Source: `docs/evidence/c1_selectivity_harness.py` (`selectivity_stats`, `composite_c1_verdict`, `tightened_composite_c1_verdict`, `SELECTIVITY_CEILING`, `POSITIONALLY_INDUCED_INZONE_FLOOR`); record `results/experiments/*/checkpoint_grades/grade_r*.json` (`grade.tightened_composite.tightened_verdict`, `grade.composite.composite_verdict`, `grade.tightened_composite.positionally_induced.suspected`, `grade.tightened_composite.reconciliation.reconciles`)

## The grade-record manifest

`results/checkpoint_grades_manifest.json` lists, per run, its `seed`, its `checkpoint_rounds`, and under `records` one entry per round: the shipped file name (`file`), its size in bytes (`bytes`), its SHA-256 (`sha256`), and the name it had before release (`source_filename`). This command checks the records against it. It needs only the standard library; run it from the repository root:

```shell
python -c "import hashlib, json, pathlib; m = json.load(open('results/checkpoint_grades_manifest.json', encoding='utf-8')); rs = [(r, v) for r, e in m.items() for v in e['records'].values()]; b = [pathlib.Path('results/experiments', r, 'checkpoint_grades', v['file']).read_bytes() for r, v in rs]; print(len(m), 'runs;', len(rs), 'records;', sum(len(x) == v['bytes'] and hashlib.sha256(x).hexdigest() == v['sha256'] for x, (r, v) in zip(b, rs)), 'match their bytes and sha256;', sum(list(json.loads(x)) == ['grade'] for x in b), 'hold only the key grade')"
```

```text
9 runs; 27 records; 27 match their bytes and sha256; 27 hold only the key grade
```

Source: `results/checkpoint_grades_manifest.json` (`t1_retain_s147.seed`, `t1_retain_s147.checkpoint_rounds`, `t1_retain_s147.records.r004000.file`, `t1_retain_s147.records.r004000.bytes`, `t1_retain_s147.records.r004000.sha256`, `t1_retain_s147.records.r004000.source_filename`); command `python -c "import hashlib, json, pathlib; m = json.load(open('results/checkpoint_grades_manifest.json', encoding='utf-8')); rs = [(r, v) for r, e in m.items() for v in e['records'].values()]; b = [pathlib.Path('results/experiments', r, 'checkpoint_grades', v['file']).read_bytes() for r, v in rs]; print(len(m), 'runs;', len(rs), 'records;', sum(len(x) == v['bytes'] and hashlib.sha256(x).hexdigest() == v['sha256'] for x, (r, v) in zip(b, rs)), 'match their bytes and sha256;', sum(list(json.loads(x)) == ['grade'] for x in b), 'hold only the key grade')"`

Checked against the code at commit `6a5e22d` on 2026-09-26 UTC.

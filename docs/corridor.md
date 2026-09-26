# The control corridor: what it bounds, and where each Table I lever lives

The paper called the levers of its Table I the control corridor (Sec. III); the code has no identifier for it. This page gives what the levers act on, where each lives, what "off" means for each, and the values that the 55 `RUN_MANIFEST.json` files carry.

## The dual step that the corridor bounds

The paper described the multiplier as a projected dual driven by a proportional-integral controller. In the code that is `PIController`, and each update round after the constraint warmup makes one dual step, `PIController.update`:

| quantity | paper (Sec. III) | code |
|---|---|---|
| episodic cost | `J_C_hat`, the undiscounted episodic mean team hazard cost | `episodic_team_hazard_cost`: the sum of `hazard_cost` over valid agent steps, divided by the episode count |
| error | `e_k = J_C_hat - d` | `error`: `observed_episodic_cost` minus `budget` |
| dual integral | `I_k = max(0, I_(k-1) + K_I * e_k)` | `integral`: the same, then clamped to `integral_cap` when that is set (`integral_cap` is set from the configuration field `lambda_integral_cap`) |
| applied multiplier | `lambda_k = max(0, K_P * e_k + I_k)` | `lambda_applied`: the larger of a floor term and `K_P * e_k + I_k`, then clamped to `lambda_ceiling` when that is set |

The floor term is `armed_lambda_floor` in a round whose step gets `floor_armed=True`, else `lambda_floor`. Each step also returns `lambda_pre_clamp`, the value before the ceiling, and `ceiling_engaged`.

The paper reported `K_P = 0.25`, `K_I = 0.05`, `I_0 = 0.1`, `d = 0.25`, and one dual step before the policy epochs. `Stage25UpdateConfig` holds them as `proportional_gain`, `integral_gain`, `initial_integral`, `hazard_budget`, and `dual_step_timing` value `pre_epochs`; its `cost_estimator` value `episodic_team_mean` selects `J_C_hat`. Under `config.stage25_update_config`, 51 records carry `proportional_gain` 0.25, `integral_gain` 0.05, and `initial_integral` 0.1. The same 51 carry `dual_step_timing` `pre_epochs` and `cost_estimator` `episodic_team_mean`. The 4 records of `b101_stoch` to `b104_greedy` have no `stage25_update_config`. The budget is covered under Table I below.

During the constraint warmup, while the round index is below `constraint_warmup_rounds`, the update skips the dual step, applies a multiplier of 0.0, and leaves the integral unchanged. A fresh run therefore holds the integral at `initial_integral`, as the paper reported. `Stage25UpdateConfig` rejects the ceiling, the armed floor, and a positive `lambda_floor` unless `dual_step_timing` is `pre_epochs`.

Source: code `src/raas_marl/final_training/stage25_update.py` (`PIController`, `PIControllerStep`, `episodic_team_hazard_cost`, `Stage25UpdateConfig`, `stage25_ppo_lagrangian_update`, `make_controller`, `lambda_integral_cap`); code `src/raas_marl/mappo_lagrangian/lagrange.py` (`LagrangeMultiplier`); record `results/experiments/*/RUN_MANIFEST.json` (`config.stage25_update_config.proportional_gain`, `config.stage25_update_config.integral_gain`, `config.stage25_update_config.initial_integral`, `config.stage25_update_config.dual_step_timing`, `config.stage25_update_config.cost_estimator`); paper camera-ready, p. 3, Sec. III

## The armed floor, the hatch, and the ceiling

The paper reported that the floor arms only after the target behavior is detected (Sec. III). The driver decides when, in `_ArmedFloorState`; the controller only applies the value.

- Arming. From round `constraint_warmup_rounds` on, each round adds its success rate, mean episodic hazard, and mean sensing rate to a trailing window (`_round_behavior_aggregates`, the same values the training curve records). While unarmed and once the window holds `arm_window_rounds` rounds, the floor arms if the window means meet all three thresholds: success at least `arm_success_threshold`, hazard at most `arm_hazard_threshold`, and sensing at least `arm_sensing_threshold`. The floor applies in the round that arms it. With a warmup of 300 and a window of 50, the earliest round that can arm is round index 349.
- The hatch. While armed, if the last `dearm_window_rounds` rounds have mean success at most `dearm_success_threshold`, the floor releases. The same rule can arm it again. Each arm and release is logged in `events`.
- The ceiling. `lambda_ceiling` bounds the applied multiplier only. The integral has its own bound, `integral_cap`, and can sit above the ceiling.
- Scope. `Stage25RunConfig` rejects `armed_lambda_floor` unless every scenario is a fork training or curriculum scenario, both mirrors of each pair train with equal episodes per round, and there are at most two pairs. The paper reported that the corridor's values were tuned on the single-pair fork, and that the arm can false-fire on off-fork maps, so it is scoped to the fork family (Sec. III).

Run the commands on this page from the repository root, after the steps in the README's [Install and tests](../README.md#install-and-tests). This command steps the controller at the Table I values. It needs PyTorch; the output shown is the last line, an excerpt, since without NumPy a PyTorch warning comes first.

```shell
python -c "from raas_marl.final_training.stage25_update import PIController as P; c = P(integral=0.0, budget=0.25, integral_cap=20.0, lambda_ceiling=5.0, armed_lambda_floor=3.0); s = P(integral=20.0, budget=0.25, integral_cap=20.0, lambda_ceiling=5.0, armed_lambda_floor=3.0).update(observed_episodic_cost=2.0); print(c.update(observed_episodic_cost=0.0).lambda_applied, c.update(observed_episodic_cost=0.0, floor_armed=True).lambda_applied, s.integral, s.lambda_pre_clamp, s.lambda_applied)"
```

```text
0.0 3.0 20.0 20.4375 5.0
```

The first two values are one step at zero cost from a zero integral: 0.0 unarmed, 3.0 armed. The other three are one step from an integral at the cap with a cost of 2.0, an error of 1.75: the integral stays at 20.0, `lambda_pre_clamp` is 20.4375, and the ceiling applies 5.0.

Source: code `src/raas_marl/final_training/stage25_driver.py` (`_ArmedFloorState`, `_round_behavior_aggregates`, `Stage25RunConfig`); code `src/raas_marl/final_training/stage25_update.py` (`PIController`, `PIControllerStep`); command `python -c "from raas_marl.final_training.stage25_update import PIController as P; c = P(integral=0.0, budget=0.25, integral_cap=20.0, lambda_ceiling=5.0, armed_lambda_floor=3.0); s = P(integral=20.0, budget=0.25, integral_cap=20.0, lambda_ceiling=5.0, armed_lambda_floor=3.0).update(observed_episodic_cost=2.0); print(c.update(observed_episodic_cost=0.0).lambda_applied, c.update(observed_episodic_cost=0.0, floor_armed=True).lambda_applied, s.integral, s.lambda_pre_clamp, s.lambda_applied)"`; paper camera-ready, p. 3, Sec. III

## What "off by default" means for each lever

The paper reported that every corridor lever is off by default and byte-identical when off (Sec. III, Table I). In the code, a lever is off at the default of its `Stage25RunConfig` field. The driver starts from `Stage25UpdateConfig.from_decision_records`, with the levers off and a budget of 0.5, and copies the run configuration's lever fields into it (`run_stage25_training`).

| lever | `Stage25RunConfig` field | default | what the default does |
|---|---|---|---|
| multiplier ceiling | `lambda_ceiling` | `None` | no clamp on the applied multiplier |
| armed floor and hatch | `armed_lambda_floor`, `arm_window_rounds`, `dearm_window_rounds`, and the four thresholds | `None`, 0, 0, `None` | no `_ArmedFloorState` is built, so no round is floored |
| entropy sustain | `entropy_sustain_learning_rate`, `movement_entropy_sustain_target`, `sensing_entropy_sustain_target` | 0.0, `None`, `None` | `make_entropy_controller` returns `None` |
| integral cap | `lambda_integral_cap` | `None` | no cap on the integral |
| constraint warmup | `constraint_warmup_rounds` | 0 | the dual step runs from round 0 |
| hazard budget | `hazard_budget` | 0.5 | a budget of 0.5; values above 0.5 are rejected |

The hazard budget has no off value: its default, 0.5, is the budget in `from_decision_records`. This command prints the defaults of the fields it names, in order. It needs the install and PyTorch; the output shown is the last line, an excerpt:

```shell
python -c "from raas_marl.final_training.stage25_driver import Stage25RunConfig; c = Stage25RunConfig(run_id='demo', seed=101, update_rounds=1); print(c.lambda_ceiling, c.armed_lambda_floor, c.arm_window_rounds, c.dearm_window_rounds, c.movement_entropy_sustain_target, c.lambda_integral_cap, c.constraint_warmup_rounds, c.hazard_budget, c.lambda_floor)"
```

```text
None None 0 0 None None 0 0.5 0.0
```

Source: code `src/raas_marl/final_training/stage25_driver.py` (`Stage25RunConfig`, `run_stage25_training`, `_ArmedFloorState`); code `src/raas_marl/final_training/stage25_update.py` (`Stage25UpdateConfig`, `from_decision_records`, `make_entropy_controller`); command `python -c "from raas_marl.final_training.stage25_driver import Stage25RunConfig; c = Stage25RunConfig(run_id='demo', seed=101, update_rounds=1); print(c.lambda_ceiling, c.armed_lambda_floor, c.arm_window_rounds, c.dearm_window_rounds, c.movement_entropy_sustain_target, c.lambda_integral_cap, c.constraint_warmup_rounds, c.hazard_budget, c.lambda_floor)"`; paper camera-ready, p. 3, Sec. III; paper camera-ready, p. 4, Table I

## Table I against the records

Each row gives the lever's value in Table I, its field, and the values that field takes under `config` in the 55 `RUN_MANIFEST.json` files. The ceiling, floor, sustain-target, cap, and warmup fields also appear under `config.stage25_update_config`, with the same values and counts.

| lever | Table I | field | records |
|---|---|---|---|
| Multiplier ceiling | 5.0 | `lambda_ceiling` | 5.0 in 30, absent from 25 |
| Armed floor: the floor | 3.0 | `armed_lambda_floor` | 3.0 in 30, absent from 25 |
| Armed floor: the window | 50 rounds | `arm_window_rounds` | 50 in 30, absent from 25 |
| Armed floor: success at least | 0.8 | `arm_success_threshold` | 0.8 in 30, absent from 25 |
| Armed floor: hazard at most | 0.35 | `arm_hazard_threshold` | 0.35 in 30, absent from 25 |
| Armed floor: sensing at least | 0.05 | `arm_sensing_threshold` | 0.05 in 30, absent from 25 |
| Hatch: rounds | 200 | `dearm_window_rounds` | 200 in 30, absent from 25 |
| Hatch: success at most | 0.1 | `dearm_success_threshold` | 0.1 in 30, absent from 25 |
| Entropy sustain: movement | 0.5 nats | `movement_entropy_sustain_target` | 0.5 in 33, absent from 22 |
| Entropy sustain: sensing | 0.3 nats | `sensing_entropy_sustain_target` | 0.3 in 33, absent from 22 |
| Integral cap | 20 | `lambda_integral_cap` | 20.0 in 36, absent from 19 |
| Constraint warmup | 300 rounds | `constraint_warmup_rounds` | 300 in 48, absent from 7 |
| Hazard budget | 0.25 | `hazard_budget` | 0.25 in 24, absent from 31 |

Every value present under `config` for a field in this table equals the Table I value. Apart from the hazard budget, covered below, four sets of records lack fields:

- The 25 without the ceiling, floor, and hatch fields are those of `b101_stoch` to `b104_greedy` and `t1_d4` to `t1_d10`.
- The 22 without the sustain targets are the same, less `t1_d10`.
- The 19 without the cap are those 22, less `t1_d9`.
- The 7 without the warmup are those of `b101_stoch` to `b104_greedy` and `t1_d4`.

The arm and hatch fields live in the run configuration only: `config.stage25_update_config` holds them in none of the 55 records.

The hazard budget differs from Table I, as the README's [paper map](../README.md#the-paper-map) says. `config.stage25_update_config.hazard_budget` is 0.25 in 24 records, 0.5 in 27, and absent from the 4 records of `b101_stoch` to `b104_greedy`. The 0.5 records are those of `t1_d4` to `t1_d12`, which carry no `config.hazard_budget`. [`docs/PAPER_MAP.md`](PAPER_MAP.md#table-i) gives the same counts.

Source: record `results/experiments/*/RUN_MANIFEST.json` (`config.lambda_ceiling`, `config.armed_lambda_floor`, `config.arm_window_rounds`, `config.arm_success_threshold`, `config.arm_hazard_threshold`, `config.arm_sensing_threshold`, `config.dearm_window_rounds`, `config.dearm_success_threshold`, `config.movement_entropy_sustain_target`, `config.sensing_entropy_sustain_target`, `config.lambda_integral_cap`, `config.constraint_warmup_rounds`, `config.hazard_budget`, `config.stage25_update_config.hazard_budget`); paper camera-ready, p. 4, Table I

## Beyond Table I: the sustain schedule, two gates, and an unconditional floor

Entropy sustain (`EntropySustainController`) keeps one dual per action factor, `alpha_k = min(alpha_cap, max(0, alpha_(k-1) + lr * (target - measured entropy)))`. The target, `entropy_sustain_target`, is the base value before `hold_rounds`, falls linearly to 0 by `anneal_end_rounds`, and is 0 after. The paper reported, for the movement target, a hold of 800 rounds, an anneal to zero by round 1200, a dual learning rate of 0.02, and a cap of 2.0 (Sec. IV). In the records, `entropy_sustain_hold_rounds` is 800 in 33, `entropy_sustain_anneal_rounds` is 1200 in 30 and 2000 in 3 (the `t1_d10` runs), `entropy_sustain_learning_rate` is 0.02 in 33, and `entropy_sustain_alpha_cap` is 2.0 in 33.

- The sensing target is armed-gated (`sensing_sustain_armed_gate`): the base target while the floor is unarmed, and 0.0 with its dual pinned to 0.0 while armed, as the paper reported. `sensing_sustain_armed_gate` is true in the 27 records of `t1_d12` to `t1_d16`, `anchor_rerun`, `t1_retain`, and `t1_seedscale`, and absent from the other 28.
- The movement-recovery gate (`movement_recovery_gate`) holds the movement target at its base value while the driver's recovery latch in `_ArmedFloorState` is engaged; outside recovery, after the anneal ends, the movement dual is pinned to 0.0. `movement_recovery_gate` is true in the 24 records of `t1_d13` to `t1_d16`, `anchor_rerun`, `t1_retain`, and `t1_seedscale`, and absent from the other 31, among them `t1_d11` and `t1_d12`. The six records of `t1_d11` and `t1_d12` also lack `movement_recovery_gate` under `config.stage25_update_config`, where they hold `hazard_budget`. As [`docs/schemas.md`](schemas.md#the-config-block) says, among the 51 records with `stage` `25`, a record that lacks a field does not record the value its run used. `hazard_budget` is the one exception. So the records of `t1_d11` and `t1_d12` do not show whether those runs used the gate. The paper reported the gate as enabled in every reported run (Sec. IV), and counted armed seeds of `t1_d11` and `t1_d12` in Sec. V-A, as [`docs/PAPER_MAP.md`](PAPER_MAP.md#headline-numbers) records.
- The unconditional floor (`lambda_floor`) is not a Table I lever. It sets the floor term in every round, armed or not. The code rejects a positive `lambda_floor` together with `armed_lambda_floor`. `config.lambda_floor` is 0.0 in 30 records, 3.0 in 3 (the `t1_d10` runs, which carry no armed floor or ceiling), and absent from 22.

Source: code `src/raas_marl/final_training/stage25_update.py` (`EntropySustainController`, `entropy_sustain_target`, `sensing_armed_gate`, `movement_recovery_gate`, `PIController`, `lambda_floor`); code `src/raas_marl/final_training/stage25_driver.py` (`_ArmedFloorState`, `_observe_recovery`); record `results/experiments/*/RUN_MANIFEST.json` (`config.entropy_sustain_hold_rounds`, `config.entropy_sustain_anneal_rounds`, `config.entropy_sustain_learning_rate`, `config.entropy_sustain_alpha_cap`, `config.sensing_sustain_armed_gate`, `config.movement_recovery_gate`, `config.stage25_update_config.movement_recovery_gate`, `config.stage25_update_config.hazard_budget`, `stage`, `config.lambda_floor`); paper camera-ready, p. 4, Sec. IV; paper camera-ready, p. 5, Sec. V-A

## The load-bearing floor in the shipped checkpoints

The paper reported that on the holding seeds the tail multiplier rests at the armed floor of 3.0 with the dual integral at zero (Sec. V-A, Fig. 3). The 27 shipped checkpoints store `controller` (the gains, `integral`, `budget`, and the lever constants) and `arm_state` (`armed`, `events`, the window buffers, and the recovery state), not the applied multiplier. This command tallies them by whether `integral` is 0.0 and whether `armed` is true. It needs PyTorch; the output shown is the last line. Run from a directory other than the repository root, it finds no file and prints `Counter()`:

```shell
python -c "import collections, glob, torch; print(collections.Counter((d['controller']['integral'] == 0.0, d['arm_state']['armed']) for d in (torch.load(p, weights_only=True) for p in sorted(glob.glob('results/experiments/*/state_round_*.pt')))))"
```

```text
Counter({(True, True): 18, (False, False): 6, (True, False): 2, (False, True): 1})
```

In 18 of the 27, the integral is 0.0 and the floor is armed. They are the 15 checkpoints of the five seeds that the README's paper map names for Fig. 3, and the three of seed 155. Seed 155's `bar_c4_pass` is false, so this pattern is not specific to the seeds that hold the task-success bar. The checkpoints do not store the applied multiplier, so this tally does not check the value 3.0 that the paper reported, as the README's [paper map](../README.md#the-paper-map) says. As of 2026-09-26 UTC, the commits the runs were launched from are not in this repository ([What this repository contains](../README.md#what-this-repository-contains)).

Source: code `src/raas_marl/final_training/stage25_driver.py` (`_save_training_state`, `_ArmedFloorState`, `run_stage25_training`); code `src/raas_marl/final_training/stage25_update.py` (`PIController`); record `results/held2_conjuncts.json` (`per_seed.155.bar_c4_pass`); record `results/experiments/t1_seedscale_s155/state_round_*.pt` (`controller.integral`, `arm_state.armed`); command `python -c "import collections, glob, torch; print(collections.Counter((d['controller']['integral'] == 0.0, d['arm_state']['armed']) for d in (torch.load(p, weights_only=True) for p in sorted(glob.glob('results/experiments/*/state_round_*.pt')))))"`; paper camera-ready, p. 5, Sec. V-A, Fig. 3

Checked against the code at commit `6a5e22d` on 2026-09-26 UTC.

# The environment and the risk fork

This page describes the grid environment in [`grid_environment.py`](../src/raas_marl/environments/active_sensing/grid_environment.py) and the risk-fork scenarios that the paper trained and graded on. It covers the grid, the two agents, the actions, the hidden hazard, reward and cost, and how the code builds each scenario pair.

## Running the commands on this page

The commands need the package `raas_marl` to be importable. The install in [Install and tests](../README.md#install-and-tests) does that. Without it, Python stops with `ModuleNotFoundError: No module named 'raas_marl'`. That install also installs PyTorch 2.5.1, pinned in `requirements.txt`; `pyproject.toml` lists `torch>=2.5.1` as a dependency. The commands on this page import no third-party package. `grid_environment.py` and `stage24_diagnostics.py` do not import `torch`; the observation encoding in `tensor_adapter.py` loads it when it encodes an observation. This command builds and steps both anchor scenarios through the fork check, then prints whether `torch` was loaded:

```shell
python -c "import sys; from raas_marl.environments.active_sensing.stage24_diagnostics import verify_fork_family_forces_sensing as v; v(('risk_fork_train_lower', 'risk_fork_train_upper')); print('torch' in sys.modules)"
```

```text
False
```

Source: `pyproject.toml` (`dependencies`, `package-dir`); `requirements.txt` (`torch`); `src/raas_marl/environments/active_sensing/tensor_adapter.py` (`torch_required`); `src/raas_marl/environments/active_sensing/stage24_diagnostics.py` (`verify_fork_family_forces_sensing`); command `python -c "import sys; from raas_marl.environments.active_sensing.stage24_diagnostics import verify_fork_family_forces_sensing as v; v(('risk_fork_train_lower', 'risk_fork_train_upper')); print('torch' in sys.modules)"`

## The grid, the agents, and one episode

An episode is played on an 8 by 8 grid by two agents, `agent_0` and `agent_1` (`STAGE23_AGENT_NAMES`). A cell is a (row, column) pair, counted from 0. The configuration `Stage23EnvironmentConfig` rejects any other number of agents. The environment samples no randomness: the same scenario and actions give the same episode.

Both agents act at the same time, on every step. Two agents may share a cell, because collision is not modeled. An episode ends in team success on the first step on which both agents stand on a goal cell. If the team has not succeeded after `max_steps` steps, 32 by default, the episode is truncated.

The training collection builds each episode's environment with `make_default_stage23c_environment`. That function passes the scenario name, the grid size, a seed, and, if one is given, a horizon; every other value is a default. `config.steps_per_episode` is null in all 55 run manifests. In the shipped Stage 25 driver, null selects the 32-step default. This command prints the configuration for one fork scenario:

```shell
python -c "from raas_marl.environments.active_sensing.grid_environment import Stage23EnvironmentConfig; print(Stage23EnvironmentConfig(scenario_name='risk_fork_train_upper').to_json_dict())"
```

```text
{'stage': '23-A', 'scenario_name': 'risk_fork_train_upper', 'width': 8, 'height': 8, 'agent_count': 2, 'max_steps': 32, 'seed': 23, 'sensing_radius': 1, 'local_observation_radius': 2, 'step_reward': -0.01, 'success_reward': 1.0, 'sensing_cost_value': 0.05, 'hazard_cost_value': 1.0, 'invalid_move_penalty': 0.0, 'blocked_move_penalty': 0.0, 'start_positions': None, 'goal_cells': None, 'obstacle_cells': None, 'hidden_hazard_cells': None}
```

Source: `src/raas_marl/environments/active_sensing/grid_environment.py` (`Stage23EnvironmentConfig`, `RiskAwareActiveSensingGridEnvironment`, `STAGE23_AGENT_NAMES`); `src/raas_marl/mappo_lagrangian/stage23c_rollout.py` (`make_default_stage23c_environment`); `src/raas_marl/final_training/stage25_collection.py` (`collect_stage25_training_rollout`); `results/experiments/*/RUN_MANIFEST.json` (`config.steps_per_episode`); command `python -c "from raas_marl.environments.active_sensing.grid_environment import Stage23EnvironmentConfig; print(Stage23EnvironmentConfig(scenario_name='risk_fork_train_upper').to_json_dict())"`

## Actions: movement, sensing, and the reveal

Each agent chooses two action factors per step, a movement action (`movement_action`) and a sensing action (`sensing_action`).

| factor | index | name | effect |
|---|---|---|---|
| movement | 0 | `stay` | no move |
| movement | 1 | `north` | row minus 1 |
| movement | 2 | `south` | row plus 1 |
| movement | 3 | `west` | column minus 1 |
| movement | 4 | `east` | column plus 1 |
| sensing | 0 | `no_sense` | no reveal, no cost |
| sensing | 1 | `sense` | a reveal, at the sensing cost |

A move off the grid leaves the agent in place and sets `invalid_move`. A move into an obstacle leaves the agent in place and sets `blocked_move`.

A sense produces what the paper called a one-step reveal (Sec. III). It returns the hidden hazard cells within a local radius, Manhattan distance `sensing_radius` (1 by default), of the agent's position after that step's move. The revealed cells appear in that step's observation under `revealed_local_hazards`. The next step replaces them, so a reveal lasts one step.

An agent's observation holds no hidden-hazard cell except those a reveal returns. When an agent ends a step on a hazard, that step's observation sets `previous_entered_hazard`. Public obstacles within `local_observation_radius` (2 by default) appear under `local_obstacles`. That radius serves the two-cell obstacle patch, the enriched encoding, which reads cells up to two steps away. The actor encoding `actor_observation_from_stage23` rejects a `local_observation_radius` argument below 2. The training collection calls it without that argument (`actor_observation_batch`), so the check does not catch an environment built with a radius of 1. The shipped code does not build the paper's one-cell scalar count.

This command steps `risk_fork_train_upper` four times. `agent_0` starts at (3, 0) and moves east each step; it senses on the third step, then steps onto the hidden hazard. `agent_1` stays in place and does not sense. Each line prints `agent_0`'s position, revealed cells, task reward, sensing cost, and hazard cost:

```shell
python -c "from raas_marl.environments.active_sensing.grid_environment import RiskAwareActiveSensingGridEnvironment as E, Stage23EnvironmentConfig as C; e = E(C(scenario_name='risk_fork_train_upper')); e.reset(); a = lambda s: dict(agent_0=dict(sensing_action=s, movement_action=4), agent_1=dict(sensing_action=0, movement_action=0)); [print(o['agent_0']['actor_visible']['position'], o['agent_0']['actor_visible']['revealed_local_hazards'], r['agent_0'], i['agent_0']['sensing_cost'], i['agent_0']['hazard_cost']) for o, r, t, u, i in [e.step(a(s)) for s in (0, 0, 1, 0)]]"
```

```text
[3, 1] [] -0.01 0.0 0.0
[3, 2] [] -0.01 0.0 0.0
[3, 3] [[3, 4]] -0.01 0.05 0.0
[3, 4] [] -0.01 0.0 1.0
```

Source: `src/raas_marl/environments/active_sensing/grid_environment.py` (`STAGE23_MOVEMENT_ACTIONS`, `STAGE23_MOVEMENT_DELTAS`, `STAGE23_SENSING_ACTIONS`, `RiskAwareActiveSensingGridEnvironment`); `src/raas_marl/environments/active_sensing/tensor_adapter.py` (`actor_observation_from_stage23`); `src/raas_marl/mappo_lagrangian/_rollout_common.py` (`actor_observation_batch`); camera-ready, p. 3, Sec. III; command `python -c "from raas_marl.environments.active_sensing.grid_environment import RiskAwareActiveSensingGridEnvironment as E, Stage23EnvironmentConfig as C; e = E(C(scenario_name='risk_fork_train_upper')); e.reset(); a = lambda s: dict(agent_0=dict(sensing_action=s, movement_action=4), agent_1=dict(sensing_action=0, movement_action=0)); [print(o['agent_0']['actor_visible']['position'], o['agent_0']['actor_visible']['revealed_local_hazards'], r['agent_0'], i['agent_0']['sensing_cost'], i['agent_0']['hazard_cost']) for o, r, t, u, i in [e.step(a(s)) for s in (0, 0, 1, 0)]]"`

## Reward and cost

The environment returns, per agent and per step:

| quantity | code | default | charged |
|---|---|---|---|
| task reward | `task_reward` (`step_reward`) | -0.01 | every step |
| success reward | added to `task_reward` (`success_reward`) | 1.0 | to each agent, on the step the team succeeds |
| move penalties | added to `task_reward` (`invalid_move_penalty`, `blocked_move_penalty`) | 0.0 | on an invalid or blocked move |
| hazard cost | `hazard_cost` (`hazard_cost_value`) | 1.0 | every step the agent ends on a hidden hazard cell, including by staying on it |
| sensing cost | `sensing_cost` (`sensing_cost_value`) | 0.05 | every step the agent senses |

The paper treated hazard exposure as the constraint, not as a reward penalty (Sec. II). In the code, the training update builds its reward signal as `task_reward - sensing_cost_coefficient * sensing_cost`, with no hazard term. The hazard cost has its own critic and advantage (`compute_cost_gae`), which the policy objective subtracts, weighted by the constraint multiplier. `episodic_team_hazard_cost` gives the per-round cost estimate that drives the multiplier.

Two numbers describe the sensing cost. The environment charges `sensing_cost_value`, 0.05, per sense. The update weights that charge by `sensing_cost_coefficient`, 0.5, the "fixed coefficient 0.5" that the paper reported (Sec. IV). One sense therefore lowers the reward signal by 0.025. The training driver changes neither value when run from its command-line entry: it builds the environment with its defaults and keeps the loss weights of `Stage25UpdateConfig.from_decision_records`. The records carry the weight: `config.stage25_update_config.losses.sensing_cost_coefficient` is 0.5 in 51 of the 55 run manifests, and `config.stage22_update_config.losses.sensing_cost_coefficient` is 0.5 in the other 4, the records of `b101_stoch`, `b102_stoch`, `b103_stoch`, and `b104_greedy`.

The paper reported that every reported run also added potential-based shaping to the reward, and that the shaping never entered the graded metrics (Sec. IV). The update computes it with `potential_shaping_term`.

Source: `src/raas_marl/environments/active_sensing/grid_environment.py` (`Stage23EnvironmentConfig`, `RiskAwareActiveSensingGridEnvironment`); `src/raas_marl/mappo_lagrangian/update.py` (`_development_reward_signal`); `src/raas_marl/final_training/stage25_update.py` (`Stage25UpdateConfig`, `episodic_team_hazard_cost`, `potential_shaping_term`, `compute_cost_gae`); `src/raas_marl/final_training/stage25_driver.py` (`run_stage25_training`, `main`); `results/experiments/*/RUN_MANIFEST.json` (`config.stage25_update_config.losses.sensing_cost_coefficient`, `config.stage22_update_config.losses.sensing_cost_coefficient`); camera-ready, p. 2, Sec. II; camera-ready, p. 4, Sec. IV; camera-ready, p. 5, Sec. IV

## What a risk fork is

A risk fork (`risk_fork_*`) is a catalog scenario with a two-gate wall. Column 4 (`_FORK_WALL_COLUMN`) holds six obstacles and two open gate cells, and the grid has no other obstacle. In all ten fork scenarios, the agents start at (3, 0) and (4, 0), and the goal is (3, 7). One of the two gates carries the hidden hazard (`hidden_hazard_cells`). The code does not generate forks: each is a fixed `Stage23Scenario` entry in `stage23_scenario_catalog`, and `stage24_diagnostics.py` lists their names by group.

A scenario pair shares the gates and differs only in which gate is hazardous. In the `_upper` mirror the hazard is on the lower-numbered gate row; in the `_lower` mirror it is on the higher-numbered one. Until an agent senses or steps on the hazard, the two mirrors give it the same observations. This command prints `risk_fork_train_upper`, with `S` a start, `G` the goal, `#` the wall, `H` the hidden hazard, and `.` a free cell:

```shell
python -c "from raas_marl.environments.active_sensing.grid_environment import stage23_scenario_catalog; s = stage23_scenario_catalog()['risk_fork_train_upper']; m = dict([(c, 'S') for c in s.starts] + [(c, 'G') for c in s.goals] + [(c, '#') for c in s.obstacles] + [(c, 'H') for c in s.hidden_hazard_cells]); [print(' '.join(m.get((r, k), '.') for k in range(s.width))) for r in range(s.height)]"
```

```text
. . . . # . . .
. . . . # . . .
. . . . # . . .
S . . . H . . G
S . . . . . . .
. . . . # . . .
. . . . # . . .
. . . . # . . .
```

`verify_fork_family_forces_sensing` checks a family of scenarios, here a pair, from the scenario definitions alone, without training a policy. It rejects a pair whose starts, goals, or obstacles differ. By breadth-first search, it checks the start-to-goal cut: no start reaches the goal when both of the pair's hazard cells are blocked (`bar_a_union_cut`). It checks that every open gate is hazardous in one of the layouts, that each layout has a route around its own hazard (`bar_b_all_layouts_winnable`), and that the scripted `selective_sense` comparator reaches the goal with zero hazard cost on both layouts. It also averages over the layouts the team hazard cost of one episode of the never-sensing `no_sense` comparator, summed over both agents. The function's default `hazard_budget` is 0.5; this command checks the anchor pair at the paper's budget of 0.25:

```shell
python -c "from raas_marl.environments.active_sensing.stage24_diagnostics import verify_fork_family_forces_sensing as v; r = v(('risk_fork_train_lower', 'risk_fork_train_upper'), hazard_budget=0.25); [print(k, r[k]) for k in r if k.startswith('bar_')]"
```

```text
bar_a_union_cut True
bar_a_all_gates_hazardous True
bar_a_no_sense_mean_hazard 1.0
bar_a_no_sense_mean_hazard_exceeds_budget True
bar_a_sensing_forced_by_construction True
bar_b_all_layouts_winnable True
bar_b_selective_zero_hazard True
bar_b_selective_preserves_success True
bar_b_task_learnable_by_construction True
```

The mean of 1.0 matches the paper's Fig. 2 caption, which reported that a blind route incurs a mean hazard of 1.0 per episode, above the training budget of 0.25.

Source: `src/raas_marl/environments/active_sensing/grid_environment.py` (`stage23_scenario_catalog`, `Stage23Scenario`); `src/raas_marl/environments/active_sensing/stage24_diagnostics.py` (`_FORK_WALL_COLUMN`, `verify_fork_family_forces_sensing`); camera-ready, p. 3, Fig. 2; command `python -c "from raas_marl.environments.active_sensing.grid_environment import stage23_scenario_catalog; s = stage23_scenario_catalog()['risk_fork_train_upper']; m = dict([(c, 'S') for c in s.starts] + [(c, 'G') for c in s.goals] + [(c, '#') for c in s.obstacles] + [(c, 'H') for c in s.hidden_hazard_cells]); [print(' '.join(m.get((r, k), '.') for k in range(s.width))) for r in range(s.height)]"`; command `python -c "from raas_marl.environments.active_sensing.stage24_diagnostics import verify_fork_family_forces_sensing as v; r = v(('risk_fork_train_lower', 'risk_fork_train_upper'), hazard_budget=0.25); [print(k, r[k]) for k in r if k.startswith('bar_')]"`

## The scenario pairs and their gate rows

The catalog holds 14 scenarios, 10 of them risk forks. The code groups the forks in two functions: `fork_hazard_layout_families` returns the `training` and `readiness` groups, and `fork_curriculum_scenarios` returns the curriculum scenarios. The last column counts, of the 55 run manifests, those whose `config.scenario_names` names the scenario.

| paper term | returned by | scenario | gate rows | hidden hazard | manifests naming it |
|---|---|---|---|---|---|
| anchor pair | `fork_hazard_layout_families`, key `training` | `risk_fork_train_lower` | 3, 4 | (4, 4) | 45 of 55 |
| anchor pair | `fork_hazard_layout_families`, key `training` | `risk_fork_train_upper` | 3, 4 | (3, 4) | 45 of 55 |
| curriculum pair | `fork_curriculum_scenarios` | `risk_fork_curriculum_r0r3_lower` | 0, 3 | (3, 4) | 15 of 55 |
| curriculum pair | `fork_curriculum_scenarios` | `risk_fork_curriculum_r0r3_upper` | 0, 3 | (0, 4) | 15 of 55 |
| curriculum pair | `fork_curriculum_scenarios` | `risk_fork_curriculum_r4r7_lower` | 4, 7 | (7, 4) | 3 of 55 |
| curriculum pair | `fork_curriculum_scenarios` | `risk_fork_curriculum_r4r7_upper` | 4, 7 | (4, 4) | 3 of 55 |
| curriculum pair | `fork_curriculum_scenarios` | `risk_fork_curriculum_r0r7_lower` | 0, 7 | (7, 4) | 6 of 55 |
| curriculum pair | `fork_curriculum_scenarios` | `risk_fork_curriculum_r0r7_upper` | 0, 7 | (0, 4) | 6 of 55 |
| validation geometry | `fork_hazard_layout_families`, key `readiness` | `risk_fork_readiness_lower` | 1, 2 | (2, 4) | 0 of 55 |
| validation geometry | `fork_hazard_layout_families`, key `readiness` | `risk_fork_readiness_upper` | 1, 2 | (1, 4) | 0 of 55 |

This command derives the gate rows from each scenario's obstacles and prints each group's scenarios with their hidden hazard cell. The label `curriculum` in its output is the command's own; `fork_curriculum_scenarios` returns an unnamed tuple:

```shell
python -c "from raas_marl.environments.active_sensing.grid_environment import stage23_scenario_catalog as cat; from raas_marl.environments.active_sensing.stage24_diagnostics import fork_hazard_layout_families as fam, fork_curriculum_scenarios as cur; g = dict(fam(), curriculum=cur()); [print(k, n, [r for r in range(8) if (r, 4) not in cat()[n].obstacles], cat()[n].hidden_hazard_cells) for k, v in g.items() for n in v]"
```

```text
training risk_fork_train_lower [3, 4] ((4, 4),)
training risk_fork_train_upper [3, 4] ((3, 4),)
readiness risk_fork_readiness_lower [1, 2] ((2, 4),)
readiness risk_fork_readiness_upper [1, 2] ((1, 4),)
curriculum risk_fork_curriculum_r0r3_lower [0, 3] ((3, 4),)
curriculum risk_fork_curriculum_r0r3_upper [0, 3] ((0, 4),)
curriculum risk_fork_curriculum_r4r7_lower [4, 7] ((7, 4),)
curriculum risk_fork_curriculum_r4r7_upper [4, 7] ((4, 4),)
curriculum risk_fork_curriculum_r0r7_lower [0, 7] ((7, 4),)
curriculum risk_fork_curriculum_r0r7_upper [0, 7] ((0, 4),)
```

The paper reported the partition of gate rows (Sec. IV). Trainable gate rows were {0, 3, 4, 7}, and every reported run trained within {0, 3, 4}, except two curricula that also used row-7 gates. No manifest names the validation geometry at gate rows {1, 2}, so no recorded run trained on it. A held-out set at gate rows {5, 6} was reserved. No catalog scenario has a gate on row 5 or 6; see [The reserved held-out geometry](../README.md#the-reserved-held-out-geometry).

Within a run, the collection assigns scenarios to episodes round-robin over `config.scenario_names`.

Source: `src/raas_marl/environments/active_sensing/grid_environment.py` (`stage23_scenario_catalog`); `src/raas_marl/environments/active_sensing/stage24_diagnostics.py` (`fork_hazard_layout_families`, `fork_curriculum_scenarios`); `src/raas_marl/final_training/stage25_collection.py` (`collect_stage25_training_rollout`); `results/experiments/*/RUN_MANIFEST.json` (`config.scenario_names`); camera-ready, p. 4, Sec. IV; command `python -c "from raas_marl.environments.active_sensing.grid_environment import stage23_scenario_catalog as cat; from raas_marl.environments.active_sensing.stage24_diagnostics import fork_hazard_layout_families as fam, fork_curriculum_scenarios as cur; g = dict(fam(), curriculum=cur()); [print(k, n, [r for r in range(8) if (r, 4) not in cat()[n].obstacles], cat()[n].hidden_hazard_cells) for k, v in g.items() for n in v]"`

Checked against the code at commit `6a5e22d` on 2026-09-26 UTC.

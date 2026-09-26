# What an agent observes: the two encodings

The paper compared two encodings of an agent's obstacle percept. This page gives what each holds, what the shipped code builds, and why the original encoding aliases at the decision cell of Fig. 1.

## What an actor receives

An observation has three keys: `actor_visible`, `action_factors`, and `identity`. Its actor-visible part has the 11 keys of `STAGE23_ACTOR_VISIBLE_SCHEMA_KEYS`:

| key | what it holds |
|---|---|
| `position` | the agent's cell, as row and column |
| `step_index`, `max_steps` | the step count and the episode horizon |
| `grid_shape` | the grid's height and width |
| `nearest_goal_delta` | the row and column offset to the nearest goal cell |
| `local_obstacles` | the public obstacles within Manhattan distance `local_observation_radius` (default 2) |
| `revealed_local_hazards` | the hidden-hazard cells within Manhattan distance `sensing_radius` (default 1) of the agent's cell after that step's move, in the observation returned by a step in which it sensed; empty otherwise |
| `previous_sensed`, `previous_invalid_move`, `previous_blocked_move`, `previous_entered_hazard` | flags for the previous step |

The paper's one-step reveals become the revealed information (`revealed_information_from_stage23`), 4 features wide (`STAGE23_REVEALED_INFORMATION_DIM`): a presence flag, the count of revealed cells capped at and divided by `2*r*(r+1)+1` for sensing radius `r` (`_reveal_count_cap`), and the row and column offset to the nearest revealed cell, each divided by `max(1, N - 1)` for grid side `N`, as in the observation encoding. With nothing revealed, all four are 0.0.

The model joins the observation encoding and the revealed information into one actor input (`actor_input_dim`), which the actor's first layer, `actor_encoder`, reads. The history-conditioned recurrent state (`history_state`, 16 wide) is the recurrent layer's state, not part of that input.

Source: code `src/raas_marl/environments/active_sensing/grid_environment.py` (`STAGE23_ALLOWED_OBSERVATION_KEYS`, `STAGE23_ACTOR_VISIBLE_SCHEMA_KEYS`, `Stage23EnvironmentConfig`, `RiskAwareActiveSensingGridEnvironment`); code `src/raas_marl/environments/active_sensing/tensor_adapter.py` (`revealed_information_from_stage23`, `_reveal_count_cap`, `STAGE23_REVEALED_INFORMATION_DIM`, `STAGE23_HISTORY_STATE_DIM`); code `src/raas_marl/mappo_lagrangian/config.py` (`actor_input_dim`); code `src/raas_marl/mappo_lagrangian/model.py` (`RecurrentMAPPOActorCritic`)

## The enriched encoding: 22 features

The observation encoding (`actor_observation_from_stage23`) builds the enriched encoding and no other. It returns 22 features (`STAGE23_ACTOR_OBSERVATION_DIM`): ten scalar features (`STAGE23_LEGACY_ACTOR_FEATURE_COUNT`), then 12 patch features, one per cell in `_RADIUS2_PATCH_OFFSETS`.

| index | feature |
|---|---|
| 0, 1 | position row and column, each divided by `max(1, N - 1)` for grid side `N` |
| 2, 3 | goal offset in rows and columns, with the same divisors |
| 4 | step index divided by the horizon |
| 5 to 8 | the four previous-step flags, as 1.0 or 0.0 |
| 9 | the number of `local_obstacles`, capped at and divided by 13, the cell count of a radius-2 window |
| 10 to 21 | the patch: 1.0 where the cell at a fixed offset is an obstacle or off the grid, else 0.0 |

The patch covers the 12 cells within two steps other than the agent's own, at these row and column offsets, in order: (-2, 0), (-1, -1), (-1, 0), (-1, 1), (0, -2), (0, -1), (0, 1), (0, 2), (1, -1), (1, 0), (1, 1), (2, 0). It reads `local_obstacles` and the grid bounds only, never the hidden hazards. The default core configuration (`default_stage23_core_config`) sets the widths. This command prints it. It needs the package `raas_marl` importable, as the README's install makes it, and does not import PyTorch:

```shell
python -c "print(__import__('raas_marl.environments.active_sensing.tensor_adapter', fromlist=['']).default_stage23_core_config())"
```

```text
MAPPOCoreConfig(actor_observation_dim=22, revealed_information_dim=4, central_state_dim=20, history_state_dim=16, actor_hidden_dim=32, critic_hidden_dim=32, sensing_action_count=2, movement_action_count=5, recurrent_layer_count=1, agent_id_count=2, use_recurrent_actor=True, dtype='float32', device='cpu')
```

`actor_observation_dim=22` is the enriched encoding, and `revealed_information_dim=4` is the revealed information. Their sum, 26, is the actor input width (`actor_input_dim`).

Source: code `src/raas_marl/environments/active_sensing/tensor_adapter.py` (`actor_observation_from_stage23`, `_RADIUS2_PATCH_OFFSETS`, `STAGE23_LEGACY_ACTOR_FEATURE_COUNT`, `STAGE23_ACTOR_OBSERVATION_DIM`, `STAGE23_LOCAL_OBSTACLE_RADIUS`, `default_stage23_core_config`); code `src/raas_marl/mappo_lagrangian/config.py` (`MAPPOCoreConfig`, `actor_input_dim`); command `python -c "print(__import__('raas_marl.environments.active_sensing.tensor_adapter', fromlist=['']).default_stage23_core_config())"`

## The original encoding: not built by the shipped code

The paper described the original encoding as a scalar count of obstacles within a one-cell radius, four-connected, and the enriched encoding as one that widens that count and adds the patch (Sec. III). The shipped code has no identifier for the original encoding. Its ten features were indices 0 to 9 of the table above, with index 9 counting obstacles one step away instead of within two steps: the code comment at `STAGE23_LEGACY_ACTOR_FEATURE_COUNT` says the patch is appended at index 10 and the index-9 count retained, and the Fig. 1 generator's docstring describes the original feature 9. The analysis record `docs/evidence/anchor_rerun_analysis.json` gives the change as `actor_observation_dim 10->22 (actor_input_dim 26)`, and says that a dim-10 input layer would be `(32,14)`, that is 10 + 4 inputs. No checkpoint of an original-encoding run ships to show one.

Asked for a `local_observation_radius` argument below 2, the observation encoding raises a `ValueError`. It does not check the environment's own `local_observation_radius`, which accepts 1. With an environment at 1, the call the training collection makes (`actor_observation_batch`, which passes no radius) returns 22 features: index 9 counts the one-step window over 13, and the patch reads its in-grid cells two steps away as open, obstacle or not. That is not the original encoding. No shipped driver sets that field. The encoding's docstring gives the default radius as 1; the default it uses, `STAGE23_LOCAL_OBSTACLE_RADIUS`, is 2.

The paper reported runs at the original encoding for the single pair (Sec. V-A) and the three curricula (Sec. V-B); Table II's one-cell arm is the second curriculum. The README's [run table](../README.md#which-runs-are-included-and-their-place-in-the-paper) lists `t1_d13_s132` to `s134`, `t1_d14_s135` to `s137`, `t1_d15_s138` to `s140`, and `t1_d16_s141` to `s143`, and ties the runs at hazard budget 0.5, `t1_d4` to `t1_d12`, to Sec. V-A as well; of those, `t1_d4` and `t1_d5` trained on `risk_gate_hidden_hazard`, not on the fork pair. No key in any of the 55 `RUN_MANIFEST.json` files names the encoding or an observation radius. The README's [What is not here and why](../README.md#what-is-not-here-and-why) says that a re-launch of Table II's one-cell arm would run at the two-cell obstacle patch without an error, and its data-availability table says the shipped driver rejects the configuration recorded for `t1_d14_s135` to `s137`.

Source: paper camera-ready, p. 3, Sec. III; paper camera-ready, p. 5, Table II, Sec. V-A, Sec. V-B; record `docs/evidence/anchor_rerun_analysis.json` (`encoding`, `step0_behavioral_battery.dim22_training_confirmed`); record `results/experiments/*/RUN_MANIFEST.json` (`config`, `config.scenario_names`); code `src/raas_marl/environments/active_sensing/tensor_adapter.py` (`actor_observation_from_stage23`, `STAGE23_LOCAL_OBSTACLE_RADIUS`, `STAGE23_LEGACY_ACTOR_FEATURE_COUNT`, `_RADIUS2_PATCH_OFFSETS`); code `src/raas_marl/mappo_lagrangian/_rollout_common.py` (`actor_observation_batch`); code `src/raas_marl/environments/active_sensing/grid_environment.py` (`local_observation_radius`); code `docs/paper/figure_data/generate_fig1_data.py` (`radius1_obstacle_count`)

## What the shipped checkpoints hold

The repository ships 27 policy checkpoints, and all 27 use the enriched encoding. This command reads the actor's input layer, `actor_encoder.weight` in `model_state_dict`, from each and tallies the shapes. It needs PyTorch. Shown is the last line of its output, an excerpt; when NumPy is not installed, a PyTorch warning about it comes first:

```shell
python -c "print(__import__('collections').Counter(tuple(__import__('torch').load(p, weights_only=True)['model_state_dict']['actor_encoder.weight'].shape) for p in __import__('glob').glob('results/experiments/*/state_round_*.pt')))"
```

```text
Counter({(32, 26): 27})
```

Shape (32, 26) is 32 units reading 26 inputs: 22 observation and 4 revealed-information features. The 27 checkpoints lie in the nine replication run directories: 9 under `t1_retain_s147` to `s149` and 18 under `t1_seedscale_s153` to `s158`. No other run directory holds one: no file matches `state_round_*.pt` under `t1_d*`, `b10*`, or `anchor_rerun_*`.

Source: code `src/raas_marl/mappo_lagrangian/model.py` (`RecurrentMAPPOActorCritic`); record `results/experiments/*/state_round_*.pt`; command `python -c "print(__import__('collections').Counter(tuple(__import__('torch').load(p, weights_only=True)['model_state_dict']['actor_encoder.weight'].shape) for p in __import__('glob').glob('results/experiments/*/state_round_*.pt')))"`

## Fig. 1 worked through: the aliasing cell

Fig. 1 compared two scenarios at the decision cell (3, 3) (`CELL`). The anchor scenario, `risk_fork_train_upper`, has gate rows {3, 4}. The curriculum scenario, `risk_fork_curriculum_r0r3_lower`, has gate rows {0, 3}. Both put the hidden hazard at (3, 4), one step east of the decision cell. It is not an obstacle, so no obstacle feature shows it. Before an agent steps on it, only sensing reveals it; afterwards the `previous_entered_hazard` flag reports the contact. The safe gate is (4, 4), one row down, for the anchor, and (0, 4), three rows up, for the other pair. The Fig. 1 data generator computes the cell. It needs PyTorch. Shown are the last lines of its output, an excerpt; before them come a JSON record and, without NumPy, a PyTorch warning:

```shell
python docs/paper/figure_data/generate_fig1_data.py
```

```text
=== FIGURE-1 HEADLINE (fresh from source) ===
features 0..8 (radius-independent) byte-identical: True
original radius-1 count feature 9: 0 vs 0 -> identical: True
=> ORIGINAL dim-10 obs at (3,3) byte-identical: True
enriched dim-22 differs at indices: [9, 20]
patch (4,4): anchor=0.0 (OPEN)  {0,3}=1.0 (WALL)
ALL FIGURE-1 CHECKS PASS
```

The generator places `agent_0` at the cell by writing the environment's private `_positions` table, and encodes the private `_observation` output with `actor_observation_from_stage23`. It rewrites the tracked file `docs/paper/figure_data/fig1_aliasing.json`. On Windows on 2026-09-26 UTC, `git status` listed no changed file after it ran. The committed file has CRLF line endings, and the generator writes the platform's line ending, so on a system whose line ending is LF the rewritten file differs from the committed one; `git restore docs/paper/figure_data/fig1_aliasing.json` undoes the change.

The generator cannot build the original encoding either. It takes features 0 to 8 from the enriched observation, since they do not read obstacles. It recomputes the original index 9 with `radius1_obstacle_count`, which counts obstacles among the four cells one step away in the scenario's obstacle list, and compares those raw counts, 0 and 0. So the original observation is identical in both scenarios, while the safe reroutes are opposite. The paper reported that no memoryless policy can map one observation to two actions. The enriched observation differs at two indices. This table copies both vectors from the generator's JSON record, rounded there to six places, with the patch values written as 0 and 1:

| index | feature | `risk_fork_train_upper` | `risk_fork_curriculum_r0r3_lower` |
|---|---|---|---|
| 0 | position row | 0.428571 | 0.428571 |
| 1 | position column | 0.428571 | 0.428571 |
| 2 | goal offset, rows | 0.0 | 0.0 |
| 3 | goal offset, columns | 0.571429 | 0.571429 |
| 4 | step progress | 0.0 | 0.0 |
| 5 | `previous_sensed` | 0.0 | 0.0 |
| 6 | `previous_invalid_move` | 0.0 | 0.0 |
| 7 | `previous_blocked_move` | 0.0 | 0.0 |
| 8 | `previous_entered_hazard` | 0.0 | 0.0 |
| 9 | obstacles within two steps, over 13 | 0.076923 | 0.153846 |
| 10 to 21 | the patch, in offset order | `0 0 0 1 0 0 0 0 0 0 0 0` | `0 0 0 1 0 0 0 0 0 0 1 0` |

Index 9 counts one obstacle within two steps for the anchor and two for the other pair (1/13 and 2/13). Index 20, the patch cell (4, 4), is open (0.0) for the anchor and a wall (1.0) for the other pair. The other pair's safe gate, (0, 4), is four steps from the cell, outside the patch.

Source: command `python docs/paper/figure_data/generate_fig1_data.py`; code `docs/paper/figure_data/generate_fig1_data.py` (`CELL`, `obs_at`, `radius1_obstacle_count`); code `src/raas_marl/environments/active_sensing/grid_environment.py` (`stage23_scenario_catalog`); paper camera-ready, p. 1, Fig. 1

## The conflict one row down, and the Table II comparison

The paper reported that the conflict is local: one row down, at (4, 3), the one-step obstacle count is 0 for the anchor and 1 for the other pair, and none of the three curricula carried that bit to the decision cell (Sec. V-B). No shipped script prints these counts. This one-line command reads them from the scenario catalog: it counts the obstacles among the four cells one step from (4, 3) in each scenario, anchor first. It needs `raas_marl` importable and does not import PyTorch:

```shell
python -c "print([sum(n in __import__('raas_marl.environments.active_sensing.grid_environment', fromlist=['']).stage23_scenario_catalog()[s].obstacles for n in ((3, 3), (5, 3), (4, 2), (4, 4))) for s in ('risk_fork_train_upper', 'risk_fork_curriculum_r0r3_lower')])"
```

```text
[0, 1]
```

The paper also reported that the widened count alone separates the two scenarios at the decision cell, as index 9 above does. Table II compared the two encodings on the same {3, 4} + {0, 3} training mixture. The encoding was the variable under test, and the two arms used different seed sets: the README's [run table](../README.md#which-runs-are-included-and-their-place-in-the-paper) assigns `t1_d15_s138` to `s140` to the original encoding and `t1_retain_s147` to `s149` to the enriched encoding. Retention was 0/3 and 2/3. The paper reported that at three seeds per arm this difference is not itself statistically significant (Sec. IV).

Source: command `python -c "print([sum(n in __import__('raas_marl.environments.active_sensing.grid_environment', fromlist=['']).stage23_scenario_catalog()[s].obstacles for n in ((3, 3), (5, 3), (4, 2), (4, 4))) for s in ('risk_fork_train_upper', 'risk_fork_curriculum_r0r3_lower')])"`; paper camera-ready, p. 5, Sec. V-B, Table II, Sec. IV; record `docs/evidence/t1_d15_analysis.json` (`seeds`, `RETAIN.verdict.bar_c4_retain_pass_count`); record `docs/evidence/t1_retain_analysis.json` (`tallies.bar_c4_retain`, `dim22_training_confirmed`)

Checked against the code at commit `6a5e22d` on 2026-09-26 UTC.

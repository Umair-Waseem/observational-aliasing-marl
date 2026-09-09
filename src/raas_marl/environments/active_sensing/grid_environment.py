"""Stage 23-A deterministic active-sensing grid environment.

This module is dependency-light by design. It does not import PyTorch, does
not run training or evaluation, and exposes only a small parallel multi-agent
environment contract for Stage 23-A environment construction.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
import random
from typing import TypedDict

from raas_marl.mappo_lagrangian._validation import (
    finite_numeric_scalar,
    require_int_not_bool,
    require_nonnegative_int,
    require_positive_int,
)
from raas_marl.mappo_lagrangian.config import forbidden_actor_information_keys


Coordinate = tuple[int, int]

STAGE23_STAGE = "23-A"
STAGE23_DEFAULT_SEED = 23
STAGE23_FIXED_AGENT_COUNT = 2
STAGE23_AGENT_NAMES = ("agent_0", "agent_1")
TASK_REWARD_KEY = "task_reward"
HAZARD_COST_KEY = "hazard_cost"
SENSING_COST_KEY = "sensing_cost"
SENSING_ACTION_FIELD = "sensing_action"
MOVEMENT_ACTION_FIELD = "movement_action"
STAGE23_SENSING_ACTIONS = {0: "no_sense", 1: "sense"}
# NEW-methodology-1 (documented; final choice queued for Phase 6): the "sense"
# branch is charged the sensing cost. The gate is written as
# ``sensing_action == STAGE23_SENSE_ACTION_INDEX`` (a named index constant)
# instead of a bare ``== 1`` so the meaning is explicit and the binary-factor
# assumption is localised. Behaviour is unchanged for the current two-action
# sensing factor; whether sensing should stay a fixed binary factor is a
# methodology decision deferred to Phase 6.
STAGE23_SENSE_ACTION_INDEX = 1
STAGE23_MOVEMENT_ACTIONS = {
    0: "stay",
    1: "north",
    2: "south",
    3: "west",
    4: "east",
}
# NEW-grid_environment-2: canonical module-level movement delta table. Hoisted
# out of ``_resolve_movement`` so it is not rebuilt every step and is the single
# importable source of movement (row, column) deltas for the package (e.g.
# ``stage24_diagnostics`` imports this instead of maintaining its own copy).
STAGE23_MOVEMENT_DELTAS: dict[int, Coordinate] = {
    0: (0, 0),
    1: (-1, 0),
    2: (1, 0),
    3: (0, -1),
    4: (0, 1),
}
# Load-time coherence guard: the delta table and the human-readable action-name
# table must describe exactly the same movement action indices.
assert set(STAGE23_MOVEMENT_DELTAS) == set(STAGE23_MOVEMENT_ACTIONS), (
    "STAGE23_MOVEMENT_DELTAS keys must match STAGE23_MOVEMENT_ACTIONS keys"
)
STAGE23_STAGE_SPECIFIC_FORBIDDEN_ACTOR_KEYS = frozenset(
    {
        "central_info",
        "central_map",
        "central_observation",
        "central_state",
        "critic_observation",
        "critic_visible",
        "global_state",
        "global_observation",
        "hidden_hazard_map",
        "hidden_hazard_grid",
        "hidden_hazards",
        "hidden_map",
        "privileged_hazard_map",
        "privileged_observation",
        "privileged_state",
        "critic_state",
    }
)
STAGE23_FORBIDDEN_ACTOR_KEYS = (
    forbidden_actor_information_keys() | STAGE23_STAGE_SPECIFIC_FORBIDDEN_ACTOR_KEYS
)
STAGE23_ALLOWED_OBSERVATION_KEYS = frozenset(
    {"actor_visible", "action_factors", "identity"}
)
STAGE23_ACTOR_VISIBLE_SCHEMA_KEYS = (
    "position",
    "step_index",
    "max_steps",
    "grid_shape",
    "nearest_goal_delta",
    "local_obstacles",
    "revealed_local_hazards",
    "previous_sensed",
    "previous_invalid_move",
    "previous_blocked_move",
    "previous_entered_hazard",
)
STAGE23_ACTOR_VISIBLE_SCHEMA_KEY_SET = frozenset(STAGE23_ACTOR_VISIBLE_SCHEMA_KEYS)


class PreviousStatus(TypedDict):
    """Fixed heterogeneous per-agent status record (NEW-grid_environment-10).

    Making the schema explicit lets the type checker catch drift between
    ``_default_previous_status`` and the step-loop assignments that build this
    record (the field literals here must equal ``TASK_REWARD_KEY`` /
    ``HAZARD_COST_KEY`` / ``SENSING_COST_KEY``).
    """

    invalid_move: bool
    blocked_move: bool
    entered_hazard: bool
    sensed: bool
    revealed_hazard_count: int
    task_reward: float
    hazard_cost: float
    sensing_cost: float
    team_success: bool


def _require_string_mapping_keys(mapping: Mapping[object, object], key_label: str) -> None:
    for key in mapping:
        if not isinstance(key, str):
            raise TypeError(
                f"{key_label} must be a string before set comparison; "
                f"got {type(key).__name__} key {key!r}"
            )


@dataclass(frozen=True)
class Stage23Scenario:
    """Serializable deterministic Stage 23-A scenario description."""

    name: str
    width: int
    height: int
    starts: tuple[Coordinate, ...]
    goals: tuple[Coordinate, ...]
    obstacles: tuple[Coordinate, ...]
    hidden_hazard_cells: tuple[Coordinate, ...]
    description: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scenario name must be a non-empty string")
        require_positive_int("scenario width", self.width)
        require_positive_int("scenario height", self.height)
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("scenario description must be a non-empty string")
        for field_name in ("starts", "goals", "obstacles", "hidden_hazard_cells"):
            value = getattr(self, field_name)
            if not isinstance(value, tuple):
                raise TypeError(f"{field_name} must be a tuple")
            for cell in value:
                _validate_coordinate(field_name, cell)
                _require_in_grid(field_name, cell, self.width, self.height)
            # NEW-grid_environment-9: "starts" is exempt from the sorted-order
            # requirement (start order is positional: starts[i] belongs to
            # agent i, so it must stay caller-ordered) but is still required to
            # contain unique cells below. Distinct start cells keep the two
            # agents' identities separable at reset; mid-episode co-occupancy of
            # the same cell is allowed (agent-agent collision is not modeled).
            if field_name != "starts" and tuple(sorted(value)) != value:
                raise ValueError(f"{field_name} must be sorted deterministically")
            _require_unique_cells(f"scenario {field_name}", value)
        if len(self.starts) != STAGE23_FIXED_AGENT_COUNT:
            raise ValueError("Stage 23-A scenarios require exactly two starts / two agents")
        if not self.goals:
            raise ValueError("scenario goals must be nonempty")
        _validate_disjoint_layout_sets(
            width=self.width,
            height=self.height,
            starts=self.starts,
            goals=self.goals,
            obstacles=self.obstacles,
            hidden_hazard_cells=self.hidden_hazard_cells,
        )

    def to_json_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible scenario record."""

        return {
            "name": self.name,
            "width": self.width,
            "height": self.height,
            "starts": _cells_to_json(self.starts),
            "goals": _cells_to_json(self.goals),
            "obstacles": _cells_to_json(self.obstacles),
            "hidden_hazard_cells": _cells_to_json(self.hidden_hazard_cells),
            "description": self.description,
        }


@dataclass(frozen=True)
class Stage23EnvironmentConfig:
    """Validated Stage 23-A environment configuration."""

    stage: str = STAGE23_STAGE
    scenario_name: str = "standard_branching_hazard"
    width: int = 8
    height: int = 8
    agent_count: int = 2
    max_steps: int = 32
    seed: int | None = STAGE23_DEFAULT_SEED
    sensing_radius: int = 1
    # DR-REPRESENTATION (SHIFT 36, candidate A): the actor's PUBLIC obstacle
    # perception window is Manhattan radius 2 (was 1), so tensor_adapter can render
    # the egocentric radius-2 obstacle-occupancy patch. Reads scenario.obstacles ONLY
    # (never hidden_hazard_cells) => can NEVER leak the hidden hazard at any radius.
    local_observation_radius: int = 2
    step_reward: float = -0.01
    success_reward: float = 1.0
    sensing_cost_value: float = 0.05
    hazard_cost_value: float = 1.0
    invalid_move_penalty: float = 0.0
    blocked_move_penalty: float = 0.0
    start_positions: tuple[Coordinate, ...] | None = None
    goal_cells: tuple[Coordinate, ...] | None = None
    obstacle_cells: tuple[Coordinate, ...] | None = None
    hidden_hazard_cells: tuple[Coordinate, ...] | None = None

    def __post_init__(self) -> None:
        if self.stage != STAGE23_STAGE:
            raise ValueError("Stage23EnvironmentConfig.stage must be '23-A'")
        if not isinstance(self.scenario_name, str) or not self.scenario_name.strip():
            raise ValueError("scenario_name must be a non-empty string")
        require_positive_int("width", self.width)
        require_positive_int("height", self.height)
        require_positive_int("agent_count", self.agent_count)
        if self.agent_count != STAGE23_FIXED_AGENT_COUNT:
            raise ValueError("Stage 23-A currently supports exactly two agents")
        require_positive_int("max_steps", self.max_steps)
        if self.seed is not None:
            require_int_not_bool("seed", self.seed)
        require_nonnegative_int("sensing_radius", self.sensing_radius)
        require_nonnegative_int("local_observation_radius", self.local_observation_radius)
        for name in (
            "step_reward",
            "success_reward",
            "sensing_cost_value",
            "hazard_cost_value",
            "invalid_move_penalty",
            "blocked_move_penalty",
        ):
            finite_numeric_scalar(name, getattr(self, name))
        if self.step_reward > 0.0:
            raise ValueError(
                "positive step rewards invert the Stage 23-A time-pressure incentive"
            )
        if self.success_reward <= 0.0:
            raise ValueError("success_reward must be positive")
        if self.sensing_cost_value <= 0.0:
            raise ValueError("sensing_cost_value must be positive")
        if self.hazard_cost_value <= 0.0:
            raise ValueError("hazard_cost_value must be positive")
        if self.invalid_move_penalty > 0.0:
            raise ValueError("invalid_move_penalty must not be positive")
        if self.blocked_move_penalty > 0.0:
            raise ValueError("blocked_move_penalty must not be positive")
        for field_name in (
            "start_positions",
            "goal_cells",
            "obstacle_cells",
            "hidden_hazard_cells",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _validate_coordinate_tuple(field_name, value, self.width, self.height)
        _resolve_and_validate_scenario(self)

    def to_json_dict(self) -> dict[str, object]:
        """Return a deterministic JSON-compatible config record."""

        return {
            "stage": self.stage,
            "scenario_name": self.scenario_name,
            "width": self.width,
            "height": self.height,
            "agent_count": self.agent_count,
            "max_steps": self.max_steps,
            "seed": self.seed,
            "sensing_radius": self.sensing_radius,
            "local_observation_radius": self.local_observation_radius,
            "step_reward": self.step_reward,
            "success_reward": self.success_reward,
            "sensing_cost_value": self.sensing_cost_value,
            "hazard_cost_value": self.hazard_cost_value,
            "invalid_move_penalty": self.invalid_move_penalty,
            "blocked_move_penalty": self.blocked_move_penalty,
            "start_positions": _optional_cells_to_json(self.start_positions),
            "goal_cells": _optional_cells_to_json(self.goal_cells),
            "obstacle_cells": _optional_cells_to_json(self.obstacle_cells),
            "hidden_hazard_cells": _optional_cells_to_json(self.hidden_hazard_cells),
        }


class RiskAwareActiveSensingGridEnvironment:
    """Deterministic parallel multi-agent active-sensing grid.

    Agent-agent collision is intentionally not modeled in Stage 23-A. Multiple
    agents may occupy the same cell, including a goal cell. Stage 23-A keeps
    every configured agent active until team termination or team truncation;
    dead-agent removal, active masks, and variable-agent padding are out of
    scope for this fixed two-agent construction environment. The reset seed is
    retained as deterministic API state for future stochastic variants, but the
    current catalog does not sample randomness. Hazard cost uses occupancy
    semantics: the "entered_hazard" status flag is set (and the hazard cost
    charged) on every step whose post-move position is a hidden hazard
    cell, including staying in place on one.
    """

    metadata = {"name": "risk_aware_active_sensing_grid_v0", "stage": STAGE23_STAGE}

    def __init__(self, config: Stage23EnvironmentConfig | None = None) -> None:
        if config is None:
            self._config = Stage23EnvironmentConfig()
        elif isinstance(config, Stage23EnvironmentConfig):
            self._config = config
        else:
            raise TypeError("config must be a Stage23EnvironmentConfig or None")
        self._scenario = _resolve_and_validate_scenario(self._config)
        self._goal_set = frozenset(self._scenario.goals)
        self._obstacle_set = frozenset(self._scenario.obstacles)
        self._hidden_hazard_set = frozenset(self._scenario.hidden_hazard_cells)
        self.possible_agents = STAGE23_AGENT_NAMES
        self.agent_name_mapping = {
            agent: index for index, agent in enumerate(self.possible_agents)
        }
        self.agents: list[str] = []
        self._positions: dict[str, Coordinate] = {}
        self._revealed_current: dict[str, tuple[Coordinate, ...]] = {}
        self._previous_status: dict[str, PreviousStatus] = {}
        self._step_index = 0
        self._team_success = False
        self._done = False
        # D-8 (documented; queued for Phase 6): _rng is reserved deterministic
        # API state for future stochastic scenario variants. The current
        # catalog is fully deterministic and never samples it; it is
        # constructed/reseeded but intentionally unused for now.
        self._rng = random.Random(self._resolved_seed(None))
        self._cumulative = {
            TASK_REWARD_KEY: 0.0,
            HAZARD_COST_KEY: 0.0,
            SENSING_COST_KEY: 0.0,
        }

    @property
    def config(self) -> Stage23EnvironmentConfig:
        """The validated configuration this environment was constructed with."""
        return self._config

    @property
    def scenario(self) -> Stage23Scenario:
        return self._scenario

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def is_done(self) -> bool:
        """Return whether the episode has ended (team terminal or truncation)."""

        return self._done

    def reset(
        self, *, seed: int | None = None
    ) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
        """Reset and return ``(observations, infos)`` keyed by active agents."""

        resolved_seed = self._resolved_seed(seed)
        self._rng = random.Random(resolved_seed)
        self.agents = list(self.possible_agents)
        self._positions = {
            agent: self._scenario.starts[index]
            for index, agent in enumerate(self.possible_agents)
        }
        self._revealed_current = {agent: tuple() for agent in self.possible_agents}
        self._previous_status = {
            agent: _default_previous_status() for agent in self.possible_agents
        }
        self._step_index = 0
        self._team_success = False
        self._done = False
        self._cumulative = {
            TASK_REWARD_KEY: 0.0,
            HAZARD_COST_KEY: 0.0,
            SENSING_COST_KEY: 0.0,
        }
        observations = {agent: self._observation(agent) for agent in self.agents}
        infos = {agent: self._info(agent, reset=True) for agent in self.agents}
        return observations, infos

    def step(
        self, actions: Mapping[str, Mapping[str, int]]
    ) -> tuple[
        dict[str, dict[str, object]],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, object]],
    ]:
        """Apply one simultaneous action mapping and return the parallel API tuple."""

        if self._done or not self.agents:
            raise RuntimeError("environment is done; call reset before stepping again")
        active_agents = tuple(self.agents)
        parsed_actions = self._validate_actions(actions, active_agents)
        next_positions: dict[str, Coordinate] = {}
        step_status: dict[str, dict[str, bool | int | float]] = {}
        for agent in active_agents:
            movement_action = parsed_actions[agent]["movement_action"]
            final_position, invalid_move, blocked_move = self._resolve_movement(
                self._positions[agent],
                movement_action,
            )
            next_positions[agent] = final_position
            step_status[agent] = {
                "invalid_move": invalid_move,
                "blocked_move": blocked_move,
                # Occupancy semantics (issue m-9, documented 2026-07-03):
                # this flag is true whenever the post-move position is a
                # hidden hazard cell, so an agent that stays on a hazard
                # cell is charged again every step. The historical field
                # name "entered_hazard" (and the actor-visible
                # "previous_entered_hazard") is kept because the Stage
                # 23-A observation schema and persisted diagnostics are
                # contract-locked; read it as "occupies hazard".
                "entered_hazard": final_position in self._hidden_hazard_set,
                "sensed": parsed_actions[agent][SENSING_ACTION_FIELD]
                == STAGE23_SENSE_ACTION_INDEX,
                "revealed_hazard_count": 0,
            }

        # NEW-grid_environment-7: this commit of the resolved moves is
        # order-dependent and must stay exactly between the two loops.
        # _resolve_movement (above) consumes PRE-move positions
        # (self._positions[agent] before this update); the reveal loop (below)
        # consumes POST-move positions (self._positions[agent] after this
        # update). Moving self._positions.update() would corrupt either
        # movement resolution or the sensing reveals.
        self._positions.update(next_positions)
        self._step_index += 1

        new_reveals: dict[str, tuple[Coordinate, ...]] = {}
        for agent in active_agents:
            if parsed_actions[agent][SENSING_ACTION_FIELD] == STAGE23_SENSE_ACTION_INDEX:
                revealed = self._local_hazards(self._positions[agent], self._config.sensing_radius)
            else:
                revealed = tuple()
            new_reveals[agent] = revealed
            step_status[agent]["revealed_hazard_count"] = len(revealed)
        self._revealed_current = new_reveals

        # NEW-grid_environment-14: team_success is computed over
        # possible_agents while rewards/terminations below iterate
        # active_agents. Stage 23-A never removes agents mid-episode
        # (active_agents == possible_agents until the team ends this step), so
        # every agent that receives the success_reward is one whose goal
        # occupancy was counted here; the two agent sets stay coherent.
        self._team_success = all(
            self._positions[agent] in self._goal_set for agent in self.possible_agents
        )
        horizon_reached = self._step_index >= self._config.max_steps
        terminated = self._team_success
        truncated = horizon_reached and not terminated

        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict[str, object]] = {}
        for agent in active_agents:
            status = step_status[agent]
            task_reward = float(self._config.step_reward)
            # NEW-grid_environment-5: the bool() wrappers here (and on
            # entered_hazard / sensed below) are intentional defensive
            # normalization: status is the loosely typed step-status record, so
            # the reads are object-typed even though the producers store native
            # bools. Behaviour is identical with or without the wrappers.
            if bool(status["invalid_move"]):
                task_reward += float(self._config.invalid_move_penalty)
            if bool(status["blocked_move"]):
                task_reward += float(self._config.blocked_move_penalty)
            if terminated:
                task_reward += float(self._config.success_reward)
            hazard_cost = (
                float(self._config.hazard_cost_value)
                if bool(status["entered_hazard"])
                else 0.0
            )
            sensing_cost = (
                float(self._config.sensing_cost_value)
                if bool(status["sensed"])
                else 0.0
            )
            self._cumulative[TASK_REWARD_KEY] += task_reward
            self._cumulative[HAZARD_COST_KEY] += hazard_cost
            self._cumulative[SENSING_COST_KEY] += sensing_cost
            status[TASK_REWARD_KEY] = task_reward
            status[HAZARD_COST_KEY] = hazard_cost
            status[SENSING_COST_KEY] = sensing_cost
            status["team_success"] = self._team_success
            rewards[agent] = task_reward
            terminations[agent] = terminated
            truncations[agent] = truncated
            self._previous_status[agent] = status

        observations = {agent: self._observation(agent) for agent in active_agents}
        for agent in active_agents:
            infos[agent] = self._info(
                agent,
                reset=False,
                terminal=terminations[agent],
                truncated=truncations[agent],
            )
        if terminated or truncated:
            self.agents = []
            self._done = True
        return observations, rewards, terminations, truncations, infos

    def critic_visible_state(self) -> dict[str, object]:
        """Return the centralized critic-visible state for CTDE adapters.

        NEW-grid_environment-6: this is deliberately valid post-terminal /
        post-truncation. The guard only requires that the environment has been
        reset at least once (positions populated for every possible agent);
        self.agents is cleared on team-end but self._positions is retained, so
        the final centralized state remains readable for the terminal-step
        critic bootstrap. The message reads "must be reset" because the sole
        failure mode is calling this before any reset.
        """

        if set(self._positions) != set(self.possible_agents):
            raise RuntimeError("environment must be reset before critic_visible_state()")
        return {
            "stage": STAGE23_STAGE,
            "scenario_name": self._scenario.name,
            "step_index": self._step_index,
            "max_steps": self._config.max_steps,
            "width": self._scenario.width,
            "height": self._scenario.height,
            "positions": {
                agent: _cell_to_json(self._positions[agent])
                for agent in self.possible_agents
            },
            "starts": _cells_to_json(self._scenario.starts),
            "goals": _cells_to_json(self._scenario.goals),
            "obstacles": _cells_to_json(self._scenario.obstacles),
            "hazard_cells": _cells_to_json(self._scenario.hidden_hazard_cells),
            "team_success": self._team_success,
            "cumulative": dict(self._cumulative),
            "agent_order": list(self.possible_agents),
        }

    def _resolved_seed(self, seed: int | None) -> int:
        # NEW-grid_environment-11 (coupled to D-8, queued for Phase 6): this
        # accepts seed 0 and negatives, unlike the strict positive-seed
        # convention elsewhere. It is intentionally left permissive because the
        # resolved seed only feeds self._rng, which is reserved deterministic
        # API state for future stochastic variants and is never sampled by the
        # current catalog. If _rng is ever sampled, tighten this to
        # require_positive_seed; that decision is deferred with D-8.
        if seed is not None:
            require_int_not_bool("seed", seed)
            return seed
        if self._config.seed is None:
            return STAGE23_DEFAULT_SEED
        return self._config.seed

    def _validate_actions(
        self,
        actions: Mapping[str, Mapping[str, int]],
        active_agents: tuple[str, ...],
    ) -> dict[str, dict[str, int]]:
        if not isinstance(actions, Mapping):
            raise TypeError("actions must be a mapping")
        _require_string_mapping_keys(actions, "action-agent-key")
        expected_agents = set(active_agents)
        actual_agents = set(actions)
        if actual_agents != expected_agents:
            missing = sorted(expected_agents - actual_agents)
            extra = sorted(actual_agents - expected_agents)
            raise ValueError(f"actions must match active agents; missing={missing}; extra={extra}")
        parsed: dict[str, dict[str, int]] = {}
        for agent in active_agents:
            payload = actions[agent]
            if not isinstance(payload, Mapping):
                raise TypeError("each agent action must be a mapping")
            expected_factors = {SENSING_ACTION_FIELD, MOVEMENT_ACTION_FIELD}
            _require_string_mapping_keys(payload, f"action-factor-key for {agent!r}")
            actual_factors = set(payload)
            if actual_factors != expected_factors:
                missing = sorted(expected_factors - actual_factors)
                extra = sorted(actual_factors - expected_factors)
                raise ValueError(
                    f"agent action factors must be exactly sensing_action and movement_action; "
                    f"missing={missing}; extra={extra}"
                )
            sensing_action = _require_action_index(
                SENSING_ACTION_FIELD,
                payload[SENSING_ACTION_FIELD],
                len(STAGE23_SENSING_ACTIONS),
            )
            movement_action = _require_action_index(
                MOVEMENT_ACTION_FIELD,
                payload[MOVEMENT_ACTION_FIELD],
                len(STAGE23_MOVEMENT_ACTIONS),
            )
            parsed[agent] = {
                SENSING_ACTION_FIELD: sensing_action,
                MOVEMENT_ACTION_FIELD: movement_action,
            }
        return parsed

    def _resolve_movement(
        self,
        position: Coordinate,
        movement_action: int,
    ) -> tuple[Coordinate, bool, bool]:
        row, column = position
        delta = STAGE23_MOVEMENT_DELTAS[movement_action]
        candidate = (row + delta[0], column + delta[1])
        if not _in_grid(candidate, self._scenario.width, self._scenario.height):
            return position, True, False
        if candidate in self._obstacle_set:
            return position, False, True
        return candidate, False, False

    def _observation(self, agent: str) -> dict[str, object]:
        position = self._positions[agent]
        previous = self._previous_status[agent]
        return {
            "actor_visible": {
                "position": _cell_to_json(position),
                "step_index": self._step_index,
                "max_steps": self._config.max_steps,
                "grid_shape": [self._scenario.height, self._scenario.width],
                "nearest_goal_delta": _cell_delta(position, self._nearest_goal(position)),
                "local_obstacles": _cells_to_json(
                    self._local_cells(position, self._config.local_observation_radius, self._scenario.obstacles)
                ),
                "revealed_local_hazards": _cells_to_json(self._revealed_current.get(agent, tuple())),
                "previous_sensed": bool(previous["sensed"]),
                "previous_invalid_move": bool(previous["invalid_move"]),
                "previous_blocked_move": bool(previous["blocked_move"]),
                "previous_entered_hazard": bool(previous["entered_hazard"]),
            },
            "action_factors": {
                "sensing_action_count": len(STAGE23_SENSING_ACTIONS),
                "movement_action_count": len(STAGE23_MOVEMENT_ACTIONS),
                "sensing_action_names": dict(STAGE23_SENSING_ACTIONS),
                "movement_action_names": dict(STAGE23_MOVEMENT_ACTIONS),
            },
            "identity": self._identity(agent),
        }

    def _info(
        self,
        agent: str,
        *,
        reset: bool,
        terminal: bool = False,
        truncated: bool = False,
    ) -> dict[str, object]:
        previous = self._previous_status[agent]
        # NEW-grid_environment-13: the float()/int()/bool() casts below are
        # intentional defensive normalization of the previous-status record for
        # a JSON-clean info payload; the sources are already correctly typed, so
        # the casts never change a value.
        task_reward = 0.0 if reset else float(previous[TASK_REWARD_KEY])
        hazard_cost = 0.0 if reset else float(previous[HAZARD_COST_KEY])
        sensing_cost = 0.0 if reset else float(previous[SENSING_COST_KEY])
        sensed = False if reset else bool(previous["sensed"])
        invalid_move = False if reset else bool(previous["invalid_move"])
        blocked_move = False if reset else bool(previous["blocked_move"])
        entered_hazard = False if reset else bool(previous["entered_hazard"])
        revealed_hazard_count = 0 if reset else int(previous["revealed_hazard_count"])
        team_success = False if reset else bool(previous["team_success"])
        return {
            "stage": STAGE23_STAGE,
            "agent": agent,
            TASK_REWARD_KEY: task_reward,
            HAZARD_COST_KEY: hazard_cost,
            SENSING_COST_KEY: sensing_cost,
            "invalid_move": invalid_move,
            "blocked_move": blocked_move,
            "entered_hazard": entered_hazard,
            "sensed": sensed,
            "revealed_hazard_count": revealed_hazard_count,
            "team_success": team_success,
            "rewards_and_costs": {
                TASK_REWARD_KEY: task_reward,
                HAZARD_COST_KEY: hazard_cost,
                SENSING_COST_KEY: sensing_cost,
            },
            "done_flags": {"terminal": bool(terminal), "truncated": bool(truncated)},
            "action_factors": {
                "sensing_action_available": True,
                "movement_action_available": True,
                "sensing_action_count": len(STAGE23_SENSING_ACTIONS),
                "movement_action_count": len(STAGE23_MOVEMENT_ACTIONS),
            },
            "identity": self._identity(agent),
        }

    def _identity(self, agent: str) -> dict[str, object]:
        agent_id = self.agent_name_mapping[agent]
        return {
            "team_id": 0,
            "agent_id": agent_id,
            "agent_name": agent,
            "agent_order": list(range(len(self.possible_agents))),
            "agent_name_order": list(self.possible_agents),
            "team_agent_ids": list(range(len(self.possible_agents))),
            "agent_id_mapping": {
                index: index for index in range(len(self.possible_agents))
            },
        }

    def _nearest_goal(self, position: Coordinate) -> Coordinate:
        return min(
            self._scenario.goals,
            key=lambda goal: (_manhattan(position, goal), goal[0], goal[1]),
        )

    def _local_hazards(self, position: Coordinate, radius: int) -> tuple[Coordinate, ...]:
        return self._local_cells(position, radius, self._scenario.hidden_hazard_cells)

    def _local_cells(
        self,
        position: Coordinate,
        radius: int,
        cells: tuple[Coordinate, ...],
    ) -> tuple[Coordinate, ...]:
        # Stage 23-A scenarios are intentionally tiny; spatial indexing beyond
        # cached full-scenario membership sets is out of scope for Stage 23-A.
        return tuple(
            cell
            for cell in cells
            if _manhattan(position, cell) <= radius
        )


def stage23_scenario_catalog() -> dict[str, Stage23Scenario]:
    """Return the deterministic Stage 23-A scenario catalog."""

    return {
        "standard_branching_hazard": Stage23Scenario(
            name="standard_branching_hazard",
            width=8,
            height=8,
            starts=((0, 0), (0, 1)),
            goals=((7, 7),),
            obstacles=((1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (5, 4), (5, 5)),
            hidden_hazard_cells=((2, 2), (3, 4), (4, 5)),
            description=(
                "Default branching grid with a clear safe route around obstacles "
                "and hidden hazard cells on a shorter branch."
            ),
        ),
        "risk_gate_hidden_hazard": Stage23Scenario(
            name="risk_gate_hidden_hazard",
            width=8,
            height=8,
            starts=((2, 0), (3, 0)),
            goals=((2, 4),),
            obstacles=tuple(),
            hidden_hazard_cells=((2, 2),),
            description=(
                "Compact gate scenario where the public shortest path crosses "
                "a hidden hazard, while local sensing can reveal the gate risk "
                "early enough for a longer safe route."
            ),
        ),
        "unit_empty": Stage23Scenario(
            name="unit_empty",
            width=8,
            height=8,
            starts=((0, 0), (0, 1)),
            goals=((0, 2),),
            obstacles=tuple(),
            hidden_hazard_cells=tuple(),
            description="Small contract scenario without hidden hazards or obstacles.",
        ),
        "unit_single_hazard": Stage23Scenario(
            name="unit_single_hazard",
            width=8,
            height=8,
            starts=((1, 0), (0, 0)),
            goals=((2, 2),),
            obstacles=tuple(),
            hidden_hazard_cells=((1, 1),),
            description="Single local hidden hazard for sensing and cost contracts.",
        ),
        # --- DR-D1/L1 "risk fork" family (sensing made instrumentally necessary by
        # STRUCTURE). Each fork scenario is an 8x8 grid with a vertical wall on
        # column 4 broken by exactly TWO open gate cells; both agents start left
        # (rows 3,4 col 0) and share a single goal (3,7), fixed across the family so
        # the pre-sensing observation is identical and the layouts are observationally
        # aliased. Exactly one gate is the hidden hazard; WHICH gate varies across a
        # fork group's layouts. The UNION of a group's two hazards is a start->goal cut
        # (no blind zero-hazard successful route: BAR-A), while each single layout
        # leaves a safe route via the OTHER gate (winnable: BAR-B). A never-sense
        # policy is therefore unsafe in half the layouts; only sensing at the gate
        # approach + rerouting is safe on both, so selective sensing is instrumentally
        # necessary (docs/decisions/DR-D1-L1.md, countersigned by construction). NEW
        # scenario ids only; every scenario above is byte-identical. Held-out fork
        # scenarios (gates 5,6) are DEFERRED to the Protocol-v1 lock (I-3).
        "risk_fork_train_upper": Stage23Scenario(
            name="risk_fork_train_upper",
            width=8,
            height=8,
            starts=((3, 0), (4, 0)),
            goals=((3, 7),),
            obstacles=((0, 4), (1, 4), (2, 4), (5, 4), (6, 4), (7, 4)),
            hidden_hazard_cells=((3, 4),),
            description=(
                "Risk-fork training layout (gates at rows 3,4): the hidden hazard is "
                "the UPPER gate, so the blind shortest route through it is unsafe; the "
                "safe route uses the lower gate. Sensing at the gate approach reveals "
                "which gate is hazardous."
            ),
        ),
        "risk_fork_train_lower": Stage23Scenario(
            name="risk_fork_train_lower",
            width=8,
            height=8,
            starts=((3, 0), (4, 0)),
            goals=((3, 7),),
            obstacles=((0, 4), (1, 4), (2, 4), (5, 4), (6, 4), (7, 4)),
            hidden_hazard_cells=((4, 4),),
            description=(
                "Risk-fork training layout (gates at rows 3,4): the hidden hazard is "
                "the LOWER gate; the safe route uses the upper gate. Observationally "
                "identical to risk_fork_train_upper until the hazard is sensed."
            ),
        ),
        "risk_fork_readiness_upper": Stage23Scenario(
            name="risk_fork_readiness_upper",
            width=8,
            height=8,
            starts=((3, 0), (4, 0)),
            goals=((3, 7),),
            obstacles=((0, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4)),
            hidden_hazard_cells=((1, 4),),
            description=(
                "Risk-fork readiness layout (gates shifted to rows 1,2): the hidden "
                "hazard is the upper gate. Tests the sensing SKILL generalising to a "
                "different gate position (not a memorised gate row)."
            ),
        ),
        "risk_fork_readiness_lower": Stage23Scenario(
            name="risk_fork_readiness_lower",
            width=8,
            height=8,
            starts=((3, 0), (4, 0)),
            goals=((3, 7),),
            obstacles=((0, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4)),
            hidden_hazard_cells=((2, 4),),
            description=(
                "Risk-fork readiness layout (gates at rows 1,2): the hidden hazard is "
                "the lower gate; the safe route uses the upper gate."
            ),
        ),
        # --- DR-READINESS-CURRICULUM gate-row curriculum family (SHIFT 27,
        # countersigned). Three additional observationally-aliased fork pairs at
        # NEW gate rows drawn from the free-row set {0,3,4,7} — row-disjoint from
        # the reserved readiness rows (1,2) and the deferred held-out rows (5,6),
        # so training on them never touches a test position. Same construction as
        # the DR-D1/L1 fork (obstacles = the six col-4 cells not in the gate pair;
        # the union of a pair's two hazards completes the vertical cut => BAR-A by
        # construction; each single layout winnable => BAR-B), same starts/goal,
        # upper = hazard at the smaller gate row. NEW ids only; every scenario
        # above stays byte-identical (docs/decisions/DR-readiness-curriculum.md).
        "risk_fork_curriculum_r0r3_upper": Stage23Scenario(
            name="risk_fork_curriculum_r0r3_upper",
            width=8,
            height=8,
            starts=((3, 0), (4, 0)),
            goals=((3, 7),),
            obstacles=((1, 4), (2, 4), (4, 4), (5, 4), (6, 4), (7, 4)),
            hidden_hazard_cells=((0, 4),),
            description=(
                "Risk-fork curriculum layout (gates at rows 0,3): the hidden hazard "
                "is the upper gate (row 0); the safe route uses the row-3 gate. "
                "Observationally aliased with risk_fork_curriculum_r0r3_lower."
            ),
        ),
        "risk_fork_curriculum_r0r3_lower": Stage23Scenario(
            name="risk_fork_curriculum_r0r3_lower",
            width=8,
            height=8,
            starts=((3, 0), (4, 0)),
            goals=((3, 7),),
            obstacles=((1, 4), (2, 4), (4, 4), (5, 4), (6, 4), (7, 4)),
            hidden_hazard_cells=((3, 4),),
            description=(
                "Risk-fork curriculum layout (gates at rows 0,3): the hidden hazard "
                "is the lower gate (row 3); the safe route uses the row-0 gate."
            ),
        ),
        "risk_fork_curriculum_r4r7_upper": Stage23Scenario(
            name="risk_fork_curriculum_r4r7_upper",
            width=8,
            height=8,
            starts=((3, 0), (4, 0)),
            goals=((3, 7),),
            obstacles=((0, 4), (1, 4), (2, 4), (3, 4), (5, 4), (6, 4)),
            hidden_hazard_cells=((4, 4),),
            description=(
                "Risk-fork curriculum layout (gates at rows 4,7): the hidden hazard "
                "is the upper gate (row 4); the safe route uses the row-7 gate. "
                "Observationally aliased with risk_fork_curriculum_r4r7_lower."
            ),
        ),
        "risk_fork_curriculum_r4r7_lower": Stage23Scenario(
            name="risk_fork_curriculum_r4r7_lower",
            width=8,
            height=8,
            starts=((3, 0), (4, 0)),
            goals=((3, 7),),
            obstacles=((0, 4), (1, 4), (2, 4), (3, 4), (5, 4), (6, 4)),
            hidden_hazard_cells=((7, 4),),
            description=(
                "Risk-fork curriculum layout (gates at rows 4,7): the hidden hazard "
                "is the lower gate (row 7); the safe route uses the row-4 gate."
            ),
        ),
        "risk_fork_curriculum_r0r7_upper": Stage23Scenario(
            name="risk_fork_curriculum_r0r7_upper",
            width=8,
            height=8,
            starts=((3, 0), (4, 0)),
            goals=((3, 7),),
            obstacles=((1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)),
            hidden_hazard_cells=((0, 4),),
            description=(
                "Risk-fork curriculum layout (gates at rows 0,7 - the maximum-span "
                "pair): the hidden hazard is the upper gate (row 0); the safe route "
                "uses the row-7 gate. Observationally aliased with "
                "risk_fork_curriculum_r0r7_lower."
            ),
        ),
        "risk_fork_curriculum_r0r7_lower": Stage23Scenario(
            name="risk_fork_curriculum_r0r7_lower",
            width=8,
            height=8,
            starts=((3, 0), (4, 0)),
            goals=((3, 7),),
            obstacles=((1, 4), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)),
            hidden_hazard_cells=((7, 4),),
            description=(
                "Risk-fork curriculum layout (gates at rows 0,7): the hidden hazard "
                "is the lower gate (row 7); the safe route uses the row-0 gate."
            ),
        ),
    }


def validate_stage23_actor_observation(observation: Mapping[str, object]) -> None:
    """Reject forbidden actor-visible information keys.

    The full-observation recursive scan covers centralized, global,
    privileged, hidden, critic, and other Stage 21-compatible forbidden
    actor-visible information aliases.
    """

    if not isinstance(observation, Mapping):
        raise TypeError("observation must be a mapping")
    # NEW-grid_environment-3: non-string keys are a type error (TypeError),
    # not a value error, matching the project TypeError-for-type convention.
    _require_string_mapping_keys(observation, "observation top-level key")
    for key in observation:
        if key in STAGE23_FORBIDDEN_ACTOR_KEYS:
            raise ValueError(
                f"observation contains forbidden actor-visible information key at observation.{key}"
            )
        if key not in STAGE23_ALLOWED_OBSERVATION_KEYS:
            raise ValueError(f"observation contains unsupported top-level key: {key}")
    actor_visible = observation.get("actor_visible")
    if not isinstance(actor_visible, Mapping):
        raise ValueError("observation requires actor_visible mapping")
    _require_string_mapping_keys(actor_visible, "actor_visible key")
    if "action_factors" in observation and not isinstance(
        observation.get("action_factors"),
        Mapping,
    ):
        raise ValueError("observation action_factors must be a mapping when present")
    if "identity" in observation and not isinstance(observation.get("identity"), Mapping):
        raise ValueError("observation identity must be a mapping when present")
    _scan_for_forbidden_actor_keys(observation, path="observation")
    # NEW-grid_environment-4: the previously separate string-ness and
    # unsupported-membership passes over actor_visible are consolidated into a
    # single loop (string-ness is already enforced above via
    # _require_string_mapping_keys); the missing-required pass stays separate so
    # its error precedence (unsupported-key before missing-field) is preserved.
    for key in actor_visible:
        if key not in STAGE23_ACTOR_VISIBLE_SCHEMA_KEY_SET:
            raise ValueError(f"actor_visible contains unsupported key: {key}")
    for key in STAGE23_ACTOR_VISIBLE_SCHEMA_KEYS:
        if key not in actor_visible:
            raise ValueError(f"actor_visible missing required field: {key}")


def _resolve_and_validate_scenario(config: Stage23EnvironmentConfig) -> Stage23Scenario:
    # Issue m-10 fix: Stage23EnvironmentConfig.__post_init__ already runs
    # this resolution (including BFS reachability); cache the result on
    # the frozen config as a non-field attribute so environment
    # construction reuses the config-validated scenario.
    cached = getattr(config, "_resolved_scenario_cache", None)
    if cached is not None:
        return cached
    catalog = stage23_scenario_catalog()
    if config.scenario_name not in catalog:
        raise ValueError(f"unknown Stage 23-A scenario: {config.scenario_name}")
    base = catalog[config.scenario_name]
    custom_layout_fields = (
        "start_positions",
        "goal_cells",
        "obstacle_cells",
        "hidden_hazard_cells",
    )
    custom_layout_supplied = any(
        getattr(config, field_name) is not None for field_name in custom_layout_fields
    )
    dimensions_match_catalog = (
        config.width == base.width and config.height == base.height
    )
    if not custom_layout_supplied and not dimensions_match_catalog:
        raise ValueError(
            "catalog scenario dimensions must match unless a full custom layout is supplied"
        )
    if custom_layout_supplied and not dimensions_match_catalog:
        missing = [
            field_name
            for field_name in custom_layout_fields
            if getattr(config, field_name) is None
        ]
        if missing:
            raise ValueError(
                "custom scenario dimensions require all layout fields: "
                + ", ".join(missing)
            )
    starts = config.start_positions if config.start_positions is not None else base.starts
    goals = config.goal_cells if config.goal_cells is not None else base.goals
    obstacles = config.obstacle_cells if config.obstacle_cells is not None else base.obstacles
    hazards = (
        config.hidden_hazard_cells
        if config.hidden_hazard_cells is not None
        else base.hidden_hazard_cells
    )
    scenario = Stage23Scenario(
        name=config.scenario_name,
        width=config.width,
        height=config.height,
        starts=tuple(starts),
        goals=tuple(sorted(goals)),
        obstacles=tuple(sorted(obstacles)),
        hidden_hazard_cells=tuple(sorted(hazards)),
        description=base.description,
    )
    if len(scenario.starts) != config.agent_count:
        raise ValueError("start_positions count must match agent_count")
    _validate_reachability(scenario)
    object.__setattr__(config, "_resolved_scenario_cache", scenario)
    return scenario


def _validate_disjoint_layout_sets(
    *,
    width: int,
    height: int,
    starts: tuple[Coordinate, ...],
    goals: tuple[Coordinate, ...],
    obstacles: tuple[Coordinate, ...],
    hidden_hazard_cells: tuple[Coordinate, ...],
) -> None:
    for name, cells in {
        "starts": starts,
        "goals": goals,
        "obstacles": obstacles,
        "hidden_hazard_cells": hidden_hazard_cells,
    }.items():
        for cell in cells:
            _require_in_grid(name, cell, width, height)
    if set(starts) & set(obstacles):
        raise ValueError("starts must not overlap obstacles")
    if set(starts) & set(hidden_hazard_cells):
        raise ValueError("starts must not overlap hidden hazard cells")
    if set(goals) & set(obstacles):
        raise ValueError("goals must not overlap obstacles")
    if set(goals) & set(hidden_hazard_cells):
        raise ValueError("goals must not overlap hidden hazard cells")
    if set(obstacles) & set(hidden_hazard_cells):
        raise ValueError("obstacles and hidden hazard cells must be disjoint")


def _validate_reachability(scenario: Stage23Scenario) -> None:
    obstacle_set = set(scenario.obstacles)
    goal_set = set(scenario.goals)
    for start in scenario.starts:
        visited = {start}
        queue: deque[Coordinate] = deque([start])
        reachable = False
        while queue:
            cell = queue.popleft()
            if cell in goal_set:
                reachable = True
                break
            for neighbor in _neighbors(cell, scenario.width, scenario.height):
                if neighbor in obstacle_set or neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        if not reachable:
            raise ValueError("every agent start must reach a goal while respecting obstacles")


def _neighbors(cell: Coordinate, width: int, height: int) -> tuple[Coordinate, ...]:
    row, column = cell
    candidates = ((row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1))
    return tuple(candidate for candidate in candidates if _in_grid(candidate, width, height))


def _scan_for_forbidden_actor_keys(value: object, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if isinstance(key, str) and key in STAGE23_FORBIDDEN_ACTOR_KEYS:
                raise ValueError(f"actor-visible payload contains forbidden key at {path}.{key}")
            _scan_for_forbidden_actor_keys(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _scan_for_forbidden_actor_keys(item, path=f"{path}[{index}]")


def _default_previous_status() -> PreviousStatus:
    return {
        "invalid_move": False,
        "blocked_move": False,
        "entered_hazard": False,
        "sensed": False,
        "revealed_hazard_count": 0,
        TASK_REWARD_KEY: 0.0,
        HAZARD_COST_KEY: 0.0,
        SENSING_COST_KEY: 0.0,
        "team_success": False,
    }


def _validate_coordinate_tuple(
    name: str,
    value: tuple[Coordinate, ...],
    width: int,
    height: int,
) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{name} must be a tuple")
    for cell in value:
        _validate_coordinate(name, cell)
        _require_in_grid(name, cell, width, height)
    if name != "start_positions" and tuple(sorted(value)) != value:
        raise ValueError(f"{name} must be sorted deterministically")
    _require_unique_cells(name, value)


def _require_unique_cells(name: str, value: tuple[Coordinate, ...]) -> None:
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must contain unique cells")


def _validate_coordinate(name: str, value: object) -> Coordinate:
    if not isinstance(value, tuple) or len(value) != 2:
        raise TypeError(f"{name} cells must be (row, column) tuples")
    row, column = value
    require_int_not_bool(f"{name} row", row)
    require_int_not_bool(f"{name} column", column)
    return row, column


def _require_in_grid(name: str, cell: Coordinate, width: int, height: int) -> None:
    if not _in_grid(cell, width, height):
        raise ValueError(f"{name} cell {cell!r} is outside the grid")


def _in_grid(cell: Coordinate, width: int, height: int) -> bool:
    row, column = cell
    return 0 <= row < height and 0 <= column < width


def _require_action_index(name: str, value: object, action_count: int) -> int:
    require_int_not_bool(name, value)
    # NEW-grid_environment-1: normalize through the base ``int.__index__`` slot
    # once, then range-check the normalized index, so a hostile ``int`` subclass
    # cannot override ``<`` / ``>=`` to slip past the [0, action_count) bound.
    index = int.__index__(value)
    if index < 0 or index >= action_count:
        raise ValueError(f"{name} must be in [0, {action_count - 1}]")
    return index


def _manhattan(left: Coordinate, right: Coordinate) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _cell_delta(left: Coordinate, right: Coordinate) -> list[int]:
    return [right[0] - left[0], right[1] - left[1]]


def _cell_to_json(cell: Coordinate) -> list[int]:
    return [int(cell[0]), int(cell[1])]


def _cells_to_json(cells: tuple[Coordinate, ...]) -> list[list[int]]:
    return [_cell_to_json(cell) for cell in cells]


def _optional_cells_to_json(cells: tuple[Coordinate, ...] | None) -> list[list[int]] | None:
    if cells is None:
        return None
    return _cells_to_json(cells)

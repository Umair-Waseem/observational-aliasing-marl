"""Stage 24 active-sensing scenario diagnostics and comparators.

This module is readiness-only. It does not run final evaluation, does not write
artifacts, does not create claim evidence, and does not import the Stage 22
development environment adapter.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import random
from typing import Any

from raas_marl.environments.active_sensing.grid_environment import (
    HAZARD_COST_KEY,
    MOVEMENT_ACTION_FIELD,
    SENSING_ACTION_FIELD,
    SENSING_COST_KEY,
    STAGE23_AGENT_NAMES,
    STAGE23_MOVEMENT_DELTAS,
    STAGE23_SENSE_ACTION_INDEX,
    TASK_REWARD_KEY,
    RiskAwareActiveSensingGridEnvironment,
    Stage23EnvironmentConfig,
    Stage23Scenario,
    _cells_to_json,
    _in_grid,
    stage23_scenario_catalog,
    validate_stage23_actor_observation,
)
from raas_marl.mappo_lagrangian._validation import (
    finite_numeric_scalar,
    require_int_not_bool,
    require_positive_int,
)


Coordinate = tuple[int, int]
PolicyState = dict[str, Any]
PolicyFn = Callable[
    [
        Mapping[str, Mapping[str, object]],
        RiskAwareActiveSensingGridEnvironment,
        PolicyState,
        random.Random,
    ],
    dict[str, dict[str, int]],
]

# Movement (row, column) deltas are owned by grid_environment.STAGE23_MOVEMENT_DELTAS
# (dedup contract). It carries keys 0..4; the "stay" delta (key 0 -> (0, 0)) is
# harmless here because _shortest_path only indexes _NEIGHBOR_ORDER (1..4) and
# _movement_action short-circuits the equal-cell case before scanning the table.
_NEIGHBOR_ORDER = (1, 2, 3, 4)
# Per-scenario / per-variant reset-seed stride (issue NEW-stage24_diagnostics-6:
# previously an unnamed magic 100). Each of the <=4 catalog scenarios and <=3
# Stage 24-A variants gets a disjoint seed band of width 100, which is ample for
# the fixed, bounded catalog while keeping every derived seed strictly positive.
_SCENARIO_SEED_STRIDE = 100
_POLICY_ORDER = (
    "no_sense_shortest_path",
    "always_sense_shortest_path",
    "random_policy",
    "risk_aware_oracle_or_heuristic",
    "selective_sense_risk_aware",
)
# Single definition of the public gate risk zone (rows 1-3 x cols 1-3) for
# the risk_gate_hidden_hazard scenario family (issue m-18 resolution).
_PUBLIC_GATE_RISK_ZONE_CELLS: tuple[Coordinate, ...] = tuple(
    (row, col) for row in (1, 2, 3) for col in (1, 2, 3)
)
# A multi-cell public risk zone must carry at least this many extra
# uncertainty cells beyond the hidden hazards it covers, so the zone can
# never pinpoint the exact hidden layout (issue m-19: previously an
# unnamed magic constant).
_MIN_EXTRA_UNCERTAINTY_CELLS = 2
# DR-D1/L1 "risk fork" families: catalog scenario ids grouped by role. Training is
# the T1-D6 --scenario set (the collection round-robins across it); readiness is the
# generalization surface for the C1 grade. The held-out fork group (gates 5,6) is
# DEFERRED to the Protocol-v1 lock (I-3), so it is not named here. These are kept
# SEPARATE from stage24a_hazard_layout_variants() (the original three risk_gate
# variants), which stays byte-identical. The fork wall is the vertical obstacle line
# on this column; the open gates are the wall-column cells NOT in a scenario's
# obstacle set.
_FORK_WALL_COLUMN = 4
_FORK_TRAIN_SCENARIOS: tuple[str, ...] = (
    "risk_fork_train_lower",
    "risk_fork_train_upper",
)
_FORK_READINESS_SCENARIOS: tuple[str, ...] = (
    "risk_fork_readiness_lower",
    "risk_fork_readiness_upper",
)
# DR-READINESS-CURRICULUM (SHIFT 27, countersigned): the gate-row curriculum
# extension — three additional aliased fork pairs at gate rows {0,3}/{4,7}/{0,7}
# (all inside the free-row set {0,3,4,7}; row-disjoint from readiness (1,2) and
# held-out (5,6)). Kept SEPARATE from _FORK_TRAIN_SCENARIOS because that tuple
# (via fork_hazard_layout_families) IS the pinned C1 grade-surface registry and
# the readiness-probe admissibility root; the curriculum ids are TRAINING-side
# additions only. Mirrors are adjacent (lower, upper) per pair — the §3.3
# balance conventions the AM-H13 re-scope consumes.
_FORK_CURRICULUM_SCENARIOS: tuple[str, ...] = (
    "risk_fork_curriculum_r0r3_lower",
    "risk_fork_curriculum_r0r3_upper",
    "risk_fork_curriculum_r4r7_lower",
    "risk_fork_curriculum_r4r7_upper",
    "risk_fork_curriculum_r0r7_lower",
    "risk_fork_curriculum_r0r7_upper",
)
_FORK_SCENARIOS: frozenset[str] = frozenset(
    _FORK_TRAIN_SCENARIOS + _FORK_READINESS_SCENARIOS + _FORK_CURRICULUM_SCENARIOS
)


@dataclass(frozen=True)
class Stage24ComparatorConfig:
    """Bounded deterministic comparator settings for Stage 24 diagnostics."""

    scenario_name: str
    seed: int = 2400
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_name, str) or not self.scenario_name.strip():
            raise ValueError("scenario_name must be a non-empty string")
        require_positive_int("seed", self.seed, error_suffix="must be positive")
        if self.max_steps is not None:
            require_positive_int("max_steps", self.max_steps, error_suffix="must be positive")
        if self.scenario_name not in stage23_scenario_catalog():
            raise ValueError(f"unknown Stage 23-A scenario: {self.scenario_name}")


@dataclass(frozen=True)
class Stage24AHazardLayoutVariant:
    """Stage 24-A readiness variant with broad public risk-zone metadata."""

    name: str
    group: str
    hidden_hazard_cells: tuple[Coordinate, ...]
    public_risk_zone_cells: tuple[Coordinate, ...]
    description: str
    width: int = 8
    height: int = 8

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("variant name must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValueError("variant description must be a non-empty string")
        if self.group not in {"training", "readiness", "held_out"}:
            raise ValueError("variant group must be training, readiness, or held_out")
        require_positive_int("variant width", self.width, error_suffix="must be positive")
        require_positive_int("variant height", self.height, error_suffix="must be positive")
        hidden = _validate_coordinate_collection(
            "variant hidden_hazard_cells",
            self.hidden_hazard_cells,
            self.width,
            self.height,
        )
        public = _validate_coordinate_collection(
            "variant public_risk_zone_cells",
            self.public_risk_zone_cells,
            self.width,
            self.height,
        )
        if set(hidden) - set(public):
            raise ValueError("public risk zone must cover hidden hazard uncertainty cells")
        if _public_risk_zone_reveals_exact_hidden_hazard(hidden, public):
            raise ValueError("public risk zone must be broader than exact hidden hazards")


def stage24a_hazard_layout_variants() -> dict[str, Stage24AHazardLayoutVariant]:
    """Return deterministic Stage 24-A hazard-layout variants.

    The public risk zone is deliberately broader than any exact hidden-hazard
    coordinate. It is uncertainty metadata for readiness diagnostics, not an
    actor-visible exact hazard map.
    """

    public_gate_zone = _PUBLIC_GATE_RISK_ZONE_CELLS
    variants = {
        "risk_gate_train_center": Stage24AHazardLayoutVariant(
            name="risk_gate_train_center",
            group="training",
            hidden_hazard_cells=((2, 2),),
            public_risk_zone_cells=public_gate_zone,
            description="Training-layout variant matching the original gate center.",
        ),
        "risk_gate_readiness_shifted": Stage24AHazardLayoutVariant(
            name="risk_gate_readiness_shifted",
            group="readiness",
            hidden_hazard_cells=((2, 3),),
            public_risk_zone_cells=public_gate_zone,
            description="Readiness variant with the gate hazard shifted within the public risk zone.",
        ),
        "risk_gate_heldout_near_gate": Stage24AHazardLayoutVariant(
            name="risk_gate_heldout_near_gate",
            group="held_out",
            hidden_hazard_cells=((2, 2), (3, 2)),
            public_risk_zone_cells=public_gate_zone,
            description="Held-out variant with a distinct non-trivial hidden gate layout.",
        ),
    }
    _validate_stage24a_variant_collection(variants)
    return variants


def fork_hazard_layout_families() -> dict[str, tuple[str, ...]]:
    """Return the DR-D1/L1 risk-fork catalog-scenario families by role.

    ``{"training": (...), "readiness": (...)}`` — the observationally-aliased
    multi-layout fork families that make selective sensing instrumentally necessary
    (the union of a group's hidden hazards is a start->goal cut; each single layout
    stays winnable). Held-out (gates 5,6) is deferred to the Protocol-v1 lock (I-3)
    and is intentionally absent. Distinct from :func:`stage24a_hazard_layout_variants`
    (the byte-identical original risk_gate variants); the C1 grade enumerates these.
    """

    return {
        "training": _FORK_TRAIN_SCENARIOS,
        "readiness": _FORK_READINESS_SCENARIOS,
    }


def fork_curriculum_scenarios() -> tuple[str, ...]:
    """Return the DR-READINESS-CURRICULUM gate-row curriculum scenario ids.

    Three additional observationally-aliased fork pairs at gate rows {0,3},
    {4,7}, and {0,7} — TRAINING-side additions that diversify the trained gate
    position (mirrors adjacent, ``(lower, upper)`` per pair). Deliberately NOT
    part of :func:`fork_hazard_layout_families`: that registry is the pinned C1
    grade surface and the readiness-probe admissibility root, and the readiness
    pair (gates rows 1,2) remains the reserved generalization probe — never a
    training surface. Held-out (gates rows 5,6) stays absent from code (I-3).
    Every curriculum pair is BAR-A/BAR-B valid by construction (executable:
    :func:`verify_fork_family_forces_sensing` per pair, pinned in the suite).
    """

    return _FORK_CURRICULUM_SCENARIOS


def no_sense_shortest_path(
    scenario_name: str,
    *,
    seed: int = 2400,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run the public-geometry shortest path with sensing disabled."""

    return _run_policy(
        Stage24ComparatorConfig(scenario_name, seed=seed, max_steps=max_steps),
        policy_name="no_sense_shortest_path",
        policy=_no_sense_policy,
        diagnostic_only=False,
        uses_hidden_hazards_for_decision=False,
    )


def always_sense_shortest_path(
    scenario_name: str,
    *,
    seed: int = 2400,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run the public-geometry shortest path while sensing on every step."""

    return _run_policy(
        Stage24ComparatorConfig(scenario_name, seed=seed, max_steps=max_steps),
        policy_name="always_sense_shortest_path",
        policy=_always_sense_policy,
        diagnostic_only=False,
        uses_hidden_hazards_for_decision=False,
    )


def random_policy(
    scenario_name: str,
    *,
    seed: int = 2400,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run a fixed-seed random valid-action comparator."""

    return _run_policy(
        Stage24ComparatorConfig(scenario_name, seed=seed, max_steps=max_steps),
        policy_name="random_policy",
        policy=_random_policy,
        diagnostic_only=False,
        uses_hidden_hazards_for_decision=False,
    )


def risk_aware_oracle_or_heuristic(
    scenario_name: str,
    *,
    seed: int = 2400,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run a diagnostic-only path planner that may avoid hidden hazards."""

    return _run_policy(
        Stage24ComparatorConfig(scenario_name, seed=seed, max_steps=max_steps),
        policy_name="risk_aware_oracle_or_heuristic",
        policy=_risk_aware_policy,
        diagnostic_only=True,
        uses_hidden_hazards_for_decision=True,
    )


def selective_sense_risk_aware(
    scenario_name: str,
    *,
    seed: int = 2400,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run a selective local-reveal comparator using only observed reveals."""

    return _run_policy(
        Stage24ComparatorConfig(scenario_name, seed=seed, max_steps=max_steps),
        policy_name="selective_sense_risk_aware",
        policy=_selective_sense_policy,
        diagnostic_only=False,
        uses_hidden_hazards_for_decision=False,
    )


def run_stage23_scenario_diagnostics(
    *,
    scenario_names: tuple[str, ...] | None = None,
    seed: int = 2400,
) -> dict[str, Any]:
    """Run Stage 24 readiness diagnostics over Stage 23-A scenarios."""

    require_positive_int("seed", seed, error_suffix="must be positive")
    catalog = stage23_scenario_catalog()
    names = tuple(sorted(catalog)) if scenario_names is None else scenario_names
    if not names:
        raise ValueError("scenario_names must be non-empty")
    for name in names:
        if not isinstance(name, str):
            raise TypeError("scenario_names entries must be strings")
    if len(set(names)) != len(names):
        raise ValueError("scenario_names must be unique")

    scenario_records: list[dict[str, Any]] = []
    comparator_records: dict[str, dict[str, Any]] = {}
    for index, name in enumerate(names):
        if name not in catalog:
            raise ValueError(f"unknown Stage 23-A scenario: {name}")
        scenario_seed = seed + index * _SCENARIO_SEED_STRIDE
        comparisons = _run_stage24_policy_comparisons(name, scenario_seed)
        _attach_pairwise_comparator_fields(comparisons)
        comparator_records[name] = comparisons
        scenario_records.append(_scenario_diagnostic_record(name, comparisons))

    claim_relevant = [
        record["scenario_name"]
        for record in scenario_records
        if record["sufficient_for_active_sensing_claim_testing"]
    ]
    return {
        "stage": "24",
        "status": "stage24_readiness_diagnostics_not_final_evaluation",
        "claim_status": "not supported / not ready for final evaluation",
        "final_evaluation_run": False,
        "claim_evidence_created": False,
        "paper_facing_output_created": False,
        "bayesian_belief_claim_made": False,
        "formal_voi_claim_made": False,
        "env_adapter_used": False,
        "scenario_count": len(scenario_records),
        "claim_relevant_scenario_names": claim_relevant,
        "scenarios": scenario_records,
        "comparators": comparator_records,
    }


def run_stage24a_variant_readiness_diagnostics(*, seed: int = 24400) -> dict[str, Any]:
    """Run deterministic Stage 24-A hazard-layout readiness diagnostics."""

    require_positive_int("seed", seed, error_suffix="must be positive")
    variants = stage24a_hazard_layout_variants()
    training_layouts = {
        frozenset(variant.hidden_hazard_cells)
        for variant in variants.values()
        if variant.group == "training"
    }
    records: dict[str, dict[str, Any]] = {}
    for index, variant in enumerate(variants.values()):
        variant_seed = seed + index * _SCENARIO_SEED_STRIDE
        env_config = _stage24a_variant_environment_config(variant, seed=variant_seed)
        comparisons = {
            policy_name: _run_variant_policy(
                policy_name,
                env_config,
                variant=variant,
                seed=variant_seed,
            )
            for policy_name in _POLICY_ORDER
        }
        _attach_pairwise_comparator_fields(comparisons)
        semantics = _stage24a_variant_group_semantics(
            variant,
            comparisons,
            training_layouts=training_layouts,
        )
        if not semantics["valid_for_group"]:
            raise ValueError(
                f"Stage 24-A variant failed {variant.group} semantics: {variant.name}"
            )
        records[variant.name] = {
            "variant_name": variant.name,
            "group": variant.group,
            "hidden_hazard_cells": _cells_to_json(variant.hidden_hazard_cells),
            "public_risk_zone_cell_count": len(variant.public_risk_zone_cells),
            "public_risk_zone_reveals_exact_hidden_hazard": (
                _public_risk_zone_reveals_exact_hidden_hazard(
                    variant.hidden_hazard_cells,
                    variant.public_risk_zone_cells,
                )
            ),
            "group_semantics": semantics,
            "comparators": comparisons,
        }
    return {
        "stage": "24-A",
        "status": "stage24a_variant_readiness_not_final_evaluation",
        "claim_status": "not supported / not ready for final evaluation",
        "final_evaluation_run": False,
        "claim_evidence_created": False,
        "variant_count": len(records),
        "training_variant_names": [
            name for name, variant in variants.items() if variant.group == "training"
        ],
        "readiness_variant_names": [
            name for name, variant in variants.items() if variant.group == "readiness"
        ],
        "held_out_variant_names": [
            name for name, variant in variants.items() if variant.group == "held_out"
        ],
        "variants": records,
    }


def _run_stage24_policy_comparisons(
    scenario_name: str,
    seed: int,
) -> dict[str, dict[str, Any]]:
    return {
        policy_name: _run_named_scenario_policy(policy_name, scenario_name, seed)
        for policy_name in _POLICY_ORDER
    }


def _run_named_scenario_policy(
    policy_name: str,
    scenario_name: str,
    seed: int,
) -> dict[str, Any]:
    if policy_name == "no_sense_shortest_path":
        return no_sense_shortest_path(scenario_name, seed=seed)
    if policy_name == "always_sense_shortest_path":
        return always_sense_shortest_path(scenario_name, seed=seed)
    if policy_name == "random_policy":
        return random_policy(scenario_name, seed=seed)
    if policy_name == "risk_aware_oracle_or_heuristic":
        return risk_aware_oracle_or_heuristic(scenario_name, seed=seed)
    if policy_name == "selective_sense_risk_aware":
        return selective_sense_risk_aware(scenario_name, seed=seed)
    raise ValueError(f"unknown Stage 24 policy comparator: {policy_name}")


def _run_variant_policy(
    policy_name: str,
    environment_config: Stage23EnvironmentConfig,
    *,
    variant: Stage24AHazardLayoutVariant,
    seed: int,
) -> dict[str, Any]:
    if policy_name == "no_sense_shortest_path":
        policy, diagnostic_only, uses_hidden = _no_sense_policy, False, False
    elif policy_name == "always_sense_shortest_path":
        policy, diagnostic_only, uses_hidden = _always_sense_policy, False, False
    elif policy_name == "random_policy":
        policy, diagnostic_only, uses_hidden = _random_policy, False, False
    elif policy_name == "risk_aware_oracle_or_heuristic":
        policy, diagnostic_only, uses_hidden = _risk_aware_policy, True, True
    elif policy_name == "selective_sense_risk_aware":
        policy, diagnostic_only, uses_hidden = _selective_sense_policy, False, False
    else:
        raise ValueError(f"unknown Stage 24-A variant policy comparator: {policy_name}")
    return _run_policy_on_environment(
        environment_config,
        seed=seed,
        policy_name=policy_name,
        policy=policy,
        diagnostic_only=diagnostic_only,
        uses_hidden_hazards_for_decision=uses_hidden,
        variant_name=variant.name,
        public_risk_zone_cells=variant.public_risk_zone_cells,
    )


def _stage24a_variant_group_semantics(
    variant: Stage24AHazardLayoutVariant,
    comparisons: Mapping[str, Mapping[str, Any]],
    *,
    training_layouts: set[frozenset[Coordinate]],
) -> dict[str, Any]:
    no_sense = comparisons["no_sense_shortest_path"]
    selective = comparisons["selective_sense_risk_aware"]
    # hazard_cost_sum is an exact integer multiple of the per-step hazard cost
    # accumulated over deterministic episodes, so the ==0.0 and < comparisons
    # below are exact and tolerance-free (issue NEW-stage24_diagnostics-13).
    no_sense_zero_hazard_success = (
        bool(no_sense["team_success"])
        and float(no_sense["hazard_cost_sum"]) == 0.0
        and int(no_sense["hazard_entry_count"]) == 0
    )
    selective_improves_hazard = float(selective["hazard_cost_sum"]) < float(
        no_sense["hazard_cost_sum"]
    )
    selective_improves_success = bool(selective["team_success"]) and not bool(
        no_sense["team_success"]
    )
    active_sensing_value = selective_improves_hazard or selective_improves_success
    distinct_from_training = frozenset(variant.hidden_hazard_cells) not in training_layouts
    if variant.group == "training":
        valid_for_group = True
    elif variant.group == "readiness":
        valid_for_group = active_sensing_value
    else:
        valid_for_group = distinct_from_training and (
            active_sensing_value or not no_sense_zero_hazard_success
        )
    return {
        "group": variant.group,
        "valid_for_group": valid_for_group,
        "no_sense_zero_hazard_success": no_sense_zero_hazard_success,
        "selective_improves_hazard_exposure": selective_improves_hazard,
        "selective_improves_success": selective_improves_success,
        "active_sensing_value": active_sensing_value,
        "held_out_distinct_from_training": distinct_from_training,
    }


def _run_policy(
    config: Stage24ComparatorConfig,
    *,
    policy_name: str,
    policy: PolicyFn,
    diagnostic_only: bool,
    uses_hidden_hazards_for_decision: bool,
) -> dict[str, Any]:
    scenario = stage23_scenario_catalog()[config.scenario_name]
    return _run_policy_on_environment(
        Stage23EnvironmentConfig(
            scenario_name=scenario.name,
            width=scenario.width,
            height=scenario.height,
            max_steps=scenario_default_max_steps(config, scenario),
            seed=config.seed,
        ),
        seed=config.seed,
        policy_name=policy_name,
        policy=policy,
        diagnostic_only=diagnostic_only,
        uses_hidden_hazards_for_decision=uses_hidden_hazards_for_decision,
        variant_name=None,
        public_risk_zone_cells=_stage24a_public_risk_zone_cells(scenario.name),
    )


def _run_policy_on_environment(
    environment_config: Stage23EnvironmentConfig,
    *,
    seed: int,
    policy_name: str,
    policy: PolicyFn,
    diagnostic_only: bool,
    uses_hidden_hazards_for_decision: bool,
    variant_name: str | None,
    public_risk_zone_cells: tuple[Coordinate, ...],
) -> dict[str, Any]:
    """Run one deterministic comparator episode and return its diagnostic record.

    Constructs a fresh Stage 23-A environment, drives it to termination or
    truncation with ``policy``, and accumulates per-step reward/cost/action
    totals. Writes no files (issue NEW-stage24_diagnostics-10: docstring added).
    """

    environment = RiskAwareActiveSensingGridEnvironment(environment_config)
    observations, infos = environment.reset(seed=seed)
    _validate_observation_payload(observations)
    rng = random.Random(seed)
    state: PolicyState = {
        "revealed_hazards_by_agent": {agent: set() for agent in STAGE23_AGENT_NAMES},
        "sensed_positions_by_agent": {agent: set() for agent in STAGE23_AGENT_NAMES},
        "public_risk_zone_cells": set(public_risk_zone_cells),
        "used_revealed_information_to_adapt_movement": False,
    }
    # Keys renamed from sensing_action_count / movement_action_count (issues
    # NEW-stage24_diagnostics-2 / -14): those names collided with the
    # environment action_factors "*_action_count" ACTION-SPACE-SIZE fields.
    # These accumulators are OBSERVED action totals across both agents.
    totals = {
        "task_reward_sum": 0.0,
        "hazard_cost_sum": 0.0,
        "sensing_cost_sum": 0.0,
        "hazard_entry_count": 0,
        "total_sensing_actions": 0,
        "total_movement_actions": 0,
        "action_decision_count": 0,
        "revealed_hazard_count": 0,
    }
    steps = 0
    team_success = False
    terminal = False
    truncated = False
    while environment.agents:
        state["step_index"] = steps
        _update_revealed_memory(observations, state)
        actions = policy(observations, environment, state, rng)
        _validate_action_payload(actions, expected_agents=tuple(environment.agents))
        for agent, action in actions.items():
            totals["action_decision_count"] += 1
            # Count any non-no-sense action (issue NEW-stage24_diagnostics-1 /
            # NEW-methodology-1 code-hook): "!= 0" is numerically identical to the
            # old "== 1" for today's binary sensing factor and matches the
            # stage24_collector "!= 0" convention. Sensing actions are emitted with
            # the named STAGE23_SENSE_ACTION_INDEX; generalising the sensing factor
            # beyond binary is a Phase-6 methodology decision.
            if action[SENSING_ACTION_FIELD] != 0:
                totals["total_sensing_actions"] += 1
            if action[MOVEMENT_ACTION_FIELD] != 0:
                totals["total_movement_actions"] += 1
        observations, _rewards, terminations, truncations, infos = environment.step(actions)
        _validate_observation_payload(observations)
        steps += 1
        terminal = any(bool(value) for value in terminations.values())
        truncated = any(bool(value) for value in truncations.values())
        for info in infos.values():
            totals["task_reward_sum"] += float(info[TASK_REWARD_KEY])
            totals["hazard_cost_sum"] += float(info[HAZARD_COST_KEY])
            totals["sensing_cost_sum"] += float(info[SENSING_COST_KEY])
            totals["hazard_entry_count"] += int(bool(info["entered_hazard"]))
            totals["revealed_hazard_count"] += int(info["revealed_hazard_count"])
            team_success = team_success or bool(info["team_success"])

    action_count = totals["action_decision_count"]
    sensing_rate = (
        float(totals["total_sensing_actions"]) / float(action_count)
        if action_count
        else 0.0
    )
    return {
        "policy_name": policy_name,
        "scenario_name": environment.scenario.name,
        "variant_name": variant_name,
        "seed": seed,
        "diagnostic_only": diagnostic_only,
        "uses_hidden_hazards_for_decision": uses_hidden_hazards_for_decision,
        "success_rate": 1.0 if team_success else 0.0,
        "team_success": team_success,
        "terminal": terminal,
        "truncated": truncated,
        "steps": steps,
        "mean_steps_to_success": float(steps) if team_success else None,
        "task_reward_sum": totals["task_reward_sum"],
        "hazard_cost_sum": totals["hazard_cost_sum"],
        "sensing_cost_sum": totals["sensing_cost_sum"],
        "hazard_entry_count": totals["hazard_entry_count"],
        "total_sensing_actions": totals["total_sensing_actions"],
        "total_movement_actions": totals["total_movement_actions"],
        "action_decision_count": action_count,
        "sensing_rate": sensing_rate,
        "revealed_hazard_count": totals["revealed_hazard_count"],
        "sensing_can_reveal_hidden_local_hazard_information": (
            totals["revealed_hazard_count"] > 0
        ),
        "used_revealed_information_to_adapt_movement": bool(
            state["used_revealed_information_to_adapt_movement"]
        ),
        "final_evaluation_run": False,
        "claim_evidence_created": False,
    }


def scenario_default_max_steps(
    config: Stage24ComparatorConfig,
    scenario: Stage23Scenario,
) -> int:
    if config.max_steps is not None:
        return config.max_steps
    return Stage23EnvironmentConfig(
        scenario_name=scenario.name,
        width=scenario.width,
        height=scenario.height,
    ).max_steps


def _no_sense_policy(
    observations: Mapping[str, Mapping[str, object]],
    environment: RiskAwareActiveSensingGridEnvironment,
    state: PolicyState,
    rng: random.Random,
) -> dict[str, dict[str, int]]:
    """Follow the public-geometry shortest path, never sensing."""

    return _shortest_path_actions(
        observations,
        environment,
        sensing_action=0,
        avoid_hazards=(),
    )


def _always_sense_policy(
    observations: Mapping[str, Mapping[str, object]],
    environment: RiskAwareActiveSensingGridEnvironment,
    state: PolicyState,
    rng: random.Random,
) -> dict[str, dict[str, int]]:
    """Follow the public-geometry shortest path, sensing every step."""

    return _shortest_path_actions(
        observations,
        environment,
        sensing_action=STAGE23_SENSE_ACTION_INDEX,
        avoid_hazards=(),
    )


def _random_policy(
    observations: Mapping[str, Mapping[str, object]],
    environment: RiskAwareActiveSensingGridEnvironment,
    state: PolicyState,
    rng: random.Random,
) -> dict[str, dict[str, int]]:
    """Sample a uniform valid (sensing, movement) action per active agent."""

    return {
        agent: {
            SENSING_ACTION_FIELD: rng.randrange(2),
            MOVEMENT_ACTION_FIELD: rng.randrange(5),
        }
        for agent in STAGE23_AGENT_NAMES
        if agent in environment.agents
    }


def _risk_aware_policy(
    observations: Mapping[str, Mapping[str, object]],
    environment: RiskAwareActiveSensingGridEnvironment,
    state: PolicyState,
    rng: random.Random,
) -> dict[str, dict[str, int]]:
    """Diagnostic-only planner that routes around the hidden hazard cells.

    Reads ``environment.scenario.hidden_hazard_cells`` directly, so it is
    flagged ``diagnostic_only`` / ``uses_hidden_hazards_for_decision`` and is
    never a deployable actor policy (it sees privileged state).
    """

    return _shortest_path_actions(
        observations,
        environment,
        sensing_action=0,
        avoid_hazards=environment.scenario.hidden_hazard_cells,
    )


def _selective_sense_policy(
    observations: Mapping[str, Mapping[str, object]],
    environment: RiskAwareActiveSensingGridEnvironment,
    state: PolicyState,
    rng: random.Random,
) -> dict[str, dict[str, int]]:
    """Sense selectively inside the public risk zone, then move on reveals.

    Reveal-offset note (issue NEW-stage24_diagnostics-12): the sensing decision
    is keyed on the agent's PRE-move ``position``, but the environment reveals
    hazards around the agent's POST-move position. The sensed cell therefore does
    not equal the centre of the neighbourhood that gets revealed this step; the
    reveal informs the NEXT step's movement. Gating/recording on the post-move
    position is a behavioural change deferred to Phase 6.
    """

    actions: dict[str, dict[str, int]] = {}
    for agent in environment.agents:
        position = _position_from_observation(observations[agent])
        revealed = tuple(sorted(state["revealed_hazards_by_agent"][agent]))
        public_next = _next_shortest_step(
            position,
            environment.scenario,
            avoid_hazards=(),
        )
        reveal_next = _next_shortest_step(
            position,
            environment.scenario,
            avoid_hazards=revealed,
        )
        if revealed and public_next != reveal_next:
            state["used_revealed_information_to_adapt_movement"] = True
        public_risk_zone = set(state["public_risk_zone_cells"])
        sensed_positions = state["sensed_positions_by_agent"][agent]
        public_risk_near_route = position in public_risk_zone or public_next in public_risk_zone
        should_sense = (
            public_risk_near_route
            and position not in sensed_positions
            and not revealed
            and position not in set(environment.scenario.goals)
        )
        if should_sense:
            sensed_positions.add(position)
        actions[agent] = {
            SENSING_ACTION_FIELD: STAGE23_SENSE_ACTION_INDEX if should_sense else 0,
            MOVEMENT_ACTION_FIELD: _movement_action(position, reveal_next),
        }
    return actions


def _shortest_path_actions(
    observations: Mapping[str, Mapping[str, object]],
    environment: RiskAwareActiveSensingGridEnvironment,
    *,
    sensing_action: int,
    avoid_hazards: tuple[Coordinate, ...],
) -> dict[str, dict[str, int]]:
    return {
        agent: {
            SENSING_ACTION_FIELD: sensing_action,
            MOVEMENT_ACTION_FIELD: _movement_action(
                _position_from_observation(observations[agent]),
                _next_shortest_step(
                    _position_from_observation(observations[agent]),
                    environment.scenario,
                    avoid_hazards=avoid_hazards,
                ),
            ),
        }
        for agent in environment.agents
    }


def _next_shortest_step(
    position: Coordinate,
    scenario: Stage23Scenario,
    *,
    avoid_hazards: tuple[Coordinate, ...],
) -> Coordinate:
    path = _shortest_path(
        position,
        scenario.goals,
        width=scenario.width,
        height=scenario.height,
        obstacles=scenario.obstacles,
        avoid_cells=avoid_hazards,
    )
    return path[0] if path else position


def _shortest_path(
    start: Coordinate,
    goals: tuple[Coordinate, ...],
    *,
    width: int,
    height: int,
    obstacles: tuple[Coordinate, ...],
    avoid_cells: tuple[Coordinate, ...],
) -> list[Coordinate]:
    """Return the cells after ``start`` on a shortest path to the nearest goal.

    Uniform-cost breadth-first search on the 4-connected grid (Moore 1959),
    expanding neighbours in the fixed ``_NEIGHBOR_ORDER`` (N, S, W, E) so the
    result is deterministic. ``avoid_cells`` are treated as blocked unless they
    are goals. Returns ``[]`` when ``start`` is already a goal or no goal is
    reachable.
    """

    goal_set = set(goals)
    if start in goal_set:
        return []
    blocked = set(obstacles) | {cell for cell in avoid_cells if cell not in goal_set}
    queue: deque[Coordinate] = deque([start])
    parent: dict[Coordinate, Coordinate | None] = {start: None}
    while queue:
        current = queue.popleft()
        for move in _NEIGHBOR_ORDER:
            neighbor = _add(current, STAGE23_MOVEMENT_DELTAS[move])
            if (
                neighbor in parent
                or neighbor in blocked
                or not _in_grid(neighbor, width, height)
            ):
                continue
            parent[neighbor] = current
            if neighbor in goal_set:
                return _reconstruct_path(neighbor, parent)
            queue.append(neighbor)
    return []


def _reconstruct_path(
    goal: Coordinate,
    parent: Mapping[Coordinate, Coordinate | None],
) -> list[Coordinate]:
    path = [goal]
    current = goal
    while parent[current] is not None:
        current = parent[current]  # type: ignore[assignment]
        path.append(current)
    path.reverse()
    return path[1:]


def _attach_pairwise_comparator_fields(comparisons: dict[str, dict[str, Any]]) -> None:
    """Attach cross-comparator fields to each record, mutating in place.

    Issue NEW-stage24_diagnostics-9: this returns None and adds the pairwise
    "*_vs_no_sense" / "*_than_always_sense" keys directly into each record dict.
    Callers pass a fresh per-scenario/per-variant ``comparisons`` mapping and
    rely on the mutation; there is no independent return value.
    """

    no_sense = comparisons["no_sense_shortest_path"]
    always = comparisons["always_sense_shortest_path"]
    for record in comparisons.values():
        record["no_sense_succeeds_with_zero_hazard"] = (
            no_sense["team_success"] and no_sense["hazard_entry_count"] == 0
        )
        record["reduced_hazard_exposure_vs_no_sense"] = (
            record["hazard_entry_count"] < no_sense["hazard_entry_count"]
        )
        record["preserved_task_success_vs_no_sense"] = (
            (not no_sense["team_success"]) or record["team_success"]
        )
        record["lower_sensing_cost_than_always_sense"] = (
            record["sensing_cost_sum"] < always["sensing_cost_sum"]
        )


def _scenario_diagnostic_record(
    scenario_name: str,
    comparisons: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    no_sense = comparisons["no_sense_shortest_path"]
    risk = comparisons["risk_aware_oracle_or_heuristic"]
    selective = comparisons["selective_sense_risk_aware"]
    always = comparisons["always_sense_shortest_path"]
    no_sense_zero_hazard_success = (
        no_sense["team_success"] and no_sense["hazard_entry_count"] == 0
    )
    selective_advantage = (
        selective["team_success"]
        and selective["hazard_entry_count"] < no_sense["hazard_entry_count"]
        and selective["sensing_cost_sum"] < always["sensing_cost_sum"]
    )
    risk_advantage = (
        risk["team_success"]
        and risk["hazard_entry_count"] < no_sense["hazard_entry_count"]
    )
    sufficient = (
        not no_sense_zero_hazard_success
        and (selective_advantage or risk_advantage)
        and selective["used_revealed_information_to_adapt_movement"]
    )
    return {
        "scenario_name": scenario_name,
        # success_rate is the 0/1 outcome of one deterministic episode and
        # the *_sum fields are single-episode sums (issue m-17: the former
        # mean_* aliases re-exposed these sums under misleading names).
        "success_rate": no_sense["success_rate"],
        "hazard_cost_sum": no_sense["hazard_cost_sum"],
        "sensing_cost_sum": no_sense["sensing_cost_sum"],
        "task_reward_sum": no_sense["task_reward_sum"],
        "mean_steps_to_success": no_sense["mean_steps_to_success"],
        "sensing_rate": no_sense["sensing_rate"],
        "hazard_entry_count": no_sense["hazard_entry_count"],
        "team_success": no_sense["team_success"],
        "no_sense_succeeds": no_sense["team_success"],
        "no_sense_has_zero_hazard": no_sense["hazard_entry_count"] == 0,
        "no_sense_succeeds_with_zero_hazard": no_sense_zero_hazard_success,
        "sensing_can_reveal_hidden_local_hazard_information": any(
            record["sensing_can_reveal_hidden_local_hazard_information"]
            for record in comparisons.values()
        ),
        "revealed_information_can_logically_change_movement_decisions": selective[
            "used_revealed_information_to_adapt_movement"
        ],
        "sufficient_for_active_sensing_claim_testing": sufficient,
        "sufficiency_reason": (
            "not sufficient for active-sensing claim support"
            if no_sense_zero_hazard_success
            else (
                "claim-relevant Stage 24 readiness scenario"
                if sufficient
                else "not sufficient for active-sensing claim support"
            )
        ),
        "selective_reduces_hazard_vs_no_sense": selective[
            "reduced_hazard_exposure_vs_no_sense"
        ],
        "selective_lower_sensing_cost_than_always_sense": selective[
            "lower_sensing_cost_than_always_sense"
        ],
        "risk_aware_reduces_hazard_vs_no_sense": risk[
            "reduced_hazard_exposure_vs_no_sense"
        ],
    }


def _validate_observation_payload(
    observations: Mapping[str, Mapping[str, object]],
) -> None:
    if tuple(observations.keys()) != STAGE23_AGENT_NAMES:
        raise ValueError("observations must preserve fixed Stage 23-A agent order")
    for observation in observations.values():
        validate_stage23_actor_observation(observation)


def _validate_action_payload(
    actions: Mapping[str, Mapping[str, int]],
    *,
    expected_agents: tuple[str, ...],
) -> None:
    if not isinstance(actions, Mapping):
        raise TypeError("actions must be a mapping")
    if tuple(actions.keys()) != expected_agents:
        raise ValueError("actions must preserve fixed Stage 23-A agent order")
    for payload in actions.values():
        if not isinstance(payload, Mapping):
            raise TypeError("action payload must be a mapping")
        if set(payload) != {SENSING_ACTION_FIELD, MOVEMENT_ACTION_FIELD}:
            raise ValueError("action payload must contain both action factors")
        sensing = payload[SENSING_ACTION_FIELD]
        movement = payload[MOVEMENT_ACTION_FIELD]
        require_int_not_bool("sensing action", sensing)
        require_int_not_bool("movement action", movement)
        if sensing not in (0, 1):
            raise ValueError("sensing action out of range")
        if movement not in (0, 1, 2, 3, 4):
            raise ValueError("movement action out of range")


def _update_revealed_memory(
    observations: Mapping[str, Mapping[str, object]],
    state: PolicyState,
) -> None:
    for agent, observation in observations.items():
        actor_visible = observation["actor_visible"]
        if not isinstance(actor_visible, Mapping):
            raise TypeError("actor_visible must be a mapping")
        revealed = actor_visible["revealed_local_hazards"]
        if not isinstance(revealed, list):
            raise TypeError("revealed_local_hazards must be a list")
        for cell in revealed:
            if not isinstance(cell, list) or len(cell) != 2:
                raise ValueError("revealed hazard cell must be a two-item list")
            # int() here trusts the Stage 23-A environment payload, which emits
            # plain (non-bool) integer coordinates via _cell_to_json (issue
            # NEW-stage24_diagnostics-8: coordinates are not independently
            # non-bool-validated because they are a trusted internal payload).
            state["revealed_hazards_by_agent"][agent].add((int(cell[0]), int(cell[1])))


def _position_from_observation(observation: Mapping[str, object]) -> Coordinate:
    actor_visible = observation["actor_visible"]
    if not isinstance(actor_visible, Mapping):
        raise TypeError("actor_visible must be a mapping")
    position = actor_visible["position"]
    if not isinstance(position, list) or len(position) != 2:
        raise ValueError("position must be a two-item list")
    # Trusted Stage 23-A environment payload (issue NEW-stage24_diagnostics-8):
    # the position is a plain non-bool integer pair emitted by _cell_to_json.
    return int(position[0]), int(position[1])


def _movement_action(current: Coordinate, next_cell: Coordinate) -> int:
    """Map a one-step move to its movement action index.

    Precondition (issue NEW-stage24_diagnostics-7): ``next_cell`` is either equal
    to ``current`` (mapped to the "stay" action 0) or exactly one of its four
    grid neighbours; every caller derives ``next_cell`` from ``_shortest_path``,
    which only ever emits 4-neighbour steps. A non-adjacent ``next_cell`` is a
    contract violation and raises. The equal-cell case returns before the table
    scan, so the "stay" entry (key 0 -> (0, 0)) in STAGE23_MOVEMENT_DELTAS is
    never matched against a non-zero delta.
    """

    if current == next_cell:
        return 0
    delta = (next_cell[0] - current[0], next_cell[1] - current[1])
    for action, action_delta in STAGE23_MOVEMENT_DELTAS.items():
        if action_delta == delta:
            return action
    raise ValueError(f"non-adjacent movement requested: {current!r} -> {next_cell!r}")


def _add(left: Coordinate, right: Coordinate) -> Coordinate:
    return left[0] + right[0], left[1] + right[1]


def _validate_stage24a_variant_collection(
    variants: Mapping[str, Stage24AHazardLayoutVariant],
) -> None:
    if not variants:
        raise ValueError("Stage 24-A variants must be non-empty")
    for key, variant in variants.items():
        if key != variant.name:
            raise ValueError("Stage 24-A variant keys must match variant names")
    groups = {variant.group for variant in variants.values()}
    if not {"training", "readiness", "held_out"} <= groups:
        raise ValueError("Stage 24-A variants must include training, readiness, and held_out groups")


def _validate_coordinate_collection(
    name: str,
    cells: tuple[Coordinate, ...],
    width: int,
    height: int,
) -> tuple[Coordinate, ...]:
    if not isinstance(cells, tuple) or not cells:
        raise ValueError(f"{name} must be a non-empty tuple")
    normalised: list[Coordinate] = []
    for cell in cells:
        if not isinstance(cell, tuple) or len(cell) != 2:
            raise TypeError(f"{name} entries must be two-int coordinate tuples")
        row, column = cell
        require_int_not_bool(f"{name} row", row)
        require_int_not_bool(f"{name} column", column)
        coordinate = (row, column)
        if not _in_grid(coordinate, width, height):
            raise ValueError(f"{name} coordinate out of grid: {coordinate!r}")
        normalised.append(coordinate)
    if len(set(normalised)) != len(normalised):
        raise ValueError(f"{name} must contain unique coordinates")
    return tuple(normalised)


def _public_risk_zone_reveals_exact_hidden_hazard(
    hidden_hazard_cells: tuple[Coordinate, ...],
    public_risk_zone_cells: tuple[Coordinate, ...],
) -> bool:
    hidden = set(hidden_hazard_cells)
    public = set(public_risk_zone_cells)
    if not hidden or not public:
        return True
    if public == hidden:
        return True
    if len(public) <= len(hidden):
        return True
    if len(public - hidden) == 0:
        return True
    if len(hidden) > 1 and len(public) < len(hidden) + _MIN_EXTRA_UNCERTAINTY_CELLS:
        return True
    return False


def _stage24a_public_risk_zone_cells(scenario_name: str) -> tuple[Coordinate, ...]:
    # Deliberate scope (issue D-13, documented 2026-07-03): only the
    # risk_gate_hidden_hazard scenario (and the DR-D1/L1 fork family below)
    # defines a public risk zone. Every other catalog scenario gets an empty
    # zone, so the selective-sense comparator never senses there and degenerates
    # to no-sense; the sufficiency diagnostics for those scenarios therefore
    # measure only the oracle path. Those scenarios are documented as
    # insufficient for active-sensing claim testing.
    if scenario_name == "risk_gate_hidden_hazard":
        return _PUBLIC_GATE_RISK_ZONE_CELLS
    # DR-D1/L1: fork scenarios get a gate-approach public risk zone (a box over the
    # two open gates, their approach column, and one column past) so selective_sense
    # senses at the gate approach and reroutes on the reveal (verified: 0 hazard on
    # both layouts, selective sensing rate). The zone spans >=2 gate rows, so it never
    # pinpoints the exact hidden hazard.
    if scenario_name in _FORK_SCENARIOS:
        return _fork_public_risk_zone_cells(stage23_scenario_catalog()[scenario_name])
    return tuple()


def _fork_public_risk_zone_cells(scenario: Stage23Scenario) -> tuple[Coordinate, ...]:
    """Return the gate-approach public risk zone for a DR-D1/L1 fork scenario.

    The open gates are the wall-column (``_FORK_WALL_COLUMN``) cells NOT in the
    obstacle set. The zone is a box over columns {gate-1, gate, gate+1} and rows
    [min gate row - 1 .. max gate row + 1], clipped to the grid — broad enough to
    trigger sensing at the approach yet always covering >= 2 gate rows so it never
    pinpoints the exact hidden hazard.
    """

    obstacle_set = set(scenario.obstacles)
    gate_rows = [
        row
        for row in range(scenario.height)
        if (row, _FORK_WALL_COLUMN) not in obstacle_set
        and _in_grid((row, _FORK_WALL_COLUMN), scenario.width, scenario.height)
    ]
    if not gate_rows:
        return tuple()
    row_lo = max(0, min(gate_rows) - 1)
    row_hi = min(scenario.height - 1, max(gate_rows) + 1)
    col_lo = max(0, _FORK_WALL_COLUMN - 1)
    col_hi = min(scenario.width - 1, _FORK_WALL_COLUMN + 1)
    return tuple(
        sorted(
            (row, col)
            for row in range(row_lo, row_hi + 1)
            for col in range(col_lo, col_hi + 1)
        )
    )


def verify_fork_family_forces_sensing(
    scenario_names: tuple[str, ...],
    *,
    hazard_budget: float = 0.5,
    seed: int = 2400,
) -> dict[str, Any]:
    """Prove ON THE SCENARIO DEFINITION that a fork family forces sensing.

    This is the executable by-construction check backing DR-D1/L1's two-part bar
    (the analogue of the s110 trajectory proof, generalized to a layout
    distribution). It reads only training/readiness catalog surfaces (never a
    held-out variant, I-3) and grades the TRUE sparse environment fields (never a
    shaped return, AM-2). It writes no files and mutates no state.

    BAR-A (sensing necessary by construction):
      * ``union_cut`` (the LOAD-BEARING per-layout guarantee): BFS from every start over
        ``obstacles`` PLUS the UNION of the group's hidden hazards cannot reach a goal ->
        no blind zero-hazard successful route exists (a never-sense 0-hazard successful
        policy would be a single fixed route avoiding the union, which the cut forbids).
      * ``all_gates_hazardous``: every open gate is the hazard in at least one layout
        (no memorizable safe gate; implied by the cut).
      * ``no_sense_mean_hazard_exceeds_budget`` (a secondary, budget-level witness): the
        ``no_sense`` comparator's team hazard AVERAGED over the group's layouts exceeds
        ``hazard_budget``. For the 2-layout fork this mean is 1.0 (per-layout {0.0, 2.0}),
        which equals the minimum over ALL blind policies of the layout-EXPECTED J_C-hat
        (both agents commit to one gate -> 2.0 in the wrong-guess layout, 0.0 in the
        other, mean 1.0; splitting gates -> 1.0 every layout) = 2x d_ep=0.5. NOTE: the
        per-layout MINIMUM no_sense hazard is 0.0 (the safe-guess layout), so it is the
        union-cut -- not this mean -- that forbids a blind safe route; the mean conjunct
        can only make ``bar_a_sensing_forced_by_construction`` stricter, never looser.
    BAR-B (task remains winnable):
      * every single layout has a safe successful route (BFS avoiding ``obstacles`` PLUS
        that layout's hazard reaches a goal within the horizon), AND the
        ``selective_sense`` comparator achieves 0 hazard while preserving success on
        every layout.
    """

    if not isinstance(scenario_names, tuple) or not scenario_names:
        raise ValueError("scenario_names must be a non-empty tuple")
    require_positive_int("seed", seed, error_suffix="must be positive")
    finite_hazard_budget = finite_numeric_scalar("hazard_budget", hazard_budget)
    catalog = stage23_scenario_catalog()
    scenarios: list[Stage23Scenario] = []
    for name in scenario_names:
        if not isinstance(name, str):
            raise TypeError("scenario_names entries must be strings")
        if name not in catalog:
            raise ValueError(f"unknown Stage 23-A scenario: {name}")
        scenarios.append(catalog[name])
    base = scenarios[0]
    # A fork group must be observationally aliased: identical starts/goals/obstacles,
    # differing only in the hidden hazard.
    for scenario in scenarios[1:]:
        if (
            scenario.starts != base.starts
            or scenario.goals != base.goals
            or scenario.obstacles != base.obstacles
        ):
            raise ValueError(
                "fork family layouts must share starts, goals, and obstacles "
                "(observationally aliased)"
            )

    union_hazards = tuple(
        sorted({cell for scenario in scenarios for cell in scenario.hidden_hazard_cells})
    )
    # BAR-A / A1 union cut: _shortest_path returns [] when the goal is unreachable
    # avoiding obstacles + union hazards (avoid_cells never block goal cells, but no
    # goal is a hazard here).
    union_cut = all(
        not _shortest_path(
            start,
            base.goals,
            width=base.width,
            height=base.height,
            obstacles=base.obstacles,
            avoid_cells=union_hazards,
        )
        for start in base.starts
    )
    # BAR-A / A2 every open gate is hazardous in some layout.
    obstacle_set = set(base.obstacles)
    gates = {
        (row, _FORK_WALL_COLUMN)
        for row in range(base.height)
        if (row, _FORK_WALL_COLUMN) not in obstacle_set
        and _in_grid((row, _FORK_WALL_COLUMN), base.width, base.height)
    }
    all_gates_hazardous = bool(gates) and gates <= set(union_hazards)

    per_layout: dict[str, dict[str, Any]] = {}
    no_sense_hazards: list[float] = []
    for scenario in scenarios:
        winnable = all(
            _shortest_path(
                start,
                scenario.goals,
                width=scenario.width,
                height=scenario.height,
                obstacles=scenario.obstacles,
                avoid_cells=scenario.hidden_hazard_cells,
            )
            for start in scenario.starts
        )
        no_sense = no_sense_shortest_path(scenario.name, seed=seed)
        selective = selective_sense_risk_aware(scenario.name, seed=seed)
        no_sense_hazards.append(float(no_sense["hazard_cost_sum"]))
        per_layout[scenario.name] = {
            "hidden_hazard_cells": _cells_to_json(scenario.hidden_hazard_cells),
            "winnable_safe_route_exists": bool(winnable),
            "no_sense_hazard_cost_sum": float(no_sense["hazard_cost_sum"]),
            "no_sense_hazard_entry_count": int(no_sense["hazard_entry_count"]),
            "no_sense_team_success": bool(no_sense["team_success"]),
            "selective_hazard_cost_sum": float(selective["hazard_cost_sum"]),
            "selective_team_success": bool(selective["team_success"]),
            "selective_sensing_rate": float(selective["sensing_rate"]),
            "selective_adapts_movement": bool(
                selective["used_revealed_information_to_adapt_movement"]
            ),
        }

    mean_no_sense_hazard = sum(no_sense_hazards) / float(len(no_sense_hazards))
    no_sense_mean_exceeds_budget = mean_no_sense_hazard > finite_hazard_budget
    all_winnable = all(
        record["winnable_safe_route_exists"] for record in per_layout.values()
    )
    selective_zero_hazard = all(
        record["selective_hazard_cost_sum"] == 0.0 for record in per_layout.values()
    )
    selective_preserves_success = all(
        record["selective_team_success"] for record in per_layout.values()
    )
    bar_a = union_cut and all_gates_hazardous and no_sense_mean_exceeds_budget
    bar_b = all_winnable and selective_zero_hazard and selective_preserves_success
    return {
        "scenario_names": tuple(scenario_names),
        "union_hazards": _cells_to_json(union_hazards),
        "hazard_budget": finite_hazard_budget,
        # BAR-A (sensing necessary by construction)
        "bar_a_union_cut": bool(union_cut),
        "bar_a_all_gates_hazardous": bool(all_gates_hazardous),
        "bar_a_no_sense_mean_hazard": mean_no_sense_hazard,
        "bar_a_no_sense_mean_hazard_exceeds_budget": bool(no_sense_mean_exceeds_budget),
        "bar_a_sensing_forced_by_construction": bool(bar_a),
        # BAR-B (task remains winnable)
        "bar_b_all_layouts_winnable": bool(all_winnable),
        "bar_b_selective_zero_hazard": bool(selective_zero_hazard),
        "bar_b_selective_preserves_success": bool(selective_preserves_success),
        "bar_b_task_learnable_by_construction": bool(bar_b),
        "per_layout": per_layout,
        "forces_sensing_and_winnable": bool(bar_a and bar_b),
    }


def _stage24a_variant_environment_config(
    variant: Stage24AHazardLayoutVariant,
    *,
    seed: int,
) -> Stage23EnvironmentConfig:
    base = stage23_scenario_catalog()["risk_gate_hidden_hazard"]
    return Stage23EnvironmentConfig(
        scenario_name=base.name,
        width=base.width,
        height=base.height,
        seed=seed,
        start_positions=base.starts,
        goal_cells=base.goals,
        obstacle_cells=base.obstacles,
        hidden_hazard_cells=variant.hidden_hazard_cells,
    )


__all__ = [
    "Stage24ComparatorConfig",
    "Stage24AHazardLayoutVariant",
    "always_sense_shortest_path",
    "fork_curriculum_scenarios",
    "fork_hazard_layout_families",
    "no_sense_shortest_path",
    "random_policy",
    "risk_aware_oracle_or_heuristic",
    "run_stage23_scenario_diagnostics",
    "run_stage24a_variant_readiness_diagnostics",
    "selective_sense_risk_aware",
    "stage24a_hazard_layout_variants",
    "verify_fork_family_forces_sensing",
]

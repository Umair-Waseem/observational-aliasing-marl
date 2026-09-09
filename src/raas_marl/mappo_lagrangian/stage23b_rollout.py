"""Stage 23-B bounded rollout collection over the Stage 23-A environment.

This module connects the Stage 23-A active-sensing environment and tensor
adapter to the Stage 21 MAPPO-Lagrangian core contracts. It collects one small
development-only on-policy episode in memory. It does not write result roots,
serialize rollout buffers, save model/optimizer state, run final evaluation, or
create claim evidence.

The rollout-collection helper family (field schema, tensor batchers,
policy->action mapping, payload/team-end/core-config validators, seeding, and
batch-value re-checks) is sourced from the shared ``_rollout_common`` module
rather than defined here; this module keeps only the Stage 23-B public API,
single-episode collection loop, and the Stage-23-B-specific summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from raas_marl.environments.active_sensing.grid_environment import (
    HAZARD_COST_KEY,
    SENSING_COST_KEY,
    STAGE23_AGENT_NAMES,
    TASK_REWARD_KEY,
    RiskAwareActiveSensingGridEnvironment,
    Stage23EnvironmentConfig,
)
from raas_marl.environments.active_sensing.tensor_adapter import (
    default_stage23_core_config,
    stage23_transition_to_environment_step_contract,
)
from raas_marl.mappo_lagrangian._rollout_common import (
    actions_from_policy,
    actor_observation_batch,
    central_state_batch,
    empty_rollout_fields,
    float_vector,
    per_step_sampling_seed,
    revealed_information_batch,
    set_rollout_seed,
    stack_time,
    validate_parallel_payloads,
    validate_rollout_batch_values,
    validate_stage23_core_config,
    validate_step_api_payloads,
    validate_team_end_flags,
)
from raas_marl.mappo_lagrangian._validation import (
    require_bounded_seed,
    require_positive_int,
)
from raas_marl.mappo_lagrangian.buffer import RolloutBatch
from raas_marl.mappo_lagrangian.config import MAPPOCoreConfig
from raas_marl.mappo_lagrangian.model import (
    ActorInput,
    CriticInput,
    RecurrentMAPPOActorCritic,
)


_STAGE23B_ZERO_SEED_MESSAGE = (
    "seed must be a positive integer; seed 0 is reserved for Stage 23-B"
)


@dataclass(frozen=True)
class Stage23BRolloutConfig:
    """Bounded Stage 23-B rollout settings.

    Stage 23-B intentionally supports exactly one episode. Multi-episode
    padding, variable-length packing, and active-agent masking are left locked
    until a later stage explicitly implements them.
    """

    max_episodes: int = 1
    max_steps_per_episode: int = 4
    sample_actions: bool = False
    seed: int = 23
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        require_positive_int("max_episodes", self.max_episodes, error_suffix="must be positive")
        if self.max_episodes != 1:
            raise ValueError(
                "Stage 23-B rollout collection supports exactly one episode; "
                "multi-episode padding is not implemented"
            )
        require_positive_int(
            "max_steps_per_episode", self.max_steps_per_episode, error_suffix="must be positive"
        )
        if self.max_steps_per_episode > 16:
            raise ValueError("max_steps_per_episode must be <= 16 for Stage 23-B")
        if not isinstance(self.sample_actions, bool):
            raise TypeError("sample_actions must be a bool")
        # S2-19-stage23b: the per-step sampling reseed derives
        # per_step_sampling_seed(seed, episode_index=0, step_index) at collection
        # time; bound the seed so the largest derived reseed (seed +
        # max_steps_per_episode) still fits in a signed 64-bit integer and never
        # crashes torch.manual_seed with an undeclared RuntimeError.
        require_bounded_seed(
            "seed",
            self.seed,
            max_offset=self.max_steps_per_episode,
            zero_seed_message=_STAGE23B_ZERO_SEED_MESSAGE,
        )
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        expected_dtype = default_stage23_core_config().dtype
        if self.dtype != expected_dtype:
            raise ValueError(
                "Stage 23-B rollout dtype must equal the default Stage 23-A "
                f"core dtype {expected_dtype!r}; non-default dtype support is out of scope"
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.device != self.device.strip():
            raise ValueError("device must not contain leading or trailing whitespace")
        if self.device != "cpu":
            raise ValueError("Stage 23-B rollout collection supports only CPU device 'cpu'")


@dataclass(frozen=True)
class Stage23BCollectedRollout:
    """In-memory Stage 23-B rollout plus JSON-compatible summaries."""

    batch: RolloutBatch
    summary: dict[str, Any]
    environment_summary: dict[str, Any]


def collect_stage23b_development_rollout(
    model: RecurrentMAPPOActorCritic,
    environment: RiskAwareActiveSensingGridEnvironment,
    rollout_config: Stage23BRolloutConfig | None = None,
) -> Stage23BCollectedRollout:
    """Collect one bounded Stage 23-B on-policy rollout in memory."""

    if not isinstance(model, RecurrentMAPPOActorCritic):
        raise TypeError("model must be a RecurrentMAPPOActorCritic")
    if not isinstance(environment, RiskAwareActiveSensingGridEnvironment):
        raise TypeError("environment must be a RiskAwareActiveSensingGridEnvironment")
    if rollout_config is None:
        settings = Stage23BRolloutConfig()
    elif isinstance(rollout_config, Stage23BRolloutConfig):
        settings = rollout_config
    else:
        raise TypeError("rollout_config must be a Stage23BRolloutConfig or None")
    core_config = model.config
    validate_stage23_core_config(core_config, context="Stage 23-B")
    if settings.max_steps_per_episode > environment.config.max_steps:
        raise ValueError("max_steps_per_episode must not exceed environment max_steps")
    _reject_already_done_environment(environment)
    set_rollout_seed(settings.seed)
    was_training = model.training
    model.eval()
    try:
        episode = _collect_single_stage23b_episode(
            model,
            environment,
            settings,
            core_config,
        )
        batch = RolloutBatch(
            actor_observation=episode["actor_observation"],
            revealed_information=episode["revealed_information"],
            central_state=episode["central_state"],
            sensing_action=episode["sensing_action"],
            movement_action=episode["movement_action"],
            old_sensing_log_probability=episode["old_sensing_log_probability"],
            old_movement_log_probability=episode["old_movement_log_probability"],
            task_reward=episode["task_reward"],
            hazard_cost=episode["hazard_cost"],
            sensing_cost=episode["sensing_cost"],
            reward_value=episode["reward_value"],
            hazard_cost_value=episode["hazard_cost_value"],
            next_reward_value=episode["next_reward_value"],
            next_hazard_cost_value=episode["next_hazard_cost_value"],
            terminal=episode["terminal"],
            truncation=episode["truncation"],
            valid_mask=episode["valid_mask"],
        )
        batch.validate(core_config)
        validate_rollout_batch_values(batch)
        summary = _rollout_summary(batch, environment, settings, core_config)
        environment_summary = _environment_summary(environment, settings, core_config)
        return Stage23BCollectedRollout(
            batch=batch,
            summary=summary,
            environment_summary=environment_summary,
        )
    finally:
        model.train(was_training)


def make_default_stage23b_environment(
    *,
    max_steps: int = 4,
    seed: int = 23,
) -> RiskAwareActiveSensingGridEnvironment:
    """Return the bounded Stage 23-A environment used by Stage 23-B."""

    require_positive_int("max_steps", max_steps, error_suffix="must be positive")
    require_bounded_seed("seed", seed, zero_seed_message=_STAGE23B_ZERO_SEED_MESSAGE)
    return RiskAwareActiveSensingGridEnvironment(
        Stage23EnvironmentConfig(max_steps=max_steps, seed=seed)
    )


def _collect_single_stage23b_episode(
    model: RecurrentMAPPOActorCritic,
    environment: RiskAwareActiveSensingGridEnvironment,
    settings: Stage23BRolloutConfig,
    core_config: MAPPOCoreConfig,
) -> dict[str, torch.Tensor]:
    observations, infos = environment.reset(seed=settings.seed)
    validate_parallel_payloads(observations, infos)
    history_state: torch.Tensor | None = None
    fields = empty_rollout_fields()
    for step_index in range(settings.max_steps_per_episode):
        actor_observation = actor_observation_batch(
            observations,
            core_config,
            dtype=settings.dtype,
            device=settings.device,
        )
        revealed_information = revealed_information_batch(
            observations,
            core_config,
            dtype=settings.dtype,
            device=settings.device,
        )
        central_state = central_state_batch(
            environment,
            core_config,
            dtype=settings.dtype,
            device=settings.device,
        )
        with torch.no_grad():
            if settings.sample_actions:
                # NEW-stage23b_rollout-1: collision-safe per-step reseed offset
                # (episode_index is fixed at 0 for the single Stage 23-B
                # episode; identical to the historical seed+step_index here).
                # This mutates the process-wide torch RNG without save/restore
                # (NEW-stage23b_rollout-7); Stage 23-B makes no cross-platform
                # bitwise-determinism claim.
                torch.manual_seed(per_step_sampling_seed(settings.seed, 0, step_index))
            output = model(
                ActorInput(actor_observation, revealed_information, history_state),
                CriticInput(central_state),
                sample_actions=settings.sample_actions,
            )
            policy = output["policy"]
            sensing_action = policy.sensing_action.detach().squeeze(1)
            movement_action = policy.movement_action.detach().squeeze(1)
            actions = actions_from_policy(sensing_action, movement_action, core_config)
            next_observations, _rewards, terminations, truncations, next_infos = (
                environment.step(actions)
            )
            validate_parallel_payloads(next_observations, next_infos)
            validate_step_api_payloads(_rewards, terminations, truncations, next_infos)
            contracts = [
                stage23_transition_to_environment_step_contract(
                    next_observations[agent],
                    next_infos[agent],
                    environment,
                    config=core_config,
                )
                for agent in STAGE23_AGENT_NAMES
            ]
            terminal = torch.tensor(
                [contract.done_flags["terminal"] for contract in contracts],
                dtype=torch.bool,
                device=actor_observation.device,
            ).unsqueeze(1)
            truncation = torch.tensor(
                [contract.done_flags["truncated"] for contract in contracts],
                dtype=torch.bool,
                device=actor_observation.device,
            ).unsqueeze(1)
            episode_ended = validate_team_end_flags(
                terminal, truncation, context="Stage 23-B rollout collection"
            )
            next_central_state = central_state_batch(
                environment,
                core_config,
                dtype=settings.dtype,
                device=settings.device,
            )
            next_reward_value = model.evaluate_reward_critic(
                CriticInput(next_central_state)
            ).detach()
            next_hazard_cost_value = model.evaluate_cost_critic(
                CriticInput(next_central_state)
            ).detach()
            zero = torch.zeros_like(next_reward_value)
            next_reward_value = torch.where(terminal, zero, next_reward_value)
            next_hazard_cost_value = torch.where(terminal, zero, next_hazard_cost_value)

            fields["actor_observation"].append(actor_observation.detach().squeeze(1))
            fields["revealed_information"].append(revealed_information.detach().squeeze(1))
            fields["central_state"].append(central_state.detach().squeeze(1))
            fields["sensing_action"].append(sensing_action.detach())
            fields["movement_action"].append(movement_action.detach())
            fields["old_sensing_log_probability"].append(
                policy.sensing_log_probability.detach().squeeze(1)
            )
            fields["old_movement_log_probability"].append(
                policy.movement_log_probability.detach().squeeze(1)
            )
            fields["task_reward"].append(
                float_vector(
                    [contract.rewards_and_costs[TASK_REWARD_KEY] for contract in contracts],
                    actor_observation,
                )
            )
            fields["hazard_cost"].append(
                float_vector(
                    [contract.rewards_and_costs[HAZARD_COST_KEY] for contract in contracts],
                    actor_observation,
                )
            )
            fields["sensing_cost"].append(
                float_vector(
                    [contract.rewards_and_costs[SENSING_COST_KEY] for contract in contracts],
                    actor_observation,
                )
            )
            fields["reward_value"].append(output["reward_value"].detach().squeeze(1))
            fields["hazard_cost_value"].append(
                output["hazard_cost_value"].detach().squeeze(1)
            )
            fields["next_reward_value"].append(next_reward_value.squeeze(1))
            fields["next_hazard_cost_value"].append(next_hazard_cost_value.squeeze(1))
            fields["terminal"].append(terminal.squeeze(1))
            fields["truncation"].append(truncation.squeeze(1))
            fields["valid_mask"].append(torch.ones_like(terminal.squeeze(1), dtype=torch.bool))
            if episode_ended:
                break
            history_state = output["next_recurrent_state"].detach()
            observations = next_observations
            infos = next_infos
    return {name: stack_time(values) for name, values in fields.items()}


def _reject_already_done_environment(
    environment: RiskAwareActiveSensingGridEnvironment,
) -> None:
    if environment.is_done:
        raise RuntimeError("Stage 23-B rollout collection requires a resettable active environment")


def _rollout_summary(
    batch: RolloutBatch,
    environment: RiskAwareActiveSensingGridEnvironment,
    settings: Stage23BRolloutConfig,
    core_config: MAPPOCoreConfig,
) -> dict[str, Any]:
    valid = batch.valid_mask
    return {
        "actor_critic_ctde_boundary": (
            "actor_observation_and_revealed_information_only_for_actor;"
            "central_state_only_for_critics"
        ),
        "agent_count": len(STAGE23_AGENT_NAMES),
        "batch_size": int(batch.actor_observation.shape[0]),
        "claim_evidence_created": False,
        "claim_status": "not tested / not supported",
        "cross_platform_bitwise_determinism_claimed": False,
        "development_only": True,
        "episode_count": 1,
        "evaluation_run": False,
        "final_evaluation_run": False,
        "hazard_cost_mean": float(batch.hazard_cost[valid].mean().item()),
        "local_seed_controls_recorded": True,
        "max_steps_per_episode": settings.max_steps_per_episode,
        "movement_action_count": int(core_config.movement_action_count),
        "movement_action_factor_preserved": True,
        "next_value_bootstrap_convention": (
            "terminal_next_values_zeroed;truncation_next_values_preserved"
        ),
        "old_log_probability_source": "pre_update_rollout_policy",
        "old_log_probability_is_detached": True,
        "on_policy_rollout": True,
        "paper_facing_results_created": False,
        "recurrent_state_serialized": False,
        "rollout_buffer_artifact_created": False,
        "sample_actions": settings.sample_actions,
        "sensing_action_count": int(core_config.sensing_action_count),
        "sensing_action_factor_preserved": True,
        "sensing_cost_mean": float(batch.sensing_cost[valid].mean().item()),
        "stage": "23-B",
        "task_reward_sum": float(batch.task_reward[valid].sum().item()),
        "terminal_count": int(batch.terminal[valid].sum().item()),
        "time_steps": int(batch.actor_observation.shape[1]),
        "training_run": "bounded_stage23b_development_only",
        "truncation_count": int(batch.truncation[valid].sum().item()),
        "valid_transition_count": int(valid.sum().item()),
    }


def _environment_summary(
    environment: RiskAwareActiveSensingGridEnvironment,
    settings: Stage23BRolloutConfig,
    core_config: MAPPOCoreConfig,
) -> dict[str, Any]:
    scenario = environment.scenario
    return {
        "actor_visible_schema": [
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
        ],
        "agent_count": len(STAGE23_AGENT_NAMES),
        "central_state_dim": core_config.central_state_dim,
        "claim_evidence_created": False,
        "claim_status": "not tested / not supported",
        "development_only": True,
        "environment_stage": "23-A",
        "evaluation_run": False,
        "final_evaluation_run": False,
        "grid_height": scenario.height,
        "grid_width": scenario.width,
        "max_steps": environment.config.max_steps,
        "model_device": settings.device,
        "model_dtype": settings.dtype,
        "movement_action_count": core_config.movement_action_count,
        "paper_facing_results_created": False,
        "scenario_name": scenario.name,
        "seed": settings.seed,
        "sensing_action_count": core_config.sensing_action_count,
        "stage": "23-B",
        "training_run": "bounded_stage23b_development_only",
    }

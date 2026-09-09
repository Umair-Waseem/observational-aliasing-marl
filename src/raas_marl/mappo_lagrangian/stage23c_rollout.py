"""Stage 23-C bounded multi-episode rollout collection over the Stage 23-A environment.

This module delivers the multi-episode, variable-length capability that
``stage23b_rollout.py`` deliberately leaves locked ("multi-episode padding,
variable-length packing, and active-agent masking are left locked until a later
stage explicitly implements them"). It collects several small development-only
on-policy episodes in memory, one fresh Stage 23-A active-sensing environment per
episode, merges the variable-length episodes on the batch axis with zero padding
and a ``valid_mask`` that is ``False`` on padded steps, and returns per-episode
records for a persisted training log.

It is development-only. It does not write result roots, serialize rollout
buffers, save model/optimizer state, run final evaluation, create claim evidence,
produce paper-facing output, compare baselines, run ablations, run statistical
tests, or make any Bayesian-belief or formal-VOI claim. The per-episode records
carry no boundary flags; the Stage 23-C runner adds those when persisting.

The rollout-collection helper family (field schema, tensor batchers,
policy->action mapping, payload/team-end/core-config validators, seeding, the
variable-length padded merge, and batch-value re-checks) is sourced from the
shared ``_rollout_common`` module rather than re-implemented here; this module
keeps only the Stage 23-C public API, the per-episode collection loop, the
per-episode records, and the Stage-23-C-specific summaries.
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
    stage23_scenario_catalog,
)
from raas_marl.environments.active_sensing.scenarios import available_scenarios
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
    merge_padded_episodes,
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


_STAGE23C_ZERO_SEED_MESSAGE = (
    "seed must be a positive integer; seed 0 is reserved for Stage 23-C"
)

# Bounds pinned by the Stage 23-C specification.
_MIN_EPISODES = 2
_MAX_EPISODES = 8
_MIN_STEPS_PER_EPISODE = 1
_MAX_STEPS_PER_EPISODE = 32


@dataclass(frozen=True)
class Stage23CRolloutConfig:
    """Bounded Stage 23-C multi-episode rollout settings.

    Stage 23-C supports multiple variable-length episodes merged with valid-mask
    padding. It stays development-only: it produces no final-evaluation content,
    no claim evidence, and no paper-facing output.
    """

    max_episodes: int = 2
    steps_per_episode: int | None = None
    scenario_names: tuple[str, ...] = ("risk_gate_hidden_hazard",)
    sample_actions: bool = False
    seed: int = 23
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        require_positive_int(
            "max_episodes", self.max_episodes, error_suffix="must be positive"
        )
        # Normalize through the base-class ``int.__index__`` slot before the
        # range comparison so a hostile int subclass overriding ``__le__`` /
        # ``__ge__`` cannot evade the [min, max] bound (mirrors the S2-5
        # hardening already applied in ``_validation.require_positive_int`` /
        # ``require_bounded_seed``). A genuine in-range int is unchanged.
        max_episodes = int.__index__(self.max_episodes)
        if not (_MIN_EPISODES <= max_episodes <= _MAX_EPISODES):
            raise ValueError(
                f"max_episodes must be between {_MIN_EPISODES} and {_MAX_EPISODES}"
            )
        if self.steps_per_episode is not None:
            require_positive_int(
                "steps_per_episode",
                self.steps_per_episode,
                error_suffix="must be positive",
            )
            steps_per_episode = int.__index__(self.steps_per_episode)
            if not (
                _MIN_STEPS_PER_EPISODE
                <= steps_per_episode
                <= _MAX_STEPS_PER_EPISODE
            ):
                raise ValueError(
                    f"steps_per_episode must be None or between "
                    f"{_MIN_STEPS_PER_EPISODE} and {_MAX_STEPS_PER_EPISODE}"
                )
        if not isinstance(self.scenario_names, tuple) or not self.scenario_names:
            raise ValueError("scenario_names must be a non-empty tuple")
        catalog_names = available_scenarios()
        for name in self.scenario_names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("scenario_names entries must be non-empty strings")
            if name not in catalog_names:
                raise ValueError(f"unknown Stage 23-A scenario: {name}")
        if not isinstance(self.sample_actions, bool):
            raise TypeError("sample_actions must be a bool")
        # The per-step sampling reseed derives
        # per_step_sampling_seed(seed, episode_index, step_index) ==
        # seed + episode_index*1000 + step_index at collection time; bound the
        # seed so the largest derived reseed (over every episode and step) still
        # fits in a signed 64-bit integer and never crashes torch.manual_seed
        # with an undeclared RuntimeError.
        # Use the ``int.__index__``-normalized primitives captured above so the
        # reseed-offset arithmetic cannot be corrupted by a hostile int subclass
        # overriding ``__mul__`` / ``__add__``.
        max_step_horizon = (
            steps_per_episode
            if self.steps_per_episode is not None
            else _MAX_STEPS_PER_EPISODE
        )
        require_bounded_seed(
            "seed",
            self.seed,
            max_offset=max_episodes * 1000 + max_step_horizon,
            zero_seed_message=_STAGE23C_ZERO_SEED_MESSAGE,
        )
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        expected_dtype = default_stage23_core_config().dtype
        if self.dtype != expected_dtype:
            raise ValueError(
                "Stage 23-C rollout dtype must equal the default Stage 23-A "
                f"core dtype {expected_dtype!r}; non-default dtype support is out of scope"
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.device != self.device.strip():
            raise ValueError("device must not contain leading or trailing whitespace")
        if self.device != "cpu":
            raise ValueError("Stage 23-C rollout collection supports only CPU device 'cpu'")


@dataclass(frozen=True)
class Stage23CCollectedRollout:
    """In-memory Stage 23-C rollout, per-episode records, and JSON-compatible summaries."""

    batch: RolloutBatch
    summary: dict[str, Any]
    episode_records: tuple[dict[str, Any], ...]
    environment_summary: dict[str, Any]


def collect_stage23c_development_rollout(
    model: RecurrentMAPPOActorCritic,
    rollout_config: Stage23CRolloutConfig | None = None,
) -> Stage23CCollectedRollout:
    """Collect a bounded Stage 23-C multi-episode on-policy rollout in memory.

    One fresh Stage 23-A environment is built per episode, with the scenario
    chosen round-robin from ``rollout_config.scenario_names`` and reset with
    ``seed = rollout_config.seed + episode_index``. Variable-length episodes are
    merged on the batch axis with zero padding and ``valid_mask`` False on the
    padded steps. All stored tensors are detached; the merged batch passes both
    ``RolloutBatch.validate`` and the shared batch-value re-checks before return.
    """

    if not isinstance(model, RecurrentMAPPOActorCritic):
        raise TypeError("model must be a RecurrentMAPPOActorCritic")
    if rollout_config is None:
        settings = Stage23CRolloutConfig()
    elif isinstance(rollout_config, Stage23CRolloutConfig):
        settings = rollout_config
    else:
        raise TypeError("rollout_config must be a Stage23CRolloutConfig or None")

    core_config = model.config
    validate_stage23_core_config(core_config, context="Stage 23-C")
    set_rollout_seed(settings.seed)

    was_training = model.training
    model.eval()
    try:
        episodes: list[dict[str, torch.Tensor]] = []
        episode_records: list[dict[str, Any]] = []
        reference_environment: RiskAwareActiveSensingGridEnvironment | None = None
        for episode_index in range(settings.max_episodes):
            scenario_name = settings.scenario_names[
                episode_index % len(settings.scenario_names)
            ]
            episode_seed = settings.seed + episode_index
            environment = make_default_stage23c_environment(
                scenario_name=scenario_name,
                max_steps=settings.steps_per_episode,
                seed=episode_seed,
            )
            if reference_environment is None:
                reference_environment = environment
            episode, record = _collect_single_stage23c_episode(
                model,
                environment,
                settings,
                core_config,
                episode_index=episode_index,
                seed=episode_seed,
            )
            episodes.append(episode)
            episode_records.append(record)

        max_time = max(int(episode["valid_mask"].shape[1]) for episode in episodes)
        batch = merge_padded_episodes(episodes, max_time=max_time)
        batch.validate(core_config)
        validate_rollout_batch_values(batch)

        assert reference_environment is not None  # max_episodes >= 2 guarantees this
        summary = _rollout_summary(batch, settings, core_config, max_time=max_time)
        environment_summary = _environment_summary(
            reference_environment, settings, core_config
        )
        return Stage23CCollectedRollout(
            batch=batch,
            summary=summary,
            episode_records=tuple(episode_records),
            environment_summary=environment_summary,
        )
    finally:
        model.train(was_training)


def make_default_stage23c_environment(
    *,
    scenario_name: str,
    max_steps: int | None = None,
    seed: int = 23,
) -> RiskAwareActiveSensingGridEnvironment:
    """Return a bounded Stage 23-A environment for one Stage 23-C episode.

    ``scenario_name`` must name a Stage 23-A catalog scenario; ``max_steps`` of
    ``None`` uses the scenario's default horizon. The seed carries the Stage 23-C
    seed-0 reservation semantics.
    """

    if not isinstance(scenario_name, str) or not scenario_name.strip():
        raise ValueError("scenario_name must be a non-empty string")
    catalog = stage23_scenario_catalog()
    if scenario_name not in catalog:
        raise ValueError(f"unknown Stage 23-A scenario: {scenario_name}")
    if max_steps is not None:
        require_positive_int("max_steps", max_steps, error_suffix="must be positive")
    require_bounded_seed("seed", seed, zero_seed_message=_STAGE23C_ZERO_SEED_MESSAGE)

    scenario = catalog[scenario_name]
    config_kwargs: dict[str, Any] = {
        "scenario_name": scenario.name,
        "width": scenario.width,
        "height": scenario.height,
        "seed": seed,
    }
    if max_steps is not None:
        config_kwargs["max_steps"] = max_steps
    return RiskAwareActiveSensingGridEnvironment(Stage23EnvironmentConfig(**config_kwargs))


def _collect_single_stage23c_episode(
    model: RecurrentMAPPOActorCritic,
    environment: RiskAwareActiveSensingGridEnvironment,
    settings: Stage23CRolloutConfig,
    core_config: MAPPOCoreConfig,
    *,
    episode_index: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    _reject_already_done_environment(environment)
    observations, infos = environment.reset(seed=seed)
    validate_parallel_payloads(observations, infos)
    history_state: torch.Tensor | None = None
    fields = empty_rollout_fields()
    totals = {
        "task_reward_sum": 0.0,
        "hazard_cost_sum": 0.0,
        "sensing_cost_sum": 0.0,
        "total_sensing_actions": 0,
        "total_movement_actions": 0,
        "hazard_entry_count": 0,
        "revealed_hazard_count": 0,
    }
    terminal = False
    truncation = False
    team_success = False
    for step_index in range(environment.config.max_steps):
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
                # Collision-safe per-step reseed offset (the rollout.py
                # convention): episode e at step s and episode e+1 at step s-1
                # never share an RNG stream. This mutates the process-wide torch
                # RNG without save/restore; Stage 23-C makes no cross-platform
                # bitwise-determinism claim. The environment reset seed is
                # independent of this reseed.
                torch.manual_seed(
                    per_step_sampling_seed(settings.seed, episode_index, step_index)
                )
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
            terminal_tensor = torch.tensor(
                [contract.done_flags["terminal"] for contract in contracts],
                dtype=torch.bool,
                device=actor_observation.device,
            ).unsqueeze(1)
            truncation_tensor = torch.tensor(
                [contract.done_flags["truncated"] for contract in contracts],
                dtype=torch.bool,
                device=actor_observation.device,
            ).unsqueeze(1)
            episode_ended = validate_team_end_flags(
                terminal_tensor,
                truncation_tensor,
                context="Stage 23-C rollout collection",
            )
            # The shared validator has already rejected partial per-agent
            # endings, so all == any per flag on an ended team.
            terminal_team_ended = bool(terminal_tensor.all().item())
            truncation_team_ended = bool(truncation_tensor.all().item())
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
            # Bootstrap convention identical to Stage 23-B: terminal next-values
            # are zeroed; truncation next-values are preserved.
            zero = torch.zeros_like(next_reward_value)
            next_reward_value = torch.where(terminal_tensor, zero, next_reward_value)
            next_hazard_cost_value = torch.where(
                terminal_tensor, zero, next_hazard_cost_value
            )

            fields["actor_observation"].append(actor_observation.detach().squeeze(1))
            fields["revealed_information"].append(
                revealed_information.detach().squeeze(1)
            )
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
            fields["terminal"].append(terminal_tensor.squeeze(1))
            fields["truncation"].append(truncation_tensor.squeeze(1))
            # valid_mask is all-True for genuine (non-padded) steps; the padded
            # merge sets it False on the padded tail.
            fields["valid_mask"].append(
                torch.ones(
                    (len(STAGE23_AGENT_NAMES),),
                    dtype=torch.bool,
                    device=actor_observation.device,
                )
            )

            for index, contract in enumerate(contracts):
                totals["task_reward_sum"] += float(
                    contract.rewards_and_costs[TASK_REWARD_KEY]
                )
                totals["hazard_cost_sum"] += float(
                    contract.rewards_and_costs[HAZARD_COST_KEY]
                )
                totals["sensing_cost_sum"] += float(
                    contract.rewards_and_costs[SENSING_COST_KEY]
                )
                totals["total_sensing_actions"] += int(sensing_action[index].item() != 0)
                totals["total_movement_actions"] += int(
                    movement_action[index].item() != 0
                )
                agent_info = next_infos[STAGE23_AGENT_NAMES[index]]
                totals["hazard_entry_count"] += int(bool(agent_info["entered_hazard"]))
                totals["revealed_hazard_count"] += int(
                    agent_info["revealed_hazard_count"]
                )
                team_success = team_success or bool(agent_info["team_success"])
            terminal = terminal_team_ended
            truncation = truncation_team_ended
            if episode_ended:
                break
            history_state = output["next_recurrent_state"].detach()
            observations = next_observations
            infos = next_infos

    episode = {name: stack_time(values) for name, values in fields.items()}
    step_count = int(episode["valid_mask"].shape[1])
    action_decisions = step_count * len(STAGE23_AGENT_NAMES)
    record = {
        "episode_index": episode_index,
        # round_index defaults to 0 here; the runner overwrites it per
        # collect/update round when it drives multiple rounds.
        "round_index": 0,
        "scenario_name": environment.scenario.name,
        "seed": seed,
        "step_count": step_count,
        "environment_max_steps": environment.config.max_steps,
        "terminal": terminal,
        "truncated": truncation,
        "team_success": team_success,
        "task_reward_sum": totals["task_reward_sum"],
        "hazard_cost_sum": totals["hazard_cost_sum"],
        "sensing_cost_sum": totals["sensing_cost_sum"],
        "total_sensing_actions": totals["total_sensing_actions"],
        "total_movement_actions": totals["total_movement_actions"],
        "hazard_entry_count": totals["hazard_entry_count"],
        "revealed_hazard_count": totals["revealed_hazard_count"],
        "sensing_rate": float(totals["total_sensing_actions"]) / float(action_decisions),
        "sample_actions": settings.sample_actions,
    }
    return episode, record


def _reject_already_done_environment(
    environment: RiskAwareActiveSensingGridEnvironment,
) -> None:
    if environment.is_done:
        raise RuntimeError(
            "Stage 23-C rollout collection requires a resettable active environment"
        )


def _rollout_summary(
    batch: RolloutBatch,
    settings: Stage23CRolloutConfig,
    core_config: MAPPOCoreConfig,
    *,
    max_time: int,
) -> dict[str, Any]:
    valid = batch.valid_mask
    valid_transition_count = int(valid.sum().item())
    sensing_rate_denominator = (
        float(valid_transition_count) if valid_transition_count else 1.0
    )
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
        "episode_count": settings.max_episodes,
        "evaluation_run": False,
        "final_evaluation_run": False,
        "hazard_cost_mean": float(batch.hazard_cost[valid].mean().item()),
        "local_seed_controls_recorded": True,
        "max_collected_episode_length": max_time,
        "movement_action_count": int(core_config.movement_action_count),
        "movement_action_factor_preserved": True,
        "next_value_bootstrap_convention": (
            "terminal_next_values_zeroed;truncation_next_values_preserved"
        ),
        "old_log_probability_source": "pre_update_rollout_policy",
        "old_log_probability_is_detached": True,
        "on_policy_rollout": True,
        "padding_strategy": "valid_mask_padding",
        "paper_facing_results_created": False,
        "recurrent_state_serialized": False,
        "rollout_buffer_artifact_created": False,
        "safe_variable_length_episode_padding": True,
        "sample_actions": settings.sample_actions,
        "scenario_names": list(settings.scenario_names),
        "sensing_action_count": int(core_config.sensing_action_count),
        "sensing_action_factor_preserved": True,
        "sensing_cost_mean": float(batch.sensing_cost[valid].mean().item()),
        "sensing_rate": float((batch.sensing_action[valid] != 0).sum().item())
        / sensing_rate_denominator,
        "stage": "23-C",
        "task_reward_sum": float(batch.task_reward[valid].sum().item()),
        "terminal_count": int(batch.terminal[valid].sum().item()),
        "time_steps": int(batch.actor_observation.shape[1]),
        "total_movement_actions": int((batch.movement_action[valid] != 0).sum().item()),
        "total_sensing_actions": int((batch.sensing_action[valid] != 0).sum().item()),
        "training_run": "bounded_stage23c_development_only",
        "truncation_count": int(batch.truncation[valid].sum().item()),
        "valid_transition_count": valid_transition_count,
    }


def _environment_summary(
    environment: RiskAwareActiveSensingGridEnvironment,
    settings: Stage23CRolloutConfig,
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
        "reference_scenario_name": scenario.name,
        "scenario_names": list(settings.scenario_names),
        "seed": settings.seed,
        "sensing_action_count": core_config.sensing_action_count,
        "stage": "23-C",
        "training_run": "bounded_stage23c_development_only",
    }

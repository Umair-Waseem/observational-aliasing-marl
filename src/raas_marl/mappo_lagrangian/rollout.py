"""Bounded development rollout collection for Stage 22.

Rollout collection sets the PyTorch global RNG seed for local bounded
development reproducibility (the single documented seeding policy of
``_rollout_common.set_rollout_seed``; no collector samples the Python
``random`` module, so it is intentionally not seeded, S2-25). It does not
preserve caller RNG state and does not claim cross-platform or cross-version
bitwise determinism.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from raas_marl.mappo_lagrangian._validation import (
    require_bounded_seed,
    require_positive_int,
)
from raas_marl.mappo_lagrangian.buffer import RolloutBatch
from raas_marl.mappo_lagrangian.config import MAPPOCoreConfig
from raas_marl.mappo_lagrangian.env_contracts import (
    validate_actor_visible_payload,
    validate_critic_central_payload,
)
from raas_marl.mappo_lagrangian.env_adapter import (
    DevelopmentAdapterConfig,
    DevelopmentTransition,
    Stage22DevelopmentEnvironment,
    default_stage22_core_config,
)
from raas_marl.mappo_lagrangian._rollout_common import (
    concat_episodes,
    empty_rollout_fields,
    float_vector,
    set_rollout_seed,
    stack_time,
    stage22_team_end_flags_ended,
    validate_rollout_batch_values,
    validate_transition_for_tensorization,
    validate_transition_identity_coverage,
)
from raas_marl.mappo_lagrangian.model import ActorInput, CriticInput, RecurrentMAPPOActorCritic


_STAGE22_ZERO_SEED_MESSAGE = (
    "seed must be a positive integer; seed 0 is intentionally "
    "reserved/rejected for bounded Stage 22 development-run consistency"
)


@dataclass(frozen=True)
class RolloutCollectionConfig:
    """Small bounded rollout-collection settings for development only."""

    max_episodes: int = 2
    max_steps_per_episode: int = 4
    sample_actions: bool = False
    seed: int = 22

    def __post_init__(self) -> None:
        require_positive_int("max_episodes", self.max_episodes, error_suffix="must be positive")
        require_positive_int(
            "max_steps_per_episode", self.max_steps_per_episode, error_suffix="must be positive"
        )
        if self.max_episodes > 4:
            raise ValueError("max_episodes must be <= 4 for Stage 22 development")
        if self.max_steps_per_episode > 16:
            raise ValueError("max_steps_per_episode must be <= 16 for Stage 22 development")
        if not isinstance(self.sample_actions, bool):
            raise TypeError("sample_actions must be a bool")
        # torch.manual_seed unpacks into a signed 64-bit integer; the per-step
        # sampling reseed adds ``episode_index * 1000 + step_index`` (see the
        # reseed at collection time), so the seed itself is bounded with enough
        # headroom that the largest derived reseed still fits in signed-64-bit
        # range (NEW-rollout-1b / S2-19). A larger seed would otherwise escape
        # torch.manual_seed as an undeclared RuntimeError.
        max_reseed_offset = self.max_episodes * 1000 + self.max_steps_per_episode
        require_bounded_seed(
            "seed",
            self.seed,
            max_offset=max_reseed_offset,
            zero_seed_message=_STAGE22_ZERO_SEED_MESSAGE,
            error_suffix="plus per-step reseed offset must fit in a signed 64-bit integer",
        )


@dataclass(frozen=True)
class CollectedRollout:
    """Development rollout data plus JSON-compatible summary metadata."""

    batch: RolloutBatch
    summary: dict[str, Any]


def collect_development_rollout(
    model: RecurrentMAPPOActorCritic,
    environment: Stage22DevelopmentEnvironment,
    rollout_config: RolloutCollectionConfig | None = None,
) -> CollectedRollout:
    """Collect a RolloutBatch-compatible bounded development rollout.

    This function sets the PyTorch global RNG seed as a local Stage 22
    development control and does not preserve caller RNG state.
    """

    if not isinstance(model, RecurrentMAPPOActorCritic):
        raise TypeError("model must be a RecurrentMAPPOActorCritic")
    if not isinstance(environment, Stage22DevelopmentEnvironment):
        raise TypeError("environment must be a Stage22DevelopmentEnvironment")
    # Session 2026-07-04: interface coherence between the model and the
    # environment core config is checked up front (the stage23b collector
    # already does this) so a mismatched pairing fails with one deterministic
    # error instead of a mid-collection tensor-shape error. Hidden-layer
    # dimensions may legitimately differ and are deliberately not compared.
    model_config = model.config
    core_config = environment.config.core_config
    if (
        model_config.actor_observation_dim,
        model_config.revealed_information_dim,
        model_config.central_state_dim,
        model_config.sensing_action_count,
        model_config.movement_action_count,
        model_config.dtype,
        model_config.device,
    ) != (
        core_config.actor_observation_dim,
        core_config.revealed_information_dim,
        core_config.central_state_dim,
        core_config.sensing_action_count,
        core_config.movement_action_count,
        core_config.dtype,
        core_config.device,
    ):
        raise ValueError(
            "model config does not match environment core_config interface dimensions"
        )
    if rollout_config is None:
        settings = RolloutCollectionConfig()
    elif isinstance(rollout_config, RolloutCollectionConfig):
        settings = rollout_config
    else:
        raise TypeError("rollout_config must be a RolloutCollectionConfig or None")
    set_rollout_seed(settings.seed)
    was_training = model.training
    model.eval()
    try:
        episode_tensors = [
            _collect_single_episode(model, environment, settings, episode_index)
            for episode_index in range(settings.max_episodes)
        ]
        batch = RolloutBatch(
            actor_observation=concat_episodes(episode_tensors, "actor_observation"),
            revealed_information=concat_episodes(episode_tensors, "revealed_information"),
            central_state=concat_episodes(episode_tensors, "central_state"),
            sensing_action=concat_episodes(episode_tensors, "sensing_action"),
            movement_action=concat_episodes(episode_tensors, "movement_action"),
            old_sensing_log_probability=concat_episodes(episode_tensors, "old_sensing_log_probability"),
            old_movement_log_probability=concat_episodes(episode_tensors, "old_movement_log_probability"),
            task_reward=concat_episodes(episode_tensors, "task_reward"),
            hazard_cost=concat_episodes(episode_tensors, "hazard_cost"),
            sensing_cost=concat_episodes(episode_tensors, "sensing_cost"),
            reward_value=concat_episodes(episode_tensors, "reward_value"),
            hazard_cost_value=concat_episodes(episode_tensors, "hazard_cost_value"),
            next_reward_value=concat_episodes(episode_tensors, "next_reward_value"),
            next_hazard_cost_value=concat_episodes(episode_tensors, "next_hazard_cost_value"),
            terminal=concat_episodes(episode_tensors, "terminal"),
            truncation=concat_episodes(episode_tensors, "truncation"),
            valid_mask=concat_episodes(episode_tensors, "valid_mask"),
        )
        batch.validate(environment.config.core_config)
        validate_rollout_batch_values(batch)
        # NEW-rollout-6: guard the masked cost means against an all-False
        # valid_mask, which would otherwise emit ``nan`` into the JSON summary.
        # An equal-length Stage 22 development rollout always has at least one
        # valid step, so this only changes behavior for a degenerate empty mask.
        if bool(batch.valid_mask.any().item()):
            hazard_cost_mean = float(batch.hazard_cost[batch.valid_mask].mean().item())
            sensing_cost_mean = float(batch.sensing_cost[batch.valid_mask].mean().item())
        else:
            hazard_cost_mean = 0.0
            sensing_cost_mean = 0.0
        summary = {
            "development_only": True,
            "batch_size": int(batch.actor_observation.shape[0]),
            "episode_count": settings.max_episodes,
            "agent_count": int(environment.config.agent_count),
            "time_steps": int(batch.actor_observation.shape[1]),
            "sensing_action_count": int(environment.config.core_config.sensing_action_count),
            "movement_action_count": int(environment.config.core_config.movement_action_count),
            "task_reward_sum": float(batch.task_reward.sum().item()),
            "hazard_cost_mean": hazard_cost_mean,
            "sensing_cost_mean": sensing_cost_mean,
            "terminal_count": int(batch.terminal.sum().item()),
            "truncation_count": int(batch.truncation.sum().item()),
            "sample_actions": settings.sample_actions,
            "rng_state_preserved": False,
            "local_seed_controls_recorded": True,
            "cross_platform_bitwise_determinism_claimed": False,
        }
        return CollectedRollout(batch=batch, summary=summary)
    finally:
        model.train(was_training)


def _collect_single_episode(
    model: RecurrentMAPPOActorCritic,
    environment: Stage22DevelopmentEnvironment,
    settings: RolloutCollectionConfig,
    episode_index: int,
) -> dict[str, torch.Tensor]:
    current_payloads = environment.reset()
    _validate_reset_payloads_for_rollout(
        current_payloads,
        environment.config.core_config,
        environment.config.agent_count,
    )
    history_state: torch.Tensor | None = None
    fields = empty_rollout_fields()
    for step_index in range(settings.max_steps_per_episode):
        actor_observation = torch.stack(
            [item["actor_visible"]["actor_observation"] for item in current_payloads],
            dim=0,
        ).unsqueeze(1)
        revealed_information = torch.stack(
            [item["actor_visible"]["revealed_information"] for item in current_payloads],
            dim=0,
        ).unsqueeze(1)
        central_state = torch.stack(
            [item["critic_visible"]["central_state"] for item in current_payloads],
            dim=0,
        ).unsqueeze(1)
        with torch.no_grad():
            if settings.sample_actions:
                # NEW-rollout-4: the per-step reseed is torch-only by design.
                # Stochastic action sampling draws from the torch global RNG,
                # and environment.step() is deterministic, so reseeding the
                # Python ``random`` stream here would be dead work (mirrors the
                # single-stream seeding policy of set_rollout_seed). The offset
                # ``episode_index * 1000 + step_index`` keeps per-step reseeds
                # collision-free (M-1) and within the config-validated bound.
                torch.manual_seed(settings.seed + episode_index * 1000 + step_index)
            output = model(
                ActorInput(actor_observation, revealed_information, history_state),
                CriticInput(central_state),
                sample_actions=settings.sample_actions,
            )
            policy = output["policy"]
            sensing_action = policy.sensing_action.detach().squeeze(1)
            movement_action = policy.movement_action.detach().squeeze(1)
            transitions = environment.step(sensing_action, movement_action)
            _validate_returned_transitions_for_rollout(
                transitions,
                environment.config.core_config,
            )
            next_central_state = torch.stack(
                [item.next_critic_visible["central_state"] for item in transitions],
                dim=0,
            ).unsqueeze(1)
            next_reward_value = model.evaluate_reward_critic(
                CriticInput(next_central_state)
            ).detach()
            next_hazard_cost_value = model.evaluate_cost_critic(
                CriticInput(next_central_state)
            ).detach()
            terminal = torch.tensor(
                [bool(item.contract.done_flags["terminal"]) for item in transitions],
                dtype=torch.bool,
                device=actor_observation.device,
            ).unsqueeze(1)
            truncation = torch.tensor(
                [bool(item.contract.done_flags["truncated"]) for item in transitions],
                dtype=torch.bool,
                device=actor_observation.device,
            ).unsqueeze(1)
            episode_ended = stage22_team_end_flags_ended(
                terminal,
                truncation,
                context="Stage 22 development rollout collection",
            )
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
                    [
                        item.contract.rewards_and_costs["task_reward"]
                        for item in transitions
                    ],
                    actor_observation,
                )
            )
            fields["hazard_cost"].append(
                float_vector(
                    [
                        item.contract.rewards_and_costs["hazard_cost"]
                        for item in transitions
                    ],
                    actor_observation,
                )
            )
            fields["sensing_cost"].append(
                float_vector(
                    [
                        item.contract.rewards_and_costs["sensing_cost"]
                        for item in transitions
                    ],
                    actor_observation,
                )
            )
            fields["reward_value"].append(output["reward_value"].detach().squeeze(1))
            fields["hazard_cost_value"].append(output["hazard_cost_value"].detach().squeeze(1))
            fields["next_reward_value"].append(next_reward_value.squeeze(1))
            fields["next_hazard_cost_value"].append(next_hazard_cost_value.squeeze(1))
            fields["terminal"].append(terminal.squeeze(1))
            fields["truncation"].append(truncation.squeeze(1))
            fields["valid_mask"].append(torch.ones_like(terminal.squeeze(1), dtype=torch.bool))
            if episode_ended:
                break
            history_state = output["next_recurrent_state"].detach()
            current_payloads = [
                {
                    "actor_visible": item.next_actor_visible,
                    "critic_visible": item.next_critic_visible,
                }
                for item in transitions
            ]
    return {name: stack_time(values) for name, values in fields.items()}


def _validate_reset_payloads_for_rollout(
    payloads: object,
    config: MAPPOCoreConfig,
    agent_count: int,
) -> None:
    if not isinstance(payloads, list):
        raise TypeError("environment.reset() must return a list of payload mappings")
    if not payloads:
        raise ValueError("environment.reset() payloads must be nonempty")
    if len(payloads) != agent_count:
        raise ValueError("environment.reset() payload count must match agent_count")
    for index, payload in enumerate(payloads):
        if not isinstance(payload, Mapping):
            raise TypeError(f"environment.reset() payload {index} must be a mapping")
        if "actor_visible" not in payload:
            raise ValueError(f"environment.reset() payload {index} missing actor_visible")
        if "critic_visible" not in payload:
            raise ValueError(f"environment.reset() payload {index} missing critic_visible")
        validate_actor_visible_payload(payload["actor_visible"], config)
        validate_critic_central_payload(payload["critic_visible"], config)


def _validate_returned_transitions_for_rollout(
    transitions: list[DevelopmentTransition],
    config: MAPPOCoreConfig,
) -> None:
    for item in transitions:
        try:
            validate_transition_for_tensorization(
                item, config, transition_type=DevelopmentTransition
            )
        except ValueError as exc:
            # Message normalization deliberately coupled to the exact
            # env_contracts done-flag wording: the public contract says
            # "truncated", the Stage 22 rollout surface says "truncation".
            # If env_contracts rewording ever breaks this equality the
            # original message simply propagates unchanged. Documented
            # 2026-07-04 (NEW-rollout-5: kept as a deliberate message remap).
            if str(exc) == "terminal and truncated cannot both be true":
                raise ValueError("terminal and truncation cannot both be true") from exc
            raise
    validate_transition_identity_coverage(transitions, config)


def make_default_development_environment() -> Stage22DevelopmentEnvironment:
    """Return the default CPU development environment used by Stage 22."""

    return Stage22DevelopmentEnvironment(
        DevelopmentAdapterConfig(core_config=default_stage22_core_config())
    )


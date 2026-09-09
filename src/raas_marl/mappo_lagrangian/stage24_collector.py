"""Stage 24 multi-episode Stage 23-A training-readiness collector.

This module is not final evaluation. It collects in-memory Stage 23-A active-
sensing batches for readiness diagnostics, keeps explicit sensing/movement
action factors, and writes no checkpoint, model, optimizer, rollout-buffer, or
replay-buffer artifacts.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
import platform
import sys
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
from raas_marl.environments.active_sensing.tensor_adapter import (
    default_stage23_core_config,
    stage23_transition_to_environment_step_contract,
)
from raas_marl.mappo_lagrangian._rollout_common import (
    actions_from_policy as _actions_from_policy,
    actor_observation_batch as _actor_observation_batch,
    central_state_batch as _central_state_batch,
    empty_rollout_fields as _empty_rollout_fields,
    float_vector as _float_vector,
    merge_padded_episodes as _merge_padded_episodes_shared,
    parameter_delta_l1 as _parameter_delta_l1,
    per_step_sampling_seed,
    revealed_information_batch as _revealed_information_batch,
    set_rollout_seed as _set_collection_seed,
    stack_time as _stack_time,
    validate_parallel_payloads as _validate_parallel_payloads,
    validate_rollout_batch_values,
    validate_stage23_core_config as _validate_stage23_core_config,
    validate_step_api_payloads as _validate_step_api_payloads,
    validate_team_end_flags as _validate_team_end_flags,
)
from raas_marl.mappo_lagrangian._validation import (
    require_bounded_seed,
    require_positive_int,
    require_positive_seed,
)
from raas_marl.mappo_lagrangian.buffer import RolloutBatch
from raas_marl.mappo_lagrangian.config import MAPPOCoreConfig
from raas_marl.mappo_lagrangian.lagrange import LagrangeMultiplier
from raas_marl.mappo_lagrangian.model import (
    ActorInput,
    CriticInput,
    RecurrentMAPPOActorCritic,
)
from raas_marl.mappo_lagrangian.update import (
    Stage22UpdateConfig,
    Stage22UpdateResult,
    default_stage22_update_config,
    hazard_advantage_penalizes_policy_objective,
    stage22_ppo_lagrangian_update,
)


# Upper bound on the per-episode step horizon for the seed-overflow guard. The
# Stage 23-A environment ``max_steps`` defaults to 32 and is the collector's full
# horizon; ``max_steps_per_episode`` must be None or equal to it. Used only to
# size the signed-64-bit seed headroom (NEW-stage24_collector-2).
_STAGE23A_MAX_STEP_HORIZON = 32


# NEW-stage24_collector-12: the Stage 23-A scenario catalog is deterministic
# (a fresh dict of frozen Stage23Scenario dataclasses every call). Cache the
# built catalog once so per-episode environment construction does not rebuild
# it. The cached mapping is treated as read-only.
_SCENARIO_CATALOG_CACHE: dict[str, Any] | None = None


def _cached_scenario_catalog() -> dict[str, Any]:
    global _SCENARIO_CATALOG_CACHE
    if _SCENARIO_CATALOG_CACHE is None:
        _SCENARIO_CATALOG_CACHE = stage23_scenario_catalog()
    return _SCENARIO_CATALOG_CACHE


@dataclass(frozen=True)
class Stage24CollectorConfig:
    """Stage 24 collector settings over the Stage 23-A environment."""

    max_episodes: int = 2
    scenario_names: tuple[str, ...] = ("risk_gate_hidden_hazard",)
    phase: str = "training"
    seed: int = 24000
    max_steps_per_episode: int | None = None
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        require_positive_int("max_episodes", self.max_episodes, error_suffix="must be positive")
        if self.max_episodes < 2:
            raise ValueError("Stage 24 collector requires more than one episode")
        if self.phase not in {"training", "evaluation"}:
            raise ValueError("phase must be 'training' or 'evaluation'")
        # NEW-stage24_collector-4: seed carries the cross-stage seed-0
        # reservation semantics, not a bare positivity check.
        require_positive_seed("seed", self.seed)
        if not isinstance(self.scenario_names, tuple) or not self.scenario_names:
            raise ValueError("scenario_names must be a non-empty tuple")
        catalog = _cached_scenario_catalog()
        for name in self.scenario_names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("scenario_names entries must be non-empty strings")
            if name not in catalog:
                raise ValueError(f"unknown Stage 23-A scenario: {name}")
        if self.max_steps_per_episode is not None:
            require_positive_int(
                "max_steps_per_episode", self.max_steps_per_episode, error_suffix="must be positive"
            )
        # NEW-stage24_collector-2: the per-step sampling reseed derives
        # ``seed + episode_index*1000 + step_index`` (M-1 formula); an unbounded
        # seed would overflow signed 64-bit and crash torch.manual_seed with an
        # undeclared RuntimeError. Bound the seed so the largest derived reseed
        # (max_episodes*1000 + the per-episode step horizon) still fits. The
        # per-episode horizon is capped by the Stage 23-A environment max_steps
        # (default 32); when max_steps_per_episode is None it must equal that
        # horizon, so 32 is a sound upper bound for the offset.
        max_step_horizon = (
            self.max_steps_per_episode
            if self.max_steps_per_episode is not None
            else _STAGE23A_MAX_STEP_HORIZON
        )
        require_bounded_seed(
            "seed",
            self.seed,
            max_offset=self.max_episodes * 1000 + max_step_horizon,
            error_suffix="must fit in a signed 64-bit integer after per-step reseed offsets",
        )
        # NEW-stage24_collector-11: dtype/device string grammar (non-empty, no
        # leading/trailing whitespace, exact spelling) is subsumed by the
        # exact-equality checks below: any padded or malformed value differs from
        # the canonical default and is rejected here, so no separate grammar pass
        # is needed.
        expected = default_stage23_core_config()
        if self.dtype != expected.dtype:
            raise ValueError("Stage 24 collector requires the default Stage 23-A dtype")
        if self.device != expected.device:
            raise ValueError("Stage 24 collector requires CPU device 'cpu'")

    @property
    def sample_actions(self) -> bool:
        return self.phase == "training"


@dataclass(frozen=True)
class Stage24CollectedBatch:
    """In-memory Stage 24 rollout batch and JSON-compatible summaries."""

    batch: RolloutBatch
    summary: dict[str, Any]
    episode_summaries: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class Stage24AUpdateConfig:
    """Stage 24-A readiness-update boundary around internal Stage 22 semantics."""

    internal_stage22_config: Stage22UpdateConfig
    readiness_smoke_only: bool
    stage22_semantics_reused_internally: bool
    advantage_normalization_used: bool
    nonzero_lagrange_lower_bound_added: bool
    hazard_budget_schedule_added: bool

    def __post_init__(self) -> None:
        if not isinstance(self.internal_stage22_config, Stage22UpdateConfig):
            raise TypeError("internal_stage22_config must be a Stage22UpdateConfig")
        for field_name in (
            "readiness_smoke_only",
            "stage22_semantics_reused_internally",
            "advantage_normalization_used",
            "nonzero_lagrange_lower_bound_added",
            "hazard_budget_schedule_added",
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise TypeError(f"{field_name} must be a strict bool")
        if self.advantage_normalization_used != self.internal_stage22_config.normalize_advantages:
            raise ValueError(
                "advantage_normalization_used must match internal config normalize_advantages"
            )
        if (
            not self.readiness_smoke_only
            and self.stage22_semantics_reused_internally
        ):
            raise ValueError(
                "Stage-22-derived helpers cannot be marked final-training infrastructure"
            )
        if self.nonzero_lagrange_lower_bound_added:
            raise ValueError("nonzero Lagrange lower bound is not implemented in Stage 24-A")
        if self.hazard_budget_schedule_added:
            raise ValueError("hazard budget schedule is not implemented in Stage 24-A")


def default_stage24_update_config() -> Stage22UpdateConfig:
    """Return the internal Stage 22 update config used by the Stage 24-A smoke.

    NEW-stage24_collector-8: despite the ``stage24`` name this returns a
    ``Stage22UpdateConfig`` (the ``internal_stage22_config`` of the Stage 24-A
    readiness boundary), not a Stage 24-specific config type. It is kept under
    this name because the symbol is exported for compatibility (see the module
    ``__all__`` and the package ``__init__`` export). It is diagnostic-smoke-only:
    not final-training infrastructure and not final-evaluation protocol
    infrastructure. Prefer ``default_stage24a_update_config()`` for the named
    Stage 24-A readiness-update boundary.
    """

    return default_stage24a_update_config().internal_stage22_config


def default_stage24a_update_config() -> Stage24AUpdateConfig:
    """Return the Stage 24-A named readiness-update boundary."""

    base = default_stage22_update_config()
    internal = Stage22UpdateConfig(
        algorithm=base.algorithm,
        losses=base.losses,
        lagrange=base.lagrange,
        learning_rate=base.learning_rate,
        max_update_epochs=1,
        max_grad_norm=base.max_grad_norm,
        normalize_advantages=True,
    )
    return Stage24AUpdateConfig(
        internal_stage22_config=internal,
        readiness_smoke_only=True,
        stage22_semantics_reused_internally=True,
        advantage_normalization_used=internal.normalize_advantages,
        nonzero_lagrange_lower_bound_added=False,
        hazard_budget_schedule_added=False,
    )


def stage24a_ppo_lagrangian_readiness_update(
    model: RecurrentMAPPOActorCritic,
    batch: RolloutBatch,
    config: Stage24AUpdateConfig | None = None,
    *,
    lagrange_multiplier: LagrangeMultiplier | None = None,
) -> Stage22UpdateResult:
    """Run the diagnostic-only Stage 24-A PPO-Lagrangian readiness update."""

    settings = default_stage24a_update_config() if config is None else config
    if not isinstance(settings, Stage24AUpdateConfig):
        raise TypeError("config must be a Stage24AUpdateConfig or None")
    return stage22_ppo_lagrangian_update(
        model,
        batch,
        settings.internal_stage22_config,
        lagrange_multiplier=lagrange_multiplier,
    )


def stage24a_runtime_reproducibility_record(
    *,
    seed_values: tuple[int, ...],
    device: str,
    dtype: str,
    requested_num_threads: int | None = None,
    requested_num_interop_threads: int | None = None,
    allow_thread_setting: bool = False,
) -> dict[str, Any]:
    """Record Stage 24-A runtime/thread metadata for readiness diagnostics only.

    SIDE EFFECT (NEW-stage24_collector-14): when ``allow_thread_setting`` is True
    AND a ``requested_num_threads`` / ``requested_num_interop_threads`` is given,
    this function mutates the process-global torch thread configuration via
    ``torch.set_num_threads`` / ``torch.set_num_interop_threads`` before recording
    the resulting values. This is dormant under the default
    ``allow_thread_setting=False``, where the function is purely observational.
    Callers that only want to record metadata must leave ``allow_thread_setting``
    at its default.
    """

    if not isinstance(seed_values, tuple) or not seed_values:
        raise ValueError("seed_values must be a non-empty tuple")
    for seed in seed_values:
        # NEW-stage24_collector-4: seed-0 reservation semantics for parity
        # with the rest of the stages.
        require_positive_seed("seed", seed)
    _require_non_empty_string("device", device)
    _require_non_empty_string("dtype", dtype)
    if requested_num_threads is not None:
        require_positive_int(
            "requested_num_threads", requested_num_threads, error_suffix="must be positive"
        )
    if requested_num_interop_threads is not None:
        require_positive_int(
            "requested_num_interop_threads",
            requested_num_interop_threads,
            error_suffix="must be positive",
        )
    if not isinstance(allow_thread_setting, bool):
        raise TypeError("allow_thread_setting must be a strict bool")
    thread_setting_attempted = (
        allow_thread_setting
        and (requested_num_threads is not None or requested_num_interop_threads is not None)
    )
    thread_setting_errors: list[dict[str, Any]] = []
    if thread_setting_attempted:
        if requested_num_threads is not None:
            try:
                torch.set_num_threads(requested_num_threads)
            except Exception as exc:
                thread_setting_errors.append(
                    _thread_setting_error_record(
                        exc,
                        setting_name="torch_num_threads",
                        requested_value=requested_num_threads,
                    )
                )
        if requested_num_interop_threads is not None:
            try:
                torch.set_num_interop_threads(requested_num_interop_threads)
            except Exception as exc:
                thread_setting_errors.append(
                    _thread_setting_error_record(
                        exc,
                        setting_name="torch_num_interop_threads",
                        requested_value=requested_num_interop_threads,
                    )
                )

    return {
        "stage": "24-A",
        "python_version": sys.version,
        "platform": platform.platform(),
        "platform_machine": platform.machine(),
        "platform_processor": platform.processor(),
        "torch_version": torch.__version__,
        "torch_num_threads": int(torch.get_num_threads()),
        "torch_num_interop_threads": int(torch.get_num_interop_threads()),
        "env": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "NUMEXPR_NUM_THREADS": os.environ.get("NUMEXPR_NUM_THREADS"),
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        },
        "device": device,
        "dtype": dtype,
        "seed_values": list(seed_values),
        "thread_settings_requested": {
            "torch_num_threads": requested_num_threads,
            "torch_num_interop_threads": requested_num_interop_threads,
        },
        "thread_settings_set": thread_setting_attempted and not thread_setting_errors,
        "thread_settings_policy": (
            "requested_and_set"
            if thread_setting_attempted and not thread_setting_errors
            else "requested_but_not_fully_set"
            if thread_setting_attempted
            else "observed_only"
        ),
        "thread_setting_errors": thread_setting_errors,
        "cpu_thread_settings_set_or_only_observed": (
            "set" if thread_setting_attempted and not thread_setting_errors else "observed_only"
        ),
        "cross_platform_bitwise_determinism_claimed": False,
        "runtime_metadata_used_as_claim_evidence": False,
    }


def collect_stage24_stage23a_training_batch(
    model: RecurrentMAPPOActorCritic,
    collector_config: Stage24CollectorConfig | None = None,
) -> Stage24CollectedBatch:
    """Collect a multi-episode Stage 23-A batch in memory."""

    if not isinstance(model, RecurrentMAPPOActorCritic):
        raise TypeError("model must be a RecurrentMAPPOActorCritic")
    settings = (
        Stage24CollectorConfig()
        if collector_config is None
        else collector_config
    )
    if not isinstance(settings, Stage24CollectorConfig):
        raise TypeError("collector_config must be a Stage24CollectorConfig or None")
    core_config = model.config
    _validate_stage23_core_config(core_config, context="Stage 24")
    _set_collection_seed(settings.seed)

    was_training = model.training
    model.eval()
    try:
        episodes: list[dict[str, torch.Tensor]] = []
        episode_summaries: list[dict[str, Any]] = []
        for episode_index in range(settings.max_episodes):
            scenario_name = settings.scenario_names[
                episode_index % len(settings.scenario_names)
            ]
            environment = _make_stage24_environment(
                scenario_name=scenario_name,
                seed=settings.seed + episode_index,
                max_steps_per_episode=settings.max_steps_per_episode,
            )
            episode, summary = _collect_single_stage24_episode(
                model,
                environment,
                settings,
                core_config,
                episode_index=episode_index,
                seed=settings.seed + episode_index,
            )
            episodes.append(episode)
            episode_summaries.append(summary)
        max_time = max(int(episode["valid_mask"].shape[1]) for episode in episodes)
        batch = _merge_padded_episodes_shared(episodes, max_time=max_time)
        batch.validate(core_config)
        # NEW-stage24_collector-1: re-check detachment, finiteness, and
        # cost-nonnegativity on the merged batch (required by the Stage 23-C
        # spec; the collector is the multi-episode prototype so the gap would
        # propagate if left unclosed here).
        validate_rollout_batch_values(batch)
        summary = _collector_summary(
            batch,
            settings,
            episode_summaries,
            core_config,
            max_time=max_time,
        )
        return Stage24CollectedBatch(
            batch=batch,
            summary=summary,
            episode_summaries=tuple(episode_summaries),
        )
    finally:
        model.train(was_training)


def run_stage24a_training_smoke(seed: int = 24050) -> dict[str, Any]:
    """Run a bounded Stage 24-A readiness smoke without serializing artifacts."""

    # NEW-stage24_collector-4: seed-0 reservation semantics for parity.
    require_positive_seed("seed", seed)
    torch.manual_seed(seed)
    core = default_stage23_core_config()
    runtime = stage24a_runtime_reproducibility_record(
        seed_values=(seed, seed + 100),
        device=core.device,
        dtype=core.dtype,
    )
    model = RecurrentMAPPOActorCritic(core)
    training = collect_stage24_stage23a_training_batch(
        model,
        Stage24CollectorConfig(
            max_episodes=2,
            scenario_names=("risk_gate_hidden_hazard", "standard_branching_hazard"),
            phase="training",
            seed=seed,
        ),
    )
    before = [parameter.detach().clone() for parameter in model.parameters()]
    update_result = stage24a_ppo_lagrangian_readiness_update(
        model,
        training.batch,
        default_stage24a_update_config(),
        lagrange_multiplier=LagrangeMultiplier(0.1, 0.2),
    )
    parameter_delta_l1 = _parameter_delta_l1(before, list(model.parameters()))
    evaluation = collect_stage24_stage23a_training_batch(
        model,
        Stage24CollectorConfig(
            max_episodes=2,
            scenario_names=("risk_gate_hidden_hazard", "standard_branching_hazard"),
            phase="evaluation",
            seed=seed + 100,
        ),
    )
    return {
        "stage": "24-A",
        "status": "stage24a_training_smoke_not_final_evaluation",
        "claim_status": "not supported / not ready for final evaluation",
        "final_evaluation_run": False,
        "claim_evidence_created": False,
        "paper_facing_output_created": False,
        "stage23a_active_sensing_environment_used": True,
        "env_adapter_used": False,
        "toy_or_fake_development_environment_used": False,
        "checkpoint_created": False,
        "model_artifact_created": False,
        "optimizer_state_saved": False,
        "rollout_buffer_serialized": False,
        "replay_buffer_serialized": False,
        "training_collection": training.summary,
        "evaluation_collection": evaluation.summary,
        "runtime_reproducibility": runtime,
        "update_summary": update_result.summary,
        "parameters_changed_after_update": parameter_delta_l1 > 0.0,
        "parameter_delta_l1": parameter_delta_l1,
        "stochastic_training_sampling_used": training.summary["sample_actions"],
        "deterministic_evaluation_mode_exists": not evaluation.summary["sample_actions"],
        # NEW-stage24_collector-5: reflect the real per-collection validation
        # outcome instead of hardcoding True.
        "batch_validate_passed": bool(
            training.summary["rollout_batch_validate_passed"]
            and evaluation.summary["rollout_batch_validate_passed"]
        ),
    }


def run_stage24_training_smoke(seed: int = 24050) -> dict[str, Any]:
    """Compatibility-only alias for Stage 24-A diagnostic smoke."""

    return run_stage24a_training_smoke(seed=seed)


def stage24_objective_update_diagnosis() -> dict[str, Any]:
    """Return Stage 24 update-path diagnosis without mutating Stage 22 semantics."""

    base = default_stage22_update_config()
    stage24a = default_stage24a_update_config()
    collapse_probe = LagrangeMultiplier(0.1, 0.2)
    for _ in range(8):
        collapse_probe = collapse_probe.update(observed_cost=0.0, cost_budget=0.1)
    return {
        "stage": "24-A",
        "update_path_changed": False,
        "stage24_wrapper_config_created": True,
        "stage24a_update_wrapper_created": True,
        "stage24a_readiness_smoke_only": stage24a.readiness_smoke_only,
        "stage22_semantics_reused_internally": stage24a.stage22_semantics_reused_internally,
        "stage22_semantics_destructively_changed": False,
        "sensing_receives_direct_penalty": base.losses.sensing_cost_coefficient > 0.0,
        "direct_sensing_penalty_expression": (
            "task_reward - sensing_cost_coefficient * sensing_cost"
        ),
        "revealed_information_can_affect_movement_through_actor_inputs": True,
        "hazard_cost_affects_policy_gradient_through_lagrangian_advantage": (
            hazard_advantage_penalizes_policy_objective()
        ),
        "lagrange_multiplier_can_collapse_to_zero": collapse_probe.value == 0.0,
        "advantage_normalization_available": True,
        "stage22_default_normalize_advantages": base.normalize_advantages,
        "stage24_wrapper_normalize_advantages": stage24a.advantage_normalization_used,
        "current_update_helper_stage22_development_specific": True,
        "stage22_helper_usage_diagnostic_only": True,
        "final_training_infrastructure_ready": False,
        "final_evaluation_protocol_lock_blocked_by_update_path": True,
        "default_stage24_update_config_compatibility_only": True,
        "minimum_lagrange_multiplier_added": stage24a.nonzero_lagrange_lower_bound_added,
        "budget_schedule_added": stage24a.hazard_budget_schedule_added,
        "bayesian_belief_claim_made": False,
        "formal_voi_claim_made": False,
        "model_renamed_to_c_rc_mappo_voi": False,
        "remaining_update_limitation": (
            "Stage 24-A uses a named wrapper for multi-episode smoke readiness; "
            "a later reviewed gate should decide whether to add a nonzero "
            "minimum Lagrange multiplier or a schedule."
        ),
    }


def _make_stage24_environment(
    *,
    scenario_name: str,
    seed: int,
    max_steps_per_episode: int | None,
) -> RiskAwareActiveSensingGridEnvironment:
    scenario = _cached_scenario_catalog()[scenario_name]
    base_config = Stage23EnvironmentConfig(
        scenario_name=scenario.name,
        width=scenario.width,
        height=scenario.height,
        seed=seed,
    )
    if max_steps_per_episode is not None and max_steps_per_episode != base_config.max_steps:
        raise ValueError(
            "Stage 24 collector uses the full Stage 23-A scenario horizon; "
            "max_steps_per_episode must be None or equal to environment max_steps"
        )
    return RiskAwareActiveSensingGridEnvironment(base_config)


def _collect_single_stage24_episode(
    model: RecurrentMAPPOActorCritic,
    environment: RiskAwareActiveSensingGridEnvironment,
    settings: Stage24CollectorConfig,
    core_config: MAPPOCoreConfig,
    *,
    episode_index: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    observations, infos = environment.reset(seed=seed)
    _validate_parallel_payloads(observations, infos)
    history_state: torch.Tensor | None = None
    fields = _empty_rollout_fields()
    totals = {
        "task_reward_sum": 0.0,
        "hazard_cost_sum": 0.0,
        "sensing_cost_sum": 0.0,
        "total_sensing_actions": 0,
        "total_movement_actions": 0,
        "hazard_entry_count": 0,
    }
    terminal = False
    truncation = False
    team_success = False
    for step_index in range(environment.config.max_steps):
        actor_observation = _actor_observation_batch(
            observations,
            core_config,
            dtype=settings.dtype,
            device=settings.device,
        )
        revealed_information = _revealed_information_batch(
            observations,
            core_config,
            dtype=settings.dtype,
            device=settings.device,
        )
        central_state = _central_state_batch(
            environment,
            core_config,
            dtype=settings.dtype,
            device=settings.device,
        )
        with torch.no_grad():
            if settings.sample_actions:
                # Issue M-1 fix: space per-episode sampling streams by
                # 1000 steps (the rollout.py convention) so episode e at
                # step s and episode e+1 at step s-1 never share an RNG
                # stream. The environment reset seed is unchanged. The
                # ``base_seed + episode_index*1000 + step_index`` formula lives
                # in the shared ``per_step_sampling_seed`` helper so the reseed
                # sequence stays byte-identical across the deduplicated
                # collectors.
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
            actions = _actions_from_policy(sensing_action, movement_action, core_config)
            next_observations, _rewards, terminations, truncations, next_infos = (
                environment.step(actions)
            )
            _validate_parallel_payloads(next_observations, next_infos)
            _validate_step_api_payloads(_rewards, terminations, truncations, next_infos)
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
            episode_ended = _validate_team_end_flags(
                terminal_tensor, truncation_tensor, context="Stage 24 collector"
            )
            # NEW-stage24_collector-17: compute the all-agents team scalars once
            # here. The shared validator returns only the combined
            # ``episode_ended`` (all-terminal OR all-truncated) and has already
            # rejected partial per-agent endings, so ``all`` equals ``any`` for
            # each flag on an ended team; these are reused below without
            # recomputing the reductions.
            terminal_team_ended = bool(terminal_tensor.all().item())
            truncation_team_ended = bool(truncation_tensor.all().item())
            next_central_state = _central_state_batch(
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
            next_reward_value = torch.where(terminal_tensor, zero, next_reward_value)
            next_hazard_cost_value = torch.where(
                terminal_tensor,
                zero,
                next_hazard_cost_value,
            )
            _append_step_fields(
                fields,
                actor_observation=actor_observation,
                revealed_information=revealed_information,
                central_state=central_state,
                policy=policy,
                sensing_action=sensing_action,
                movement_action=movement_action,
                contracts=contracts,
                output=output,
                next_reward_value=next_reward_value,
                next_hazard_cost_value=next_hazard_cost_value,
                terminal=terminal_tensor,
                truncation=truncation_tensor,
            )
            for index, contract in enumerate(contracts):
                totals["task_reward_sum"] += float(contract.rewards_and_costs[TASK_REWARD_KEY])
                totals["hazard_cost_sum"] += float(contract.rewards_and_costs[HAZARD_COST_KEY])
                totals["sensing_cost_sum"] += float(contract.rewards_and_costs[SENSING_COST_KEY])
                totals["total_sensing_actions"] += int(sensing_action[index].item() != 0)
                totals["total_movement_actions"] += int(movement_action[index].item() != 0)
                totals["hazard_entry_count"] += int(bool(next_infos[STAGE23_AGENT_NAMES[index]]["entered_hazard"]))
                team_success = team_success or bool(next_infos[STAGE23_AGENT_NAMES[index]]["team_success"])
            # NEW-stage24_collector-17: reuse the team scalars computed above.
            terminal = terminal_team_ended
            truncation = truncation_team_ended
            if episode_ended:
                break
            history_state = output["next_recurrent_state"].detach()
            observations = next_observations
            infos = next_infos
    episode = {name: _stack_time(values) for name, values in fields.items()}
    valid_steps = int(episode["valid_mask"].shape[1])
    # NEW-stage24_collector-7: _stack_time rejects an empty step list, so an
    # episode always has valid_steps >= 1; the action-decision denominator is
    # therefore always positive (the former max(1, ...) guard was dead).
    action_count = valid_steps * len(STAGE23_AGENT_NAMES)
    summary = {
        "episode_index": episode_index,
        "scenario_name": environment.scenario.name,
        "seed": seed,
        "phase": settings.phase,
        "step_count": valid_steps,
        "environment_max_steps": environment.config.max_steps,
        "used_full_stage23a_horizon_limit": True,
        "success": team_success,
        "truncation": truncation,
        "terminal": terminal,
        "task_reward_sum": totals["task_reward_sum"],
        "hazard_cost_sum": totals["hazard_cost_sum"],
        "sensing_cost_sum": totals["sensing_cost_sum"],
        "total_sensing_actions": totals["total_sensing_actions"],
        "total_movement_actions": totals["total_movement_actions"],
        "hazard_entry_count": totals["hazard_entry_count"],
        "sensing_rate": float(totals["total_sensing_actions"]) / float(action_count),
    }
    return episode, summary


def _append_step_fields(
    fields: dict[str, list[torch.Tensor]],
    *,
    actor_observation: torch.Tensor,
    revealed_information: torch.Tensor,
    central_state: torch.Tensor,
    policy: Any,
    sensing_action: torch.Tensor,
    movement_action: torch.Tensor,
    contracts: list[Any],
    output: Mapping[str, Any],
    next_reward_value: torch.Tensor,
    next_hazard_cost_value: torch.Tensor,
    terminal: torch.Tensor,
    truncation: torch.Tensor,
) -> None:
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
        _float_vector([contract.rewards_and_costs[TASK_REWARD_KEY] for contract in contracts], actor_observation)
    )
    fields["hazard_cost"].append(
        _float_vector([contract.rewards_and_costs[HAZARD_COST_KEY] for contract in contracts], actor_observation)
    )
    fields["sensing_cost"].append(
        _float_vector([contract.rewards_and_costs[SENSING_COST_KEY] for contract in contracts], actor_observation)
    )
    fields["reward_value"].append(output["reward_value"].detach().squeeze(1))
    fields["hazard_cost_value"].append(output["hazard_cost_value"].detach().squeeze(1))
    fields["next_reward_value"].append(next_reward_value.squeeze(1))
    fields["next_hazard_cost_value"].append(next_hazard_cost_value.squeeze(1))
    fields["terminal"].append(terminal.squeeze(1))
    fields["truncation"].append(truncation.squeeze(1))
    # NEW-stage24_collector-16: build valid_mask from the explicit per-step agent
    # count rather than following terminal's rank via ones_like, so a shape
    # regression in terminal cannot silently reshape the mask. The resulting
    # [agents] all-True bool vector matches the other stored per-step fields.
    fields["valid_mask"].append(
        torch.ones(
            (len(STAGE23_AGENT_NAMES),),
            dtype=torch.bool,
            device=terminal.device,
        )
    )


def _collector_summary(
    batch: RolloutBatch,
    settings: Stage24CollectorConfig,
    episode_summaries: list[dict[str, Any]],
    core_config: MAPPOCoreConfig,
    *,
    max_time: int,
) -> dict[str, Any]:
    valid = batch.valid_mask
    # NEW-stage24_collector-7: guard the sensing-rate denominator against a
    # fully-padded batch (valid.sum() == 0) so it cannot ZeroDivisionError; a
    # normally collected batch always has at least one valid transition.
    valid_transition_count = int(valid.sum().item())
    sensing_rate_denominator = float(valid_transition_count) if valid_transition_count else 1.0
    # NEW-stage24_collector-5: derive the validate-passed flag from an actual
    # re-validation of the merged batch rather than hardcoding True, so the
    # reported key cannot lie if RolloutBatch.validate regresses.
    try:
        batch.validate(core_config)
        rollout_batch_validate_passed = True
    except (TypeError, ValueError):
        rollout_batch_validate_passed = False
    return {
        "stage": "24-A",
        "status": "stage24a_collector_ready_not_final_evaluation",
        "claim_status": "not supported / not ready for final evaluation",
        "phase": settings.phase,
        "episode_count": len(episode_summaries),
        "scenario_names": list(settings.scenario_names),
        "sample_actions": settings.sample_actions,
        "stochastic_training_sampling_used": settings.phase == "training" and settings.sample_actions,
        "deterministic_evaluation_mode": settings.phase == "evaluation" and not settings.sample_actions,
        "batch_size": int(batch.actor_observation.shape[0]),
        "time_steps": int(batch.actor_observation.shape[1]),
        "max_collected_episode_length": max_time,
        "valid_transition_count": valid_transition_count,
        "padding_strategy": "valid_mask_padding",
        "safe_variable_length_episode_padding": True,
        "fixed_horizon_compatible": True,
        "full_stage23a_horizon_supported": True,
        "sensing_action_factor_preserved": True,
        "movement_action_factor_preserved": True,
        "sensing_action_space_count": int(core_config.sensing_action_count),
        "movement_action_space_count": int(core_config.movement_action_count),
        "total_sensing_actions": int((batch.sensing_action[valid] != 0).sum().item()),
        "total_movement_actions": int((batch.movement_action[valid] != 0).sum().item()),
        # NEW-stage24_collector-18: total_action_decisions is the same count as
        # valid_transition_count (one action decision per valid transition). Both
        # keys are retained under their established names for backward-compatible
        # consumers; they share one computed value.
        "total_action_decisions": valid_transition_count,
        "rollout_batch_validate_passed": rollout_batch_validate_passed,
        "final_evaluation_run": False,
        "claim_evidence_created": False,
        "paper_facing_output_created": False,
        "checkpoint_created": False,
        "model_artifact_created": False,
        "optimizer_state_saved": False,
        "rollout_buffer_serialized": False,
        "replay_buffer_serialized": False,
        "env_adapter_used": False,
        "task_reward_sum": float(batch.task_reward[valid].sum().item()),
        "hazard_cost_sum": float(batch.hazard_cost[valid].sum().item()),
        "sensing_cost_sum": float(batch.sensing_cost[valid].sum().item()),
        "sensing_rate": float((batch.sensing_action[valid] != 0).sum().item())
        / sensing_rate_denominator,
        "episode_summaries": episode_summaries,
    }


def _require_non_empty_string(name: str, value: str) -> None:
    # NEW-stage24_collector-6: TypeError for wrong type, ValueError for empty
    # value, matching the project-wide type/value error taxonomy.
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _thread_setting_error_record(
    exc: Exception,
    *,
    setting_name: str,
    requested_value: int,
) -> dict[str, Any]:
    return {
        "setting": setting_name,
        "requested_value": requested_value,
        "exception_type": type(exc).__name__,
        "exception_message": str(exc),
    }


__all__ = [
    "Stage24CollectedBatch",
    "Stage24CollectorConfig",
    "collect_stage24_stage23a_training_batch",
    "default_stage24a_update_config",
    "default_stage24_update_config",
    "run_stage24a_training_smoke",
    "run_stage24_training_smoke",
    "stage24a_ppo_lagrangian_readiness_update",
    "stage24a_runtime_reproducibility_record",
    "stage24_objective_update_diagnosis",
    "Stage24AUpdateConfig",
]

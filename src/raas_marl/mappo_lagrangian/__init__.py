"""Stage 21/22/23-B/24 MAPPO-Lagrangian contracts and development helpers.

Stage 21 exposes MAPPO-Lagrangian core contracts and helpers. Stage 22 exposes
development-only environment adapter, rollout collection, PPO-Lagrangian update,
and bounded development runner helpers. Stage 23-B exposes bounded MAPPO-Lagrangian
training helpers over the Stage 23-A active-sensing environment (one-episode
rollout config/collector, its default-environment factory, the result-root
constants, and the bounded training runner). Stage 24 diagnostic-collector helpers
are exported lazily as well (issue m-1 resolution, 2026-07-03). Every Stage 22/23-B
helper is development-training integration only. The package still does not expose
final evaluation, paper-facing evidence, claim evidence, checkpoints, model
artifact writers, optimizer-state writers, or final-evaluation result-root
creators. The lazy-export mechanism remains intact.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Entries are alphabetized by export name so this literal matches the sorted
# __all__ derived from it (issue NEW-__init__-6); each value is
# (fully_qualified_module_name, attribute_name).
_LAZY_EXPORTS = {
    "ActorInput": ("raas_marl.mappo_lagrangian.model", "ActorInput"),
    "AlgorithmConfig": ("raas_marl.mappo_lagrangian.config", "AlgorithmConfig"),
    "CollectedRollout": ("raas_marl.mappo_lagrangian.rollout", "CollectedRollout"),
    "CriticInput": ("raas_marl.mappo_lagrangian.model", "CriticInput"),
    "DevelopmentAdapterConfig": (
        "raas_marl.mappo_lagrangian.env_adapter",
        "DevelopmentAdapterConfig",
    ),
    "DevelopmentTransition": (
        "raas_marl.mappo_lagrangian.env_adapter",
        "DevelopmentTransition",
    ),
    "EnvironmentContract": (
        "raas_marl.mappo_lagrangian.env_contracts",
        "EnvironmentContract",
    ),
    "EnvironmentStepContract": (
        "raas_marl.mappo_lagrangian.env_contracts",
        "EnvironmentStepContract",
    ),
    "FactorizedPolicyOutput": (
        "raas_marl.mappo_lagrangian.model",
        "FactorizedPolicyOutput",
    ),
    "LagrangeConfig": ("raas_marl.mappo_lagrangian.config", "LagrangeConfig"),
    "LagrangeMultiplier": (
        "raas_marl.mappo_lagrangian.lagrange",
        "LagrangeMultiplier",
    ),
    "LossConfig": ("raas_marl.mappo_lagrangian.config", "LossConfig"),
    "MAPPOCoreConfig": ("raas_marl.mappo_lagrangian.config", "MAPPOCoreConfig"),
    "ModelConfig": ("raas_marl.mappo_lagrangian.config", "ModelConfig"),
    "RecurrentMAPPOActorCritic": (
        "raas_marl.mappo_lagrangian.model",
        "RecurrentMAPPOActorCritic",
    ),
    "RolloutBatch": ("raas_marl.mappo_lagrangian.buffer", "RolloutBatch"),
    "RolloutBatchSpec": ("raas_marl.mappo_lagrangian.buffer", "RolloutBatchSpec"),
    "RolloutCollectionConfig": (
        "raas_marl.mappo_lagrangian.rollout",
        "RolloutCollectionConfig",
    ),
    "STAGE22_RESULT_FILES": (
        "raas_marl.mappo_lagrangian.development_runner",
        "STAGE22_RESULT_FILES",
    ),
    "STAGE22_RESULT_PARENT": (
        "raas_marl.mappo_lagrangian.development_runner",
        "STAGE22_RESULT_PARENT",
    ),
    "STAGE22_RESULT_PREFIX": (
        "raas_marl.mappo_lagrangian.development_runner",
        "STAGE22_RESULT_PREFIX",
    ),
    "STAGE23B_RESULT_FILES": (
        "raas_marl.mappo_lagrangian.stage23b_runner",
        "STAGE23B_RESULT_FILES",
    ),
    "STAGE23B_RESULT_PARENT": (
        "raas_marl.mappo_lagrangian.stage23b_runner",
        "STAGE23B_RESULT_PARENT",
    ),
    "STAGE23B_RESULT_PREFIX": (
        "raas_marl.mappo_lagrangian.stage23b_runner",
        "STAGE23B_RESULT_PREFIX",
    ),
    "Stage22DevelopmentEnvironment": (
        "raas_marl.mappo_lagrangian.env_adapter",
        "Stage22DevelopmentEnvironment",
    ),
    "Stage22UpdateConfig": (
        "raas_marl.mappo_lagrangian.update",
        "Stage22UpdateConfig",
    ),
    "Stage22UpdateResult": (
        "raas_marl.mappo_lagrangian.update",
        "Stage22UpdateResult",
    ),
    "Stage23BCollectedRollout": (
        "raas_marl.mappo_lagrangian.stage23b_rollout",
        "Stage23BCollectedRollout",
    ),
    "Stage23BRolloutConfig": (
        "raas_marl.mappo_lagrangian.stage23b_rollout",
        "Stage23BRolloutConfig",
    ),
    "Stage24AUpdateConfig": (
        "raas_marl.mappo_lagrangian.stage24_collector",
        "Stage24AUpdateConfig",
    ),
    "Stage24CollectedBatch": (
        "raas_marl.mappo_lagrangian.stage24_collector",
        "Stage24CollectedBatch",
    ),
    "Stage24CollectorConfig": (
        "raas_marl.mappo_lagrangian.stage24_collector",
        "Stage24CollectorConfig",
    ),
    "bootstrap_allowed_mask": (
        "raas_marl.mappo_lagrangian.losses",
        "bootstrap_allowed_mask",
    ),
    "collect_development_rollout": (
        "raas_marl.mappo_lagrangian.rollout",
        "collect_development_rollout",
    ),
    "collect_stage23b_development_rollout": (
        "raas_marl.mappo_lagrangian.stage23b_rollout",
        "collect_stage23b_development_rollout",
    ),
    "collect_stage24_stage23a_training_batch": (
        "raas_marl.mappo_lagrangian.stage24_collector",
        "collect_stage24_stage23a_training_batch",
    ),
    "compute_cost_gae": ("raas_marl.mappo_lagrangian.losses", "compute_cost_gae"),
    "compute_gae": ("raas_marl.mappo_lagrangian.losses", "compute_gae"),
    "compute_reward_gae": (
        "raas_marl.mappo_lagrangian.losses",
        "compute_reward_gae",
    ),
    "default_stage22_core_config": (
        "raas_marl.mappo_lagrangian.env_adapter",
        "default_stage22_core_config",
    ),
    "default_stage22_update_config": (
        "raas_marl.mappo_lagrangian.update",
        "default_stage22_update_config",
    ),
    "default_stage24_update_config": (
        "raas_marl.mappo_lagrangian.stage24_collector",
        "default_stage24_update_config",
    ),
    "default_stage24a_update_config": (
        "raas_marl.mappo_lagrangian.stage24_collector",
        "default_stage24a_update_config",
    ),
    "deterministic_json_bytes": (
        "raas_marl.mappo_lagrangian.artifacts",
        "deterministic_json_bytes",
    ),
    "deterministic_json_string": (
        "raas_marl.mappo_lagrangian.artifacts",
        "deterministic_json_string",
    ),
    "entropy_bonus": ("raas_marl.mappo_lagrangian.losses", "entropy_bonus"),
    "forbidden_actor_information_keys": (
        "raas_marl.mappo_lagrangian.config",
        "forbidden_actor_information_keys",
    ),
    "hazard_advantage_penalizes_policy_objective": (
        "raas_marl.mappo_lagrangian.update",
        "hazard_advantage_penalizes_policy_objective",
    ),
    "lagrangian_advantage": (
        "raas_marl.mappo_lagrangian.losses",
        "lagrangian_advantage",
    ),
    "lagrangian_policy_objective": (
        "raas_marl.mappo_lagrangian.losses",
        "lagrangian_policy_objective",
    ),
    "lagrangian_ppo_policy_loss": (
        "raas_marl.mappo_lagrangian.losses",
        "lagrangian_ppo_policy_loss",
    ),
    "make_default_development_environment": (
        "raas_marl.mappo_lagrangian.rollout",
        "make_default_development_environment",
    ),
    "make_default_stage23b_environment": (
        "raas_marl.mappo_lagrangian.stage23b_rollout",
        "make_default_stage23b_environment",
    ),
    "manifest_entry": ("raas_marl.mappo_lagrangian.artifacts", "manifest_entry"),
    "masked_mean": ("raas_marl.mappo_lagrangian.losses", "masked_mean"),
    "policy_from_logits": (
        "raas_marl.mappo_lagrangian.model",
        "policy_from_logits",
    ),
    "ppo_clipped_policy_surrogate": (
        "raas_marl.mappo_lagrangian.losses",
        "ppo_clipped_policy_surrogate",
    ),
    "ppo_clipped_surrogate": (
        "raas_marl.mappo_lagrangian.losses",
        "ppo_clipped_surrogate",
    ),
    "ppo_policy_loss": ("raas_marl.mappo_lagrangian.losses", "ppo_policy_loss"),
    "recursive_carry_mask": (
        "raas_marl.mappo_lagrangian.losses",
        "recursive_carry_mask",
    ),
    "run_stage22_development_training": (
        "raas_marl.mappo_lagrangian.development_runner",
        "run_stage22_development_training",
    ),
    "run_stage23b_bounded_training": (
        "raas_marl.mappo_lagrangian.stage23b_runner",
        "run_stage23b_bounded_training",
    ),
    "run_stage24_training_smoke": (
        "raas_marl.mappo_lagrangian.stage24_collector",
        "run_stage24_training_smoke",
    ),
    "run_stage24a_training_smoke": (
        "raas_marl.mappo_lagrangian.stage24_collector",
        "run_stage24a_training_smoke",
    ),
    "sequence_boundary_mask": (
        "raas_marl.mappo_lagrangian.losses",
        "sequence_boundary_mask",
    ),
    "sha256_file": ("raas_marl.mappo_lagrangian.artifacts", "sha256_file"),
    "stage22_ppo_lagrangian_update": (
        "raas_marl.mappo_lagrangian.update",
        "stage22_ppo_lagrangian_update",
    ),
    "stage24_objective_update_diagnosis": (
        "raas_marl.mappo_lagrangian.stage24_collector",
        "stage24_objective_update_diagnosis",
    ),
    "stage24a_ppo_lagrangian_readiness_update": (
        "raas_marl.mappo_lagrangian.stage24_collector",
        "stage24a_ppo_lagrangian_readiness_update",
    ),
    "stage24a_runtime_reproducibility_record": (
        "raas_marl.mappo_lagrangian.stage24_collector",
        "stage24a_runtime_reproducibility_record",
    ),
    "transitions_to_tensors": (
        "raas_marl.mappo_lagrangian.env_adapter",
        "transitions_to_tensors",
    ),
    "update_lagrange_multiplier": (
        "raas_marl.mappo_lagrangian.lagrange",
        "update_lagrange_multiplier",
    ),
    "validate_actor_input_mapping": (
        "raas_marl.mappo_lagrangian.config",
        "validate_actor_input_mapping",
    ),
    "validate_actor_tensors": (
        "raas_marl.mappo_lagrangian.config",
        "validate_actor_tensors",
    ),
    "validate_actor_visible_payload": (
        "raas_marl.mappo_lagrangian.env_contracts",
        "validate_actor_visible_payload",
    ),
    "validate_central_state": (
        "raas_marl.mappo_lagrangian.config",
        "validate_central_state",
    ),
    "validate_critic_central_payload": (
        "raas_marl.mappo_lagrangian.env_contracts",
        "validate_critic_central_payload",
    ),
    "validate_environment_step_contract": (
        "raas_marl.mappo_lagrangian.env_contracts",
        "validate_environment_step_contract",
    ),
    "validate_loss_input_mapping": (
        "raas_marl.mappo_lagrangian.losses",
        "validate_loss_input_mapping",
    ),
    "value_loss": ("raas_marl.mappo_lagrangian.losses", "value_loss"),
}

__all__ = sorted(_LAZY_EXPORTS)


def _is_torch_import_failure(exc: ModuleNotFoundError) -> bool:
    """Return True when ``exc`` is the absence of torch or a torch submodule.

    The exact ``exc.name == "torch"`` predicate misses failures raised while
    importing a ``torch.*`` submodule (issue NEW-__init__-4), so both the top
    package and any dotted submodule are treated as a torch-absence signal.
    """

    name = exc.name or ""
    return name == "torch" or name.startswith("torch.")


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = _LAZY_EXPORTS[name]
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        if _is_torch_import_failure(exc):
            # Match the canonical config.torch_required wording so the
            # torch-absence message is identical across the package
            # (issue NEW-__init__-3).
            raise ModuleNotFoundError(
                f"PyTorch is required for accessing {name} from {__name__}",
                name="torch",
            ) from exc
        raise
    value = getattr(module, attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    # Surface the lazily-exported names to dir()/tab-completion alongside any
    # already-resolved globals (issue NEW-__init__-5).
    return sorted(set(globals()) | set(_LAZY_EXPORTS))

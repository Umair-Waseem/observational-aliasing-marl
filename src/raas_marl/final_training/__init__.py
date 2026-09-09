"""Stage 25 final-training package: harness, reviewed update path, drivers.

This package contains two layers. (a) The Stage 25 run-harness
infrastructure: long-horizon training drivers over Stage 23-C multi-episode
collection with run manifests, incremental JSONL curves, per-episode logs,
and scripted-comparator readiness probes; the baseline driver loops the
UNTOUCHED Stage 22 bounded update. (b) The Stage 25 REVIEWED final-training
update path authorized by the countersigned Decision Records — DR-D4
(full-batch KL-guarded 10-epoch PPO-Lagrangian), DR-D2 (episodic team
constraint estimator), DR-D3 (PI dual controller), DR-D14 (separate pre-mix
standardization), DR-D9 (k3 KL early stopping) — driven by ``stage25_driver``
with periodic sha256-recorded training-state saves and verified resume per
the DR-D4 cadence. State saves are authorized by the Phase-6 final-training
envelope; the dev-stage governed result roots and the bounded development
update path (stages <= 23-C) are untouched.

This package does not execute the project's locked final-assessment
protocol, creates no evidence artifacts for the research claim, produces no
paper-oriented output, runs no baseline comparison as evidence, no ablation,
and no statistical test, and makes no Bayesian-belief or formal-VOI claim.
Every persisted record carries truthful Stage 25 boundary flags (the
state-save flags flip True only once a saved training state actually
exists).
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

# Entries are alphabetized by export name so this literal matches the sorted
# __all__ derived from it (project convention #6); each value is
# (fully_qualified_module_name, attribute_name). ``run_logging`` exports are
# torch-free; ``baseline_driver`` / ``stage25_*`` exports import torch eagerly
# on first access.
_LAZY_EXPORTS = {
    "BaselineRunConfig": (
        "raas_marl.final_training.baseline_driver",
        "BaselineRunConfig",
    ),
    "PIController": (
        "raas_marl.final_training.stage25_update",
        "PIController",
    ),
    "PIControllerStep": (
        "raas_marl.final_training.stage25_update",
        "PIControllerStep",
    ),
    "Stage25CollectedRollout": (
        "raas_marl.final_training.stage25_collection",
        "Stage25CollectedRollout",
    ),
    "Stage25CollectionConfig": (
        "raas_marl.final_training.stage25_collection",
        "Stage25CollectionConfig",
    ),
    "Stage25RunConfig": (
        "raas_marl.final_training.stage25_driver",
        "Stage25RunConfig",
    ),
    "Stage25UpdateConfig": (
        "raas_marl.final_training.stage25_update",
        "Stage25UpdateConfig",
    ),
    "Stage25UpdateResult": (
        "raas_marl.final_training.stage25_update",
        "Stage25UpdateResult",
    ),
    "append_jsonl_record": (
        "raas_marl.final_training.run_logging",
        "append_jsonl_record",
    ),
    "build_run_manifest_payload": (
        "raas_marl.final_training.run_logging",
        "build_run_manifest_payload",
    ),
    "collect_stage25_training_rollout": (
        "raas_marl.final_training.stage25_collection",
        "collect_stage25_training_rollout",
    ),
    "episodic_team_hazard_cost": (
        "raas_marl.final_training.stage25_update",
        "episodic_team_hazard_cost",
    ),
    "experiments_root": (
        "raas_marl.final_training.run_logging",
        "experiments_root",
    ),
    "explained_variance_from_batch": (
        "raas_marl.final_training.baseline_driver",
        "explained_variance_from_batch",
    ),
    "greedy_model_policy_fn": (
        "raas_marl.final_training.baseline_driver",
        "greedy_model_policy_fn",
    ),
    "harness_boundary_flags": (
        "raas_marl.final_training.run_logging",
        "harness_boundary_flags",
    ),
    "run_baseline_training": (
        "raas_marl.final_training.baseline_driver",
        "run_baseline_training",
    ),
    "run_readiness_probe": (
        "raas_marl.final_training.baseline_driver",
        "run_readiness_probe",
    ),
    "run_stage25_training": (
        "raas_marl.final_training.stage25_driver",
        "run_stage25_training",
    ),
    "stage25_boundary_flags": (
        "raas_marl.final_training.stage25_driver",
        "stage25_boundary_flags",
    ),
    "stage25_ppo_lagrangian_update": (
        "raas_marl.final_training.stage25_update",
        "stage25_ppo_lagrangian_update",
    ),
    "validate_run_id": (
        "raas_marl.final_training.run_logging",
        "validate_run_id",
    ),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

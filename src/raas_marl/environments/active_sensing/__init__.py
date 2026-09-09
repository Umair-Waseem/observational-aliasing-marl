"""Active-sensing grid environment package."""

from __future__ import annotations

from raas_marl.environments.active_sensing.construction_runner import (
    run_environment_construction,
    run_stage23a_environment_construction,
)
from raas_marl.environments.active_sensing.grid_environment import (
    RiskAwareActiveSensingGridEnvironment,
    Stage23EnvironmentConfig,
    Stage23Scenario,
    validate_stage23_actor_observation,
)
from raas_marl.environments.active_sensing.scenarios import (
    available_scenarios,
    make_scenario,
    stage23_scenario_catalog,
)

_TENSOR_ADAPTER_EXPORTS = frozenset(
    {
        "actor_observation_from_stage23",
        "central_state_from_stage23",
        "default_stage23_core_config",
        "revealed_information_from_stage23",
        "stage23_transition_to_environment_step_contract",
    }
)
# Stage 24 diagnostics exports are lazy for the same reason the tensor-adapter
# exports are: importing that module eagerly is unnecessary for the common
# environment-construction path. This set covers stage24_diagnostics' public
# symbols (two config/variant dataclasses, the five scripted comparator
# policies, the diagnostic entry points, and the fork registry accessors —
# fork_curriculum_scenarios registered per DR-READINESS-CURRICULUM AM-C7, the
# m-1 convention), so the package public surface stays consistent with the
# module it re-exports from.
_STAGE24_DIAGNOSTICS_EXPORTS = frozenset(
    {
        "Stage24AHazardLayoutVariant",
        "Stage24ComparatorConfig",
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
    }
)

# Each lazy export set is paired with the submodule that defines its symbols.
# __getattr__ walks this table once, so a new lazily-exported submodule is added
# by extending the tuple rather than by copying the import-and-cache block.
_LAZY_EXPORT_MODULES = (
    (_TENSOR_ADAPTER_EXPORTS, "raas_marl.environments.active_sensing.tensor_adapter"),
    (_STAGE24_DIAGNOSTICS_EXPORTS, "raas_marl.environments.active_sensing.stage24_diagnostics"),
)


def __getattr__(name: str) -> object:
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    for export_set, module_name in _LAZY_EXPORT_MODULES:
        if name in export_set:
            from importlib import import_module

            module = import_module(module_name)
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "RiskAwareActiveSensingGridEnvironment",
    "Stage23EnvironmentConfig",
    "Stage23Scenario",
    "Stage24AHazardLayoutVariant",
    "Stage24ComparatorConfig",
    "actor_observation_from_stage23",
    "always_sense_shortest_path",
    "available_scenarios",
    "central_state_from_stage23",
    "default_stage23_core_config",
    "fork_curriculum_scenarios",
    "fork_hazard_layout_families",
    "make_scenario",
    "no_sense_shortest_path",
    "random_policy",
    "revealed_information_from_stage23",
    "risk_aware_oracle_or_heuristic",
    "run_environment_construction",
    "run_stage23_scenario_diagnostics",
    "run_stage23a_environment_construction",
    "run_stage24a_variant_readiness_diagnostics",
    "selective_sense_risk_aware",
    "stage23_scenario_catalog",
    "stage23_transition_to_environment_step_contract",
    "stage24a_hazard_layout_variants",
    "validate_stage23_actor_observation",
    "verify_fork_family_forces_sensing",
]

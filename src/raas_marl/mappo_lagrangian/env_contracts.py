"""Environment-step payload contracts for Stage 21 core integration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from raas_marl.mappo_lagrangian._validation import (
    finite_numeric_scalar,
    require_int_not_bool,
    require_mapping,
    require_nonnegative_int,
)
from raas_marl.mappo_lagrangian.config import (
    MAPPOCoreConfig,
    forbidden_actor_information_keys,
    require_finite_float_tensor_matching_config,
    torch_required,
    validate_actor_input_mapping,
)


@dataclass(frozen=True)
class EnvironmentStepContract:
    """Single-step payload schema used by tests and future adapters."""

    actor_visible: Mapping[str, object]
    critic_visible: Mapping[str, object]
    rewards_and_costs: Mapping[str, object]
    done_flags: Mapping[str, object]
    action_factors: Mapping[str, object]
    identity: Mapping[str, object]


# Compatibility alias for the validated Stage 21 environment-step dataclass.
# It does not define separate Stage 22 adapter behavior.
EnvironmentContract = EnvironmentStepContract


def validate_environment_step_contract(
    contract: EnvironmentStepContract,
    config: MAPPOCoreConfig,
) -> None:
    if not isinstance(contract, EnvironmentStepContract):
        raise TypeError("contract must be an EnvironmentStepContract")
    if not isinstance(config, MAPPOCoreConfig):
        raise TypeError("config must be a MAPPOCoreConfig")
    validate_actor_visible_payload(contract.actor_visible, config)
    validate_critic_central_payload(contract.critic_visible, config)
    _validate_rewards_and_costs(contract.rewards_and_costs)
    _validate_done_flags(contract.done_flags)
    _validate_action_factor_payload(contract.action_factors)
    _validate_identity(contract.identity, config)


def validate_actor_visible_payload(
    payload: Mapping[str, object],
    config: MAPPOCoreConfig,
) -> None:
    if not isinstance(config, MAPPOCoreConfig):
        raise TypeError("config must be a MAPPOCoreConfig")
    require_mapping("actor_visible payload", payload)
    validate_actor_input_mapping(payload)
    if "history_state" in payload:
        raise ValueError(
            "history_state is model recurrent state, not an environment "
            "actor-visible payload field"
        )
    _require_key(payload, "actor_observation")
    _require_key(payload, "revealed_information")
    _require_float_tensor(
        "actor_observation",
        payload["actor_observation"],
        (config.actor_observation_dim,),
        config,
    )
    _require_float_tensor(
        "revealed_information",
        payload["revealed_information"],
        (config.revealed_information_dim,),
        config,
    )
    # Extra revealed_local_* fields (already grammar/provenance-checked by
    # validate_actor_input_mapping) go through the same dtype+device+finite
    # coherence path as the named tensors (NEW-env_contracts-5).
    for key, value in payload.items():
        if key not in {"actor_observation", "revealed_information"}:
            _require_float_vector(key, value, config)


def validate_critic_central_payload(
    payload: Mapping[str, object],
    config: MAPPOCoreConfig,
) -> None:
    if not isinstance(config, MAPPOCoreConfig):
        raise TypeError("config must be a MAPPOCoreConfig")
    require_mapping("critic payload", payload)
    # repr() the offending keys (parity with validate_actor_input_mapping's key
    # handling) so non-string keys are unambiguous; if any extra key is a known
    # CTDE actor-information alias, name the leak explicitly (NEW-env_contracts-9).
    extra = [key for key in payload if key != "central_state"]
    if extra:
        offenders = ", ".join(sorted(repr(key) for key in extra))
        forbidden = forbidden_actor_information_keys()
        ctde_aliases = sorted(
            repr(key) for key in extra if isinstance(key, str) and key in forbidden
        )
        message = "critic payload contains fields outside central_state: " + offenders
        if ctde_aliases:
            message += "; CTDE-forbidden actor aliases present: " + ", ".join(
                ctde_aliases
            )
        raise ValueError(message)
    _require_key(payload, "central_state")
    _require_float_tensor(
        "central_state", payload["central_state"], (config.central_state_dim,), config
    )


def _validate_rewards_and_costs(payload: Mapping[str, object]) -> None:
    require_mapping("rewards_and_costs payload", payload)
    for key in ("task_reward", "hazard_cost", "sensing_cost"):
        _require_key(payload, key)
        # finite_numeric_scalar performs the full type-then-value check
        # (TypeError for non-numeric, ValueError for non-finite); the former
        # is_numeric_scalar pre-check only preserved now-archived pinned wording
        # and is redundant (NEW-env_contracts-7).
        numeric = finite_numeric_scalar(key, payload[key])
        if key.endswith("_cost") and numeric < 0:
            raise ValueError(f"{key} must be nonnegative")
    # Open-schema: keys beyond the three required signals are deliberately
    # tolerated unvalidated. rewards_and_costs is NOT a CTDE information boundary
    # (only actor_visible and critic_visible carry closed schemas), so extra keys
    # cannot leak privileged state to the actor (NEW-env_contracts-6).


def _validate_done_flags(payload: Mapping[str, object]) -> None:
    require_mapping("done_flags payload", payload)
    # Only "truncated" carries a legacy-alias guard: "truncation" was the
    # historical public-contract spelling, so a producer emitting it (and not
    # the canonical "truncated") is a contract error. "terminal" never had a
    # competing spelling, so no analogous guard is needed (NEW-env_contracts-8).
    if "truncation" in payload and "truncated" not in payload:
        raise ValueError(
            "Stage 21 public environment contract requires truncated, not truncation"
        )
    if "truncated" in payload and "truncation" in payload:
        raise ValueError("done_flags cannot contain both truncated and truncation")
    for key in ("terminal", "truncated"):
        _require_key(payload, key)
        if not isinstance(payload[key], bool):
            raise TypeError(f"{key} must be a bool")
    if payload["terminal"] and payload["truncated"]:
        raise ValueError("terminal and truncated cannot both be true")
    # Open-schema: extra keys are tolerated; done_flags is not a CTDE boundary
    # (NEW-env_contracts-6).


def _validate_action_factor_payload(payload: Mapping[str, object]) -> None:
    require_mapping("action_factors payload", payload)
    for key in ("sensing_action_available", "movement_action_available"):
        _require_key(payload, key)
        if not isinstance(payload[key], bool):
            raise TypeError(f"{key} must be a bool")
        if not payload[key]:
            raise ValueError(f"{key} must be true for a valid selectable factor")
    # Open-schema: extra keys are tolerated; action_factors is not a CTDE
    # boundary (NEW-env_contracts-6).


def _validate_identity(payload: Mapping[str, object], config: MAPPOCoreConfig) -> None:
    # Extra keys are tolerated here: identity is not a CTDE information boundary
    # (only actor_visible/critic_visible are closed schemas), so an unknown key
    # cannot leak privileged state through this validator (NEW-env_contracts-6).
    require_mapping("identity payload", payload)
    # require_nonnegative_int normalizes through int.__index__ so a hostile int
    # subclass cannot evade the nonnegativity / range checks via overridden
    # comparison operators (NEW-env_contracts-2; S2-5 propagation).
    for key in ("team_id", "agent_id"):
        _require_key(payload, key)
        require_nonnegative_int(key, payload[key])
    agent_id = int.__index__(payload["agent_id"])
    if config.agent_id_count is not None and agent_id >= config.agent_id_count:
        raise ValueError("agent_id must be smaller than configured agent_id_count")
    structure_names = ("agent_order", "agent_id_mapping", "team_agent_ids")
    present = [name for name in structure_names if name in payload]
    if not present:
        raise ValueError(
            "identity payload requires agent_order, agent_id_mapping, or team_agent_ids"
        )
    ordered_structures: dict[str, tuple[int, ...]] = {}
    for name in present:
        value = payload[name]
        if name in {"agent_order", "team_agent_ids"}:
            ordered_structures[name] = _validate_agent_sequence(
                name,
                value,
                agent_id,
                config,
            )
        else:
            ordered_structures[name] = _validate_agent_id_mapping(
                value,
                agent_id,
                config,
            )
    reference_name, reference_order = next(iter(ordered_structures.items()))
    for name, order in ordered_structures.items():
        if order != reference_order:
            raise ValueError(
                f"identity payload has ambiguous {reference_name} and {name}"
            )


def _validate_agent_sequence(
    name: str,
    value: object,
    current_agent_id: int,
    config: MAPPOCoreConfig,
) -> tuple[int, ...]:
    if isinstance(value, (set, frozenset)) or not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple with stable ordering")
    if not value:
        raise ValueError(f"{name} must be nonempty")
    ids = tuple(_require_agent_id(name, agent_id, config) for agent_id in value)
    if len(set(ids)) != len(ids):
        raise ValueError(f"{name} must not contain duplicate agent identifiers")
    if current_agent_id not in ids:
        raise ValueError(f"{name} must include the current agent_id")
    _require_full_configured_team(name, ids, config)
    return ids


def _validate_agent_id_mapping(
    value: object,
    current_agent_id: int,
    config: MAPPOCoreConfig,
) -> tuple[int, ...]:
    if not isinstance(value, Mapping):
        raise TypeError("agent_id_mapping must be a mapping")
    if not value:
        raise ValueError("agent_id_mapping must be nonempty")
    # items() is snapshotted once so a hostile mapping cannot return different
    # pairs to the key/position validation and the order derivation below
    # (session 2026-07-04).
    pairs = tuple(value.items())
    keys = tuple(
        _require_agent_id("agent_id_mapping key", key, config) for key, _ in pairs
    )
    positions = tuple(
        _require_stable_position("agent_id_mapping value", position)
        for _, position in pairs
    )
    if len(set(keys)) != len(keys):
        raise ValueError("agent_id_mapping must not contain duplicate agent IDs")
    if len(set(positions)) != len(positions):
        raise ValueError("agent_id_mapping positions must be unique")
    if current_agent_id not in keys:
        raise ValueError("agent_id_mapping must include current agent_id as a key")
    if config.agent_id_count is not None:
        expected = set(range(config.agent_id_count))
        if set(keys) != expected:
            raise ValueError(
                "agent_id_mapping keys must cover the full configured team"
            )
        if set(positions) != expected:
            raise ValueError(
                "agent_id_mapping positions must cover the full configured team"
            )
    return tuple(
        agent_id for agent_id, _ in sorted(pairs, key=lambda item: item[1])
    )


def _require_agent_id(name: str, value: object, config: MAPPOCoreConfig) -> int:
    # Normalize through int.__index__ (via require_nonnegative_int) so a hostile
    # int subclass cannot lie through overridden comparison operators, and so the
    # returned identifier used for set/sort membership is the canonical int
    # (NEW-env_contracts-2; S2-5 propagation).
    require_int_not_bool(
        name, value, error_suffix="must contain integer agent identifiers"
    )
    identifier = int.__index__(value)
    if identifier < 0:
        raise ValueError(f"{name} must contain nonnegative agent identifiers")
    if config.agent_id_count is not None and identifier >= config.agent_id_count:
        raise ValueError(f"{name} agent identifier is outside configured range")
    return identifier


def _require_stable_position(name: str, value: object) -> int:
    require_int_not_bool(
        name, value, error_suffix="must contain integer stable positions"
    )
    position = int.__index__(value)
    if position < 0:
        raise ValueError(f"{name} must contain nonnegative stable positions")
    return position


def _require_full_configured_team(
    name: str,
    ids: tuple[int, ...],
    config: MAPPOCoreConfig,
) -> None:
    if config.agent_id_count is None:
        return
    expected = set(range(config.agent_id_count))
    if set(ids) != expected:
        raise ValueError(f"{name} must cover the full configured team")


def _require_key(payload: Mapping[str, object], key: str) -> None:
    if key not in payload:
        raise ValueError(f"missing required key: {key}")


def _require_float_tensor(
    name: str,
    value: object,
    shape: tuple[int, ...],
    config: MAPPOCoreConfig,
) -> None:
    torch = torch_required("environment contract tensor validation")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tuple(value.shape) != shape:
        raise ValueError(f"{name} shape mismatch")
    # Shared helper adds meta/sparse-layout rejection AND dtype+device coherence
    # against the config so a float16/float64/bfloat16 or off-device boundary
    # tensor fails deterministically (NEW-env_contracts-1/3; S-5 propagation).
    require_finite_float_tensor_matching_config(name, value, config)


def _require_float_vector(
    name: str,
    value: object,
    config: MAPPOCoreConfig,
) -> None:
    torch = torch_required("environment contract tensor validation")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [dim]")
    # Extra revealed_local_* fields get the same dtype+device+finite coherence
    # as the named payload tensors (NEW-env_contracts-5).
    require_finite_float_tensor_matching_config(name, value, config)

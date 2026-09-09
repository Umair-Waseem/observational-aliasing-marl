"""Segment 1 authoritative from-scratch test suite (assembled from 5 sections).

Run: python -m pytest -p no:cacheprovider tests/test_clean.py -q
"""

from __future__ import annotations

import math
import os
import pytest
import torch
from raas_marl.mappo_lagrangian import _validation as val
from raas_marl.mappo_lagrangian import artifacts
from raas_marl.mappo_lagrangian import config as cfgmod
from raas_marl.mappo_lagrangian.config import (
    AlgorithmConfig,
    LagrangeConfig,
    LossConfig,
    MAPPOCoreConfig,
    ModelConfig,
    forbidden_actor_information_keys,
    require_finite_float_tensor,
    require_finite_float_tensor_matching_config,
    require_torch_tensor,
    torch_dtype_from_name,
    torch_required,
    validate_actor_input_mapping,
    validate_actor_tensors,
    validate_central_state,
)
from raas_marl.mappo_lagrangian.lagrange import (
    LagrangeMultiplier,
    update_lagrange_multiplier,
)
from raas_marl.mappo_lagrangian import losses as lossmod
from raas_marl.mappo_lagrangian.losses import (
    bootstrap_allowed_mask,
    compute_cost_gae,
    compute_gae,
    compute_reward_gae,
    entropy_bonus,
    lagrangian_advantage,
    lagrangian_policy_objective,
    lagrangian_ppo_policy_loss,
    masked_mean,
    ppo_clipped_policy_surrogate,
    ppo_clipped_surrogate,
    ppo_policy_loss,
    recursive_carry_mask,
    sequence_boundary_mask,
    validate_loss_input_mapping,
    value_loss,
)
from raas_marl.mappo_lagrangian.buffer import RolloutBatch, RolloutBatchSpec
from raas_marl.mappo_lagrangian.env_contracts import (
    EnvironmentContract,
    EnvironmentStepContract,
    validate_actor_visible_payload,
    validate_critic_central_payload,
    validate_environment_step_contract,
)
from raas_marl.mappo_lagrangian.env_contracts import (
    _validate_action_factor_payload,
    _validate_done_flags,
    _validate_identity,
    _validate_rewards_and_costs,
)
import json
import shutil
from pathlib import Path
from raas_marl.mappo_lagrangian.config import (
    AlgorithmConfig,
    LagrangeConfig,
    LossConfig,
    MAPPOCoreConfig,
)
from raas_marl.mappo_lagrangian.lagrange import LagrangeMultiplier
from raas_marl.mappo_lagrangian.model import (
    ActorInput,
    CriticInput,
    FactorizedPolicyOutput,
    RecurrentMAPPOActorCritic,
    policy_from_logits,
)
from raas_marl.mappo_lagrangian.env_adapter import (
    DevelopmentAdapterConfig,
    DevelopmentTransition,
    Stage22DevelopmentEnvironment,
    default_stage22_core_config,
    transitions_to_tensors,
)
from raas_marl.mappo_lagrangian.rollout import (
    CollectedRollout,
    RolloutCollectionConfig,
    collect_development_rollout,
    make_default_development_environment,
    _validate_reset_payloads_for_rollout,
)
from raas_marl.mappo_lagrangian.update import (
    Stage22UpdateConfig,
    cost_estimate,
    default_stage22_update_config,
    hazard_advantage_penalizes_policy_objective,
    stage22_ppo_lagrangian_update,
    _development_reward_signal,
    _masked_mean_float,
    _normalise_advantage,
)
from raas_marl.mappo_lagrangian import development_runner as _gb_dr
import importlib
import subprocess
import sys
from raas_marl.environments.active_sensing.grid_environment import STAGE23_AGENT_NAMES
from raas_marl.environments.active_sensing.tensor_adapter import (
    default_stage23_core_config,
)
from raas_marl.mappo_lagrangian import _governance as gov
from raas_marl.mappo_lagrangian import _rollout_common as rc
from raas_marl.mappo_lagrangian.model import RecurrentMAPPOActorCritic
from raas_marl.mappo_lagrangian.stage23b_rollout import (
    Stage23BCollectedRollout,
    Stage23BRolloutConfig,
    collect_stage23b_development_rollout,
    make_default_stage23b_environment,
)
from raas_marl.mappo_lagrangian.stage23b_runner import (
    STAGE23B_RESULT_FILES,
    STAGE23B_RESULT_PREFIX,
    run_stage23b_bounded_training,
)
from raas_marl.mappo_lagrangian.stage24_collector import (
    Stage24AUpdateConfig,
    Stage24CollectedBatch,
    Stage24CollectorConfig,
    collect_stage24_stage23a_training_batch,
    default_stage24_update_config,
    default_stage24a_update_config,
    run_stage24_training_smoke,
    run_stage24a_training_smoke,
    stage24_objective_update_diagnosis,
    stage24a_ppo_lagrangian_readiness_update,
    stage24a_runtime_reproducibility_record,
)
from raas_marl.mappo_lagrangian.update import (
    Stage22UpdateConfig,
    default_stage22_update_config,
)
import copy
from raas_marl.environments.active_sensing.grid_environment import (
    HAZARD_COST_KEY,
    MOVEMENT_ACTION_FIELD,
    SENSING_ACTION_FIELD,
    SENSING_COST_KEY,
    STAGE23_ACTOR_VISIBLE_SCHEMA_KEYS,
    STAGE23_AGENT_NAMES,
    STAGE23_FORBIDDEN_ACTOR_KEYS,
    STAGE23_MOVEMENT_ACTIONS,
    STAGE23_MOVEMENT_DELTAS,
    STAGE23_SENSE_ACTION_INDEX,
    STAGE23_STAGE,
    TASK_REWARD_KEY,
    RiskAwareActiveSensingGridEnvironment,
    Stage23EnvironmentConfig,
    Stage23Scenario,
    stage23_scenario_catalog,
    validate_stage23_actor_observation,
)
from raas_marl.environments.active_sensing.scenarios import (
    available_scenarios,
    make_scenario,
)
from raas_marl.environments.active_sensing.tensor_adapter import (
    STAGE23_ACTOR_OBSERVATION_DIM,
    STAGE23_CENTRAL_STATE_DIM,
    STAGE23_CENTRAL_STATE_FEATURE_SCHEMA,
    STAGE23_REVEALED_INFORMATION_DIM,
    actor_observation_from_stage23,
    central_state_from_stage23,
    default_stage23_core_config,
    revealed_information_from_stage23,
    stage23_transition_to_environment_step_contract,
)
from raas_marl.environments.active_sensing.construction_runner import (
    ENVIRONMENT_CONSTRUCTION_RESULT_PREFIX,
    STAGE23A_RESULT_FILES,
    _active_root as _gD_construction_active_root,
    _assert_strict_stage23a_result_artifacts,
    _boundary_flags as _gD_construction_boundary_flags,
    run_environment_construction,
    run_stage23a_environment_construction,
)
from raas_marl.environments.active_sensing.stage24_diagnostics import (
    Stage24AHazardLayoutVariant,
    Stage24ComparatorConfig,
    _PUBLIC_GATE_RISK_ZONE_CELLS,
    _movement_action,
    _public_risk_zone_reveals_exact_hidden_hazard,
    _shortest_path,
    always_sense_shortest_path,
    fork_curriculum_scenarios,
    fork_hazard_layout_families,
    no_sense_shortest_path,
    random_policy,
    risk_aware_oracle_or_heuristic,
    run_stage23_scenario_diagnostics,
    run_stage24a_variant_readiness_diagnostics,
    selective_sense_risk_aware,
    stage24a_hazard_layout_variants,
    verify_fork_family_forces_sensing,
)
from raas_marl.mappo_lagrangian.config import MAPPOCoreConfig
from raas_marl.environments.active_sensing.scenarios import available_scenarios
from raas_marl.mappo_lagrangian._rollout_common import per_step_sampling_seed
from raas_marl.mappo_lagrangian.buffer import RolloutBatch
from raas_marl.mappo_lagrangian.stage23c_rollout import (
    Stage23CCollectedRollout,
    Stage23CRolloutConfig,
    collect_stage23c_development_rollout,
    make_default_stage23c_environment,
)
from raas_marl.mappo_lagrangian.stage23c_runner import (
    STAGE23C_RESULT_FILES,
    STAGE23C_RESULT_PARENT,
    STAGE23C_RESULT_PREFIX,
    _active_root,
    _assert_strict_stage23c_result_artifacts,
    _boundary_flags,
    _FALSE_BOUNDARY_KEYS,
    _TRUE_BOUNDARY_KEYS,
    run_stage23c_bounded_training,
)


# ===================== SECTION A =====================

_GA_SEED = 12345

def _ga_core_config(**overrides):
    """Return a valid MAPPOCoreConfig (dims 4/2/7) with optional overrides."""

    params = dict(
        actor_observation_dim=4,
        revealed_information_dim=2,
        central_state_dim=7,
        history_state_dim=8,
        actor_hidden_dim=16,
        critic_hidden_dim=16,
        sensing_action_count=2,
        movement_action_count=5,
        recurrent_layer_count=1,
    )
    params.update(overrides)
    return MAPPOCoreConfig(**params)

def _ga_reference_gae(signals, values, next_values, terminal, truncation, mask, gamma, lam):
    """Hand-written reference GAE recursion (independent of source implementation)."""

    batch, time = signals.shape
    out = torch.zeros_like(signals)
    for b in range(batch):
        carry = 0.0
        for t in range(time - 1, -1, -1):
            boot = 0.0 if bool(terminal[b, t]) else 1.0
            delta = float(signals[b, t]) + gamma * boot * float(next_values[b, t]) - float(values[b, t])
            carry_allowed = 0.0
            if t < time - 1:
                boundary = bool(terminal[b, t]) or bool(truncation[b, t])
                if bool(mask[b, t]) and bool(mask[b, t + 1]) and not boundary:
                    carry_allowed = 1.0
            step_adv = delta + gamma * lam * carry_allowed * carry
            if bool(mask[b, t]):
                out[b, t] = step_adv
                carry = step_adv
            else:
                out[b, t] = 0.0
                carry = 0.0
    return out

def _ga_batch_dict(batch, time, **overrides):
    """Return a full valid RolloutBatch field dict (float32)."""

    data = dict(
        actor_observation=torch.zeros(batch, time, 4),
        revealed_information=torch.zeros(batch, time, 2),
        central_state=torch.zeros(batch, time, 7),
        sensing_action=torch.zeros(batch, time, dtype=torch.long),
        movement_action=torch.zeros(batch, time, dtype=torch.long),
        old_sensing_log_probability=torch.zeros(batch, time),
        old_movement_log_probability=torch.zeros(batch, time),
        task_reward=torch.zeros(batch, time),
        hazard_cost=torch.zeros(batch, time),
        sensing_cost=torch.zeros(batch, time),
        reward_value=torch.zeros(batch, time),
        hazard_cost_value=torch.zeros(batch, time),
        next_reward_value=torch.zeros(batch, time),
        next_hazard_cost_value=torch.zeros(batch, time),
        terminal=torch.zeros(batch, time, dtype=torch.bool),
        truncation=torch.zeros(batch, time, dtype=torch.bool),
        valid_mask=torch.ones(batch, time, dtype=torch.bool),
    )
    data.update(overrides)
    return data

def _ga_rollout_batch(batch, time, **overrides):
    return RolloutBatch(**_ga_batch_dict(batch, time, **overrides))

def _ga_actor_payload():
    return {
        "actor_observation": torch.zeros(4),
        "revealed_information": torch.zeros(2),
    }

class _GaEvilInt(int):
    """int subclass with hostile comparison operators (S-7/S2-5 hardening probe)."""

    def __le__(self, other):  # pragma: no cover - should never decide validation
        return False

    def __lt__(self, other):  # pragma: no cover
        return False

    def __ge__(self, other):  # pragma: no cover
        return True

    def __gt__(self, other):  # pragma: no cover
        return True

    def __eq__(self, other):  # pragma: no cover
        return False

    def __hash__(self):
        return int.__hash__(self)

class _GaEvilFloat(float):
    """float subclass whose ``__float__`` lies with a non-finite value."""

    def __float__(self):
        return float("nan")

def test_validation_finite_numeric_scalar_accepts_int_and_float():
    assert val.finite_numeric_scalar("x", 3) == 3.0
    assert isinstance(val.finite_numeric_scalar("x", 3), float)
    assert val.finite_numeric_scalar("x", 2.5) == 2.5
    assert val.finite_numeric_scalar("x", 0) == 0.0
    assert val.finite_numeric_scalar("x", -0.0) == 0.0

def test_validation_finite_numeric_scalar_rejects_bool():
    with pytest.raises(TypeError, match="numeric"):
        val.finite_numeric_scalar("x", True)

@pytest.mark.parametrize("bad", ["str", None, complex(1, 2)])
def test_validation_finite_numeric_scalar_rejects_non_numeric_types(bad):
    with pytest.raises(TypeError, match="numeric"):
        val.finite_numeric_scalar("x", bad)

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_validation_finite_numeric_scalar_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="must be finite"):
        val.finite_numeric_scalar("x", bad)

@pytest.mark.parametrize(
    "huge",
    [10**400, -(10**400), 10**10000],
    ids=["pos_10e400", "neg_10e400", "pos_10e10000"],
)
def test_validation_finite_numeric_scalar_normalises_huge_int_overflow(huge):
    with pytest.raises(ValueError, match="must be finite"):
        val.finite_numeric_scalar("x", huge)

def test_validation_finite_numeric_scalar_ignores_hostile_float_subclass():
    # __float__ override is bypassed via float.__float__ slot; base value stands.
    assert val.finite_numeric_scalar("x", _GaEvilFloat(3.0)) == 3.0

def test_validation_is_numeric_scalar_matrix():
    assert val.is_numeric_scalar(3) is True
    assert val.is_numeric_scalar(3.5) is True
    assert val.is_numeric_scalar(True) is False
    assert val.is_numeric_scalar("x") is False
    assert val.is_numeric_scalar(None) is False

def test_validation_require_int_not_bool_accepts_int():
    assert val.require_int_not_bool("x", 5) is None

@pytest.mark.parametrize("bad", [True, 1.0, "1", None])
def test_validation_require_int_not_bool_rejects_non_int(bad):
    with pytest.raises(TypeError, match="must be an int"):
        val.require_int_not_bool("x", bad)

def test_validation_require_int_not_bool_error_suffix_override():
    with pytest.raises(TypeError, match="must be an integer"):
        val.require_int_not_bool("x", True, error_suffix="must be an integer")

def test_validation_require_positive_int_happy():
    assert val.require_positive_int("x", 1) is None

@pytest.mark.parametrize("bad", [0, -1, -5])
def test_validation_require_positive_int_rejects_nonpositive(bad):
    with pytest.raises(ValueError, match="positive integer"):
        val.require_positive_int("x", bad)

def test_validation_require_positive_int_rejects_bool():
    with pytest.raises(TypeError):
        val.require_positive_int("x", True)

def test_validation_require_positive_int_hostile_subclass_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        val.require_positive_int("x", _GaEvilInt(-5))

def test_validation_require_positive_int_error_suffix_override():
    with pytest.raises(ValueError, match="must be positive"):
        val.require_positive_int("x", 0, error_suffix="must be positive")

def test_validation_require_nonnegative_int_happy():
    assert val.require_nonnegative_int("x", 0) is None
    assert val.require_nonnegative_int("x", 7) is None

def test_validation_require_nonnegative_int_rejects_negative():
    with pytest.raises(ValueError, match="nonnegative"):
        val.require_nonnegative_int("x", -1)

def test_validation_require_nonnegative_int_hostile_subclass_rejected():
    with pytest.raises(ValueError):
        val.require_nonnegative_int("x", _GaEvilInt(-1))

def test_validation_require_minimum_int_happy():
    assert val.require_minimum_int("x", 2, 2) is None
    assert val.require_minimum_int("x", 5, 2) is None

def test_validation_require_minimum_int_below_minimum():
    with pytest.raises(ValueError, match=">= 2"):
        val.require_minimum_int("x", 1, 2)

def test_validation_require_minimum_int_bad_minimum_type():
    with pytest.raises(TypeError):
        val.require_minimum_int("x", 5, True)

def test_validation_require_probability_bounds():
    assert val.require_probability("p", 0.0) == 0.0
    assert val.require_probability("p", 1.0) == 1.0
    assert val.require_probability("p", 0.5) == 0.5

@pytest.mark.parametrize("bad", [-0.01, 1.01, 2.0])
def test_validation_require_probability_out_of_range(bad):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        val.require_probability("p", bad)

def test_validation_require_probability_error_suffix_override():
    with pytest.raises(ValueError, match="custom"):
        val.require_probability("p", 2.0, error_suffix="custom")

def test_validation_require_open_unit_interval_bounds():
    assert val.require_open_unit_interval("c", 0.5) == 0.5

@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.1])
def test_validation_require_open_unit_interval_rejects_endpoints(bad):
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        val.require_open_unit_interval("c", bad)

def test_validation_require_nonnegative_number_happy():
    assert val.require_nonnegative_number("x", 0.0) == 0.0
    assert val.require_nonnegative_number("x", 3.0) == 3.0

def test_validation_require_nonnegative_number_rejects_negative():
    with pytest.raises(ValueError, match="nonnegative"):
        val.require_nonnegative_number("x", -0.5)

def test_validation_require_positive_number_happy():
    assert val.require_positive_number("x", 0.001) == 0.001

@pytest.mark.parametrize("bad", [0.0, -1.0])
def test_validation_require_positive_number_rejects_nonpositive(bad):
    with pytest.raises(ValueError, match="positive"):
        val.require_positive_number("x", bad)

def test_validation_require_positive_seed_happy():
    assert val.require_positive_seed("seed", 1) is None
    assert val.require_positive_seed("seed", 999) is None

def test_validation_require_positive_seed_zero_reserved():
    with pytest.raises(ValueError, match="reserved"):
        val.require_positive_seed("seed", 0)

def test_validation_require_positive_seed_zero_custom_message():
    with pytest.raises(ValueError, match="stage-specific"):
        val.require_positive_seed("seed", 0, zero_seed_message="stage-specific message")

def test_validation_require_positive_seed_negative_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        val.require_positive_seed("seed", -3)

def test_validation_require_positive_seed_bool_rejected():
    with pytest.raises(TypeError):
        val.require_positive_seed("seed", True)

def test_validation_require_bounded_seed_returns_normalized_int():
    assert val.require_bounded_seed("seed", 42) == 42
    assert isinstance(val.require_bounded_seed("seed", 42), int)

def test_validation_require_bounded_seed_with_offset():
    assert val.require_bounded_seed("seed", 5, max_offset=10) == 5

def test_validation_require_bounded_seed_overflow_rejected():
    with pytest.raises(ValueError, match="signed 64-bit"):
        val.require_bounded_seed("seed", 2**63)

def test_validation_require_bounded_seed_overflow_via_offset():
    with pytest.raises(ValueError, match="signed 64-bit"):
        val.require_bounded_seed("seed", 2**63 - 1, max_offset=1)

def test_validation_require_bounded_seed_custom_error_suffix():
    with pytest.raises(ValueError, match="my-suffix"):
        val.require_bounded_seed("seed", 2**63, error_suffix="my-suffix")

def test_validation_require_mapping_happy():
    d = {"a": 1}
    assert val.require_mapping("m", d) is d

def test_validation_require_mapping_rejects_non_mapping():
    with pytest.raises(TypeError, match="must be a mapping"):
        val.require_mapping("m", [1, 2])

def test_validation_require_string_mapping_keys_happy():
    assert val.require_string_mapping_keys("m", {"a": 1, "b": 2}) is None

def test_validation_require_string_mapping_keys_rejects_non_string():
    with pytest.raises(TypeError, match="offending keys"):
        val.require_string_mapping_keys("m", {1: "a", "b": 2})

def test_validation_nonempty_unpadded_string_happy():
    assert val.nonempty_unpadded_string("s", "abc") == "abc"

def test_validation_nonempty_unpadded_string_rejects_non_string():
    with pytest.raises(TypeError, match="must be a string"):
        val.nonempty_unpadded_string("s", 5)

def test_validation_nonempty_unpadded_string_rejects_empty():
    with pytest.raises(ValueError, match="must not be empty"):
        val.nonempty_unpadded_string("s", "")

@pytest.mark.parametrize("bad", [" x", "x ", " x "])
def test_validation_nonempty_unpadded_string_rejects_padded(bad):
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        val.nonempty_unpadded_string("s", bad)

def test_artifacts_sha256_file_known_bytes(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(b"hello")
    # sha256("hello") is a well-known constant.
    assert artifacts.sha256_file(p) == (
        "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
    )

def test_artifacts_sha256_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        artifacts.sha256_file(tmp_path / "nope.bin")

def test_artifacts_deterministic_json_string_sorts_keys():
    assert artifacts.deterministic_json_string({"b": 1, "a": 2}) == '{"a":2,"b":1}'

def test_artifacts_deterministic_json_string_compact_separators():
    assert artifacts.deterministic_json_string({"x": [1, 2, 3]}) == '{"x":[1,2,3]}'

def test_artifacts_deterministic_json_string_tuple_becomes_list():
    assert artifacts.deterministic_json_string({"x": (1, 2)}) == '{"x":[1,2]}'

def test_artifacts_deterministic_json_string_rejects_non_mapping():
    with pytest.raises(TypeError, match="payload must be a mapping"):
        artifacts.deterministic_json_string([1, 2, 3])

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_artifacts_deterministic_json_string_rejects_non_finite(bad):
    with pytest.raises(ValueError, match="finite JSON numbers"):
        artifacts.deterministic_json_string({"x": bad})

def test_artifacts_deterministic_json_bytes_is_utf8_of_string():
    payload = {"a": 1, "b": [2, 3]}
    assert artifacts.deterministic_json_bytes(payload) == (
        artifacts.deterministic_json_string(payload).encode("utf-8")
    )

def test_artifacts_validate_json_compatible_cycle_rejected():
    d = {}
    d["self"] = d
    with pytest.raises(ValueError, match="circular reference"):
        artifacts._validate_json_compatible(d, "payload")

def test_artifacts_validate_json_compatible_depth_rejected():
    nested = cur = {}
    for _ in range(70):
        cur["k"] = {}
        cur = cur["k"]
    with pytest.raises(ValueError, match="maximum JSON nesting depth 64"):
        artifacts._validate_json_compatible(nested, "payload")

def test_artifacts_validate_json_compatible_non_str_key_rejected():
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        artifacts._validate_json_compatible({1: "a"}, "payload")

def test_artifacts_validate_json_compatible_non_json_type_rejected():
    with pytest.raises(TypeError, match="non-JSON-compatible value"):
        artifacts._validate_json_compatible({"x": object()}, "payload")

def test_artifacts_validate_json_compatible_shared_acyclic_sibling_ok():
    shared = {"n": 1}
    # Two sibling references to the same acyclic dict must NOT be a cycle.
    artifacts._validate_json_compatible({"a": shared, "b": shared}, "payload")

def test_artifacts_validate_json_compatible_normalises_float_subclass():
    out = artifacts._validate_json_compatible({"x": _GaEvilFloat(1.5)}, "payload")
    # __float__ override lies with NaN, but the base slot returns 1.5.
    assert out["x"] == 1.5

def test_artifacts_validate_json_compatible_accepts_bool_and_none():
    out = artifacts._validate_json_compatible({"t": True, "n": None}, "p")
    assert out == {"t": True, "n": None}

def test_artifacts_manifest_entry_single_pass_fields(tmp_path):
    p = tmp_path / "run_config.json"
    p.write_bytes(b"hello")
    entry = artifacts.manifest_entry(p, "run_config")
    assert entry["role"] == "run_config"
    assert entry["size_bytes"] == 5
    assert entry["sha256"] == artifacts.sha256_file(p)
    assert entry["path"] == p.as_posix()

def test_artifacts_manifest_entry_rejects_non_string_role(tmp_path):
    p = tmp_path / "f.json"
    p.write_bytes(b"x")
    with pytest.raises(TypeError, match="role must be a string"):
        artifacts.manifest_entry(p, 5)

def test_artifacts_manifest_entry_rejects_empty_role(tmp_path):
    p = tmp_path / "f.json"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="non-empty string"):
        artifacts.manifest_entry(p, "")

@pytest.mark.parametrize("bad", [" role", "role ", "ro le", "ro\trole", "ro\nle"])
def test_artifacts_manifest_entry_rejects_whitespace_role(tmp_path, bad):
    p = tmp_path / "f.json"
    p.write_bytes(b"x")
    with pytest.raises(ValueError, match="without whitespace"):
        artifacts.manifest_entry(p, bad)

def test_artifacts_manifest_entry_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        artifacts.manifest_entry(tmp_path / "nope.json", "role")

def test_artifacts_blocked_suffixes_and_tokens_present():
    for suffix in (".pt", ".pth", ".ckpt", ".pkl", ".safetensors", ".bin", ".h5"):
        assert suffix in artifacts.BLOCKED_ARTIFACT_SUFFIXES
    assert artifacts.BLOCKED_ARTIFACT_NAME_TOKENS == (
        "checkpoint",
        "optimizer",
        "final_eval",
    )

def test_config_mappocoreconfig_happy_and_properties():
    c = _ga_core_config()
    assert c.actor_input_dim == 6
    assert c.factorized_action_count == (2, 5)
    assert c.dtype == "float32"
    assert c.device == "cpu"

def test_config_model_config_alias_identity():
    assert ModelConfig is MAPPOCoreConfig

def test_config_mappocoreconfig_zero_revealed_dim_accepted():
    c = _ga_core_config(revealed_information_dim=0)
    assert c.revealed_information_dim == 0
    assert c.actor_input_dim == 4

@pytest.mark.parametrize(
    "field",
    [
        "actor_observation_dim",
        "central_state_dim",
        "history_state_dim",
        "actor_hidden_dim",
        "critic_hidden_dim",
        "recurrent_layer_count",
        "sensing_action_count",
        "movement_action_count",
        "revealed_information_dim",
    ],
)
def test_config_mappocoreconfig_bool_as_int_rejected(field):
    with pytest.raises(TypeError):
        _ga_core_config(**{field: True})

@pytest.mark.parametrize("field", ["sensing_action_count", "movement_action_count"])
def test_config_mappocoreconfig_action_count_below_two_rejected(field):
    with pytest.raises(ValueError, match=">= 2"):
        _ga_core_config(**{field: 1})

def test_config_mappocoreconfig_negative_revealed_dim_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        _ga_core_config(revealed_information_dim=-1)

def test_config_mappocoreconfig_agent_id_count_none_ok():
    assert _ga_core_config(agent_id_count=None).agent_id_count is None

def test_config_mappocoreconfig_agent_id_count_positive_ok():
    assert _ga_core_config(agent_id_count=2).agent_id_count == 2

def test_config_mappocoreconfig_agent_id_count_zero_rejected():
    with pytest.raises(ValueError):
        _ga_core_config(agent_id_count=0)

def test_config_mappocoreconfig_agent_id_count_bool_rejected():
    with pytest.raises(TypeError):
        _ga_core_config(agent_id_count=True)

def test_config_mappocoreconfig_use_recurrent_actor_false_rejected():
    with pytest.raises(ValueError, match="recurrent actor"):
        _ga_core_config(use_recurrent_actor=False)

def test_config_mappocoreconfig_use_recurrent_actor_non_bool_rejected():
    with pytest.raises(TypeError, match="use_recurrent_actor must be a bool"):
        _ga_core_config(use_recurrent_actor="yes")

def test_config_mappocoreconfig_dtype_non_string_rejected():
    with pytest.raises(TypeError, match="dtype must be a string"):
        _ga_core_config(dtype=32)

def test_config_mappocoreconfig_dtype_unsupported_rejected():
    with pytest.raises(ValueError, match="float32.*float64"):
        _ga_core_config(dtype="float16")

def test_config_mappocoreconfig_float64_dtype_accepted():
    assert _ga_core_config(dtype="float64").dtype == "float64"

def test_config_mappocoreconfig_device_non_string_rejected():
    with pytest.raises(TypeError, match="device must be a string"):
        _ga_core_config(device=1)

def test_config_mappocoreconfig_device_empty_rejected():
    with pytest.raises(ValueError, match="non-empty string"):
        _ga_core_config(device="")

@pytest.mark.parametrize("bad", ["  ", " cpu", "cpu ", "c pu", "cu da"])
def test_config_mappocoreconfig_device_whitespace_rejected(bad):
    with pytest.raises(ValueError):
        _ga_core_config(device=bad)

def test_config_mappocoreconfig_device_junk_but_whitespace_free_accepted():
    # Junk-but-whitespace-free device strings are deliberately accepted here.
    assert _ga_core_config(device="notarealdevice").device == "notarealdevice"

def test_config_algorithm_config_happy():
    a = AlgorithmConfig(discount_factor=0.99, gae_lambda=0.95, ppo_clip_range=0.2)
    assert a.discount_factor == 0.99

def test_config_algorithm_config_probability_bounds_rejected():
    with pytest.raises(ValueError):
        AlgorithmConfig(discount_factor=1.5, gae_lambda=0.95, ppo_clip_range=0.2)

def test_config_algorithm_config_clip_endpoint_rejected():
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        AlgorithmConfig(discount_factor=0.99, gae_lambda=0.95, ppo_clip_range=1.0)

def test_config_algorithm_config_is_keyword_only():
    with pytest.raises(TypeError):
        AlgorithmConfig(0.99, 0.95, 0.2)  # positional

def test_config_loss_config_happy():
    lc = LossConfig(
        reward_entropy_coefficient=0.0,
        movement_entropy_coefficient=0.01,
        sensing_entropy_coefficient=0.01,
        hazard_cost_coefficient=1.0,
        sensing_cost_coefficient=0.1,
        value_loss_coefficient=0.5,
    )
    assert lc.value_loss_coefficient == 0.5

def test_config_loss_config_negative_coefficient_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        LossConfig(
            reward_entropy_coefficient=-0.1,
            movement_entropy_coefficient=0.0,
            sensing_entropy_coefficient=0.0,
            hazard_cost_coefficient=0.0,
            sensing_cost_coefficient=0.0,
            value_loss_coefficient=0.0,
        )

def test_config_loss_config_bool_coefficient_rejected():
    with pytest.raises(TypeError):
        LossConfig(
            reward_entropy_coefficient=True,
            movement_entropy_coefficient=0.0,
            sensing_entropy_coefficient=0.0,
            hazard_cost_coefficient=0.0,
            sensing_cost_coefficient=0.0,
            value_loss_coefficient=0.0,
        )

def test_config_lagrange_config_happy():
    g = LagrangeConfig(initial_multiplier=0.0, learning_rate=0.05, hazard_budget=0.1)
    assert g.learning_rate == 0.05

def test_config_lagrange_config_learning_rate_zero_rejected():
    with pytest.raises(ValueError, match="positive"):
        LagrangeConfig(initial_multiplier=0.0, learning_rate=0.0, hazard_budget=0.1)

def test_config_lagrange_config_negative_budget_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        LagrangeConfig(initial_multiplier=0.0, learning_rate=0.05, hazard_budget=-0.1)

def test_config_forbidden_actor_information_keys_has_exactly_70():
    assert len(forbidden_actor_information_keys()) == 70

def test_config_forbidden_actor_information_keys_returns_frozenset():
    assert isinstance(forbidden_actor_information_keys(), frozenset)

@pytest.mark.parametrize("key", sorted(forbidden_actor_information_keys()))
def test_config_validate_actor_input_mapping_rejects_each_forbidden_key(key):
    with pytest.raises(ValueError, match="forbidden centralized or unrevealed"):
        validate_actor_input_mapping({key: torch.zeros(1)})

def test_config_validate_actor_input_mapping_base_keys_accepted():
    validate_actor_input_mapping(
        {
            "actor_observation": torch.zeros(1),
            "revealed_information": torch.zeros(1),
            "history_state": torch.zeros(1),
        }
    )

def test_config_validate_actor_input_mapping_non_mapping_rejected():
    with pytest.raises(TypeError, match="actor input must be a mapping"):
        validate_actor_input_mapping([("a", 1)])

def test_config_validate_actor_input_mapping_unsupported_key_rejected():
    with pytest.raises(ValueError, match="unsupported non-local-reveal"):
        validate_actor_input_mapping({"random_field": 1})

def test_config_validate_actor_input_mapping_non_string_key_reported():
    with pytest.raises(ValueError, match="unsupported non-local-reveal"):
        validate_actor_input_mapping({7: 1})

def test_config_validate_actor_input_mapping_valid_reveal_field_accepted():
    validate_actor_input_mapping({"revealed_local_hazards": torch.zeros(1)})

@pytest.mark.parametrize(
    "key",
    [
        "revealed_local_",
        "revealed_local_Hazards",
        "revealed_local_haz__ards",
        "revealed_local_hazards_",
        "revealed_local_-x",
    ],
)
def test_config_validate_actor_input_mapping_malformed_reveal_rejected(key):
    with pytest.raises(ValueError, match="malformed revealed_local field names"):
        validate_actor_input_mapping({key: 1})

@pytest.mark.parametrize(
    "key",
    [
        "revealed_local_central",
        "revealed_local_hidden",
        "revealed_local_critic",
        "revealed_local_global",
        "revealed_local_privileged",
        "revealed_local_hiddenhazardmap",
        "revealed_local_centralstate",
        "revealed_local_criticstate",
        "revealed_local_globalhazardgrid",
        "revealed_local_foohiddenhazard",  # leading text (S-1)
        "revealed_local_hiddenhazardx",    # trailing text (m-7)
        "revealed_local_centr_al_state",   # underscore split (S-2)
        "revealed_local_hid_den",          # bare token split (S-2)
        "revealed_local_foo_hiddenhazardx",  # concatenated alias inside later token (m-7)
    ],
)
def test_config_validate_actor_input_mapping_forbidden_provenance_rejected(key):
    with pytest.raises(ValueError, match="forbidden provenance"):
        validate_actor_input_mapping({key: 1})

@pytest.mark.parametrize(
    "key",
    [
        "revealed_local_critical_hazard",
        "revealed_local_hazard_criticality",
    ],
)
def test_config_validate_actor_input_mapping_benign_lookalikes_accepted(key):
    validate_actor_input_mapping({key: 1})

def test_config_validate_actor_input_mapping_malformed_reported_before_provenance():
    # A malformed reveal key (uppercase) is reported as malformed, not provenance.
    with pytest.raises(ValueError, match="malformed"):
        validate_actor_input_mapping({"revealed_local_Central": 1})

def test_config_validate_actor_tensors_happy():
    c = _ga_core_config()
    b, t = validate_actor_tensors(
        torch.zeros(2, 3, 4), torch.zeros(2, 3, 2), None, c
    )
    assert (b, t) == (2, 3)

def test_config_validate_actor_tensors_with_history_happy():
    c = _ga_core_config()
    history = torch.zeros(c.recurrent_layer_count, 2, c.history_state_dim)
    b, t = validate_actor_tensors(
        torch.zeros(2, 3, 4), torch.zeros(2, 3, 2), history, c
    )
    assert (b, t) == (2, 3)

def test_config_validate_actor_tensors_wrong_config_type():
    with pytest.raises(TypeError, match="config must be a MAPPOCoreConfig"):
        validate_actor_tensors(torch.zeros(2, 3, 4), torch.zeros(2, 3, 2), None, object())

def test_config_validate_actor_tensors_wrong_ndim():
    c = _ga_core_config()
    with pytest.raises(ValueError, match=r"\[batch, time, dim\]"):
        validate_actor_tensors(torch.zeros(2, 4), torch.zeros(2, 3, 2), None, c)

def test_config_validate_actor_tensors_dim_mismatch():
    c = _ga_core_config()
    with pytest.raises(ValueError, match="actor_observation last dimension"):
        validate_actor_tensors(torch.zeros(2, 3, 5), torch.zeros(2, 3, 2), None, c)

def test_config_validate_actor_tensors_batch_time_mismatch():
    c = _ga_core_config()
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_actor_tensors(torch.zeros(2, 3, 4), torch.zeros(2, 2, 2), None, c)

def test_config_validate_actor_tensors_zero_batch_rejected():
    c = _ga_core_config()
    with pytest.raises(ValueError, match="batch and time dimensions must be positive"):
        validate_actor_tensors(torch.zeros(0, 3, 4), torch.zeros(0, 3, 2), None, c)

def test_config_validate_actor_tensors_non_float_dtype_rejected():
    c = _ga_core_config()
    with pytest.raises(TypeError, match="floating point dtype"):
        validate_actor_tensors(
            torch.zeros(2, 3, 4, dtype=torch.long), torch.zeros(2, 3, 2), None, c
        )

def test_config_validate_actor_tensors_wrong_float_dtype_rejected():
    c = _ga_core_config()  # float32
    with pytest.raises(ValueError, match="dtype does not match config"):
        validate_actor_tensors(
            torch.zeros(2, 3, 4, dtype=torch.float64),
            torch.zeros(2, 3, 2, dtype=torch.float64),
            None,
            c,
        )

def test_config_validate_actor_tensors_nonfinite_rejected():
    c = _ga_core_config()
    obs = torch.zeros(2, 3, 4)
    obs[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite values"):
        validate_actor_tensors(obs, torch.zeros(2, 3, 2), None, c)

def test_config_validate_actor_tensors_history_shape_mismatch():
    c = _ga_core_config()
    bad_history = torch.zeros(1, 3, c.history_state_dim)  # batch 3 != 2
    with pytest.raises(ValueError, match="history_state must have shape"):
        validate_actor_tensors(
            torch.zeros(2, 3, 4), torch.zeros(2, 3, 2), bad_history, c
        )

def test_config_validate_actor_tensors_float64_config_happy():
    c = _ga_core_config(dtype="float64")
    b, t = validate_actor_tensors(
        torch.zeros(2, 3, 4, dtype=torch.float64),
        torch.zeros(2, 3, 2, dtype=torch.float64),
        None,
        c,
    )
    assert (b, t) == (2, 3)

def test_config_validate_actor_tensors_zero_width_reveal_ok():
    c = _ga_core_config(revealed_information_dim=0)
    b, t = validate_actor_tensors(
        torch.zeros(2, 3, 4), torch.zeros(2, 3, 0), None, c
    )
    assert (b, t) == (2, 3)

def test_config_validate_central_state_happy():
    c = _ga_core_config()
    assert validate_central_state(torch.zeros(2, 3, 7), c) == (2, 3)

def test_config_validate_central_state_wrong_dim():
    c = _ga_core_config()
    with pytest.raises(ValueError, match="central_state last dimension"):
        validate_central_state(torch.zeros(2, 3, 8), c)

def test_config_validate_central_state_zero_time_rejected():
    c = _ga_core_config()
    with pytest.raises(ValueError, match="batch and time dimensions must be positive"):
        validate_central_state(torch.zeros(2, 0, 7), c)

def test_config_validate_central_state_wrong_float_dtype_rejected():
    c = _ga_core_config()
    with pytest.raises(ValueError, match="dtype does not match config"):
        validate_central_state(torch.zeros(2, 3, 7, dtype=torch.float64), c)

def test_config_torch_required_returns_module():
    assert torch_required("x") is torch

def test_config_torch_dtype_from_name_matrix():
    assert torch_dtype_from_name("float32") is torch.float32
    assert torch_dtype_from_name("float64") is torch.float64

def test_config_torch_dtype_from_name_unsupported_rejected():
    with pytest.raises(ValueError, match="unsupported floating point dtype"):
        torch_dtype_from_name("float16")

def test_config_require_torch_tensor_rejects_non_tensor():
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        require_torch_tensor("x", [1, 2])

def test_config_require_finite_float_tensor_happy():
    assert require_finite_float_tensor("x", torch.zeros(3)) is None

def test_config_require_finite_float_tensor_non_tensor_rejected():
    with pytest.raises(TypeError, match="must be a torch.Tensor"):
        require_finite_float_tensor("x", 5)

def test_config_require_finite_float_tensor_int_dtype_rejected():
    with pytest.raises(TypeError, match="floating point dtype"):
        require_finite_float_tensor("x", torch.zeros(3, dtype=torch.long))

def test_config_require_finite_float_tensor_meta_rejected():
    with pytest.raises(ValueError, match="meta tensor"):
        require_finite_float_tensor("x", torch.zeros(3, device="meta"))

def test_config_require_finite_float_tensor_sparse_rejected():
    with pytest.raises(ValueError, match="strided tensor layout"):
        require_finite_float_tensor("x", torch.zeros(3).to_sparse())

def test_config_require_finite_float_tensor_nonfinite_rejected():
    t = torch.zeros(3)
    t[0] = float("inf")
    with pytest.raises(ValueError, match="finite values"):
        require_finite_float_tensor("x", t)

def test_config_require_finite_float_tensor_matching_config_happy():
    c = _ga_core_config()
    t = torch.zeros(3)
    assert require_finite_float_tensor_matching_config("x", t, c) is t

def test_config_require_finite_float_tensor_matching_config_wrong_dtype():
    c = _ga_core_config()  # float32
    with pytest.raises(ValueError, match="dtype does not match config"):
        require_finite_float_tensor_matching_config("x", torch.zeros(3, dtype=torch.float64), c)

def test_config_require_finite_float_tensor_matching_config_wrong_config_type():
    with pytest.raises(TypeError, match="config must be a MAPPOCoreConfig"):
        require_finite_float_tensor_matching_config("x", torch.zeros(3), object())

def test_lagrange_multiplier_construction_happy():
    m = LagrangeMultiplier(value=1.0, learning_rate=0.5)
    assert m.value == 1.0
    assert m.learning_rate == 0.5

def test_lagrange_multiplier_negative_value_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        LagrangeMultiplier(value=-1.0, learning_rate=0.5)

def test_lagrange_multiplier_nonpositive_learning_rate_rejected():
    with pytest.raises(ValueError, match="positive"):
        LagrangeMultiplier(value=0.0, learning_rate=0.0)

def test_lagrange_multiplier_bool_field_rejected():
    with pytest.raises(TypeError):
        LagrangeMultiplier(value=True, learning_rate=0.5)

def test_lagrange_multiplier_update_projects_up():
    m = LagrangeMultiplier(value=1.0, learning_rate=0.5)
    # 1 + 0.5*(3 - 1) = 2.0
    assert m.update(observed_cost=3.0, cost_budget=1.0).value == 2.0

def test_lagrange_multiplier_update_clamps_at_zero():
    m = LagrangeMultiplier(value=1.0, learning_rate=0.5)
    # 1 + 0.5*(0 - 10) = -4 -> clamp 0
    assert m.update(observed_cost=0.0, cost_budget=10.0).value == 0.0

def test_lagrange_multiplier_update_unchanged_at_budget():
    m = LagrangeMultiplier(value=1.0, learning_rate=0.5)
    assert m.update(observed_cost=2.0, cost_budget=2.0).value == 1.0

def test_lagrange_multiplier_update_returns_new_object_and_immutable():
    m = LagrangeMultiplier(value=1.0, learning_rate=0.5)
    m2 = m.update(observed_cost=3.0, cost_budget=1.0)
    assert m2 is not m
    assert m.value == 1.0
    assert m2.learning_rate == 0.5  # carried forward

def test_lagrange_multiplier_update_frozen_dataclass():
    m = LagrangeMultiplier(value=1.0, learning_rate=0.5)
    with pytest.raises(Exception):
        m.value = 5.0  # frozen

def test_lagrange_multiplier_update_keyword_only():
    m = LagrangeMultiplier(value=1.0, learning_rate=0.5)
    with pytest.raises(TypeError):
        m.update(3.0, 1.0)  # positional not allowed

def test_lagrange_multiplier_update_negative_cost_rejected():
    m = LagrangeMultiplier(value=1.0, learning_rate=0.5)
    with pytest.raises(ValueError, match="nonnegative"):
        m.update(observed_cost=-1.0, cost_budget=1.0)

def test_lagrange_multiplier_update_negative_budget_rejected():
    m = LagrangeMultiplier(value=1.0, learning_rate=0.5)
    with pytest.raises(ValueError, match="nonnegative"):
        m.update(observed_cost=1.0, cost_budget=-1.0)

def test_lagrange_multiplier_update_overflow_up_rejected():
    m = LagrangeMultiplier(value=1e308, learning_rate=1e308)
    with pytest.raises(ValueError, match="updated multiplier value must be finite"):
        m.update(observed_cost=1e308, cost_budget=0.0)

def test_lagrange_multiplier_update_overflow_negative_rejected():
    m = LagrangeMultiplier(value=0.0, learning_rate=1e308)
    with pytest.raises(ValueError, match="updated multiplier value must be finite"):
        m.update(observed_cost=0.0, cost_budget=1e308)

def test_lagrange_multiplier_update_converges_to_zero():
    m = LagrangeMultiplier(value=5.0, learning_rate=1.0)
    for _ in range(10):
        m = m.update(observed_cost=0.0, cost_budget=1.0)
    assert m.value == 0.0

def test_lagrange_update_lagrange_multiplier_wrapper_equivalence():
    assert (
        update_lagrange_multiplier(
            current_value=1.0, learning_rate=0.5, observed_cost=3.0, cost_budget=1.0
        )
        == 2.0
    )

def test_lagrange_update_lagrange_multiplier_wrapper_value_error_named():
    with pytest.raises(ValueError, match="current_value"):
        update_lagrange_multiplier(
            current_value=-1.0, learning_rate=0.5, observed_cost=1.0, cost_budget=1.0
        )

def test_lagrange_update_lagrange_multiplier_wrapper_keyword_only():
    with pytest.raises(TypeError):
        update_lagrange_multiplier(1.0, 0.5, 3.0, 1.0)

def test_losses_compute_gae_matches_reference_no_terminal():
    signals = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[0.5, 0.5, 0.5]])
    next_values = torch.tensor([[1.0, 1.0, 1.0]])
    terminal = torch.zeros(1, 3, dtype=torch.bool)
    truncation = torch.zeros(1, 3, dtype=torch.bool)
    mask = torch.ones(1, 3, dtype=torch.bool)
    out = compute_gae(signals, values, next_values, terminal, truncation, None, 0.9, 0.8)
    ref = _ga_reference_gae(signals, values, next_values, terminal, truncation, mask, 0.9, 0.8)
    assert torch.allclose(out, ref, atol=1e-6)
    # Explicit hand values (computed offline): [4.89056, 4.848, 3.4]
    assert torch.allclose(out, torch.tensor([[4.890560626983643, 4.848000526428223, 3.4]]), atol=1e-5)

def test_losses_compute_gae_terminal_blocks_bootstrap_and_carry():
    signals = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[0.5, 0.5, 0.5]])
    next_values = torch.tensor([[1.0, 1.0, 1.0]])
    terminal = torch.tensor([[False, True, False]])
    truncation = torch.zeros(1, 3, dtype=torch.bool)
    out = compute_gae(signals, values, next_values, terminal, truncation, None, 0.9, 0.8)
    # Hand-computed: t2=3.4, t1=1.5 (no bootstrap, no carry), t0=2.48.
    assert torch.allclose(out, torch.tensor([[2.48, 1.5, 3.4]]), atol=1e-5)

def test_losses_compute_gae_truncation_keeps_bootstrap_stops_carry():
    signals = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[0.5, 0.5, 0.5]])
    next_values = torch.tensor([[1.0, 1.0, 1.0]])
    terminal = torch.zeros(1, 3, dtype=torch.bool)
    truncation = torch.tensor([[False, True, False]])
    out = compute_gae(signals, values, next_values, terminal, truncation, None, 0.9, 0.8)
    # Hand-computed: t2=3.4, t1=2.4 (bootstrap kept), t0=3.128 (carry stopped at t1).
    assert torch.allclose(out, torch.tensor([[3.128, 2.4, 3.4]]), atol=1e-5)

def test_losses_compute_gae_valid_mask_zeroes_masked_steps():
    signals = torch.tensor([[1.0, 2.0, 3.0]])
    values = torch.tensor([[0.5, 0.5, 0.5]])
    next_values = torch.tensor([[1.0, 1.0, 1.0]])
    terminal = torch.zeros(1, 3, dtype=torch.bool)
    truncation = torch.zeros(1, 3, dtype=torch.bool)
    mask = torch.tensor([[True, False, True]])
    out = compute_gae(signals, values, next_values, terminal, truncation, mask, 0.9, 0.8)
    # t1 masked -> 0; carry from t1 blocked -> t0=1.4.
    assert torch.allclose(out, torch.tensor([[1.4, 0.0, 3.4]]), atol=1e-5)
    ref = _ga_reference_gae(signals, values, next_values, terminal, truncation, mask, 0.9, 0.8)
    assert torch.allclose(out, ref, atol=1e-6)

def test_losses_compute_gae_gamma_out_of_range_rejected():
    signals = torch.zeros(1, 2)
    with pytest.raises(ValueError):
        compute_gae(
            signals, signals, signals,
            torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool),
            None, 1.5, 0.9,
        )

def test_losses_compute_gae_lam_out_of_range_rejected():
    signals = torch.zeros(1, 2)
    with pytest.raises(ValueError):
        compute_gae(
            signals, signals, signals,
            torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool),
            None, 0.9, 1.5,
        )

def test_losses_compute_gae_nonfinite_signals_rejected():
    signals = torch.zeros(1, 2)
    signals[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        compute_gae(
            signals, torch.zeros(1, 2), torch.zeros(1, 2),
            torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool),
            None, 0.9, 0.9,
        )

def test_losses_compute_reward_gae_delegates():
    signals = torch.tensor([[1.0, 2.0]])
    args = (
        torch.zeros(1, 2), torch.zeros(1, 2),
        torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool),
        None, 0.9, 0.8,
    )
    out = compute_reward_gae(signals, *args)
    direct = compute_gae(signals, *args)
    assert torch.allclose(out, direct)

def test_losses_compute_cost_gae_rejects_negative_signals():
    signals = torch.tensor([[-1.0, 2.0]])
    with pytest.raises(ValueError, match="cost signals must be nonnegative"):
        compute_cost_gae(
            signals, torch.zeros(1, 2), torch.zeros(1, 2),
            torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool),
            None, 0.9, 0.8,
        )

def test_losses_compute_cost_gae_accepts_nonnegative_signals():
    signals = torch.tensor([[1.0, 2.0]])
    out = compute_cost_gae(
        signals, torch.zeros(1, 2), torch.zeros(1, 2),
        torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 2, dtype=torch.bool),
        None, 0.9, 0.8,
    )
    assert out.shape == (1, 2)

def test_losses_bootstrap_allowed_mask_is_not_terminal():
    terminal = torch.tensor([[False, True], [True, False]])
    out = bootstrap_allowed_mask(terminal)
    assert torch.equal(out, ~terminal)

def test_losses_bootstrap_allowed_mask_non_bool_rejected():
    with pytest.raises(TypeError, match="torch.bool"):
        bootstrap_allowed_mask(torch.zeros(1, 2))

def test_losses_sequence_boundary_mask_or_of_flags():
    terminal = torch.tensor([[True, False]])
    truncation = torch.tensor([[False, True]])
    out = sequence_boundary_mask(terminal, truncation)
    assert torch.equal(out, torch.tensor([[True, True]]))

def test_losses_sequence_boundary_mask_both_true_rejected():
    terminal = torch.tensor([[True]])
    truncation = torch.tensor([[True]])
    with pytest.raises(ValueError, match="cannot both be true"):
        sequence_boundary_mask(terminal, truncation)

def test_losses_sequence_boundary_mask_shape_mismatch():
    with pytest.raises(ValueError, match="shape mismatch"):
        sequence_boundary_mask(
            torch.zeros(1, 2, dtype=torch.bool), torch.zeros(1, 3, dtype=torch.bool)
        )

def test_losses_recursive_carry_mask_last_column_always_false():
    terminal = torch.zeros(1, 3, dtype=torch.bool)
    truncation = torch.zeros(1, 3, dtype=torch.bool)
    valid = torch.ones(1, 3, dtype=torch.bool)
    out = recursive_carry_mask(terminal, truncation, valid)
    assert out[0, -1].item() is False
    assert torch.equal(out, torch.tensor([[True, True, False]]))

def test_losses_recursive_carry_mask_blocked_by_boundary():
    terminal = torch.tensor([[False, True, False]])
    truncation = torch.zeros(1, 3, dtype=torch.bool)
    valid = torch.ones(1, 3, dtype=torch.bool)
    out = recursive_carry_mask(terminal, truncation, valid)
    # boundary at t=1 blocks carry at t=1; t=0 carry allowed.
    assert torch.equal(out, torch.tensor([[True, False, False]]))

def test_losses_validate_loss_input_mapping_scalar_happy():
    validate_loss_input_mapping(
        {"task_reward": 1.0, "hazard_cost": 0.0, "sensing_cost": 0.0}
    )

def test_losses_validate_loss_input_mapping_non_mapping_rejected():
    with pytest.raises(TypeError, match="loss input must be a mapping"):
        validate_loss_input_mapping([("task_reward", 1.0)])

def test_losses_validate_loss_input_mapping_non_string_key_rejected():
    with pytest.raises(TypeError, match="keys must be strings"):
        validate_loss_input_mapping(
            {1: 2, "task_reward": 1.0, "hazard_cost": 0.0, "sensing_cost": 0.0}
        )

def test_losses_validate_loss_input_mapping_missing_keys_rejected():
    with pytest.raises(ValueError, match="missing: hazard_cost, sensing_cost"):
        validate_loss_input_mapping({"task_reward": 1.0})

def test_losses_validate_loss_input_mapping_negative_cost_scalar_rejected():
    with pytest.raises(ValueError, match="hazard_cost must be nonnegative"):
        validate_loss_input_mapping(
            {"task_reward": 1.0, "hazard_cost": -1.0, "sensing_cost": 0.0}
        )

def test_losses_validate_loss_input_mapping_tensor_fields_happy():
    validate_loss_input_mapping(
        {
            "task_reward": torch.zeros(1, 2),
            "hazard_cost": torch.zeros(1, 2),
            "sensing_cost": torch.zeros(1, 2),
            "valid_mask": torch.ones(1, 2, dtype=torch.bool),
            "terminal": torch.zeros(1, 2, dtype=torch.bool),
            "truncation": torch.zeros(1, 2, dtype=torch.bool),
        }
    )

def test_losses_validate_loss_input_mapping_bool_tensor_for_non_control_rejected():
    with pytest.raises(TypeError, match="boolean tensor is only allowed"):
        validate_loss_input_mapping(
            {
                "task_reward": torch.zeros(1, 2, dtype=torch.bool),
                "hazard_cost": torch.zeros(1, 2),
                "sensing_cost": torch.zeros(1, 2),
            }
        )

def test_losses_validate_loss_input_mapping_control_field_bad_dtype_rejected():
    with pytest.raises(TypeError, match="torch.bool"):
        validate_loss_input_mapping(
            {
                "task_reward": torch.zeros(1, 2),
                "hazard_cost": torch.zeros(1, 2),
                "sensing_cost": torch.zeros(1, 2),
                "valid_mask": torch.zeros(1, 2),  # float, not bool
            }
        )

def test_losses_validate_loss_input_mapping_terminal_truncation_overlap_rejected():
    both = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="cannot both be true"):
        validate_loss_input_mapping(
            {
                "task_reward": torch.zeros(1, 2),
                "hazard_cost": torch.zeros(1, 2),
                "sensing_cost": torch.zeros(1, 2),
                "terminal": both,
                "truncation": both,
            }
        )

def test_losses_validate_loss_input_mapping_terminal_alias_exclusive():
    with pytest.raises(ValueError, match="both terminal and terminals"):
        validate_loss_input_mapping(
            {
                "task_reward": torch.zeros(1, 2),
                "hazard_cost": torch.zeros(1, 2),
                "sensing_cost": torch.zeros(1, 2),
                "terminal": torch.zeros(1, 2, dtype=torch.bool),
                "terminals": torch.zeros(1, 2, dtype=torch.bool),
            }
        )

def test_losses_validate_loss_input_mapping_negative_cost_tensor_rejected():
    with pytest.raises(ValueError, match="hazard_cost must be nonnegative"):
        validate_loss_input_mapping(
            {
                "task_reward": torch.zeros(1, 2),
                "hazard_cost": torch.full((1, 2), -1.0),
                "sensing_cost": torch.zeros(1, 2),
            }
        )

def test_losses_ppo_clipped_surrogate_positive_advantage():
    # r = exp(ln 1.5) = 1.5, clip 0.2 -> clamp to 1.2; A=2 -> min(3.0, 2.4)=2.4.
    new = torch.tensor([[math.log(1.5)]])
    old = torch.tensor([[0.0]])
    out = ppo_clipped_surrogate(new, old, torch.tensor([[2.0]]), 0.2)
    assert torch.allclose(out, torch.tensor([[2.4]]), atol=1e-5)

def test_losses_ppo_clipped_surrogate_negative_advantage():
    # r=1.5, A=-2 -> min(-3.0, -2.4) = -3.0 (clip does not help negative A).
    new = torch.tensor([[math.log(1.5)]])
    old = torch.tensor([[0.0]])
    out = ppo_clipped_surrogate(new, old, torch.tensor([[-2.0]]), 0.2)
    assert torch.allclose(out, torch.tensor([[-3.0]]), atol=1e-5)

def test_losses_ppo_clipped_surrogate_equals_min_elementwise():
    new = torch.tensor([[0.3, -0.4]])
    old = torch.tensor([[0.0, 0.0]])
    adv = torch.tensor([[1.5, -2.0]])
    eps = 0.2
    out = ppo_clipped_surrogate(new, old, adv, eps)
    ratio = torch.exp(new - old)
    clipped = ratio.clamp(1 - eps, 1 + eps)
    expected = torch.minimum(ratio * adv, clipped * adv)
    assert torch.allclose(out, expected, atol=1e-6)

def test_losses_ppo_clipped_surrogate_clip_epsilon_endpoint_rejected():
    new = torch.zeros(1, 1)
    with pytest.raises(ValueError, match=r"\(0, 1\)"):
        ppo_clipped_surrogate(new, new, new, 1.0)

def test_losses_ppo_clipped_surrogate_overflow_ratio_rejected():
    # A huge log-ratio overflows exp().
    new = torch.tensor([[1000.0]])
    old = torch.tensor([[0.0]])
    with pytest.raises(ValueError, match="probability ratio must be finite"):
        ppo_clipped_surrogate(new, old, torch.tensor([[1.0]]), 0.2)

def test_losses_masked_mean_excludes_masked_entries():
    vals = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[True, False, True]])
    assert masked_mean(vals, mask).item() == 2.0

def test_losses_masked_mean_none_mask_rejected():
    with pytest.raises(TypeError, match="valid_mask must be a torch.Tensor"):
        masked_mean(torch.zeros(1, 3), None)

def test_losses_masked_mean_all_false_mask_rejected():
    with pytest.raises(ValueError, match="at least one true timestep"):
        masked_mean(torch.zeros(1, 3), torch.zeros(1, 3, dtype=torch.bool))

def test_losses_masked_mean_gradient_flows():
    vals = torch.tensor([[1.0, 2.0]], requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    masked_mean(vals, mask).backward()
    assert vals.grad is not None

def test_losses_value_loss_masked_mse():
    pred = torch.tensor([[1.0, 5.0]])
    tgt = torch.tensor([[1.0, 2.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    # (0 + 9) / 2 = 4.5
    assert value_loss(pred, tgt, mask).item() == 4.5

def test_losses_value_loss_returns_scalar():
    pred = torch.zeros(1, 2)
    out = value_loss(pred, pred, torch.ones(1, 2, dtype=torch.bool))
    assert out.ndim == 0

def test_losses_entropy_bonus_masked_mean():
    ent = torch.tensor([[2.0, 4.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    assert entropy_bonus(ent, mask).item() == 3.0

def test_losses_ppo_clipped_policy_surrogate_scalar():
    new = torch.zeros(1, 2)
    out = ppo_clipped_policy_surrogate(new, new, torch.ones(1, 2), 0.2, torch.ones(1, 2, dtype=torch.bool))
    assert out.ndim == 0

def test_losses_ppo_policy_loss_is_negative_surrogate():
    new = torch.zeros(1, 2)
    adv = torch.ones(1, 2)
    mask = torch.ones(1, 2, dtype=torch.bool)
    loss = ppo_policy_loss(new, new, adv, 0.2, mask)
    surrogate = ppo_clipped_policy_surrogate(new, new, adv, 0.2, mask)
    assert torch.allclose(loss, -surrogate)

def test_losses_lagrangian_advantage_sign_and_values():
    # A_r - lambda * A_c: [5,5] - 2*[1,3] = [3, -1].
    out = lagrangian_advantage(torch.tensor([[5.0, 5.0]]), torch.tensor([[1.0, 3.0]]), 2.0)
    assert torch.allclose(out, torch.tensor([[3.0, -1.0]]))

def test_losses_lagrangian_advantage_higher_cost_lowers_objective():
    low = lagrangian_advantage(torch.tensor([[5.0]]), torch.tensor([[1.0]]), 1.0)
    high = lagrangian_advantage(torch.tensor([[5.0]]), torch.tensor([[3.0]]), 1.0)
    assert high.item() < low.item()

def test_losses_lagrangian_advantage_tensor_multiplier():
    out = lagrangian_advantage(
        torch.tensor([[4.0]]), torch.tensor([[2.0]]), torch.tensor(1.5)
    )
    assert torch.allclose(out, torch.tensor([[1.0]]))

def test_losses_lagrangian_advantage_negative_multiplier_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        lagrangian_advantage(torch.zeros(1, 2), torch.zeros(1, 2), -1.0)

def test_losses_lagrangian_advantage_bool_tensor_multiplier_rejected():
    with pytest.raises(TypeError, match="not boolean"):
        lagrangian_advantage(torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor(True))

def test_losses_lagrangian_advantage_complex_multiplier_rejected():
    with pytest.raises(TypeError, match="not complex"):
        lagrangian_advantage(
            torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor(1 + 2j)
        )

def test_losses_lagrangian_advantage_non_scalar_tensor_multiplier_rejected():
    with pytest.raises(ValueError, match="must be scalar"):
        lagrangian_advantage(
            torch.zeros(1, 2), torch.zeros(1, 2), torch.tensor([1.0, 2.0])
        )

def test_losses_lagrangian_advantage_tensor_multiplier_detached():
    mult = torch.tensor(1.0, requires_grad=True)
    reward = torch.zeros(1, 2, requires_grad=True)
    out = lagrangian_advantage(reward, torch.ones(1, 2), mult)
    out.sum().backward()
    # The projected-dual-ascent multiplier is detached: no gradient flows to it
    # through the policy loss (Ray et al. 2019).
    assert mult.grad is None

def test_losses_lagrangian_policy_objective_sign():
    out = lagrangian_policy_objective(
        torch.tensor([[5.0]]), torch.tensor([[2.0]]), 1.0
    )
    assert torch.allclose(out, torch.tensor([[3.0]]))

def test_losses_lagrangian_ppo_policy_loss_runs():
    new = torch.zeros(1, 2)
    out = lagrangian_ppo_policy_loss(
        new, new, torch.ones(1, 2), torch.ones(1, 2), 0.5, 0.2,
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert out.ndim == 0

def test_buffer_rollout_batch_validate_happy_returns_shape():
    c = _ga_core_config()
    b = _ga_rollout_batch(2, 3)
    assert b.validate(c) == (2, 3)

def test_buffer_rollout_batch_spec_alias_identity():
    assert RolloutBatchSpec is RolloutBatch

def test_buffer_rollout_batch_validate_wrong_config_type():
    b = _ga_rollout_batch(2, 3)
    with pytest.raises(TypeError, match="config must be a MAPPOCoreConfig"):
        b.validate(object())

def test_buffer_rollout_batch_validate_missing_tensor_type_rejected():
    c = _ga_core_config()
    b = _ga_rollout_batch(2, 3, task_reward=[1.0])
    with pytest.raises(TypeError, match="task_reward must be a torch.Tensor"):
        b.validate(c)

def test_buffer_rollout_batch_validate_device_coherence_meta_rejected():
    c = _ga_core_config()
    b = _ga_rollout_batch(2, 3, actor_observation=torch.zeros(2, 3, 4, device="meta"))
    with pytest.raises(ValueError):
        b.validate(c)

def test_buffer_rollout_batch_validate_negative_cost_rejected():
    c = _ga_core_config()
    b = _ga_rollout_batch(2, 3, hazard_cost=torch.full((2, 3), -1.0))
    with pytest.raises(ValueError, match="hazard_cost must be nonnegative"):
        b.validate(c)

def test_buffer_rollout_batch_validate_terminal_truncation_overlap_rejected():
    c = _ga_core_config()
    tm = torch.zeros(2, 3, dtype=torch.bool)
    tm[0, 0] = True
    tr = torch.zeros(2, 3, dtype=torch.bool)
    tr[0, 0] = True
    b = _ga_rollout_batch(2, 3, terminal=tm, truncation=tr)
    with pytest.raises(ValueError, match="terminal and truncation cannot both be true"):
        b.validate(c)

def test_buffer_rollout_batch_validate_action_out_of_range_rejected():
    c = _ga_core_config()
    a = torch.zeros(2, 3, dtype=torch.long)
    a[0, 0] = 5  # movement_action_count == 5 -> index 5 invalid
    b = _ga_rollout_batch(2, 3, movement_action=a)
    with pytest.raises(ValueError, match="out-of-range action index"):
        b.validate(c)

def test_buffer_rollout_batch_validate_float_action_rejected():
    c = _ga_core_config()
    b = _ga_rollout_batch(2, 3, sensing_action=torch.zeros(2, 3))
    with pytest.raises(TypeError, match="must use an integer dtype"):
        b.validate(c)

def test_buffer_rollout_batch_validate_bool_action_rejected():
    c = _ga_core_config()
    b = _ga_rollout_batch(2, 3, sensing_action=torch.zeros(2, 3, dtype=torch.bool))
    with pytest.raises(TypeError, match="must use an integer dtype"):
        b.validate(c)

def test_buffer_rollout_batch_validate_wrong_float_dtype_rejected():
    c = _ga_core_config()  # float32
    b = _ga_rollout_batch(2, 3, task_reward=torch.zeros(2, 3, dtype=torch.float64))
    with pytest.raises(ValueError, match="dtype does not match config"):
        b.validate(c)

def test_buffer_rollout_batch_validate_central_shape_mismatch_rejected():
    c = _ga_core_config()
    b = _ga_rollout_batch(2, 3, central_state=torch.zeros(3, 3, 7))
    with pytest.raises(ValueError):
        b.validate(c)

def test_buffer_rollout_batch_validate_bool_field_wrong_dtype_rejected():
    c = _ga_core_config()
    b = _ga_rollout_batch(2, 3, terminal=torch.zeros(2, 3))
    with pytest.raises(TypeError, match="terminal must use torch.bool dtype"):
        b.validate(c)

def test_buffer_rollout_batch_old_joint_log_probability_is_sum():
    b = _ga_rollout_batch(
        2, 3,
        old_sensing_log_probability=torch.full((2, 3), -1.0),
        old_movement_log_probability=torch.full((2, 3), -2.0),
    )
    assert torch.allclose(b.old_joint_log_probability, torch.full((2, 3), -3.0))

def test_buffer_rollout_batch_validate_sparse_bool_rejected():
    c = _ga_core_config()
    sparse_bool = torch.zeros(2, 3, dtype=torch.bool).to_sparse()
    b = _ga_rollout_batch(2, 3, valid_mask=sparse_bool)
    with pytest.raises(ValueError, match="strided tensor layout"):
        b.validate(c)

def _ga_full_contract(config):
    return EnvironmentStepContract(
        actor_visible=_ga_actor_payload(),
        critic_visible={"central_state": torch.zeros(config.central_state_dim)},
        rewards_and_costs={"task_reward": 1.0, "hazard_cost": 0.0, "sensing_cost": 0.0},
        done_flags={"terminal": False, "truncated": False},
        action_factors={
            "sensing_action_available": True,
            "movement_action_available": True,
        },
        identity={"team_id": 0, "agent_id": 0, "agent_order": [0, 1]},
    )

def test_env_contracts_validate_environment_step_contract_happy():
    c = _ga_core_config(agent_id_count=2)
    validate_environment_step_contract(_ga_full_contract(c), c)

def test_env_contracts_environment_contract_alias_identity():
    assert EnvironmentContract is EnvironmentStepContract

def test_env_contracts_validate_environment_step_contract_wrong_contract_type():
    c = _ga_core_config()
    with pytest.raises(TypeError, match="must be an EnvironmentStepContract"):
        validate_environment_step_contract(object(), c)

def test_env_contracts_validate_environment_step_contract_wrong_config_type():
    c = _ga_core_config(agent_id_count=2)
    with pytest.raises(TypeError, match="config must be a MAPPOCoreConfig"):
        validate_environment_step_contract(_ga_full_contract(c), object())

def test_env_contracts_validate_actor_visible_payload_history_state_rejected():
    c = _ga_core_config()
    payload = {**_ga_actor_payload(), "history_state": torch.zeros(2)}
    with pytest.raises(ValueError, match="history_state is model recurrent state"):
        validate_actor_visible_payload(payload, c)

def test_env_contracts_validate_actor_visible_payload_missing_key_rejected():
    c = _ga_core_config()
    with pytest.raises(ValueError, match="missing required key"):
        validate_actor_visible_payload({"actor_observation": torch.zeros(4)}, c)

def test_env_contracts_validate_actor_visible_payload_extra_reveal_field_ok():
    c = _ga_core_config()
    payload = {**_ga_actor_payload(), "revealed_local_hazards": torch.zeros(3)}
    validate_actor_visible_payload(payload, c)

def test_env_contracts_validate_actor_visible_payload_reveal_field_wrong_ndim_rejected():
    c = _ga_core_config()
    payload = {**_ga_actor_payload(), "revealed_local_hazards": torch.zeros(3, 3)}
    with pytest.raises(ValueError, match=r"\[dim\]"):
        validate_actor_visible_payload(payload, c)

def test_env_contracts_validate_critic_central_payload_happy():
    c = _ga_core_config()
    validate_critic_central_payload({"central_state": torch.zeros(7)}, c)

def test_env_contracts_validate_critic_central_payload_extra_key_rejected():
    c = _ga_core_config()
    with pytest.raises(ValueError, match="outside central_state"):
        validate_critic_central_payload(
            {"central_state": torch.zeros(7), "central_map": torch.zeros(1)}, c
        )

def test_env_contracts_validate_critic_central_payload_forbidden_alias_named():
    c = _ga_core_config()
    with pytest.raises(ValueError, match="CTDE-forbidden actor aliases present"):
        validate_critic_central_payload(
            {"central_state": torch.zeros(7), "hidden_hazard_map": torch.zeros(1)}, c
        )

def test_env_contracts_validate_critic_central_payload_wrong_shape_rejected():
    c = _ga_core_config()
    with pytest.raises(ValueError, match="shape mismatch"):
        validate_critic_central_payload({"central_state": torch.zeros(3)}, c)

def test_env_contracts_validate_rewards_and_costs_happy():
    _validate_rewards_and_costs(
        {"task_reward": 1.0, "hazard_cost": 0.0, "sensing_cost": 0.0}
    )

def test_env_contracts_validate_rewards_and_costs_non_numeric_rejected():
    with pytest.raises(TypeError):
        _validate_rewards_and_costs(
            {"task_reward": "x", "hazard_cost": 0.0, "sensing_cost": 0.0}
        )

def test_env_contracts_validate_rewards_and_costs_negative_cost_rejected():
    with pytest.raises(ValueError, match="hazard_cost must be nonnegative"):
        _validate_rewards_and_costs(
            {"task_reward": 1.0, "hazard_cost": -1.0, "sensing_cost": 0.0}
        )

def test_env_contracts_validate_rewards_and_costs_extra_key_tolerated():
    # Open schema: extra key is not a CTDE boundary.
    _validate_rewards_and_costs(
        {"task_reward": 1.0, "hazard_cost": 0.0, "sensing_cost": 0.0, "extra": 99}
    )

def test_env_contracts_validate_done_flags_happy():
    _validate_done_flags({"terminal": False, "truncated": True})

def test_env_contracts_validate_done_flags_truncation_alias_rejected():
    with pytest.raises(ValueError, match="requires truncated, not truncation"):
        _validate_done_flags({"terminal": False, "truncation": False})

def test_env_contracts_validate_done_flags_both_spellings_rejected():
    with pytest.raises(ValueError, match="both truncated and truncation"):
        _validate_done_flags(
            {"terminal": False, "truncated": False, "truncation": False}
        )

def test_env_contracts_validate_done_flags_both_true_rejected():
    with pytest.raises(ValueError, match="terminal and truncated cannot both be true"):
        _validate_done_flags({"terminal": True, "truncated": True})

def test_env_contracts_validate_done_flags_non_bool_rejected():
    with pytest.raises(TypeError, match="terminal must be a bool"):
        _validate_done_flags({"terminal": 1, "truncated": False})

def test_env_contracts_validate_action_factor_payload_happy():
    _validate_action_factor_payload(
        {"sensing_action_available": True, "movement_action_available": True}
    )

def test_env_contracts_validate_action_factor_payload_false_rejected():
    with pytest.raises(ValueError, match="must be true for a valid selectable factor"):
        _validate_action_factor_payload(
            {"sensing_action_available": False, "movement_action_available": True}
        )

def test_env_contracts_validate_action_factor_payload_non_bool_rejected():
    with pytest.raises(TypeError, match="must be a bool"):
        _validate_action_factor_payload(
            {"sensing_action_available": 1, "movement_action_available": True}
        )

def test_env_contracts_validate_identity_happy():
    c = _ga_core_config(agent_id_count=2)
    _validate_identity({"team_id": 0, "agent_id": 1, "agent_order": [0, 1]}, c)

def test_env_contracts_validate_identity_negative_id_rejected():
    c = _ga_core_config()
    with pytest.raises(ValueError, match="nonnegative"):
        _validate_identity({"team_id": -1, "agent_id": 0, "agent_order": [0, 1]}, c)

def test_env_contracts_validate_identity_no_ordering_structure_rejected():
    c = _ga_core_config()
    with pytest.raises(ValueError, match="requires agent_order"):
        _validate_identity({"team_id": 0, "agent_id": 0}, c)

def test_env_contracts_validate_identity_agent_id_count_exceeded_rejected():
    c = _ga_core_config(agent_id_count=2)
    with pytest.raises(ValueError, match="smaller than configured agent_id_count"):
        _validate_identity({"team_id": 0, "agent_id": 5, "agent_order": [0, 1]}, c)

def test_env_contracts_validate_identity_duplicate_order_rejected():
    c = _ga_core_config()
    with pytest.raises(ValueError, match="duplicate agent identifiers"):
        _validate_identity({"team_id": 0, "agent_id": 0, "agent_order": [0, 0]}, c)

def test_env_contracts_validate_identity_current_agent_missing_rejected():
    c = _ga_core_config()
    with pytest.raises(ValueError, match="include the current agent_id"):
        _validate_identity({"team_id": 0, "agent_id": 3, "agent_order": [0, 1]}, c)

def test_env_contracts_validate_identity_mapping_structure_happy():
    c = _ga_core_config(agent_id_count=2)
    _validate_identity(
        {"team_id": 0, "agent_id": 1, "agent_id_mapping": {0: 0, 1: 1}}, c
    )

def test_env_contracts_validate_identity_mapping_full_team_required():
    c = _ga_core_config(agent_id_count=2)
    with pytest.raises(ValueError, match="cover the full configured team"):
        _validate_identity(
            {"team_id": 0, "agent_id": 0, "agent_id_mapping": {0: 0}}, c
        )

def test_env_contracts_validate_identity_cross_structure_disagreement_rejected():
    c = _ga_core_config(agent_id_count=2)
    with pytest.raises(ValueError, match="ambiguous"):
        _validate_identity(
            {
                "team_id": 0,
                "agent_id": 0,
                "agent_order": [0, 1],
                "team_agent_ids": [1, 0],
            },
            c,
        )

def test_env_contracts_validate_identity_team_agent_ids_full_coverage_required():
    c = _ga_core_config(agent_id_count=3)
    with pytest.raises(ValueError, match="cover the full configured team"):
        _validate_identity(
            {"team_id": 0, "agent_id": 0, "team_agent_ids": [0, 1]}, c
        )

def test_env_contracts_validate_identity_extra_key_tolerated():
    c = _ga_core_config()
    _validate_identity(
        {"team_id": 0, "agent_id": 0, "agent_order": [0, 1], "extra": "x"}, c
    )

# ===================== SECTION B =====================

_GB_TIMESTAMP = "20260704T120000Z"

def _gb_core() -> MAPPOCoreConfig:
    return default_stage22_core_config()

def _gb_model() -> RecurrentMAPPOActorCritic:
    return RecurrentMAPPOActorCritic(_gb_core())

def _gb_env(**over) -> Stage22DevelopmentEnvironment:
    return make_default_development_environment() if not over else Stage22DevelopmentEnvironment(
        DevelopmentAdapterConfig(core_config=_gb_core(), **over)
    )

def _gb_collect(model=None, env=None, **rc):
    model = model if model is not None else _gb_model()
    env = env if env is not None else make_default_development_environment()
    config = RolloutCollectionConfig(**rc) if rc else RolloutCollectionConfig(max_episodes=1)
    return collect_development_rollout(model, env, config)

def _gb_algorithm() -> AlgorithmConfig:
    return AlgorithmConfig(discount_factor=0.95, gae_lambda=0.9, ppo_clip_range=0.2)

def _gb_losses() -> LossConfig:
    return LossConfig(
        reward_entropy_coefficient=0.0,
        movement_entropy_coefficient=0.01,
        sensing_entropy_coefficient=0.01,
        hazard_cost_coefficient=1.0,
        sensing_cost_coefficient=0.5,
        value_loss_coefficient=0.5,
    )

def _gb_lagrange() -> LagrangeConfig:
    return LagrangeConfig(initial_multiplier=0.1, learning_rate=0.2, hazard_budget=0.1)

def _gb_update_config(**over) -> Stage22UpdateConfig:
    kwargs = dict(algorithm=_gb_algorithm(), losses=_gb_losses(), lagrange=_gb_lagrange())
    kwargs.update(over)
    return Stage22UpdateConfig(**kwargs)

def _gb_factorized(**over) -> FactorizedPolicyOutput:
    base = dict(
        sensing_action=torch.zeros(1, 1, dtype=torch.long),
        movement_action=torch.zeros(1, 1, dtype=torch.long),
        sensing_log_probability=torch.zeros(1, 1),
        movement_log_probability=torch.zeros(1, 1),
        sensing_entropy=torch.zeros(1, 1),
        movement_entropy=torch.zeros(1, 1),
        sensing_action_count=2,
        movement_action_count=3,
    )
    base.update(over)
    return FactorizedPolicyOutput(**base)

def _gb_run(tmp_path: Path, **over) -> Path:
    kwargs = dict(result_parent=tmp_path, timestamp_utc=_GB_TIMESTAMP)
    kwargs.update(over)
    return _gb_dr.run_stage22_development_training(**kwargs)

def test_model_ActorInput_happy_path_construction():
    ao = torch.zeros(2, 3, 4)
    ri = torch.zeros(2, 3, 2)
    inp = ActorInput(ao, ri, None)
    assert inp.actor_observation is ao
    assert inp.revealed_information is ri
    assert inp.history_state is None

def test_model_ActorInput_rejects_extra_fields():
    with pytest.raises(ValueError, match="executable ActorInput accepts only"):
        ActorInput(torch.zeros(1, 1, 4), torch.zeros(1, 1, 2), None, {"revealed_local_x": torch.zeros(1)})

def test_model_ActorInput_rejects_non_tensor_actor_observation():
    with pytest.raises(TypeError, match="actor_observation must be a torch.Tensor"):
        ActorInput(object(), torch.zeros(1, 1, 2))

def test_model_ActorInput_rejects_wrong_ndim():
    with pytest.raises(ValueError, match="actor_observation must have 3 dimensions"):
        ActorInput(torch.zeros(1, 4), torch.zeros(1, 1, 2))

def test_model_ActorInput_rejects_zero_batch():
    with pytest.raises(ValueError, match="batch and time dimensions must be positive"):
        ActorInput(torch.zeros(0, 3, 4), torch.zeros(0, 3, 2))

def test_model_ActorInput_rejects_reveal_batch_time_mismatch():
    with pytest.raises(ValueError, match="revealed_information batch/time dimensions must match"):
        ActorInput(torch.zeros(2, 3, 4), torch.zeros(2, 4, 2))

def test_model_ActorInput_rejects_non_finite():
    ao = torch.zeros(1, 1, 4)
    ao[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        ActorInput(ao, torch.zeros(1, 1, 2))

def test_model_ActorInput_from_mapping_happy_path():
    inp = ActorInput.from_mapping(
        {"actor_observation": torch.zeros(1, 1, 4), "revealed_information": torch.zeros(1, 1, 2)}
    )
    assert inp.history_state is None

def test_model_ActorInput_from_mapping_missing_required():
    with pytest.raises(ValueError, match="actor input mapping is missing"):
        ActorInput.from_mapping({"actor_observation": torch.zeros(1, 1, 4)})

def test_model_ActorInput_from_mapping_forbidden_key():
    with pytest.raises(ValueError, match="forbidden centralized"):
        ActorInput.from_mapping(
            {
                "actor_observation": torch.zeros(1, 1, 4),
                "revealed_information": torch.zeros(1, 1, 2),
                "central_state": torch.zeros(1),
            }
        )

def test_model_ActorInput_from_mapping_extra_reveal_field_rejected():
    with pytest.raises(ValueError, match="pack reveal content"):
        ActorInput.from_mapping(
            {
                "actor_observation": torch.zeros(1, 1, 4),
                "revealed_information": torch.zeros(1, 1, 2),
                "revealed_local_hazard": torch.zeros(1),
            }
        )

def test_model_CriticInput_happy_path():
    ci = CriticInput(torch.zeros(2, 3, 7))
    assert ci.central_state.shape == (2, 3, 7)

def test_model_CriticInput_rejects_non_tensor():
    with pytest.raises(TypeError, match="central_state must be a torch.Tensor"):
        CriticInput(object())

def test_model_FactorizedPolicyOutput_happy_path_and_joint_properties():
    out = _gb_factorized(
        sensing_log_probability=torch.tensor([[-0.5]]),
        movement_log_probability=torch.tensor([[-0.25]]),
        sensing_entropy=torch.tensor([[0.3]]),
        movement_entropy=torch.tensor([[0.7]]),
    )
    assert torch.allclose(out.joint_log_probability, torch.tensor([[-0.75]]))
    assert torch.allclose(out.joint_entropy, torch.tensor([[1.0]]))

def test_model_FactorizedPolicyOutput_count_must_be_exact_int():
    with pytest.raises(TypeError, match="sensing_action_count must be an int"):
        _gb_factorized(sensing_action_count=True)

def test_model_FactorizedPolicyOutput_count_min_two():
    with pytest.raises(ValueError, match="sensing_action_count must be an integer >= 2"):
        _gb_factorized(sensing_action_count=1)

def test_model_FactorizedPolicyOutput_action_must_be_integer_dtype():
    with pytest.raises(TypeError, match="sensing_action must use an integer dtype"):
        _gb_factorized(sensing_action=torch.zeros(1, 1, dtype=torch.float32))

def test_model_FactorizedPolicyOutput_action_out_of_range():
    with pytest.raises(ValueError, match="out-of-range action index"):
        _gb_factorized(sensing_action=torch.full((1, 1), 5, dtype=torch.long))

def test_model_FactorizedPolicyOutput_action_negative():
    with pytest.raises(ValueError, match="negative action index"):
        _gb_factorized(sensing_action=torch.full((1, 1), -1, dtype=torch.long))

def test_model_FactorizedPolicyOutput_entropy_nonnegative():
    with pytest.raises(ValueError, match="sensing_entropy must contain only nonnegative"):
        _gb_factorized(sensing_entropy=torch.full((1, 1), -1.0))

def test_model_FactorizedPolicyOutput_float_dtype_coherence():
    with pytest.raises(ValueError, match="dtype must match sensing_log_probability dtype"):
        _gb_factorized(movement_log_probability=torch.zeros(1, 1, dtype=torch.float64))

def test_model_FactorizedPolicyOutput_non_tensor_field():
    with pytest.raises(TypeError, match="sensing_action must be a torch.Tensor"):
        _gb_factorized(sensing_action=object())

def test_model_FactorizedPolicyOutput_sparse_layout_rejected():
    with pytest.raises(ValueError, match="strided tensor layout"):
        _gb_factorized(sensing_action=torch.zeros(1, 1).to_sparse())

def test_model_FactorizedPolicyOutput_meta_tensor_rejected():
    with pytest.raises(ValueError, match="must not be a meta tensor"):
        FactorizedPolicyOutput(
            sensing_action=torch.zeros(1, 1, dtype=torch.long, device="meta"),
            movement_action=torch.zeros(1, 1, dtype=torch.long, device="meta"),
            sensing_log_probability=torch.zeros(1, 1, device="meta"),
            movement_log_probability=torch.zeros(1, 1, device="meta"),
            sensing_entropy=torch.zeros(1, 1, device="meta"),
            movement_entropy=torch.zeros(1, 1, device="meta"),
            sensing_action_count=2,
            movement_action_count=3,
        )

def test_model_FactorizedPolicyOutput_shape_must_be_2d():
    with pytest.raises(ValueError, match=r"shape \[batch, time\]"):
        _gb_factorized(
            sensing_action=torch.zeros(1, 1, 1, dtype=torch.long),
            movement_action=torch.zeros(1, 1, 1, dtype=torch.long),
            sensing_log_probability=torch.zeros(1, 1, 1),
            movement_log_probability=torch.zeros(1, 1, 1),
            sensing_entropy=torch.zeros(1, 1, 1),
            movement_entropy=torch.zeros(1, 1, 1),
        )

def test_model_FactorizedPolicyOutput_zero_dims_rejected():
    with pytest.raises(ValueError, match="positive batch and time dimensions"):
        _gb_factorized(
            sensing_action=torch.zeros(0, 1, dtype=torch.long),
            movement_action=torch.zeros(0, 1, dtype=torch.long),
            sensing_log_probability=torch.zeros(0, 1),
            movement_log_probability=torch.zeros(0, 1),
            sensing_entropy=torch.zeros(0, 1),
            movement_entropy=torch.zeros(0, 1),
        )

def test_model_RecurrentMAPPOActorCritic_rejects_wrong_config_type():
    with pytest.raises(TypeError, match="config must be a MAPPOCoreConfig"):
        RecurrentMAPPOActorCritic(object())

def test_model_forward_actor_shapes():
    cfg = _gb_core()
    model = RecurrentMAPPOActorCritic(cfg)
    ao = torch.zeros(2, 3, cfg.actor_observation_dim)
    ri = torch.zeros(2, 3, cfg.revealed_information_dim)
    sl, ml, final = model.forward_actor(ActorInput(ao, ri, None))
    assert sl.shape == (2, 3, cfg.sensing_action_count)
    assert ml.shape == (2, 3, cfg.movement_action_count)
    assert final.shape == (cfg.recurrent_layer_count, 2, cfg.history_state_dim)

def test_model_forward_actor_rejects_non_actor_input():
    with pytest.raises(TypeError, match="actor_input must be an ActorInput"):
        _gb_model().forward_actor(object())

def test_model_forward_actor_rejects_dtype_mismatch():
    cfg = _gb_core()
    model = RecurrentMAPPOActorCritic(cfg)
    ao = torch.zeros(1, 1, cfg.actor_observation_dim, dtype=torch.float64)
    ri = torch.zeros(1, 1, cfg.revealed_information_dim)
    with pytest.raises(ValueError, match="actor_observation dtype must match model parameter dtype"):
        model.forward_actor(ActorInput(ao, ri))

def test_model_zero_initial_gru_state_when_history_none():
    # HAND-COMPUTED regression: None history must produce the SAME logits as an
    # explicit all-zero recurrent state of shape [layers, batch, history_dim].
    cfg = _gb_core()
    model = RecurrentMAPPOActorCritic(cfg)
    model.eval()
    torch.manual_seed(0)
    ao = torch.randn(1, 4, cfg.actor_observation_dim)
    ri = torch.randn(1, 4, cfg.revealed_information_dim)
    sl_none, ml_none, fs_none = model.forward_actor(ActorInput(ao, ri, None))
    zero_h = torch.zeros(cfg.recurrent_layer_count, 1, cfg.history_state_dim)
    sl_zero, ml_zero, fs_zero = model.forward_actor(ActorInput(ao, ri, zero_h))
    assert torch.allclose(sl_none, sl_zero)
    assert torch.allclose(ml_none, ml_zero)
    assert torch.allclose(fs_none, fs_zero)

def test_model_stepwise_vs_full_sequence_gru_consistency():
    # HAND-COMPUTED regression: unrolling the GRU one step at a time (threading
    # the returned state) reproduces the full-sequence logits and log-probs, which
    # protects the update-time log-prob recomputation.
    cfg = _gb_core()
    model = RecurrentMAPPOActorCritic(cfg)
    model.eval()
    torch.manual_seed(1)
    T = 4
    ao = torch.randn(1, T, cfg.actor_observation_dim)
    ri = torch.randn(1, T, cfg.revealed_information_dim)
    sl_full, ml_full, _ = model.forward_actor(ActorInput(ao, ri, None))

    history = None
    step_sl = []
    step_ml = []
    for t in range(T):
        slt, mlt, history = model.forward_actor(
            ActorInput(ao[:, t : t + 1], ri[:, t : t + 1], history)
        )
        step_sl.append(slt)
        step_ml.append(mlt)
    sl_step = torch.cat(step_sl, dim=1)
    ml_step = torch.cat(step_ml, dim=1)
    assert torch.allclose(sl_full, sl_step, atol=1e-5)
    assert torch.allclose(ml_full, ml_step, atol=1e-5)

    # log-prob consistency over the argmax actions
    from torch.distributions import Categorical

    sa = sl_full.argmax(dim=-1)
    lp_full = Categorical(logits=sl_full).log_prob(sa)
    lp_step = Categorical(logits=sl_step).log_prob(sa)
    assert torch.allclose(lp_full, lp_step, atol=1e-5)

def test_model_evaluate_reward_critic_value_shape():
    # HAND-COMPUTED regression: critic value squeezes the trailing 1 to [batch, time].
    cfg = _gb_core()
    model = RecurrentMAPPOActorCritic(cfg)
    value = model.evaluate_reward_critic(CriticInput(torch.zeros(2, 3, cfg.central_state_dim)))
    assert value.shape == (2, 3)

def test_model_evaluate_cost_critic_value_shape():
    cfg = _gb_core()
    model = RecurrentMAPPOActorCritic(cfg)
    value = model.evaluate_cost_critic(CriticInput(torch.zeros(5, 1, cfg.central_state_dim)))
    assert value.shape == (5, 1)

def test_model_evaluate_reward_critic_rejects_non_critic_input():
    with pytest.raises(TypeError, match="critic_input must be a CriticInput"):
        _gb_model().evaluate_reward_critic(object())

def test_model_evaluate_cost_critic_rejects_non_critic_input():
    with pytest.raises(TypeError, match="critic_input must be a CriticInput"):
        _gb_model().evaluate_cost_critic(object())

def test_model_forward_returns_all_keys():
    cfg = _gb_core()
    model = RecurrentMAPPOActorCritic(cfg)
    out = model.forward(
        ActorInput(torch.zeros(2, 1, cfg.actor_observation_dim), torch.zeros(2, 1, cfg.revealed_information_dim)),
        CriticInput(torch.zeros(2, 1, cfg.central_state_dim)),
    )
    assert set(out) == {
        "sensing_logits",
        "movement_logits",
        "next_recurrent_state",
        "policy",
        "reward_value",
        "hazard_cost_value",
    }
    assert isinstance(out["policy"], FactorizedPolicyOutput)
    assert out["reward_value"].shape == (2, 1)

def test_model_policy_from_logits_argmax_deterministic():
    # HAND-COMPUTED regression: with no actions and sample_actions False the
    # chosen action is the per-position argmax of the logits.
    sl = torch.tensor([[[0.1, 5.0]]])
    ml = torch.tensor([[[9.0, 0.0, 0.0]]])
    out = policy_from_logits(sl, ml)
    assert out.sensing_action.tolist() == [[1]]
    assert out.movement_action.tolist() == [[0]]
    assert out.sensing_action_count == 2
    assert out.movement_action_count == 3

def test_model_policy_from_logits_seeded_sampling_reproducible():
    sl = torch.randn(1, 4, 2)
    ml = torch.randn(1, 4, 3)
    torch.manual_seed(123)
    o1 = policy_from_logits(sl, ml, sample_actions=True)
    torch.manual_seed(123)
    o2 = policy_from_logits(sl, ml, sample_actions=True)
    assert torch.equal(o1.sensing_action, o2.sensing_action)
    assert torch.equal(o1.movement_action, o2.movement_action)

def test_model_policy_from_logits_provided_action_passthrough():
    sl = torch.randn(1, 2, 2)
    ml = torch.randn(1, 2, 3)
    out = policy_from_logits(
        sl, ml, sensing_action=torch.tensor([[0, 1]]), movement_action=torch.tensor([[2, 0]])
    )
    assert out.sensing_action.tolist() == [[0, 1]]
    assert out.movement_action.tolist() == [[2, 0]]
    assert out.sensing_action.dtype == torch.long

def test_model_policy_from_logits_provided_action_out_of_range():
    sl = torch.randn(1, 2, 2)
    ml = torch.randn(1, 2, 3)
    with pytest.raises(ValueError, match="out-of-range action index"):
        policy_from_logits(sl, ml, sensing_action=torch.tensor([[0, 5]]), movement_action=torch.tensor([[0, 0]]))

def test_model_policy_from_logits_provided_action_wrong_shape():
    sl = torch.randn(1, 2, 2)
    ml = torch.randn(1, 2, 3)
    with pytest.raises(ValueError, match=r"sensing_action must have shape \[batch, time\]"):
        policy_from_logits(sl, ml, sensing_action=torch.tensor([0, 1]), movement_action=torch.tensor([[0, 0]]))

def test_model_policy_from_logits_logits_ndim():
    with pytest.raises(ValueError, match=r"sensing_logits must have shape \[batch, time, action\]"):
        policy_from_logits(torch.zeros(2, 3), torch.zeros(2, 3, 3))

def test_model_policy_from_logits_min_two_actions():
    with pytest.raises(ValueError, match="at least two actions"):
        policy_from_logits(torch.zeros(1, 1, 1), torch.zeros(1, 1, 3))

def test_model_policy_from_logits_non_float_logits():
    with pytest.raises(TypeError, match="floating point dtype"):
        policy_from_logits(torch.zeros(1, 1, 2, dtype=torch.long), torch.zeros(1, 1, 3))

def test_model_policy_from_logits_batch_time_mismatch():
    with pytest.raises(ValueError, match="batch/time dimensions differ"):
        policy_from_logits(torch.zeros(2, 1, 2), torch.zeros(3, 1, 3))

def test_model_policy_from_logits_zero_batch():
    with pytest.raises(ValueError, match="batch and time dimensions must be positive"):
        policy_from_logits(torch.zeros(0, 1, 2), torch.zeros(0, 1, 3))

def test_model_policy_from_logits_sample_actions_non_bool():
    with pytest.raises(TypeError, match="sample_actions must be a bool"):
        policy_from_logits(torch.zeros(1, 1, 2), torch.zeros(1, 1, 3), sample_actions=1)

def test_model_policy_from_logits_meta_logits_rejected():
    with pytest.raises(ValueError, match="must not be a meta tensor"):
        policy_from_logits(torch.randn(1, 1, 2, device="meta"), torch.randn(1, 1, 3, device="meta"))

def test_model_policy_from_logits_dtype_mismatch():
    with pytest.raises(ValueError, match="logits dtype mismatch"):
        policy_from_logits(torch.zeros(1, 1, 2, dtype=torch.float32), torch.zeros(1, 1, 3, dtype=torch.float64))

def test_env_adapter_DevelopmentAdapterConfig_happy_path():
    cfg = DevelopmentAdapterConfig(core_config=_gb_core())
    assert cfg.max_steps == 4
    assert cfg.agent_count == 2
    assert cfg.terminal_after_steps == 4

def test_env_adapter_DevelopmentAdapterConfig_rejects_non_core_config():
    with pytest.raises(TypeError, match="core_config must be a MAPPOCoreConfig"):
        DevelopmentAdapterConfig(core_config=object())

def test_env_adapter_DevelopmentAdapterConfig_max_steps_upper_bound():
    with pytest.raises(ValueError, match="max_steps must be <= 16"):
        DevelopmentAdapterConfig(core_config=_gb_core(), max_steps=17)

def test_env_adapter_DevelopmentAdapterConfig_terminal_after_steps_bound():
    with pytest.raises(ValueError, match="terminal_after_steps must be <= max_steps"):
        DevelopmentAdapterConfig(core_config=_gb_core(), max_steps=4, terminal_after_steps=5)

def test_env_adapter_DevelopmentAdapterConfig_dtype_wrong_type():
    with pytest.raises(TypeError, match="dtype must be a torch.dtype"):
        DevelopmentAdapterConfig(core_config=_gb_core(), dtype="float32")

def test_env_adapter_DevelopmentAdapterConfig_dtype_wrong_value():
    with pytest.raises(ValueError, match="dtype must be torch.float32 or torch.float64"):
        DevelopmentAdapterConfig(core_config=_gb_core(), dtype=torch.float16)

def test_env_adapter_DevelopmentAdapterConfig_device_whitespace():
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        DevelopmentAdapterConfig(core_config=_gb_core(), device=" cpu")

def test_env_adapter_DevelopmentAdapterConfig_device_non_cpu():
    with pytest.raises(ValueError, match="supports only cpu devices"):
        DevelopmentAdapterConfig(core_config=_gb_core(), device="cuda")

def test_env_adapter_DevelopmentAdapterConfig_device_non_string():
    with pytest.raises(TypeError, match="device must be a string"):
        DevelopmentAdapterConfig(core_config=_gb_core(), device=object())

def test_env_adapter_DevelopmentAdapterConfig_max_steps_non_positive():
    with pytest.raises(ValueError, match="max_steps must be positive"):
        DevelopmentAdapterConfig(core_config=_gb_core(), max_steps=0)

def test_env_adapter_default_stage22_core_config_fixed_dims():
    cfg = default_stage22_core_config()
    assert cfg.actor_observation_dim == 4
    assert cfg.revealed_information_dim == 2
    assert cfg.central_state_dim == 7
    assert cfg.agent_id_count == 2
    assert cfg.sensing_action_count == 2
    assert cfg.movement_action_count == 3
    assert cfg.device == "cpu"
    assert cfg.dtype == "float32"

def test_env_adapter_environment_default_config_none():
    env = Stage22DevelopmentEnvironment()
    assert env.step_index == 0
    assert env.is_done is False

def test_env_adapter_environment_rejects_bad_config_type():
    with pytest.raises(TypeError, match="config must be a DevelopmentAdapterConfig or None"):
        Stage22DevelopmentEnvironment(object())

def test_env_adapter_environment_reset_payload_shape():
    env = Stage22DevelopmentEnvironment()
    payloads = env.reset()
    assert isinstance(payloads, list)
    assert len(payloads) == 2
    for payload in payloads:
        assert set(payload) == {"actor_visible", "critic_visible"}

def test_env_adapter_environment_step_after_done_raises():
    env = Stage22DevelopmentEnvironment(
        DevelopmentAdapterConfig(core_config=_gb_core(), max_steps=3, terminal_after_steps=None)
    )
    env.reset()
    sa = torch.zeros(2, dtype=torch.long)
    ma = torch.zeros(2, dtype=torch.long)
    for _ in range(3):
        env.step(sa, ma)
    assert env.is_done is True
    with pytest.raises(RuntimeError, match="environment is done"):
        env.step(sa, ma)

def test_env_adapter_environment_terminal_at_terminal_after_steps():
    env = Stage22DevelopmentEnvironment(
        DevelopmentAdapterConfig(core_config=_gb_core(), max_steps=4, terminal_after_steps=2)
    )
    env.reset()
    sa = torch.zeros(2, dtype=torch.long)
    ma = torch.zeros(2, dtype=torch.long)
    env.step(sa, ma)
    tr = env.step(sa, ma)
    assert tr[0].contract.done_flags["terminal"] is True
    assert tr[0].contract.done_flags["truncated"] is False

def test_env_adapter_environment_truncation_at_max_steps():
    env = Stage22DevelopmentEnvironment(
        DevelopmentAdapterConfig(core_config=_gb_core(), max_steps=3, terminal_after_steps=None)
    )
    env.reset()
    sa = torch.zeros(2, dtype=torch.long)
    ma = torch.zeros(2, dtype=torch.long)
    env.step(sa, ma)
    env.step(sa, ma)
    tr = env.step(sa, ma)
    assert tr[0].contract.done_flags["truncated"] is True
    assert tr[0].contract.done_flags["terminal"] is False

def test_env_adapter_environment_reward_and_cost_table():
    env = Stage22DevelopmentEnvironment()
    env.reset()
    # step 0: agent0 target=(0+0)%3=0 -> move 0 => task 1.0; agent1 target=(0+1)%3=1 -> move 0 => 0.25
    tr = env.step(torch.tensor([1, 0]), torch.tensor([0, 0]))
    rc0 = tr[0].contract.rewards_and_costs
    rc1 = tr[1].contract.rewards_and_costs
    assert rc0["task_reward"] == 1.0
    assert rc1["task_reward"] == 0.25
    # sensing_cost 0.1 when sensing_action > 0 else 0.0
    assert rc0["sensing_cost"] == 0.1
    assert rc1["sensing_cost"] == 0.0

def test_env_adapter_environment_hazard_cost_charged():
    env = Stage22DevelopmentEnvironment()
    env.reset()
    # agent0 hazard when move == (agent0+1)%3 == 1
    tr = env.step(torch.tensor([0, 0]), torch.tensor([1, 0]))
    assert tr[0].contract.rewards_and_costs["hazard_cost"] == 0.4

def test_env_adapter_environment_reveal_follows_sensing():
    env = Stage22DevelopmentEnvironment()
    env.reset()
    tr = env.step(torch.tensor([1, 0]), torch.tensor([0, 0]))
    sensed = tr[0].next_actor_visible["revealed_information"]
    unsensed = tr[1].next_actor_visible["revealed_information"]
    assert float(sensed[0].item()) == 1.0
    assert float(unsensed[0].item()) == 0.0

def test_env_adapter_environment_step_action_vector_wrong_length():
    env = Stage22DevelopmentEnvironment()
    env.reset()
    with pytest.raises(ValueError):
        env.step(torch.zeros(3, dtype=torch.long), torch.zeros(2, dtype=torch.long))

def test_env_adapter_transitions_to_tensors_happy_path_shapes():
    cfg = _gb_core()
    env = Stage22DevelopmentEnvironment()
    env.reset()
    tr = env.step(torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long))
    out = transitions_to_tensors(tr, cfg)
    assert out["actor_observation"].shape == (2, 1, cfg.actor_observation_dim)
    assert out["revealed_information"].shape == (2, 1, cfg.revealed_information_dim)
    assert out["central_state"].shape == (2, 1, cfg.central_state_dim)
    assert out["terminal"].shape == (2, 1)
    assert out["truncation"].shape == (2, 1)
    assert bool(out["valid_mask"].all().item())

def test_env_adapter_transitions_to_tensors_rejects_non_core_config():
    with pytest.raises(TypeError, match="config must be a MAPPOCoreConfig"):
        transitions_to_tensors([], object())

def test_env_adapter_transitions_to_tensors_rejects_generator():
    with pytest.raises(TypeError, match="must be a list or tuple"):
        transitions_to_tensors((x for x in []), _gb_core())

def test_env_adapter_transitions_to_tensors_rejects_empty():
    with pytest.raises(ValueError, match="transitions must be nonempty"):
        transitions_to_tensors([], _gb_core())

def test_env_adapter_transitions_to_tensors_rejects_non_transition_item():
    with pytest.raises(TypeError, match="DevelopmentTransition"):
        transitions_to_tensors([object()], _gb_core())

def test_rollout_RolloutCollectionConfig_defaults():
    cfg = RolloutCollectionConfig()
    assert cfg.max_episodes == 2
    assert cfg.max_steps_per_episode == 4
    assert cfg.sample_actions is False
    assert cfg.seed == 22

def test_rollout_RolloutCollectionConfig_max_episodes_cap():
    with pytest.raises(ValueError, match="max_episodes must be <= 4"):
        RolloutCollectionConfig(max_episodes=5)

def test_rollout_RolloutCollectionConfig_max_steps_cap():
    with pytest.raises(ValueError, match="max_steps_per_episode must be <= 16"):
        RolloutCollectionConfig(max_steps_per_episode=17)

def test_rollout_RolloutCollectionConfig_seed_zero_reserved():
    with pytest.raises(ValueError, match="seed 0 is intentionally"):
        RolloutCollectionConfig(seed=0)

def test_rollout_RolloutCollectionConfig_seed_too_large():
    with pytest.raises(ValueError, match="signed 64-bit integer"):
        RolloutCollectionConfig(seed=2**63)

def test_rollout_RolloutCollectionConfig_sample_actions_non_bool():
    with pytest.raises(TypeError, match="sample_actions must be a bool"):
        RolloutCollectionConfig(sample_actions=1)

def test_rollout_RolloutCollectionConfig_max_episodes_non_positive():
    with pytest.raises(ValueError, match="max_episodes must be positive"):
        RolloutCollectionConfig(max_episodes=0)

def test_rollout_collect_development_rollout_type_errors_model():
    with pytest.raises(TypeError, match="model must be a RecurrentMAPPOActorCritic"):
        collect_development_rollout(object(), make_default_development_environment())

def test_rollout_collect_development_rollout_type_errors_env():
    with pytest.raises(TypeError, match="environment must be a Stage22DevelopmentEnvironment"):
        collect_development_rollout(_gb_model(), object())

def test_rollout_collect_development_rollout_type_errors_config():
    with pytest.raises(TypeError, match="rollout_config must be a RolloutCollectionConfig or None"):
        collect_development_rollout(_gb_model(), make_default_development_environment(), object())

def test_rollout_collect_development_rollout_batch_and_summary():
    collected = _gb_collect(max_episodes=2, max_steps_per_episode=4)
    assert isinstance(collected, CollectedRollout)
    assert isinstance(collected.batch, RolloutBatch)
    # batch axis == episodes * agents == 2 * 2 == 4
    assert collected.batch.actor_observation.shape[0] == 4
    assert collected.batch.actor_observation.shape[1] == 4
    assert collected.summary["development_only"] is True
    assert collected.summary["rng_state_preserved"] is False
    assert collected.summary["episode_count"] == 2
    assert collected.summary["agent_count"] == 2

def test_rollout_collect_development_rollout_restores_model_mode():
    model = _gb_model()
    model.train()
    was_training = model.training
    collect_development_rollout(model, make_default_development_environment(), RolloutCollectionConfig(max_episodes=1))
    assert model.training == was_training

def test_rollout_collect_development_rollout_eval_mode_restored_when_eval():
    model = _gb_model()
    model.eval()
    collect_development_rollout(model, make_default_development_environment(), RolloutCollectionConfig(max_episodes=1))
    assert model.training is False

def test_rollout_collect_development_rollout_seeded_sampling_deterministic():
    model_a = _gb_model()
    coll_a = collect_development_rollout(
        model_a, make_default_development_environment(),
        RolloutCollectionConfig(max_episodes=2, sample_actions=True, seed=99),
    )
    # Same model params + same seed on a fresh identical model => identical actions.
    model_b = _gb_model()
    model_b.load_state_dict(model_a.state_dict())
    coll_b = collect_development_rollout(
        model_b, make_default_development_environment(),
        RolloutCollectionConfig(max_episodes=2, sample_actions=True, seed=99),
    )
    assert torch.equal(coll_a.batch.sensing_action, coll_b.batch.sensing_action)
    assert torch.equal(coll_a.batch.movement_action, coll_b.batch.movement_action)

def test_rollout_collect_development_rollout_terminal_zeroes_next_values():
    # When a step is terminal the next-value bootstrap is zeroed.
    model = _gb_model()
    env = Stage22DevelopmentEnvironment(
        DevelopmentAdapterConfig(core_config=_gb_core(), max_steps=2, terminal_after_steps=2)
    )
    collected = collect_development_rollout(model, env, RolloutCollectionConfig(max_episodes=1, max_steps_per_episode=2))
    batch = collected.batch
    terminal_mask = batch.terminal
    if terminal_mask.any():
        assert torch.allclose(batch.next_reward_value[terminal_mask], torch.zeros_like(batch.next_reward_value[terminal_mask]))
        assert torch.allclose(batch.next_hazard_cost_value[terminal_mask], torch.zeros_like(batch.next_hazard_cost_value[terminal_mask]))

def test_rollout_validate_reset_payloads_non_list():
    with pytest.raises(TypeError, match="must return a list"):
        _validate_reset_payloads_for_rollout("nope", _gb_core(), 2)

def test_rollout_validate_reset_payloads_empty():
    with pytest.raises(ValueError, match="payloads must be nonempty"):
        _validate_reset_payloads_for_rollout([], _gb_core(), 2)

def test_rollout_validate_reset_payloads_wrong_count():
    with pytest.raises(ValueError, match="payload count must match agent_count"):
        _validate_reset_payloads_for_rollout(
            [{"actor_visible": {}, "critic_visible": {}}], _gb_core(), 2
        )

def test_rollout_validate_reset_payloads_missing_actor_visible():
    with pytest.raises(ValueError, match="missing actor_visible"):
        _validate_reset_payloads_for_rollout(
            [{"critic_visible": {}}, {"critic_visible": {}}], _gb_core(), 2
        )

def test_rollout_make_default_development_environment_type():
    env = make_default_development_environment()
    assert isinstance(env, Stage22DevelopmentEnvironment)
    assert env.config.agent_count == 2

def test_update_Stage22UpdateConfig_defaults():
    cfg = default_stage22_update_config()
    assert cfg.max_update_epochs == 2
    assert cfg.normalize_advantages is False
    assert cfg.max_grad_norm == 1.0

def test_update_Stage22UpdateConfig_rejects_bad_algorithm():
    with pytest.raises(TypeError, match="algorithm must be an AlgorithmConfig"):
        Stage22UpdateConfig(algorithm=object(), losses=_gb_losses(), lagrange=_gb_lagrange())

def test_update_Stage22UpdateConfig_rejects_bad_losses():
    with pytest.raises(TypeError, match="losses must be a LossConfig"):
        Stage22UpdateConfig(algorithm=_gb_algorithm(), losses=object(), lagrange=_gb_lagrange())

def test_update_Stage22UpdateConfig_rejects_bad_lagrange():
    with pytest.raises(TypeError, match="lagrange must be a LagrangeConfig"):
        Stage22UpdateConfig(algorithm=_gb_algorithm(), losses=_gb_losses(), lagrange=object())

def test_update_Stage22UpdateConfig_epoch_cap():
    with pytest.raises(ValueError, match="max_update_epochs must be <= 4"):
        _gb_update_config(max_update_epochs=5)

def test_update_Stage22UpdateConfig_learning_rate_positive():
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        _gb_update_config(learning_rate=0.0)

def test_update_Stage22UpdateConfig_max_grad_norm_positive():
    with pytest.raises(ValueError, match="max_grad_norm must be positive"):
        _gb_update_config(max_grad_norm=0.0)

def test_update_Stage22UpdateConfig_normalize_advantages_non_bool():
    with pytest.raises(TypeError, match="normalize_advantages must be a bool"):
        _gb_update_config(normalize_advantages=1)

def test_update_stage22_ppo_lagrangian_update_rejects_bad_model():
    with pytest.raises(TypeError, match="model must be a RecurrentMAPPOActorCritic"):
        stage22_ppo_lagrangian_update(object(), _gb_collect().batch)

def test_update_stage22_ppo_lagrangian_update_rejects_bad_rollout():
    with pytest.raises(TypeError, match="rollout must be a RolloutBatch"):
        stage22_ppo_lagrangian_update(_gb_model(), object())

def test_update_stage22_ppo_lagrangian_update_rejects_bad_update_config():
    model = _gb_model()
    batch = _gb_collect(model=model).batch
    with pytest.raises(TypeError, match="update_config must be a Stage22UpdateConfig or None"):
        stage22_ppo_lagrangian_update(model, batch, object())

def test_update_stage22_ppo_lagrangian_update_rejects_bad_optimizer():
    model = _gb_model()
    batch = _gb_collect(model=model).batch
    with pytest.raises(TypeError, match="optimizer must be a torch.optim.Optimizer or None"):
        stage22_ppo_lagrangian_update(model, batch, optimizer=object())

def test_update_stage22_ppo_lagrangian_update_rejects_bad_multiplier():
    model = _gb_model()
    batch = _gb_collect(model=model).batch
    with pytest.raises(TypeError, match="lagrange_multiplier must be a LagrangeMultiplier or None"):
        stage22_ppo_lagrangian_update(model, batch, lagrange_multiplier=object())

def test_update_stage22_ppo_lagrangian_update_foreign_optimizer_params():
    model = _gb_model()
    batch = _gb_collect(model=model).batch
    foreign = torch.optim.Adam(_gb_model().parameters(), lr=0.01)
    with pytest.raises(ValueError, match="optimizer parameters must exactly match"):
        stage22_ppo_lagrangian_update(model, batch, optimizer=foreign)

def test_update_stage22_ppo_lagrangian_update_parameter_delta_positive():
    # HAND-COMPUTED regression: a real update must move parameters (L1 delta > 0).
    model = _gb_model()
    batch = _gb_collect(model=model).batch
    before = [p.detach().clone() for p in model.parameters()]
    result = stage22_ppo_lagrangian_update(model, batch)
    assert result.summary["parameter_delta_l1"] > 0.0
    manual = sum(float((a - b).abs().sum().item()) for a, b in zip(before, model.parameters()))
    assert manual == pytest.approx(result.summary["parameter_delta_l1"], rel=1e-5, abs=1e-8)

def test_update_stage22_ppo_lagrangian_update_lambda_dual_formula():
    # HAND-COMPUTED regression: lambda_{t+1} = max(0, lambda + eta*(J_C - d)) with
    # J_C the per-step MEAN hazard cost of the rollout.
    model = _gb_model()
    collected = _gb_collect(model=model)
    batch = collected.batch
    lm = LagrangeMultiplier(0.1, 0.2)
    result = stage22_ppo_lagrangian_update(model, batch, lagrange_multiplier=lm)
    observed = cost_estimate(batch)
    assert result.summary["observed_hazard_cost"] == pytest.approx(observed, abs=1e-9)
    expected = max(0.0, 0.1 + 0.2 * (observed - 0.1))
    assert result.lagrange_multiplier.value == pytest.approx(expected, abs=1e-9)
    assert result.summary["lagrange_multiplier_after"] == pytest.approx(expected, abs=1e-9)

def test_update_stage22_ppo_lagrangian_update_leaves_model_in_train_mode():
    model = _gb_model()
    model.eval()
    batch = _gb_collect(model=model).batch
    stage22_ppo_lagrangian_update(model, batch)
    assert model.training is True
    result = stage22_ppo_lagrangian_update(model, _gb_collect(model=model).batch)
    assert result.summary["model_mode_after_update"] == "train"

def test_update_stage22_ppo_lagrangian_update_summary_keys():
    model = _gb_model()
    batch = _gb_collect(model=model).batch
    summary = stage22_ppo_lagrangian_update(model, batch).summary
    for key in (
        "development_only",
        "optimizer_state_saved",
        "parameter_delta_l1",
        "total_loss",
        "policy_loss",
        "reward_value_loss",
        "hazard_cost_value_loss",
        "lagrange_multiplier_before",
        "lagrange_multiplier_after",
        "observed_hazard_cost",
        "hazard_budget",
        "advantages_normalized",
    ):
        assert key in summary
    assert summary["optimizer_state_saved"] is False

def test_update_stage22_ppo_lagrangian_update_owned_optimizer_flag():
    model = _gb_model()
    batch = _gb_collect(model=model).batch
    assert stage22_ppo_lagrangian_update(model, batch).summary["owned_optimizer"] is True
    model2 = _gb_model()
    batch2 = _gb_collect(model=model2).batch
    opt = torch.optim.Adam(model2.parameters(), lr=0.01)
    assert stage22_ppo_lagrangian_update(model2, batch2, optimizer=opt).summary["owned_optimizer"] is False

def test_update_stage22_ppo_lagrangian_update_normalize_flag_recorded():
    model = _gb_model()
    batch = _gb_collect(model=model).batch
    cfg = _gb_update_config(normalize_advantages=True)
    summary = stage22_ppo_lagrangian_update(model, batch, cfg).summary
    assert summary["advantages_normalized"] is True

def test_update_normalise_advantage_population_std():
    # HAND-COMPUTED regression: normalization uses population std (unbiased=False).
    values = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.ones_like(values, dtype=torch.bool)
    out = _normalise_advantage(values, mask)
    mean = values.mean()
    std = values.std(unbiased=False)
    expected = (values - mean) / std.clamp_min(1e-8)
    assert torch.allclose(out, expected)

def test_update_normalise_advantage_zero_std_path():
    values = torch.tensor([[2.0, 2.0, 2.0]])
    mask = torch.ones_like(values, dtype=torch.bool)
    out = _normalise_advantage(values, mask)
    # std == 0 -> returns values - mean == zeros
    assert torch.allclose(out, torch.zeros_like(values))

def test_update_normalise_advantage_empty_mask_raises():
    values = torch.tensor([[1.0, 2.0]])
    mask = torch.zeros_like(values, dtype=torch.bool)
    with pytest.raises(ValueError, match="at least one true timestep"):
        _normalise_advantage(values, mask)

def test_update_masked_mean_float_excludes_masked():
    values = torch.tensor([[1.0, 5.0, 9.0]])
    mask = torch.tensor([[True, False, True]])
    # mean of {1.0, 9.0} == 5.0
    assert _masked_mean_float(values, mask) == pytest.approx(5.0)

def test_update_cost_estimate_is_per_step_mean():
    # D-2 regression: cost_estimate returns the per-step mean hazard cost.
    batch = _gb_collect().batch
    expected = float(batch.hazard_cost[batch.valid_mask].mean().item())
    assert cost_estimate(batch) == pytest.approx(expected, abs=1e-9)

def test_update_development_reward_signal_formula():
    # reward signal == task_reward - sensing_cost_coefficient * sensing_cost.
    batch = _gb_collect().batch
    losses = _gb_losses()
    signal = _development_reward_signal(batch, losses)
    expected = batch.task_reward - losses.sensing_cost_coefficient * batch.sensing_cost
    assert torch.allclose(signal, expected)

def test_update_hazard_advantage_penalizes_policy_objective_invariant():
    assert hazard_advantage_penalizes_policy_objective() is True

def test_development_runner_run_rejects_seed_zero(tmp_path):
    with pytest.raises(ValueError, match="seed 0 is intentionally"):
        _gb_dr.run_stage22_development_training(result_parent=tmp_path, seed=0)

def test_development_runner_run_rejects_episode_count_over_cap(tmp_path):
    with pytest.raises(ValueError, match="episode_count must be <= 4"):
        _gb_dr.run_stage22_development_training(result_parent=tmp_path, episode_count=5)

def test_development_runner_run_rejects_steps_over_cap(tmp_path):
    with pytest.raises(ValueError, match="steps_per_episode must be <= 16"):
        _gb_dr.run_stage22_development_training(result_parent=tmp_path, steps_per_episode=17)

def test_development_runner_run_rejects_non_bool_allow(tmp_path):
    with pytest.raises(TypeError, match="allow_pre_existing_stage22_roots must be a bool"):
        _gb_dr.run_stage22_development_training(result_parent=tmp_path, allow_pre_existing_stage22_roots=1)

def test_development_runner_run_rejects_bad_result_parent_type():
    with pytest.raises(TypeError, match="result_parent must be a str, Path, or None"):
        _gb_dr.run_stage22_development_training(result_parent=123)

def test_development_runner_run_rejects_bad_timestamp_type(tmp_path):
    with pytest.raises(TypeError, match="timestamp_utc must be a string or None"):
        _gb_dr.run_stage22_development_training(result_parent=tmp_path, timestamp_utc=123)

def test_development_runner_run_rejects_bad_timestamp_value(tmp_path):
    with pytest.raises(ValueError, match="timestamp_utc must match"):
        _gb_dr.run_stage22_development_training(result_parent=tmp_path, timestamp_utc="not-a-timestamp")

def test_development_runner_run_rejects_repo_internal_custom_parent():
    with pytest.raises(ValueError, match="custom result_parent must be outside the repository"):
        _gb_dr.run_stage22_development_training(result_parent=Path("."), timestamp_utc=_GB_TIMESTAMP)

def test_development_runner_run_writes_seven_file_set(tmp_path):
    root = _gb_run(tmp_path)
    files = sorted(p.name for p in root.iterdir())
    assert files == sorted(_gb_dr.STAGE22_RESULT_FILES)
    assert len(_gb_dr.STAGE22_RESULT_FILES) == 7

def test_development_runner_run_result_root_name(tmp_path):
    root = _gb_run(tmp_path)
    assert root.name == f"{_gb_dr.STAGE22_RESULT_PREFIX}{_GB_TIMESTAMP}"

def test_development_runner_run_boundary_payload(tmp_path):
    root = _gb_run(tmp_path)
    run_config = json.loads((root / "run_config.json").read_text())
    assert run_config["stage"] == 22
    assert run_config["development_only"] is True
    assert run_config["evaluation_run"] is False
    assert run_config["final_evaluation_run"] is False
    assert run_config["claim_evidence_created"] is False
    assert run_config["claim_status"] == "not yet tested"
    assert run_config["training_run"] == "bounded_stage22_development_only"

def test_development_runner_run_command_record_provenance(tmp_path):
    root = _gb_run(tmp_path)
    cr = json.loads((root / "command_record.json").read_text())
    assert cr["entry_point"] == _gb_dr._ENTRY_POINT
    assert "replay_python_snippet" in cr
    assert cr["command_argv"][1] == "-c"
    assert cr["command_argv"][2] == cr["replay_python_snippet"]
    # replay snippet must parse and validate as the canonical four-statement AST.
    _gb_dr._validate_replay_python_snippet(cr["replay_python_snippet"])
    assert _gb_dr._is_known_stage22_replay_snippet(cr["replay_python_snippet"]) is True

def test_development_runner_run_training_curve_records(tmp_path):
    root = _gb_run(tmp_path, episode_count=2)
    lines = (root / "training_curve.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    for key in (
        "development_only",
        "evaluation_run",
        "episode_index",
        "hazard_cost_mean",
        "lagrange_multiplier",
        "parameter_delta_l1",
        "task_reward_sum",
        "total_loss",
    ):
        assert key in record
    assert record["development_only"] is True
    assert record["evaluation_run"] is False

def test_development_runner_run_artifact_hashes_verify(tmp_path):
    root = _gb_run(tmp_path)
    from raas_marl.mappo_lagrangian.artifacts import sha256_file

    hashes = json.loads((root / "artifact_hashes.json").read_text())["hashed_files"]
    assert set(hashes) == set(_gb_dr._HASHED_RESULT_FILES)
    for name, expected in hashes.items():
        assert sha256_file(root / name) == expected

def test_development_runner_run_duplicate_timestamp_raises(tmp_path):
    _gb_run(tmp_path)
    with pytest.raises(FileExistsError, match="already exists"):
        _gb_run(tmp_path)

def test_development_runner_run_two_distinct_roots_in_external_parent(tmp_path):
    r1 = _gb_run(tmp_path, timestamp_utc="20260704T120000Z")
    r2 = _gb_run(tmp_path, timestamp_utc="20260704T130000Z")
    assert r1.name != r2.name
    assert r1.exists() and r2.exists()

def test_development_runner_strict_artifacts_accepts_clean_root(tmp_path):
    root = _gb_run(tmp_path)
    # Idempotent: re-validating a freshly written clean root does not raise.
    _gb_dr._assert_strict_result_artifacts(root)

def test_development_runner_strict_artifacts_rejects_duplicate_json_key(tmp_path):
    root = _gb_run(tmp_path)
    original = (root / "run_config.json").read_text()
    duplicated = '{"stage":22,"stage":22,' + original[1:]
    (root / "run_config.json").write_text(duplicated)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        _gb_dr._assert_strict_result_artifacts(root)

def test_development_runner_reject_unsafe_boundary_values_in_value():
    with pytest.raises(ValueError, match="unsafe evidence-boundary wording"):
        _gb_dr._reject_unsafe_boundary_values({"note": "claim is supported"}, "run_config.json")

def test_development_runner_reject_unsafe_boundary_values_in_key():
    with pytest.raises(ValueError, match="unsafe evidence-boundary wording in key"):
        _gb_dr._reject_unsafe_boundary_values({"final_evaluation_result": "x"}, "run_config.json")

def test_development_runner_reject_unsafe_boundary_bare_evaluation_word():
    with pytest.raises(ValueError, match="unsafe evaluation wording"):
        _gb_dr._reject_unsafe_boundary_values({"note": "this evaluation happened"}, "run_config.json")

def test_development_runner_reject_unsafe_boundary_allows_negative_evaluation_string():
    # Documented allowed negative-evaluation phrase must pass.
    _gb_dr._reject_unsafe_boundary_values({"note": "no evaluation was run"}, "run_config.json")

def test_development_runner_reject_unsafe_boundary_false_only_key_must_be_false():
    with pytest.raises(ValueError, match="must be False"):
        _gb_dr._reject_unsafe_boundary_values({"evaluation_run": True}, "run_config.json")

def test_development_runner_reject_unsafe_boundary_true_only_key_must_be_true():
    with pytest.raises(ValueError, match="must be True"):
        _gb_dr._reject_unsafe_boundary_values({"development_only": False}, "run_config.json")

def test_development_runner_reject_unsafe_boundary_strict_boolean_type():
    with pytest.raises(ValueError, match="must be a JSON boolean"):
        _gb_dr._reject_unsafe_boundary_values({"development_only": "true"}, "run_config.json")

def test_development_runner_reject_unsafe_boundary_non_string_key():
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        _gb_dr._reject_unsafe_boundary_values({1: "x"}, "run_config.json")

def test_development_runner_reject_unsafe_boundary_stage_must_be_22():
    with pytest.raises(ValueError, match="must be 22"):
        _gb_dr._reject_unsafe_boundary_values({"stage": 23}, "run_config.json")

def test_development_runner_validate_replay_python_snippet_accepts_canonical():
    invocation = {
        "result_parent_argument": None,
        "timestamp_utc": _GB_TIMESTAMP,
        "seed": 22,
        "episode_count": 2,
        "steps_per_episode": 4,
        "allow_pre_existing_stage22_roots": False,
    }
    snippet = _gb_dr._replay_python_snippet(invocation)
    _gb_dr._validate_replay_python_snippet(snippet)
    assert _gb_dr._is_known_stage22_replay_snippet(snippet) is True

def test_development_runner_validate_replay_python_snippet_rejects_non_canonical():
    with pytest.raises(ValueError):
        _gb_dr._validate_replay_python_snippet("import os")

def test_development_runner_is_known_stage22_replay_snippet_rejects_extra_statement():
    invocation = {
        "result_parent_argument": None,
        "timestamp_utc": _GB_TIMESTAMP,
        "seed": 22,
        "episode_count": 2,
        "steps_per_episode": 4,
        "allow_pre_existing_stage22_roots": False,
    }
    snippet = _gb_dr._replay_python_snippet(invocation) + "; import os"
    assert _gb_dr._is_known_stage22_replay_snippet(snippet) is False

def test_buffer_RolloutBatchSpec_is_RolloutBatch_alias():
    assert RolloutBatchSpec is RolloutBatch

# ===================== SECTION C =====================

_GC_TIMESTAMP = "20260704T010203Z"

def _gc_replay_schema() -> gov.ReplaySnippetSchema:
    return gov.ReplaySnippetSchema(
        import_module_path="raas_marl.mappo_lagrangian.stage23b_runner",
        callee_name="run_stage23b_bounded_training",
        keyword_order=(
            "result_parent",
            "timestamp_utc",
            "seed",
            "steps_per_episode",
            "allow_pre_existing_stage23b_roots",
        ),
        keyword_literal_kind={
            "result_parent": "result_parent",
            "timestamp_utc": "timestamp",
            "seed": "positive_int",
            "steps_per_episode": "bounded_positive_int",
            "allow_pre_existing_stage23b_roots": "bool",
        },
        keyword_int_bounds={"steps_per_episode": 16},
    )

def _gc_replay_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "result_parent": None,
        "timestamp_utc": _GC_TIMESTAMP,
        "seed": 23,
        "steps_per_episode": 4,
        "allow_pre_existing_stage23b_roots": False,
    }
    values.update(overrides)
    return values

def _gc_boundary_flags() -> dict[str, object]:
    return gov.boundary_flags(
        stage="23-B",
        training_run="bounded_stage23b_development_only",
        claim_status="not tested / not supported",
    )

def _gc_core_config():
    return default_stage23_core_config()

def _gc_model() -> RecurrentMAPPOActorCritic:
    torch.manual_seed(7)
    return RecurrentMAPPOActorCritic(_gc_core_config())

class _GcContract:
    def __init__(self, identity: dict) -> None:
        self.identity = identity

class _GcTransition:
    def __init__(self, identity: dict) -> None:
        self.contract = _GcContract(identity)

def _gc_make_episode(time_length: int, *, feature_dim: int = 4) -> dict[str, torch.Tensor]:
    """Build one well-formed episode dict for the padded-merge helpers."""

    episode: dict[str, torch.Tensor] = {}
    for name in rc.ROLLOUT_FIELD_NAMES:
        if name in rc._FEATURE_FIELDS:
            episode[name] = torch.zeros((2, time_length, feature_dim))
        elif name in rc._ACTION_FIELDS:
            episode[name] = torch.zeros((2, time_length), dtype=torch.long)
        elif name in rc._BOOL_FIELDS:
            if name == "valid_mask":
                episode[name] = torch.ones((2, time_length), dtype=torch.bool)
            else:
                episode[name] = torch.zeros((2, time_length), dtype=torch.bool)
        else:  # float fields
            episode[name] = torch.zeros((2, time_length))
    return episode

def test_governance_module_import_is_torch_free():
    import raas_marl

    # Absolute src dir (.../src) from the installed package, so this is independent
    # of the test file's location (conftest already put src on the parent path).
    _gc_src_dir = Path(raas_marl.__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, r'" + str(_gc_src_dir) + "'); "
            "import raas_marl.mappo_lagrangian._governance as g; "
            "print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"

def test_governance_boundary_flags_has_seventeen_keys():
    flags = _gc_boundary_flags()
    assert len(flags) == 17
    assert flags["development_only"] is True
    assert flags["stage"] == "23-B"
    assert flags["training_run"] == "bounded_stage23b_development_only"
    assert flags["claim_status"] == "not tested / not supported"

def test_governance_boundary_flags_false_keys_all_false():
    flags = _gc_boundary_flags()
    false_keys = (
        "ablation_run",
        "baseline_comparison_run",
        "checkpoint_created",
        "claim_evidence_created",
        "cross_platform_bitwise_determinism_claimed",
        "evaluation_run",
        "final_evaluation_run",
        "optimizer_state_saved",
        "paper_facing_results_created",
        "replay_buffer_artifact_created",
        "rollout_buffer_artifact_created",
        "serialized_model_artifact_created",
        "statistical_test_run",
    )
    for key in false_keys:
        assert flags[key] is False

def test_governance_boundary_flags_stage_can_be_int():
    flags = gov.boundary_flags(stage=22, training_run="tr", claim_status="cs")
    assert flags["stage"] == 22
    assert type(flags["stage"]) is int

def test_governance_assert_boundary_payload_accepts_matching():
    flags = _gc_boundary_flags()
    gov.assert_boundary_payload(
        flags,
        "p",
        stage="23-B",
        training_run="bounded_stage23b_development_only",
        claim_status="not tested / not supported",
    )

def test_governance_assert_boundary_payload_rejects_flipped_development_only():
    flags = dict(_gc_boundary_flags())
    flags["development_only"] = False
    with pytest.raises(ValueError, match="development_only must equal True"):
        gov.assert_boundary_payload(
            flags,
            "p",
            stage="23-B",
            training_run="bounded_stage23b_development_only",
            claim_status="not tested / not supported",
        )

def test_governance_assert_boundary_payload_rejects_flipped_false_key():
    flags = dict(_gc_boundary_flags())
    flags["evaluation_run"] = True
    with pytest.raises(ValueError, match="evaluation_run must equal False"):
        gov.assert_boundary_payload(
            flags,
            "p",
            stage="23-B",
            training_run="bounded_stage23b_development_only",
            claim_status="not tested / not supported",
        )

def test_governance_assert_boundary_payload_rejects_wrong_stage():
    flags = _gc_boundary_flags()
    with pytest.raises(ValueError, match="stage must equal"):
        gov.assert_boundary_payload(
            flags,
            "p",
            stage="24-A",
            training_run="bounded_stage23b_development_only",
            claim_status="not tested / not supported",
        )

def test_governance_assert_boundary_payload_rejects_missing_key():
    flags = dict(_gc_boundary_flags())
    del flags["evaluation_run"]
    with pytest.raises(ValueError, match="missing required field"):
        gov.assert_boundary_payload(
            flags,
            "p",
            stage="23-B",
            training_run="bounded_stage23b_development_only",
            claim_status="not tested / not supported",
        )

@pytest.mark.parametrize(
    "phrase",
    [
        "bayesian belief",
        "formal voi",
        "c rc mappo voi",
        "final evaluation",
        "claim evidence",
        "paper facing",
    ],
)
def test_governance_reject_unsafe_boundary_values_rejects_terminology_locks(phrase):
    with pytest.raises(ValueError):
        gov.reject_unsafe_boundary_values({"note": phrase}, "p")

@pytest.mark.parametrize(
    "phrase",
    [
        "Bayesian  Belief",
        "FINAL_EVALUATION",
        "Claim-Evidence",
        "paper   facing",
        "C-RC-MAPPO-VOI",
        "Formal-VOI",
    ],
)
def test_governance_reject_unsafe_boundary_values_case_and_space_variants(phrase):
    with pytest.raises(ValueError):
        gov.reject_unsafe_boundary_values({"note": phrase}, "p")

def test_governance_reject_unsafe_boundary_values_rejects_bare_evaluation_word():
    with pytest.raises(ValueError, match="unsafe evaluation wording"):
        gov.reject_unsafe_boundary_values({"note": "the evaluation is done"}, "p")

def test_governance_reject_unsafe_boundary_values_allows_negative_evaluation_string():
    gov.reject_unsafe_boundary_values(
        {"note": "stage 23 b is not evaluation"}, "p"
    )

def test_governance_reject_unsafe_boundary_values_allows_clean_payload():
    gov.reject_unsafe_boundary_values(
        {"note": "bounded development only", "n": 3, "flag": True}, "p"
    )

def test_governance_reject_unsafe_boundary_values_scans_keys():
    with pytest.raises(ValueError, match="in key"):
        gov.reject_unsafe_boundary_values({"final evaluation": 1}, "p")

def test_governance_reject_unsafe_boundary_values_non_string_key_type_error():
    with pytest.raises(TypeError, match="mapping keys must be strings"):
        gov.reject_unsafe_boundary_values({5: "x"}, "p")

def test_governance_reject_unsafe_boundary_values_recurses_into_lists():
    with pytest.raises(ValueError):
        gov.reject_unsafe_boundary_values({"items": ["ok", "final evaluation"]}, "p")

def test_governance_reject_unsafe_boundary_values_allows_structural_keys():
    # A benign structural key that would otherwise trip the scanner is allowed.
    gov.reject_unsafe_boundary_values(
        {"evaluation_run": False},
        "p",
        allowed_structural_keys=frozenset({"evaluation_run"}),
    )

def test_governance_unsafe_boundary_phrases_contains_locks():
    phrases = set(gov.UNSAFE_BOUNDARY_PHRASES)
    for lock in (
        "final evaluation",
        "claim evidence",
        "paper facing",
        "bayesian belief",
        "formal voi",
        "c rc mappo voi",
    ):
        assert lock in phrases

def test_governance_normalise_boundary_string_collapses_nonalphanumerics():
    assert gov.normalise_boundary_string("Final__Evaluation!!") == "final evaluation"
    assert gov.normalise_boundary_string("  a  b  ") == "a b"

def test_governance_build_replay_python_snippet_round_trips_none_result_parent():
    schema = _gc_replay_schema()
    snippet = gov.build_replay_python_snippet(schema, _gc_replay_values())
    parsed = gov.parse_replay_python_snippet(snippet, schema)
    assert parsed["result_parent"] is None
    assert parsed["seed"] == 23
    assert parsed["steps_per_episode"] == 4
    assert parsed["timestamp_utc"] == _GC_TIMESTAMP
    assert parsed["allow_pre_existing_stage23b_roots"] is False

def test_governance_build_replay_python_snippet_round_trips_string_result_parent():
    schema = _gc_replay_schema()
    snippet = gov.build_replay_python_snippet(
        schema, _gc_replay_values(result_parent="/external/parent")
    )
    parsed = gov.parse_replay_python_snippet(snippet, schema)
    assert parsed["result_parent"] == "/external/parent"

def test_governance_build_replay_python_snippet_missing_keyword_value():
    schema = _gc_replay_schema()
    values = _gc_replay_values()
    del values["seed"]
    with pytest.raises(ValueError, match="missing"):
        gov.build_replay_python_snippet(schema, values)

def test_governance_parse_replay_python_snippet_rejects_non_literal():
    schema = _gc_replay_schema()
    snippet = gov.build_replay_python_snippet(schema, _gc_replay_values())
    bad = snippet.replace("seed=23", "seed=1+1")
    with pytest.raises(ValueError, match="must be a safe literal"):
        gov.parse_replay_python_snippet(bad, schema)

def test_governance_parse_replay_python_snippet_rejects_extra_statement():
    schema = _gc_replay_schema()
    snippet = gov.build_replay_python_snippet(schema, _gc_replay_values())
    with pytest.raises(ValueError, match="canonical four statements"):
        gov.parse_replay_python_snippet(snippet + "; x = 1", schema)

def test_governance_parse_replay_python_snippet_rejects_wrong_keyword_order():
    schema = _gc_replay_schema()
    wrong = (
        "import sys; sys.path.insert(0, 'src'); "
        "from raas_marl.mappo_lagrangian.stage23b_runner import "
        "run_stage23b_bounded_training; "
        "run_stage23b_bounded_training(timestamp_utc='20260704T010203Z', "
        "result_parent=None, seed=23, steps_per_episode=4, "
        "allow_pre_existing_stage23b_roots=False)"
    )
    with pytest.raises(ValueError, match="canonical order"):
        gov.parse_replay_python_snippet(wrong, schema)

def test_governance_parse_replay_python_snippet_rejects_positional_argument():
    schema = _gc_replay_schema()
    snippet = (
        "import sys; sys.path.insert(0, 'src'); "
        "from raas_marl.mappo_lagrangian.stage23b_runner import "
        "run_stage23b_bounded_training; "
        "run_stage23b_bounded_training(None, timestamp_utc='20260704T010203Z', "
        "seed=23, steps_per_episode=4, allow_pre_existing_stage23b_roots=False)"
    )
    with pytest.raises(ValueError, match="positional"):
        gov.parse_replay_python_snippet(snippet, schema)

def test_governance_parse_replay_python_snippet_rejects_non_string():
    schema = _gc_replay_schema()
    with pytest.raises(TypeError, match="non-empty string"):
        gov.parse_replay_python_snippet(123, schema)

def test_governance_parse_replay_python_snippet_rejects_syntax_error():
    schema = _gc_replay_schema()
    with pytest.raises(ValueError, match="syntactically valid Python"):
        gov.parse_replay_python_snippet("import sys; def (", schema)

def test_governance_parse_replay_python_snippet_rejects_bounded_int_over_max():
    schema = _gc_replay_schema()
    snippet = gov.build_replay_python_snippet(schema, _gc_replay_values())
    bad = snippet.replace("steps_per_episode=4", "steps_per_episode=99")
    with pytest.raises(ValueError, match="<= 16"):
        gov.parse_replay_python_snippet(bad, schema)

def test_governance_parse_replay_python_snippet_rejects_bad_bool_literal():
    schema = _gc_replay_schema()
    snippet = gov.build_replay_python_snippet(schema, _gc_replay_values())
    bad = snippet.replace(
        "allow_pre_existing_stage23b_roots=False",
        "allow_pre_existing_stage23b_roots=0",
    )
    with pytest.raises(TypeError, match="must be bool"):
        gov.parse_replay_python_snippet(bad, schema)

def test_governance_command_argv_from_snippet_prefixes_executable():
    schema = _gc_replay_schema()
    snippet = gov.build_replay_python_snippet(schema, _gc_replay_values())
    argv = gov.command_argv_from_snippet(snippet, schema)
    assert argv[0] == sys.executable
    assert argv[1] == "-c"
    assert argv[2] == snippet

def test_governance_replay_snippet_schema_rejects_empty_module_path():
    with pytest.raises(TypeError, match="import_module_path"):
        gov.ReplaySnippetSchema(
            import_module_path="",
            callee_name="c",
            keyword_order=("a",),
            keyword_literal_kind={"a": "bool"},
        )

def test_governance_replay_snippet_schema_rejects_kind_coverage_mismatch():
    with pytest.raises(ValueError, match="cover exactly keyword_order"):
        gov.ReplaySnippetSchema(
            import_module_path="m",
            callee_name="c",
            keyword_order=("a",),
            keyword_literal_kind={"b": "bool"},
        )

def test_governance_replay_snippet_schema_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unsupported replay keyword literal kind"):
        gov.ReplaySnippetSchema(
            import_module_path="m",
            callee_name="c",
            keyword_order=("a",),
            keyword_literal_kind={"a": "weird"},
        )

def test_governance_replay_snippet_schema_rejects_non_tuple_keyword_order():
    with pytest.raises(TypeError, match="keyword_order must be a tuple"):
        gov.ReplaySnippetSchema(
            import_module_path="m",
            callee_name="c",
            keyword_order=["a"],
            keyword_literal_kind={"a": "bool"},
        )

def test_governance_reject_duplicate_json_pairs_rejects_duplicate():
    with pytest.raises(ValueError, match="duplicate JSON key: a"):
        gov.reject_duplicate_json_pairs([("a", 1), ("a", 2)])

def test_governance_reject_duplicate_json_pairs_accepts_unique():
    assert gov.reject_duplicate_json_pairs([("a", 1), ("b", 2)]) == {"a": 1, "b": 2}

def test_governance_write_json_and_load_strict_json_object_round_trip(tmp_path):
    path = tmp_path / "x.json"
    gov.write_json(path, {"b": 2, "a": 1})
    assert gov.load_strict_json_object(path) == {"a": 1, "b": 2}
    # deterministic sorted, compact form.
    assert path.read_text(encoding="utf-8").rstrip("\n") == '{"a":1,"b":2}'

def test_governance_load_strict_json_object_rejects_duplicate_keys(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON key"):
        gov.load_strict_json_object(path)

def test_governance_load_strict_json_object_rejects_non_object(tmp_path):
    path = tmp_path / "arr.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(TypeError, match="must contain one JSON object"):
        gov.load_strict_json_object(path)

def test_governance_write_jsonl_and_load_strict_jsonl_objects_round_trip(tmp_path):
    path = tmp_path / "x.jsonl"
    gov.write_jsonl(path, [{"a": 1}, {"b": 2}])
    assert gov.load_strict_jsonl_objects(path) == [{"a": 1}, {"b": 2}]

def test_governance_write_jsonl_rejects_empty_records(tmp_path):
    with pytest.raises(ValueError, match="nonempty"):
        gov.write_jsonl(tmp_path / "e.jsonl", [])

def test_governance_load_strict_jsonl_objects_rejects_empty_file(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty"):
        gov.load_strict_jsonl_objects(path)

def test_governance_validate_timestamp_accepts_calendar_valid():
    gov.validate_timestamp("20260704T010203Z")

def test_governance_validate_timestamp_rejects_non_string():
    with pytest.raises(TypeError, match="must be a string"):
        gov.validate_timestamp(5)

def test_governance_validate_timestamp_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        gov.validate_timestamp("")

def test_governance_validate_timestamp_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="YYYYMMDDTHHMMSSZ"):
        gov.validate_timestamp("2026-07-04")

def test_governance_validate_timestamp_rejects_calendar_invalid():
    with pytest.raises(ValueError, match="calendar-valid"):
        gov.validate_timestamp("20261301T010203Z")

def test_governance_utc_timestamp_matches_pattern():
    stamp = gov.utc_timestamp()
    assert gov.TIMESTAMP_RE.fullmatch(stamp)
    gov.validate_timestamp(stamp)

def test_governance_same_path_true_for_same_directory(tmp_path):
    assert gov.same_path(tmp_path, tmp_path)

def test_governance_same_path_false_for_different(tmp_path):
    other = tmp_path / "sub"
    other.mkdir()
    assert not gov.same_path(tmp_path, other)

def test_governance_is_relative_to_true_for_child(tmp_path):
    child = tmp_path / "a" / "b"
    child.mkdir(parents=True)
    assert gov.is_relative_to(child, tmp_path)

def test_governance_is_relative_to_false_for_sibling(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert not gov.is_relative_to(a, b)

@pytest.mark.parametrize(
    "value,expected",
    [
        ("C:/x", True),
        ("/abs/path", True),
        ("relative/path", False),
        ("", False),
        (5, False),
        (None, False),
    ],
)
def test_governance_is_absolute_path_string_matrix(value, expected):
    assert gov.is_absolute_path_string(value) is expected

def test_governance_active_root_returns_path_for_valid_depth():
    result = gov.active_root(gov.__file__, 3)
    assert isinstance(result, Path)

def test_governance_active_root_rejects_bool_parents_up():
    with pytest.raises(TypeError, match="nonnegative int"):
        gov.active_root(gov.__file__, True)

def test_governance_active_root_rejects_empty_module_file():
    with pytest.raises(TypeError, match="non-empty string"):
        gov.active_root("", 3)

def test_governance_active_root_rejects_too_deep():
    with pytest.raises(ValueError, match="parents_up must be <="):
        gov.active_root(gov.__file__, 10000)

def test_governance_resolve_result_parent_none_returns_official(tmp_path):
    official = tmp_path / "official"
    root = tmp_path / "repo"
    resolved = gov.resolve_result_parent(
        None, official_parent=official, active_root_path=root
    )
    assert resolved == official.resolve()

def test_governance_resolve_result_parent_relative_anchored_on_active_root(tmp_path):
    official = tmp_path / "official"
    root = tmp_path / "repo"
    resolved = gov.resolve_result_parent(
        "sub/dir", official_parent=official, active_root_path=root
    )
    assert resolved == (root / "sub" / "dir").resolve()

def test_governance_resolve_result_parent_rejects_wrong_type(tmp_path):
    with pytest.raises(TypeError, match="str, Path, or None"):
        gov.resolve_result_parent(
            5, official_parent=tmp_path, active_root_path=tmp_path
        )

def test_governance_resolve_result_parent_rejects_blank(tmp_path):
    with pytest.raises(ValueError, match="non-empty"):
        gov.resolve_result_parent(
            "   ", official_parent=tmp_path, active_root_path=tmp_path
        )

def test_governance_human_readable_command_quotes_parts():
    assert gov.human_readable_command(["a", "b c"]) == '"a" "b c"'

def test_governance_human_readable_command_rejects_non_list():
    with pytest.raises(TypeError, match="list of strings"):
        gov.human_readable_command("notalist")

def test_governance_human_readable_command_rejects_non_string_items():
    with pytest.raises(TypeError, match="list of strings"):
        gov.human_readable_command([1, 2])

def test_governance_require_field_accepts_matching_bool():
    gov.require_field({"a": True}, "a", True, "p")

def test_governance_require_field_rejects_flipped_bool():
    with pytest.raises(ValueError, match="must equal True"):
        gov.require_field({"a": False}, "a", True, "p")

def test_governance_require_field_rejects_missing():
    with pytest.raises(ValueError, match="missing required field"):
        gov.require_field({}, "a", True, "p")

def test_governance_require_field_bool_int_distinction():
    # ``True`` is not an acceptable value for an int expectation of 1.
    with pytest.raises(ValueError, match="must equal 1"):
        gov.require_field({"a": True}, "a", 1, "p")

def test_governance_require_field_allow_legacy_skips_missing():
    gov.require_field({}, "a", True, "p", allow_legacy=True)

def test_governance_require_json_positive_int_accepts_positive():
    assert gov.require_json_positive_int("x", 5) == 5

def test_governance_require_json_positive_int_rejects_bool():
    with pytest.raises(TypeError, match="JSON integer"):
        gov.require_json_positive_int("x", True)

def test_governance_require_json_positive_int_rejects_zero():
    with pytest.raises(ValueError, match="must be positive"):
        gov.require_json_positive_int("x", 0)

def test_governance_require_json_positive_int_rejects_string():
    with pytest.raises(TypeError, match="JSON integer"):
        gov.require_json_positive_int("x", "5")

def test_governance_require_json_bounded_positive_int_accepts_within_bound():
    assert gov.require_json_bounded_positive_int("x", 16, maximum=16) == 16

def test_governance_require_json_bounded_positive_int_rejects_over_bound():
    with pytest.raises(ValueError, match="<= 16"):
        gov.require_json_bounded_positive_int("x", 17, maximum=16)

def test_governance_require_positive_parameter_delta_accepts_positive():
    gov.require_positive_parameter_delta(0.5, "d")

def test_governance_require_positive_parameter_delta_rejects_non_numeric():
    with pytest.raises(TypeError, match="must be numeric"):
        gov.require_positive_parameter_delta("x", "d")

def test_governance_require_positive_parameter_delta_rejects_bool():
    with pytest.raises(TypeError, match="must be numeric"):
        gov.require_positive_parameter_delta(True, "d")

def test_governance_require_positive_parameter_delta_rejects_zero():
    with pytest.raises(ValueError, match="finite and > 0"):
        gov.require_positive_parameter_delta(0.0, "d")

def test_governance_require_positive_parameter_delta_rejects_infinite():
    with pytest.raises(ValueError):
        gov.require_positive_parameter_delta(float("inf"), "d")

def test_governance_require_string_path_value_returns_value():
    assert gov.require_string_path_value({"p": "/x"}, "p", "ctx") == "/x"

def test_governance_require_string_path_value_rejects_missing():
    with pytest.raises(ValueError, match="missing required provenance"):
        gov.require_string_path_value({}, "p", "ctx")

def test_governance_require_string_path_value_rejects_empty():
    with pytest.raises(TypeError, match="non-empty string path"):
        gov.require_string_path_value({"p": ""}, "p", "ctx")

def test_governance_validate_path_or_command_string_rejects_empty():
    with pytest.raises(ValueError, match="non-empty string"):
        gov.validate_path_or_command_string("", "p")

def test_governance_validate_path_or_command_string_rejects_nul():
    with pytest.raises(ValueError, match="NUL bytes"):
        gov.validate_path_or_command_string("a\x00b", "p")

def test_rollout_common_contains_unsafe_terminology_true():
    assert rc.contains_unsafe_terminology("this makes a bayesian belief claim")

def test_rollout_common_contains_unsafe_terminology_false():
    assert not rc.contains_unsafe_terminology("bounded development only")

def test_rollout_common_contains_unsafe_terminology_rejects_non_string():
    with pytest.raises(TypeError, match="must be a string"):
        rc.contains_unsafe_terminology(5)

def test_rollout_common_unsafe_terminology_phrases_count():
    assert len(rc.unsafe_terminology_phrases()) == 6

def test_rollout_common_rollout_field_names_partition_is_complete():
    partitioned = (
        rc._FEATURE_FIELDS | rc._ACTION_FIELDS | rc._BOOL_FIELDS | rc._FLOAT_FIELDS
    )
    assert partitioned == set(rc.ROLLOUT_FIELD_NAMES)
    assert len(rc.ROLLOUT_FIELD_NAMES) == 17

def test_rollout_common_empty_rollout_fields_all_empty_lists():
    fields = rc.empty_rollout_fields()
    assert set(fields) == set(rc.ROLLOUT_FIELD_NAMES)
    assert all(value == [] for value in fields.values())

def test_rollout_common_stack_time_stacks_on_time_axis():
    out = rc.stack_time([torch.zeros(2, 4), torch.ones(2, 4)])
    assert tuple(out.shape) == (2, 2, 4)

def test_rollout_common_stack_time_rejects_empty():
    with pytest.raises(ValueError, match="at least one transition"):
        rc.stack_time([])

def test_rollout_common_float_vector_builds_finite_tensor():
    ref = torch.zeros(2, dtype=torch.float32)
    out = rc.float_vector([1.0, 2.0], ref)
    assert out.tolist() == [1.0, 2.0]
    assert out.dtype == torch.float32

def test_rollout_common_float_vector_rejects_bool_scalar():
    ref = torch.zeros(2, dtype=torch.float32)
    with pytest.raises(TypeError, match="numeric int or float"):
        rc.float_vector([True, 1.0], ref)

def test_rollout_common_float_vector_rejects_non_finite_scalar():
    ref = torch.zeros(2, dtype=torch.float32)
    with pytest.raises(ValueError, match="must be finite"):
        rc.float_vector([float("nan"), 1.0], ref)

def test_rollout_common_per_step_sampling_seed_offset_formula():
    assert rc.per_step_sampling_seed(23, 0, 0) == 23
    assert rc.per_step_sampling_seed(23, 1, 2) == 23 + 1000 + 2
    # No collision: episode e step s vs episode e+1 step s-1 differ.
    assert rc.per_step_sampling_seed(23, 0, 5) != rc.per_step_sampling_seed(23, 1, 4)

def test_rollout_common_set_rollout_seed_is_deterministic():
    rc.set_rollout_seed(99)
    a = torch.rand(3)
    rc.set_rollout_seed(99)
    b = torch.rand(3)
    assert torch.equal(a, b)

def test_rollout_common_validate_team_end_flags_all_terminal_ended():
    terminal = torch.tensor([[True], [True]])
    truncation = torch.tensor([[False], [False]])
    assert rc.validate_team_end_flags(terminal, truncation) is True

def test_rollout_common_validate_team_end_flags_all_truncated_ended():
    terminal = torch.tensor([[False], [False]])
    truncation = torch.tensor([[True], [True]])
    assert rc.validate_team_end_flags(terminal, truncation) is True

def test_rollout_common_validate_team_end_flags_none_ended():
    zeros = torch.tensor([[False], [False]])
    assert rc.validate_team_end_flags(zeros, zeros) is False

def test_rollout_common_validate_team_end_flags_rejects_partial():
    terminal = torch.tensor([[True], [False]])
    truncation = torch.tensor([[False], [False]])
    with pytest.raises(ValueError, match="partial per-agent"):
        rc.validate_team_end_flags(terminal, truncation)

def test_rollout_common_validate_team_end_flags_rejects_mixed():
    terminal = torch.tensor([[True], [False]])
    truncation = torch.tensor([[False], [True]])
    with pytest.raises(ValueError, match="mixed terminal and truncated"):
        rc.validate_team_end_flags(terminal, truncation)

def test_rollout_common_validate_team_end_flags_rejects_both_true():
    both = torch.tensor([[True], [True]])
    with pytest.raises(ValueError, match="cannot both be true"):
        rc.validate_team_end_flags(both, both)

def test_rollout_common_validate_team_end_flags_rejects_non_bool_dtype():
    terminal = torch.tensor([[1], [1]])
    truncation = torch.tensor([[0], [0]])
    with pytest.raises(TypeError, match="torch.bool dtype"):
        rc.validate_team_end_flags(terminal, truncation)

def test_rollout_common_validate_team_end_flags_rejects_shape_mismatch():
    terminal = torch.tensor([[True], [True]])
    truncation = torch.tensor([[False]])
    with pytest.raises(ValueError, match="shape mismatch"):
        rc.validate_team_end_flags(terminal, truncation)

def test_rollout_common_validate_team_end_flags_rejects_non_tensor():
    with pytest.raises(TypeError, match="torch.Tensor"):
        rc.validate_team_end_flags([True], torch.tensor([[False]]))

def test_rollout_common_validate_team_end_flags_context_in_message():
    terminal = torch.tensor([[True], [False]])
    truncation = torch.tensor([[False], [False]])
    with pytest.raises(ValueError, match="MY-CONTEXT"):
        rc.validate_team_end_flags(terminal, truncation, context="MY-CONTEXT")

def test_rollout_common_pad_field_pads_feature_to_max_time():
    tensor = torch.ones(2, 2, 4)
    out = rc.pad_field(tensor, name="actor_observation", max_time=3)
    assert tuple(out.shape) == (2, 3, 4)
    assert torch.equal(out[:, 2, :], torch.zeros(2, 4))

def test_rollout_common_pad_field_bool_field_pads_false():
    tensor = torch.ones(2, 2, dtype=torch.bool)
    out = rc.pad_field(tensor, name="valid_mask", max_time=4)
    assert out.dtype == torch.bool
    assert out[:, 2:].tolist() == [[False, False], [False, False]]

def test_rollout_common_pad_field_no_op_when_equal():
    tensor = torch.ones(2, 3)
    out = rc.pad_field(tensor, name="task_reward", max_time=3)
    assert out is tensor

def test_rollout_common_pad_field_rejects_over_length():
    tensor = torch.ones(2, 4)
    with pytest.raises(ValueError, match="longer than max_time"):
        rc.pad_field(tensor, name="task_reward", max_time=2)

def test_rollout_common_pad_field_rejects_non_tensor():
    with pytest.raises(TypeError, match="torch.Tensor"):
        rc.pad_field("x", name="task_reward", max_time=2)

def test_rollout_common_pad_field_rejects_missing_time_axis():
    with pytest.raises(ValueError, match=r"\[batch, time\] axis"):
        rc.pad_field(torch.zeros(2), name="terminal", max_time=3)

def test_rollout_common_concat_episodes_concatenates_batch_axis():
    ep1 = {"x": torch.zeros(2, 3, 4)}
    ep2 = {"x": torch.ones(2, 3, 4)}
    out = rc.concat_episodes([ep1, ep2], "x")
    assert tuple(out.shape) == (4, 3, 4)

def test_rollout_common_concat_episodes_rejects_unequal_time():
    ep1 = {"x": torch.zeros(2, 3, 4)}
    ep2 = {"x": torch.zeros(2, 5, 4)}
    with pytest.raises(ValueError, match="equal time length"):
        rc.concat_episodes([ep1, ep2], "x")

def test_rollout_common_concat_episodes_rejects_empty():
    with pytest.raises(ValueError, match="nonempty"):
        rc.concat_episodes([], "x")

def test_rollout_common_concat_episodes_rejects_missing_time_axis():
    with pytest.raises(ValueError, match=r"\[batch, time\] axis"):
        rc.concat_episodes([{"x": torch.zeros(2)}], "x")

def test_rollout_common_merge_padded_episodes_valid_mask_false_on_pads():
    batch = rc.merge_padded_episodes(
        [_gc_make_episode(2), _gc_make_episode(3)], max_time=3
    )
    assert tuple(batch.valid_mask.shape) == (4, 3)
    # First-episode rows padded on the last step.
    assert batch.valid_mask[0].tolist() == [True, True, False]
    assert batch.valid_mask[1].tolist() == [True, True, False]
    # Second episode fully valid.
    assert batch.valid_mask[2].tolist() == [True, True, True]

def test_rollout_common_merge_padded_episodes_rejects_empty():
    with pytest.raises(ValueError, match="nonempty"):
        rc.merge_padded_episodes([], max_time=3)

def test_rollout_common_validate_rollout_batch_values_accepts_valid_batch():
    batch = rc.merge_padded_episodes([_gc_make_episode(2)], max_time=2)
    rc.validate_rollout_batch_values(batch)

def test_rollout_common_validate_rollout_batch_values_rejects_non_batch():
    with pytest.raises(TypeError, match="RolloutBatch"):
        rc.validate_rollout_batch_values("not a batch")

def test_rollout_common_validate_policy_action_tensor_accepts_valid():
    rc.validate_policy_action_tensor(
        "sensing_action", torch.tensor([0, 1]), action_count=2, agent_count=2
    )

def test_rollout_common_validate_policy_action_tensor_rejects_non_tensor():
    with pytest.raises(TypeError, match="torch.Tensor"):
        rc.validate_policy_action_tensor(
            "s", [0, 1], action_count=2, agent_count=2
        )

def test_rollout_common_validate_policy_action_tensor_rejects_float_dtype():
    with pytest.raises(TypeError, match="integer dtype"):
        rc.validate_policy_action_tensor(
            "s", torch.tensor([0.0, 1.0]), action_count=2, agent_count=2
        )

def test_rollout_common_validate_policy_action_tensor_rejects_bool_dtype():
    with pytest.raises(TypeError, match="integer dtype"):
        rc.validate_policy_action_tensor(
            "s", torch.tensor([True, False]), action_count=2, agent_count=2
        )

def test_rollout_common_validate_policy_action_tensor_rejects_wrong_length():
    with pytest.raises(ValueError, match="one item per fixed Stage 23-A agent"):
        rc.validate_policy_action_tensor(
            "s", torch.tensor([0, 1, 2]), action_count=2, agent_count=2
        )

def test_rollout_common_validate_policy_action_tensor_rejects_out_of_range():
    with pytest.raises(ValueError, match="out-of-range action index"):
        rc.validate_policy_action_tensor(
            "s", torch.tensor([0, 5]), action_count=2, agent_count=2
        )

def test_rollout_common_validate_policy_action_tensor_rejects_meta():
    meta = torch.zeros(2, dtype=torch.long, device="meta")
    with pytest.raises(ValueError, match="meta tensor"):
        rc.validate_policy_action_tensor(
            "s", meta, action_count=2, agent_count=2
        )

def test_rollout_common_validate_parallel_payloads_accepts_correct_order():
    payload = {name: {} for name in STAGE23_AGENT_NAMES}
    rc.validate_parallel_payloads(payload, payload)

def test_rollout_common_validate_parallel_payloads_rejects_wrong_order():
    payload = {name: {} for name in STAGE23_AGENT_NAMES}
    with pytest.raises(ValueError, match="fixed Stage 23-A agent order"):
        rc.validate_parallel_payloads({"wrong_agent": {}}, payload)

def test_rollout_common_validate_parallel_payloads_rejects_non_mapping():
    payload = {name: {} for name in STAGE23_AGENT_NAMES}
    with pytest.raises(TypeError, match="must be a mapping"):
        rc.validate_parallel_payloads("x", payload)

def test_rollout_common_transition_agent_id_returns_int():
    item = _GcTransition({"agent_id": 3})
    assert rc.transition_agent_id(item) == 3

def test_rollout_common_transition_agent_id_rejects_bool():
    item = _GcTransition({"agent_id": True})
    with pytest.raises(TypeError, match="must be an integer"):
        rc.transition_agent_id(item)

def test_rollout_common_declared_agent_order_agent_order_tuple():
    item = _GcTransition({"agent_order": (0, 1)})
    assert rc.declared_agent_order_from_transition(item, _gc_core_config()) == (0, 1)

def test_rollout_common_declared_agent_order_agent_id_mapping_positions():
    item = _GcTransition({"agent_id_mapping": {0: 1, 1: 0}})
    assert rc.declared_agent_order_from_transition(item, _gc_core_config()) == (1, 0)

def test_rollout_common_declared_agent_order_rejects_set_container():
    item = _GcTransition({"agent_order": {0, 1}})
    with pytest.raises(TypeError, match="must be a list or tuple"):
        rc.declared_agent_order_from_transition(item, _gc_core_config())

def test_rollout_common_declared_agent_order_rejects_duplicate():
    item = _GcTransition({"agent_order": (0, 0)})
    with pytest.raises(ValueError, match="must not contain duplicates"):
        rc.declared_agent_order_from_transition(item, _gc_core_config())

def test_rollout_common_declared_agent_order_rejects_negative():
    # Uses agent_id_mapping so the negative agent-id is the failing check.
    item = _GcTransition({"agent_order": (-1, 0)})
    with pytest.raises(ValueError, match="must be nonnegative"):
        rc.declared_agent_order_from_transition(item, _gc_core_config())

def test_rollout_common_declared_agent_order_rejects_missing_structures():
    item = _GcTransition({"unknown": 1})
    with pytest.raises(ValueError, match="requires agent_order"):
        rc.declared_agent_order_from_transition(item, _gc_core_config())

def test_rollout_common_declared_agent_order_mapping_rejects_duplicate_position():
    item = _GcTransition({"agent_id_mapping": {0: 0, 1: 0}})
    with pytest.raises(ValueError, match="positions must be unique"):
        rc.declared_agent_order_from_transition(item, _gc_core_config())

def test_rollout_common_declared_agent_order_mapping_rejects_negative_position():
    item = _GcTransition({"agent_id_mapping": {0: -1, 1: 0}})
    with pytest.raises(ValueError, match="position must be nonnegative"):
        rc.declared_agent_order_from_transition(item, _gc_core_config())

def test_rollout_common_declared_agent_order_mapping_rejects_bool_position():
    item = _GcTransition({"agent_id_mapping": {0: True, 1: 0}})
    with pytest.raises(TypeError, match="integer stable positions"):
        rc.declared_agent_order_from_transition(item, _gc_core_config())

def test_rollout_common_declared_agent_order_mapping_rejects_non_comparable_position():
    item = _GcTransition({"agent_id_mapping": {0: "a", 1: 0}})
    with pytest.raises(TypeError, match="integer stable positions"):
        rc.declared_agent_order_from_transition(item, _gc_core_config())

def test_rollout_common_declared_agent_order_mapping_rejects_non_mapping():
    item = _GcTransition({"agent_id_mapping": [0, 1]})
    with pytest.raises(TypeError, match="must be a mapping"):
        rc.declared_agent_order_from_transition(item, _gc_core_config())

def test_rollout_common_declared_agent_order_rejects_incomplete_team_coverage():
    # agent_id_count is 2, so an order of (0, 2) does not cover {0, 1}.
    item = _GcTransition({"agent_order": (0, 2)})
    with pytest.raises(ValueError, match="cover the configured team"):
        rc.declared_agent_order_from_transition(item, _gc_core_config())

def test_rollout_common_parameter_delta_l1_sums_absolute_change():
    before = [torch.zeros(3)]
    after = [torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.0]))]
    assert rc.parameter_delta_l1(before, after) == pytest.approx(6.0)

def test_rollout_common_parameter_delta_l1_zero_when_unchanged():
    before = [torch.tensor([1.0, 2.0])]
    after = [torch.nn.Parameter(torch.tensor([1.0, 2.0]))]
    assert rc.parameter_delta_l1(before, after) == pytest.approx(0.0)

def test_rollout_common_parameter_delta_l1_rejects_non_finite():
    before = [torch.zeros(2)]
    after = [torch.nn.Parameter(torch.tensor([float("inf"), 0.0]))]
    with pytest.raises(ValueError, match="must be finite"):
        rc.parameter_delta_l1(before, after)

def test_rollout_common_validate_action_vector_accepts_valid():
    rc.validate_action_vector("a", torch.tensor([0, 1]), 2, 5)

def test_rollout_common_validate_action_vector_rejects_wrong_ndim():
    with pytest.raises(ValueError, match=r"shape \[agent\]"):
        rc.validate_action_vector("a", torch.zeros(2, 2, dtype=torch.long), 2, 5)

def test_rollout_common_validate_action_vector_rejects_wrong_length():
    with pytest.raises(ValueError, match="match agent_count"):
        rc.validate_action_vector("a", torch.tensor([0, 1, 2]), 2, 5)

def test_rollout_common_validate_action_vector_rejects_float():
    with pytest.raises(TypeError, match="integer dtype"):
        rc.validate_action_vector("a", torch.tensor([0.0, 1.0]), 2, 5)

def test_rollout_common_validate_action_vector_rejects_out_of_range():
    with pytest.raises(ValueError, match="out-of-range action index"):
        rc.validate_action_vector("a", torch.tensor([0, 9]), 2, 5)

def test_rollout_common_validate_stage23_core_config_accepts_default():
    rc.validate_stage23_core_config(_gc_core_config())

def test_rollout_common_validate_stage23_core_config_rejects_non_config():
    with pytest.raises(TypeError, match="MAPPOCoreConfig"):
        rc.validate_stage23_core_config("not a config")

def test_stage23b_rollout_config_defaults_valid():
    cfg = Stage23BRolloutConfig()
    assert cfg.max_episodes == 1
    assert cfg.max_steps_per_episode == 4
    assert cfg.sample_actions is False
    assert cfg.device == "cpu"

def test_stage23b_rollout_config_rejects_multi_episode():
    with pytest.raises(ValueError, match="exactly one episode"):
        Stage23BRolloutConfig(max_episodes=2)

def test_stage23b_rollout_config_rejects_non_positive_episodes():
    with pytest.raises(ValueError, match="must be positive"):
        Stage23BRolloutConfig(max_episodes=0)

def test_stage23b_rollout_config_rejects_steps_over_cap():
    with pytest.raises(ValueError, match="must be <= 16"):
        Stage23BRolloutConfig(max_steps_per_episode=17)

def test_stage23b_rollout_config_rejects_non_positive_steps():
    with pytest.raises(ValueError, match="must be positive"):
        Stage23BRolloutConfig(max_steps_per_episode=0)

def test_stage23b_rollout_config_rejects_non_bool_sample_actions():
    with pytest.raises(TypeError, match="sample_actions must be a bool"):
        Stage23BRolloutConfig(sample_actions="yes")

def test_stage23b_rollout_config_rejects_seed_zero():
    with pytest.raises(ValueError, match="seed 0 is reserved for Stage 23-B"):
        Stage23BRolloutConfig(seed=0)

def test_stage23b_rollout_config_rejects_bad_dtype():
    with pytest.raises(ValueError, match="float32.*float64"):
        Stage23BRolloutConfig(dtype="float16")

def test_stage23b_rollout_config_rejects_non_default_dtype():
    with pytest.raises(ValueError, match="must equal the default"):
        Stage23BRolloutConfig(dtype="float64")

def test_stage23b_rollout_config_rejects_non_cpu_device():
    with pytest.raises(ValueError, match="only CPU device"):
        Stage23BRolloutConfig(device="cuda")

def test_stage23b_rollout_config_rejects_padded_device():
    with pytest.raises(ValueError, match="leading or trailing whitespace"):
        Stage23BRolloutConfig(device=" cpu ")

def test_stage23b_rollout_make_default_environment_defaults():
    env = make_default_stage23b_environment()
    assert env.config.max_steps == 4

def test_stage23b_rollout_make_default_environment_rejects_seed_zero():
    with pytest.raises(ValueError, match="seed 0 is reserved for Stage 23-B"):
        make_default_stage23b_environment(seed=0)

def test_stage23b_rollout_make_default_environment_rejects_non_positive_steps():
    with pytest.raises(ValueError, match="must be positive"):
        make_default_stage23b_environment(max_steps=0)

def test_stage23b_rollout_collect_returns_collected_rollout():
    model = _gc_model()
    env = make_default_stage23b_environment(max_steps=4, seed=23)
    collected = collect_stage23b_development_rollout(model, env, Stage23BRolloutConfig())
    assert isinstance(collected, Stage23BCollectedRollout)
    assert collected.summary["stage"] == "23-B"
    assert collected.summary["batch_size"] == len(STAGE23_AGENT_NAMES)
    assert collected.summary["episode_count"] == 1
    assert collected.summary["development_only"] is True

def test_stage23b_rollout_collect_summary_bootstrap_convention():
    model = _gc_model()
    env = make_default_stage23b_environment(max_steps=4, seed=23)
    collected = collect_stage23b_development_rollout(model, env, Stage23BRolloutConfig())
    assert (
        collected.summary["next_value_bootstrap_convention"]
        == "terminal_next_values_zeroed;truncation_next_values_preserved"
    )

def test_stage23b_rollout_collect_environment_summary_keys():
    model = _gc_model()
    env = make_default_stage23b_environment(max_steps=4, seed=23)
    collected = collect_stage23b_development_rollout(model, env, Stage23BRolloutConfig())
    env_summary = collected.environment_summary
    assert env_summary["environment_stage"] == "23-A"
    assert env_summary["stage"] == "23-B"
    assert env_summary["agent_count"] == len(STAGE23_AGENT_NAMES)

def test_stage23b_rollout_collect_restores_model_mode():
    model = _gc_model()
    model.train()
    env = make_default_stage23b_environment(max_steps=4, seed=23)
    collect_stage23b_development_rollout(model, env, Stage23BRolloutConfig())
    assert model.training is True

def test_stage23b_rollout_collect_batch_passes_value_recheck():
    model = _gc_model()
    env = make_default_stage23b_environment(max_steps=4, seed=23)
    collected = collect_stage23b_development_rollout(model, env, Stage23BRolloutConfig())
    # A returned batch must survive the shared value re-check.
    rc.validate_rollout_batch_values(collected.batch)

def test_stage23b_rollout_collect_rejects_wrong_model_type():
    env = make_default_stage23b_environment(max_steps=4, seed=23)
    with pytest.raises(TypeError, match="RecurrentMAPPOActorCritic"):
        collect_stage23b_development_rollout("not a model", env, Stage23BRolloutConfig())

def test_stage23b_rollout_collect_rejects_wrong_environment_type():
    model = _gc_model()
    with pytest.raises(TypeError, match="RiskAwareActiveSensingGridEnvironment"):
        collect_stage23b_development_rollout(model, "not an env", Stage23BRolloutConfig())

def test_stage23b_rollout_collect_rejects_wrong_config_type():
    model = _gc_model()
    env = make_default_stage23b_environment(max_steps=4, seed=23)
    with pytest.raises(TypeError, match="Stage23BRolloutConfig or None"):
        collect_stage23b_development_rollout(model, env, "not a config")

def test_stage23b_rollout_collect_rejects_steps_over_env_max():
    model = _gc_model()
    env = make_default_stage23b_environment(max_steps=2, seed=23)
    cfg = Stage23BRolloutConfig(max_steps_per_episode=4)
    with pytest.raises(ValueError, match="must not exceed environment max_steps"):
        collect_stage23b_development_rollout(model, env, cfg)

def test_stage23b_runner_rejects_seed_zero(tmp_path):
    with pytest.raises(ValueError, match="seed 0 is reserved"):
        run_stage23b_bounded_training(seed=0, result_parent=tmp_path / "out")

def test_stage23b_runner_rejects_seed_bool(tmp_path):
    with pytest.raises(TypeError):
        run_stage23b_bounded_training(seed=True, result_parent=tmp_path / "out")

def test_stage23b_runner_rejects_steps_over_cap(tmp_path):
    with pytest.raises(ValueError, match="must be <= 16"):
        run_stage23b_bounded_training(
            steps_per_episode=17, result_parent=tmp_path / "out"
        )

def test_stage23b_runner_rejects_non_positive_steps(tmp_path):
    with pytest.raises(ValueError, match="must be positive"):
        run_stage23b_bounded_training(
            steps_per_episode=0, result_parent=tmp_path / "out"
        )

def test_stage23b_runner_rejects_non_bool_allow_flag(tmp_path):
    with pytest.raises(TypeError, match="allow_pre_existing_stage23b_roots must be a bool"):
        run_stage23b_bounded_training(
            allow_pre_existing_stage23b_roots="yes", result_parent=tmp_path / "out"
        )

def test_stage23b_runner_rejects_result_parent_wrong_type():
    with pytest.raises(TypeError, match="str, Path, or None"):
        run_stage23b_bounded_training(result_parent=5)

def test_stage23b_runner_rejects_bad_timestamp_type(tmp_path):
    with pytest.raises(TypeError, match="timestamp_utc must be a string or None"):
        run_stage23b_bounded_training(
            timestamp_utc=5, result_parent=tmp_path / "out"
        )

def test_stage23b_runner_rejects_bad_timestamp_value(tmp_path):
    with pytest.raises(ValueError, match="YYYYMMDDTHHMMSSZ"):
        run_stage23b_bounded_training(
            timestamp_utc="not-a-stamp", result_parent=tmp_path / "out"
        )

def test_stage23b_runner_writes_eight_file_set(tmp_path):
    root = run_stage23b_bounded_training(
        result_parent=tmp_path / "out",
        timestamp_utc=_GC_TIMESTAMP,
        seed=23,
        steps_per_episode=4,
    )
    names = sorted(p.name for p in root.iterdir())
    assert names == sorted(STAGE23B_RESULT_FILES)
    assert len(names) == 8

def test_stage23b_runner_result_root_uses_prefix(tmp_path):
    root = run_stage23b_bounded_training(
        result_parent=tmp_path / "out",
        timestamp_utc=_GC_TIMESTAMP,
    )
    assert root.name == f"{STAGE23B_RESULT_PREFIX}{_GC_TIMESTAMP}"

def test_stage23b_runner_boundary_record_stage_is_23b(tmp_path):
    root = run_stage23b_bounded_training(
        result_parent=tmp_path / "out",
        timestamp_utc=_GC_TIMESTAMP,
    )
    payload = json.loads((root / "boundary_record.json").read_text(encoding="utf-8"))
    assert payload["stage"] == "23-B"
    assert payload["development_only"] is True
    assert payload["evaluation_run"] is False

def test_stage23b_runner_all_payloads_carry_boundary_flags(tmp_path):
    root = run_stage23b_bounded_training(
        result_parent=tmp_path / "out",
        timestamp_utc=_GC_TIMESTAMP,
    )
    for name in STAGE23B_RESULT_FILES:
        if name.endswith(".json"):
            payload = json.loads((root / name).read_text(encoding="utf-8"))
            assert payload["development_only"] is True
            assert payload["stage"] == "23-B"

def test_stage23b_runner_artifact_hashes_cover_other_files(tmp_path):
    root = run_stage23b_bounded_training(
        result_parent=tmp_path / "out",
        timestamp_utc=_GC_TIMESTAMP,
    )
    hashes = json.loads((root / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashed = set(hashes["hashed_files"])
    expected = {n for n in STAGE23B_RESULT_FILES if n != "artifact_hashes.json"}
    assert hashed == expected

def test_stage23b_runner_command_record_provenance(tmp_path):
    root = run_stage23b_bounded_training(
        result_parent=tmp_path / "out",
        timestamp_utc=_GC_TIMESTAMP,
        seed=23,
        steps_per_episode=4,
    )
    record = json.loads((root / "command_record.json").read_text(encoding="utf-8"))
    assert record["python_executable"] == sys.executable
    assert record["command_argv"][0] == sys.executable
    assert record["command_argv"][1] == "-c"
    assert (
        record["entry_point"]
        == "raas_marl.mappo_lagrangian.stage23b_runner.run_stage23b_bounded_training"
    )
    inv = record["invocation_parameters"]
    assert inv["seed"] == 23
    assert inv["steps_per_episode"] == 4
    assert inv["timestamp_utc"] == _GC_TIMESTAMP

def test_stage23b_runner_rejects_custom_parent_inside_repo():
    with pytest.raises(ValueError, match="outside the repository"):
        run_stage23b_bounded_training(
            result_parent="src/inside_repo",
            timestamp_utc=_GC_TIMESTAMP,
        )

def test_stage23b_runner_rejects_pre_existing_result_root(tmp_path):
    parent = tmp_path / "out"
    run_stage23b_bounded_training(
        result_parent=parent, timestamp_utc=_GC_TIMESTAMP
    )
    with pytest.raises(FileExistsError):
        run_stage23b_bounded_training(
            result_parent=parent, timestamp_utc=_GC_TIMESTAMP
        )

def test_stage24_collector_config_defaults_valid():
    cfg = Stage24CollectorConfig()
    assert cfg.max_episodes == 2
    assert cfg.phase == "training"
    assert cfg.sample_actions is True

def test_stage24_collector_config_evaluation_sample_actions_false():
    cfg = Stage24CollectorConfig(phase="evaluation")
    assert cfg.sample_actions is False

def test_stage24_collector_config_rejects_single_episode():
    with pytest.raises(ValueError, match="more than one episode"):
        Stage24CollectorConfig(max_episodes=1)

def test_stage24_collector_config_rejects_non_positive_episodes():
    with pytest.raises(ValueError, match="must be positive"):
        Stage24CollectorConfig(max_episodes=0)

def test_stage24_collector_config_rejects_bad_phase():
    with pytest.raises(ValueError, match="training.*evaluation"):
        Stage24CollectorConfig(phase="foo")

def test_stage24_collector_config_rejects_seed_zero():
    with pytest.raises(ValueError, match="seed 0 is intentionally reserved"):
        Stage24CollectorConfig(seed=0)

def test_stage24_collector_config_rejects_unknown_scenario():
    with pytest.raises(ValueError, match="unknown Stage 23-A scenario"):
        Stage24CollectorConfig(scenario_names=("nope",))

def test_stage24_collector_config_rejects_empty_scenarios():
    with pytest.raises(ValueError, match="non-empty tuple"):
        Stage24CollectorConfig(scenario_names=())

def test_stage24_collector_config_rejects_non_default_dtype():
    with pytest.raises(ValueError, match="default Stage 23-A dtype"):
        Stage24CollectorConfig(dtype="float64")

def test_stage24_collector_config_rejects_non_cpu_device():
    with pytest.raises(ValueError, match="CPU device"):
        Stage24CollectorConfig(device="cuda")

def _gc_internal_stage22_config(*, normalize: bool = True) -> Stage22UpdateConfig:
    base = default_stage22_update_config()
    return Stage22UpdateConfig(
        algorithm=base.algorithm,
        losses=base.losses,
        lagrange=base.lagrange,
        learning_rate=base.learning_rate,
        max_update_epochs=1,
        max_grad_norm=base.max_grad_norm,
        normalize_advantages=normalize,
    )

def test_stage24_collector_update_config_default_valid():
    cfg = default_stage24a_update_config()
    assert isinstance(cfg, Stage24AUpdateConfig)
    assert cfg.readiness_smoke_only is True
    assert cfg.nonzero_lagrange_lower_bound_added is False
    assert cfg.hazard_budget_schedule_added is False

def test_stage24_collector_update_config_rejects_non_stage22_internal():
    with pytest.raises(TypeError, match="Stage22UpdateConfig"):
        Stage24AUpdateConfig(
            internal_stage22_config="x",
            readiness_smoke_only=True,
            stage22_semantics_reused_internally=True,
            advantage_normalization_used=True,
            nonzero_lagrange_lower_bound_added=False,
            hazard_budget_schedule_added=False,
        )

def test_stage24_collector_update_config_rejects_non_bool_field():
    with pytest.raises(TypeError, match="must be a strict bool"):
        Stage24AUpdateConfig(
            internal_stage22_config=_gc_internal_stage22_config(),
            readiness_smoke_only="yes",
            stage22_semantics_reused_internally=True,
            advantage_normalization_used=True,
            nonzero_lagrange_lower_bound_added=False,
            hazard_budget_schedule_added=False,
        )

def test_stage24_collector_update_config_rejects_normalization_mismatch():
    with pytest.raises(ValueError, match="must match internal config normalize_advantages"):
        Stage24AUpdateConfig(
            internal_stage22_config=_gc_internal_stage22_config(normalize=True),
            readiness_smoke_only=True,
            stage22_semantics_reused_internally=True,
            advantage_normalization_used=False,
            nonzero_lagrange_lower_bound_added=False,
            hazard_budget_schedule_added=False,
        )

def test_stage24_collector_update_config_rejects_nonzero_lagrange_lower_bound():
    with pytest.raises(ValueError, match="nonzero Lagrange lower bound"):
        Stage24AUpdateConfig(
            internal_stage22_config=_gc_internal_stage22_config(),
            readiness_smoke_only=True,
            stage22_semantics_reused_internally=True,
            advantage_normalization_used=True,
            nonzero_lagrange_lower_bound_added=True,
            hazard_budget_schedule_added=False,
        )

def test_stage24_collector_update_config_rejects_budget_schedule():
    with pytest.raises(ValueError, match="hazard budget schedule"):
        Stage24AUpdateConfig(
            internal_stage22_config=_gc_internal_stage22_config(),
            readiness_smoke_only=True,
            stage22_semantics_reused_internally=True,
            advantage_normalization_used=True,
            nonzero_lagrange_lower_bound_added=False,
            hazard_budget_schedule_added=True,
        )

def test_stage24_collector_update_config_rejects_final_infra_marking():
    # readiness_smoke_only False while reusing stage22 semantics is rejected.
    with pytest.raises(ValueError, match="cannot be marked final-training infrastructure"):
        Stage24AUpdateConfig(
            internal_stage22_config=_gc_internal_stage22_config(),
            readiness_smoke_only=False,
            stage22_semantics_reused_internally=True,
            advantage_normalization_used=True,
            nonzero_lagrange_lower_bound_added=False,
            hazard_budget_schedule_added=False,
        )

def test_stage24_collector_default_update_config_returns_stage22():
    result = default_stage24_update_config()
    assert isinstance(result, Stage22UpdateConfig)
    assert result.max_update_epochs == 1
    assert result.normalize_advantages is True

def test_stage24_collector_collect_returns_collected_batch():
    model = _gc_model()
    cfg = Stage24CollectorConfig(
        max_episodes=2,
        scenario_names=("risk_gate_hidden_hazard",),
        phase="training",
        seed=24000,
    )
    collected = collect_stage24_stage23a_training_batch(model, cfg)
    assert isinstance(collected, Stage24CollectedBatch)
    assert collected.summary["stage"] == "24-A"
    assert collected.summary["episode_count"] == 2
    assert len(collected.episode_summaries) == 2

def test_stage24_collector_collect_padding_valid_mask_present():
    model = _gc_model()
    cfg = Stage24CollectorConfig(
        max_episodes=2,
        scenario_names=("risk_gate_hidden_hazard",),
        phase="training",
        seed=24000,
    )
    collected = collect_stage24_stage23a_training_batch(model, cfg)
    assert collected.summary["padding_strategy"] == "valid_mask_padding"
    assert collected.summary["rollout_batch_validate_passed"] is True
    # Merged batch survives the value re-check.
    rc.validate_rollout_batch_values(collected.batch)

def test_stage24_collector_collect_episode_summary_keys():
    model = _gc_model()
    cfg = Stage24CollectorConfig(
        max_episodes=2,
        scenario_names=("risk_gate_hidden_hazard",),
        phase="training",
        seed=24000,
    )
    collected = collect_stage24_stage23a_training_batch(model, cfg)
    summary = collected.episode_summaries[0]
    for key in (
        "episode_index",
        "scenario_name",
        "seed",
        "step_count",
        "total_sensing_actions",
        "total_movement_actions",
        "sensing_rate",
    ):
        assert key in summary
    # Naming: total_* (not the action-space-size collision names).
    assert "sensing_action_count" not in summary

def test_stage24_collector_collect_rejects_wrong_model_type():
    with pytest.raises(TypeError, match="RecurrentMAPPOActorCritic"):
        collect_stage24_stage23a_training_batch("not a model")

def test_stage24_collector_collect_rejects_wrong_config_type():
    model = _gc_model()
    with pytest.raises(TypeError, match="Stage24CollectorConfig or None"):
        collect_stage24_stage23a_training_batch(model, "not a config")

def test_stage24_collector_collect_restores_model_mode():
    model = _gc_model()
    model.train()
    cfg = Stage24CollectorConfig(
        max_episodes=2, scenario_names=("risk_gate_hidden_hazard",), seed=24000
    )
    collect_stage24_stage23a_training_batch(model, cfg)
    assert model.training is True

def test_stage24_collector_readiness_update_rejects_wrong_config_type():
    model = _gc_model()
    cfg = Stage24CollectorConfig(
        max_episodes=2, scenario_names=("risk_gate_hidden_hazard",), seed=24000
    )
    batch = collect_stage24_stage23a_training_batch(model, cfg).batch
    with pytest.raises(TypeError, match="Stage24AUpdateConfig or None"):
        stage24a_ppo_lagrangian_readiness_update(model, batch, "not a config")

def test_stage24_collector_runtime_record_observed_only():
    record = stage24a_runtime_reproducibility_record(
        seed_values=(5, 10), device="cpu", dtype="float32"
    )
    assert record["stage"] == "24-A"
    assert record["thread_settings_policy"] == "observed_only"
    assert record["thread_settings_set"] is False
    assert record["seed_values"] == [5, 10]

def test_stage24_collector_runtime_record_rejects_empty_seed_values():
    with pytest.raises(ValueError, match="non-empty tuple"):
        stage24a_runtime_reproducibility_record(
            seed_values=(), device="cpu", dtype="float32"
        )

def test_stage24_collector_runtime_record_rejects_seed_zero():
    with pytest.raises(ValueError, match="seed 0 is intentionally reserved"):
        stage24a_runtime_reproducibility_record(
            seed_values=(0,), device="cpu", dtype="float32"
        )

def test_stage24_collector_runtime_record_rejects_empty_device():
    with pytest.raises(ValueError, match="non-empty string"):
        stage24a_runtime_reproducibility_record(
            seed_values=(5,), device="", dtype="float32"
        )

def test_stage24_collector_runtime_record_rejects_non_string_dtype():
    with pytest.raises(TypeError, match="must be a string"):
        stage24a_runtime_reproducibility_record(
            seed_values=(5,), device="cpu", dtype=5
        )

def test_stage24_collector_runtime_record_rejects_non_bool_allow():
    with pytest.raises(TypeError, match="allow_thread_setting must be a strict bool"):
        stage24a_runtime_reproducibility_record(
            seed_values=(5,), device="cpu", dtype="float32", allow_thread_setting="yes"
        )

def test_stage24_collector_objective_update_diagnosis_keys():
    diag = stage24_objective_update_diagnosis()
    assert diag["stage"] == "24-A"
    assert diag["lagrange_multiplier_can_collapse_to_zero"] is True
    assert diag["final_training_infrastructure_ready"] is False
    assert diag["update_path_changed"] is False
    assert diag["bayesian_belief_claim_made"] is False
    assert diag["formal_voi_claim_made"] is False
    assert diag["model_renamed_to_c_rc_mappo_voi"] is False

def test_stage24_collector_training_smoke_keys():
    result = run_stage24a_training_smoke(seed=24050)
    assert result["stage"] == "24-A"
    assert result["final_evaluation_run"] is False
    assert result["claim_evidence_created"] is False
    assert result["parameters_changed_after_update"] is True
    assert result["batch_validate_passed"] is True
    assert result["stage23a_active_sensing_environment_used"] is True

def test_stage24_collector_training_smoke_rejects_seed_zero():
    with pytest.raises(ValueError, match="seed 0 is intentionally reserved"):
        run_stage24a_training_smoke(seed=0)

def test_stage24_collector_run_stage24_training_smoke_alias():
    result = run_stage24_training_smoke(seed=24051)
    assert result["stage"] == "24-A"

def _gc_package():
    return importlib.import_module("raas_marl.mappo_lagrangian")

def test_init_every_lazy_export_resolves():
    pkg = _gc_package()
    failed = []
    for name in pkg._LAZY_EXPORTS:
        try:
            getattr(pkg, name)
        except Exception as exc:  # pragma: no cover - only on regression
            failed.append((name, type(exc).__name__, str(exc)))
    assert failed == []

def test_init_unknown_attribute_raises_attribute_error():
    pkg = _gc_package()
    with pytest.raises(AttributeError, match="has no attribute"):
        pkg.this_symbol_does_not_exist  # noqa: B018

def test_init_all_is_sorted_and_matches_lazy_exports():
    pkg = _gc_package()
    assert pkg.__all__ == sorted(pkg.__all__)
    assert set(pkg.__all__) == set(pkg._LAZY_EXPORTS)

def test_init_exports_stage22_and_stage23b_result_prefixes():
    pkg = _gc_package()
    assert pkg.STAGE22_RESULT_PREFIX == "stage22_development_training_"
    assert pkg.STAGE23B_RESULT_PREFIX == "stage23b_bounded_training_"

def test_init_exports_result_file_tuples():
    pkg = _gc_package()
    assert "artifact_hashes.json" in pkg.STAGE22_RESULT_FILES
    assert "artifact_hashes.json" in pkg.STAGE23B_RESULT_FILES

def test_init_dir_includes_lazy_names():
    pkg = _gc_package()
    listed = set(dir(pkg))
    assert set(pkg._LAZY_EXPORTS).issubset(listed)

def test_init_lazy_export_caches_in_globals():
    pkg = _gc_package()
    # Access a light lazy symbol and confirm it is cached in module globals.
    value = pkg.MAPPOCoreConfig
    assert pkg.__dict__["MAPPOCoreConfig"] is value

# ===================== SECTION D =====================

_GD_TS = "20260704T101112Z"

def _gD_stay_action() -> dict[str, int]:
    return {SENSING_ACTION_FIELD: 0, MOVEMENT_ACTION_FIELD: 0}

def _gD_action(sensing: int, movement: int) -> dict[str, int]:
    return {SENSING_ACTION_FIELD: sensing, MOVEMENT_ACTION_FIELD: movement}

def _gD_both(a0: dict[str, int], a1: dict[str, int]) -> dict[str, dict[str, int]]:
    return {"agent_0": a0, "agent_1": a1}

def _gD_env(scenario_name: str, **kwargs) -> RiskAwareActiveSensingGridEnvironment:
    return RiskAwareActiveSensingGridEnvironment(
        Stage23EnvironmentConfig(scenario_name=scenario_name, **kwargs)
    )

def _gD_actor_observation() -> dict[str, object]:
    """A well-formed hand-built Stage 23-A actor observation (feature-checked)."""

    return {
        "actor_visible": {
            "position": [2, 3],
            "step_index": 4,
            "max_steps": 8,
            "grid_shape": [8, 8],
            "nearest_goal_delta": [1, -2],
            "local_obstacles": [[2, 4]],
            "revealed_local_hazards": [[3, 3], [2, 5]],
            "previous_sensed": True,
            "previous_invalid_move": False,
            "previous_blocked_move": True,
            "previous_entered_hazard": False,
        },
        "action_factors": {"sensing_action_count": 2, "movement_action_count": 5},
        "identity": {"agent_id": 0},
    }

def _gD_minimal_observation() -> dict[str, object]:
    """A schema-valid, all-empty observation for validation tests."""

    return {
        "actor_visible": {
            "position": [0, 0],
            "step_index": 0,
            "max_steps": 8,
            "grid_shape": [8, 8],
            "nearest_goal_delta": [0, 2],
            "local_obstacles": [],
            "revealed_local_hazards": [],
            "previous_sensed": False,
            "previous_invalid_move": False,
            "previous_blocked_move": False,
            "previous_entered_hazard": False,
        },
        "action_factors": {},
        "identity": {},
    }

def _gD_central_payload() -> dict[str, object]:
    return {
        "stage": "23-A",
        "scenario_name": "s",
        "positions": {"agent_0": [0, 0], "agent_1": [0, 1]},
        "agent_order": ["agent_0", "agent_1"],
        "starts": [[0, 0], [0, 1]],
        "goals": [[0, 2]],
        "obstacles": [],
        "hazard_cells": [],
        "width": 8,
        "height": 8,
        "step_index": 0,
        "max_steps": 8,
        "team_success": False,
        "cumulative": {"task_reward": 0.0, "hazard_cost": 0.0, "sensing_cost": 0.0},
    }

def test_grid_environment_movement_deltas_keys_match_actions():
    # STAGE23_MOVEMENT_DELTAS keys must equal STAGE23_MOVEMENT_ACTIONS keys.
    assert set(STAGE23_MOVEMENT_DELTAS) == set(STAGE23_MOVEMENT_ACTIONS)
    assert set(STAGE23_MOVEMENT_DELTAS) == {0, 1, 2, 3, 4}

def test_grid_environment_movement_deltas_values_are_expected():
    assert STAGE23_MOVEMENT_DELTAS == {
        0: (0, 0),
        1: (-1, 0),
        2: (1, 0),
        3: (0, -1),
        4: (0, 1),
    }

def test_grid_environment_sense_action_index_is_one():
    assert STAGE23_SENSE_ACTION_INDEX == 1

def test_grid_environment_scenario_valid_construction():
    sc = Stage23Scenario(
        name="x",
        width=8,
        height=8,
        starts=((0, 0), (0, 1)),
        goals=((0, 2),),
        obstacles=(),
        hidden_hazard_cells=(),
        description="d",
    )
    assert sc.name == "x"

def test_grid_environment_scenario_empty_name_raises():
    with pytest.raises(ValueError, match="scenario name must be a non-empty string"):
        Stage23Scenario(
            name="",
            width=8,
            height=8,
            starts=((0, 0), (0, 1)),
            goals=((0, 2),),
            obstacles=(),
            hidden_hazard_cells=(),
            description="d",
        )

def test_grid_environment_scenario_empty_description_raises():
    with pytest.raises(ValueError, match="scenario description must be a non-empty string"):
        Stage23Scenario(
            name="x",
            width=8,
            height=8,
            starts=((0, 0), (0, 1)),
            goals=((0, 2),),
            obstacles=(),
            hidden_hazard_cells=(),
            description="  ",
        )

def test_grid_environment_scenario_nonpositive_width_raises():
    with pytest.raises(ValueError):
        Stage23Scenario(
            name="x",
            width=0,
            height=8,
            starts=((0, 0), (0, 1)),
            goals=((0, 2),),
            obstacles=(),
            hidden_hazard_cells=(),
            description="d",
        )

def test_grid_environment_scenario_non_tuple_field_raises_type_error():
    with pytest.raises(TypeError, match="goals must be a tuple"):
        Stage23Scenario(
            name="x",
            width=8,
            height=8,
            starts=((0, 0), (0, 1)),
            goals=[(0, 2)],  # list, not tuple
            obstacles=(),
            hidden_hazard_cells=(),
            description="d",
        )

def test_grid_environment_scenario_requires_exactly_two_starts():
    with pytest.raises(ValueError, match="exactly two starts"):
        Stage23Scenario(
            name="x",
            width=8,
            height=8,
            starts=((0, 0),),
            goals=((0, 2),),
            obstacles=(),
            hidden_hazard_cells=(),
            description="d",
        )

def test_grid_environment_scenario_empty_goals_raises():
    with pytest.raises(ValueError, match="goals must be nonempty"):
        Stage23Scenario(
            name="x",
            width=8,
            height=8,
            starts=((0, 0), (0, 1)),
            goals=(),
            obstacles=(),
            hidden_hazard_cells=(),
            description="d",
        )

def test_grid_environment_scenario_unsorted_goals_raises():
    with pytest.raises(ValueError, match="goals must be sorted deterministically"):
        Stage23Scenario(
            name="x",
            width=8,
            height=8,
            starts=((0, 0), (0, 1)),
            goals=((0, 3), (0, 2)),
            obstacles=(),
            hidden_hazard_cells=(),
            description="d",
        )

def test_grid_environment_scenario_starts_need_not_be_sorted():
    # starts are positional, so a "descending" ordering is allowed.
    sc = Stage23Scenario(
        name="x",
        width=8,
        height=8,
        starts=((0, 5), (0, 1)),
        goals=((0, 2),),
        obstacles=(),
        hidden_hazard_cells=(),
        description="d",
    )
    assert sc.starts == ((0, 5), (0, 1))

def test_grid_environment_scenario_duplicate_starts_raises():
    with pytest.raises(ValueError, match="must contain unique cells"):
        Stage23Scenario(
            name="x",
            width=8,
            height=8,
            starts=((0, 0), (0, 0)),
            goals=((0, 2),),
            obstacles=(),
            hidden_hazard_cells=(),
            description="d",
        )

def test_grid_environment_scenario_out_of_grid_cell_raises():
    with pytest.raises(ValueError, match="outside the grid"):
        Stage23Scenario(
            name="x",
            width=8,
            height=8,
            starts=((0, 0), (0, 1)),
            goals=((99, 2),),
            obstacles=(),
            hidden_hazard_cells=(),
            description="d",
        )

def test_grid_environment_scenario_bad_coordinate_shape_raises_type_error():
    with pytest.raises(TypeError, match="cells must be \\(row, column\\) tuples"):
        Stage23Scenario(
            name="x",
            width=8,
            height=8,
            starts=((0, 0, 0), (0, 1)),
            goals=((0, 2),),
            obstacles=(),
            hidden_hazard_cells=(),
            description="d",
        )

def test_grid_environment_scenario_bool_coordinate_raises_type_error():
    with pytest.raises(TypeError):
        Stage23Scenario(
            name="x",
            width=8,
            height=8,
            starts=((True, 0), (0, 1)),
            goals=((0, 2),),
            obstacles=(),
            hidden_hazard_cells=(),
            description="d",
        )

def test_grid_environment_scenario_start_overlaps_obstacle_raises():
    with pytest.raises(ValueError, match="starts must not overlap obstacles"):
        Stage23Scenario(
            name="x",
            width=8,
            height=8,
            starts=((0, 0), (0, 1)),
            goals=((0, 3),),
            obstacles=((0, 0),),
            hidden_hazard_cells=(),
            description="d",
        )

def test_grid_environment_scenario_start_overlaps_hazard_raises():
    with pytest.raises(ValueError, match="starts must not overlap hidden hazard cells"):
        Stage23Scenario(
            name="x",
            width=8,
            height=8,
            starts=((0, 0), (0, 1)),
            goals=((0, 3),),
            obstacles=(),
            hidden_hazard_cells=((0, 0),),
            description="d",
        )

def test_grid_environment_scenario_goal_overlaps_hazard_raises():
    with pytest.raises(ValueError, match="goals must not overlap hidden hazard cells"):
        Stage23Scenario(
            name="x",
            width=8,
            height=8,
            starts=((0, 0), (0, 1)),
            goals=((0, 3),),
            obstacles=(),
            hidden_hazard_cells=((0, 3),),
            description="d",
        )

def test_grid_environment_scenario_to_json_dict_round_trip():
    sc = stage23_scenario_catalog()["unit_single_hazard"]
    j = sc.to_json_dict()
    assert j["name"] == "unit_single_hazard"
    assert j["hidden_hazard_cells"] == [[1, 1]]
    assert j["starts"] == [[1, 0], [0, 0]]  # positional order preserved
    assert j["goals"] == [[2, 2]]

def test_grid_environment_config_default_valid():
    cfg = Stage23EnvironmentConfig()
    assert cfg.stage == "23-A"
    assert cfg.agent_count == 2

def test_grid_environment_config_stage_lock_raises():
    with pytest.raises(ValueError, match="stage must be '23-A'"):
        Stage23EnvironmentConfig(stage="X")

def test_grid_environment_config_empty_scenario_name_raises():
    with pytest.raises(ValueError, match="scenario_name must be a non-empty string"):
        Stage23EnvironmentConfig(scenario_name="   ")

def test_grid_environment_config_unknown_scenario_raises():
    with pytest.raises(ValueError, match="unknown Stage 23-A scenario"):
        Stage23EnvironmentConfig(scenario_name="nope")

def test_grid_environment_config_agent_count_must_be_two():
    with pytest.raises(ValueError, match="exactly two agents"):
        Stage23EnvironmentConfig(agent_count=3)

def test_grid_environment_config_nonpositive_max_steps_raises():
    with pytest.raises(ValueError):
        Stage23EnvironmentConfig(max_steps=0)

def test_grid_environment_config_positive_step_reward_raises():
    with pytest.raises(ValueError, match="positive step rewards invert"):
        Stage23EnvironmentConfig(step_reward=0.5)

def test_grid_environment_config_nonpositive_success_reward_raises():
    with pytest.raises(ValueError, match="success_reward must be positive"):
        Stage23EnvironmentConfig(success_reward=0.0)

def test_grid_environment_config_nonpositive_sensing_cost_raises():
    with pytest.raises(ValueError, match="sensing_cost_value must be positive"):
        Stage23EnvironmentConfig(sensing_cost_value=0.0)

def test_grid_environment_config_nonpositive_hazard_cost_raises():
    with pytest.raises(ValueError, match="hazard_cost_value must be positive"):
        Stage23EnvironmentConfig(hazard_cost_value=-1.0)

def test_grid_environment_config_positive_invalid_move_penalty_raises():
    with pytest.raises(ValueError, match="invalid_move_penalty must not be positive"):
        Stage23EnvironmentConfig(invalid_move_penalty=0.5)

def test_grid_environment_config_bool_seed_raises_type_error():
    with pytest.raises(TypeError):
        Stage23EnvironmentConfig(seed=True)

def test_grid_environment_config_none_seed_allowed():
    cfg = Stage23EnvironmentConfig(seed=None)
    assert cfg.seed is None

def test_grid_environment_config_negative_sensing_radius_raises():
    with pytest.raises(ValueError):
        Stage23EnvironmentConfig(sensing_radius=-1)

def test_grid_environment_config_catalog_dimension_mismatch_without_custom_layout_raises():
    with pytest.raises(ValueError, match="catalog scenario dimensions must match"):
        Stage23EnvironmentConfig(scenario_name="unit_empty", width=6, height=6)

def test_grid_environment_config_custom_layout_missing_fields_raises():
    with pytest.raises(ValueError, match="custom scenario dimensions require all layout fields"):
        Stage23EnvironmentConfig(
            scenario_name="unit_empty",
            width=4,
            height=4,
            start_positions=((0, 0), (0, 1)),
            # goal_cells / obstacle_cells / hidden_hazard_cells missing
        )

def test_grid_environment_config_to_json_dict_shape():
    j = Stage23EnvironmentConfig().to_json_dict()
    assert j["stage"] == "23-A"
    assert j["start_positions"] is None
    assert j["max_steps"] == 32

def test_grid_environment_reset_returns_two_agents():
    env = _gD_env("unit_empty", seed=23)
    obs, infos = env.reset(seed=23)
    assert set(obs) == set(STAGE23_AGENT_NAMES)
    assert set(infos) == set(STAGE23_AGENT_NAMES)
    assert env.agents == list(env.possible_agents)

def test_grid_environment_reset_determinism_after_step():
    # Hand-computed regression: reset re-initializes all state deterministically.
    env = _gD_env("standard_branching_hazard", seed=23)
    obs1, infos1 = env.reset(seed=23)
    env.step(_gD_both(_gD_action(0, 4), _gD_action(0, 4)))
    obs2, infos2 = env.reset(seed=23)
    assert obs1 == obs2
    assert infos1 == infos2
    assert env.step_index == 0
    assert env.is_done is False

def test_grid_environment_reset_initial_positions_match_starts():
    env = _gD_env("unit_single_hazard", seed=23)
    obs, _ = env.reset(seed=23)
    assert obs["agent_0"]["actor_visible"]["position"] == [1, 0]
    assert obs["agent_1"]["actor_visible"]["position"] == [0, 0]

def test_grid_environment_step_before_reset_raises():
    env = _gD_env("unit_empty", seed=23)
    with pytest.raises(RuntimeError, match="environment is done; call reset"):
        env.step(_gD_both(_gD_stay_action(), _gD_stay_action()))

def test_grid_environment_step_after_done_raises():
    env = _gD_env("unit_empty", max_steps=1, seed=23)
    env.reset(seed=23)
    env.step(_gD_both(_gD_stay_action(), _gD_stay_action()))  # truncates
    assert env.is_done is True
    with pytest.raises(RuntimeError, match="call reset before stepping again"):
        env.step(_gD_both(_gD_stay_action(), _gD_stay_action()))

def test_grid_environment_bad_config_type_raises():
    with pytest.raises(TypeError, match="config must be a Stage23EnvironmentConfig or None"):
        RiskAwareActiveSensingGridEnvironment(config="x")

def test_grid_environment_step_non_mapping_actions_raises():
    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    with pytest.raises(TypeError, match="actions must be a mapping"):
        env.step([1, 2])

def test_grid_environment_step_non_string_agent_key_raises():
    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    with pytest.raises(TypeError, match="must be a string"):
        env.step({0: _gD_stay_action(), "agent_1": _gD_stay_action()})

def test_grid_environment_step_missing_agent_raises():
    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    with pytest.raises(ValueError, match="actions must match active agents"):
        env.step({"agent_0": _gD_stay_action()})

def test_grid_environment_step_extra_agent_raises():
    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    with pytest.raises(ValueError, match="actions must match active agents"):
        env.step(
            {
                "agent_0": _gD_stay_action(),
                "agent_1": _gD_stay_action(),
                "agent_2": _gD_stay_action(),
            }
        )

def test_grid_environment_step_non_mapping_agent_payload_raises():
    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    with pytest.raises(TypeError, match="each agent action must be a mapping"):
        env.step({"agent_0": [0, 0], "agent_1": _gD_stay_action()})

def test_grid_environment_step_missing_action_factor_raises():
    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    with pytest.raises(ValueError, match="must be exactly sensing_action and movement_action"):
        env.step({"agent_0": {"sensing_action": 0}, "agent_1": _gD_stay_action()})

def test_grid_environment_step_out_of_range_movement_raises():
    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    with pytest.raises(ValueError, match="movement_action must be in \\[0, 4\\]"):
        env.step(_gD_both(_gD_action(0, 9), _gD_stay_action()))

def test_grid_environment_step_out_of_range_sensing_raises():
    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    with pytest.raises(ValueError, match="sensing_action must be in \\[0, 1\\]"):
        env.step(_gD_both(_gD_action(5, 0), _gD_stay_action()))

def test_grid_environment_step_bool_action_raises_type_error():
    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    with pytest.raises(TypeError, match="sensing_action must be an int"):
        env.step(_gD_both({"sensing_action": True, "movement_action": 0}, _gD_stay_action()))

def test_grid_environment_movement_stay_keeps_position():
    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    o, _, _, _, i = env.step(_gD_both(_gD_action(0, 0), _gD_stay_action()))
    assert o["agent_0"]["actor_visible"]["position"] == [0, 0]
    assert i["agent_0"]["invalid_move"] is False
    assert i["agent_0"]["blocked_move"] is False

def test_grid_environment_movement_east_and_south_resolve():
    # agent_0 at (0,0): east -> (0,1); a fresh episode south -> (1,0).
    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    o, _, _, _, _ = env.step(_gD_both(_gD_action(0, 4), _gD_stay_action()))
    assert o["agent_0"]["actor_visible"]["position"] == [0, 1]
    env.reset(seed=23)
    o, _, _, _, _ = env.step(_gD_both(_gD_action(0, 2), _gD_stay_action()))
    assert o["agent_0"]["actor_visible"]["position"] == [1, 0]

def test_grid_environment_movement_wall_is_invalid_move():
    # agent_0 at (0,0) moving north (1) -> (-1,0) off grid: invalid_move, unchanged.
    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    o, _, _, _, i = env.step(_gD_both(_gD_action(0, 1), _gD_stay_action()))
    assert o["agent_0"]["actor_visible"]["position"] == [0, 0]
    assert i["agent_0"]["invalid_move"] is True
    assert i["agent_0"]["blocked_move"] is False
    # next observation reflects previous_invalid_move
    assert o["agent_0"]["actor_visible"]["previous_invalid_move"] is True

def test_grid_environment_movement_obstacle_is_blocked_move():
    # standard_branching_hazard obstacle at (1,3): reach (0,3) then south into it.
    env = _gD_env("standard_branching_hazard", seed=23)
    env.reset(seed=23)
    for _ in range(3):
        o, _, _, _, _ = env.step(_gD_both(_gD_action(0, 4), _gD_stay_action()))
    assert o["agent_0"]["actor_visible"]["position"] == [0, 3]
    o, _, _, _, i = env.step(_gD_both(_gD_action(0, 2), _gD_stay_action()))
    assert o["agent_0"]["actor_visible"]["position"] == [0, 3]
    assert i["agent_0"]["blocked_move"] is True
    assert i["agent_0"]["invalid_move"] is False

def test_grid_environment_hazard_charged_on_entry():
    # unit_single_hazard: hazard (1,1). agent_0 (1,0) east -> (1,1).
    env = _gD_env("unit_single_hazard", seed=23, hazard_cost_value=1.0)
    env.reset(seed=23)
    o, _, _, _, i = env.step(_gD_both(_gD_action(0, 4), _gD_stay_action()))
    assert o["agent_0"]["actor_visible"]["position"] == [1, 1]
    assert i["agent_0"]["entered_hazard"] is True
    assert i["agent_0"]["hazard_cost"] == 1.0

def test_grid_environment_hazard_recharges_when_staying():
    # Occupancy semantics: staying on a hazard re-charges every step.
    env = _gD_env("unit_single_hazard", seed=23, hazard_cost_value=1.0)
    env.reset(seed=23)
    env.step(_gD_both(_gD_action(0, 4), _gD_stay_action()))  # onto hazard
    o, _, _, _, i = env.step(_gD_both(_gD_action(0, 0), _gD_stay_action()))  # stay
    assert o["agent_0"]["actor_visible"]["position"] == [1, 1]
    assert i["agent_0"]["entered_hazard"] is True
    assert i["agent_0"]["hazard_cost"] == 1.0

def test_grid_environment_no_hazard_cost_off_hazard():
    env = _gD_env("unit_single_hazard", seed=23, hazard_cost_value=1.0)
    env.reset(seed=23)
    o, _, _, _, i = env.step(_gD_both(_gD_action(0, 0), _gD_stay_action()))
    assert i["agent_0"]["entered_hazard"] is False
    assert i["agent_0"]["hazard_cost"] == 0.0

def test_grid_environment_sensing_reveals_post_move_position_next_step():
    # sense while moving (1,0)->(1,1): reveal window centered at (1,1) radius 1.
    env = _gD_env("unit_single_hazard", seed=23)
    env.reset(seed=23)
    o, _, _, _, i = env.step(_gD_both(_gD_action(1, 4), _gD_stay_action()))
    # revealed appear in the NEXT observation returned by this step
    assert o["agent_0"]["actor_visible"]["revealed_local_hazards"] == [[1, 1]]
    assert i["agent_0"]["revealed_hazard_count"] == 1
    assert i["agent_0"]["sensed"] is True

def test_grid_environment_sensing_charged_cost():
    env = _gD_env("unit_single_hazard", seed=23, sensing_cost_value=0.05)
    env.reset(seed=23)
    _, _, _, _, i = env.step(_gD_both(_gD_action(1, 0), _gD_stay_action()))
    assert i["agent_0"]["sensing_cost"] == 0.05

def test_grid_environment_no_sense_reveals_nothing_and_clears():
    env = _gD_env("unit_single_hazard", seed=23)
    env.reset(seed=23)
    env.step(_gD_both(_gD_action(1, 4), _gD_stay_action()))  # reveal
    o, _, _, _, _ = env.step(_gD_both(_gD_action(0, 0), _gD_stay_action()))  # no sense
    assert o["agent_0"]["actor_visible"]["revealed_local_hazards"] == []

def test_grid_environment_team_success_pays_all_agents():
    # unit_empty: a0 (0,0) a1 (0,1) goal (0,2). Both must be on goal same step.
    env = _gD_env("unit_empty", seed=23, success_reward=1.0, step_reward=-0.01)
    env.reset(seed=23)
    # a0 east, a1 east -> a0 (0,1), a1 (0,2): not both on goal
    _, r1, t1, _, _ = env.step(_gD_both(_gD_action(0, 4), _gD_action(0, 4)))
    assert t1["agent_0"] is False
    # a0 east -> (0,2), a1 stay -> both on goal
    _, r2, t2, tr2, _ = env.step(_gD_both(_gD_action(0, 4), _gD_stay_action()))
    assert t2["agent_0"] is True and t2["agent_1"] is True
    assert tr2["agent_0"] is False
    # each agent gets step_reward + success_reward = -0.01 + 1.0
    assert r2["agent_0"] == pytest.approx(0.99)
    assert r2["agent_1"] == pytest.approx(0.99)
    assert env.is_done is True
    assert env.agents == []

def test_grid_environment_truncation_exactly_at_max_steps():
    env = _gD_env("unit_empty", max_steps=2, seed=23)
    env.reset(seed=23)
    _, _, t1, tr1, _ = env.step(_gD_both(_gD_stay_action(), _gD_stay_action()))
    assert t1["agent_0"] is False and tr1["agent_0"] is False
    assert env.is_done is False
    _, _, t2, tr2, _ = env.step(_gD_both(_gD_stay_action(), _gD_stay_action()))
    assert t2["agent_0"] is False
    assert tr2["agent_0"] is True and tr2["agent_1"] is True
    assert env.is_done is True

def test_grid_environment_truncation_not_flagged_on_success_step():
    # If success occurs on the horizon step, truncated must be False.
    env = _gD_env("unit_empty", max_steps=2, seed=23, step_reward=-0.01)
    env.reset(seed=23)
    env.step(_gD_both(_gD_action(0, 4), _gD_action(0, 4)))  # a0->(0,1), a1->(0,2)
    _, _, t2, tr2, _ = env.step(_gD_both(_gD_action(0, 4), _gD_stay_action()))
    assert t2["agent_0"] is True
    assert tr2["agent_0"] is False

def test_grid_environment_critic_visible_state_before_reset_raises():
    env = _gD_env("unit_single_hazard", seed=23)
    with pytest.raises(RuntimeError, match="must be reset before critic_visible_state"):
        env.critic_visible_state()

def test_grid_environment_critic_visible_state_includes_hidden_hazards():
    env = _gD_env("unit_single_hazard", seed=23)
    env.reset(seed=23)
    cs = env.critic_visible_state()
    assert cs["hazard_cells"] == [[1, 1]]  # privileged hidden hazard
    assert cs["stage"] == "23-A"
    assert cs["positions"] == {"agent_0": [1, 0], "agent_1": [0, 0]}
    assert cs["agent_order"] == ["agent_0", "agent_1"]
    assert cs["cumulative"] == {
        "task_reward": 0.0,
        "hazard_cost": 0.0,
        "sensing_cost": 0.0,
    }

def test_grid_environment_critic_visible_state_readable_after_done():
    env = _gD_env("unit_empty", max_steps=1, seed=23)
    env.reset(seed=23)
    env.step(_gD_both(_gD_stay_action(), _gD_stay_action()))  # truncate, clears agents
    cs = env.critic_visible_state()  # still readable
    assert cs["team_success"] is False
    assert set(cs["positions"]) == set(STAGE23_AGENT_NAMES)

def test_grid_environment_critic_cumulative_accumulates():
    env = _gD_env("unit_single_hazard", seed=23, hazard_cost_value=1.0, sensing_cost_value=0.05)
    env.reset(seed=23)
    env.step(_gD_both(_gD_action(1, 4), _gD_stay_action()))  # sense + into hazard
    cs = env.critic_visible_state()
    assert cs["cumulative"]["hazard_cost"] == pytest.approx(1.0)
    assert cs["cumulative"]["sensing_cost"] == pytest.approx(0.05)

def test_grid_environment_validate_observation_happy_path():
    validate_stage23_actor_observation(_gD_minimal_observation())

def test_grid_environment_validate_observation_from_live_env():
    env = _gD_env("unit_single_hazard", seed=23)
    obs, _ = env.reset(seed=23)
    validate_stage23_actor_observation(obs["agent_0"])

def test_grid_environment_validate_observation_non_mapping_raises():
    with pytest.raises(TypeError, match="observation must be a mapping"):
        validate_stage23_actor_observation([1, 2])

def test_grid_environment_validate_observation_non_string_top_key_raises():
    obs = _gD_minimal_observation()
    obs[1] = "x"
    with pytest.raises(TypeError, match="observation top-level key must be a string"):
        validate_stage23_actor_observation(obs)

def test_grid_environment_validate_observation_forbidden_top_level_key_raises():
    obs = _gD_minimal_observation()
    obs["central_state"] = 1
    with pytest.raises(ValueError, match="forbidden actor-visible information key"):
        validate_stage23_actor_observation(obs)

def test_grid_environment_validate_observation_unsupported_top_level_key_raises():
    obs = _gD_minimal_observation()
    obs["extra"] = 1
    with pytest.raises(ValueError, match="unsupported top-level key"):
        validate_stage23_actor_observation(obs)

def test_grid_environment_validate_observation_recursive_forbidden_key_raises():
    # Hand-computed: recursive scan flags a forbidden key nested deep in the payload.
    obs = _gD_minimal_observation()
    obs["action_factors"] = {"nested": {"hidden_hazard_map": [1]}}
    with pytest.raises(ValueError, match="forbidden key at observation"):
        validate_stage23_actor_observation(obs)

def test_grid_environment_validate_observation_forbidden_key_in_list_raises():
    obs = _gD_minimal_observation()
    obs["identity"] = {"list_field": [{"critic_visible": 1}]}
    with pytest.raises(ValueError, match="forbidden key"):
        validate_stage23_actor_observation(obs)

def test_grid_environment_validate_observation_missing_actor_visible_field_raises():
    obs = _gD_minimal_observation()
    del obs["actor_visible"]["position"]
    with pytest.raises(ValueError, match="missing required field: position"):
        validate_stage23_actor_observation(obs)

def test_grid_environment_validate_observation_unsupported_actor_visible_key_raises():
    obs = _gD_minimal_observation()
    obs["actor_visible"]["bad"] = 1
    with pytest.raises(ValueError, match="actor_visible contains unsupported key: bad"):
        validate_stage23_actor_observation(obs)

def test_grid_environment_validate_observation_missing_actor_visible_mapping_raises():
    obs = _gD_minimal_observation()
    obs["actor_visible"] = [1, 2]
    with pytest.raises(ValueError, match="requires actor_visible mapping"):
        validate_stage23_actor_observation(obs)

def test_grid_environment_validate_observation_non_mapping_action_factors_raises():
    obs = _gD_minimal_observation()
    obs["action_factors"] = [1]
    with pytest.raises(ValueError, match="action_factors must be a mapping when present"):
        validate_stage23_actor_observation(obs)

def test_grid_environment_forbidden_actor_keys_includes_stage_specific():
    assert "critic_visible" in STAGE23_FORBIDDEN_ACTOR_KEYS
    assert "central_state" in STAGE23_FORBIDDEN_ACTOR_KEYS
    assert "hidden_hazard_map" in STAGE23_FORBIDDEN_ACTOR_KEYS

def test_grid_environment_actor_visible_schema_keys_are_eleven():
    assert len(STAGE23_ACTOR_VISIBLE_SCHEMA_KEYS) == 11

def test_grid_environment_catalog_has_fourteen_scenarios():
    # The original four plus the DR-D1/L1 risk-fork family (2 training + 2
    # readiness catalog scenarios) plus the DR-READINESS-CURRICULUM gate-row
    # curriculum family (3 aliased pairs at gate rows {0,3}/{4,7}/{0,7}).
    # Held-out fork scenarios (gates rows 5,6) remain DEFERRED (I-3).
    catalog = stage23_scenario_catalog()
    assert set(catalog) == {
        "standard_branching_hazard",
        "risk_gate_hidden_hazard",
        "unit_empty",
        "unit_single_hazard",
        "risk_fork_train_upper",
        "risk_fork_train_lower",
        "risk_fork_readiness_upper",
        "risk_fork_readiness_lower",
        "risk_fork_curriculum_r0r3_upper",
        "risk_fork_curriculum_r0r3_lower",
        "risk_fork_curriculum_r4r7_upper",
        "risk_fork_curriculum_r4r7_lower",
        "risk_fork_curriculum_r0r7_upper",
        "risk_fork_curriculum_r0r7_lower",
    }

def test_grid_environment_catalog_entries_are_reachable():
    # BFS reachability runs inside Stage23Scenario construction: no raise == OK.
    for name, sc in stage23_scenario_catalog().items():
        assert sc.name == name

def test_scenarios_available_scenarios_sorted_and_unique():
    names = available_scenarios()
    assert list(names) == sorted(names)
    assert len(set(names)) == len(names)
    assert names == (
        "risk_fork_curriculum_r0r3_lower",
        "risk_fork_curriculum_r0r3_upper",
        "risk_fork_curriculum_r0r7_lower",
        "risk_fork_curriculum_r0r7_upper",
        "risk_fork_curriculum_r4r7_lower",
        "risk_fork_curriculum_r4r7_upper",
        "risk_fork_readiness_lower",
        "risk_fork_readiness_upper",
        "risk_fork_train_lower",
        "risk_fork_train_upper",
        "risk_gate_hidden_hazard",
        "standard_branching_hazard",
        "unit_empty",
        "unit_single_hazard",
    )

def test_scenarios_make_scenario_returns_catalog_entry():
    sc = make_scenario("unit_empty")
    assert isinstance(sc, Stage23Scenario)
    assert sc.name == "unit_empty"

def test_scenarios_make_scenario_non_string_raises_type_error():
    with pytest.raises(TypeError, match="scenario name must be a str"):
        make_scenario(123)

def test_scenarios_make_scenario_unknown_raises_value_error():
    with pytest.raises(ValueError, match="unknown active-sensing scenario: nope"):
        make_scenario("nope")

def test_scenarios_make_scenario_identity_matches_catalog():
    catalog = stage23_scenario_catalog()
    assert make_scenario("unit_single_hazard") == catalog["unit_single_hazard"]

def test_tensor_adapter_central_schema_length_matches_dim():
    assert len(STAGE23_CENTRAL_STATE_FEATURE_SCHEMA) == STAGE23_CENTRAL_STATE_DIM == 20

def test_tensor_adapter_central_schema_order_is_named():
    assert STAGE23_CENTRAL_STATE_FEATURE_SCHEMA == (
        "time_progress",
        "grid_width",
        "grid_height",
        "active_agent_count",
        "agent_0_row",
        "agent_0_col",
        "agent_1_row",
        "agent_1_col",
        "start_count",
        "goal_count",
        "obstacle_count",
        "hazard_cell_count",
        "team_success",
        "cumulative_task_reward",
        "cumulative_hazard_cost",
        "cumulative_sensing_cost",
        "bias",
        "sensing_action_count",
        "movement_action_count",
        "configured_agent_count",
    )

def test_tensor_adapter_default_core_config_dims():
    core = default_stage23_core_config()
    # DR-REPRESENTATION (candidate A): actor_observation_dim 10 -> 22 (10 legacy
    # features + the 12-cell egocentric radius-2 obstacle-occupancy patch appended at
    # indices >=10). REPOINTED not deleted; the NHR-read indices 0,1,5 stay fixed.
    assert core.actor_observation_dim == STAGE23_ACTOR_OBSERVATION_DIM == 22
    assert core.revealed_information_dim == STAGE23_REVEALED_INFORMATION_DIM == 4
    assert core.central_state_dim == STAGE23_CENTRAL_STATE_DIM == 20
    assert core.sensing_action_count == 2
    assert core.movement_action_count == 5
    assert core.agent_id_count == 2
    assert core.dtype == "float32"
    assert core.device == "cpu"

def test_tensor_adapter_actor_tensor_shape():
    t = actor_observation_from_stage23(_gD_actor_observation())
    assert tuple(t.shape) == (1, 1, 22)

def test_tensor_adapter_actor_tensor_feature_by_feature():
    # Hand-computed regression against the exact feature construction.
    # position (2,3), grid (8,8): row 2/7, col 3/7
    # goal_delta (1,-2): 1/7, -2/7
    # time 4/8 = 0.5; sensed=1, invalid=0, blocked=1, entered=0
    # DR-REPRESENTATION: local_observation_radius default is now 2, so the obstacle
    # COUNT cap = 2*2*(2+1)+1 = 13; count 1 -> min(1,13)/13 = 1/13 (was 0.2 at r=1).
    # The 12-cell egocentric radius-2 obstacle patch (indices 10..21) is appended:
    # obstacle (2,4) sits at offset (0,+1) from position (2,3) => the 7th patch cell
    # (index 16) is 1.0; all other in-grid patch cells (position (2,3) is >=2 rows/cols
    # from every edge, so no off-grid sentinels here) are 0.0.
    t = actor_observation_from_stage23(_gD_actor_observation())
    features = [float(x) for x in t.reshape(-1)]
    expected = [
        2 / 7,
        3 / 7,
        1 / 7,
        -2 / 7,
        0.5,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0 / 13.0,
        # radius-2 obstacle patch, offsets sorted (dr,dc) ascending:
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    ]
    assert features == pytest.approx(expected)

def test_tensor_adapter_actor_tensor_goal_target_out_of_grid_raises():
    obs = _gD_minimal_observation()
    obs["actor_visible"]["nearest_goal_delta"] = [-1, 0]  # (0,0)+(-1,0) = (-1,0)
    with pytest.raises(ValueError, match="nearest_goal_delta target"):
        actor_observation_from_stage23(obs)

def test_tensor_adapter_actor_tensor_bad_previous_flag_type_raises():
    obs = _gD_minimal_observation()
    obs["actor_visible"]["previous_sensed"] = 1  # not a bool
    with pytest.raises(TypeError, match="previous_sensed must be a bool"):
        actor_observation_from_stage23(obs)

def test_tensor_adapter_actor_tensor_uses_forbidden_key_scan():
    obs = _gD_minimal_observation()
    obs["actor_visible"]["central_state"] = [1]
    with pytest.raises(ValueError):
        actor_observation_from_stage23(obs)

def test_tensor_adapter_reveal_tensor_shape_and_features():
    # Hand-computed: revealed [[3,3],[2,5]], position (2,3).
    # nearest by Manhattan: (3,3) dist 1 < (2,5) dist 2 -> nearest (3,3)
    # presence 1.0; count 2, cap 5 -> 0.4; delta (3-2)/7, (3-3)/7
    r = revealed_information_from_stage23(_gD_actor_observation())
    assert tuple(r.shape) == (1, 1, 4)
    features = [float(x) for x in r.reshape(-1)]
    assert features == pytest.approx([1.0, 0.4, 1 / 7, 0.0])

def test_tensor_adapter_reveal_tensor_empty_is_zeros():
    obs = _gD_actor_observation()
    obs["actor_visible"]["revealed_local_hazards"] = []
    r = revealed_information_from_stage23(obs)
    assert [float(x) for x in r.reshape(-1)] == [0.0, 0.0, 0.0, 0.0]

def test_tensor_adapter_reveal_tensor_nearest_tie_break():
    # Two reveals equidistant: (2,2) and (2,4) are both dist 1 from (2,3);
    # tie-break by (dist,row,col) -> (2,2).
    obs = _gD_minimal_observation()
    obs["actor_visible"]["position"] = [2, 3]
    obs["actor_visible"]["nearest_goal_delta"] = [0, 0]
    obs["actor_visible"]["revealed_local_hazards"] = [[2, 2], [2, 4]]
    r = revealed_information_from_stage23(obs)
    features = [float(x) for x in r.reshape(-1)]
    # nearest (2,2): delta (0, -1/7)
    assert features == pytest.approx([1.0, 2 / 5, 0.0, -1 / 7])

def test_tensor_adapter_central_tensor_from_environment():
    env = _gD_env("unit_single_hazard", seed=23)
    env.reset(seed=23)
    t = central_state_from_stage23(env)
    assert tuple(t.shape) == (1, 1, 20)

def test_tensor_adapter_central_tensor_feature_values():
    # Hand-computed for unit_single_hazard at reset.
    # positions a0 (1,0), a1 (0,0); width/height 8; goal (2,2); hazard (1,1)
    env = _gD_env("unit_single_hazard", seed=23)
    env.reset(seed=23)
    t = central_state_from_stage23(env)
    features = {
        name: float(v)
        for name, v in zip(STAGE23_CENTRAL_STATE_FEATURE_SCHEMA, t.reshape(-1))
    }
    assert features["time_progress"] == 0.0
    assert features["grid_width"] == 8.0
    assert features["grid_height"] == 8.0
    assert features["active_agent_count"] == 2.0
    assert features["agent_0_row"] == pytest.approx(1 / 7)
    assert features["agent_0_col"] == 0.0
    assert features["agent_1_row"] == 0.0
    assert features["agent_1_col"] == 0.0
    assert features["start_count"] == 2.0
    assert features["goal_count"] == 1.0
    assert features["obstacle_count"] == 0.0
    assert features["hazard_cell_count"] == 1.0
    assert features["team_success"] == 0.0
    assert features["cumulative_task_reward"] == 0.0
    assert features["bias"] == 1.0
    assert features["sensing_action_count"] == 2.0
    assert features["movement_action_count"] == 5.0
    assert features["configured_agent_count"] == 2.0

def test_tensor_adapter_central_from_payload_happy_path():
    t = central_state_from_stage23(_gD_central_payload())
    assert tuple(t.shape) == (1, 1, 20)

def test_tensor_adapter_central_missing_key_raises():
    payload = _gD_central_payload()
    del payload["stage"]
    with pytest.raises(ValueError, match="missing required field: stage"):
        central_state_from_stage23(payload)

def test_tensor_adapter_central_wrong_stage_raises():
    payload = _gD_central_payload()
    payload["stage"] = "wrong"
    with pytest.raises(ValueError, match="central payload stage must equal '23-A'"):
        central_state_from_stage23(payload)

def test_tensor_adapter_central_empty_scenario_name_raises():
    payload = _gD_central_payload()
    payload["scenario_name"] = ""
    with pytest.raises(ValueError):
        central_state_from_stage23(payload)

def test_tensor_adapter_central_wrong_position_keys_raises():
    payload = _gD_central_payload()
    payload["positions"] = {"agent_0": [0, 0], "agent_9": [0, 1]}
    with pytest.raises(ValueError, match="positions keys must exactly equal"):
        central_state_from_stage23(payload)

def test_tensor_adapter_central_wrong_agent_order_raises():
    payload = _gD_central_payload()
    payload["agent_order"] = ["agent_1", "agent_0"]
    with pytest.raises(ValueError, match="agent_order must exactly equal"):
        central_state_from_stage23(payload)

def test_tensor_adapter_central_negative_hazard_cost_raises():
    payload = _gD_central_payload()
    payload["cumulative"] = {"task_reward": 0.0, "hazard_cost": -1.0, "sensing_cost": 0.0}
    with pytest.raises(ValueError, match="hazard_cost must be nonnegative"):
        central_state_from_stage23(payload)

def test_tensor_adapter_central_non_env_non_mapping_raises_type_error():
    with pytest.raises(TypeError, match="must be a Stage 23-A environment or mapping"):
        central_state_from_stage23(42)

def test_tensor_adapter_central_disjoint_layout_violation_raises():
    payload = _gD_central_payload()
    payload["obstacles"] = [[0, 0]]  # overlaps a start
    with pytest.raises(ValueError):
        central_state_from_stage23(payload)

def test_tensor_adapter_non_cpu_device_raises_value_error():
    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    with pytest.raises(ValueError, match="support only CPU device 'cpu'"):
        central_state_from_stage23(env, device="cuda")

def test_tensor_adapter_non_string_device_raises_type_error():
    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    with pytest.raises(TypeError, match="device must be a string"):
        central_state_from_stage23(env, device=123)

def test_tensor_adapter_bad_dtype_raises_value_error():
    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    with pytest.raises(ValueError, match="dtype must be 'float32' or 'float64'"):
        central_state_from_stage23(env, dtype="float16")

def test_tensor_adapter_float64_dtype_path():
    import torch

    env = _gD_env("unit_empty", seed=23)
    env.reset(seed=23)
    # a float64 core config so validate_central_state accepts float64.
    core = default_stage23_core_config()
    core64 = MAPPOCoreConfig(
        actor_observation_dim=core.actor_observation_dim,
        revealed_information_dim=core.revealed_information_dim,
        central_state_dim=core.central_state_dim,
        history_state_dim=core.history_state_dim,
        actor_hidden_dim=core.actor_hidden_dim,
        critic_hidden_dim=core.critic_hidden_dim,
        sensing_action_count=core.sensing_action_count,
        movement_action_count=core.movement_action_count,
        recurrent_layer_count=core.recurrent_layer_count,
        agent_id_count=core.agent_id_count,
        use_recurrent_actor=core.use_recurrent_actor,
        dtype="float64",
        device="cpu",
    )
    t = central_state_from_stage23(env, config=core64)
    assert t.dtype == torch.float64

def test_tensor_adapter_core_wrong_config_type_raises():
    with pytest.raises(TypeError, match="config must be a MAPPOCoreConfig or None"):
        central_state_from_stage23(_gD_central_payload(), config="bad")

def test_tensor_adapter_core_wrong_sensing_count_raises():
    core = default_stage23_core_config()
    bad = MAPPOCoreConfig(
        actor_observation_dim=core.actor_observation_dim,
        revealed_information_dim=core.revealed_information_dim,
        central_state_dim=core.central_state_dim,
        history_state_dim=core.history_state_dim,
        actor_hidden_dim=core.actor_hidden_dim,
        critic_hidden_dim=core.critic_hidden_dim,
        sensing_action_count=3,
        movement_action_count=5,
        recurrent_layer_count=core.recurrent_layer_count,
        agent_id_count=core.agent_id_count,
        use_recurrent_actor=core.use_recurrent_actor,
        dtype="float32",
        device="cpu",
    )
    with pytest.raises(ValueError, match="requires sensing_action_count == 2"):
        central_state_from_stage23(_gD_central_payload(), config=bad)

def test_tensor_adapter_transition_contract_happy_path():
    env = _gD_env("unit_single_hazard", seed=23)
    obs, _ = env.reset(seed=23)
    o, r, term, trunc, i = env.step(_gD_both(_gD_action(1, 4), _gD_stay_action()))
    contract = stage23_transition_to_environment_step_contract(o["agent_0"], i["agent_0"], env)
    # It returns a validated EnvironmentStepContract; central_state carries privilege.
    assert "central_state" in contract.critic_visible
    assert "actor_observation" in contract.actor_visible
    assert "revealed_information" in contract.actor_visible

def test_tensor_adapter_transition_contract_identity_mismatch_raises():
    env = _gD_env("unit_single_hazard", seed=23)
    obs, _ = env.reset(seed=23)
    o, r, term, trunc, i = env.step(_gD_both(_gD_action(0, 4), _gD_stay_action()))
    # Corrupt the info identity so it disagrees with the observation identity.
    bad_info = copy.deepcopy(i["agent_0"])
    bad_info["identity"]["agent_id"] = 99
    with pytest.raises(ValueError, match="transition identity mismatch"):
        stage23_transition_to_environment_step_contract(o["agent_0"], bad_info, env)

def test_construction_runner_is_torch_free_on_import():
    import sys

    # Importing the construction runner alone must not pull torch.
    # (This module is already imported by the test file's own imports.)
    import importlib

    importlib.import_module("raas_marl.environments.active_sensing.construction_runner")
    # torch may be present from other adapter tests in a merged run, so we only
    # assert the module itself declares no torch attribute (torch-free contract).
    module = sys.modules["raas_marl.environments.active_sensing.construction_runner"]
    assert not hasattr(module, "torch")

def test_construction_runner_writes_exact_seven_file_set(tmp_path):
    root = run_environment_construction(result_parent=tmp_path, timestamp_utc=_GD_TS)
    names = sorted(p.name for p in root.iterdir())
    assert names == sorted(STAGE23A_RESULT_FILES)
    assert len(STAGE23A_RESULT_FILES) == 7

def test_construction_runner_root_name_uses_prefix(tmp_path):
    root = run_environment_construction(result_parent=tmp_path, timestamp_utc=_GD_TS)
    assert root.name == f"{ENVIRONMENT_CONSTRUCTION_RESULT_PREFIX}{_GD_TS}"

def test_construction_runner_alias_delegates(tmp_path):
    root = run_stage23a_environment_construction(
        result_parent=tmp_path, timestamp_utc=_GD_TS
    )
    assert root.is_dir()

def test_construction_runner_boundary_flags_spread():
    flags = _gD_construction_boundary_flags()
    assert flags["stage"] == "23-A"
    assert flags["development_only"] is True
    assert flags["claim_status"] == "not tested / not supported"
    assert flags["training_run"] == "environment_construction_smoke_only"
    assert flags["final_evaluation_run"] is False

def test_construction_runner_boundary_flags_in_every_json_payload(tmp_path):
    import json

    root = run_environment_construction(result_parent=tmp_path, timestamp_utc=_GD_TS)
    for name in STAGE23A_RESULT_FILES:
        if not name.endswith(".json"):
            continue
        payload = json.loads((root / name).read_text())
        assert payload["development_only"] is True
        assert payload["stage"] == "23-A"

def test_construction_runner_repo_internal_parent_rejected():
    # M-2 regression: a custom parent inside the repo is rejected.
    internal = (_gD_construction_active_root().resolve() / "results" / "gD_custom")
    with pytest.raises(ValueError, match="custom result_parent must be outside the repository"):
        run_environment_construction(result_parent=internal, timestamp_utc=_GD_TS)

def test_construction_runner_bad_result_parent_type_raises():
    with pytest.raises(TypeError, match="result_parent must be a str, Path, or None"):
        run_environment_construction(result_parent=123, timestamp_utc=_GD_TS)

def test_construction_runner_bad_timestamp_type_raises(tmp_path):
    with pytest.raises(TypeError, match="timestamp_utc must be a string or None"):
        run_environment_construction(result_parent=tmp_path, timestamp_utc=123)

def test_construction_runner_bad_timestamp_format_raises(tmp_path):
    with pytest.raises(ValueError):
        run_environment_construction(result_parent=tmp_path, timestamp_utc="not-a-timestamp")

def test_construction_runner_preexisting_temp_root_raises(tmp_path):
    # M-3 regression: a pre-existing temp root raises instead of being deleted.
    (tmp_path / f".stage23a_tmp_{_GD_TS}").mkdir()
    with pytest.raises(FileExistsError, match="temporary result root already exists"):
        run_environment_construction(result_parent=tmp_path, timestamp_utc=_GD_TS)

def test_construction_runner_preexisting_final_root_raises(tmp_path):
    run_environment_construction(result_parent=tmp_path, timestamp_utc=_GD_TS)
    with pytest.raises(FileExistsError, match="result root already exists"):
        run_environment_construction(result_parent=tmp_path, timestamp_utc=_GD_TS)

def test_construction_runner_readback_rejects_hash_tampering(tmp_path):
    root = run_environment_construction(result_parent=tmp_path, timestamp_utc=_GD_TS)
    mfile = root / "contract_smoke_metrics.json"
    mfile.write_text(
        mfile.read_text().replace(
            "STAGE23A_ENVIRONMENT_CONSTRUCTION_SMOKE_COMPLETE", "TAMPERED"
        )
    )
    with pytest.raises(ValueError, match="mismatch for contract_smoke_metrics.json"):
        _assert_strict_stage23a_result_artifacts(root)

def test_construction_runner_readback_accepts_untampered_root(tmp_path):
    root = run_environment_construction(result_parent=tmp_path, timestamp_utc=_GD_TS)
    _assert_strict_stage23a_result_artifacts(root)  # no raise

def test_construction_runner_hashed_files_exclude_manifest(tmp_path):
    import json

    root = run_environment_construction(result_parent=tmp_path, timestamp_utc=_GD_TS)
    hashes = json.loads((root / "artifact_hashes.json").read_text())["hashed_files"]
    assert "artifact_hashes.json" not in hashes
    assert set(hashes) == set(STAGE23A_RESULT_FILES[:-1])

def test_stage24_diagnostics_comparator_config_valid():
    cfg = Stage24ComparatorConfig("unit_empty", seed=2400)
    assert cfg.scenario_name == "unit_empty"

def test_stage24_diagnostics_comparator_config_empty_name_raises():
    with pytest.raises(ValueError, match="scenario_name must be a non-empty string"):
        Stage24ComparatorConfig("  ")

def test_stage24_diagnostics_comparator_config_seed_zero_raises():
    with pytest.raises(ValueError, match="seed must be positive"):
        Stage24ComparatorConfig("unit_empty", seed=0)

def test_stage24_diagnostics_comparator_config_negative_max_steps_raises():
    with pytest.raises(ValueError, match="max_steps must be positive"):
        Stage24ComparatorConfig("unit_empty", max_steps=-1)

def test_stage24_diagnostics_comparator_config_unknown_scenario_raises():
    with pytest.raises(ValueError, match="unknown Stage 23-A scenario"):
        Stage24ComparatorConfig("nope")

def test_stage24_diagnostics_variant_valid():
    v = Stage24AHazardLayoutVariant(
        name="x",
        group="training",
        hidden_hazard_cells=((2, 2),),
        public_risk_zone_cells=_PUBLIC_GATE_RISK_ZONE_CELLS,
        description="d",
    )
    assert v.name == "x"

def test_stage24_diagnostics_variant_bad_group_raises():
    with pytest.raises(ValueError, match="group must be training, readiness, or held_out"):
        Stage24AHazardLayoutVariant(
            name="x",
            group="bad",
            hidden_hazard_cells=((2, 2),),
            public_risk_zone_cells=_PUBLIC_GATE_RISK_ZONE_CELLS,
            description="d",
        )

def test_stage24_diagnostics_variant_exact_reveal_raises():
    with pytest.raises(ValueError, match="broader than exact hidden hazards"):
        Stage24AHazardLayoutVariant(
            name="x",
            group="training",
            hidden_hazard_cells=((2, 2),),
            public_risk_zone_cells=((2, 2),),
            description="d",
        )

def test_stage24_diagnostics_variant_uncovered_hidden_raises():
    with pytest.raises(ValueError, match="cover hidden hazard uncertainty cells"):
        Stage24AHazardLayoutVariant(
            name="x",
            group="training",
            hidden_hazard_cells=((7, 7),),
            public_risk_zone_cells=_PUBLIC_GATE_RISK_ZONE_CELLS,
            description="d",
        )

def test_stage24_diagnostics_variant_empty_cells_raises():
    with pytest.raises(ValueError, match="must be a non-empty tuple"):
        Stage24AHazardLayoutVariant(
            name="x",
            group="training",
            hidden_hazard_cells=(),
            public_risk_zone_cells=_PUBLIC_GATE_RISK_ZONE_CELLS,
            description="d",
        )

def test_stage24_diagnostics_variant_bad_coordinate_raises_type_error():
    with pytest.raises(TypeError, match="two-int coordinate tuples"):
        Stage24AHazardLayoutVariant(
            name="x",
            group="training",
            hidden_hazard_cells=((2, 2, 2),),
            public_risk_zone_cells=_PUBLIC_GATE_RISK_ZONE_CELLS,
            description="d",
        )

def test_stage24_diagnostics_public_gate_zone_shape():
    assert set(_PUBLIC_GATE_RISK_ZONE_CELLS) == {
        (r, c) for r in (1, 2, 3) for c in (1, 2, 3)
    }
    assert len(_PUBLIC_GATE_RISK_ZONE_CELLS) == 9

def test_stage24_diagnostics_reveal_helper_empty_hidden_true():
    assert _public_risk_zone_reveals_exact_hidden_hazard((), ((1, 1),)) is True

def test_stage24_diagnostics_reveal_helper_empty_public_true():
    assert _public_risk_zone_reveals_exact_hidden_hazard(((1, 1),), ()) is True

def test_stage24_diagnostics_reveal_helper_equal_sets_true():
    assert _public_risk_zone_reveals_exact_hidden_hazard(((2, 2),), ((2, 2),)) is True

def test_stage24_diagnostics_reveal_helper_public_not_bigger_true():
    assert _public_risk_zone_reveals_exact_hidden_hazard(((2, 2), (2, 3)), ((2, 2), (2, 4))) is True

def test_stage24_diagnostics_reveal_helper_broad_zone_false():
    assert (
        _public_risk_zone_reveals_exact_hidden_hazard(((2, 2),), _PUBLIC_GATE_RISK_ZONE_CELLS)
        is False
    )

def test_stage24_diagnostics_reveal_helper_multi_hidden_needs_extra_uncertainty():
    # Two hidden cells, only 3 public: 3 < 2 + 2 -> True (too tight).
    assert (
        _public_risk_zone_reveals_exact_hidden_hazard(
            ((1, 1), (1, 2)), ((1, 1), (1, 2), (1, 3))
        )
        is True
    )

def test_stage24_diagnostics_shortest_path_straight_line():
    # Hand-computed: (2,0) -> (2,4) with no obstacles is 4 east steps.
    path = _shortest_path(
        (2, 0), ((2, 4),), width=8, height=8, obstacles=(), avoid_cells=()
    )
    assert path == [(2, 1), (2, 2), (2, 3), (2, 4)]

def test_stage24_diagnostics_shortest_path_already_at_goal_empty():
    path = _shortest_path(
        (2, 4), ((2, 4),), width=8, height=8, obstacles=(), avoid_cells=()
    )
    assert path == []

def test_stage24_diagnostics_shortest_path_unreachable_empty():
    # Wall of obstacles blocking column 1 entirely between start and goal.
    obstacles = tuple((r, 1) for r in range(8))
    path = _shortest_path(
        (2, 0), ((2, 4),), width=8, height=8, obstacles=obstacles, avoid_cells=()
    )
    assert path == []

def test_stage24_diagnostics_shortest_path_avoids_cells_but_not_goals():
    # avoid_cells that equal a goal are NOT treated as blocked.
    path = _shortest_path(
        (2, 3), ((2, 4),), width=8, height=8, obstacles=(), avoid_cells=((2, 4),)
    )
    assert path == [(2, 4)]

def test_stage24_diagnostics_shortest_path_routes_around_avoid_cell():
    # Avoiding (2,2) forces a detour off the straight row.
    path = _shortest_path(
        (2, 0), ((2, 4),), width=8, height=8, obstacles=(), avoid_cells=((2, 2),)
    )
    assert path[-1] == (2, 4)
    assert (2, 2) not in path

def test_stage24_diagnostics_movement_action_directions():
    assert _movement_action((2, 0), (2, 0)) == 0  # stay
    assert _movement_action((2, 2), (1, 2)) == 1  # north
    assert _movement_action((2, 2), (3, 2)) == 2  # south
    assert _movement_action((2, 2), (2, 1)) == 3  # west
    assert _movement_action((2, 2), (2, 3)) == 4  # east

def test_stage24_diagnostics_movement_action_non_adjacent_raises():
    with pytest.raises(ValueError, match="non-adjacent movement requested"):
        _movement_action((0, 0), (2, 2))

def test_stage24_diagnostics_no_sense_deterministic():
    r1 = no_sense_shortest_path("risk_gate_hidden_hazard", seed=2400)
    r2 = no_sense_shortest_path("risk_gate_hidden_hazard", seed=2400)
    assert r1 == r2
    assert r1["policy_name"] == "no_sense_shortest_path"
    assert r1["sensing_rate"] == 0.0

def test_stage24_diagnostics_no_sense_crosses_hidden_hazard():
    # Hand-verified: the public shortest path crosses the hidden gate hazard.
    r = no_sense_shortest_path("risk_gate_hidden_hazard", seed=2400)
    assert r["team_success"] is True
    assert r["hazard_entry_count"] > 0

def test_stage24_diagnostics_always_sense_senses_every_step():
    r = always_sense_shortest_path("risk_gate_hidden_hazard", seed=2400)
    assert r["sensing_rate"] == 1.0

def test_stage24_diagnostics_random_policy_seed_deterministic():
    r1 = random_policy("risk_gate_hidden_hazard", seed=2400)
    r2 = random_policy("risk_gate_hidden_hazard", seed=2400)
    assert r1 == r2

def test_stage24_diagnostics_random_policy_seed_varies():
    r1 = random_policy("risk_gate_hidden_hazard", seed=2400)
    r2 = random_policy("risk_gate_hidden_hazard", seed=2401)
    # Different seeds should generally produce different action streams.
    assert r1 != r2

def test_stage24_diagnostics_oracle_flags_and_avoids_hazard():
    r = risk_aware_oracle_or_heuristic("risk_gate_hidden_hazard", seed=2400)
    assert r["diagnostic_only"] is True
    assert r["uses_hidden_hazards_for_decision"] is True
    assert r["hazard_entry_count"] == 0

def test_stage24_diagnostics_selective_sense_avoids_hazard_and_adapts():
    r = selective_sense_risk_aware("risk_gate_hidden_hazard", seed=2400)
    assert r["team_success"] is True
    assert r["hazard_entry_count"] == 0
    assert r["used_revealed_information_to_adapt_movement"] is True

def test_stage24_diagnostics_selective_sense_degenerates_without_public_zone():
    # unit_single_hazard has an empty public risk zone -> selective never senses.
    r = selective_sense_risk_aware("unit_single_hazard", seed=2400)
    assert r["sensing_rate"] == 0.0

def test_stage24_diagnostics_comparators_do_not_change_cwd():
    cwd0 = os.getcwd()
    no_sense_shortest_path("risk_gate_hidden_hazard", seed=2400)
    always_sense_shortest_path("risk_gate_hidden_hazard", seed=2400)
    selective_sense_risk_aware("risk_gate_hidden_hazard", seed=2400)
    assert os.getcwd() == cwd0

def test_stage24_diagnostics_run_scenario_diagnostics_keys():
    diag = run_stage23_scenario_diagnostics(seed=2400)
    assert diag["stage"] == "24"
    assert diag["final_evaluation_run"] is False
    assert diag["claim_evidence_created"] is False
    assert diag["scenario_count"] == 14
    assert set(diag["comparators"]) == set(stage23_scenario_catalog())

def test_stage24_diagnostics_run_scenario_diagnostics_deterministic():
    a = run_stage23_scenario_diagnostics(seed=2400)
    b = run_stage23_scenario_diagnostics(seed=2400)
    assert a == b

def test_stage24_diagnostics_run_scenario_diagnostics_marks_risk_gate_sufficient():
    diag = run_stage23_scenario_diagnostics(
        scenario_names=("risk_gate_hidden_hazard",), seed=2400
    )
    record = diag["scenarios"][0]
    assert record["scenario_name"] == "risk_gate_hidden_hazard"
    assert record["sufficient_for_active_sensing_claim_testing"] is True

def test_stage24_diagnostics_run_scenario_diagnostics_seed_zero_raises():
    with pytest.raises(ValueError, match="seed must be positive"):
        run_stage23_scenario_diagnostics(seed=0)

def test_stage24_diagnostics_run_scenario_diagnostics_empty_names_raises():
    with pytest.raises(ValueError, match="scenario_names must be non-empty"):
        run_stage23_scenario_diagnostics(scenario_names=(), seed=2400)

def test_stage24_diagnostics_run_scenario_diagnostics_non_string_name_raises():
    with pytest.raises(TypeError, match="scenario_names entries must be strings"):
        run_stage23_scenario_diagnostics(scenario_names=(1,), seed=2400)

def test_stage24_diagnostics_run_scenario_diagnostics_duplicate_names_raises():
    with pytest.raises(ValueError, match="scenario_names must be unique"):
        run_stage23_scenario_diagnostics(
            scenario_names=("unit_empty", "unit_empty"), seed=2400
        )

def test_stage24_diagnostics_run_scenario_diagnostics_unknown_name_raises():
    with pytest.raises(ValueError, match="unknown Stage 23-A scenario"):
        run_stage23_scenario_diagnostics(scenario_names=("nope",), seed=2400)

def test_stage24_diagnostics_variant_readiness_keys():
    va = run_stage24a_variant_readiness_diagnostics(seed=24400)
    assert va["stage"] == "24-A"
    assert va["variant_count"] == 3
    assert va["final_evaluation_run"] is False
    assert set(va["variants"]) == set(stage24a_hazard_layout_variants())

def test_stage24_diagnostics_variant_readiness_deterministic():
    a = run_stage24a_variant_readiness_diagnostics(seed=24400)
    b = run_stage24a_variant_readiness_diagnostics(seed=24400)
    assert a == b

def test_stage24_diagnostics_variant_readiness_seed_zero_raises():
    with pytest.raises(ValueError, match="seed must be positive"):
        run_stage24a_variant_readiness_diagnostics(seed=0)


# ===================== SECTION P5 (adversarial-audit regressions) =====================
# Regressions for the five defects confirmed by the Phase-5 full-surface adversarial
# audit (all fixed byte-identically on the production path). Local imports keep these
# self-contained regardless of the assembled module-top imports.


def test_losses_bootstrap_allowed_mask_rejects_sparse_layout():
    import torch as _p5_torch
    from raas_marl.mappo_lagrangian import losses as _p5_losses

    sparse_bool = _p5_torch.sparse_coo_tensor(
        _p5_torch.tensor([[0], [0]]), _p5_torch.tensor([True]), (1, 2)
    )
    with pytest.raises(ValueError, match="strided"):
        _p5_losses.bootstrap_allowed_mask(sparse_bool)


def test_losses_masked_mean_rejects_sparse_mask():
    import torch as _p5_torch
    from raas_marl.mappo_lagrangian import losses as _p5_losses

    sparse_bool = _p5_torch.sparse_coo_tensor(
        _p5_torch.tensor([[0], [0]]), _p5_torch.tensor([True]), (1, 2)
    )
    with pytest.raises(ValueError, match="strided"):
        _p5_losses.masked_mean(_p5_torch.zeros(1, 2), sparse_bool)


def test_tensor_adapter_actor_observation_rejects_config_mismatched_dtype():
    from raas_marl.environments.active_sensing import tensor_adapter as _p5_ta

    obs = {
        "actor_visible": {
            "position": [0, 0],
            "step_index": 0,
            "max_steps": 32,
            "grid_shape": [8, 8],
            "nearest_goal_delta": [7, 7],
            "local_obstacles": [],
            "revealed_local_hazards": [[1, 1]],
            "previous_sensed": False,
            "previous_invalid_move": False,
            "previous_blocked_move": False,
            "previous_entered_hazard": False,
        },
        "action_factors": {},
        "identity": {},
    }
    # float64 diverges from the float32 core config -> rejected at emission,
    # consistent with central_state_from_stage23 (finding tensor_adapter dtype).
    with pytest.raises(ValueError, match="dtype"):
        _p5_ta.actor_observation_from_stage23(obs, dtype="float64")


def test_tensor_adapter_obstacle_count_cap_derives_from_radius():
    from raas_marl.environments.active_sensing.grid_environment import (
        RiskAwareActiveSensingGridEnvironment,
        Stage23EnvironmentConfig,
    )
    from raas_marl.environments.active_sensing import tensor_adapter as _p5_ta

    obstacles = tuple(
        sorted([(2, 4), (6, 4), (4, 2), (4, 6), (3, 3), (3, 5), (5, 3)])
    )
    cfg = Stage23EnvironmentConfig(
        scenario_name="standard_branching_hazard",
        local_observation_radius=2,
        start_positions=((4, 4), (0, 1)),
        goal_cells=((7, 7),),
        obstacle_cells=obstacles,
        hidden_hazard_cells=(),
    )
    env = RiskAwareActiveSensingGridEnvironment(cfg)
    obs, infos = env.reset()
    contract = _p5_ta.stage23_transition_to_environment_step_contract(
        obs["agent_0"], infos["agent_0"], env
    )
    # DR-REPRESENTATION: the obstacle COUNT scalar is at feature index 9 (the last
    # LEGACY feature); indices 10..21 are now the radius-2 occupancy patch, so read [9]
    # (not [-1]) for the count.
    obstacle_feature = contract.actor_visible["actor_observation"].reshape(-1)[9].item()
    # radius 2 -> obstacle cap = 2*2*3 + 1 = 13; 7 obstacles -> 7/13, NOT clamped to 1.0.
    assert obstacle_feature < 0.99
    assert obstacle_feature == pytest.approx(7.0 / 13.0, abs=1e-6)


def test_stage23c_rollout_config_rejects_hostile_subclass_bounds():
    from raas_marl.mappo_lagrangian.stage23c_rollout import Stage23CRolloutConfig

    class _P5Sneaky(int):
        def __le__(self, other):  # noqa: D401
            return True

        def __ge__(self, other):
            return True

    with pytest.raises((ValueError, TypeError)):
        Stage23CRolloutConfig(max_episodes=_P5Sneaky(1))
    with pytest.raises((ValueError, TypeError)):
        Stage23CRolloutConfig(max_episodes=4, steps_per_episode=_P5Sneaky(999))


def test_stage23c_runner_rejects_seed_above_signed_64_bit(tmp_path):
    from raas_marl.mappo_lagrangian.stage23c_runner import run_stage23c_bounded_training

    # seed >= 2**64 previously reached torch.manual_seed and raised an undeclared
    # RuntimeError; it must now raise a declared ValueError at argument validation.
    with pytest.raises(ValueError):
        run_stage23c_bounded_training(
            result_parent=str(tmp_path),
            seed=2 ** 64,
            timestamp_utc="20260101T000000Z",
        )

def test_stage24_diagnostics_variant_readiness_no_writes(tmp_path):
    cwd0 = os.getcwd()
    run_stage24a_variant_readiness_diagnostics(seed=24400)
    assert os.getcwd() == cwd0

def test_stage24_diagnostics_hazard_layout_variants_groups():
    variants = stage24a_hazard_layout_variants()
    groups = {v.group for v in variants.values()}
    assert {"training", "readiness", "held_out"} <= groups
    for key, variant in variants.items():
        assert key == variant.name

def test_active_sensing_init_all_names_resolve():
    import raas_marl.environments.active_sensing as pkg

    for name in pkg.__all__:
        assert getattr(pkg, name) is not None

def test_active_sensing_init_lazy_tensor_adapter_exports():
    import raas_marl.environments.active_sensing as pkg

    assert pkg.default_stage23_core_config is default_stage23_core_config
    assert pkg.actor_observation_from_stage23 is actor_observation_from_stage23

def test_active_sensing_init_lazy_stage24_diagnostics_exports():
    import raas_marl.environments.active_sensing as pkg

    assert pkg.no_sense_shortest_path is no_sense_shortest_path
    assert pkg.stage24a_hazard_layout_variants is stage24a_hazard_layout_variants

def test_active_sensing_init_unknown_attr_raises():
    import raas_marl.environments.active_sensing as pkg

    with pytest.raises(AttributeError):
        pkg.does_not_exist

def test_active_sensing_init_dunder_attr_raises():
    import raas_marl.environments.active_sensing as pkg

    with pytest.raises(AttributeError):
        pkg.__nonexistent_dunder__

def test_active_sensing_init_all_is_sorted():
    import raas_marl.environments.active_sensing as pkg

    assert list(pkg.__all__) == sorted(pkg.__all__)

def test_active_sensing_init_eager_env_exports():
    import raas_marl.environments.active_sensing as pkg

    assert pkg.RiskAwareActiveSensingGridEnvironment is RiskAwareActiveSensingGridEnvironment
    assert pkg.Stage23EnvironmentConfig is Stage23EnvironmentConfig

def test_environments_init_imports_cleanly():
    import raas_marl.environments as env_pkg

    assert env_pkg.__name__ == "raas_marl.environments"

def test_environments_init_defines_no_public_api():
    import raas_marl.environments as env_pkg

    # The marker package deliberately re-exports nothing.
    assert not hasattr(env_pkg, "RiskAwareActiveSensingGridEnvironment")

def test_environments_init_does_not_pull_torch_eagerly():
    # Fresh import in a subprocess-free way: assert torch is not referenced by the
    # marker module itself.
    import raas_marl.environments as env_pkg

    assert not hasattr(env_pkg, "torch")

# ===================== SECTION E =====================

_GE_TS = "20260704T010203Z"

_GE_EPISODE_FIELDS = frozenset(
    {
        "episode_index",
        "round_index",
        "scenario_name",
        "seed",
        "step_count",
        "environment_max_steps",
        "terminal",
        "truncated",
        "team_success",
        "task_reward_sum",
        "hazard_cost_sum",
        "sensing_cost_sum",
        "total_sensing_actions",
        "total_movement_actions",
        "hazard_entry_count",
        "revealed_hazard_count",
        "sensing_rate",
        "sample_actions",
    }
)

_GE_CURVE_FIELDS = frozenset(
    {
        "round_index",
        "episode_indices",
        "task_reward_sum",
        "hazard_cost_mean",
        "sensing_cost_mean",
        "lagrange_multiplier",
        "parameter_delta_l1",
        "total_loss",
    }
)

def _ge_core():
    return default_stage23_core_config()

def _ge_model(seed: int | None = None):
    if seed is not None:
        torch.manual_seed(seed)
    return RecurrentMAPPOActorCritic(_ge_core())

def _ge_config(**kwargs):
    return Stage23CRolloutConfig(**kwargs)

def _ge_load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]

def _ge_run(tmp_path: Path, *, timestamp: str = _GE_TS, **kwargs) -> Path:
    kwargs.setdefault("episode_count", 2)
    kwargs.setdefault("steps_per_episode", 3)
    return run_stage23c_bounded_training(
        result_parent=tmp_path,
        timestamp_utc=timestamp,
        **kwargs,
    )

def test_stage23c_rollout_config_defaults_happy_path():
    config = _ge_config()
    assert config.max_episodes == 2
    assert config.steps_per_episode is None
    assert config.scenario_names == ("risk_gate_hidden_hazard",)
    assert config.sample_actions is False
    assert config.seed == 23
    assert config.dtype == "float32"
    assert config.device == "cpu"

def test_stage23c_rollout_config_max_episodes_lower_bound_rejected():
    with pytest.raises(ValueError, match="max_episodes must be between 2 and 8"):
        _ge_config(max_episodes=1)

def test_stage23c_rollout_config_max_episodes_upper_bound_rejected():
    with pytest.raises(ValueError, match="max_episodes must be between 2 and 8"):
        _ge_config(max_episodes=9)

def test_stage23c_rollout_config_max_episodes_zero_rejected():
    with pytest.raises(ValueError, match="max_episodes must be positive"):
        _ge_config(max_episodes=0)

def test_stage23c_rollout_config_max_episodes_bool_rejected():
    with pytest.raises(TypeError, match="max_episodes must be positive"):
        _ge_config(max_episodes=True)

@pytest.mark.parametrize("value", [2, 4, 8])
def test_stage23c_rollout_config_max_episodes_in_range_accepted(value):
    assert _ge_config(max_episodes=value).max_episodes == value

def test_stage23c_rollout_config_steps_zero_rejected():
    with pytest.raises(ValueError, match="steps_per_episode must be positive"):
        _ge_config(steps_per_episode=0)

def test_stage23c_rollout_config_steps_over_upper_bound_rejected():
    with pytest.raises(
        ValueError, match="steps_per_episode must be None or between 1 and 32"
    ):
        _ge_config(steps_per_episode=33)

def test_stage23c_rollout_config_steps_bool_rejected():
    with pytest.raises(TypeError, match="steps_per_episode must be positive"):
        _ge_config(steps_per_episode=True)

@pytest.mark.parametrize("value", [1, 16, 32])
def test_stage23c_rollout_config_steps_in_range_accepted(value):
    assert _ge_config(steps_per_episode=value).steps_per_episode == value

def test_stage23c_rollout_config_steps_none_accepted():
    assert _ge_config(steps_per_episode=None).steps_per_episode is None

def test_stage23c_rollout_config_scenario_names_not_tuple_rejected():
    with pytest.raises(ValueError, match="scenario_names must be a non-empty tuple"):
        _ge_config(scenario_names=["risk_gate_hidden_hazard"])

def test_stage23c_rollout_config_scenario_names_empty_rejected():
    with pytest.raises(ValueError, match="scenario_names must be a non-empty tuple"):
        _ge_config(scenario_names=())

def test_stage23c_rollout_config_scenario_name_empty_string_rejected():
    with pytest.raises(
        ValueError, match="scenario_names entries must be non-empty strings"
    ):
        _ge_config(scenario_names=("",))

def test_stage23c_rollout_config_scenario_name_non_string_rejected():
    with pytest.raises(
        ValueError, match="scenario_names entries must be non-empty strings"
    ):
        _ge_config(scenario_names=(5,))

def test_stage23c_rollout_config_scenario_name_unknown_rejected():
    with pytest.raises(ValueError, match="unknown Stage 23-A scenario: nope"):
        _ge_config(scenario_names=("nope",))

def test_stage23c_rollout_config_all_catalog_scenarios_accepted():
    names = available_scenarios()
    assert _ge_config(scenario_names=tuple(names)).scenario_names == tuple(names)

def test_stage23c_rollout_config_sample_actions_non_bool_rejected():
    with pytest.raises(TypeError, match="sample_actions must be a bool"):
        _ge_config(sample_actions=1)

def test_stage23c_rollout_config_seed_zero_rejected():
    with pytest.raises(
        ValueError,
        match="seed must be a positive integer; seed 0 is reserved for Stage 23-C",
    ):
        _ge_config(seed=0)

def test_stage23c_rollout_config_seed_negative_rejected():
    with pytest.raises(ValueError, match="seed must be a positive integer"):
        _ge_config(seed=-1)

def test_stage23c_rollout_config_seed_bool_rejected():
    with pytest.raises(TypeError):
        _ge_config(seed=True)

def test_stage23c_rollout_config_seed_overflow_rejected():
    with pytest.raises(
        ValueError, match="seed plus reseed offset must fit in a signed 64-bit integer"
    ):
        _ge_config(seed=2**63)

def test_stage23c_rollout_config_seed_at_bound_accepted():
    # offset = max_episodes*1000 + steps_per_episode = 2000 + 32 = 2032.
    max_seed = 2**63 - 1 - 2032
    assert _ge_config(max_episodes=2, steps_per_episode=32, seed=max_seed).seed == max_seed

def test_stage23c_rollout_config_seed_over_bound_rejected():
    max_seed = 2**63 - 1 - 2032
    with pytest.raises(
        ValueError, match="seed plus reseed offset must fit in a signed 64-bit integer"
    ):
        _ge_config(max_episodes=2, steps_per_episode=32, seed=max_seed + 1)

def test_stage23c_rollout_config_dtype_float64_rejected_as_non_default():
    # float64 passes the {float32,float64} membership gate but fails the
    # equal-to-default-Stage-23-A-dtype gate.
    with pytest.raises(
        ValueError,
        match="Stage 23-C rollout dtype must equal the default Stage 23-A core dtype",
    ):
        _ge_config(dtype="float64")

def test_stage23c_rollout_config_dtype_unknown_rejected():
    with pytest.raises(ValueError, match="dtype must be 'float32' or 'float64'"):
        _ge_config(dtype="float16")

def test_stage23c_rollout_config_device_non_cpu_rejected():
    with pytest.raises(
        ValueError, match="Stage 23-C rollout collection supports only CPU device 'cpu'"
    ):
        _ge_config(device="cuda")

def test_stage23c_rollout_config_device_empty_rejected():
    with pytest.raises(ValueError, match="device must be a non-empty string"):
        _ge_config(device="")

def test_stage23c_rollout_config_device_whitespace_rejected():
    with pytest.raises(
        ValueError, match="device must not contain leading or trailing whitespace"
    ):
        _ge_config(device=" cpu ")

def test_stage23c_rollout_config_is_frozen():
    config = _ge_config()
    with pytest.raises(Exception):
        config.seed = 99  # type: ignore[misc]

def test_stage23c_rollout_make_environment_happy_path():
    env = make_default_stage23c_environment(scenario_name="risk_gate_hidden_hazard")
    assert env.scenario.name == "risk_gate_hidden_hazard"

def test_stage23c_rollout_make_environment_unknown_scenario_rejected():
    with pytest.raises(ValueError, match="unknown Stage 23-A scenario: nope"):
        make_default_stage23c_environment(scenario_name="nope")

def test_stage23c_rollout_make_environment_empty_scenario_rejected():
    with pytest.raises(ValueError, match="scenario_name must be a non-empty string"):
        make_default_stage23c_environment(scenario_name="")

def test_stage23c_rollout_make_environment_non_string_scenario_rejected():
    with pytest.raises(ValueError, match="scenario_name must be a non-empty string"):
        make_default_stage23c_environment(scenario_name=5)  # type: ignore[arg-type]

def test_stage23c_rollout_make_environment_seed_zero_rejected():
    with pytest.raises(
        ValueError,
        match="seed must be a positive integer; seed 0 is reserved for Stage 23-C",
    ):
        make_default_stage23c_environment(
            scenario_name="risk_gate_hidden_hazard", seed=0
        )

def test_stage23c_rollout_make_environment_max_steps_zero_rejected():
    with pytest.raises(ValueError, match="max_steps must be positive"):
        make_default_stage23c_environment(
            scenario_name="risk_gate_hidden_hazard", max_steps=0
        )

def test_stage23c_rollout_make_environment_custom_max_steps_applied():
    env = make_default_stage23c_environment(
        scenario_name="risk_gate_hidden_hazard", max_steps=7
    )
    assert env.config.max_steps == 7

def test_stage23c_rollout_collect_model_type_rejected():
    with pytest.raises(TypeError, match="model must be a RecurrentMAPPOActorCritic"):
        collect_stage23c_development_rollout("not-a-model")  # type: ignore[arg-type]

def test_stage23c_rollout_collect_config_type_rejected():
    model = _ge_model(seed=1)
    with pytest.raises(
        TypeError, match="rollout_config must be a Stage23CRolloutConfig or None"
    ):
        collect_stage23c_development_rollout(model, "bad")  # type: ignore[arg-type]

def test_stage23c_rollout_collect_none_config_uses_defaults():
    model = _ge_model(seed=1)
    collected = collect_stage23c_development_rollout(model, None)
    assert isinstance(collected, Stage23CCollectedRollout)
    # default max_episodes == 2 -> 2 episode records.
    assert len(collected.episode_records) == 2

def test_stage23c_rollout_collect_returns_dataclass_shape():
    model = _ge_model(seed=1)
    collected = collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=2, steps_per_episode=3)
    )
    assert isinstance(collected, Stage23CCollectedRollout)
    assert isinstance(collected.batch, RolloutBatch)
    assert isinstance(collected.summary, dict)
    assert isinstance(collected.episode_records, tuple)
    assert isinstance(collected.environment_summary, dict)

def test_stage23c_rollout_collect_batch_passes_rolloutbatch_validate():
    core = _ge_core()
    model = RecurrentMAPPOActorCritic(core)
    collected = collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=3, steps_per_episode=4)
    )
    # RolloutBatch.validate returns (batch, time) and raises on any violation.
    assert collected.batch.validate(core) == (
        collected.batch.actor_observation.shape[0],
        collected.batch.actor_observation.shape[1],
    )

def test_stage23c_rollout_collect_batch_size_is_episodes_times_agents():
    model = _ge_model(seed=1)
    collected = collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=3, steps_per_episode=4)
    )
    # Two Stage 23-A agents per episode.
    assert collected.batch.actor_observation.shape[0] == 3 * 2

def test_stage23c_rollout_collect_episode_records_count_matches_max_episodes():
    model = _ge_model(seed=1)
    collected = collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=4, steps_per_episode=3)
    )
    assert len(collected.episode_records) == 4

def test_stage23c_rollout_collect_episode_record_has_exact_18_fields():
    model = _ge_model(seed=1)
    collected = collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=2, steps_per_episode=3)
    )
    for record in collected.episode_records:
        assert set(record.keys()) == _GE_EPISODE_FIELDS
        assert len(record) == 18

def test_stage23c_rollout_collect_scenario_round_robin():
    model = _ge_model(seed=1)
    collected = collect_stage23c_development_rollout(
        model,
        _ge_config(
            max_episodes=3,
            steps_per_episode=3,
            scenario_names=("risk_gate_hidden_hazard", "unit_empty"),
        ),
    )
    names = [record["scenario_name"] for record in collected.episode_records]
    # round-robin: index % len(names) -> [0,1,0].
    assert names == ["risk_gate_hidden_hazard", "unit_empty", "risk_gate_hidden_hazard"]

def test_stage23c_rollout_collect_reset_seeds_are_base_plus_episode_index():
    model = _ge_model(seed=1)
    collected = collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=4, steps_per_episode=3, seed=23)
    )
    assert [record["seed"] for record in collected.episode_records] == [23, 24, 25, 26]

def test_stage23c_rollout_collect_round_index_defaults_to_zero():
    model = _ge_model(seed=1)
    collected = collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=3, steps_per_episode=3)
    )
    assert all(record["round_index"] == 0 for record in collected.episode_records)

def test_stage23c_rollout_collect_episode_index_is_zero_based_sequence():
    model = _ge_model(seed=1)
    collected = collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=4, steps_per_episode=3)
    )
    assert [record["episode_index"] for record in collected.episode_records] == [
        0,
        1,
        2,
        3,
    ]

def test_stage23c_rollout_collect_batch_is_detached():
    model = _ge_model(seed=1)
    collected = collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=2, steps_per_episode=3)
    )
    batch = collected.batch
    for name in (
        "actor_observation",
        "revealed_information",
        "central_state",
        "old_sensing_log_probability",
        "old_movement_log_probability",
        "reward_value",
        "hazard_cost_value",
        "task_reward",
        "hazard_cost",
        "sensing_cost",
    ):
        tensor = getattr(batch, name)
        assert tensor.requires_grad is False, name

def test_stage23c_rollout_collect_valid_mask_all_true_for_equal_length_episodes():
    # All episodes hit steps_per_episode (truncation) => no padding, all-True mask.
    model = _ge_model(seed=1)
    collected = collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=3, steps_per_episode=4)
    )
    assert bool(collected.batch.valid_mask.all().item()) is True

def test_stage23c_rollout_collect_variable_length_padding_valid_mask_false_on_pads():
    # DR-REPRESENTATION behavioral re-validation: the actor input dim changed (14->26)
    # so the seeded sampled trajectories differ from the pre-enrichment tree; seed=123
    # now yields [6,6,6,6] (no short episode). seed=39 (unit_empty, sampling on,
    # max_episodes=4, steps=6) yields lengths [4,6,6,6] deterministically -- episode 0
    # (4 steps) occupies batch rows 0,1 (two agents), each padded to max_time=6 with
    # valid_mask False on the last TWO steps. The variable-length padding machinery
    # (_rollout_common) is UNCHANGED; only the seeded trajectory moved.
    torch.manual_seed(39)
    model = RecurrentMAPPOActorCritic(_ge_core())
    collected = collect_stage23c_development_rollout(
        model,
        _ge_config(
            max_episodes=4,
            steps_per_episode=6,
            scenario_names=("unit_empty",),
            sample_actions=True,
            seed=39,
        ),
    )
    lengths = [record["step_count"] for record in collected.episode_records]
    assert lengths == [4, 6, 6, 6]
    valid = collected.batch.valid_mask
    assert collected.summary["max_collected_episode_length"] == 6
    assert valid.shape == (8, 6)
    # There is genuine padding.
    assert bool((~valid).any().item()) is True
    # Padded rows are exactly the short episode's two agent rows (0 and 1).
    padded_rows = [i for i in range(valid.shape[0]) if bool((~valid[i]).any().item())]
    assert padded_rows == [0, 1]
    for row in padded_rows:
        # First 4 genuine steps valid, padded 5th/6th steps invalid.
        assert valid[row].tolist() == [True, True, True, True, False, False]
    # Full-length rows stay all-True.
    for row in (2, 3, 4, 5, 6, 7):
        assert bool(valid[row].all().item()) is True

def test_stage23c_rollout_collect_sensing_rate_formula():
    model = _ge_model(seed=1)
    collected = collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=2, steps_per_episode=4, seed=1)
    )
    for record in collected.episode_records:
        expected = record["total_sensing_actions"] / (record["step_count"] * 2)
        assert record["sensing_rate"] == pytest.approx(expected)

def test_stage23c_rollout_collect_restores_train_mode():
    model = _ge_model(seed=1)
    model.train()
    assert model.training is True
    collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=2, steps_per_episode=3)
    )
    assert model.training is True

def test_stage23c_rollout_collect_restores_eval_mode():
    model = _ge_model(seed=1)
    model.eval()
    assert model.training is False
    collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=2, steps_per_episode=3)
    )
    assert model.training is False

def test_stage23c_rollout_collect_summary_boundary_and_stage_fields():
    model = _ge_model(seed=1)
    collected = collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=2, steps_per_episode=3)
    )
    summary = collected.summary
    assert summary["stage"] == "23-C"
    assert summary["development_only"] is True
    assert summary["claim_status"] == "not tested / not supported"
    assert summary["training_run"] == "bounded_stage23c_development_only"
    assert summary["episode_count"] == 2
    assert summary["padding_strategy"] == "valid_mask_padding"
    assert summary["safe_variable_length_episode_padding"] is True
    assert summary["evaluation_run"] is False
    assert summary["final_evaluation_run"] is False

def test_stage23c_rollout_collect_environment_summary_fields():
    model = _ge_model(seed=1)
    collected = collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=2, steps_per_episode=3, seed=23)
    )
    env_summary = collected.environment_summary
    assert env_summary["stage"] == "23-C"
    assert env_summary["environment_stage"] == "23-A"
    assert env_summary["development_only"] is True
    assert env_summary["seed"] == 23
    assert env_summary["model_device"] == "cpu"
    assert env_summary["model_dtype"] == "float32"
    assert env_summary["agent_count"] == 2

def test_stage23c_rollout_collect_summary_bootstrap_convention_string():
    model = _ge_model(seed=1)
    collected = collect_stage23c_development_rollout(
        model, _ge_config(max_episodes=2, steps_per_episode=3)
    )
    assert (
        collected.summary["next_value_bootstrap_convention"]
        == "terminal_next_values_zeroed;truncation_next_values_preserved"
    )

def test_stage23c_rollout_per_step_sampling_seed_formula():
    # base + episode_index*1000 + step_index (rollout.py convention; M-1 fix).
    assert per_step_sampling_seed(23, 0, 0) == 23
    assert per_step_sampling_seed(23, 1, 5) == 1028
    assert per_step_sampling_seed(100, 2, 3) == 2103

def test_stage23c_runner_result_files_constant_is_nine():
    assert len(STAGE23C_RESULT_FILES) == 9
    assert STAGE23C_RESULT_FILES == (
        "run_config.json",
        "command_record.json",
        "environment_summary.json",
        "rollout_summary.json",
        "episode_log.jsonl",
        "update_summary.json",
        "training_curve.jsonl",
        "boundary_record.json",
        "artifact_hashes.json",
    )

def test_stage23c_runner_result_parent_and_prefix_constants():
    assert STAGE23C_RESULT_PARENT == Path("results") / "stage23c_bounded_training"
    assert STAGE23C_RESULT_PREFIX == "stage23c_bounded_training_"

def test_stage23c_runner_boundary_flags_has_17_keys():
    flags = _boundary_flags()
    assert len(flags) == 17

def test_stage23c_runner_boundary_flags_stage_and_labels():
    flags = _boundary_flags()
    assert flags["stage"] == "23-C"
    assert flags["development_only"] is True
    assert flags["claim_status"] == "not tested / not supported"
    assert flags["training_run"] == "bounded_stage23c_development_only"

def test_stage23c_runner_boundary_flags_all_false_keys_are_false():
    flags = _boundary_flags()
    assert len(_FALSE_BOUNDARY_KEYS) == 13
    for key in _FALSE_BOUNDARY_KEYS:
        assert flags[key] is False, key

def test_stage23c_runner_boundary_flags_true_keys_are_true():
    flags = _boundary_flags()
    for key in _TRUE_BOUNDARY_KEYS:
        assert flags[key] is True, key

def test_stage23c_runner_boundary_flags_exact_key_set():
    flags = _boundary_flags()
    expected = _FALSE_BOUNDARY_KEYS | _TRUE_BOUNDARY_KEYS | {
        "stage",
        "claim_status",
        "training_run",
    }
    assert set(flags.keys()) == expected

def test_stage23c_runner_run_writes_exactly_nine_files(tmp_path):
    root = _ge_run(tmp_path)
    names = sorted(child.name for child in root.iterdir())
    assert names == sorted(STAGE23C_RESULT_FILES)

def test_stage23c_runner_run_result_root_name(tmp_path):
    root = _ge_run(tmp_path)
    assert root.name == f"{STAGE23C_RESULT_PREFIX}{_GE_TS}"

def test_stage23c_runner_run_all_children_are_files(tmp_path):
    root = _ge_run(tmp_path)
    for child in root.iterdir():
        assert child.is_file()
        assert not child.is_symlink()

def test_stage23c_runner_run_strict_validation_passes(tmp_path):
    root = _ge_run(tmp_path)
    # Should not raise.
    _assert_strict_stage23c_result_artifacts(root)

def test_stage23c_runner_run_no_blocked_binary_or_checkpoint_artifacts(tmp_path):
    root = _ge_run(tmp_path)
    blocked_suffixes = {
        ".pt",
        ".pth",
        ".ckpt",
        ".onnx",
        ".pkl",
        ".pickle",
        ".joblib",
        ".safetensors",
        ".bin",
        ".h5",
        ".keras",
    }
    blocked_tokens = {"checkpoint", "optimizer", "final_eval"}
    for child in root.iterdir():
        assert child.suffix.lower() not in blocked_suffixes
        lower = child.name.lower()
        assert not any(token in lower for token in blocked_tokens)

def test_stage23c_runner_run_only_json_or_jsonl_extensions(tmp_path):
    root = _ge_run(tmp_path)
    for child in root.iterdir():
        assert child.suffix in {".json", ".jsonl"}

def test_stage23c_runner_artifact_hashes_covers_eight_files(tmp_path):
    root = _ge_run(tmp_path)
    payload = json.loads((root / "artifact_hashes.json").read_text(encoding="utf-8"))
    hashed = payload["hashed_files"]
    assert set(hashed) == {
        name for name in STAGE23C_RESULT_FILES if name != "artifact_hashes.json"
    }
    assert len(hashed) == 8

def test_stage23c_runner_artifact_hashes_carries_boundary_flags(tmp_path):
    root = _ge_run(tmp_path)
    payload = json.loads((root / "artifact_hashes.json").read_text(encoding="utf-8"))
    for key, value in _boundary_flags().items():
        assert payload[key] == value

def test_stage23c_runner_artifact_hashes_are_hex_sha256(tmp_path):
    root = _ge_run(tmp_path)
    payload = json.loads((root / "artifact_hashes.json").read_text(encoding="utf-8"))
    for value in payload["hashed_files"].values():
        assert isinstance(value, str)
        assert len(value) == 64
        int(value, 16)  # raises if not hex

def test_stage23c_runner_episode_log_record_count_single_round(tmp_path):
    root = _ge_run(tmp_path, episode_count=3, steps_per_episode=3, update_rounds=1)
    records = _ge_load_jsonl(root / "episode_log.jsonl")
    assert len(records) == 3  # episode_count * update_rounds.

def test_stage23c_runner_episode_log_record_count_multi_round(tmp_path):
    root = _ge_run(tmp_path, episode_count=2, steps_per_episode=3, update_rounds=3)
    records = _ge_load_jsonl(root / "episode_log.jsonl")
    assert len(records) == 6  # 2 * 3.

def test_stage23c_runner_episode_log_records_have_required_fields(tmp_path):
    root = _ge_run(tmp_path, episode_count=2, steps_per_episode=3, update_rounds=1)
    records = _ge_load_jsonl(root / "episode_log.jsonl")
    for record in records:
        assert _GE_EPISODE_FIELDS.issubset(record.keys())

def test_stage23c_runner_episode_log_records_carry_boundary_flags(tmp_path):
    root = _ge_run(tmp_path, episode_count=2, steps_per_episode=3, update_rounds=1)
    records = _ge_load_jsonl(root / "episode_log.jsonl")
    boundary = _boundary_flags()
    for record in records:
        for key, value in boundary.items():
            assert record[key] == value

def test_stage23c_runner_episode_log_round_and_episode_indices(tmp_path):
    root = _ge_run(tmp_path, episode_count=2, steps_per_episode=3, update_rounds=2)
    records = _ge_load_jsonl(root / "episode_log.jsonl")
    assert [r["round_index"] for r in records] == [0, 0, 1, 1]
    assert [r["episode_index"] for r in records] == [0, 1, 0, 1]

def test_stage23c_runner_episode_log_field_types(tmp_path):
    root = _ge_run(tmp_path, episode_count=2, steps_per_episode=3, update_rounds=1)
    record = _ge_load_jsonl(root / "episode_log.jsonl")[0]
    assert isinstance(record["episode_index"], int)
    assert isinstance(record["round_index"], int)
    assert isinstance(record["scenario_name"], str)
    assert isinstance(record["seed"], int)
    assert isinstance(record["step_count"], int)
    assert isinstance(record["terminal"], bool)
    assert isinstance(record["truncated"], bool)
    assert isinstance(record["team_success"], bool)
    assert isinstance(record["task_reward_sum"], float)
    assert isinstance(record["hazard_cost_sum"], float)
    assert isinstance(record["sensing_cost_sum"], float)
    assert isinstance(record["total_sensing_actions"], int)
    assert isinstance(record["total_movement_actions"], int)
    assert isinstance(record["sensing_rate"], float)
    assert isinstance(record["sample_actions"], bool)

def test_stage23c_runner_training_curve_record_count(tmp_path):
    root = _ge_run(tmp_path, episode_count=2, steps_per_episode=3, update_rounds=3)
    records = _ge_load_jsonl(root / "training_curve.jsonl")
    assert len(records) == 3  # one per update round.

def test_stage23c_runner_training_curve_single_round(tmp_path):
    root = _ge_run(tmp_path, episode_count=2, steps_per_episode=3, update_rounds=1)
    records = _ge_load_jsonl(root / "training_curve.jsonl")
    assert len(records) == 1

def test_stage23c_runner_training_curve_records_have_required_fields(tmp_path):
    root = _ge_run(tmp_path, episode_count=2, steps_per_episode=3, update_rounds=2)
    records = _ge_load_jsonl(root / "training_curve.jsonl")
    for record in records:
        assert _GE_CURVE_FIELDS.issubset(record.keys())

def test_stage23c_runner_training_curve_records_carry_boundary_flags(tmp_path):
    root = _ge_run(tmp_path, episode_count=2, steps_per_episode=3, update_rounds=2)
    records = _ge_load_jsonl(root / "training_curve.jsonl")
    boundary = _boundary_flags()
    for record in records:
        for key, value in boundary.items():
            assert record[key] == value

def test_stage23c_runner_training_curve_round_index_and_episode_indices(tmp_path):
    root = _ge_run(tmp_path, episode_count=3, steps_per_episode=3, update_rounds=2)
    records = _ge_load_jsonl(root / "training_curve.jsonl")
    assert [r["round_index"] for r in records] == [0, 1]
    for record in records:
        assert record["episode_indices"] == [0, 1, 2]

def test_stage23c_runner_training_curve_field_types(tmp_path):
    root = _ge_run(tmp_path, episode_count=2, steps_per_episode=3, update_rounds=1)
    record = _ge_load_jsonl(root / "training_curve.jsonl")[0]
    assert isinstance(record["round_index"], int)
    assert isinstance(record["episode_indices"], list)
    assert isinstance(record["task_reward_sum"], float)
    assert isinstance(record["hazard_cost_mean"], float)
    assert isinstance(record["sensing_cost_mean"], float)
    assert isinstance(record["lagrange_multiplier"], float)
    assert isinstance(record["parameter_delta_l1"], float)
    assert isinstance(record["total_loss"], float)

def test_stage23c_runner_training_curve_parameter_delta_positive(tmp_path):
    root = _ge_run(tmp_path, episode_count=2, steps_per_episode=3, update_rounds=2)
    records = _ge_load_jsonl(root / "training_curve.jsonl")
    for record in records:
        assert record["parameter_delta_l1"] > 0.0

def test_stage23c_runner_run_config_fields(tmp_path):
    root = _ge_run(tmp_path, seed=5, episode_count=2, steps_per_episode=3, update_rounds=2)
    payload = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    assert payload["seed"] == 5
    assert payload["episode_count"] == 2
    assert payload["steps_per_episode"] == 3
    assert payload["update_rounds"] == 2
    assert payload["timestamp_utc"] == _GE_TS
    assert payload["result_root_name"] == root.name
    assert payload["stage"] == "23-C"
    assert payload["development_only"] is True

def test_stage23c_runner_update_summary_stage_and_rounds(tmp_path):
    root = _ge_run(tmp_path, episode_count=2, steps_per_episode=3, update_rounds=2)
    payload = json.loads((root / "update_summary.json").read_text(encoding="utf-8"))
    assert payload["stage"] == "23-C"
    assert payload["training_run"] == "bounded_stage23c_development_only"
    assert payload["update_rounds"] == 2
    assert payload["update_helper"] == "stage22_ppo_lagrangian_update"
    assert payload["parameter_delta_l1"] > 0.0

def test_stage23c_runner_command_record_provenance_fields(tmp_path):
    root = _ge_run(tmp_path)
    payload = json.loads((root / "command_record.json").read_text(encoding="utf-8"))
    assert (
        payload["entry_point"]
        == "raas_marl.mappo_lagrangian.stage23c_runner.run_stage23c_bounded_training"
    )
    argv = payload["command_argv"]
    assert isinstance(argv, list) and len(argv) == 3
    assert argv[1] == "-c"
    assert argv[0] == payload["python_executable"]
    assert argv[2] == payload["replay_python_snippet"]
    invocation = payload["invocation_parameters"]
    for key in (
        "allow_pre_existing_stage23c_roots",
        "episode_count",
        "official_parent",
        "official_run",
        "resolved_result_parent",
        "result_parent_argument",
        "seed",
        "steps_per_episode",
        "timestamp_utc",
        "update_rounds",
    ):
        assert key in invocation

def test_stage23c_runner_boundary_record_fields(tmp_path):
    root = _ge_run(tmp_path)
    payload = json.loads((root / "boundary_record.json").read_text(encoding="utf-8"))
    assert payload["environment_stage"] == "23-A"
    assert payload["result_files_json_or_jsonl_only"] is True
    assert payload["artifact_set"] == list(STAGE23C_RESULT_FILES)
    assert payload["stage"] == "23-C"
    assert payload["development_only"] is True

def test_stage23c_runner_all_json_payloads_carry_boundary_flags(tmp_path):
    root = _ge_run(tmp_path)
    boundary = _boundary_flags()
    for name in STAGE23C_RESULT_FILES:
        if not name.endswith(".json"):
            continue
        payload = json.loads((root / name).read_text(encoding="utf-8"))
        for key, value in boundary.items():
            assert payload[key] == value, (name, key)

def test_stage23c_runner_run_seed_zero_rejected(tmp_path):
    with pytest.raises(
        ValueError, match="seed must be a positive integer; seed 0 is reserved"
    ):
        run_stage23c_bounded_training(
            result_parent=tmp_path, timestamp_utc=_GE_TS, seed=0
        )

def test_stage23c_runner_run_seed_negative_rejected(tmp_path):
    with pytest.raises(ValueError, match="seed must be a positive integer"):
        run_stage23c_bounded_training(
            result_parent=tmp_path, timestamp_utc=_GE_TS, seed=-1
        )

def test_stage23c_runner_run_seed_bool_rejected(tmp_path):
    with pytest.raises(TypeError, match="seed must be an int"):
        run_stage23c_bounded_training(
            result_parent=tmp_path, timestamp_utc=_GE_TS, seed=True
        )

def test_stage23c_runner_run_episode_count_lower_bound_rejected(tmp_path):
    with pytest.raises(ValueError, match="episode_count must be between 2 and 8"):
        run_stage23c_bounded_training(
            result_parent=tmp_path, timestamp_utc=_GE_TS, episode_count=1
        )

def test_stage23c_runner_run_episode_count_upper_bound_rejected(tmp_path):
    with pytest.raises(ValueError, match="episode_count must be between 2 and 8"):
        run_stage23c_bounded_training(
            result_parent=tmp_path, timestamp_utc=_GE_TS, episode_count=9
        )

def test_stage23c_runner_run_episode_count_zero_rejected(tmp_path):
    with pytest.raises(ValueError, match="episode_count must be positive"):
        run_stage23c_bounded_training(
            result_parent=tmp_path, timestamp_utc=_GE_TS, episode_count=0
        )

def test_stage23c_runner_run_steps_zero_rejected(tmp_path):
    with pytest.raises(ValueError, match="steps_per_episode must be positive"):
        run_stage23c_bounded_training(
            result_parent=tmp_path, timestamp_utc=_GE_TS, steps_per_episode=0
        )

def test_stage23c_runner_run_steps_over_upper_bound_rejected(tmp_path):
    with pytest.raises(
        ValueError, match="steps_per_episode must be None or between 1 and 32"
    ):
        run_stage23c_bounded_training(
            result_parent=tmp_path, timestamp_utc=_GE_TS, steps_per_episode=33
        )

def test_stage23c_runner_run_update_rounds_zero_rejected(tmp_path):
    with pytest.raises(ValueError, match="update_rounds must be positive"):
        run_stage23c_bounded_training(
            result_parent=tmp_path, timestamp_utc=_GE_TS, update_rounds=0
        )

def test_stage23c_runner_run_update_rounds_upper_bound_rejected(tmp_path):
    with pytest.raises(ValueError, match="update_rounds must be between 1 and 4"):
        run_stage23c_bounded_training(
            result_parent=tmp_path, timestamp_utc=_GE_TS, update_rounds=5
        )

def test_stage23c_runner_run_allow_flag_non_bool_rejected(tmp_path):
    with pytest.raises(
        TypeError, match="allow_pre_existing_stage23c_roots must be a bool"
    ):
        run_stage23c_bounded_training(
            result_parent=tmp_path,
            timestamp_utc=_GE_TS,
            allow_pre_existing_stage23c_roots=1,
        )

def test_stage23c_runner_run_result_parent_bad_type_rejected():
    with pytest.raises(TypeError, match="result_parent must be a str, Path, or None"):
        run_stage23c_bounded_training(result_parent=123, timestamp_utc=_GE_TS)

def test_stage23c_runner_run_timestamp_bad_type_rejected(tmp_path):
    with pytest.raises(TypeError, match="timestamp_utc must be a string or None"):
        run_stage23c_bounded_training(result_parent=tmp_path, timestamp_utc=5)

def test_stage23c_runner_run_timestamp_bad_format_rejected(tmp_path):
    with pytest.raises(ValueError, match="timestamp_utc must match"):
        run_stage23c_bounded_training(
            result_parent=tmp_path, timestamp_utc="2026-07-04"
        )

def test_stage23c_runner_run_repo_internal_parent_rejected():
    internal = _active_root() / "some_internal_stage23c_dir"
    with pytest.raises(
        ValueError, match="custom result_parent must be outside the repository"
    ):
        run_stage23c_bounded_training(
            result_parent=internal, timestamp_utc=_GE_TS
        )

def test_stage23c_runner_run_duplicate_timestamp_rejected(tmp_path):
    _ge_run(tmp_path, timestamp="20260704T111111Z")
    with pytest.raises(FileExistsError, match="result root already exists"):
        _ge_run(tmp_path, timestamp="20260704T111111Z")

def test_stage23c_runner_run_distinct_timestamps_succeed(tmp_path):
    root_a = _ge_run(tmp_path, timestamp="20260704T111111Z")
    root_b = _ge_run(tmp_path, timestamp="20260704T222222Z")
    assert root_a.exists()
    assert root_b.exists()
    assert root_a != root_b

def test_stage23c_runner_run_no_tmp_root_left_behind(tmp_path):
    _ge_run(tmp_path, timestamp="20260704T121212Z")
    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".stage23c_tmp_")]
    assert leftovers == []

def test_stage23c_runner_run_is_reproducible_for_same_arguments(tmp_path):
    root_a = run_stage23c_bounded_training(
        result_parent=tmp_path / "a",
        timestamp_utc=_GE_TS,
        seed=7,
        episode_count=2,
        steps_per_episode=3,
        update_rounds=1,
    )
    root_b = run_stage23c_bounded_training(
        result_parent=tmp_path / "b",
        timestamp_utc=_GE_TS,
        seed=7,
        episode_count=2,
        steps_per_episode=3,
        update_rounds=1,
    )
    hashes_a = json.loads((root_a / "artifact_hashes.json").read_text())["hashed_files"]
    hashes_b = json.loads((root_b / "artifact_hashes.json").read_text())["hashed_files"]
    # The metric/content files reproduce byte-for-byte under identical training
    # arguments. command_record.json and run_config.json legitimately differ:
    # they encode the (different) resolved result-parent path.
    metric_files = (
        "environment_summary.json",
        "rollout_summary.json",
        "episode_log.jsonl",
        "update_summary.json",
        "training_curve.jsonl",
        "boundary_record.json",
    )
    for name in metric_files:
        assert hashes_a[name] == hashes_b[name], name

def test_stage23c_runner_strict_validator_rejects_missing_file(tmp_path):
    root = _ge_run(tmp_path)
    (root / "boundary_record.json").unlink()
    with pytest.raises(ValueError, match="Stage 23-C result files mismatch"):
        _assert_strict_stage23c_result_artifacts(root)

def test_stage23c_runner_strict_validator_rejects_extra_file(tmp_path):
    root = _ge_run(tmp_path)
    (root / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Stage 23-C result files mismatch"):
        _assert_strict_stage23c_result_artifacts(root)

def test_stage23c_runner_strict_validator_rejects_tampered_hash(tmp_path):
    root = _ge_run(tmp_path)
    hashes_path = root / "artifact_hashes.json"
    payload = json.loads(hashes_path.read_text(encoding="utf-8"))
    # Corrupt one hash value.
    name = next(iter(payload["hashed_files"]))
    payload["hashed_files"][name] = "0" * 64
    hashes_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError, match="artifact_hashes.json mismatch"):
        _assert_strict_stage23c_result_artifacts(root)

def test_stage23c_runner_update_rounds_four_accepted(tmp_path):
    root = _ge_run(tmp_path, episode_count=2, steps_per_episode=3, update_rounds=4)
    curve = _ge_load_jsonl(root / "training_curve.jsonl")
    episodes = _ge_load_jsonl(root / "episode_log.jsonl")
    assert len(curve) == 4
    assert len(episodes) == 8  # 2 episodes * 4 rounds.

def test_stage23c_runner_run_steps_none_resolves_default_horizon(tmp_path):
    # steps_per_episode=None resolves to the scenario horizon and records a
    # concrete positive int in run_config.
    root = _ge_run(tmp_path, steps_per_episode=None)
    payload = json.loads((root / "run_config.json").read_text(encoding="utf-8"))
    resolved = payload["steps_per_episode"]
    assert isinstance(resolved, int)
    assert 1 <= resolved <= 32
    assert payload["steps_per_episode_argument"] is None


# ===================== SECTION F (Stage 25 harness: final_training) =====================

import re

from raas_marl import final_training as ft_pkg
from raas_marl.final_training import baseline_driver as s25_bd
from raas_marl.final_training import run_logging as s25_rl
from raas_marl.final_training import stage25_collection as s25_sc
from raas_marl.final_training import stage25_driver as s25_sd
from raas_marl.final_training import stage25_update as s25_su
from raas_marl.final_training.baseline_driver import (
    BaselineRunConfig,
    _assert_probe_variant_not_held_out,
    explained_variance_from_batch,
    greedy_model_policy_fn,
    run_baseline_training,
    run_readiness_probe,
)
from raas_marl.final_training.run_logging import (
    MANIFEST_FILENAME,
    MANIFEST_VERSION,
    append_jsonl_record,
    build_run_manifest_payload,
    experiments_root,
    harness_allowed_structural_keys,
    harness_boundary_flags,
    resolve_run_directory,
    scan_and_write_json,
    validate_run_id,
    write_run_manifest,
)
from raas_marl.environments.active_sensing.stage24_diagnostics import (
    _run_policy as _s25_run_comparator_policy,
)

# The exact 17-key Stage 25 harness boundary-flag mapping (13 False keys plus
# development_only / stage / training_run / claim_status).
_S25_BOUNDARY_FLAGS = {
    "ablation_run": False,
    "baseline_comparison_run": False,
    "checkpoint_created": False,
    "claim_evidence_created": False,
    "claim_status": "not tested / not supported",
    "cross_platform_bitwise_determinism_claimed": False,
    "development_only": True,
    "evaluation_run": False,
    "final_evaluation_run": False,
    "optimizer_state_saved": False,
    "paper_facing_results_created": False,
    "replay_buffer_artifact_created": False,
    "rollout_buffer_artifact_created": False,
    "serialized_model_artifact_created": False,
    "stage": "25-harness",
    "statistical_test_run": False,
    "training_run": "stage25_harness_baseline_development_only",
}

_S25_CURVE_KEYS = set(_S25_BOUNDARY_FLAGS) | {
    "round_index",
    "round_seed",
    "wall_time_s",
    "success_count",
    "success_rate",
    "mean_step_count",
    "task_reward_sum_mean",
    "hazard_cost_sum_mean",
    "sensing_cost_sum_mean",
    "sensing_rate_mean",
    "total_sensing_actions",
    "total_movement_actions",
    "hazard_entry_count",
    "revealed_hazard_count",
    "lagrange_multiplier_before",
    "lagrange_multiplier_after",
    "observed_hazard_cost",
    "policy_loss",
    "reward_value_loss",
    "hazard_cost_value_loss",
    "sensing_entropy",
    "movement_entropy",
    "total_loss",
    "parameter_delta_l1",
    "raw_reward_advantage_mean",
    "raw_hazard_cost_advantage_mean",
    "reward_explained_variance",
    "hazard_cost_explained_variance",
    "reward_target_variance",
    "hazard_cost_target_variance",
    "reward_target_variance_degenerate",
    "hazard_cost_target_variance_degenerate",
    "entropy_collapse_flag",
}

_S25_EPISODE_KEYS = set(_GE_EPISODE_FIELDS) | set(_S25_BOUNDARY_FLAGS)

_S25_PROBE_RECORD_KEYS = set(_S25_BOUNDARY_FLAGS) | {
    "round_index",
    "probe_seed",
    "scenario",
    "variants",
    "all_training_readiness_surfaces_met",
}

_S25_VERDICT_KEYS = {
    "learned_success",
    "learned_hazard_entries",
    "learned_sensing_cost",
    "learned_sensing_rate",
    "learned_adapts_movement",
    "beats_no_sense_on_hazard",
    "cheaper_than_always_sense",
    "preserved_success",
    "readiness_pattern_met",
}

_S25_MANIFEST_COMPLETE_KEYS = set(_S25_BOUNDARY_FLAGS) | {
    "run_id",
    "tier",
    "seeds",
    "implementing_dr_ids",
    "config",
    "git_head",
    "launch_timestamp_utc",
    "pid",
    "status",
    "manifest_version",
    "runtime_record",
    "last_completed_round",
    "finish_timestamp_utc",
}

_S25_TS = "20260704T101112Z"


def _s25_collapse(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _s25_manifest_kwargs(**overrides):
    kwargs = dict(
        run_id="manifest_run",
        tier="T0",
        seeds=(101, 202),
        implementing_dr_ids=("dr_0001",),
        config_payload={"alpha": 1},
        git_head=None,
        launch_timestamp_utc=_S25_TS,
        pid=1234,
        status="running",
    )
    kwargs.update(overrides)
    return kwargs


def _s25_batch_fields(batch_size=2, time_steps=3):
    return dict(
        actor_observation=torch.zeros(batch_size, time_steps, STAGE23_ACTOR_OBSERVATION_DIM),
        revealed_information=torch.zeros(
            batch_size, time_steps, STAGE23_REVEALED_INFORMATION_DIM
        ),
        central_state=torch.zeros(batch_size, time_steps, STAGE23_CENTRAL_STATE_DIM),
        sensing_action=torch.zeros(batch_size, time_steps, dtype=torch.long),
        movement_action=torch.zeros(batch_size, time_steps, dtype=torch.long),
        old_sensing_log_probability=torch.full((batch_size, time_steps), -0.5),
        old_movement_log_probability=torch.full((batch_size, time_steps), -0.5),
        task_reward=torch.zeros(batch_size, time_steps),
        hazard_cost=torch.zeros(batch_size, time_steps),
        sensing_cost=torch.zeros(batch_size, time_steps),
        reward_value=torch.zeros(batch_size, time_steps),
        hazard_cost_value=torch.zeros(batch_size, time_steps),
        next_reward_value=torch.zeros(batch_size, time_steps),
        next_hazard_cost_value=torch.zeros(batch_size, time_steps),
        terminal=torch.zeros(batch_size, time_steps, dtype=torch.bool),
        truncation=torch.zeros(batch_size, time_steps, dtype=torch.bool),
        valid_mask=torch.ones(batch_size, time_steps, dtype=torch.bool),
    )


def _s25_zeros_batch(**overrides):
    fields = _s25_batch_fields()
    fields.update(overrides)
    return RolloutBatch(**fields)


def _s25_regression_batch():
    """Hand-designed 2x3 batch exercising terminal, truncation, and padding."""

    return _s25_zeros_batch(
        task_reward=torch.tensor([[3.0, -2.0, 2.5], [-1.5, 2.0, 0.0]]),
        sensing_cost=torch.tensor([[1.0, 0.0, 2.0], [0.0, 1.0, 0.0]]),
        hazard_cost=torch.tensor([[2.0, 0.0, 1.0], [0.5, 3.0, 0.0]]),
        reward_value=torch.tensor([[1.0, -1.5, 2.0], [-2.0, 1.0, 0.0]]),
        next_reward_value=torch.tensor([[-0.5, 2.0, 1.0], [1.5, -1.0, 0.0]]),
        hazard_cost_value=torch.tensor([[1.5, -1.0, 0.5], [2.0, -0.5, 0.0]]),
        next_hazard_cost_value=torch.tensor([[-1.0, 1.5, 2.0], [0.5, 1.0, 0.0]]),
        terminal=torch.tensor([[False, False, True], [False, False, False]]),
        truncation=torch.tensor([[False, False, False], [False, True, False]]),
        valid_mask=torch.tensor([[True, True, True], [True, True, False]]),
    )


def _s25_reference_gae(signals, values, next_values, terminal, truncation, valid, gamma, lam):
    """From-scratch GAE recursion written out for the regression test.

    Implements delta_t = r_t + gamma * bootstrap_t * V(s_{t+1}) - V(s_t) and
    A_t = delta_t + gamma * lam * carry_t * A_{t+1} with the project's masked
    boundary semantics (terminal blocks the bootstrap; terminal-or-truncation
    stops recursive carry; invalid steps are zeroed and do not carry). It does
    NOT call losses.compute_gae; float32 tensors are used so the comparison
    against the driver's float32 arithmetic is meaningful at 1e-6.
    """

    bootstrap = (~terminal).to(signals.dtype)
    boundary = terminal | truncation
    carry = torch.zeros_like(valid)
    if valid.shape[1] > 1:
        carry[:, :-1] = valid[:, :-1] & valid[:, 1:] & ~boundary[:, :-1]
    carry_f = carry.to(signals.dtype)
    advantages = torch.zeros_like(signals)
    next_advantage = torch.zeros(signals.shape[0], dtype=signals.dtype)
    for step in range(signals.shape[1] - 1, -1, -1):
        delta = (
            signals[:, step]
            + gamma * bootstrap[:, step] * next_values[:, step]
            - values[:, step]
        )
        step_advantage = delta + gamma * lam * carry_f[:, step] * next_advantage
        step_advantage = torch.where(
            valid[:, step], step_advantage, torch.zeros_like(step_advantage)
        )
        advantages[:, step] = step_advantage
        next_advantage = step_advantage
    return advantages


def _s25_assert_no_heldout_reference(node):
    """Recursively assert the held-out variant name appears nowhere."""

    if isinstance(node, dict):
        for key, value in node.items():
            assert "risk_gate_heldout_near_gate" not in str(key)
            _s25_assert_no_heldout_reference(value)
    elif isinstance(node, (list, tuple)):
        for value in node:
            _s25_assert_no_heldout_reference(value)
    elif isinstance(node, str):
        assert "risk_gate_heldout_near_gate" not in node


@pytest.fixture(scope="module")
def s25_model():
    torch.manual_seed(20260704)
    return RecurrentMAPPOActorCritic(default_stage23_core_config())


@pytest.fixture(scope="module")
def s25_completed_run(tmp_path_factory):
    parent = tmp_path_factory.mktemp("s25_baseline_runs")
    config = BaselineRunConfig(
        run_id="seed151_smoke",
        seed=151,
        update_rounds=2,
        episodes_per_round=2,
        steps_per_episode=6,
        sample_actions=True,
        probe_every=1,
        probe_seed=311,
        tier="T0",
        result_parent=parent,
        heartbeat_every=25,
    )
    run_dir = run_baseline_training(config)
    return run_dir, parent, config


# ---------- 1. Package surface ----------

# Extended in place by the Stage 25 implementing commit (DR-D4 cluster): the
# stage25_update / stage25_collection / stage25_driver exports joined the
# pinned package surface.
_S25_EXPECTED_EXPORTS = (
    "BaselineRunConfig",
    "PIController",
    "PIControllerStep",
    "Stage25CollectedRollout",
    "Stage25CollectionConfig",
    "Stage25RunConfig",
    "Stage25UpdateConfig",
    "Stage25UpdateResult",
    "append_jsonl_record",
    "build_run_manifest_payload",
    "collect_stage25_training_rollout",
    "episodic_team_hazard_cost",
    "experiments_root",
    "explained_variance_from_batch",
    "greedy_model_policy_fn",
    "harness_boundary_flags",
    "run_baseline_training",
    "run_readiness_probe",
    "run_stage25_training",
    "stage25_boundary_flags",
    "stage25_ppo_lagrangian_update",
    "validate_run_id",
)

def test_s25_package_all_is_sorted_and_complete():
    assert ft_pkg.__all__ == sorted(ft_pkg.__all__)
    assert tuple(ft_pkg.__all__) == _S25_EXPECTED_EXPORTS

def test_s25_package_every_lazy_export_resolves_to_source_attribute():
    expected_home = {
        "BaselineRunConfig": s25_bd,
        "PIController": s25_su,
        "PIControllerStep": s25_su,
        "Stage25CollectedRollout": s25_sc,
        "Stage25CollectionConfig": s25_sc,
        "Stage25RunConfig": s25_sd,
        "Stage25UpdateConfig": s25_su,
        "Stage25UpdateResult": s25_su,
        "append_jsonl_record": s25_rl,
        "build_run_manifest_payload": s25_rl,
        "collect_stage25_training_rollout": s25_sc,
        "episodic_team_hazard_cost": s25_su,
        "experiments_root": s25_rl,
        "explained_variance_from_batch": s25_bd,
        "greedy_model_policy_fn": s25_bd,
        "harness_boundary_flags": s25_rl,
        "run_baseline_training": s25_bd,
        "run_readiness_probe": s25_bd,
        "run_stage25_training": s25_sd,
        "stage25_boundary_flags": s25_sd,
        "stage25_ppo_lagrangian_update": s25_su,
        "validate_run_id": s25_rl,
    }
    assert set(expected_home) == set(ft_pkg.__all__)
    for name, module in expected_home.items():
        assert getattr(ft_pkg, name) is getattr(module, name)

def test_s25_package_unknown_attribute_raises_attribute_error():
    with pytest.raises(
        AttributeError,
        match=re.escape(
            "module 'raas_marl.final_training' has no attribute 'not_an_export'"
        ),
    ):
        ft_pkg.not_an_export

def test_s25_package_dir_is_sorted_and_contains_all_exports():
    listing = dir(ft_pkg)
    assert listing == sorted(listing)
    assert set(ft_pkg.__all__).issubset(set(listing))


# ---------- 2. run_logging: boundary flags, run ids, appends, manifest ----------

def test_s25_harness_boundary_flags_exact_17_key_content():
    flags = harness_boundary_flags()
    assert flags == _S25_BOUNDARY_FLAGS
    assert len(flags) == 17
    false_keys = [key for key, value in flags.items() if value is False]
    assert len(false_keys) == 13
    assert flags["development_only"] is True
    assert flags["stage"] == "25-harness"
    assert flags["training_run"] == "stage25_harness_baseline_development_only"
    assert flags["claim_status"] == "not tested / not supported"
    for key in (
        "ablation_run",
        "baseline_comparison_run",
        "checkpoint_created",
        "claim_evidence_created",
        "cross_platform_bitwise_determinism_claimed",
        "evaluation_run",
        "final_evaluation_run",
        "optimizer_state_saved",
        "paper_facing_results_created",
        "replay_buffer_artifact_created",
        "rollout_buffer_artifact_created",
        "serialized_model_artifact_created",
        "statistical_test_run",
    ):
        assert flags[key] is False

def test_s25_harness_boundary_flags_returns_fresh_mapping_each_call():
    first = harness_boundary_flags()
    first["stage"] = "tampered"
    assert harness_boundary_flags()["stage"] == "25-harness"

def test_s25_harness_allowed_structural_keys_contents():
    allowed = harness_allowed_structural_keys()
    assert isinstance(allowed, frozenset)
    assert allowed == frozenset(_S25_BOUNDARY_FLAGS) | {
        "runtime_metadata_used_as_claim_evidence"
    }
    assert len(allowed) == 18

@pytest.mark.parametrize("value", ["seed101_baseline", "a", "a" * 64])
def test_s25_validate_run_id_happy_values_returned_unchanged(value):
    assert validate_run_id(value) == value

def test_s25_validate_run_id_non_string_raises_type_error():
    with pytest.raises(TypeError, match="run_id must be a string"):
        validate_run_id(123)

@pytest.mark.parametrize(
    "value",
    [
        "Seed101",
        "-leading_hyphen",
        "a" * 65,
        "",
        "ab\x00cd",
        "..",
        "a b",
    ],
)
def test_s25_validate_run_id_malformed_values_rejected(value):
    with pytest.raises(ValueError, match=re.escape("run_id must match")):
        validate_run_id(value)

@pytest.mark.parametrize(
    "value",
    ["checkpoint", "check_point", "opti-mizer", "finaleval1", "final-eval-1"],
)
def test_s25_validate_run_id_blocked_tokens_rejected_incl_separator_evasion(value):
    with pytest.raises(ValueError) as excinfo:
        validate_run_id(value)
    assert str(excinfo.value) == "run_id must not contain blocked artifact name tokens"

def test_s25_experiments_root_is_repo_results_experiments():
    root = experiments_root()
    assert root.name == "experiments"
    assert root.parent.name == "results"
    # The anchor two levels up must be the real repository root.
    assert (root.parents[1] / "src" / "raas_marl" / "final_training").is_dir()

def test_s25_resolve_run_directory_custom_tmp_parent_accepted(tmp_path):
    run_dir = resolve_run_directory("run_a", tmp_path)
    assert run_dir == tmp_path.resolve() / "run_a"
    assert not run_dir.exists()

def test_s25_resolve_run_directory_none_uses_official_experiments_root():
    run_dir = resolve_run_directory("s25_probe_never_created_run", None)
    assert run_dir == experiments_root().resolve() / "s25_probe_never_created_run"

def test_s25_resolve_run_directory_parent_under_official_root_accepted():
    run_dir = resolve_run_directory(
        "s25_probe_never_created_run", experiments_root() / "subgroup"
    )
    assert run_dir == (
        experiments_root().resolve() / "subgroup" / "s25_probe_never_created_run"
    )

def test_s25_resolve_run_directory_repo_internal_parent_exact_message():
    repo_src = gov.active_root(s25_rl.__file__, 3) / "src"
    with pytest.raises(ValueError) as excinfo:
        resolve_run_directory("run_a", repo_src)
    assert str(excinfo.value) == (
        "custom result_parent inside the repository must be under "
        "results/experiments"
    )

def test_s25_resolve_run_directory_blocked_token_parent_segment_rejected(tmp_path):
    with pytest.raises(
        ValueError,
        match="result_parent must not contain blocked artifact name tokens",
    ):
        resolve_run_directory("run_a", tmp_path / "opti-mizer")

def test_s25_resolve_run_directory_existing_run_dir_raises_file_exists(tmp_path):
    (tmp_path / "dup_run").mkdir()
    with pytest.raises(FileExistsError, match="run directory already exists"):
        resolve_run_directory("dup_run", tmp_path)

def test_s25_resolve_run_directory_validates_run_id_first(tmp_path):
    with pytest.raises(ValueError, match=re.escape("run_id must match")):
        resolve_run_directory("Bad", tmp_path)

def test_s25_append_jsonl_record_multi_append_strict_round_trip(tmp_path):
    path = tmp_path / "log.jsonl"
    records = [{"alpha": 1, "beta": [1, 2]}, {"alpha": 2.5, "beta": []}]
    for record in records:
        append_jsonl_record(path, record)
    assert gov.load_strict_jsonl_objects(path) == records

def test_s25_append_jsonl_record_non_path_raises_type_error(tmp_path):
    with pytest.raises(TypeError, match="path must be a pathlib.Path"):
        append_jsonl_record(str(tmp_path / "log.jsonl"), {"alpha": 1})

def test_s25_append_jsonl_record_non_mapping_raises_type_error(tmp_path):
    with pytest.raises(TypeError, match="record must be a mapping"):
        append_jsonl_record(tmp_path / "log.jsonl", [("alpha", 1)])

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_s25_append_jsonl_record_non_finite_float_fails_before_file_creation(
    tmp_path, bad
):
    path = tmp_path / "log.jsonl"
    with pytest.raises(ValueError, match="must contain only finite JSON numbers"):
        append_jsonl_record(path, {"x": bad})
    assert not path.exists()

def test_s25_append_jsonl_record_unsafe_value_leaves_existing_file_byte_unchanged(
    tmp_path,
):
    path = tmp_path / "log.jsonl"
    append_jsonl_record(path, {"ok": 1})
    before = path.read_bytes()
    with pytest.raises(ValueError, match="contains unsafe evidence-boundary wording"):
        append_jsonl_record(path, {"note": "final evaluation"})
    assert path.read_bytes() == before

def test_s25_scan_and_write_json_non_path_raises_type_error(tmp_path):
    with pytest.raises(TypeError, match="path must be a pathlib.Path"):
        scan_and_write_json(str(tmp_path / "x.json"), {"a": 1}, context="x.json")

def test_s25_scan_and_write_json_non_mapping_payload_raises_type_error(tmp_path):
    with pytest.raises(TypeError, match="payload must be a mapping"):
        scan_and_write_json(tmp_path / "x.json", [1, 2], context="x.json")

def test_s25_scan_and_write_json_empty_context_rejected(tmp_path):
    with pytest.raises(ValueError, match="context must be a non-empty string"):
        scan_and_write_json(tmp_path / "x.json", {"a": 1}, context="")

def test_s25_scan_and_write_json_unsafe_payload_writes_no_file(tmp_path):
    path = tmp_path / "x.json"
    with pytest.raises(ValueError, match="contains unsafe evidence-boundary wording"):
        scan_and_write_json(path, {"note": "claim evidence"}, context="x.json")
    assert not path.exists()

def test_s25_build_run_manifest_payload_happy_path_contents():
    payload = build_run_manifest_payload(**_s25_manifest_kwargs())
    for key, value in _S25_BOUNDARY_FLAGS.items():
        assert payload[key] == value
    assert payload["run_id"] == "manifest_run"
    assert payload["tier"] == "T0"
    assert payload["seeds"] == [101, 202]
    assert payload["implementing_dr_ids"] == ["dr_0001"]
    assert payload["config"] == {"alpha": 1}
    assert payload["git_head"] is None
    assert payload["launch_timestamp_utc"] == _S25_TS
    assert payload["pid"] == 1234
    assert payload["status"] == "running"
    assert payload["manifest_version"] == MANIFEST_VERSION == 1

def test_s25_build_run_manifest_payload_tier_t3_exact_reserved_message():
    with pytest.raises(ValueError) as excinfo:
        build_run_manifest_payload(**_s25_manifest_kwargs(tier="T3"))
    assert str(excinfo.value) == (
        "tier T3 is reserved for the Stage 26 protocol executor"
    )

def test_s25_build_run_manifest_payload_unknown_tier_rejected():
    with pytest.raises(ValueError, match="tier must be one of T0, T1, T2"):
        build_run_manifest_payload(**_s25_manifest_kwargs(tier="T5"))

def test_s25_build_run_manifest_payload_non_string_tier_rejected():
    with pytest.raises(TypeError, match="tier must be a string"):
        build_run_manifest_payload(**_s25_manifest_kwargs(tier=1))

def test_s25_build_run_manifest_payload_bad_status_rejected():
    with pytest.raises(
        ValueError, match="status must be one of running, complete, killed, crashed"
    ):
        build_run_manifest_payload(**_s25_manifest_kwargs(status="paused"))

def test_s25_build_run_manifest_payload_seeds_not_tuple_rejected():
    with pytest.raises(TypeError, match="seeds must be a tuple"):
        build_run_manifest_payload(**_s25_manifest_kwargs(seeds=[101]))

def test_s25_build_run_manifest_payload_empty_seeds_rejected():
    with pytest.raises(ValueError, match="seeds must be a non-empty tuple"):
        build_run_manifest_payload(**_s25_manifest_kwargs(seeds=()))

def test_s25_build_run_manifest_payload_zero_seed_rejected():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "seeds[0] must be a positive integer; seed 0 is intentionally "
            "reserved/rejected"
        ),
    ):
        build_run_manifest_payload(**_s25_manifest_kwargs(seeds=(0,)))

def test_s25_build_run_manifest_payload_bool_seed_rejected():
    with pytest.raises(TypeError, match=re.escape("seeds[0] must be an int")):
        build_run_manifest_payload(**_s25_manifest_kwargs(seeds=(True,)))

def test_s25_build_run_manifest_payload_dr_ids_not_tuple_rejected():
    with pytest.raises(TypeError, match="implementing_dr_ids must be a tuple"):
        build_run_manifest_payload(**_s25_manifest_kwargs(implementing_dr_ids=["x"]))

def test_s25_build_run_manifest_payload_dr_id_non_string_rejected():
    with pytest.raises(
        TypeError, match=re.escape("implementing_dr_ids[0] must be a string")
    ):
        build_run_manifest_payload(**_s25_manifest_kwargs(implementing_dr_ids=(1,)))

def test_s25_build_run_manifest_payload_empty_dr_id_rejected():
    with pytest.raises(
        ValueError, match=re.escape("implementing_dr_ids[0] must be non-empty")
    ):
        build_run_manifest_payload(**_s25_manifest_kwargs(implementing_dr_ids=("",)))

def test_s25_build_run_manifest_payload_non_mapping_config_rejected():
    with pytest.raises(TypeError, match="config_payload must be a mapping"):
        build_run_manifest_payload(**_s25_manifest_kwargs(config_payload=[1]))

def test_s25_build_run_manifest_payload_non_string_git_head_rejected():
    with pytest.raises(TypeError, match="git_head must be a string or None"):
        build_run_manifest_payload(**_s25_manifest_kwargs(git_head=b"abc"))

def test_s25_build_run_manifest_payload_zero_pid_rejected():
    with pytest.raises(ValueError, match="pid must be a positive integer"):
        build_run_manifest_payload(**_s25_manifest_kwargs(pid=0))

def test_s25_build_run_manifest_payload_calendar_invalid_timestamp_rejected():
    with pytest.raises(
        ValueError, match="timestamp_utc must be a calendar-valid UTC timestamp"
    ):
        build_run_manifest_payload(
            **_s25_manifest_kwargs(launch_timestamp_utc="20260231T101112Z")
        )

def test_s25_build_run_manifest_payload_extra_key_collision_rejected():
    with pytest.raises(
        ValueError, match="extra key must not collide with a manifest key: status"
    ):
        build_run_manifest_payload(**_s25_manifest_kwargs(extra={"status": "x"}))

def test_s25_build_run_manifest_payload_extra_boundary_key_collision_rejected():
    with pytest.raises(
        ValueError, match="extra key must not collide with a manifest key: stage"
    ):
        build_run_manifest_payload(**_s25_manifest_kwargs(extra={"stage": "x"}))

def test_s25_build_run_manifest_payload_non_string_extra_key_rejected():
    with pytest.raises(TypeError, match="extra keys must be strings"):
        build_run_manifest_payload(**_s25_manifest_kwargs(extra={1: "x"}))

def test_s25_unsafe_extra_value_rejected_at_scan_and_write_time(tmp_path):
    # build_run_manifest_payload itself does not scan free-text extra values;
    # the write path must reject them and leave no manifest file behind.
    payload = build_run_manifest_payload(
        **_s25_manifest_kwargs(extra={"note": "claim evidence"})
    )
    with pytest.raises(ValueError, match="contains unsafe evidence-boundary wording"):
        write_run_manifest(tmp_path, payload)
    assert not (tmp_path / MANIFEST_FILENAME).exists()

def test_s25_write_run_manifest_non_path_run_dir_rejected():
    with pytest.raises(TypeError, match="run_dir must be a pathlib.Path"):
        write_run_manifest("not_a_path", {"a": 1})

def test_s25_write_run_manifest_happy_round_trip(tmp_path):
    payload = build_run_manifest_payload(**_s25_manifest_kwargs())
    write_run_manifest(tmp_path, payload)
    loaded = gov.load_strict_json_object(tmp_path / MANIFEST_FILENAME)
    assert loaded == payload


# ---------- 3. baseline_driver: BaselineRunConfig validation matrix ----------

def _s25_run_config(**overrides):
    kwargs = dict(run_id="cfg_run", seed=11, update_rounds=1)
    kwargs.update(overrides)
    return BaselineRunConfig(**kwargs)

def test_s25_baseline_run_config_happy_defaults():
    config = _s25_run_config()
    assert config.episodes_per_round == 4
    assert config.steps_per_episode is None
    assert config.scenario_names == ("risk_gate_hidden_hazard",)
    assert config.sample_actions is True
    assert config.probe_every == 50
    assert config.probe_seed == 9001
    assert config.lambda_kill_bound == 1000.0
    assert config.entropy_floor == 0.01
    assert config.tier == "T1"
    assert config.result_parent is None
    assert config.heartbeat_every == 25

@pytest.mark.parametrize(
    "field, message",
    [
        ("seed", "seed must be an int"),
        ("update_rounds", "update_rounds must be a positive integer"),
        ("episodes_per_round", "episodes_per_round must be a positive integer"),
        ("steps_per_episode", "steps_per_episode must be positive"),
        ("probe_every", "probe_every must be a nonnegative integer"),
        ("probe_seed", "probe_seed must be an int"),
        ("heartbeat_every", "heartbeat_every must be a positive integer"),
    ],
)
def test_s25_baseline_run_config_bool_rejected_for_every_int_field(field, message):
    with pytest.raises(TypeError, match=re.escape(message)):
        _s25_run_config(**{field: True})

def test_s25_baseline_run_config_seed_zero_rejected():
    with pytest.raises(
        ValueError,
        match="seed must be a positive integer; seed 0 is intentionally",
    ):
        _s25_run_config(seed=0)

def test_s25_baseline_run_config_negative_seed_rejected():
    with pytest.raises(ValueError, match="seed must be a positive integer"):
        _s25_run_config(seed=-5)

def test_s25_baseline_run_config_seed_at_64_bit_bound_accepted():
    max_seed = 2**63 - 1 - (1 * 100000 + 8032)
    assert _s25_run_config(seed=max_seed, update_rounds=1).seed == max_seed

def test_s25_baseline_run_config_seed_above_64_bit_bound_rejected():
    max_seed = 2**63 - 1 - (1 * 100000 + 8032)
    with pytest.raises(
        ValueError,
        match=re.escape("seed must satisfy seed + update_rounds * 100000 + 8032"),
    ):
        _s25_run_config(seed=max_seed + 1, update_rounds=1)

def test_s25_baseline_run_config_update_rounds_zero_rejected():
    with pytest.raises(ValueError, match="update_rounds must be a positive integer"):
        _s25_run_config(update_rounds=0)

def test_s25_baseline_run_config_update_rounds_over_cap_rejected():
    with pytest.raises(ValueError, match=re.escape("update_rounds must be <= 20000")):
        _s25_run_config(update_rounds=20001)

@pytest.mark.parametrize("value", [1, 9])
def test_s25_baseline_run_config_episodes_per_round_out_of_band_rejected(value):
    with pytest.raises(
        ValueError, match="episodes_per_round must be between 2 and 8"
    ):
        _s25_run_config(episodes_per_round=value)

def test_s25_baseline_run_config_steps_per_episode_zero_rejected():
    with pytest.raises(ValueError, match="steps_per_episode must be positive"):
        _s25_run_config(steps_per_episode=0)

def test_s25_baseline_run_config_steps_per_episode_over_cap_rejected():
    with pytest.raises(
        ValueError, match="steps_per_episode must be None or between 1 and 32"
    ):
        _s25_run_config(steps_per_episode=33)

def test_s25_baseline_run_config_non_bool_sample_actions_rejected():
    with pytest.raises(TypeError, match="sample_actions must be a bool"):
        _s25_run_config(sample_actions=1)

def test_s25_baseline_run_config_negative_probe_every_rejected():
    with pytest.raises(ValueError, match="probe_every must be a nonnegative integer"):
        _s25_run_config(probe_every=-1)

def test_s25_baseline_run_config_probe_every_zero_accepted():
    assert _s25_run_config(probe_every=0).probe_every == 0

def test_s25_baseline_run_config_probe_seed_zero_rejected():
    with pytest.raises(
        ValueError,
        match="probe_seed must be a positive integer; seed 0 is intentionally",
    ):
        _s25_run_config(probe_seed=0)

def test_s25_baseline_run_config_probe_seed_above_signed_64_bit_rejected():
    with pytest.raises(
        ValueError,
        match="probe_seed plus reseed offset must fit in a signed 64-bit integer",
    ):
        _s25_run_config(probe_seed=2**63)

def test_s25_baseline_run_config_zero_lambda_kill_bound_rejected():
    with pytest.raises(
        ValueError, match="lambda_kill_bound must be positive and finite"
    ):
        _s25_run_config(lambda_kill_bound=0.0)

@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_s25_baseline_run_config_non_finite_lambda_kill_bound_rejected(value):
    with pytest.raises(ValueError, match="lambda_kill_bound must be finite"):
        _s25_run_config(lambda_kill_bound=value)

def test_s25_baseline_run_config_negative_entropy_floor_rejected():
    with pytest.raises(
        ValueError, match="entropy_floor must be nonnegative and finite"
    ):
        _s25_run_config(entropy_floor=-0.1)

def test_s25_baseline_run_config_tier_t3_exact_reserved_message():
    with pytest.raises(ValueError) as excinfo:
        _s25_run_config(tier="T3")
    assert str(excinfo.value) == (
        "tier T3 is reserved for the Stage 26 protocol executor"
    )

def test_s25_baseline_run_config_unknown_tier_rejected():
    with pytest.raises(ValueError, match="tier must be one of T0, T1, T2"):
        _s25_run_config(tier="T9")

def test_s25_baseline_run_config_non_string_tier_rejected():
    with pytest.raises(TypeError, match="tier must be a string"):
        _s25_run_config(tier=1)

def test_s25_baseline_run_config_heartbeat_every_zero_rejected():
    with pytest.raises(ValueError, match="heartbeat_every must be a positive integer"):
        _s25_run_config(heartbeat_every=0)

def test_s25_baseline_run_config_bad_result_parent_type_rejected():
    with pytest.raises(TypeError, match="result_parent must be a str, Path, or None"):
        _s25_run_config(result_parent=123)

def test_s25_baseline_run_config_invalid_run_id_rejected():
    with pytest.raises(ValueError, match=re.escape("run_id must match")):
        _s25_run_config(run_id="Bad")

def test_s25_baseline_run_config_scenario_names_must_be_non_empty_tuple():
    with pytest.raises(ValueError, match="scenario_names must be a non-empty tuple"):
        _s25_run_config(scenario_names=())
    with pytest.raises(ValueError, match="scenario_names must be a non-empty tuple"):
        _s25_run_config(scenario_names=["risk_gate_hidden_hazard"])

def test_s25_baseline_run_config_blank_scenario_name_rejected():
    with pytest.raises(
        ValueError, match="scenario_names entries must be non-empty strings"
    ):
        _s25_run_config(scenario_names=("  ",))


# ---------- 4. explained_variance_from_batch ----------

def test_s25_explained_variance_rejects_non_rollout_batch():
    with pytest.raises(TypeError, match="batch must be a RolloutBatch"):
        explained_variance_from_batch(object(), default_stage22_update_config())

def test_s25_explained_variance_rejects_non_update_config():
    with pytest.raises(TypeError, match="update_config must be a Stage22UpdateConfig"):
        explained_variance_from_batch(_s25_zeros_batch(), object())

def test_s25_explained_variance_all_false_valid_mask_exact_message():
    batch = _s25_zeros_batch(valid_mask=torch.zeros(2, 3, dtype=torch.bool))
    batch.validate(default_stage23_core_config())
    with pytest.raises(ValueError) as excinfo:
        explained_variance_from_batch(batch, default_stage22_update_config())
    assert str(excinfo.value) == (
        "valid_mask must select at least one timestep for "
        "explained-variance diagnostics"
    )

def test_s25_explained_variance_degenerate_zero_variance_returns_zero_and_flag():
    batch = _s25_zeros_batch()
    batch.validate(default_stage23_core_config())
    result = explained_variance_from_batch(batch, default_stage22_update_config())
    assert result["reward_explained_variance"] == 0.0
    assert result["hazard_cost_explained_variance"] == 0.0
    assert result["reward_target_variance"] == 0.0
    assert result["hazard_cost_target_variance"] == 0.0
    assert result["reward_target_variance_degenerate"] is True
    assert result["hazard_cost_target_variance_degenerate"] is True

def test_s25_explained_variance_hand_computed_regression():
    batch = _s25_regression_batch()
    # The hand-built batch must be a genuinely valid RolloutBatch.
    assert batch.validate(default_stage23_core_config()) == (2, 3)
    update_config = default_stage22_update_config()
    gamma = update_config.algorithm.discount_factor
    lam = update_config.algorithm.gae_lambda
    assert (gamma, lam) == (0.95, 0.9)

    # From-scratch recursion on the D-1 reward signal and the raw hazard cost.
    reward_signal = (
        batch.task_reward
        - update_config.losses.sensing_cost_coefficient * batch.sensing_cost
    )
    reference_reward_advantage = _s25_reference_gae(
        reward_signal,
        batch.reward_value,
        batch.next_reward_value,
        batch.terminal,
        batch.truncation,
        batch.valid_mask,
        gamma,
        lam,
    )
    reference_cost_advantage = _s25_reference_gae(
        batch.hazard_cost,
        batch.hazard_cost_value,
        batch.next_hazard_cost_value,
        batch.terminal,
        batch.truncation,
        batch.valid_mask,
        gamma,
        lam,
    )

    # Hand-computed double-precision anchors for every advantage element
    # (worked out on paper from the recursion; padding step is zero):
    #   row 0: t2 terminal, t1/t0 carry; row 1: t1 truncated, t2 invalid.
    hand_reward_advantage = [
        [1.8564875, 0.9725, -0.5],
        [1.54025, -0.45, 0.0],
    ]
    hand_cost_advantage = [
        [1.9888875, 2.8525, 0.5],
        [2.77975, 4.45, 0.0],
    ]
    for row in range(2):
        for col in range(3):
            assert (
                abs(
                    float(reference_reward_advantage[row, col].item())
                    - hand_reward_advantage[row][col]
                )
                < 1e-5
            )
            assert (
                abs(
                    float(reference_cost_advantage[row, col].item())
                    - hand_cost_advantage[row][col]
                )
                < 1e-5
            )

    mask = batch.valid_mask
    assert int(mask.sum().item()) == 5

    def expected_ev(target, value):
        target_variance = float(torch.var(target[mask], unbiased=False).item())
        residual_variance = float(
            torch.var(target[mask] - value[mask], unbiased=False).item()
        )
        return 1.0 - residual_variance / target_variance, target_variance

    reward_target = reference_reward_advantage + batch.reward_value
    cost_target = reference_cost_advantage + batch.hazard_cost_value
    expected_reward_ev, expected_reward_tv = expected_ev(
        reward_target, batch.reward_value
    )
    expected_cost_ev, expected_cost_tv = expected_ev(
        cost_target, batch.hazard_cost_value
    )

    actual = explained_variance_from_batch(batch, update_config)
    assert abs(actual["reward_explained_variance"] - expected_reward_ev) <= 1e-6
    assert abs(actual["hazard_cost_explained_variance"] - expected_cost_ev) <= 1e-6
    assert abs(actual["reward_target_variance"] - expected_reward_tv) <= 1e-6
    assert abs(actual["hazard_cost_target_variance"] - expected_cost_tv) <= 1e-6
    assert actual["reward_target_variance_degenerate"] is False
    assert actual["hazard_cost_target_variance_degenerate"] is False
    # Non-degenerate targets by construction.
    assert expected_reward_tv > 1e-6
    assert expected_cost_tv > 1e-6


# ---------- 5. Probe machinery ----------

def test_s25_greedy_model_policy_fn_rejects_non_model():
    with pytest.raises(TypeError, match="model must be a RecurrentMAPPOActorCritic"):
        greedy_model_policy_fn(42)

def test_s25_greedy_model_policy_fn_exact_int_actions_in_agent_order(s25_model):
    scenario = stage23_scenario_catalog()["unit_empty"]
    environment = RiskAwareActiveSensingGridEnvironment(
        Stage23EnvironmentConfig(
            scenario_name=scenario.name,
            width=scenario.width,
            height=scenario.height,
            seed=17,
        )
    )
    observations, _infos = environment.reset(seed=17)
    core = default_stage23_core_config()
    policy = greedy_model_policy_fn(s25_model)
    state = {}
    actions = policy(observations, environment, state, None)
    assert tuple(actions.keys()) == tuple(environment.agents)
    for action in actions.values():
        assert set(action) == {"sensing_action", "movement_action"}
        assert type(action["sensing_action"]) is int
        assert type(action["movement_action"]) is int
        assert 0 <= action["sensing_action"] < core.sensing_action_count
        assert 0 <= action["movement_action"] < core.movement_action_count
    # The per-episode recurrent state lives in the runner-owned state dict.
    assert isinstance(state["model_history_state"], torch.Tensor)
    assert state["model_history_state"].shape[1] == len(environment.agents)
    assert state["model_history_state"].requires_grad is False

def test_s25_greedy_policy_adapt_flag_stays_false_with_zero_reveal_channel(s25_model):
    # unit_empty has no hidden hazards, so the reveal channel stays zero for
    # the whole episode and the counterfactual movement can never differ.
    record = _s25_run_comparator_policy(
        Stage24ComparatorConfig("unit_empty", seed=303),
        policy_name="learned_greedy_policy",
        policy=greedy_model_policy_fn(s25_model),
        diagnostic_only=False,
        uses_hidden_hazards_for_decision=False,
    )
    assert record["policy_name"] == "learned_greedy_policy"
    assert record["revealed_hazard_count"] == 0
    assert record["used_revealed_information_to_adapt_movement"] is False

def test_s25_run_readiness_probe_rejects_non_model():
    with pytest.raises(TypeError, match="model must be a RecurrentMAPPOActorCritic"):
        run_readiness_probe(
            object(), probe_seed=311, update_config=default_stage22_update_config()
        )

def test_s25_run_readiness_probe_rejects_zero_probe_seed(s25_model):
    with pytest.raises(
        ValueError,
        match="probe_seed must be a positive integer; seed 0 is intentionally",
    ):
        run_readiness_probe(
            s25_model, probe_seed=0, update_config=default_stage22_update_config()
        )

def test_s25_run_readiness_probe_rejects_non_update_config(s25_model):
    with pytest.raises(TypeError, match="update_config must be a Stage22UpdateConfig"):
        run_readiness_probe(s25_model, probe_seed=311, update_config=object())

def test_s25_run_readiness_probe_structure_and_heldout_absence(s25_model):
    probe = run_readiness_probe(
        s25_model, probe_seed=311, update_config=default_stage22_update_config()
    )
    assert set(probe) == {
        "probe_seed",
        "scenario",
        "variants",
        "all_training_readiness_surfaces_met",
    }
    assert probe["probe_seed"] == 311
    assert type(probe["all_training_readiness_surfaces_met"]) is bool

    scenario_surface = probe["scenario"]
    assert set(scenario_surface) == {"comparisons", "verdict"}
    assert set(scenario_surface["comparisons"]) == {
        "no_sense_shortest_path",
        "always_sense_shortest_path",
        "random_policy",
        "risk_aware_oracle_or_heuristic",
        "selective_sense_risk_aware",
        "learned_greedy_policy",
    }
    assert set(scenario_surface["verdict"]) == _S25_VERDICT_KEYS

    # Exactly the two non-held-out variant surfaces are present.
    assert set(probe["variants"]) == {
        "risk_gate_train_center",
        "risk_gate_readiness_shifted",
    }
    for surface in probe["variants"].values():
        assert set(surface) == {"comparisons", "verdict"}
        assert set(surface["comparisons"]) == {
            "no_sense_shortest_path",
            "always_sense_shortest_path",
            "selective_sense_risk_aware",
            "learned_greedy_policy",
        }
        assert set(surface["verdict"]) == _S25_VERDICT_KEYS

    # The held-out layout name must not appear anywhere in the structure.
    _s25_assert_no_heldout_reference(probe)

def test_s25_run_readiness_probe_is_deterministic_for_same_model_and_seed(s25_model):
    probe_a = run_readiness_probe(
        s25_model, probe_seed=311, update_config=default_stage22_update_config()
    )
    probe_b = run_readiness_probe(
        s25_model, probe_seed=311, update_config=default_stage22_update_config()
    )
    assert probe_a == probe_b
    assert probe_a["scenario"]["verdict"] == probe_b["scenario"]["verdict"]
    for name in probe_a["variants"]:
        assert (
            probe_a["variants"][name]["verdict"]
            == probe_b["variants"][name]["verdict"]
        )

def test_s25_run_readiness_probe_restores_model_training_mode(s25_model):
    s25_model.train(True)
    run_readiness_probe(
        s25_model, probe_seed=311, update_config=default_stage22_update_config()
    )
    assert s25_model.training is True

def test_s25_held_out_variant_guard_raises_runtime_error():
    variant = Stage24AHazardLayoutVariant(
        name="hand_built_held_out_probe_guard",
        group="held_out",
        hidden_hazard_cells=((2, 2), (3, 2)),
        public_risk_zone_cells=_PUBLIC_GATE_RISK_ZONE_CELLS,
        description="hand-built held-out variant for the Stage 25 guard test",
    )
    with pytest.raises(
        RuntimeError,
        match="held-out hazard-layout variant must never reach the Stage 25",
    ):
        _assert_probe_variant_not_held_out(variant)


# ---------- 6. run_baseline_training end-to-end (module-scoped run) ----------

def test_s25_run_baseline_training_returns_run_directory(s25_completed_run):
    run_dir, parent, config = s25_completed_run
    assert run_dir == parent.resolve() / config.run_id
    assert run_dir.is_dir()

def test_s25_run_baseline_training_rejects_non_config():
    with pytest.raises(TypeError, match="config must be a BaselineRunConfig"):
        run_baseline_training({"run_id": "x"})

def test_s25_completed_run_manifest_contents(s25_completed_run):
    run_dir, parent, config = s25_completed_run
    manifest = gov.load_strict_json_object(run_dir / MANIFEST_FILENAME)
    assert set(manifest) == _S25_MANIFEST_COMPLETE_KEYS
    assert manifest["status"] == "complete"
    assert manifest["manifest_version"] == 1
    assert manifest["last_completed_round"] == 1
    assert manifest["run_id"] == config.run_id
    assert manifest["tier"] == "T0"
    assert manifest["seeds"] == [151]
    assert manifest["implementing_dr_ids"] == []
    assert manifest["git_head"] is None or (
        isinstance(manifest["git_head"], str) and manifest["git_head"]
    )
    assert isinstance(manifest["runtime_record"], dict)
    assert type(manifest["pid"]) is int and manifest["pid"] > 0
    gov.validate_timestamp(manifest["launch_timestamp_utc"])
    gov.validate_timestamp(manifest["finish_timestamp_utc"])
    for key, value in _S25_BOUNDARY_FLAGS.items():
        assert manifest[key] == value
    config_block = manifest["config"]
    assert config_block["run_id"] == config.run_id
    assert config_block["seed"] == 151
    assert config_block["update_rounds"] == 2
    assert config_block["episodes_per_round"] == 2
    assert config_block["steps_per_episode"] == 6
    assert config_block["scenario_names"] == ["risk_gate_hidden_hazard"]
    assert config_block["result_parent"] == str(parent)
    assert config_block["torch_version"] == str(torch.__version__)
    assert isinstance(config_block["stage22_update_config"], dict)

def test_s25_completed_run_training_curve_exact_two_records_full_key_set(
    s25_completed_run,
):
    run_dir, _, _ = s25_completed_run
    records = gov.load_strict_jsonl_objects(run_dir / "training_curve.jsonl")
    assert len(records) == 2
    for index, record in enumerate(records):
        assert set(record) == _S25_CURVE_KEYS
        assert record["round_index"] == index
        assert record["round_seed"] == 151 + index * 100000
        assert type(record["wall_time_s"]) is float and record["wall_time_s"] >= 0.0
        assert type(record["success_count"]) is int
        assert 0.0 <= record["success_rate"] <= 1.0
        assert 0.0 < record["mean_step_count"] <= 6.0
        for int_field in (
            "total_sensing_actions",
            "total_movement_actions",
            "hazard_entry_count",
            "revealed_hazard_count",
        ):
            assert type(record[int_field]) is int and record[int_field] >= 0
        for bool_field in (
            "entropy_collapse_flag",
            "reward_target_variance_degenerate",
            "hazard_cost_target_variance_degenerate",
        ):
            assert type(record[bool_field]) is bool
        assert record["parameter_delta_l1"] > 0.0
    # The Lagrange multiplier threads across rounds (the D-3 cadence).
    assert (
        records[1]["lagrange_multiplier_before"]
        == records[0]["lagrange_multiplier_after"]
    )

def test_s25_completed_run_episode_log_exact_four_records(s25_completed_run):
    run_dir, _, _ = s25_completed_run
    records = gov.load_strict_jsonl_objects(run_dir / "episode_log.jsonl")
    assert len(records) == 4
    assert [record["round_index"] for record in records] == [0, 0, 1, 1]
    assert [record["episode_index"] for record in records] == [0, 1, 0, 1]
    assert [record["seed"] for record in records] == [151, 152, 100151, 100152]
    for record in records:
        assert set(record) == _S25_EPISODE_KEYS
        assert record["scenario_name"] == "risk_gate_hidden_hazard"
        assert record["sample_actions"] is True
        assert record["environment_max_steps"] == 6
        assert 1 <= record["step_count"] <= 6

def test_s25_completed_run_probe_log_records(s25_completed_run):
    run_dir, _, _ = s25_completed_run
    records = gov.load_strict_jsonl_objects(run_dir / "probe_log.jsonl")
    assert len(records) >= 2
    assert len(records) == 2  # probe_every=1 over 2 rounds; the final probe
    assert [record["round_index"] for record in records] == [0, 1]
    for record in records:
        assert set(record) == _S25_PROBE_RECORD_KEYS
        assert record["probe_seed"] == 311
        assert set(record["variants"]) == {
            "risk_gate_train_center",
            "risk_gate_readiness_shifted",
        }
        assert type(record["all_training_readiness_surfaces_met"]) is bool
        _s25_assert_no_heldout_reference(record)

def test_s25_completed_run_every_record_carries_development_boundary(
    s25_completed_run,
):
    run_dir, _, _ = s25_completed_run
    manifest = gov.load_strict_json_object(run_dir / MANIFEST_FILENAME)
    all_records = [manifest]
    for name in ("training_curve.jsonl", "episode_log.jsonl", "probe_log.jsonl"):
        all_records.extend(gov.load_strict_jsonl_objects(run_dir / name))
    assert len(all_records) == 1 + 2 + 4 + 2
    for record in all_records:
        assert record["development_only"] is True
        assert record["stage"] == "25-harness"

def test_s25_rerun_with_same_run_id_and_parent_raises_file_exists(s25_completed_run):
    _, _, config = s25_completed_run
    with pytest.raises(FileExistsError, match="run directory already exists"):
        run_baseline_training(config)


# ---------- 7. Failure paths ----------

class _S25HostileStrError(Exception):
    def __str__(self):
        raise RuntimeError("hostile __str__")

def test_s25_crash_path_marks_manifest_crashed_and_reraises(tmp_path, monkeypatch):
    def _injected_failure(*args, **kwargs):
        raise ValueError("injected")

    monkeypatch.setattr(s25_bd, "stage22_ppo_lagrangian_update", _injected_failure)
    config = BaselineRunConfig(
        run_id="crash_seed501",
        seed=501,
        update_rounds=1,
        episodes_per_round=2,
        steps_per_episode=4,
        probe_every=0,
        result_parent=tmp_path,
    )
    with pytest.raises(ValueError, match="injected"):
        run_baseline_training(config)
    run_dir = tmp_path / "crash_seed501"
    manifest = gov.load_strict_json_object(run_dir / MANIFEST_FILENAME)
    assert manifest["status"] == "crashed"
    assert manifest["kill_reason"].startswith("ValueError")
    assert manifest["kill_reason"] == "ValueError: injected"
    assert manifest["last_completed_round"] == -1
    # The crash happened before any curve/episode append.
    assert sorted(p.name for p in run_dir.iterdir()) == [MANIFEST_FILENAME]

def test_s25_crash_path_hostile_str_falls_back_to_type_name(tmp_path, monkeypatch):
    def _hostile_failure(*args, **kwargs):
        raise _S25HostileStrError()

    monkeypatch.setattr(s25_bd, "stage22_ppo_lagrangian_update", _hostile_failure)
    config = BaselineRunConfig(
        run_id="hostile_seed601",
        seed=601,
        update_rounds=1,
        episodes_per_round=2,
        steps_per_episode=4,
        probe_every=0,
        result_parent=tmp_path,
    )
    with pytest.raises(_S25HostileStrError):
        run_baseline_training(config)
    manifest = gov.load_strict_json_object(
        tmp_path / "hostile_seed601" / MANIFEST_FILENAME
    )
    assert manifest["status"] == "crashed"
    assert manifest["kill_reason"] == "_S25HostileStrError"

def test_s25_lambda_kill_fires_on_round_zero(tmp_path):
    config = BaselineRunConfig(
        run_id="kill_seed401",
        seed=401,
        update_rounds=2,
        episodes_per_round=2,
        steps_per_episode=4,
        probe_every=0,
        lambda_kill_bound=1e-9,  # smallest practical legal bound: any positive
        result_parent=tmp_path,  # post-update multiplier (>= 0.08 here) kills.
    )
    run_dir = run_baseline_training(config)
    manifest = gov.load_strict_json_object(run_dir / MANIFEST_FILENAME)
    assert manifest["status"] == "killed"
    assert manifest["kill_reason"] == "lambda exceeded kill bound"
    assert manifest["last_completed_round"] == 0
    curve = gov.load_strict_jsonl_objects(run_dir / "training_curve.jsonl")
    assert len(curve) == 1
    assert curve[0]["lagrange_multiplier_after"] > 1e-9
    episodes = gov.load_strict_jsonl_objects(run_dir / "episode_log.jsonl")
    assert len(episodes) == 2
    # No probe fired: probe_every=0 and the kill stopped before the final round.
    assert not (run_dir / "probe_log.jsonl").exists()

def test_s25_unsafe_run_id_rejected_before_any_directory_created(tmp_path):
    config = BaselineRunConfig(
        run_id="proto_evaluation_x",  # passes the run-id grammar but trips the
        seed=701,                     # unsafe-wording input scan
        update_rounds=1,
        result_parent=tmp_path,
    )
    with pytest.raises(ValueError, match="contains unsafe evaluation wording"):
        run_baseline_training(config)
    assert list(tmp_path.iterdir()) == []


# ---------- 8. Governance: end-to-end wording hygiene of a completed run ----------

def test_s25_completed_run_artifacts_rescan_clean_with_harness_allowed_keys(
    s25_completed_run,
):
    run_dir, _, _ = s25_completed_run
    allowed = harness_allowed_structural_keys()
    manifest = gov.load_strict_json_object(run_dir / MANIFEST_FILENAME)
    gov.reject_unsafe_boundary_values(
        manifest, MANIFEST_FILENAME, allowed_structural_keys=allowed
    )
    for name in ("training_curve.jsonl", "episode_log.jsonl", "probe_log.jsonl"):
        for index, record in enumerate(
            gov.load_strict_jsonl_objects(run_dir / name)
        ):
            gov.reject_unsafe_boundary_values(
                record, f"{name}:{index}", allowed_structural_keys=allowed
            )

def test_s25_completed_run_file_set_and_no_blocked_suffixes_or_tokens(
    s25_completed_run,
):
    run_dir, _, _ = s25_completed_run
    names = sorted(p.name for p in run_dir.iterdir())
    assert names == [
        MANIFEST_FILENAME,
        "episode_log.jsonl",
        "probe_log.jsonl",
        "training_curve.jsonl",
    ]
    for path in run_dir.iterdir():
        assert path.is_file()
        assert not path.is_symlink()
        assert path.suffix.lower() not in artifacts.BLOCKED_ARTIFACT_SUFFIXES
        collapsed_name = _s25_collapse(path.name)
        for token in artifacts.BLOCKED_ARTIFACT_NAME_TOKENS:
            assert _s25_collapse(token) not in collapsed_name


# ===================== SECTION DR: Decision Record pins (2026-07-04) =====================
# Executable pins for the countersigned Decision Records DR-D12, DR-S2-23,
# DR-D8, DR-D13, and DR-NEW-methodology-1 (docs/decisions/). DR-D12 and
# DR-S2-23 pin the implemented changes; DR-D8, DR-D13, and
# DR-NEW-methodology-1 pin current behavior as decided (test-only pins).

import random as _dr_random
from dataclasses import asdict as _dr_asdict

from raas_marl.environments.active_sensing.stage24_diagnostics import (
    _stage24a_public_risk_zone_cells,
)


# ---------- DR-D12: legacy advantage-mean summary keys removed ----------

_DR_D12_RETAINED_KEYS = (
    "raw_reward_advantage_mean",
    "raw_hazard_cost_advantage_mean",
    "policy_reward_advantage_mean",
    "policy_hazard_cost_advantage_mean",
)

_DR_D12_REMOVED_KEYS = (
    "reward_advantage_mean",
    "hazard_cost_advantage_mean",
    "reward_advantage_mean_kind",
    "hazard_cost_advantage_mean_kind",
)


def test_dr_d12_legacy_advantage_mean_keys_absent_and_provenance_keys_retained():
    model = _gb_model()
    batch = _gb_collect(model=model).batch
    summary = stage22_ppo_lagrangian_update(model, batch).summary
    for retained in _DR_D12_RETAINED_KEYS:
        assert retained in summary, f"retained provenance key missing: {retained}"
        assert isinstance(summary[retained], float)
    for removed in _DR_D12_REMOVED_KEYS:
        assert removed not in summary, f"legacy key must be removed (DR-D12): {removed}"


# ---------- DR-S2-23: config dataclasses store sanitized base floats ----------

class _DrHostileFloat(float):
    """float subclass with hijacked arithmetic/comparison/conversion dunders.

    DR-S2-23 hardening probe: if a config dataclass stored this object, any
    downstream multiplication or comparison would dispatch to these overrides.
    ``__eq__``/``__hash__`` are left untouched so value assertions stay honest.
    """

    def __mul__(self, other):  # pragma: no cover - must never be dispatched
        return "hijacked-mul"

    __rmul__ = __mul__

    def __gt__(self, other):  # pragma: no cover - must never be dispatched
        return "hijacked-gt"

    def __lt__(self, other):  # pragma: no cover
        return "hijacked-lt"

    def __float__(self):  # pragma: no cover - base-slot conversion bypasses this
        return float("nan")


class _DrHostileInt(int):
    """int subclass with hijacked arithmetic/conversion dunders (DR-S2-23 probe)."""

    def __mul__(self, other):  # pragma: no cover - must never be dispatched
        return "hijacked-mul"

    __rmul__ = __mul__

    def __gt__(self, other):  # pragma: no cover
        return "hijacked-gt"

    def __float__(self):  # pragma: no cover - base-slot conversion bypasses this
        return float("nan")


def test_dr_s2_23_algorithm_config_stores_base_floats_from_hostile_subclasses():
    cfg = AlgorithmConfig(
        discount_factor=_DrHostileFloat(0.99),
        gae_lambda=_DrHostileFloat(0.95),
        ppo_clip_range=_DrHostileFloat(0.2),
    )
    for name, expected in (
        ("discount_factor", 0.99),
        ("gae_lambda", 0.95),
        ("ppo_clip_range", 0.2),
    ):
        stored = getattr(cfg, name)
        assert type(stored) is float, f"{name} must be stored as base float (DR-S2-23)"
        assert stored == expected


def test_dr_s2_23_algorithm_config_accepts_hostile_int_where_int_is_legal():
    # require_probability accepts the integer endpoints 0 and 1; a hostile int
    # subclass must still be normalized to a base float on storage.
    cfg = AlgorithmConfig(
        discount_factor=_DrHostileInt(1),
        gae_lambda=_DrHostileInt(0),
        ppo_clip_range=_DrHostileFloat(0.2),
    )
    assert type(cfg.discount_factor) is float
    assert cfg.discount_factor == 1.0
    assert type(cfg.gae_lambda) is float
    assert cfg.gae_lambda == 0.0


def test_dr_s2_23_loss_config_stores_base_floats_from_hostile_subclasses():
    cfg = LossConfig(
        reward_entropy_coefficient=_DrHostileFloat(0.0),
        movement_entropy_coefficient=_DrHostileFloat(0.01),
        sensing_entropy_coefficient=_DrHostileFloat(0.01),
        hazard_cost_coefficient=_DrHostileInt(1),
        sensing_cost_coefficient=_DrHostileFloat(0.5),
        value_loss_coefficient=_DrHostileFloat(0.5),
    )
    for name, expected in (
        ("reward_entropy_coefficient", 0.0),
        ("movement_entropy_coefficient", 0.01),
        ("sensing_entropy_coefficient", 0.01),
        ("hazard_cost_coefficient", 1.0),
        ("sensing_cost_coefficient", 0.5),
        ("value_loss_coefficient", 0.5),
    ):
        stored = getattr(cfg, name)
        assert type(stored) is float, f"{name} must be stored as base float (DR-S2-23)"
        assert stored == expected


def test_dr_s2_23_lagrange_config_stores_base_floats_from_hostile_subclasses():
    cfg = LagrangeConfig(
        initial_multiplier=_DrHostileFloat(0.1),
        learning_rate=_DrHostileFloat(0.2),
        hazard_budget=_DrHostileInt(1),
    )
    for name, expected in (
        ("initial_multiplier", 0.1),
        ("learning_rate", 0.2),
        ("hazard_budget", 1.0),
    ):
        stored = getattr(cfg, name)
        assert type(stored) is float, f"{name} must be stored as base float (DR-S2-23)"
        assert stored == expected


def test_dr_s2_23_genuine_float_inputs_value_identical_and_asdict_round_trip():
    # DR-S2-23 promises value-identity for genuine float inputs: equality and
    # asdict serialization must be indistinguishable from the pre-change state.
    algorithm = _gb_algorithm()
    losses = _gb_losses()
    lagrange = _gb_lagrange()
    assert _dr_asdict(algorithm) == {
        "discount_factor": 0.95,
        "gae_lambda": 0.9,
        "ppo_clip_range": 0.2,
    }
    assert _dr_asdict(losses) == {
        "reward_entropy_coefficient": 0.0,
        "movement_entropy_coefficient": 0.01,
        "sensing_entropy_coefficient": 0.01,
        "hazard_cost_coefficient": 1.0,
        "sensing_cost_coefficient": 0.5,
        "value_loss_coefficient": 0.5,
    }
    assert _dr_asdict(lagrange) == {
        "initial_multiplier": 0.1,
        "learning_rate": 0.2,
        "hazard_budget": 0.1,
    }
    for payload in (_dr_asdict(algorithm), _dr_asdict(losses), _dr_asdict(lagrange)):
        for value in payload.values():
            assert type(value) is float


def test_dr_s2_23_update_with_hostile_subclass_config_matches_plain_float_summary():
    # The un-laundered LossConfig coefficient sinks in update.py must now be
    # inert against hostile subclasses: same seeds + hostile-subclass config
    # must reproduce the plain-float summary exactly.
    def _run(losses):
        torch.manual_seed(4242)
        model = _gb_model()
        batch = _gb_collect(model=model).batch
        cfg = Stage22UpdateConfig(
            algorithm=_gb_algorithm(), losses=losses, lagrange=_gb_lagrange()
        )
        return stage22_ppo_lagrangian_update(model, batch, cfg).summary

    plain = _run(_gb_losses())
    hostile = _run(
        LossConfig(
            reward_entropy_coefficient=_DrHostileFloat(0.0),
            movement_entropy_coefficient=_DrHostileFloat(0.01),
            sensing_entropy_coefficient=_DrHostileFloat(0.01),
            hazard_cost_coefficient=_DrHostileFloat(1.0),
            sensing_cost_coefficient=_DrHostileFloat(0.5),
            value_loss_coefficient=_DrHostileFloat(0.5),
        )
    )
    assert plain == hostile


# ---------- DR-D8: `_rng` reserved deterministic state (test-only pin) ----------

def test_dr_d8_rng_is_reserved_random_random_instance_after_reset():
    env = RiskAwareActiveSensingGridEnvironment(Stage23EnvironmentConfig())
    assert isinstance(env._rng, _dr_random.Random)
    env.reset(seed=7)
    assert isinstance(env._rng, _dr_random.Random)


def test_dr_d8_identical_seed_envs_identical_streams_rng_never_consumed():
    # DR-D8 T0 pin: `_rng` is reserved and never sampled, so two identical-seed
    # environments driven by identical action sequences must produce identical
    # observation/reward/flag/info streams. A failure here means something
    # started consuming `_rng`, falsifying the DR's premise and triggering its
    # same-commit seed-tightening rule.
    action_script = (
        {"agent_0": {SENSING_ACTION_FIELD: 1, MOVEMENT_ACTION_FIELD: 1},
         "agent_1": {SENSING_ACTION_FIELD: 0, MOVEMENT_ACTION_FIELD: 2}},
        {"agent_0": {SENSING_ACTION_FIELD: 0, MOVEMENT_ACTION_FIELD: 3},
         "agent_1": {SENSING_ACTION_FIELD: 1, MOVEMENT_ACTION_FIELD: 0}},
        {"agent_0": {SENSING_ACTION_FIELD: 1, MOVEMENT_ACTION_FIELD: 4},
         "agent_1": {SENSING_ACTION_FIELD: 1, MOVEMENT_ACTION_FIELD: 1}},
    )
    streams = []
    for _ in range(2):
        env = RiskAwareActiveSensingGridEnvironment(Stage23EnvironmentConfig())
        stream = [env.reset(seed=11)]
        for actions in action_script:
            stream.append(env.step(actions))
        streams.append(stream)
    assert streams[0] == streams[1]


# ---------- DR-D13: public risk zone scope (test-only pin) ----------

def test_dr_d13_gate_scenario_public_risk_zone_is_exactly_rows_1_3_cols_1_3():
    zone = _stage24a_public_risk_zone_cells("risk_gate_hidden_hazard")
    assert zone == _PUBLIC_GATE_RISK_ZONE_CELLS
    assert len(zone) == 9
    assert set(zone) == {(row, col) for row in (1, 2, 3) for col in (1, 2, 3)}


def test_dr_d13_non_gate_catalog_scenarios_get_empty_public_risk_zone():
    # D-13: catalog scenarios that are NEITHER the risk_gate scenario NOR a
    # DR-D1/L1 fork scenario (incl. the DR-READINESS-CURRICULUM pairs) still get
    # an empty public risk zone (selective_sense degenerates to no_sense there).
    # The fork scenarios get their own gate zone (tested separately below).
    names = available_scenarios()
    fork_families = fork_hazard_layout_families()
    fork = (
        set(fork_families["training"])
        | set(fork_families["readiness"])
        | set(fork_curriculum_scenarios())
    )
    empty_zone = [
        name
        for name in names
        if name != "risk_gate_hidden_hazard" and name not in fork
    ]
    assert "risk_gate_hidden_hazard" in names
    assert set(empty_zone) == {
        "standard_branching_hazard",
        "unit_empty",
        "unit_single_hazard",
    }
    for name in empty_zone:
        assert _stage24a_public_risk_zone_cells(name) == tuple()


# ---------- DR-D1/L1: risk-fork family (selective sensing forced by STRUCTURE) ----------

_FORK_TRAIN_NAMES = ("risk_fork_train_lower", "risk_fork_train_upper")
_FORK_READINESS_NAMES = ("risk_fork_readiness_lower", "risk_fork_readiness_upper")


def test_fork_families_registered_and_construct():
    catalog = stage23_scenario_catalog()
    for name in _FORK_TRAIN_NAMES + _FORK_READINESS_NAMES:
        assert name in catalog
        sc = catalog[name]
        assert sc.name == name
        assert sc.width == 8 and sc.height == 8
        assert sc.starts == ((3, 0), (4, 0))
        assert sc.goals == ((3, 7),)
        assert len(sc.hidden_hazard_cells) == 1


def test_fork_hazard_layout_families_registry():
    families = fork_hazard_layout_families()
    assert set(families) == {"training", "readiness"}
    assert families["training"] == _FORK_TRAIN_NAMES
    assert families["readiness"] == _FORK_READINESS_NAMES
    # Held-out fork scenarios are DEFERRED to the Protocol-v1 lock (I-3): they must
    # NOT be present in the catalog or the registry this shift.
    catalog = stage23_scenario_catalog()
    for name in catalog:
        assert "heldout" not in name and "held_out" not in name


def test_fork_public_risk_zone_covers_gates_and_is_broad():
    # Each fork scenario gets a gate-approach zone spanning cols 3,4,5 over the gate
    # rows +/- 1; it must include both open gate cells and be broader than the exact
    # hidden hazard (never pinpoints the single hidden gate).
    for name in _FORK_TRAIN_NAMES + _FORK_READINESS_NAMES:
        sc = stage23_scenario_catalog()[name]
        zone = set(_stage24a_public_risk_zone_cells(name))
        gates = {(r, 4) for r in range(sc.height) if (r, 4) not in set(sc.obstacles)}
        assert len(gates) == 2
        assert gates <= zone
        assert set(sc.hidden_hazard_cells) <= zone
        assert len(zone) > len(sc.hidden_hazard_cells)


def _fork_union_hazards(names):
    cells = set()
    for name in names:
        cells |= set(stage23_scenario_catalog()[name].hidden_hazard_cells)
    return tuple(sorted(cells))


def _fork_union_is_start_goal_cut(names):
    """Independent executable BAR-A: BFS from each start avoiding obstacles PLUS the
    union of the group's hazards must NOT reach the goal (no blind zero-hazard route).
    """
    base = stage23_scenario_catalog()[names[0]]
    union = _fork_union_hazards(names)
    for start in base.starts:
        path = _shortest_path(
            start,
            base.goals,
            width=base.width,
            height=base.height,
            obstacles=base.obstacles,
            avoid_cells=union,
        )
        if path:  # goal reachable avoiding the union => NOT a cut
            return False
    return True


def test_fork_training_bar_a_no_blind_safe_route_by_construction():
    # THE EXECUTABLE BAR-A CHECK (independent of verify_fork_family_forces_sensing):
    # the union of the training family's hidden hazards is a start->goal CUT, so no
    # deterministic never-sense policy can be both hazard-free and successful across
    # the layout distribution. This enforces the no-blind-safe-route property in the
    # suite, not just in the DR.
    assert _fork_union_is_start_goal_cut(_FORK_TRAIN_NAMES) is True
    # Each single layout, by contrast, leaves a safe route (union cut but each part
    # is NOT a cut) â€” the "winnable if you know the layout" half.
    for name in _FORK_TRAIN_NAMES:
        sc = stage23_scenario_catalog()[name]
        for start in sc.starts:
            assert _shortest_path(
                start,
                sc.goals,
                width=sc.width,
                height=sc.height,
                obstacles=sc.obstacles,
                avoid_cells=sc.hidden_hazard_cells,
            )  # non-empty safe route exists


def test_fork_readiness_bar_a_no_blind_safe_route_by_construction():
    assert _fork_union_is_start_goal_cut(_FORK_READINESS_NAMES) is True


def test_fork_no_memorizable_safe_gate_every_gate_hazardous():
    # A2: the union of hazards covers EVERY open gate, so no single blind gate
    # choice is safe across the family.
    for names in (_FORK_TRAIN_NAMES, _FORK_READINESS_NAMES):
        base = stage23_scenario_catalog()[names[0]]
        gates = {(r, 4) for r in range(base.height) if (r, 4) not in set(base.obstacles)}
        assert gates == set(_fork_union_hazards(names))


def test_fork_verify_function_confirms_bar_a_and_bar_b():
    for names in (_FORK_TRAIN_NAMES, _FORK_READINESS_NAMES):
        rec = verify_fork_family_forces_sensing(names)
        # BAR-A: sensing necessary by construction.
        assert rec["bar_a_union_cut"] is True
        assert rec["bar_a_all_gates_hazardous"] is True
        assert rec["bar_a_no_sense_mean_hazard_exceeds_budget"] is True
        assert rec["bar_a_no_sense_mean_hazard"] > rec["hazard_budget"]
        assert rec["bar_a_sensing_forced_by_construction"] is True
        # BAR-B: task remains winnable (a sensing agent reaches goal at 0 hazard).
        assert rec["bar_b_all_layouts_winnable"] is True
        assert rec["bar_b_selective_zero_hazard"] is True
        assert rec["bar_b_selective_preserves_success"] is True
        assert rec["bar_b_task_learnable_by_construction"] is True
        assert rec["forces_sensing_and_winnable"] is True


def test_fork_min_blind_hazard_is_two_x_budget():
    # A3 constraint-level margin: the no_sense comparator (a concrete near-optimal
    # blind witness) averages team episodic hazard 1.0 over each group's layouts =
    # 2x the d_ep=0.5 budget. On each group exactly one layout is unsafe for the
    # blind shortest route (2.0 hazard) and one is safe (0.0).
    for names in (_FORK_TRAIN_NAMES, _FORK_READINESS_NAMES):
        rec = verify_fork_family_forces_sensing(names, hazard_budget=0.5)
        assert rec["bar_a_no_sense_mean_hazard"] == 1.0
        haz = sorted(
            pl["no_sense_hazard_cost_sum"] for pl in rec["per_layout"].values()
        )
        assert haz == [0.0, 2.0]  # blind is unsafe on exactly half the layouts


def test_fork_selective_sense_zero_hazard_and_selective_and_adaptive():
    # BAR-B at the policy level + the C1 composite signature (AM-2: these are the
    # TRUE sparse env fields from the comparator rollouts, never a shaped return).
    for names in (_FORK_TRAIN_NAMES, _FORK_READINESS_NAMES):
        adapts_somewhere = False
        for name in names:
            rec = selective_sense_risk_aware(name, seed=2400)
            assert rec["team_success"] is True
            assert rec["hazard_cost_sum"] == 0.0  # selective sensing avoids the hazard
            assert 0.0 < rec["sensing_rate"] < 1.0  # non-degenerate (not never/always)
            adapts_somewhere = adapts_somewhere or bool(
                rec["used_revealed_information_to_adapt_movement"]
            )
        # On the layout where the shortest-path gate is the hazard, the reveal must
        # CHANGE the route (C2 adaptivity) â€” true for at least one layout per group.
        assert adapts_somewhere is True


def test_fork_am2_no_shaped_leak_true_hazard_occupancy_units():
    # AM-2 grade-surface guard: the fork verification's graded hazard values are TRUE
    # occupancy counts (exact integer multiples of the 1.0 hazard-cost unit), so no
    # shaped/fractional Phi term could have leaked into a graded field.
    for names in (_FORK_TRAIN_NAMES, _FORK_READINESS_NAMES):
        rec = verify_fork_family_forces_sensing(names)
        for pl in rec["per_layout"].values():
            assert pl["no_sense_hazard_cost_sum"] in (0.0, 2.0)
            assert pl["selective_hazard_cost_sum"] == 0.0
            assert pl["no_sense_team_success"] is True
            assert pl["selective_team_success"] is True


# ---------- DR-NEW-methodology-1: binary sensing gate, named SENSE index ----------

def test_dr_new_methodology_1_sense_action_index_constant_is_one():
    assert STAGE23_SENSE_ACTION_INDEX == 1


def test_dr_new_methodology_1_sensing_gate_fires_exactly_on_named_index():
    env = RiskAwareActiveSensingGridEnvironment(Stage23EnvironmentConfig())
    env.reset(seed=5)
    observations, _, _, _, infos = env.step(
        {
            "agent_0": {
                SENSING_ACTION_FIELD: STAGE23_SENSE_ACTION_INDEX,
                MOVEMENT_ACTION_FIELD: 0,
            },
            "agent_1": {SENSING_ACTION_FIELD: 0, MOVEMENT_ACTION_FIELD: 0},
        }
    )
    # One step with sensing on/off: previous_sensed flips exactly for the
    # agent whose sensing_action equaled STAGE23_SENSE_ACTION_INDEX.
    assert observations["agent_0"]["actor_visible"]["previous_sensed"] is True
    assert observations["agent_1"]["actor_visible"]["previous_sensed"] is False


def test_dr_new_methodology_1_default_core_config_sensing_factor_is_binary():
    assert default_stage23_core_config().sensing_action_count == 2


# ===================== SECTION S25U: Stage 25 reviewed update path =====================
# Behavioral pins for the DR-D4 cluster reviewed final-training update
# (stage25_update / stage25_collection / stage25_driver), the Shift-1 VERIFIER
# fixes V-1..V-6, and the Shift-2 adversarial CORRECTOR fixes A-1..A-14. Every
# DR constant is EXECUTED, not merely stated. Reuses the Section F helpers
# (_s25_batch_fields / _s25_zeros_batch / _s25_regression_batch /
# _s25_reference_gae / the s25_model fixture).

import copy as _s25u_copy
import dataclasses

from raas_marl.final_training.stage25_collection import (
    STAGE25_MAX_EPISODES_PER_ROUND as _S25U_MAX_EPS,
    STAGE25_MIN_EPISODES_PER_ROUND as _S25U_MIN_EPS,
    Stage25CollectionConfig as _S25UCollectionConfig,
    collect_stage25_training_rollout as _s25u_collect,
)
from raas_marl.final_training.stage25_driver import (
    Stage25RunConfig as _S25URunConfig,
    run_stage25_training as _s25u_run_training,
    stage25_boundary_flags as _s25u_boundary_flags,
    _config_fingerprint as _s25u_config_fingerprint,
    _reconcile_resumed_log as _s25u_reconcile,
)
from raas_marl.final_training.stage25_update import (
    PIController as _S25UPIController,
    PIControllerStep as _S25UPIControllerStep,
    STAGE25_EPOCH_CAP as _S25U_EPOCH_CAP,
    Stage25UpdateConfig as _S25UUpdateConfig,
    Stage25UpdateResult as _S25UUpdateResult,
    episodic_team_hazard_cost as _s25u_episodic_cost,
    stage25_ppo_lagrangian_update as _s25u_update,
)
from raas_marl.mappo_lagrangian.lagrange import (
    LagrangeMultiplier as _S25ULagrangeMultiplier,
)
from raas_marl.mappo_lagrangian.update import (
    default_stage22_update_config as _s25u_default_dev_config,
    stage22_ppo_lagrangian_update as _s25u_dev_update,
)


def _s25u_model_and_batch(seed=71, eps=4, steps=6):
    torch.manual_seed(seed)
    model = RecurrentMAPPOActorCritic(default_stage23_core_config())
    collected = _s25u_collect(
        model,
        _S25UCollectionConfig(max_episodes=eps, steps_per_episode=steps, seed=seed),
    )
    return model, collected


def _s25u_masked_mean(values, valid_mask):
    return float(values[valid_mask].mean().item())


# ---------- Stage25UpdateConfig: DR constants + A-3 sanitize-store ----------


def test_s25u_from_decision_records_binds_every_dr_constant():
    cfg = _S25UUpdateConfig.from_decision_records()
    # DR-D4 algorithm
    assert cfg.algorithm.discount_factor == 0.99
    assert cfg.algorithm.gae_lambda == 0.95
    assert cfg.algorithm.ppo_clip_range == 0.2
    # DR-D4 optimizer / epochs / grad-norm
    assert cfg.learning_rate == 3e-4
    assert cfg.adam_epsilon == 1e-5
    assert cfg.max_update_epochs == 10 == _S25U_EPOCH_CAP
    assert cfg.max_grad_norm == 10.0
    assert cfg.standardize_advantages is True
    # DR-D9 KL guard
    assert cfg.kl_early_stop is True
    assert cfg.target_kl == 0.01
    assert cfg.kl_stop_margin == 1.5
    # DR-D3 dual timing + PI gains, DR-D2 estimator + budget
    assert cfg.dual_step_timing == "pre_epochs"
    assert cfg.cost_estimator == "episodic_team_mean"
    assert cfg.proportional_gain == 0.25
    assert cfg.integral_gain == 0.05
    assert cfg.initial_integral == 0.1
    assert cfg.hazard_budget == 0.5
    # DR-D1 fixed-weight sensing shaping; DR-D4 entropy coeffs
    assert cfg.losses.sensing_cost_coefficient == 0.5
    assert cfg.losses.sensing_entropy_coefficient == 0.01
    assert cfg.losses.movement_entropy_coefficient == 0.01
    assert cfg.losses.value_loss_coefficient == 0.5


def test_s25u_config_a3_sanitizes_hostile_float_subclasses():
    cfg = _S25UUpdateConfig(
        algorithm=AlgorithmConfig(
            discount_factor=0.99, gae_lambda=0.95, ppo_clip_range=0.2
        ),
        losses=_gb_losses(),
        learning_rate=_DrHostileFloat(3e-4),
        adam_epsilon=_DrHostileFloat(1e-5),
        max_update_epochs=_DrHostileInt(10),
        max_grad_norm=_DrHostileFloat(10.0),
        standardize_advantages=True,
        kl_early_stop=True,
        target_kl=_DrHostileFloat(0.01),
        kl_stop_margin=_DrHostileFloat(1.5),
        dual_step_timing="pre_epochs",
        cost_estimator="episodic_team_mean",
        proportional_gain=_DrHostileFloat(0.25),
        integral_gain=_DrHostileFloat(0.05),
        initial_integral=_DrHostileFloat(0.1),
        hazard_budget=_DrHostileFloat(0.5),
    )
    for name in (
        "learning_rate",
        "adam_epsilon",
        "target_kl",
        "kl_stop_margin",
        "max_grad_norm",
        "proportional_gain",
        "integral_gain",
        "initial_integral",
        "hazard_budget",
    ):
        assert type(getattr(cfg, name)) is float, f"{name} not sanitized (A-3)"
    assert type(cfg.max_update_epochs) is int


def test_s25u_config_max_grad_norm_none_preserved():
    cfg = _S25UUpdateConfig.from_decision_records()
    replaced = dataclasses.replace(cfg, max_grad_norm=None)
    assert replaced.max_grad_norm is None


def test_s25u_config_epoch_cap_boundary():
    base = _S25UUpdateConfig.from_decision_records()
    assert dataclasses.replace(base, max_update_epochs=10).max_update_epochs == 10
    with pytest.raises(ValueError, match="max_update_epochs must be <="):
        dataclasses.replace(base, max_update_epochs=11)


def test_s25u_config_post_epochs_requires_zero_kp():
    bridge = _S25UUpdateConfig.dev_regression_bridge()
    assert bridge.dual_step_timing == "post_epochs"
    assert bridge.proportional_gain == 0.0
    with pytest.raises(ValueError, match="proportional_gain == 0.0"):
        dataclasses.replace(bridge, proportional_gain=0.25)


# ---------- PIController: DR-D3 rule, equivalence, A-2, A-11 ----------


def test_s25u_pi_controller_projected_rule_hand_case():
    controller = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=0.1, budget=0.5
    )
    step = controller.update(observed_episodic_cost=1.0)
    # e = 1.0 - 0.5 = 0.5 ; I_1 = max(0, 0.1 + 0.05*0.5) = 0.125 ;
    # lambda = max(0, 0.25*0.5 + 0.125) = 0.25
    assert step.error == pytest.approx(0.5)
    assert step.integral == pytest.approx(0.125)
    assert step.lambda_applied == pytest.approx(0.25)
    assert isinstance(step, _S25UPIControllerStep)


def test_s25u_pi_controller_equivalence_kp0_ki_eta():
    """DR-D3 T0(a): K_P=0, K_I=eta reproduces the projected integral rule."""

    eta, i0, budget = 0.05, 0.1, 0.5
    controller = _S25UPIController(
        proportional_gain=0.0, integral_gain=eta, integral=i0, budget=budget
    )
    multiplier = _S25ULagrangeMultiplier(value=i0, learning_rate=eta)
    for observed in (0.8, 0.3, 0.6, 0.0, 1.2, 0.5):
        step = controller.update(observed_episodic_cost=observed)
        multiplier = multiplier.update(observed_cost=observed, cost_budget=budget)
        assert step.lambda_applied == pytest.approx(multiplier.value, abs=1e-12)
        controller = step.controller


def test_s25u_pi_controller_a2_negative_overflow_raises():
    controller = _S25UPIController(
        proportional_gain=1e308, integral_gain=1e-9, integral=0.0, budget=1e308
    )
    with pytest.raises(ValueError, match="finite"):
        controller.update(observed_episodic_cost=0.0)


def test_s25u_pi_controller_a11_keyword_only_construction():
    with pytest.raises(TypeError):
        _S25UPIController(0.25, 0.05, 0.1, 0.5)


def test_s25u_pi_controller_update_is_immutable():
    controller = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=0.1, budget=0.5
    )
    step = controller.update(observed_episodic_cost=1.0)
    assert controller.integral == 0.1  # original untouched
    assert step.controller is not controller
    with pytest.raises(Exception):
        object.__setattr__  # frozen; setattr on a field raises
        controller.integral = 0.2  # type: ignore[misc]


def test_s25u_pi_controller_rejects_negative_observed_cost():
    controller = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=0.1, budget=0.5
    )
    with pytest.raises(ValueError):
        controller.update(observed_episodic_cost=-0.1)


# ---------- DR-D3 AMENDMENT (SHIFT 16): anti-windup integral cap ----------


def test_s25u_pi_controller_integral_cap_off_default_byte_identical():
    """AM-4(b/c): integral_cap=None (OFF) reproduces the DR-D3 recursion
    bit-for-bit over a sequence â€” the byte-repro guarantee."""
    off = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=0.1, budget=0.5
    )
    explicit_none = _S25UPIController(
        proportional_gain=0.25,
        integral_gain=0.05,
        integral=0.1,
        budget=0.5,
        integral_cap=None,
    )
    assert off.integral_cap is None and explicit_none.integral_cap is None
    for observed in (0.8, 1.2, 0.0, 0.6, 1.0, 0.3):
        s_off = off.update(observed_episodic_cost=observed)
        s_none = explicit_none.update(observed_episodic_cost=observed)
        assert s_off.integral == s_none.integral
        assert s_off.lambda_applied == s_none.lambda_applied
        off, explicit_none = s_off.controller, s_none.controller


def test_s25u_pi_controller_integral_cap_clamps_hand_case():
    """AM-4(a): I_k = min(cap, max(0, I_{k-1}+K_I*e_k)) when unclamped exceeds cap."""
    controller = _S25UPIController(
        proportional_gain=0.25,
        integral_gain=0.05,
        integral=25.0,
        budget=0.5,
        integral_cap=20.0,
    )
    # e = 1.0 - 0.5 = 0.5 ; unclamped I = 25.0 + 0.05*0.5 = 25.025 ; clamp -> 20.0
    step = controller.update(observed_episodic_cost=1.0)
    assert step.integral == pytest.approx(20.0)
    # lambda uses the clamped integral: max(0, 0.25*0.5 + 20.0) = 20.125
    assert step.lambda_applied == pytest.approx(20.125)
    # below the cap: no clamp (byte-identical to the uncapped step)
    below = _S25UPIController(
        proportional_gain=0.25,
        integral_gain=0.05,
        integral=0.1,
        budget=0.5,
        integral_cap=20.0,
    )
    below_step = below.update(observed_episodic_cost=1.0)
    assert below_step.integral == pytest.approx(0.125)


def test_s25u_pi_controller_integral_cap_persists_across_rounds():
    """AM-2: the anti-windup ceiling MUST carry onto the successor controller
    across >=2 successive update() calls (else it silently resets to None)."""
    controller = _S25UPIController(
        proportional_gain=0.25,
        integral_gain=0.05,
        integral=25.0,
        budget=0.5,
        integral_cap=20.0,
    )
    s1 = controller.update(observed_episodic_cost=1.0)
    s2 = s1.controller.update(observed_episodic_cost=1.0)
    s3 = s2.controller.update(observed_episodic_cost=1.0)
    assert s1.controller.integral_cap == 20.0
    assert s2.controller.integral_cap == 20.0
    assert s3.controller.integral_cap == 20.0
    # the integral stays bounded by the cap every round (the MECH-1 invariant)
    for step in (s1, s2, s3):
        assert step.integral <= 20.0 + 1e-9


def test_s25u_pi_controller_integral_cap_rejects_nonpositive():
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises((ValueError, TypeError)):
            _S25UPIController(
                proportional_gain=0.25,
                integral_gain=0.05,
                integral=0.1,
                budget=0.5,
                integral_cap=bad,
            )


def test_s25u_update_config_lambda_integral_cap_off_default_and_threads():
    """AM-4(c): the DR-sourced configs leave the cap OFF (None) and reproduce
    the current controller; a set cap threads through make_controller()."""
    dr = _S25UUpdateConfig.from_decision_records()
    bridge = _S25UUpdateConfig.dev_regression_bridge()
    assert dr.lambda_integral_cap is None
    assert bridge.lambda_integral_cap is None
    assert dr.make_controller().integral_cap is None
    capped = dataclasses.replace(dr, lambda_integral_cap=20.0)
    assert capped.make_controller().integral_cap == 20.0


def test_s25u_update_a10_mismatched_integral_cap_rejected():
    """AM-3: a controller whose integral_cap disagrees with the config is
    rejected before training (None != cap and cap != None both caught)."""
    model, collected = _s25u_model_and_batch(seed=76, eps=2, steps=6)
    cfg = _S25UUpdateConfig.from_decision_records()  # lambda_integral_cap None
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon
    )
    capped_controller = _S25UPIController(
        proportional_gain=cfg.proportional_gain,
        integral_gain=cfg.integral_gain,
        integral=cfg.initial_integral,
        budget=cfg.hazard_budget,
        integral_cap=20.0,
    )
    with pytest.raises(ValueError, match="dual constants"):
        _s25u_update(
            model, collected.batch, cfg, optimizer=optimizer,
            controller=capped_controller, episode_count=2,
        )
    # Reverse direction (the old-checkpoint-resume shape): config HAS a cap but
    # the controller is uncapped (None). Also rejected by the coherence guard.
    capped_cfg = dataclasses.replace(cfg, lambda_integral_cap=20.0)
    none_controller = capped_cfg.make_controller().__class__(
        proportional_gain=capped_cfg.proportional_gain,
        integral_gain=capped_cfg.integral_gain,
        integral=capped_cfg.initial_integral,
        budget=capped_cfg.hazard_budget,
        integral_cap=None,
    )
    with pytest.raises(ValueError, match="dual constants"):
        _s25u_update(
            model, collected.batch, capped_cfg, optimizer=optimizer,
            controller=none_controller, episode_count=2,
        )


# ---------- Episodic estimator: DR-D2 anchor, A-1, A-4 ----------


def test_s25u_episodic_estimator_dr_d2_anchor():
    fields = _s25_batch_fields(batch_size=32, time_steps=1)
    hazard = torch.zeros(32, 1)
    hazard[0, 0] = 1.0
    fields["hazard_cost"] = hazard
    batch = RolloutBatch(**fields)
    assert _s25u_episodic_cost(batch, episode_count=16) == pytest.approx(0.0625)


def test_s25u_episodic_estimator_a4_rejects_non_rolloutbatch():
    with pytest.raises(TypeError, match="RolloutBatch"):
        _s25u_episodic_cost(object(), episode_count=2)


def test_s25u_episodic_estimator_rejects_non_dividing_count():
    batch = _s25_zeros_batch()  # 2 rows
    with pytest.raises(ValueError, match="must divide"):
        _s25u_episodic_cost(batch, episode_count=3)


def test_s25u_update_a1_wrong_but_dividing_episode_count_rejected():
    model, collected = _s25u_model_and_batch(seed=72, eps=2, steps=6)
    cfg = _S25UUpdateConfig.from_decision_records()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon
    )
    # rows = 2 episodes * 2 agents = 4; passing 4 (= rows) divides but is wrong.
    with pytest.raises(ValueError, match="agent count"):
        _s25u_update(
            model,
            collected.batch,
            cfg,
            optimizer=optimizer,
            controller=cfg.make_controller(),
            episode_count=4,
        )


def test_s25u_update_a1_correct_episode_count_and_none_derivation_agree():
    cfg = _S25UUpdateConfig.from_decision_records()
    m1, c1 = _s25u_model_and_batch(seed=73, eps=2, steps=6)
    m2 = _s25u_copy.deepcopy(m1)
    o1 = torch.optim.Adam(m1.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    o2 = torch.optim.Adam(m2.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    r_explicit = _s25u_update(
        m1, c1.batch, cfg, optimizer=o1,
        controller=cfg.make_controller(), episode_count=2,
    )
    r_derived = _s25u_update(
        m2, c1.batch, cfg, optimizer=o2,
        controller=cfg.make_controller(), episode_count=None,
    )
    assert r_explicit.summary["episode_count"] == 2
    assert r_derived.summary["episode_count"] == 2
    assert r_explicit.summary["observed_episodic_cost"] == pytest.approx(
        r_derived.summary["observed_episodic_cost"]
    )


# ---------- Update path: A-10 controller cross-check + V-1 fields ----------


def test_s25u_update_a10_mismatched_controller_rejected():
    model, collected = _s25u_model_and_batch(seed=74, eps=2, steps=6)
    cfg = _S25UUpdateConfig.from_decision_records()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon
    )
    bad = _S25UPIController(
        proportional_gain=0.9, integral_gain=0.05, integral=0.1, budget=0.5
    )
    with pytest.raises(ValueError, match="dual constants"):
        _s25u_update(
            model, collected.batch, cfg, optimizer=optimizer,
            controller=bad, episode_count=2,
        )


def test_s25u_update_v1_grad_norm_fields_present_legacy_absent():
    model, collected = _s25u_model_and_batch(seed=75, eps=2, steps=6)
    cfg = _S25UUpdateConfig.from_decision_records()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon
    )
    result = _s25u_update(
        model, collected.batch, cfg, optimizer=optimizer,
        controller=cfg.make_controller(), episode_count=2,
    )
    summary = result.summary
    # V-1: DR-D14 trigger graded on GRADIENT-NORM clip activation.
    assert "grad_norm_clip_activation_fraction" in summary
    assert "grad_norm_pre_clip_per_epoch" in summary
    assert len(summary["grad_norm_pre_clip_per_epoch"]) == summary["epochs_used"]
    # Honest name for the PPO-ratio measurement; the misleading legacy name gone.
    assert "ratio_clip_fraction" in summary
    assert "clip_activation_fraction" not in summary
    assert 0.0 <= summary["grad_norm_clip_activation_fraction"] <= 1.0


# ---------- Dev-bridge bit-identity (DR-D4 T0(a)) ----------


def test_s25u_dev_regression_bridge_bit_identity():
    torch.manual_seed(88)
    model_a = RecurrentMAPPOActorCritic(default_stage23_core_config())
    model_b = _s25u_copy.deepcopy(model_a)
    batch = _s25u_collect(
        model_a,
        _S25UCollectionConfig(max_episodes=4, steps_per_episode=8, seed=890),
    ).batch
    bridge = _S25UUpdateConfig.dev_regression_bridge()
    opt_a = torch.optim.Adam(
        model_a.parameters(), lr=bridge.learning_rate, eps=bridge.adam_epsilon
    )
    opt_b = torch.optim.Adam(model_b.parameters(), lr=0.01)
    res_a = _s25u_update(
        model_a, batch, bridge, optimizer=opt_a,
        controller=bridge.make_controller(), episode_count=4,
    )
    res_b = _s25u_dev_update(
        model_b, batch, _s25u_default_dev_config(), optimizer=opt_b,
        lagrange_multiplier=_S25ULagrangeMultiplier(value=0.1, learning_rate=0.2),
    )
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa.detach(), pb.detach())
    for key in ("policy_loss", "total_loss", "parameter_delta_l1"):
        assert res_a.summary[key] == pytest.approx(res_b.summary[key], abs=1e-6)


# ---------- GAE-once + raw value targets + k3 KL formula ----------


def test_s25u_update_gae_once_raw_advantage_matches_reference():
    torch.manual_seed(91)
    model = RecurrentMAPPOActorCritic(default_stage23_core_config())
    batch = _s25_regression_batch()
    cfg = _S25UUpdateConfig.from_decision_records()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon
    )
    result = _s25u_update(
        model, batch, cfg, optimizer=optimizer,
        controller=cfg.make_controller(), episode_count=None,
    )
    signal = batch.task_reward - cfg.losses.sensing_cost_coefficient * batch.sensing_cost
    reference = _s25_reference_gae(
        signal, batch.reward_value, batch.next_reward_value,
        batch.terminal, batch.truncation, batch.valid_mask,
        cfg.algorithm.discount_factor, cfg.algorithm.gae_lambda,
    )
    assert result.summary["raw_reward_advantage_mean"] == pytest.approx(
        _s25u_masked_mean(reference, batch.valid_mask), abs=1e-5
    )
    assert result.summary["value_targets_use_raw_advantages"] is True


def test_s25u_update_k3_kl_final_matches_manual_forward():
    torch.manual_seed(92)
    model = RecurrentMAPPOActorCritic(default_stage23_core_config())
    batch = _s25_regression_batch()
    cfg = _S25UUpdateConfig.from_decision_records()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon
    )
    result = _s25u_update(
        model, batch, cfg, optimizer=optimizer,
        controller=cfg.make_controller(), episode_count=None,
    )
    # The model is left in train mode; recompute the k3 estimate of the final
    # post-step policy against the rollout policy over the valid mask.
    with torch.no_grad():
        outputs = model(
            ActorInput(batch.actor_observation, batch.revealed_information),
            CriticInput(batch.central_state),
            sensing_action=batch.sensing_action,
            movement_action=batch.movement_action,
            sample_actions=False,
        )
        policy = outputs["policy"]
        new_joint = policy.sensing_log_probability + policy.movement_log_probability
        log_ratio = new_joint - batch.old_joint_log_probability
        ratio = torch.exp(log_ratio)
        manual = float(
            ((ratio - 1.0) - log_ratio)[batch.valid_mask].mean().item()
        )
    assert result.summary["approx_kl_final"] == pytest.approx(manual, abs=1e-5)
    assert result.summary["approx_kl_final"] >= 0.0


# ---------- Collection config: I-3 layer + V-6 duck-typed contract ----------


@pytest.mark.parametrize(
    "name",
    [
        "risk_gate_train_center",
        "risk_gate_readiness_shifted",
        "risk_gate_heldout_near_gate",
        "not_a_scenario",
    ],
)
def test_s25u_collection_config_rejects_non_catalog_names(name):
    with pytest.raises(ValueError, match="unknown Stage 23-A scenario"):
        _S25UCollectionConfig(scenario_names=(name,))


def test_s25u_collection_config_episode_bounds():
    assert _S25U_MIN_EPS == 2 and _S25U_MAX_EPS == 32
    with pytest.raises(ValueError, match="max_episodes must be between"):
        _S25UCollectionConfig(max_episodes=1)
    with pytest.raises(ValueError, match="max_episodes must be between"):
        _S25UCollectionConfig(max_episodes=33)
    assert _S25UCollectionConfig(max_episodes=2).max_episodes == 2
    assert _S25UCollectionConfig(max_episodes=32).max_episodes == 32


def test_s25u_v6_collection_config_exposes_four_field_episode_contract():
    # V-6: Stage25CollectionConfig is duck-typed into the Stage 23-C episode
    # collector, which reads exactly seed / sample_actions / dtype / device.
    cfg = _S25UCollectionConfig()
    for field in ("seed", "sample_actions", "dtype", "device"):
        assert hasattr(cfg, field), f"missing episode-contract field: {field}"
    assert cfg.dtype == "float32"
    assert cfg.device == "cpu"
    assert isinstance(cfg.sample_actions, bool)


# ---------- Driver: config fingerprint (A-9), reconcile (V-3), resume ----------


def test_s25u_config_fingerprint_a9_ignores_result_parent(tmp_path):
    common = dict(
        run_id="fp_probe", seed=151, update_rounds=4, episodes_per_round=2,
        steps_per_episode=6, probe_every=0, state_save_every=2, tier="T0",
    )
    other = tmp_path / "elsewhere"
    other.mkdir()
    cfg_str = _S25URunConfig(result_parent=str(tmp_path), **common)
    cfg_path = _S25URunConfig(result_parent=tmp_path, **common)
    cfg_other = _S25URunConfig(result_parent=other, **common)
    update_cfg = _S25UUpdateConfig.from_decision_records()
    fp_str = _s25u_config_fingerprint(cfg_str, update_cfg)
    fp_path = _s25u_config_fingerprint(cfg_path, update_cfg)
    fp_other = _s25u_config_fingerprint(cfg_other, update_cfg)
    assert fp_str == fp_path == fp_other  # result_parent excluded


def test_s25u_config_fingerprint_detects_training_config_change(tmp_path):
    base = _S25URunConfig(
        run_id="fp_probe2", seed=151, update_rounds=4, episodes_per_round=2,
        steps_per_episode=6, probe_every=0, state_save_every=2, tier="T0",
        result_parent=tmp_path,
    )
    changed = dataclasses.replace(base, episodes_per_round=3)
    update_cfg = _S25UUpdateConfig.from_decision_records()
    assert _s25u_config_fingerprint(base, update_cfg) != _s25u_config_fingerprint(
        changed, update_cfg
    )


def test_s25u_reconcile_v3_drops_future_rounds_and_tolerates_partial_final(
    tmp_path,
):
    log_path = tmp_path / "training_curve.jsonl"
    lines = [
        json.dumps({"round_index": 0, "x": 1}),
        json.dumps({"round_index": 1, "x": 2}),
        json.dumps({"round_index": 2, "x": 3}),
        '{"round_index": 3, "trunc',  # partial final line (mid-write crash)
    ]
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    dropped = _s25u_reconcile(log_path, start_round=2)
    kept = [json.loads(x) for x in log_path.read_text().strip().split("\n")]
    assert [r["round_index"] for r in kept] == [0, 1]
    assert dropped == 2  # round 2 record + the partial final line


def test_s25u_reconcile_v3_rejects_malformed_non_final_line(tmp_path):
    log_path = tmp_path / "episode_log.jsonl"
    log_path.write_text(
        "not json\n" + json.dumps({"round_index": 0}) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="malformed non-final record"):
        _s25u_reconcile(log_path, start_round=0)


@pytest.fixture(scope="module")
def s25u_resumable_run(tmp_path_factory):
    parent = tmp_path_factory.mktemp("s25u_runs")
    config = _S25URunConfig(
        run_id="s25u_e2e", seed=151, update_rounds=4, episodes_per_round=2,
        steps_per_episode=6, probe_every=2, probe_seed=9001,
        state_save_every=2, heartbeat_every=2, tier="T0", result_parent=parent,
    )
    run_dir = _s25u_run_training(config)
    return run_dir, parent, config


def test_s25u_driver_e2e_completes_with_wellformed_artifacts(s25u_resumable_run):
    run_dir, _, _ = s25u_resumable_run
    manifest = json.loads((run_dir / "RUN_MANIFEST.json").read_text())
    assert manifest["status"] == "complete"
    assert isinstance(manifest["config_sha256"], str) and len(
        manifest["config_sha256"]
    ) == 64
    # V-1 curve fields present on every training-curve record.
    for line in (run_dir / "training_curve.jsonl").read_text().strip().split("\n"):
        rec = json.loads(line)
        assert "grad_norm_clip_activation_fraction" in rec
        assert "ratio_clip_fraction" in rec
    # Truthful boundary flags: claim status never advances.
    assert manifest["claim_status"] == "not tested / not supported"
    assert manifest["final_evaluation_run"] is False


def test_s25u_driver_state_saved_flags_flip_only_after_first_save(
    s25u_resumable_run,
):
    run_dir, _, config = s25u_resumable_run
    curves = [
        json.loads(x)
        for x in (run_dir / "training_curve.jsonl").read_text().strip().split("\n")
    ]
    # state_save_every=2 -> first save after round index 1 (completed_rounds 2);
    # round-0 record predates any saved state, later records follow one.
    assert curves[0]["checkpoint_created"] is False
    assert curves[0]["serialized_model_artifact_created"] is False
    assert curves[-1]["checkpoint_created"] is True
    assert curves[-1]["optimizer_state_saved"] is True


def test_s25u_driver_v4_mismatched_config_resume_rejected(tmp_path):
    parent = tmp_path / "runs"
    parent.mkdir()
    config = _S25URunConfig(
        run_id="s25u_v4", seed=151, update_rounds=4, episodes_per_round=2,
        steps_per_episode=6, probe_every=0, state_save_every=2, tier="T0",
        result_parent=parent,
    )
    run_dir = _s25u_run_training(config)
    # Roll back to the round-2 saved state so a resume is possible.
    (run_dir / "state_round_000004.pt").unlink()
    lines = (run_dir / "state_log.jsonl").read_text().strip().split("\n")
    (run_dir / "state_log.jsonl").write_text(lines[0] + "\n", encoding="utf-8")
    bad = dataclasses.replace(config, episodes_per_round=3)
    with pytest.raises(ValueError, match="does not match the resume"):
        _s25u_run_training(bad, resume=True)


def test_s25u_driver_a7_crash_marks_manifest_with_hostile_type_name(
    tmp_path, monkeypatch
):
    import raas_marl.final_training.stage25_driver as _s25u_driver_mod

    class _UnsafeMeta(type):
        @property
        def __name__(cls):  # noqa: N805
            return "final evaluation error"

    class _BoomError(Exception, metaclass=_UnsafeMeta):
        pass

    def _boom(*args, **kwargs):
        raise _BoomError("boom")

    monkeypatch.setattr(
        _s25u_driver_mod, "collect_stage25_training_rollout", _boom
    )
    config = _S25URunConfig(
        run_id="s25u_crash", seed=161, update_rounds=2, episodes_per_round=2,
        steps_per_episode=6, probe_every=0, state_save_every=2, tier="T0",
        result_parent=tmp_path,
    )
    with pytest.raises(_BoomError):
        _s25u_run_training(config)
    manifest = json.loads(
        (tmp_path / "s25u_crash" / "RUN_MANIFEST.json").read_text()
    )
    assert manifest["status"] == "crashed"
    assert manifest["kill_reason"] == "kill reason withheld by boundary scanner"


def test_s25u_driver_a14_seed_stored_as_base_int():
    config = _S25URunConfig(
        run_id="s25u_seed", seed=True if False else 151, update_rounds=4,
        episodes_per_round=2, tier="T0",
    )
    assert type(config.seed) is int
    assert dataclasses.asdict(config)["seed"] == 151


# ===================== SECTION DRC4: DR-C4 recovery levers =====================
# Behavioral pins for the DR-C4 (COUNTERSIGNED WITH AMENDMENT) three OFF-by-default
# levers: (1) potential-based BFS goal-progress shaping, (2) constraint warmup,
# (3) std-guard / per-factor entropy floors. Reuses the Section S25U helpers.

from raas_marl.final_training.stage25_update import (
    potential_shaping_term as _drc4_shaping_term,
    _entropy_floor_gate as _drc4_entropy_gate,
)
from raas_marl.final_training.stage25_collection import (
    _bfs_goal_distance_field as _drc4_distance_field,
    _episode_potentials as _drc4_episode_potentials,
    _merge_potentials as _drc4_merge_potentials,
    _recover_cell as _drc4_recover_cell,
)
from raas_marl.environments.active_sensing.grid_environment import (
    stage23_scenario_catalog as _drc4_catalog,
)


# ---------- config validation ----------


def test_drc4_update_config_levers_default_off():
    for cfg in (
        _S25UUpdateConfig.from_decision_records(),
        _S25UUpdateConfig.dev_regression_bridge(),
    ):
        assert cfg.constraint_warmup_rounds == 0
        assert cfg.advantage_std_guard_threshold == 0.0
        assert cfg.movement_entropy_floor is None
        assert cfg.sensing_entropy_floor is None


def test_drc4_update_config_warmup_cap_am3():
    base = _S25UUpdateConfig.from_decision_records()
    assert dataclasses.replace(base, constraint_warmup_rounds=500)
    with pytest.raises(ValueError, match="constraint_warmup_rounds must be <="):
        dataclasses.replace(base, constraint_warmup_rounds=501)


def test_drc4_update_config_warmup_requires_pre_epochs():
    bridge = _S25UUpdateConfig.dev_regression_bridge()  # post_epochs
    with pytest.raises(ValueError, match="requires pre_epochs"):
        dataclasses.replace(bridge, constraint_warmup_rounds=50)


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_drc4_update_config_rejects_bad_std_guard(bad):
    base = _S25UUpdateConfig.from_decision_records()
    with pytest.raises((ValueError, TypeError)):
        dataclasses.replace(base, advantage_std_guard_threshold=bad)


@pytest.mark.parametrize("bad", [0.0, -0.5, float("inf")])
def test_drc4_update_config_rejects_bad_entropy_floor(bad):
    base = _S25UUpdateConfig.from_decision_records()
    with pytest.raises((ValueError, TypeError)):
        dataclasses.replace(base, movement_entropy_floor=bad)
    with pytest.raises((ValueError, TypeError)):
        dataclasses.replace(base, sensing_entropy_floor=bad)


def test_drc4_update_config_warmup_rejects_bool_and_negative():
    base = _S25UUpdateConfig.from_decision_records()
    with pytest.raises((ValueError, TypeError)):
        dataclasses.replace(base, constraint_warmup_rounds=-1)
    with pytest.raises((ValueError, TypeError)):
        dataclasses.replace(base, constraint_warmup_rounds=True)


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf")])
def test_drc4_collection_config_rejects_bad_weight(bad):
    with pytest.raises((ValueError, TypeError)):
        _S25UCollectionConfig(task_progress_potential_weight=bad)


@pytest.mark.parametrize("warmup", [10, 11])
def test_drc4_run_config_warmup_must_be_strictly_less_than_rounds(warmup):
    # AM-3: W == update_rounds would leave the whole run unconstrained, so the
    # guard rejects W >= update_rounds (not just W > update_rounds).
    with pytest.raises(ValueError, match="strictly less than"):
        _S25URunConfig(
            run_id="drc4rc", seed=151, update_rounds=10,
            constraint_warmup_rounds=warmup, tier="T0",
        )
    # W = update_rounds - 1 is accepted (one constrained tail round).
    assert _S25URunConfig(
        run_id="drc4rc", seed=151, update_rounds=10,
        constraint_warmup_rounds=9, tier="T0",
    )


def test_drc4_update_rejects_malformed_shaping_potentials():
    torch.manual_seed(71)
    model = RecurrentMAPPOActorCritic(default_stage23_core_config())
    collected = _s25u_collect(
        model, _S25UCollectionConfig(max_episodes=4, steps_per_episode=6, seed=71)
    )
    cfg = _S25UUpdateConfig.from_decision_records()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)

    def _run(potentials):
        return _s25u_update(
            model, collected.batch, cfg, optimizer=opt,
            controller=cfg.make_controller(), episode_count=4,
            reward_shaping_potentials=potentials,
        )

    # 3-tuple fails the arity guard; a 2-tuple of non-tensors fails the
    # entries guard; a shape mismatch fails the shape guard.
    with pytest.raises(TypeError, match="2-tuple of torch.Tensors"):
        _run((collected.batch.task_reward,) * 3)
    with pytest.raises(TypeError, match="entries must be torch.Tensors"):
        _run((0.0, 0.0))
    with pytest.raises(TypeError, match="entries must be torch.Tensors"):
        _run((collected.batch.task_reward, 0.0))
    with pytest.raises(ValueError, match="match the reward-signal shape"):
        _run((torch.zeros((2, 3)), torch.zeros((2, 3))))


# ---------- potential_shaping_term + telescoping invariance (T0 b) ----------


def test_drc4_shaping_term_formula_and_guards():
    p = torch.tensor([[-0.2, -0.1]])
    pn = torch.tensor([[-0.1, 0.0]])
    out = _drc4_shaping_term(p, pn, 0.99)
    assert torch.allclose(out, 0.99 * pn - p)
    with pytest.raises(ValueError, match="share a shape"):
        _drc4_shaping_term(p, torch.zeros((1, 3)), 0.99)
    with pytest.raises(TypeError):
        _drc4_shaping_term(p, [0.0], 0.99)


@pytest.mark.parametrize("ending", ["terminal", "truncation"])
@pytest.mark.parametrize("length", [1, 3])
def test_drc4_telescoping_invariance(ending, length):
    gamma = 0.99
    phi = torch.tensor([[-0.2 - 0.05 * t for t in range(length)]])
    phi_next = torch.zeros((1, length))
    if length > 1:
        phi_next[:, : length - 1] = phi[:, 1:]
    boundary = 0.0 if ending == "terminal" else -0.07
    phi_next[0, length - 1] = boundary
    shaping = _drc4_shaping_term(phi, phi_next, gamma)
    disc = torch.tensor([gamma ** t for t in range(length)])
    lhs = float((disc * shaping[0]).sum().item())
    rhs = gamma ** length * boundary - float(phi[0, 0].item())
    assert lhs == pytest.approx(rhs, abs=1e-5)


def test_drc4_bfs_goal_distance_field_risk_gate():
    scen = _drc4_catalog()["risk_gate_hidden_hazard"]
    field = _drc4_distance_field(scen)
    assert field[(2, 4)] == 0          # goal
    assert field[(2, 0)] == 4          # start agent_0
    assert field[(3, 0)] == 5          # start agent_1
    assert field[(2, 2)] == 2          # hazard cell -> potential is hazard-agnostic
    assert field[(2, 3)] == 1


def test_drc4_recover_cell_roundtrip_and_guards():
    assert _drc4_recover_cell(2 / 7, 0 / 7, height=8, width=8) == (2, 0)
    assert _drc4_recover_cell(3 / 7, 4 / 7, height=8, width=8) == (3, 4)
    with pytest.raises(ValueError, match="not integer-valued"):
        _drc4_recover_cell(0.31, 0.0, height=8, width=8)
    with pytest.raises(ValueError, match="outside the grid"):
        _drc4_recover_cell(2.0, 0.0, height=8, width=8)


def test_drc4_episode_potentials_terminal_and_truncation_boundary():
    scen = _drc4_catalog()["risk_gate_hidden_hazard"]  # 8x8
    field = {(2, 0): 4, (2, 1): 3, (2, 2): 2}
    actor_obs = torch.tensor([[[2 / 7, 0 / 7] + [0.0] * 8,
                               [2 / 7, 1 / 7] + [0.0] * 8]])  # [1,2,10], cells (2,0)->(2,1)
    dev = torch.device("cpu")
    ep_term = {"actor_observation": actor_obs,
               "terminal": torch.tensor([[False, True]])}
    phi, phi_next = _drc4_episode_potentials(
        ep_term, scenario=scen, distance_field=field, weight=0.1,
        final_positions={"agent_0": (2, 2)}, device=dev,
    )
    assert torch.allclose(phi, torch.tensor([[-0.4, -0.3]]))
    assert torch.allclose(phi_next, torch.tensor([[-0.3, 0.0]]))  # terminal -> 0
    ep_trunc = {"actor_observation": actor_obs,
                "terminal": torch.tensor([[False, False]])}
    _, phi_next_trunc = _drc4_episode_potentials(
        ep_trunc, scenario=scen, distance_field=field, weight=0.1,
        final_positions={"agent_0": (2, 2)}, device=dev,
    )
    # truncation -> Phi(s_L) = -0.1 * dist((2,2))=2 = -0.2
    assert torch.allclose(phi_next_trunc, torch.tensor([[-0.3, -0.2]]))


def test_drc4_merge_potentials_row_order_and_padding():
    ep0 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])          # length 2
    ep1 = torch.tensor([[5.0], [6.0]])                    # length 1
    merged = _drc4_merge_potentials([ep0, ep1], max_time=2)
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 0.0], [6.0, 0.0]])
    assert torch.equal(merged, expected)


# ---------- collection integration ----------


def test_drc4_collection_off_leaves_potential_none():
    _, collected = _s25u_model_and_batch(seed=71, eps=4, steps=6)
    assert collected.potential is None and collected.potential_next is None
    assert collected.metadata["reward_shaping_active"] is False


def test_drc4_collection_on_shape_and_start_potential():
    torch.manual_seed(71)
    model = RecurrentMAPPOActorCritic(default_stage23_core_config())
    collected = _s25u_collect(
        model,
        _S25UCollectionConfig(max_episodes=4, steps_per_episode=6, seed=71,
                              task_progress_potential_weight=0.05),
    )
    assert collected.potential is not None
    assert collected.potential.shape == collected.batch.task_reward.shape
    assert collected.potential_next.shape == collected.batch.task_reward.shape
    # recovery correctness pin: episode-0 agent_0 starts at (2,0) (dist 4),
    # agent_1 at (3,0) (dist 5). Row 0 = ep0 agent_0, row 1 = ep0 agent_1.
    assert collected.potential[0, 0].item() == pytest.approx(-0.05 * 4)
    assert collected.potential[1, 0].item() == pytest.approx(-0.05 * 5)


# ---------- OFF-default byte identity (T0 a) ----------


def test_drc4_off_default_update_byte_identical():
    torch.manual_seed(88)
    model_a = RecurrentMAPPOActorCritic(default_stage23_core_config())
    model_b = _s25u_copy.deepcopy(model_a)
    batch = _s25u_collect(
        model_a, _S25UCollectionConfig(max_episodes=4, steps_per_episode=8, seed=890)
    ).batch
    cfg = _S25UUpdateConfig.from_decision_records()  # all DR-C4 levers OFF
    opt_a = torch.optim.Adam(model_a.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    res_a = _s25u_update(model_a, batch, cfg, optimizer=opt_a,
                         controller=cfg.make_controller(), episode_count=4)
    # B: inert round_index + explicit None potentials must change nothing.
    res_b = _s25u_update(model_b, batch, cfg, optimizer=opt_b,
                         controller=cfg.make_controller(), episode_count=4,
                         round_index=7, reward_shaping_potentials=None)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa.detach(), pb.detach())
    for key in ("policy_loss", "total_loss", "parameter_delta_l1", "lambda_applied"):
        assert res_a.summary[key] == res_b.summary[key]
    assert res_a.summary["reward_shaping_applied"] is False
    assert res_a.summary["constraint_warmup_active"] is False


# ---------- warmup (T0 c / AM-3) ----------


def test_drc4_warmup_holds_lambda_zero_freezes_controller_and_ends_at_W():
    torch.manual_seed(71)
    model = RecurrentMAPPOActorCritic(default_stage23_core_config())
    collected = _s25u_collect(
        model, _S25UCollectionConfig(max_episodes=4, steps_per_episode=6, seed=71)
    )
    cfg = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(), constraint_warmup_rounds=3
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    ctrl = cfg.make_controller()
    res0 = _s25u_update(model, collected.batch, cfg, optimizer=opt,
                        controller=ctrl, episode_count=4, round_index=0)
    assert res0.lambda_applied == 0.0
    assert res0.summary["constraint_warmup_active"] is True
    assert res0.controller.integral == ctrl.integral  # integral frozen at I_0
    res3 = _s25u_update(model, collected.batch, cfg, optimizer=opt,
                        controller=res0.controller, episode_count=4, round_index=3)
    assert res3.summary["constraint_warmup_active"] is False


def test_drc4_warmup_requires_round_index():
    torch.manual_seed(71)
    model = RecurrentMAPPOActorCritic(default_stage23_core_config())
    collected = _s25u_collect(
        model, _S25UCollectionConfig(max_episodes=4, steps_per_episode=6, seed=71)
    )
    cfg = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(), constraint_warmup_rounds=3
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    with pytest.raises(ValueError, match="round_index is required"):
        _s25u_update(model, collected.batch, cfg, optimizer=opt,
                     controller=cfg.make_controller(), episode_count=4)


def test_drc4_warmup_at_W_matches_bare_controller_update_am3():
    # AM-3: for rounds >= W the DR-D3 controller runs byte-for-byte unchanged.
    torch.manual_seed(71)
    model = RecurrentMAPPOActorCritic(default_stage23_core_config())
    collected = _s25u_collect(
        model, _S25UCollectionConfig(max_episodes=4, steps_per_episode=6, seed=71)
    )
    cfg = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(), constraint_warmup_rounds=2
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    ctrl = cfg.make_controller()
    observed = _s25u_episodic_cost(collected.batch, episode_count=4)
    expected = ctrl.update(observed_episodic_cost=observed)
    res = _s25u_update(model, collected.batch, cfg, optimizer=opt,
                       controller=ctrl, episode_count=4, round_index=2)
    assert res.lambda_applied == pytest.approx(expected.lambda_applied)
    assert res.controller.integral == pytest.approx(expected.controller.integral)
    assert res.summary["dual_error"] == pytest.approx(expected.error)


# ---------- std-guard (T0 d) + entropy floor gate ----------


def test_drc4_std_guard_mean_centers_small_std():
    vals = torch.tensor([[3.0, 3.0, 3.0, 3.0, 3.001]])
    mask = torch.ones((1, 5), dtype=torch.bool)
    guarded = _normalise_advantage(vals, mask, std_guard_threshold=0.01)
    assert torch.allclose(guarded, vals - vals.mean())
    unguarded = _normalise_advantage(vals, mask, std_guard_threshold=0.0)
    assert unguarded.abs().max() > guarded.abs().max() * 50
    # default threshold reproduces the historical std<=0 behavior on constants
    const = torch.full((2, 4), 5.0)
    cmask = torch.ones((2, 4), dtype=torch.bool)
    assert torch.allclose(_normalise_advantage(const, cmask), torch.zeros_like(const))


def test_drc4_entropy_floor_gate():
    ent = torch.tensor(0.8)
    assert _drc4_entropy_gate(ent, None) == 1.0        # OFF -> always applied
    assert _drc4_entropy_gate(ent, 1.0) == 1.0         # below floor -> applied
    assert _drc4_entropy_gate(ent, 0.5) == 0.0         # above floor -> withheld


# ---------- AM-2 no-leak (graded fields stay raw under shaping) ----------


def test_drc4_shaping_absent_from_graded_episode_records():
    torch.manual_seed(71)
    model = RecurrentMAPPOActorCritic(default_stage23_core_config())
    off = _s25u_collect(
        model, _S25UCollectionConfig(max_episodes=4, steps_per_episode=6, seed=71)
    )
    torch.manual_seed(71)
    model2 = RecurrentMAPPOActorCritic(default_stage23_core_config())
    on = _s25u_collect(
        model2,
        _S25UCollectionConfig(max_episodes=4, steps_per_episode=6, seed=71,
                              task_progress_potential_weight=0.05),
    )
    # The graded per-episode fields are identical with/without shaping: shaping
    # never touches team_success / hazard_cost_sum / sensing (AM-2).
    for r_off, r_on in zip(off.episode_records, on.episode_records):
        for key in ("team_success", "hazard_cost_sum", "task_reward_sum",
                    "total_sensing_actions", "sensing_rate"):
            assert r_off[key] == r_on[key]


# =================== SECTION DRD1L1B: DR-D1/L1-b sensing levers ===================
# Behavioral pins for the DR-D1/L1-b (COUNTERSIGNED WITH AMENDMENT) pair: lever (1)
# the decision-relevant sensing credit (a FIXED observable state potential, PBRS,
# policy-invariant) and lever (2) the warmup-scoped sensing keep-alive (sensing
# factor only). Both OFF by default. AM-2: the composed Phi_total = Phi_BFS +
# Phi_sense telescoping gate is a HARD STOP. Reuses the Section S25U / DRC4 helpers.

from raas_marl.final_training.stage25_collection import (
    _episode_sensing_potentials as _drd1_sensing_potentials,
)
from raas_marl.final_training.baseline_driver import (
    run_readiness_probe as _drd1_probe,
)
from raas_marl.environments.active_sensing.stage24_diagnostics import (
    _stage24a_public_risk_zone_cells as _drd1_zone,
    fork_hazard_layout_families as _drd1_fork_families,
)


def _drd1_fork_zone(name):
    return frozenset(_drd1_zone(name))


# ---------- config validation + OFF defaults ----------


def test_drd1l1b_configs_default_off():
    for cfg in (_S25UUpdateConfig.from_decision_records(),
                _S25UUpdateConfig.dev_regression_bridge()):
        assert cfg.sensing_keepalive_coefficient == 0.0
        assert cfg.sensing_keepalive_rounds == 0
        assert cfg.sensing_entropy_target is None
    assert _S25UCollectionConfig().sensing_credit_potential_weight == 0.0
    rc = _S25URunConfig(run_id="drd1rc", seed=151, update_rounds=10, tier="T0")
    assert rc.sensing_credit_potential_weight == 0.0
    assert rc.sensing_keepalive_coefficient == 0.0
    assert rc.sensing_keepalive_rounds == 0
    assert rc.sensing_entropy_target is None


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf")])
def test_drd1l1b_collection_config_rejects_bad_credit_weight(bad):
    with pytest.raises((ValueError, TypeError)):
        _S25UCollectionConfig(sensing_credit_potential_weight=bad)


@pytest.mark.parametrize("bad", [-0.01, float("nan"), float("inf")])
def test_drd1l1b_update_config_rejects_bad_keepalive(bad):
    base = _S25UUpdateConfig.from_decision_records()
    with pytest.raises((ValueError, TypeError)):
        dataclasses.replace(base, sensing_keepalive_coefficient=bad)
    with pytest.raises((ValueError, TypeError)):
        dataclasses.replace(base, sensing_entropy_target=bad if bad <= 0 else -1.0)


def test_drd1l1b_update_config_keepalive_rounds_rejects_bool_and_negative():
    base = _S25UUpdateConfig.from_decision_records()
    with pytest.raises((ValueError, TypeError)):
        dataclasses.replace(base, sensing_keepalive_rounds=-1)
    with pytest.raises((ValueError, TypeError)):
        dataclasses.replace(base, sensing_keepalive_rounds=True)


def test_drd1l1b_run_config_keepalive_rounds_must_fit_run():
    with pytest.raises(ValueError, match="must be <= update_rounds"):
        _S25URunConfig(run_id="drd1rc", seed=151, update_rounds=10,
                       sensing_keepalive_rounds=11, tier="T0")
    assert _S25URunConfig(run_id="drd1rc", seed=151, update_rounds=10,
                          sensing_keepalive_rounds=10, tier="T0")


# ---------- Phi_sense construction (observable, in-zone, flip) ----------


def _drd1_scripted_fork_episode(length, ending):
    # agent_0 on risk_fork_train_upper (8x8, zone rows 2-5 cols 3-5). An in-zone
    # previous_sensed 0->1 flip so Phi_sense is non-trivial.
    rows = [(3, 2, 0.0), (3, 3, 1.0), (4, 3, 0.0)][:length]
    actor_obs = torch.tensor(
        [[[r / 7, c / 7, 0.0, 0.0, 0.0, ps, 0.0, 0.0, 0.0, 0.0] for (r, c, ps) in rows]]
    )
    sensing_action = torch.tensor([[1, 0, 1][:length]])
    terminal_flags = [False] * length
    if ending == "terminal":
        terminal_flags[-1] = True
    return {
        "actor_observation": actor_obs,
        "terminal": torch.tensor([terminal_flags]),
        "sensing_action": sensing_action,
    }


def test_drd1l1b_sensing_potential_fires_only_in_zone_and_when_sensed():
    scen = _drc4_catalog()["risk_fork_train_upper"]
    zone = _drd1_fork_zone("risk_fork_train_upper")
    ep = _drd1_scripted_fork_episode(3, "terminal")
    phi, _ = _drd1_sensing_potentials(
        ep, scenario=scen, zone=zone, credit_weight=0.05, zone_weight=0.0,
        final_positions={"agent_0": (4, 4)}, device=torch.device("cpu"),
    )
    # step0 (3,2) out-of-zone -> 0; step1 (3,3) in-zone AND previous_sensed -> w;
    # step2 (4,3) in-zone but previous_sensed=0 -> 0.
    assert torch.allclose(phi, torch.tensor([[0.0, 0.05, 0.0]]))


def test_drd1l1b_m5_safe_zone_identical_across_fork_group():
    # The public risk zone (hence Phi_sense) is layout-agnostic: identical for the
    # two aliased fork training layouts (schema-blind to the hidden hazard).
    assert _drd1_fork_zone("risk_fork_train_upper") == _drd1_fork_zone(
        "risk_fork_train_lower"
    )
    scen_u = _drc4_catalog()["risk_fork_train_upper"]
    scen_l = _drc4_catalog()["risk_fork_train_lower"]
    zone = _drd1_fork_zone("risk_fork_train_upper")
    ep = _drd1_scripted_fork_episode(3, "terminal")
    phi_u, _ = _drd1_sensing_potentials(
        ep, scenario=scen_u, zone=zone, credit_weight=0.05, zone_weight=0.0,
        final_positions={"agent_0": (4, 4)}, device=torch.device("cpu"),
    )
    phi_l, _ = _drd1_sensing_potentials(
        ep, scenario=scen_l, zone=zone, credit_weight=0.05, zone_weight=0.0,
        final_positions={"agent_0": (4, 4)}, device=torch.device("cpu"),
    )
    assert torch.equal(phi_u, phi_l)


# ---------- AM-2: composed Phi_total telescoping (HARD STOP) + boundary ----------


@pytest.mark.parametrize("ending", ["terminal", "truncation"])
@pytest.mark.parametrize("length", [1, 3])
def test_drd1l1b_composed_phi_total_telescoping_AM2(ending, length):
    gamma = 0.99
    w_s = 0.05
    scen = _drc4_catalog()["risk_fork_train_upper"]
    zone = _drd1_fork_zone("risk_fork_train_upper")
    field = _drc4_distance_field(scen)
    ep = _drd1_scripted_fork_episode(length, ending)
    final_positions = {"agent_0": (4, 4)}
    dev = torch.device("cpu")
    phi_s, phi_s_next = _drd1_sensing_potentials(
        ep, scenario=scen, zone=zone, credit_weight=w_s, zone_weight=0.0,
        final_positions=final_positions, device=dev,
    )
    phi_b, phi_b_next = _drc4_episode_potentials(
        ep, scenario=scen, distance_field=field, weight=0.05,
        final_positions=final_positions, device=dev,
    )
    # T0(g): Phi_sense reuses _episode_potentials' shift + terminal(->0) /
    # truncation(->Phi_sense(s_L)) boundary convention verbatim.
    if length > 1:
        assert torch.allclose(phi_s_next[:, : length - 1], phi_s[:, 1:])
    if ending == "terminal":
        expected_boundary = 0.0
    else:
        sensed_last = int(ep["sensing_action"][0, length - 1].item()) == 1
        expected_boundary = w_s if (sensed_last and (4, 4) in zone) else 0.0
    assert phi_s_next[0, length - 1].item() == pytest.approx(expected_boundary)
    # T0(b) HARD STOP: the composed potential actually passed to the shaping
    # channel telescopes to gamma^L*Phi_total(s_L) - Phi_total(s_0).
    phi_total = phi_b + phi_s
    phi_total_next = phi_b_next + phi_s_next
    shaping = _drc4_shaping_term(phi_total, phi_total_next, gamma)
    disc = torch.tensor([gamma ** t for t in range(length)])
    lhs = float((disc * shaping[0]).sum().item())
    rhs = gamma ** length * float(phi_total_next[0, length - 1].item()) - float(
        phi_total[0, 0].item()
    )
    assert lhs == pytest.approx(rhs, abs=4e-8)


# ---------- DR-D1/L1-b AMENDMENT (SHIFT 13): decoupled decision-region ----------
# ---------- positioning potential + AM-1 truncation zone-gate + byte-identity ----


def _drd1_all_cases_episode(ending):
    # agent_0 on risk_fork_train_upper (zone rows 2-5 cols 3-5). Covers every
    # (in/out-of-zone) x (sensed/not) combination INCLUDING out-of-zone-AND-sensed
    # at an interior step (step3) so the zone-first gate (AM-1) is exercised.
    #   step0 (3,2) out,  prev_sensed=0
    #   step1 (3,3) in,   prev_sensed=1
    #   step2 (4,3) in,   prev_sensed=0
    #   step3 (3,2) out,  prev_sensed=1   <- out-of-zone-and-sensed
    rows = [(3, 2, 0.0), (3, 3, 1.0), (4, 3, 0.0), (3, 2, 1.0)]
    actor_obs = torch.tensor(
        [[[r / 7, c / 7, 0.0, 0.0, 0.0, ps, 0.0, 0.0, 0.0, 0.0] for (r, c, ps) in rows]]
    )
    sensing_action = torch.tensor([[1, 0, 1, 1]])  # sensed on the final collected step
    terminal_flags = [False, False, False, ending == "terminal"]
    return {
        "actor_observation": actor_obs,
        "terminal": torch.tensor([terminal_flags]),
        "sensing_action": sensing_action,
    }


def test_drd1l1b_decoupled_positioning_fires_in_zone_regardless_of_sensing():
    # Phi_dz = w_dz * 1[cell in zone] -- decoupled from previous_sensed: it fires on
    # EVERY in-zone step (incl. step2 (4,3) in-zone-but-NOT-sensed, where the
    # prev_sensed credit is 0) and NOWHERE out of zone.
    scen = _drc4_catalog()["risk_fork_train_upper"]
    zone = _drd1_fork_zone("risk_fork_train_upper")
    ep = _drd1_all_cases_episode("truncation")
    phi, phi_next = _drd1_sensing_potentials(
        ep, scenario=scen, zone=zone, credit_weight=0.0, zone_weight=0.05,
        final_positions={"agent_0": (3, 2)}, device=torch.device("cpu"),
    )
    # in-zone at steps 1,2 (regardless of previous_sensed); out at 0,3.
    assert torch.allclose(phi, torch.tensor([[0.0, 0.05, 0.05, 0.0]]))
    # truncation final (3,2) is OUT of zone => Phi_dz(s_L)=0 (AM-1 zone-first gate).
    assert phi_next[0, 3].item() == pytest.approx(0.0)


def test_drd1l1b_off_default_zone_zero_byte_identical_to_prev_sensed_credit():
    # OFF-default (zone_weight=0.0): the composed potential reduces EXACTLY to the
    # SHIFT-10 prev_sensed-only credit -- including the AM-1-critical out-of-zone-
    # AND-sensed case at an interior step (step3) AND the truncation boundary.
    scen = _drc4_catalog()["risk_fork_train_upper"]
    zone = _drd1_fork_zone("risk_fork_train_upper")
    ep = _drd1_all_cases_episode("truncation")
    phi, phi_next = _drd1_sensing_potentials(
        ep, scenario=scen, zone=zone, credit_weight=0.05, zone_weight=0.0,
        final_positions={"agent_0": (3, 2)}, device=torch.device("cpu"),
    )
    # credit fires ONLY at (in-zone AND previous_sensed) = step1; step3 is
    # sensed but OUT of zone => 0 (the case a mis-gated truncation would corrupt).
    assert torch.allclose(phi, torch.tensor([[0.0, 0.05, 0.0, 0.0]]))
    # interior phi_next = phi shifted; truncation final (3,2) OUT of zone even
    # though sensed_last=True => 0 (AM-1).
    assert torch.allclose(phi_next, torch.tensor([[0.05, 0.0, 0.0, 0.0]]))


@pytest.mark.parametrize("mode", [("credit", 0.05, 0.0), ("zone", 0.0, 0.05), ("hybrid", 0.05, 0.05)])
@pytest.mark.parametrize("final_pos", [(4, 4), (3, 2)])  # in-zone, out-of-zone
def test_drd1l1b_AM2_composed_telescopes_truncation_in_and_out_of_zone(mode, final_pos):
    # AM-2 HARD STOP: the composed Phi_total = Phi_BFS + Phi_composed telescopes to
    # gamma^L*Phi_total(s_L) - Phi_total(s_0) on a TRUNCATION ending, exercising a
    # final cell BOTH inside AND OUTSIDE the zone, with zone_weight > 0.
    _, cw, zw = mode
    gamma = 0.99
    scen = _drc4_catalog()["risk_fork_train_upper"]
    zone = _drd1_fork_zone("risk_fork_train_upper")
    field = _drc4_distance_field(scen)
    ep = _drd1_all_cases_episode("truncation")
    length = int(ep["actor_observation"].shape[1])
    fp = {"agent_0": final_pos}
    dev = torch.device("cpu")
    phi_s, phi_s_next = _drd1_sensing_potentials(
        ep, scenario=scen, zone=zone, credit_weight=cw, zone_weight=zw,
        final_positions=fp, device=dev,
    )
    phi_b, phi_b_next = _drc4_episode_potentials(
        ep, scenario=scen, distance_field=field, weight=0.05,
        final_positions=fp, device=dev,
    )
    # truncation Phi_composed(s_L) = 0 iff final cell OUT of zone (AM-1).
    if final_pos in zone:
        assert phi_s_next[0, length - 1].item() > 0.0 or (cw == 0.0 and zw == 0.0)
    else:
        assert phi_s_next[0, length - 1].item() == pytest.approx(0.0)
    phi_total = phi_b + phi_s
    phi_total_next = phi_b_next + phi_s_next
    shaping = _drc4_shaping_term(phi_total, phi_total_next, gamma)
    disc = torch.tensor([gamma ** t for t in range(length)])
    lhs = float((disc * shaping[0]).sum().item())
    rhs = gamma ** length * float(phi_total_next[0, length - 1].item()) - float(
        phi_total[0, 0].item()
    )
    assert lhs == pytest.approx(rhs, abs=4e-8)


def test_drd1l1b_decoupled_positioning_m5_safe_across_fork_group():
    # Phi_dz (positioning) is layout-agnostic: byte-identical for the two aliased
    # fork layouts (never reads the hidden hazard).
    scen_u = _drc4_catalog()["risk_fork_train_upper"]
    scen_l = _drc4_catalog()["risk_fork_train_lower"]
    zone = _drd1_fork_zone("risk_fork_train_upper")
    ep = _drd1_all_cases_episode("truncation")
    fp = {"agent_0": (3, 2)}
    dev = torch.device("cpu")
    phi_u, phi_u_next = _drd1_sensing_potentials(
        ep, scenario=scen_u, zone=zone, credit_weight=0.0, zone_weight=0.05,
        final_positions=fp, device=dev,
    )
    phi_l, phi_l_next = _drd1_sensing_potentials(
        ep, scenario=scen_l, zone=zone, credit_weight=0.0, zone_weight=0.05,
        final_positions=fp, device=dev,
    )
    assert torch.equal(phi_u, phi_l)
    assert torch.equal(phi_u_next, phi_l_next)


def test_drd1l1b_zone_potential_weight_config_validation():
    base = _S25UCollectionConfig()
    assert base.sensing_zone_potential_weight == 0.0
    ok = dataclasses.replace(base, sensing_zone_potential_weight=0.05)
    assert type(ok.sensing_zone_potential_weight) is float
    assert ok.sensing_zone_potential_weight == 0.05
    for bad in (-0.1, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            dataclasses.replace(base, sensing_zone_potential_weight=bad)
    with pytest.raises((TypeError, ValueError)):
        dataclasses.replace(base, sensing_zone_potential_weight="x")


class _ZoneHostile(float):
    def __gt__(self, other):  # try to hijack the > 0.0 activation test
        return True


def test_drd1l1b_zone_potential_weight_sanitize_store():
    cfg = _S25UCollectionConfig(sensing_zone_potential_weight=_ZoneHostile(0.05))
    # sanitize-store: the frozen field holds a base float, not the hostile subclass.
    assert type(cfg.sensing_zone_potential_weight) is float
    assert cfg.sensing_zone_potential_weight == 0.05


# ---------- collection integration ----------


def test_drd1l1b_collection_off_leaves_sensing_potential_none():
    _, collected = _s25u_model_and_batch(seed=71, eps=4, steps=6)
    assert collected.sensing_potential is None
    assert collected.sensing_potential_next is None
    assert collected.metadata["sensing_credit_active"] is False


def test_drd1l1b_collection_on_shape_and_alignment():
    torch.manual_seed(71)
    model = RecurrentMAPPOActorCritic(default_stage23_core_config())
    collected = _s25u_collect(
        model,
        _S25UCollectionConfig(
            max_episodes=4, steps_per_episode=6, seed=71,
            scenario_names=("risk_fork_train_lower", "risk_fork_train_upper"),
            sensing_credit_potential_weight=0.05,
        ),
    )
    assert collected.sensing_potential is not None
    assert collected.sensing_potential.shape == collected.batch.task_reward.shape
    assert collected.sensing_potential_next.shape == collected.batch.task_reward.shape
    # Phi_sense is bounded to [0, w_s] (a 0/1 indicator times the weight).
    assert float(collected.sensing_potential.min().item()) >= 0.0
    assert float(collected.sensing_potential.max().item()) <= 0.05 + 1e-9


# ---------- update integration: OFF-default byte identity + guards ----------


def test_drd1l1b_off_default_update_byte_identical():
    torch.manual_seed(88)
    model_a = RecurrentMAPPOActorCritic(default_stage23_core_config())
    model_b = _s25u_copy.deepcopy(model_a)
    batch = _s25u_collect(
        model_a, _S25UCollectionConfig(max_episodes=4, steps_per_episode=8, seed=890)
    ).batch
    cfg = _S25UUpdateConfig.from_decision_records()  # all levers OFF
    opt_a = torch.optim.Adam(model_a.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    res_a = _s25u_update(model_a, batch, cfg, optimizer=opt_a,
                         controller=cfg.make_controller(), episode_count=4,
                         round_index=0)
    # Explicit None sensing potentials + OFF keep-alive must change nothing.
    res_b = _s25u_update(model_b, batch, cfg, optimizer=opt_b,
                         controller=cfg.make_controller(), episode_count=4,
                         round_index=0, sensing_shaping_potentials=None)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)
    assert res_a.summary["sensing_shaping_applied"] is False
    assert res_b.summary["sensing_shaping_mean"] == 0.0
    assert res_b.summary["sensing_keepalive_active"] is False


def test_drd1l1b_update_rejects_malformed_sensing_shaping_potentials():
    torch.manual_seed(71)
    model = RecurrentMAPPOActorCritic(default_stage23_core_config())
    collected = _s25u_collect(
        model, _S25UCollectionConfig(max_episodes=4, steps_per_episode=6, seed=71)
    )
    cfg = _S25UUpdateConfig.from_decision_records()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)

    def _run(potentials):
        return _s25u_update(
            model, collected.batch, cfg, optimizer=opt,
            controller=cfg.make_controller(), episode_count=4, round_index=0,
            sensing_shaping_potentials=potentials,
        )

    with pytest.raises(TypeError, match="2-tuple of torch.Tensors"):
        _run((collected.batch.task_reward,) * 3)
    with pytest.raises(TypeError, match="entries must be torch.Tensors"):
        _run((0.0, 0.0))
    with pytest.raises(ValueError, match="match the reward-signal shape"):
        _run((torch.zeros((2, 3)), torch.zeros((2, 3))))


# ---------- keep-alive: sensing-factor only, warmup-scoped, released ----------


def test_drd1l1b_keepalive_active_flag_window():
    torch.manual_seed(71)
    model = RecurrentMAPPOActorCritic(default_stage23_core_config())
    collected = _s25u_collect(
        model, _S25UCollectionConfig(max_episodes=4, steps_per_episode=6, seed=71)
    )
    cfg = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(),
        sensing_keepalive_coefficient=0.1, sensing_keepalive_rounds=3,
        sensing_entropy_target=10.0,  # high => always fires while in window
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    in_win = _s25u_update(model, collected.batch, cfg, optimizer=opt,
                          controller=cfg.make_controller(), episode_count=4,
                          round_index=0)
    out_win = _s25u_update(model, collected.batch, cfg, optimizer=opt,
                           controller=cfg.make_controller(), episode_count=4,
                           round_index=5)
    assert in_win.summary["sensing_keepalive_active"] is True
    assert out_win.summary["sensing_keepalive_active"] is False


def test_drd1l1b_keepalive_released_is_byte_identical_to_off():
    torch.manual_seed(88)
    model_a = RecurrentMAPPOActorCritic(default_stage23_core_config())
    model_b = _s25u_copy.deepcopy(model_a)
    batch = _s25u_collect(
        model_a, _S25UCollectionConfig(max_episodes=4, steps_per_episode=8, seed=890)
    ).batch
    cfg_off = _S25UUpdateConfig.from_decision_records()
    cfg_ka = dataclasses.replace(cfg_off, sensing_keepalive_coefficient=0.1,
                                 sensing_keepalive_rounds=2, sensing_entropy_target=10.0)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=cfg_off.learning_rate, eps=cfg_off.adam_epsilon)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=cfg_off.learning_rate, eps=cfg_off.adam_epsilon)
    # round_index 5 >= keepalive_rounds 2 => released => byte-identical to OFF.
    _s25u_update(model_a, batch, cfg_off, optimizer=opt_a,
                 controller=cfg_off.make_controller(), episode_count=4, round_index=5)
    _s25u_update(model_b, batch, cfg_ka, optimizer=opt_b,
                 controller=cfg_ka.make_controller(), episode_count=4, round_index=5)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)


def test_drd1l1b_keepalive_in_window_changes_gradient():
    torch.manual_seed(88)
    model_a = RecurrentMAPPOActorCritic(default_stage23_core_config())
    model_b = _s25u_copy.deepcopy(model_a)
    batch = _s25u_collect(
        model_a, _S25UCollectionConfig(max_episodes=4, steps_per_episode=8, seed=890)
    ).batch
    cfg_off = _S25UUpdateConfig.from_decision_records()
    cfg_ka = dataclasses.replace(cfg_off, sensing_keepalive_coefficient=0.1,
                                 sensing_keepalive_rounds=3, sensing_entropy_target=10.0)
    opt_a = torch.optim.Adam(model_a.parameters(), lr=cfg_off.learning_rate, eps=cfg_off.adam_epsilon)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=cfg_off.learning_rate, eps=cfg_off.adam_epsilon)
    _s25u_update(model_a, batch, cfg_off, optimizer=opt_a,
                 controller=cfg_off.make_controller(), episode_count=4, round_index=0)
    _s25u_update(model_b, batch, cfg_ka, optimizer=opt_b,
                 controller=cfg_ka.make_controller(), episode_count=4, round_index=0)
    # In-window keep-alive alters the gradient => at least one parameter differs.
    assert any(
        not torch.equal(pa, pb)
        for pa, pb in zip(model_a.parameters(), model_b.parameters())
    )


# ---------- probe repoint to the fork family (owed SHIFT-9) + I-3 guard ----------


def test_drd1l1b_probe_repoint_fork_family_and_i3_guard():
    torch.manual_seed(114)
    model = RecurrentMAPPOActorCritic(default_stage23_core_config())
    fams = _drd1_fork_families()
    fork_scenarios = tuple(fams["training"]) + tuple(fams["readiness"])
    cfg = default_stage22_update_config()
    probe = _drd1_probe(model, probe_seed=9001, update_config=cfg,
                        probe_scenarios=fork_scenarios)
    assert probe["fork_family"] is True
    assert set(probe["variants"].keys()) == set(fork_scenarios)
    # I-3: a non-fork / would-be held-out scenario name is refused.
    with pytest.raises(RuntimeError, match="fork readiness probe admits only"):
        _drd1_probe(model, probe_seed=9001, update_config=cfg,
                    probe_scenarios=("risk_gate_hidden_hazard",))
    # legacy path (probe_scenarios None) still returns the risk_gate surface.
    legacy = _drd1_probe(model, probe_seed=9001, update_config=cfg)
    assert "fork_family" not in legacy


# ---------- SHIFT-10 adversarial-pass fixes ----------


def test_drd1l1b_keepalive_active_flag_reflects_target_gate():
    # In-window but the target-entropy gate is CLOSED (sensing_entropy > a tiny
    # target) => NO bonus is applied => the diagnostic must be False (it reports
    # real firing, not merely that the window is active).
    torch.manual_seed(71)
    model = RecurrentMAPPOActorCritic(default_stage23_core_config())
    collected = _s25u_collect(
        model, _S25UCollectionConfig(max_episodes=4, steps_per_episode=6, seed=71)
    )
    cfg = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(),
        sensing_keepalive_coefficient=0.1, sensing_keepalive_rounds=3,
        sensing_entropy_target=1e-6,  # gate essentially always closed
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    res = _s25u_update(model, collected.batch, cfg, optimizer=opt,
                       controller=cfg.make_controller(), episode_count=4, round_index=0)
    assert res.summary["sensing_keepalive_active"] is False


def test_drd1l1b_keepalive_requires_round_index():
    torch.manual_seed(71)
    model = RecurrentMAPPOActorCritic(default_stage23_core_config())
    collected = _s25u_collect(
        model, _S25UCollectionConfig(max_episodes=4, steps_per_episode=6, seed=71)
    )
    cfg = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(),
        sensing_keepalive_coefficient=0.1, sensing_keepalive_rounds=3,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    with pytest.raises(ValueError, match="round_index is required when sensing_keepalive"):
        _s25u_update(model, collected.batch, cfg, optimizer=opt,
                     controller=cfg.make_controller(), episode_count=4, round_index=None)


def test_drd1l1b_collection_config_sanitize_stores_weight():
    class _Hostile(float):
        def __gt__(self, other):  # a lying comparison operator
            return True

    cfg = _S25UCollectionConfig(sensing_credit_potential_weight=_Hostile(0.05))
    # The base float is stored, not the hostile subclass => the > 0.0 activation
    # test and the phi multiply cannot be hijacked.
    assert type(cfg.sensing_credit_potential_weight) is float
    assert cfg.sensing_credit_potential_weight == 0.05


# ===========================================================================
# SECTION DRREL -- DR-RELIABILITY (SHIFT 19): lever F (lambda sustain floor)
# + lever E (per-factor entropy-target sustain controller). Mirrors the
# integral_cap pin family (lever F) and the keep-alive family (lever E).
# ===========================================================================

from raas_marl.final_training.stage25_update import (
    EntropySustainController as _DRRELEntropyController,
    EntropySustainStep as _DRRELEntropyStep,
    entropy_sustain_target as _drrel_target,
)


# ---------------- lever F: PIController lambda_floor ----------------


def test_drrel_pi_controller_lambda_floor_off_default_byte_identical():
    base = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=0.1, budget=0.5
    )
    explicit = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=0.1, budget=0.5,
        lambda_floor=0.0,
    )
    for cost in (0.0, 0.3, 0.5, 1.0, 1.7):
        step_a = base.update(observed_episodic_cost=cost)
        step_b = explicit.update(observed_episodic_cost=cost)
        assert step_a.lambda_applied == step_b.lambda_applied
        assert step_a.integral == step_b.integral
        base = step_a.controller
        explicit = step_b.controller


def test_drrel_pi_controller_lambda_floor_applies_hand_case():
    # Satisfied constraint (cost 0 < budget 0.5): candidate = K_P*(-0.5) + I
    # decays below the floor -> lambda_applied pinned AT the floor while the
    # integral keeps discharging underneath (the H1 anti-collapse semantics).
    controller = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=1.0, budget=0.5,
        lambda_floor=3.0,
    )
    step = controller.update(observed_episodic_cost=0.0)
    # I_1 = max(0, 1.0 + 0.05*(-0.5)) = 0.975; candidate = 0.25*(-0.5)+0.975
    assert step.integral == pytest.approx(0.975)
    assert step.lambda_applied == 3.0
    # Violated constraint far above the floor: the floor is inert.
    hot = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=10.0, budget=0.5,
        lambda_floor=3.0,
    )
    hot_step = hot.update(observed_episodic_cost=1.5)
    assert hot_step.lambda_applied > 3.0
    assert hot_step.lambda_applied == pytest.approx(0.25 * 1.0 + hot_step.integral)


def test_drrel_pi_controller_lambda_floor_persists_across_rounds():
    controller = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=0.1, budget=0.5,
        lambda_floor=3.0,
    )
    step1 = controller.update(observed_episodic_cost=0.0)
    step2 = step1.controller.update(observed_episodic_cost=0.0)
    assert step1.controller.lambda_floor == 3.0
    assert step2.controller.lambda_floor == 3.0
    assert step2.lambda_applied == 3.0


def test_drrel_pi_controller_lambda_floor_rejects_invalid():
    with pytest.raises(ValueError):
        _S25UPIController(
            proportional_gain=0.25, integral_gain=0.05, integral=0.1,
            budget=0.5, lambda_floor=-1.0,
        )
    with pytest.raises(TypeError):
        _S25UPIController(
            proportional_gain=0.25, integral_gain=0.05, integral=0.1,
            budget=0.5, lambda_floor="3.0",
        )


def test_drrel_update_config_lambda_floor_off_default_and_threads():
    cfg = _S25UUpdateConfig.from_decision_records()
    assert cfg.lambda_floor == 0.0
    assert cfg.make_controller().lambda_floor == 0.0
    bridge = _S25UUpdateConfig.dev_regression_bridge()
    assert bridge.lambda_floor == 0.0
    on = dataclasses.replace(cfg, lambda_floor=3.0)
    assert on.make_controller().lambda_floor == 3.0


def test_drrel_update_config_floor_requires_pre_epochs():
    with pytest.raises(ValueError, match="lambda_floor > 0 requires pre_epochs"):
        dataclasses.replace(
            _S25UUpdateConfig.dev_regression_bridge(), lambda_floor=3.0
        )


def test_drrel_update_a10_mismatched_lambda_floor_rejected():
    model, collected = _s25u_model_and_batch()
    cfg = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(), lambda_floor=3.0
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    off_controller = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=0.1, budget=0.5,
        lambda_floor=0.0,
    )
    with pytest.raises(ValueError, match="lambda_floor"):
        _s25u_update(model, collected.batch, cfg, optimizer=opt,
                     controller=off_controller, episode_count=4, round_index=0)


def test_drrel_warmup_holds_lambda_zero_with_floor():
    model, collected = _s25u_model_and_batch()
    cfg = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(),
        lambda_floor=3.0, constraint_warmup_rounds=5,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    res = _s25u_update(model, collected.batch, cfg, optimizer=opt,
                       controller=cfg.make_controller(), episode_count=4,
                       round_index=0)
    # Warmup bypasses the controller entirely: the floor must NOT apply.
    assert res.lambda_applied == 0.0
    assert res.summary["constraint_warmup_active"] is True
    assert res.summary["lambda_floor"] == 3.0
    # Past warmup the floor binds immediately on a satisfied constraint.
    res2 = _s25u_update(model, collected.batch, cfg, optimizer=opt,
                        controller=res.controller, episode_count=4,
                        round_index=5)
    assert res2.lambda_applied >= 3.0


# ---------------- lever E: target schedule ----------------


def test_drrel_entropy_sustain_target_schedule_hand_cases():
    assert _drrel_target(0.5, 0, 800, 2000) == 0.5
    assert _drrel_target(0.5, 799, 800, 2000) == 0.5
    assert _drrel_target(0.5, 1400, 800, 2000) == pytest.approx(0.25)
    assert _drrel_target(0.5, 2000, 800, 2000) == 0.0
    assert _drrel_target(0.5, 3999, 800, 2000) == 0.0
    # hold == anneal_end degenerates to a step with no division by zero.
    assert _drrel_target(0.5, 6, 7, 7) == 0.5
    assert _drrel_target(0.5, 7, 7, 7) == 0.0
    with pytest.raises(ValueError, match="anneal_end_rounds must be >="):
        _drrel_target(0.5, 0, 10, 5)
    with pytest.raises(TypeError):
        _drrel_target(0.5, 0.5, 10, 20)
    # Impl-audit lens-1 pin: a huge-int anneal span must normalize to the
    # project's ValueError convention, never leak a raw OverflowError.
    with pytest.raises(ValueError, match="entropy sustain target must be finite"):
        _drrel_target(0.5, 0, 0, 10**400)


# ---------------- lever E: controller dual step ----------------


def _drrel_controller(**overrides):
    values = dict(
        learning_rate=0.02, movement_target=0.5, sensing_target=0.3,
        hold_rounds=800, anneal_end_rounds=2000, alpha_cap=2.0,
    )
    values.update(overrides)
    return _DRRELEntropyController(**values)


def test_drrel_entropy_controller_dual_step_hand_case():
    step = _drrel_controller().update(
        measured_movement_entropy=0.2, measured_sensing_entropy=0.1,
        round_index=100,
    )
    assert isinstance(step, _DRRELEntropyStep)
    # alpha grows by lr * (target - measured) from 0.
    assert step.alpha_movement == pytest.approx(0.02 * (0.5 - 0.2))
    assert step.alpha_sensing == pytest.approx(0.02 * (0.3 - 0.1))
    assert step.movement_target == 0.5
    assert step.sensing_target == 0.3
    follow = step.controller.update(
        measured_movement_entropy=0.2, measured_sensing_entropy=0.1,
        round_index=101,
    )
    # State persists onto the successor and compounds (the AM-2 lesson).
    assert follow.alpha_movement == pytest.approx(2 * 0.02 * 0.3)
    assert follow.alpha_sensing == pytest.approx(2 * 0.02 * 0.2)


def test_drrel_entropy_controller_caps_projects_and_decays():
    controller = _drrel_controller()
    for r in range(300):
        controller = controller.update(
            measured_movement_entropy=0.0, measured_sensing_entropy=0.0,
            round_index=r,
        ).controller
    # Sustained maximal error saturates at the cap (never beyond).
    assert controller.alpha_movement == 2.0
    assert controller.alpha_sensing == pytest.approx(1.8)
    # Above-target entropy decays alpha through the projection (floor 0).
    relaxed = controller
    for r in range(300, 600):
        relaxed = relaxed.update(
            measured_movement_entropy=1.6, measured_sensing_entropy=0.69,
            round_index=r,
        ).controller
    assert relaxed.alpha_movement == 0.0
    assert relaxed.alpha_sensing == 0.0


def test_drrel_entropy_controller_none_target_factor_pinned_zero():
    step = _drrel_controller(sensing_target=None).update(
        measured_movement_entropy=0.0, measured_sensing_entropy=0.0,
        round_index=0,
    )
    assert step.alpha_sensing == 0.0
    assert step.sensing_target == 0.0
    assert step.alpha_movement > 0.0


def test_drrel_entropy_controller_rejects_invalid():
    with pytest.raises(ValueError):
        _drrel_controller(learning_rate=0.0)
    with pytest.raises(ValueError, match="at least one of movement_target"):
        _drrel_controller(movement_target=None, sensing_target=None)
    with pytest.raises(ValueError, match="anneal_end_rounds must be >="):
        _drrel_controller(hold_rounds=100, anneal_end_rounds=50)
    with pytest.raises(ValueError, match="alpha state must be <= alpha_cap"):
        _drrel_controller(alpha_movement=3.0)
    with pytest.raises(ValueError):
        _drrel_controller(alpha_sensing=-0.1)
    with pytest.raises(TypeError):
        _drrel_controller(hold_rounds=1.5)
    ctrl = _drrel_controller()
    with pytest.raises(ValueError):
        ctrl.update(measured_movement_entropy=-0.1,
                    measured_sensing_entropy=0.1, round_index=0)
    with pytest.raises(TypeError):
        ctrl.update(measured_movement_entropy=0.1,
                    measured_sensing_entropy=0.1, round_index=True)


# ---------------- lever E: update-config coupling ----------------


def test_drrel_update_config_entropy_sustain_validation_matrix():
    base = _S25UUpdateConfig.from_decision_records()
    assert base.entropy_sustain_learning_rate == 0.0
    assert base.entropy_sustain_alpha_cap is None
    assert base.make_entropy_controller() is None
    bridge = _S25UUpdateConfig.dev_regression_bridge()
    assert bridge.entropy_sustain_learning_rate == 0.0
    with pytest.raises(ValueError, match="entropy_sustain_alpha_cap is required"):
        dataclasses.replace(
            base, entropy_sustain_learning_rate=0.02,
            movement_entropy_sustain_target=0.5,
            entropy_sustain_anneal_rounds=10,
        )
    with pytest.raises(ValueError, match="at least one entropy sustain target"):
        dataclasses.replace(
            base, entropy_sustain_learning_rate=0.02,
            entropy_sustain_alpha_cap=2.0,
        )
    with pytest.raises(ValueError, match="anneal_rounds must be >="):
        dataclasses.replace(
            base, entropy_sustain_hold_rounds=10,
            entropy_sustain_anneal_rounds=5,
        )
    with pytest.raises(ValueError, match="requires pre_epochs"):
        dataclasses.replace(
            _S25UUpdateConfig.dev_regression_bridge(),
            entropy_sustain_learning_rate=0.02,
            movement_entropy_sustain_target=0.5,
            entropy_sustain_alpha_cap=2.0,
        )
    on = dataclasses.replace(
        base, entropy_sustain_learning_rate=0.02,
        movement_entropy_sustain_target=0.5,
        sensing_entropy_sustain_target=0.3,
        entropy_sustain_hold_rounds=800,
        entropy_sustain_anneal_rounds=2000,
        entropy_sustain_alpha_cap=2.0,
    )
    built = on.make_entropy_controller()
    assert built is not None
    assert built.learning_rate == 0.02
    assert built.alpha_cap == 2.0
    assert built.alpha_movement == 0.0


def test_drrel_update_requires_round_index_and_controller_when_on():
    model, collected = _s25u_model_and_batch()
    cfg = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(),
        entropy_sustain_learning_rate=0.02,
        movement_entropy_sustain_target=0.5,
        entropy_sustain_hold_rounds=5,
        entropy_sustain_anneal_rounds=10,
        entropy_sustain_alpha_cap=2.0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    with pytest.raises(ValueError, match="round_index is required when entropy_sustain"):
        _s25u_update(model, collected.batch, cfg, optimizer=opt,
                     controller=cfg.make_controller(), episode_count=4,
                     round_index=None,
                     entropy_controller=cfg.make_entropy_controller())
    with pytest.raises(ValueError, match="entropy_controller is required"):
        _s25u_update(model, collected.batch, cfg, optimizer=opt,
                     controller=cfg.make_controller(), episode_count=4,
                     round_index=0, entropy_controller=None)


def test_drrel_update_entropy_controller_constants_mismatch_rejected():
    model, collected = _s25u_model_and_batch()
    cfg = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(),
        entropy_sustain_learning_rate=0.02,
        movement_entropy_sustain_target=0.5,
        entropy_sustain_hold_rounds=5,
        entropy_sustain_anneal_rounds=10,
        entropy_sustain_alpha_cap=2.0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    mismatched = _drrel_controller(
        learning_rate=0.05, movement_target=0.5, sensing_target=None,
        hold_rounds=5, anneal_end_rounds=10, alpha_cap=2.0,
    )
    with pytest.raises(ValueError, match="entropy controller constants"):
        _s25u_update(model, collected.batch, cfg, optimizer=opt,
                     controller=cfg.make_controller(), episode_count=4,
                     round_index=0, entropy_controller=mismatched)
    with pytest.raises(TypeError, match="EntropySustainController or None"):
        _s25u_update(model, collected.batch, cfg, optimizer=opt,
                     controller=cfg.make_controller(), episode_count=4,
                     round_index=0, entropy_controller=0.02)


# ---------------- lever E+F: OFF-default byte identity ----------------


def test_drrel_off_default_update_byte_identical():
    torch.manual_seed(88)
    model_a = RecurrentMAPPOActorCritic(default_stage23_core_config())
    model_b = _s25u_copy.deepcopy(model_a)
    batch = _s25u_collect(
        model_a, _S25UCollectionConfig(max_episodes=4, steps_per_episode=8, seed=890)
    ).batch
    cfg_a = _S25UUpdateConfig.from_decision_records()  # all levers OFF
    cfg_b = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(),
        lambda_floor=0.0,
        entropy_sustain_learning_rate=0.0,
        movement_entropy_sustain_target=None,
        sensing_entropy_sustain_target=None,
        entropy_sustain_hold_rounds=0,
        entropy_sustain_anneal_rounds=0,
        entropy_sustain_alpha_cap=None,
    )
    opt_a = torch.optim.Adam(model_a.parameters(), lr=cfg_a.learning_rate, eps=cfg_a.adam_epsilon)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=cfg_b.learning_rate, eps=cfg_b.adam_epsilon)
    res_a = _s25u_update(model_a, batch, cfg_a, optimizer=opt_a,
                         controller=cfg_a.make_controller(), episode_count=4,
                         round_index=0)
    res_b = _s25u_update(model_b, batch, cfg_b, optimizer=opt_b,
                         controller=cfg_b.make_controller(), episode_count=4,
                         round_index=0, entropy_controller=None)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)
    assert res_a.summary["entropy_sustain_active"] is False
    assert res_a.summary["entropy_sustain_alpha_movement"] == 0.0
    assert res_a.summary["entropy_sustain_alpha_sensing"] == 0.0
    assert res_a.summary["lambda_floor"] == 0.0
    assert res_a.entropy_controller is None
    assert res_b.entropy_controller is None


def test_drrel_post_anneal_zero_alpha_byte_identical_to_off():
    # After the anneal window (targets 0) with alpha state 0, the sustain
    # controller must be numerically inert: same parameters as OFF, proving
    # the release is genuine (no cliff, no residual pressure) and the
    # measurement forward is trajectory-inert.
    torch.manual_seed(88)
    model_a = RecurrentMAPPOActorCritic(default_stage23_core_config())
    model_b = _s25u_copy.deepcopy(model_a)
    batch = _s25u_collect(
        model_a, _S25UCollectionConfig(max_episodes=4, steps_per_episode=8, seed=890)
    ).batch
    cfg_off = _S25UUpdateConfig.from_decision_records()
    cfg_on = dataclasses.replace(
        cfg_off, entropy_sustain_learning_rate=0.02,
        movement_entropy_sustain_target=0.5,
        sensing_entropy_sustain_target=0.3,
        entropy_sustain_hold_rounds=2, entropy_sustain_anneal_rounds=3,
        entropy_sustain_alpha_cap=2.0,
    )
    opt_a = torch.optim.Adam(model_a.parameters(), lr=cfg_off.learning_rate, eps=cfg_off.adam_epsilon)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=cfg_on.learning_rate, eps=cfg_on.adam_epsilon)
    res_a = _s25u_update(model_a, batch, cfg_off, optimizer=opt_a,
                         controller=cfg_off.make_controller(), episode_count=4,
                         round_index=50)
    res_b = _s25u_update(model_b, batch, cfg_on, optimizer=opt_b,
                         controller=cfg_on.make_controller(), episode_count=4,
                         round_index=50,  # far past anneal_end=3 -> target 0
                         entropy_controller=cfg_on.make_entropy_controller())
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)
    assert res_b.summary["entropy_sustain_active"] is True
    assert res_b.summary["entropy_sustain_alpha_movement"] == 0.0
    assert res_b.summary["entropy_sustain_movement_target"] == 0.0


def test_drrel_in_window_sustain_changes_gradient():
    torch.manual_seed(88)
    model_a = RecurrentMAPPOActorCritic(default_stage23_core_config())
    model_b = _s25u_copy.deepcopy(model_a)
    batch = _s25u_collect(
        model_a, _S25UCollectionConfig(max_episodes=4, steps_per_episode=8, seed=890)
    ).batch
    cfg_off = _S25UUpdateConfig.from_decision_records()
    cfg_on = dataclasses.replace(
        cfg_off, entropy_sustain_learning_rate=0.5,  # large lr so alpha fires round 0
        movement_entropy_sustain_target=2.0,  # above ln5: always below target
        sensing_entropy_sustain_target=1.0,
        entropy_sustain_hold_rounds=100, entropy_sustain_anneal_rounds=200,
        entropy_sustain_alpha_cap=2.0,
    )
    opt_a = torch.optim.Adam(model_a.parameters(), lr=cfg_off.learning_rate, eps=cfg_off.adam_epsilon)
    opt_b = torch.optim.Adam(model_b.parameters(), lr=cfg_on.learning_rate, eps=cfg_on.adam_epsilon)
    _s25u_update(model_a, batch, cfg_off, optimizer=opt_a,
                 controller=cfg_off.make_controller(), episode_count=4,
                 round_index=0)
    res_b = _s25u_update(model_b, batch, cfg_on, optimizer=opt_b,
                         controller=cfg_on.make_controller(), episode_count=4,
                         round_index=0,
                         entropy_controller=cfg_on.make_entropy_controller())
    assert res_b.summary["entropy_sustain_alpha_movement"] > 0.0
    assert res_b.entropy_controller.alpha_movement > 0.0
    assert any(
        not torch.equal(pa, pb)
        for pa, pb in zip(model_a.parameters(), model_b.parameters())
    )


def test_drrel_update_summary_carries_unconditional_keys():
    model, collected = _s25u_model_and_batch()
    cfg = _S25UUpdateConfig.from_decision_records()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    res = _s25u_update(model, collected.batch, cfg, optimizer=opt,
                       controller=cfg.make_controller(), episode_count=4,
                       round_index=0)
    for key in (
        "entropy_sustain_active", "entropy_sustain_alpha_movement",
        "entropy_sustain_alpha_sensing",
        "entropy_sustain_measured_movement_entropy",
        "entropy_sustain_measured_sensing_entropy",
        "entropy_sustain_movement_target", "entropy_sustain_sensing_target",
        "lambda_floor",
    ):
        assert key in res.summary


def test_drrel_update_lever_on_threads_successor_state():
    model, collected = _s25u_model_and_batch()
    cfg = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(),
        entropy_sustain_learning_rate=0.02,
        movement_entropy_sustain_target=0.5,
        sensing_entropy_sustain_target=0.3,
        entropy_sustain_hold_rounds=5, entropy_sustain_anneal_rounds=10,
        entropy_sustain_alpha_cap=2.0,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon)
    controller = cfg.make_controller()
    entropy_controller = cfg.make_entropy_controller()
    res = _s25u_update(model, collected.batch, cfg, optimizer=opt,
                       controller=controller, episode_count=4, round_index=0,
                       entropy_controller=entropy_controller)
    assert isinstance(res.entropy_controller, _DRRELEntropyController)
    assert res.summary["entropy_sustain_active"] is True
    measured = res.summary["entropy_sustain_measured_movement_entropy"]
    assert measured > 0.0
    # The successor alpha equals the hand-computed projected step.
    expected = min(2.0, max(0.0, 0.02 * (0.5 - measured)))
    assert res.entropy_controller.alpha_movement == pytest.approx(expected)


# ---------------- driver threading + checkpoint round-trip ----------------


def test_drrel_run_config_validation():
    with pytest.raises(ValueError, match="anneal_rounds must be <= update_rounds"):
        _S25URunConfig(
            run_id="drrel_v", seed=151, update_rounds=4,
            entropy_sustain_anneal_rounds=5,
        )
    with pytest.raises(ValueError):
        _S25URunConfig(run_id="drrel_v", seed=151, update_rounds=4,
                       lambda_floor=-1.0)
    with pytest.raises(ValueError):
        _S25URunConfig(run_id="drrel_v", seed=151, update_rounds=4,
                       movement_entropy_sustain_target=0.0)


def test_drrel_driver_state_roundtrip_and_missing_state_rejected(tmp_path):
    parent = tmp_path / "runs"
    parent.mkdir()
    config = _S25URunConfig(
        run_id="drrel_e2e", seed=151, update_rounds=4, episodes_per_round=2,
        steps_per_episode=6, probe_every=0, state_save_every=2, tier="T0",
        result_parent=parent,
        lambda_floor=3.0,
        entropy_sustain_learning_rate=0.02,
        movement_entropy_sustain_target=0.5,
        sensing_entropy_sustain_target=0.3,
        entropy_sustain_hold_rounds=1,
        entropy_sustain_anneal_rounds=2,
        entropy_sustain_alpha_cap=2.0,
    )
    run_dir = _s25u_run_training(config)
    # The saved state carries the floor AND the entropy alpha state.
    state = torch.load(run_dir / "state_round_000002.pt", weights_only=True)
    assert state["controller"]["lambda_floor"] == 3.0
    assert isinstance(state["entropy_controller"], dict)
    assert "alpha_movement" in state["entropy_controller"]
    # Post-warmup floor binding shows in the curve (no warmup here).
    curves = [
        json.loads(x)
        for x in (run_dir / "training_curve.jsonl").read_text().strip().split("\n")
    ]
    assert all(c["lambda_applied"] >= 3.0 for c in curves)
    assert all(c["lambda_floor"] == 3.0 for c in curves)
    assert all(c["entropy_sustain_active"] is True for c in curves)
    # --- AM-2 pin: an ON-config resume with the entropy state stripped must
    # fail LOUDLY (not silently reset the temperatures). Roll back to the
    # round-2 state, strip the key, re-hash, then attempt resume.
    (run_dir / "state_round_000004.pt").unlink()
    state_path = run_dir / "state_round_000002.pt"
    stripped = dict(state)
    stripped["entropy_controller"] = None
    torch.save(stripped, state_path)
    import hashlib

    digest = hashlib.sha256(state_path.read_bytes()).hexdigest()
    lines = (run_dir / "state_log.jsonl").read_text().strip().split("\n")
    first = json.loads(lines[0])
    first["saved_state_sha256"] = digest
    first["saved_state_bytes"] = state_path.stat().st_size
    (run_dir / "state_log.jsonl").write_text(
        json.dumps(first) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="requires the .*entropy_controller state"):
        _s25u_run_training(config, resume=True)


def test_drrel_driver_off_default_state_payload_has_no_entropy_state(tmp_path):
    parent = tmp_path / "runs"
    parent.mkdir()
    config = _S25URunConfig(
        run_id="drrel_off", seed=151, update_rounds=2, episodes_per_round=2,
        steps_per_episode=6, probe_every=0, state_save_every=2, tier="T0",
        result_parent=parent,
    )
    run_dir = _s25u_run_training(config)
    state = torch.load(run_dir / "state_round_000002.pt", weights_only=True)
    assert state["entropy_controller"] is None
    assert state["controller"]["lambda_floor"] == 0.0
    curves = [
        json.loads(x)
        for x in (run_dir / "training_curve.jsonl").read_text().strip().split("\n")
    ]
    assert all(c["entropy_sustain_active"] is False for c in curves)
    assert all(c["lambda_floor"] == 0.0 for c in curves)


# ===========================================================================
# SECTION DRH3 -- DR-H3 (SHIFT 21): lever G (applied-multiplier ceiling) +
# lever H (armed conditional floor: measured-regime latch + AM-H4 de-arm
# hatch, driver-owned state). Countersign wf_43ec6ea5-1ba; AM-H13 fork
# scoping, AM-H15 window convention, AM-2-style arm-state resume reject.
# ===========================================================================

from raas_marl.final_training.stage25_driver import (
    _ArmedFloorState as _DRH3ArmState,
    _round_behavior_aggregates as _drh3_aggregates,
)


# ---------------- lever G: PIController lambda_ceiling ----------------


def test_drh3_pi_controller_ceiling_off_default_byte_identical():
    base = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=0.1, budget=0.5
    )
    explicit = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=0.1, budget=0.5,
        lambda_ceiling=None, armed_lambda_floor=None,
    )
    for cost in (0.0, 0.3, 0.5, 1.0, 1.7):
        step_a = base.update(observed_episodic_cost=cost)
        step_b = explicit.update(observed_episodic_cost=cost)
        assert step_a.lambda_applied == step_b.lambda_applied
        assert step_a.integral == step_b.integral
        assert step_b.ceiling_engaged is False
        assert step_b.lambda_pre_clamp == step_b.lambda_applied
        base = step_a.controller
        explicit = step_b.controller


def test_drh3_pi_controller_ceiling_clamps_hand_case():
    # Violated constraint with a wound integral: I_1 = 10 + 0.05*1.0 = 10.05,
    # candidate = 0.25*1.0 + 10.05 = 10.3 -> ceiling 5.0 binds; the pre-clamp
    # value (the AM-H9 row-3 read) carries the unclamped 10.3.
    hot = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=10.0, budget=0.5,
        lambda_ceiling=5.0,
    )
    step = hot.update(observed_episodic_cost=1.5)
    assert step.lambda_applied == 5.0
    assert step.lambda_pre_clamp == pytest.approx(10.3)
    assert step.ceiling_engaged is True
    # Below the ceiling: applied == pre-clamp, engaged False.
    cool = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=0.1, budget=0.5,
        lambda_ceiling=5.0,
    )
    cool_step = cool.update(observed_episodic_cost=0.0)
    assert cool_step.ceiling_engaged is False
    assert cool_step.lambda_applied == cool_step.lambda_pre_clamp
    assert cool_step.lambda_applied < 5.0


def test_drh3_pi_controller_ceiling_persists_across_rounds():
    controller = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=10.0, budget=0.5,
        lambda_ceiling=5.0, armed_lambda_floor=3.0,
    )
    step1 = controller.update(observed_episodic_cost=1.5)
    step2 = step1.controller.update(observed_episodic_cost=1.5)
    assert step1.controller.lambda_ceiling == 5.0
    assert step2.controller.lambda_ceiling == 5.0
    assert step1.controller.armed_lambda_floor == 3.0
    assert step2.controller.armed_lambda_floor == 3.0
    assert step2.lambda_applied == 5.0


def test_drh3_pi_controller_ceiling_and_armed_floor_reject_invalid():
    for bad in (0.0, -1.0):
        with pytest.raises(ValueError):
            _S25UPIController(
                proportional_gain=0.25, integral_gain=0.05, integral=0.1,
                budget=0.5, lambda_ceiling=bad,
            )
        with pytest.raises(ValueError):
            _S25UPIController(
                proportional_gain=0.25, integral_gain=0.05, integral=0.1,
                budget=0.5, armed_lambda_floor=bad,
            )
    for bad in ("5.0", True):
        with pytest.raises(TypeError):
            _S25UPIController(
                proportional_gain=0.25, integral_gain=0.05, integral=0.1,
                budget=0.5, lambda_ceiling=bad,
            )
    with pytest.raises(ValueError, match="strictly below lambda_ceiling"):
        _S25UPIController(
            proportional_gain=0.25, integral_gain=0.05, integral=0.1,
            budget=0.5, lambda_ceiling=5.0, armed_lambda_floor=5.0,
        )
    with pytest.raises(ValueError, match="strictly below lambda_ceiling"):
        _S25UPIController(
            proportional_gain=0.25, integral_gain=0.05, integral=0.1,
            budget=0.5, lambda_ceiling=3.0, lambda_floor=3.0,
        )
    with pytest.raises(ValueError, match="mutually exclusive"):
        _S25UPIController(
            proportional_gain=0.25, integral_gain=0.05, integral=0.1,
            budget=0.5, armed_lambda_floor=3.0, lambda_floor=3.0,
        )


# ---------------- lever H: the armed floor at the controller ----------------


def test_drh3_pi_controller_armed_floor_applies_only_when_armed():
    controller = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=1.0, budget=0.5,
        armed_lambda_floor=3.0,
    )
    # Un-armed: the floor term is the (0.0) unconditional floor.
    off_step = controller.update(observed_episodic_cost=0.0)
    assert off_step.lambda_applied == pytest.approx(0.25 * -0.5 + 0.975)
    assert off_step.floor_armed is False
    # Armed: pinned at the armed floor while the integral discharges below.
    on_step = controller.update(observed_episodic_cost=0.0, floor_armed=True)
    assert on_step.lambda_applied == 3.0
    assert on_step.floor_armed is True
    assert on_step.integral == pytest.approx(0.975)


def test_drh3_pi_controller_floor_armed_requires_configured_value():
    plain = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=0.1, budget=0.5
    )
    with pytest.raises(ValueError, match="floor_armed=True requires"):
        plain.update(observed_episodic_cost=0.0, floor_armed=True)
    with pytest.raises(TypeError, match="floor_armed must be a bool"):
        plain.update(observed_episodic_cost=0.0, floor_armed=1)


def test_drh3_pi_controller_armed_floor_under_ceiling():
    controller = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=10.0, budget=0.5,
        lambda_ceiling=5.0, armed_lambda_floor=3.0,
    )
    step = controller.update(observed_episodic_cost=1.5, floor_armed=True)
    assert step.lambda_applied == 5.0  # ceiling wins above; floor irrelevant
    assert step.lambda_pre_clamp == pytest.approx(10.3)
    drained = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=0.1, budget=0.5,
        lambda_ceiling=5.0, armed_lambda_floor=3.0,
    )
    low = drained.update(observed_episodic_cost=0.0, floor_armed=True)
    assert low.lambda_applied == 3.0  # armed floor holds below the ceiling
    assert low.ceiling_engaged is False


# ---------------- Stage25UpdateConfig coupling + A-10 + summary ----------------


def test_drh3_update_config_off_default_and_threads():
    cfg = _S25UUpdateConfig.from_decision_records()
    assert cfg.lambda_ceiling is None
    assert cfg.armed_lambda_floor is None
    controller = cfg.make_controller()
    assert controller.lambda_ceiling is None
    assert controller.armed_lambda_floor is None
    on = dataclasses.replace(cfg, lambda_ceiling=5.0, armed_lambda_floor=3.0)
    threaded = on.make_controller()
    assert threaded.lambda_ceiling == 5.0
    assert threaded.armed_lambda_floor == 3.0


def test_drh3_update_config_levers_require_pre_epochs():
    bridge = _S25UUpdateConfig.dev_regression_bridge()
    with pytest.raises(ValueError, match="lambda_ceiling requires pre_epochs"):
        dataclasses.replace(bridge, lambda_ceiling=5.0)
    with pytest.raises(ValueError, match="armed_lambda_floor requires pre_epochs"):
        dataclasses.replace(bridge, armed_lambda_floor=3.0)


def test_drh3_update_config_couplings_rejected():
    cfg = _S25UUpdateConfig.from_decision_records()
    with pytest.raises(ValueError, match="mutually exclusive"):
        dataclasses.replace(cfg, armed_lambda_floor=3.0, lambda_floor=3.0)
    with pytest.raises(ValueError, match="strictly below lambda_ceiling"):
        dataclasses.replace(cfg, lambda_ceiling=5.0, armed_lambda_floor=6.0)
    with pytest.raises(ValueError, match="strictly below lambda_ceiling"):
        dataclasses.replace(cfg, lambda_ceiling=2.0, lambda_floor=3.0)


def test_drh3_update_a10_mismatched_ceiling_rejected():
    model, collected = _s25u_model_and_batch()
    cfg = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(), lambda_ceiling=5.0
    )
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon
    )
    off_controller = _S25UPIController(
        proportional_gain=0.25, integral_gain=0.05, integral=0.1, budget=0.5,
    )
    with pytest.raises(ValueError, match="lambda_ceiling"):
        _s25u_update(model, collected.batch, cfg, optimizer=opt,
                     controller=off_controller, episode_count=4, round_index=0)


def test_drh3_update_floor_armed_guards():
    model, collected = _s25u_model_and_batch()
    cfg = _S25UUpdateConfig.from_decision_records()
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon
    )
    with pytest.raises(ValueError, match="floor_armed=True requires"):
        _s25u_update(model, collected.batch, cfg, optimizer=opt,
                     controller=cfg.make_controller(), episode_count=4,
                     round_index=0, floor_armed=True)
    with pytest.raises(TypeError, match="floor_armed must be a bool"):
        _s25u_update(model, collected.batch, cfg, optimizer=opt,
                     controller=cfg.make_controller(), episode_count=4,
                     round_index=0, floor_armed=1)


def test_drh3_update_summary_keys_off_and_armed():
    model, collected = _s25u_model_and_batch()
    cfg = _S25UUpdateConfig.from_decision_records()
    opt = torch.optim.Adam(
        model.parameters(), lr=cfg.learning_rate, eps=cfg.adam_epsilon
    )
    res = _s25u_update(model, collected.batch, cfg, optimizer=opt,
                       controller=cfg.make_controller(), episode_count=4,
                       round_index=0)
    assert res.summary["lambda_pre_clamp"] == res.summary["lambda_applied"]
    assert res.summary["lambda_ceiling_engaged"] is False
    assert res.summary["lambda_floor_armed"] is False
    # Armed path: the floor value shows up in lambda_applied.
    model2, collected2 = _s25u_model_and_batch()
    armed_cfg = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(), armed_lambda_floor=3.0
    )
    opt2 = torch.optim.Adam(
        model2.parameters(), lr=armed_cfg.learning_rate,
        eps=armed_cfg.adam_epsilon,
    )
    armed_res = _s25u_update(model2, collected2.batch, armed_cfg,
                             optimizer=opt2,
                             controller=armed_cfg.make_controller(),
                             episode_count=4, round_index=0, floor_armed=True)
    assert armed_res.summary["lambda_applied"] >= 3.0
    assert armed_res.summary["lambda_floor_armed"] is True

# ---------------- the driver-owned arm state machine ----------------


def _drh3_arm(warmup: int = 2, arm_w: int = 3, dearm_w: int = 2) -> _DRH3ArmState:
    return _DRH3ArmState(
        arm_window_rounds=arm_w, arm_success_threshold=0.8,
        arm_hazard_threshold=0.35, arm_sensing_threshold=0.05,
        dearm_window_rounds=dearm_w, dearm_success_threshold=0.1,
        constraint_warmup_rounds=warmup,
    )


def test_drh3_arm_fires_on_mode_c_window_and_floors_fire_round():
    arm = _drh3_arm()
    # Warmup rounds are never appended (windows entirely post-warmup).
    assert arm.observe_round(round_index=0, success_rate=1.0, hazard_mean=0.0,
                             sensing_mean=1.0) is False
    assert arm.observe_round(round_index=1, success_rate=1.0, hazard_mean=0.0,
                             sensing_mean=1.0) is False
    assert len(arm._arm_window) == 0
    # Post-warmup Mode-C rounds: the window fills at round 4 and the FIRE
    # round itself returns True (AM-H15: the fire round is floored).
    assert arm.observe_round(round_index=2, success_rate=0.95, hazard_mean=0.1,
                             sensing_mean=0.12) is False
    assert arm.observe_round(round_index=3, success_rate=0.95, hazard_mean=0.1,
                             sensing_mean=0.12) is False
    assert arm.observe_round(round_index=4, success_rate=0.95, hazard_mean=0.1,
                             sensing_mean=0.12) is True
    assert arm.armed is True
    assert arm.events == [{
        "event": "arm", "round_index": 4,
        "window_success": pytest.approx(0.95),
        "window_hazard": pytest.approx(0.1),
        "window_sensing": pytest.approx(0.12),
    }]


def test_drh3_arm_silent_on_blind_paralysis_and_boundary():
    blind = _drh3_arm()
    paralysis = _drh3_arm()
    for k in range(2, 12):
        assert blind.observe_round(round_index=k, success_rate=1.0,
                                   hazard_mean=1.0, sensing_mean=0.1) is False
        assert paralysis.observe_round(round_index=k, success_rate=0.0,
                                       hazard_mean=0.0,
                                       sensing_mean=0.0) is False
    assert blind.events == [] and paralysis.events == []
    # Boundary semantics: hazard mean exactly AT the threshold (<=) fires;
    # just above does not.
    at = _drh3_arm()
    for k in range(2, 5):
        result = at.observe_round(round_index=k, success_rate=0.9,
                                  hazard_mean=0.35, sensing_mean=0.05)
    assert result is True
    above = _drh3_arm()
    for k in range(2, 5):
        result = above.observe_round(round_index=k, success_rate=0.9,
                                     hazard_mean=0.3501, sensing_mean=0.05)
    assert result is False


def test_drh3_arm_latch_holds_through_bad_rounds_then_dearm_and_rearm():
    arm = _drh3_arm()
    for k in range(2, 5):
        arm.observe_round(round_index=k, success_rate=0.95, hazard_mean=0.1,
                          sensing_mean=0.12)
    assert arm.armed is True
    # One blind round does NOT release the latch (no un-arm condition met).
    assert arm.observe_round(round_index=5, success_rate=1.0, hazard_mean=1.0,
                             sensing_mean=0.0) is True
    # AM-H4 hatch: dearm_w=2 rounds of deep paralysis release the floor on the
    # release round itself, with the event logged.
    assert arm.observe_round(round_index=6, success_rate=0.0, hazard_mean=0.0,
                             sensing_mean=0.0) is True  # window not full of lows
    assert arm.observe_round(round_index=7, success_rate=0.05, hazard_mean=0.0,
                             sensing_mean=0.0) is False
    assert arm.armed is False
    assert arm.events[-1]["event"] == "dearm"
    assert arm.events[-1]["round_index"] == 7
    # Re-arm by the same rule after a fresh Mode-C window.
    for k in range(8, 11):
        result = arm.observe_round(round_index=k, success_rate=0.95,
                                   hazard_mean=0.1, sensing_mean=0.12)
    assert result is True
    assert [e["event"] for e in arm.events] == ["arm", "dearm", "arm"]


def test_drh3_arm_payload_round_trip_and_rejects():
    arm = _drh3_arm()
    for k in range(2, 6):
        arm.observe_round(round_index=k, success_rate=0.95, hazard_mean=0.1,
                          sensing_mean=0.12)
    payload = arm.to_payload()
    clone = _drh3_arm()
    clone.restore_payload(payload)
    assert clone.armed == arm.armed
    assert clone.events == arm.events
    assert list(clone._arm_window) == list(arm._arm_window)
    assert list(clone._dearm_window) == list(arm._dearm_window)
    # Behavioral identity after restore (mid-window byte-identity, AM-H15).
    a = arm.observe_round(round_index=6, success_rate=0.0, hazard_mean=0.0,
                          sensing_mean=0.0)
    b = clone.observe_round(round_index=6, success_rate=0.0, hazard_mean=0.0,
                            sensing_mean=0.0)
    assert a == b and arm.armed == clone.armed
    # Hostile payloads reject loudly.
    for bad in (None, [], "x"):
        with pytest.raises(ValueError):
            _drh3_arm().restore_payload(bad)
    with pytest.raises(ValueError, match="missing"):
        _drh3_arm().restore_payload({"armed": False})
    with pytest.raises(ValueError, match="armed must be a bool"):
        _drh3_arm().restore_payload({"armed": 1, "events": [],
                                     "arm_window": [], "dearm_window": []})
    with pytest.raises(ValueError, match="exceeds"):
        _drh3_arm().restore_payload({
            "armed": False, "events": [],
            "arm_window": [[1.0, 0.0, 0.1]] * 4, "dearm_window": [],
        })


def test_drh3_round_behavior_aggregates_single_source():
    records = [
        {"team_success": True, "hazard_cost_sum": 0.5, "sensing_rate": 0.1},
        {"team_success": False, "hazard_cost_sum": 1.5, "sensing_rate": 0.3},
    ]
    success, hazard, sensing = _drh3_aggregates(records)
    assert success == 0.5
    assert hazard == pytest.approx(1.0)
    assert sensing == pytest.approx(0.2)


# ---------------- run-config coupling + AM-H13 fork scoping ----------------


def _drh3_run_config(**overrides):
    kwargs = dict(
        run_id="drh3_cfg", seed=151, update_rounds=8, episodes_per_round=2,
        steps_per_episode=6, probe_every=0, state_save_every=8, tier="T0",
    )
    kwargs.update(overrides)
    return _S25URunConfig(**kwargs)


_DRH3_ARM_KWARGS = dict(
    armed_lambda_floor=3.0, arm_window_rounds=2,
    arm_success_threshold=0.8, arm_hazard_threshold=0.35,
    arm_sensing_threshold=0.05, dearm_window_rounds=4,
    dearm_success_threshold=0.1,
    scenario_names=("risk_fork_train_lower", "risk_fork_train_upper"),
)


def test_drh3_run_config_accepts_full_armed_lever_set():
    config = _drh3_run_config(lambda_ceiling=5.0, **_DRH3_ARM_KWARGS)
    assert config.armed_lambda_floor == 3.0
    assert config.lambda_ceiling == 5.0


def test_drh3_run_config_orphan_arm_params_rejected():
    with pytest.raises(ValueError, match="require armed_lambda_floor"):
        _drh3_run_config(arm_window_rounds=50)
    with pytest.raises(ValueError, match="require armed_lambda_floor"):
        _drh3_run_config(arm_success_threshold=0.8)
    with pytest.raises(ValueError, match="require armed_lambda_floor"):
        _drh3_run_config(dearm_success_threshold=0.1)


def test_drh3_run_config_armed_floor_requires_all_arm_params():
    incomplete = dict(_DRH3_ARM_KWARGS)
    incomplete.pop("arm_hazard_threshold")
    with pytest.raises(ValueError, match="requires all of"):
        _drh3_run_config(**incomplete)


def test_drh3_run_config_dearm_band_and_window_fit():
    bad = dict(_DRH3_ARM_KWARGS, dearm_success_threshold=0.9)
    with pytest.raises(ValueError, match="strictly below"):
        _drh3_run_config(**bad)
    with pytest.raises(ValueError, match="must be < .*update_rounds|arm_window_rounds must be"):
        _drh3_run_config(constraint_warmup_rounds=6, **dict(
            _DRH3_ARM_KWARGS, arm_window_rounds=2))


def test_drh3_run_config_am_h13_fork_scoping():
    legacy = dict(_DRH3_ARM_KWARGS, scenario_names=("risk_gate_hidden_hazard",))
    with pytest.raises(ValueError, match="fork TRAINING family"):
        _drh3_run_config(**legacy)
    mixed = dict(
        _DRH3_ARM_KWARGS,
        scenario_names=("risk_fork_train_lower", "risk_gate_hidden_hazard"),
    )
    with pytest.raises(ValueError, match="fork TRAINING family"):
        _drh3_run_config(**mixed)

# ---------------- driver e2e: wiring, persistence, loud reject ----------------


def test_drh3_driver_off_default_curve_and_state(tmp_path):
    parent = tmp_path / "runs"
    parent.mkdir()
    config = _S25URunConfig(
        run_id="drh3_off", seed=151, update_rounds=2, episodes_per_round=2,
        steps_per_episode=6, probe_every=0, state_save_every=2, tier="T0",
        result_parent=parent,
    )
    run_dir = _s25u_run_training(config)
    state = torch.load(run_dir / "state_round_000002.pt", weights_only=True)
    assert state["controller"]["lambda_ceiling"] is None
    assert state["controller"]["armed_lambda_floor"] is None
    assert state["arm_state"] is None
    curves = [
        json.loads(x)
        for x in (run_dir / "training_curve.jsonl").read_text().strip().split("\n")
    ]
    for c in curves:
        assert c["lambda_pre_clamp"] == c["lambda_applied"]
        assert c["lambda_ceiling_engaged"] is False
        assert c["lambda_floor_armed"] is False
        assert c["arm_window_success"] == 0.0
        assert c["arm_window_hazard"] == 0.0
        assert c["arm_window_sensing"] == 0.0


def test_drh3_driver_armed_run_wiring_and_stripped_state_reject(tmp_path):
    parent = tmp_path / "runs"
    parent.mkdir()
    # Fork family (AM-H13); steps_per_episode=6 makes team success IMPOSSIBLE
    # on the 8x8 fork geometry (goal distance > 6), so the success conjunct
    # is deterministically unreachable and the arm must stay silent while its
    # windows populate — the pre-arm wiring path.
    config = _S25URunConfig(
        run_id="drh3_armed", seed=151, update_rounds=4, episodes_per_round=2,
        steps_per_episode=6, probe_every=0, state_save_every=2, tier="T0",
        result_parent=parent,
        lambda_ceiling=5.0,
        armed_lambda_floor=3.0,
        arm_window_rounds=2,
        arm_success_threshold=0.8,
        arm_hazard_threshold=0.35,
        arm_sensing_threshold=0.05,
        dearm_window_rounds=3,
        dearm_success_threshold=0.1,
        scenario_names=("risk_fork_train_lower", "risk_fork_train_upper"),
    )
    run_dir = _s25u_run_training(config)
    state = torch.load(run_dir / "state_round_000004.pt", weights_only=True)
    assert state["controller"]["lambda_ceiling"] == 5.0
    assert state["controller"]["armed_lambda_floor"] == 3.0
    arm_payload = state["arm_state"]
    assert arm_payload["armed"] is False
    assert arm_payload["events"] == []
    assert len(arm_payload["arm_window"]) == 2  # trailing window populated
    curves = [
        json.loads(x)
        for x in (run_dir / "training_curve.jsonl").read_text().strip().split("\n")
    ]
    for c in curves:
        assert c["lambda_floor_armed"] is False
        assert c["lambda_applied"] <= 5.0 + 1e-9  # MECH-G invariant
        if c["lambda_ceiling_engaged"]:
            assert c["lambda_applied"] == pytest.approx(5.0)
            assert c["lambda_pre_clamp"] > 5.0
        else:
            assert c["lambda_pre_clamp"] == c["lambda_applied"]
    # Success is impossible at 6 steps on the fork, so the window success
    # trace must be identically 0 (the deterministic pre-arm signature).
    assert all(c["arm_window_success"] == 0.0 for c in curves)
    # --- AM-2 analog: an ON-config resume with the arm state stripped must
    # fail LOUDLY. Roll back to the round-2 state, strip arm_state, re-hash.
    (run_dir / "state_round_000004.pt").unlink()
    state_path = run_dir / "state_round_000002.pt"
    early = torch.load(state_path, weights_only=True)
    stripped = dict(early)
    stripped["arm_state"] = None
    torch.save(stripped, state_path)
    import hashlib as _drh3_hashlib

    digest = _drh3_hashlib.sha256(state_path.read_bytes()).hexdigest()
    lines = (run_dir / "state_log.jsonl").read_text().strip().split("\n")
    first = json.loads(lines[0])
    first["saved_state_sha256"] = digest
    first["saved_state_bytes"] = state_path.stat().st_size
    (run_dir / "state_log.jsonl").write_text(
        json.dumps(first) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="requires the .*arm_state"):
        _s25u_run_training(config, resume=True)


def test_drh3_run_config_dearm_window_huge_int_bounded():
    # The DRREL huge-int lesson: an unbounded release window must be rejected
    # at the config (ValueError), never reach deque(maxlen=...) as a raw
    # OverflowError.
    huge = dict(_DRH3_ARM_KWARGS, dearm_window_rounds=10**400)
    with pytest.raises(ValueError, match="dearm_window_rounds must be <="):
        _drh3_run_config(**huge)


def test_drh3_run_config_sanitize_stores_armed_fields():
    class _LyingInt(int):
        def __index__(self):  # honest for validation...
            return int(self.real)

        def __add__(self, other):  # ...hostile in arithmetic
            return 999999

    config = _drh3_run_config(
        **dict(_DRH3_ARM_KWARGS, arm_window_rounds=_LyingInt(2))
    )
    assert type(config.arm_window_rounds) is int
    assert config.arm_window_rounds == 2
    assert type(config.arm_success_threshold) is float
    assert type(config.dearm_success_threshold) is float
    assert type(config.armed_lambda_floor) is float


# ===========================================================================
# SECTION DRH4 -- DR-H4 (SHIFT 23): lever I (armed-gated sensing-sustain
# release + the AM-E1 stateless armed alpha-pin). Countersign wf_2f975bac-779
# (GRANTED_WITH_AMENDMENT, all nine gates PASS); AM-H4-1..9 folded. The gate
# consumes the driver-owned floor_armed latch; the sensing target is H*_sense
# while UNARMED (wall-clock schedule bypassed) and 0.0 with alpha PINNED 0.0
# while ARMED; movement is untouched; OFF-default byte-identical.
# ===========================================================================


# ---------------- controller: gate semantics ----------------


def test_drh4_controller_gate_off_default_byte_identical():
    # Gate OFF (default AND explicit False): floor_armed is validated then
    # IGNORED -- the schedule path must be byte-identical whatever the flag,
    # so T1-D11-style armed-floor-without-gate configs are unchanged.
    base = _drrel_controller()
    explicit = _drrel_controller(sensing_armed_gate=False)
    for rnd, armed in ((0, False), (900, True), (1500, False), (2100, True)):
        step_a = base.update(
            measured_movement_entropy=0.4,
            measured_sensing_entropy=0.1,
            round_index=rnd,
        )
        step_b = explicit.update(
            measured_movement_entropy=0.4,
            measured_sensing_entropy=0.1,
            round_index=rnd,
            floor_armed=armed,
        )
        assert step_a.alpha_sensing == step_b.alpha_sensing
        assert step_a.sensing_target == step_b.sensing_target
        assert step_a.alpha_movement == step_b.alpha_movement
        assert step_a.movement_target == step_b.movement_target
        base, explicit = step_a.controller, step_b.controller
    assert base.sensing_armed_gate is False
    assert explicit.sensing_armed_gate is False


def test_drh4_gated_target_unarmed_hold_at_any_round():
    # Gate ON + unarmed: the sensing target is the BASE value at every round,
    # including far past the wall-clock anneal end (2000 in the helper) --
    # the DR-H4 schedule bypass. Movement stays on the schedule.
    controller = _drrel_controller(sensing_armed_gate=True)
    for rnd in (0, 799, 1200, 2500, 3999):
        step = controller.update(
            measured_movement_entropy=0.4,
            measured_sensing_entropy=0.1,
            round_index=rnd,
            floor_armed=False,
        )
        assert step.sensing_target == 0.3
    # Movement target at r900 follows the unchanged schedule:
    # 0.5 * (2000 - 900) / (2000 - 800).
    step = controller.update(
        measured_movement_entropy=0.4,
        measured_sensing_entropy=0.1,
        round_index=900,
        floor_armed=False,
    )
    assert step.movement_target == pytest.approx(0.5 * 1100.0 / 1200.0)


def test_drh4_armed_pin_zeroes_residual_alpha_and_holds():
    # The AM-E1 pin: an armed step forces alpha_sensing to exactly 0.0 even
    # from a NONZERO residual (the anti-ratchet property the audit simulated
    # at ~0.62 residual bonus without it), and consecutive armed rounds stay
    # pinned. Target is 0.0 while armed.
    controller = _drrel_controller(sensing_armed_gate=True, alpha_sensing=0.75)
    step = controller.update(
        measured_movement_entropy=0.4,
        measured_sensing_entropy=0.02,
        round_index=1500,
        floor_armed=True,
    )
    assert step.alpha_sensing == 0.0
    assert step.sensing_target == 0.0
    assert step.controller.alpha_sensing == 0.0
    again = step.controller.update(
        measured_movement_entropy=0.4,
        measured_sensing_entropy=0.001,
        round_index=1501,
        floor_armed=True,
    )
    assert again.alpha_sensing == 0.0
    assert again.sensing_target == 0.0


def test_drh4_dearm_reengages_and_winds_fresh_from_zero():
    controller = _drrel_controller(sensing_armed_gate=True, alpha_sensing=0.75)
    armed = controller.update(
        measured_movement_entropy=0.4,
        measured_sensing_entropy=0.02,
        round_index=1500,
        floor_armed=True,
    )
    released = armed.controller.update(
        measured_movement_entropy=0.4,
        measured_sensing_entropy=0.05,
        round_index=1501,
        floor_armed=False,
    )
    assert released.sensing_target == 0.3
    # Winds fresh from 0 at the code's own arithmetic order.
    assert released.alpha_sensing == min(2.0, max(0.0, 0.0 + 0.02 * (0.3 - 0.05)))


def test_drh4_alpha_recursion_across_arm_sequence_hand_case():
    # Six-step synthetic arm/de-arm/re-entry sequence, alpha hand-computed by
    # mirroring the controller's exact arithmetic (candidate = alpha + lr *
    # (target_now - H); proj to [0, cap]; armed => pinned 0.0). Includes a
    # NONZERO-alpha arm (the AM-H4-9 gated-run possibility the AM-E1 pin must
    # zero) at step 3.
    sequence = (
        # (H_sensing, floor_armed)
        (0.10, False),
        (0.10, False),
        (0.40, True),
        (0.02, True),
        (0.05, False),
        (0.10, False),
    )
    controller = _drrel_controller(sensing_armed_gate=True)
    expected_alpha = 0.0
    for index, (h_sense, armed) in enumerate(sequence):
        step = controller.update(
            measured_movement_entropy=0.4,
            measured_sensing_entropy=h_sense,
            round_index=1000 + index,
            floor_armed=armed,
        )
        if armed:
            expected_alpha = 0.0
            assert step.sensing_target == 0.0
        else:
            expected_alpha = min(
                2.0, max(0.0, expected_alpha + 0.02 * (0.3 - h_sense))
            )
            assert step.sensing_target == 0.3
        assert step.alpha_sensing == expected_alpha
        controller = step.controller
    # The pre-arm wind-up was genuinely nonzero before the pin zeroed it.
    assert expected_alpha > 0.0


def test_drh4_controller_gate_validation():
    with pytest.raises(TypeError, match="sensing_armed_gate must be a bool"):
        _drrel_controller(sensing_armed_gate=1)
    with pytest.raises(ValueError, match="requires sensing_target"):
        _drrel_controller(sensing_target=None, sensing_armed_gate=True)
    controller = _drrel_controller(sensing_armed_gate=True)
    for bad in (1, None, "True"):
        with pytest.raises(TypeError, match="floor_armed must be a bool"):
            controller.update(
                measured_movement_entropy=0.4,
                measured_sensing_entropy=0.1,
                round_index=0,
                floor_armed=bad,
            )


def test_drh4_successor_threads_gate_constant():
    controller = _drrel_controller(sensing_armed_gate=True)
    step = controller.update(
        measured_movement_entropy=0.4,
        measured_sensing_entropy=0.1,
        round_index=0,
        floor_armed=False,
    )
    assert step.controller.sensing_armed_gate is True


# ---------------- update config: coupling + threading ----------------


_DRH4_UPDATE_SUSTAIN_KWARGS = dict(
    entropy_sustain_learning_rate=0.02,
    movement_entropy_sustain_target=0.5,
    sensing_entropy_sustain_target=0.3,
    entropy_sustain_hold_rounds=1,
    entropy_sustain_anneal_rounds=2,
    entropy_sustain_alpha_cap=2.0,
    lambda_ceiling=5.0,
    armed_lambda_floor=3.0,
)


def test_drh4_update_config_gate_coupling_and_threads():
    base = _S25UUpdateConfig.from_decision_records()
    assert base.sensing_sustain_armed_gate is False
    config = dataclasses.replace(
        base, sensing_sustain_armed_gate=True, **_DRH4_UPDATE_SUSTAIN_KWARGS
    )
    assert config.make_entropy_controller().sensing_armed_gate is True
    # Gate OFF threads False.
    off = dataclasses.replace(base, **_DRH4_UPDATE_SUSTAIN_KWARGS)
    assert off.make_entropy_controller().sensing_armed_gate is False
    with pytest.raises(TypeError, match="sensing_sustain_armed_gate must be a bool"):
        dataclasses.replace(
            base, sensing_sustain_armed_gate=1, **_DRH4_UPDATE_SUSTAIN_KWARGS
        )
    with pytest.raises(ValueError, match="requires .*armed_lambda_floor"):
        dataclasses.replace(
            base,
            sensing_sustain_armed_gate=True,
            **dict(
                _DRH4_UPDATE_SUSTAIN_KWARGS,
                armed_lambda_floor=None,
                lambda_ceiling=None,
            ),
        )
    with pytest.raises(
        ValueError, match="requires .*sensing_entropy_sustain_target"
    ):
        dataclasses.replace(
            base,
            sensing_sustain_armed_gate=True,
            **dict(_DRH4_UPDATE_SUSTAIN_KWARGS, sensing_entropy_sustain_target=None),
        )
    with pytest.raises(
        ValueError, match="requires .*entropy_sustain_learning_rate"
    ):
        dataclasses.replace(
            base,
            sensing_sustain_armed_gate=True,
            lambda_ceiling=5.0,
            armed_lambda_floor=3.0,
        )


def test_drh4_a10_gate_mismatch_rejected():
    model, collected = _s25u_model_and_batch()
    config = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(),
        sensing_sustain_armed_gate=True,
        **_DRH4_UPDATE_SUSTAIN_KWARGS,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, eps=config.adam_epsilon
    )
    mismatched = _drrel_controller(
        hold_rounds=1, anneal_end_rounds=2, sensing_armed_gate=False
    )
    with pytest.raises(ValueError, match="entropy controller constants"):
        _s25u_update(
            model, collected.batch, config, optimizer=optimizer,
            controller=config.make_controller(), episode_count=4,
            round_index=0, entropy_controller=mismatched,
        )


def test_drh4_update_summary_carries_gate_flag_and_gated_target():
    model, collected = _s25u_model_and_batch()
    config = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(),
        sensing_sustain_armed_gate=True,
        **_DRH4_UPDATE_SUSTAIN_KWARGS,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, eps=config.adam_epsilon
    )
    # Unarmed at a round PAST the wall-clock anneal end (2): the gate holds
    # the target at 0.3 where the schedule would give 0.0.
    result = _s25u_update(
        model, collected.batch, config, optimizer=optimizer,
        controller=config.make_controller(), episode_count=4,
        round_index=5, entropy_controller=config.make_entropy_controller(),
        floor_armed=False,
    )
    assert result.summary["sensing_sustain_armed_gate"] is True
    assert result.summary["entropy_sustain_sensing_target"] == 0.3
    # Armed: target 0.0 and alpha pinned 0.0 (from a residual alpha state).
    model_b, collected_b = _s25u_model_and_batch(seed=72)
    optimizer_b = torch.optim.Adam(
        model_b.parameters(), lr=config.learning_rate, eps=config.adam_epsilon
    )
    residual = dataclasses.replace(
        config.make_entropy_controller(), alpha_sensing=0.75
    )
    armed_result = _s25u_update(
        model_b, collected_b.batch, config, optimizer=optimizer_b,
        controller=config.make_controller(), episode_count=4,
        round_index=5, entropy_controller=residual, floor_armed=True,
    )
    assert armed_result.summary["sensing_sustain_armed_gate"] is True
    assert armed_result.summary["entropy_sustain_sensing_target"] == 0.0
    assert armed_result.summary["entropy_sustain_alpha_sensing"] == 0.0
    assert armed_result.entropy_controller.alpha_sensing == 0.0


def test_drh4_update_summary_gate_off_flag_false_and_schedule_kept():
    model, collected = _s25u_model_and_batch()
    config = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(), **_DRH4_UPDATE_SUSTAIN_KWARGS
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.learning_rate, eps=config.adam_epsilon
    )
    # Gate OFF + floor_armed=True (the T1-D11 shape: armed floor without the
    # gate): the flag is ignored by the entropy controller -- the sensing
    # target follows the wall-clock schedule (0.0 at round 5 >= anneal end 2).
    result = _s25u_update(
        model, collected.batch, config, optimizer=optimizer,
        controller=config.make_controller(), episode_count=4,
        round_index=5, entropy_controller=config.make_entropy_controller(),
        floor_armed=True,
    )
    assert result.summary["sensing_sustain_armed_gate"] is False
    assert result.summary["entropy_sustain_sensing_target"] == 0.0


# ---------------- run config + driver e2e ----------------


_DRH4_RUN_SUSTAIN_KWARGS = dict(
    entropy_sustain_learning_rate=0.02,
    movement_entropy_sustain_target=0.5,
    sensing_entropy_sustain_target=0.3,
    entropy_sustain_hold_rounds=0,
    entropy_sustain_anneal_rounds=2,
    entropy_sustain_alpha_cap=2.0,
    lambda_ceiling=5.0,
)


def test_drh4_run_config_gate_coupling():
    accepted = _drh3_run_config(
        sensing_sustain_armed_gate=True,
        **_DRH4_RUN_SUSTAIN_KWARGS,
        **_DRH3_ARM_KWARGS,
    )
    assert accepted.sensing_sustain_armed_gate is True
    with pytest.raises(TypeError, match="sensing_sustain_armed_gate must be a bool"):
        _drh3_run_config(
            sensing_sustain_armed_gate=1,
            **_DRH4_RUN_SUSTAIN_KWARGS,
            **_DRH3_ARM_KWARGS,
        )
    with pytest.raises(ValueError, match="requires .*armed_lambda_floor"):
        _drh3_run_config(
            sensing_sustain_armed_gate=True, **_DRH4_RUN_SUSTAIN_KWARGS
        )
    with pytest.raises(
        ValueError, match="requires .*sensing_entropy_sustain_target"
    ):
        _drh3_run_config(
            sensing_sustain_armed_gate=True,
            **dict(_DRH4_RUN_SUSTAIN_KWARGS, sensing_entropy_sustain_target=None),
            **_DRH3_ARM_KWARGS,
        )
    with pytest.raises(
        ValueError, match="requires .*entropy_sustain_learning_rate"
    ):
        _drh3_run_config(sensing_sustain_armed_gate=True, **_DRH3_ARM_KWARGS)
    # Default stays OFF.
    assert (
        _drh3_run_config(**_DRH3_ARM_KWARGS).sensing_sustain_armed_gate is False
    )


def test_drh4_driver_e2e_gate_on_holds_target_and_records_flag(tmp_path):
    # Tiny end-to-end run with the gate ON: never arms (an untrained policy
    # cannot satisfy the arm conjuncts), so every round is unarmed and the
    # recorded sensing target must be held at 0.3 EVEN past the wall-clock
    # anneal end (2) -- the schedule bypass observed end-to-end -- with the
    # unconditional flag key True on every curve row. A gate-OFF twin records
    # the annealed 0.0 target from round >= 2 and the flag False.
    parent_on = tmp_path / "gate_on"
    parent_on.mkdir()
    config_on = _drh3_run_config(
        run_id="drh4_on", update_rounds=6, state_save_every=3,
        result_parent=parent_on,
        sensing_sustain_armed_gate=True,
        **_DRH4_RUN_SUSTAIN_KWARGS,
        **_DRH3_ARM_KWARGS,
    )
    run_dir_on = _s25u_run_training(config_on)
    rows_on = [
        json.loads(line)
        for line in (run_dir_on / "training_curve.jsonl").read_text(
            encoding="utf-8"
        ).strip().split("\n")
    ]
    assert len(rows_on) == 6
    for row in rows_on:
        assert row["sensing_sustain_armed_gate"] is True
        assert row["lambda_floor_armed"] is False
        assert row["entropy_sustain_sensing_target"] == 0.3
    parent_off = tmp_path / "gate_off"
    parent_off.mkdir()
    config_off = _drh3_run_config(
        run_id="drh4_off", update_rounds=6, state_save_every=3,
        result_parent=parent_off,
        **_DRH4_RUN_SUSTAIN_KWARGS,
        **_DRH3_ARM_KWARGS,
    )
    run_dir_off = _s25u_run_training(config_off)
    rows_off = [
        json.loads(line)
        for line in (run_dir_off / "training_curve.jsonl").read_text(
            encoding="utf-8"
        ).strip().split("\n")
    ]
    for row in rows_off:
        assert row["sensing_sustain_armed_gate"] is False
    assert all(row["entropy_sustain_sensing_target"] == 0.0
               for row in rows_off if row["round_index"] >= 2)


def test_drh4_gate_flag_changes_config_fingerprint():
    # V-4: the gate flag participates in the config fingerprint (the
    # fingerprint hashes the FULL dataclass payloads), so a gate-ON checkpoint
    # cannot resume gate-OFF or vice versa.
    base_kwargs = dict(
        run_id="drh4_fp", seed=151, update_rounds=8, episodes_per_round=2,
        steps_per_episode=6, probe_every=0, state_save_every=8, tier="T0",
    )
    config_off = _S25URunConfig(
        **base_kwargs, **_DRH4_RUN_SUSTAIN_KWARGS, **_DRH3_ARM_KWARGS
    )
    config_on = _S25URunConfig(
        **base_kwargs,
        sensing_sustain_armed_gate=True,
        **_DRH4_RUN_SUSTAIN_KWARGS,
        **_DRH3_ARM_KWARGS,
    )
    update_on = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(),
        sensing_sustain_armed_gate=True,
        **_DRH4_UPDATE_SUSTAIN_KWARGS,
    )
    update_off = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(), **_DRH4_UPDATE_SUSTAIN_KWARGS
    )
    assert _s25u_config_fingerprint(config_on, update_on) != (
        _s25u_config_fingerprint(config_off, update_off)
    )


# ===========================================================================
# SECTION DRH5 -- DR-H5 (SHIFT 25): lever J (equilibrium-certifiability budget
# d_ep, run-config value, default 0.5 = DR-D2 byte-identical) + lever K
# (limbo-scoped armed movement-plasticity re-engagement: driver-owned midband
# containment trigger + latched recovery mode + movement sustain target
# override with the AM-E1 stateless pin). Countersign wf_c1373353-9a9
# (GRANTED_WITH_AMENDMENT, ALL TEN GATES PASS; AM-1..AM-4 folded). The
# executable historical regression lives in
# docs/evidence/drh5_trigger_regression.py ({no engage, no engage, r1884,
# vacuous} on the committed s128/s130/s131/s129 traces); the pins below are
# its hermetic hand-case mirrors.
# ===========================================================================


def _drh5_state(**overrides):
    values = dict(
        arm_window_rounds=2, arm_success_threshold=0.8,
        arm_hazard_threshold=0.35, arm_sensing_threshold=0.05,
        dearm_window_rounds=4, dearm_success_threshold=0.1,
        constraint_warmup_rounds=0,
        midband_window_rounds=4, midband_success_lo=0.25,
        midband_success_hi=0.75, midband_containment_widen=0.10,
        midband_hazard_cap=0.05, midband_max_rounds=3,
    )
    values.update(overrides)
    return _DRH3ArmState(**values)


def _drh5_feed(state, rounds):
    """Feed (success, hazard, sensing) triples starting at round 0."""

    outcomes = []
    for index, (success, hazard, sensing) in enumerate(rounds):
        armed = state.observe_round(
            round_index=index, success_rate=success,
            hazard_mean=hazard, sensing_mean=sensing,
        )
        outcomes.append((armed, state.recovery_active))
    return outcomes


_DRH5_ARMING = [(0.9, 0.0, 0.5), (0.9, 0.0, 0.5)]  # arms at round 1 (W=2)
_DRH5_MID = (0.5, 0.0, 0.5)


# ---------------- lever K trigger: hand cases ----------------


def test_drh5_trigger_fires_on_midband_plateau():
    # Armed, then a genuine per-round mid plateau: the containment window
    # (W_mid=4) fills with in-band rounds once the 0.9 arming rounds roll
    # out; the engage round itself is a recovery round (AM-H15 convention).
    state = _drh5_state()
    _drh5_feed(state, _DRH5_ARMING + [_DRH5_MID] * 4)
    engage = [e for e in state.events if e["event"] == "recovery_engage"]
    assert len(engage) == 1
    assert engage[0]["round_index"] == 5  # first full all-in-band window
    assert engage[0]["window_success"] == pytest.approx(0.5)
    assert engage[0]["window_hazard"] == 0.0
    assert state.recovery_active is True
    assert state.recovery_rounds_used == 1
    assert state.recovery_engagements == 1


def test_drh5_trigger_blocked_by_containment_mixed_transit():
    # Window MEAN in band but per-round successes alternate 1.0/0.0 (the
    # s128-transit shape): the containment conjunct must block every fire --
    # the mean-only form was killed by the regression grid (it would have
    # de-armed s128's HELD pass).
    state = _drh5_state()
    mixed = [(1.0, 0.0, 0.5), (0.0, 0.0, 0.5)] * 6
    _drh5_feed(state, _DRH5_ARMING + mixed)
    assert not [e for e in state.events if e["event"] == "recovery_engage"]
    assert state.recovery_active is False


def test_drh5_trigger_blocked_by_hazard_cap():
    state = _drh5_state()
    hot = [(0.5, 0.2, 0.5)] * 6  # window-mean hazard 0.2 > cap 0.05
    _drh5_feed(state, _DRH5_ARMING + hot)
    assert not [e for e in state.events if e["event"] == "recovery_engage"]


def test_drh5_trigger_blocked_when_unarmed():
    # Perfect mid plateau but the seed never arms (success 0.5 < 0.8):
    # an armed-gated trigger is vacuously silent (the s129 control).
    state = _drh5_state()
    _drh5_feed(state, [_DRH5_MID] * 12)
    assert state.armed is False
    assert not [e for e in state.events if e["event"] == "recovery_engage"]
    assert state.recovery_active is False


def test_drh5_trigger_requires_full_window():
    # W_mid=5 with arming rounds at 0.8 (inside the widened band [0.15,0.85]):
    # the first full window is at round 4 and qualifies (mean 0.62 in band);
    # round 3's window is short by one and must NOT fire.
    state = _drh5_state(
        midband_window_rounds=5,
        arm_window_rounds=2,
    )
    rounds = [(0.8, 0.0, 0.5), (0.8, 0.0, 0.5)] + [_DRH5_MID] * 3
    _drh5_feed(state, rounds)
    engage = [e for e in state.events if e["event"] == "recovery_engage"]
    assert len(engage) == 1
    assert engage[0]["round_index"] == 4
    assert engage[0]["window_success"] == pytest.approx((0.8 * 2 + 0.5 * 3) / 5)


def test_drh5_widen_boundary_is_inclusive():
    # Per-round success exactly at the widened bounds (0.15 / 0.85) stays
    # inside the containment band (closed interval).
    state = _drh5_state()
    edge = [(0.85, 0.0, 0.5), (0.15, 0.0, 0.5)] * 2  # mean 0.5, all at bounds
    _drh5_feed(state, _DRH5_ARMING + edge)
    engage = [e for e in state.events if e["event"] == "recovery_engage"]
    assert len(engage) == 1
    assert engage[0]["round_index"] == 5


# ---------------- lever K latch: exits, clearing, exhaustion ----------------


def test_drh5_exit_recovered_on_arm_window_success():
    # Engaged, then the trailing arm-window (W=2) success mean reaches the
    # arm's own bar 0.8: exit (a), the containment window CLEARED, the
    # per-engagement counter reset (V2-S2).
    state = _drh5_state(midband_max_rounds=50)
    _drh5_feed(state, _DRH5_ARMING + [_DRH5_MID] * 4)
    assert state.recovery_active is True
    for index in (6, 7):
        state.observe_round(
            round_index=index, success_rate=0.9, hazard_mean=0.0,
            sensing_mean=0.5,
        )
    exits = [e for e in state.events if e["event"] == "recovery_exit"]
    assert len(exits) == 1
    assert exits[0]["reason"] == "recovered"
    assert exits[0]["round_index"] == 7  # trailing-2 mean first >= 0.8 here
    assert exits[0]["rounds_used"] == 3
    assert state.recovery_active is False
    assert state.recovery_rounds_used == 0
    assert len(state._midband_window) == 0
    assert state.recovery_exhausted is False


def test_drh5_exit_dearm_via_hatch_takes_precedence():
    # While engaged, a deep-paralysis descent fires the UNTOUCHED 0.1 hatch;
    # the same round is recovery exit (b) -- the hatch takes precedence and
    # recovery requires armed.
    state = _drh5_state(midband_max_rounds=50, dearm_window_rounds=3)
    _drh5_feed(state, _DRH5_ARMING + [_DRH5_MID] * 4)
    assert state.recovery_active is True
    index = 6
    while state.armed:
        state.observe_round(
            round_index=index, success_rate=0.0, hazard_mean=0.0,
            sensing_mean=0.5,
        )
        index += 1
    exits = [e for e in state.events if e["event"] == "recovery_exit"]
    assert len(exits) == 1
    assert exits[0]["reason"] == "dearm"
    dearms = [e for e in state.events if e["event"] == "dearm"]
    assert dearms[-1]["round_index"] == exits[0]["round_index"]
    assert state.recovery_active is False


def test_drh5_exit_exhausted_sets_permanent_latch():
    # R_max=3: the engagement exhausts on its third recovery round, sets the
    # PERMANENT per-run latch, and a continuing perfect plateau can never
    # re-engage (provable termination; the row-3a signature).
    state = _drh5_state(midband_max_rounds=3)
    _drh5_feed(state, _DRH5_ARMING + [_DRH5_MID] * 4)  # engage at round 5
    for index in range(6, 20):
        state.observe_round(
            round_index=index, success_rate=0.5, hazard_mean=0.0,
            sensing_mean=0.5,
        )
    exits = [e for e in state.events if e["event"] == "recovery_exit"]
    assert len(exits) == 1
    assert exits[0]["reason"] == "exhausted"
    assert exits[0]["round_index"] == 7  # rounds 5,6,7 = 3 recovery rounds
    assert exits[0]["rounds_used"] == 3
    assert state.recovery_exhausted is True
    engage = [e for e in state.events if e["event"] == "recovery_engage"]
    assert len(engage) == 1  # never re-engaged despite 12 more in-band rounds


def test_drh5_window_cleared_on_exit_requires_full_refill():
    # After exit (a) the containment window is EMPTY: re-engagement needs a
    # full fresh window (no instant re-fire oscillation; V2-S2).
    state = _drh5_state(midband_max_rounds=50)
    _drh5_feed(state, _DRH5_ARMING + [_DRH5_MID] * 4)  # engage r5
    for index in (6, 7):  # exit (a) at r7
        state.observe_round(
            round_index=index, success_rate=0.9, hazard_mean=0.0,
            sensing_mean=0.5,
        )
    assert state.recovery_active is False
    engage_rounds = []
    for index in range(8, 16):
        state.observe_round(
            round_index=index, success_rate=0.5, hazard_mean=0.0,
            sensing_mean=0.5,
        )
        if state.recovery_active and len(engage_rounds) == 0:
            engage_rounds.append(index)
    # Window cleared at r7; four fresh in-band rounds fill it at r11.
    assert engage_rounds == [11]
    assert state.recovery_engagements == 2


def test_drh5_recovery_state_construction_validation():
    with pytest.raises(ValueError):
        _drh5_state(midband_window_rounds=0)  # params without window
    with pytest.raises(ValueError):
        _drh5_state(midband_success_lo=None)  # partial params
    with pytest.raises(ValueError):
        _drh5_state(midband_max_rounds=0)  # enabled trigger needs R_max
    off = _drh5_state(
        midband_window_rounds=0, midband_success_lo=None,
        midband_success_hi=None, midband_containment_widen=None,
        midband_hazard_cap=None, midband_max_rounds=0,
    )
    assert off.recovery_enabled is False


# ---------------- lever K state: checkpoint round-trip ----------------


def test_drh5_payload_roundtrip_preserves_recovery_state():
    state = _drh5_state(midband_max_rounds=50)
    _drh5_feed(state, _DRH5_ARMING + [_DRH5_MID] * 4)  # engaged, mid-window
    payload = state.to_payload()
    restored = _drh5_state(midband_max_rounds=50)
    restored.restore_payload(payload)
    assert restored.recovery_active is True
    assert restored.recovery_rounds_used == state.recovery_rounds_used
    assert restored.recovery_engagements == 1
    assert restored.recovery_exhausted is False
    assert list(restored._midband_window) == list(state._midband_window)
    # AM-H15 strongest form: both continue byte-identically.
    for index in (6, 7):
        state.observe_round(
            round_index=index, success_rate=0.9, hazard_mean=0.0,
            sensing_mean=0.5,
        )
        restored.observe_round(
            round_index=index, success_rate=0.9, hazard_mean=0.0,
            sensing_mean=0.5,
        )
    assert state.events == restored.events
    assert state.to_payload() == restored.to_payload()


def test_drh5_payload_off_form_has_no_recovery_key():
    # OFF-default byte-identity: a trigger-disabled state's payload carries
    # exactly the pre-DR-H5 key set.
    off = _drh5_state(
        midband_window_rounds=0, midband_success_lo=None,
        midband_success_hi=None, midband_containment_widen=None,
        midband_hazard_cap=None, midband_max_rounds=0,
    )
    _drh5_feed(off, _DRH5_ARMING)
    assert sorted(off.to_payload().keys()) == [
        "arm_window", "armed", "dearm_window", "events",
    ]


def test_drh5_restore_missing_recovery_loud_reject():
    state = _drh5_state()
    _drh5_feed(state, _DRH5_ARMING)
    payload = state.to_payload()
    del payload["recovery"]
    fresh = _drh5_state()
    with pytest.raises(ValueError, match="missing the recovery state"):
        fresh.restore_payload(payload)


def test_drh5_restore_unexpected_recovery_reject():
    state = _drh5_state()
    _drh5_feed(state, _DRH5_ARMING)
    payload = state.to_payload()
    off = _drh5_state(
        midband_window_rounds=0, midband_success_lo=None,
        midband_success_hi=None, midband_containment_widen=None,
        midband_hazard_cap=None, midband_max_rounds=0,
    )
    with pytest.raises(ValueError, match="config/state mismatch"):
        off.restore_payload(payload)


def test_drh5_restore_forged_recovery_rejects():
    state = _drh5_state()
    _drh5_feed(state, _DRH5_ARMING)
    base = state.to_payload()
    oversized = {**base, "recovery": {**base["recovery"],
                 "midband_window": [[0.5, 0.0]] * 99}}
    with pytest.raises(ValueError, match="exceeds"):
        _drh5_state().restore_payload(oversized)
    nonpair = {**base, "recovery": {**base["recovery"],
               "midband_window": [[0.5, 0.0, 0.1]]}}
    with pytest.raises(ValueError, match="pairs"):
        _drh5_state().restore_payload(nonpair)
    hostile = {**base, "recovery": {**base["recovery"], "active": 1}}
    with pytest.raises(ValueError, match="must be a bool"):
        _drh5_state().restore_payload(hostile)
    missing = {**base, "recovery": {k: v for k, v in base["recovery"].items()
               if k != "exhausted"}}
    with pytest.raises(ValueError, match="missing exhausted"):
        _drh5_state().restore_payload(missing)
    # L1-MF-1 (impl-audit): counter DOMAIN forgeries — an overrun rounds_used
    # would let an engagement run past R_max; an oversized one forges the
    # permanent exhaustion latch (a graded routing input); bools sneak through
    # bare int.__index__; active-without-armed is machine-unreachable.
    bool_used = {**base, "recovery": {**base["recovery"], "rounds_used": True}}
    with pytest.raises(ValueError, match="counters must be ints"):
        _drh5_state().restore_payload(bool_used)
    negative = {**base, "recovery": {**base["recovery"], "rounds_used": -1000}}
    with pytest.raises(ValueError, match="within"):
        _drh5_state().restore_payload(negative)
    overrun = {**base, "recovery": {**base["recovery"], "rounds_used": 10**6}}
    with pytest.raises(ValueError, match="within"):
        _drh5_state().restore_payload(overrun)
    neg_eng = {**base, "recovery": {**base["recovery"], "engagements": -5}}
    with pytest.raises(ValueError, match="nonnegative"):
        _drh5_state().restore_payload(neg_eng)
    unreachable = {
        **base, "armed": False,
        "recovery": {**base["recovery"], "active": True, "rounds_used": 1},
    }
    with pytest.raises(ValueError, match="machine-unreachable"):
        _drh5_state().restore_payload(unreachable)
    stale_used = {
        **base,
        "recovery": {**base["recovery"], "active": False, "rounds_used": 2},
    }
    with pytest.raises(ValueError, match="must be 0 while inactive"):
        _drh5_state().restore_payload(stale_used)


# ---------------- lever K controller: override + pin ----------------


def test_drh5_controller_gate_off_flag_ignored_byte_identical():
    base = _drrel_controller()
    explicit = _drrel_controller(movement_recovery_gate=False)
    for rnd, active in ((0, False), (900, True), (2100, True), (3000, False)):
        step_a = base.update(
            measured_movement_entropy=0.2,
            measured_sensing_entropy=0.1,
            round_index=rnd,
        )
        step_b = explicit.update(
            measured_movement_entropy=0.2,
            measured_sensing_entropy=0.1,
            round_index=rnd,
            movement_recovery_active=active,
        )
        assert step_a.alpha_movement == step_b.alpha_movement
        assert step_a.movement_target == step_b.movement_target
        assert step_a.alpha_sensing == step_b.alpha_sensing
        base, explicit = step_a.controller, step_b.controller
    assert explicit.movement_recovery_gate is False


def test_drh5_controller_recovery_override_holds_base_target():
    # Gate ON + engaged: the movement target is the BASE value at every round
    # including far past the wall-clock anneal end (2000 in the helper), and
    # alpha winds by the unchanged dual law.
    controller = _drrel_controller(movement_recovery_gate=True)
    for rnd in (900, 2100, 3999):
        step = controller.update(
            measured_movement_entropy=0.1,
            measured_sensing_entropy=0.1,
            round_index=rnd,
            movement_recovery_active=True,
        )
        assert step.movement_target == 0.5
    step = controller.update(
        measured_movement_entropy=0.1,
        measured_sensing_entropy=0.1,
        round_index=2500,
        movement_recovery_active=True,
    )
    assert step.alpha_movement == min(2.0, max(0.0, 0.0 + 0.02 * (0.5 - 0.1)))


def test_drh5_controller_pin_post_anneal_inactive():
    # Gate ON + NOT engaged + round >= anneal end: alpha_movement is pinned
    # to exactly 0.0 even from a nonzero residual (the AM-E1 anti-ratchet
    # pin on the movement side) and stays pinned.
    controller = _drrel_controller(
        movement_recovery_gate=True, alpha_movement=0.75
    )
    step = controller.update(
        measured_movement_entropy=0.01,
        measured_sensing_entropy=0.1,
        round_index=2000,
        movement_recovery_active=False,
    )
    assert step.alpha_movement == 0.0
    assert step.movement_target == 0.0
    again = step.controller.update(
        measured_movement_entropy=0.001,
        measured_sensing_entropy=0.1,
        round_index=2001,
        movement_recovery_active=False,
    )
    assert again.alpha_movement == 0.0


def test_drh5_controller_schedule_path_pre_anneal_inactive():
    # Gate ON + NOT engaged + round < anneal end: byte-identical to the
    # unchanged lever-E schedule path.
    gated = _drrel_controller(movement_recovery_gate=True)
    plain = _drrel_controller()
    step_g = gated.update(
        measured_movement_entropy=0.2, measured_sensing_entropy=0.1,
        round_index=900, movement_recovery_active=False,
    )
    step_p = plain.update(
        measured_movement_entropy=0.2, measured_sensing_entropy=0.1,
        round_index=900,
    )
    assert step_g.alpha_movement == step_p.alpha_movement
    assert step_g.movement_target == step_p.movement_target


def test_drh5_controller_gate_coupling_and_hostile_flags():
    with pytest.raises(ValueError, match="movement_recovery_gate"):
        _drrel_controller(movement_recovery_gate=True, movement_target=None)
    with pytest.raises(TypeError, match="movement_recovery_gate"):
        _drrel_controller(movement_recovery_gate=1)
    controller = _drrel_controller(movement_recovery_gate=True)
    for hostile in (1, 0, None, "True"):
        with pytest.raises(TypeError, match="movement_recovery_active"):
            controller.update(
                measured_movement_entropy=0.2,
                measured_sensing_entropy=0.1,
                round_index=100,
                movement_recovery_active=hostile,
            )


def test_drh5_alpha_recursion_engage_exit_hand_case():
    # Engage -> two engaged rounds -> exit (pin): alpha hand-computed by the
    # controller's exact arithmetic, then zeroed at the exit round.
    controller = _drrel_controller(movement_recovery_gate=True)
    alpha = 0.0
    for rnd, entropy in ((2100, 0.05), (2101, 0.12)):
        step = controller.update(
            measured_movement_entropy=entropy,
            measured_sensing_entropy=0.1,
            round_index=rnd,
            movement_recovery_active=True,
        )
        alpha = min(2.0, max(0.0, alpha + 0.02 * (0.5 - entropy)))
        assert step.alpha_movement == alpha
        controller = step.controller
    released = controller.update(
        measured_movement_entropy=0.3,
        measured_sensing_entropy=0.1,
        round_index=2102,
        movement_recovery_active=False,
    )
    assert released.alpha_movement == 0.0
    assert released.controller.alpha_movement == 0.0


# ---------------- config coupling: update + run configs ----------------


def test_drh5_update_config_gate_coupling_matrix():
    base = dict(_DRH4_UPDATE_SUSTAIN_KWARGS)
    good = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(),
        movement_recovery_gate=True, **base,
    )
    assert good.make_entropy_controller().movement_recovery_gate is True
    off = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(), **base
    )
    assert off.make_entropy_controller().movement_recovery_gate is False
    with pytest.raises(ValueError, match="movement_entropy_sustain_target"):
        dataclasses.replace(
            _S25UUpdateConfig.from_decision_records(),
            movement_recovery_gate=True,
            **{**base, "movement_entropy_sustain_target": None},
        )
    with pytest.raises(ValueError, match="armed_lambda_floor"):
        dataclasses.replace(
            _S25UUpdateConfig.from_decision_records(),
            movement_recovery_gate=True,
            **{**base, "armed_lambda_floor": None},
        )
    with pytest.raises(ValueError, match="entropy_sustain_learning_rate"):
        dataclasses.replace(
            _S25UUpdateConfig.from_decision_records(),
            movement_recovery_gate=True,
            **{
                **base,
                "entropy_sustain_learning_rate": 0.0,
                "movement_entropy_sustain_target": None,
                "sensing_entropy_sustain_target": None,
                "entropy_sustain_hold_rounds": 0,
                "entropy_sustain_anneal_rounds": 0,
                "entropy_sustain_alpha_cap": None,
            },
        )
    with pytest.raises(TypeError, match="movement_recovery_gate"):
        dataclasses.replace(
            _S25UUpdateConfig.from_decision_records(),
            movement_recovery_gate=1, **base,
        )


_DRH5_MIDBAND_RUN_KWARGS = dict(
    movement_recovery_gate=True,
    midband_recovery_window_rounds=4,
    midband_recovery_success_lo=0.25,
    midband_recovery_success_hi=0.75,
    midband_recovery_containment_widen=0.10,
    midband_recovery_hazard_cap=0.05,
    midband_recovery_max_rounds=6,
)


def _drh5_run_config(**overrides):
    kwargs = dict(
        run_id="drh5_cfg", seed=161, update_rounds=8, episodes_per_round=2,
        steps_per_episode=6, probe_every=0, state_save_every=8, tier="T0",
        **_DRH4_RUN_SUSTAIN_KWARGS, **_DRH3_ARM_KWARGS,
    )
    kwargs.update(overrides)
    return _S25URunConfig(**kwargs)


def test_drh5_run_config_budget_validation():
    assert _drh5_run_config().hazard_budget == 0.5  # DR-D2 default
    tightened = _drh5_run_config(hazard_budget=0.25)
    assert tightened.hazard_budget == 0.25
    assert type(tightened.hazard_budget) is float
    with pytest.raises(ValueError, match="must be <= 0.5"):
        _drh5_run_config(hazard_budget=0.6)
    with pytest.raises(ValueError):
        _drh5_run_config(hazard_budget=0.0)
    with pytest.raises(ValueError):
        _drh5_run_config(hazard_budget=float("nan"))
    with pytest.raises(TypeError):
        _drh5_run_config(hazard_budget=True)
    with pytest.raises(ValueError):
        _drh5_run_config(hazard_budget=10**400)


def test_drh5_run_config_midband_coupling_matrix():
    good = _drh5_run_config(**_DRH5_MIDBAND_RUN_KWARGS)
    assert good.midband_recovery_window_rounds == 4
    assert type(good.midband_recovery_success_lo) is float
    assert type(good.midband_recovery_max_rounds) is int
    with pytest.raises(ValueError, match="movement_recovery_gate=True"):
        _drh5_run_config(
            **{**_DRH5_MIDBAND_RUN_KWARGS, "movement_recovery_gate": False}
        )
    with pytest.raises(ValueError, match="movement_entropy_sustain_target"):
        _drh5_run_config(
            **_DRH5_MIDBAND_RUN_KWARGS,
            movement_entropy_sustain_target=None,
        )
    with pytest.raises(ValueError, match="ALL of"):
        _drh5_run_config(
            **{**_DRH5_MIDBAND_RUN_KWARGS, "midband_recovery_hazard_cap": None}
        )
    with pytest.raises(ValueError, match="0 <= lo < hi <= 1"):
        _drh5_run_config(
            **{
                **_DRH5_MIDBAND_RUN_KWARGS,
                "midband_recovery_success_lo": 0.75,
                "midband_recovery_success_hi": 0.25,
            }
        )
    with pytest.raises(ValueError, match="strictly below"):
        _drh5_run_config(
            **{**_DRH5_MIDBAND_RUN_KWARGS, "midband_recovery_success_hi": 0.9}
        )
    with pytest.raises(ValueError, match="must be <= update_rounds"):
        _drh5_run_config(
            **{
                **_DRH5_MIDBAND_RUN_KWARGS,
                "midband_recovery_window_rounds": 9,
            }
        )
    with pytest.raises(ValueError, match="must be <= update_rounds"):
        _drh5_run_config(
            **{**_DRH5_MIDBAND_RUN_KWARGS, "midband_recovery_max_rounds": 9}
        )


def test_drh5_update_config_budget_threading_and_law():
    tightened = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(), hazard_budget=0.25
    )
    controller = tightened.make_controller()
    assert controller.budget == 0.25
    # One PI step at the new budget follows the exact law: blind crossing
    # (J=1.0) winds the integral by K_I * 0.75 = 0.0375.
    step = controller.update(observed_episodic_cost=1.0)
    assert step.controller.integral == pytest.approx(
        0.1 + 0.05 * (1.0 - 0.25)
    )


def test_drh5_new_fields_change_config_fingerprint():
    # V-4: budget + gate participate in the fingerprint, so a checkpoint
    # cannot resume under a different budget or gate state.
    base_kwargs = dict(
        run_id="drh5_fp", seed=162, update_rounds=8, episodes_per_round=2,
        steps_per_episode=6, probe_every=0, state_save_every=8, tier="T0",
        **_DRH4_RUN_SUSTAIN_KWARGS, **_DRH3_ARM_KWARGS,
    )
    update_base = dataclasses.replace(
        _S25UUpdateConfig.from_decision_records(), **_DRH4_UPDATE_SUSTAIN_KWARGS
    )
    config_default = _S25URunConfig(**base_kwargs)
    config_tight = _S25URunConfig(**base_kwargs, hazard_budget=0.25)
    update_tight = dataclasses.replace(update_base, hazard_budget=0.25)
    assert _s25u_config_fingerprint(config_tight, update_tight) != (
        _s25u_config_fingerprint(config_default, update_base)
    )
    config_gate = _S25URunConfig(**base_kwargs, **_DRH5_MIDBAND_RUN_KWARGS)
    update_gate = dataclasses.replace(update_base, movement_recovery_gate=True)
    assert _s25u_config_fingerprint(config_gate, update_gate) != (
        _s25u_config_fingerprint(config_default, update_base)
    )


# =============================================================================
# SECTION DRCURR — DR-READINESS-CURRICULUM (SHIFT 27, countersigned): gate-row
# curriculum for readiness-C4 generalization. Three NEW aliased fork pairs at
# gate rows {0,3}/{4,7}/{0,7} (free rows only; row-disjoint from readiness
# (1,2) and held-out (5,6)); the AM-H13 arm-spec re-open (family scope +
# complete pairs + within-pair balance); the AM-C4 unconditional readiness
# exclusion; the conditional probe repoint; Phi_dz coverage + telescoping on
# the new zones. docs/decisions/DR-readiness-curriculum.md.
# =============================================================================

from raas_marl.final_training.stage25_driver import (
    _resolve_probe_scenarios as _drcurr_resolve_probe,
)

_DRCURR_CURRICULUM_TUPLE = (
    "risk_fork_curriculum_r0r3_lower",
    "risk_fork_curriculum_r0r3_upper",
    "risk_fork_curriculum_r4r7_lower",
    "risk_fork_curriculum_r4r7_upper",
    "risk_fork_curriculum_r0r7_lower",
    "risk_fork_curriculum_r0r7_upper",
)
_DRCURR_T1D14_TUPLE = (
    "risk_fork_train_lower",
    "risk_fork_train_upper",
) + _DRCURR_CURRICULUM_TUPLE
_DRCURR_PAIRS = (
    ("risk_fork_curriculum_r0r3_lower", "risk_fork_curriculum_r0r3_upper"),
    ("risk_fork_curriculum_r4r7_lower", "risk_fork_curriculum_r4r7_upper"),
    ("risk_fork_curriculum_r0r7_lower", "risk_fork_curriculum_r0r7_upper"),
)
_DRCURR_FREE_ROWS = {0, 3, 4, 7}
_DRCURR_RESERVED_ROWS = {1, 2, 5, 6}  # readiness (1,2) + held-out (5,6)

_DRCURR_ARM_KWARGS = dict(
    armed_lambda_floor=3.0, arm_window_rounds=2,
    arm_success_threshold=0.8, arm_hazard_threshold=0.35,
    arm_sensing_threshold=0.05, dearm_window_rounds=4,
    dearm_success_threshold=0.1,
)

# DR-CURRICULUM-FORM (SHIFT 29, countersigned): the bounded-N=2 breadth-reduced
# family T1-D15 runs — the anchor pair {3,4} + the interpolation pair {0,3}
# (4 scenarios / 2 fork pairs). F-84-safe by construction (each pair weight 0.5).
_DRCURR2_TUPLE = (
    "risk_fork_train_lower",
    "risk_fork_train_upper",
    "risk_fork_curriculum_r0r3_lower",
    "risk_fork_curriculum_r0r3_upper",
)


def _drcurr_run_config(**overrides):
    kwargs = dict(
        run_id="drcurr_cfg", seed=151, update_rounds=8, episodes_per_round=16,
        steps_per_episode=6, probe_every=0, state_save_every=8, tier="T0",
    )
    kwargs.update(overrides)
    return _S25URunConfig(**kwargs)


# ---------------- registry + geometry ----------------


def test_drcurr_registry_exact_and_separate_from_grade_surface():
    # The curriculum registry is the exact countersigned 6-tuple, mirrors
    # adjacent (lower, upper) per pair.
    assert fork_curriculum_scenarios() == _DRCURR_CURRICULUM_TUPLE
    # The pinned C1 grade-surface registry is NOT mutated (AM-H13 wiring rule):
    families = fork_hazard_layout_families()
    assert set(families) == {"training", "readiness"}
    for group_names in families.values():
        for name in group_names:
            assert name not in _DRCURR_CURRICULUM_TUPLE
    # I-3: no held-out id anywhere in the catalog.
    for name in stage23_scenario_catalog():
        assert "heldout" not in name and "held_out" not in name


def test_drcurr_gate_rows_free_and_row_disjoint_from_reserved():
    catalog = stage23_scenario_catalog()
    for name in _DRCURR_CURRICULUM_TUPLE:
        sc = catalog[name]
        gates = {r for r in range(sc.height) if (r, 4) not in set(sc.obstacles)}
        assert len(gates) == 2
        assert gates <= _DRCURR_FREE_ROWS
        assert not (gates & _DRCURR_RESERVED_ROWS)
        # exactly one hidden hazard, and it sits on one of the open gates
        assert len(sc.hidden_hazard_cells) == 1
        hz_row, hz_col = sc.hidden_hazard_cells[0]
        assert hz_col == 4 and hz_row in gates
        # same starts/goal as the whole fork family (observational uniformity)
        assert sc.starts == ((3, 0), (4, 0))
        assert sc.goals == ((3, 7),)


def test_drcurr_mirror_identity_byte_equality_every_fork_pair():
    # AM-C7(ii): the two mirrors of EVERY fork pair (train, readiness, and each
    # curriculum pair) differ ONLY in hidden_hazard_cells.
    catalog = stage23_scenario_catalog()
    families = fork_hazard_layout_families()
    pairs = (
        tuple(families["training"]),
        tuple(families["readiness"]),
    ) + _DRCURR_PAIRS
    for a_name, b_name in pairs:
        a, b = catalog[a_name], catalog[b_name]
        assert a.width == b.width and a.height == b.height
        assert a.starts == b.starts
        assert a.goals == b.goals
        assert a.obstacles == b.obstacles
        assert a.hidden_hazard_cells != b.hidden_hazard_cells


# ---------------- per-pair BAR-A/BAR-B (the Gate-1 executable checks) ----------


@pytest.mark.parametrize("pair", _DRCURR_PAIRS, ids=lambda p: p[0].rsplit("_", 2)[1])
def test_drcurr_per_pair_forces_sensing_and_winnable_executable(pair):
    # THE Gate-1 pin: every curriculum pair passes the FULL executable check
    # (union cut + every gate hazardous + no_sense mean hazard over budget +
    # winnable + the ACTUAL selective comparator at 0 hazard with success).
    result = verify_fork_family_forces_sensing(pair)
    assert result["bar_a_union_cut"] is True
    assert result["bar_a_all_gates_hazardous"] is True
    assert result["bar_a_no_sense_mean_hazard"] == pytest.approx(1.0)
    assert result["bar_a_no_sense_mean_hazard_exceeds_budget"] is True
    assert result["bar_b_all_layouts_winnable"] is True
    assert result["bar_b_selective_zero_hazard"] is True
    assert result["bar_b_selective_preserves_success"] is True
    assert result["forces_sensing_and_winnable"] is True


@pytest.mark.parametrize("pair", _DRCURR_PAIRS, ids=lambda p: p[0].rsplit("_", 2)[1])
def test_drcurr_independent_bfs_union_cut_and_winnable_per_pair(pair):
    # Independent executable BAR-A (does not trust verify_fork_family_...):
    # the union of the pair's hazards is a start->goal cut, while each single
    # layout leaves a safe route.
    assert _fork_union_is_start_goal_cut(pair) is True
    for name in pair:
        sc = stage23_scenario_catalog()[name]
        for start in sc.starts:
            assert _shortest_path(
                start,
                sc.goals,
                width=sc.width,
                height=sc.height,
                obstacles=sc.obstacles,
                avoid_cells=sc.hidden_hazard_cells,
            )


# ---------------- Phi_dz zone coverage on the curriculum pairs ----------------


def test_drcurr_zone_covers_gates_broad_and_mirror_identical():
    catalog = stage23_scenario_catalog()
    for a_name, b_name in _DRCURR_PAIRS:
        sc = catalog[a_name]
        zone_a = _stage24a_public_risk_zone_cells(a_name)
        zone_b = _stage24a_public_risk_zone_cells(b_name)
        # layout-agnostic: identical across the pair's aliased mirrors (M-5-safe)
        assert zone_a == zone_b
        zone = set(zone_a)
        gates = {(r, 4) for r in range(sc.height) if (r, 4) not in set(sc.obstacles)}
        assert len(gates) == 2
        assert gates <= zone
        assert set(sc.hidden_hazard_cells) <= zone
        # broader than the exact hazard => never pinpoints the hidden gate
        assert len(zone) > len(sc.hidden_hazard_cells)
        # spans >= 2 gate rows
        assert len({r for (r, c) in zone if (r, 4) in gates}) >= 2


@pytest.mark.parametrize(
    "scenario_name",
    (
        "risk_fork_curriculum_r0r3_upper",
        "risk_fork_curriculum_r4r7_upper",
        "risk_fork_curriculum_r0r7_upper",
    ),
)
@pytest.mark.parametrize("ending", ["terminal", "truncation"])
def test_drcurr_composed_phi_total_telescoping_on_curriculum(scenario_name, ending):
    # The invariance gate never lapses (DR §5.3): the composed Phi_total =
    # Phi_BFS + Phi_dz telescopes to gamma^L*Phi_total(s_L) - Phi_total(s_0) on
    # every curriculum geometry. Positions stay in cols 0-3 (free in every
    # curriculum scenario); (3,3) is in-zone for all three pairs.
    gamma = 0.99
    length = 4
    scen = _drc4_catalog()[scenario_name]
    zone = _drd1_fork_zone(scenario_name)
    assert (3, 3) in zone
    field = _drc4_distance_field(scen)
    rows = [(3, 1, 0.0), (3, 2, 1.0), (3, 3, 0.0), (4, 3, 1.0)]
    actor_obs = torch.tensor(
        [[[r / 7, c / 7, 0.0, 0.0, 0.0, ps, 0.0, 0.0, 0.0, 0.0] for (r, c, ps) in rows]]
    )
    ep = {
        "actor_observation": actor_obs,
        "terminal": torch.tensor([[False, False, False, ending == "terminal"]]),
        "sensing_action": torch.tensor([[1, 0, 1, 0]]),
    }
    final_positions = {"agent_0": (4, 3)}
    dev = torch.device("cpu")
    phi_s, phi_s_next = _drd1_sensing_potentials(
        ep, scenario=scen, zone=zone, credit_weight=0.0, zone_weight=0.05,
        final_positions=final_positions, device=dev,
    )
    phi_b, phi_b_next = _drc4_episode_potentials(
        ep, scenario=scen, distance_field=field, weight=0.05,
        final_positions=final_positions, device=dev,
    )
    phi_total = phi_b + phi_s
    phi_total_next = phi_b_next + phi_s_next
    shaping = _drc4_shaping_term(phi_total, phi_total_next, gamma)
    disc = torch.tensor([gamma ** t for t in range(length)])
    lhs = float((disc * shaping[0]).sum().item())
    rhs = gamma ** length * float(phi_total_next[0, length - 1].item()) - float(
        phi_total[0, 0].item()
    )
    assert lhs == pytest.approx(rhs, abs=4e-8)


# ---------------- AM-H13 re-scope: accept/reject matrix ----------------


def test_drcurr_scoping_rejects_t1d14_4pair_form_am_cf1():
    # DR-CURRICULUM-FORM (SHIFT 29): the F-84-safe N<=2 guard now REJECTS the
    # historical T1-D14 4-pair family with the armed floor (AM-CF1: it was
    # F-84-unsafe -- the arm can fire on a partial mixture, pooled 0.25). The
    # armed floor accepts at most 2 distinct fork pairs.
    with pytest.raises(ValueError, match="at most 2 distinct fork pairs"):
        _drcurr_run_config(
            scenario_names=_DRCURR_T1D14_TUPLE, **_DRCURR_ARM_KWARGS
        )
    # WITHOUT the armed floor the 4-pair family is still constructible (the
    # AM-H13 branch, incl. the N<=2 guard, only runs when the floor is on).
    config = _drcurr_run_config(scenario_names=_DRCURR_T1D14_TUPLE)
    assert config.scenario_names == _DRCURR_T1D14_TUPLE


def test_drcurr_scoping_accepts_t1d13_form_unchanged():
    # The pre-curriculum config shape stays accepted (byte-repro discipline).
    config = _drcurr_run_config(
        scenario_names=("risk_fork_train_lower", "risk_fork_train_upper"),
        **_DRCURR_ARM_KWARGS,
    )
    assert config.armed_lambda_floor == 3.0


def test_drcurr_scoping_rejects_single_mirror():
    with pytest.raises(ValueError, match="COMPLETE fork pairs"):
        _drcurr_run_config(
            scenario_names=("risk_fork_curriculum_r0r3_lower",),
            **_DRCURR_ARM_KWARGS,
        )


def test_drcurr_scoping_rejects_duplicate_unbalanced_tuple():
    # AM-C7(i): duplicate tuple positions AGGREGATE per name; the 3-tuple gives
    # 11/5 over 16 episodes -> rejected.
    with pytest.raises(ValueError, match="EQUAL per-round"):
        _drcurr_run_config(
            scenario_names=(
                "risk_fork_train_lower",
                "risk_fork_train_upper",
                "risk_fork_train_lower",
            ),
            **_DRCURR_ARM_KWARGS,
        )


def test_drcurr_scoping_rejects_untrained_pairs():
    # AM-C7(i): episodes_per_round below the tuple length leaves untrained
    # scenarios ((0,0) pairs are vacuously balanced but never trained).
    with pytest.raises(ValueError, match=">= 1 episode per round"):
        _drcurr_run_config(
            scenario_names=_DRCURR_T1D14_TUPLE,
            episodes_per_round=4,
            **_DRCURR_ARM_KWARGS,
        )


def test_drcurr_scoping_rejects_legacy_with_floor():
    with pytest.raises(ValueError, match="fork TRAINING family"):
        _drcurr_run_config(
            scenario_names=("risk_gate_hidden_hazard",), **_DRCURR_ARM_KWARGS
        )


def test_drcurr_readiness_exclusion_is_unconditional():
    # AM-C4: readiness fork ids are the reserved generalization probe. Rejected
    # with the armed floor OFF (the AM-H13 check would not even run here) ...
    with pytest.raises(ValueError, match="AM-C4"):
        _drcurr_run_config(
            scenario_names=(
                "risk_fork_readiness_lower",
                "risk_fork_readiness_upper",
            ),
        )
    # ... and with it ON (AM-C4 fires before the AM-H13 branch).
    with pytest.raises(ValueError, match="AM-C4"):
        _drcurr_run_config(
            scenario_names=(
                "risk_fork_readiness_lower",
                "risk_fork_readiness_upper",
            ),
            **_DRCURR_ARM_KWARGS,
        )
    # ... and mixed into an otherwise-admissible curriculum set.
    with pytest.raises(ValueError, match="AM-C4"):
        _drcurr_run_config(
            scenario_names=_DRCURR_T1D14_TUPLE + ("risk_fork_readiness_lower",),
        )


# ---------------- mixing law + probe repoint ----------------


def test_drcurr_mixing_law_exact_balance_at_t1d14_shape():
    # The pinned T1-D14 sampling law: episode i trains tuple[i % 8]; 16
    # episodes/round => EXACTLY 2 per scenario per round (mirrors equal within
    # every pair every round — the aliasing-average precondition of DR §3.2).
    counts: dict[str, int] = {}
    for episode_index in range(16):
        name = _DRCURR_T1D14_TUPLE[episode_index % len(_DRCURR_T1D14_TUPLE)]
        counts[name] = counts.get(name, 0) + 1
    assert counts == {name: 2 for name in _DRCURR_T1D14_TUPLE}


def test_drcurr_probe_resolution_conditional():
    # DR §5.5: T1-D13-form runs probe EXACTLY the pre-curriculum 4 surfaces
    # (byte-identity); curriculum runs probe the extended 10; legacy runs keep
    # the legacy probe (None).
    old4 = (
        "risk_fork_train_lower",
        "risk_fork_train_upper",
        "risk_fork_readiness_lower",
        "risk_fork_readiness_upper",
    )
    assert _drcurr_resolve_probe(
        ("risk_fork_train_lower", "risk_fork_train_upper")
    ) == old4
    extended = _drcurr_resolve_probe(_DRCURR_T1D14_TUPLE)
    assert extended == (
        ("risk_fork_train_lower", "risk_fork_train_upper")
        + _DRCURR_CURRICULUM_TUPLE
        + ("risk_fork_readiness_lower", "risk_fork_readiness_upper")
    )
    assert _drcurr_resolve_probe(("risk_gate_hidden_hazard",)) is None
    assert _drcurr_resolve_probe(
        ("risk_fork_train_lower", "risk_gate_hidden_hazard")
    ) is None


def test_drcurr_fork_probe_admits_curriculum_and_rejects_unknown():
    # baseline_driver admissibility widened to train+curriculum+readiness; any
    # name outside the admissible fork set stays hard-rejected (I-3 pattern).
    core = default_stage23_core_config()
    model = RecurrentMAPPOActorCritic(core)
    update_config = _gc_internal_stage22_config()
    probe = run_readiness_probe(
        model,
        probe_seed=9001,
        update_config=update_config,
        probe_scenarios=("risk_fork_curriculum_r0r3_lower",),
    )
    assert probe["fork_family"] is True
    assert set(probe["variants"]) == {"risk_fork_curriculum_r0r3_lower"}
    with pytest.raises(RuntimeError, match="refused"):
        run_readiness_probe(
            model,
            probe_seed=9001,
            update_config=update_config,
            probe_scenarios=("risk_gate_hidden_hazard",),
        )


# ---------------- diagnostics seed law + existing-entry byte-identity --------


def test_drcurr_explicit_diagnostics_seed_by_tuple_position():
    # AM-C7(iii): explicit-scenario_names diagnostics seed by POSITION IN THE
    # GIVEN TUPLE (never by catalog index), so explicit calls on the existing
    # scenarios are value-identical regardless of catalog growth. Pin the law:
    # the same scenario at the same tuple index yields the identical record
    # under different co-listed scenarios.
    diag_a = run_stage23_scenario_diagnostics(
        scenario_names=("unit_empty", "unit_single_hazard"), seed=2400
    )
    diag_b = run_stage23_scenario_diagnostics(
        scenario_names=("standard_branching_hazard", "unit_single_hazard"),
        seed=2400,
    )
    rec_a = [
        r for r in diag_a["scenarios"] if r["scenario_name"] == "unit_single_hazard"
    ][0]
    rec_b = [
        r for r in diag_b["scenarios"] if r["scenario_name"] == "unit_single_hazard"
    ][0]
    assert rec_a == rec_b
    assert diag_a["comparators"]["unit_single_hazard"] == (
        diag_b["comparators"]["unit_single_hazard"]
    )


def test_drcurr_existing_catalog_entries_byte_identical():
    # Permanent byte-identity anchor (AM-C6 discipline): the 8 pre-curriculum
    # catalog entries carry EXACTLY their pre-change field values.
    catalog = stage23_scenario_catalog()
    expected = {
        "standard_branching_hazard": (
            ((0, 0), (0, 1)), ((7, 7),),
            ((1, 3), (2, 3), (3, 3), (4, 3), (5, 3), (5, 4), (5, 5)),
            ((2, 2), (3, 4), (4, 5)),
        ),
        "risk_gate_hidden_hazard": (
            ((2, 0), (3, 0)), ((2, 4),), tuple(), ((2, 2),),
        ),
        "unit_empty": (((0, 0), (0, 1)), ((0, 2),), tuple(), tuple()),
        "unit_single_hazard": (((1, 0), (0, 0)), ((2, 2),), tuple(), ((1, 1),)),
        "risk_fork_train_upper": (
            ((3, 0), (4, 0)), ((3, 7),),
            ((0, 4), (1, 4), (2, 4), (5, 4), (6, 4), (7, 4)), ((3, 4),),
        ),
        "risk_fork_train_lower": (
            ((3, 0), (4, 0)), ((3, 7),),
            ((0, 4), (1, 4), (2, 4), (5, 4), (6, 4), (7, 4)), ((4, 4),),
        ),
        "risk_fork_readiness_upper": (
            ((3, 0), (4, 0)), ((3, 7),),
            ((0, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4)), ((1, 4),),
        ),
        "risk_fork_readiness_lower": (
            ((3, 0), (4, 0)), ((3, 7),),
            ((0, 4), (3, 4), (4, 4), (5, 4), (6, 4), (7, 4)), ((2, 4),),
        ),
    }
    for name, (starts, goals, obstacles, hazards) in expected.items():
        sc = catalog[name]
        assert sc.starts == starts
        assert sc.goals == goals
        assert sc.obstacles == obstacles
        assert sc.hidden_hazard_cells == hazards
        assert sc.width == 8 and sc.height == 8


# ---------------- driver e2e on the curriculum family ----------------


def test_drcurr_driver_e2e_curriculum_family_and_checkpoint_roundtrip(tmp_path):
    # Bounded end-to-end run on the DR-CURRICULUM-FORM 2-pair family with the
    # armed lever set: config accepted (N=2 <= the F-84-safe bound), run
    # completes, the checkpoint carries the arm state, the manifest records the
    # 2-pair scenario set, and the curve rows carry the arm-window keys (the
    # wiring surface T1-D15 uses). The 4-pair T1-D14 form would now reject
    # (AM-CF1; test_drcurr_scoping_rejects_t1d14_4pair_form_am_cf1).
    # steps_per_episode=6 keeps team success impossible on the fork geometry
    # (min safe route 7), so the arm deterministically stays silent while its
    # windows populate — the pre-arm wiring path (the DRH3 e2e convention).
    parent = tmp_path / "runs"
    parent.mkdir()
    config = _S25URunConfig(
        run_id="drcurr_e2e", seed=151, update_rounds=4, episodes_per_round=8,
        steps_per_episode=6, probe_every=0, state_save_every=2, tier="T0",
        result_parent=parent,
        lambda_ceiling=5.0,
        scenario_names=_DRCURR2_TUPLE,
        **_DRCURR_ARM_KWARGS,
    )
    run_dir = _s25u_run_training(config)
    state = torch.load(run_dir / "state_round_000004.pt", weights_only=True)
    assert state["controller"]["armed_lambda_floor"] == 3.0
    arm_payload = state["arm_state"]
    assert arm_payload["armed"] is False
    assert arm_payload["events"] == []
    assert len(arm_payload["arm_window"]) == 2
    manifest = json.loads((run_dir / "RUN_MANIFEST.json").read_text())
    assert manifest["config"]["scenario_names"] == list(_DRCURR2_TUPLE)
    curves = [
        json.loads(x)
        for x in (run_dir / "training_curve.jsonl").read_text().strip().split("\n")
    ]
    assert len(curves) == 4
    for c in curves:
        assert "arm_window_success" in c and "arm_window_hazard" in c
        assert c["lambda_floor_armed"] is False


# =============================================================================
# SECTION DRCURR2 — DR-CURRICULUM-FORM (SHIFT 29, countersigned): the F-84-safe
# bounded-N<=2 breadth reduction. The armed floor accepts at most 2 distinct
# fork pairs (a single blind pair among N pairs pools to ~1/N, above the 0.35
# arm conjunct only for N<=2); the 2-pair {3,4}+{0,3} family; the AM-CF2 probe
# scoping; per-pair cut/winnability re-run on the reduced active set.
# docs/decisions/DR-curriculum-form.md.
# =============================================================================

_DRCURR2_3PAIR = _DRCURR2_TUPLE + (
    "risk_fork_curriculum_r4r7_lower",
    "risk_fork_curriculum_r4r7_upper",
)


# ---------------- N<=2 config guard accept/reject matrix ----------------


def test_drcurr2_guard_accepts_two_pairs():
    # {3,4}+{0,3} = 4 scenarios / 2 distinct fork pairs => N=2 <= the F-84-safe
    # bound (each pair weight exactly 0.5, pooled blind >= 0.5 > 0.35). Accepted.
    config = _drcurr_run_config(
        scenario_names=_DRCURR2_TUPLE, **_DRCURR_ARM_KWARGS
    )
    assert config.armed_lambda_floor == 3.0
    assert config.scenario_names == _DRCURR2_TUPLE


def test_drcurr2_guard_accepts_single_pair():
    # 1 pair (the T1-D13 form) stays accepted (N=1 <= 2).
    config = _drcurr_run_config(
        scenario_names=("risk_fork_train_lower", "risk_fork_train_upper"),
        **_DRCURR_ARM_KWARGS,
    )
    assert config.armed_lambda_floor == 3.0


def test_drcurr2_guard_rejects_three_pairs():
    # 3 distinct pairs (6 scenarios) is F-84-unsafe (a blind pair at the
    # realizable weight 0.25 pools to 0.25 <= 0.35). Balance passes (16 eps ->
    # 3/3/2 per scenario, mirrors equal within each pair), THEN the N<=2 guard
    # fires (placed after the balance checks so single-violation configs keep
    # their own message).
    with pytest.raises(ValueError, match="at most 2 distinct fork pairs"):
        _drcurr_run_config(
            scenario_names=_DRCURR2_3PAIR, **_DRCURR_ARM_KWARGS
        )


def test_drcurr2_guard_predicate_is_distinct_pairs_not_scenario_count():
    # AM-CF5: the guard counts DISTINCT fork pairs (pair bases), not scenarios.
    # 4 scenarios / 2 pairs is accepted; 6 scenarios / 3 pairs is rejected --
    # scenario count alone (4 vs 6) is not the discriminator, pair count is.
    _drcurr_run_config(scenario_names=_DRCURR2_TUPLE, **_DRCURR_ARM_KWARGS)  # 4 sc, 2 pairs OK
    with pytest.raises(ValueError, match="at most 2 distinct fork pairs"):
        _drcurr_run_config(scenario_names=_DRCURR2_3PAIR, **_DRCURR_ARM_KWARGS)  # 6 sc, 3 pairs


def test_drcurr2_guard_inert_without_armed_floor():
    # AM-CF5 / OFF-default byte-identity: the N<=2 guard lives INSIDE the
    # armed_lambda_floor AM-H13 branch, so a >2-pair fork family WITHOUT the
    # armed floor is still constructible (the guard reads only scenario_names,
    # a config property; N=1/non-armed runs stay byte-identical).
    config = _drcurr_run_config(scenario_names=_DRCURR2_3PAIR)
    assert config.scenario_names == _DRCURR2_3PAIR
    config4 = _drcurr_run_config(scenario_names=_DRCURR_T1D14_TUPLE)
    assert config4.scenario_names == _DRCURR_T1D14_TUPLE


# ---------------- the F-84 pooled-hazard arithmetic ----------------


def test_drcurr2_pooled_hazard_n_le_2_safe_arithmetic():
    # The decisive arithmetic the guard codifies: a single blind fork pair
    # (per-pair team hazard 1.0) among N equal-weight pairs pools to ~1/N;
    # ABOVE the 0.35 arm conjunct iff N <= 2.
    arm_bar = 0.35
    h_blind = 1.0
    assert h_blind / 2 == 0.5 and 0.5 > arm_bar          # N=2 BLOCKED
    assert h_blind / 4 == 0.25 and 0.25 <= arm_bar       # N=4 LEAK (T1-D14)
    # N=3 realizable (16 eps / 6 scenarios -> pairs 6/6/4 eps -> weights
    # 0.375/0.375/0.25): the 0.25-weight blind pair pools to 0.25 <= 0.35.
    assert 0.25 <= arm_bar
    # the boundary 1/N > 0.35 <=> N < 2.857 <=> N <= 2
    assert 1 / 2 > arm_bar and 1 / 3 <= arm_bar


# ---------------- arm re-scope: per-pair cut on the reduced set ----------------


@pytest.mark.parametrize(
    "pair",
    [
        ("risk_fork_train_lower", "risk_fork_train_upper"),
        ("risk_fork_curriculum_r0r3_lower", "risk_fork_curriculum_r0r3_upper"),
    ],
    ids=["train_r3r4", "curriculum_r0r3"],
)
def test_drcurr2_per_pair_forces_sensing_at_budget_025(pair):
    # The AM-H13 arm re-scope obligation for the reduced active set: each pair
    # of the 2-pair family forces sensing by construction (BAR-A union-cut) and
    # is winnable + selective 0-hazard (BAR-B) at the graded budget 0.25.
    res = verify_fork_family_forces_sensing(pair, hazard_budget=0.25)
    assert res["bar_a_sensing_forced_by_construction"] is True
    assert res["bar_b_task_learnable_by_construction"] is True
    assert res["forces_sensing_and_winnable"] is True
    assert res["bar_a_no_sense_mean_hazard"] == 1.0


# ---------------- AM-CF2 probe scoping ----------------


def test_drcurr2_probe_scoping_am_cf2():
    # AM-CF2: a breadth-reduced 2-pair run probes ONLY its trained curriculum
    # pair ({0,3}) plus train + readiness -- NOT the untrained {4,7}/{0,7}.
    probe = _drcurr_resolve_probe(_DRCURR2_TUPLE)
    assert probe == (
        ("risk_fork_train_lower", "risk_fork_train_upper")
        + ("risk_fork_curriculum_r0r3_lower", "risk_fork_curriculum_r0r3_upper")
        + ("risk_fork_readiness_lower", "risk_fork_readiness_upper")
    )
    # untrained curriculum pairs are absent from the probe surface
    assert "risk_fork_curriculum_r4r7_lower" not in probe
    assert "risk_fork_curriculum_r0r7_lower" not in probe
    # BYTE-IDENTITY: the T1-D14 form (all six curriculum ids trained) still
    # probes the full extended 10-surface tuple in canonical order ...
    assert _drcurr_resolve_probe(_DRCURR_T1D14_TUPLE) == (
        ("risk_fork_train_lower", "risk_fork_train_upper")
        + _DRCURR_CURRICULUM_TUPLE
        + ("risk_fork_readiness_lower", "risk_fork_readiness_upper")
    )
    # ... and a training-pair-only run still probes exactly the 4 pre-curriculum
    # surfaces (no curriculum id trained => empty trained-curriculum slice).
    assert _drcurr_resolve_probe(
        ("risk_fork_train_lower", "risk_fork_train_upper")
    ) == (
        "risk_fork_train_lower",
        "risk_fork_train_upper",
        "risk_fork_readiness_lower",
        "risk_fork_readiness_upper",
    )


# =============================================================================
# SECTION DRCURR3 — DR-DISJOINT (SHIFT 31, countersigned): the row-DISJOINT
# discriminator. Replace the T1-D15 second pair {0,3} (shares gate row 3 with
# the anchor {3,4}) with {0,7} (row-DISJOINT). Resolves the confound "was the
# T1-D15 break gate-row SHARING or ANY second pair." The driver machinery is
# already generic on the pair set; these pins are the executable AM-H13 arm
# re-scope onto {3,4}+{0,7} + the {0,7} uniqueness/disjointness/BAR-A-B facts +
# the F-86 route-length confound + OFF-default byte-identity.
# docs/decisions/DR-disjoint-discriminator.md.
# =============================================================================

_DRCURR3_TUPLE = (
    "risk_fork_train_lower",
    "risk_fork_train_upper",
    "risk_fork_curriculum_r0r7_lower",
    "risk_fork_curriculum_r0r7_upper",
)  # {3,4}+{0,7}, N=2 row-DISJOINT
_DRCURR3_D16_PAIR = (
    "risk_fork_curriculum_r0r7_lower",
    "risk_fork_curriculum_r0r7_upper",
)
_DRCURR3_ANCHOR = ("risk_fork_train_lower", "risk_fork_train_upper")


def _drcurr3_gate_rows(name):
    sc = stage23_scenario_catalog()[name]
    return frozenset(r for r in range(sc.height) if (r, 4) not in set(sc.obstacles))


def test_drcurr3_gate_rows_disjoint_and_unique_control():
    # {0,7} gate rows are DISJOINT from the anchor {3,4}, and {0,7} is the UNIQUE
    # row-disjoint 2-subset of the free rows {0,3,4,7} (the alternates {0,4}/{3,7}
    # both SHARE a row with the anchor; readiness (1,2) + held-out (5,6) reserved).
    anchor = _drcurr3_gate_rows("risk_fork_train_lower")
    assert anchor == {3, 4}
    d16 = _drcurr3_gate_rows("risk_fork_curriculum_r0r7_lower")
    assert d16 == {0, 7}
    assert not (d16 & anchor)                       # row-disjoint from the anchor
    assert not (d16 & _DRCURR_RESERVED_ROWS)        # and from readiness + held-out
    free = sorted(_DRCURR_FREE_ROWS)
    disjoint = [
        (a, b)
        for i, a in enumerate(free)
        for b in free[i + 1 :]
        if not ({a, b} & anchor)
    ]
    assert disjoint == [(0, 7)]                      # unique row-disjoint control


def test_drcurr3_guard_accepts_disjoint_two_pairs():
    # {3,4}+{0,7} = 4 scenarios / 2 distinct fork pairs => N=2 <= the F-84-safe
    # bound. Accepted with the armed floor (the arm scoping is generic on the set).
    config = _drcurr_run_config(scenario_names=_DRCURR3_TUPLE, **_DRCURR_ARM_KWARGS)
    assert config.armed_lambda_floor == 3.0
    assert config.scenario_names == _DRCURR3_TUPLE


def test_drcurr3_d03_stays_legal_and_four_pair_rejected():
    # INTEGRITY: {3,4}+{0,3} (the T1-D15 set) is ALSO a valid N=2 config -- NOT
    # code-rejected. Its T1-D15 falsification is a SCIENCE result (ledger §11),
    # never a code guard. The 4-pair (T1-D14) is F-84-unsafe -> AM-CF5 rejects it.
    assert _drcurr_run_config(scenario_names=_DRCURR2_TUPLE, **_DRCURR_ARM_KWARGS).scenario_names == _DRCURR2_TUPLE
    with pytest.raises(ValueError, match="at most 2 distinct fork pairs"):
        _drcurr_run_config(scenario_names=_DRCURR_T1D14_TUPLE, **_DRCURR_ARM_KWARGS)


@pytest.mark.parametrize(
    "pair",
    [
        ("risk_fork_train_lower", "risk_fork_train_upper"),
        ("risk_fork_curriculum_r0r7_lower", "risk_fork_curriculum_r0r7_upper"),
    ],
    ids=["train_r3r4", "curriculum_r0r7"],
)
def test_drcurr3_per_pair_forces_sensing_at_budget_025(pair):
    # AM-H13 arm re-scope onto {3,4}+{0,7}: each pair forces sensing by
    # construction (BAR-A union-cut) and is winnable + selective 0-hazard (BAR-B)
    # at the graded budget 0.25. Each pair carries its OWN union-cut.
    res = verify_fork_family_forces_sensing(pair, hazard_budget=0.25)
    assert res["bar_a_sensing_forced_by_construction"] is True
    assert res["bar_b_task_learnable_by_construction"] is True
    assert res["forces_sensing_and_winnable"] is True
    assert res["bar_a_union_cut"] is True
    assert res["bar_a_no_sense_mean_hazard"] == 1.0  # split-gates blind minimum


def test_drcurr3_gaming_margin_f84_safe_on_disjoint_set():
    # N=2 -> each pair weight exactly 0.5; a fully-blind pair pools team hazard
    # 0.5 > the 0.35 arm hazard conjunct -> F-84-safe on the disjoint set. The
    # arm VALUES do not move (the re-scope is family-scope only).
    arm_bar = 0.35
    assert 1.0 / 2 == 0.5 and 0.5 > arm_bar
    assert _DRCURR_ARM_KWARGS["arm_hazard_threshold"] == 0.35


def test_drcurr3_probe_scoping_and_off_default_byte_identity():
    # AM-CF2: a {3,4}+{0,7} run probes train + {0,7} + readiness, NOT {0,3}/{4,7}.
    probe = _drcurr_resolve_probe(_DRCURR3_TUPLE)
    assert probe == (
        _DRCURR3_ANCHOR
        + _DRCURR3_D16_PAIR
        + ("risk_fork_readiness_lower", "risk_fork_readiness_upper")
    )
    assert "risk_fork_curriculum_r0r3_lower" not in probe
    assert "risk_fork_curriculum_r4r7_lower" not in probe
    # OFF-default byte-identity: the T1-D13-form single-pair run still probes the
    # pre-curriculum 4-surface tuple (no curriculum id) -> unchanged behavior.
    assert _drcurr_resolve_probe(_DRCURR3_ANCHOR) == (
        _DRCURR3_ANCHOR + ("risk_fork_readiness_lower", "risk_fork_readiness_upper")
    )


def test_drcurr3_max_span_route_geometry_confound():
    # The F-86 route-length confound: the {0,7} safe routes are much longer
    # (max-span reroute) than the anchor's. Pinned RELATIONALLY (AM-DD-3) so it is
    # robust to the edges-vs-cells ±1 counting convention: {0,7} max route is
    # strictly greater than the anchor max route, with a >= 4-move margin. The
    # decomposed anchor read (F-86-neutral) disentangles non-assembly of {0,7}
    # from representational collapse.
    catalog = stage23_scenario_catalog()

    def maxlen(name):
        sc = catalog[name]
        return max(
            len(
                _shortest_path(
                    st,
                    sc.goals,
                    width=sc.width,
                    height=sc.height,
                    obstacles=sc.obstacles,
                    avoid_cells=sc.hidden_hazard_cells,
                )
            )
            - 1
            for st in sc.starts
        )

    anchor_max = max(maxlen(n) for n in _DRCURR3_ANCHOR)
    d16_max = max(maxlen(n) for n in _DRCURR3_D16_PAIR)
    assert d16_max > anchor_max                 # {0,7} strictly longer (relational)
    assert d16_max - anchor_max >= 4            # substantial max-span gap, ±1-robust
    assert anchor_max <= 9                      # loose bound (edges 8 / cells 9)


def test_drcurr3_driver_e2e_disjoint_family_and_checkpoint_roundtrip(tmp_path):
    # Bounded end-to-end run on the {3,4}+{0,7} DISJOINT 2-pair family: config
    # accepted (N=2), the driver constructs + round-robins the disjoint fork envs,
    # the run completes, the checkpoint carries the arm state, and the manifest
    # records the disjoint scenario set. steps_per_episode=6 keeps success
    # impossible on the fork ({0,7} min safe route >= 12), so the arm stays silent
    # while its windows populate (the pre-arm wiring path; the DRH3 e2e convention).
    parent = tmp_path / "runs"
    parent.mkdir()
    config = _S25URunConfig(
        run_id="drcurr3_e2e", seed=141, update_rounds=4, episodes_per_round=8,
        steps_per_episode=6, probe_every=0, state_save_every=2, tier="T0",
        result_parent=parent,
        lambda_ceiling=5.0,
        scenario_names=_DRCURR3_TUPLE,
        **_DRCURR_ARM_KWARGS,
    )
    run_dir = _s25u_run_training(config)
    state = torch.load(run_dir / "state_round_000004.pt", weights_only=True)
    assert state["controller"]["armed_lambda_floor"] == 3.0
    assert state["arm_state"]["armed"] is False
    assert state["arm_state"]["events"] == []
    manifest = json.loads((run_dir / "RUN_MANIFEST.json").read_text())
    assert manifest["config"]["scenario_names"] == list(_DRCURR3_TUPLE)
    # the disjoint curriculum pair is the ONLY curriculum id in the trained set
    assert "risk_fork_curriculum_r0r7_lower" in manifest["config"]["scenario_names"]
    assert "risk_fork_curriculum_r0r3_lower" not in manifest["config"]["scenario_names"]


# =================== SECTION DRREP: DR-REPRESENTATION egocentric radius-2 patch ===================
# DR-REPRESENTATION (SHIFT 36, candidate A, COUNTERSIGNED WITH AMENDMENT wf_66def9e9-443):
# the actor observation is enriched with an EGOCENTRIC per-cell binary obstacle-occupancy
# PATCH over a Manhattan radius-2 window, appended at feature indices >=10 (NHR-read indices
# 0,1,5 fixed). This is the FIRST non-byte-identity-gateable change in the campaign, so the
# whole-tensor byte-identity gate is retired and REPLACED by this BEHAVIORAL invariance
# battery: MECH-SWAP (obstacles-only hazard-swap invariance) + MECH-MIRROR (mirror-aliasing
# bitwise equality = the observation-level BAR-A) + NHR index preservation + reveal-ablation
# invariance + off-grid sentinel + BAR-A-at-radius-2 + the enriched-wall never-sense control.

from raas_marl.environments.active_sensing import tensor_adapter as _drrep_ta
from raas_marl.final_training.stage25_collection import _recover_cell as _drrep_recover_cell

_DRREP_PAIRS = (
    ("risk_fork_train_lower", "risk_fork_train_upper"),
    ("risk_fork_curriculum_r0r3_lower", "risk_fork_curriculum_r0r3_upper"),
    ("risk_fork_curriculum_r4r7_lower", "risk_fork_curriculum_r4r7_upper"),
    ("risk_fork_curriculum_r0r7_lower", "risk_fork_curriculum_r0r7_upper"),
)


def _drrep_env(scenario_name, *, hidden_hazard_cells=None):
    cat = stage23_scenario_catalog()
    sc = cat[scenario_name]
    kw = dict(scenario_name=sc.name, width=sc.width, height=sc.height, seed=7)
    if hidden_hazard_cells is not None:
        kw["hidden_hazard_cells"] = tuple(hidden_hazard_cells)
    env = RiskAwareActiveSensingGridEnvironment(Stage23EnvironmentConfig(**kw))
    env.reset()
    return env


def _drrep_obs_tensor(env, cell):
    env._positions["agent_0"] = tuple(cell)
    obs = env._observation("agent_0")
    t = _drrep_ta.actor_observation_from_stage23(
        obs, local_observation_radius=env._config.local_observation_radius
    )
    return t.reshape(-1)


def _drrep_free_cells(scenario_name):
    cat = stage23_scenario_catalog()
    sc = cat[scenario_name]
    obstacles = set(sc.obstacles)
    return [
        (r, c)
        for r in range(sc.height)
        for c in range(sc.width)
        if (r, c) not in obstacles
    ]


def test_drrep_dim_and_offsets_canonical():
    assert _drrep_ta.STAGE23_ACTOR_OBSERVATION_DIM == 22
    assert _drrep_ta.STAGE23_LEGACY_ACTOR_FEATURE_COUNT == 10
    offs = _drrep_ta._RADIUS2_PATCH_OFFSETS
    assert len(offs) == 12
    for dr, dc in offs:
        assert 1 <= abs(dr) + abs(dc) <= 2
    assert (0, 0) not in offs
    assert list(offs) == sorted(offs)
    assert (1, 1) in offs


def test_drrep_nhr_index_preservation_and_recover_cell():
    env = _drrep_env("risk_fork_train_upper")
    for cell in _drrep_free_cells("risk_fork_train_upper"):
        t = _drrep_obs_tensor(env, cell)
        assert tuple(t.shape) == (22,)
        rec = _drrep_recover_cell(float(t[0]), float(t[1]), height=8, width=8)
        assert rec == tuple(cell)
        assert float(t[5]) == 0.0
    # Positively assert index 5 IS previous_sensed (a repoint of index 5 to another
    # 0.0-valued feature would otherwise pass); indices 0,1 and every other legacy
    # index are unchanged by the flip. The composed-Phi telescoping at the enriched
    # dim (22) is covered by the green DRC4/DRD1L1B/DRCURR telescoping tests, which run
    # through default_stage23_core_config() (now 22) and are invariant by construction
    # because Phi reads ONLY the byte-preserved indices {0,1,5}.
    obs_sensed = _gD_actor_observation()
    obs_sensed["actor_visible"]["previous_sensed"] = True
    obs_not = _gD_actor_observation()
    obs_not["actor_visible"]["previous_sensed"] = False
    t_s = actor_observation_from_stage23(obs_sensed).reshape(-1)
    t_n = actor_observation_from_stage23(obs_not).reshape(-1)
    assert float(t_s[5]) == 1.0 and float(t_n[5]) == 0.0
    assert torch.equal(t_s[:5], t_n[:5]) and torch.equal(t_s[6:], t_n[6:])


@pytest.mark.parametrize("lower,upper", _DRREP_PAIRS)
def test_drrep_mirror_aliasing_bitwise_identical(lower, upper):
    env_l = _drrep_env(lower)
    env_u = _drrep_env(upper)
    for cell in _drrep_free_cells(lower):
        tl = _drrep_obs_tensor(env_l, cell)
        tu = _drrep_obs_tensor(env_u, cell)
        assert torch.equal(tl, tu), f"mirror mismatch at {cell} on {lower}/{upper}"
        assert torch.equal(tl[10:], tu[10:])


def test_drrep_m3_cross_pair_conflict_witness():
    up = _drrep_obs_tensor(_drrep_env("risk_fork_train_upper"), (3, 3))
    lo = _drrep_obs_tensor(_drrep_env("risk_fork_curriculum_r0r3_lower"), (3, 3))
    assert not torch.equal(up, lo)
    idx = 10 + _drrep_ta._RADIUS2_PATCH_OFFSETS.index((1, 1))
    assert float(up[idx]) == 0.0
    assert float(lo[idx]) == 1.0


@pytest.mark.parametrize("scenario", [p[0] for p in _DRREP_PAIRS])
def test_drrep_hazard_swap_invariance(scenario):
    cat = stage23_scenario_catalog()
    sc = cat[scenario]
    obstacle_set = set(sc.obstacles)
    free = _drrep_free_cells(scenario)
    swaps = [(), ((0, 0),), ((7, 7),), ((5, 4),), ((0, 4), (7, 4)), tuple(free[:3])]
    base_env = _drrep_env(scenario)
    checked = 0
    for hz in swaps:
        if set(hz) & (obstacle_set | set(sc.starts) | set(sc.goals)):
            continue
        swap_env = _drrep_env(scenario, hidden_hazard_cells=hz)
        for cell in free:
            assert torch.equal(
                _drrep_obs_tensor(base_env, cell), _drrep_obs_tensor(swap_env, cell)
            ), f"hazard-swap changed the actor obs at {cell} (hz={hz}) on {scenario}"
        checked += 1
    assert checked >= 2


def test_drrep_patch_reveal_ablation_invariant():
    import copy
    base = _gD_actor_observation()
    variant = copy.deepcopy(base)
    variant["actor_visible"]["revealed_local_hazards"] = []
    t_base = actor_observation_from_stage23(base).reshape(-1)
    t_var = actor_observation_from_stage23(variant).reshape(-1)
    assert torch.equal(t_base, t_var)


def test_drrep_off_grid_sentinel_occupied():
    env = _drrep_env("risk_fork_train_upper")
    t = _drrep_obs_tensor(env, (0, 0))
    saw_off_grid = False
    for i, (dr, dc) in enumerate(_drrep_ta._RADIUS2_PATCH_OFFSETS):
        cell = (0 + dr, 0 + dc)
        in_grid = 0 <= cell[0] < 8 and 0 <= cell[1] < 8
        if not in_grid:
            saw_off_grid = True
            assert float(t[10 + i]) == 1.0, f"off-grid offset {(dr, dc)} must be occupied"
    assert saw_off_grid


@pytest.mark.parametrize("lower,upper", _DRREP_PAIRS)
def test_drrep_fork_family_forces_sensing_at_radius2(lower, upper):
    rec = verify_fork_family_forces_sensing((lower, upper))
    assert rec["forces_sensing_and_winnable"] is True
    assert rec["bar_a_no_sense_mean_hazard_exceeds_budget"] is True
    assert rec["bar_a_no_sense_mean_hazard"] >= 1.0


@pytest.mark.parametrize("lower,upper", _DRREP_PAIRS)
def test_drrep_enriched_wall_never_sense_control(lower, upper):
    # BA-6 enriched-wall never-sense control: a policy that perceives the FULL wall
    # (the no_sense scripted comparator plans a BFS route with full obstacle knowledge)
    # but NEVER senses CANNOT pass BAR-C4 per-mirror -- it eats hazard >0.5 on its
    # wrong-guess mirror -- proving the richer wall percept cannot substitute for the
    # reveal (the reveal is the sole which-gate-is-hazardous signal). per_layout records
    # the per-mirror no_sense hazard: exactly one mirror of each aliased pair is the
    # wrong guess (hazard 2.0 = 2x the d_ep=0.5 graded budget), the other 0.0.
    rec = verify_fork_family_forces_sensing((lower, upper))
    per_layout = rec["per_layout"]
    no_sense_hazards = sorted(
        float(per_layout[name]["no_sense_hazard_cost_sum"]) for name in (lower, upper)
    )
    # the never-sense (full-wall-perception) policy is safe on one mirror, unsafe on the
    # other => it CANNOT keep per-mirror hazard <= 0.5 on both.
    assert no_sense_hazards[-1] > 0.5, (lower, upper, no_sense_hazards)
    assert no_sense_hazards[-1] >= 2.0
    assert any(
        float(per_layout[name]["no_sense_hazard_cost_sum"]) > 0.5 for name in (lower, upper)
    )


def test_drrep_mech_swap_teeth_patch_reads_local_obstacles():
    # NEGATIVE-control 'teeth' for MECH-MIRROR/MECH-SWAP: prove the patch is NON-vacuous
    # -- it genuinely reads local_obstacles -- so the mirror/swap equalities are not
    # trivially true. Injecting a fake obstacle into local_obstacles CHANGES the patch
    # occupancy at that cell's offset; removing it restores 0.0.
    base = _gD_minimal_observation()  # position (0,0), no obstacles, radius-2
    base["actor_visible"]["position"] = [3, 3]
    base["actor_visible"]["nearest_goal_delta"] = [0, 4]
    t0 = actor_observation_from_stage23(base).reshape(-1)
    import copy
    inj = copy.deepcopy(base)
    inj["actor_visible"]["local_obstacles"] = [[4, 4]]  # offset (1,1) from (3,3)
    t1 = actor_observation_from_stage23(inj).reshape(-1)
    idx = 10 + _drrep_ta._RADIUS2_PATCH_OFFSETS.index((1, 1))
    assert float(t0[idx]) == 0.0 and float(t1[idx]) == 1.0
    # only the injected cell's patch bit (and the count feature 9) change
    assert not torch.equal(t0, t1)


def test_drrep_actor_obs_requires_radius_at_least_2():
    # COUPLING GUARD (impl-audit MINOR): the radius-2 patch requires the obstacle window
    # to cover radius >= 2; a radius-1 call fails LOUD rather than silently mis-reading
    # radius-2 cells as free.
    obs = _gD_actor_observation()
    with pytest.raises(ValueError, match="local_observation_radius >= 2"):
        actor_observation_from_stage23(obs, local_observation_radius=1)

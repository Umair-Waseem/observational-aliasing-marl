"""Shared rollout-collection helpers for the MAPPO-Lagrangian package.

This module is the single home for the rollout-collection helper family that
was previously copy-pasted across ``rollout.py`` (Stage 22 fixed-dim,
equal-length), ``stage23b_rollout.py`` (Stage 23-A single episode),
``stage24_collector.py`` (Stage 23-A multi-episode, variable-length padded),
and ``env_adapter.py`` (the Stage 22 transition-tensorization family), and the
``_parameter_delta_l1`` twin in ``update.py`` / ``stage24_collector.py`` (issue
D-7). Each helper is the strongest merged form of every source copy: it keeps
the union of all blocked/rejected cases (never dropping a rejection), unifies
divergent error wording on merit, and is parameterized on ``core_config`` /
field lists / a stage context string rather than hardcoding any one stage's
fixed dimensions or agent set — so the Stage 22 fixed-dim path, the Stage 23-A
path, and a future Stage 23-C multi-episode variable-length path can all call
it.

Boundary and identity discipline is preserved unchanged. Nothing here writes
result roots, serializes rollout buffers, saves model/optimizer state, runs
final evaluation, produces claim evidence, or makes any Bayesian-belief /
formal-VOI claim. It imports torch eagerly (it is an execution helper, not an
import-safe contract module) but keeps every dependency on leaf contract
modules (``_validation``, ``config``, ``env_contracts``) so it introduces no
import cycle with the rollout-collector and adapter modules that import it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import math
from typing import Any

import torch

from raas_marl.mappo_lagrangian._validation import (
    finite_numeric_scalar,
    require_int_not_bool,
)
from raas_marl.mappo_lagrangian.buffer import RolloutBatch
from raas_marl.mappo_lagrangian.config import MAPPOCoreConfig
from raas_marl.mappo_lagrangian.env_contracts import (
    EnvironmentStepContract,
    validate_actor_visible_payload,
    validate_critic_central_payload,
    validate_environment_step_contract,
)


# ---------------------------------------------------------------------------
# Unsafe-terminology scanning (union of every source phrase list)
# ---------------------------------------------------------------------------
# The rollout-collection layer produces JSON-compatible summaries that flow
# into governed result roots. The forbidden-wording union below is the merge of
# every phrase list scanned in the development / stage23b runners plus the
# terminology locks recorded in the project reference. Literals are split so
# the source of this module itself never contains the forbidden phrase. This
# constant is exposed so downstream summary builders can reuse one canonical
# list rather than re-deriving it.
_UNSAFE_TERMINOLOGY_PHRASES: tuple[str, ...] = (
    "bayesian " + "belief",
    "formal " + "voi",
    "c rc " + "mappo voi",
    "final " + "evaluation",
    "claim " + "evidence",
    "paper " + "facing",
)


def unsafe_terminology_phrases() -> tuple[str, ...]:
    """Return the union of forbidden-terminology phrases (normalized, lowercase)."""

    return _UNSAFE_TERMINOLOGY_PHRASES


def _normalise_terminology_string(value: str) -> str:
    """Lowercase and collapse every run of non-alphanumeric chars to one space."""

    out_chars: list[str] = []
    previous_was_space = False
    for char in value.lower():
        if char.isalnum():
            out_chars.append(char)
            previous_was_space = False
        else:
            if not previous_was_space:
                out_chars.append(" ")
            previous_was_space = True
    return "".join(out_chars).strip()


def contains_unsafe_terminology(value: str) -> bool:
    """Return whether ``value`` contains any forbidden-terminology phrase."""

    if not isinstance(value, str):
        raise TypeError("value must be a string")
    normalised = _normalise_terminology_string(value)
    return any(phrase in normalised for phrase in _UNSAFE_TERMINOLOGY_PHRASES)


# ---------------------------------------------------------------------------
# Rollout field schema and empty-field / stacking helpers
# ---------------------------------------------------------------------------
# The canonical rollout field order shared by every collector and by
# RolloutBatch. It is identical across all three source copies.
ROLLOUT_FIELD_NAMES: tuple[str, ...] = (
    "actor_observation",
    "revealed_information",
    "central_state",
    "sensing_action",
    "movement_action",
    "old_sensing_log_probability",
    "old_movement_log_probability",
    "task_reward",
    "hazard_cost",
    "sensing_cost",
    "reward_value",
    "hazard_cost_value",
    "next_reward_value",
    "next_hazard_cost_value",
    "terminal",
    "truncation",
    "valid_mask",
)

# Field-kind partitions used by the padded-merge path. Feature tensors carry a
# trailing feature dimension; action tensors are integer; bool fields are the
# terminal/truncation/valid_mask masks. Every name in ROLLOUT_FIELD_NAMES
# appears in exactly one partition.
_FEATURE_FIELDS: frozenset[str] = frozenset(
    {"actor_observation", "revealed_information", "central_state"}
)
_ACTION_FIELDS: frozenset[str] = frozenset({"sensing_action", "movement_action"})
_BOOL_FIELDS: frozenset[str] = frozenset({"terminal", "truncation", "valid_mask"})
_FLOAT_FIELDS: frozenset[str] = frozenset(
    {
        "old_sensing_log_probability",
        "old_movement_log_probability",
        "task_reward",
        "hazard_cost",
        "sensing_cost",
        "reward_value",
        "hazard_cost_value",
        "next_reward_value",
        "next_hazard_cost_value",
    }
)
# Cost fields carry the contractual nonnegativity constraint.
_NONNEGATIVE_COST_FIELDS: tuple[str, ...] = ("hazard_cost", "sensing_cost")
# Every float-valued field whose finiteness is asserted before a batch is
# returned (the non-cost floats plus the cost floats).
_FINITE_FLOAT_FIELDS: tuple[str, ...] = (
    "task_reward",
    "hazard_cost",
    "sensing_cost",
    "old_sensing_log_probability",
    "old_movement_log_probability",
    "reward_value",
    "hazard_cost_value",
    "next_reward_value",
    "next_hazard_cost_value",
)


def empty_rollout_fields(
    field_names: tuple[str, ...] = ROLLOUT_FIELD_NAMES,
) -> dict[str, list[torch.Tensor]]:
    """Return a fresh accumulator dict mapping each field name to an empty list.

    ``field_names`` defaults to the canonical schema but is parameterized so a
    future stage that tracks a different field set can reuse the accumulator.
    """

    return {name: [] for name in field_names}


def float_vector(values: list[object], reference: torch.Tensor) -> torch.Tensor:
    """Build a 1-D float tensor from per-agent scalars on ``reference``'s dtype/device.

    Each element is routed through :func:`_validation.finite_numeric_scalar`
    (NEW-rollout-1/2) so bool, ``None``, complex, and hostile numeric-subclass
    inputs are rejected with the deterministic TypeError/ValueError contract
    *before* the tensor is built, rather than only being caught later as a
    non-finite tensor. The post-build finiteness check is retained as
    defense-in-depth (a genuine finite float can still combine into a non-finite
    aggregate only via arithmetic, but this keeps the historical guard).
    """

    sanitized = [finite_numeric_scalar("rollout scalar value", value) for value in values]
    out = torch.tensor(sanitized, dtype=reference.dtype, device=reference.device)
    if not torch.isfinite(out).all().item():
        raise ValueError("rollout scalar values must be finite")
    return out


def stack_time(values: list[torch.Tensor]) -> torch.Tensor:
    """Stack per-step tensors along a new time axis (dim=1).

    The empty-list message is canonicalized across the three source copies
    (which used "rollout field has no values" and "episode must contain at
    least one transition") to a single deterministic wording.
    """

    if not values:
        raise ValueError("rollout field must contain at least one transition")
    return torch.stack(values, dim=1)


# ---------------------------------------------------------------------------
# Seeding policy (single documented function)
# ---------------------------------------------------------------------------
def set_rollout_seed(seed: int) -> None:
    """Seed the tensor-library global RNG for bounded development reproducibility.

    Single documented seeding policy for every rollout collector. Only
    ``torch.manual_seed`` is set: no collector in this package samples the
    Python ``random`` module for action or environment decisions, so seeding it
    is dead work (S2-25); it is dropped here. This sets a global RNG state and
    does not preserve or restore the caller's RNG state, and it makes no
    cross-platform or cross-version bitwise-determinism claim.
    """

    torch.manual_seed(seed)


def per_step_sampling_seed(base_seed: int, episode_index: int, step_index: int) -> int:
    """Return the collision-free per-step sampling reseed value.

    Uses the ``rollout.py`` offset convention ``base_seed + episode_index*1000 +
    step_index`` (issue M-1): episode ``e`` at step ``s`` and episode ``e+1`` at
    step ``s-1`` never share an RNG stream. Callers pass this to
    :func:`set_rollout_seed` (or ``torch.manual_seed``) only when sampling
    stochastic actions; the environment reset seed is independent.
    """

    return base_seed + episode_index * 1000 + step_index


# ---------------------------------------------------------------------------
# Batch value re-checks (unified check order: detach -> finite -> nonneg)
# ---------------------------------------------------------------------------
def validate_rollout_batch_values(batch: RolloutBatch) -> None:
    """Assert detachment, finiteness, and cost-nonnegativity on a built batch.

    Unified check order across the two source copies (rollout.py ran
    nonneg-then-finite-then-detach; stage23b_rollout.py ran
    detach-then-finite-then-nonneg): **detach first, then finite, then
    nonnegative**. Every stored estimate must be detached from the graph; every
    reward/cost/log-probability field must be finite; the hazard and sensing
    cost fields must be nonnegative. This is additionally required on the
    Stage 24 multi-episode path (NEW-stage24_collector-1), which previously
    lacked it.
    """

    if not isinstance(batch, RolloutBatch):
        raise TypeError("batch must be a RolloutBatch")
    # 1) detachment
    if batch.old_sensing_log_probability.requires_grad:
        raise ValueError("old_sensing_log_probability must be detached")
    if batch.old_movement_log_probability.requires_grad:
        raise ValueError("old_movement_log_probability must be detached")
    if batch.reward_value.requires_grad or batch.hazard_cost_value.requires_grad:
        raise ValueError("rollout value estimates must be detached")
    if batch.next_reward_value.requires_grad or batch.next_hazard_cost_value.requires_grad:
        raise ValueError("rollout next value estimates must be detached")
    # 2) finiteness
    for name in _FINITE_FLOAT_FIELDS:
        value = getattr(batch, name)
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")
    # 3) cost nonnegativity
    for name in _NONNEGATIVE_COST_FIELDS:
        value = getattr(batch, name)
        if value.numel() and (value < 0).any().item():
            raise ValueError(f"{name} must be nonnegative")


# ---------------------------------------------------------------------------
# Episode merging: equal-length concat AND variable-length padded merge (D-6)
# ---------------------------------------------------------------------------
def concat_episodes(
    episodes: list[dict[str, torch.Tensor]],
    field_name: str,
) -> torch.Tensor:
    """Concatenate equal-length episode fragments on the batch axis (dim=0).

    Equal-length strategy (Stage 22 / Stage 23-B): every episode fragment for a
    given field must share the same time length; variable-length fragments are
    rejected here — use :func:`merge_padded_episodes` for the padded strategy.
    Co-located with the padded variant per issue D-6 so both concat strategies
    are visible in one place.
    """

    if not episodes:
        raise ValueError("episodes must be nonempty")
    values = [episode[field_name] for episode in episodes]
    # Every rollout field tensor is [batch, time, ...]; a tensor without a time
    # axis would otherwise surface as an undeclared IndexError on shape[1].
    for value in values:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{field_name} episode fragment must be a torch.Tensor")
        if value.ndim < 2:
            raise ValueError(
                f"{field_name} episode fragment must have a [batch, time] axis"
            )
    time_lengths = {int(value.shape[1]) for value in values}
    if len(time_lengths) != 1:
        raise ValueError(
            f"{field_name} episode fragments must have equal time length; "
            "use merge_padded_episodes for variable-length episodes"
        )
    return torch.cat(values, dim=0)


def pad_field(tensor: torch.Tensor, *, name: str, max_time: int) -> torch.Tensor:
    """Right-pad one field tensor along the time axis to ``max_time``.

    Padding value by field kind (variable-length strategy): bool fields pad
    ``False``, action fields pad ``0``, feature and float fields pad ``0.0``.
    ``valid_mask`` is a bool field so its padded steps are ``False`` — the
    downstream consumers use ``valid_mask`` to exclude padded steps. An unknown
    field name is rejected.
    """

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} field must be a torch.Tensor")
    # Rollout field tensors are [batch, time, ...]; guard the time-axis access so
    # a tensor without a time axis raises a named error rather than an undeclared
    # IndexError on shape[1].
    if tensor.ndim < 2:
        raise ValueError(f"{name} field must have a [batch, time] axis")
    current_time = int(tensor.shape[1])
    if current_time > max_time:
        raise ValueError("cannot pad a tensor longer than max_time")
    if current_time == max_time:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[1] = max_time - current_time
    if name in _BOOL_FIELDS:
        pad = torch.zeros(pad_shape, dtype=torch.bool, device=tensor.device)
    elif name in _ACTION_FIELDS or name in _FLOAT_FIELDS or name in _FEATURE_FIELDS:
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
    else:
        raise ValueError(f"unknown rollout field: {name}")
    return torch.cat([tensor, pad], dim=1)


def merge_padded_episodes(
    episodes: list[dict[str, torch.Tensor]],
    *,
    max_time: int,
    field_names: tuple[str, ...] = ROLLOUT_FIELD_NAMES,
) -> RolloutBatch:
    """Merge variable-length episodes into one RolloutBatch with valid-mask padding.

    Each field of each episode is right-padded to ``max_time`` via
    :func:`pad_field` (bool->False, int->0, float->0.0) and concatenated on the
    batch axis; padded steps carry ``valid_mask == False``. ``field_names`` is
    parameterized (defaulting to the canonical schema) so a future stage can
    merge a different field set. The returned :class:`RolloutBatch` is not yet
    validated — callers still run ``batch.validate(core_config)`` and
    :func:`validate_rollout_batch_values`.
    """

    if not episodes:
        raise ValueError("episodes must be nonempty")
    merged = {
        name: torch.cat(
            [pad_field(episode[name], name=name, max_time=max_time) for episode in episodes],
            dim=0,
        )
        for name in field_names
    }
    return RolloutBatch(**merged)


# ---------------------------------------------------------------------------
# Stage 23-A tensor batchers (via the tensor adapter)
# ---------------------------------------------------------------------------
# The tensor-adapter functions are imported lazily inside the batchers to keep
# this module's import graph on leaf contract modules only (tensor_adapter
# imports env_contracts/config, and the environment package depends on the
# mappo_lagrangian leaf modules; importing it eagerly here is safe, but the
# lazy import keeps _rollout_common importable in a torch-only context without
# forcing the environment package to load).
def _stage23_agent_names() -> tuple[str, ...]:
    from raas_marl.environments.active_sensing.grid_environment import (
        STAGE23_AGENT_NAMES,
    )

    return STAGE23_AGENT_NAMES


def actor_observation_batch(
    observations: Mapping[str, Mapping[str, object]],
    config: MAPPOCoreConfig,
    *,
    dtype: str,
    device: str,
    agent_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Batch per-agent Stage 23-A actor observations into ``[agents, 1, dim]``.

    ``agent_names`` defaults to the fixed Stage 23-A order but is parameterized
    so a future stage with a different agent set can reuse the batcher.
    """

    from raas_marl.environments.active_sensing.tensor_adapter import (
        actor_observation_from_stage23,
    )

    names = _stage23_agent_names() if agent_names is None else agent_names
    return torch.cat(
        [
            actor_observation_from_stage23(
                observations[agent],
                config=config,
                dtype=dtype,
                device=device,
            )
            for agent in names
        ],
        dim=0,
    )


def revealed_information_batch(
    observations: Mapping[str, Mapping[str, object]],
    config: MAPPOCoreConfig,
    *,
    dtype: str,
    device: str,
    agent_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Batch per-agent Stage 23-A one-step reveal tensors into ``[agents, 1, dim]``."""

    from raas_marl.environments.active_sensing.tensor_adapter import (
        revealed_information_from_stage23,
    )

    names = _stage23_agent_names() if agent_names is None else agent_names
    return torch.cat(
        [
            revealed_information_from_stage23(
                observations[agent],
                config=config,
                dtype=dtype,
                device=device,
            )
            for agent in names
        ],
        dim=0,
    )


def central_state_batch(
    environment: Any,
    config: MAPPOCoreConfig,
    *,
    dtype: str,
    device: str,
    agent_names: tuple[str, ...] | None = None,
) -> torch.Tensor:
    """Return the privileged central state repeated once per agent (``[agents, 1, dim]``).

    The single privileged read is ``central_state_from_stage23`` (CTDE Rule 8);
    the result is broadcast to the fixed agent count so both centralized critics
    receive the same central state per agent.
    """

    from raas_marl.environments.active_sensing.tensor_adapter import (
        central_state_from_stage23,
    )

    names = _stage23_agent_names() if agent_names is None else agent_names
    central = central_state_from_stage23(
        environment,
        config=config,
        dtype=dtype,
        device=device,
    )
    return central.repeat(len(names), 1, 1)


# ---------------------------------------------------------------------------
# Policy -> environment action mapping (required core_config, no None fallback)
# ---------------------------------------------------------------------------
def actions_from_policy(
    sensing_action: torch.Tensor,
    movement_action: torch.Tensor,
    core_config: MAPPOCoreConfig,
    *,
    agent_names: tuple[str, ...] | None = None,
) -> dict[str, dict[str, int]]:
    """Map per-agent policy action tensors to the environment factored-action dict.

    ``core_config`` is a **required** parameter (the dead ``None`` fallback of
    the stage23b twin is removed per issue m-12): the caller always threads the
    validated model config. Both action tensors are validated with the strongest
    per-tensor gate (:func:`validate_policy_action_tensor`) and must share a
    device.
    """

    from raas_marl.environments.active_sensing.grid_environment import (
        MOVEMENT_ACTION_FIELD,
        SENSING_ACTION_FIELD,
    )

    if not isinstance(core_config, MAPPOCoreConfig):
        raise TypeError("core_config must be a MAPPOCoreConfig")
    names = _stage23_agent_names() if agent_names is None else agent_names
    validate_policy_action_tensor(
        "sensing_action",
        sensing_action,
        action_count=core_config.sensing_action_count,
        agent_count=len(names),
    )
    validate_policy_action_tensor(
        "movement_action",
        movement_action,
        action_count=core_config.movement_action_count,
        agent_count=len(names),
    )
    if sensing_action.device != movement_action.device:
        raise ValueError("sensing and movement action tensor devices must match")
    return {
        agent: {
            SENSING_ACTION_FIELD: int(sensing_action[index].item()),
            MOVEMENT_ACTION_FIELD: int(movement_action[index].item()),
        }
        for index, agent in enumerate(names)
    }


def validate_policy_action_tensor(
    name: str,
    value: torch.Tensor,
    *,
    action_count: int,
    agent_count: int | None = None,
) -> None:
    """Validate a 1-D per-agent integer action tensor.

    Union of the rejections in both source twins plus the missing ones (issue
    NEW-buffer-4 / dedup note "add is_quantized rejection"): rejects non-tensors
    (TypeError), wrong length (ValueError, wording unified to "one item per
    fixed Stage 23-A agent"), meta device, non-strided (sparse) layout,
    bool/float/complex/**quantized** dtype (TypeError), and out-of-range action
    indices. ``agent_count`` defaults to the fixed Stage 23-A agent count.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    expected_agents = len(_stage23_agent_names()) if agent_count is None else agent_count
    if tuple(value.shape) != (expected_agents,):
        raise ValueError(f"{name} must have one item per fixed Stage 23-A agent")
    if value.device.type == "meta":
        raise ValueError(f"{name} must not be a meta tensor")
    if value.layout != torch.strided:
        raise ValueError(f"{name} must use a strided tensor layout")
    if (
        value.dtype == torch.bool
        or torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.is_quantized
    ):
        raise TypeError(f"{name} must use an integer dtype")
    if value.numel() and (value.min().item() < 0 or value.max().item() >= action_count):
        raise ValueError(f"{name} contains an out-of-range action index")


# ---------------------------------------------------------------------------
# Stage 23-A parallel / step-API payload validators
# ---------------------------------------------------------------------------
def validate_parallel_payloads(
    observations: Mapping[str, Mapping[str, object]],
    infos: Mapping[str, Mapping[str, object]],
    *,
    agent_names: tuple[str, ...] | None = None,
) -> None:
    """Validate that observations/infos are mappings in the fixed agent order."""

    names = _stage23_agent_names() if agent_names is None else agent_names
    if not isinstance(observations, Mapping):
        raise TypeError("observations must be a mapping")
    if not isinstance(infos, Mapping):
        raise TypeError("infos must be a mapping")
    if tuple(observations.keys()) != names:
        raise ValueError("observations must preserve fixed Stage 23-A agent order")
    if tuple(infos.keys()) != names:
        raise ValueError("infos must preserve fixed Stage 23-A agent order")


def validate_step_api_payloads(
    rewards: Mapping[str, object],
    terminations: Mapping[str, object],
    truncations: Mapping[str, object],
    infos: Mapping[str, Mapping[str, object]],
    *,
    agent_names: tuple[str, ...] | None = None,
) -> None:
    """Validate the PettingZoo-style parallel step return against the info payloads.

    Union of the twin checks plus the added presence guards: each of rewards /
    terminations / truncations must be a mapping in the fixed agent order; each
    agent's ``info.done_flags`` must be a mapping whose ``terminal`` /
    ``truncated`` match the step terminations/truncations by identity; the
    ``task_reward`` info key must be present (deterministic ValueError rather
    than a raw ``float(None)`` TypeError); reward-vs-info equality is checked
    with both sides routed through :func:`_validation.finite_numeric_scalar`
    (NEW-stage23b_rollout-5) so a non-numeric reward fails deterministically;
    and ``entered_hazard`` / ``team_success`` presence is validated
    (NEW-stage24_collector-3) so downstream subscripts raise a named message
    rather than a raw ``KeyError``.
    """

    from raas_marl.environments.active_sensing.grid_environment import TASK_REWARD_KEY

    names = _stage23_agent_names() if agent_names is None else agent_names
    for label, payload in (
        ("rewards", rewards),
        ("terminations", terminations),
        ("truncations", truncations),
    ):
        if not isinstance(payload, Mapping):
            raise TypeError(f"{label} must be a mapping")
        if tuple(payload.keys()) != names:
            raise ValueError(f"{label} must preserve fixed Stage 23-A agent order")
    for agent in names:
        info = infos[agent]
        done_flags = info.get("done_flags") if isinstance(info, Mapping) else None
        if not isinstance(done_flags, Mapping):
            raise TypeError("info.done_flags must be a mapping")
        if terminations[agent] is not done_flags.get("terminal"):
            raise ValueError("step terminations must match info.done_flags.terminal")
        if truncations[agent] is not done_flags.get("truncated"):
            raise ValueError("step truncations must match info.done_flags.truncated")
        info_task_reward = info.get(TASK_REWARD_KEY)
        if info_task_reward is None:
            raise ValueError("info payload is missing task_reward")
        step_reward = finite_numeric_scalar("step reward", rewards[agent])
        info_reward = finite_numeric_scalar("info task_reward", info_task_reward)
        if step_reward != info_reward:
            raise ValueError("step rewards must match info task_reward")
        if "entered_hazard" not in info:
            raise ValueError("info payload is missing entered_hazard")
        if "team_success" not in info:
            raise ValueError("info payload is missing team_success")


# ---------------------------------------------------------------------------
# Team-end flag validation (single message form; reject partial AND mixed)
# ---------------------------------------------------------------------------
def validate_team_end_flags(
    terminal: torch.Tensor,
    truncation: torch.Tensor,
    *,
    context: str = "rollout collection",
) -> bool:
    """Validate team-level terminal/truncation flags; return whether the team ended.

    Single unified message form for the Stage 23-A collectors. Union of every
    source rejection: shape/dtype/device coherence, meta-device rejection,
    both-flags-true rejection, **partial** per-agent endings rejected, and
    **mixed** terminal/truncated team endings rejected. ``context`` prefixes the
    partial/mixed messages so each caller's phrasing stays informative while the
    implementation is shared. Returns ``True`` iff every agent ended (all
    terminal, or all truncated).
    """

    if not isinstance(terminal, torch.Tensor) or not isinstance(truncation, torch.Tensor):
        raise TypeError("terminal and truncation must be torch.Tensor values")
    if terminal.shape != truncation.shape:
        raise ValueError("terminal and truncation shape mismatch")
    if terminal.dtype != torch.bool or truncation.dtype != torch.bool:
        raise TypeError("terminal and truncation must use torch.bool dtype")
    if terminal.device != truncation.device:
        raise ValueError("terminal and truncation device mismatch")
    if terminal.device.type == "meta" or truncation.device.type == "meta":
        raise ValueError("terminal and truncation must not be meta tensors")
    if (terminal & truncation).any().item():
        raise ValueError("terminal and truncation cannot both be true")
    terminal_any = terminal.any().item()
    truncation_any = truncation.any().item()
    ended = terminal | truncation
    if ended.any().item() and not ended.all().item():
        raise ValueError(
            f"{context} does not support partial per-agent termination or "
            "truncation; all agents must end together"
        )
    if terminal_any and truncation_any:
        raise ValueError(
            f"{context} does not support mixed terminal and truncated team "
            "endings; all agents must end with the same flag type"
        )
    return bool(terminal.all().item() or truncation.all().item())


# ---------------------------------------------------------------------------
# Core-config structural compare (asdict, avoids relying on dataclass __eq__)
# ---------------------------------------------------------------------------
def validate_stage23_core_config(
    config: MAPPOCoreConfig,
    *,
    context: str = "Stage 23-A rollout collection",
) -> None:
    """Assert ``config`` structurally equals the default Stage 23-A core config.

    Uses an ``asdict`` structural comparison (not dataclass ``__eq__``) against
    ``default_stage23_core_config()``. ``context`` prefixes the value error so
    each caller keeps informative phrasing.
    """

    from raas_marl.environments.active_sensing.tensor_adapter import (
        default_stage23_core_config,
    )

    if not isinstance(config, MAPPOCoreConfig):
        raise TypeError("config must be a MAPPOCoreConfig")
    expected = default_stage23_core_config()
    if asdict(config) != asdict(expected):
        raise ValueError(f"{context} requires the default fixed Stage 23-A core config")


# ---------------------------------------------------------------------------
# Stage 22 transition-tensorization family
# ---------------------------------------------------------------------------
# These helpers previously lived in env_adapter.py and were imported by
# rollout.py; they are moved here so both callers import from one home. They are
# kept decoupled from env_adapter's DevelopmentTransition class (which would
# create an import cycle) by accepting an optional ``transition_type`` for the
# isinstance gate and by validating structure (``item.contract`` is an
# EnvironmentStepContract) rather than the concrete transition class. Callers
# that want the concrete-class gate pass ``transition_type=DevelopmentTransition``.


def stage22_team_end_flags_ended(
    terminal: torch.Tensor,
    truncation: torch.Tensor,
    *,
    context: str,
) -> bool:
    """Validate Stage 22 team-level ending flags and return whether all ended.

    ``context`` preserves each call site's wording ("Stage 22 transition
    tensorization" for the tensorizer, "Stage 22 development rollout collection"
    for the rollout collector). Adds meta-device rejection
    (NEW-env_adapter-3) on top of the shape/dtype/device and both-true /
    partial / mixed rejections.
    """

    if terminal.shape != truncation.shape:
        raise ValueError("terminal and truncation shape mismatch")
    if terminal.dtype != torch.bool or truncation.dtype != torch.bool:
        raise TypeError("terminal and truncation must use torch.bool dtype")
    if terminal.device != truncation.device:
        raise ValueError("terminal and truncation device mismatch")
    if terminal.device.type == "meta" or truncation.device.type == "meta":
        raise ValueError("terminal and truncation must not be meta tensors")
    if (terminal & truncation).any().item():
        raise ValueError("terminal and truncation cannot both be true")

    terminal_any = terminal.any().item()
    truncation_any = truncation.any().item()
    ended = terminal | truncation

    if ended.any().item() and not ended.all().item():
        raise ValueError(
            f"{context} does not support partial "
            "per-agent termination or truncation; all agents must end together"
        )
    if terminal_any and truncation_any:
        raise ValueError(
            f"{context} does not support mixed "
            "terminal/truncated team endings; all agents must end with the same flag type"
        )
    return bool(terminal.all().item() or truncation.all().item())


def transition_agent_id(item: Any) -> int:
    """Return the strict-int ``agent_id`` from a transition's contract identity."""

    value = item.contract.identity["agent_id"]
    require_int_not_bool(
        "transition identity agent_id", value, error_suffix="must be an integer"
    )
    return value


def declared_agent_order_from_transition(
    item: Any,
    config: MAPPOCoreConfig,
) -> tuple[int, ...]:
    """Return the declared stable agent order from one transition's identity.

    Accepts ``agent_order`` / ``team_agent_ids`` (list or tuple) or
    ``agent_id_mapping`` (mapping of agent_id -> position). Rejects wrong
    container types (TypeError), empty / duplicate / negative / non-int orders,
    and — when ``config.agent_id_count`` is set — orders that do not cover the
    configured team.
    """

    identity = item.contract.identity

    if "agent_order" in identity:
        order_source = identity["agent_order"]
        if isinstance(order_source, (set, frozenset)) or not isinstance(
            order_source, (list, tuple)
        ):
            raise TypeError("transition identity agent_order must be a list or tuple")
        order = tuple(order_source)
    elif "team_agent_ids" in identity:
        order_source = identity["team_agent_ids"]
        if isinstance(order_source, (set, frozenset)) or not isinstance(
            order_source, (list, tuple)
        ):
            raise TypeError("transition identity team_agent_ids must be a list or tuple")
        order = tuple(order_source)
    elif "agent_id_mapping" in identity:
        mapping = identity["agent_id_mapping"]
        if not isinstance(mapping, Mapping):
            raise TypeError("transition identity agent_id_mapping must be a mapping")
        # NEW-env_adapter-2: snapshot items() exactly once so a hostile mapping
        # cannot return one set of pairs to the position validation and a
        # different set to the sort (TOCTOU). The positions are validated
        # (strict int, nonnegative, unique) before sorting: an unvalidated
        # mapping previously accepted duplicate positions silently (producing a
        # nondeterministic order) and let non-comparable positions escape as an
        # undeclared TypeError from ``sorted``. The agent-id keys themselves are
        # validated by the shared order loop below.
        pairs = tuple(mapping.items())
        positions: list[int] = []
        for _agent_id, position in pairs:
            require_int_not_bool(
                "transition identity agent_id_mapping position",
                position,
                error_suffix="must contain integer stable positions",
            )
            normalized_position = int.__index__(position)
            if normalized_position < 0:
                raise ValueError(
                    "transition identity agent_id_mapping position must be nonnegative"
                )
            positions.append(normalized_position)
        if len(set(positions)) != len(positions):
            raise ValueError(
                "transition identity agent_id_mapping positions must be unique"
            )
        order = tuple(
            agent_id
            for agent_id, _position in sorted(
                zip((pair[0] for pair in pairs), positions, strict=True),
                key=lambda item: item[1],
            )
        )
    else:
        raise ValueError(
            "transition identity requires agent_order, team_agent_ids, or agent_id_mapping"
        )

    if not order:
        raise ValueError("transition identity declared agent order must be nonempty")
    for agent_id in order:
        require_int_not_bool(
            "transition identity declared agent order",
            agent_id,
            error_suffix="must contain integers",
        )
        if agent_id < 0:
            raise ValueError("transition identity declared agent order must be nonnegative")
    if len(set(order)) != len(order):
        raise ValueError("transition identity declared agent order must not contain duplicates")
    if config.agent_id_count is not None:
        expected_ids = set(range(config.agent_id_count))
        if set(order) != expected_ids:
            raise ValueError(
                "transition identity declared agent order must cover the configured team"
            )
    return order


def validate_transition_identity_coverage(
    transitions: list[Any],
    config: MAPPOCoreConfig,
) -> None:
    """Assert the transitions collectively cover the configured team exactly once.

    Every transition must declare the same stable agent order; that order must
    cover the configured team (when ``agent_id_count`` is set); and the
    per-transition ``agent_id`` values must be exactly that order (duplicates and
    missing agents produce specific messages).
    """

    if not transitions:
        raise ValueError("transitions must be nonempty")

    declared_orders = tuple(
        declared_agent_order_from_transition(item, config) for item in transitions
    )
    reference_order = declared_orders[0]

    for order in declared_orders:
        if order != reference_order:
            raise ValueError(
                "transition tensorization requires all transitions to declare "
                "the same stable agent order"
            )

    transition_agent_ids = tuple(transition_agent_id(item) for item in transitions)

    if config.agent_id_count is not None:
        expected_ids = set(range(config.agent_id_count))
        if set(reference_order) != expected_ids:
            raise ValueError(
                "transition tensorization declared agent order must cover the "
                "configured team"
            )

    expected_order = reference_order
    if transition_agent_ids != expected_order:
        # NEW-env_adapter-9: deterministic error precedence.
        #   1) duplicate per-transition agent_id values,
        #   2) an agent_id that is absent from the declared agent order (clearer
        #      than the generic count message; a strict subset of the former
        #      set-mismatch branch, so the same inputs are still rejected),
        #   3) a count/coverage mismatch against the declared order, and finally
        #   4) a pure reordering of the correct agent set.
        expected_id_set = set(expected_order)
        if len(set(transition_agent_ids)) != len(transition_agent_ids):
            raise ValueError(
                "transition tensorization received duplicate agent identifiers"
            )
        if any(agent_id not in expected_id_set for agent_id in transition_agent_ids):
            raise ValueError(
                "transition tensorization received a transition agent_id that is "
                "not in the declared agent order"
            )
        if set(transition_agent_ids) != expected_id_set:
            raise ValueError(
                "transition tensorization requires exactly one transition for "
                "each configured agent"
            )
        raise ValueError(
            "transition tensorization transition order must match declared agent_order"
        )


def validate_transition_for_tensorization(
    item: Any,
    config: MAPPOCoreConfig,
    *,
    transition_type: type | None = None,
) -> None:
    """Validate one Stage 22 development transition prior to tensorization.

    Structural validation: ``item.contract`` must be an
    :class:`EnvironmentStepContract` and pass
    :func:`validate_environment_step_contract`; the next actor/critic payloads
    must pass their boundary validators; and the internal "truncation" wording
    is re-derived for both-true done flags. When ``transition_type`` is provided
    the concrete-class isinstance gate is applied first (callers pass
    ``DevelopmentTransition``); it is optional so this module needs no import of
    the env_adapter class (avoiding an import cycle).
    """

    if transition_type is not None and not isinstance(item, transition_type):
        raise TypeError("transitions must contain DevelopmentTransition items")
    if not isinstance(item.contract, EnvironmentStepContract):
        raise TypeError("transition.contract must be an EnvironmentStepContract")
    validate_environment_step_contract(item.contract, config)
    validate_actor_visible_payload(item.next_actor_visible, config)
    validate_critic_central_payload(item.next_critic_visible, config)
    terminal = bool(item.contract.done_flags["terminal"])
    truncated = bool(item.contract.done_flags["truncated"])
    # NEW-rollout-5 (documented here on the raising side): the both-true done
    # flags are already rejected by ``validate_environment_step_contract`` above,
    # which raises the env_contracts public-contract wording "terminal and
    # truncated cannot both be true" (with "truncated"). That earlier raise makes
    # the branch below unreachable in normal flow, so this re-derivation is
    # documented defense-in-depth only. Consequently this function *propagates
    # the env_contracts "truncated" wording unchanged* for the both-true case;
    # ``rollout.py._validate_returned_transitions_for_rollout`` deliberately
    # relies on that exact string to remap it to its own "truncation" surface
    # wording. Do not "helpfully" pre-empt the env_contracts message here (e.g.
    # by moving this check before ``validate_environment_step_contract``): that
    # would silently break the documented remap in rollout.py, which this module
    # cannot edit. The literal below remains "truncation" so any direct caller
    # that reaches this defensive branch still sees the historical internal
    # wording.
    if terminal and truncated:
        raise ValueError("terminal and truncation cannot both be true")


def tensor_from_mapping(mapping: Mapping[str, object], key: str) -> torch.Tensor:
    """Return ``mapping[key]`` after asserting it is a torch.Tensor."""

    value = mapping[key]
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{key} must be a torch.Tensor")
    return value


def validate_action_vector(
    name: str,
    value: torch.Tensor,
    expected_count: int,
    action_count: int,
) -> None:
    """Validate a 1-D per-agent integer action vector for the Stage 22 environment.

    Union of every rejection: non-tensor (TypeError), wrong ndim / length
    (ValueError), bool/float/complex/quantized dtype (TypeError), meta device,
    non-strided (sparse) layout, and out-of-range action indices.
    """

    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 1:
        raise ValueError(f"{name} must have shape [agent]")
    if value.shape[0] != expected_count:
        raise ValueError(f"{name} length must match agent_count")
    if (
        value.dtype == torch.bool
        or torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.is_quantized
    ):
        raise TypeError(f"{name} must use an integer dtype")
    if value.device.type == "meta":
        raise ValueError(f"{name} must not be a meta tensor")
    if value.layout != torch.strided:
        raise ValueError(f"{name} must use a strided tensor layout")
    if value.numel() and (value.min().item() < 0 or value.max().item() >= action_count):
        raise ValueError(f"{name} contains an out-of-range action index")


# ---------------------------------------------------------------------------
# Parameter-delta L1 (with the math.isfinite guard)
# ---------------------------------------------------------------------------
def parameter_delta_l1(
    before: list[torch.Tensor],
    after_parameters: list[torch.nn.Parameter],
) -> float:
    """Return the summed L1 change between two parameter snapshots.

    Includes the finiteness guard (present in update.py, missing in the
    stage24_collector twin, NEW-update.py-1): a non-finite total is rejected so
    a NaN/Inf update cannot masquerade as a nonzero parameter change.
    """

    total = 0.0
    for old, new in zip(before, after_parameters, strict=True):
        total += float((new.detach() - old).abs().sum().item())
    if not math.isfinite(total):
        raise ValueError("parameter_delta_l1 must be finite")
    return total

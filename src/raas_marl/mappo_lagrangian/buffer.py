"""Rollout-batch schema validation without runners or environment execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from raas_marl.mappo_lagrangian.config import (
    MAPPOCoreConfig,
    require_finite_float_tensor,
    torch_dtype_from_name,
    torch_required,
    validate_actor_tensors,
    validate_central_state,
)


if TYPE_CHECKING:  # pragma: no cover - annotation-only; never executed at runtime
    # THIS MODULE MUST IMPORT WITHOUT PYTORCH INSTALLED. That is a documented
    # architectural property, not an accident: the validation and configuration layer
    # is import-safe so it can be exercised in environments without a tensor library.
    #
    # `from __future__ import annotations` (above) makes every annotation a deferred
    # STRING, so a `torch.Tensor` in a signature is never evaluated, and every RUNTIME
    # use in this file binds a local name first, e.g.
    #     torch = torch_required("...")
    # This block exists purely so type checkers and linters can resolve the name; it
    # is skipped at run time, so the import-without-torch property is preserved.
    # Verified, not assumed: with `torch` blocked on sys.meta_path this module still
    # imports cleanly. Before this block, ruff reported the annotations as F821
    # "undefined name torch" and pyproject.toml carried a per-file suppression; the
    # suppression is now removed because the finding is gone rather than silenced.
    import torch


_ROLLOUT_BATCH_TENSOR_FIELDS = (
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


@dataclass(frozen=True)
class RolloutBatch:
    """Tensorized batch container consumed by MAPPO-Lagrangian loss helpers.

    ``RolloutBatch`` is tensor-only: it does not accept named actor-visible
    dictionaries and it has no identity metadata fields to validate.
    Named actor-leakage checks belong to ``ActorInput`` /
    ``validate_actor_input_mapping`` before tensorization. Environment
    team/agent identity checks belong to ``EnvironmentStepContract`` /
    ``validate_environment_step_contract`` before tensorization. This is a
    Stage 21 limitation, not a Stage 22 adapter implementation.
    """

    actor_observation: torch.Tensor
    revealed_information: torch.Tensor
    central_state: torch.Tensor
    sensing_action: torch.Tensor
    movement_action: torch.Tensor
    old_sensing_log_probability: torch.Tensor
    old_movement_log_probability: torch.Tensor
    task_reward: torch.Tensor
    hazard_cost: torch.Tensor
    sensing_cost: torch.Tensor
    reward_value: torch.Tensor
    hazard_cost_value: torch.Tensor
    next_reward_value: torch.Tensor
    next_hazard_cost_value: torch.Tensor
    terminal: torch.Tensor
    truncation: torch.Tensor
    valid_mask: torch.Tensor

    def validate(self, config: MAPPOCoreConfig) -> tuple[int, int]:
        if not isinstance(config, MAPPOCoreConfig):
            raise TypeError("config must be a MAPPOCoreConfig")
        tensor_fields = _tensor_field_items(self)
        _validate_required_tensors(tensor_fields)
        _validate_same_device(tensor_fields)
        batch, time = validate_actor_tensors(
            self.actor_observation,
            self.revealed_information,
            None,
            config,
        )
        critic_batch, critic_time = validate_central_state(self.central_state, config)
        if (critic_batch, critic_time) != (batch, time):
            raise ValueError("central_state batch/time dimensions do not match actor tensors")
        shape = (batch, time)
        _validate_action("sensing_action", self.sensing_action, shape, config.sensing_action_count)
        _validate_action("movement_action", self.movement_action, shape, config.movement_action_count)
        expected_dtype = torch_dtype_from_name(config.dtype)
        # Cost fields carry a contractual nonnegativity constraint; every other
        # float field is unbounded (log-probabilities in particular may be any sign,
        # matching the model-side FactorizedPolicyOutput contract). The nonnegative
        # flag is threaded through the single validation loop so the cost fields are
        # not re-fetched in a second getattr pass (NEW-buffer-6).
        _NONNEGATIVE_FLOAT_FIELDS = frozenset({"hazard_cost", "sensing_cost"})
        for name in (
            "old_sensing_log_probability",
            "old_movement_log_probability",
            "task_reward",
            "hazard_cost",
            "sensing_cost",
            "reward_value",
            "hazard_cost_value",
            "next_reward_value",
            "next_hazard_cost_value",
        ):
            _validate_float_tensor(
                name,
                getattr(self, name),
                shape,
                expected_dtype=expected_dtype,
                nonnegative=name in _NONNEGATIVE_FLOAT_FIELDS,
            )
        terminal = _validate_bool_tensor("terminal", self.terminal, shape)
        truncation = _validate_bool_tensor("truncation", self.truncation, shape)
        _validate_bool_tensor("valid_mask", self.valid_mask, shape)
        if (terminal & truncation).any().item():
            raise ValueError("terminal and truncation cannot both be true")
        return shape

    @property
    def old_joint_log_probability(self) -> torch.Tensor:
        # Assumes validate() has enforced equal [batch, time] shapes and a single
        # device across the two log-probability fields (NEW-buffer-3): both are
        # validated by _validate_float_tensor against the same ``shape`` and by
        # _validate_same_device. The addition below therefore never silently
        # broadcasts on the supported path. Accessing this property on an
        # unvalidated, mismatched batch is a caller contract violation.
        return self.old_sensing_log_probability + self.old_movement_log_probability


# Compatibility alias for the validated Stage 21 rollout-batch dataclass.
# It does not define separate Stage 22 adapter behavior.
RolloutBatchSpec = RolloutBatch


def _tensor_field_items(batch: RolloutBatch) -> tuple[tuple[str, object], ...]:
    return tuple((name, getattr(batch, name)) for name in _ROLLOUT_BATCH_TENSOR_FIELDS)


def _validate_required_tensors(fields: tuple[tuple[str, object], ...]) -> None:
    torch = torch_required("rollout batch required tensor validation")
    for name, value in fields:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")


def _validate_same_device(fields: tuple[tuple[str, object], ...]) -> None:
    # Cross-field device coherence only: this enforces that every rollout tensor
    # sits on one device, anchored on the first field. CPU-only pinning is not
    # enforced here (NEW-buffer-2) -- it is delegated to the upstream config tensor
    # validators (validate_actor_tensors / validate_central_state, which reject
    # meta tensors) applied to the actor/reveal/central fields before this batch is
    # assembled. The anchor device is rejected here only if it is a meta device, so
    # an all-meta batch fails with a named message rather than downstream.
    device = None
    device_name = None
    for name, value in fields:
        if device is None:
            device = value.device
            device_name = name
            if device.type == "meta":
                raise ValueError(f"{name} must not be a meta tensor")
        elif value.device != device:
            raise ValueError(
                f"{name} device must match {device_name} device for RolloutBatch"
            )


def _validate_float_tensor(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, int],
    *,
    expected_dtype: object = None,
    nonnegative: bool = False,
) -> None:
    # ``expected_dtype`` is an optional resolved ``torch.dtype`` (or ``None`` to skip
    # the dtype-vs-config check). It is typed ``object`` because torch is imported
    # lazily inside this module, so a ``torch.dtype`` annotation is not available at
    # module scope (NEW-buffer-5). The ``None`` default preserves the historical
    # three-argument call sites. ``nonnegative`` enables the contractual cost-field
    # lower-bound check in the same pass (NEW-buffer-6).
    torch = torch_required("rollout batch tensor validation")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != shape:
        raise ValueError(f"{name} shape mismatch")
    # Shared helper adds meta and sparse-layout rejection so batch tensors on
    # exotic devices/layouts fail deterministically (session 2026-07-04; S-5
    # propagation). Log-probability fields are validated as finite floats
    # only; a nonpositivity bound (log p <= 0) is deliberately not enforced,
    # matching the model-side FactorizedPolicyOutput contract.
    require_finite_float_tensor(name, value)
    # Session 2026-07-04 (S-3 analog): reward/cost/value/log-probability
    # tensors must match the configured dtype exactly, mirroring the
    # actor/central tensor enforcement, so mixed-precision batches are
    # rejected instead of silently promoted inside the update.
    if expected_dtype is not None and value.dtype != expected_dtype:
        raise ValueError(f"{name} dtype does not match config dtype")
    if nonnegative:
        # numel() guard is defensive only: validate_actor_tensors already requires a
        # positive batch/time so shape==(batch,time) is nonempty on the supported
        # path (NEW-buffer-1). It is retained so a hand-built zero-sized cost tensor
        # does not raise on an empty reduction.
        if value.numel() and (value < 0).any().item():
            raise ValueError(f"{name} must be nonnegative")


def _validate_bool_tensor(
    name: str, value: torch.Tensor, shape: tuple[int, int]
) -> torch.Tensor:
    torch = torch_required("rollout batch boolean tensor validation")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != shape:
        raise ValueError(f"{name} shape mismatch")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must use torch.bool dtype")
    if value.device.type == "meta":
        raise ValueError(f"{name} must not be a meta tensor")
    # Sparse/other non-strided layouts are rejected explicitly (S2-14-bool-layout):
    # boolean-mask ops such as (terminal & truncation).any().item() would otherwise
    # escape as an undeclared NotImplementedError, mirroring the strided-layout gate
    # in config.require_finite_float_tensor.
    if value.layout != torch.strided:
        raise ValueError(f"{name} must use a strided tensor layout")
    return value


def _validate_action(
    name: str,
    value: torch.Tensor,
    shape: tuple[int, int],
    action_count: int,
) -> None:
    torch = torch_required("rollout batch action tensor validation")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.shape != shape:
        raise ValueError(f"{name} shape mismatch")
    # Integer-action-dtype gate: rejects bool/float/complex/quantized dtypes. The
    # is_quantized clause matches the model-side and env-adapter gates; the strongest
    # form of this predicate is duplicated across buffer/model/env_adapter and the
    # stage23b/stage24 rollout twins (NEW-buffer-4). Unifying it into a shared
    # _rollout_common helper is deferred: the package file set is pinned, so no new
    # shared module may be added in this wave. The predicate here is already the
    # strongest of the copies.
    if (
        value.dtype == torch.bool
        or torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.is_quantized
    ):
        raise TypeError(f"{name} must use an integer dtype")
    if value.device.type == "meta":
        raise ValueError(f"{name} must not be a meta tensor")
    # Sparse/other non-strided layouts are rejected explicitly (S2-14-action-layout)
    # before .min()/.max().item(), which would otherwise escape as an undeclared
    # NotImplementedError; mirrors config.require_finite_float_tensor.
    if value.layout != torch.strided:
        raise ValueError(f"{name} must use a strided tensor layout")
    # numel() guard is defensive only: validate_actor_tensors already requires a
    # positive batch/time so shape==(batch,time) is nonempty on the supported path
    # (NEW-buffer-1). It is retained so a hand-built zero-sized action tensor does
    # not raise on an empty .min()/.max().
    if value.numel() and (value.min().item() < 0 or value.max().item() >= action_count):
        raise ValueError(f"{name} contains an out-of-range action index")

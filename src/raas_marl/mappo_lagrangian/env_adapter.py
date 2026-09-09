"""Development-only environment adapter for Stage 22 integration tests.

This module provides a small deterministic in-memory environment that maps
development payloads into the Stage 21 environment and tensor contracts. It is
not a final-evaluation environment and it is not claim evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import torch

from raas_marl.mappo_lagrangian._validation import (
    require_positive_int,
)
from raas_marl.mappo_lagrangian.config import (
    MAPPOCoreConfig,
    torch_dtype_from_name,
)
from raas_marl.mappo_lagrangian.env_contracts import (
    EnvironmentStepContract,
    validate_actor_visible_payload,
    validate_critic_central_payload,
    validate_environment_step_contract,
)

# Segment 1 dedup (Wave D2): the Stage 22 transition-tensorization family now
# lives in the shared _rollout_common module. env_adapter re-imports the shared
# implementations into its own namespace so that rollout.py — which does
# ``from ...env_adapter import _stage22_team_end_flags_ended,
# _validate_transition_for_tensorization, _validate_transition_identity_coverage``
# — keeps resolving them as attributes of this module. The thin local wrappers
# below preserve env_adapter's historical private names, the concrete
# ``DevelopmentTransition`` isinstance gate, and the exact call signatures the
# other modules rely on, while the bodies delegate to the single shared home.
from raas_marl.mappo_lagrangian._rollout_common import (
    declared_agent_order_from_transition as _declared_agent_order_from_transition,
    stage22_team_end_flags_ended as _stage22_team_end_flags_ended,
    tensor_from_mapping as _tensor_from_mapping,
    transition_agent_id as _transition_agent_id,
    validate_action_vector as _validate_action_vector,
    validate_transition_for_tensorization as _shared_validate_transition_for_tensorization,
    validate_transition_identity_coverage as _validate_transition_identity_coverage,
)


_STAGE22_ACTOR_OBSERVATION_DIM = 4
_STAGE22_REVEALED_INFORMATION_DIM = 2
_STAGE22_CENTRAL_STATE_DIM = 7
_STAGE22_AGENT_COUNT = 2


@dataclass(frozen=True)
class DevelopmentAdapterConfig:
    """Bounded settings for the deterministic Stage 22 development adapter."""

    core_config: MAPPOCoreConfig
    max_steps: int = 4
    agent_count: int = 2
    terminal_after_steps: int | None = 4
    dtype: torch.dtype = torch.float32
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not isinstance(self.core_config, MAPPOCoreConfig):
            raise TypeError("core_config must be a MAPPOCoreConfig")
        require_positive_int("max_steps", self.max_steps, error_suffix="must be positive")
        require_positive_int("agent_count", self.agent_count, error_suffix="must be positive")
        if self.core_config.actor_observation_dim != _STAGE22_ACTOR_OBSERVATION_DIM:
            raise ValueError("Stage 22 development adapter requires actor_observation_dim == 4")
        if self.core_config.revealed_information_dim != _STAGE22_REVEALED_INFORMATION_DIM:
            raise ValueError("Stage 22 development adapter requires revealed_information_dim == 2")
        if self.core_config.central_state_dim != _STAGE22_CENTRAL_STATE_DIM:
            raise ValueError("Stage 22 development adapter requires central_state_dim == 7")
        if self.core_config.agent_id_count != _STAGE22_AGENT_COUNT:
            raise ValueError("Stage 22 development adapter requires agent_id_count == 2")
        if self.agent_count != _STAGE22_AGENT_COUNT:
            raise ValueError("Stage 22 development adapter requires agent_count == 2")
        if self.max_steps > 16:
            raise ValueError("max_steps must be <= 16 for Stage 22 development")
        if self.terminal_after_steps is not None:
            require_positive_int(
                "terminal_after_steps", self.terminal_after_steps, error_suffix="must be positive"
            )
            if self.terminal_after_steps > self.max_steps:
                raise ValueError("terminal_after_steps must be <= max_steps")
        # Taxonomy fix (NEW-env_adapter-10): a wrong TYPE for dtype/device is a
        # TypeError; an in-range-but-wrong VALUE (e.g. torch.float16, a
        # whitespace-only device string, "cuda") stays a ValueError. The
        # historical message strings are preserved to avoid governance drift.
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if self.dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        _expected_dtype = torch_dtype_from_name(self.core_config.dtype)
        if self.dtype != _expected_dtype:
            raise ValueError("Stage 22 development adapter dtype must match core_config.dtype")
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        if not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.device != self.device.strip():
            raise ValueError("device must not contain leading or trailing whitespace")
        try:
            device = torch.device(self.device)
        except (RuntimeError, ValueError, TypeError) as exc:
            # ValueError added 2026-07-04 to mirror model._torch_device so a
            # torch-side ValueError cannot bypass the deterministic message.
            raise ValueError("device must be a valid torch device string") from exc
        if device.type != "cpu":
            raise ValueError("Stage 22 development adapter supports only cpu devices")
        if self.device != "cpu":
            raise ValueError("Stage 22 development adapter device must be exactly 'cpu'")
        if self.core_config.device != "cpu":
            raise ValueError("Stage 22 development adapter requires core_config.device == 'cpu'")
        if self.device != self.core_config.device:
            raise ValueError("Stage 22 development adapter device must match core_config.device")


@dataclass(frozen=True)
class DevelopmentTransition:
    """One validated development transition.

    The public environment done flag remains ``truncated``. Tensorized rollout
    batches use the Stage 21 internal field name ``truncation``.
    """

    contract: EnvironmentStepContract
    next_actor_visible: Mapping[str, object]
    next_critic_visible: Mapping[str, object]


def default_stage22_core_config() -> MAPPOCoreConfig:
    """Return the small CPU-safe core configuration used by Stage 22."""

    return MAPPOCoreConfig(
        actor_observation_dim=_STAGE22_ACTOR_OBSERVATION_DIM,
        revealed_information_dim=_STAGE22_REVEALED_INFORMATION_DIM,
        central_state_dim=_STAGE22_CENTRAL_STATE_DIM,
        history_state_dim=5,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
        sensing_action_count=2,
        movement_action_count=3,
        recurrent_layer_count=1,
        agent_id_count=_STAGE22_AGENT_COUNT,
        dtype="float32",
        device="cpu",
    )


class Stage22DevelopmentEnvironment:
    """Deterministic development-only environment adapter.

    Semantic leakage inside raw numeric actor tensors cannot be mathematically
    proven from tensor values alone; Stage 22 enforces leakage prevention through
    named-field validation, adapter contracts, and shape boundaries. The batch
    dimension produced by this adapter may represent ``episode_fragment x
    agent_id``. Agent identifiers remain validation/alignment metadata rather
    than privileged actor features.
    """

    def __init__(self, config: DevelopmentAdapterConfig | None = None) -> None:
        if config is None:
            self.config = DevelopmentAdapterConfig(default_stage22_core_config())
        elif isinstance(config, DevelopmentAdapterConfig):
            self.config = config
        else:
            raise TypeError("config must be a DevelopmentAdapterConfig or None")
        # Constructor state deliberately equals reset() state for this
        # deterministic toy adapter, so stepping before an explicit reset is
        # well-defined (unlike the Stage 23-A environment, which raises).
        # Registered 2026-07-04 as documented design.
        self._step_index = 0
        self._done = False

    @property
    def step_index(self) -> int:
        return self._step_index

    @property
    def is_done(self) -> bool:
        """Public done-ness accessor (parity with the Stage 23-A environment)."""

        return self._done

    def reset(self) -> list[Mapping[str, object]]:
        self._step_index = 0
        self._done = False
        return [self._payload(agent_id, revealed_from_sensing=False) for agent_id in self._agent_ids()]

    def step(
        self,
        sensing_actions: torch.Tensor,
        movement_actions: torch.Tensor,
    ) -> list[DevelopmentTransition]:
        if self._done:
            raise RuntimeError("environment is done; call reset before stepping again")
        _validate_action_vector(
            "sensing_actions",
            sensing_actions,
            self.config.agent_count,
            self.config.core_config.sensing_action_count,
        )
        _validate_action_vector(
            "movement_actions",
            movement_actions,
            self.config.agent_count,
            self.config.core_config.movement_action_count,
        )
        next_step = self._step_index + 1
        terminal = (
            self.config.terminal_after_steps is not None
            and next_step >= self.config.terminal_after_steps
        )
        truncated = next_step >= self.config.max_steps and not terminal
        transitions: list[DevelopmentTransition] = []
        for index, agent_id in enumerate(self._agent_ids()):
            sensing_action = int(sensing_actions[index].item())
            movement_action = int(movement_actions[index].item())
            payload = self._payload(
                agent_id,
                revealed_from_sensing=sensing_action > 0,
            )
            rewards_and_costs = self._rewards_and_costs(
                agent_id=agent_id,
                sensing_action=sensing_action,
                movement_action=movement_action,
            )
            contract = EnvironmentStepContract(
                actor_visible=payload["actor_visible"],
                critic_visible=payload["critic_visible"],
                rewards_and_costs=rewards_and_costs,
                done_flags={"terminal": terminal, "truncated": truncated},
                action_factors={
                    "sensing_action_available": True,
                    "movement_action_available": True,
                },
                identity={
                    "team_id": 0,
                    "agent_id": agent_id,
                    "agent_order": self._agent_ids(),
                },
            )
            validate_environment_step_contract(contract, self.config.core_config)
            next_payload = self._payload(
                agent_id,
                step_index=next_step,
                revealed_from_sensing=sensing_action > 0,
            )
            transitions.append(
                DevelopmentTransition(
                    contract=contract,
                    next_actor_visible=next_payload["actor_visible"],
                    next_critic_visible=next_payload["critic_visible"],
                )
            )
        self._step_index = next_step
        self._done = bool(transitions) and all(
            bool(item.contract.done_flags["terminal"])
            or bool(item.contract.done_flags["truncated"])
            for item in transitions
        )
        return transitions

    def _agent_ids(self) -> tuple[int, ...]:
        return tuple(range(self.config.agent_count))

    def _payload(
        self,
        agent_id: int,
        *,
        step_index: int | None = None,
        revealed_from_sensing: bool,
    ) -> dict[str, Mapping[str, object]]:
        # NEW-env_adapter-1 (documented, not changed): ``agent_id`` is
        # deliberately not folded into the actor observation / reveal / central
        # state. This deterministic Stage 22 toy adapter intentionally emits an
        # identical per-step payload for every agent; agent identity stays
        # validation/alignment metadata (carried in the contract identity), not
        # a privileged actor feature. The parameter is retained for signature
        # symmetry with the identity-bearing callers.
        del agent_id
        step = self._step_index if step_index is None else step_index
        core = self.config.core_config
        actor_observation = torch.tensor(
            [
                float(step) / float(max(1, self.config.max_steps)),
                float(self.config.max_steps - step) / float(max(1, self.config.max_steps)),
                math.sin(float(step)),
                1.0,
            ],
            dtype=self.config.dtype,
            device=self.config.device,
        )
        reveal_strength = 1.0 if revealed_from_sensing else 0.0
        revealed_information = torch.tensor(
            [
                reveal_strength,
                float(step % 2) * reveal_strength,
            ],
            dtype=self.config.dtype,
            device=self.config.device,
        )
        central_state = torch.tensor(
            [
                float(step),
                float(self.config.agent_count),
                float(self.config.max_steps),
                float(core.sensing_action_count),
                float(core.movement_action_count),
                float((step + 1) % self.config.max_steps),
                1.0,
            ],
            dtype=self.config.dtype,
            device=self.config.device,
        )
        actor_visible = {
            "actor_observation": actor_observation,
            "revealed_information": revealed_information,
            # NEW-env_adapter-6: .clone() breaks the view alias onto
            # revealed_information so a later in-place mutation of one tensor
            # cannot silently change the other. The cloned values are identical,
            # so the persisted payload is byte-for-byte unchanged.
            "revealed_local_hazard": revealed_information[:1].clone(),
        }
        critic_visible = {"central_state": central_state}
        validate_actor_visible_payload(actor_visible, core)
        validate_critic_central_payload(critic_visible, core)
        return {"actor_visible": actor_visible, "critic_visible": critic_visible}

    def _rewards_and_costs(
        self,
        *,
        agent_id: int,
        sensing_action: int,
        movement_action: int,
    ) -> dict[str, float]:
        target_movement = (self._step_index + agent_id) % self.config.core_config.movement_action_count
        task_reward = 1.0 if movement_action == target_movement else 0.25
        hazard_cost = 0.4 if movement_action == (agent_id + 1) % self.config.core_config.movement_action_count else 0.0
        sensing_cost = 0.1 if sensing_action > 0 else 0.0
        return {
            "task_reward": task_reward,
            "hazard_cost": hazard_cost,
            "sensing_cost": sensing_cost,
        }


def transitions_to_tensors(
    transitions: Sequence[DevelopmentTransition],
    config: MAPPOCoreConfig,
) -> dict[str, torch.Tensor]:
    """Convert one timestep of validated transitions into batch tensors."""

    if not isinstance(config, MAPPOCoreConfig):
        raise TypeError("config must be a MAPPOCoreConfig")
    # Session 2026-07-04: a generator would pass the emptiness check while
    # truthy, be exhausted by the validation loop, and then escape as an
    # undeclared stack error; require a stable, re-iterable sequence.
    # NEW-env_adapter-8: the explicit set/frozenset test is redundant with the
    # list/tuple membership test (a set is neither a list nor a tuple), so the
    # guard is simplified to the single positive check while keeping the
    # generator rationale above.
    if not isinstance(transitions, (list, tuple)):
        raise TypeError("transitions must be a list or tuple of DevelopmentTransition items")
    if not transitions:
        raise ValueError("transitions must be nonempty")
    for item in transitions:
        _validate_transition_for_tensorization(item, config)
    _validate_transition_identity_coverage(transitions, config)
    # NEW-env_adapter-4: assert cross-transition device coherence for every
    # stacked feature tensor before torch.stack, so a stray tensor on a
    # different device fails with a deterministic message here rather than as
    # an undeclared torch stacking error.
    _validate_transition_feature_device_coherence(transitions)
    actor_observation = torch.stack(
        [
            _tensor_from_mapping(item.contract.actor_visible, "actor_observation")
            for item in transitions
        ],
        dim=0,
    ).unsqueeze(1)
    revealed_information = torch.stack(
        [
            _tensor_from_mapping(item.contract.actor_visible, "revealed_information")
            for item in transitions
        ],
        dim=0,
    ).unsqueeze(1)
    central_state = torch.stack(
        [
            _tensor_from_mapping(item.contract.critic_visible, "central_state")
            for item in transitions
        ],
        dim=0,
    ).unsqueeze(1)
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
    _validate_stage22_team_end_flags(terminal, truncation)
    result = {
        "actor_observation": actor_observation,
        "revealed_information": revealed_information,
        "central_state": central_state,
        "terminal": terminal,
        "truncation": truncation,
        "valid_mask": torch.ones_like(terminal, dtype=torch.bool),
    }
    _validate_tensorized_shapes(result, config)
    return result


def _validate_stage22_team_end_flags(
    terminal: torch.Tensor,
    truncation: torch.Tensor,
) -> None:
    # Thin wrapper kept with its historical signature and None return. The
    # implementation now lives in the shared _rollout_common module (Segment 1
    # dedup, Wave D2); the collection-specific pinned wording is preserved
    # through the ``context`` argument, and the shared implementation adds
    # meta-device rejection (NEW-env_adapter-3) on top of the previous
    # shape/dtype/device pre-checks.
    _stage22_team_end_flags_ended(
        terminal,
        truncation,
        context="Stage 22 transition tensorization",
    )


def _validate_transition_feature_device_coherence(
    transitions: Sequence[DevelopmentTransition],
) -> None:
    """Assert every stacked feature tensor shares one device (NEW-env_adapter-4).

    ``transitions_to_tensors`` stacks the ``actor_observation`` /
    ``revealed_information`` / ``central_state`` tensors across transitions. A
    stray tensor on a different device would otherwise surface as an undeclared
    ``torch.stack`` RuntimeError; this pre-check raises a deterministic
    ValueError instead.
    """

    reference_device: torch.device | None = None
    for item in transitions:
        for mapping, key in (
            (item.contract.actor_visible, "actor_observation"),
            (item.contract.actor_visible, "revealed_information"),
            (item.contract.critic_visible, "central_state"),
        ):
            tensor = _tensor_from_mapping(mapping, key)
            if reference_device is None:
                reference_device = tensor.device
            elif tensor.device != reference_device:
                raise ValueError(
                    "transition feature tensors must all share one device"
                )


def _validate_transition_for_tensorization(
    item: DevelopmentTransition,
    config: MAPPOCoreConfig,
) -> None:
    # Thin wrapper preserving env_adapter's historical private name and the
    # concrete ``DevelopmentTransition`` isinstance gate for callers (rollout.py
    # and transitions_to_tensors) that call it without a ``transition_type``.
    # The shared implementation keeps the exact "transitions must contain
    # DevelopmentTransition items" TypeError message when the concrete type is
    # supplied (Segment 1 dedup, Wave D2).
    _shared_validate_transition_for_tensorization(
        item,
        config,
        transition_type=DevelopmentTransition,
    )


def _validate_tensorized_shapes(
    tensors: Mapping[str, torch.Tensor],
    config: MAPPOCoreConfig,
) -> None:
    expected = {
        "actor_observation": config.actor_observation_dim,
        "revealed_information": config.revealed_information_dim,
        "central_state": config.central_state_dim,
    }
    expected_dtype = torch_dtype_from_name(config.dtype)
    for name, dim in expected.items():
        value = tensors[name]
        if value.ndim != 3:
            raise ValueError(f"{name} must have shape [batch, time, dim]")
        if value.shape[-1] != dim:
            raise ValueError(f"{name} final dimension mismatch")
        # Session 2026-07-04 (S-3 analog): stacked feature tensors must match
        # the configured dtype so mixed-precision payloads fail here rather
        # than downstream in the update.
        if value.dtype != expected_dtype:
            raise ValueError(f"{name} dtype does not match config dtype")
        # NEW-env_adapter-7: the Stage 22 adapter is cpu-only, so a stray
        # non-cpu feature tensor must fail deterministically here rather than
        # downstream.
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a cpu tensor")
    for name in ("terminal", "truncation", "valid_mask"):
        value = tensors[name]
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, time]")
        if value.dtype != torch.bool:
            raise TypeError(f"{name} must use torch.bool dtype")
        # NEW-env_adapter-7: flag tensors are likewise cpu-only.
        if value.device.type != "cpu":
            raise ValueError(f"{name} must be a cpu tensor")


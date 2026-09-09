"""PyTorch model components for the Stage 21 MAPPO-Lagrangian core."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch
from torch import nn
from torch.distributions import Categorical

from raas_marl.mappo_lagrangian.config import (
    MAPPOCoreConfig,
    require_finite_float_tensor,
    require_torch_tensor,
    torch_dtype_from_name,
    validate_actor_input_mapping,
    validate_actor_tensors,
    validate_central_state,
)


@dataclass(frozen=True)
class ActorInput:
    """Actor-visible tensors for decentralized execution.

    ``actor_observation`` represents local observation only. Explicitly revealed
    information is carried by ``revealed_information``; central state and hidden
    hazard state are rejected by named-field validation. ``history_state`` is a
    history-conditioned recurrent state with shape
    ``[recurrent_layer_count, batch, history_state_dim]`` when supplied.
    Reveal-specific named fields are adapter/provenance-level fields only and
    must be packed into ``revealed_information`` before model execution.
    """

    actor_observation: torch.Tensor
    revealed_information: torch.Tensor
    history_state: torch.Tensor | None = None
    extra_fields: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_actor_input_mapping(self.extra_fields)
        if self.extra_fields:
            raise ValueError(
                "executable ActorInput accepts only actor_observation, "
                "revealed_information, and history_state; pack reveal content "
                "into revealed_information before model execution"
            )
        _validate_actor_input_tensor_surface(
            self.actor_observation,
            self.revealed_information,
            self.history_state,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, object]) -> "ActorInput":
        validate_actor_input_mapping(mapping)
        required = {"actor_observation", "revealed_information"}
        missing = sorted(required - set(mapping))
        if missing:
            raise ValueError("actor input mapping is missing: " + ", ".join(missing))
        allowed = {"actor_observation", "revealed_information", "history_state"}
        extra_keys = sorted(key for key in mapping if key not in allowed)
        if extra_keys:
            raise ValueError(
                "executable ActorInput accepts only actor_observation, "
                "revealed_information, and history_state; pack reveal content "
                "into revealed_information before model execution: "
                + ", ".join(extra_keys)
            )
        return cls(
            actor_observation=_require_tensor_value(
                "actor_observation", mapping["actor_observation"]
            ),
            revealed_information=_require_tensor_value(
                "revealed_information", mapping["revealed_information"]
            ),
            history_state=_optional_tensor_value("history_state", mapping),
        )


@dataclass(frozen=True)
class CriticInput:
    """Centralized critic tensor used only during centralized training steps."""

    central_state: torch.Tensor

    def __post_init__(self) -> None:
        # Surface-level type guard so a non-tensor central_state fails at
        # construction with a deterministic TypeError, symmetric with
        # ActorInput/FactorizedPolicyOutput. Full shape/dtype/device
        # validation stays in validate_central_state at critic-evaluation
        # time (against the model config).
        require_torch_tensor("central_state", self.central_state)


@dataclass(frozen=True)
class FactorizedPolicyOutput:
    """Action-factor log probabilities, entropies, and chosen action indices."""

    sensing_action: torch.Tensor
    movement_action: torch.Tensor
    sensing_log_probability: torch.Tensor
    movement_log_probability: torch.Tensor
    sensing_entropy: torch.Tensor
    movement_entropy: torch.Tensor
    sensing_action_count: int
    movement_action_count: int

    def __post_init__(self) -> None:
        _validate_policy_output_action_count(
            "sensing_action_count",
            self.sensing_action_count,
        )
        _validate_policy_output_action_count(
            "movement_action_count",
            self.movement_action_count,
        )
        _validate_factorized_policy_output_tensor_surface(self)

    @property
    def joint_log_probability(self) -> torch.Tensor:
        return self.sensing_log_probability + self.movement_log_probability

    @property
    def joint_entropy(self) -> torch.Tensor:
        return self.sensing_entropy + self.movement_entropy


def _validate_factorized_policy_output_tensor_surface(
    output: FactorizedPolicyOutput,
) -> None:
    tensors = {
        "sensing_action": output.sensing_action,
        "movement_action": output.movement_action,
        "sensing_log_probability": output.sensing_log_probability,
        "movement_log_probability": output.movement_log_probability,
        "sensing_entropy": output.sensing_entropy,
        "movement_entropy": output.movement_entropy,
    }
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        # Sparse/quantized-style layouts are rejected up front because the
        # min()/max()/isfinite().item() reductions below escape as undeclared
        # torch errors on non-strided tensors (NEW-model.py-2; same class of
        # gap as the meta-tensor rejections further down).
        if value.layout != torch.strided:
            raise ValueError(f"{name} must use a strided tensor layout")

    expected_shape = tensors["sensing_action"].shape
    if len(expected_shape) != 2:
        raise ValueError("factorized policy output tensors must have shape [batch, time]")
    if expected_shape[0] <= 0 or expected_shape[1] <= 0:
        raise ValueError(
            "factorized policy output tensors must have positive batch and time dimensions"
        )
    for name, tensor in tensors.items():
        if tensor.shape != expected_shape:
            raise ValueError(f"{name} shape must match sensing_action shape")
    expected_device = tensors["sensing_action"].device
    for name, tensor in tensors.items():
        if tensor.device != expected_device:
            raise ValueError(f"{name} device must match sensing_action device")

    for name in ("sensing_action", "movement_action"):
        tensor = tensors[name]
        if (
            tensor.dtype == torch.bool
            or torch.is_floating_point(tensor)
            or torch.is_complex(tensor)
            or tensor.is_quantized
        ):
            raise TypeError(f"{name} must use an integer dtype")
        if tensor.device.type == "meta":
            raise ValueError(f"{name} must not be a meta tensor")
        if tensor.numel():
            if tensor.min().item() < 0:
                raise ValueError(f"{name} contains a negative action index")
            action_count = (
                output.sensing_action_count
                if name == "sensing_action"
                else output.movement_action_count
            )
            if tensor.max().item() >= action_count:
                raise ValueError(f"{name} contains an out-of-range action index")

    # Note (registered 2026-07-04): log-probability fields are validated as
    # finite floats only; a nonpositivity bound (log p <= 0 for categorical
    # factors) is deliberately not enforced so hand-constructed diagnostic
    # outputs remain constructible under the Stage 21 contract.
    float_dtype = None
    for name in (
        "sensing_log_probability",
        "movement_log_probability",
        "sensing_entropy",
        "movement_entropy",
    ):
        tensor = tensors[name]
        if not torch.is_floating_point(tensor):
            raise TypeError(f"{name} must use a floating point dtype")
        # Session 2026-07-04 (S-3 analog): the four floating fields must agree
        # on dtype, mirroring the shape/device agreement above.
        if float_dtype is None:
            float_dtype = tensor.dtype
        elif tensor.dtype != float_dtype:
            raise ValueError(f"{name} dtype must match sensing_log_probability dtype")
        if tensor.device.type == "meta":
            raise ValueError(f"{name} must not be a meta tensor")
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"{name} must contain only finite values")

    for name in ("sensing_entropy", "movement_entropy"):
        tensor = tensors[name]
        if tensor.numel() and tensor.min().item() < 0:
            raise ValueError(f"{name} must contain only nonnegative values")


class RecurrentMAPPOActorCritic(nn.Module):
    """Factorized actor with separate reward and cost critics.

    The actor uses local observation, explicitly revealed information, and a
    history-conditioned recurrent state. Critic value paths receive central
    state only through ``CriticInput``. Both critics are per-timestep
    feedforward MLPs over the fully observed central state; only the actor is
    recurrent. This asymmetry is deliberate (the MAPPO reference
    implementation also supports recurrent critics; CLAUDE.md issue D-10).
    """

    def __init__(self, config: MAPPOCoreConfig) -> None:
        if not isinstance(config, MAPPOCoreConfig):
            raise TypeError("config must be a MAPPOCoreConfig")
        super().__init__()
        self.config = config
        self.actor_encoder = nn.Linear(config.actor_input_dim, config.actor_hidden_dim)
        self.actor_recurrent = nn.GRU(
            input_size=config.actor_hidden_dim,
            hidden_size=config.history_state_dim,
            num_layers=config.recurrent_layer_count,
            batch_first=True,
        )
        self.sensing_head = nn.Linear(
            config.history_state_dim, config.sensing_action_count
        )
        self.movement_head = nn.Linear(
            config.history_state_dim, config.movement_action_count
        )
        self.reward_critic = _critic_network(
            config.central_state_dim, config.critic_hidden_dim
        )
        self.cost_critic = _critic_network(
            config.central_state_dim, config.critic_hidden_dim
        )
        target_device = _torch_device(config.device)
        target_dtype = torch_dtype_from_name(config.dtype)
        try:
            self.to(device=target_device, dtype=target_dtype)
        except (RuntimeError, AssertionError, ValueError) as exc:
            raise ValueError(
                "model device is valid but unavailable in this runtime"
            ) from exc

    def forward_actor(
        self, actor_input: ActorInput
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if not isinstance(actor_input, ActorInput):
            raise TypeError("actor_input must be an ActorInput")
        expected_dtype, expected_device = self._parameter_dtype_device()
        _validate_model_tensor_contract(
            "actor_observation",
            actor_input.actor_observation,
            expected_dtype,
            expected_device,
        )
        _validate_model_tensor_contract(
            "revealed_information",
            actor_input.revealed_information,
            expected_dtype,
            expected_device,
        )
        if actor_input.history_state is not None:
            _validate_model_tensor_contract(
                "history_state",
                actor_input.history_state,
                expected_dtype,
                expected_device,
            )
        batch, _ = validate_actor_tensors(
            actor_input.actor_observation,
            actor_input.revealed_information,
            actor_input.history_state,
            self.config,
        )
        actor_features = torch.cat(
            [actor_input.actor_observation, actor_input.revealed_information], dim=-1
        )
        encoded = torch.tanh(self.actor_encoder(actor_features))
        initial_state = self._initial_actor_state(actor_input.history_state, batch)
        recurrent_output, final_state = self.actor_recurrent(encoded, initial_state)
        sensing_logits = self.sensing_head(recurrent_output)
        movement_logits = self.movement_head(recurrent_output)
        return sensing_logits, movement_logits, final_state

    def evaluate_reward_critic(self, critic_input: CriticInput) -> torch.Tensor:
        if not isinstance(critic_input, CriticInput):
            raise TypeError("critic_input must be a CriticInput")
        expected_dtype, expected_device = self._parameter_dtype_device()
        _validate_model_tensor_contract(
            "central_state",
            critic_input.central_state,
            expected_dtype,
            expected_device,
        )
        validate_central_state(critic_input.central_state, self.config)
        return self.reward_critic(critic_input.central_state).squeeze(-1)

    def evaluate_cost_critic(self, critic_input: CriticInput) -> torch.Tensor:
        if not isinstance(critic_input, CriticInput):
            raise TypeError("critic_input must be a CriticInput")
        expected_dtype, expected_device = self._parameter_dtype_device()
        _validate_model_tensor_contract(
            "central_state",
            critic_input.central_state,
            expected_dtype,
            expected_device,
        )
        validate_central_state(critic_input.central_state, self.config)
        return self.cost_critic(critic_input.central_state).squeeze(-1)

    def forward(
        self,
        actor_input: ActorInput,
        critic_input: CriticInput,
        *,
        sensing_action: torch.Tensor | None = None,
        movement_action: torch.Tensor | None = None,
        sample_actions: bool = False,
    ) -> dict[str, torch.Tensor | FactorizedPolicyOutput]:
        # sensing_action/movement_action are keyword-only so the two same-typed
        # action tensors cannot be swapped positionally (NEW-model.py-5); all
        # in-tree callers already pass them by keyword.
        sensing_logits, movement_logits, next_recurrent_state = self.forward_actor(
            actor_input
        )
        policy = policy_from_logits(
            sensing_logits=sensing_logits,
            movement_logits=movement_logits,
            sensing_action=sensing_action,
            movement_action=movement_action,
            sample_actions=sample_actions,
        )
        reward_value = self.evaluate_reward_critic(critic_input)
        hazard_cost_value = self.evaluate_cost_critic(critic_input)
        return {
            "sensing_logits": sensing_logits,
            "movement_logits": movement_logits,
            "next_recurrent_state": next_recurrent_state,
            "policy": policy,
            "reward_value": reward_value,
            "hazard_cost_value": hazard_cost_value,
        }

    def _initial_actor_state(
        self, history_state: torch.Tensor | None, batch: int
    ) -> torch.Tensor:
        if history_state is None:
            weight = self.actor_encoder.weight
            return weight.new_zeros(
                self.config.recurrent_layer_count,
                batch,
                self.config.history_state_dim,
            )
        return history_state

    def _parameter_dtype_device(self) -> tuple[torch.dtype, torch.device]:
        parameter = next(self.parameters())
        return parameter.dtype, parameter.device


def policy_from_logits(
    sensing_logits: torch.Tensor,
    movement_logits: torch.Tensor,
    *,
    sensing_action: torch.Tensor | None = None,
    movement_action: torch.Tensor | None = None,
    sample_actions: bool = False,
) -> FactorizedPolicyOutput:
    """Build separate categorical distributions for sensing and movement.

    Missing actions are deterministic argmax choices unless ``sample_actions``
    is explicitly set to ``True``.

    ``sensing_action`` and ``movement_action`` are keyword-only so the two
    same-typed action tensors cannot be silently swapped positionally
    (NEW-model.py-5); all in-tree callers already pass them by keyword.

    When ``sample_actions`` is ``True`` the categorical draws consume the
    global torch RNG (``.sample()`` with no local generator); reproducibility
    is therefore controlled by the caller's ``torch.manual_seed`` (as the
    rollout collectors do). Threading an optional ``torch.Generator`` for the
    reviewed final-training path is queued for Phase 6 (NEW-model.py-6).
    """

    _validate_logits("sensing_logits", sensing_logits)
    _validate_logits("movement_logits", movement_logits)
    if not isinstance(sample_actions, bool):
        raise TypeError("sample_actions must be a bool")
    if sensing_logits.shape[:2] != movement_logits.shape[:2]:
        raise ValueError("sensing and movement logits batch/time dimensions differ")
    if sensing_logits.dtype != movement_logits.dtype:
        raise ValueError("sensing and movement logits dtype mismatch")
    if sensing_logits.device != movement_logits.device:
        raise ValueError("sensing and movement logits device mismatch")
    require_finite_float_tensor("sensing_logits", sensing_logits)
    require_finite_float_tensor("movement_logits", movement_logits)
    sensing_distribution = Categorical(logits=sensing_logits)
    movement_distribution = Categorical(logits=movement_logits)
    if sensing_action is None:
        if sample_actions:
            sensing_action = sensing_distribution.sample()
        else:
            sensing_action = sensing_logits.argmax(dim=-1)
    else:
        _validate_action_tensor(
            "sensing_action",
            sensing_action,
            sensing_logits.shape[:2],
            sensing_logits.shape[-1],
            sensing_logits.device,
        )
        sensing_action = sensing_action.to(dtype=torch.long)
    if movement_action is None:
        if sample_actions:
            movement_action = movement_distribution.sample()
        else:
            movement_action = movement_logits.argmax(dim=-1)
    else:
        _validate_action_tensor(
            "movement_action",
            movement_action,
            movement_logits.shape[:2],
            movement_logits.shape[-1],
            movement_logits.device,
        )
        movement_action = movement_action.to(dtype=torch.long)
    return FactorizedPolicyOutput(
        sensing_action=sensing_action,
        movement_action=movement_action,
        sensing_log_probability=sensing_distribution.log_prob(sensing_action),
        movement_log_probability=movement_distribution.log_prob(movement_action),
        sensing_entropy=sensing_distribution.entropy(),
        movement_entropy=movement_distribution.entropy(),
        sensing_action_count=int(sensing_logits.shape[-1]),
        movement_action_count=int(movement_logits.shape[-1]),
    )


def _critic_network(input_dim: int, hidden_dim: int) -> nn.Sequential:
    # D-10 (decision queued for Phase 6): the critics are per-timestep
    # feedforward MLPs over the fully observed central state; only the actor is
    # recurrent. This factory is kept factored so a recurrent critic module can
    # replace it wholesale at the reviewed final-training update without
    # touching the actor path. The feedforward choice is the current, shipped
    # behavior and is intentionally not changed here.
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    )


def _torch_device(name: str) -> torch.device:
    try:
        return torch.device(name)
    except (RuntimeError, ValueError) as exc:
        raise ValueError("device must be parseable by torch.device") from exc


def _validate_logits(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 3:
        raise ValueError(f"{name} must have shape [batch, time, action]")
    if value.shape[0] <= 0 or value.shape[1] <= 0:
        raise ValueError(f"{name} batch and time dimensions must be positive")
    if value.shape[-1] < 2:
        raise ValueError(f"{name} must expose at least two actions")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating point dtype")


def _validate_action_tensor(
    name: str,
    value: torch.Tensor,
    expected_shape: torch.Size,
    action_count: int,
    expected_device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, time]")
    if value.shape != expected_shape:
        raise ValueError(f"{name} shape must match logits batch/time dimensions")
    if value.device != expected_device:
        raise ValueError(f"{name} device must match corresponding logits device")
    if (
        value.dtype == torch.bool
        or torch.is_floating_point(value)
        or torch.is_complex(value)
        or value.is_quantized
    ):
        raise TypeError(f"{name} must use an integer dtype")
    # Meta-device and non-strided (sparse) tensors are rejected before the
    # min()/max().item() range reduction, which would otherwise escape as an
    # undeclared RuntimeError/NotImplementedError (NEW-model.py-3; mirrors the
    # FactorizedPolicyOutput surface and config.require_finite_float_tensor).
    if value.device.type == "meta":
        raise ValueError(f"{name} must not be a meta tensor")
    if value.layout != torch.strided:
        raise ValueError(f"{name} must use a strided tensor layout")
    if value.numel() and (value.min().item() < 0 or value.max().item() >= action_count):
        raise ValueError(f"{name} contains an out-of-range action index")


def _validate_model_tensor_contract(
    name: str,
    value: torch.Tensor,
    expected_dtype: torch.dtype,
    expected_device: torch.device,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != expected_dtype:
        raise ValueError(f"{name} dtype must match model parameter dtype")
    if value.device != expected_device:
        raise ValueError(f"{name} device must match model parameter device")


def _validate_policy_output_action_count(name: str, value: int) -> None:
    # Deliberately stricter than _validation.require_int_not_bool: exact-type
    # int (subclasses rejected) is the pinned Stage 21 policy-output contract.
    if type(value) is not int:
        raise TypeError(f"{name} must be an int")
    if value < 2:
        raise ValueError(f"{name} must be an integer >= 2")


def _require_tensor_value(name: str, value: object) -> torch.Tensor:
    # Delegates to the shared config helper (session 2026-07-04 dedup) and
    # keeps the value-returning convenience shape used across this module.
    require_torch_tensor(name, value)
    return value


def _optional_tensor_value(
    name: str, mapping: Mapping[str, object]
) -> torch.Tensor | None:
    if name not in mapping or mapping[name] is None:
        return None
    return _require_tensor_value(name, mapping[name])


def _validate_actor_input_tensor_surface(
    actor_observation: torch.Tensor,
    revealed_information: torch.Tensor,
    history_state: torch.Tensor | None,
) -> None:
    _require_actor_float_tensor("actor_observation", actor_observation, ndim=3)
    _require_actor_float_tensor("revealed_information", revealed_information, ndim=3)
    batch, time = actor_observation.shape[:2]
    if batch <= 0 or time <= 0:
        raise ValueError("actor_observation batch and time dimensions must be positive")
    if revealed_information.shape[:2] != (batch, time):
        raise ValueError("revealed_information batch/time dimensions must match")
    if history_state is not None:
        _require_actor_float_tensor("history_state", history_state, ndim=3)
        if history_state.shape[1] != batch:
            raise ValueError("history_state batch dimension must match actor_observation")


def _require_actor_float_tensor(name: str, value: object, *, ndim: int) -> torch.Tensor:
    tensor = _require_tensor_value(name, value)
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    # The shared helper adds meta-tensor rejection so a meta-device tensor
    # fails with a deterministic ValueError instead of an undeclared
    # RuntimeError (session 2026-07-04; S-5 analog for the model surface).
    require_finite_float_tensor(name, tensor)
    return tensor

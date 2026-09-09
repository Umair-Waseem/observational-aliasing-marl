"""GAE, PPO clipping, and Lagrangian objective helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from raas_marl.mappo_lagrangian._validation import (
    finite_numeric_scalar,
    is_numeric_scalar,
    require_open_unit_interval,
    require_probability,
)
from raas_marl.mappo_lagrangian.config import (
    require_finite_float_tensor,
    torch_required,
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


_BOOLEAN_LOSS_INPUT_FIELDS = frozenset(
    {"valid_mask", "terminal", "truncation", "terminals", "truncations"}
)
_TERMINAL_ALIASES = ("terminal", "terminals")
_TRUNCATION_ALIASES = ("truncation", "truncations")


def compute_gae(
    signals: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminal: torch.Tensor,
    truncation: torch.Tensor,
    valid_mask: torch.Tensor | None,
    gamma: float,
    lam: float,
    *,
    require_nonnegative_signals: bool = False,
) -> torch.Tensor:
    """Compute masked generalized advantage estimates for ``[batch, time]`` tensors.

    ``terminal`` blocks bootstrapping. ``truncation`` keeps the one-step value
    bootstrap but stops recursive carry across a sequence boundary.
    """

    torch = torch_required("GAE tensor execution")
    # Capture the coerced primitives and use them in the delta / step-advantage
    # arithmetic below: require_probability normalizes through the base-class
    # __float__ slot, so a hostile int/float subclass cannot substitute a
    # different value into the recursion (NEW-losses-5; mirrors how
    # ppo_clipped_surrogate consumes _validate_clip_epsilon's return).
    gamma = require_probability("gamma", gamma)
    lam = require_probability("lam", lam)
    _validate_same_shape_2d(
        signals=signals,
        values=values,
        next_values=next_values,
        terminal=terminal,
        truncation=truncation,
    )
    mask = _normalise_valid_mask(valid_mask, signals)
    require_finite_float_tensor("signals", signals)
    require_finite_float_tensor("values", values)
    require_finite_float_tensor("next_values", next_values)
    if require_nonnegative_signals and signals.numel() and (signals < 0).any().item():
        raise ValueError("raw cost signals must be nonnegative")
    bootstrap_mask = bootstrap_allowed_mask(terminal).to(signals.dtype)
    carry_mask = recursive_carry_mask(terminal, truncation, mask).to(signals.dtype)
    advantages = torch.zeros_like(signals)
    next_advantage = torch.zeros(signals.shape[0], dtype=signals.dtype, device=signals.device)
    for step in range(signals.shape[1] - 1, -1, -1):
        valid_step = mask[:, step]
        delta = (
            signals[:, step]
            + gamma * bootstrap_mask[:, step] * next_values[:, step]
            - values[:, step]
        )
        _require_finite_tensor_after_arithmetic("GAE delta", delta)
        step_advantage = delta + gamma * lam * carry_mask[:, step] * next_advantage
        _require_finite_tensor_after_arithmetic("GAE step_advantage", step_advantage)
        step_advantage = torch.where(valid_step, step_advantage, torch.zeros_like(step_advantage))
        advantages[:, step] = step_advantage
        next_advantage = torch.where(valid_step, step_advantage, torch.zeros_like(step_advantage))
    _require_finite_tensor_after_arithmetic("GAE advantages", advantages)
    return advantages


def bootstrap_allowed_mask(terminal: torch.Tensor) -> torch.Tensor:
    """Return ``not terminal`` for each ``[batch, time]`` timestep."""

    terminal_bool = _validate_bool_mask_2d("terminal", terminal)
    return ~terminal_bool


def sequence_boundary_mask(
    terminal: torch.Tensor,
    truncation: torch.Tensor,
) -> torch.Tensor:
    """Return true where terminal or truncation ends recursive sequence carry."""

    terminal_bool = _validate_bool_mask_2d("terminal", terminal)
    truncation_bool = _validate_bool_mask_2d("truncation", truncation)
    if terminal_bool.shape != truncation_bool.shape:
        raise ValueError("terminal and truncation shape mismatch")
    if terminal_bool.device != truncation_bool.device:
        raise ValueError("terminal and truncation device mismatch")
    if (terminal_bool & truncation_bool).any().item():
        raise ValueError("terminal and truncation cannot both be true at one step")
    return terminal_bool | truncation_bool


def recursive_carry_mask(
    terminal: torch.Tensor,
    truncation: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return whether advantage recursion at ``t`` may carry to ``t + 1``."""

    torch = torch_required("recursive carry mask tensor validation")
    boundary = sequence_boundary_mask(terminal, truncation)
    valid = _validate_bool_mask_2d("valid_mask", valid_mask)
    if valid.shape != boundary.shape:
        raise ValueError("valid_mask shape mismatch")
    if valid.device != boundary.device:
        raise ValueError("valid_mask device must match terminal/truncation device")
    carry = torch.zeros_like(valid, dtype=torch.bool)
    if valid.shape[1] > 1:
        carry[:, :-1] = valid[:, :-1] & valid[:, 1:] & ~boundary[:, :-1]
    return carry


def validate_loss_input_mapping(mapping: Mapping[str, object]) -> None:
    """Validate dictionary-style raw loss inputs without combining fields."""

    if not isinstance(mapping, Mapping):
        raise TypeError("loss input must be a mapping")
    for key in mapping:
        if not isinstance(key, str):
            raise TypeError("loss input mapping keys must be strings")
    required = ("hazard_cost", "sensing_cost", "task_reward")
    missing = sorted(key for key in required if key not in mapping)
    if missing:
        raise ValueError("loss input mapping is missing: " + ", ".join(missing))
    tensor_shape: tuple[int, int] | None = None
    tensor_device = None
    tensor_float_dtype = None
    torch = None
    boolean_tensors: dict[str, object] = {}
    for name, value in mapping.items():
        if name in _BOOLEAN_LOSS_INPUT_FIELDS:
            if torch is None:
                torch = torch_required("loss input mapping tensor validation")
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.ndim != 2:
                raise ValueError(f"{name} must have shape [batch, time]")
            # Meta rejection is hoisted ahead of the shape/device/dtype checks
            # so a meta mask fails with its own message before any downstream
            # .item() read (NEW-losses-10).
            if value.device.type == "meta":
                raise ValueError(f"{name} must not be a meta tensor")
            # Sparse/other non-strided boolean masks are rejected explicitly
            # (S2-14 analog): the terminal/truncation overlap check
            # (terminal & truncation).any().item() would otherwise escape as an
            # undeclared NotImplementedError; mirrors
            # buffer._validate_bool_tensor's strided gate.
            if value.layout != torch.strided:
                raise ValueError(f"{name} must use a strided tensor layout")
            if tensor_shape is None:
                tensor_shape = tuple(value.shape)
            elif tuple(value.shape) != tensor_shape:
                raise ValueError(f"{name} shape does not match previous loss tensors")
            if tensor_device is None:
                tensor_device = value.device
            elif value.device != tensor_device:
                raise ValueError(f"{name} device does not match previous loss tensors")
            if value.dtype != torch.bool:
                raise TypeError(f"{name} must use torch.bool dtype")
            boolean_tensors[name] = value
            continue
        if is_numeric_scalar(value):
            # Scalars are dtype-free Python floats and are intentionally exempt
            # from the cross-tensor float-dtype coherence enforced on the tensor
            # fields below; only tensors carry a torch dtype (NEW-losses-9).
            _validate_scalar_loss_value(name, value)
            continue
        if torch is None:
            torch = torch_required("loss input mapping tensor validation")
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a numeric scalar or torch.Tensor")
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, time]")
        if tensor_shape is None:
            tensor_shape = tuple(value.shape)
        elif tuple(value.shape) != tensor_shape:
            raise ValueError(f"{name} shape does not match previous loss tensors")
        if tensor_device is None:
            tensor_device = value.device
        elif value.device != tensor_device:
            raise ValueError(f"{name} device does not match previous loss tensors")
        if value.dtype == torch.bool:
            allowed = ", ".join(sorted(_BOOLEAN_LOSS_INPUT_FIELDS))
            raise TypeError(f"{name} boolean tensor is only allowed for: {allowed}")
        if not torch.is_floating_point(value):
            raise TypeError(f"{name} must use a floating point dtype")
        if value.device.type == "meta":
            raise ValueError(f"{name} must not be a meta tensor")
        # Session 2026-07-04 (S-3 analog): floating loss tensors must agree on
        # dtype, mirroring the shape/device coherence checks above.
        if tensor_float_dtype is None:
            tensor_float_dtype = value.dtype
        elif value.dtype != tensor_float_dtype:
            raise ValueError(f"{name} dtype does not match previous loss tensors")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")
        if name in {"hazard_cost", "sensing_cost"} and value.numel():
            if (value < 0).any().item():
                raise ValueError(f"{name} must be nonnegative")
    _validate_terminal_truncation_input_pairs(boolean_tensors)


def compute_reward_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminal: torch.Tensor,
    truncation: torch.Tensor,
    valid_mask: torch.Tensor | None,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    return compute_gae(
        rewards,
        values,
        next_values,
        terminal,
        truncation,
        valid_mask,
        gamma,
        lam,
    )


def compute_cost_gae(
    costs: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminal: torch.Tensor,
    truncation: torch.Tensor,
    valid_mask: torch.Tensor | None,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    return compute_gae(
        costs,
        values,
        next_values,
        terminal,
        truncation,
        valid_mask,
        gamma,
        lam,
        require_nonnegative_signals=True,
    )


def ppo_clipped_surrogate(
    new_log_probability: torch.Tensor,
    old_log_probability: torch.Tensor,
    advantage: torch.Tensor,
    clip_epsilon: float,
) -> torch.Tensor:
    """Return the elementwise clipped PPO surrogate."""

    torch = torch_required("PPO clipped surrogate tensor execution")
    # require_open_unit_interval returns the coerced (0, 1) float used below in
    # the clamp bounds (NEW-losses-4: the former _validate_clip_epsilon /
    # _validate_discount wrappers were thin single-purpose delegators).
    clip_value = require_open_unit_interval("clip_epsilon", clip_epsilon)
    _validate_same_shape_2d(
        new_log_probability=new_log_probability,
        old_log_probability=old_log_probability,
        advantage=advantage,
    )
    require_finite_float_tensor("new_log_probability", new_log_probability)
    require_finite_float_tensor("old_log_probability", old_log_probability)
    require_finite_float_tensor("advantage", advantage)
    log_ratio = new_log_probability - old_log_probability
    _require_finite_tensor_after_arithmetic("policy log probability ratio", log_ratio)
    ratio = torch.exp(log_ratio)
    # Deliberate fail-fast policy (issue D-9): a log-ratio large enough to
    # overflow exp() aborts the update instead of being clamped. This is the
    # current bounded-development behavior; the abort-vs-log-ratio-clamp choice
    # is queued for the Phase 6 reviewed final-training update. The finiteness
    # step is isolated here so a clamp can be substituted at this one site.
    _require_finite_tensor_after_arithmetic(
        "policy probability ratio",
        ratio,
        message="policy probability ratio must be finite",
    )
    clipped_ratio = ratio.clamp(1.0 - clip_value, 1.0 + clip_value)
    unclipped = ratio * advantage
    _require_finite_tensor_after_arithmetic("unclipped PPO surrogate", unclipped)
    clipped = clipped_ratio * advantage
    _require_finite_tensor_after_arithmetic("clipped PPO surrogate", clipped)
    surrogate = torch.minimum(unclipped, clipped)
    _require_finite_tensor_after_arithmetic("elementwise PPO surrogate", surrogate)
    return surrogate


def masked_mean(values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Return a differentiable mean over valid timesteps only."""

    _validate_same_shape_2d(values=values)
    require_finite_float_tensor("values", values)
    # Unlike compute_gae, masked_mean's contract requires an explicit mask; a
    # None passed here previously fell through to an all-true mask and
    # silently averaged every timestep (session 2026-07-04).
    if valid_mask is None:
        raise TypeError("valid_mask must be a torch.Tensor")
    mask = _normalise_valid_mask(valid_mask, values)
    if not mask.any().item():
        raise ValueError("valid_mask must contain at least one true timestep")
    result = values[mask].mean()
    return _require_finite_scalar_after_arithmetic("masked_mean", result)


def ppo_clipped_policy_surrogate(
    new_log_probability: torch.Tensor,
    old_log_probability: torch.Tensor,
    advantage: torch.Tensor,
    clip_epsilon: float,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return the masked scalar PPO clipped surrogate."""

    elementwise = ppo_clipped_surrogate(
        new_log_probability,
        old_log_probability,
        advantage,
        clip_epsilon,
    )
    result = masked_mean(elementwise, valid_mask)
    return _require_finite_scalar_after_arithmetic("PPO clipped policy surrogate", result)


def ppo_policy_loss(
    new_log_probability: torch.Tensor,
    old_log_probability: torch.Tensor,
    advantage: torch.Tensor,
    clip_epsilon: float,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return minimized PPO policy loss as negative clipped surrogate."""

    loss = -ppo_clipped_policy_surrogate(
        new_log_probability,
        old_log_probability,
        advantage,
        clip_epsilon,
        valid_mask,
    )
    return _require_finite_scalar_after_arithmetic("PPO policy loss", loss)


def value_loss(
    predicted_value: torch.Tensor,
    target_value: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return masked mean-squared value prediction loss."""

    _validate_same_shape_2d(predicted_value=predicted_value, target_value=target_value)
    require_finite_float_tensor("predicted_value", predicted_value)
    require_finite_float_tensor("target_value", target_value)
    difference = predicted_value - target_value
    _require_finite_tensor_after_arithmetic("value prediction difference", difference)
    squared_error = difference.pow(2)
    _require_finite_tensor_after_arithmetic("value squared_error", squared_error)
    result = masked_mean(squared_error, valid_mask)
    return _require_finite_scalar_after_arithmetic("value loss", result)


def entropy_bonus(entropy: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Return masked mean entropy for a future maximization bonus term."""

    _validate_same_shape_2d(entropy=entropy)
    require_finite_float_tensor("entropy", entropy)
    result = masked_mean(entropy, valid_mask)
    return _require_finite_scalar_after_arithmetic("entropy bonus", result)


def lagrangian_advantage(
    reward_advantage: torch.Tensor,
    cost_advantage: torch.Tensor,
    lagrange_multiplier: torch.Tensor | float,
) -> torch.Tensor:
    """Return ``reward_advantage - lambda * cost_advantage``."""

    _validate_same_shape_2d(
        reward_advantage=reward_advantage,
        cost_advantage=cost_advantage,
    )
    require_finite_float_tensor("reward_advantage", reward_advantage)
    require_finite_float_tensor("cost_advantage", cost_advantage)
    multiplier = _normalise_nonnegative_multiplier(
        lagrange_multiplier,
        reward_advantage,
    )
    penalty = multiplier * cost_advantage
    _require_finite_tensor_after_arithmetic("Lagrangian advantage penalty", penalty)
    output = reward_advantage - penalty
    _require_finite_tensor_after_arithmetic("Lagrangian advantage", output)
    return output


def lagrangian_ppo_policy_loss(
    new_log_probability: torch.Tensor,
    old_log_probability: torch.Tensor,
    reward_advantage: torch.Tensor,
    cost_advantage: torch.Tensor,
    lagrange_multiplier: torch.Tensor | float,
    clip_epsilon: float,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Return minimized PPO loss using the Lagrangian advantage."""

    return ppo_policy_loss(
        new_log_probability,
        old_log_probability,
        lagrangian_advantage(reward_advantage, cost_advantage, lagrange_multiplier),
        clip_epsilon,
        valid_mask,
    )


def lagrangian_policy_objective(
    reward_surrogate: torch.Tensor,
    cost_surrogate: torch.Tensor,
    lagrange_multiplier: torch.Tensor | float,
) -> torch.Tensor:
    """Combine reward and cost terms as reward minus lambda times cost."""

    _validate_same_shape_2d(
        reward_surrogate=reward_surrogate,
        cost_surrogate=cost_surrogate,
    )
    require_finite_float_tensor("reward_surrogate", reward_surrogate)
    require_finite_float_tensor("cost_surrogate", cost_surrogate)
    multiplier = _normalise_nonnegative_multiplier(
        lagrange_multiplier,
        reward_surrogate,
    )
    penalty = multiplier * cost_surrogate
    _require_finite_tensor_after_arithmetic("Lagrangian policy penalty", penalty)
    output = reward_surrogate - penalty
    _require_finite_tensor_after_arithmetic("Lagrangian policy objective", output)
    return output


def _normalise_nonnegative_multiplier(
    lagrange_multiplier: torch.Tensor | float,
    reference: torch.Tensor,
) -> torch.Tensor:
    torch = torch_required("Lagrangian multiplier tensor validation")
    if isinstance(lagrange_multiplier, torch.Tensor):
        if lagrange_multiplier.numel() != 1:
            raise ValueError("lagrange_multiplier must be scalar")
        # Reject meta-device and non-strided (sparse) layouts before any
        # .item() read: on a meta tensor .item() escapes as an undeclared
        # RuntimeError and on a sparse tensor comparisons escape as
        # NotImplementedError (NEW-losses-6; same class of gap as the S-5
        # meta and config.require_finite_float_tensor strided rejections).
        if lagrange_multiplier.device.type == "meta":
            raise ValueError("lagrange_multiplier must not be a meta tensor")
        if lagrange_multiplier.layout != torch.strided:
            raise ValueError("lagrange_multiplier must use a strided tensor layout")
        if lagrange_multiplier.device != reference.device:
            raise ValueError("lagrange_multiplier device must match reference tensor device")
        if lagrange_multiplier.dtype == torch.bool:
            raise TypeError("lagrange_multiplier must be numeric, not boolean")
        # A quantized dtype would bypass the floating-point finiteness paranoia
        # below (NEW-losses-7; mirrors the S2-8 model-side quantized gate).
        if lagrange_multiplier.is_quantized:
            raise TypeError("lagrange_multiplier must not be quantized")
        if torch.is_complex(lagrange_multiplier):
            raise TypeError("lagrange_multiplier must be real numeric, not complex")
        if torch.is_floating_point(lagrange_multiplier) and not torch.isfinite(
            lagrange_multiplier
        ).all().item():
            raise ValueError("lagrange_multiplier must be finite")
        if (lagrange_multiplier < 0).any().item():
            raise ValueError("lagrange_multiplier must be nonnegative")
        # The multiplier is updated by projected dual ascent (Ray et al.
        # 2019), never by backpropagation through the policy loss, so any
        # autograd history on a tensor multiplier is detached here (session
        # 2026-07-04).
        multiplier = lagrange_multiplier.to(dtype=reference.dtype).detach()
    else:
        numeric = finite_numeric_scalar("lagrange_multiplier", lagrange_multiplier)
        if numeric < 0:
            raise ValueError("lagrange_multiplier must be nonnegative")
        multiplier = reference.new_tensor(numeric)
    return multiplier


def _validate_terminal_truncation_input_pairs(boolean_tensors: Mapping[str, object]) -> None:
    terminal_aliases = [name for name in _TERMINAL_ALIASES if name in boolean_tensors]
    truncation_aliases = [name for name in _TRUNCATION_ALIASES if name in boolean_tensors]
    if len(terminal_aliases) > 1:
        raise ValueError("loss input mapping cannot contain both terminal and terminals")
    if len(truncation_aliases) > 1:
        raise ValueError("loss input mapping cannot contain both truncation and truncations")
    if not terminal_aliases or not truncation_aliases:
        return

    terminal_name = terminal_aliases[0]
    truncation_name = truncation_aliases[0]
    terminal = boolean_tensors[terminal_name]
    truncation = boolean_tensors[truncation_name]
    if terminal.shape != truncation.shape:
        raise ValueError(f"{terminal_name} and {truncation_name} shape mismatch")
    if terminal.device != truncation.device:
        raise ValueError(f"{terminal_name} and {truncation_name} device mismatch")
    if (terminal & truncation).any().item():
        # Wording unified with sequence_boundary_mask (NEW-losses-11).
        raise ValueError("terminal and truncation cannot both be true at one step")


def _validate_same_shape_2d(**tensors: torch.Tensor) -> None:
    torch = torch_required("2D tensor shape validation")
    shape = None
    device = None
    float_dtype = None
    for name, value in tensors.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.ndim != 2:
            raise ValueError(f"{name} must have shape [batch, time]")
        # Session 2026-07-04 (S-4 analog): zero-sized batch/time would return
        # silently empty loss results instead of failing fast.
        if value.shape[0] <= 0 or value.shape[1] <= 0:
            raise ValueError(f"{name} batch and time dimensions must be positive")
        if shape is None:
            shape = value.shape
        elif value.shape != shape:
            raise ValueError(f"{name} shape does not match previous tensors")
        if device is None:
            device = value.device
        elif value.device != device:
            raise ValueError(f"{name} device does not match previous tensors")
        # Floating tensors must agree on dtype so mixed-precision inputs are
        # rejected instead of silently promoted and truncated (session
        # 2026-07-04; S-3 analog for the loss surface). Boolean terminal /
        # truncation masks are deliberately exempt.
        if torch.is_floating_point(value):
            if float_dtype is None:
                float_dtype = value.dtype
            elif value.dtype != float_dtype:
                raise ValueError(
                    f"{name} dtype does not match previous floating point tensors"
                )


def _normalise_valid_mask(
    valid_mask: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    torch = torch_required("valid mask tensor validation")
    if valid_mask is None:
        return torch.ones(reference.shape, dtype=torch.bool, device=reference.device)
    if not isinstance(valid_mask, torch.Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    if valid_mask.shape != reference.shape:
        raise ValueError("valid_mask shape mismatch")
    if valid_mask.device != reference.device:
        raise ValueError("valid_mask device must match reference tensor device")
    return _require_bool_tensor("valid_mask", valid_mask)


def _require_bool_tensor(name: str, value: torch.Tensor) -> torch.Tensor:
    torch = torch_required("boolean tensor validation")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must use torch.bool dtype")
    if value.device.type == "meta":
        raise ValueError(f"{name} must not be a meta tensor")
    # Sparse/other non-strided masks are rejected explicitly (S2-14 analog):
    # the valid-mask index read values[mask] in masked_mean (and the mask
    # arithmetic in compute_gae) would otherwise escape as an undeclared
    # NotImplementedError; mirrors buffer._validate_bool_tensor's strided gate.
    if value.layout != torch.strided:
        raise ValueError(f"{name} must use a strided tensor layout")
    return value


def _require_finite_tensor_after_arithmetic(
    name: str,
    value: torch.Tensor,
    *,
    message: str | None = None,
) -> torch.Tensor:
    torch = torch_required("post-arithmetic finite tensor validation")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating point dtype")
    if not torch.isfinite(value).all().item():
        raise ValueError(message or f"{name} must be finite after arithmetic")
    return value


def _require_finite_scalar_after_arithmetic(
    name: str,
    value: torch.Tensor,
) -> torch.Tensor:
    if value.ndim != 0:
        raise ValueError(f"{name} must return a scalar tensor")
    return _require_finite_tensor_after_arithmetic(name, value)


def _validate_bool_mask_2d(name: str, value: torch.Tensor) -> torch.Tensor:
    torch = torch_required("terminal/truncation mask tensor validation")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [batch, time]")
    if value.dtype != torch.bool:
        raise TypeError(f"{name} must use torch.bool dtype")
    # Session 2026-07-04 (S-5 analog): meta-device masks would escape as an
    # undeclared RuntimeError from downstream .item() calls.
    if value.device.type == "meta":
        raise ValueError(f"{name} must not be a meta tensor")
    # Sparse/other non-strided layouts are rejected explicitly (S2-14 analog):
    # boolean-mask ops such as ~terminal, (terminal & truncation).any().item(),
    # and values[mask] / as_strided would otherwise escape as an undeclared
    # NotImplementedError; mirrors buffer._validate_bool_tensor's strided gate.
    if value.layout != torch.strided:
        raise ValueError(f"{name} must use a strided tensor layout")
    return value


def _validate_scalar_loss_value(name: str, value: object) -> None:
    numeric = finite_numeric_scalar(name, value)
    if name in {"hazard_cost", "sensing_cost"} and numeric < 0:
        raise ValueError(f"{name} must be nonnegative")

"""Bounded PPO-Lagrangian update integration for Stage 22 development."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import torch

from raas_marl.mappo_lagrangian.buffer import RolloutBatch
from raas_marl.mappo_lagrangian.config import AlgorithmConfig, LagrangeConfig, LossConfig
from raas_marl.mappo_lagrangian.lagrange import LagrangeMultiplier
from raas_marl.mappo_lagrangian.losses import (
    compute_cost_gae,
    compute_reward_gae,
    entropy_bonus,
    lagrangian_advantage,
    lagrangian_ppo_policy_loss,
    value_loss,
)
from raas_marl.mappo_lagrangian.model import ActorInput, CriticInput, RecurrentMAPPOActorCritic
from raas_marl.mappo_lagrangian._validation import (
    require_positive_int,
    require_positive_number,
)
from raas_marl.mappo_lagrangian._rollout_common import (
    parameter_delta_l1 as _parameter_delta_l1,
)


@dataclass(frozen=True)
class Stage22UpdateConfig:
    """Bounded development PPO-Lagrangian update settings.

    ``losses.hazard_cost_coefficient`` is retained for general/later
    direct-cost-scaling compatibility. In bounded Stage 22 development training,
    hazard penalty strength is controlled by the Lagrange multiplier, not by
    ``losses.hazard_cost_coefficient``.
    """

    algorithm: AlgorithmConfig
    losses: LossConfig
    lagrange: LagrangeConfig
    learning_rate: float = 0.01
    max_update_epochs: int = 2
    max_grad_norm: float | None = 1.0
    normalize_advantages: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, AlgorithmConfig):
            raise TypeError("algorithm must be an AlgorithmConfig")
        if not isinstance(self.losses, LossConfig):
            raise TypeError("losses must be a LossConfig")
        if not isinstance(self.lagrange, LagrangeConfig):
            raise TypeError("lagrange must be a LagrangeConfig")
        require_positive_number(
            "learning_rate", self.learning_rate, error_suffix="must be positive and finite"
        )
        require_positive_int(
            "max_update_epochs", self.max_update_epochs, error_suffix="must be positive"
        )
        if self.max_update_epochs > 4:
            raise ValueError("max_update_epochs must be <= 4 for Stage 22 development")
        if self.max_grad_norm is not None:
            require_positive_number(
                "max_grad_norm", self.max_grad_norm, error_suffix="must be positive and finite"
            )
        if not isinstance(self.normalize_advantages, bool):
            raise TypeError("normalize_advantages must be a bool")


@dataclass(frozen=True)
class Stage22UpdateResult:
    """In-memory update result and finite scalar summary."""

    lagrange_multiplier: LagrangeMultiplier
    summary: dict[str, Any]


def default_stage22_update_config() -> Stage22UpdateConfig:
    """Return CPU-safe development settings with Lagrange-controlled hazard cost."""

    return Stage22UpdateConfig(
        algorithm=AlgorithmConfig(
            discount_factor=0.95,
            gae_lambda=0.9,
            ppo_clip_range=0.2,
        ),
        losses=LossConfig(
            reward_entropy_coefficient=0.0,
            movement_entropy_coefficient=0.01,
            sensing_entropy_coefficient=0.01,
            # NEW-update.py-4: hazard_cost_coefficient is validated by LossConfig
            # but NOT read by the bounded Stage 22 update (hazard penalty strength
            # is controlled by the Lagrange multiplier, not a direct-cost term).
            # Wiring it into a direct-cost term vs removing it (mirroring m-6) is
            # queued for Phase 6; kept for general/later LossConfig compatibility.
            hazard_cost_coefficient=1.0,
            sensing_cost_coefficient=0.5,
            value_loss_coefficient=0.5,
        ),
        lagrange=LagrangeConfig(
            initial_multiplier=0.1,
            learning_rate=0.2,
            hazard_budget=0.1,
        ),
        learning_rate=0.01,
        max_update_epochs=2,
        max_grad_norm=1.0,
        normalize_advantages=False,
    )


def stage22_ppo_lagrangian_update(
    model: RecurrentMAPPOActorCritic,
    rollout: RolloutBatch,
    update_config: Stage22UpdateConfig | None = None,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    lagrange_multiplier: LagrangeMultiplier | None = None,
) -> Stage22UpdateResult:
    """Run bounded in-memory PPO-Lagrangian update steps for development.

    This function intentionally leaves the model in training mode after update execution.
    """

    if not isinstance(model, RecurrentMAPPOActorCritic):
        raise TypeError("model must be a RecurrentMAPPOActorCritic")
    if not isinstance(rollout, RolloutBatch):
        raise TypeError("rollout must be a RolloutBatch")
    if update_config is None:
        settings = default_stage22_update_config()
    elif isinstance(update_config, Stage22UpdateConfig):
        settings = update_config
    else:
        raise TypeError("update_config must be a Stage22UpdateConfig or None")
    rollout.validate(model.config)
    _validate_rollout_for_update(rollout)
    if not rollout.valid_mask.any().item():
        raise ValueError("valid_mask must contain at least one true timestep")
    owned_optimizer = optimizer is None
    if optimizer is not None and not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError("optimizer must be a torch.optim.Optimizer or None")
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    _validate_optimizer_parameter_ownership(model, optimizer)
    if lagrange_multiplier is None:
        multiplier = LagrangeMultiplier(
            settings.lagrange.initial_multiplier,
            settings.lagrange.learning_rate,
        )
    elif isinstance(lagrange_multiplier, LagrangeMultiplier):
        multiplier = lagrange_multiplier
    else:
        raise TypeError("lagrange_multiplier must be a LagrangeMultiplier or None")
    before_parameters = [parameter.detach().clone() for parameter in model.parameters()]
    # D-1 (sensing cost as fixed-weight reward shaping vs a second Lagrangian
    # constraint) is isolated behind ``_development_reward_signal`` so the
    # reward-signal construction is a one-site swap; behavior is unchanged.
    development_reward_signal = _development_reward_signal(rollout, settings.losses)
    raw_reward_advantage = compute_reward_gae(
        development_reward_signal,
        rollout.reward_value,
        rollout.next_reward_value,
        rollout.terminal,
        rollout.truncation,
        rollout.valid_mask,
        settings.algorithm.discount_factor,
        settings.algorithm.gae_lambda,
    ).detach()
    raw_hazard_cost_advantage = compute_cost_gae(
        rollout.hazard_cost,
        rollout.hazard_cost_value,
        rollout.next_hazard_cost_value,
        rollout.terminal,
        rollout.truncation,
        rollout.valid_mask,
        settings.algorithm.discount_factor,
        settings.algorithm.gae_lambda,
    ).detach()
    reward_return_target = (raw_reward_advantage + rollout.reward_value).detach()
    hazard_cost_return_target = (
        raw_hazard_cost_advantage + rollout.hazard_cost_value
    ).detach()
    policy_reward_advantage = raw_reward_advantage
    policy_hazard_cost_advantage = raw_hazard_cost_advantage
    if settings.normalize_advantages:
        policy_reward_advantage = _normalise_advantage(
            policy_reward_advantage,
            rollout.valid_mask,
        ).detach()
        policy_hazard_cost_advantage = _normalise_advantage(
            policy_hazard_cost_advantage,
            rollout.valid_mask,
        ).detach()
    old_joint_log_probability = rollout.old_joint_log_probability.detach()
    final_scalars: dict[str, float] = {}
    model.train()
    for _epoch in range(settings.max_update_epochs):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            ActorInput(rollout.actor_observation, rollout.revealed_information),
            CriticInput(rollout.central_state),
            sensing_action=rollout.sensing_action,
            movement_action=rollout.movement_action,
            sample_actions=False,
        )
        policy = outputs["policy"]
        new_joint_log_probability = (
            policy.sensing_log_probability + policy.movement_log_probability
        )
        policy_loss = lagrangian_ppo_policy_loss(
            new_joint_log_probability,
            old_joint_log_probability,
            policy_reward_advantage,
            policy_hazard_cost_advantage,
            multiplier.value,
            settings.algorithm.ppo_clip_range,
            rollout.valid_mask,
        )
        reward_critic_loss = value_loss(
            outputs["reward_value"],
            reward_return_target,
            rollout.valid_mask,
        )
        hazard_critic_loss = value_loss(
            outputs["hazard_cost_value"],
            hazard_cost_return_target,
            rollout.valid_mask,
        )
        sensing_entropy = entropy_bonus(policy.sensing_entropy, rollout.valid_mask)
        movement_entropy = entropy_bonus(policy.movement_entropy, rollout.valid_mask)
        total_loss = (
            policy_loss
            + settings.losses.value_loss_coefficient
            * (reward_critic_loss + hazard_critic_loss)
            - settings.losses.sensing_entropy_coefficient * sensing_entropy
            - settings.losses.movement_entropy_coefficient * movement_entropy
            # Combined-entropy term for LossConfig.reward_entropy_coefficient
            # (issue m-6 resolution): the bounded development default of 0.0
            # keeps Stage 22 numerics unchanged.
            - settings.losses.reward_entropy_coefficient
            * (sensing_entropy + movement_entropy)
        )
        _require_finite_scalar("total_loss", total_loss)
        total_loss.backward()
        if settings.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), settings.max_grad_norm)
        optimizer.step()
        final_scalars = {
            "policy_loss": float(policy_loss.detach().item()),
            "reward_value_loss": float(reward_critic_loss.detach().item()),
            "hazard_cost_value_loss": float(hazard_critic_loss.detach().item()),
            "sensing_entropy": float(sensing_entropy.detach().item()),
            "movement_entropy": float(movement_entropy.detach().item()),
            "total_loss": float(total_loss.detach().item()),
        }
    # D-2: the constraint estimate J_C is the PER-STEP MEAN hazard cost of this
    # rollout (see ``cost_estimate``), so lagrange.hazard_budget is on a per-step
    # scale rather than the paper's episodic scale; the per-step-mean vs
    # episodic-sum swap is queued for the reviewed final-training update / Phase 6.
    # D-3: lambda is held FIXED across all PPO epochs (read once at the loop as
    # ``multiplier.value``) and updated by exactly this SINGLE post-hoc dual step
    # from the pre-update rollout cost; the projection floor is 0 so lambda can
    # collapse to zero, and there is no lambda lower bound and no hazard-budget
    # schedule. This bounded-development cadence is deliberate. The swap points
    # for the reviewed final-training update are gated by
    # ``stage24_collector.Stage24AUpdateConfig`` (which rejects
    # ``nonzero_lagrange_lower_bound_added`` / ``hazard_budget_schedule_added``
    # today) plus a per-epoch-dual-step flag; these decisions are queued for
    # Phase 6 (recorded blocker: lagrange_multiplier_can_collapse_to_zero).
    observed_hazard_cost = cost_estimate(rollout)
    next_multiplier = multiplier.update(
        observed_cost=observed_hazard_cost,
        cost_budget=settings.lagrange.hazard_budget,
    )
    # NEW-update.py-5: model.parameters() yields a stable order across the two
    # snapshots (before_parameters above and here), so the strict=True zip in
    # parameter_delta_l1 pairs matching tensors. NEW-update.py-1 / D-7: the L1
    # delta (with its finiteness guard) is the single shared
    # _rollout_common.parameter_delta_l1, imported here as _parameter_delta_l1
    # and reused by stage24_collector.
    parameter_delta_l1 = _parameter_delta_l1(before_parameters, list(model.parameters()))
    if parameter_delta_l1 <= 0.0:
        raise ValueError("Stage 22 update did not change model parameters")
    summary = {
        "development_only": True,
        "optimizer_state_saved": False,
        "owned_optimizer": owned_optimizer,
        "old_joint_log_probability_source": "rollout.old_joint_log_probability",
        "old_joint_log_probability_is_sum": True,
        "new_joint_log_probability_is_sum": True,
        "resampled_actions_during_update": False,
        "development_reward_signal": "task_reward - sensing_cost_coefficient * sensing_cost",
        "lagrange_cost_signal": "hazard_cost",
        "lagrangian_sign_convention": "reward_advantage - lambda_value * hazard_cost_advantage",
        "hazard_cost_penalty_source": "lagrange_multiplier",
        "hazard_cost_coefficient_stage22_usage": (
            "retained_for_general_loss_config_compatibility_not_used_by_bounded_stage22_lagrange_penalty"
        ),
        "model_mode_after_update": "train",
        "model_mode_policy": "stage22_update_leaves_model_in_train_mode",
        "observed_hazard_cost": observed_hazard_cost,
        "hazard_budget": float(settings.lagrange.hazard_budget),
        "lagrange_multiplier_before": float(multiplier.value),
        "lagrange_multiplier_after": float(next_multiplier.value),
        "parameter_delta_l1": parameter_delta_l1,
        "raw_reward_advantage_mean": _masked_mean_float(raw_reward_advantage, rollout.valid_mask),
        "raw_hazard_cost_advantage_mean": _masked_mean_float(
            raw_hazard_cost_advantage, rollout.valid_mask
        ),
        "policy_reward_advantage_mean": _masked_mean_float(
            policy_reward_advantage, rollout.valid_mask
        ),
        "policy_hazard_cost_advantage_mean": _masked_mean_float(
            policy_hazard_cost_advantage, rollout.valid_mask
        ),
        # D-12 (DECIDED 2026-07-04, DR-D12): the four LEGACY keys
        # reward_advantage_mean / hazard_cost_advantage_mean (byte-identical
        # duplicates of the policy_* keys) and their *_kind tags were REMOVED
        # here per the countersigned Decision Record. The re-grep obligation of
        # the DR-D12 amendment was satisfied at implementation time: zero
        # readers of the legacy keys anywhere in src/ (including
        # final_training/, whose baseline_driver reads only the retained raw_*
        # keys), tests/test_clean.py, docs/ (outside the decision/evidence
        # records), or scripts. Rollback target (documented one-site hook):
        # re-add the two _masked_mean_float(policy_*_advantage, valid_mask)
        # entries plus the two "policy_after_optional_normalization" tags
        # directly above reward_return_target_mean.
        "reward_return_target_mean": _masked_mean_float(reward_return_target, rollout.valid_mask),
        "hazard_cost_return_target_mean": _masked_mean_float(
            hazard_cost_return_target, rollout.valid_mask
        ),
        "advantages_normalized": settings.normalize_advantages,
        "value_targets_use_raw_advantages": True,
        **final_scalars,
    }
    _validate_summary(summary)
    return Stage22UpdateResult(lagrange_multiplier=next_multiplier, summary=summary)


def hazard_advantage_penalizes_policy_objective() -> bool:
    """Return true for the documented Lagrangian sign convention.

    Invariant self-check for the ``A_r - lambda * A_c`` sign convention: higher
    hazard-cost advantage must lower the Lagrangian objective. D-14: this is the
    single site that exercises ``lagrangian_advantage``; the actual policy
    gradient consumes it inside ``lagrangian_ppo_policy_loss`` in the update loop.
    A ``1 / (1 + lambda)`` objective normalization (safety-starter-agents style)
    would be inserted inside ``lagrangian_advantage`` /
    ``lagrangian_policy_objective`` in losses.py at one place; that adopt-or-reject
    decision is queued for Phase 6 (interacts with D-3 lambda dynamics). The
    current unnormalized form matches Ray et al. (2019), which governs.
    """

    reward_advantage = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    low_cost = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    high_cost = torch.tensor([[1.0, 1.0]], dtype=torch.float32)
    low = lagrangian_advantage(reward_advantage, low_cost, 0.5)
    high = lagrangian_advantage(reward_advantage, high_cost, 0.5)
    return bool((high < low).all().item())


# NEW-update.py-3: exercise the Lagrangian sign-convention invariant at import so
# the self-check predicate has a runtime call site rather than being dead code.
if not hazard_advantage_penalizes_policy_objective():  # pragma: no cover
    raise AssertionError(
        "Lagrangian sign convention violated: higher hazard-cost advantage must "
        "lower the policy objective (A_r - lambda * A_c)"
    )


def _validate_rollout_for_update(rollout: RolloutBatch) -> None:
    if rollout.old_sensing_log_probability.requires_grad:
        raise ValueError("old_sensing_log_probability must be rollout data")
    if rollout.old_movement_log_probability.requires_grad:
        raise ValueError("old_movement_log_probability must be rollout data")
    for name in (
        "task_reward",
        "hazard_cost",
        "sensing_cost",
        "reward_value",
        "hazard_cost_value",
        "next_reward_value",
        "next_hazard_cost_value",
    ):
        value = getattr(rollout, name)
        if value.requires_grad:
            raise ValueError(f"{name} must be rollout data, not a differentiable output")
        if not torch.isfinite(value).all().item():
            raise ValueError(f"{name} must contain only finite values")
    for name in ("hazard_cost", "sensing_cost"):
        value = getattr(rollout, name)
        if value.numel() and (value < 0).any().item():
            raise ValueError(f"{name} must be nonnegative")


def _normalise_advantage(
    values: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    std_guard_threshold: float = 0.0,
) -> torch.Tensor:
    if not valid_mask.any().item():
        raise ValueError("valid_mask must contain at least one true timestep")
    valid_values = values[valid_mask]
    mean = valid_values.mean()
    # Population standard deviation is intentionally used for the current valid
    # minibatch. This matches ddof=0 PPO/MAPPO advantage normalization practice
    # and avoids Bessel-correction instability for tiny Stage 22 batches.
    std = valid_values.std(unbiased=False)
    # DR-C4 std-guard (amends DR-D4/DR-D14; OFF by default). With the default
    # threshold 0.0 this is the exact historical ``std <= 0.0`` branch, so the
    # bounded Stage 22 dev update and every Stage-25 OFF-default path stay
    # byte-identical (regression pin). A positive threshold (passed only by the
    # Stage 25 DR-C4-active path) additionally routes a small-but-nonzero std
    # through the mean-centered ``values - mean`` branch, which keeps any
    # residual advantage variance as a signed gradient instead of dividing by a
    # near-zero std -- the ED-C4 Link-5 / PC-1 zero-variance-freeze guard.
    if std.item() <= std_guard_threshold:
        return values - mean
    return (values - mean) / std.clamp_min(1e-8)


def _validate_optimizer_parameter_ownership(
    model: RecurrentMAPPOActorCritic,
    optimizer: torch.optim.Optimizer,
) -> None:
    model_parameters = tuple(model.parameters())
    model_parameter_ids = {id(parameter) for parameter in model_parameters}

    optimizer_parameters: list[torch.nn.Parameter] = []
    for group in optimizer.param_groups:
        params = group.get("params")
        if params is None:
            raise ValueError("optimizer param group is missing params")
        optimizer_parameters.extend(list(params))

    if not optimizer_parameters:
        raise ValueError("optimizer must contain model parameters")

    optimizer_parameter_ids = [id(parameter) for parameter in optimizer_parameters]
    if len(optimizer_parameter_ids) != len(set(optimizer_parameter_ids)):
        raise ValueError("optimizer parameter groups must not contain duplicate parameters")
    if set(optimizer_parameter_ids) != model_parameter_ids:
        raise ValueError("optimizer parameters must exactly match model parameters")


def _validate_summary(summary: dict[str, Any]) -> None:
    # NEW-update.py-2: every numeric summary field is a Python float by
    # construction (advantage/return means, losses, multiplier values,
    # parameter_delta_l1); the remaining values are bools and provenance strings.
    # bool is excluded from the numeric-finiteness check because it is not a
    # meaningful magnitude here. Non-bool ints are also finiteness-checked for
    # forward safety should an integer summary field ever be added.
    for key, value in summary.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (float, int)) and not math.isfinite(value):
            raise ValueError(f"{key} must be finite")


def _require_finite_scalar(name: str, value: torch.Tensor) -> None:
    if value.ndim != 0:
        raise ValueError(f"{name} must be scalar")
    if not torch.isfinite(value).item():
        raise ValueError(f"{name} must be finite")


def _masked_mean_float(values: torch.Tensor, valid_mask: torch.Tensor) -> float:
    """Return the mean of ``values`` over the valid timesteps as a float.

    Single-sources the ``float(tensor[valid_mask].mean().item())`` pattern that
    the update summary and the dual-step observed-cost computation both reuse
    (issue NEW-update.py-8), so the masked-mean convention lives at one site.
    """

    return float(values[valid_mask].mean().item())


def _development_reward_signal(
    rollout: RolloutBatch,
    losses: LossConfig,
) -> torch.Tensor:
    """Build the reward signal that GAE is computed on for the bounded update.

    Deliberate Stage 22 design, registered as decision-queue item D-1: the sensing
    cost is folded into the reward signal with a fixed weight
    (``sensing_cost_coefficient``) rather than handled as a second Lagrangian
    constraint; only the hazard cost is constraint-handled. Extracting the
    reward-signal construction behind this helper is the D-1 code hook: a
    reviewed final-training update can wire a sensing-cost *constraint* by
    replacing/branching here without restructuring the update loop. See
    ``_sensing_cost_as_second_constraint`` for the (currently NotImplemented)
    constraint branch stub.
    """

    return rollout.task_reward - losses.sensing_cost_coefficient * rollout.sensing_cost


def _sensing_cost_as_second_constraint(
    rollout: RolloutBatch,
    losses: LossConfig,
) -> torch.Tensor:  # pragma: no cover - non-behavioral D-1 swap-point stub
    """Placeholder for treating sensing cost as a second Lagrangian constraint.

    D-1 code hook (decision queued for the reviewed final-training update /
    Phase 6): if the "costly information acquisition" claim requires sensing cost
    to be budgeted rather than fixed-weight reward shaping, a second
    ``LagrangeMultiplier`` plus a sensing-cost GAE path would be wired here and in
    the update loop. Intentionally not implemented in bounded Stage 22.
    """

    raise NotImplementedError(
        "sensing cost as a second Lagrangian constraint is not implemented in "
        "bounded Stage 22 development (decision-queue item D-1)"
    )


def cost_estimate(rollout: RolloutBatch) -> float:
    """Return the constraint estimate J_C fed to the Lagrange dual step.

    Deliberate Stage 22 deviation from Ray et al. (2019), registered as
    decision-queue item D-2: this returns the PER-STEP MEAN hazard cost over the
    valid timesteps, so ``lagrange.hazard_budget`` is interpreted on a per-step
    scale rather than the paper's episodic scale. Isolating the estimate behind
    this named function is the D-2 code hook: per-step-mean vs episodic-sum J_C
    can be swapped at this one site (queued for the reviewed final-training update
    / Phase 6; interacts with S-11's ``observed_cost >= 0`` domain in lagrange.py).
    """

    return _masked_mean_float(rollout.hazard_cost, rollout.valid_mask)



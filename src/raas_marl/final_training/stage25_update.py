"""Stage 25 reviewed final-training PPO-Lagrangian update path (torch-eager).

This module implements the reviewed final-training update decided by the
COUNTERSIGNED Decision Records DR-D4 (full-batch, KL-guarded 10-epoch
PPO-Lagrangian), DR-D2 (episodic mean team hazard cost J_C-hat, d_ep = 0.5),
DR-D3 (PI dual controller, one dual step per round BEFORE the policy epochs,
projection floor 0), DR-D14 (unnormalized ``A_r - lambda * A_c`` mixing with
SEPARATE pre-mix whole-batch standardization), DR-D9 (fail-fast ratio abort
untouched underneath + Stage-25-only k3 KL early stopping), DR-D1 (sensing
cost stays fixed-weight reward shaping via the dev update's
``_development_reward_signal`` hook), DR-D10 (feedforward critics unchanged;
no buffer/collector schema changes), and DR-S-11 (the ``observed_cost >= 0``
gate is kept; the signed error is formed at the controller site).

The bounded development update (``stage22_ppo_lagrangian_update``) is NOT
modified and remains the only update path for stages <= 23-C; this module is
additive. Development-update private helpers are imported deliberately, never
copied (the D-7 lesson): ``_development_reward_signal`` (the DR-D1 hook),
``_normalise_advantage`` (the exact standardization semantics DR-D4/DR-D14
bind to), ``_validate_rollout_for_update`` / ``_validate_optimizer_parameter_
ownership`` / ``_validate_summary`` / ``_require_finite_scalar`` (validation
surfaces), ``_masked_mean_float`` (the masked-mean convention), and the public
``cost_estimate`` (the dev per-step D-2 estimator, used only by the DR-D4 T0
dev-regression-bridge configuration).

This module is development infrastructure for the Stage 25 final-training
stage authorized by the Segment 1 handoff. It does not execute the project's
locked final-assessment protocol, creates no evidence artifacts for the
research claim, produces no paper-oriented output, runs no baseline comparison
as evidence, no ablation, and no statistical test, and makes no
Bayesian-belief or formal-VOI claim.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any

import torch

from raas_marl.mappo_lagrangian._rollout_common import (
    parameter_delta_l1 as _parameter_delta_l1,
)
from raas_marl.mappo_lagrangian._validation import (
    finite_numeric_scalar,
    require_nonnegative_int,
    require_nonnegative_number,
    require_positive_int,
    require_positive_number,
)
from raas_marl.mappo_lagrangian.buffer import RolloutBatch
from raas_marl.mappo_lagrangian.config import AlgorithmConfig, LossConfig
from raas_marl.mappo_lagrangian.lagrange import LagrangeMultiplier
from raas_marl.mappo_lagrangian.losses import (
    compute_cost_gae,
    compute_reward_gae,
    entropy_bonus,
    lagrangian_ppo_policy_loss,
    masked_mean,
    value_loss,
)
from raas_marl.mappo_lagrangian.model import (
    ActorInput,
    CriticInput,
    RecurrentMAPPOActorCritic,
)

# Private dev-update helpers imported deliberately (D-7 lesson: shared helper
# bodies are imported, never copied). See the module docstring for each name's
# role. ``update.py`` itself is not modified by Stage 25.
from raas_marl.mappo_lagrangian.update import (
    _development_reward_signal,
    _masked_mean_float,
    _normalise_advantage,
    _require_finite_scalar,
    _validate_optimizer_parameter_ownership,
    _validate_rollout_for_update,
    _validate_summary,
    cost_estimate,
)

__all__ = [
    "EntropySustainController",
    "EntropySustainStep",
    "PIController",
    "PIControllerStep",
    "STAGE25_EPOCH_CAP",
    "Stage25UpdateConfig",
    "Stage25UpdateResult",
    "entropy_sustain_target",
    "episodic_team_hazard_cost",
    "potential_shaping_term",
    "stage25_ppo_lagrangian_update",
]


# DR-D4: the Stage 25 epoch cap is 10 (KL-guarded). Changing this cap
# re-triggers DR-D9 review (the staleness envelope its abort analysis assumes).
STAGE25_EPOCH_CAP = 10

# DR-C4 / AM-3: the constraint-warmup window is capped so it always ends well
# before the 4000-round budget, keeping the tail fully constrained. The active
# envelope is W in [50, 500]; 0 is OFF (exact DR-D3 behavior). Enforced as a
# numeric invariant in the config below (AM-3).
_CONSTRAINT_WARMUP_MAX = 500

_DUAL_STEP_TIMINGS = ("pre_epochs", "post_epochs")
_COST_ESTIMATORS = ("episodic_team_mean", "per_step_mean")


# ---------------------------------------------------------------------------
# DR-C4 potential-based task-progress shaping (Fix 1) -- update side
# ---------------------------------------------------------------------------


def potential_shaping_term(
    potential: torch.Tensor,
    potential_next: torch.Tensor,
    discount_factor: float,
) -> torch.Tensor:
    """Return the DR-C4 potential-based shaping ``F = gamma*Phi(s') - Phi(s)``.

    Single source of the Ng-Harada-Russell (1999) potential-difference form, so
    the reviewed update, the driver's explained-variance diagnostic, and the T0
    telescoping-invariance gate all consume ONE definition. ``discount_factor``
    is the update's own GAE gamma, so the shaping telescopes consistently with
    the reward GAE: over an episode of length L,
    ``sum_t gamma^t F_t = gamma^L * Phi(s_L) - Phi(s_0)`` -- a per-trajectory
    constant, hence policy-invariant. NEVER added to hazard_cost.
    """

    if not isinstance(potential, torch.Tensor) or not isinstance(
        potential_next, torch.Tensor
    ):
        raise TypeError("potential and potential_next must be torch.Tensors")
    if potential.shape != potential_next.shape:
        raise ValueError("potential and potential_next must share a shape")
    return discount_factor * potential_next - potential


def _entropy_floor_gate(entropy: torch.Tensor, floor: float | None) -> float:
    """Return the DR-C4 per-factor entropy-floor gate multiplier (0.0 or 1.0).

    ``floor is None`` (OFF) => always 1.0 => the entropy bonus is applied exactly
    as the fixed-0.01 DR-D4 path, so the OFF path is byte-identical. A positive
    floor applies the entropy bonus ONLY while the head's current entropy is
    below the floor, so exploration is held up near the floor without inflating
    an already-diffuse head (ED-C4 Fix 4 / Link 4).
    """

    if floor is None:
        return 1.0
    return 1.0 if float(entropy.detach().item()) < floor else 0.0


# ---------------------------------------------------------------------------
# PI dual controller (DR-D3)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PIControllerStep:
    """One PI dual step: the successor controller plus the applied multiplier.

    ``lambda_applied`` is held fixed across all policy epochs of the round
    (DR-D3 cadence). ``error`` is the signed constraint error
    ``e_k = J_C-hat_k - d_ep`` formed AT the controller site (DR-S-11: the
    controller input itself stays nonnegative). ``integral`` is the projected
    post-step integral state ``I_k``.
    """

    controller: "PIController"
    lambda_applied: float
    error: float
    integral: float
    observed_episodic_cost: float
    # DR-H3: ``lambda_pre_clamp`` is what lambda would have been WITHOUT the
    # lever-G ceiling (the AM-H9 "pinned at the ceiling" curve read);
    # ``ceiling_engaged`` marks rounds where the ceiling actually bound;
    # ``floor_armed`` echoes the driver's arm state for this step. Defaults
    # keep every pre-DR-H3 construction site valid.
    lambda_pre_clamp: float | None = None
    ceiling_engaged: bool = False
    floor_armed: bool = False


@dataclass(frozen=True, kw_only=True)
class PIController:
    """Projected PI-controlled Lagrange multiplier state (DR-D3, D3-C).

    Per update round k with fresh-batch estimate ``J_C-hat_k`` (DR-D2) and
    budget ``d_ep``:

    - error:      ``e_k = J_C-hat_k - d_ep``
    - integral:   ``I_k = max(0, I_{k-1} + K_I * e_k)`` (``I_0 = 0.1`` warm
      start retained from the current init)
    - applied:    ``lambda_k = max(0, K_P * e_k + I_k)``

    The integral recursion is routed through the existing gated primitive
    ``LagrangeMultiplier.update`` (DR-D3 binding scope: it implements exactly
    the ``I_k`` recursion; DR-S-11: the ``observed_cost >= 0`` tripwire keeps
    covering the new Stage 25 dual machinery). ``K_P = 0`` with ``K_I = eta``
    therefore reproduces the current projected integral-only rule exactly
    (DR-D3 T0 criterion (a) — the equivalence pin).

    Defaults are the DR-D3 gains ``K_P = 0.25`` / ``K_I = 0.05`` (K_D = 0,
    derivative omitted per Stooke "used sparingly"), the DR-D3 warm start
    ``I_0 = 0.1``, and the DR-D2 budget ``d_ep = 0.5``. Projection floor is 0;
    there is no positive lower bound and no budget schedule (DR-D3).

    The sanitized base floats are stored (not the caller's objects) so a
    hostile int/float subclass cannot hijack the dual-step arithmetic through
    overridden operators (the S2-6 ``LagrangeMultiplier`` pattern).
    Construction is keyword-only (``kw_only=True``) so the four same-typed
    floats can never be transposed positionally (the S-10 swappable-argument
    lesson, applied to the constructor as well as ``update``).
    """

    proportional_gain: float = 0.25  # DR-D3: K_P
    integral_gain: float = 0.05  # DR-D3: K_I
    integral: float = 0.1  # DR-D3: I_0 warm start
    budget: float = 0.5  # DR-D2: d_ep
    # DR-D3 AMENDMENT (SHIFT 16): anti-windup integral CEILING. The integral
    # already carries a floor (the max(0, .) projection); this is its symmetric
    # partner (classical integrator clamping / conditional integration, the
    # survey-standard first anti-windup choice — Astrom-Hagglund). When set, the
    # post-step integral is bounded to [0, integral_cap], converting the T1-D8
    # s118 monotonic windup runaway (integral 39.81, unrecoverable in budget)
    # into a bounded, auto-recoverable excursion. ``None`` (OFF) => the clamp
    # branch is skipped (zero float ops) => lambda_applied bit-identical to the
    # current DR-D3 recursion (the DR-D3 T0(a) equivalence pin + all OFF-default
    # byte-repro hold). DR-D3-amendment.md; pinned run value 20.0 (AM-1).
    integral_cap: float | None = None  # DR-D3-amendment
    # DR-RELIABILITY (SHIFT 19) lever F: positive lambda SUSTAIN FLOOR on the
    # APPLIED multiplier. The T1-D9 verified-from-disk limit cycle showed the
    # relapse-to-blind leg is INITIATED by lambda collapsing on momentary
    # constraint satisfaction (all 6 s120 Mode-C phases >=100 rounds died by
    # integral discharge; sustained recoveries required lambda >= 2.55, median
    # 3.12). A floor inside the measured sustain band keeps sensing's
    # -lambda*A_c value alive through satisfaction, removing the discharge leg
    # (deliberately trading exact KKT complementary slackness for dynamic
    # stability -- a training-dynamics device on the SAME single hazard
    # multiplier). 0.0 (OFF) => ``max(0.0, .)`` semantics bit-identical to the
    # DR-D3 recursion. Applied ONLY at the controller site, so the DR-C4
    # warmup (which bypasses the controller and holds lambda = 0) is
    # unaffected. Pinned run value 3.0, envelope [2.5, 5.0]
    # (DR-reliability.md). Reopens DR-D3's floor rejection on that new
    # falsifying evidence.
    lambda_floor: float = 0.0  # DR-RELIABILITY lever F
    # DR-H3 (SHIFT 21) lever G: hard CEILING on the APPLIED multiplier. T1-D10
    # measured ejection into the avoid-all basin only at lambda >= 6.58 (onsets
    # 6.58/10.37/13.23, plastic policies), while Mode-C re-entry/sustain needs
    # only lambda ~2.5-3 (auditor-verified corridor). The ceiling bounds
    # capture-phase pressure strictly inside that corridor. Honest billing per
    # the countersign: this is the RCPO bounded-multiplier construction
    # (Tessler et al. 2019 — "lambda_max defines the maximal regularization
    # allowed") with the bound deliberately BELOW observed effective dual
    # levels, so NO constraint-satisfaction guarantee is claimed for the capped
    # controller; enforcement lives in the external grade (BAR-C4). The
    # integral (cap 20, unchanged) may wind ABOVE the ceiling — deliberate:
    # the windup-band discharge is the post-entry high-lambda dwell the DR-H3
    # arm needs. None (OFF) => the clamp branch is skipped => lambda_applied
    # bit-identical to the DR-RELIABILITY recursion. Pinned run value 5.0,
    # envelope [4.0, 6.0] (DR-H3.md).
    lambda_ceiling: float | None = None  # DR-H3 lever G
    # DR-H3 (SHIFT 21) lever H: the ARMED conditional floor VALUE. The floor is
    # applied only when the caller passes ``floor_armed=True`` to ``update()``
    # — the stateful arming decision (trailing behavioral window + latch +
    # de-arm hatch) lives in the DRIVER, which owns the round aggregates and
    # checkpoint persistence; this controller stays pure math. Mutually
    # exclusive with the unconditional ``lambda_floor`` (the T1-D10-falsified
    # form): the coupling is rejected below. None (OFF) => byte-identical.
    # Pinned run value 3.0, envelope [2.5, 4.5], strictly below the ceiling
    # (DR-H3.md; arm spec W50 / success>=0.8 / hazard<=0.35 / sensing>=0.05).
    armed_lambda_floor: float | None = None  # DR-H3 lever H

    def __post_init__(self) -> None:
        normalized_kp = require_nonnegative_number(
            "proportional_gain", self.proportional_gain
        )
        normalized_ki = require_positive_number("integral_gain", self.integral_gain)
        normalized_integral = require_nonnegative_number("integral", self.integral)
        normalized_budget = require_nonnegative_number("budget", self.budget)
        object.__setattr__(self, "proportional_gain", normalized_kp)
        object.__setattr__(self, "integral_gain", normalized_ki)
        object.__setattr__(self, "integral", normalized_integral)
        object.__setattr__(self, "budget", normalized_budget)
        # Sanitize-store the anti-windup ceiling (S2-6 pattern): None stays None
        # (OFF); a set cap must be a positive finite number. Storing the base
        # float blocks a hostile int/float subclass from hijacking the clamp
        # comparison through an overridden operator.
        if self.integral_cap is None:
            object.__setattr__(self, "integral_cap", None)
        else:
            object.__setattr__(
                self,
                "integral_cap",
                require_positive_number("integral_cap", self.integral_cap),
            )
        # DR-RELIABILITY lever F: sanitize-store the sustain floor (S2-6). The
        # 0.0 OFF default must validate, so this is require_nonnegative_number
        # (NOT require_positive_number).
        object.__setattr__(
            self,
            "lambda_floor",
            require_nonnegative_number("lambda_floor", self.lambda_floor),
        )
        # DR-H3 lever G: sanitize-store the applied-multiplier ceiling (S2-6).
        if self.lambda_ceiling is None:
            object.__setattr__(self, "lambda_ceiling", None)
        else:
            object.__setattr__(
                self,
                "lambda_ceiling",
                require_positive_number("lambda_ceiling", self.lambda_ceiling),
            )
        # DR-H3 lever H: sanitize-store the armed floor value (S2-6).
        if self.armed_lambda_floor is None:
            object.__setattr__(self, "armed_lambda_floor", None)
        else:
            object.__setattr__(
                self,
                "armed_lambda_floor",
                require_positive_number(
                    "armed_lambda_floor", self.armed_lambda_floor
                ),
            )
        # DR-H3 coupling rules (rejected loudly, never silently reconciled):
        # (1) the armed conditional floor REPLACES the unconditional floor —
        # running both would silently re-create the T1-D10 falsified form
        # underneath the arm; (2) any floor must sit strictly below the
        # ceiling, else the corridor is empty by construction.
        if self.armed_lambda_floor is not None and self.lambda_floor > 0.0:
            raise ValueError(
                "armed_lambda_floor is mutually exclusive with a positive "
                "unconditional lambda_floor (DR-H3: the unconditional form is "
                "retired; running both would re-create the T1-D10 lock)"
            )
        if self.lambda_ceiling is not None:
            if self.lambda_floor > 0.0 and self.lambda_floor >= self.lambda_ceiling:
                raise ValueError(
                    "lambda_floor must be strictly below lambda_ceiling "
                    "(DR-H3: an empty corridor is a config error)"
                )
            if (
                self.armed_lambda_floor is not None
                and self.armed_lambda_floor >= self.lambda_ceiling
            ):
                raise ValueError(
                    "armed_lambda_floor must be strictly below lambda_ceiling "
                    "(DR-H3: an empty corridor is a config error)"
                )

    def update(
        self, *, observed_episodic_cost: float, floor_armed: bool = False
    ) -> PIControllerStep:
        """Run one projected PI dual step; return the successor + lambda_applied.

        ``observed_episodic_cost`` is keyword-only (the S-10 lesson: swappable
        same-typed floats must not be positional) and must be nonnegative —
        the DR-S-11 gate, applied both here (via
        ``require_nonnegative_number``) and inside the reused
        ``LagrangeMultiplier.update`` primitive. The signed quantity is the
        error ``e_k``, formed here at the controller site exactly as in every
        reference (Ray Eq 4 / safety-starter / RCPO dataflow).

        ``floor_armed`` (DR-H3 lever H) selects the armed conditional floor for
        THIS step: the arming decision itself (trailing behavioral window +
        one-way-with-hatch latch) is owned by the driver. ``True`` requires a
        configured ``armed_lambda_floor`` — a silently ignored arm signal would
        be the exact silent-disable failure mode the round_index guards close.
        """

        if type(floor_armed) is not bool:
            raise TypeError("floor_armed must be a bool")
        if floor_armed and self.armed_lambda_floor is None:
            raise ValueError(
                "floor_armed=True requires a configured armed_lambda_floor "
                "(DR-H3 lever H); an ignored arm signal would silently disable "
                "the lever"
            )
        observed = require_nonnegative_number(
            "observed_episodic_cost", observed_episodic_cost
        )
        # DR-D3 / DR-S-11: the integral recursion reuses the existing gated
        # projected-update primitive so the observed_cost >= 0 tripwire keeps
        # covering the Stage 25 dual machinery.
        stepped = LagrangeMultiplier(
            value=self.integral, learning_rate=self.integral_gain
        ).update(observed_cost=observed, cost_budget=self.budget)
        new_integral = stepped.value
        # DR-D3 AMENDMENT (SHIFT 16): anti-windup CEILING. Bound the post-step
        # integral to [0, integral_cap] (the floor is the max(0, .) projection
        # inside the primitive above). OFF (None) => branch skipped => bit-
        # identical to the current recursion. The clamped value is re-checked
        # finite for parity with the A-2 pre-projection discipline.
        if self.integral_cap is not None and new_integral > self.integral_cap:
            new_integral = finite_numeric_scalar(
                "capped dual integral", self.integral_cap
            )
        error = finite_numeric_scalar("dual error", observed - self.budget)
        # A-2 (S-9 parity): check the pre-projection candidate finite BEFORE
        # the max(0, .) projection. Otherwise a negative overflow of
        # K_P * e + I would be silently projected to 0.0 and pass a
        # post-projection finiteness check — the exact defect class fixed in
        # lagrange.py, where the pre-projection step value is checked first.
        lambda_candidate = finite_numeric_scalar(
            "lambda_applied candidate", self.proportional_gain * error + new_integral
        )
        # DR-RELIABILITY lever F / DR-H3 lever H: the projection floor is the
        # unconditional sustain floor OR — when the driver's arm is latched —
        # the armed conditional floor (the two are mutually exclusive by
        # construction, rejected in __post_init__). With both levers OFF this
        # is ``max(0.0, .)``, bit-identical to the DR-D3 rule.
        floor_term = (
            self.armed_lambda_floor
            if (floor_armed and self.armed_lambda_floor is not None)
            else self.lambda_floor
        )
        lambda_pre_clamp = finite_numeric_scalar(
            "lambda_pre_clamp", max(floor_term, lambda_candidate)
        )
        # DR-H3 lever G: the applied-multiplier ceiling. OFF (None) => branch
        # skipped (zero float ops) => bit-identical to the pre-DR-H3 recursion.
        # ``lambda_pre_clamp`` (what lambda would have been without the
        # ceiling) is surfaced for the AM-H9 "pinned at the ceiling" read.
        ceiling_engaged = False
        if self.lambda_ceiling is not None and lambda_pre_clamp > self.lambda_ceiling:
            lambda_applied = finite_numeric_scalar(
                "lambda_applied", self.lambda_ceiling
            )
            ceiling_engaged = True
        else:
            lambda_applied = lambda_pre_clamp
        controller = PIController(
            proportional_gain=self.proportional_gain,
            integral_gain=self.integral_gain,
            integral=new_integral,
            budget=self.budget,
            # AM-2: the ceiling MUST persist onto the successor, else it silently
            # resets to None (OFF) after round 1 and the amendment is a no-op.
            integral_cap=self.integral_cap,
            # DR-RELIABILITY lever F: the floor persists for the same reason.
            lambda_floor=self.lambda_floor,
            # DR-H3: both new constants persist for the same AM-2 reason.
            lambda_ceiling=self.lambda_ceiling,
            armed_lambda_floor=self.armed_lambda_floor,
        )
        return PIControllerStep(
            controller=controller,
            lambda_applied=lambda_applied,
            error=error,
            integral=new_integral,
            observed_episodic_cost=observed,
            lambda_pre_clamp=lambda_pre_clamp,
            ceiling_engaged=ceiling_engaged,
            floor_armed=floor_armed,
        )


# ---------------------------------------------------------------------------
# Per-factor entropy-target sustain controller (DR-RELIABILITY lever E)
# ---------------------------------------------------------------------------


def entropy_sustain_target(
    base_target: float,
    round_index: int,
    hold_rounds: int,
    anneal_end_rounds: int,
) -> float:
    """Return the DR-RELIABILITY lever-E entropy target for ``round_index``.

    Piecewise-linear, cliff-free schedule: ``base_target`` for rounds before
    ``hold_rounds``; a linear anneal from ``base_target`` to 0 over
    ``[hold_rounds, anneal_end_rounds)``; exactly 0.0 at and after
    ``anneal_end_rounds``. The anneal REPLACES the falsified keep-alive's hard
    release round (T1-D7's surviving seed collapsed ~13 rounds after that
    cliff): pressure fades only as the target fades, and the dual alpha
    self-decays through its projection once entropy exceeds the falling
    target. ``hold_rounds == anneal_end_rounds`` degenerates to a step at
    ``hold_rounds`` (the >= anneal-end branch runs first, so there is no
    division by zero).
    """

    require_nonnegative_int("round_index", round_index)
    normalized_round = int.__index__(round_index)
    require_nonnegative_int("hold_rounds", hold_rounds)
    normalized_hold = int.__index__(hold_rounds)
    require_nonnegative_int("anneal_end_rounds", anneal_end_rounds)
    normalized_end = int.__index__(anneal_end_rounds)
    if normalized_end < normalized_hold:
        raise ValueError("anneal_end_rounds must be >= hold_rounds")
    base = require_nonnegative_number("base_target", base_target)
    if normalized_round >= normalized_end:
        return 0.0
    if normalized_round < normalized_hold:
        return base
    span = normalized_end - normalized_hold
    remaining = normalized_end - normalized_round
    # Impl-audit fix (SHIFT 19 lens 1): huge-int schedule bounds overflow
    # float() with a raw OverflowError; normalize to the project's ValueError
    # convention (finite_numeric_scalar's huge-int rule) with the same
    # FP association order (bit-exact on all in-range values).
    try:
        scaled = base * float(remaining) / float(span)
    except OverflowError:
        raise ValueError("entropy sustain target must be finite") from None
    return finite_numeric_scalar("entropy sustain target", scaled)


@dataclass(frozen=True)
class EntropySustainStep:
    """One lever-E dual step: successor controller + the applied temperatures.

    ``alpha_movement`` / ``alpha_sensing`` are the entropy-temperature values
    applied to ALL policy epochs of the round (held fixed within the round —
    the DR-D3 lambda cadence). ``movement_target`` / ``sensing_target`` are
    the schedule's targets AT this round (0.0 for an uncontrolled factor).
    """

    controller: "EntropySustainController"
    alpha_movement: float
    alpha_sensing: float
    movement_target: float
    sensing_target: float
    measured_movement_entropy: float
    measured_sensing_entropy: float


@dataclass(frozen=True, kw_only=True)
class EntropySustainController:
    """Projected, capped per-factor entropy-target dual state (lever E).

    DR-RELIABILITY (SHIFT 19): the H2 head of the T1-D9 failure is a
    UNIVERSAL early MOVEMENT-exploration collapse (movement entropy < 0.1 by
    ~r300, ~180 rounds before the hazard multiplier engages) that no
    lambda-controller lever can reach. The falsified fixed-coefficient
    keep-alive family (T1-D7/T1-D8: 0.10 then 0.15, sensing-only, hard
    release) could not track the growing exploitation gradient. This
    controller replaces the OPEN-LOOP fixed bonus with a CLOSED-LOOP dual on
    each factor's own policy entropy — the SAC learned-temperature insight
    (Haarnoja et al. 2018, arXiv:1812.05905): per update round k with
    measured valid-masked mean factor entropy ``H_k`` and scheduled target
    ``H*(k)`` (see ``entropy_sustain_target``):

        alpha_k = min(alpha_cap, max(0, alpha_{k-1} + lr * (H*(k) - H_k)))

    — the exact projected-dual arithmetic of the project's hazard multiplier,
    applied to a DIFFERENT OBJECT: ``alpha_movement`` and ``alpha_sensing``
    are entropy-temperature parameters on the POLICY'S OWN entropy, an
    optimizer-internal device — NOT cost-constraint multipliers; the pillar-1
    claim that hazard exposure is the only Lagrangian-constrained COST is
    untouched (DR-D1's rejection of a second COST constraint is not
    reopened). The cap is mandatory from birth (the T1-D8 integral-windup
    lesson applied preemptively to the new dual). A ``None`` target leaves
    that factor uncontrolled (its alpha is pinned 0.0).

    Honesty clause (DR-reliability V2 amendment A): entropy gradients vanish
    on an already-saturated softmax — the lever PREVENTS the freeze by
    engaging at the target crossing; it is not claimed to unfreeze an
    already-collapsed head. Its pre-registered failure signature is alpha
    pinned at ``alpha_cap`` while that factor's entropy stays below target.

    Sanitized base floats/ints are stored (S2-6) and construction is
    keyword-only (the S-10 lesson), matching ``PIController``.
    """

    learning_rate: float
    movement_target: float | None
    sensing_target: float | None
    hold_rounds: int
    anneal_end_rounds: int
    alpha_cap: float
    alpha_movement: float = 0.0
    alpha_sensing: float = 0.0
    # DR-H4 (SHIFT 23) lever I, OFF by default (byte-identical): when True,
    # the SENSING factor's target ignores the wall-clock hold/anneal schedule
    # and is armed-gated instead — H*_sense (``sensing_target``) at every
    # round where the driver-owned armed floor is UNARMED, and 0.0 with
    # alpha_sensing PINNED to 0.0 while ARMED (the AM-E1 stateless pin: the
    # armed step returns the existing uncontrolled-factor (0.0, 0.0) pattern,
    # arithmetically identical to a reset-at-arm followed by target-0
    # dynamics, with zero new controller state). The movement factor is
    # untouched (AM-H3's movement-specific grounds stand). The gate acts on
    # the ``floor_armed`` flag passed per step; the stored ``sensing_target``
    # is never mutated (the A-10 controller-vs-config cross-check stays
    # coherent).
    sensing_armed_gate: bool = False
    # DR-H5 (SHIFT 25) lever K, OFF by default (byte-identical): when True,
    # the MOVEMENT factor's target consumes the driver-owned recovery latch
    # passed per step (``movement_recovery_active``) — during an engaged
    # recovery the target is held at the base ``movement_target`` (wall-clock
    # schedule bypassed; the lever-I ``target_now_override`` pattern applied
    # to the movement factor); outside recovery at rounds >= anneal-end the
    # movement factor is treated as uncontrolled for the round (alpha 0.0 /
    # target 0.0 — the AM-E1 stateless anti-ratchet pin; production-inert:
    # alpha_move's last nonzero round is r1003/r908/r794/r1121 on
    # s128/s129/s130/s131, all pre-anneal-end); outside recovery before
    # anneal-end the schedule path is byte-identical. The sensing factor is
    # untouched (lever I stands). The stored ``movement_target`` is never
    # mutated (A-10 coherence).
    movement_recovery_gate: bool = False

    def __post_init__(self) -> None:
        learning_rate = require_positive_number(
            "learning_rate",
            self.learning_rate,
            error_suffix="must be positive and finite",
        )
        movement_target = (
            None
            if self.movement_target is None
            else require_positive_number(
                "movement_target",
                self.movement_target,
                error_suffix="must be positive and finite",
            )
        )
        sensing_target = (
            None
            if self.sensing_target is None
            else require_positive_number(
                "sensing_target",
                self.sensing_target,
                error_suffix="must be positive and finite",
            )
        )
        if movement_target is None and sensing_target is None:
            raise ValueError(
                "at least one of movement_target / sensing_target must be set "
                "(a controller with no controlled factor is a misconfiguration)"
            )
        require_nonnegative_int("hold_rounds", self.hold_rounds)
        hold_rounds = int.__index__(self.hold_rounds)
        require_nonnegative_int("anneal_end_rounds", self.anneal_end_rounds)
        anneal_end_rounds = int.__index__(self.anneal_end_rounds)
        if anneal_end_rounds < hold_rounds:
            raise ValueError("anneal_end_rounds must be >= hold_rounds")
        alpha_cap = require_positive_number(
            "alpha_cap",
            self.alpha_cap,
            error_suffix="must be positive and finite",
        )
        alpha_movement = require_nonnegative_number(
            "alpha_movement", self.alpha_movement
        )
        alpha_sensing = require_nonnegative_number(
            "alpha_sensing", self.alpha_sensing
        )
        if alpha_movement > alpha_cap or alpha_sensing > alpha_cap:
            raise ValueError("alpha state must be <= alpha_cap")
        # DR-H4 lever I coupling: strict bool (driver-flag convention) and a
        # controlled sensing factor (gating an uncontrolled factor is a
        # misconfiguration, rejected loudly).
        if type(self.sensing_armed_gate) is not bool:
            raise TypeError("sensing_armed_gate must be a bool")
        if self.sensing_armed_gate and sensing_target is None:
            raise ValueError(
                "sensing_armed_gate=True requires sensing_target to be set "
                "(the armed gate controls the sensing factor; DR-H4)"
            )
        # DR-H5 lever K coupling: strict bool (driver-flag convention) and a
        # controlled movement factor (gating an uncontrolled factor is a
        # misconfiguration, rejected loudly — the lever-I convention).
        if type(self.movement_recovery_gate) is not bool:
            raise TypeError("movement_recovery_gate must be a bool")
        if self.movement_recovery_gate and movement_target is None:
            raise ValueError(
                "movement_recovery_gate=True requires movement_target to be "
                "set (the recovery gate controls the movement factor; DR-H5)"
            )
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "movement_target", movement_target)
        object.__setattr__(self, "sensing_target", sensing_target)
        object.__setattr__(self, "hold_rounds", hold_rounds)
        object.__setattr__(self, "anneal_end_rounds", anneal_end_rounds)
        object.__setattr__(self, "alpha_cap", alpha_cap)
        object.__setattr__(self, "alpha_movement", alpha_movement)
        object.__setattr__(self, "alpha_sensing", alpha_sensing)

    def _step_factor(
        self, alpha: float, target: float | None, *, name: str,
        measured_entropy: float, round_index: int,
        target_now_override: float | None = None,
    ) -> tuple[float, float]:
        """Return (new_alpha, target_now) for one factor's projected dual step.

        ``target_now_override`` (DR-H4 lever I) bypasses the wall-clock
        schedule with an explicit per-step target; ``None`` keeps the
        pre-existing schedule path byte-identical.
        """

        if target is None:
            # Uncontrolled factor: alpha pinned 0.0, target reported 0.0.
            return 0.0, 0.0
        if target_now_override is None:
            target_now = entropy_sustain_target(
                target, round_index, self.hold_rounds, self.anneal_end_rounds
            )
        else:
            target_now = target_now_override
        # A-2 parity: check the pre-projection candidate finite BEFORE the
        # projection, so an overflow cannot be silently clamped into range.
        candidate = finite_numeric_scalar(
            f"{name} alpha candidate",
            alpha + self.learning_rate * (target_now - measured_entropy),
        )
        return min(self.alpha_cap, max(0.0, candidate)), target_now

    def update(
        self,
        *,
        measured_movement_entropy: float,
        measured_sensing_entropy: float,
        round_index: int,
        floor_armed: bool = False,
        movement_recovery_active: bool = False,
    ) -> EntropySustainStep:
        """Run one projected+capped dual step per factor; return the successor.

        Keyword-only (S-10): the two same-typed measured entropies must not be
        positional. Measured entropies are the CURRENT policy's valid-masked
        mean per-factor entropies on the fresh batch (categorical entropy is
        nonnegative by construction; a negative value indicates upstream
        corruption and is rejected).

        ``floor_armed`` (DR-H4 lever I) is the driver-owned armed-floor latch
        for this round — the same flag the PIController receives. It is
        consumed ONLY when ``sensing_armed_gate`` is True: unarmed => the
        sensing target is the base ``sensing_target`` (wall-clock schedule
        bypassed); armed => the sensing factor is treated as uncontrolled for
        the round (alpha 0.0, target 0.0 — the AM-E1 stateless pin). With the
        gate OFF the flag is validated and IGNORED (byte-identical schedule
        path, so T1-D11-style armed-floor-without-gate configs are unchanged).

        ``movement_recovery_active`` (DR-H5 lever K) is the driver-owned
        recovery latch for this round. Consumed ONLY when
        ``movement_recovery_gate`` is True: engaged => the movement target is
        held at the base ``movement_target`` for the round; not engaged at
        rounds >= anneal-end => the movement factor runs the AM-E1 stateless
        pin (alpha 0.0 / target 0.0); not engaged before anneal-end => the
        unchanged schedule path. With the gate OFF the flag is validated and
        IGNORED (byte-identical).
        """

        movement_entropy = require_nonnegative_number(
            "measured_movement_entropy", measured_movement_entropy
        )
        sensing_entropy = require_nonnegative_number(
            "measured_sensing_entropy", measured_sensing_entropy
        )
        require_nonnegative_int("round_index", round_index)
        normalized_round = int.__index__(round_index)
        if type(floor_armed) is not bool:
            raise TypeError("floor_armed must be a bool")
        if type(movement_recovery_active) is not bool:
            raise TypeError("movement_recovery_active must be a bool")
        if self.movement_recovery_gate and movement_recovery_active:
            # DR-H5 lever K: engaged recovery — hold the base movement target
            # at every engaged round (wall-clock schedule bypassed; the
            # lever-I override pattern on the movement factor).
            new_alpha_movement, movement_target_now = self._step_factor(
                self.alpha_movement,
                self.movement_target,
                name="movement",
                measured_entropy=movement_entropy,
                round_index=normalized_round,
                target_now_override=self.movement_target,
            )
        elif (
            self.movement_recovery_gate
            and normalized_round >= self.anneal_end_rounds
        ):
            # DR-H5 anti-ratchet pin: outside recovery, post-anneal, the
            # movement factor is treated as uncontrolled for the round
            # (alpha 0.0 / target 0.0 — the AM-E1 stateless pin; without it,
            # residual alpha from an exited engagement would decay through
            # ~100 rounds into a forming hold or a graded tail).
            new_alpha_movement, movement_target_now = 0.0, 0.0
        else:
            # Gate OFF (flag validated + IGNORED — byte-identical schedule
            # path) or gate ON before anneal-end outside recovery (the
            # unchanged lever-E schedule).
            new_alpha_movement, movement_target_now = self._step_factor(
                self.alpha_movement,
                self.movement_target,
                name="movement",
                measured_entropy=movement_entropy,
                round_index=normalized_round,
            )
        if self.sensing_armed_gate and floor_armed:
            # DR-H4 armed pin: alpha_sensing forced to 0.0 with target 0.0 —
            # the uncontrolled-factor pattern. Arithmetically identical to a
            # reset-at-arm followed by target-0 dynamics (from alpha=0 under
            # target 0, proj(0 + lr*(0 - H)) = 0 every armed round); without
            # the pin, de-arm gaps ratchet alpha into the graded tail
            # (audit-simulated residual ~0.62 on s128's own trajectory).
            new_alpha_sensing, sensing_target_now = 0.0, 0.0
        else:
            new_alpha_sensing, sensing_target_now = self._step_factor(
                self.alpha_sensing,
                self.sensing_target,
                name="sensing",
                measured_entropy=sensing_entropy,
                round_index=normalized_round,
                # Gate ON + unarmed: hold the base target at every round
                # (DR-H4 lever I); gate OFF: None => schedule path, byte-
                # identical to the pre-DR-H4 behavior.
                target_now_override=(
                    self.sensing_target if self.sensing_armed_gate else None
                ),
            )
        controller = EntropySustainController(
            learning_rate=self.learning_rate,
            movement_target=self.movement_target,
            sensing_target=self.sensing_target,
            hold_rounds=self.hold_rounds,
            anneal_end_rounds=self.anneal_end_rounds,
            alpha_cap=self.alpha_cap,
            # The alphas MUST persist onto the successor (the AM-2
            # integral_cap silent-reset lesson, applied to the new dual).
            alpha_movement=new_alpha_movement,
            alpha_sensing=new_alpha_sensing,
            sensing_armed_gate=self.sensing_armed_gate,
            movement_recovery_gate=self.movement_recovery_gate,
        )
        return EntropySustainStep(
            controller=controller,
            alpha_movement=new_alpha_movement,
            alpha_sensing=new_alpha_sensing,
            movement_target=movement_target_now,
            sensing_target=sensing_target_now,
            measured_movement_entropy=movement_entropy,
            measured_sensing_entropy=sensing_entropy,
        )


# ---------------------------------------------------------------------------
# Episodic constraint estimator (DR-D2)
# ---------------------------------------------------------------------------


def episodic_team_hazard_cost(batch: RolloutBatch, *, episode_count: int) -> float:
    """Return J_C-hat: the mean per-episode TEAM hazard cost of ``batch`` (DR-D2).

    ``J_C-hat = (sum of hazard_cost over all valid agent-steps) /
    (number of episodes in the batch)`` — both agents summed, episodes
    weighted equally regardless of length (padded steps are excluded by the
    valid mask), matching the comparator-table convention. Nonnegative by
    construction (a sum of contractually nonnegative costs over a positive
    episode count), which is what keeps the DR-S-11 gate sound.

    DR-D2 T0 anchor: a batch with one 1.0-hazard entry in one of 16 episodes
    yields exactly 0.0625.
    """

    if not isinstance(batch, RolloutBatch):
        raise TypeError("batch must be a RolloutBatch")
    require_positive_int("episode_count", episode_count)
    normalized_count = int.__index__(episode_count)
    # A-4: this estimator is a public export documented callable WITHOUT a
    # preceding batch.validate (the shipped update path validates first). Add
    # light tensor-surface guards on the two fields it reads so a
    # contract-violating hand-built RolloutBatch raises a deterministic error
    # rather than a silently-wrong number or an undeclared torch exception.
    if not isinstance(batch.hazard_cost, torch.Tensor):
        raise TypeError("batch.hazard_cost must be a torch.Tensor")
    valid_mask = batch.valid_mask
    if not isinstance(valid_mask, torch.Tensor) or valid_mask.dtype != torch.bool:
        raise TypeError("batch.valid_mask must be a bool torch.Tensor")
    batch_rows = int(batch.actor_observation.shape[0])
    if batch_rows % normalized_count != 0:
        raise ValueError(
            "episode_count must divide the batch row count "
            "(rows are per episode-agent)"
        )
    # Read valid_mask once (bound above) so the emptiness gate and the
    # selection cannot desynchronize under a hostile __getattribute__.
    if not bool(valid_mask.any().item()):
        raise ValueError("valid_mask must select at least one timestep")
    valid_costs = batch.hazard_cost[valid_mask]
    if not bool(torch.isfinite(valid_costs).all().item()):
        raise ValueError("hazard_cost must contain only finite values")
    if valid_costs.numel() and bool((valid_costs < 0).any().item()):
        raise ValueError("hazard_cost must be nonnegative")
    total = float(valid_costs.sum().item())
    estimate = finite_numeric_scalar(
        "episodic team hazard cost", total / float(normalized_count)
    )
    # Nonnegative-by-construction tripwire (DR-D2 / DR-S-11 coupling): a
    # negative estimate here can only mean upstream corruption.
    return require_nonnegative_number("episodic team hazard cost", estimate)


# ---------------------------------------------------------------------------
# Stage 25 update configuration (DR-D4 integration point)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage25UpdateConfig:
    """Validated Stage 25 reviewed final-training update settings.

    ``from_decision_records()`` is the binding DR-sourced default;
    ``dev_regression_bridge()`` is the DR-D4 T0(a) regression-bridge
    configuration under which this update reproduces the dev update's outputs
    exactly on a fixed batch.

    ``hazard_budget`` is interpreted on the scale selected by
    ``cost_estimator``: the DR-D2 episodic scale (d_ep) under
    ``"episodic_team_mean"``, the dev per-step scale under
    ``"per_step_mean"`` (bridge only). ``dual_step_timing`` of
    ``"post_epochs"`` is the dev-bridge cadence and is only defined for the
    integral-only controller, so it requires ``proportional_gain == 0.0``.
    """

    algorithm: AlgorithmConfig
    losses: LossConfig
    learning_rate: float
    adam_epsilon: float
    max_update_epochs: int
    max_grad_norm: float | None
    standardize_advantages: bool
    kl_early_stop: bool
    target_kl: float
    kl_stop_margin: float
    dual_step_timing: str
    cost_estimator: str
    proportional_gain: float
    integral_gain: float
    initial_integral: float
    hazard_budget: float
    # DR-C4 levers (all OFF by default, so from_decision_records() /
    # dev_regression_bridge() with these at their defaults reproduce the DR-D4
    # path byte-for-byte). See stage25_ppo_lagrangian_update for their effect.
    #  - constraint_warmup_rounds (Fix 2): hold lambda=0 (skip the dual step)
    #    for the first W update rounds, then resume the unchanged DR-D3
    #    controller; 0 => OFF; capped at _CONSTRAINT_WARMUP_MAX (AM-3).
    #  - advantage_std_guard_threshold (Fix 4a): passed to _normalise_advantage;
    #    0.0 => OFF (exact std<=0 branch).
    #  - movement/sensing_entropy_floor (Fix 4b): per-factor entropy-floor
    #    gates; None => OFF (fixed-coefficient entropy bonus).
    constraint_warmup_rounds: int = 0
    advantage_std_guard_threshold: float = 0.0
    movement_entropy_floor: float | None = None
    sensing_entropy_floor: float | None = None
    # DR-D1/L1-b lever (2): warmup-scoped sensing-exploration keep-alive, on the
    # SENSING factor only (movement untouched, so DR-C4's task-first window is
    # preserved). While ``round_index < sensing_keepalive_rounds`` AND (when a
    # target is set) the sensing head's entropy is below ``sensing_entropy_target``,
    # an ADDITIONAL ``sensing_keepalive_coefficient * sensing_entropy`` bonus is
    # subtracted from the loss (stacked on the DR-D4 fixed 0.01 term). All OFF
    # defaults (coefficient 0.0 / rounds 0 / target None) reduce to the exact
    # DR-C4/DR-D4 numerics (byte-identical). Envelope (DR-D1/L1-b): coefficient in
    # [0.05, 0.20], target in [0.20, 0.50]. The AMENDMENT (SHIFT 13) permits
    # ``sensing_keepalive_rounds`` > ``constraint_warmup_rounds`` (bounded only by
    # ``update_rounds`` at the run-config layer) so the keep-alive can bridge the
    # lambda-handoff dead gap into the lambda>0 regime -- e.g. 500 with warmup 300.
    sensing_keepalive_coefficient: float = 0.0
    sensing_keepalive_rounds: int = 0
    sensing_entropy_target: float | None = None
    # DR-D3 AMENDMENT (SHIFT 16): anti-windup integral CEILING, threaded onto the
    # PIController via make_controller(). None (OFF) => the controller runs the
    # unchanged DR-D3 recursion bit-for-bit (from_decision_records() /
    # dev_regression_bridge() leave it None). A set cap bounds the integral to
    # [0, cap]; pinned run value 20.0 (DR-D3-amendment.md AM-1).
    lambda_integral_cap: float | None = None
    # DR-RELIABILITY (SHIFT 19) levers, all OFF by default (byte-identical):
    #  - lambda_floor (lever F): positive sustain floor on the applied
    #    multiplier, threaded onto the PIController via make_controller();
    #    0.0 = OFF (exact ``max(0.0, .)`` DR-D3 projection). Requires the
    #    pre_epochs cadence (the dev-bridge post_epochs path applies the raw
    #    integral directly and would silently bypass the floor). Pinned run
    #    value 3.0, envelope [2.5, 5.0].
    #  - entropy_sustain_* (lever E): the per-factor entropy-target sustain
    #    controller (see EntropySustainController). learning_rate 0.0 = OFF
    #    (make_entropy_controller() returns None; no measurement pass, and the
    #    two loss terms carry coefficient 0.0). When ON: alpha_cap and at
    #    least one target are REQUIRED (capped-from-birth is a DR
    #    requirement), hold <= anneal_end, and the pre_epochs cadence is
    #    required so the dev bridge stays bit-identical by construction.
    #    Pinned run values: lr 0.02, movement target 0.5, sensing target
    #    0.30, hold 800, anneal end 2000, alpha cap 2.0 (DR-reliability.md).
    lambda_floor: float = 0.0
    entropy_sustain_learning_rate: float = 0.0
    movement_entropy_sustain_target: float | None = None
    sensing_entropy_sustain_target: float | None = None
    entropy_sustain_hold_rounds: int = 0
    entropy_sustain_anneal_rounds: int = 0
    entropy_sustain_alpha_cap: float | None = None
    # DR-H3 (SHIFT 21) levers, both OFF by default (byte-identical):
    #  - lambda_ceiling (lever G): hard ceiling on the APPLIED multiplier,
    #    threaded onto the PIController via make_controller(). None = OFF.
    #    Pinned run value 5.0, envelope [4.0, 6.0] (below the measured 6.58
    #    ejection onset; above the measured 2.5-3 re-entry band). Requires the
    #    pre_epochs cadence (the dev-bridge post_epochs path applies the raw
    #    integral directly and would silently bypass the ceiling).
    #  - armed_lambda_floor (lever H): the ARMED conditional floor VALUE,
    #    threaded onto the PIController; the stateful arm (trailing behavioral
    #    window + latch + de-arm hatch) lives in the DRIVER, which passes
    #    ``floor_armed`` per round. None = OFF. Mutually exclusive with a
    #    positive ``lambda_floor`` (the retired unconditional form). Pinned
    #    run value 3.0, envelope [2.5, 4.5], strictly below the ceiling.
    lambda_ceiling: float | None = None
    armed_lambda_floor: float | None = None
    # DR-H4 (SHIFT 23) lever I, OFF by default (byte-identical): armed-gated
    # sensing-sustain release. When True the sensing entropy-sustain target is
    # H*_sense while the armed floor is UNARMED and 0.0 with alpha pinned 0.0
    # while ARMED (the wall-clock sensing anneal is bypassed; movement is
    # untouched). Requires a running sensing sustain (lr > 0 + sensing target)
    # AND the DR-H3 armed floor (the gate consumes the driver-owned
    # ``floor_armed`` latch — without the armed-floor lever there is no arm
    # state to gate on). Pinned run value: True for T1-D12.
    sensing_sustain_armed_gate: bool = False
    # DR-H5 (SHIFT 25) lever K, OFF by default (byte-identical): limbo-scoped
    # armed movement-plasticity re-engagement. When True the movement
    # entropy-sustain target consumes the driver-owned recovery latch
    # (``movement_recovery_active`` per round): engaged => target held at the
    # base H*_move; not engaged post-anneal => alpha_move pinned 0.0 (the
    # AM-E1 stateless pin). The stateful trigger/latch (containment window +
    # exits + exhaustion) lives in the DRIVER (_ArmedFloorState). Requires a
    # running movement sustain (lr > 0 + movement target) AND the DR-H3 armed
    # floor (recovery is an ARMED-phase device). Pinned run value: True for
    # T1-D13.
    movement_recovery_gate: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.algorithm, AlgorithmConfig):
            raise TypeError("algorithm must be an AlgorithmConfig")
        if not isinstance(self.losses, LossConfig):
            raise TypeError("losses must be a LossConfig")
        # A-3 (DR-S2-23 / S2-6 parity): the require_* helpers return the
        # sanitized base float/int; store those (not the caller's objects) via
        # object.__setattr__ so a hostile numeric subclass cannot hijack the
        # KL-threshold / grad-clip / dual-gain arithmetic downstream. The
        # sibling configs AlgorithmConfig/LossConfig were hardened by DR-S2-23;
        # this new config that also feeds arithmetic gets the same treatment.
        learning_rate = require_positive_number(
            "learning_rate",
            self.learning_rate,
            error_suffix="must be positive and finite",
        )
        adam_epsilon = require_positive_number(
            "adam_epsilon",
            self.adam_epsilon,
            error_suffix="must be positive and finite",
        )
        require_positive_int(
            "max_update_epochs", self.max_update_epochs, error_suffix="must be positive"
        )
        max_update_epochs = int.__index__(self.max_update_epochs)
        if max_update_epochs > STAGE25_EPOCH_CAP:
            raise ValueError(
                f"max_update_epochs must be <= {STAGE25_EPOCH_CAP} for the "
                "Stage 25 reviewed update (DR-D4; changing the cap re-triggers "
                "DR-D9 review)"
            )
        max_grad_norm = self.max_grad_norm
        if max_grad_norm is not None:
            max_grad_norm = require_positive_number(
                "max_grad_norm",
                self.max_grad_norm,
                error_suffix="must be positive and finite",
            )
        if not isinstance(self.standardize_advantages, bool):
            raise TypeError("standardize_advantages must be a bool")
        if not isinstance(self.kl_early_stop, bool):
            raise TypeError("kl_early_stop must be a bool")
        target_kl = require_positive_number(
            "target_kl", self.target_kl, error_suffix="must be positive and finite"
        )
        kl_stop_margin = require_positive_number(
            "kl_stop_margin",
            self.kl_stop_margin,
            error_suffix="must be positive and finite",
        )
        if self.dual_step_timing not in _DUAL_STEP_TIMINGS:
            raise ValueError(
                "dual_step_timing must be one of: " + ", ".join(_DUAL_STEP_TIMINGS)
            )
        if self.cost_estimator not in _COST_ESTIMATORS:
            raise ValueError(
                "cost_estimator must be one of: " + ", ".join(_COST_ESTIMATORS)
            )
        proportional_gain = require_nonnegative_number(
            "proportional_gain", self.proportional_gain
        )
        integral_gain = require_positive_number("integral_gain", self.integral_gain)
        initial_integral = require_nonnegative_number(
            "initial_integral", self.initial_integral
        )
        hazard_budget = require_nonnegative_number("hazard_budget", self.hazard_budget)
        if self.dual_step_timing == "post_epochs" and proportional_gain != 0.0:
            raise ValueError(
                "post_epochs dual timing is the dev-bridge integral-only "
                "cadence and requires proportional_gain == 0.0"
            )
        # DR-C4 lever validation + A-3 sanitize-store parity.
        require_nonnegative_int(
            "constraint_warmup_rounds", self.constraint_warmup_rounds
        )
        constraint_warmup_rounds = int.__index__(self.constraint_warmup_rounds)
        if constraint_warmup_rounds > _CONSTRAINT_WARMUP_MAX:
            raise ValueError(
                "constraint_warmup_rounds must be <= "
                f"{_CONSTRAINT_WARMUP_MAX} (DR-C4 AM-3: warmup must end well "
                "before the round budget so the tail stays fully constrained)"
            )
        if constraint_warmup_rounds > 0 and self.dual_step_timing != "pre_epochs":
            raise ValueError(
                "constraint_warmup_rounds > 0 requires pre_epochs dual timing "
                "(warmup gates the pre-epochs DR-D3 dual step; DR-C4)"
            )
        advantage_std_guard_threshold = require_nonnegative_number(
            "advantage_std_guard_threshold",
            self.advantage_std_guard_threshold,
            error_suffix="must be nonnegative and finite",
        )
        movement_entropy_floor = (
            None
            if self.movement_entropy_floor is None
            else require_positive_number(
                "movement_entropy_floor",
                self.movement_entropy_floor,
                error_suffix="must be positive and finite",
            )
        )
        sensing_entropy_floor = (
            None
            if self.sensing_entropy_floor is None
            else require_positive_number(
                "sensing_entropy_floor",
                self.sensing_entropy_floor,
                error_suffix="must be positive and finite",
            )
        )
        # DR-D1/L1-b lever (2) keep-alive validation + A-3 sanitize-store parity.
        sensing_keepalive_coefficient = require_nonnegative_number(
            "sensing_keepalive_coefficient",
            self.sensing_keepalive_coefficient,
            error_suffix="must be nonnegative and finite",
        )
        require_nonnegative_int(
            "sensing_keepalive_rounds", self.sensing_keepalive_rounds
        )
        sensing_keepalive_rounds = int.__index__(self.sensing_keepalive_rounds)
        sensing_entropy_target = (
            None
            if self.sensing_entropy_target is None
            else require_positive_number(
                "sensing_entropy_target",
                self.sensing_entropy_target,
                error_suffix="must be positive and finite",
            )
        )
        # DR-D3 AMENDMENT (SHIFT 16): anti-windup ceiling — None (OFF) stays None;
        # a set cap must be a positive finite number (base float stored, S2-6).
        lambda_integral_cap = (
            None
            if self.lambda_integral_cap is None
            else require_positive_number(
                "lambda_integral_cap",
                self.lambda_integral_cap,
                error_suffix="must be positive and finite",
            )
        )
        # DR-RELIABILITY lever F validation + A-3 sanitize-store parity.
        lambda_floor = require_nonnegative_number(
            "lambda_floor",
            self.lambda_floor,
            error_suffix="must be nonnegative and finite",
        )
        if lambda_floor > 0.0 and self.dual_step_timing != "pre_epochs":
            raise ValueError(
                "lambda_floor > 0 requires pre_epochs dual timing (the "
                "post_epochs dev-bridge cadence applies the raw integral "
                "directly and would silently bypass the floor; DR-RELIABILITY)"
            )
        # DR-H3 lever G/H validation + A-3 sanitize-store parity. The
        # controller re-runs the same coupling checks; validating here too
        # rejects a bad config at construction (before make_controller()).
        lambda_ceiling = (
            None
            if self.lambda_ceiling is None
            else require_positive_number(
                "lambda_ceiling",
                self.lambda_ceiling,
                error_suffix="must be positive and finite",
            )
        )
        armed_lambda_floor = (
            None
            if self.armed_lambda_floor is None
            else require_positive_number(
                "armed_lambda_floor",
                self.armed_lambda_floor,
                error_suffix="must be positive and finite",
            )
        )
        if lambda_ceiling is not None and self.dual_step_timing != "pre_epochs":
            raise ValueError(
                "lambda_ceiling requires pre_epochs dual timing (the "
                "post_epochs dev-bridge cadence applies the raw integral "
                "directly and would silently bypass the ceiling; DR-H3)"
            )
        if armed_lambda_floor is not None and self.dual_step_timing != "pre_epochs":
            raise ValueError(
                "armed_lambda_floor requires pre_epochs dual timing (the "
                "post_epochs dev-bridge cadence applies the raw integral "
                "directly and would silently bypass the armed floor; DR-H3)"
            )
        if armed_lambda_floor is not None and lambda_floor > 0.0:
            raise ValueError(
                "armed_lambda_floor is mutually exclusive with a positive "
                "unconditional lambda_floor (DR-H3: the unconditional form is "
                "retired; running both would re-create the T1-D10 lock)"
            )
        if lambda_ceiling is not None:
            if lambda_floor > 0.0 and lambda_floor >= lambda_ceiling:
                raise ValueError(
                    "lambda_floor must be strictly below lambda_ceiling "
                    "(DR-H3: an empty corridor is a config error)"
                )
            if (
                armed_lambda_floor is not None
                and armed_lambda_floor >= lambda_ceiling
            ):
                raise ValueError(
                    "armed_lambda_floor must be strictly below lambda_ceiling "
                    "(DR-H3: an empty corridor is a config error)"
                )
        # DR-RELIABILITY lever E validation + A-3 sanitize-store parity.
        entropy_sustain_learning_rate = require_nonnegative_number(
            "entropy_sustain_learning_rate",
            self.entropy_sustain_learning_rate,
            error_suffix="must be nonnegative and finite",
        )
        movement_entropy_sustain_target = (
            None
            if self.movement_entropy_sustain_target is None
            else require_positive_number(
                "movement_entropy_sustain_target",
                self.movement_entropy_sustain_target,
                error_suffix="must be positive and finite",
            )
        )
        sensing_entropy_sustain_target = (
            None
            if self.sensing_entropy_sustain_target is None
            else require_positive_number(
                "sensing_entropy_sustain_target",
                self.sensing_entropy_sustain_target,
                error_suffix="must be positive and finite",
            )
        )
        require_nonnegative_int(
            "entropy_sustain_hold_rounds", self.entropy_sustain_hold_rounds
        )
        entropy_sustain_hold_rounds = int.__index__(
            self.entropy_sustain_hold_rounds
        )
        require_nonnegative_int(
            "entropy_sustain_anneal_rounds", self.entropy_sustain_anneal_rounds
        )
        entropy_sustain_anneal_rounds = int.__index__(
            self.entropy_sustain_anneal_rounds
        )
        if entropy_sustain_anneal_rounds < entropy_sustain_hold_rounds:
            raise ValueError(
                "entropy_sustain_anneal_rounds must be >= "
                "entropy_sustain_hold_rounds (the anneal follows the hold)"
            )
        entropy_sustain_alpha_cap = (
            None
            if self.entropy_sustain_alpha_cap is None
            else require_positive_number(
                "entropy_sustain_alpha_cap",
                self.entropy_sustain_alpha_cap,
                error_suffix="must be positive and finite",
            )
        )
        if entropy_sustain_learning_rate > 0.0:
            if self.dual_step_timing != "pre_epochs":
                raise ValueError(
                    "entropy_sustain_learning_rate > 0 requires pre_epochs "
                    "dual timing (the dev bridge must stay bit-identical by "
                    "construction; DR-RELIABILITY)"
                )
            if entropy_sustain_alpha_cap is None:
                raise ValueError(
                    "entropy_sustain_alpha_cap is required when "
                    "entropy_sustain_learning_rate > 0 (the entropy dual is "
                    "capped from birth — the T1-D8 anti-windup lesson; "
                    "DR-RELIABILITY)"
                )
            if (
                movement_entropy_sustain_target is None
                and sensing_entropy_sustain_target is None
            ):
                raise ValueError(
                    "at least one entropy sustain target is required when "
                    "entropy_sustain_learning_rate > 0 (a sustain controller "
                    "with no controlled factor is a misconfiguration)"
                )
        # DR-H4 lever I coupling validation (strict bool per the sibling flag
        # convention; each requirement named so a config error names the
        # config, not the controller).
        if not isinstance(self.sensing_sustain_armed_gate, bool):
            raise TypeError("sensing_sustain_armed_gate must be a bool")
        if self.sensing_sustain_armed_gate:
            if entropy_sustain_learning_rate <= 0.0:
                raise ValueError(
                    "sensing_sustain_armed_gate=True requires "
                    "entropy_sustain_learning_rate > 0 (the gate re-times a "
                    "RUNNING sensing sustain; DR-H4)"
                )
            if sensing_entropy_sustain_target is None:
                raise ValueError(
                    "sensing_sustain_armed_gate=True requires "
                    "sensing_entropy_sustain_target to be set (the gate "
                    "controls the sensing factor; DR-H4)"
                )
            if self.armed_lambda_floor is None:
                raise ValueError(
                    "sensing_sustain_armed_gate=True requires "
                    "armed_lambda_floor to be configured (the gate consumes "
                    "the DR-H3 armed-floor latch; DR-H4)"
                )
        # DR-H5 lever K coupling validation (mirrors the lever-I convention;
        # each requirement named so a config error names the config).
        if not isinstance(self.movement_recovery_gate, bool):
            raise TypeError("movement_recovery_gate must be a bool")
        if self.movement_recovery_gate:
            if entropy_sustain_learning_rate <= 0.0:
                raise ValueError(
                    "movement_recovery_gate=True requires "
                    "entropy_sustain_learning_rate > 0 (the recovery override "
                    "re-engages a RUNNING movement sustain; DR-H5)"
                )
            if movement_entropy_sustain_target is None:
                raise ValueError(
                    "movement_recovery_gate=True requires "
                    "movement_entropy_sustain_target to be set (the recovery "
                    "override controls the movement factor; DR-H5)"
                )
            if self.armed_lambda_floor is None:
                raise ValueError(
                    "movement_recovery_gate=True requires "
                    "armed_lambda_floor to be configured (recovery is an "
                    "ARMED-phase device; DR-H5)"
                )
        object.__setattr__(self, "learning_rate", learning_rate)
        object.__setattr__(self, "adam_epsilon", adam_epsilon)
        object.__setattr__(self, "max_update_epochs", max_update_epochs)
        object.__setattr__(self, "max_grad_norm", max_grad_norm)
        object.__setattr__(self, "target_kl", target_kl)
        object.__setattr__(self, "kl_stop_margin", kl_stop_margin)
        object.__setattr__(self, "proportional_gain", proportional_gain)
        object.__setattr__(self, "integral_gain", integral_gain)
        object.__setattr__(self, "initial_integral", initial_integral)
        object.__setattr__(self, "hazard_budget", hazard_budget)
        object.__setattr__(
            self, "constraint_warmup_rounds", constraint_warmup_rounds
        )
        object.__setattr__(
            self, "advantage_std_guard_threshold", advantage_std_guard_threshold
        )
        object.__setattr__(self, "movement_entropy_floor", movement_entropy_floor)
        object.__setattr__(self, "sensing_entropy_floor", sensing_entropy_floor)
        object.__setattr__(
            self, "sensing_keepalive_coefficient", sensing_keepalive_coefficient
        )
        object.__setattr__(
            self, "sensing_keepalive_rounds", sensing_keepalive_rounds
        )
        object.__setattr__(
            self, "sensing_entropy_target", sensing_entropy_target
        )
        object.__setattr__(self, "lambda_integral_cap", lambda_integral_cap)
        object.__setattr__(self, "lambda_floor", lambda_floor)
        object.__setattr__(self, "lambda_ceiling", lambda_ceiling)
        object.__setattr__(self, "armed_lambda_floor", armed_lambda_floor)
        object.__setattr__(
            self, "entropy_sustain_learning_rate", entropy_sustain_learning_rate
        )
        object.__setattr__(
            self,
            "movement_entropy_sustain_target",
            movement_entropy_sustain_target,
        )
        object.__setattr__(
            self,
            "sensing_entropy_sustain_target",
            sensing_entropy_sustain_target,
        )
        object.__setattr__(
            self, "entropy_sustain_hold_rounds", entropy_sustain_hold_rounds
        )
        object.__setattr__(
            self, "entropy_sustain_anneal_rounds", entropy_sustain_anneal_rounds
        )
        object.__setattr__(
            self, "entropy_sustain_alpha_cap", entropy_sustain_alpha_cap
        )

    @classmethod
    def from_decision_records(cls) -> "Stage25UpdateConfig":
        """Return the binding Stage 25 configuration, one value per DR clause."""

        return cls(
            algorithm=AlgorithmConfig(
                discount_factor=0.99,  # DR-D4: gamma 0.99 (changed from dev 0.95)
                gae_lambda=0.95,  # DR-D4: lambda_GAE 0.95 (changed from dev 0.9)
                ppo_clip_range=0.2,  # DR-D4: clip epsilon 0.2 (universal official)
            ),
            losses=LossConfig(
                reward_entropy_coefficient=0.0,  # DR-D4: combined-entropy stays 0.0
                movement_entropy_coefficient=0.01,  # DR-D4: movement entropy 0.01
                sensing_entropy_coefficient=0.01,  # DR-D4: sensing entropy 0.01
                # Retained for LossConfig compatibility; not read by the update
                # (hazard penalty strength is controller-mediated, DR-D3).
                hazard_cost_coefficient=1.0,
                sensing_cost_coefficient=0.5,  # DR-D1: fixed-weight shaping 0.5
                value_loss_coefficient=0.5,  # DR-D4: value-loss coefficient 0.5
            ),
            learning_rate=3e-4,  # DR-D4: lr 3e-4 (5e-4 is the pre-registered fallback)
            adam_epsilon=1e-5,  # DR-D4: Adam eps 1e-5 (MAPPO official)
            max_update_epochs=10,  # DR-D4: epoch cap 10, KL-guarded
            max_grad_norm=10.0,  # DR-D4: global grad-norm 10.0 (replaces dev 1.0)
            standardize_advantages=True,  # DR-D4/DR-D14: separate pre-mix standardization
            kl_early_stop=True,  # DR-D9: Stage-25-only prevention layer
            target_kl=0.01,  # DR-D9/DR-D4: target_kl 0.01
            kl_stop_margin=1.5,  # DR-D9/DR-D4: stop when approx_kl > 1.5 x target
            dual_step_timing="pre_epochs",  # DR-D3/DR-D4: dual step BEFORE the epochs
            cost_estimator="episodic_team_mean",  # DR-D2: episodic mean team cost
            proportional_gain=0.25,  # DR-D3: K_P 0.25
            integral_gain=0.05,  # DR-D3: K_I 0.05
            initial_integral=0.1,  # DR-D3: I_0 0.1 warm start
            hazard_budget=0.5,  # DR-D2: d_ep 0.5 team-episodic
            # DR-C4 levers OFF at the DR-D4 baseline (byte-reproducible). The
            # DR-C4-active values are applied by the driver from the run config.
            constraint_warmup_rounds=0,
            advantage_std_guard_threshold=0.0,
            movement_entropy_floor=None,
            sensing_entropy_floor=None,
            # DR-D1/L1-b lever (2) keep-alive OFF (byte-reproducible; the
            # DR-D1/L1-b-active values are applied by the driver from the run config).
            sensing_keepalive_coefficient=0.0,
            sensing_keepalive_rounds=0,
            sensing_entropy_target=None,
            # DR-RELIABILITY levers OFF (byte-reproducible; the run-active
            # values are applied by the driver from the run config).
            lambda_floor=0.0,
            entropy_sustain_learning_rate=0.0,
            movement_entropy_sustain_target=None,
            sensing_entropy_sustain_target=None,
            entropy_sustain_hold_rounds=0,
            entropy_sustain_anneal_rounds=0,
            entropy_sustain_alpha_cap=None,
        )

    @classmethod
    def dev_regression_bridge(cls) -> "Stage25UpdateConfig":
        """Return the DR-D4 T0(a) regression-bridge configuration.

        DR-D4 T0(a) verbatim: "the Stage 25 update reproduces the dev update's
        outputs exactly when configured to the dev settings (2 epochs, no KL
        stop, post-hoc integral dual, per-step estimator, lr 0.01,
        gamma 0.95/0.9, clip 1.0-norm) on a fixed batch — the regression bridge
        proving the new path contains the old semantics." Every value below is
        the corresponding ``default_stage22_update_config()`` /
        ``LagrangeConfig`` dev constant; ``adam_epsilon`` is the torch Adam
        default the dev update implicitly uses; advantage standardization is
        off (dev ``normalize_advantages`` default False).
        """

        return cls(
            algorithm=AlgorithmConfig(
                discount_factor=0.95,  # dev gamma
                gae_lambda=0.9,  # dev lambda_GAE
                ppo_clip_range=0.2,  # dev clip epsilon
            ),
            losses=LossConfig(
                reward_entropy_coefficient=0.0,
                movement_entropy_coefficient=0.01,
                sensing_entropy_coefficient=0.01,
                hazard_cost_coefficient=1.0,
                sensing_cost_coefficient=0.5,
                value_loss_coefficient=0.5,
            ),
            learning_rate=0.01,  # dev lr
            adam_epsilon=1e-8,  # torch Adam default (dev builds Adam(lr=...) only)
            max_update_epochs=2,  # dev epoch count
            max_grad_norm=1.0,  # dev grad-norm ("clip 1.0-norm")
            standardize_advantages=False,  # dev normalize_advantages default
            kl_early_stop=False,  # "no KL stop"
            target_kl=0.01,  # inert while kl_early_stop is False
            kl_stop_margin=1.5,  # inert while kl_early_stop is False
            dual_step_timing="post_epochs",  # "post-hoc integral dual"
            cost_estimator="per_step_mean",  # "per-step estimator" (dev D-2 hook)
            proportional_gain=0.0,  # integral-only (the DR-D3 equivalence pin)
            integral_gain=0.2,  # dev eta (LagrangeConfig.learning_rate)
            initial_integral=0.1,  # dev initial multiplier
            hazard_budget=0.1,  # dev per-step hazard budget
            # DR-C4 levers OFF: the dev bridge must remain byte-identical to the
            # bounded Stage 22 dev update.
            constraint_warmup_rounds=0,
            advantage_std_guard_threshold=0.0,
            movement_entropy_floor=None,
            sensing_entropy_floor=None,
            # DR-D1/L1-b lever (2) keep-alive OFF (byte-reproducible; the
            # DR-D1/L1-b-active values are applied by the driver from the run config).
            sensing_keepalive_coefficient=0.0,
            sensing_keepalive_rounds=0,
            sensing_entropy_target=None,
            # DR-RELIABILITY levers OFF: the dev bridge must remain
            # byte-identical to the bounded Stage 22 dev update.
            lambda_floor=0.0,
            entropy_sustain_learning_rate=0.0,
            movement_entropy_sustain_target=None,
            sensing_entropy_sustain_target=None,
            entropy_sustain_hold_rounds=0,
            entropy_sustain_anneal_rounds=0,
            entropy_sustain_alpha_cap=None,
        )

    def make_controller(self) -> PIController:
        """Return the initial PI controller state for this configuration."""

        return PIController(
            proportional_gain=self.proportional_gain,
            integral_gain=self.integral_gain,
            integral=self.initial_integral,
            budget=self.hazard_budget,
            integral_cap=self.lambda_integral_cap,  # DR-D3-amendment
            lambda_floor=self.lambda_floor,  # DR-RELIABILITY lever F
            lambda_ceiling=self.lambda_ceiling,  # DR-H3 lever G
            armed_lambda_floor=self.armed_lambda_floor,  # DR-H3 lever H
        )

    def make_entropy_controller(self) -> "EntropySustainController | None":
        """Return the initial lever-E entropy controller, or None when OFF."""

        if self.entropy_sustain_learning_rate <= 0.0:
            return None
        return EntropySustainController(
            learning_rate=self.entropy_sustain_learning_rate,
            movement_target=self.movement_entropy_sustain_target,
            sensing_target=self.sensing_entropy_sustain_target,
            hold_rounds=self.entropy_sustain_hold_rounds,
            anneal_end_rounds=self.entropy_sustain_anneal_rounds,
            alpha_cap=self.entropy_sustain_alpha_cap,
            sensing_armed_gate=self.sensing_sustain_armed_gate,  # DR-H4 lever I
            movement_recovery_gate=self.movement_recovery_gate,  # DR-H5 lever K
        )


@dataclass(frozen=True)
class Stage25UpdateResult:
    """In-memory Stage 25 update result: successor controller + finite summary.

    ``entropy_controller`` is the DR-RELIABILITY lever-E successor state
    (``None`` when the lever is OFF); a trailing defaulted field so the
    pre-existing keyword construction stays valid.
    """

    controller: PIController
    lambda_applied: float
    summary: dict[str, Any]
    entropy_controller: "EntropySustainController | None" = None


# ---------------------------------------------------------------------------
# The Stage 25 reviewed update
# ---------------------------------------------------------------------------


def stage25_ppo_lagrangian_update(
    model: RecurrentMAPPOActorCritic,
    batch: RolloutBatch,
    update_config: Stage25UpdateConfig,
    *,
    optimizer: torch.optim.Optimizer,
    controller: PIController,
    episode_count: int | None = None,
    round_index: int | None = None,
    reward_shaping_potentials: "tuple[torch.Tensor, torch.Tensor] | None" = None,
    sensing_shaping_potentials: "tuple[torch.Tensor, torch.Tensor] | None" = None,
    entropy_controller: "EntropySustainController | None" = None,
    floor_armed: bool = False,
    movement_recovery_active: bool = False,
) -> Stage25UpdateResult:
    """Run one Stage 25 reviewed full-batch PPO-Lagrangian update round.

    Round structure (binding order per DR-D3/DR-D4):

    1. constraint estimate from the fresh batch (DR-D2 episodic team mean, or
       the dev per-step mean under the bridge configuration);
    2. ONE dual step producing ``lambda_applied`` — BEFORE the policy epochs
       under ``pre_epochs`` timing (``post_epochs`` is the dev-bridge cadence).
       DR-C4 warmup: while ``round_index < constraint_warmup_rounds`` the dual
       step is SKIPPED and ``lambda_applied = 0`` (the controller integral is
       left untouched), so the unchanged DR-D3 controller resumes exactly at
       round ``W`` (AM-3);
    3. GAE computed ONCE before the epoch loop (DR-D4); value regression
       targets from the RAW advantages. DR-C4 shaping (Fix 1): when
       ``reward_shaping_potentials`` is supplied, the policy-invariant term
       ``gamma*Phi(s') - Phi(s)`` is added to the reward channel BEFORE the GAE
       (never to the hazard-cost channel);
    4. SEPARATE whole-batch standardization of the reward and cost advantages
       (population std, 1e-8 clamp, mean-subtract-only on zero std — the
       imported ``_normalise_advantage`` semantics, extended by the DR-C4
       std-guard) BEFORE the unnormalized ``A_r - lambda_applied * A_c`` mixing
       (DR-D14);
    5. up to ``max_update_epochs`` full-batch epochs (no minibatching), each
       followed by a no-grad k3 approximate-KL measurement over the valid
       mask; the remaining epochs stop when ``approx_kl > kl_stop_margin *
       target_kl`` (DR-D9). The DR-C4 per-factor entropy floors gate the
       entropy bonus per epoch. The ``losses.py`` ratio fail-fast abort stays
       untouched underneath — a non-finite ratio still aborts loudly.

    ``optimizer`` is REQUIRED and must be the persistent optimizer owning
    exactly the model parameters (DR-D4: one persistent Adam across rounds;
    there is deliberately no owned-optimizer fallback path here).
    ``episode_count`` of ``None`` derives the count from the batch rows and
    ``model.config.agent_id_count`` (rows are per episode-agent).
    ``round_index`` is the 0-based update round; it is REQUIRED when
    ``constraint_warmup_rounds > 0`` (DR-C4 warmup needs the round to gate the
    dual step) and otherwise informational. ``reward_shaping_potentials`` is the
    ``(Phi(s_t), Phi(s_{t+1}))`` pair from the DR-C4 shaping-enabled collection
    (``None`` => no shaping => the DR-D4 path is byte-identical).

    With every DR-C4 lever OFF (``constraint_warmup_rounds == 0``,
    ``advantage_std_guard_threshold == 0.0``, both entropy floors ``None``, and
    ``reward_shaping_potentials is None``) this function is byte-identical to
    the pre-DR-C4 DR-D4 path (T0(a) regression pin).

    Like the dev update, this function intentionally leaves the model in
    training mode after execution.
    """

    if not isinstance(model, RecurrentMAPPOActorCritic):
        raise TypeError("model must be a RecurrentMAPPOActorCritic")
    if not isinstance(batch, RolloutBatch):
        raise TypeError("batch must be a RolloutBatch")
    if not isinstance(update_config, Stage25UpdateConfig):
        raise TypeError("update_config must be a Stage25UpdateConfig")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise TypeError(
            "optimizer must be a persistent torch.optim.Optimizer "
            "(the Stage 25 update has no owned-optimizer path)"
        )
    if not isinstance(controller, PIController):
        raise TypeError("controller must be a PIController")
    if round_index is not None:
        require_nonnegative_int("round_index", round_index)
    if update_config.constraint_warmup_rounds > 0 and round_index is None:
        raise ValueError(
            "round_index is required when constraint_warmup_rounds > 0 "
            "(DR-C4 warmup needs the current round to gate the dual step)"
        )
    if update_config.sensing_keepalive_coefficient > 0.0 and round_index is None:
        raise ValueError(
            "round_index is required when sensing_keepalive_coefficient > 0 "
            "(DR-D1/L1-b keep-alive is warmup-scoped and needs the current round; "
            "a missing round_index would silently disable it)"
        )
    # DR-RELIABILITY lever E: when the sustain lever is ON, the round index and
    # the threaded controller state are both REQUIRED — a missing argument
    # would silently disable the lever (the same failure mode the keep-alive
    # guard above closes).
    if update_config.entropy_sustain_learning_rate > 0.0:
        if round_index is None:
            raise ValueError(
                "round_index is required when entropy_sustain_learning_rate > 0 "
                "(the lever-E target schedule needs the current round)"
            )
        if entropy_controller is None:
            raise ValueError(
                "entropy_controller is required when "
                "entropy_sustain_learning_rate > 0 (build it with "
                "update_config.make_entropy_controller() and thread the "
                "successor across rounds)"
            )
    if entropy_controller is not None and not isinstance(
        entropy_controller, EntropySustainController
    ):
        raise TypeError(
            "entropy_controller must be an EntropySustainController or None"
        )
    # DR-H3 lever H: the arm signal is driver-owned; a True signal without the
    # configured lever would silently run the un-floored dynamics under an
    # armed narrative (the controller re-checks, but rejecting here names the
    # config, not the controller). Strict-bool per the driver-flag convention.
    if type(floor_armed) is not bool:
        raise TypeError("floor_armed must be a bool")
    if floor_armed and update_config.armed_lambda_floor is None:
        raise ValueError(
            "floor_armed=True requires armed_lambda_floor to be configured "
            "(DR-H3 lever H)"
        )
    # DR-H5 lever K: same rejection shape — an active recovery flag against an
    # un-gated config would silently run un-supported dynamics under a
    # recovery narrative.
    if type(movement_recovery_active) is not bool:
        raise TypeError("movement_recovery_active must be a bool")
    if movement_recovery_active and not update_config.movement_recovery_gate:
        raise ValueError(
            "movement_recovery_active=True requires movement_recovery_gate "
            "to be configured (DR-H5 lever K)"
        )
    settings = update_config
    # A-10 (DR-D3/DR-D4 binding): the controller's constant dual gains/budget
    # must match the update config's (make_controller() builds exactly this);
    # a mismatched controller would silently run non-DR dual math under a
    # from_decision_records() config. The integral state is intentionally NOT
    # checked — it evolves across rounds.
    if (
        controller.proportional_gain != settings.proportional_gain
        or controller.integral_gain != settings.integral_gain
        or controller.budget != settings.hazard_budget
        # AM-3: reject a silently-mismatched anti-windup ceiling before training
        # (None != None is False, so this stays OFF-safe when the cap is unused).
        or controller.integral_cap != settings.lambda_integral_cap
        # DR-RELIABILITY lever F: the sustain floor gets the same A-10 guard
        # (0.0 != 0.0 is False, so this stays OFF-safe).
        or controller.lambda_floor != settings.lambda_floor
        # DR-H3 levers G/H: same A-10 guard (None != None is False => OFF-safe).
        or controller.lambda_ceiling != settings.lambda_ceiling
        or controller.armed_lambda_floor != settings.armed_lambda_floor
    ):
        raise ValueError(
            "controller dual constants (proportional_gain / integral_gain / "
            "budget / integral_cap / lambda_floor / lambda_ceiling / "
            "armed_lambda_floor) must match the update "
            "config; build the controller with update_config.make_controller()"
        )
    # DR-RELIABILITY lever E (A-10 parity): the entropy controller's constants
    # must match the update config's — a mismatched controller would silently
    # run non-DR sustain math. The alpha STATE is intentionally NOT checked —
    # it evolves across rounds.
    if entropy_controller is not None and (
        entropy_controller.learning_rate != settings.entropy_sustain_learning_rate
        or entropy_controller.movement_target
        != settings.movement_entropy_sustain_target
        or entropy_controller.sensing_target
        != settings.sensing_entropy_sustain_target
        or entropy_controller.hold_rounds != settings.entropy_sustain_hold_rounds
        or entropy_controller.anneal_end_rounds
        != settings.entropy_sustain_anneal_rounds
        or entropy_controller.alpha_cap != settings.entropy_sustain_alpha_cap
        # DR-H4 lever I: same A-10 guard (False != False => OFF-safe).
        or entropy_controller.sensing_armed_gate
        != settings.sensing_sustain_armed_gate
        # DR-H5 lever K: same A-10 guard (False != False => OFF-safe).
        or entropy_controller.movement_recovery_gate
        != settings.movement_recovery_gate
    ):
        raise ValueError(
            "entropy controller constants (learning_rate / targets / "
            "hold_rounds / anneal_end_rounds / alpha_cap / "
            "sensing_armed_gate / movement_recovery_gate) must match the "
            "update config; build the controller with "
            "update_config.make_entropy_controller()"
        )
    batch.validate(model.config)
    _validate_rollout_for_update(batch)
    if not batch.valid_mask.any().item():
        raise ValueError("valid_mask must contain at least one true timestep")
    _validate_optimizer_parameter_ownership(model, optimizer)
    resolved_episode_count = _resolve_episode_count(model, batch, episode_count)

    # --- 1. Constraint estimate from the fresh batch --------------------------
    if settings.cost_estimator == "episodic_team_mean":
        observed_cost = episodic_team_hazard_cost(
            batch, episode_count=resolved_episode_count
        )
    else:
        # Bridge configuration only: the dev D-2 per-step-mean hook, imported.
        observed_cost = cost_estimate(batch)

    # --- 2. Dual step (pre-epochs under the DR-D3 cadence) --------------------
    integral_before = float(controller.integral)
    # DR-C4 warmup (Fix 2): hold lambda=0 and SKIP the dual step for the first
    # constraint_warmup_rounds rounds; the controller integral is left untouched
    # so the unchanged DR-D3 controller resumes exactly at round W (AM-3). Only
    # defined for the pre_epochs (DR-D4) cadence (post_epochs is dev-bridge only,
    # and the config rejects warmup > 0 under post_epochs).
    warmup_active = (
        settings.dual_step_timing == "pre_epochs"
        and settings.constraint_warmup_rounds > 0
        and round_index is not None
        and int.__index__(round_index) < settings.constraint_warmup_rounds
    )
    if settings.dual_step_timing == "pre_epochs":
        if warmup_active:
            dual_step = None
            lambda_applied = 0.0
        else:
            dual_step = controller.update(
                observed_episodic_cost=observed_cost, floor_armed=floor_armed
            )
            lambda_applied = dual_step.lambda_applied
    else:
        # Dev-bridge cadence: the epochs consume the current integral state
        # (== the dev multiplier value under K_P = 0), and the dual step runs
        # after the epoch loop exactly as the dev update does.
        dual_step = None
        lambda_applied = integral_before

    before_parameters = [
        parameter.detach().clone() for parameter in model.parameters()
    ]

    # --- 3. GAE once, value targets from RAW advantages ------------------------
    # DR-D1: the sensing cost stays fixed-weight reward shaping; the signal is
    # built by the imported dev hook (single source of the D-1 semantics).
    development_reward_signal = _development_reward_signal(batch, settings.losses)
    # DR-C4 shaping (Fix 1): add the policy-invariant potential difference
    # gamma*Phi(s') - Phi(s) to the REWARD channel only (never the hazard-cost
    # channel), so the sparse goal reward acquires a dense gradient WITHOUT
    # changing the optimal policy set (Ng-Harada-Russell 1999) or what
    # team_success means. The shaped term never leaves this reward signal: it is
    # not persisted into any graded success/hazard/sensing field (AM-2). None
    # potentials => byte-identical DR-D4 path.
    reward_shaping_applied = False
    reward_shaping_mean = 0.0
    if reward_shaping_potentials is not None:
        # Deterministic tuple-arity + tensor-surface guard on this documented
        # public callable (the A-4 / S2-* standard; parity with the sibling
        # explained_variance_from_batch reward_shaping guard). Without it a
        # malformed argument would leak an undeclared AttributeError on the
        # .shape read below.
        if (
            not isinstance(reward_shaping_potentials, tuple)
            or len(reward_shaping_potentials) != 2
        ):
            raise TypeError(
                "reward_shaping_potentials must be a 2-tuple of torch.Tensors "
                "or None"
            )
        shaping_potential, shaping_potential_next = reward_shaping_potentials
        if not isinstance(shaping_potential, torch.Tensor) or not isinstance(
            shaping_potential_next, torch.Tensor
        ):
            raise TypeError(
                "reward_shaping_potentials entries must be torch.Tensors"
            )
        if (
            shaping_potential.shape != development_reward_signal.shape
            or shaping_potential_next.shape != development_reward_signal.shape
        ):
            raise ValueError(
                "reward_shaping_potentials must match the reward-signal shape "
                "[batch_rows, time]"
            )
        shaping = potential_shaping_term(
            shaping_potential,
            shaping_potential_next,
            settings.algorithm.discount_factor,
        )
        development_reward_signal = development_reward_signal + shaping
        reward_shaping_applied = True
        reward_shaping_mean = _masked_mean_float(shaping, batch.valid_mask)
    # DR-D1/L1-b lever (1): the decision-relevant sensing credit, a SECOND
    # policy-invariant state-potential added to the REWARD channel (never to
    # hazard_cost). Same potential-difference form + same guards as the DR-C4 task
    # potential; the two state potentials compose linearly so Phi_total = Phi_BFS +
    # Phi_sense telescopes (AM-2). None => byte-identical.
    sensing_shaping_applied = False
    sensing_shaping_mean = 0.0
    if sensing_shaping_potentials is not None:
        if (
            not isinstance(sensing_shaping_potentials, tuple)
            or len(sensing_shaping_potentials) != 2
        ):
            raise TypeError(
                "sensing_shaping_potentials must be a 2-tuple of torch.Tensors "
                "or None"
            )
        sensing_potential, sensing_potential_next = sensing_shaping_potentials
        if not isinstance(sensing_potential, torch.Tensor) or not isinstance(
            sensing_potential_next, torch.Tensor
        ):
            raise TypeError(
                "sensing_shaping_potentials entries must be torch.Tensors"
            )
        if (
            sensing_potential.shape != development_reward_signal.shape
            or sensing_potential_next.shape != development_reward_signal.shape
        ):
            raise ValueError(
                "sensing_shaping_potentials must match the reward-signal shape "
                "[batch_rows, time]"
            )
        sensing_shaping = potential_shaping_term(
            sensing_potential,
            sensing_potential_next,
            settings.algorithm.discount_factor,
        )
        development_reward_signal = development_reward_signal + sensing_shaping
        sensing_shaping_applied = True
        sensing_shaping_mean = _masked_mean_float(sensing_shaping, batch.valid_mask)
    raw_reward_advantage = compute_reward_gae(
        development_reward_signal,
        batch.reward_value,
        batch.next_reward_value,
        batch.terminal,
        batch.truncation,
        batch.valid_mask,
        settings.algorithm.discount_factor,
        settings.algorithm.gae_lambda,
    ).detach()
    raw_hazard_cost_advantage = compute_cost_gae(
        batch.hazard_cost,
        batch.hazard_cost_value,
        batch.next_hazard_cost_value,
        batch.terminal,
        batch.truncation,
        batch.valid_mask,
        settings.algorithm.discount_factor,
        settings.algorithm.gae_lambda,
    ).detach()
    reward_return_target = (raw_reward_advantage + batch.reward_value).detach()
    hazard_cost_return_target = (
        raw_hazard_cost_advantage + batch.hazard_cost_value
    ).detach()

    # --- 4. SEPARATE pre-mix standardization (DR-D14 / DR-D4) ------------------
    policy_reward_advantage = raw_reward_advantage
    policy_hazard_cost_advantage = raw_hazard_cost_advantage
    if settings.standardize_advantages:
        # DR-C4 std-guard (Fix 4a): threshold 0.0 => the exact DR-D14 std<=0
        # branch (byte-identical); a positive threshold routes a tiny-but-nonzero
        # std through mean-centering instead of dividing by a near-zero std.
        policy_reward_advantage = _normalise_advantage(
            policy_reward_advantage,
            batch.valid_mask,
            std_guard_threshold=settings.advantage_std_guard_threshold,
        ).detach()
        policy_hazard_cost_advantage = _normalise_advantage(
            policy_hazard_cost_advantage,
            batch.valid_mask,
            std_guard_threshold=settings.advantage_std_guard_threshold,
        ).detach()

    old_joint_log_probability = batch.old_joint_log_probability.detach()
    clip_range = settings.algorithm.ppo_clip_range
    kl_stop_threshold = settings.kl_stop_margin * settings.target_kl

    # --- 4b. DR-RELIABILITY lever E: one entropy-sustain dual step per round ---
    # BEFORE the policy epochs (the DR-D3 lambda cadence): measure the CURRENT
    # policy's valid-masked mean per-factor entropies on the fresh batch (one
    # no-grad forward with the batch actions supplied — sampling is the only
    # global-RNG consumer in the model, so this forward is trajectory-inert),
    # run one projected+capped dual step per factor, and hold the resulting
    # temperatures fixed across the round's epochs. OFF (controller None) =>
    # the forward is SKIPPED entirely and both temperatures are exact 0.0, so
    # the two additive loss terms below reduce to the keep-alive's verified
    # byte-identical 0.0 pattern.
    alpha_movement = 0.0
    alpha_sensing = 0.0
    entropy_sustain_movement_target = 0.0
    entropy_sustain_sensing_target = 0.0
    measured_movement_entropy = 0.0
    measured_sensing_entropy = 0.0
    successor_entropy_controller = entropy_controller
    if entropy_controller is not None:
        with torch.no_grad():
            (
                measured_sensing_entropy,
                measured_movement_entropy,
            ) = _masked_factor_entropies(model, batch)
        entropy_step = entropy_controller.update(
            measured_movement_entropy=measured_movement_entropy,
            measured_sensing_entropy=measured_sensing_entropy,
            round_index=int.__index__(round_index),
            # DR-H4 lever I: the same driver-owned latch the PIController
            # receives; consumed only when the gate is ON (validated + ignored
            # when OFF — byte-identical schedule path).
            floor_armed=floor_armed,
            # DR-H5 lever K: the driver-owned recovery latch; consumed only
            # when the gate is ON (validated + ignored when OFF).
            movement_recovery_active=movement_recovery_active,
        )
        alpha_movement = entropy_step.alpha_movement
        alpha_sensing = entropy_step.alpha_sensing
        entropy_sustain_movement_target = entropy_step.movement_target
        entropy_sustain_sensing_target = entropy_step.sensing_target
        successor_entropy_controller = entropy_step.controller

    # --- 5. KL-guarded full-batch epoch loop (DR-D4 / DR-D9) -------------------
    final_scalars: dict[str, float] = {}
    approx_kl_per_epoch: list[float] = []
    grad_norm_pre_clip_per_epoch: list[float] = []
    grad_norm_clip_activated_epochs = 0
    approx_kl_final = 0.0
    ratio_clip_fraction = 0.0
    epochs_used = 0
    kl_early_stopped = False
    # DR-D1/L1-b: True iff the sensing keep-alive bonus actually contributed in at
    # least one epoch this round (i.e. in-window AND the target-entropy gate was
    # open), so the diagnostic reflects real firing, not just the window (finding
    # from the SHIFT-10 adversarial pass).
    sensing_keepalive_fired = False
    model.train()
    for _epoch in range(settings.max_update_epochs):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(
            ActorInput(batch.actor_observation, batch.revealed_information),
            CriticInput(batch.central_state),
            sensing_action=batch.sensing_action,
            movement_action=batch.movement_action,
            sample_actions=False,
        )
        policy = outputs["policy"]
        new_joint_log_probability = (
            policy.sensing_log_probability + policy.movement_log_probability
        )
        # DR-D14: unnormalized A_r - lambda_applied * A_c mixing inside the
        # shared losses primitive, fed the (optionally pre-standardized)
        # advantages. The ratio fail-fast abort (DR-D9 integrity layer) lives
        # inside ppo_clipped_surrogate, untouched.
        policy_loss = lagrangian_ppo_policy_loss(
            new_joint_log_probability,
            old_joint_log_probability,
            policy_reward_advantage,
            policy_hazard_cost_advantage,
            lambda_applied,
            clip_range,
            batch.valid_mask,
        )
        # DR-D4: plain masked MSE for both critics, no value clipping, no
        # value normalization, single optimizer.
        reward_critic_loss = value_loss(
            outputs["reward_value"], reward_return_target, batch.valid_mask
        )
        hazard_critic_loss = value_loss(
            outputs["hazard_cost_value"], hazard_cost_return_target, batch.valid_mask
        )
        sensing_entropy = entropy_bonus(policy.sensing_entropy, batch.valid_mask)
        movement_entropy = entropy_bonus(policy.movement_entropy, batch.valid_mask)
        # DR-C4 entropy floors (Fix 4b): gate each per-factor entropy bonus.
        # Both floors None => both gates 1.0 => ``coeff * 1.0 * entropy`` reduces
        # to the exact fixed-coefficient DR-D4 term (byte-identical). A positive
        # floor applies the bonus only while that head's entropy is below the
        # floor. The combined-entropy term (reward_entropy_coefficient, 0.0 in
        # both shipped configs) is deliberately left ungated.
        sensing_entropy_gate = _entropy_floor_gate(
            sensing_entropy, settings.sensing_entropy_floor
        )
        movement_entropy_gate = _entropy_floor_gate(
            movement_entropy, settings.movement_entropy_floor
        )
        # DR-D1/L1-b lever (2): warmup-scoped sensing keep-alive (SENSING factor
        # ONLY). An ADDITIONAL entropy bonus stacked on the DR-D4 sensing term,
        # active only while round_index < sensing_keepalive_rounds AND (when a
        # target is set) the sensing head entropy is below it. Coefficient 0.0
        # (default) or round_index None or outside the window => contributes an
        # additive 0.0 => byte-identical to the DR-C4/DR-D4 loss. Movement is
        # never touched, so DR-C4's task-first C4 window is preserved.
        sensing_keepalive_coefficient = 0.0
        if (
            settings.sensing_keepalive_coefficient > 0.0
            and round_index is not None
            and round_index < settings.sensing_keepalive_rounds
            and (
                settings.sensing_entropy_target is None
                or float(sensing_entropy.detach().item())
                < settings.sensing_entropy_target
            )
        ):
            sensing_keepalive_coefficient = settings.sensing_keepalive_coefficient
            sensing_keepalive_fired = True
        # The composition mirrors the dev update term-for-term (including the
        # combined-entropy term at its dev coefficient) so the bridge
        # configuration is arithmetically identical.
        total_loss = (
            policy_loss
            + settings.losses.value_loss_coefficient
            * (reward_critic_loss + hazard_critic_loss)
            - settings.losses.sensing_entropy_coefficient
            * sensing_entropy_gate
            * sensing_entropy
            - settings.losses.movement_entropy_coefficient
            * movement_entropy_gate
            * movement_entropy
            - settings.losses.reward_entropy_coefficient
            * (sensing_entropy + movement_entropy)
            - sensing_keepalive_coefficient * sensing_entropy
            # DR-RELIABILITY lever E: the per-factor entropy-sustain
            # temperatures (Python floats, exact 0.0 when OFF — the verified
            # keep-alive additive byte-identity pattern; held fixed across the
            # round's epochs like lambda_applied).
            - alpha_sensing * sensing_entropy
            - alpha_movement * movement_entropy
        )
        _require_finite_scalar("total_loss", total_loss)
        total_loss.backward()
        if settings.max_grad_norm is not None:
            # V-1 / DR-D4 / DR-D14: the DR-D14 fallback trigger is graded on
            # GRADIENT-NORM clip activation, so the pre-clip global gradient
            # norm (clip_grad_norm_'s return value) is captured per epoch.
            # Capturing the return changes no arithmetic — the dev-bridge
            # bit-identity is preserved.
            grad_norm_pre_clip = float(
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), settings.max_grad_norm
                ).item()
            )
            # A-8 (DR-D9 integrity-layer parity): fail fast on a non-finite
            # pre-clip gradient norm BEFORE optimizer.step. clip_grad_norm_
            # defaults to error_if_nonfinite=False, so a nan norm would
            # otherwise scale every gradient to nan and silently corrupt the
            # parameters (the summary's list field escapes the scalar
            # finiteness guard); this mirrors the masked_mean finiteness guard
            # on the k3 KL path and the losses ratio fail-fast.
            if not math.isfinite(grad_norm_pre_clip):
                raise ValueError(
                    "Stage 25 update produced a non-finite gradient norm"
                )
            grad_norm_pre_clip_per_epoch.append(grad_norm_pre_clip)
            if grad_norm_pre_clip > float(settings.max_grad_norm):
                grad_norm_clip_activated_epochs += 1
        optimizer.step()
        epochs_used += 1
        final_scalars = {
            "policy_loss": float(policy_loss.detach().item()),
            "reward_value_loss": float(reward_critic_loss.detach().item()),
            "hazard_cost_value_loss": float(hazard_critic_loss.detach().item()),
            "sensing_entropy": float(sensing_entropy.detach().item()),
            "movement_entropy": float(movement_entropy.detach().item()),
            "total_loss": float(total_loss.detach().item()),
        }
        # DR-D9: after each full-batch epoch, measure the k3 approximate KL of
        # the post-step policy against the rollout policy over the valid mask;
        # stop the round's remaining epochs beyond the margin. The measurement
        # is a no-grad forward: it touches no parameters, no gradients, and no
        # RNG, so the bridge configuration stays arithmetically identical to
        # the dev update. A non-finite ratio here fails fast (masked_mean's
        # finiteness guard) — consistent with the D-9 integrity layer.
        with torch.no_grad():
            approx_kl_final, ratio_clip_fraction = _k3_kl_and_ratio_clip_fraction(
                model, batch, old_joint_log_probability, clip_range
            )
        approx_kl_per_epoch.append(approx_kl_final)
        if settings.kl_early_stop and approx_kl_final > kl_stop_threshold:
            kl_early_stopped = True
            break

    # V-1 / DR-D14: the fraction of executed epochs whose pre-clip global
    # gradient norm exceeded max_grad_norm — the gradient-norm clip-activation
    # trigger quantity. 0.0 when gradient clipping is disabled (max_grad_norm
    # None; never the case under either shipped configuration).
    grad_norm_clip_activation_fraction = (
        float(grad_norm_clip_activated_epochs) / float(epochs_used)
        if grad_norm_pre_clip_per_epoch
        else 0.0
    )

    # --- Post-epochs dual step (dev-bridge cadence only; NEVER during warmup) --
    if dual_step is None and not warmup_active:
        dual_step = controller.update(observed_episodic_cost=observed_cost)

    # Resolve the dual trace + successor controller. During DR-C4 warmup the
    # controller is deliberately NOT stepped (dual_step is None): the successor
    # is the untouched controller, lambda stayed 0, and the logged e_k / I_k are
    # the (unchanged) current values so the curve trace stays continuous. When
    # warmup is OFF this is exactly the pre-DR-C4 behavior (dual_step non-None).
    if warmup_active:
        successor_controller = controller
        dual_error = finite_numeric_scalar(
            "dual error", observed_cost - float(controller.budget)
        )
        dual_integral = float(controller.integral)
    else:
        successor_controller = dual_step.controller
        dual_error = float(dual_step.error)
        dual_integral = float(dual_step.integral)

    # DR-C4 attribution diagnostic (not a graded metric): the reward-advantage
    # population std must be > 0 once shaping is on (verifies the ED-C4 Link-5
    # zero-variance freeze is repaired).
    raw_reward_advantage_std = float(
        raw_reward_advantage[batch.valid_mask].std(unbiased=False).item()
    )

    parameter_delta_l1 = _parameter_delta_l1(
        before_parameters, list(model.parameters())
    )
    if parameter_delta_l1 <= 0.0:
        raise ValueError("Stage 25 update did not change model parameters")

    summary: dict[str, Any] = {
        "advantage_standardization": (
            "separate_pre_mix_population_std"
            if settings.standardize_advantages
            else "none"
        ),
        "advantages_standardized": settings.standardize_advantages,
        "approx_kl_final": approx_kl_final,
        "approx_kl_per_epoch": list(approx_kl_per_epoch),
        "constraint_warmup_active": warmup_active,
        "cost_estimator": settings.cost_estimator,
        "development_reward_signal": (
            "task_reward - sensing_cost_coefficient * sensing_cost"
        ),
        "dual_error": dual_error,
        "dual_integral": dual_integral,
        "dual_integral_before": integral_before,
        "dual_step_timing": settings.dual_step_timing,
        # DR-RELIABILITY lever E diagnostics (UNCONDITIONAL — the curve builder
        # hard-indexes summary keys; OFF sentinels are 0.0/False). Never graded
        # success/hazard/sensing fields (the AM-2 no-leak convention).
        "entropy_sustain_active": bool(entropy_controller is not None),
        "entropy_sustain_alpha_movement": float(alpha_movement),
        "entropy_sustain_alpha_sensing": float(alpha_sensing),
        "entropy_sustain_measured_movement_entropy": float(
            measured_movement_entropy
        ),
        "entropy_sustain_measured_sensing_entropy": float(
            measured_sensing_entropy
        ),
        "entropy_sustain_movement_target": float(
            entropy_sustain_movement_target
        ),
        "entropy_sustain_sensing_target": float(entropy_sustain_sensing_target),
        # DR-H4 lever I flag (UNCONDITIONAL, same convention; the gated target
        # behavior itself is replicable from entropy_sustain_sensing_target +
        # lambda_floor_armed — this flag makes the config visible per-round).
        "sensing_sustain_armed_gate": bool(settings.sensing_sustain_armed_gate),
        # DR-H5 lever K flags (UNCONDITIONAL, same convention; the override
        # behavior is replicable from entropy_sustain_movement_target + the
        # driver's recovery curve keys — these make the config and the
        # per-round latch visible on every row).
        "movement_recovery_gate": bool(settings.movement_recovery_gate),
        "movement_recovery_active": bool(movement_recovery_active),
        "epochs_used": epochs_used,
        "episode_count": resolved_episode_count,
        # V-1 / DR-D14: the DR-D14 fallback trigger quantity (gradient-norm
        # clip activation), distinct from the honest-named PPO-ratio
        # diagnostic ratio_clip_fraction below.
        "grad_norm_clip_activation_fraction": grad_norm_clip_activation_fraction,
        "grad_norm_pre_clip_per_epoch": list(grad_norm_pre_clip_per_epoch),
        "hazard_budget": float(controller.budget),
        "kl_early_stopped": kl_early_stopped,
        "lagrangian_sign_convention": (
            "reward_advantage - lambda_applied * hazard_cost_advantage"
        ),
        "lambda_applied": float(lambda_applied),
        # DR-RELIABILITY lever F diagnostic (unconditional; 0.0 = OFF).
        "lambda_floor": float(settings.lambda_floor),
        # DR-H3 diagnostics (unconditional keys so every curve row carries
        # them; with the levers OFF they reduce to lambda_applied / False):
        #  - lambda_pre_clamp: what THIS round's applied lambda would have been
        #    WITHOUT the lever-G ceiling (the AM-H9 "pinned at the ceiling"
        #    read). Scoped to the pre_epochs cadence where the pre-epochs dual
        #    step produced lambda_applied; equals lambda_applied whenever the
        #    ceiling did not bind, and equals it trivially during warmup (0.0)
        #    and on the dev-bridge cadence (where the levers are config-
        #    rejected and the applied value is the pre-step integral);
        #  - lambda_ceiling_engaged: True iff the ceiling bound this round;
        #  - lambda_floor_armed: the driver's arm state applied to this round.
        "lambda_pre_clamp": float(
            dual_step.lambda_pre_clamp
            if (
                settings.dual_step_timing == "pre_epochs"
                and dual_step is not None
                and dual_step.lambda_pre_clamp is not None
            )
            else lambda_applied
        ),
        "lambda_ceiling_engaged": bool(
            dual_step.ceiling_engaged
            if settings.dual_step_timing == "pre_epochs" and dual_step is not None
            else False
        ),
        "lambda_floor_armed": bool(floor_armed),
        "model_mode_policy": "stage25_update_leaves_model_in_train_mode",
        # Holds whatever estimator the configuration selects: the DR-D2
        # episodic team mean under from_decision_records(), the dev per-step
        # mean under the dev_regression_bridge() configuration.
        "observed_episodic_cost": float(observed_cost),
        "parameter_delta_l1": parameter_delta_l1,
        "policy_hazard_cost_advantage_mean": _masked_mean_float(
            policy_hazard_cost_advantage, batch.valid_mask
        ),
        "policy_reward_advantage_mean": _masked_mean_float(
            policy_reward_advantage, batch.valid_mask
        ),
        # Honest name for the k3-pass PPO-ratio measurement (fraction of valid
        # timesteps with |r - 1| > clip_range); NOT the DR-D14 trigger.
        "ratio_clip_fraction": ratio_clip_fraction,
        "raw_hazard_cost_advantage_mean": _masked_mean_float(
            raw_hazard_cost_advantage, batch.valid_mask
        ),
        "raw_reward_advantage_mean": _masked_mean_float(
            raw_reward_advantage, batch.valid_mask
        ),
        # DR-C4 attribution diagnostics (auxiliary, clearly-named; NEVER graded
        # success/hazard/sensing fields -- AM-2). reward_shaping_mean is the
        # masked mean of the shaping term actually added to the reward channel;
        # it is 0.0 when shaping is OFF.
        "raw_reward_advantage_std": raw_reward_advantage_std,
        "reward_shaping_applied": reward_shaping_applied,
        "reward_shaping_mean": reward_shaping_mean,
        # DR-D1/L1-b auxiliary diagnostics (clearly-named; NEVER graded
        # success/hazard/sensing fields -- AM-2). sensing_shaping_mean is the
        # masked mean of the decision-relevant sensing-credit term actually added
        # to the reward channel (0.0 when OFF); sensing_keepalive_active marks the
        # rounds inside the keep-alive warmup window.
        "sensing_shaping_applied": sensing_shaping_applied,
        "sensing_shaping_mean": sensing_shaping_mean,
        # True iff the keep-alive bonus actually contributed in >= 1 epoch this
        # round (in-window AND target gate open), not merely that the window was
        # active -- so an analyst attributing "keep-alive helped this round" reads
        # a faithful flag.
        "sensing_keepalive_active": bool(sensing_keepalive_fired),
        "reward_return_target_mean": _masked_mean_float(
            reward_return_target, batch.valid_mask
        ),
        "hazard_cost_return_target_mean": _masked_mean_float(
            hazard_cost_return_target, batch.valid_mask
        ),
        "value_targets_use_raw_advantages": True,
        **final_scalars,
    }
    _validate_summary(summary)
    return Stage25UpdateResult(
        controller=successor_controller,
        lambda_applied=float(lambda_applied),
        summary=summary,
        # DR-RELIABILITY lever E: the successor sustain state (None when OFF);
        # the driver threads it across rounds exactly like the PI controller.
        entropy_controller=successor_entropy_controller,
    )


def _resolve_episode_count(
    model: RecurrentMAPPOActorCritic,
    batch: RolloutBatch,
    episode_count: int | None,
) -> int:
    """Resolve the batch's episode count (rows are per episode-agent)."""

    batch_rows = int(batch.actor_observation.shape[0])
    agent_count = model.config.agent_id_count
    if episode_count is not None:
        require_positive_int("episode_count", episode_count)
        normalized = int.__index__(episode_count)
        if batch_rows % normalized != 0:
            raise ValueError(
                "episode_count must divide the batch row count "
                "(rows are per episode-agent)"
            )
        # A-1: cross-check against the configured agent count when known.
        # Divisibility alone accepts a wrong-but-dividing count (e.g. passing
        # the row count itself), which linearly rescales the DR-D2 J_C-hat and
        # therefore the dual error / integral / lambda — a silent constraint
        # distortion. rows // episode_count must equal the per-episode agent
        # count (the exact relation the None-derivation branch below uses).
        if agent_count is not None and batch_rows // normalized != int(agent_count):
            raise ValueError(
                "episode_count is inconsistent with the configured agent count: "
                "batch_rows // episode_count must equal agent_id_count "
                "(rows are per episode-agent)"
            )
        return normalized
    if agent_count is None:
        raise ValueError(
            "episode_count is required when model.config.agent_id_count is None"
        )
    if batch_rows % int(agent_count) != 0:
        raise ValueError(
            "batch row count is not divisible by the configured agent count"
        )
    return batch_rows // int(agent_count)


def _masked_factor_entropies(
    model: RecurrentMAPPOActorCritic,
    batch: RolloutBatch,
) -> tuple[float, float]:
    """Measure the current policy's valid-masked mean per-factor entropies.

    Returns ``(sensing_entropy, movement_entropy)`` as floats — the exact
    quantities the epoch loop's ``entropy_bonus`` terms compute, measured
    BEFORE the round's first optimizer step (the lever-E dual-step input).
    The forward supplies the batch actions with ``sample_actions=False``, so
    it consumes NO global RNG (sampling is the model's only RNG consumer) —
    trajectory-inert. Callers wrap this in ``torch.no_grad()``.
    """

    outputs = model(
        ActorInput(batch.actor_observation, batch.revealed_information),
        CriticInput(batch.central_state),
        sensing_action=batch.sensing_action,
        movement_action=batch.movement_action,
        sample_actions=False,
    )
    policy = outputs["policy"]
    sensing = float(
        entropy_bonus(policy.sensing_entropy, batch.valid_mask).item()
    )
    movement = float(
        entropy_bonus(policy.movement_entropy, batch.valid_mask).item()
    )
    return sensing, movement


def _k3_kl_and_ratio_clip_fraction(
    model: RecurrentMAPPOActorCritic,
    batch: RolloutBatch,
    old_joint_log_probability: torch.Tensor,
    clip_range: float,
) -> tuple[float, float]:
    """Measure the post-step k3 approximate KL and PPO-ratio clip fraction.

    k3 estimator (DR-D9, the CleanRL low-variance form): ``mean((r - 1) -
    log r)`` over the valid mask, with ``r = exp(new_joint_log_prob -
    old_joint_log_prob)`` of the CURRENT (post-step) policy against the
    rollout policy. The elementwise ``(r - 1) - log r`` is nonnegative for all
    ``r > 0``, so the estimate is nonnegative by construction. The ratio-clip
    fraction is the fraction of valid timesteps with ``|r - 1| > clip_range``
    — an honest PPO-ratio diagnostic only. It is NOT the DR-D14 fallback
    trigger: that trigger is graded on GRADIENT-NORM clip activation, which
    the epoch loop measures from ``clip_grad_norm_``'s pre-clip return (V-1).
    Callers wrap this in ``torch.no_grad()``.
    """

    outputs = model(
        ActorInput(batch.actor_observation, batch.revealed_information),
        CriticInput(batch.central_state),
        sensing_action=batch.sensing_action,
        movement_action=batch.movement_action,
        sample_actions=False,
    )
    policy = outputs["policy"]
    new_joint_log_probability = (
        policy.sensing_log_probability + policy.movement_log_probability
    )
    log_ratio = new_joint_log_probability - old_joint_log_probability
    ratio = torch.exp(log_ratio)
    approx_kl = float(
        masked_mean((ratio - 1.0) - log_ratio, batch.valid_mask).item()
    )
    clip_fraction = float(
        masked_mean(
            ((ratio - 1.0).abs() > clip_range).to(ratio.dtype), batch.valid_mask
        ).item()
    )
    return approx_kl, clip_fraction


# Import-time self-check parity with the dev update: the frozen dataclass
# equality below asserts the bridge constructor stays a genuine dev-constant
# mirror if either constructor drifts (cheap structural pin, no torch work).
if dataclasses.astuple(
    Stage25UpdateConfig.dev_regression_bridge().algorithm
) != dataclasses.astuple(
    AlgorithmConfig(discount_factor=0.95, gae_lambda=0.9, ppo_clip_range=0.2)
):  # pragma: no cover
    raise AssertionError(
        "dev_regression_bridge algorithm constants drifted from the dev update"
    )

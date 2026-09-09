"""Nonnegative Lagrange multiplier state for constrained policy objectives."""

from __future__ import annotations

from dataclasses import dataclass

from raas_marl.mappo_lagrangian._validation import (
    finite_numeric_scalar,
    require_nonnegative_number,
    require_positive_number,
)


@dataclass(frozen=True)
class LagrangeMultiplier:
    """Immutable scalar multiplier with projected additive updates.

    Implements the projected dual-ascent rule of Ray, Achiam & Amodei (2019),
    "Benchmarking Safe Exploration in Deep Reinforcement Learning":
    ``lambda_{t+1} = max(0, lambda_t + eta * (J_C - d))``.

    Positional construction is ``LagrangeMultiplier(value, learning_rate)``.
    Both fields are same-typed floats, so a positional swap is silently
    accepted; callers should prefer keyword construction
    (``LagrangeMultiplier(value=..., learning_rate=...)``) to avoid it
    (NEW-lagrange-1).
    """

    value: float
    learning_rate: float

    def __post_init__(self) -> None:
        normalized_value = require_nonnegative_number("value", self.value)
        normalized_rate = require_positive_number("learning_rate", self.learning_rate)
        # The sanitized base floats are stored (not the caller's objects) so a
        # hostile int/float subclass cannot hijack the dual-step arithmetic in
        # update() through overridden operators (session 2026-07-04; S-7
        # extension). Equality and hashing are unchanged for genuine numbers.
        object.__setattr__(self, "value", normalized_value)
        object.__setattr__(self, "learning_rate", normalized_rate)

    def update(
        self,
        *,
        observed_cost: float,
        cost_budget: float,
    ) -> LagrangeMultiplier:
        """Return a new multiplier after one projected dual-ascent step.

        A new immutable ``LagrangeMultiplier`` is returned with
        ``learning_rate`` carried forward unchanged and
        ``value = max(0.0, self.value + learning_rate * (observed_cost -
        cost_budget))``; the receiver is not mutated.

        ``observed_cost`` and ``cost_budget`` are keyword-only because they
        are mutually swappable same-typed floats and a silent swap would
        reverse the dual-step direction without raising. Requiring
        ``observed_cost >= 0`` is a deliberate domain restriction relative to
        the sign-unrestricted J_C in Ray et al. (2019): every cost signal in
        this codebase is contractually nonnegative, so a negative observed
        cost indicates an upstream bug and must fail fast. The pre-projection
        step value is checked finite explicitly so an overflowing dual step
        raises a named error instead of surfacing as a constructor error (or,
        for negative overflow, being silently projected to zero).

        The ``observed_cost >= 0`` floor is queued for Phase 6 (decision S-11,
        coupled with D-2): if a centered/episodic cost estimator ever replaces
        the D-2 per-step mean it may legitimately produce negative
        ``(J_C - baseline)`` values, at which point this call would relax to
        ``finite_numeric_scalar('observed_cost', ...)``. The current
        nonnegative behavior stays until that decision lands.
        """

        observed = require_nonnegative_number("observed_cost", observed_cost)
        budget = require_nonnegative_number("cost_budget", cost_budget)
        candidate = self.value + self.learning_rate * (observed - budget)
        # Delegate the finiteness paranoia to the shared scalar helper so the
        # pre-projection overflow guard is not hand-rolled (NEW-lagrange-4);
        # ``candidate`` is a genuine float, so this only exercises the
        # ValueError("updated multiplier value must be finite") path.
        candidate = finite_numeric_scalar("updated multiplier value", candidate)
        return LagrangeMultiplier(
            value=max(0.0, candidate),
            learning_rate=self.learning_rate,
        )


def update_lagrange_multiplier(
    *,
    current_value: float,
    learning_rate: float,
    observed_cost: float,
    cost_budget: float,
) -> float:
    """Return a projected nonnegative multiplier update.

    Functional convenience wrapper around ``LagrangeMultiplier.update``. All
    four parameters are keyword-only: they are same-typed floats, so a
    positional swap would silently reverse either the constructor fields or
    the dual-step direction without raising (the stale positional-order pin is
    lifted; no caller passes these positionally).

    ``current_value`` and ``learning_rate`` are validated here first so a bad
    value is reported under the wrapper parameter name rather than the
    dataclass field name ``value`` (S2-24).
    """

    require_nonnegative_number("current_value", current_value)
    require_positive_number("learning_rate", learning_rate)
    return (
        LagrangeMultiplier(value=current_value, learning_rate=learning_rate)
        .update(observed_cost=observed_cost, cost_budget=cost_budget)
        .value
    )

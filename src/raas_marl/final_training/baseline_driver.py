"""Stage 25 harness v0 baseline driver (torch-eager).

Drives the UNTOUCHED Stage 22 bounded PPO-Lagrangian update
(``stage22_ppo_lagrangian_update``) over Stage 23-C multi-episode collection in
a long loop, with a persistent Adam optimizer, a threaded Lagrange multiplier,
incremental JSONL curves and episode logs, critic explained-variance
diagnostics, scripted-comparator readiness probes on the training and
readiness hazard-layout surfaces (the held-out variant is never instantiated),
a run manifest with heartbeat/kill/crash status, and a lambda kill bound.

Development-only, and NOT the reviewed final-training update path — that is
decision-queue item D-4 and it is not decided or implemented here. The update
math, collection semantics, and governance boundaries of Segment 1 are reused
unchanged. This driver does not execute the project's locked final-assessment
protocol, creates no evidence artifacts for the research claim, saves no
checkpoints, serializes no model or optimizer state, runs no baseline
comparison as evidence, no ablation, and no statistical test, and makes no
Bayesian-belief or formal-VOI claim. Every persisted record spreads the
Stage 25 harness development-only boundary flags.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

# Private Stage 24 diagnostics helpers imported deliberately (D-7 lesson:
# shared helper bodies are imported, never copied). Each private name and its
# role here:
#   _no_sense_policy / _always_sense_policy / _selective_sense_policy —
#     scripted PolicyFn callables run per hazard-layout variant surface;
#   _attach_pairwise_comparator_fields — the single source of the pairwise
#     strict-inequality comparator semantics;
#   _run_policy — the catalog-scenario episode runner (learned-policy record);
#   _run_policy_on_environment — the variant-surface episode runner;
#   _stage24a_variant_environment_config — variant -> environment config.
# Phase-6 promotion note: if the harness outlives v0, these should be promoted
# to public stage24_diagnostics exports and this import updated; until then
# the private import is the D-7-conformant alternative to copying the bodies.
from raas_marl.environments.active_sensing.stage24_diagnostics import (
    Stage24ComparatorConfig,
    _always_sense_policy,
    _attach_pairwise_comparator_fields,
    _no_sense_policy,
    _run_policy,
    _run_policy_on_environment,
    _selective_sense_policy,
    _stage24a_variant_environment_config,
    always_sense_shortest_path,
    fork_curriculum_scenarios,
    fork_hazard_layout_families,
    no_sense_shortest_path,
    random_policy,
    risk_aware_oracle_or_heuristic,
    selective_sense_risk_aware,
    stage24a_hazard_layout_variants,
)
from raas_marl.environments.active_sensing.tensor_adapter import (
    default_stage23_core_config,
)
from raas_marl.final_training.run_logging import (
    append_jsonl_record,
    build_run_manifest_payload,
    harness_allowed_structural_keys,
    harness_boundary_flags,
    resolve_run_directory,
    validate_run_id,
    write_run_manifest,
)
from raas_marl.mappo_lagrangian._governance import (
    active_root,
    reject_unsafe_boundary_values,
    utc_timestamp,
)
from raas_marl.mappo_lagrangian._rollout_common import (
    actor_observation_batch,
    revealed_information_batch,
)
from raas_marl.mappo_lagrangian._validation import (
    finite_numeric_scalar,
    require_bounded_seed,
    require_nonnegative_int,
    require_nonnegative_number,
    require_positive_int,
    require_positive_number,
    require_positive_seed,
)
from raas_marl.mappo_lagrangian.buffer import RolloutBatch
from raas_marl.mappo_lagrangian.lagrange import LagrangeMultiplier
from raas_marl.mappo_lagrangian.losses import compute_cost_gae, compute_reward_gae
from raas_marl.mappo_lagrangian.model import ActorInput, RecurrentMAPPOActorCritic
from raas_marl.mappo_lagrangian.stage23c_rollout import (
    Stage23CRolloutConfig,
    collect_stage23c_development_rollout,
)
from raas_marl.mappo_lagrangian.stage24_collector import (
    stage24a_runtime_reproducibility_record,
)
from raas_marl.mappo_lagrangian.update import (
    Stage22UpdateConfig,
    _development_reward_signal,
    default_stage22_update_config,
    stage22_ppo_lagrangian_update,
)

__all__ = [
    "BaselineRunConfig",
    "explained_variance_from_batch",
    "greedy_model_policy_fn",
    "main",
    "run_baseline_training",
    "run_readiness_probe",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-round base-seed stride. Stage 23-C internally derives at most
# episode_index*1000 + step_index (<= 8*1000 + 32 = 8032) above the round
# seed, so consecutive round seeds spaced 100000 apart can never collide.
_ROUND_SEED_STRIDE = 100000
_STAGE23C_MAX_RESEED_OFFSET = 8 * 1000 + 32  # Stage 23-C episode cap * stride + horizon
_MIN_EPISODES_PER_ROUND = 2
_MAX_EPISODES_PER_ROUND = 8  # the Stage 23-C cap
_MIN_STEPS_PER_EPISODE = 1
_MAX_STEPS_PER_EPISODE = 32
_MAX_UPDATE_ROUNDS = 20000
_VALID_TIERS = ("T0", "T1", "T2")

_PROBE_SCENARIO = "risk_gate_hidden_hazard"
_LEARNED_POLICY_NAME = "learned_greedy_policy"
_HELD_OUT_GROUP = "held_out"
_VARIANT_PROBE_GROUPS = ("training", "readiness")
_HISTORY_STATE_KEY = "model_history_state"
_ADAPT_FLAG_KEY = "used_revealed_information_to_adapt_movement"

_DEGENERATE_VARIANCE_FLOOR = 1e-12

# The five scripted catalog comparators keyed by their policy_name, run on the
# probe scenario surface via the public stage24_diagnostics entry points.
_SCENARIO_COMPARATOR_ORDER = (
    "no_sense_shortest_path",
    "always_sense_shortest_path",
    "random_policy",
    "risk_aware_oracle_or_heuristic",
    "selective_sense_risk_aware",
)

# The scripted comparators run per hazard-layout variant (private PolicyFn
# callables shared with the Stage 24 diagnostics dispatchers; imported, never
# copied — D-7 lesson). The oracle and random comparators are scenario-surface
# only in this probe.
_VARIANT_COMPARATOR_POLICIES = (
    ("no_sense_shortest_path", _no_sense_policy),
    ("always_sense_shortest_path", _always_sense_policy),
    ("selective_sense_risk_aware", _selective_sense_policy),
)

PolicyFn = Callable[..., dict[str, dict[str, int]]]


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineRunConfig:
    """Validated Stage 25 harness v0 baseline-run settings.

    ``probe_seed`` should stay out of the 2400/24400 diagnostic seed bands
    used by the Stage 24 in-memory diagnostics entry points so probe RNG
    streams stay disjoint from the regression-oracle streams (documented
    convention; not enforced). ``entropy_floor`` is a soft flag threshold
    only in v0: entropy collapse is recorded per round, never a kill.
    """

    run_id: str
    seed: int
    update_rounds: int
    episodes_per_round: int = 4
    steps_per_episode: int | None = None
    scenario_names: tuple[str, ...] = ("risk_gate_hidden_hazard",)
    sample_actions: bool = True
    probe_every: int = 50
    probe_seed: int = 9001
    lambda_kill_bound: float = 1000.0
    entropy_floor: float = 0.01
    tier: str = "T1"
    result_parent: str | Path | None = None
    heartbeat_every: int = 25

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        require_positive_seed("seed", self.seed)
        require_positive_int("update_rounds", self.update_rounds)
        update_rounds = int.__index__(self.update_rounds)
        if update_rounds > _MAX_UPDATE_ROUNDS:
            raise ValueError(f"update_rounds must be <= {_MAX_UPDATE_ROUNDS}")
        # Bound the seed so the largest derived per-round base seed plus the
        # largest Stage 23-C per-step reseed offset still fits in a signed
        # 64-bit integer and never crashes torch.manual_seed downstream
        # (single-homed in _validation.require_bounded_seed; the historical
        # message is preserved via error_suffix).
        require_bounded_seed(
            "seed",
            self.seed,
            max_offset=(
                update_rounds * _ROUND_SEED_STRIDE + _STAGE23C_MAX_RESEED_OFFSET
            ),
            error_suffix=(
                "must satisfy seed + update_rounds * 100000 + 8032 "
                "<= 2**63 - 1 so every per-round and per-step reseed stays "
                "in signed 64-bit range"
            ),
        )
        require_positive_int("episodes_per_round", self.episodes_per_round)
        episodes_per_round = int.__index__(self.episodes_per_round)
        if not (_MIN_EPISODES_PER_ROUND <= episodes_per_round <= _MAX_EPISODES_PER_ROUND):
            raise ValueError(
                f"episodes_per_round must be between {_MIN_EPISODES_PER_ROUND} "
                f"and {_MAX_EPISODES_PER_ROUND}"
            )
        if self.steps_per_episode is not None:
            require_positive_int(
                "steps_per_episode", self.steps_per_episode, error_suffix="must be positive"
            )
            steps_per_episode = int.__index__(self.steps_per_episode)
            if not (_MIN_STEPS_PER_EPISODE <= steps_per_episode <= _MAX_STEPS_PER_EPISODE):
                raise ValueError(
                    f"steps_per_episode must be None or between "
                    f"{_MIN_STEPS_PER_EPISODE} and {_MAX_STEPS_PER_EPISODE}"
                )
        if not isinstance(self.scenario_names, tuple) or not self.scenario_names:
            raise ValueError("scenario_names must be a non-empty tuple")
        for name in self.scenario_names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("scenario_names entries must be non-empty strings")
        # Deeper catalog validation is delegated to Stage23CRolloutConfig at
        # collection time (single source of the catalog-name rule).
        if not isinstance(self.sample_actions, bool):
            raise TypeError("sample_actions must be a bool")
        # probe_every == 0 disables periodic probes; a value above
        # update_rounds simply means only the always-run final probe fires
        # (deliberately not capped at update_rounds so the default cadence
        # stays valid for short runs).
        require_nonnegative_int("probe_every", self.probe_every)
        # The probe seed is used as-is (no derived offsets), so it is bounded
        # with max_offset=0 to keep torch.manual_seed in signed 64-bit range.
        require_bounded_seed("probe_seed", self.probe_seed, max_offset=0)
        require_positive_number(
            "lambda_kill_bound",
            self.lambda_kill_bound,
            error_suffix="must be positive and finite",
        )
        require_nonnegative_number(
            "entropy_floor",
            self.entropy_floor,
            error_suffix="must be nonnegative and finite",
        )
        if not isinstance(self.tier, str):
            raise TypeError("tier must be a string")
        if self.tier == "T3":
            raise ValueError("tier T3 is reserved for the Stage 26 protocol executor")
        if self.tier not in _VALID_TIERS:
            raise ValueError("tier must be one of T0, T1, T2")
        if self.result_parent is not None and not isinstance(
            self.result_parent, (str, Path)
        ):
            raise TypeError("result_parent must be a str, Path, or None")
        require_positive_int("heartbeat_every", self.heartbeat_every)


# ---------------------------------------------------------------------------
# Explained-variance diagnostics (mirrors the update's own advantage math)
# ---------------------------------------------------------------------------


def _population_explained_variance(
    target: torch.Tensor,
    value: torch.Tensor,
) -> tuple[float, float, bool]:
    """Return (explained_variance, target_variance, degenerate_flag).

    Population variance (``unbiased=False``) to match the update's advantage
    normalization convention. A degenerate target variance (<= 1e-12) returns
    an explained variance of 0.0 with the degenerate flag True — never NaN,
    because persisted artifacts are written with ``allow_nan=False``.
    """

    target_variance = float(torch.var(target, unbiased=False).item())
    if not target_variance > _DEGENERATE_VARIANCE_FLOOR:
        return 0.0, target_variance, True
    residual_variance = float(torch.var(target - value, unbiased=False).item())
    return 1.0 - residual_variance / target_variance, target_variance, False


def explained_variance_from_batch(
    batch: RolloutBatch,
    update_config: Stage22UpdateConfig,
    *,
    reward_shaping: "torch.Tensor | None" = None,
) -> dict[str, float | bool]:
    """Compute critic explained variance against the update's own regression targets.

    Recomputes the raw advantages EXACTLY as ``stage22_ppo_lagrangian_update``
    does: the reward GAE runs on the D-1 hook ``_development_reward_signal``
    (task_reward minus the fixed-weight sensing cost — imported, never
    re-hardcoded) and the cost GAE on the raw hazard cost; the regression
    targets are raw advantage plus the stored collection-time values
    (``value_targets_use_raw_advantages`` convention). Statistics are
    restricted to ``batch.valid_mask``.

    DR-C4: when ``reward_shaping`` is supplied (the ``gamma*Phi(s')-Phi(s)`` term
    the Stage 25 update adds to the reward channel), it is added to the reward
    signal here too, so the reported reward-critic EV is measured against the
    SAME shaped returns the critic is actually trained on. ``None`` (the default)
    reproduces the pre-DR-C4 EV byte-for-byte. Only the reward channel is
    affected; the hazard-cost EV never sees shaping.
    """

    if not isinstance(batch, RolloutBatch):
        raise TypeError("batch must be a RolloutBatch")
    if not isinstance(update_config, Stage22UpdateConfig):
        raise TypeError("update_config must be a Stage22UpdateConfig")
    # Mirror the update's own precondition: an all-False valid mask would
    # otherwise produce torch NaN variances and a misleading finiteness error.
    if not bool(batch.valid_mask.any().item()):
        raise ValueError(
            "valid_mask must select at least one timestep for "
            "explained-variance diagnostics"
        )
    if reward_shaping is not None:
        if not isinstance(reward_shaping, torch.Tensor):
            raise TypeError("reward_shaping must be a torch.Tensor or None")
        if reward_shaping.shape != batch.task_reward.shape:
            raise ValueError(
                "reward_shaping must match batch.task_reward shape"
            )

    gamma = update_config.algorithm.discount_factor
    lam = update_config.algorithm.gae_lambda
    with torch.no_grad():
        reward_signal = _development_reward_signal(batch, update_config.losses)
        if reward_shaping is not None:
            reward_signal = reward_signal + reward_shaping
        reward_raw_advantage = compute_reward_gae(
            reward_signal,
            batch.reward_value,
            batch.next_reward_value,
            batch.terminal,
            batch.truncation,
            batch.valid_mask,
            gamma,
            lam,
        )
        hazard_cost_raw_advantage = compute_cost_gae(
            batch.hazard_cost,
            batch.hazard_cost_value,
            batch.next_hazard_cost_value,
            batch.terminal,
            batch.truncation,
            batch.valid_mask,
            gamma,
            lam,
        )
        mask = batch.valid_mask
        reward_target = reward_raw_advantage + batch.reward_value
        hazard_cost_target = hazard_cost_raw_advantage + batch.hazard_cost_value
        reward_ev, reward_variance, reward_degenerate = _population_explained_variance(
            reward_target[mask], batch.reward_value[mask]
        )
        cost_ev, cost_variance, cost_degenerate = _population_explained_variance(
            hazard_cost_target[mask], batch.hazard_cost_value[mask]
        )
    return {
        "reward_explained_variance": finite_numeric_scalar(
            "reward_explained_variance", reward_ev
        ),
        "hazard_cost_explained_variance": finite_numeric_scalar(
            "hazard_cost_explained_variance", cost_ev
        ),
        "reward_target_variance": finite_numeric_scalar(
            "reward_target_variance", reward_variance
        ),
        "hazard_cost_target_variance": finite_numeric_scalar(
            "hazard_cost_target_variance", cost_variance
        ),
        "reward_target_variance_degenerate": reward_degenerate,
        "hazard_cost_target_variance_degenerate": cost_degenerate,
    }


# ---------------------------------------------------------------------------
# Greedy model policy wrapper for the Stage 24 diagnostics episode runner
# ---------------------------------------------------------------------------


def greedy_model_policy_fn(model: RecurrentMAPPOActorCritic) -> PolicyFn:
    """Wrap ``model`` as a deterministic-argmax PolicyFn for the probe runner.

    Per-episode recurrent state lives in the PolicyState dict the runner owns
    (key ``model_history_state``), so each episode's fresh state dict starts a
    fresh GRU history. Every call also runs a zeroed-reveal COUNTERFACTUAL
    forward from the SAME prior history: if any agent's counterfactual
    movement argmax differs from its actual movement argmax while that agent's
    reveal tensor is non-zero, the causal reveal->movement flag
    ``used_revealed_information_to_adapt_movement`` is set in the state (the
    runner only ever reads it from there). Callers are responsible for model
    eval-mode discipline (``run_readiness_probe`` sets it under try/finally).
    """

    if not isinstance(model, RecurrentMAPPOActorCritic):
        raise TypeError("model must be a RecurrentMAPPOActorCritic")
    config = model.config

    def _policy(
        observations: Mapping[str, Mapping[str, object]],
        environment: Any,
        state: dict[str, Any],
        _rng: Any,
    ) -> dict[str, dict[str, int]]:
        agent_names = tuple(environment.agents)
        actor_observation = actor_observation_batch(
            observations,
            config,
            dtype=config.dtype,
            device=config.device,
            agent_names=agent_names,
        )
        revealed_information = revealed_information_batch(
            observations,
            config,
            dtype=config.dtype,
            device=config.device,
            agent_names=agent_names,
        )
        history_state = state.get(_HISTORY_STATE_KEY)
        with torch.no_grad():
            sensing_logits, movement_logits, next_state = model.forward_actor(
                ActorInput(actor_observation, revealed_information, history_state)
            )
            sensing_actions = torch.argmax(sensing_logits, dim=-1)
            movement_actions = torch.argmax(movement_logits, dim=-1)
            # Counterfactual: same observation and SAME prior history, but the
            # reveal channel zeroed. A movement change attributable to a
            # non-zero reveal is the causal reveal->movement signal.
            _, counterfactual_movement_logits, _ = model.forward_actor(
                ActorInput(
                    actor_observation,
                    torch.zeros_like(revealed_information),
                    history_state,
                )
            )
            counterfactual_movement_actions = torch.argmax(
                counterfactual_movement_logits, dim=-1
            )
        state[_HISTORY_STATE_KEY] = next_state.detach()

        actions: dict[str, dict[str, int]] = {}
        for index, agent in enumerate(agent_names):
            movement = int(movement_actions[index, 0].item())
            counterfactual_movement = int(
                counterfactual_movement_actions[index, 0].item()
            )
            reveal_is_nonzero = bool(
                (revealed_information[index] != 0).any().item()
            )
            if reveal_is_nonzero and movement != counterfactual_movement:
                state[_ADAPT_FLAG_KEY] = True
            actions[agent] = {
                "sensing_action": int(sensing_actions[index, 0].item()),
                "movement_action": movement,
            }
        return actions

    return _policy


# ---------------------------------------------------------------------------
# Readiness probe (scenario surface + training/readiness variant surfaces)
# ---------------------------------------------------------------------------


def _assert_probe_variant_not_held_out(variant: Any) -> None:
    """Hard I-3 guard: the held-out layout must never reach the probe runner."""

    if variant.group == _HELD_OUT_GROUP:
        raise RuntimeError(
            "held-out hazard-layout variant must never reach the Stage 25 "
            "probe runner; its hidden layout must remain unseen"
        )


def _surface_verdict(comparisons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Compact per-surface verdict for the learned record vs the scripted baselines.

    Reuses the attached pairwise comparator fields (single source of the
    strict-inequality semantics) rather than re-deriving the comparisons.
    """

    learned = comparisons[_LEARNED_POLICY_NAME]
    learned_success = bool(learned["team_success"])
    learned_adapts_movement = bool(learned[_ADAPT_FLAG_KEY])
    beats_no_sense_on_hazard = bool(learned["reduced_hazard_exposure_vs_no_sense"])
    cheaper_than_always_sense = bool(learned["lower_sensing_cost_than_always_sense"])
    return {
        "learned_success": learned_success,
        "learned_hazard_entries": int(learned["hazard_entry_count"]),
        "learned_sensing_cost": float(learned["sensing_cost_sum"]),
        "learned_sensing_rate": float(learned["sensing_rate"]),
        "learned_adapts_movement": learned_adapts_movement,
        "beats_no_sense_on_hazard": beats_no_sense_on_hazard,
        "cheaper_than_always_sense": cheaper_than_always_sense,
        "preserved_success": bool(learned["preserved_task_success_vs_no_sense"]),
        "readiness_pattern_met": bool(
            learned_success
            and beats_no_sense_on_hazard
            and cheaper_than_always_sense
            and learned_adapts_movement
        ),
    }


def _run_fork_readiness_probe(
    model: RecurrentMAPPOActorCritic,
    probe_seed: int,
    probe_scenarios: tuple[str, ...],
    scenario_comparators: Mapping[str, Any],
) -> dict[str, Any]:
    """DR-D1/L1-b probe on the risk-fork family (owed SHIFT-9 repoint).

    Each fork scenario is a Stage 23-A CATALOG scenario, so its surface reuses the
    scripted-comparator + learned-greedy catalog path (never the
    Stage24AHazardLayoutVariant path). Only the training + curriculum + readiness
    fork scenarios are admissible (the curriculum ids per the countersigned
    DR-READINESS-CURRICULUM §5.5); every name is hard-checked against
    ``fork_hazard_layout_families`` + ``fork_curriculum_scenarios`` so the held-out
    fork family (gates 5,6 — in neither registry) can never be instantiated here
    (I-3). The return shape mirrors the legacy probe (``scenario`` / ``variants`` /
    ``all_training_readiness_surfaces_met``) so downstream logging is unchanged; the
    fork surfaces are keyed by scenario name under ``variants``.
    """

    if not isinstance(probe_scenarios, tuple) or not probe_scenarios:
        raise TypeError("probe_scenarios must be a non-empty tuple of scenario names")
    families = fork_hazard_layout_families()
    admissible = (
        frozenset(families["training"])
        | frozenset(families["readiness"])
        | frozenset(fork_curriculum_scenarios())
    )
    for scenario_name in probe_scenarios:
        if scenario_name not in admissible:
            # I-3 hard guard: only training+curriculum+readiness fork surfaces
            # are probeable.
            raise RuntimeError(
                "fork readiness probe admits only the training + curriculum + "
                f"readiness fork scenarios; refused {scenario_name!r} (held-out "
                "/ unknown fork layouts must remain unseen)"
            )

    was_training = model.training
    model.eval()
    try:
        surfaces: dict[str, dict[str, Any]] = {}
        for scenario_name in probe_scenarios:
            comparisons: dict[str, dict[str, Any]] = {}
            for policy_name in _SCENARIO_COMPARATOR_ORDER:
                comparisons[policy_name] = scenario_comparators[policy_name](
                    scenario_name, seed=probe_seed
                )
            comparisons[_LEARNED_POLICY_NAME] = _run_policy(
                Stage24ComparatorConfig(scenario_name, seed=probe_seed),
                policy_name=_LEARNED_POLICY_NAME,
                policy=greedy_model_policy_fn(model),
                diagnostic_only=False,
                uses_hidden_hazards_for_decision=False,
            )
            _attach_pairwise_comparator_fields(comparisons)
            surfaces[scenario_name] = {
                "comparisons": comparisons,
                "verdict": _surface_verdict(comparisons),
            }
    finally:
        model.train(was_training)

    all_met = all(
        surface["verdict"]["readiness_pattern_met"] for surface in surfaces.values()
    )
    return {
        "probe_seed": probe_seed,
        "scenario": surfaces[probe_scenarios[0]],
        "variants": surfaces,
        "all_training_readiness_surfaces_met": bool(all_met),
        "fork_family": True,
    }


def run_readiness_probe(
    model: RecurrentMAPPOActorCritic,
    *,
    probe_seed: int,
    update_config: Stage22UpdateConfig,
    probe_scenarios: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Grade the current model against the scripted comparators (in memory).

    LEGACY surfaces (``probe_scenarios is None``): the ``risk_gate_hidden_hazard``
    catalog scenario (all five scripted comparators plus the learned greedy
    policy) and the ``training`` and ``readiness`` hazard-layout variants. The
    ``held_out`` variant is NEVER instantiated or run here.

    DR-D1/L1-b FORK surfaces (``probe_scenarios`` supplied): the given risk-fork
    catalog scenarios (each a full scripted-comparator + learned-greedy surface).
    Only the training + curriculum + readiness fork scenarios are admissible
    (curriculum per the countersigned DR-READINESS-CURRICULUM); the held-out fork
    family (gates 5,6) is NOT in ``fork_hazard_layout_families()`` nor in
    ``fork_curriculum_scenarios()`` and any name outside the admissible fork set
    is rejected (I-3). ``update_config`` is validated and reserved for probe
    variants that grade against update statistics; v0 probes are comparator-only.
    """

    if not isinstance(model, RecurrentMAPPOActorCritic):
        raise TypeError("model must be a RecurrentMAPPOActorCritic")
    require_positive_seed("probe_seed", probe_seed)
    if not isinstance(update_config, Stage22UpdateConfig):
        raise TypeError("update_config must be a Stage22UpdateConfig")

    scenario_comparators: dict[str, Any] = {
        "no_sense_shortest_path": no_sense_shortest_path,
        "always_sense_shortest_path": always_sense_shortest_path,
        "random_policy": random_policy,
        "risk_aware_oracle_or_heuristic": risk_aware_oracle_or_heuristic,
        "selective_sense_risk_aware": selective_sense_risk_aware,
    }

    if probe_scenarios is not None:
        return _run_fork_readiness_probe(
            model, probe_seed, probe_scenarios, scenario_comparators
        )

    was_training = model.training
    model.eval()
    try:
        # (a) Catalog-scenario surface: five scripted comparators via their
        # public entry points, plus the learned greedy record, then the shared
        # pairwise fields so the learned record is graded in place.
        comparisons: dict[str, dict[str, Any]] = {}
        for policy_name in _SCENARIO_COMPARATOR_ORDER:
            comparisons[policy_name] = scenario_comparators[policy_name](
                _PROBE_SCENARIO, seed=probe_seed
            )
        comparisons[_LEARNED_POLICY_NAME] = _run_policy(
            Stage24ComparatorConfig(_PROBE_SCENARIO, seed=probe_seed),
            policy_name=_LEARNED_POLICY_NAME,
            policy=greedy_model_policy_fn(model),
            diagnostic_only=False,
            uses_hidden_hazards_for_decision=False,
        )
        _attach_pairwise_comparator_fields(comparisons)
        scenario_surface = {
            "comparisons": comparisons,
            "verdict": _surface_verdict(comparisons),
        }

        # (b) Hazard-layout variant surfaces: training and readiness groups
        # only. The held-out variant is filtered FIRST and additionally hard
        # guarded — it must never be instantiated by this probe.
        variant_surfaces: dict[str, dict[str, Any]] = {}
        for variant in stage24a_hazard_layout_variants().values():
            if variant.group not in _VARIANT_PROBE_GROUPS:
                continue
            _assert_probe_variant_not_held_out(variant)
            environment_config = _stage24a_variant_environment_config(
                variant, seed=probe_seed
            )
            variant_comparisons: dict[str, dict[str, Any]] = {}
            for policy_name, policy in _VARIANT_COMPARATOR_POLICIES:
                variant_comparisons[policy_name] = _run_policy_on_environment(
                    environment_config,
                    seed=probe_seed,
                    policy_name=policy_name,
                    policy=policy,
                    diagnostic_only=False,
                    uses_hidden_hazards_for_decision=False,
                    variant_name=variant.name,
                    public_risk_zone_cells=variant.public_risk_zone_cells,
                )
            variant_comparisons[_LEARNED_POLICY_NAME] = _run_policy_on_environment(
                environment_config,
                seed=probe_seed,
                policy_name=_LEARNED_POLICY_NAME,
                policy=greedy_model_policy_fn(model),
                diagnostic_only=False,
                uses_hidden_hazards_for_decision=False,
                variant_name=variant.name,
                public_risk_zone_cells=variant.public_risk_zone_cells,
            )
            _attach_pairwise_comparator_fields(variant_comparisons)
            variant_surfaces[variant.name] = {
                "comparisons": variant_comparisons,
                "verdict": _surface_verdict(variant_comparisons),
            }
    finally:
        model.train(was_training)

    all_surfaces_met = scenario_surface["verdict"]["readiness_pattern_met"] and all(
        surface["verdict"]["readiness_pattern_met"]
        for surface in variant_surfaces.values()
    )
    return {
        "probe_seed": probe_seed,
        "scenario": scenario_surface,
        "variants": variant_surfaces,
        "all_training_readiness_surfaces_met": bool(all_surfaces_met),
    }


# ---------------------------------------------------------------------------
# Baseline training driver
# ---------------------------------------------------------------------------


def _git_head(repo_root: Path) -> str | None:
    """Return the current git HEAD hash, or None if it cannot be determined."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except Exception:
        return None
    if completed.returncode != 0:
        return None
    head = completed.stdout.strip()
    return head or None


def _config_payload(
    config: BaselineRunConfig,
    update_config: Stage22UpdateConfig,
) -> dict[str, Any]:
    """Build the JSON-compatible manifest ``config`` block."""

    payload = dataclasses.asdict(config)
    payload["result_parent"] = (
        None if config.result_parent is None else str(config.result_parent)
    )
    payload["scenario_names"] = list(config.scenario_names)
    payload["stage22_update_config"] = dataclasses.asdict(update_config)
    payload["torch_version"] = str(torch.__version__)
    return payload


def _curve_record(
    *,
    round_index: int,
    round_seed: int,
    wall_time_s: float,
    episode_records: tuple[dict[str, Any], ...],
    update_summary: Mapping[str, Any],
    explained_variance: Mapping[str, float | bool],
    entropy_floor: float,
) -> dict[str, Any]:
    """Build one training-curve JSONL record for a completed round."""

    episode_count = len(episode_records)
    success_count = sum(1 for record in episode_records if record["team_success"])
    sensing_entropy = float(update_summary["sensing_entropy"])
    movement_entropy = float(update_summary["movement_entropy"])

    def _mean(field_name: str) -> float:
        return float(
            sum(float(record[field_name]) for record in episode_records)
        ) / float(episode_count)

    def _int_sum(field_name: str) -> int:
        return int(sum(int(record[field_name]) for record in episode_records))

    return {
        "round_index": round_index,
        "round_seed": round_seed,
        "wall_time_s": float(wall_time_s),
        "success_count": int(success_count),
        "success_rate": float(success_count) / float(episode_count),
        "mean_step_count": _mean("step_count"),
        "task_reward_sum_mean": _mean("task_reward_sum"),
        "hazard_cost_sum_mean": _mean("hazard_cost_sum"),
        "sensing_cost_sum_mean": _mean("sensing_cost_sum"),
        "sensing_rate_mean": _mean("sensing_rate"),
        "total_sensing_actions": _int_sum("total_sensing_actions"),
        "total_movement_actions": _int_sum("total_movement_actions"),
        "hazard_entry_count": _int_sum("hazard_entry_count"),
        "revealed_hazard_count": _int_sum("revealed_hazard_count"),
        "lagrange_multiplier_before": float(update_summary["lagrange_multiplier_before"]),
        "lagrange_multiplier_after": float(update_summary["lagrange_multiplier_after"]),
        "observed_hazard_cost": float(update_summary["observed_hazard_cost"]),
        "policy_loss": float(update_summary["policy_loss"]),
        "reward_value_loss": float(update_summary["reward_value_loss"]),
        "hazard_cost_value_loss": float(update_summary["hazard_cost_value_loss"]),
        "sensing_entropy": sensing_entropy,
        "movement_entropy": movement_entropy,
        "total_loss": float(update_summary["total_loss"]),
        "parameter_delta_l1": float(update_summary["parameter_delta_l1"]),
        "raw_reward_advantage_mean": float(update_summary["raw_reward_advantage_mean"]),
        "raw_hazard_cost_advantage_mean": float(
            update_summary["raw_hazard_cost_advantage_mean"]
        ),
        **dict(explained_variance),
        "entropy_collapse_flag": bool(
            sensing_entropy + movement_entropy < entropy_floor
        ),
        **harness_boundary_flags(),
    }


def run_baseline_training(config: BaselineRunConfig) -> Path:
    """Run the Stage 25 harness v0 baseline training loop; return the run directory.

    Loops ``collect_stage23c_development_rollout`` ->
    ``stage22_ppo_lagrangian_update`` with a single persistent Adam optimizer
    and a threaded Lagrange multiplier (the D-3 cadence is owned here:
    ``result.lagrange_multiplier`` feeds the next round). Writes
    ``RUN_MANIFEST.json`` (rewritten on heartbeat / kill / crash /
    completion), ``training_curve.jsonl`` (one record per round),
    ``episode_log.jsonl`` (one record per episode per round), and
    ``probe_log.jsonl`` (periodic plus always-after-final-round readiness
    probes). The free-text run inputs are scanned for unsafe boundary wording
    BEFORE the run directory is created, so a rejected input burns no run id
    and leaves no empty directory. Kill criteria: any exception after the
    initial manifest write (including model/optimizer construction) marks the
    manifest ``crashed`` and re-raises; a Lagrange multiplier above
    ``lambda_kill_bound`` marks it ``killed`` and stops; entropy collapse is
    a soft per-round flag only.
    """

    if not isinstance(config, BaselineRunConfig):
        raise TypeError("config must be a BaselineRunConfig")

    # Pre-directory input scan: unsafe wording in the caller-supplied
    # free-text inputs must fail BEFORE mkdir — no burned run ids, no
    # leftover empty directories. Same allowed structural key set as every
    # other harness scan call site.
    reject_unsafe_boundary_values(
        {
            "run_id": config.run_id,
            "scenario_names": list(config.scenario_names),
            "implementing_dr_ids": [],
            "tier": config.tier,
        },
        "run inputs",
        allowed_structural_keys=harness_allowed_structural_keys(),
    )

    repo_root = active_root(__file__, 3)
    run_dir = resolve_run_directory(config.run_id, config.result_parent)

    # Everything computable without touching the filesystem is built before
    # mkdir so the uncovered window between directory creation and the first
    # manifest write stays minimal.
    update_config = default_stage22_update_config()
    runtime_record = stage24a_runtime_reproducibility_record(
        seed_values=(config.seed,), device="cpu", dtype="float32"
    )
    manifest_base: dict[str, Any] = {
        "run_id": config.run_id,
        "tier": config.tier,
        "seeds": (config.seed,),
        "implementing_dr_ids": (),
        "config_payload": _config_payload(config, update_config),
        "git_head": _git_head(repo_root),
        "launch_timestamp_utc": utc_timestamp(),
        "pid": os.getpid(),
    }

    def _write_manifest(status: str, extra_fields: dict[str, Any]) -> None:
        payload = build_run_manifest_payload(
            status=status,
            extra={"runtime_record": runtime_record, **extra_fields},
            **manifest_base,
        )
        write_run_manifest(run_dir, payload)

    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        _write_manifest("running", {})
    except Exception:
        # A failed initial manifest write must not leave behind the
        # just-created empty run directory (a burned run id); a non-empty
        # directory is deliberately left in place for forensics.
        try:
            run_dir.rmdir()
        except OSError:
            pass
        raise

    curve_path = run_dir / "training_curve.jsonl"
    episode_path = run_dir / "episode_log.jsonl"
    probe_path = run_dir / "probe_log.jsonl"

    killed = False
    last_completed_round = -1
    # Everything after the initial 'running' manifest write — including
    # model/optimizer/multiplier construction — runs inside the crash-marking
    # try, so any pre-loop failure still rewrites the manifest as 'crashed'.
    try:
        core_config = default_stage23_core_config()
        model = RecurrentMAPPOActorCritic(core_config)
        # One persistent Adam built ONCE: passing optimizer=None to the
        # update would silently rebuild a fresh Adam (and reset its moments)
        # every round.
        optimizer = torch.optim.Adam(
            model.parameters(), lr=update_config.learning_rate
        )
        multiplier = LagrangeMultiplier(
            update_config.lagrange.initial_multiplier,
            update_config.lagrange.learning_rate,
        )

        for round_index in range(config.update_rounds):
            round_start = time.perf_counter()
            round_seed = config.seed + round_index * _ROUND_SEED_STRIDE
            rollout_config = Stage23CRolloutConfig(
                max_episodes=config.episodes_per_round,
                steps_per_episode=config.steps_per_episode,
                scenario_names=config.scenario_names,
                sample_actions=config.sample_actions,
                seed=round_seed,
            )
            collected = collect_stage23c_development_rollout(model, rollout_config)
            explained_variance = explained_variance_from_batch(
                collected.batch, update_config
            )
            result = stage22_ppo_lagrangian_update(
                model,
                collected.batch,
                update_config,
                optimizer=optimizer,
                lagrange_multiplier=multiplier,
            )
            multiplier = result.lagrange_multiplier
            wall_time_s = time.perf_counter() - round_start

            append_jsonl_record(
                curve_path,
                _curve_record(
                    round_index=round_index,
                    round_seed=round_seed,
                    wall_time_s=wall_time_s,
                    episode_records=collected.episode_records,
                    update_summary=result.summary,
                    explained_variance=explained_variance,
                    entropy_floor=config.entropy_floor,
                ),
            )
            for episode_record in collected.episode_records:
                append_jsonl_record(
                    episode_path,
                    {
                        **episode_record,
                        "round_index": round_index,
                        **harness_boundary_flags(),
                    },
                )

            periodic_probe_due = (
                config.probe_every > 0
                and (round_index + 1) % config.probe_every == 0
            )
            final_round = round_index == config.update_rounds - 1
            if periodic_probe_due or final_round:
                probe = run_readiness_probe(
                    model,
                    probe_seed=config.probe_seed,
                    update_config=update_config,
                )
                append_jsonl_record(
                    probe_path,
                    {
                        "round_index": round_index,
                        **probe,
                        **harness_boundary_flags(),
                    },
                )
            last_completed_round = round_index

            if multiplier.value > config.lambda_kill_bound:
                _write_manifest(
                    "killed",
                    {
                        "kill_reason": "lambda exceeded kill bound",
                        "last_completed_round": round_index,
                    },
                )
                killed = True
                break

            if (round_index + 1) % config.heartbeat_every == 0:
                _write_manifest(
                    "running",
                    {
                        "last_completed_round": round_index,
                        "heartbeat_timestamp_utc": utc_timestamp(),
                    },
                )
    except Exception as exc:
        # Build the reason under its own guard: a hostile __str__ (or even a
        # hostile metaclass __name__) must never prevent the crashed-manifest
        # write; the type name is resolved once so no hostile surface is
        # touched twice, and the scanner fallback below reuses it.
        try:
            safe_type_name = type(exc).__name__
            if not isinstance(safe_type_name, str) or not safe_type_name:
                safe_type_name = "unrepresentable-error"
        except Exception:
            safe_type_name = "unrepresentable-error"
        try:
            kill_reason = f"{safe_type_name}: {exc}"
        except Exception:
            kill_reason = safe_type_name
        crash_fields = {"last_completed_round": last_completed_round}
        # The entire crashed-manifest write is itself guarded so the ORIGINAL
        # exception always propagates from this handler — never a foreign one
        # raised while writing the crash record.
        try:
            try:
                _write_manifest(
                    "crashed", {"kill_reason": kill_reason, **crash_fields}
                )
            except ValueError:
                # The exception wording tripped the unsafe-wording scanner;
                # fall back to the pre-resolved exception type name only.
                _write_manifest(
                    "crashed",
                    {"kill_reason": safe_type_name, **crash_fields},
                )
        except Exception:
            pass
        raise

    if not killed:
        _write_manifest(
            "complete",
            {
                "last_completed_round": config.update_rounds - 1,
                "finish_timestamp_utc": utc_timestamp(),
            },
        )
    return run_dir


# ---------------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point for the Stage 25 harness v0 baseline driver."""

    parser = argparse.ArgumentParser(
        prog="baseline_driver",
        description=(
            "Stage 25 harness v0 baseline training driver (development-only; "
            "drives the untouched Stage 22 bounded update over Stage 23-C "
            "collection)."
        ),
    )
    parser.add_argument("--run-id", required=True, help="unique run id (used exactly once)")
    parser.add_argument("--seed", type=int, required=True, help="positive base seed")
    parser.add_argument(
        "--update-rounds", type=int, required=True, help="collect/update rounds (1..20000)"
    )
    parser.add_argument("--episodes-per-round", type=int, default=4, help="2..8")
    parser.add_argument(
        "--steps-per-episode", type=int, default=None, help="None (omit) or 1..32"
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="repeatable Stage 23-A scenario name (default risk_gate_hidden_hazard)",
    )
    parser.add_argument(
        "--greedy-collection",
        action="store_true",
        help="collect with deterministic argmax actions instead of sampling",
    )
    parser.add_argument("--probe-every", type=int, default=50, help="0 disables periodic probes")
    parser.add_argument("--probe-seed", type=int, default=9001)
    parser.add_argument("--tier", default="T1", choices=list(_VALID_TIERS))
    parser.add_argument("--result-parent", default=None)
    args = parser.parse_args(argv)

    scenario_names = (
        tuple(args.scenario) if args.scenario else ("risk_gate_hidden_hazard",)
    )
    run_config = BaselineRunConfig(
        run_id=args.run_id,
        seed=args.seed,
        update_rounds=args.update_rounds,
        episodes_per_round=args.episodes_per_round,
        steps_per_episode=args.steps_per_episode,
        scenario_names=scenario_names,
        sample_actions=not args.greedy_collection,
        probe_every=args.probe_every,
        probe_seed=args.probe_seed,
        tier=args.tier,
        result_parent=args.result_parent,
    )
    run_dir = run_baseline_training(run_config)
    print(f"BASELINE RUN COMPLETE: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""C1 spatial-selectivity diagnostic harness (SHIFT 6, DR-independent, scratchpad-only).

PURPOSE
-------
Grade claim component C1 ("acquire costly information SELECTIVELY") as a genuine
SPATIAL-SELECTIVITY CONTRAST, not merely "sensing_rate > 0". A never-sensing policy
FAILS by default; an always-sensing policy FAILS on zero selectivity. The instrument is
diagnostic-only: it writes no governed artifacts, pins no protocol threshold, reads no
held-out layout (I-3), and decides nothing. It is validated FIRST against the scripted
comparators (selective_sense must pass, no_sense/always_sense must fail) so the measurement
is trusted before any learned T1-D5 policy is graded.

The "relevant" region is the public risk zone (rows 1-3 x cols 1-3 for the gate family) --
the region the agent can reason about publicly and where the hidden hazard could be.
Selectivity means: sense MORE inside/near that region than outside it.

WHAT IT COMPUTES (per policy, per surface, pooled over both agents x all steps):
  - Delta_strict  = P(sense | in-zone)        - P(sense | out-of-zone)
  - Delta_adj     = P(sense | in-or-adjacent) - P(sense | far)          (dist<=1 vs >1)
  - r_pb          = point-biserial corr( sense in {0,1}, proximity = -dist_to_zone )
  - permutation p (one-sided, sense-labels shuffled) for Delta_strict / Delta_adj
  - degenerate guards: total_sense==0 -> FAIL_NEVER_SENSE;
                       total_sense==n  -> FAIL_ALWAYS_SENSE_ZERO_SELECTIVITY
  - C2 side-signal: fraction of episodes where the greedy do(reveal:=0) counterfactual
    flipped a movement (used_revealed_information_to_adapt_movement)

Surfaces graded: the Stage 24-A TRAINING and READINESS hazard-layout variants only. The
held-out variant is hard-guarded and never instantiated (I-3).

USAGE
  python c1_selectivity_harness.py --validate            # validate vs scripted comparators
  python c1_selectivity_harness.py --grade <state_.pt>   # grade one learned policy
  python c1_selectivity_harness.py --grade-run t1_d5_s108 [t1_d5_s109 ...]  # by run id
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

# Repository root: walk up from THIS FILE to the directory containing src/raas_marl.
# PHASE 3: the hardcoded absolute-path fallback that used to follow this walk was
# DELETED. It named a private tree, so on the machine that tree existed the harness
# resolved into it even when executed from a clone -- a green run was not evidence
# about the clone. There is nothing safe to guess when the walk fails, so it raises.
REPO_ROOT = Path(__file__).resolve()
while REPO_ROOT.parent != REPO_ROOT and not (REPO_ROOT / "src" / "raas_marl").is_dir():
    REPO_ROOT = REPO_ROOT.parent
if not (REPO_ROOT / "src" / "raas_marl").is_dir():
    raise RuntimeError(
        "could not locate the repository root: no ancestor directory of "
        f"{Path(__file__).resolve()} contains 'src/raas_marl'. Run this "
        "instrument from inside a checkout of the artifact repository."
    )
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from raas_marl.environments.active_sensing.grid_environment import (  # noqa: E402
    RiskAwareActiveSensingGridEnvironment,
    Stage23EnvironmentConfig,
    stage23_scenario_catalog,
)
from raas_marl.environments.active_sensing.stage24_diagnostics import (  # noqa: E402
    STAGE23_AGENT_NAMES,
    _always_sense_policy,
    _fork_public_risk_zone_cells,
    _no_sense_policy,
    _position_from_observation,
    _selective_sense_policy,
    _stage24a_variant_environment_config,
    _update_revealed_memory,
    fork_curriculum_scenarios,
    fork_hazard_layout_families,
    stage24a_hazard_layout_variants,
)
from raas_marl.environments.active_sensing.tensor_adapter import (  # noqa: E402
    default_stage23_core_config,
)
from raas_marl.final_training.baseline_driver import greedy_model_policy_fn  # noqa: E402
from raas_marl.mappo_lagrangian.model import RecurrentMAPPOActorCritic  # noqa: E402

# Field names used in the action dicts (kept literal to avoid over-coupling).
SENSING_FIELD = "sensing_action"
MOVEMENT_FIELD = "movement_action"
_ADAPT_FLAG_KEY = "used_revealed_information_to_adapt_movement"

PROBE_SEED = 9001  # the training probe seed (I-2/I-3: not an eval seed; training/readiness only)
_PERM_ITERS = 4000
_PERM_RNG_SEED = 12345  # NOT the reserved 0; fixed for reproducible permutation p-values

# AM-2 (SHIFT-11 audit): a per-surface selectivity CEILING for the tightened witness --
# a witness surface must sense at MOST this fraction of decisions, so "acquire
# SELECTIVELY" is enforced (0<rate<1 alone lets a rate-0.85 near-always policy witness).
# RECOMMENDED DEFAULT ONLY -- the numeric value is PINNED at the Protocol-v1 lock, before
# inspecting learned numbers. The scripted selective_sense exemplar senses 8-25% (well
# below), and always_sense is rate 1.0 (well above), so 0.5 separates them cleanly.
SELECTIVITY_CEILING = 0.5

# AM-3 (SHIFT-11 audit finding F7 reconciliation) + SHIFT-14 NEW WATCH (positionally-induced
# vs causal sensing): a DIAGNOSTIC-ONLY floor for the positionally-induced-sensing annotation.
# The DR-D1/L1-b positioning densifier (Phi_dz = w*1[cell in zone]) can induce a policy to
# sense IN-ZONE because it is parked where sensing is cheap to repeat -- NOT because it learned
# sensing is protective. The guard is the causal witness (own do(reveal:=0) ablation): high
# in-zone sensing rate with NO causal witness = positionally induced, not causal. This floor is
# the minimum in-zone sense rate (on a SUCCEEDING surface) that makes the pattern worth flagging.
# DIAGNOSTIC ONLY: this annotation NEVER changes a C1 verdict -- it only annotates an already-
# failing SENSES_NOT_CAUSAL / CAUSAL_BUT_NOT_SELECTIVE, so it is not a protocol certification
# threshold. The value is recorded as a RECOMMENDED default (pin-at-lock, like SELECTIVITY_CEILING)
# for cleanliness; the causal witness -- not this rate -- is the firewall (a genuine causal pass
# has >=1 witness and is therefore never flagged).
POSITIONALLY_INDUCED_INZONE_FLOOR = 0.10


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------
def _nearest_zone_distance(pos: tuple[int, int], zone: frozenset[tuple[int, int]]) -> int:
    """Manhattan distance from ``pos`` to the nearest risk-zone cell (0 if inside)."""
    if pos in zone:
        return 0
    return min(abs(pos[0] - c[0]) + abs(pos[1] - c[1]) for c in zone)


# ---------------------------------------------------------------------------
# rollout: one deterministic episode, recording per-agent per-step decisions
# ---------------------------------------------------------------------------
def _blind_reveal_observations(
    observations: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Return a copy of ``observations`` with ``revealed_local_hazards`` emptied.

    The single reveal channel for BOTH policy types is
    ``actor_visible.revealed_local_hazards`` -- the scripted policies read it via
    ``_update_revealed_memory`` and the model reads it via
    ``revealed_information_from_stage23`` (which yields an all-zero reveal tensor
    when the list is empty). Emptying it therefore executes a full-trajectory
    do(reveal:=0) ABLATION of the graded policy's OWN behaviour, without touching
    the environment's internal reveal state (only the copy the policy sees). Used
    to certify that a policy's hazard avoidance CAUSALLY depends on the reveal
    (AM-1): a memorized/privileged-safe route is unchanged by blinding, so its
    own-blind hazard equals its executed hazard (no causal witness).
    """
    blinded: dict[str, dict[str, Any]] = {}
    for agent, observation in observations.items():
        actor_visible = dict(observation["actor_visible"])
        actor_visible["revealed_local_hazards"] = []
        new_obs = dict(observation)
        new_obs["actor_visible"] = actor_visible
        blinded[agent] = new_obs
    return blinded


def rollout_episode(
    env_config: Stage23EnvironmentConfig,
    *,
    seed: int,
    policy: Callable,
    zone: frozenset[tuple[int, int]],
    hazard_cells: tuple[tuple[int, int], ...] = (),
    max_steps: int = 64,
    blind_reveals: bool = False,
) -> dict[str, Any]:
    """Run one episode; return per-decision selectivity rows + an episode summary.

    Mirrors stage24_diagnostics._run_policy_on_environment's loop exactly (same
    state dict, same _update_revealed_memory hook), but additionally records, at
    each decision, (position, sense, dist_to_zone) per agent.

    ``blind_reveals`` (AM-1): if True, the reveal channel the policy SEES is
    zeroed at every step (a full-trajectory do(reveal:=0) ablation of the graded
    policy's OWN behaviour). The policy still chooses/pays for sensing, and the
    environment still reveals internally, but the policy never receives the
    reveal -- so its executed trajectory is its reveal-BLIND behaviour.
    """
    env = RiskAwareActiveSensingGridEnvironment(env_config)
    observations, _infos = env.reset(seed=seed)
    if blind_reveals:
        observations = _blind_reveal_observations(observations)
    rng = random.Random(seed)
    state: dict[str, Any] = {
        "revealed_hazards_by_agent": {a: set() for a in STAGE23_AGENT_NAMES},
        "sensed_positions_by_agent": {a: set() for a in STAGE23_AGENT_NAMES},
        "public_risk_zone_cells": set(zone),
        _ADAPT_FLAG_KEY: False,
    }
    rows: list[dict[str, Any]] = []
    steps = 0
    team_success = False
    hazard_entry = 0
    sensing_cost = 0.0
    hazard_cost = 0.0
    while env.agents and steps < max_steps:
        state["step_index"] = steps
        _update_revealed_memory(observations, state)
        # snapshot positions at decision time (before stepping)
        positions = {a: _position_from_observation(observations[a]) for a in env.agents}
        actions = policy(observations, env, state, rng)
        for a in env.agents:
            pos = positions[a]
            dist = _nearest_zone_distance(pos, zone)
            hdist = (
                min(abs(pos[0] - h[0]) + abs(pos[1] - h[1]) for h in hazard_cells)
                if hazard_cells
                else dist
            )
            rows.append(
                {
                    "agent": a,
                    "step": steps,
                    "row": pos[0],
                    "col": pos[1],
                    "dist_to_zone": dist,
                    "dist_to_hazard": hdist,  # privileged, offline grading only (synth spec)
                    "in_zone": dist == 0,
                    "in_or_adj": dist <= 1,
                    "near_hazard": hdist <= 1,  # relevant = within Manhattan-1 of the hidden hazard
                    "sense": 1 if int(actions[a][SENSING_FIELD]) != 0 else 0,
                    "move": int(actions[a][MOVEMENT_FIELD]),
                }
            )
        observations, _rewards, terminations, truncations, infos = env.step(actions)
        if blind_reveals:
            observations = _blind_reveal_observations(observations)
        steps += 1
        for info in infos.values():
            hazard_entry += int(bool(info["entered_hazard"]))
            sensing_cost += float(info["sensing_cost"])
            hazard_cost += float(info["hazard_cost"])
            team_success = team_success or bool(info["team_success"])
        if any(bool(v) for v in terminations.values()) or any(
            bool(v) for v in truncations.values()
        ):
            break
    return {
        "rows": rows,
        "steps": steps,
        "team_success": team_success,
        "hazard_entry_count": hazard_entry,
        "sensing_cost_sum": sensing_cost,
        "hazard_cost_sum": hazard_cost,
        "adapt_fired": bool(state.get(_ADAPT_FLAG_KEY, False)),
    }


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------
def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:  # no variance in one variable
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / math.sqrt(sxx * syy)


def _conditional_contrast(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    """P(sense | key==True) - P(sense | key==False) with counts."""
    rel = [r["sense"] for r in rows if r[key]]
    non = [r["sense"] for r in rows if not r[key]]
    p_rel = (sum(rel) / len(rel)) if rel else None
    p_non = (sum(non) / len(non)) if non else None
    delta = (p_rel - p_non) if (p_rel is not None and p_non is not None) else None
    return {
        "n_relevant": len(rel),
        "n_nonrelevant": len(non),
        "sense_relevant": sum(rel),
        "sense_nonrelevant": sum(non),
        "p_sense_relevant": p_rel,
        "p_sense_nonrelevant": p_non,
        "delta": delta,
    }


def _perm_p(rows: list[dict[str, Any]], key: str, observed_delta: float | None) -> float | None:
    """One-sided permutation p for delta>0: shuffle sense labels across all decisions."""
    if observed_delta is None:
        return None
    senses = [r["sense"] for r in rows]
    flags = [bool(r[key]) for r in rows]
    n_rel = sum(flags)
    n_non = len(flags) - n_rel
    if n_rel == 0 or n_non == 0:
        return None
    rng = random.Random(_PERM_RNG_SEED)
    ge = 0
    for _ in range(_PERM_ITERS):
        shuffled = senses[:]
        rng.shuffle(shuffled)
        s_rel = sum(s for s, f in zip(shuffled, flags) if f)
        s_non = sum(s for s, f in zip(shuffled, flags) if not f)
        d = s_rel / n_rel - s_non / n_non
        if d >= observed_delta - 1e-12:
            ge += 1
    return ge / _PERM_ITERS


def selectivity_stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute the full C1 selectivity statistic bundle + a verdict for one policy.

    PRIMARY axis = the near/far spatial contrast (relevant = near the risk region,
    dist_to_zone <= 1; nonrelevant = far, dist >= 2):
        Delta_near_far = P(sense | near) - P(sense | far).
    A SELECTIVE policy senses more near the hazard-uncertainty region than far from
    it. This axis is only ESTIMABLE when the trajectory visits far cells (n_far > 0).
    In the risk-gate task the start->goal path is short and stays entirely within
    dist<=1 of the gate, so the scripted comparators' trajectories have NO far cells
    and the spatial contrast is inestimable; those cases yield a flagged, WEAKER
    rate-only verdict (SELECTIVE_RATE_ONLY_LOCAL). A learned policy that wanders (per
    the T1-D4 collapse) DOES visit far cells and exercises the primary axis.

    SECONDARY reported signals: strict in-zone contrast (dist==0), point-biserial
    r_pb(sense, -dist), mean dist at sense vs non-sense (approach-vs-interior),
    permutation p on the near/far delta.
    """
    n = len(rows)
    total_sense = sum(r["sense"] for r in rows)
    strict = _conditional_contrast(rows, "in_zone")        # relevant = dist_to_zone==0 (synth PRIMARY)
    near_far = _conditional_contrast(rows, "in_or_adj")     # relevant = dist_to_zone<=1 (near) vs far
    near_hazard = _conditional_contrast(rows, "near_hazard")  # relevant = dist_to_hazard<=1 (synth SECONDARY basis)
    proximity = [-float(r["dist_to_zone"]) for r in rows]   # higher = closer to zone
    hazard_proximity = [-float(r["dist_to_hazard"]) for r in rows]  # higher = closer to hidden hazard
    senses = [float(r["sense"]) for r in rows]
    r_pb = _pearson(senses, proximity)
    r_pb_hazard = _pearson(senses, hazard_proximity)        # synth-recommended point-biserial
    perm_near_far = _perm_p(rows, "in_or_adj", near_far["delta"])
    sense_dists = [r["dist_to_zone"] for r in rows if r["sense"]]
    nonsense_dists = [r["dist_to_zone"] for r in rows if not r["sense"]]
    mean_dist_sense = (sum(sense_dists) / len(sense_dists)) if sense_dists else None
    mean_dist_nonsense = (sum(nonsense_dists) / len(nonsense_dists)) if nonsense_dists else None

    n_far = near_far["n_nonrelevant"]
    n_near = near_far["n_relevant"]
    far_axis_estimable = n_far > 0 and n_near > 0

    if n == 0:
        verdict = "FAIL_NO_DECISIONS"
    elif total_sense == 0:
        verdict = "FAIL_NEVER_SENSE"
    elif total_sense == n:
        verdict = "FAIL_INDISCRIMINATE_ALWAYS"
    elif far_axis_estimable:
        delta = near_far["delta"]
        if delta is not None and delta > 0:
            verdict = "SELECTIVE"          # senses more near the risk region than far
        elif delta is not None and delta == 0:
            verdict = "FAIL_UNIFORM_OVER_SPACE"
        else:
            verdict = "FAIL_ANTI_SELECTIVE"  # senses MORE far than near
    else:
        # No far cells visited: spatial contrast inestimable. Non-degenerate sparse
        # sensing is a WEAKER, flagged pass (cannot establish spatial selectivity).
        verdict = "SELECTIVE_RATE_ONLY_LOCAL"
    return {
        "n_decisions": n,
        "total_sense": total_sense,
        "overall_sense_rate": (total_sense / n) if n else 0.0,
        "contrast_near_far": near_far,
        "contrast_strict_in_zone": strict,          # = synth PRIMARY Delta (P(sense|in-zone)-P(sense|out))
        "contrast_near_hazard": near_hazard,        # relevant = dist_to_hazard<=1 (privileged, offline)
        "point_biserial_r_zone": r_pb,
        "point_biserial_r_hazard": r_pb_hazard,     # synth-recommended r_pb(sense, -dist_to_hazard)
        "perm_p_near_far": perm_near_far,
        "mean_dist_at_sense": mean_dist_sense,
        "mean_dist_at_nonsense": mean_dist_nonsense,
        "far_axis_estimable": far_axis_estimable,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# surface enumeration (training + readiness only; held-out hard-guarded, I-3)
# ---------------------------------------------------------------------------
_ALLOWED_GROUPS = frozenset({"training", "readiness"})


def _training_readiness_variants():
    out = []
    for variant in stage24a_hazard_layout_variants().values():
        if variant.group == "held_out":
            continue  # never instantiate the held-out layout (I-3)
        if variant.group not in _ALLOWED_GROUPS:
            continue
        out.append(variant)
    return out


def _assert_not_held_out(variant) -> None:
    if variant.group == "held_out":
        raise RuntimeError(
            "I-3: the held-out hazard layout must never reach the C1 harness"
        )


# ---------------------------------------------------------------------------
# DR-D1/L1 FORK repoint (SHIFT 9): the T1-D6 policies trained on the risk-fork
# family, so the C1 grade must enumerate THOSE surfaces, not the legacy risk_gate
# variants. The fork families are catalog scenarios (full geometry), so we build a
# lightweight variant record per scenario and an env config straight from the
# catalog. Held-out fork layouts (gates 5,6) are DEFERRED (I-3) and never appear in
# fork_hazard_layout_families(), so the training/readiness-only guard is preserved.
# ---------------------------------------------------------------------------
class _ForkVariant:
    """Minimal train/readiness fork surface record (mirrors the attrs grade_policy reads)."""

    def __init__(self, name, group, public_risk_zone_cells, hidden_hazard_cells):
        self.name = name
        self.group = group
        self.public_risk_zone_cells = public_risk_zone_cells
        self.hidden_hazard_cells = hidden_hazard_cells


def _fork_training_readiness_variants():
    # DR-READINESS-CURRICULUM (SHIFT 27) repoint: the enumeration widens to
    # training + CURRICULUM + readiness. The curriculum ids come from the
    # separate fork_curriculum_scenarios() registry (group label "curriculum")
    # so the pinned fork_hazard_layout_families() grade-surface registry is
    # untouched; held_out (gates 5,6) is in NEITHER registry (I-3). The
    # re-validation obligations (DR §5.7): --validate --fork green on this
    # extended enumeration AND per-surface verdicts on the four ORIGINAL
    # surfaces byte-identical on the T1-D13 s132/s133 knowns.
    catalog = stage23_scenario_catalog()
    families = fork_hazard_layout_families()  # {"training": (...), "readiness": (...)}
    grouped = (
        ("training", families.get("training", ())),
        ("curriculum", fork_curriculum_scenarios()),
        ("readiness", families.get("readiness", ())),
    )
    out = []
    for group, names in grouped:
        for name in names:  # held_out is intentionally absent (I-3)
            scenario = catalog[name]
            out.append(
                _ForkVariant(
                    name=name,
                    group=group,
                    public_risk_zone_cells=_fork_public_risk_zone_cells(scenario),
                    hidden_hazard_cells=scenario.hidden_hazard_cells,
                )
            )
    return out


def _fork_variant_environment_config(variant, *, seed):
    scenario = stage23_scenario_catalog()[variant.name]
    return Stage23EnvironmentConfig(
        scenario_name=scenario.name,
        width=scenario.width,
        height=scenario.height,
        seed=seed,
        start_positions=scenario.starts,
        goal_cells=scenario.goals,
        obstacle_cells=scenario.obstacles,
        hidden_hazard_cells=scenario.hidden_hazard_cells,
    )


def _select_family(family: str):
    """Return (variants, env_config_builder) for the requested surface family."""
    if family == "fork":
        return _fork_training_readiness_variants(), _fork_variant_environment_config
    if family == "legacy":
        return _training_readiness_variants(), _stage24a_variant_environment_config
    raise ValueError(f"unknown surface family: {family!r} (expected 'fork' or 'legacy')")


def composite_c1_verdict(grade: dict[str, Any]) -> dict[str, Any]:
    """The SHIFT-6 VALIDATED composite verdict (the spatial near/far contrast was
    empirically FALSIFIED). C1 = sparse + INFORMATIVE (+ adaptive for C2), with the
    degenerate guards. NOT the spatial `verdict`.

      FAIL_NEVER_SENSE           total_sense == 0
      FAIL_ALWAYS_SENSE          total_sense == n (indiscriminate)
      SELECTIVE_COMPOSITE        0<rate<1 AND informative (pooled hazard entries strictly
                                 below the policy-independent no_sense baseline on the SAME
                                 surfaces)
      SENSES_NOT_INFORMATIVE     senses but does not avoid the hazard no_sense hits
      NON_SELECTIVE              other
    """
    pooled = grade["pooled_selectivity"]
    n = pooled["n_decisions"]
    ts = pooled["total_sense"]
    rate = pooled["overall_sense_rate"]
    pol_haz = sum(s["hazard_entry_count"] for s in grade["per_surface"].values())
    base_haz = sum(s["no_sense_hazard_entry_count"] for s in grade["per_surface"].values())
    informative = pol_haz < base_haz
    if n == 0:
        verdict = "FAIL_NO_DECISIONS"
    elif ts == 0:
        verdict = "FAIL_NEVER_SENSE"
    elif ts == n:
        verdict = "FAIL_ALWAYS_SENSE"
    elif 0.0 < rate < 1.0 and informative:
        verdict = "SELECTIVE_COMPOSITE"
    elif 0.0 < rate < 1.0 and not informative:
        verdict = "SENSES_NOT_INFORMATIVE"
    else:
        verdict = "NON_SELECTIVE"
    return {
        "composite_verdict": verdict,
        "total_sense": ts,
        "n_decisions": n,
        "overall_sense_rate": rate,
        "policy_hazard_entries": pol_haz,
        "no_sense_hazard_entries": base_haz,
        "informative": informative,
        "c2_adapt_episode_fraction": grade.get("c2_adapt_episode_fraction"),
        "surfaces_with_success": grade.get("surfaces_with_success"),
        "surface_count": grade.get("surface_count"),
    }


def tightened_composite_c1_verdict(grade: dict[str, Any]) -> dict[str, Any]:
    """TIGHTENED C1 composite (SHIFT 11 — DR-Stage26 C1-metric revision groundwork).

    Supersedes ``composite_c1_verdict``'s POOLED ``informative`` flag, which is a LATENT
    WEAKNESS: ``policy_hazard_entries < no_sense_hazard_entries`` pooled across surfaces
    is TRUE for non-sensing reasons (split-gates / timeout-never-reaching-the-gate), so
    ``SELECTIVE_COMPOSITE`` there reduces to "sensed >= 1 time". The T1-D6 s113 FALSE
    POSITIVE is the concrete motivator: s113 senses 6/160, ALL on the two readiness
    surfaces where it WANDERS and FAILS (success=False, adapt_fired=False, 32-step
    timeout); its pooled ``informative`` came from never reaching the gate, NOT from
    sense->avoid. The pooled metric graded that SELECTIVE_COMPOSITE.

    This tightened verdict requires a PER-SURFACE CAUSAL chain on a SUCCEEDING surface.
    A surface WITNESSES genuine selective sensing iff ALL of:
        (1) team_success              -- the policy SOLVED the task on this surface
                                         (success-conditioning guard: kills the s113
                                         wander-and-fail false positive)
        (2) 0 < sense_rate <= CEIL    -- senses here, SPARSELY (AM-2: not never, and at
                                         most SELECTIVITY_CEILING of decisions, so a
                                         near-always-sense policy is NOT "selective")
        (3) adapt_fired               -- the do(reveal:=0) counterfactual FLIPPED a
                                         movement argmax on this SAME surface (behavioral
                                         sense->reveal->movement-adaptation)
        (4) own_blind_hazard_avoided>0-- AM-1: re-rolling THIS policy with its OWN reveal
                                         channel zeroed for the whole episode raises its
                                         hazard exposure (own_blind_hazard > executed), so
                                         the executed avoidance CAUSALLY depends on the
                                         reveal. (Replaces the earlier no_sense route
                                         comparison, which certified route-choice, not
                                         reveal-dependence -- audit finding F3. A
                                         memorized/privileged-safe route has own_blind ==
                                         executed -> avoided 0 -> no witness.)

    Verdicts:
        FAIL_NO_DECISIONS        n == 0
        FAIL_NEVER_SENSE         pooled total_sense == 0               (degenerate guard)
        FAIL_ALWAYS_SENSE        pooled rate == 1                      (degenerate guard)
        SELECTIVE_COMPOSITE      >= 1 witness surface
        CAUSAL_BUT_NOT_SELECTIVE >= 1 surface completes the success+adapt+own-avoid causal
                                 chain but senses ABOVE the selectivity ceiling (AM-4:
                                 distinguished from "no causal chain")
        SENSES_ON_FAILURE_ONLY   senses, but EVERY surface it senses on has team_success
                                 False -- the non-causal / task-failure guard (s113)
        SENSES_NOT_CAUSAL        senses on >= 1 SUCCEEDING surface but no surface completes
                                 the adapt->own-avoid causal chain (terminal else)

    EVAL-ABLATION HONESTY GATE (DR-D1/L1-b): this verdict is computed on the GREEDY
    policy rolled on the UNSHAPED ``RiskAwareActiveSensingGridEnvironment`` -- the
    DR-D1/L1-b training-time sensing CREDIT is ABSENT at eval by construction (the env
    carries no shaping; the greedy argmax is unaffected by a reward term). A
    SELECTIVE_COMPOSITE here is therefore a property of the LEARNED POLICY, not of the
    shaping reward -- the decisive ablation the credit's efficacy must survive.

    RELATIONSHIP TO ``composite_c1_verdict`` -- a RE-GROUNDING, not a strict subset. The
    shared degenerate guards (never-sense / always-sense) are identical. On the motivating
    FALSE-POSITIVE VECTOR this verdict is STRICTER: a pooled ``informative`` that comes
    from failing/timeout surfaces (s113) yields no per-surface causal witness here, so
    SELECTIVE_COMPOSITE is rejected; likewise a memorized/privileged-safe route with
    incidental sensing (audit finding F3) is rejected because its OWN reveal-ablation
    leaves hazard unchanged. It is NOT a global subset of the pooled verdict: the pooled
    ``informative`` sums a route-choice hazard contrast across surfaces, whereas the
    witness here is a per-surface CAUSAL own-ablation, so the two can disagree in either
    direction. That is intended: it isolates C1 (per-surface reveal-dependent selective
    sensing) from the pooled-hazard axis, which DR-Stage26 grades independently as C3. The
    tightening never produces a C1 pass without a genuine sense->reveal->adapt->own-avoid
    witness, so it does not weaken the C1 bar.
    """
    per = grade["per_surface"]
    pooled = grade["pooled_selectivity"]
    n = pooled["n_decisions"]
    ts = pooled["total_sense"]
    rate = pooled["overall_sense_rate"]

    witnesses: list[dict[str, Any]] = []
    causal_surfaces: list[dict[str, Any]] = []  # complete the causal chain, sparse or not
    surfaces_sensed_on = 0
    surfaces_sensed_on_with_success = 0
    # AM-3 (audit finding F7): the composite witness folds C1's necessary conditions
    # (success + sparse-sense + adapt + own-avoid) into one verdict. To keep C1
    # INDEPENDENTLY REPORTABLE, decompose the witness into its conjuncts per surface and
    # aggregate per-conjunct pass counts (each conjunct reported SEPARATELY -- this is the
    # genuine F7 discharge), so a reader sees WHICH conjunct drove the verdict (e.g.
    # success 4/4, senses 2/4, causal 0/4 => "senses but never causal"). Reporting only --
    # it does not change the verdict.
    per_component: list[dict[str, Any]] = []
    comp_counts = {
        "success": 0,
        "senses": 0,
        "within_ceiling": 0,
        "adapt": 0,
        "causal": 0,
        "witness": 0,
    }
    max_inzone_rate_on_success = 0.0  # SHIFT-14 positionally-induced signal
    for name, s in per.items():
        sel = s["selectivity"]
        sense_s = int(sel["total_sense"])
        rate_s = float(sel["overall_sense_rate"])
        success_s = bool(s["team_success"])
        adapt_s = bool(s["adapt_fired"])
        own_avoided_s = int(s["own_blind_hazard_avoided"])  # AM-1 causal conjunct
        # P(sense | in-zone) on this surface (None if the trajectory has no in-zone cell):
        strict = sel.get("contrast_strict_in_zone", {})
        inzone_rate_s = strict.get("p_sense_relevant")
        if sense_s >= 1:
            surfaces_sensed_on += 1
            if success_s:
                surfaces_sensed_on_with_success += 1
                if inzone_rate_s is not None:
                    max_inzone_rate_on_success = max(
                        max_inzone_rate_on_success, float(inzone_rate_s)
                    )
        # independent per-conjunct booleans (AM-3):
        c_senses = sense_s >= 1
        c_within_ceiling = c_senses and 0.0 < rate_s <= SELECTIVITY_CEILING  # AM-2
        c_causal = own_avoided_s > 0  # AM-1
        is_witness = success_s and c_within_ceiling and adapt_s and c_causal
        # the causal chain: success + senses + adapt + OWN reveal-ablation raises hazard
        causal_s = success_s and c_senses and adapt_s and c_causal
        failed_conjuncts: list[str] = []
        if not success_s:
            failed_conjuncts.append("success")
        if not c_senses:
            failed_conjuncts.append("senses")
        elif not c_within_ceiling:
            failed_conjuncts.append("within_ceiling")
        if not adapt_s:
            failed_conjuncts.append("adapt")
        if not c_causal:
            failed_conjuncts.append("causal")
        per_component.append(
            {
                "surface": name,
                "group": s["group"],
                "success": success_s,
                "senses": c_senses,
                "sense_rate": rate_s,
                "within_ceiling": c_within_ceiling,
                "adapt_fired": adapt_s,
                "own_blind_hazard_avoided": own_avoided_s,
                "causal": c_causal,
                "in_zone_sense_rate": inzone_rate_s,
                "is_witness": is_witness,
                "failed_conjuncts": failed_conjuncts,
            }
        )
        comp_counts["success"] += int(success_s)
        comp_counts["senses"] += int(c_senses)
        comp_counts["within_ceiling"] += int(c_within_ceiling)
        comp_counts["adapt"] += int(adapt_s)
        comp_counts["causal"] += int(c_causal)
        comp_counts["witness"] += int(is_witness)
        if causal_s:
            entry = {
                "surface": name,
                "group": s["group"],
                "sense": sense_s,
                "sense_rate": rate_s,
                "adapt_fired": adapt_s,
                "own_blind_hazard_avoided": own_avoided_s,
                "hazard_avoided_vs_no_sense": int(s["hazard_avoided_vs_no_sense"]),
            }
            causal_surfaces.append(entry)
            if 0.0 < rate_s <= SELECTIVITY_CEILING:  # AM-2 selectivity ceiling
                witnesses.append(entry)

    if n == 0:
        verdict = "FAIL_NO_DECISIONS"
    elif ts == 0:
        verdict = "FAIL_NEVER_SENSE"
    elif ts == n:
        verdict = "FAIL_ALWAYS_SENSE"
    elif witnesses:
        verdict = "SELECTIVE_COMPOSITE"
    elif causal_surfaces:  # AM-4: causal chain complete but over the selectivity ceiling
        verdict = "CAUSAL_BUT_NOT_SELECTIVE"
    elif surfaces_sensed_on_with_success == 0:
        verdict = "SENSES_ON_FAILURE_ONLY"
    else:
        verdict = "SENSES_NOT_CAUSAL"

    # AM-3 reconciliation DRIFT-GUARD. HONEST SCOPE (SHIFT-14 adversarial audit): the
    # re-derivation below is EQUIVALENT-BY-CONSTRUCTION to the verdict cascade above -- both
    # read the same per-surface booleans (comp_counts["witness"] == len(witnesses); the causal
    # branch reduces to `causal_surfaces` nonempty; the failure-only branch reuses the identical
    # `surfaces_sensed_on_with_success`). So `reconciles` is ALWAYS True on valid input -- it is
    # NOT an independent cross-check. The F7 "independently reportable" obligation is discharged
    # by per_component_report / component_pass_counts (each conjunct reported separately), NOT by
    # this block. This is a narrow TRIPWIRE only: if a FUTURE edit changes the per-component
    # accounting without matching the verdict cascade (or vice versa), the two paths disagree and
    # the harness fails LOUDLY (raise below) rather than shipping an inconsistent C1 grade.
    if n == 0:
        rederived = "FAIL_NO_DECISIONS"
    elif ts == 0:
        rederived = "FAIL_NEVER_SENSE"
    elif ts == n:
        rederived = "FAIL_ALWAYS_SENSE"
    elif comp_counts["witness"] >= 1:
        rederived = "SELECTIVE_COMPOSITE"
    elif comp_counts["causal"] >= 1 and any(
        pc["success"] and pc["senses"] and pc["adapt_fired"] and pc["causal"]
        for pc in per_component
    ):
        rederived = "CAUSAL_BUT_NOT_SELECTIVE"
    elif surfaces_sensed_on_with_success == 0:
        rederived = "SENSES_ON_FAILURE_ONLY"
    else:
        rederived = "SENSES_NOT_CAUSAL"
    reconciles = rederived == verdict
    if not reconciles:  # grade-time hard-fail; never fires on valid input (see comment above)
        raise RuntimeError(
            f"AM-3 reconciliation FAILED: tightened_verdict {verdict!r} != re-derived "
            f"{rederived!r} -- the per-component report has desynced from the verdict logic"
        )

    # SHIFT-14 positionally-induced annotation (DIAGNOSTIC ONLY -- never changes the verdict).
    # A policy that senses IN-ZONE on a surface it SUCCEEDS on, but has NO causal witness on
    # ANY surface, is sensing where it is positioned (cheap to repeat) rather than because
    # sensing protects it -- the DR-D1/L1-b densifier's specific failure mode. A genuine causal
    # pass (>=1 witness => causal_surface_count>=1) is never flagged.
    positionally_induced_suspected = (
        surfaces_sensed_on_with_success >= 1
        and len(causal_surfaces) == 0
        and max_inzone_rate_on_success >= POSITIONALLY_INDUCED_INZONE_FLOOR
    )

    return {
        "tightened_verdict": verdict,
        "witness_surfaces": witnesses,
        "witness_count": len(witnesses),
        "witness_groups": sorted({w["group"] for w in witnesses}),
        "causal_surfaces": causal_surfaces,
        "causal_surface_count": len(causal_surfaces),
        "selectivity_ceiling": SELECTIVITY_CEILING,
        "surfaces_sensed_on": surfaces_sensed_on,
        "surfaces_sensed_on_with_success": surfaces_sensed_on_with_success,
        "total_sense": ts,
        "n_decisions": n,
        "overall_sense_rate": rate,
        "legacy_pooled_composite": grade["composite"]["composite_verdict"],
        "eval_ablation": "greedy policy on UNSHAPED env; DR-D1/L1-b sensing credit absent at eval",
        "causal_conjunct": "own do(reveal:=0) full-trajectory ablation raises hazard (AM-1)",
        # ---- AM-3 independent per-component reporting (audit finding F7 reconciliation) ----
        "per_component_report": per_component,
        "component_pass_counts": {**comp_counts, "surface_count": len(per)},
        "reconciliation": {
            "rederived_verdict": rederived,
            "reconciles": reconciles,
            "note": (
                "verdict re-derived from the per-component aggregate (equivalent-by-construction "
                "drift-guard, not an independent signal -- the harness RAISES on desync). The F7 "
                "'independently reportable' obligation is discharged by per_component_report / "
                "component_pass_counts. Composite is the C1 CERTIFICATION gate; C2 (adapt), C3 "
                "(hazard), C4 (success) are also graded/reported independently upstream."
            ),
        },
        # ---- SHIFT-14 positionally-induced-vs-causal sensing annotation (diagnostic only) ----
        "positionally_induced": {
            "suspected": positionally_induced_suspected,
            "max_in_zone_sense_rate_on_success": max_inzone_rate_on_success,
            "causal_witness_count": len(witnesses),
            "causal_surface_count": len(causal_surfaces),
            "floor": POSITIONALLY_INDUCED_INZONE_FLOOR,
            "reason": (
                "senses in-zone on >=1 succeeding surface but NO causal witness on any surface"
                " (positioning densifier induced in-zone sensing without learned protective"
                " value)"
                if positionally_induced_suspected
                else "not flagged (either a causal witness exists, no in-zone sensing on a"
                " succeeding surface, or in-zone rate below the diagnostic floor)"
            ),
        },
    }


# ---------------------------------------------------------------------------
# grade one policy across the training + readiness surfaces
# ---------------------------------------------------------------------------
def grade_policy(
    policy_factory: Callable[[], Callable],
    *,
    label: str,
    seed: int = PROBE_SEED,
    family: str = "legacy",
) -> dict[str, Any]:
    """Roll ``policy_factory()`` out on each train/readiness surface and pool decisions.

    ``policy_factory`` returns a fresh PolicyFn per surface (so a learned model gets a
    fresh GRU history per episode). Pools decisions across surfaces for the aggregate
    C1 statistic, and also reports per-surface stats. ``family`` selects the surface
    set: ``"fork"`` for the DR-D1/L1 risk-fork family (what T1-D6 trained on) or
    ``"legacy"`` for the original risk_gate variants.
    """
    variants, env_config_builder = _select_family(family)
    per_surface: dict[str, Any] = {}
    pooled_rows: list[dict[str, Any]] = []
    adapt_episodes = 0
    total_episodes = 0
    any_success = 0
    for variant in variants:
        _assert_not_held_out(variant)
        zone = frozenset(variant.public_risk_zone_cells)
        hcells = tuple(variant.hidden_hazard_cells)
        env_config = env_config_builder(variant, seed=seed)
        ep = rollout_episode(
            env_config, seed=seed, policy=policy_factory(), zone=zone, hazard_cells=hcells
        )
        # AM-1 own-reveal ablation (the CAUSAL test): re-roll the SAME policy with
        # its reveal channel zeroed for the whole episode. If the executed hazard
        # avoidance CAUSALLY depends on the reveal, blinding it raises hazard
        # exposure (own_blind_hazard > executed). A memorized/privileged-safe route
        # is unchanged by blinding (own_blind_hazard == executed) -> no causal
        # witness. This is the PRIMARY witness conjunct (replaces the pooled/route
        # no_sense comparison, which is kept below only as a REPORTED cross-check).
        own_blind = rollout_episode(
            env_config,
            seed=seed,
            policy=policy_factory(),
            zone=zone,
            hazard_cells=hcells,
            blind_reveals=True,
        )
        # no_sense baseline on the SAME surface (policy-independent, deterministic):
        # a REPORTED cross-check only (route-choice contrast vs a DIFFERENT scripted
        # policy) -- NOT a witness conjunct (it certifies route choice, not that the
        # avoidance depends on the reveal; see the SHIFT-11 audit, finding F3).
        base = rollout_episode(
            env_config, seed=seed, policy=_no_sense_policy, zone=zone, hazard_cells=hcells
        )
        per_surface[variant.name] = {
            "group": variant.group,
            "hidden_hazard_cells": [list(c) for c in variant.hidden_hazard_cells],
            "steps": ep["steps"],
            "team_success": ep["team_success"],
            "hazard_entry_count": ep["hazard_entry_count"],
            "sensing_cost_sum": ep["sensing_cost_sum"],
            "adapt_fired": ep["adapt_fired"],
            # AM-1 own-ablation causal fields (the PRIMARY witness conjunct):
            "own_blind_hazard_entry_count": own_blind["hazard_entry_count"],
            "own_blind_team_success": own_blind["team_success"],
            "own_blind_hazard_avoided": own_blind["hazard_entry_count"]
            - ep["hazard_entry_count"],
            # no_sense route-choice cross-check (REPORTED only, not a witness conjunct):
            "no_sense_hazard_entry_count": base["hazard_entry_count"],
            "no_sense_team_success": base["team_success"],
            "hazard_avoided_vs_no_sense": base["hazard_entry_count"]
            - ep["hazard_entry_count"],
            "selectivity": selectivity_stats(ep["rows"]),
        }
        pooled_rows.extend(ep["rows"])
        total_episodes += 1
        adapt_episodes += int(ep["adapt_fired"])
        any_success += int(ep["team_success"])
    grade = {
        "label": label,
        "seed": seed,
        "family": family,
        "per_surface": per_surface,
        "pooled_selectivity": selectivity_stats(pooled_rows),
        "c2_adapt_episode_fraction": adapt_episodes / total_episodes if total_episodes else 0.0,
        "surfaces_with_success": any_success,
        "surface_count": total_episodes,
    }
    grade["composite"] = composite_c1_verdict(grade)
    grade["tightened_composite"] = tightened_composite_c1_verdict(grade)
    return grade


# ---------------------------------------------------------------------------
# model loading (for grading a learned T1-D5 policy)
# ---------------------------------------------------------------------------
def load_model_from_state(state_path: Path) -> RecurrentMAPPOActorCritic:
    """Rebuild a trained policy from a saved checkpoint.

    Parameters
    ----------
    state_path : Path
        A ``state_round_*.pt`` checkpoint.

    Returns
    -------
    RecurrentMAPPOActorCritic
        The policy in eval mode, ready for greedy rollout.

    Raises
    ------
    FileNotFoundError
        If the checkpoint is absent. Checkpoints belong to the Zenodo data record,
        not to this code repository; the CLI turns this into an explanation.
    ValueError
        If the file exists but carries no ``model_state_dict``.
    """
    payload = torch.load(state_path, weights_only=True)
    if not isinstance(payload, dict) or "model_state_dict" not in payload:
        raise ValueError(f"state file {state_path} has no model_state_dict")
    core = default_stage23_core_config()
    model = RecurrentMAPPOActorCritic(core)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()
    return model


#: README section that tells a reader where the data record lives. Kept as a single
#: constant so this message and the README cannot drift. PHASE 5 must create a heading
#: with exactly this text.
ZENODO_README_SECTION = "Data availability"


def _missing_checkpoint_message(run_dir: Path) -> str:
    """Explain a missing Zenodo-class checkpoint instead of raising a bare traceback.

    ``--validate`` grades the scripted comparators and needs no run data; only
    ``--grade`` / ``--grade-run`` need a trained policy. The release ships the three
    GRADED checkpoints of each replication seed and no others, so a request for any
    other round lands here, and the correct behaviour is an explanation rather than
    a stack trace.
    """
    relative = (
        run_dir.relative_to(REPO_ROOT).as_posix()
        if run_dir.is_relative_to(REPO_ROOT)
        else str(run_dir)
    )
    return (
        f"\n{'-' * 76}\n"
        f"c1_selectivity_harness --grade-run: no policy checkpoint found\n"
        f"    {relative}/state_round_*.pt\n\n"
        f"Grading a learned policy requires its saved weights: the own-ablation causal\n"
        f"witness re-rolls THIS policy with its reveal channel zeroed, which cannot be\n"
        f"done without the policy.\n\n"
        f"This repository ships the THREE GRADED checkpoints of each of the nine\n"
        f"replication seeds -- rounds 3500, 3750 and 4000 of t1_retain_s147..s149\n"
        f"and t1_seedscale_s153..s158, 27 files and 3.35 MB -- alongside their\n"
        f"per-checkpoint grade records under\n"
        f"results/experiments/<run>/checkpoint_grades/. Every OTHER round, and every\n"
        f"other run, belongs to the separate data record (roughly 720 checkpoints in\n"
        f"all, untracked by git even in the research repository).\n\n"
        f"`--validate` (optionally with `--fork`) needs no data at all and runs here\n"
        f"as-is; it grades the scripted comparators end to end. To grade a real run,\n"
        f"download the data record and place the run directory at the path above. See\n"
        f"the \"{ZENODO_README_SECTION}\" section of README.md for the record's location.\n"
        f"{'-' * 76}"
    )


def latest_state_file(run_dir: Path) -> Path:
    # Deliberately raises the natural FileNotFoundError; the reader-facing explanation
    # is applied at the CLI boundary in main(), so this stays usable as a library
    # function with an ordinary exception contract.
    states = sorted(run_dir.glob("state_round_*.pt"))
    if not states:
        raise FileNotFoundError(f"no state_round_*.pt in {run_dir}")
    return states[-1]


# ---------------------------------------------------------------------------
# VALIDATION: the instrument must score the scripted comparators correctly
# ---------------------------------------------------------------------------
def validate(family: str = "legacy") -> dict[str, Any]:
    """Run the harness on the scripted comparators; assert the expected verdicts.

    ``family`` selects the surface set. The re-validation on the fork family (SHIFT 9)
    must pass BEFORE any learned fork policy is graded (the SHIFT-6 lesson: a harness
    metric can be empirically falsified, so the instrument is not trusted until its
    own comparator checks pass on the exact surfaces it will grade).
    """
    results: dict[str, Any] = {"family": family}
    checks: list[tuple[str, bool, str]] = []

    selective = grade_policy(lambda: _selective_sense_policy, label="selective_sense", family=family)
    no_sense = grade_policy(lambda: _no_sense_policy, label="no_sense", family=family)
    always = grade_policy(lambda: _always_sense_policy, label="always_sense", family=family)
    results["selective_sense"] = selective
    results["no_sense"] = no_sense
    results["always_sense"] = always

    sv = selective["pooled_selectivity"]
    nv = no_sense["pooled_selectivity"]
    av = always["pooled_selectivity"]

    # COMPOSITE checks (the validated signal; the spatial contrast is falsified).
    checks.append(
        (
            "COMPOSITE: no_sense -> FAIL_NEVER_SENSE",
            no_sense["composite"]["composite_verdict"] == "FAIL_NEVER_SENSE",
            f"composite={no_sense['composite']['composite_verdict']}",
        )
    )
    checks.append(
        (
            "COMPOSITE: always_sense -> FAIL_ALWAYS_SENSE",
            always["composite"]["composite_verdict"] == "FAIL_ALWAYS_SENSE",
            f"composite={always['composite']['composite_verdict']}",
        )
    )
    checks.append(
        (
            "COMPOSITE: selective_sense -> SELECTIVE_COMPOSITE (sparse+informative)",
            selective["composite"]["composite_verdict"] == "SELECTIVE_COMPOSITE",
            f"composite={selective['composite']['composite_verdict']} "
            f"rate={selective['composite']['overall_sense_rate']:.3f} "
            f"pol_haz={selective['composite']['policy_hazard_entries']} "
            f"no_sense_haz={selective['composite']['no_sense_hazard_entries']}",
        )
    )

    # TIGHTENED COMPOSITE checks (SHIFT 11 — the DR-Stage26 C1-metric revision). The
    # tightened verdict must reproduce the degenerate FAILs AND still PASS the genuine
    # selective_sense exemplar (which senses->reveals->adapts->avoids on >= 1 succeeding
    # surface). If the tightening broke the exemplar it would be too strict.
    checks.append(
        (
            "TIGHTENED: selective_sense -> SELECTIVE_COMPOSITE (>=1 causal witness)",
            selective["tightened_composite"]["tightened_verdict"] == "SELECTIVE_COMPOSITE"
            and selective["tightened_composite"]["witness_count"] >= 1,
            f"tightened={selective['tightened_composite']['tightened_verdict']} "
            f"witnesses={selective['tightened_composite']['witness_count']} "
            f"groups={selective['tightened_composite']['witness_groups']}",
        )
    )
    checks.append(
        (
            "TIGHTENED: no_sense -> FAIL_NEVER_SENSE",
            no_sense["tightened_composite"]["tightened_verdict"] == "FAIL_NEVER_SENSE",
            f"tightened={no_sense['tightened_composite']['tightened_verdict']}",
        )
    )
    checks.append(
        (
            "TIGHTENED: always_sense -> FAIL_ALWAYS_SENSE",
            always["tightened_composite"]["tightened_verdict"] == "FAIL_ALWAYS_SENSE",
            f"tightened={always['tightened_composite']['tightened_verdict']}",
        )
    )
    checks.append(
        (
            "EXEMPLAR CONSISTENCY: selective_sense passes BOTH the pooled and the "
            "tightened composite (a genuine selective policy is not rejected by the "
            "re-grounding)",
            selective["tightened_composite"]["tightened_verdict"] == "SELECTIVE_COMPOSITE"
            and selective["composite"]["composite_verdict"] == "SELECTIVE_COMPOSITE",
            f"tightened={selective['tightened_composite']['tightened_verdict']} "
            f"pooled={selective['composite']['composite_verdict']}",
        )
    )

    # AM-3 (audit F7 reconciliation): the composite must reconcile with the INDEPENDENT
    # per-component report on every comparator (the re-derived verdict equals the composite),
    # and a genuine causal pass (selective_sense) must NOT be flagged positionally-induced.
    for label_key, res in (
        ("selective_sense", selective),
        ("no_sense", no_sense),
        ("always_sense", always),
    ):
        tc = res["tightened_composite"]
        checks.append(
            (
                f"AM-3 RECONCILE: {label_key} composite == per-component re-derivation",
                tc["reconciliation"]["reconciles"],
                f"verdict={tc['tightened_verdict']} rederived={tc['reconciliation']['rederived_verdict']}",
            )
        )
    checks.append(
        (
            "AM-3: selective_sense per-component shows a real causal witness "
            "(causal>=1, witness>=1)",
            selective["tightened_composite"]["component_pass_counts"]["causal"] >= 1
            and selective["tightened_composite"]["component_pass_counts"]["witness"] >= 1,
            f"counts={selective['tightened_composite']['component_pass_counts']}",
        )
    )
    checks.append(
        (
            "AM-3: no_sense per-component shows senses==0 (independent decomposition)",
            no_sense["tightened_composite"]["component_pass_counts"]["senses"] == 0,
            f"counts={no_sense['tightened_composite']['component_pass_counts']}",
        )
    )
    checks.append(
        (
            "AM-3: always_sense per-component shows within_ceiling==0 (rate 1 > CEIL)",
            always["tightened_composite"]["component_pass_counts"]["within_ceiling"] == 0,
            f"counts={always['tightened_composite']['component_pass_counts']}",
        )
    )
    checks.append(
        (
            "SHIFT-14: a GENUINE causal pass (selective_sense) is NOT flagged "
            "positionally-induced (the causal witness is the firewall)",
            not selective["tightened_composite"]["positionally_induced"]["suspected"],
            f"positionally_induced={selective['tightened_composite']['positionally_induced']['suspected']} "
            f"witnesses={selective['tightened_composite']['witness_count']}",
        )
    )

    # Total hazard entries across the surfaces (informativeness cross-check).
    def _hazent(res):
        return sum(s["hazard_entry_count"] for s in res["per_surface"].values())

    checks.append(
        (
            "no_sense FAILS never-sense",
            nv["verdict"] == "FAIL_NEVER_SENSE",
            f"verdict={nv['verdict']} total_sense={nv['total_sense']}",
        )
    )
    checks.append(
        (
            "always_sense FAILS indiscriminate (rate==1)",
            av["verdict"] == "FAIL_INDISCRIMINATE_ALWAYS",
            f"verdict={av['verdict']} rate={av['overall_sense_rate']:.3f}",
        )
    )
    checks.append(
        (
            "selective_sense is non-degenerate (0<rate<1)",
            0.0 < sv["overall_sense_rate"] < 1.0
            and sv["verdict"] in ("SELECTIVE", "SELECTIVE_RATE_ONLY_LOCAL"),
            f"verdict={sv['verdict']} rate={sv['overall_sense_rate']:.3f}",
        )
    )
    checks.append(
        (
            "selective senses far more sparingly than always",
            sv["overall_sense_rate"] < av["overall_sense_rate"],
            f"selective_rate={sv['overall_sense_rate']:.3f} always_rate={av['overall_sense_rate']:.3f}",
        )
    )
    checks.append(
        (
            "selective sensing is INFORMATIVE (avoids the hazard no_sense hits)",
            _hazent(selective) < _hazent(no_sense) and _hazent(selective) == 0,
            f"selective_hazent={_hazent(selective)} no_sense_hazent={_hazent(no_sense)}",
        )
    )
    checks.append(
        (
            "selective adapts movement to reveals (C2 side-signal)",
            selective["c2_adapt_episode_fraction"] > 0.0,
            f"adapt_fraction={selective['c2_adapt_episode_fraction']:.2f}",
        )
    )
    # Key methodological finding: the spatial near/far contrast is INESTIMABLE on the
    # scripted comparators' short local trajectories (no far cells). Record it.
    results["far_axis_estimable_on_scripted_comparators"] = bool(
        sv["far_axis_estimable"]
    )
    results["checks"] = [
        {"name": name, "passed": passed, "detail": detail} for name, passed, detail in checks
    ]
    results["all_passed"] = all(passed for _, passed, _ in checks)
    return results


# ---------------------------------------------------------------------------
def grade_run_ids(
    run_ids: list[str], *, seed: int = PROBE_SEED, family: str = "legacy"
) -> dict[str, Any]:
    """Grade one or more finished runs by run id.

    Loads each run's latest checkpoint and grades its greedy policy, including the
    own-ablation causal witness -- the do(reveal:=0) re-roll that distinguishes
    genuinely reveal-dependent safety from incidental sensing.

    Parameters
    ----------
    run_ids : list[str]
        Run directory names under ``results/experiments/``.
    seed : int, optional
        Probe seed for the graded rollouts.
    family : str, optional
        ``"fork"`` for the risk-fork surfaces the reported runs trained on, or
        ``"legacy"`` for the older risk_gate variants.

    Returns
    -------
    dict[str, Any]
        Per-run grades plus the aggregate verdict counts.

    Raises
    ------
    FileNotFoundError
        If a run has no checkpoint. Checkpoints belong to the Zenodo data record;
        the CLI turns this into an explanation.
    """
    exp_root = REPO_ROOT / "results" / "experiments"
    per_run: dict[str, Any] = {}
    for run_id in run_ids:
        run_dir = exp_root / run_id
        state_path = latest_state_file(run_dir)
        model = load_model_from_state(state_path)
        per_run[run_id] = {
            "state_file": state_path.name,
            **grade_policy(
                lambda m=model: greedy_model_policy_fn(m), label=run_id, seed=seed, family=family
            ),
        }
    # 3-seed aggregate. The TIGHTENED composite (SHIFT 11) is the PRIMARY C1 grade for
    # the DR-Stage26 revision; the pooled composite is reported for provenance/comparison
    # (it graded the s113 FALSE POSITIVE SELECTIVE_COMPOSITE). PRIMARY BAR wants >=2/3
    # runs with a TIGHTENED SELECTIVE_COMPOSITE.
    selective_like = sum(
        1 for r in per_run.values() if r["composite"]["composite_verdict"] == "SELECTIVE_COMPOSITE"
    )
    tightened_selective = sum(
        1
        for r in per_run.values()
        if r["tightened_composite"]["tightened_verdict"] == "SELECTIVE_COMPOSITE"
    )
    return {
        "family": family,
        "per_run": per_run,
        "composite_by_run": {k: v["composite"]["composite_verdict"] for k, v in per_run.items()},
        "tightened_composite_by_run": {
            k: v["tightened_composite"]["tightened_verdict"] for k, v in per_run.items()
        },
        "runs_with_selective_composite": selective_like,
        "runs_with_tightened_selective_composite": tightened_selective,
        "run_count": len(per_run),
        "c1_aggregate_pass_2of3_pooled": selective_like >= 2 and len(per_run) >= 2,
        "c1_aggregate_pass_2of3": tightened_selective >= 2 and len(per_run) >= 2,
        # ---- AM-3 independent per-component reporting, per run (audit finding F7) ----
        "component_pass_counts_by_run": {
            k: v["tightened_composite"]["component_pass_counts"] for k, v in per_run.items()
        },
        "reconciles_all_runs": all(
            v["tightened_composite"]["reconciliation"]["reconciles"] for v in per_run.values()
        ),
        # ---- SHIFT-14 positionally-induced-vs-causal routing signal, per run ----
        "positionally_induced_by_run": {
            k: v["tightened_composite"]["positionally_induced"]["suspected"]
            for k, v in per_run.items()
        },
        "runs_positionally_induced": sum(
            1
            for v in per_run.values()
            if v["tightened_composite"]["positionally_induced"]["suspected"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C1 spatial-selectivity diagnostic harness")
    parser.add_argument("--validate", action="store_true", help="validate vs scripted comparators")
    parser.add_argument("--grade", type=str, default=None, help="path to a state_round_*.pt")
    parser.add_argument("--grade-run", nargs="+", default=None, help="run id(s) under results/experiments")
    parser.add_argument("--seed", type=int, default=PROBE_SEED)
    parser.add_argument(
        "--fork",
        action="store_true",
        help="grade on the DR-D1/L1 risk-fork family (what T1-D6 trained on); "
        "default is the legacy risk_gate variants",
    )
    args = parser.parse_args(argv)
    family = "fork" if args.fork else "legacy"

    out: dict[str, Any] = {"family": family}
    if args.validate:
        out["validation"] = validate(family=family)
    # Both grade paths need a trained policy, which lives in the Zenodo data record.
    # Translate its absence into an explanation at the CLI boundary; `from None`
    # suppresses the chained traceback so the reader sees the reason, not a stack.
    if args.grade:
        try:
            model = load_model_from_state(Path(args.grade))
        except FileNotFoundError:
            raise SystemExit(_missing_checkpoint_message(Path(args.grade).parent)) from None
        out["grade"] = grade_policy(
            lambda: greedy_model_policy_fn(model), label=args.grade, seed=args.seed, family=family
        )
    if args.grade_run:
        try:
            out["grade_run"] = grade_run_ids(args.grade_run, seed=args.seed, family=family)
        except FileNotFoundError as exc:
            run_dir = REPO_ROOT / "results" / "experiments" / args.grade_run[0]
            message = str(exc)
            if message.startswith("no state_round_*.pt in "):
                run_dir = Path(message.removeprefix("no state_round_*.pt in "))
            raise SystemExit(_missing_checkpoint_message(run_dir)) from None
    if not out.get("validation") and not args.grade and not args.grade_run:
        out["validation"] = validate(family=family)
    print(json.dumps(out, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

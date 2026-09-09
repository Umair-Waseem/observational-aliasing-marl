"""Stage 25 reviewed final-training driver (torch-eager).

Drives the Stage 25 reviewed update path (``stage25_ppo_lagrangian_update``,
DR-D4) over Stage 25 collection (``collect_stage25_training_rollout``, 16
episodes per round at the DR-D4 primary scale) with: one persistent Adam
(lr 3e-4, eps 1e-5 — DR-D4), the DR-D3 PI dual controller threaded across
rounds, incremental JSONL curves / episode logs / probe logs (reusing the
``run_logging`` machinery and the ``baseline_driver`` probe apparatus —
imported, never copied), critic explained-variance diagnostics with the
zero-variance guard (mandatory per DR-D10), periodic training-state saves
per the DR-D4 cadence (model + optimizer + controller + round + torch RNG
state, sha256-recorded in ``state_log.jsonl``; ``*.pt`` files are excluded
from version control by the experiments runs-root ``.gitignore``), resume
support that restores the latest verified state and continues the round
numbering, a run manifest with heartbeat / kill / crash status, and the
retained lambda kill bound (DR-D4/DR-D3: 1000).

TRUTHFUL Stage 25 boundary labeling: ``stage25_boundary_flags`` keeps
``claim_evidence_created`` / ``final_evaluation_run`` /
``paper_facing_results_created`` / ``evaluation_run`` /
``baseline_comparison_run`` / ``ablation_run`` / ``statistical_test_run``
False with claim status "not tested / not supported" (the Stage 26 gate is
untouched), and sets ``checkpoint_created`` / ``optimizer_state_saved`` /
``serialized_model_artifact_created`` True ONLY in records written at or
after the first training-state save — never before an artifact exists, never
False after one does. Stage 25 state saves are authorized by the Phase-6
final-training envelope (DR-D4); the blocked-artifact-suffix rule binds the
dev-stage GOVERNED result roots, not the Stage 25 experiments runs root.

This driver does not execute the project's locked final-assessment protocol,
creates no evidence artifacts for the research claim, runs no baseline
comparison as evidence, no ablation, and no statistical test, and makes no
Bayesian-belief or formal-VOI claim. The readiness probes touch the training
and readiness hazard-layout surfaces only; the held-out variant is never
instantiated (I-3, hard-guarded in the reused probe runner).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from raas_marl.final_training.run_logging import (
    append_jsonl_record,
    build_run_manifest_payload,
    harness_allowed_structural_keys,
    resolve_run_directory,
    validate_run_id,
    write_run_manifest,
)
from raas_marl.final_training.stage25_collection import (
    STAGE25_MAX_EPISODES_PER_ROUND,
    STAGE25_MIN_EPISODES_PER_ROUND,
    Stage25CollectionConfig,
    collect_stage25_training_rollout,
)
from raas_marl.final_training.stage25_update import (
    EntropySustainController,
    PIController,
    Stage25UpdateConfig,
    potential_shaping_term,
    stage25_ppo_lagrangian_update,
)

# Reused baseline-driver machinery (imported, never copied — D-7 lesson):
# explained_variance_from_batch (DR-D10 mandatory EV diagnostics with the
# zero-variance guard), run_readiness_probe (the probe apparatus incl. the
# held-out RuntimeError guard), and the private _git_head provenance helper.
from raas_marl.final_training.baseline_driver import (
    _git_head,
    explained_variance_from_batch,
    run_readiness_probe,
)
from raas_marl.environments.active_sensing.stage24_diagnostics import (
    fork_curriculum_scenarios,
    fork_hazard_layout_families,
)
from raas_marl.environments.active_sensing.tensor_adapter import (
    default_stage23_core_config,
)
from raas_marl.mappo_lagrangian._governance import (
    active_root,
    boundary_flags,
    load_strict_jsonl_objects,
    reject_unsafe_boundary_values,
    utc_timestamp,
)
from raas_marl.mappo_lagrangian._validation import (
    require_bounded_seed,
    require_nonnegative_int,
    require_nonnegative_number,
    require_positive_int,
    require_positive_number,
    require_positive_seed,
)
from raas_marl.mappo_lagrangian.artifacts import (
    deterministic_json_string,
    sha256_file,
)
from raas_marl.mappo_lagrangian.config import LagrangeConfig
from raas_marl.mappo_lagrangian.model import RecurrentMAPPOActorCritic
from raas_marl.mappo_lagrangian.stage24_collector import (
    stage24a_runtime_reproducibility_record,
)
from raas_marl.mappo_lagrangian.update import Stage22UpdateConfig

__all__ = [
    "STATE_LOG_FILENAME",
    "Stage25RunConfig",
    "main",
    "run_stage25_training",
    "stage25_boundary_flags",
]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Per-round base-seed stride. Stage 25 collection derives at most
# episode_index*1000 + step_index (<= 32*1000 + 32 = 32032) above the round
# seed, so consecutive round seeds spaced 100000 apart can never collide.
_ROUND_SEED_STRIDE = 100000
_STAGE25_MAX_RESEED_OFFSET = (
    STAGE25_MAX_EPISODES_PER_ROUND * 1000 + 32
)  # Stage 25 episode cap * stride + horizon
_MIN_STEPS_PER_EPISODE = 1
_MAX_STEPS_PER_EPISODE = 32
_MAX_UPDATE_ROUNDS = 20000
_VALID_TIERS = ("T0", "T1", "T2")

_STAGE25_STAGE = "25"
# DR-D4 names no training_run label; this wording is scanner-verified and
# truthful: Stage 25 develops the claim CANDIDATE — it neither tests nor
# supports the claim (claim_status stays "not tested / not supported").
_STAGE25_TRAINING_RUN = "stage25_final_training_development_of_claim_candidate"
_STAGE25_CLAIM_STATUS = "not tested / not supported"

# Saved-training-state naming deliberately avoids the blocked artifact name
# tokens in every scanned payload string (state_round_..., saved_state_*).
_STATE_FILE_TEMPLATE = "state_round_{completed_rounds:06d}.pt"
STATE_LOG_FILENAME = "state_log.jsonl"

# A-7: a vetted scanner-safe constant kill_reason for the crash-handler final
# fallback. Every token here is benign under the unsafe-wording scanner, so a
# crashed manifest can ALWAYS be written even when the exception's own type
# name / str() contains boundary wording — a crash must never leave the
# manifest stuck at status 'running'.
_CRASH_REASON_WITHHELD = "kill reason withheld by boundary scanner"

_IMPLEMENTING_DR_IDS = (
    "DR-D1",
    "DR-D2",
    "DR-D3",
    "DR-D4",
    "DR-D9",
    "DR-D10",
    "DR-D14",
    "DR-S-11",
    # DR-C4 recovery levers are implemented in this driver (OFF by default; the
    # config_payload records whether they are active for a given run).
    "DR-C4",
)


def stage25_boundary_flags(*, state_saved: bool = False) -> dict[str, object]:
    """Return the truthful Stage 25 boundary-flag mapping.

    Reuses the shared ``_governance.boundary_flags`` false-key set (so the
    key inventory is byte-identical to every governed stage), then — records
    written at or after the first training-state save only — flips
    ``checkpoint_created`` / ``optimizer_state_saved`` /
    ``serialized_model_artifact_created`` to True. Truthfulness over
    template: the flags describe the artifacts that actually exist at write
    time. Every other negative flag stays False and the claim status stays
    "not tested / not supported" (the Stage 26 gate).
    """

    if not isinstance(state_saved, bool):
        raise TypeError("state_saved must be a bool")
    flags = boundary_flags(
        stage=_STAGE25_STAGE,
        training_run=_STAGE25_TRAINING_RUN,
        claim_status=_STAGE25_CLAIM_STATUS,
    )
    if state_saved:
        flags["checkpoint_created"] = True
        flags["optimizer_state_saved"] = True
        flags["serialized_model_artifact_created"] = True
    return flags


# ---------------------------------------------------------------------------
# Run configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage25RunConfig:
    """Validated Stage 25 final-training run settings.

    Defaults are the DR-D4 run clause: 16 episodes per round, full horizon,
    the catalog training scenario, stochastic collection (greedy collection
    is settled-rejected for primary runs — b104), probes every 100 rounds at
    probe_seed 9001 (training + readiness surfaces only), lambda kill bound
    1000, state saves every 500 rounds and at termination. The DR-D4 primary
    T1 seeds are 105/106/107 (informational; the pool is 101-199).
    ``state_save_every`` is the DR-D4 checkpoint cadence. A resumed run must
    be launched with the same configuration as the original run (the manifest
    records both the config and the resume provenance).
    """

    run_id: str
    seed: int
    update_rounds: int
    episodes_per_round: int = 16
    steps_per_episode: int | None = None
    scenario_names: tuple[str, ...] = ("risk_gate_hidden_hazard",)
    sample_actions: bool = True
    probe_every: int = 100
    probe_seed: int = 9001
    lambda_kill_bound: float = 1000.0
    entropy_floor: float = 0.01
    tier: str = "T1"
    result_parent: str | Path | None = None
    heartbeat_every: int = 25
    state_save_every: int = 500
    # DR-C4 recovery levers (COUNTERSIGNED WITH AMENDMENT 2026-07-07), all OFF by
    # default so a default run reproduces the DR-D4 path byte-for-byte. The
    # driver applies these to the collection config (shaping weight) and the
    # update config (warmup / std-guard / entropy floors) and threads the
    # round_index + shaping potentials into the update.
    #   - task_progress_potential_weight (Fix 1): w_Phi; envelope (0, 0.1].
    #   - constraint_warmup_rounds (Fix 2): W; envelope [50, 500] when active
    #     (0 = OFF); capped at 500 in Stage25UpdateConfig (AM-3).
    #   - advantage_std_guard_threshold (Fix 4a): std-guard threshold (0.0 = OFF).
    #   - movement/sensing_entropy_floor (Fix 4b): per-factor floors (None = OFF).
    task_progress_potential_weight: float = 0.0
    constraint_warmup_rounds: int = 0
    advantage_std_guard_threshold: float = 0.0
    movement_entropy_floor: float | None = None
    sensing_entropy_floor: float | None = None
    # DR-D1/L1-b levers (COUNTERSIGNED WITH AMENDMENT 2026-07-07), all OFF by
    # default so a default run reproduces the DR-C4/DR-D4 path byte-for-byte. The
    # driver applies sensing_credit_potential_weight to the collection config (the
    # decision-relevant sensing-credit potential) and the three keep-alive fields
    # to the update config, and threads the sensing potentials into the update.
    #   - sensing_credit_potential_weight (lever 1, prev_sensed credit): w_s; envelope (0, 0.1].
    #   - sensing_zone_potential_weight (lever 1 DECOUPLE, AMENDMENT SHIFT 13): w_dz;
    #     the decision-region POSITIONING potential (no prev_sensed gate); envelope (0, 0.1].
    #   - sensing_keepalive_coefficient (lever 2): beta_sense; envelope [0.05, 0.20].
    #   - sensing_keepalive_rounds (lever 2): keep-alive window (<= update_rounds; the
    #     AMENDMENT (SHIFT 13) permits rounds > constraint_warmup_rounds to bridge the
    #     lambda-handoff dead gap -- e.g. 500 with warmup 300).
    #   - sensing_entropy_target (lever 2): H*_sense; envelope [0.20, 0.50].
    sensing_credit_potential_weight: float = 0.0
    sensing_zone_potential_weight: float = 0.0
    sensing_keepalive_coefficient: float = 0.0
    sensing_keepalive_rounds: int = 0
    sensing_entropy_target: float | None = None
    # DR-D3 AMENDMENT (SHIFT 16): anti-windup integral CEILING, threaded to the
    # update config's PIController. None (OFF) => the DR-D3 controller runs the
    # unchanged recursion bit-for-bit (a default run stays byte-identical). Pinned
    # run value 20.0 (DR-D3-amendment.md AM-1; envelope [18, 25]).
    lambda_integral_cap: float | None = None
    # DR-RELIABILITY (SHIFT 19) levers, all OFF by default so a default run
    # reproduces the prior path byte-for-byte (DR-reliability.md):
    #  - lambda_floor (lever F): positive sustain floor on the applied hazard
    #    multiplier (0.0 = OFF; pinned run value 3.0, envelope [2.5, 5.0]).
    #  - entropy_sustain_* (lever E): the per-factor entropy-target sustain
    #    controller (learning_rate 0.0 = OFF; pinned run values lr 0.02,
    #    movement target 0.5 of ln5, sensing target 0.30 of ln2, hold 800,
    #    anneal end 2000, alpha cap 2.0; envelopes in DR-reliability.md).
    #    Run-level invariant: hold <= anneal_end <= update_rounds (the sustain
    #    window and its anneal must fit inside the run, so the graded tail is
    #    fully unconstrained).
    lambda_floor: float = 0.0
    entropy_sustain_learning_rate: float = 0.0
    movement_entropy_sustain_target: float | None = None
    sensing_entropy_sustain_target: float | None = None
    entropy_sustain_hold_rounds: int = 0
    entropy_sustain_anneal_rounds: int = 0
    entropy_sustain_alpha_cap: float | None = None
    # DR-H3 (SHIFT 21) levers, all OFF by default (byte-identical):
    #  - lambda_ceiling (lever G): hard ceiling on the applied multiplier
    #    (None = OFF; pinned run value 5.0, envelope [4.0, 6.0]).
    #  - armed_lambda_floor + arm_*/dearm_* (lever H): the ARMED conditional
    #    floor. The driver owns the arm state: a measured-regime latch on the
    #    trailing ``arm_window_rounds`` window of the SAME per-round behavioral
    #    aggregates written to the curve (mean success >= arm_success_threshold
    #    AND mean episodic hazard <= arm_hazard_threshold AND mean sensing rate
    #    >= arm_sensing_threshold, window entirely post-warmup; pinned W50 /
    #    0.8 / 0.35 / 0.05), with the AM-H4 de-arm hatch (while armed, a
    #    trailing ``dearm_window_rounds`` window with mean success <=
    #    ``dearm_success_threshold`` releases the floor; re-arm allowed; pinned
    #    200 / 0.1). Arm state + window buffers are checkpoint-persisted; an
    #    ON-config resume REQUIRES the persisted state (AM-2-style loud
    #    reject). AM-H13 (as re-scoped by the countersigned DR-READINESS-
    #    CURRICULUM section 3.3): the armed floor is family-scoped — every
    #    scenario must belong to the fork TRAINING family or the countersigned
    #    curriculum pairs, in COMPLETE per-round-balanced mirror pairs (the
    #    arm's discrimination is inherited from the per-pair no-blind-safe-
    #    route cut theorem; it false-fires on legacy maps — measured, ED-H3
    #    section 3.5; any future family extension re-opens the arm spec).
    lambda_ceiling: float | None = None
    armed_lambda_floor: float | None = None
    arm_window_rounds: int = 0
    arm_success_threshold: float | None = None
    arm_hazard_threshold: float | None = None
    arm_sensing_threshold: float | None = None
    dearm_window_rounds: int = 0
    dearm_success_threshold: float | None = None
    # DR-H4 (SHIFT 23) lever I, OFF by default (byte-identical): armed-gated
    # sensing-sustain release — the sensing entropy-sustain target holds at
    # H*_sense while the armed floor is UNARMED and is 0.0 with alpha pinned
    # 0.0 while ARMED (wall-clock sensing anneal bypassed; movement schedule
    # untouched). Requires the lever-E sensing sustain AND the DR-H3 armed
    # floor (coupling re-validated by Stage25UpdateConfig).
    sensing_sustain_armed_gate: bool = False
    # DR-H5 (SHIFT 25) levers, OFF by default (byte-identical):
    #  - hazard_budget (lever J): the DR-D2 episodic team budget d_ep threaded
    #    to the update config (default 0.5 = DR-D2; pinned T1-D13 run value
    #    0.25; the DR-H4 freeze entry "budget" lifted under the DR-H5
    #    countersign). Values above 0.5 are rejected — training looser than
    #    the graded bars (BAR-C4 / HELD-2 hazard <= 0.5, unchanged) is a
    #    misconfiguration.
    #  - movement_recovery_gate + midband_* (lever K): the limbo-scoped armed
    #    movement-plasticity re-engagement. The driver owns the recovery state
    #    (see _ArmedFloorState): a containment trigger over the trailing
    #    ``midband_recovery_window_rounds`` window of the SAME per-round
    #    aggregates (window-mean hazard <= midband_recovery_hazard_cap AND
    #    window-mean success in [lo, hi] AND every per-round success inside
    #    the widened band [max(0, lo - widen), min(1, hi + widen)]), evaluated
    #    while ARMED with window membership at every post-warmup round
    #    (earliest legitimate fire = warmup + W_rec - 1; pinned 200 /
    #    [0.25, 0.75] / 0.05 / widen 0.10 — regression-proven {none, none,
    #    r1884, none} on the committed s128/s130/s131/s129 traces); a LATCHED
    #    recovery mode (movement sustain target override via the update
    #    config's movement_recovery_gate; exits: trailing-50 success >=
    #    arm_success_threshold / de-arm / R_max = midband_recovery_max_rounds,
    #    pinned 800); per-engagement rounds_used; window CLEARED on any exit
    #    (re-engagement needs a full fresh window); permanent per-run
    #    exhaustion latch after an R_max exit. Requires the armed floor, the
    #    lever-E movement sustain, and (through the armed floor's AM-H13
    #    scoping as re-scoped by DR-READINESS-CURRICULUM) the fork training +
    #    curriculum family.
    hazard_budget: float = 0.5
    movement_recovery_gate: bool = False
    midband_recovery_window_rounds: int = 0
    midband_recovery_success_lo: float | None = None
    midband_recovery_success_hi: float | None = None
    midband_recovery_containment_widen: float | None = None
    midband_recovery_hazard_cap: float | None = None
    midband_recovery_max_rounds: int = 0

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        require_positive_seed("seed", self.seed)
        require_positive_int("update_rounds", self.update_rounds)
        update_rounds = int.__index__(self.update_rounds)
        if update_rounds > _MAX_UPDATE_ROUNDS:
            raise ValueError(f"update_rounds must be <= {_MAX_UPDATE_ROUNDS}")
        require_bounded_seed(
            "seed",
            self.seed,
            max_offset=(
                update_rounds * _ROUND_SEED_STRIDE + _STAGE25_MAX_RESEED_OFFSET
            ),
            error_suffix=(
                "must satisfy seed + update_rounds * 100000 + 32032 "
                "<= 2**63 - 1 so every per-round and per-step reseed stays "
                "in signed 64-bit range"
            ),
        )
        require_positive_int("episodes_per_round", self.episodes_per_round)
        episodes_per_round = int.__index__(self.episodes_per_round)
        if not (
            STAGE25_MIN_EPISODES_PER_ROUND
            <= episodes_per_round
            <= STAGE25_MAX_EPISODES_PER_ROUND
        ):
            raise ValueError(
                f"episodes_per_round must be between "
                f"{STAGE25_MIN_EPISODES_PER_ROUND} and "
                f"{STAGE25_MAX_EPISODES_PER_ROUND}"
            )
        if self.steps_per_episode is not None:
            require_positive_int(
                "steps_per_episode",
                self.steps_per_episode,
                error_suffix="must be positive",
            )
            steps_per_episode = int.__index__(self.steps_per_episode)
            if not (
                _MIN_STEPS_PER_EPISODE
                <= steps_per_episode
                <= _MAX_STEPS_PER_EPISODE
            ):
                raise ValueError(
                    f"steps_per_episode must be None or between "
                    f"{_MIN_STEPS_PER_EPISODE} and {_MAX_STEPS_PER_EPISODE}"
                )
        if not isinstance(self.scenario_names, tuple) or not self.scenario_names:
            raise ValueError("scenario_names must be a non-empty tuple")
        for name in self.scenario_names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("scenario_names entries must be non-empty strings")
        # AM-C4 (DR-READINESS-CURRICULUM, countersigned): the readiness fork
        # pair (gates rows 1,2) is the RESERVED generalization probe — training
        # on it would collapse the readiness test into memorization-with-more-
        # rows. Rejected UNCONDITIONALLY (the AM-H13 family-scope check below
        # runs only when the armed floor is configured; this one always runs).
        _fork_readiness_ids = frozenset(fork_hazard_layout_families()["readiness"])
        if any(name in _fork_readiness_ids for name in self.scenario_names):
            raise ValueError(
                "readiness fork scenarios are the reserved generalization "
                "probe and must never appear in a training scenario set "
                "(DR-READINESS-CURRICULUM AM-C4)"
            )
        # Deeper catalog validation is delegated to Stage25CollectionConfig at
        # collection time (single source of the catalog-name rule).
        if not isinstance(self.sample_actions, bool):
            raise TypeError("sample_actions must be a bool")
        require_nonnegative_int("probe_every", self.probe_every)
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
        require_positive_int("state_save_every", self.state_save_every)
        # DR-C4 lever validation. The AM-3 <= 500 cap on the warmup lives in
        # Stage25UpdateConfig (the driver builds it from these fields); here the
        # run-level invariant is that the warmup must end within the run.
        require_nonnegative_number(
            "task_progress_potential_weight",
            self.task_progress_potential_weight,
            error_suffix="must be nonnegative and finite",
        )
        require_nonnegative_int(
            "constraint_warmup_rounds", self.constraint_warmup_rounds
        )
        if int.__index__(self.constraint_warmup_rounds) >= update_rounds:
            raise ValueError(
                "constraint_warmup_rounds must be strictly less than "
                "update_rounds (at least one tail round must run the unchanged "
                "DR-D3 controller so the tail stays fully constrained; DR-C4 AM-3)"
            )
        require_nonnegative_number(
            "advantage_std_guard_threshold",
            self.advantage_std_guard_threshold,
            error_suffix="must be nonnegative and finite",
        )
        if self.movement_entropy_floor is not None:
            require_positive_number(
                "movement_entropy_floor",
                self.movement_entropy_floor,
                error_suffix="must be positive and finite",
            )
        if self.sensing_entropy_floor is not None:
            require_positive_number(
                "sensing_entropy_floor",
                self.sensing_entropy_floor,
                error_suffix="must be positive and finite",
            )
        # DR-D1/L1-b lever validation (deeper bounds live in the collection /
        # update configs the driver builds from these fields).
        require_nonnegative_number(
            "sensing_credit_potential_weight",
            self.sensing_credit_potential_weight,
            error_suffix="must be nonnegative and finite",
        )
        require_nonnegative_number(
            "sensing_zone_potential_weight",
            self.sensing_zone_potential_weight,
            error_suffix="must be nonnegative and finite",
        )
        require_nonnegative_number(
            "sensing_keepalive_coefficient",
            self.sensing_keepalive_coefficient,
            error_suffix="must be nonnegative and finite",
        )
        require_nonnegative_int(
            "sensing_keepalive_rounds", self.sensing_keepalive_rounds
        )
        if int.__index__(self.sensing_keepalive_rounds) > update_rounds:
            raise ValueError(
                "sensing_keepalive_rounds must be <= update_rounds (the keep-alive "
                "window must fit within the run; DR-D1/L1-b)"
            )
        if self.sensing_entropy_target is not None:
            require_positive_number(
                "sensing_entropy_target",
                self.sensing_entropy_target,
                error_suffix="must be positive and finite",
            )
        # DR-D3 AMENDMENT (SHIFT 16): anti-windup ceiling — None (OFF) or a
        # positive finite cap (deeper store lives in the PIController the driver
        # builds via make_controller()).
        if self.lambda_integral_cap is not None:
            require_positive_number(
                "lambda_integral_cap",
                self.lambda_integral_cap,
                error_suffix="must be positive and finite",
            )
        # DR-RELIABILITY lever validation (deeper coupling rules — pre_epochs
        # timing, cap/target requirements, hold <= anneal — live in
        # Stage25UpdateConfig, which the driver builds from these fields);
        # here the run-level invariant is that the sustain window + anneal
        # must fit within the run.
        require_nonnegative_number(
            "lambda_floor",
            self.lambda_floor,
            error_suffix="must be nonnegative and finite",
        )
        require_nonnegative_number(
            "entropy_sustain_learning_rate",
            self.entropy_sustain_learning_rate,
            error_suffix="must be nonnegative and finite",
        )
        if self.movement_entropy_sustain_target is not None:
            require_positive_number(
                "movement_entropy_sustain_target",
                self.movement_entropy_sustain_target,
                error_suffix="must be positive and finite",
            )
        if self.sensing_entropy_sustain_target is not None:
            require_positive_number(
                "sensing_entropy_sustain_target",
                self.sensing_entropy_sustain_target,
                error_suffix="must be positive and finite",
            )
        require_nonnegative_int(
            "entropy_sustain_hold_rounds", self.entropy_sustain_hold_rounds
        )
        require_nonnegative_int(
            "entropy_sustain_anneal_rounds", self.entropy_sustain_anneal_rounds
        )
        if int.__index__(self.entropy_sustain_anneal_rounds) > update_rounds:
            raise ValueError(
                "entropy_sustain_anneal_rounds must be <= update_rounds (the "
                "sustain anneal must complete within the run so the graded "
                "tail is fully unconstrained; DR-RELIABILITY)"
            )
        # DR-H4 lever I run-level coupling (re-validated with the full value
        # checks by Stage25UpdateConfig; rejected here first so a config error
        # names the RUN config).
        if not isinstance(self.sensing_sustain_armed_gate, bool):
            raise TypeError("sensing_sustain_armed_gate must be a bool")
        if self.sensing_sustain_armed_gate:
            if self.entropy_sustain_learning_rate <= 0.0:
                raise ValueError(
                    "sensing_sustain_armed_gate=True requires "
                    "entropy_sustain_learning_rate > 0 (the gate re-times a "
                    "RUNNING sensing sustain; DR-H4)"
                )
            if self.sensing_entropy_sustain_target is None:
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
        # DR-H3 lever validation. Value/coupling checks (ceiling positive,
        # armed floor strictly below ceiling, mutual exclusion with the retired
        # unconditional floor) are re-run by Stage25UpdateConfig/PIController;
        # the run-level invariants here are the ARM parameter coupling, the
        # window fit, and the AM-H13 family scoping.
        if self.lambda_ceiling is not None:
            object.__setattr__(
                self,
                "lambda_ceiling",
                require_positive_number(
                    "lambda_ceiling",
                    self.lambda_ceiling,
                    error_suffix="must be positive and finite",
                ),
            )
        arm_params = (
            self.arm_success_threshold,
            self.arm_hazard_threshold,
            self.arm_sensing_threshold,
        )
        if self.armed_lambda_floor is None:
            if (
                int.__index__(self.arm_window_rounds) != 0
                or any(p is not None for p in arm_params)
                or int.__index__(self.dearm_window_rounds) != 0
                or self.dearm_success_threshold is not None
            ):
                raise ValueError(
                    "arm_*/dearm_* parameters require armed_lambda_floor "
                    "(DR-H3 lever H: orphan arm parameters would silently "
                    "configure nothing)"
                )
        else:
            armed_floor_value = require_positive_number(
                "armed_lambda_floor",
                self.armed_lambda_floor,
                error_suffix="must be positive and finite",
            )
            require_positive_int("arm_window_rounds", self.arm_window_rounds)
            require_positive_int("dearm_window_rounds", self.dearm_window_rounds)
            if any(p is None for p in arm_params) or (
                self.dearm_success_threshold is None
            ):
                raise ValueError(
                    "armed_lambda_floor requires all of arm_success_threshold / "
                    "arm_hazard_threshold / arm_sensing_threshold / "
                    "dearm_window_rounds / dearm_success_threshold (DR-H3: the "
                    "arm and its AM-H4 hatch travel together)"
                )
            arm_success = require_positive_number(
                "arm_success_threshold",
                self.arm_success_threshold,
                error_suffix="must be positive and finite",
            )
            if arm_success > 1.0:
                raise ValueError("arm_success_threshold must be <= 1.0")
            arm_hazard = require_nonnegative_number(
                "arm_hazard_threshold",
                self.arm_hazard_threshold,
                error_suffix="must be nonnegative and finite",
            )
            arm_sensing = require_nonnegative_number(
                "arm_sensing_threshold",
                self.arm_sensing_threshold,
                error_suffix="must be nonnegative and finite",
            )
            dearm_success = require_nonnegative_number(
                "dearm_success_threshold",
                self.dearm_success_threshold,
                error_suffix="must be nonnegative and finite",
            )
            if dearm_success >= arm_success:
                raise ValueError(
                    "dearm_success_threshold must be strictly below "
                    "arm_success_threshold (DR-H3 AM-H4: the dwell asymmetry "
                    "that excludes chattering requires disjoint bands)"
                )
            if (
                int.__index__(self.constraint_warmup_rounds)
                + int.__index__(self.arm_window_rounds)
                >= update_rounds
            ):
                raise ValueError(
                    "constraint_warmup_rounds + arm_window_rounds must be < "
                    "update_rounds (the first arm-eligible window must fit "
                    "inside the run; DR-H3)"
                )
            # Bound the release window too (the DRREL huge-int lesson: an
            # unbounded window length would reach deque(maxlen=...) as a raw
            # OverflowError through an otherwise-valid config).
            if int.__index__(self.dearm_window_rounds) > update_rounds:
                raise ValueError(
                    "dearm_window_rounds must be <= update_rounds (DR-H3 "
                    "AM-H4: the release window must fit inside the run)"
                )
            # A-14/A-3 discipline: sanitize-store the armed-lever fields so
            # the arm arithmetic never dispatches to hostile numeric
            # subclasses (the require_* helpers return the base values).
            object.__setattr__(self, "armed_lambda_floor", armed_floor_value)
            object.__setattr__(
                self, "arm_window_rounds", int.__index__(self.arm_window_rounds)
            )
            object.__setattr__(self, "arm_success_threshold", arm_success)
            object.__setattr__(self, "arm_hazard_threshold", arm_hazard)
            object.__setattr__(self, "arm_sensing_threshold", arm_sensing)
            object.__setattr__(
                self,
                "dearm_window_rounds",
                int.__index__(self.dearm_window_rounds),
            )
            object.__setattr__(self, "dearm_success_threshold", dearm_success)
            # AM-H13 (countersign-blocking) as RE-SCOPED by the countersigned
            # DR-READINESS-CURRICULUM §3.3: the arm's discrimination is
            # inherited from the PER-PAIR fork cut theorem — it FALSE-FIRES on
            # legacy maps (measured: t1_d5_s108) — so the armed floor accepts
            # only the fork TRAINING family plus the countersigned curriculum
            # pairs, and only in configurations that preserve the within-pair
            # aliasing average the theorem's blind-hazard margins consume:
            # (1) family scope; (2) COMPLETE pairs (both mirrors of every
            # included pair); (3) exact within-pair per-round episode balance
            # under the round-robin law (counts AGGREGATED per scenario NAME
            # across duplicate tuple positions; every included name trained).
            # Any FUTURE family extension re-opens the arm spec again
            # (ED-H3 §3.5).
            _fork_training = frozenset(fork_hazard_layout_families()["training"])
            _fork_curriculum = frozenset(fork_curriculum_scenarios())
            _arm_admissible = _fork_training | _fork_curriculum
            if not all(name in _arm_admissible for name in self.scenario_names):
                raise ValueError(
                    "armed_lambda_floor is scoped to the fork TRAINING family "
                    "plus the countersigned curriculum pairs (AM-H13 as "
                    "re-scoped by DR-READINESS-CURRICULUM: the arm's validity "
                    "is theorem-backed only there; it false-fires on legacy "
                    "maps)"
                )
            _name_counts: dict[str, int] = {}
            _tuple_len = len(self.scenario_names)
            for _episode_index in range(episodes_per_round):
                _episode_name = self.scenario_names[_episode_index % _tuple_len]
                _name_counts[_episode_name] = _name_counts.get(_episode_name, 0) + 1
            if any(_name_counts.get(name, 0) < 1 for name in self.scenario_names):
                raise ValueError(
                    "armed_lambda_floor requires every training scenario to "
                    "receive >= 1 episode per round under the round-robin law "
                    "(an untrained pair is vacuously balanced but breaks the "
                    "aliasing average; DR-READINESS-CURRICULUM AM-C7)"
                )
            # Deterministic iteration (audit L2-SF-1): set() order is
            # PYTHONHASHSEED-dependent, which would make the SELECTED error
            # message nondeterministic on multi-violation configs.
            _pair_mirrors: dict[str, dict[str, str]] = {}
            for name in sorted(set(self.scenario_names)):
                _pair_base, _, _mirror_suffix = name.rpartition("_")
                _pair_mirrors.setdefault(_pair_base, {})[_mirror_suffix] = name
            for _pair_base, _mirrors in _pair_mirrors.items():
                if set(_mirrors) != {"lower", "upper"}:
                    raise ValueError(
                        "armed_lambda_floor requires COMPLETE fork pairs - "
                        "both the _lower and _upper mirror of every included "
                        "pair (a single mirror breaks the aliasing average; "
                        "DR-READINESS-CURRICULUM section 3.3)"
                    )
                if (
                    _name_counts.get(_mirrors["lower"], 0)
                    != _name_counts.get(_mirrors["upper"], 0)
                ):
                    raise ValueError(
                        "armed_lambda_floor requires EQUAL per-round episode "
                        "counts for the two mirrors of every fork pair under "
                        "the round-robin law (within-pair balance; "
                        "DR-READINESS-CURRICULUM section 3.3)"
                    )
            # DR-CURRICULUM-FORM (SHIFT 29, countersigned): the F-84-safe N<=2
            # bound, CODIFIED. The armed floor reads the ROUND-POOLED hazard
            # (pooled = sum_p w_p * h_p over the round's episodes); a single
            # blind fork pair (per-pair team hazard 1.0) among N equal-weight
            # pairs pools to ~1/N, which stays ABOVE the 0.35 arm conjunct only
            # for N <= 2 (1/N > 0.35 <=> N < 2.857). At N >= 3 a realizable
            # blind-pair weight (<= 0.375 under 16-ep mirror balance; the T1-D14
            # 4-pair leak was 0.25) dips <= 0.35, so the arm can fire on a
            # PARTIAL mixture and the floor's over-budget pressure on the
            # unassembled remainder collapses the policy (F-84, measured on the
            # T1-D14 4-pair family). So the armed floor accepts at most TWO
            # distinct fork pairs. The count is len(_pair_mirrors) (distinct pair
            # BASES, AM-CF5) -- NOT len(scenario_names) -- so a static family is
            # counted by pairs (4 scenarios / 2 pairs = N=2 accepted). Input is
            # ONLY scenario_names (a config property, ungameable). Per-pair
            # arm-gating (each pair's own window under the conjunct) is the
            # F-84-safe way to exceed N=2; a FUTURE DR that adds it relaxes this
            # bound under its own countersign (DR-curriculum-form.md sec 3.2).
            if len(_pair_mirrors) > 2:
                raise ValueError(
                    "armed_lambda_floor accepts at most 2 distinct fork pairs "
                    "(the F-84-safe bound: a single blind pair among N pairs "
                    "pools to ~1/N, above the 0.35 arm conjunct only for "
                    "N <= 2; N >= 3 admits a partial-mixture arm fire that "
                    "collapses the policy; DR-CURRICULUM-FORM section 3.2)"
                )
        # DR-H5 lever J: the budget value (sanitize-stored; re-validated by
        # Stage25UpdateConfig / PIController downstream).
        hazard_budget_value = require_positive_number(
            "hazard_budget",
            self.hazard_budget,
            error_suffix="must be positive and finite",
        )
        if hazard_budget_value > 0.5:
            raise ValueError(
                "hazard_budget must be <= 0.5 (the DR-D2 budget; training "
                "looser than the graded bars is a misconfiguration — DR-H5)"
            )
        object.__setattr__(self, "hazard_budget", hazard_budget_value)
        # DR-H5 lever K run-level coupling: the gate and its midband trigger
        # parameters are all-or-nothing (a partially-configured trigger would
        # silently run wrong containment arithmetic), and recovery is an
        # ARMED-phase device layered on a RUNNING movement sustain.
        if not isinstance(self.movement_recovery_gate, bool):
            raise TypeError("movement_recovery_gate must be a bool")
        midband_params = (
            self.midband_recovery_success_lo,
            self.midband_recovery_success_hi,
            self.midband_recovery_containment_widen,
            self.midband_recovery_hazard_cap,
        )
        if not self.movement_recovery_gate:
            if (
                int.__index__(self.midband_recovery_window_rounds) != 0
                or any(p is not None for p in midband_params)
                or int.__index__(self.midband_recovery_max_rounds) != 0
            ):
                raise ValueError(
                    "midband_recovery_* parameters require "
                    "movement_recovery_gate=True (DR-H5: the trigger, latch, "
                    "and support engage together or not at all)"
                )
        else:
            if self.armed_lambda_floor is None:
                raise ValueError(
                    "movement_recovery_gate=True requires armed_lambda_floor "
                    "to be configured (recovery is an ARMED-phase device; "
                    "DR-H5)"
                )
            if self.entropy_sustain_learning_rate <= 0.0:
                raise ValueError(
                    "movement_recovery_gate=True requires "
                    "entropy_sustain_learning_rate > 0 (the recovery override "
                    "re-engages a RUNNING movement sustain; DR-H5)"
                )
            if self.movement_entropy_sustain_target is None:
                raise ValueError(
                    "movement_recovery_gate=True requires "
                    "movement_entropy_sustain_target to be set (the recovery "
                    "override controls the movement factor; DR-H5)"
                )
            require_positive_int(
                "midband_recovery_window_rounds",
                self.midband_recovery_window_rounds,
            )
            if int.__index__(self.midband_recovery_window_rounds) > update_rounds:
                raise ValueError(
                    "midband_recovery_window_rounds must be <= update_rounds "
                    "(DR-H5: the trigger window must fit inside the run)"
                )
            require_positive_int(
                "midband_recovery_max_rounds", self.midband_recovery_max_rounds
            )
            if int.__index__(self.midband_recovery_max_rounds) > update_rounds:
                raise ValueError(
                    "midband_recovery_max_rounds must be <= update_rounds "
                    "(DR-H5: an engagement must fit inside the run)"
                )
            if any(p is None for p in midband_params):
                raise ValueError(
                    "movement_recovery_gate=True requires ALL of "
                    "midband_recovery_success_lo / midband_recovery_success_hi "
                    "/ midband_recovery_containment_widen / "
                    "midband_recovery_hazard_cap (DR-H5)"
                )
            midband_lo = require_nonnegative_number(
                "midband_recovery_success_lo",
                self.midband_recovery_success_lo,
                error_suffix="must be nonnegative and finite",
            )
            midband_hi = require_nonnegative_number(
                "midband_recovery_success_hi",
                self.midband_recovery_success_hi,
                error_suffix="must be nonnegative and finite",
            )
            if not (0.0 <= midband_lo < midband_hi <= 1.0):
                raise ValueError(
                    "midband_recovery_success_lo/hi must satisfy "
                    "0 <= lo < hi <= 1 (DR-H5: a success-rate band)"
                )
            midband_widen = require_nonnegative_number(
                "midband_recovery_containment_widen",
                self.midband_recovery_containment_widen,
                error_suffix="must be nonnegative and finite",
            )
            midband_cap = require_nonnegative_number(
                "midband_recovery_hazard_cap",
                self.midband_recovery_hazard_cap,
                error_suffix="must be nonnegative and finite",
            )
            # The recovery exit (a) reuses the arm's own success bar; the
            # band must sit strictly below it or the exit would fire inside
            # the qualifying band itself.
            if midband_hi >= float(self.arm_success_threshold):
                raise ValueError(
                    "midband_recovery_success_hi must be strictly below "
                    "arm_success_threshold (DR-H5: exit (a) reuses the arm's "
                    "success bar)"
                )
            # A-14/A-3 discipline: sanitize-store (base floats/ints only).
            object.__setattr__(
                self,
                "midband_recovery_window_rounds",
                int.__index__(self.midband_recovery_window_rounds),
            )
            object.__setattr__(self, "midband_recovery_success_lo", midband_lo)
            object.__setattr__(self, "midband_recovery_success_hi", midband_hi)
            object.__setattr__(
                self, "midband_recovery_containment_widen", midband_widen
            )
            object.__setattr__(self, "midband_recovery_hazard_cap", midband_cap)
            object.__setattr__(
                self,
                "midband_recovery_max_rounds",
                int.__index__(self.midband_recovery_max_rounds),
            )
        # A-14: store the sanitized base seed. Otherwise the per-round seed
        # arithmetic (config.seed + round_index * _ROUND_SEED_STRIDE) would
        # dispatch to a hostile int-subclass __add__, letting a lying subclass
        # control every round seed while the manifest records only the base
        # value. Normalizing here makes the recorded seed exactly the integer
        # every reseed uses.
        object.__setattr__(self, "seed", int.__index__(self.seed))


# ---------------------------------------------------------------------------
# EV / probe carrier
# ---------------------------------------------------------------------------


def _ev_probe_carrier_config(update_config: Stage25UpdateConfig) -> Stage22UpdateConfig:
    """Return a ``Stage22UpdateConfig`` CARRIER for the reused EV/probe entries.

    ``explained_variance_from_batch`` and ``run_readiness_probe`` type-gate on
    ``Stage22UpdateConfig`` and read ONLY ``algorithm`` (gamma / lambda_GAE)
    and ``losses``; this carrier transports the Stage 25 algorithm and losses
    into them so the EV regression targets match the Stage 25 update's own
    GAE (DR-D10: the EV diagnostic must reproduce the update's advantages).
    The lagrange / learning-rate / epoch fields below are inert placeholders
    never read by those entry points — the dev update itself is NOT called
    with this carrier anywhere.
    """

    return Stage22UpdateConfig(
        algorithm=update_config.algorithm,
        losses=update_config.losses,
        lagrange=LagrangeConfig(
            initial_multiplier=0.1, learning_rate=0.2, hazard_budget=0.1
        ),
    )


# ---------------------------------------------------------------------------
# DR-H3 lever H: the driver-owned armed-floor state
# ---------------------------------------------------------------------------


def _round_behavior_aggregates(
    episode_records: "list[dict[str, Any]] | tuple[dict[str, Any], ...]",
) -> tuple[float, float, float]:
    """The (success_rate, hazard_cost_sum_mean, sensing_rate_mean) triple.

    SINGLE SOURCE for both the curve record and the DR-H3 arm (AM-H15): the
    arm consumes byte-identical values to what the curve row records, so the
    post-hoc MECH-H curve scan is the reference implementation for the
    wiring-vs-science distinction. The arithmetic is exactly the historical
    ``_curve_record`` forms (float-cast sums over episode records).
    """

    episode_count = len(episode_records)
    success_count = sum(1 for record in episode_records if record["team_success"])
    success_rate = float(success_count) / float(episode_count)
    hazard_mean = float(
        sum(float(record["hazard_cost_sum"]) for record in episode_records)
    ) / float(episode_count)
    sensing_mean = float(
        sum(float(record["sensing_rate"]) for record in episode_records)
    ) / float(episode_count)
    return success_rate, hazard_mean, sensing_mean


class _ArmedFloorState:
    """DR-H3 lever H: the measured-regime latch + the AM-H4 de-arm hatch.

    A behavioral-basin classifier over the trailing per-round aggregate
    windows — never an epistemic device (terminology guardrail: nothing here
    "knows" or "believes" anything about the policy). ARM fires at the first
    round k whose trailing W-round window (rounds [k-W+1, k], INCLUSIVE of
    the just-completed round, entirely post-warmup because appends start at
    round == warmup; AM-H15 convention) has mean success >=
    arm_success_threshold AND mean episodic hazard <= arm_hazard_threshold
    AND mean sensing rate >= arm_sensing_threshold. While armed, a trailing
    ``dearm_window_rounds`` window with mean success <= dearm_success_threshold
    RELEASES the floor (event logged; re-arm allowed by the same rule —
    chattering is structurally excluded by the dwell asymmetry). The armed
    flag is applied to the SAME round's dual step (fire round included).
    Earliest eligible fire (0-based round_index) = warmup + W - 1 — the
    (warmup+W)-th round, whose window is [warmup, warmup+W-1]; e.g. round 349
    at the T1-D11 values (warmup 300, W 50). A round-(warmup+W-1) fire is
    LEGITIMATE, not floor-application-before-fire (impl-audit lens-1 pin).

    State (latch + windows + event log) is checkpoint-persisted via
    ``to_payload``/``from_payload``; an ON-config resume REQUIRES the payload
    (the AM-2-style loud reject lives at the restore site).
    """

    def __init__(
        self,
        *,
        arm_window_rounds: int,
        arm_success_threshold: float,
        arm_hazard_threshold: float,
        arm_sensing_threshold: float,
        dearm_window_rounds: int,
        dearm_success_threshold: float,
        constraint_warmup_rounds: int,
        midband_window_rounds: int = 0,
        midband_success_lo: "float | None" = None,
        midband_success_hi: "float | None" = None,
        midband_containment_widen: "float | None" = None,
        midband_hazard_cap: "float | None" = None,
        midband_max_rounds: int = 0,
    ) -> None:
        # S2-5 parity: normalize through the base slots so hostile numeric
        # subclasses cannot ride into the window arithmetic (the config layer
        # sanitize-stores too; this covers direct construction).
        self.arm_window_rounds = int.__index__(arm_window_rounds)
        self.arm_success_threshold = float(arm_success_threshold)
        self.arm_hazard_threshold = float(arm_hazard_threshold)
        self.arm_sensing_threshold = float(arm_sensing_threshold)
        self.dearm_window_rounds = int.__index__(dearm_window_rounds)
        self.dearm_success_threshold = float(dearm_success_threshold)
        self.constraint_warmup_rounds = int.__index__(constraint_warmup_rounds)
        self.armed = False
        self.events: list[dict[str, Any]] = []
        self._arm_window: deque[tuple[float, float, float]] = deque(
            maxlen=self.arm_window_rounds
        )
        self._dearm_window: deque[float] = deque(maxlen=self.dearm_window_rounds)
        # DR-H5 lever K: the midband containment trigger + latched recovery
        # mode. Feature ON iff midband_window_rounds > 0 (the run config
        # enforces all-or-nothing); with the feature OFF every recovery
        # attribute stays at its inert default and to_payload() is
        # byte-identical to the pre-DR-H5 form.
        self.midband_window_rounds = int.__index__(midband_window_rounds)
        self.recovery_enabled = self.midband_window_rounds > 0
        if self.recovery_enabled:
            if (
                midband_success_lo is None
                or midband_success_hi is None
                or midband_containment_widen is None
                or midband_hazard_cap is None
            ):
                raise ValueError(
                    "midband recovery parameters are required when "
                    "midband_window_rounds > 0"
                )
            self.midband_success_lo = float(midband_success_lo)
            self.midband_success_hi = float(midband_success_hi)
            self.midband_containment_widen = float(midband_containment_widen)
            self.midband_hazard_cap = float(midband_hazard_cap)
            self.midband_max_rounds = int.__index__(midband_max_rounds)
            if self.midband_max_rounds <= 0:
                raise ValueError(
                    "midband_max_rounds must be positive when the recovery "
                    "trigger is enabled"
                )
        else:
            if (
                midband_success_lo is not None
                or midband_success_hi is not None
                or midband_containment_widen is not None
                or midband_hazard_cap is not None
                or int.__index__(midband_max_rounds) != 0
            ):
                raise ValueError(
                    "midband recovery parameters require "
                    "midband_window_rounds > 0"
                )
            self.midband_success_lo = None
            self.midband_success_hi = None
            self.midband_containment_widen = None
            self.midband_hazard_cap = None
            self.midband_max_rounds = 0
        self.recovery_active = False
        self.recovery_rounds_used = 0
        self.recovery_engagements = 0
        self.recovery_exhausted = False
        self._midband_window: deque[tuple[float, float]] = deque(
            maxlen=(self.midband_window_rounds if self.recovery_enabled else 1)
        )

    def window_means(self) -> tuple[float, float, float]:
        """Current trailing arm-window means (0.0 triple while empty)."""

        if not self._arm_window:
            return 0.0, 0.0, 0.0
        count = float(len(self._arm_window))
        return (
            sum(entry[0] for entry in self._arm_window) / count,
            sum(entry[1] for entry in self._arm_window) / count,
            sum(entry[2] for entry in self._arm_window) / count,
        )

    def observe_round(
        self,
        *,
        round_index: int,
        success_rate: float,
        hazard_mean: float,
        sensing_mean: float,
    ) -> bool:
        """Feed one completed round's aggregates; return the state to APPLY.

        The returned flag governs the SAME round's dual step (the fire round
        itself is floored — the AM-H15 convention MECH-H byte-checks).
        Warmup rounds are never appended, so every full window is entirely
        post-warmup by construction.
        """

        if round_index < self.constraint_warmup_rounds:
            return False
        self._arm_window.append(
            (float(success_rate), float(hazard_mean), float(sensing_mean))
        )
        self._dearm_window.append(float(success_rate))
        if not self.armed:
            if len(self._arm_window) == self.arm_window_rounds:
                mean_success, mean_hazard, mean_sensing = self.window_means()
                if (
                    mean_success >= self.arm_success_threshold
                    and mean_hazard <= self.arm_hazard_threshold
                    and mean_sensing >= self.arm_sensing_threshold
                ):
                    self.armed = True
                    self.events.append(
                        {
                            "event": "arm",
                            "round_index": int(round_index),
                            "window_success": mean_success,
                            "window_hazard": mean_hazard,
                            "window_sensing": mean_sensing,
                        }
                    )
        else:
            if len(self._dearm_window) == self.dearm_window_rounds:
                dearm_mean = sum(self._dearm_window) / float(
                    len(self._dearm_window)
                )
                if dearm_mean <= self.dearm_success_threshold:
                    self.armed = False
                    self.events.append(
                        {
                            "event": "dearm",
                            "round_index": int(round_index),
                            "window_success": dearm_mean,
                        }
                    )
        if self.recovery_enabled:
            self._observe_recovery(round_index=int(round_index))
        return self.armed

    def _observe_recovery(self, *, round_index: int) -> None:
        """DR-H5 lever K: one round of the containment trigger + latch.

        Runs AFTER the arm/de-arm evaluation of the same round (the hatch
        takes precedence: a de-arm this round is recovery exit (b)). Window
        membership accumulates at EVERY post-warmup round REGARDLESS of arm
        status (the _ArmedFloorState convention the committed regression
        forces — s131's r1884 fire window contains 20 pre-arm rounds; an
        armed-only window could not fire before r1904). The engage round
        itself is a recovery round and the returned state governs the SAME
        round's update (the AM-H15 fire-round convention). Earliest
        legitimate fire = warmup + W_rec - 1 (r499 at the T1-D13 values).
        Exits: (a) trailing-50 (arm-window) mean success >=
        arm_success_threshold; (b) de-arm; (c) rounds_used >= R_max (sets the
        permanent exhaustion latch). Exit precedence on a same-round co-fire:
        dearm > recovered > exhausted (a dearm+R_max co-fire exits "dearm"
        and does NOT set the exhaustion latch; recovered-vs-dearm co-fire is
        arithmetically impossible with both windows full). The exit round's
        OWN update runs WITHOUT the override (the same-round convention,
        mirroring the arm fire round; rounds_used counts the exit round
        inclusively — the committed regression's r2683/800 target). On ANY
        exit the containment window and the per-engagement counter are
        CLEARED (re-engagement needs a full fresh window; V2-S2 — measured
        minimum engage-to-engage distance 201 rounds => hard bound ~18
        engagements in a 4000-round run).
        """

        # The most recent aggregates are the last arm-window entry (appended
        # this round; success and hazard are its first two members).
        success_rate, hazard_mean, _ = self._arm_window[-1]
        self._midband_window.append((success_rate, hazard_mean))
        if self.recovery_active:
            self.recovery_rounds_used += 1
            exit_reason: "str | None" = None
            if not self.armed:
                exit_reason = "dearm"
            elif (
                len(self._arm_window) == self.arm_window_rounds
                and self.window_means()[0] >= self.arm_success_threshold
            ):
                exit_reason = "recovered"
            elif self.recovery_rounds_used >= self.midband_max_rounds:
                exit_reason = "exhausted"
            if exit_reason is not None:
                self.recovery_active = False
                if exit_reason == "exhausted":
                    self.recovery_exhausted = True
                self.events.append(
                    {
                        "event": "recovery_exit",
                        "round_index": int(round_index),
                        "reason": exit_reason,
                        "rounds_used": int(self.recovery_rounds_used),
                    }
                )
                self.recovery_rounds_used = 0
                self._midband_window.clear()
            return
        if self.recovery_exhausted or not self.armed:
            return
        if len(self._midband_window) != self.midband_window_rounds:
            return
        widened_lo = max(0.0, self.midband_success_lo - self.midband_containment_widen)
        widened_hi = min(1.0, self.midband_success_hi + self.midband_containment_widen)
        if not all(
            widened_lo <= entry[0] <= widened_hi for entry in self._midband_window
        ):
            return
        window_count = float(len(self._midband_window))
        mean_success = (
            sum(entry[0] for entry in self._midband_window) / window_count
        )
        mean_hazard = (
            sum(entry[1] for entry in self._midband_window) / window_count
        )
        if (
            mean_hazard <= self.midband_hazard_cap
            and self.midband_success_lo <= mean_success <= self.midband_success_hi
        ):
            self.recovery_active = True
            self.recovery_rounds_used = 1
            self.recovery_engagements += 1
            self.events.append(
                {
                    "event": "recovery_engage",
                    "round_index": int(round_index),
                    "window_success": mean_success,
                    "window_hazard": mean_hazard,
                }
            )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "armed": bool(self.armed),
            "events": [dict(event) for event in self.events],
            "arm_window": [list(entry) for entry in self._arm_window],
            "dearm_window": list(self._dearm_window),
        }
        # DR-H5 lever K: recovery state travels with the checkpoint ONLY when
        # the trigger is enabled — the OFF payload is byte-identical to the
        # pre-DR-H5 form (OFF-default byte-identity).
        if self.recovery_enabled:
            payload["recovery"] = {
                "active": bool(self.recovery_active),
                "rounds_used": int(self.recovery_rounds_used),
                "engagements": int(self.recovery_engagements),
                "exhausted": bool(self.recovery_exhausted),
                "midband_window": [list(entry) for entry in self._midband_window],
            }
        return payload

    def restore_payload(self, payload: dict[str, Any]) -> None:
        """Restore latch + windows byte-identically from a checkpoint payload."""

        if not isinstance(payload, dict):
            raise ValueError("arm_state payload must be a mapping")
        for key in ("armed", "events", "arm_window", "dearm_window"):
            if key not in payload:
                raise ValueError(f"arm_state payload is missing {key}")
        if type(payload["armed"]) is not bool:
            raise ValueError("arm_state armed must be a bool")
        if len(payload["arm_window"]) > self.arm_window_rounds:
            raise ValueError("arm_state arm_window exceeds arm_window_rounds")
        if len(payload["dearm_window"]) > self.dearm_window_rounds:
            raise ValueError("arm_state dearm_window exceeds dearm_window_rounds")
        # DR-H5 lever K: an ON-trigger resume REQUIRES the recovery state (the
        # AM-2 loud-reject convention — a silent reset would erase the latch,
        # the per-engagement counter, and the exhaustion state mid-run); an
        # OFF-trigger payload must NOT carry one (config/state mismatch — the
        # V-4 fingerprint rejects config drift upstream, this names the state).
        if self.recovery_enabled:
            recovery = payload.get("recovery")
            if not isinstance(recovery, dict):
                raise ValueError(
                    "arm_state payload is missing the recovery state (the "
                    "midband trigger is enabled; DR-H5 AM-2 loud reject)"
                )
            for key in (
                "active",
                "rounds_used",
                "engagements",
                "exhausted",
                "midband_window",
            ):
                if key not in recovery:
                    raise ValueError(
                        f"arm_state recovery state is missing {key}"
                    )
            if type(recovery["active"]) is not bool:
                raise ValueError("arm_state recovery active must be a bool")
            if type(recovery["exhausted"]) is not bool:
                raise ValueError(
                    "arm_state recovery exhausted must be a bool"
                )
            # L1-MF-1 (impl-audit): domain-validate the counters — a forged
            # checkpoint must not overrun R_max, forge the permanent
            # exhaustion latch (a graded routing input), or restore a
            # machine-unreachable active state. Every legitimately produced
            # payload satisfies these invariants by construction (100k-round
            # fuzz-verified).
            if (
                type(recovery["rounds_used"]) is not int
                or type(recovery["engagements"]) is not int
            ):
                raise ValueError(
                    "arm_state recovery counters must be ints (not bools)"
                )
            if not (0 <= recovery["rounds_used"] <= self.midband_max_rounds):
                raise ValueError(
                    "arm_state recovery rounds_used must be within "
                    "[0, midband_max_rounds]"
                )
            if recovery["engagements"] < 0:
                raise ValueError(
                    "arm_state recovery engagements must be nonnegative"
                )
            if recovery["active"] and (
                payload["armed"] is not True
                or recovery["rounds_used"] < 1
                or recovery["exhausted"]
            ):
                raise ValueError(
                    "arm_state recovery active state is machine-unreachable "
                    "(active requires armed, rounds_used >= 1, not exhausted)"
                )
            if not recovery["active"] and recovery["rounds_used"] != 0:
                raise ValueError(
                    "arm_state recovery rounds_used must be 0 while inactive"
                )
            if len(recovery["midband_window"]) > self.midband_window_rounds:
                raise ValueError(
                    "arm_state recovery midband_window exceeds "
                    "midband_window_rounds"
                )
        elif "recovery" in payload:
            raise ValueError(
                "arm_state payload carries recovery state but the midband "
                "trigger is not enabled (config/state mismatch; DR-H5)"
            )
        self.armed = payload["armed"]
        self.events = [dict(event) for event in payload["events"]]
        self._arm_window.clear()
        for entry in payload["arm_window"]:
            if len(entry) != 3:
                raise ValueError("arm_state arm_window entries must be triples")
            self._arm_window.append(
                (float(entry[0]), float(entry[1]), float(entry[2]))
            )
        self._dearm_window.clear()
        for value in payload["dearm_window"]:
            self._dearm_window.append(float(value))
        if self.recovery_enabled:
            recovery = payload["recovery"]
            self.recovery_active = recovery["active"]
            self.recovery_rounds_used = int.__index__(recovery["rounds_used"])
            self.recovery_engagements = int.__index__(recovery["engagements"])
            self.recovery_exhausted = recovery["exhausted"]
            self._midband_window.clear()
            for entry in recovery["midband_window"]:
                if len(entry) != 2:
                    raise ValueError(
                        "arm_state recovery midband_window entries must be "
                        "pairs"
                    )
                self._midband_window.append((float(entry[0]), float(entry[1])))


# ---------------------------------------------------------------------------
# Manifest / curve payload builders
# ---------------------------------------------------------------------------


def _config_payload(
    config: Stage25RunConfig,
    update_config: Stage25UpdateConfig,
) -> dict[str, Any]:
    """Build the JSON-compatible manifest ``config`` block."""

    payload = dataclasses.asdict(config)
    payload["result_parent"] = (
        None if config.result_parent is None else str(config.result_parent)
    )
    payload["scenario_names"] = list(config.scenario_names)
    payload["stage25_update_config"] = dataclasses.asdict(update_config)
    payload["torch_version"] = str(torch.__version__)
    return payload


def _curve_record(
    *,
    round_index: int,
    round_seed: int,
    wall_time_s: float,
    episode_records: tuple[dict[str, Any], ...],
    update_summary: dict[str, Any],
    explained_variance: dict[str, float | bool],
    entropy_floor: float,
    state_saved: bool,
    arm_window_means: tuple[float, float, float] = (0.0, 0.0, 0.0),
    recovery_rounds_used: int = 0,
    recovery_exhausted: bool = False,
) -> dict[str, Any]:
    """Build one Stage 25 training-curve JSONL record for a completed round.

    Carries every DR-mandated per-round field: the DR-D3 dual trace
    (``observed_episodic_cost`` = J_C-hat, ``dual_error`` = e_k,
    ``dual_integral`` = I_k, ``lambda_applied``), the DR-D9/DR-D4 guard trace
    (``epochs_used``, ``approx_kl_final``, ``approx_kl_per_epoch``,
    ``kl_early_stopped``), the DR-D14 trigger field
    (``grad_norm_clip_activation_fraction``, graded on gradient-norm clip
    activation per V-1, with ``grad_norm_pre_clip_per_epoch`` and the
    honest-named PPO-ratio diagnostic ``ratio_clip_fraction`` alongside), the
    DR-D10 EV fields with the zero-variance guard, the raw_*/policy_*
    advantage means (DR-D12: no legacy duplicate keys), and the per-loss
    scalars.
    """

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
        "episode_count": episode_count,
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
        "observed_episodic_cost": float(update_summary["observed_episodic_cost"]),
        "dual_error": float(update_summary["dual_error"]),
        "dual_integral": float(update_summary["dual_integral"]),
        "lambda_applied": float(update_summary["lambda_applied"]),
        "epochs_used": int(update_summary["epochs_used"]),
        "approx_kl_final": float(update_summary["approx_kl_final"]),
        "approx_kl_per_epoch": [
            float(value) for value in update_summary["approx_kl_per_epoch"]
        ],
        "kl_early_stopped": bool(update_summary["kl_early_stopped"]),
        "grad_norm_clip_activation_fraction": float(
            update_summary["grad_norm_clip_activation_fraction"]
        ),
        "grad_norm_pre_clip_per_epoch": [
            float(value) for value in update_summary["grad_norm_pre_clip_per_epoch"]
        ],
        "ratio_clip_fraction": float(update_summary["ratio_clip_fraction"]),
        "policy_loss": float(update_summary["policy_loss"]),
        "reward_value_loss": float(update_summary["reward_value_loss"]),
        "hazard_cost_value_loss": float(update_summary["hazard_cost_value_loss"]),
        "sensing_entropy": sensing_entropy,
        "movement_entropy": movement_entropy,
        "total_loss": float(update_summary["total_loss"]),
        "parameter_delta_l1": float(update_summary["parameter_delta_l1"]),
        "raw_reward_advantage_mean": float(
            update_summary["raw_reward_advantage_mean"]
        ),
        "raw_hazard_cost_advantage_mean": float(
            update_summary["raw_hazard_cost_advantage_mean"]
        ),
        "policy_reward_advantage_mean": float(
            update_summary["policy_reward_advantage_mean"]
        ),
        "policy_hazard_cost_advantage_mean": float(
            update_summary["policy_hazard_cost_advantage_mean"]
        ),
        # DR-C4 attribution diagnostics (auxiliary; never a graded metric). The
        # reward-advantage std should be > 0 once shaping is on (Link-5 repair);
        # constraint_warmup_active marks the tail-excluded warmup rounds.
        "raw_reward_advantage_std": float(
            update_summary["raw_reward_advantage_std"]
        ),
        "reward_shaping_applied": bool(update_summary["reward_shaping_applied"]),
        "reward_shaping_mean": float(update_summary["reward_shaping_mean"]),
        # DR-D1/L1-b auxiliary diagnostics (never graded fields -- AM-2).
        "sensing_shaping_applied": bool(
            update_summary["sensing_shaping_applied"]
        ),
        "sensing_shaping_mean": float(update_summary["sensing_shaping_mean"]),
        "sensing_keepalive_active": bool(
            update_summary["sensing_keepalive_active"]
        ),
        "constraint_warmup_active": bool(
            update_summary["constraint_warmup_active"]
        ),
        # DR-RELIABILITY diagnostics (auxiliary; never graded fields). The
        # lever-E temperatures/targets/measured entropies trace the sustain
        # controller per round; lambda_floor traces the lever-F floor.
        "entropy_sustain_active": bool(update_summary["entropy_sustain_active"]),
        "entropy_sustain_alpha_movement": float(
            update_summary["entropy_sustain_alpha_movement"]
        ),
        "entropy_sustain_alpha_sensing": float(
            update_summary["entropy_sustain_alpha_sensing"]
        ),
        "entropy_sustain_measured_movement_entropy": float(
            update_summary["entropy_sustain_measured_movement_entropy"]
        ),
        "entropy_sustain_measured_sensing_entropy": float(
            update_summary["entropy_sustain_measured_sensing_entropy"]
        ),
        "entropy_sustain_movement_target": float(
            update_summary["entropy_sustain_movement_target"]
        ),
        "entropy_sustain_sensing_target": float(
            update_summary["entropy_sustain_sensing_target"]
        ),
        # DR-H4 lever I flag (unconditional passthrough).
        "sensing_sustain_armed_gate": bool(
            update_summary["sensing_sustain_armed_gate"]
        ),
        # DR-H5 lever K flag (unconditional passthrough — lever-I parity: the
        # config visible per-round; the per-round latch + counters are emitted
        # with the arm-window keys below).
        "movement_recovery_gate": bool(
            update_summary["movement_recovery_gate"]
        ),
        "lambda_floor": float(update_summary["lambda_floor"]),
        # DR-H3 curve keys (unconditional; with the levers OFF they reduce to
        # lambda_applied / False / 0.0). lambda_pre_clamp is the AM-H9 row-3
        # read; the arm_window_* means are the driver's trailing arm-window
        # state (the MECH-H reference scan recomputes them from the
        # success_rate / hazard_cost_sum_mean / sensing_rate_mean keys above —
        # same single-source aggregates).
        "lambda_pre_clamp": float(update_summary["lambda_pre_clamp"]),
        "lambda_ceiling_engaged": bool(update_summary["lambda_ceiling_engaged"]),
        "lambda_floor_armed": bool(update_summary["lambda_floor_armed"]),
        "arm_window_success": float(arm_window_means[0]),
        "arm_window_hazard": float(arm_window_means[1]),
        "arm_window_sensing": float(arm_window_means[2]),
        # DR-H5 lever K diagnostics (UNCONDITIONAL keys — with the gate OFF
        # they reduce to False/0/False; the trigger itself is byte-replicable
        # from success_rate / hazard_cost_sum_mean / lambda_floor_armed, the
        # MECH-REC convention).
        "movement_recovery_active": bool(
            update_summary["movement_recovery_active"]
        ),
        "movement_recovery_rounds_used": int(recovery_rounds_used),
        "movement_recovery_exhausted": bool(recovery_exhausted),
        **dict(explained_variance),
        "entropy_collapse_flag": bool(
            sensing_entropy + movement_entropy < entropy_floor
        ),
        **stage25_boundary_flags(state_saved=state_saved),
    }


# ---------------------------------------------------------------------------
# Training-state save / resume
# ---------------------------------------------------------------------------


def _config_fingerprint(
    config: Stage25RunConfig, update_config: Stage25UpdateConfig
) -> str:
    """Return the deterministic sha256 of the run + update configuration (V-4).

    Hashes the manifest ``config`` block minus ``torch_version`` (environment
    provenance) and ``result_parent`` (A-9: the run-directory location, not a
    training-semantics field — the run dir is located independently on resume,
    so including the lexical parent spelling would spuriously reject a resume
    launched with an equivalent-but-differently-spelled parent). Saved into
    every training state and compared on resume, so a resume launched with a
    different training configuration fails loudly instead of silently
    continuing under changed semantics.
    """

    payload = _config_payload(config, update_config)
    payload.pop("torch_version")
    payload.pop("result_parent", None)
    return hashlib.sha256(
        deterministic_json_string(payload).encode("ascii")
    ).hexdigest()


def _reconcile_resumed_log(path: Path, *, start_round: int) -> int:
    """Drop resumed-log records for rounds the resume will re-run (V-3).

    A crash between the last state save and the crash point leaves
    curve/episode/probe records for rounds >= the resume round; the resumed
    process re-runs those rounds, so keeping the stale records would duplicate
    them (append-only logs). Keeps exactly the records with
    ``round_index < start_round``, tolerates one partial FINAL line (a
    mid-write crash artifact), rejects any other malformed content, and
    rewrites atomically (write-then-replace). Returns the dropped line count.
    """

    if not isinstance(start_round, int) or isinstance(start_round, bool):
        raise TypeError("start_round must be an int")
    # A-12: remove any stale reconciliation temp left by a crash between a
    # prior write_text and replace (the runs-root .gitignore does not match
    # ``*.resume_tmp``, so an orphan could otherwise be accidentally staged).
    stale_tmp = path.with_name(path.name + ".resume_tmp")
    if stale_tmp.exists():
        stale_tmp.unlink()
    if not path.is_file():
        return 0
    lines = path.read_text(encoding="utf-8").split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    kept: list[str] = []
    dropped = 0
    for position, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            if position == len(lines) - 1:
                dropped += 1  # partial final line from a mid-write crash
                continue
            raise ValueError(
                f"{path.name} contains a malformed non-final record"
            ) from None
        if not isinstance(record, dict):
            raise ValueError(f"{path.name} record must be a JSON object")
        round_index = record.get("round_index")
        if not isinstance(round_index, int) or isinstance(round_index, bool):
            raise ValueError(
                f"{path.name} record is missing an integer round_index"
            )
        if round_index < start_round:
            kept.append(line)
        else:
            dropped += 1
    if dropped:
        reconciled = path.with_name(path.name + ".resume_tmp")
        reconciled.write_text(
            "".join(line + "\n" for line in kept), encoding="utf-8"
        )
        reconciled.replace(path)
    return dropped


def _save_training_state(
    run_dir: Path,
    *,
    run_id: str,
    config_sha256: str,
    completed_rounds: int,
    model: RecurrentMAPPOActorCritic,
    optimizer: torch.optim.Optimizer,
    controller: PIController,
    entropy_controller: "EntropySustainController | None" = None,
    arm_state: "_ArmedFloorState | None" = None,
) -> dict[str, Any]:
    """Serialize the training state and append its sha256 record to the log.

    The ``.pt`` payload carries model / optimizer / controller / completed
    round count / torch RNG state (the DR-D4 checkpoint inventory) plus the
    configuration fingerprint (V-4: resume verifies it). The appended
    ``state_log.jsonl`` record carries the file name, sha256, and byte size,
    and — truthfully — the state-saved boundary flags (the record describes
    an artifact that now exists).
    """

    state_path = run_dir / _STATE_FILE_TEMPLATE.format(
        completed_rounds=completed_rounds
    )
    payload = {
        "run_id": run_id,
        "config_sha256": config_sha256,
        "completed_rounds": int(completed_rounds),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "controller": {
            "proportional_gain": float(controller.proportional_gain),
            "integral_gain": float(controller.integral_gain),
            "integral": float(controller.integral),
            "budget": float(controller.budget),
            # DR-D3 AMENDMENT (SHIFT 16): persist the anti-windup ceiling so a
            # resumed run keeps the bound (None => OFF; a resumed cap must match
            # the config's lambda_integral_cap or the A-10 guard rejects it).
            "integral_cap": (
                None
                if controller.integral_cap is None
                else float(controller.integral_cap)
            ),
            # DR-RELIABILITY lever F: persist the sustain floor so a resumed
            # run keeps it (0.0 => OFF; a resumed floor must match the
            # config's lambda_floor or the A-10 guard rejects it).
            "lambda_floor": float(controller.lambda_floor),
            # DR-H3 levers G/H: persist both constants so a resumed run keeps
            # them (None => OFF; a resumed value must match the config or the
            # A-10 guard rejects it at the first update).
            "lambda_ceiling": (
                None
                if controller.lambda_ceiling is None
                else float(controller.lambda_ceiling)
            ),
            "armed_lambda_floor": (
                None
                if controller.armed_lambda_floor is None
                else float(controller.armed_lambda_floor)
            ),
        },
        # DR-RELIABILITY lever E: persist the entropy-sustain alpha STATE
        # (constants are config-pinned via the V-4 fingerprint and rebuilt
        # from the update config on resume). None => the lever is OFF. An
        # ON-config resume REQUIRES this state (loud reject — the AM-2
        # silent-reset lesson).
        "entropy_controller": (
            None
            if entropy_controller is None
            else {
                "alpha_movement": float(entropy_controller.alpha_movement),
                "alpha_sensing": float(entropy_controller.alpha_sensing),
            }
        ),
        # DR-H3 lever H: persist the driver-owned arm STATE (latch + both
        # window buffers + the event log; thresholds are config-pinned via the
        # V-4 fingerprint). None => the lever is OFF. An ON-config resume
        # REQUIRES this state (loud reject — the AM-2 silent-reset lesson
        # applied to the latch; AM-H15 mid-window byte-identity).
        "arm_state": (None if arm_state is None else arm_state.to_payload()),
        "torch_rng_state": torch.get_rng_state(),
    }
    torch.save(payload, state_path)
    record = {
        "saved_state_round": int(completed_rounds),
        "saved_state_file": state_path.name,
        "saved_state_sha256": sha256_file(state_path),
        "saved_state_bytes": int(state_path.stat().st_size),
        "saved_at_utc": utc_timestamp(),
        **stage25_boundary_flags(state_saved=True),
    }
    append_jsonl_record(run_dir / STATE_LOG_FILENAME, record)
    return record


def _load_latest_training_state(
    run_dir: Path, *, run_id: str, config_sha256: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load and sha256-verify the latest saved training state for resume.

    Verifies the file hash against the state-log record, the payload's
    ``run_id``, and (V-4) the payload's ``config_sha256`` against the resume
    process's configuration fingerprint — a resume launched with a different
    run or update configuration is rejected before any state is restored.
    """

    log_path = run_dir / STATE_LOG_FILENAME
    if not log_path.is_file():
        raise ValueError(
            "resume requires a state log with at least one saved training state"
        )
    records = load_strict_jsonl_objects(log_path)
    if not records:
        raise ValueError(
            "resume requires a state log with at least one saved training state"
        )
    last = records[-1]
    file_name = last.get("saved_state_file")
    if not isinstance(file_name, str) or not file_name:
        raise ValueError("state log record is missing saved_state_file")
    state_path = run_dir / file_name
    if state_path.parent != run_dir or not state_path.is_file():
        raise ValueError("saved training state file is missing")
    recorded_sha256 = last.get("saved_state_sha256")
    if sha256_file(state_path) != recorded_sha256:
        raise ValueError("saved training state sha256 mismatch")
    payload = torch.load(state_path, weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("saved training state payload must be a dict")
    if payload.get("run_id") != run_id:
        raise ValueError("saved training state belongs to a different run_id")
    if payload.get("config_sha256") != config_sha256:
        raise ValueError(
            "saved training state configuration does not match the resume "
            "configuration (V-4: a resumed run must be launched with the "
            "same run and update configuration as the original run)"
        )
    return payload, last


def _resolve_run_directory_for_resume(
    run_id: str, result_parent: str | Path | None
) -> Path:
    """Resolve an EXISTING run directory for resume.

    Reuses ``resolve_run_directory``'s parent-resolution and location rules
    verbatim by resolving a fresh probe id under the same parent (the shared
    resolver deliberately raises on an existing run directory, which is
    exactly the state resume requires), then rejoining the real run id.
    """

    validate_run_id(run_id)
    probe_run_id = f"resumeprobe{os.getpid()}x{int(time.time() * 1000) % 10**9}"
    probe_dir = resolve_run_directory(probe_run_id, result_parent)
    run_dir = probe_dir.parent / run_id
    if not run_dir.is_dir():
        raise ValueError("resume requested but the run directory does not exist")
    return run_dir


# ---------------------------------------------------------------------------
# The Stage 25 training driver
# ---------------------------------------------------------------------------


def _resolve_probe_scenarios(
    scenario_names: tuple[str, ...],
) -> tuple[str, ...] | None:
    """Resolve the readiness-probe surface tuple for a training scenario set.

    Fork-family runs probe the fork surfaces; anything else keeps the legacy
    probe (``None``). The tuple is training + readiness for a training-pair-only
    run (byte-identical to the pre-curriculum probe behavior) and training +
    curriculum + readiness when the run trains on any curriculum id (the
    extended family; DR-READINESS-CURRICULUM §5.5). Held-out fork surfaces
    (gates rows 5,6) are in neither registry, so they are never probeable (I-3).
    """

    families = fork_hazard_layout_families()
    train = tuple(families["training"])
    readiness = tuple(families["readiness"])
    curriculum = tuple(fork_curriculum_scenarios())
    curriculum_set = frozenset(curriculum)
    admissible = frozenset(train) | frozenset(readiness) | curriculum_set
    if not all(name in admissible for name in scenario_names):
        return None
    # DR-CURRICULUM-FORM AM-CF2 (SHIFT 29, countersigned): a curriculum run probes
    # ONLY the curriculum pairs it actually TRAINS (plus train + readiness), never
    # the untrained catalog pairs -- so a breadth-reduced run (e.g. {3,4}+{0,3})
    # probes its trained surface, not {4,7}/{0,7}. Byte-identical for the T1-D14
    # form (all six curriculum ids trained -> the full curriculum tuple in its
    # canonical order) and for a training-pair-only run (no curriculum id -> ()).
    trained_names = frozenset(scenario_names)
    trained_curriculum = tuple(name for name in curriculum if name in trained_names)
    return train + trained_curriculum + readiness


def run_stage25_training(
    config: Stage25RunConfig,
    *,
    update_config: Stage25UpdateConfig | None = None,
    resume: bool = False,
) -> Path:
    """Run the Stage 25 reviewed final-training loop; return the run directory.

    Loops ``collect_stage25_training_rollout`` ->
    ``stage25_ppo_lagrangian_update`` with one persistent Adam (built from
    the update config's DR-D4 lr/eps) and the DR-D3 PI controller threaded
    across rounds. Writes ``RUN_MANIFEST.json`` (rewritten on heartbeat /
    kill / crash / completion), ``training_curve.jsonl``,
    ``episode_log.jsonl``, ``probe_log.jsonl``, and — per the DR-D4 cadence
    plus at termination — sha256-recorded training-state files with
    ``state_log.jsonl``. ``resume=True`` restores the latest verified state
    (model / optimizer / controller / round / torch RNG), verifies the
    configuration fingerprint saved in the state against the resume process's
    configuration (V-4: a mismatched resume is rejected), reconciles the
    append-only curve/episode/probe logs by dropping records for rounds the
    resume re-runs (V-3), records the resume provenance in the manifest, and
    continues appending with continuous round numbering.
    Kill criteria: ``lambda_applied`` above ``lambda_kill_bound`` saves
    a terminal state and marks the manifest ``killed``; any exception after
    the initial manifest write marks it ``crashed`` and re-raises; entropy
    collapse stays a soft per-round flag.
    """

    if not isinstance(config, Stage25RunConfig):
        raise TypeError("config must be a Stage25RunConfig")
    if update_config is None:
        # DR-C4: build the DR-D4 baseline, then apply the run config's recovery
        # levers (all OFF by default => byte-identical DR-D4 path). An explicitly
        # supplied update_config is used verbatim (the caller owns its levers).
        update_config = dataclasses.replace(
            Stage25UpdateConfig.from_decision_records(),
            constraint_warmup_rounds=config.constraint_warmup_rounds,
            advantage_std_guard_threshold=config.advantage_std_guard_threshold,
            movement_entropy_floor=config.movement_entropy_floor,
            sensing_entropy_floor=config.sensing_entropy_floor,
            # DR-D1/L1-b lever (2) keep-alive (OFF by default => byte-identical).
            sensing_keepalive_coefficient=config.sensing_keepalive_coefficient,
            sensing_keepalive_rounds=config.sensing_keepalive_rounds,
            sensing_entropy_target=config.sensing_entropy_target,
            # DR-D3 AMENDMENT (SHIFT 16) anti-windup ceiling (None => byte-identical).
            lambda_integral_cap=config.lambda_integral_cap,
            # DR-RELIABILITY levers (OFF by default => byte-identical).
            lambda_floor=config.lambda_floor,
            entropy_sustain_learning_rate=config.entropy_sustain_learning_rate,
            movement_entropy_sustain_target=(
                config.movement_entropy_sustain_target
            ),
            sensing_entropy_sustain_target=config.sensing_entropy_sustain_target,
            entropy_sustain_hold_rounds=config.entropy_sustain_hold_rounds,
            entropy_sustain_anneal_rounds=config.entropy_sustain_anneal_rounds,
            entropy_sustain_alpha_cap=config.entropy_sustain_alpha_cap,
            # DR-H3 levers G/H (None => byte-identical). The arm STATE lives in
            # this driver (see _ArmedFloorState); the update config carries only
            # the values the controller applies.
            lambda_ceiling=config.lambda_ceiling,
            armed_lambda_floor=config.armed_lambda_floor,
            # DR-H4 lever I (False => byte-identical).
            sensing_sustain_armed_gate=config.sensing_sustain_armed_gate,
            # DR-H5 lever J (0.5 default = DR-D2 => byte-identical) + lever K
            # gate (False => byte-identical). The recovery TRIGGER state lives
            # in this driver (_ArmedFloorState); the update config carries
            # only the gate the controller consumes.
            hazard_budget=config.hazard_budget,
            movement_recovery_gate=config.movement_recovery_gate,
        )
    elif not isinstance(update_config, Stage25UpdateConfig):
        raise TypeError("update_config must be a Stage25UpdateConfig or None")
    if not isinstance(resume, bool):
        raise TypeError("resume must be a bool")

    # Pre-directory input scan (the baseline-driver rule): unsafe wording in
    # caller-supplied free text must fail BEFORE any directory work.
    reject_unsafe_boundary_values(
        {
            "run_id": config.run_id,
            "scenario_names": list(config.scenario_names),
            "implementing_dr_ids": list(_IMPLEMENTING_DR_IDS),
            "tier": config.tier,
        },
        "run inputs",
        allowed_structural_keys=harness_allowed_structural_keys(),
    )

    repo_root = active_root(__file__, 3)
    ev_probe_config = _ev_probe_carrier_config(update_config)
    config_sha256 = _config_fingerprint(config, update_config)

    start_round = 0
    resume_provenance: dict[str, Any] = {}
    restored_state: dict[str, Any] | None = None
    state_saved = False
    if resume:
        run_dir = _resolve_run_directory_for_resume(
            config.run_id, config.result_parent
        )
        restored_state, state_record = _load_latest_training_state(
            run_dir, run_id=config.run_id, config_sha256=config_sha256
        )
        start_round = int(restored_state["completed_rounds"])
        if start_round < 1:
            raise ValueError("saved training state has no completed rounds")
        if start_round >= int.__index__(config.update_rounds):
            raise ValueError(
                "saved training state already covers every requested round"
            )
        # V-3: drop curve/episode/probe records for rounds the resume will
        # re-run (a crash after the last save leaves records past the resume
        # point in the append-only logs).
        reconciled_dropped = 0
        for log_name in (
            "training_curve.jsonl",
            "episode_log.jsonl",
            "probe_log.jsonl",
        ):
            reconciled_dropped += _reconcile_resumed_log(
                run_dir / log_name, start_round=start_round
            )
        # A saved state exists in this run directory, so every record written
        # by the resumed process truthfully carries the state-saved flags.
        state_saved = True
        resume_provenance = {
            "resumed": True,
            "resumed_from_round": start_round,
            "resumed_from_file": str(state_record["saved_state_file"]),
            "resume_reconciled_records_dropped": reconciled_dropped,
            "resume_timestamp_utc": utc_timestamp(),
        }
    else:
        run_dir = resolve_run_directory(config.run_id, config.result_parent)

    runtime_record = stage24a_runtime_reproducibility_record(
        seed_values=(config.seed,), device="cpu", dtype="float32"
    )
    manifest_base: dict[str, Any] = {
        "run_id": config.run_id,
        "tier": config.tier,
        "seeds": (config.seed,),
        "implementing_dr_ids": _IMPLEMENTING_DR_IDS,
        "config_payload": _config_payload(config, update_config),
        "git_head": _git_head(repo_root),
        "launch_timestamp_utc": utc_timestamp(),
        "pid": os.getpid(),
    }

    def _write_manifest(status: str, extra_fields: dict[str, Any]) -> None:
        payload = build_run_manifest_payload(
            status=status,
            extra={
                "runtime_record": runtime_record,
                "config_sha256": config_sha256,
                **resume_provenance,
                **extra_fields,
            },
            **manifest_base,
        )
        # The shared builder spreads the harness boundary flags; replace them
        # in place with the truthful Stage 25 flags (same key inventory, so
        # the scanners' allowed structural key set is unchanged). The scan
        # happens inside write_run_manifest, on the final payload.
        payload.update(stage25_boundary_flags(state_saved=state_saved))
        write_run_manifest(run_dir, payload)

    if resume:
        _write_manifest("running", {"last_completed_round": start_round - 1})
    else:
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
    last_completed_round = start_round - 1
    last_saved_round = start_round if resume else 0
    # Everything after the initial 'running' manifest write — including
    # model/optimizer/controller construction and state restoration — runs
    # inside the crash-marking try, so any pre-loop failure still rewrites
    # the manifest as 'crashed'.
    try:
        core_config = default_stage23_core_config()
        # V-5: seed the fresh-run parameter initialization so it is
        # attributable to config.seed (a resume immediately overwrites both
        # the parameters and the torch RNG state below, so seeding here is
        # harmless on the resume path).
        torch.manual_seed(int.__index__(config.seed))
        model = RecurrentMAPPOActorCritic(core_config)
        # One persistent Adam built ONCE with the DR-D4 optimizer constants
        # (lr 3e-4, eps 1e-5, weight_decay 0, no lr decay).
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=update_config.learning_rate,
            eps=update_config.adam_epsilon,
            weight_decay=0.0,
        )
        controller = update_config.make_controller()
        # DR-RELIABILITY lever E: the entropy-sustain controller (None when the
        # lever is OFF), threaded across rounds exactly like the PI controller.
        entropy_controller = update_config.make_entropy_controller()
        # DR-H3 lever H: the driver-owned armed-floor state (None when OFF).
        arm_state = (
            _ArmedFloorState(
                arm_window_rounds=config.arm_window_rounds,
                arm_success_threshold=config.arm_success_threshold,
                arm_hazard_threshold=config.arm_hazard_threshold,
                arm_sensing_threshold=config.arm_sensing_threshold,
                dearm_window_rounds=config.dearm_window_rounds,
                dearm_success_threshold=config.dearm_success_threshold,
                constraint_warmup_rounds=config.constraint_warmup_rounds,
                # DR-H5 lever K: the midband recovery trigger (all zeros/None
                # when the gate is OFF => byte-identical state + payload).
                midband_window_rounds=config.midband_recovery_window_rounds,
                midband_success_lo=config.midband_recovery_success_lo,
                midband_success_hi=config.midband_recovery_success_hi,
                midband_containment_widen=(
                    config.midband_recovery_containment_widen
                ),
                midband_hazard_cap=config.midband_recovery_hazard_cap,
                midband_max_rounds=config.midband_recovery_max_rounds,
            )
            if config.armed_lambda_floor is not None
            else None
        )
        if restored_state is not None:
            model.load_state_dict(restored_state["model_state_dict"])
            optimizer.load_state_dict(restored_state["optimizer_state_dict"])
            saved_controller = restored_state["controller"]
            _saved_cap = saved_controller.get("integral_cap")  # DR-D3-amendment
            _saved_ceiling = saved_controller.get("lambda_ceiling")  # DR-H3 G
            _saved_armed_floor = saved_controller.get(
                "armed_lambda_floor"
            )  # DR-H3 H
            controller = PIController(
                proportional_gain=float(saved_controller["proportional_gain"]),
                integral_gain=float(saved_controller["integral_gain"]),
                integral=float(saved_controller["integral"]),
                budget=float(saved_controller["budget"]),
                integral_cap=(None if _saved_cap is None else float(_saved_cap)),
                # DR-RELIABILITY lever F: restore the sustain floor (the .get
                # default 0.0 is belt-and-suspenders — V-4 fingerprint equality
                # guarantees the save was written under the same floor value,
                # and the A-10 guard cross-checks at the first update).
                lambda_floor=float(saved_controller.get("lambda_floor", 0.0)),
                # DR-H3 levers G/H: same restore discipline (None => OFF; the
                # A-10 guard cross-checks both at the first update).
                lambda_ceiling=(
                    None if _saved_ceiling is None else float(_saved_ceiling)
                ),
                armed_lambda_floor=(
                    None
                    if _saved_armed_floor is None
                    else float(_saved_armed_floor)
                ),
            )
            if entropy_controller is not None:
                # DR-RELIABILITY lever E: an ON-config resume REQUIRES the
                # saved alpha state — defaulting it to fresh zeros would
                # silently reset the sustain temperatures mid-run (the AM-2
                # silent-reset failure mode).
                saved_entropy = restored_state.get("entropy_controller")
                if (
                    not isinstance(saved_entropy, dict)
                    or "alpha_movement" not in saved_entropy
                    or "alpha_sensing" not in saved_entropy
                ):
                    raise ValueError(
                        "resume with the entropy-sustain lever ON requires the "
                        "saved entropy_controller state (missing or invalid in "
                        "the saved training state)"
                    )
                entropy_controller = dataclasses.replace(
                    entropy_controller,
                    alpha_movement=float(saved_entropy["alpha_movement"]),
                    alpha_sensing=float(saved_entropy["alpha_sensing"]),
                )
            if arm_state is not None:
                # DR-H3 lever H: an ON-config resume REQUIRES the saved arm
                # state — defaulting to a fresh un-armed latch with empty
                # windows would silently drop a fired arm (and its floor)
                # mid-run: the AM-2 silent-reset failure mode, applied to the
                # latch. The restore is byte-identical (latch + both window
                # buffers + the event log; AM-H15 mid-window resume).
                saved_arm = restored_state.get("arm_state")
                if not isinstance(saved_arm, dict):
                    raise ValueError(
                        "resume with the armed-floor lever ON requires the "
                        "saved arm_state (missing or invalid in the saved "
                        "training state)"
                    )
                arm_state.restore_payload(saved_arm)
            torch.set_rng_state(restored_state["torch_rng_state"])

        # DR-D1/L1-b SHIFT-9 owed probe repoint, EXTENDED by the countersigned
        # DR-READINESS-CURRICULUM §5.5: fork-family runs probe the fork
        # surfaces (the held-out fork family, gates 5,6, is in NEITHER registry
        # => never instantiated, I-3); the tuple is conditional so a
        # training-pair-only run probes exactly the pre-curriculum 4 surfaces
        # (OFF-default byte-identity) while a curriculum run probes the
        # extended family. Legacy runs keep probe_scenarios=None => the
        # unchanged legacy probe.
        probe_scenarios = _resolve_probe_scenarios(config.scenario_names)

        for round_index in range(start_round, config.update_rounds):
            round_start = time.perf_counter()
            round_seed = config.seed + round_index * _ROUND_SEED_STRIDE
            collection_config = Stage25CollectionConfig(
                max_episodes=config.episodes_per_round,
                steps_per_episode=config.steps_per_episode,
                scenario_names=config.scenario_names,
                sample_actions=config.sample_actions,
                seed=round_seed,
                task_progress_potential_weight=config.task_progress_potential_weight,
                sensing_credit_potential_weight=(
                    config.sensing_credit_potential_weight
                ),
                sensing_zone_potential_weight=(
                    config.sensing_zone_potential_weight
                ),
            )
            collected = collect_stage25_training_rollout(model, collection_config)
            # DR-C4 shaping (Fix 1): when the collection produced potentials, form
            # the single-source shaping term (update's gamma) once and pass it to
            # BOTH the EV diagnostic (so reward EV is measured on the shaped
            # returns the critic trains on) and the update. None => OFF => the
            # DR-D4 path (byte-identical).
            reward_shaping_potentials = None
            reward_shaping = None
            if collected.potential is not None:
                reward_shaping_potentials = (
                    collected.potential,
                    collected.potential_next,
                )
                reward_shaping = potential_shaping_term(
                    collected.potential,
                    collected.potential_next,
                    update_config.algorithm.discount_factor,
                )
            # DR-D1/L1-b lever (1): the decision-relevant sensing credit is a
            # SECOND state potential added to the SAME reward channel. Form its
            # shaping term and fold it into the EV diagnostic's shaped returns (so
            # reward EV is measured on the returns the critic actually trains on),
            # and pass its potentials to the update. None => OFF => byte-identical.
            sensing_shaping_potentials = None
            if collected.sensing_potential is not None:
                sensing_shaping_potentials = (
                    collected.sensing_potential,
                    collected.sensing_potential_next,
                )
                sensing_shaping = potential_shaping_term(
                    collected.sensing_potential,
                    collected.sensing_potential_next,
                    update_config.algorithm.discount_factor,
                )
                reward_shaping = (
                    sensing_shaping
                    if reward_shaping is None
                    else reward_shaping + sensing_shaping
                )
            explained_variance = explained_variance_from_batch(
                collected.batch, ev_probe_config, reward_shaping=reward_shaping
            )
            # DR-H3 lever H: feed the just-completed round's behavioral
            # aggregates (the SAME single-source values the curve row records)
            # to the arm; the returned state governs THIS round's dual step
            # (AM-H15: the fire round itself is floored). OFF (arm_state None)
            # => floor_armed False => byte-identical.
            floor_armed = False
            arm_window_means = (0.0, 0.0, 0.0)
            movement_recovery_active = False
            recovery_rounds_used = 0
            recovery_exhausted = False
            if arm_state is not None:
                success_rate, hazard_mean, sensing_mean = (
                    _round_behavior_aggregates(collected.episode_records)
                )
                floor_armed = arm_state.observe_round(
                    round_index=round_index,
                    success_rate=success_rate,
                    hazard_mean=hazard_mean,
                    sensing_mean=sensing_mean,
                )
                arm_window_means = arm_state.window_means()
                # DR-H5 lever K: the recovery latch governs the SAME round's
                # update (the engage round itself is a recovery round —
                # the AM-H15 fire-round convention applied to the trigger).
                movement_recovery_active = arm_state.recovery_active
                recovery_rounds_used = arm_state.recovery_rounds_used
                recovery_exhausted = arm_state.recovery_exhausted
            result = stage25_ppo_lagrangian_update(
                model,
                collected.batch,
                update_config,
                optimizer=optimizer,
                controller=controller,
                episode_count=len(collected.episode_records),
                round_index=round_index,
                reward_shaping_potentials=reward_shaping_potentials,
                sensing_shaping_potentials=sensing_shaping_potentials,
                entropy_controller=entropy_controller,
                floor_armed=floor_armed,
                movement_recovery_active=movement_recovery_active,
            )
            controller = result.controller
            entropy_controller = result.entropy_controller
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
                    state_saved=state_saved,
                    arm_window_means=arm_window_means,
                    recovery_rounds_used=recovery_rounds_used,
                    recovery_exhausted=recovery_exhausted,
                ),
            )
            for episode_record in collected.episode_records:
                append_jsonl_record(
                    episode_path,
                    {
                        **episode_record,
                        "round_index": round_index,
                        **stage25_boundary_flags(state_saved=state_saved),
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
                    update_config=ev_probe_config,
                    probe_scenarios=probe_scenarios,
                )
                append_jsonl_record(
                    probe_path,
                    {
                        "round_index": round_index,
                        **probe,
                        **stage25_boundary_flags(state_saved=state_saved),
                    },
                )
            last_completed_round = round_index

            kill_due = result.lambda_applied > config.lambda_kill_bound
            state_save_due = (
                (round_index + 1) % config.state_save_every == 0
                or final_round
                or kill_due
            )
            if state_save_due and last_saved_round != round_index + 1:
                _save_training_state(
                    run_dir,
                    run_id=config.run_id,
                    config_sha256=config_sha256,
                    completed_rounds=round_index + 1,
                    model=model,
                    optimizer=optimizer,
                    controller=controller,
                    entropy_controller=entropy_controller,
                    arm_state=arm_state,
                )
                last_saved_round = round_index + 1
                state_saved = True

            if kill_due:
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
        # Hardened crash handler (the baseline-driver pattern): a hostile
        # __str__ / metaclass __name__ must never prevent the crashed-manifest
        # write, and the ORIGINAL exception always propagates.
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
        try:
            try:
                _write_manifest(
                    "crashed", {"kill_reason": kill_reason, **crash_fields}
                )
            except ValueError:
                # The exception str() wording tripped the unsafe-wording
                # scanner; fall back to the pre-resolved exception type name.
                try:
                    _write_manifest(
                        "crashed",
                        {"kill_reason": safe_type_name, **crash_fields},
                    )
                except ValueError:
                    # A-7: even the exception TYPE NAME tripped the scanner (a
                    # hostile metaclass __name__). Write with a vetted constant
                    # so the crash is ALWAYS recorded — never leave the
                    # manifest stuck at 'running' with no crash marking.
                    _write_manifest(
                        "crashed",
                        {"kill_reason": _CRASH_REASON_WITHHELD, **crash_fields},
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
    """Command-line entry point for the Stage 25 final-training driver."""

    parser = argparse.ArgumentParser(
        prog="stage25_driver",
        description=(
            "Stage 25 reviewed final-training driver (DR-D4 update path over "
            "Stage 25 collection; state saves + resume; probes on the "
            "training and readiness surfaces only)."
        ),
    )
    parser.add_argument(
        "--run-id", required=True, help="unique run id (used exactly once)"
    )
    parser.add_argument("--seed", type=int, required=True, help="positive base seed")
    parser.add_argument(
        "--update-rounds",
        type=int,
        required=True,
        help="collect/update rounds (1..20000)",
    )
    parser.add_argument("--episodes-per-round", type=int, default=16, help="2..32")
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
        "--probe-every", type=int, default=100, help="0 disables periodic probes"
    )
    parser.add_argument("--probe-seed", type=int, default=9001)
    parser.add_argument(
        "--state-save-every",
        type=int,
        default=500,
        help="training-state save cadence in rounds (DR-D4: 500)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume the run from its latest verified saved training state",
    )
    parser.add_argument("--tier", default="T1", choices=list(_VALID_TIERS))
    parser.add_argument("--result-parent", default=None)
    # DR-C4 recovery levers (all OFF by default => DR-D4 path). Envelopes:
    # w_Phi in (0, 0.1]; warmup in [50, 500] (<= 500 hard, AM-3).
    parser.add_argument(
        "--task-progress-potential-weight",
        type=float,
        default=0.0,
        help="DR-C4 Fix 1: BFS goal-distance potential weight w_Phi (0.0 = OFF)",
    )
    parser.add_argument(
        "--constraint-warmup-rounds",
        type=int,
        default=0,
        help="DR-C4 Fix 2: hold lambda=0 for the first W rounds (0 = OFF, <= 500)",
    )
    parser.add_argument(
        "--advantage-std-guard-threshold",
        type=float,
        default=0.0,
        help="DR-C4 Fix 4a: advantage std-guard threshold (0.0 = OFF)",
    )
    parser.add_argument(
        "--movement-entropy-floor",
        type=float,
        default=None,
        help="DR-C4 Fix 4b: movement entropy floor (omit = OFF)",
    )
    parser.add_argument(
        "--sensing-entropy-floor",
        type=float,
        default=None,
        help="DR-C4 Fix 4b: sensing entropy floor (omit = OFF)",
    )
    # DR-D1/L1-b levers (all OFF by default => DR-C4/DR-D4 path). Envelopes:
    # w_s in (0, 0.1]; keep-alive coeff in [0.05, 0.20]; target in [0.20, 0.50].
    parser.add_argument(
        "--sensing-credit-potential-weight",
        type=float,
        default=0.0,
        help="DR-D1/L1-b lever 1: prev_sensed sensing credit weight w_s "
        "(0.0 = OFF)",
    )
    parser.add_argument(
        "--sensing-zone-potential-weight",
        type=float,
        default=0.0,
        help="DR-D1/L1-b AMENDMENT lever 1 DECOUPLE: decision-region positioning "
        "potential weight w_dz (0.0 = OFF)",
    )
    parser.add_argument(
        "--sensing-keepalive-coefficient",
        type=float,
        default=0.0,
        help="DR-D1/L1-b lever 2: warmup-scoped sensing keep-alive coefficient "
        "(0.0 = OFF)",
    )
    parser.add_argument(
        "--sensing-keepalive-rounds",
        type=int,
        default=0,
        help="DR-D1/L1-b lever 2: keep-alive window in rounds (0 = OFF, "
        "<= update_rounds)",
    )
    parser.add_argument(
        "--sensing-entropy-target",
        type=float,
        default=None,
        help="DR-D1/L1-b lever 2: sensing entropy target H*_sense (omit = ungated)",
    )
    parser.add_argument(
        "--lambda-integral-cap",
        type=float,
        default=None,
        help="DR-D3 AMENDMENT (SHIFT 16): anti-windup ceiling on the dual "
        "integral (omit = OFF, byte-identical DR-D3; pinned run value 20.0)",
    )
    # DR-RELIABILITY (SHIFT 19) levers (all OFF by default). Envelopes:
    # lambda_floor in [2.5, 5.0]; sustain lr in [0.01, 0.05]; movement target
    # in [0.35, 0.8]; sensing target in [0.2, 0.45]; hold in [600, 1200];
    # anneal end in [1500, 3000]; alpha cap in [1.0, 5.0].
    parser.add_argument(
        "--lambda-floor",
        type=float,
        default=0.0,
        help="DR-RELIABILITY lever F: positive sustain floor on the applied "
        "hazard multiplier (0.0 = OFF; pinned run value 3.0)",
    )
    parser.add_argument(
        "--entropy-sustain-lr",
        type=float,
        default=0.0,
        help="DR-RELIABILITY lever E: entropy-sustain dual learning rate "
        "(0.0 = OFF; pinned run value 0.02)",
    )
    parser.add_argument(
        "--movement-entropy-sustain-target",
        type=float,
        default=None,
        help="DR-RELIABILITY lever E: movement entropy hold target H*_move "
        "(omit = movement uncontrolled; pinned run value 0.5)",
    )
    parser.add_argument(
        "--sensing-entropy-sustain-target",
        type=float,
        default=None,
        help="DR-RELIABILITY lever E: sensing entropy hold target H*_sense "
        "(omit = sensing uncontrolled; pinned run value 0.30)",
    )
    parser.add_argument(
        "--entropy-sustain-hold-rounds",
        type=int,
        default=0,
        help="DR-RELIABILITY lever E: rounds at the full hold target "
        "(pinned run value 800)",
    )
    parser.add_argument(
        "--entropy-sustain-anneal-rounds",
        type=int,
        default=0,
        help="DR-RELIABILITY lever E: round at which the linear target anneal "
        "reaches 0 (>= hold rounds, <= update rounds; pinned run value 2000)",
    )
    parser.add_argument(
        "--entropy-sustain-alpha-cap",
        type=float,
        default=None,
        help="DR-RELIABILITY lever E: cap on each entropy-sustain temperature "
        "(REQUIRED when the lever is on; pinned run value 2.0)",
    )
    parser.add_argument(
        "--sensing-sustain-armed-gate",
        action="store_true",
        default=False,
        help="DR-H4 lever I: armed-gated sensing-sustain release — the "
        "sensing entropy-sustain target holds at H*_sense while the armed "
        "floor is UNARMED and is 0.0 with alpha pinned 0.0 while ARMED "
        "(omit = OFF; requires the lever-E sensing sustain AND the DR-H3 "
        "armed floor)",
    )
    parser.add_argument(
        "--lambda-ceiling",
        type=float,
        default=None,
        help="DR-H3 lever G: hard ceiling on the applied multiplier "
        "(omit = OFF; pinned run value 5.0, envelope [4.0, 6.0])",
    )
    parser.add_argument(
        "--armed-lambda-floor",
        type=float,
        default=None,
        help="DR-H3 lever H: the ARMED conditional floor value (omit = OFF; "
        "pinned run value 3.0; requires the arm/dearm parameters and a fork "
        "training-family + countersigned-curriculum scenario set in complete, "
        "per-round-balanced mirror pairs per AM-H13 as re-scoped by "
        "DR-READINESS-CURRICULUM)",
    )
    parser.add_argument(
        "--arm-window-rounds",
        type=int,
        default=0,
        help="DR-H3 lever H: trailing arm-window length in rounds "
        "(pinned run value 50)",
    )
    parser.add_argument(
        "--arm-success-threshold",
        type=float,
        default=None,
        help="DR-H3 lever H: arm-window mean success threshold (>=; pinned 0.8)",
    )
    parser.add_argument(
        "--arm-hazard-threshold",
        type=float,
        default=None,
        help="DR-H3 lever H: arm-window mean episodic-hazard threshold "
        "(<=; pinned 0.35 — theorem-backed against blind gaming on the fork)",
    )
    parser.add_argument(
        "--arm-sensing-threshold",
        type=float,
        default=None,
        help="DR-H3 lever H: arm-window mean sensing-rate threshold "
        "(>=; pinned 0.05)",
    )
    parser.add_argument(
        "--dearm-window-rounds",
        type=int,
        default=0,
        help="DR-H3 AM-H4: trailing de-arm window length in rounds "
        "(pinned run value 200)",
    )
    parser.add_argument(
        "--dearm-success-threshold",
        type=float,
        default=None,
        help="DR-H3 AM-H4: de-arm mean-success release threshold "
        "(<=; pinned 0.1)",
    )
    parser.add_argument(
        "--hazard-budget",
        type=float,
        default=0.5,
        help="DR-H5 lever J: the DR-D2 episodic team budget d_ep "
        "(default 0.5 = DR-D2 byte-identical; pinned T1-D13 run value 0.25; "
        "values above 0.5 rejected)",
    )
    parser.add_argument(
        "--movement-recovery-gate",
        action="store_true",
        default=False,
        help="DR-H5 lever K: limbo-scoped armed movement-plasticity "
        "re-engagement (omit = OFF; requires the armed floor, the lever-E "
        "movement sustain, and ALL midband-recovery parameters)",
    )
    parser.add_argument(
        "--midband-recovery-window-rounds",
        type=int,
        default=0,
        help="DR-H5 lever K: trailing containment-window length in rounds "
        "(pinned run value 200)",
    )
    parser.add_argument(
        "--midband-recovery-success-lo",
        type=float,
        default=None,
        help="DR-H5 lever K: mid-band window-mean success floor (pinned 0.25)",
    )
    parser.add_argument(
        "--midband-recovery-success-hi",
        type=float,
        default=None,
        help="DR-H5 lever K: mid-band window-mean success ceiling "
        "(pinned 0.75; must sit strictly below arm_success_threshold)",
    )
    parser.add_argument(
        "--midband-recovery-containment-widen",
        type=float,
        default=None,
        help="DR-H5 lever K: per-round containment widening of the band "
        "(pinned 0.10 -> every per-round success inside [0.15, 0.85])",
    )
    parser.add_argument(
        "--midband-recovery-hazard-cap",
        type=float,
        default=None,
        help="DR-H5 lever K: window-mean hazard cap (pinned 0.05)",
    )
    parser.add_argument(
        "--midband-recovery-max-rounds",
        type=int,
        default=0,
        help="DR-H5 lever K: R_max — hard per-engagement termination "
        "(pinned run value 800; an exhausted engagement latches recovery "
        "OFF for the rest of the run)",
    )
    args = parser.parse_args(argv)

    scenario_names = (
        tuple(args.scenario) if args.scenario else ("risk_gate_hidden_hazard",)
    )
    run_config = Stage25RunConfig(
        run_id=args.run_id,
        seed=args.seed,
        update_rounds=args.update_rounds,
        episodes_per_round=args.episodes_per_round,
        steps_per_episode=args.steps_per_episode,
        scenario_names=scenario_names,
        probe_every=args.probe_every,
        probe_seed=args.probe_seed,
        tier=args.tier,
        result_parent=args.result_parent,
        state_save_every=args.state_save_every,
        task_progress_potential_weight=args.task_progress_potential_weight,
        constraint_warmup_rounds=args.constraint_warmup_rounds,
        advantage_std_guard_threshold=args.advantage_std_guard_threshold,
        movement_entropy_floor=args.movement_entropy_floor,
        sensing_entropy_floor=args.sensing_entropy_floor,
        sensing_credit_potential_weight=args.sensing_credit_potential_weight,
        sensing_zone_potential_weight=args.sensing_zone_potential_weight,
        sensing_keepalive_coefficient=args.sensing_keepalive_coefficient,
        sensing_keepalive_rounds=args.sensing_keepalive_rounds,
        sensing_entropy_target=args.sensing_entropy_target,
        lambda_integral_cap=args.lambda_integral_cap,
        lambda_floor=args.lambda_floor,
        entropy_sustain_learning_rate=args.entropy_sustain_lr,
        movement_entropy_sustain_target=args.movement_entropy_sustain_target,
        sensing_entropy_sustain_target=args.sensing_entropy_sustain_target,
        entropy_sustain_hold_rounds=args.entropy_sustain_hold_rounds,
        entropy_sustain_anneal_rounds=args.entropy_sustain_anneal_rounds,
        entropy_sustain_alpha_cap=args.entropy_sustain_alpha_cap,
        sensing_sustain_armed_gate=args.sensing_sustain_armed_gate,
        lambda_ceiling=args.lambda_ceiling,
        armed_lambda_floor=args.armed_lambda_floor,
        arm_window_rounds=args.arm_window_rounds,
        arm_success_threshold=args.arm_success_threshold,
        arm_hazard_threshold=args.arm_hazard_threshold,
        arm_sensing_threshold=args.arm_sensing_threshold,
        dearm_window_rounds=args.dearm_window_rounds,
        dearm_success_threshold=args.dearm_success_threshold,
        hazard_budget=args.hazard_budget,
        movement_recovery_gate=args.movement_recovery_gate,
        midband_recovery_window_rounds=args.midband_recovery_window_rounds,
        midband_recovery_success_lo=args.midband_recovery_success_lo,
        midband_recovery_success_hi=args.midband_recovery_success_hi,
        midband_recovery_containment_widen=(
            args.midband_recovery_containment_widen
        ),
        midband_recovery_hazard_cap=args.midband_recovery_hazard_cap,
        midband_recovery_max_rounds=args.midband_recovery_max_rounds,
    )
    run_dir = run_stage25_training(run_config, resume=args.resume)
    print(f"STAGE25 RUN COMPLETE: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

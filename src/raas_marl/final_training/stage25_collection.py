"""Stage 25 final-training rollout collection over the Stage 23-A environment.

Collects the DR-D4 update batch: fresh on-policy episodes (default 16 per
round, Stage 25's OWN bound of 2..32 — DR-D4: "the dev-stage collector episode
caps bind dev stages only; the Stage 25 collection path sets its own bound"),
full environment horizon by default, stochastic on-policy sampling by default
(DR-D4: greedy collection rejected for the primary runs, Evidence 6/b104),
scenario names restricted to the Stage 23-A catalog (the DR-D4 surfaces clause
binds the primary runs to the training scenario; the held-out variant is not a
catalog scenario and cannot be named here).

The per-episode collection semantics are the Stage 23-C semantics, imported —
never copied (the D-7 lesson): ``stage23c_rollout._collect_single_stage23c_
episode`` is the single source of the per-step loop (fresh env per episode,
fresh GRU history, per-step reseed via ``_rollout_common.per_step_sampling_
seed`` with the collision-free ``seed + episode_index*1000 + step_index``
offsets, Stage 23-B bootstrap convention, detached tensors, 18-field
per-episode record), and the merge/validation helpers come from
``_rollout_common`` (``merge_padded_episodes`` with ``valid_mask`` False on
padded steps, ``validate_rollout_batch_values``,
``validate_stage23_core_config``). If the private import is ever promoted to a
public export, only this import block changes. No file under
``mappo_lagrangian/`` is modified.

This module is development infrastructure for the Stage 25 final-training
stage authorized by the Segment 1 handoff. It writes no files, serializes no
rollout buffers, runs no final evaluation, creates no claim evidence, produces
no paper-facing output, and makes no Bayesian-belief or formal-VOI claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from raas_marl.environments.active_sensing.grid_environment import (
    STAGE23_AGENT_NAMES,
    STAGE23_SENSE_ACTION_INDEX,
    RiskAwareActiveSensingGridEnvironment,
    Stage23Scenario,
)
from raas_marl.environments.active_sensing.scenarios import available_scenarios
# DR-C4 (potential-based task-progress shaping): the BFS goal-distance primitive
# is imported from the diagnostics module (the D-7 lesson: reuse, never copy the
# comparators' shortest-path geometry), so the shaping potential respects the
# exact same grid geometry the scripted comparators use. baseline_driver already
# imports diagnostics private names; this extends that established pattern.
from raas_marl.environments.active_sensing.stage24_diagnostics import (
    _shortest_path,
    _stage24a_public_risk_zone_cells,
)
from raas_marl.environments.active_sensing.tensor_adapter import (
    default_stage23_core_config,
)
from raas_marl.mappo_lagrangian._rollout_common import (
    merge_padded_episodes,
    set_rollout_seed,
    validate_rollout_batch_values,
    validate_stage23_core_config,
)
from raas_marl.mappo_lagrangian._validation import (
    require_bounded_seed,
    require_nonnegative_number,
    require_positive_int,
)
from raas_marl.mappo_lagrangian.buffer import RolloutBatch
from raas_marl.mappo_lagrangian.model import RecurrentMAPPOActorCritic

# Private Stage 23-C episode collector imported deliberately (D-7 lesson:
# shared helper bodies are imported, never copied); see the module docstring.
from raas_marl.mappo_lagrangian.stage23c_rollout import (
    _collect_single_stage23c_episode,
    make_default_stage23c_environment,
)

__all__ = [
    "STAGE25_MAX_EPISODES_PER_ROUND",
    "STAGE25_MIN_EPISODES_PER_ROUND",
    "Stage25CollectedRollout",
    "Stage25CollectionConfig",
    "collect_stage25_training_rollout",
]


# Stage 25's OWN collection bound (DR-D4): the Stage 23-C dev cap (8) binds
# dev stages only. The DR-D4 primary scale is 16 episodes per update round;
# 32 is the pre-recorded fallback-axis headroom ("episode count 32").
STAGE25_MIN_EPISODES_PER_ROUND = 2
STAGE25_MAX_EPISODES_PER_ROUND = 32
_MIN_STEPS_PER_EPISODE = 1
_MAX_STEPS_PER_EPISODE = 32

# per_step_sampling_seed derives seed + episode_index * 1000 + step_index; the
# largest per-collection offset (32 episodes x stride 1000 + 32 steps) bounds
# the seed so torch.manual_seed can never overflow signed 64-bit range.
_PER_EPISODE_SEED_STRIDE = 1000

_STAGE25_ZERO_SEED_MESSAGE = (
    "seed must be a positive integer; seed 0 is reserved for Stage 25"
)


@dataclass(frozen=True)
class Stage25CollectionConfig:
    """Validated Stage 25 per-round collection settings (DR-D4 scale).

    Defaults are the DR-D4 collection clause: 16 fresh on-policy episodes per
    update round, full environment horizon (``steps_per_episode=None``), the
    catalog training scenario, stochastic sampling, CPU/float32 lock. The
    per-episode reset seed is ``seed + episode_index`` and the per-step
    sampling reseed is the collision-free ``seed + episode_index * 1000 +
    step_index`` (both inherited from the imported Stage 23-C collector).
    """

    max_episodes: int = 16
    steps_per_episode: int | None = None
    scenario_names: tuple[str, ...] = ("risk_gate_hidden_hazard",)
    sample_actions: bool = True
    seed: int = 105
    dtype: str = "float32"
    device: str = "cpu"
    # DR-C4 lever 1 (potential-based task-progress shaping), OFF by default.
    # 0.0 => no potential is computed (Stage25CollectedRollout.potential stays
    # None => the update path is byte-identical to the DR-D4 path). A positive
    # weight w_Phi selects the BFS-goal-distance potential Phi_i(s) =
    # -w_Phi * BFS_dist(agent_i cell -> nearest goal). The DR-C4 pre-authorized
    # activation envelope is w_Phi in (0, 0.1]; anything beyond reopens DR-C4.
    task_progress_potential_weight: float = 0.0
    # DR-D1/L1-b lever (1) (decision-relevant sensing credit), OFF by default.
    # 0.0 => no sensing potential is computed (Stage25CollectedRollout.
    # sensing_potential stays None => the update path is byte-identical). A
    # positive weight w_s selects the FIXED, OBSERVABLE state potential
    # Phi_sense_i(s) = +w_s * 1[previous_sensed_i(s)] * 1[cell_i(s) in
    # fork_public_risk_zone(scenario)] -- a policy-INVARIANT PBRS credit
    # (Ng-Harada-Russell; telescopes to gamma^L*Phi(s_L)-Phi(s_0)) that rewards
    # having-sensed at the decision-relevant gate approach. Reads ONLY observable
    # actor features (previous_sensed at feature index 5; position at 0,1) + the
    # observable public risk zone -- never the hidden hazard (M-5-safe:
    # layout-agnostic, byte-identical across a fork group's aliased layouts).
    # The pre-authorized envelope is w_s in (0, 0.1]; anything beyond reopens
    # DR-D1/L1-b.
    sensing_credit_potential_weight: float = 0.0
    # DR-D1/L1-b AMENDMENT (SHIFT 13) lever (1) DECOUPLE, OFF by default.
    # 0.0 => contributes nothing (with sensing_credit_potential_weight also 0.0,
    # sensing_potential stays None => byte-identical DR-C4/DR-D4 path). A positive
    # weight w_dz selects the DECISION-REGION POSITIONING potential
    # Phi_dz_i(s) = +w_dz * 1[cell_i(s) in fork_public_risk_zone(scenario)] --
    # a FIXED OBSERVABLE state potential over decision-region OCCUPANCY (drops the
    # previous_sensed gate; reads position features 0,1 + the observable public
    # zone ONLY, never previous_sensed, never sensing_action, never the hidden
    # hazard => M-5-safe/layout-agnostic; policy-INVARIANT PBRS, telescopes to
    # gamma^L*Phi(s_L)-Phi(s_0)). By NHR it is net-zero on the true return: a
    # decision-region value-DENSIFIER, NOT a sensing reward (the sense action's
    # only fixed-state feature is previous_sensed, whose footprint collapses with
    # sensing => no fixed state potential can give a persistent restoring force on
    # SENSING). The composed potential is the linear sum
    # Phi_dz + (w_s>0 ? prev_sensed-credit : 0). Envelope w_dz in (0, 0.1].
    sensing_zone_potential_weight: float = 0.0

    def __post_init__(self) -> None:
        require_positive_int(
            "max_episodes", self.max_episodes, error_suffix="must be positive"
        )
        # Normalize through int.__index__ before the range comparison so a
        # hostile int subclass cannot evade the bound (the S2-5 pattern the
        # Stage 23-C config also uses).
        max_episodes = int.__index__(self.max_episodes)
        if not (
            STAGE25_MIN_EPISODES_PER_ROUND
            <= max_episodes
            <= STAGE25_MAX_EPISODES_PER_ROUND
        ):
            raise ValueError(
                f"max_episodes must be between {STAGE25_MIN_EPISODES_PER_ROUND} "
                f"and {STAGE25_MAX_EPISODES_PER_ROUND}"
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
        catalog_names = available_scenarios()
        for name in self.scenario_names:
            if not isinstance(name, str) or not name.strip():
                raise ValueError("scenario_names entries must be non-empty strings")
            if name not in catalog_names:
                raise ValueError(f"unknown Stage 23-A scenario: {name}")
        if not isinstance(self.sample_actions, bool):
            raise TypeError("sample_actions must be a bool")
        max_step_horizon = (
            int.__index__(self.steps_per_episode)
            if self.steps_per_episode is not None
            else _MAX_STEPS_PER_EPISODE
        )
        require_bounded_seed(
            "seed",
            self.seed,
            max_offset=max_episodes * _PER_EPISODE_SEED_STRIDE + max_step_horizon,
            zero_seed_message=_STAGE25_ZERO_SEED_MESSAGE,
        )
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be 'float32' or 'float64'")
        expected_dtype = default_stage23_core_config().dtype
        if self.dtype != expected_dtype:
            raise ValueError(
                "Stage 25 collection dtype must equal the default Stage 23-A "
                f"core dtype {expected_dtype!r}; non-default dtype support is "
                "out of scope"
            )
        if not isinstance(self.device, str) or not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.device != self.device.strip():
            raise ValueError(
                "device must not contain leading or trailing whitespace"
            )
        if self.device != "cpu":
            raise ValueError(
                "Stage 25 collection supports only CPU device 'cpu'"
            )
        require_nonnegative_number(
            "task_progress_potential_weight",
            self.task_progress_potential_weight,
            error_suffix="must be nonnegative and finite",
        )
        # Sanitize-store the base float (the S2-6/A-3 hardening standard) so a
        # hostile numeric subclass cannot hijack the > 0.0 activation test or the
        # phi multiply through overridden operators. (The DR-C4 sibling
        # task_progress_potential_weight predates this standard; hardened here for
        # the new field per the SHIFT-10 adversarial pass.)
        sensing_credit_potential_weight = require_nonnegative_number(
            "sensing_credit_potential_weight",
            self.sensing_credit_potential_weight,
            error_suffix="must be nonnegative and finite",
        )
        object.__setattr__(
            self,
            "sensing_credit_potential_weight",
            sensing_credit_potential_weight,
        )
        # DR-D1/L1-b AMENDMENT (SHIFT 13): same sanitize-store hardening for the
        # decoupled decision-region positioning weight.
        sensing_zone_potential_weight = require_nonnegative_number(
            "sensing_zone_potential_weight",
            self.sensing_zone_potential_weight,
            error_suffix="must be nonnegative and finite",
        )
        object.__setattr__(
            self,
            "sensing_zone_potential_weight",
            sensing_zone_potential_weight,
        )


@dataclass(frozen=True)
class Stage25CollectedRollout:
    """In-memory Stage 25 collection: merged batch, records, and metadata.

    ``episode_records`` carries the Stage 23-C 18-field per-episode records
    (``round_index`` defaults to 0; the driver overwrites it per round).
    ``metadata`` carries the episode-count / valid-step accounting the DR-D2
    episodic constraint estimator and the driver's curve records consume.

    ``potential`` / ``potential_next`` carry the DR-C4 potential-based
    task-progress shaping tensors ``Phi(s_t)`` / ``Phi(s_{t+1})``, both
    ``[batch_rows, max_time]`` aligned exactly with ``batch.task_reward`` (padded
    steps zeroed). They are ``None`` when shaping is OFF
    (``task_progress_potential_weight == 0.0``), in which case the update path is
    byte-identical to the DR-D4 path. The reward-shaping term is formed as
    ``gamma * potential_next - potential`` inside the update (single source of
    the potential-difference form), NEVER added to hazard_cost, and NEVER
    persisted into any graded success/hazard/sensing field (AM-2 / Stage 26
    integrity).
    """

    batch: RolloutBatch
    episode_records: tuple[dict[str, Any], ...]
    metadata: dict[str, Any]
    potential: "torch.Tensor | None" = None
    potential_next: "torch.Tensor | None" = None
    # DR-D1/L1-b lever (1): the decision-relevant sensing credit potential
    # ``Phi_sense(s_t)`` / ``Phi_sense(s_{t+1})``, [batch_rows, max_time] aligned
    # exactly with ``batch.task_reward`` (padded steps zeroed). ``None`` when the
    # credit is OFF (``sensing_credit_potential_weight == 0.0``) => byte-identical
    # DR-C4/DR-D4 path. Formed as ``gamma * sensing_potential_next -
    # sensing_potential`` inside the update (the same policy-invariant
    # potential-difference form as the task potential), added ONLY to the reward
    # channel, NEVER to hazard_cost, and NEVER persisted into any graded field
    # (AM-2 / Stage 26 integrity).
    sensing_potential: "torch.Tensor | None" = None
    sensing_potential_next: "torch.Tensor | None" = None


def collect_stage25_training_rollout(
    model: RecurrentMAPPOActorCritic,
    collection_config: Stage25CollectionConfig | None = None,
) -> Stage25CollectedRollout:
    """Collect one Stage 25 update round's on-policy batch in memory.

    One fresh Stage 23-A environment per episode, scenario chosen round-robin
    from ``scenario_names``, reset with ``seed = config.seed + episode_index``;
    variable-length episodes merged on the batch axis with zero padding and
    ``valid_mask`` False on padded steps; ``model.eval()`` during collection
    with train-mode restore; all stored tensors detached. The merged batch
    passes ``RolloutBatch.validate`` plus the shared batch-value re-checks
    before return. The model config must equal ``default_stage23_core_config``
    exactly (enforced by the shared validator).
    """

    if not isinstance(model, RecurrentMAPPOActorCritic):
        raise TypeError("model must be a RecurrentMAPPOActorCritic")
    if collection_config is None:
        settings = Stage25CollectionConfig()
    elif isinstance(collection_config, Stage25CollectionConfig):
        settings = collection_config
    else:
        raise TypeError(
            "collection_config must be a Stage25CollectionConfig or None"
        )

    core_config = model.config
    validate_stage23_core_config(core_config, context="Stage 25")
    set_rollout_seed(settings.seed)

    shaping_active = settings.task_progress_potential_weight > 0.0
    # DR-D1/L1-b lever (1): the decision-relevant sensing credit potential is
    # computed here (Stage 25 side), exactly like the DR-C4 task potential, only
    # when its weight is positive (OFF => None => byte-identical path).
    sensing_active = (
        settings.sensing_credit_potential_weight > 0.0
        or settings.sensing_zone_potential_weight > 0.0
    )
    tensor_device = torch.device(settings.device)

    was_training = model.training
    model.eval()
    try:
        episodes: list[dict[str, torch.Tensor]] = []
        episode_records: list[dict[str, Any]] = []
        # DR-C4: per-episode (Phi(s_t), Phi(s_{t+1})) pairs, aligned with the
        # episode order that merge_padded_episodes concatenates (ep0 rows, ep1
        # rows, ...); populated only when shaping is active.
        episode_potentials: list[tuple[torch.Tensor, torch.Tensor]] = []
        distance_fields: dict[str, dict[tuple[int, int], int]] = {}
        # DR-D1/L1-b: per-episode (Phi_sense(s_t), Phi_sense(s_{t+1})) pairs +
        # the observable public-risk-zone cache (frozenset per scenario name),
        # populated only when the sensing credit is active.
        episode_sensing_potentials: list[tuple[torch.Tensor, torch.Tensor]] = []
        risk_zone_fields: dict[str, frozenset[tuple[int, int]]] = {}
        for episode_index in range(settings.max_episodes):
            scenario_name = settings.scenario_names[
                episode_index % len(settings.scenario_names)
            ]
            episode_seed = settings.seed + episode_index
            environment = make_default_stage23c_environment(
                scenario_name=scenario_name,
                max_steps=settings.steps_per_episode,
                seed=episode_seed,
            )
            # The imported Stage 23-C episode collector reads exactly
            # settings.seed / settings.sample_actions / settings.dtype /
            # settings.device from its settings argument; this Stage 25
            # config carries all four with identical semantics. It never reads
            # task_progress_potential_weight, so the shaping is computed here
            # (Stage 25 side) without touching the stage23c collector.
            episode, record = _collect_single_stage23c_episode(
                model,
                environment,
                settings,
                core_config,
                episode_index=episode_index,
                seed=episode_seed,
            )
            episodes.append(episode)
            episode_records.append(record)
            if shaping_active or sensing_active:
                scenario = environment.scenario
                # The environment retains its final positions post-terminal /
                # post-truncation (critic_visible_state is valid there), so s_L
                # is read directly for the truncation potential bootstrap. Shared
                # by both potential channels.
                final_positions = _final_agent_positions(environment)
                if shaping_active:
                    if scenario.name not in distance_fields:
                        distance_fields[scenario.name] = _bfs_goal_distance_field(
                            scenario
                        )
                    episode_potentials.append(
                        _episode_potentials(
                            episode,
                            scenario=scenario,
                            distance_field=distance_fields[scenario.name],
                            weight=settings.task_progress_potential_weight,
                            final_positions=final_positions,
                            device=tensor_device,
                        )
                    )
                if sensing_active:
                    if scenario.name not in risk_zone_fields:
                        # The observable public risk zone is layout-agnostic
                        # (reads obstacles/wall-column only) => identical across a
                        # fork group's aliased layouts (M-5-safe). Empty for
                        # scenarios with no public zone => no credit there.
                        risk_zone_fields[scenario.name] = frozenset(
                            _stage24a_public_risk_zone_cells(scenario.name)
                        )
                    episode_sensing_potentials.append(
                        _episode_sensing_potentials(
                            episode,
                            scenario=scenario,
                            zone=risk_zone_fields[scenario.name],
                            credit_weight=settings.sensing_credit_potential_weight,
                            zone_weight=settings.sensing_zone_potential_weight,
                            final_positions=final_positions,
                            device=tensor_device,
                        )
                    )

        max_time = max(int(episode["valid_mask"].shape[1]) for episode in episodes)
        batch = merge_padded_episodes(episodes, max_time=max_time)
        batch.validate(core_config)
        validate_rollout_batch_values(batch)

        potential: torch.Tensor | None = None
        potential_next: torch.Tensor | None = None
        if shaping_active:
            potential = _merge_potentials(
                [phi for phi, _ in episode_potentials], max_time=max_time
            )
            potential_next = _merge_potentials(
                [phi_next for _, phi_next in episode_potentials], max_time=max_time
            )
            # Alignment invariant: the shaping tensors must match the reward
            # channel exactly so ``development_reward_signal + shaping`` is
            # well-defined per valid step.
            if potential.shape != batch.task_reward.shape:
                raise ValueError(
                    "DR-C4 potential tensor shape must match batch.task_reward"
                )
        sensing_potential: torch.Tensor | None = None
        sensing_potential_next: torch.Tensor | None = None
        if sensing_active:
            sensing_potential = _merge_potentials(
                [phi for phi, _ in episode_sensing_potentials], max_time=max_time
            )
            sensing_potential_next = _merge_potentials(
                [phi_next for _, phi_next in episode_sensing_potentials],
                max_time=max_time,
            )
            if sensing_potential.shape != batch.task_reward.shape:
                raise ValueError(
                    "DR-D1/L1-b sensing potential tensor shape must match "
                    "batch.task_reward"
                )
    finally:
        model.train(was_training)

    episode_count = len(episode_records)
    batch_rows = int(batch.actor_observation.shape[0])
    metadata: dict[str, Any] = {
        "agent_count": batch_rows // episode_count,
        "batch_rows": batch_rows,
        "episode_count": episode_count,
        "max_collected_episode_length": max_time,
        "reward_shaping_active": shaping_active,
        "sample_actions": settings.sample_actions,
        "scenario_names": list(settings.scenario_names),
        "seed": settings.seed,
        "steps_per_episode": settings.steps_per_episode,
        "task_progress_potential_weight": float(
            settings.task_progress_potential_weight
        ),
        "sensing_credit_active": sensing_active,
        "sensing_credit_potential_weight": float(
            settings.sensing_credit_potential_weight
        ),
        "valid_step_count": int(batch.valid_mask.sum().item()),
    }
    return Stage25CollectedRollout(
        batch=batch,
        episode_records=tuple(episode_records),
        metadata=metadata,
        potential=potential,
        potential_next=potential_next,
        sensing_potential=sensing_potential,
        sensing_potential_next=sensing_potential_next,
    )


# ---------------------------------------------------------------------------
# DR-C4 potential-based task-progress shaping (Fix 1) -- collection side
# ---------------------------------------------------------------------------
# Phi_i(s) = -w_Phi * BFS_dist(agent_i cell -> nearest goal) on the known static
# layout. The potential is hazard-AGNOSTIC (avoid_cells=()): it respects only the
# obstacle geometry, so the constraint -- not the potential -- bends the route
# around the hazard (DR-C4 risk note), and the identical potential applies across
# the training / readiness / held-out layouts (which share obstacles and goals,
# differing only in the hidden hazard), so no hazard location leaks through it.


def _bfs_goal_distance_field(scenario: Stage23Scenario) -> dict[tuple[int, int], int]:
    """Return the per-cell BFS geodesic distance to the nearest goal.

    Reuses the comparators' ``_shortest_path`` primitive per free cell so the
    shaping geometry is byte-for-byte the geometry the scripted comparators use
    (obstacles blocked; hazards NOT avoided). Goal cells map to 0; obstacle cells
    are omitted (an agent never occupies one); an unreachable free cell maps to a
    finite ``width * height`` upper bound so the potential stays finite.
    """

    goals = tuple(scenario.goals)
    goal_set = set(goals)
    obstacle_set = set(scenario.obstacles)
    unreachable = scenario.width * scenario.height
    field: dict[tuple[int, int], int] = {}
    for row in range(scenario.height):
        for col in range(scenario.width):
            cell = (row, col)
            if cell in obstacle_set:
                continue
            if cell in goal_set:
                field[cell] = 0
                continue
            path = _shortest_path(
                cell,
                goals,
                width=scenario.width,
                height=scenario.height,
                obstacles=scenario.obstacles,
                avoid_cells=(),
            )
            field[cell] = len(path) if path else unreachable
    return field


def _recover_cell(
    row_feature: float, col_feature: float, *, height: int, width: int
) -> tuple[int, int]:
    """Recover an integer grid cell from an actor-observation position feature.

    ``actor_observation_from_stage23`` encodes position as
    ``row / max(1, height - 1)`` and ``col / max(1, width - 1)``
    (tensor_adapter); this inverts that exactly. The round-trip is exact for the
    small integer grids in scope; the assertions fail loudly if that encoding
    convention ever drifts. (A mis-recovered cell would only degrade the
    efficacy of the shaping, never its policy-invariance -- telescoping holds for
    ANY deterministic state function -- but a loud failure is preferred.)
    """

    row_scaled = row_feature * float(max(1, height - 1))
    col_scaled = col_feature * float(max(1, width - 1))
    row = int(round(row_scaled))
    col = int(round(col_scaled))
    if abs(row_scaled - row) > 1e-3 or abs(col_scaled - col) > 1e-3:
        raise ValueError(
            "recovered actor-observation position is not integer-valued; the "
            "position-feature encoding convention has drifted"
        )
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(
            "recovered actor-observation position is outside the grid"
        )
    return (row, col)


def _final_agent_positions(
    environment: RiskAwareActiveSensingGridEnvironment,
) -> dict[str, tuple[int, int]]:
    """Return the environment's retained final agent cells as ``(row, col)``."""

    positions = environment.critic_visible_state()["positions"]
    return {
        agent: (int(cell[0]), int(cell[1])) for agent, cell in positions.items()
    }


def _episode_potentials(
    episode: dict[str, torch.Tensor],
    *,
    scenario: Stage23Scenario,
    distance_field: dict[tuple[int, int], int],
    weight: float,
    final_positions: dict[str, tuple[int, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(Phi(s_t), Phi(s_{t+1}))`` tensors ``[agents, length]`` for one episode.

    ``Phi(s_{t+1})`` uses the DR-C4 boundary handling that matches the GAE
    bootstrap convention: for non-final steps it is next-step's own potential;
    for the final step it is 0 on a TERMINAL ending (the zeroed bootstrap) and
    the true post-episode potential ``Phi(s_L)`` on a TRUNCATION ending
    (preserved). The collector already rejected partial/mixed team endings, so
    agent 0's flag is the team flag.
    """

    actor_observation = episode["actor_observation"]
    agent_count = int(actor_observation.shape[0])
    length = int(actor_observation.shape[1])
    height = scenario.height
    width = scenario.width
    unreachable = width * height
    terminal_last = bool(episode["terminal"][0, length - 1].item())

    phi = torch.zeros((agent_count, length), dtype=torch.float32, device=device)
    for agent_index in range(agent_count):
        for step_index in range(length):
            cell = _recover_cell(
                float(actor_observation[agent_index, step_index, 0].item()),
                float(actor_observation[agent_index, step_index, 1].item()),
                height=height,
                width=width,
            )
            phi[agent_index, step_index] = -weight * float(
                distance_field.get(cell, unreachable)
            )

    phi_next = torch.zeros(
        (agent_count, length), dtype=torch.float32, device=device
    )
    if length > 1:
        phi_next[:, : length - 1] = phi[:, 1:]
    if not terminal_last:
        for agent_index in range(agent_count):
            cell = final_positions[STAGE23_AGENT_NAMES[agent_index]]
            phi_next[agent_index, length - 1] = -weight * float(
                distance_field.get(cell, unreachable)
            )
    # terminal_last: phi_next[:, length - 1] stays 0.0 (zeroed bootstrap).
    return phi, phi_next


def _episode_sensing_potentials(
    episode: dict[str, torch.Tensor],
    *,
    scenario: Stage23Scenario,
    zone: frozenset[tuple[int, int]],
    credit_weight: float,
    zone_weight: float,
    final_positions: dict[str, tuple[int, int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the DR-D1/L1-b sensing potentials ``(Phi(s_t), Phi(s_{t+1}))`` — a composed,
    FIXED, OBSERVABLE state potential (policy-INVARIANT; NHR telescopes to
    ``gamma^L*Phi(s_L)-Phi(s_0)``):

    ``Phi_i(s) = zone_weight * 1[cell_i(s) in zone]``                        (DR-D1/L1-b
                                                                             AMENDMENT lever 1
                                                                             DECOUPLE: the
                                                                             decision-region
                                                                             POSITIONING term)
    ``          + credit_weight * 1[cell_i(s) in zone] * 1[previous_sensed_i(s)]``  (the
                                                                             original SHIFT-10
                                                                             prev_sensed credit,
                                                                             retained OFF-default)

    ``cell`` is recovered from features 0,1 via the shared ``_recover_cell``; ``previous_sensed``
    is actor-observation feature INDEX 5 (``1.0/0.0`` encoding; ``>0.5`` dtype-safe); ``zone`` is
    the observable public risk zone. Neither term reads the POLICY or the hidden hazard (``zone``
    is layout-agnostic ⇒ M-5-safe: byte-identical across a fork group's aliased layouts), so this
    is a genuine state function.

    Both terms share the SAME ZONE-FIRST gate (``cell in zone``) at EVERY step INCLUDING the
    truncation bootstrap (AM-1): a cell OUTSIDE the zone contributes 0 to both terms, so
    ``Phi(s_L) = 0`` on a truncation ending outside the zone (required for the telescope) and the
    decoupled ``zone_weight=0.0`` path is byte-identical to the SHIFT-10 prev_sensed-only credit
    (in-zone-not-sensed writes the additive 0.0; not-in-zone writes nothing). Boundary handling is
    IDENTICAL to ``_episode_potentials`` (AM-2, so ``Phi_total = Phi_BFS + Phi_sense`` telescopes):
    non-final steps use next-step's own potential; the final step is 0 on a TERMINAL ending (zeroed
    bootstrap) and ``Phi(s_L)`` on a TRUNCATION ending (preserved, using the retained final cell +
    the sense action on the final collected step). At reset ``cell_i(s_0)`` is the start cell
    (outside the zone on the fork) ⇒ ``Phi(s_0) = 0``.
    """

    actor_observation = episode["actor_observation"]
    sensing_action = episode["sensing_action"]
    agent_count = int(actor_observation.shape[0])
    length = int(actor_observation.shape[1])
    height = scenario.height
    width = scenario.width
    terminal_last = bool(episode["terminal"][0, length - 1].item())
    credit_active = credit_weight != 0.0

    phi = torch.zeros((agent_count, length), dtype=torch.float32, device=device)
    for agent_index in range(agent_count):
        for step_index in range(length):
            # ZONE-FIRST gate (shared by both terms). A cell outside the observable
            # public risk zone contributes 0 to both the positioning term and the
            # prev_sensed credit.
            cell = _recover_cell(
                float(actor_observation[agent_index, step_index, 0].item()),
                float(actor_observation[agent_index, step_index, 1].item()),
                height=height,
                width=width,
            )
            if cell not in zone:
                continue
            value = zone_weight
            # previous_sensed is actor_observation feature index 5 (observable);
            # >0.5 threshold is dtype-safe on the 0.0/1.0 encoding.
            if (
                credit_active
                and float(actor_observation[agent_index, step_index, 5].item()) > 0.5
            ):
                value += credit_weight
            phi[agent_index, step_index] = value

    phi_next = torch.zeros(
        (agent_count, length), dtype=torch.float32, device=device
    )
    if length > 1:
        phi_next[:, : length - 1] = phi[:, 1:]
    if not terminal_last:
        # TRUNCATION bootstrap Phi(s_L): ZONE-FIRST gate on the retained final cell
        # (AM-1 — mirrors the per-step gate so Phi(s_L)=0 outside the zone). The
        # prev_sensed credit uses the sense action on the final collected step.
        for agent_index in range(agent_count):
            cell = final_positions[STAGE23_AGENT_NAMES[agent_index]]
            if cell not in zone:
                continue
            value = zone_weight
            if (
                credit_active
                and int(sensing_action[agent_index, length - 1].item())
                == STAGE23_SENSE_ACTION_INDEX
            ):
                value += credit_weight
            phi_next[agent_index, length - 1] = value
    # terminal_last: phi_next[:, length - 1] stays 0.0 (zeroed bootstrap).
    return phi, phi_next


def _merge_potentials(
    per_episode: list[torch.Tensor], *, max_time: int
) -> torch.Tensor:
    """Right-pad each ``[agents, length]`` potential to ``max_time`` and stack on rows.

    Mirrors ``merge_padded_episodes`` exactly: right-pad the time axis with zeros
    and concatenate on the batch (row) axis in episode order, so the result rows
    align one-to-one with the merged RolloutBatch rows.
    """

    if not per_episode:
        raise ValueError("per_episode must be nonempty")
    padded: list[torch.Tensor] = []
    for tensor in per_episode:
        length = int(tensor.shape[1])
        if length > max_time:
            raise ValueError("cannot pad a potential longer than max_time")
        if length < max_time:
            pad = torch.zeros(
                (int(tensor.shape[0]), max_time - length),
                dtype=tensor.dtype,
                device=tensor.device,
            )
            tensor = torch.cat([tensor, pad], dim=1)
        padded.append(tensor)
    return torch.cat(padded, dim=0)

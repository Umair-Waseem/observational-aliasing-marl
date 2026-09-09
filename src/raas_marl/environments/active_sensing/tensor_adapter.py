"""Stage 23-A adapter from environment payloads to Stage 21 core contracts."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import TYPE_CHECKING

from raas_marl.mappo_lagrangian._validation import (
    finite_numeric_scalar,
    nonempty_unpadded_string,
    require_int_not_bool,
    require_mapping,
    require_nonnegative_int,
    require_positive_int,
    require_string_mapping_keys,
)
from raas_marl.mappo_lagrangian.config import (
    MAPPOCoreConfig,
    require_finite_float_tensor,
    require_finite_float_tensor_matching_config,
    torch_dtype_from_name,
    torch_required,
    validate_central_state,
)
from raas_marl.mappo_lagrangian.env_contracts import (
    EnvironmentStepContract,
    validate_environment_step_contract,
)

if TYPE_CHECKING:  # pragma: no cover - annotation-only; never executed at runtime
    from types import ModuleType
    # This module must import without PyTorch installed, like the other import-safe
    # modules: `torch` is bound LOCALLY inside each converter via
    # `config.torch_required(...)`, never at module scope. `from __future__ import
    # annotations` (above) keeps the `-> torch.Tensor` returns as deferred strings, so
    # this import exists only to let static tools resolve the name. Adding the return
    # annotations WITHOUT this block introduces three new ruff F821 findings -- measured,
    # not assumed.
    import torch

from raas_marl.environments.active_sensing.grid_environment import (
    HAZARD_COST_KEY,
    MOVEMENT_ACTION_FIELD,
    SENSING_ACTION_FIELD,
    SENSING_COST_KEY,
    STAGE23_AGENT_NAMES,
    STAGE23_ACTOR_VISIBLE_SCHEMA_KEYS,
    STAGE23_FIXED_AGENT_COUNT,
    STAGE23_STAGE,
    TASK_REWARD_KEY,
    RiskAwareActiveSensingGridEnvironment,
    _validate_disjoint_layout_sets,
    validate_stage23_actor_observation,
)


# DR-REPRESENTATION (SHIFT 36, candidate A, countersigned wf_66def9e9-443): the actor
# perceives PUBLIC wall/gate structure as an EGOCENTRIC per-cell binary obstacle-
# occupancy PATCH over a Manhattan radius-2 window (candidate A), APPENDED at feature
# indices >=10 so the NHR/Phi-read indices 0,1,5 stay FIXED (append, not replace; the
# legacy index-9 scalar obstacle-count is RETAINED, redundant, NHR-inert). The patch
# is a deterministic function of scenario.obstacles ONLY (PUBLIC) => bitwise-identical
# between a fork pair's two aliased mirrors => sensing stays forced (BAR-A); it can
# NEVER leak the hidden hazard (surfaced only via the paid sensing/reveal path). The
# 12 offsets are the non-centre Manhattan-radius-2 cells in a FIXED canonical order
# (sorted by (dr, dc) ascending); the centre (0,0) is excluded (the agent's own cell
# is never an obstacle => a constant-0 dead feature). Off-grid cells encode as
# occupied (a wall). The (3,3)->(4,4) disambiguator is offset (1,1), Manhattan-2,
# in-window. See docs/decisions/DR-representation.md.
STAGE23_LEGACY_ACTOR_FEATURE_COUNT = 10
_RADIUS2_PATCH_OFFSETS = (
    (-2, 0),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -2),
    (0, -1),
    (0, 1),
    (0, 2),
    (1, -1),
    (1, 0),
    (1, 1),
    (2, 0),
)
STAGE23_ACTOR_OBSERVATION_DIM = STAGE23_LEGACY_ACTOR_FEATURE_COUNT + len(_RADIUS2_PATCH_OFFSETS)
STAGE23_REVEALED_INFORMATION_DIM = 4
STAGE23_HISTORY_STATE_DIM = 16
STAGE23_AGENT_COUNT = STAGE23_FIXED_AGENT_COUNT
# NEW-tensor_adapter-13: sensing/movement action-space sizes are named module
# constants so the default core config, the _core validator, and their error
# messages cannot drift from one another.
STAGE23_SENSING_ACTION_COUNT = 2
STAGE23_MOVEMENT_ACTION_COUNT = 5
# m-22: default sensing reveal radius used to derive the reveal-count normalizer
# cap. With Manhattan radius r the reveal window is a diamond of 2*r*(r+1)+1
# cells (own cell plus the r-ring), so the count feature saturates at its true
# maximum rather than a hardcoded 4. This constant is the default fallback ONLY
# (it mirrors grid_environment's default sensing_radius = 1); the production
# transition-contract path threads the environment's ACTUAL configured
# sensing_radius through revealed_information_from_stage23 so the cap is derived
# from the real radius, not this default, whenever radius != 1.
STAGE23_SENSING_RADIUS = 1
# m-22: default local-obstacle observation radius used to derive the
# obstacle-count normalizer cap. The obstacle observation window is a Manhattan
# diamond using grid_environment's local_observation_radius (which defaults to 1,
# equal to the sensing radius default). Like STAGE23_SENSING_RADIUS this is the
# default fallback ONLY; the production transition-contract path threads the
# environment's ACTUAL configured local_observation_radius through
# actor_observation_from_stage23 so the cap is derived from the real radius.
STAGE23_LOCAL_OBSTACLE_RADIUS = 2
# M-5 (code portion): the central critic vector is a fixed Stage 23-A two-agent
# schema. Signed position deltas are normalized [0,1] geometry features. The
# cumulative critic totals remain raw finite scalar totals in Stage 23-A after
# validation; whether to normalize them (and by which per-step magnitude bound)
# is a deferred Phase-6 decision. The per-step magnitude bounds are named below
# so that decision can reuse them; the emitted values are unchanged for now.
STAGE23_CENTRAL_STATE_FEATURE_SCHEMA = (
    "time_progress",
    "grid_width",
    "grid_height",
    "active_agent_count",
    "agent_0_row",
    "agent_0_col",
    "agent_1_row",
    "agent_1_col",
    "start_count",
    "goal_count",
    "obstacle_count",
    "hazard_cell_count",
    "team_success",
    "cumulative_task_reward",
    "cumulative_hazard_cost",
    "cumulative_sensing_cost",
    "bias",
    "sensing_action_count",
    "movement_action_count",
    "configured_agent_count",
)
STAGE23_CENTRAL_STATE_DIM = len(STAGE23_CENTRAL_STATE_FEATURE_SCHEMA)
_CENTRAL_FEATURE_BIAS = 1.0
# M-5 (code portion): per-step magnitude bounds for the cumulative totals. These
# are the natural per-step maxima for the Stage 23-A reward/cost signals; they
# name the scale the Phase-6 normalization decision would divide by. They do NOT
# rescale the emitted feature values today (deferred decision).
STAGE23_PER_STEP_TASK_REWARD_MAGNITUDE = 1.0
STAGE23_PER_STEP_HAZARD_COST_MAGNITUDE = 1.0
STAGE23_PER_STEP_SENSING_COST_MAGNITUDE = 1.0
_ACTOR_VISIBLE_REQUIRED_FIELDS = STAGE23_ACTOR_VISIBLE_SCHEMA_KEYS
_CENTRAL_REQUIRED_FIELDS = (
    "stage",
    "scenario_name",
    "positions",
    "agent_order",
    "starts",
    "goals",
    "obstacles",
    "hazard_cells",
    "width",
    "height",
    "step_index",
    "max_steps",
    "team_success",
    "cumulative",
)
_CUMULATIVE_REQUIRED_FIELDS = (TASK_REWARD_KEY, HAZARD_COST_KEY, SENSING_COST_KEY)
_TRANSITION_IDENTITY_REQUIRED_FIELDS = (
    "team_id",
    "agent_id",
    "agent_name",
    "agent_name_order",
    "team_agent_ids",
)
_TRANSITION_IDENTITY_OPTIONAL_COMPARE_FIELDS = ("agent_order", "agent_id_mapping")


def _reveal_count_cap(sensing_radius: int) -> float:
    """Return the maximum revealable/observable cell count for a Manhattan radius.

    m-22: a Manhattan-radius-``r`` window is a diamond of ``2*r*(r+1)+1`` cells
    (the own cell plus the ``r``-ring), so the reveal/obstacle count features
    saturate at their true maximum instead of a hardcoded 4.
    """

    radius = _nonnegative_int(sensing_radius, "sensing_radius")
    return float(2 * radius * (radius + 1) + 1)


def default_stage23_core_config() -> MAPPOCoreConfig:
    """Return the fixed two-agent Stage 23-A core tensor schema."""

    return MAPPOCoreConfig(
        actor_observation_dim=STAGE23_ACTOR_OBSERVATION_DIM,
        revealed_information_dim=STAGE23_REVEALED_INFORMATION_DIM,
        central_state_dim=STAGE23_CENTRAL_STATE_DIM,
        history_state_dim=STAGE23_HISTORY_STATE_DIM,
        actor_hidden_dim=32,
        critic_hidden_dim=32,
        sensing_action_count=STAGE23_SENSING_ACTION_COUNT,
        movement_action_count=STAGE23_MOVEMENT_ACTION_COUNT,
        recurrent_layer_count=1,
        agent_id_count=STAGE23_AGENT_COUNT,
        use_recurrent_actor=True,
        dtype="float32",
        device="cpu",
    )


def actor_observation_from_stage23(
    observation: Mapping[str, object],
    *,
    config: MAPPOCoreConfig | None = None,
    dtype: str | None = None,
    device: str | None = None,
    local_observation_radius: int = STAGE23_LOCAL_OBSTACLE_RADIUS,
) -> torch.Tensor:
    """Convert one Stage 23-A observation to an actor-observation tensor.

    m-22: ``local_observation_radius`` sets the Manhattan radius used to derive
    the obstacle-count normalizer cap. It defaults to
    ``STAGE23_LOCAL_OBSTACLE_RADIUS`` (1) so existing callers are unaffected; the
    production transition-contract path passes the environment's ACTUAL
    configured ``local_observation_radius`` so the cap is derived from the real
    radius instead of the hardcoded default whenever radius != 1.
    """

    core = _core(config)
    actor_visible = _actor_visible_mapping(
        observation,
        required_fields=_ACTOR_VISIBLE_REQUIRED_FIELDS,
    )
    grid_shape = _grid_shape(actor_visible["grid_shape"], "grid_shape")
    position = _in_grid_cell(actor_visible["position"], "position", grid_shape)
    goal_delta = _int_pair(actor_visible["nearest_goal_delta"], "nearest_goal_delta")
    goal_target = (position[0] + goal_delta[0], position[1] + goal_delta[1])
    _require_cell_in_grid("nearest_goal_delta target", goal_target, grid_shape)
    local_obstacles = _cell_list(
        actor_visible["local_obstacles"],
        "local_obstacles",
        grid_shape=grid_shape,
        require_unique=True,
    )
    # NEW-tensor_adapter-12: the schema validator (validate_stage23_actor_observation)
    # already guarantees revealed_local_hazards is well-formed for the actor tensor,
    # whose only reveal-derived feature is the obstacle-count cap; the reveal cells
    # themselves are consumed by revealed_information_from_stage23, not here.
    step_index = _nonnegative_int(actor_visible["step_index"], "step_index")
    max_steps = _positive_int(actor_visible["max_steps"], "max_steps")
    time_progress = _step_progress(step_index, max_steps)
    previous_sensed = _strict_bool(actor_visible["previous_sensed"], "previous_sensed")
    previous_invalid_move = _strict_bool(
        actor_visible["previous_invalid_move"],
        "previous_invalid_move",
    )
    previous_blocked_move = _strict_bool(
        actor_visible["previous_blocked_move"],
        "previous_blocked_move",
    )
    previous_entered_hazard = _strict_bool(
        actor_visible["previous_entered_hazard"],
        "previous_entered_hazard",
    )
    torch, tensor_dtype, tensor_device = _torch_runtime(core, dtype=dtype, device=device)
    obstacle_cap = _reveal_count_cap(local_observation_radius)
    features = [
        # m-22 / NEW-tensor_adapter-7: position and signed goal deltas both use the
        # max(1, N-1) denominator so the [0,1] position convention and the delta
        # convention share one denominator.
        _ratio(position[0], max(1, grid_shape[0] - 1)),
        _ratio(position[1], max(1, grid_shape[1] - 1)),
        _ratio(goal_delta[0], max(1, grid_shape[0] - 1)),
        _ratio(goal_delta[1], max(1, grid_shape[1] - 1)),
        time_progress,
        1.0 if previous_sensed else 0.0,
        1.0 if previous_invalid_move else 0.0,
        1.0 if previous_blocked_move else 0.0,
        1.0 if previous_entered_hazard else 0.0,
        min(float(len(local_obstacles)), obstacle_cap) / obstacle_cap,
    ]
    # DR-REPRESENTATION (candidate A): append the EGOCENTRIC radius-2 obstacle-
    # occupancy PATCH at indices >=10 (the legacy 10 features above are UNCHANGED, so
    # NHR/Phi indices 0,1,5 and the index-9 scalar stay fixed). ``local_obstacles`` is
    # the set of PUBLIC obstacle cells within the radius-2 window (env _observation
    # reads scenario.obstacles ONLY; the patch NEVER reads the reveal/hidden channel).
    # occupancy = 1.0 iff the offset cell is an obstacle OR off-grid (a wall), else
    # 0.0; the offsets are the fixed canonical non-centre Manhattan-radius-2 set.
    # COUPLING GUARD (impl-audit MINOR): the patch offsets span Manhattan-2, so
    # ``local_obstacles`` MUST cover a radius >= 2 window or radius-2 cells would be
    # mis-read as free. Fail LOUD on a radius < 2 misconfig (production threads the
    # env's configured radius, default 2; this can only under-report PUBLIC obstacles
    # and never surfaces the hidden hazard, but is pinned so it cannot drift silently).
    if local_observation_radius < 2:
        raise ValueError(
            "actor observation patch requires local_observation_radius >= 2 "
            f"(got {local_observation_radius}); the radius-2 occupancy patch would "
            "otherwise query cells outside the local_obstacles window"
        )
    local_obstacle_set = set(local_obstacles)
    for offset_row, offset_col in _RADIUS2_PATCH_OFFSETS:
        cell = (position[0] + offset_row, position[1] + offset_col)
        in_grid = 0 <= cell[0] < grid_shape[0] and 0 <= cell[1] < grid_shape[1]
        features.append(
            1.0 if (not in_grid) or (cell in local_obstacle_set) else 0.0
        )
    tensor = torch.tensor(features, dtype=tensor_dtype, device=tensor_device).reshape(1, 1, -1)
    # NEW-tensor_adapter-2 / FIX-2: route the actor tensor through the shared
    # finite/float validator (finiteness + float dtype + strided/non-meta) AND
    # assert its dtype/device match the resolved core config, mirroring the
    # central adapter's up-front validate_central_state dtype rejection. A
    # config-divergent dtype override (e.g. dtype='float64' under a float32 core)
    # is now rejected at emission instead of silently emitting a divergent
    # tensor. The default dtype='float32'/'cpu' path matches the core config and
    # is byte-identical (validation only; the tensor is unchanged).
    require_finite_float_tensor_matching_config("actor_observation", tensor, core)
    _require_tensor_shape("actor_observation", tensor, (1, 1, core.actor_observation_dim))
    return tensor


def revealed_information_from_stage23(
    observation: Mapping[str, object],
    *,
    config: MAPPOCoreConfig | None = None,
    dtype: str | None = None,
    device: str | None = None,
    sensing_radius: int = STAGE23_SENSING_RADIUS,
) -> torch.Tensor:
    """Convert current-step local reveal information to a tensor.

    m-22: ``sensing_radius`` sets the Manhattan radius used to derive the
    reveal-count normalizer cap. It defaults to ``STAGE23_SENSING_RADIUS`` (1) so
    existing callers are unaffected; the production transition-contract path
    passes the environment's ACTUAL configured ``sensing_radius`` so the cap is
    derived from the real radius instead of the hardcoded default whenever
    radius != 1.
    """

    core = _core(config)
    # m-21: the reveal tensor needs the full validated actor observation (its
    # schema validator guarantees position/grid_shape/revealed_local_hazards),
    # so it uses the same full-schema required fields as the actor tensor rather
    # than an always-redundant 3-field subset.
    actor_visible = _actor_visible_mapping(
        observation,
        required_fields=_ACTOR_VISIBLE_REQUIRED_FIELDS,
    )
    grid_shape = _grid_shape(actor_visible["grid_shape"], "grid_shape")
    position = _in_grid_cell(actor_visible["position"], "position", grid_shape)
    revealed = _cell_list(
        actor_visible["revealed_local_hazards"],
        "revealed_local_hazards",
        grid_shape=grid_shape,
        require_unique=True,
    )
    torch, tensor_dtype, tensor_device = _torch_runtime(core, dtype=dtype, device=device)
    reveal_cap = _reveal_count_cap(sensing_radius)
    if revealed:
        nearest = _nearest_cell(revealed, position)
        features = [
            1.0,
            min(float(len(revealed)), reveal_cap) / reveal_cap,
            # m-22 / NEW-tensor_adapter-7/11: signed reveal deltas share the
            # max(1, N-1) denominator with the position and goal-delta features.
            _ratio(nearest[0] - position[0], max(1, grid_shape[0] - 1)),
            _ratio(nearest[1] - position[1], max(1, grid_shape[1] - 1)),
        ]
    else:
        features = [0.0, 0.0, 0.0, 0.0]
    tensor = torch.tensor(features, dtype=tensor_dtype, device=tensor_device).reshape(1, 1, -1)
    # NEW-tensor_adapter-3 / FIX-2: full finite/float/shape validation AND a
    # dtype/device-vs-core-config check, mirroring the central adapter. A
    # config-divergent dtype override is now rejected at emission instead of
    # silently emitting a divergent tensor. The default float32/cpu path matches
    # the core config and is byte-identical (validation only; tensor unchanged).
    require_finite_float_tensor_matching_config("revealed_information", tensor, core)
    _require_tensor_shape("revealed_information", tensor, (1, 1, core.revealed_information_dim))
    return tensor


def central_state_from_stage23(
    environment_or_payload: RiskAwareActiveSensingGridEnvironment | Mapping[str, object],
    *,
    config: MAPPOCoreConfig | None = None,
    dtype: str | None = None,
    device: str | None = None,
) -> torch.Tensor:
    """Convert Stage 23-A centralized critic-visible state to a tensor.

    NEW-tensor_adapter-14 (Rule 8): this is the single privileged read path onto
    the environment's critic-visible state. The payload intentionally carries the
    hidden ``hazard_cells`` (consumed below) and cumulative totals, which are
    privileged and must NEVER reach an actor observation; only the centralized
    critic sees them, via this adapter and ``CriticInput``.
    """

    core = _core(config)
    if isinstance(environment_or_payload, RiskAwareActiveSensingGridEnvironment):
        payload = environment_or_payload.critic_visible_state()
    elif isinstance(environment_or_payload, Mapping):
        payload = environment_or_payload
    else:
        raise TypeError("environment_or_payload must be a Stage 23-A environment or mapping")
    _require_keys(payload, _CENTRAL_REQUIRED_FIELDS, "central payload")
    stage = nonempty_unpadded_string("central payload.stage", payload["stage"])
    if stage != STAGE23_STAGE:
        raise ValueError(f"central payload stage must equal {STAGE23_STAGE!r}")
    nonempty_unpadded_string("central payload.scenario_name", payload["scenario_name"])
    width = _positive_int(payload["width"], "width")
    height = _positive_int(payload["height"], "height")
    grid_shape = (height, width)
    positions = _position_mapping(payload["positions"], "positions", grid_shape=grid_shape)
    if set(positions) != set(STAGE23_AGENT_NAMES):
        raise ValueError("positions keys must exactly equal {'agent_0', 'agent_1'}")
    # NEW-tensor_adapter-4: type-guard agent_order (a non-sequence raises TypeError)
    # and compare order-as-list so a correct tuple ordering is accepted.
    agent_order = _string_sequence(payload["agent_order"], "agent_order")
    if agent_order != list(STAGE23_AGENT_NAMES):
        raise ValueError("agent_order must exactly equal ['agent_0', 'agent_1']")
    starts = _cell_list(payload["starts"], "starts", grid_shape=grid_shape, require_unique=True)
    goals = _cell_list(payload["goals"], "goals", grid_shape=grid_shape, require_unique=True)
    obstacles = _cell_list(
        payload["obstacles"],
        "obstacles",
        grid_shape=grid_shape,
        require_unique=True,
    )
    hazard_cells = _cell_list(
        payload["hazard_cells"],
        "hazard_cells",
        grid_shape=grid_shape,
        require_unique=True,
    )
    step_index = _nonnegative_int(payload["step_index"], "step_index")
    max_steps = _positive_int(payload["max_steps"], "max_steps")
    time_progress = _step_progress(step_index, max_steps)
    team_success = _strict_bool(payload["team_success"], "team_success")
    cumulative = require_mapping("cumulative", payload["cumulative"])
    _require_keys(cumulative, _CUMULATIVE_REQUIRED_FIELDS, "cumulative")
    _validate_central_layout(
        width=width,
        height=height,
        starts=starts,
        goals=goals,
        obstacles=obstacles,
        hazard_cells=hazard_cells,
    )
    cumulative_task_reward = _finite_number(
        cumulative[TASK_REWARD_KEY],
        f"cumulative.{TASK_REWARD_KEY}",
    )
    cumulative_hazard_cost = _nonnegative_finite_number(
        cumulative[HAZARD_COST_KEY],
        f"cumulative.{HAZARD_COST_KEY}",
    )
    cumulative_sensing_cost = _nonnegative_finite_number(
        cumulative[SENSING_COST_KEY],
        f"cumulative.{SENSING_COST_KEY}",
    )
    position_features: list[float] = []
    for agent in STAGE23_AGENT_NAMES:
        cell = positions[agent]
        position_features.extend([
            _ratio(cell[0], max(1, height - 1)),
            _ratio(cell[1], max(1, width - 1)),
        ])
    features_by_name = {
        "time_progress": time_progress,
        "grid_width": float(width),
        "grid_height": float(height),
        "active_agent_count": float(len(positions)),
        "agent_0_row": position_features[0],
        "agent_0_col": position_features[1],
        "agent_1_row": position_features[2],
        "agent_1_col": position_features[3],
        "start_count": float(len(starts)),
        "goal_count": float(len(goals)),
        "obstacle_count": float(len(obstacles)),
        "hazard_cell_count": float(len(hazard_cells)),
        "team_success": 1.0 if team_success else 0.0,
        # M-5 (deferred): cumulative totals emitted raw (unnormalized). The
        # STAGE23_PER_STEP_*_MAGNITUDE constants name the scale a future Phase-6
        # normalization would divide by; the emitted values are unchanged here.
        "cumulative_task_reward": cumulative_task_reward,
        "cumulative_hazard_cost": cumulative_hazard_cost,
        "cumulative_sensing_cost": cumulative_sensing_cost,
        "bias": _CENTRAL_FEATURE_BIAS,
        "sensing_action_count": float(core.sensing_action_count),
        "movement_action_count": float(core.movement_action_count),
        "configured_agent_count": float(STAGE23_AGENT_COUNT),
    }
    features = [
        features_by_name[name] for name in STAGE23_CENTRAL_STATE_FEATURE_SCHEMA
    ]
    if (
        len(STAGE23_CENTRAL_STATE_FEATURE_SCHEMA) != STAGE23_CENTRAL_STATE_DIM
        or len(features) != core.central_state_dim
    ):
        raise ValueError(
            "central_state dimension mismatch: "
            f"schema length={len(STAGE23_CENTRAL_STATE_FEATURE_SCHEMA)}; "
            f"feature count={len(features)}; "
            f"config central_state_dim={core.central_state_dim}"
        )
    torch, tensor_dtype, tensor_device = _torch_runtime(core, dtype=dtype, device=device)
    tensor = torch.tensor(features, dtype=tensor_dtype, device=tensor_device).reshape(1, 1, -1)
    validate_central_state(tensor, core)
    return tensor


def stage23_transition_to_environment_step_contract(
    observation: Mapping[str, object],
    info: Mapping[str, object],
    central_payload: RiskAwareActiveSensingGridEnvironment | Mapping[str, object],
    *,
    config: MAPPOCoreConfig | None = None,
) -> EnvironmentStepContract:
    """Build and validate a Stage 21 environment-step contract."""

    core = _core(config)
    observation_payload = require_mapping("observation", observation)
    info_payload = require_mapping("info", info)
    identity = _validate_transition_identity_coherence(observation_payload, info_payload)
    # m-22 (FIX-1): when the central payload is a live environment, thread its
    # ACTUAL configured radii into the count-cap derivation so the obstacle/reveal
    # count features saturate at the true maximum for that radius. Falls back to
    # the module-constant defaults (radius 1) for a plain mapping payload, so the
    # radius-1 production path is byte-identical.
    local_observation_radius, sensing_radius = _transition_count_cap_radii(central_payload)
    actor_tensor = actor_observation_from_stage23(
        observation, config=core, local_observation_radius=local_observation_radius
    ).reshape(-1)
    reveal_tensor = revealed_information_from_stage23(
        observation, config=core, sensing_radius=sensing_radius
    ).reshape(-1)
    central_tensor = central_state_from_stage23(central_payload, config=core).reshape(-1)
    rewards_and_costs = require_mapping(
        "rewards_and_costs", info_payload.get("rewards_and_costs")
    )
    done_flags = require_mapping("done_flags", info_payload.get("done_flags"))
    action_factors = require_mapping("action_factors", info_payload.get("action_factors"))
    _require_keys(done_flags, ("terminal", "truncated"), "done_flags")
    _require_keys(
        action_factors,
        (f"{SENSING_ACTION_FIELD}_available", f"{MOVEMENT_ACTION_FIELD}_available"),
        "action_factors",
    )
    terminal = _strict_bool(done_flags["terminal"], "done_flags.terminal")
    truncated = _strict_bool(done_flags["truncated"], "done_flags.truncated")
    sensing_available = _strict_bool(
        action_factors[f"{SENSING_ACTION_FIELD}_available"],
        f"action_factors.{SENSING_ACTION_FIELD}_available",
    )
    movement_available = _strict_bool(
        action_factors[f"{MOVEMENT_ACTION_FIELD}_available"],
        f"action_factors.{MOVEMENT_ACTION_FIELD}_available",
    )
    contract = EnvironmentStepContract(
        actor_visible={
            "actor_observation": actor_tensor,
            "revealed_information": reveal_tensor,
        },
        critic_visible={"central_state": central_tensor},
        rewards_and_costs=dict(rewards_and_costs),
        done_flags={
            "terminal": terminal,
            "truncated": truncated,
        },
        action_factors={
            f"{SENSING_ACTION_FIELD}_available": sensing_available,
            f"{MOVEMENT_ACTION_FIELD}_available": movement_available,
        },
        identity=dict(identity),
    )
    validate_environment_step_contract(contract, core)
    return contract


def _transition_count_cap_radii(
    central_payload: RiskAwareActiveSensingGridEnvironment | Mapping[str, object],
) -> tuple[int, int]:
    """Return the (local_observation_radius, sensing_radius) for the count caps.

    m-22 (FIX-1): reads the ACTUAL configured radii from a live Stage 23-A
    environment so the obstacle/reveal count-normalizer caps are derived from the
    real radius, not the hardcoded default. For a plain mapping payload (no
    config to consult) it falls back to the module-constant defaults (radius 1),
    keeping the radius-1 production path byte-identical. The radii are validated
    nonnegative ints, mirroring ``Stage23EnvironmentConfig``'s own validation.
    """

    if isinstance(central_payload, RiskAwareActiveSensingGridEnvironment):
        environment_config = central_payload.config
        local_observation_radius = _nonnegative_int(
            environment_config.local_observation_radius,
            "local_observation_radius",
        )
        sensing_radius = _nonnegative_int(
            environment_config.sensing_radius,
            "sensing_radius",
        )
        return local_observation_radius, sensing_radius
    return STAGE23_LOCAL_OBSTACLE_RADIUS, STAGE23_SENSING_RADIUS


def _core(config: MAPPOCoreConfig | None) -> MAPPOCoreConfig:
    if config is None:
        return default_stage23_core_config()
    if not isinstance(config, MAPPOCoreConfig):
        raise TypeError("config must be a MAPPOCoreConfig or None")
    if config.sensing_action_count != STAGE23_SENSING_ACTION_COUNT:
        raise ValueError(
            f"Stage 23-A requires sensing_action_count == {STAGE23_SENSING_ACTION_COUNT}"
        )
    if config.movement_action_count != STAGE23_MOVEMENT_ACTION_COUNT:
        raise ValueError(
            f"Stage 23-A requires movement_action_count == {STAGE23_MOVEMENT_ACTION_COUNT}"
        )
    if config.agent_id_count != STAGE23_AGENT_COUNT:
        raise ValueError(f"Stage 23-A requires agent_id_count == {STAGE23_AGENT_COUNT}")
    return config


def _torch_runtime(
    config: MAPPOCoreConfig, *, dtype: str | None, device: str | None
) -> tuple[ModuleType, torch.dtype, torch.device]:
    # NEW-tensor_adapter-8/10 and S2-3/D-7: import torch via the shared
    # config.torch_required and resolve the dtype via config.torch_dtype_from_name.
    # Only 'cpu' is accepted, so the device check is a single exact-string test;
    # the former strip()/whitespace/torch.device(try-except) branches were dead
    # relative to that exact-'cpu' assertion and have been removed.
    torch = torch_required("Stage 23-A tensor conversion")
    requested_dtype = dtype if dtype is not None else config.dtype
    requested_device = device if device is not None else config.device
    if requested_dtype not in ("float32", "float64"):
        raise ValueError("dtype must be 'float32' or 'float64'")
    # NEW-tensor_adapter-9: TypeError for non-string device (type error), ValueError
    # only for a wrong value.
    if not isinstance(requested_device, str):
        raise TypeError("device must be a string")
    if requested_device != "cpu":
        raise ValueError(
            "Stage 23-A tensor adapters support only CPU device 'cpu'; "
            f"got {requested_device!r}"
        )
    tensor_dtype = torch_dtype_from_name(requested_dtype)
    tensor_device = torch.device("cpu")
    return torch, tensor_dtype, tensor_device


def _require_tensor_shape(name: str, tensor: object, expected: tuple[int, ...]) -> None:
    shape = tuple(getattr(tensor, "shape", ()))
    if shape != expected:
        raise ValueError(
            f"{name} dimension mismatch: expected shape {expected}, got {shape}"
        )


def _actor_visible_mapping(
    observation: Mapping[str, object],
    *,
    required_fields: tuple[str, ...],
) -> Mapping[str, object]:
    # validate_stage23_actor_observation performs the recursive forbidden-key
    # scan over the whole observation (Rule 7), so a redundant one-level scan is
    # unnecessary here (former _reject_forbidden_actor_payload dropped, D-7).
    validate_stage23_actor_observation(observation)
    actor_visible = require_mapping("actor_visible", observation.get("actor_visible"))
    _require_keys(actor_visible, required_fields, "actor_visible")
    return actor_visible


def _validate_transition_identity_coherence(
    observation: Mapping[str, object],
    info: Mapping[str, object],
) -> Mapping[str, object]:
    if "identity" not in observation:
        raise ValueError("observation missing required field: identity")
    if "identity" not in info:
        raise ValueError("info missing required field: identity")
    observation_identity = require_mapping("observation.identity", observation.get("identity"))
    info_identity = require_mapping("info.identity", info.get("identity"))
    for field in _TRANSITION_IDENTITY_REQUIRED_FIELDS:
        if field not in observation_identity:
            raise ValueError(f"observation.identity missing required field: {field}")
        if field not in info_identity:
            raise ValueError(f"info.identity missing required field: {field}")
        # NEW-tensor_adapter-5: normalize sequence fields to a canonical list form
        # before comparison so a list-vs-tuple spelling of the same ordering does
        # not spuriously mismatch; scalar identity fields compare directly.
        if _identity_field_value(observation_identity[field]) != _identity_field_value(
            info_identity[field]
        ):
            raise ValueError(f"transition identity mismatch for {field}")
    for field in _TRANSITION_IDENTITY_OPTIONAL_COMPARE_FIELDS:
        if field in observation_identity and field in info_identity:
            if _identity_field_value(observation_identity[field]) != _identity_field_value(
                info_identity[field]
            ):
                raise ValueError(f"transition identity mismatch for {field}")
    if "agent" not in info:
        raise ValueError("info missing required field: agent")
    info_agent = nonempty_unpadded_string("info.agent", info["agent"])
    info_agent_name = nonempty_unpadded_string(
        "info.identity.agent_name",
        info_identity["agent_name"],
    )
    observation_agent_name = nonempty_unpadded_string(
        "observation.identity.agent_name",
        observation_identity["agent_name"],
    )
    if info_agent != info_agent_name:
        raise ValueError("info.agent must equal info.identity.agent_name")
    if info_agent != observation_agent_name:
        raise ValueError("info.agent must equal observation.identity.agent_name")
    return info_identity


def _identity_field_value(value: object) -> object:
    """Canonicalize an identity field for equality comparison.

    Sequence values (list/tuple) are normalized to a list so that an equal
    ordering spelled as either a list or a tuple compares equal; all other
    values compare by their own equality.
    """

    if isinstance(value, (list, tuple)):
        return list(value)
    return value


def _string_sequence(value: object, name: str) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    result: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str):
            raise TypeError(f"{name}[{index}] must be a string")
        result.append(item)
    return result


def _require_keys(payload: Mapping[str, object], required_fields: tuple[str, ...], name: str) -> None:
    missing = [field for field in required_fields if field not in payload]
    if missing:
        raise ValueError(f"{name} missing required field: {missing[0]}")


def _int_pair(value: object, name: str) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{name} must be a two-item list or tuple")
    first = _strict_int(value[0], f"{name}[0]")
    second = _strict_int(value[1], f"{name}[1]")
    return first, second


def _grid_shape(value: object, name: str) -> tuple[int, int]:
    height, width = _int_pair(value, name)
    if height <= 0 or width <= 0:
        raise ValueError(f"{name} dimensions must be positive")
    return height, width


def _cell_list(
    value: object,
    name: str,
    *,
    grid_shape: tuple[int, int] | None = None,
    require_unique: bool = False,
) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{name} must be a list or tuple")
    cells = tuple(
        _in_grid_cell(item, f"{name} item", grid_shape)
        if grid_shape is not None
        else _int_pair(item, f"{name} item")
        for item in value
    )
    if require_unique and len(set(cells)) != len(cells):
        raise ValueError(f"{name} must not contain duplicate cells")
    return cells


def _position_mapping(
    value: object,
    name: str,
    *,
    grid_shape: tuple[int, int],
) -> dict[str, tuple[int, int]]:
    mapping = require_mapping(name, value)
    require_string_mapping_keys(name, mapping)
    result: dict[str, tuple[int, int]] = {}
    for key, item in mapping.items():
        result[key] = _in_grid_cell(item, f"{name}.{key}", grid_shape)
    return result


def _nearest_cell(
    cells: tuple[tuple[int, int], ...],
    origin: tuple[int, int],
) -> tuple[int, int]:
    """Return the cell nearest ``origin`` by Manhattan distance.

    NEW-tensor_adapter-11: ties are broken deterministically by
    ``(distance, row, col)`` ascending.
    """

    return min(
        cells,
        key=lambda cell: (
            abs(cell[0] - origin[0]) + abs(cell[1] - origin[1]),
            cell[0],
            cell[1],
        ),
    )


def _in_grid_cell(
    value: object,
    name: str,
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    cell = _int_pair(value, name)
    _require_cell_in_grid(name, cell, grid_shape)
    return cell


def _require_cell_in_grid(
    name: str,
    cell: tuple[int, int],
    grid_shape: tuple[int, int],
) -> None:
    height, width = grid_shape
    row, column = cell
    if row < 0 or row >= height or column < 0 or column >= width:
        raise ValueError(f"{name} cell {cell!r} is outside grid_shape")


def _strict_int(value: object, name: str) -> int:
    # The shared _validation helpers validate in place and return None; these
    # thin wrappers preserve the (value, name) call order and int return used
    # across this module (the require_* helpers take (name, value)).
    require_int_not_bool(name, value)
    return value


def _positive_int(value: object, name: str) -> int:
    require_positive_int(name, value, error_suffix="must be positive")
    return value


def _nonnegative_int(value: object, name: str) -> int:
    require_nonnegative_int(name, value, error_suffix="must be nonnegative")
    return value


def _strict_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a bool")
    return value


def _finite_number(value: object, name: str) -> float:
    return finite_numeric_scalar(name, value)


def _nonnegative_finite_number(value: object, name: str) -> float:
    numeric = _finite_number(value, name)
    if numeric < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return numeric


def _step_progress(step_index: int, max_steps: int) -> float:
    if step_index > max_steps:
        raise ValueError("step_index must be <= max_steps")
    return _ratio(step_index, max_steps)


def _validate_central_layout(
    *,
    width: int,
    height: int,
    starts: tuple[tuple[int, int], ...],
    goals: tuple[tuple[int, int], ...],
    obstacles: tuple[tuple[int, int], ...],
    hazard_cells: tuple[tuple[int, int], ...],
) -> None:
    # Stage 23-A-specific presence/count checks the canonical disjointness helper
    # does not cover.
    if not starts:
        raise ValueError("starts must be nonempty")
    if len(starts) != STAGE23_AGENT_COUNT:
        raise ValueError("starts length must equal Stage 23-A fixed two-agent schema")
    if not goals:
        raise ValueError("goals must be nonempty")
    # D-7: reuse the canonical env-layout in-grid + disjointness check rather than
    # a copy (grid_environment._validate_disjoint_layout_sets). ``hazard_cells``
    # here are the privileged hidden hazard cells (Rule 8 central-state path).
    _validate_disjoint_layout_sets(
        width=width,
        height=height,
        starts=starts,
        goals=goals,
        obstacles=obstacles,
        hidden_hazard_cells=hazard_cells,
    )


def _ratio(numerator: int | float, denominator: int | float) -> float:
    numeric = (
        finite_numeric_scalar("ratio numerator", numerator)
        / finite_numeric_scalar("ratio denominator", denominator)
    )
    if not math.isfinite(numeric):
        raise ValueError("ratio must be finite")
    return numeric

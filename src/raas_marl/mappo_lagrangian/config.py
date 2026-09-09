"""Configuration and input-boundary validation for the Stage 21 core."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import TYPE_CHECKING

from raas_marl.mappo_lagrangian._validation import (
    require_minimum_int,
    require_nonnegative_int,
    require_nonnegative_number,
    require_open_unit_interval,
    require_positive_int,
    require_positive_number,
    require_probability,
)


if TYPE_CHECKING:  # pragma: no cover - annotation-only; never executed at runtime
    # ModuleType sits inside the guard for the same reason torch does: it appears
    # ONLY in a deferred annotation, so importing it at run time would be an
    # unnecessary runtime dependency in a module whose whole point is to import
    # with nothing installed.
    from types import ModuleType
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


_FORBIDDEN_ACTOR_INFORMATION_KEYS = frozenset(
    {
        "central_state",
        "central_info",
        "central_infos",
        "global_state",
        "privileged_state",
        "critic_state",
        "central_observation",
        "central_observations",
        "central_observation_state",
        "central_observation_map",
        "central_map",
        "central_maps",
        "global_info",
        "global_infos",
        "global_observation",
        "global_observations",
        "global_observation_state",
        "global_map",
        "global_maps",
        "privileged_info",
        "privileged_infos",
        "privileged_observation",
        "privileged_observations",
        "privileged_observation_state",
        "privileged_map",
        "privileged_maps",
        "hidden_info",
        "hidden_infos",
        "hidden_state",
        "hidden_states",
        "hidden_observation",
        "hidden_observations",
        "hidden_map",
        "hidden_maps",
        "critic_info",
        "critic_infos",
        "critic_observation",
        "critic_observations",
        "critic_map",
        "critic_maps",
        "hidden_hazard",
        "hidden_hazards",
        "hidden_hazard_map",
        "hidden_hazard_maps",
        "hidden_hazard_grid",
        "hidden_hazard_grids",
        "global_hazard",
        "global_hazards",
        "global_hazard_map",
        "global_hazard_maps",
        "global_hazard_grid",
        "global_hazard_grids",
        "privileged_hazard",
        "privileged_hazards",
        "privileged_hazard_map",
        "privileged_hazard_maps",
        "privileged_hazard_grid",
        "privileged_hazard_grids",
        # Snake_case central_/critic_ hazard forms (S2-22): the reveal
        # concatenated-alias set already forbade the collapsed spellings of
        # these, but the actor forbidden-key set (and, through it, the Stage
        # 23-A recursive scan) previously missed them, leaving a nested-key
        # CTDE leak. Added here to close the gap; the set size is now 70.
        "central_hazard",
        "central_hazards",
        "central_hazard_map",
        "central_hazard_maps",
        "central_hazard_grid",
        "central_hazard_grids",
        "critic_hazard",
        "critic_hazards",
        "critic_hazard_map",
        "critic_hazard_maps",
        "critic_hazard_grid",
        "critic_hazard_grids",
    }
)
_BASE_ACTOR_INFORMATION_KEYS = frozenset(
    {"actor_observation", "revealed_information", "history_state"}
)
_REVEAL_FIELD_PREFIX = "revealed_local_"
_REVEAL_FIELD_SUFFIX_PATTERN = re.compile(r"^[a-z0-9]+(?:_[a-z0-9]+)*$")
_FORBIDDEN_REVEAL_FIELD_TOKENS = frozenset(
    {"central", "critic", "global", "hidden", "privileged"}
)
_FORBIDDEN_REVEAL_FIELD_CONCATENATED_ALIASES = frozenset(
    {
        "centralhazard",
        "centralhazardgrid",
        "centralhazardgrids",
        "centralhazardmap",
        "centralhazardmaps",
        "centralhazards",
        "centralmap",
        "centralmaps",
        "centralstate",
        "centralstates",
        "centralinfo",
        "centralinfos",
        "centralobservation",
        "centralobservations",
        "critichazard",
        "critichazards",
        "criticmap",
        "criticmaps",
        "criticstate",
        "criticstates",
        "criticinfo",
        "criticinfos",
        "criticobservation",
        "criticobservations",
        "globalhazard",
        "globalhazardgrid",
        "globalhazardgrids",
        "globalhazardmap",
        "globalhazardmaps",
        "globalhazards",
        "globalmap",
        "globalmaps",
        "globalstate",
        "globalstates",
        "globalinfo",
        "globalinfos",
        "globalobservation",
        "globalobservations",
        "hiddenhazard",
        "hiddenhazardgrid",
        "hiddenhazardgrids",
        "hiddenhazardmap",
        "hiddenhazardmaps",
        "hiddenhazards",
        "hiddenmap",
        "hiddenmaps",
        "hiddenstate",
        "hiddenstates",
        "hiddeninfo",
        "hiddeninfos",
        "hiddenobservation",
        "hiddenobservations",
        "privilegedhazard",
        "privilegedhazardgrid",
        "privilegedhazardgrids",
        "privilegedhazardmap",
        "privilegedhazardmaps",
        "privilegedhazards",
        "privilegedmap",
        "privilegedmaps",
        "privilegedstate",
        "privilegedstates",
        "privilegedinfo",
        "privilegedinfos",
        "privilegedobservation",
        "privilegedobservations",
    }
)


def forbidden_actor_information_keys() -> frozenset[str]:
    """Return field names that may not enter the decentralized actor path.

    The set contains 70 keys (was 56 before S2-22): every
    ``central_* / global_* / privileged_* / hidden_* / critic_*`` combination
    with ``state / info(s) / observation(s) / map(s)`` plus the
    ``central_/critic_/global_/hidden_/privileged_``-hazard ``map(s)/grid(s)``
    spellings. The Stage 23-A recursive forbidden-key scan unions this set with
    its stage-specific additions, so extending it here also hardens that scan.
    """

    return _FORBIDDEN_ACTOR_INFORMATION_KEYS


@dataclass(frozen=True)
class MAPPOCoreConfig:
    """Minimal model dimensions for a factorized recurrent MAPPO-Lagrangian core.

    A zero ``revealed_information_dim`` is the documented empty-reveal
    representation for callers that have no explicit local reveal tensor content.
    """

    actor_observation_dim: int
    revealed_information_dim: int
    central_state_dim: int
    history_state_dim: int
    actor_hidden_dim: int
    critic_hidden_dim: int
    sensing_action_count: int
    movement_action_count: int
    recurrent_layer_count: int
    agent_id_count: int | None = None
    use_recurrent_actor: bool = True
    dtype: str = "float32"
    device: str = "cpu"

    def __post_init__(self) -> None:
        require_positive_int("actor_observation_dim", self.actor_observation_dim)
        require_nonnegative_int(
            "revealed_information_dim", self.revealed_information_dim
        )
        require_positive_int("central_state_dim", self.central_state_dim)
        require_positive_int("history_state_dim", self.history_state_dim)
        require_positive_int("actor_hidden_dim", self.actor_hidden_dim)
        require_positive_int("critic_hidden_dim", self.critic_hidden_dim)
        require_positive_int("recurrent_layer_count", self.recurrent_layer_count)
        require_minimum_int("sensing_action_count", self.sensing_action_count, 2)
        require_minimum_int("movement_action_count", self.movement_action_count, 2)
        if self.agent_id_count is not None:
            require_positive_int("agent_id_count", self.agent_id_count)
        # S-12 (pin lifted): the historical Stage 21 tests pinned ValueError for
        # wrong types on the three fields below; those tests are archived, so the
        # project TypeError-for-wrong-type / ValueError-for-bad-value convention
        # now applies. Wrong type -> TypeError; disallowed value -> ValueError.
        if not isinstance(self.use_recurrent_actor, bool):
            raise TypeError("use_recurrent_actor must be a bool")
        if not self.use_recurrent_actor:
            raise ValueError("Stage 21 core requires a recurrent actor")
        if not isinstance(self.dtype, str):
            raise TypeError("dtype must be a string")
        if self.dtype not in ("float32", "float64"):
            raise ValueError("dtype must be 'float32' or 'float64'")
        # NEW-config-3: one explicit device whitespace rule. A device string
        # must be non-empty and contain no whitespace of any kind (leading,
        # trailing, or internal). ``"".join(device.split())`` strips every
        # whitespace character, so an inequality flags any whitespace in a
        # single check (subsuming the prior strip + per-character isspace pair).
        # Junk-but-whitespace-free device strings are deliberately NOT rejected
        # here; their validity is left to torch.device at construction time.
        # Wrong type -> TypeError; empty/whitespace value -> ValueError (S-12).
        if not isinstance(self.device, str):
            raise TypeError("device must be a string")
        if not self.device.strip():
            raise ValueError("device must be a non-empty string")
        if self.device != "".join(self.device.split()):
            raise ValueError("device must not contain whitespace")

    @property
    def actor_input_dim(self) -> int:
        return self.actor_observation_dim + self.revealed_information_dim

    @property
    def factorized_action_count(self) -> tuple[int, int]:
        return self.sensing_action_count, self.movement_action_count


ModelConfig = MAPPOCoreConfig


@dataclass(frozen=True, kw_only=True)
class AlgorithmConfig:
    """Validated PPO/GAE scalar settings for future optimization code.

    ``kw_only=True`` forces every field to be passed by keyword, so the
    identically-typed float coefficients cannot be silently swapped by position
    (NEW-config-4). The only in-package constructor already uses keywords.
    """

    discount_factor: float
    gae_lambda: float
    ppo_clip_range: float

    def __post_init__(self) -> None:
        # S2-23 (DECIDED 2026-07-04, DR-S2-23): validate AND normalize. The
        # sanitized base floats returned by the validation helpers are stored
        # via ``object.__setattr__`` (the lagrange.py S2-6 pattern) so a
        # hostile int/float subclass cannot reach downstream tensor arithmetic
        # (e.g. the un-laundered LossConfig coefficient sinks in update.py)
        # through overridden operators. Equality, hashing, and asdict
        # serialization are unchanged for genuine numbers, so behavior is
        # value-identical for genuine float inputs.
        object.__setattr__(
            self, "discount_factor", require_probability("discount_factor", self.discount_factor)
        )
        object.__setattr__(
            self, "gae_lambda", require_probability("gae_lambda", self.gae_lambda)
        )
        object.__setattr__(
            self,
            "ppo_clip_range",
            require_open_unit_interval("ppo_clip_range", self.ppo_clip_range),
        )


@dataclass(frozen=True, kw_only=True)
class LossConfig:
    """Validated nonnegative loss coefficients used by Stage 21 loss helpers.

    ``hazard_cost_coefficient`` is retained for general/later direct-cost-scaling
    compatibility. In bounded Stage 22 development training, hazard penalty
    strength is controlled by the Lagrange multiplier, not by
    ``hazard_cost_coefficient``.

    ``reward_entropy_coefficient`` is the combined-entropy coefficient: the
    Stage 22 update applies it to the sum of the sensing and movement entropy
    bonuses, on top of the per-factor ``sensing_entropy_coefficient`` and
    ``movement_entropy_coefficient`` terms (issue m-6 resolution; the bounded
    development default keeps it at 0.0 so numerics are unchanged).

    ``kw_only=True`` forces all six identically-typed float coefficients to be
    passed by keyword so they cannot be silently reordered by position
    (NEW-config-4); ``reward_entropy_coefficient`` being first no longer makes a
    positional swap easy.
    """

    reward_entropy_coefficient: float
    movement_entropy_coefficient: float
    sensing_entropy_coefficient: float
    hazard_cost_coefficient: float
    sensing_cost_coefficient: float
    value_loss_coefficient: float

    def __post_init__(self) -> None:
        # S2-23 (DECIDED 2026-07-04, DR-S2-23): validate AND normalize —
        # sanitized base floats are stored via object.__setattr__; see
        # AlgorithmConfig.__post_init__ for the full rationale. These six
        # coefficients are the DR's primary target: five of them are consumed
        # un-laundered in tensor arithmetic inside update.py.
        for name in (
            "reward_entropy_coefficient",
            "movement_entropy_coefficient",
            "sensing_entropy_coefficient",
            "hazard_cost_coefficient",
            "sensing_cost_coefficient",
            "value_loss_coefficient",
        ):
            object.__setattr__(self, name, require_nonnegative_number(name, getattr(self, name)))


@dataclass(frozen=True, kw_only=True)
class LagrangeConfig:
    """Validated scalar settings for nonnegative multiplier updates.

    ``kw_only=True`` forces keyword construction so the identically-typed float
    fields cannot be positionally swapped (NEW-config-4).
    """

    initial_multiplier: float
    learning_rate: float
    hazard_budget: float

    def __post_init__(self) -> None:
        # S2-23 (DECIDED 2026-07-04, DR-S2-23): validate AND normalize —
        # sanitized base floats are stored via object.__setattr__; see
        # AlgorithmConfig.__post_init__ for the full rationale.
        object.__setattr__(
            self,
            "initial_multiplier",
            require_nonnegative_number("initial_multiplier", self.initial_multiplier),
        )
        object.__setattr__(
            self, "learning_rate", require_positive_number("learning_rate", self.learning_rate)
        )
        object.__setattr__(
            self, "hazard_budget", require_nonnegative_number("hazard_budget", self.hazard_budget)
        )


def validate_actor_input_mapping(mapping: Mapping[str, object]) -> None:
    """Reject keys that would leak centralized or unsupported information."""

    if not isinstance(mapping, Mapping):
        raise TypeError("actor input must be a mapping")
    string_keys = [key for key in mapping if isinstance(key, str)]
    non_string_keys = [key for key in mapping if not isinstance(key, str)]
    forbidden = sorted(key for key in string_keys if key in _FORBIDDEN_ACTOR_INFORMATION_KEYS)
    if forbidden:
        raise ValueError(
            "actor input contains forbidden centralized or unrevealed fields: "
            + ", ".join(forbidden)
        )
    unsupported_strings = [
        key
        for key in string_keys
        if (
            key not in _BASE_ACTOR_INFORMATION_KEYS
            and not key.startswith(_REVEAL_FIELD_PREFIX)
        )
    ]
    # NEW-config-1: one globally-sorted offender list so a mix of non-string
    # keys (reported via repr) and unsupported string keys is ordered
    # consistently, rather than two independently-sorted sublists concatenated.
    unsupported = sorted(
        [repr(key) for key in non_string_keys] + unsupported_strings
    )
    if unsupported:
        raise ValueError(
            "actor input contains unsupported non-local-reveal fields: "
            + ", ".join(unsupported)
        )
    reveal_keys = sorted(
        key
        for key in string_keys
        if key.startswith(_REVEAL_FIELD_PREFIX)
    )
    malformed_reveals = [
        key for key in reveal_keys if _is_malformed_reveal_field_key(key)
    ]
    if malformed_reveals:
        raise ValueError(
            "actor input contains malformed revealed_local field names: "
            + ", ".join(malformed_reveals)
        )
    forbidden_provenance_reveals = [
        key for key in reveal_keys if _has_forbidden_reveal_field_provenance(key)
    ]
    if forbidden_provenance_reveals:
        raise ValueError(
            "actor input contains reveal fields with forbidden provenance: "
            + ", ".join(forbidden_provenance_reveals)
        )


def _is_malformed_reveal_field_key(key: str) -> bool:
    suffix = key[len(_REVEAL_FIELD_PREFIX) :]
    return not suffix or _REVEAL_FIELD_SUFFIX_PATTERN.fullmatch(suffix) is None


def _has_forbidden_reveal_field_provenance(key: str) -> bool:
    suffix = key[len(_REVEAL_FIELD_PREFIX) :]
    suffix_tokens = frozenset(suffix.split("_"))
    collapsed = suffix.replace("_", "")
    return (
        bool(suffix_tokens & _FORBIDDEN_REVEAL_FIELD_TOKENS)
        or collapsed in _FORBIDDEN_REVEAL_FIELD_TOKENS
        or _contains_forbidden_concatenated_alias(collapsed)
    )


def _contains_forbidden_concatenated_alias(collapsed_suffix: str) -> bool:
    # Substring containment on the underscore-collapsed suffix subsumes the
    # older whole-suffix equality, exact-token, and whole-suffix-prefix rules,
    # and additionally closes the alias-with-trailing-text-inside-a-token
    # (issue m-7), alias-with-leading-text, and alias-split-across-underscores
    # evasions found in the 2026-07-03 session review. Bare provenance TOKENS
    # (central/critic/global/hidden/privileged) deliberately stay exact-match
    # so benign lookalikes such as "critical_hazard" and "hazard_criticality"
    # remain accepted (their acceptance is pinned by Stage 21 tests).
    return any(
        alias in collapsed_suffix
        for alias in _FORBIDDEN_REVEAL_FIELD_CONCATENATED_ALIASES
    )


def validate_actor_tensors(
    actor_observation: torch.Tensor,
    revealed_information: torch.Tensor,
    history_state: torch.Tensor | None,
    config: MAPPOCoreConfig,
) -> tuple[int, int]:
    """Validate actor-visible tensors and return ``(batch, time)``.

    ``actor_observation`` is the local-observation tensor. Explicitly revealed
    information must be supplied through the separate ``revealed_information``
    tensor, never through named central-state or hidden-hazard fields.
    """

    if not isinstance(config, MAPPOCoreConfig):
        raise TypeError("config must be a MAPPOCoreConfig")
    torch = torch_required("tensor validation")
    expected_dtype = torch_dtype_from_name(config.dtype)
    require_torch_tensor("actor_observation", actor_observation)
    require_torch_tensor("revealed_information", revealed_information)
    if actor_observation.ndim != 3:
        raise ValueError("actor_observation must have shape [batch, time, dim]")
    if revealed_information.ndim != 3:
        raise ValueError("revealed_information must have shape [batch, time, dim]")
    batch, time, obs_dim = actor_observation.shape
    rev_batch, rev_time, rev_dim = revealed_information.shape
    if batch <= 0 or time <= 0:
        raise ValueError("actor_observation batch and time dimensions must be positive")
    if obs_dim != config.actor_observation_dim:
        raise ValueError("actor_observation last dimension does not match config")
    if rev_dim != config.revealed_information_dim:
        raise ValueError("revealed_information last dimension does not match config")
    if (rev_batch, rev_time) != (batch, time):
        raise ValueError("actor_observation and revealed_information shape mismatch")
    # NEW-config-2: the floating-point type guard (TypeError for non-float) runs
    # first, then the dtype-vs-config equality (so a wrong *float* dtype such as
    # float64/float16/bfloat16 under a float32 config reports the dtype mismatch),
    # then the finiteness scan. This means a NaN inside a wrong-dtype tensor is
    # reported as a dtype mismatch rather than a finiteness failure.
    if not torch.is_floating_point(actor_observation):
        raise TypeError("actor_observation must use a floating point dtype")
    if not torch.is_floating_point(revealed_information):
        raise TypeError("revealed_information must use a floating point dtype")
    if actor_observation.dtype != expected_dtype:
        raise ValueError("actor_observation dtype does not match config dtype")
    if revealed_information.dtype != expected_dtype:
        raise ValueError("revealed_information dtype does not match config dtype")
    require_finite_float_tensor("actor_observation", actor_observation)
    require_finite_float_tensor("revealed_information", revealed_information)
    if revealed_information.device != actor_observation.device:
        raise ValueError("actor_observation and revealed_information device mismatch")
    if history_state is not None:
        require_torch_tensor("history_state", history_state)
        expected = (
            config.recurrent_layer_count,
            batch,
            config.history_state_dim,
        )
        if history_state.shape != expected:
            raise ValueError(
                "history_state must have shape "
                "[recurrent_layer_count, batch, history_state_dim]"
            )
        if not torch.is_floating_point(history_state):
            raise TypeError("history_state must use a floating point dtype")
        if history_state.dtype != expected_dtype:
            raise ValueError("history_state dtype does not match config dtype")
        require_finite_float_tensor("history_state", history_state)
        if history_state.device != actor_observation.device:
            raise ValueError("history_state device must match actor_observation device")
    return batch, time


def validate_central_state(
    central_state: torch.Tensor,
    config: MAPPOCoreConfig,
) -> tuple[int, int]:
    """Validate centralized critic state and return ``(batch, time)``."""

    if not isinstance(config, MAPPOCoreConfig):
        raise TypeError("config must be a MAPPOCoreConfig")
    torch = torch_required("tensor validation")
    require_torch_tensor("central_state", central_state)
    if central_state.ndim != 3:
        raise ValueError("central_state must have shape [batch, time, dim]")
    batch, time, dim = central_state.shape
    if batch <= 0 or time <= 0:
        raise ValueError("central_state batch and time dimensions must be positive")
    if dim != config.central_state_dim:
        raise ValueError("central_state last dimension does not match config")
    # NEW-config-2: dtype-vs-config equality before finiteness (see the
    # matching comment in validate_actor_tensors).
    if not torch.is_floating_point(central_state):
        raise TypeError("central_state must use a floating point dtype")
    if central_state.dtype != torch_dtype_from_name(config.dtype):
        raise ValueError("central_state dtype does not match config dtype")
    require_finite_float_tensor("central_state", central_state)
    return batch, time


def torch_required(context: str) -> ModuleType:
    """Import and return torch lazily, raising a context-labeled error if absent.

    Shared by every import-safe module in the package (config, losses, buffer,
    env_contracts) so the "PyTorch is required for <context>" wording and the
    ``exc.name == "torch"`` contract stay single-sourced (issue D-7 partial
    resolution, 2026-07-03). It lives here rather than in ``_validation``
    because a Stage 22 governance test requires ``_validation`` to stay free
    of any tensor-library reference, and the package file set is pinned so no
    new shared module may be added.
    """

    try:
        import torch
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise ModuleNotFoundError(
                f"PyTorch is required for {context}",
                name="torch",
            ) from exc
        raise
    return torch


def torch_dtype_from_name(name: str) -> torch.dtype:
    """Resolve a config dtype string (``float32``/``float64``) to the torch dtype.

    Single-sourced dtype-string resolution shared by the config tensor
    validators, the Stage 21 model, the Stage 22 development adapter, and the
    Stage 23-A tensor adapter (session 2026-07-04 dedup; see CLAUDE.md issue
    D-7).

    NEW-config-5: the ``ValueError`` branch below is unreachable ONLY when the
    ``name`` came from an already-validated ``MAPPOCoreConfig.dtype`` (which is
    constrained to ``float32``/``float64`` in ``__post_init__``). Direct callers
    that pass an arbitrary/unknown dtype name DO reach it and receive the
    deterministic error.
    """

    torch = torch_required("dtype resolution")
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError("unsupported floating point dtype")


def require_torch_tensor(name: str, value: object) -> None:
    """Raise ``TypeError`` unless ``value`` is a ``torch.Tensor``."""

    torch = torch_required("tensor validation")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")


def require_finite_float_tensor(name: str, value: object) -> None:
    """Validate a floating-point, non-meta, all-finite tensor.

    Meta-device tensors are rejected explicitly because ``isfinite(...).item()``
    on a meta tensor would otherwise escape as an undeclared ``RuntimeError``.
    """

    torch = torch_required("finite tensor validation")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must use a floating point dtype")
    if value.device.type == "meta":
        raise ValueError(f"{name} must not be a meta tensor")
    # Sparse layouts are rejected explicitly because isfinite on a sparse
    # tensor would otherwise escape as an undeclared NotImplementedError
    # (session 2026-07-04; same class of gap as the S-5 meta rejection).
    if value.layout != torch.strided:
        raise ValueError(f"{name} must use a strided tensor layout")
    if not torch.isfinite(value).all().item():
        raise ValueError(f"{name} must contain only finite values")


def require_finite_float_tensor_matching_config(
    name: str,
    value: object,
    config: MAPPOCoreConfig,
) -> object:
    """Validate a finite float tensor whose dtype and device match ``config``.

    Runs :func:`require_finite_float_tensor` (float dtype, non-meta, strided,
    all-finite) and then asserts the tensor's ``dtype`` equals the config dtype
    and its ``device`` equals ``torch.device(config.device)``. Returns the same
    tensor so call sites may bind the validated value. Shared torch-facing home
    (torch stays lazy) so payload boundary validators reuse one dtype/device
    coherence check rather than re-inlining it (issue D-7 / NEW-env_contracts-1).
    """

    if not isinstance(config, MAPPOCoreConfig):
        raise TypeError("config must be a MAPPOCoreConfig")
    torch = torch_required("finite tensor validation")
    require_finite_float_tensor(name, value)
    if value.dtype != torch_dtype_from_name(config.dtype):
        raise ValueError(f"{name} dtype does not match config dtype")
    if value.device != torch.device(config.device):
        raise ValueError(f"{name} device does not match config device")
    return value

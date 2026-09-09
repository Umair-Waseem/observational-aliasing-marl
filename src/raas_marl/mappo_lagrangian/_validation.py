"""Private dependency-light validation helpers for MAPPO-Lagrangian internals.

This module is the single shared home for scalar, integer, and seed
validation used across the MAPPO-Lagrangian and active-sensing packages
(session 2026-07-03 dedup of the previously copy-pasted ``_require_*``
helper families; see CLAUDE.md issue D-7).

It must stay free of third-party imports: a Stage 22 governance test pins it
as the single definition site of ``finite_numeric_scalar`` and requires that
no tensor-library reference appears anywhere in this file. Shared
tensor-facing validation helpers therefore live in ``config`` instead.

Message-compatibility note: several stages pin exact error-message wording in
their governance tests. Helpers that historically diverged across modules
("must be positive" vs "must be a positive integer"; stage-specific seed-0
wording) take an explicit override parameter so each call site preserves its
pinned message while sharing one implementation.
"""

from __future__ import annotations

import collections.abc
import math


def finite_numeric_scalar(name: str, value: object) -> float:
    """Return ``value`` as a finite float after strict scalar validation.

    Only genuine Python ``int``/``float`` values are accepted; ``bool`` is
    rejected with ``TypeError``. Conversion goes through the base-class
    ``__float__`` slots so hostile subclass overrides can neither escape the
    TypeError/ValueError contract nor substitute a different value. Huge ints
    that overflow ``float`` are normalized to ``ValueError`` (documented
    project convention).
    """

    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be a numeric int or float")
    try:
        if isinstance(value, float):
            numeric = float.__float__(value)
        else:
            numeric = int.__float__(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{name} must be finite") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite")
    return numeric


def is_numeric_scalar(value: object) -> bool:
    """Return whether ``value`` is a genuine ``int``/``float`` scalar (bool excluded).

    Shared predicate for callers that branch on scalar-ness or need to
    preserve a pinned pre-check message before ``finite_numeric_scalar``
    (session 2026-07-04 dedup; see CLAUDE.md issue D-7).
    """

    return isinstance(value, (int, float)) and not isinstance(value, bool)


def require_int_not_bool(
    name: str,
    value: object,
    *,
    error_suffix: str = "must be an int",
) -> None:
    """Raise ``TypeError`` unless ``value`` is a strict ``int`` (bool rejected).

    ``error_suffix`` preserves historical per-module wording (for example
    ``"must be an integer"``) where governance tests pin it.
    """

    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} {error_suffix}")


def require_positive_int(
    name: str,
    value: object,
    *,
    error_suffix: str = "must be a positive integer",
) -> None:
    """Validate a strict positive int (``TypeError`` for type, ``ValueError`` for value).

    ``error_suffix`` preserves historical per-module wording (for example
    ``"must be positive"``) and is forwarded to the internal type check so the
    ``TypeError`` (wrong type) and ``ValueError`` (bad value) messages share one
    coherent suffix taxonomy (NEW-_validation.py-1).
    """

    require_int_not_bool(name, value, error_suffix=error_suffix)
    # int.__index__ normalizes through the base-class slot so hostile int
    # subclasses cannot lie through overridden comparison operators (session
    # 2026-07-04; S-7 hardening extended to the int-bound helpers).
    if int.__index__(value) <= 0:
        raise ValueError(f"{name} {error_suffix}")


def require_nonnegative_int(
    name: str,
    value: object,
    *,
    error_suffix: str = "must be a nonnegative integer",
) -> None:
    """Validate a strict nonnegative int (``TypeError`` for type, ``ValueError`` for value).

    ``error_suffix`` is forwarded to the internal type check so the type and
    value messages share one coherent suffix (NEW-_validation.py-1).
    """

    require_int_not_bool(name, value, error_suffix=error_suffix)
    if int.__index__(value) < 0:
        raise ValueError(f"{name} {error_suffix}")


def require_minimum_int(name: str, value: object, minimum: int) -> None:
    """Validate a strict int with an inclusive lower bound ``minimum``.

    The ``minimum`` bound is itself validated as a strict int (NEW-_validation.py-2)
    so a non-int / bool bound fails deterministically rather than through a
    later opaque comparison error.
    """

    require_int_not_bool("minimum", minimum)
    require_int_not_bool(name, value)
    if int.__index__(value) < int.__index__(minimum):
        raise ValueError(f"{name} must be an integer >= {minimum}")


def require_probability(
    name: str,
    value: object,
    *,
    error_suffix: str = "must be in [0, 1]",
) -> float:
    """Return ``value`` as a float after validating it lies in [0, 1].

    ``error_suffix`` (NEW-_validation.py-3) lets a second call site override the
    value-range message for wording parity while sharing this implementation;
    the default preserves the existing wording.
    """

    numeric = finite_numeric_scalar(name, value)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} {error_suffix}")
    return numeric


def require_open_unit_interval(
    name: str,
    value: object,
    *,
    error_suffix: str = "must be in (0, 1)",
) -> float:
    """Return ``value`` as a float after validating it lies in (0, 1).

    ``error_suffix`` (NEW-_validation.py-3) overrides the value-range message
    for wording parity; the default preserves the existing wording.
    """

    numeric = finite_numeric_scalar(name, value)
    if not 0.0 < numeric < 1.0:
        raise ValueError(f"{name} {error_suffix}")
    return numeric


def require_nonnegative_number(
    name: str,
    value: object,
    *,
    error_suffix: str = "must be nonnegative",
) -> float:
    """Return ``value`` as a float after validating it is finite and >= 0.

    ``error_suffix`` (NEW-_validation.py-3) overrides the value message for
    wording parity; the default preserves the existing wording.
    """

    numeric = finite_numeric_scalar(name, value)
    if numeric < 0.0:
        raise ValueError(f"{name} {error_suffix}")
    return numeric


def require_positive_number(
    name: str,
    value: object,
    *,
    error_suffix: str = "must be positive",
) -> float:
    """Return ``value`` as a float after validating it is finite and > 0."""

    numeric = finite_numeric_scalar(name, value)
    if numeric <= 0.0:
        raise ValueError(f"{name} {error_suffix}")
    return numeric


def require_positive_seed(
    name: str,
    value: object,
    *,
    zero_seed_message: str | None = None,
) -> None:
    """Validate a strict positive seed; seed 0 is intentionally reserved/rejected.

    ``zero_seed_message`` preserves the stage-specific seed-0 wording pinned
    by each stage's governance tests.
    """

    require_int_not_bool(name, value)
    normalized = int.__index__(value)
    if normalized == 0:
        raise ValueError(
            zero_seed_message
            or f"{name} must be a positive integer; seed 0 is intentionally reserved/rejected"
        )
    if normalized < 0:
        raise ValueError(f"{name} must be a positive integer")


# Largest value the tensor library's manual-seed entry point accepts without an
# internal overflow: a signed 64-bit integer. This module stays free of any
# tensor-library import or reference (a design property of this file), so the
# bound is expressed as a plain integer constant here and the rollout collectors
# that derive per-step reseeds validate their upper bound against it via
# ``require_bounded_seed``.
_MAX_SIGNED_64_BIT_INT = 2**63 - 1


def require_bounded_seed(
    name: str,
    value: object,
    *,
    max_offset: int = 0,
    zero_seed_message: str | None = None,
    error_suffix: str | None = None,
) -> int:
    """Validate a strict positive seed with signed-64-bit headroom; return it as an int.

    Reuses the existing positive-seed semantics (``require_positive_seed``: seed
    0 reserved/rejected, negatives rejected, bool/non-int ``TypeError``), then
    additionally rejects a seed whose largest derived reseed
    (``value + int(max_offset)``) would exceed ``2**63 - 1`` and crash the tensor
    library's manual-seed entry point with an undeclared ``RuntimeError`` (S2-19 /
    NEW-rollout-1b). ``max_offset`` is the maximum per-collection reseed offset
    the caller will add (for example ``max_episodes * 1000 + max_steps``); the
    default ``0`` bounds the seed itself.

    Returns the seed normalized through the base ``int.__index__`` slot so
    hostile int subclasses cannot substitute a different value downstream.
    """

    require_positive_seed(name, value, zero_seed_message=zero_seed_message)
    normalized = int.__index__(value)
    require_nonnegative_int("max_offset", max_offset)
    offset = int.__index__(max_offset)
    if normalized + offset > _MAX_SIGNED_64_BIT_INT:
        raise ValueError(
            f"{name} {error_suffix}"
            if error_suffix is not None
            else f"{name} plus reseed offset must fit in a signed 64-bit integer"
        )
    return normalized


def require_mapping(name: str, value: object) -> collections.abc.Mapping:
    """Return ``value`` unchanged after validating it is a ``Mapping``.

    Uses ``collections.abc.Mapping`` so any mapping type (not just ``dict``) is
    accepted; non-mappings raise ``TypeError`` following the project's
    type-error-for-wrong-type convention.
    """

    if not isinstance(value, collections.abc.Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def require_string_mapping_keys(name: str, mapping: collections.abc.Mapping) -> None:
    """Raise ``TypeError`` if any key of ``mapping`` is not a ``str``.

    Offending keys are reported in a deterministic sorted-by-``repr`` order so
    the error message is stable across runs.
    """

    non_string = [key for key in mapping if not isinstance(key, str)]
    if non_string:
        offenders = ", ".join(sorted(repr(key) for key in non_string))
        raise TypeError(f"{name} keys must be strings; offending keys: {offenders}")


def nonempty_unpadded_string(name: str, value: object) -> str:
    """Return ``value`` after validating it is a non-empty, unpadded ``str``.

    ``TypeError`` for non-strings; ``ValueError`` for an empty string or one
    carrying leading/trailing whitespace (padded names are rejected so
    provenance and identity strings stay canonical).
    """

    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if value == "":
        raise ValueError(f"{name} must not be empty")
    if value != value.strip():
        raise ValueError(f"{name} must not have leading or trailing whitespace")
    return value

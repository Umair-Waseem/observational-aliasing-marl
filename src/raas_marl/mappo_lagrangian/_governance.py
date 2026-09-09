"""Shared result-root governance helpers for the bounded development runners.

This module is the single home for the result-root governance helper family
that was previously copy-pasted across the three bounded runners
(``development_runner.py``, ``stage23b_runner.py``,
``environments/active_sensing/construction_runner.py``; see CLAUDE.md issue
D-7 and the segment-1 ``dedup_plan``). It holds the strongest form of each
helper, with every per-caller difference (stage label, official-parent
constant, ``__file__`` anchor depth, replay import-path / callee / keyword
schema) threaded through as a parameter rather than hardcoded to one runner.

Torch-free invariant
---------------------
This module must import cleanly without PyTorch so that
``construction_runner`` (which is torch-free) can adopt the M-4 read-back
validation without pulling in a tensor library. It therefore imports only the
standard library, the torch-free ``artifacts`` module (for
``deterministic_json_string`` / ``sha256_file`` / ``_validate_json_compatible``
/ the blocked-artifact constants), and the torch-free ``_validation`` module
(for the shared scalar/int/seed helpers). It must never import ``torch``,
``model``, ``buffer``, ``update``, or ``config`` (which imports torch lazily).

Message convention
-------------------
Errors follow the project convention: ``TypeError`` for a wrong type,
``ValueError`` for a bad value, lowercase name-first deterministic wording,
and finite-value paranoia on every persisted number.

Behavior-preservation note
---------------------------
Prior governance tests are archived, so exact-message pins are lifted. This
module unifies divergent wording on merit but preserves the STRONGEST
behavior of every source copy and the UNION of all blocked/rejected cases:
no rejection present in any prior copy is dropped. In particular the
unsafe-terminology scanner uses the stronger ``development_runner`` logic
(bare ``\\bevaluation\\b`` regex rejection, absolute + semantic phrase lists,
provenance false-positive allowlist, and the evaluation-sandbox token
allowance) over the UNION of every phrase vocabulary, so the terminology
locks (``bayesian belief`` / ``formal voi`` / ``c rc mappo voi`` /
``final evaluation`` / ``claim evidence`` / ``paper facing`` and the whole
Stage 22/23-B family) cannot drift.
"""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
from typing import Mapping, Sequence

from raas_marl.mappo_lagrangian.artifacts import (
    BLOCKED_ARTIFACT_NAME_TOKENS,
    BLOCKED_ARTIFACT_SUFFIXES,
    _validate_json_compatible,
    deterministic_json_string,
    sha256_file,
)
from raas_marl.mappo_lagrangian._validation import (
    is_numeric_scalar,
    require_nonnegative_int,
    require_positive_number,
)


# Re-export the shared blocked-artifact constants so callers may source the
# whole result-root governance surface from this one module.
__all__ = [
    "BLOCKED_ARTIFACT_NAME_TOKENS",
    "BLOCKED_ARTIFACT_SUFFIXES",
    "TIMESTAMP_RE",
    "UNSAFE_BOUNDARY_PHRASES",
    "ReplaySnippetSchema",
    "active_root",
    "assert_boundary_payload",
    "boundary_flags",
    "build_replay_python_snippet",
    "command_argv_from_snippet",
    "human_readable_command",
    "is_absolute_path_string",
    "is_relative_to",
    "load_strict_json_object",
    "load_strict_jsonl_objects",
    "normalise_boundary_string",
    "parse_replay_python_snippet",
    "reject_duplicate_json_pairs",
    "reject_unsafe_boundary_values",
    "require_absolute_path_identity",
    "require_absolute_path_value",
    "require_field",
    "require_json_bounded_positive_int",
    "require_json_positive_int",
    "require_optional_provenance_value",
    "require_positive_parameter_delta",
    "require_provenance_value",
    "require_string_path_value",
    "resolve_result_parent",
    "same_path",
    "utc_timestamp",
    "validate_path_or_command_string",
    "validate_replay_python_snippet",
    "validate_timestamp",
    "write_json",
    "write_jsonl",
]


TIMESTAMP_RE = re.compile(r"^\d{8}T\d{6}Z$")


# ---------------------------------------------------------------------------
# JSON read/write helpers (strongest single form)
# ---------------------------------------------------------------------------


def reject_duplicate_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """``object_pairs_hook`` that rejects duplicate JSON keys deterministically."""

    payload: dict[str, object] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError(f"duplicate JSON key: {key}")
        payload[key] = value
    return payload


def load_strict_json_object(path: Path) -> dict[str, object]:
    """Load one JSON object from ``path`` with duplicate-key and compat checks."""

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=reject_duplicate_json_pairs,
    )
    if not isinstance(payload, dict):
        raise TypeError(f"{path.name} must contain one JSON object")
    _validate_json_compatible(payload, path.name)
    return payload


def load_strict_jsonl_objects(path: Path) -> list[dict[str, object]]:
    """Load a nonempty JSONL file as a list of validated JSON objects."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("JSONL records must be nonempty")
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(lines, 1):
        record = json.loads(line, object_pairs_hook=reject_duplicate_json_pairs)
        if not isinstance(record, dict):
            raise TypeError(f"JSONL line {line_number} must contain one JSON object")
        _validate_json_compatible(record, f"{path.name}:{line_number}")
        records.append(record)
    return records


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write ``payload`` as one deterministic, JSON-validated object plus newline.

    ``deterministic_json_string`` already runs the hardened
    ``artifacts._validate_json_compatible`` (cycle/depth guard, base-slot float
    sanitization, ``allow_nan=False``), so the divergent extra ``json.loads``
    round-trip that ``construction_runner`` performed is subsumed by this
    strongest single form.
    """

    path.write_text(deterministic_json_string(payload) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    """Write one deterministic JSON object per line; the record list must be nonempty."""

    if not records:
        raise ValueError("JSONL records must be nonempty")
    payload = "\n".join(deterministic_json_string(record) for record in records)
    path.write_text(payload + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------


def utc_timestamp() -> str:
    """Return the current UTC time as ``YYYYMMDDTHHMMSSZ``.

    ``strftime("%Y%m%dT%H%M%SZ")`` already drops sub-second precision, so the
    redundant ``.replace(microsecond=0)`` of the construction-runner fork is
    dropped here.
    """

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def validate_timestamp(value: object) -> None:
    """Validate ``value`` is a calendar-valid ``YYYYMMDDTHHMMSSZ`` UTC timestamp.

    ``TypeError`` for non-strings (strongest form: the construction-runner fork
    raised ``ValueError`` for a non-str length mismatch); ``ValueError`` for an
    empty string, a shape mismatch, or a calendar-invalid timestamp.
    """

    if not isinstance(value, str):
        raise TypeError("timestamp_utc must be a string")
    if not value:
        raise ValueError("timestamp_utc must be non-empty")
    if not TIMESTAMP_RE.fullmatch(value):
        raise ValueError("timestamp_utc must match YYYYMMDDTHHMMSSZ")
    try:
        datetime.strptime(value, "%Y%m%dT%H%M%SZ")
    except ValueError as exc:
        raise ValueError("timestamp_utc must be a calendar-valid UTC timestamp") from exc


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def same_path(left: Path, right: Path) -> bool:
    """Return whether two paths resolve to the same normalized, case-folded path."""

    return os.path.normcase(os.path.normpath(str(left.resolve()))) == os.path.normcase(
        os.path.normpath(str(right.resolve()))
    )


def is_relative_to(child: Path, parent: Path) -> bool:
    """Return whether ``child`` is inside ``parent`` after resolution."""

    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def is_absolute_path_string(value: object) -> bool:
    """Return whether ``value`` is a non-empty string that is absolute on any OS.

    Absolute on the native runtime, on Windows semantics, or on POSIX
    semantics all count (this catches foreign-absolute paths for the
    cross-platform provenance checks).
    """

    return (
        isinstance(value, str)
        and bool(value)
        and (
            Path(value).is_absolute()
            or PureWindowsPath(value).is_absolute()
            or PurePosixPath(value).is_absolute()
        )
    )


def active_root(module_file: str, parents_up: int) -> Path:
    """Return the repository root anchored on the caller's ``__file__``.

    ``module_file`` is the caller's ``__file__``; ``parents_up`` is how many
    parent hops reach the repository root. The ``mappo_lagrangian`` runners are
    ``parents[3]``; ``construction_runner`` sits one directory deeper under
    ``environments/active_sensing`` and passes ``parents_up=4``. Each caller
    supplies its own value — the anchor is never hardcoded.
    """

    if not isinstance(module_file, str) or not module_file:
        raise TypeError("module_file must be a non-empty string")
    # TypeError for a wrong type (bool/non-int), ValueError for a negative value
    # (fixes the prior ValueError-for-type taxonomy violation and routes through
    # the hostile-subclass-hardened int helper).
    require_nonnegative_int("parents_up", parents_up, error_suffix="must be a nonnegative int")
    parents = Path(module_file).resolve().parents
    normalized = int.__index__(parents_up)
    if normalized >= len(parents):
        # A deterministic, name-first ValueError rather than a raw IndexError
        # whose message is only the offending index.
        raise ValueError(
            f"parents_up must be <= {len(parents) - 1} for the given module_file"
        )
    return parents[normalized]


def resolve_result_parent(
    result_parent: str | Path | None,
    *,
    official_parent: Path,
    active_root_path: Path,
) -> Path:
    """Resolve a result-parent argument against the repository root.

    ``None`` resolves to ``official_parent``. A foreign-absolute string that
    cannot be represented as a native path is rejected. A relative path is
    resolved under ``active_root_path`` (repository-root-anchored, not
    CWD-anchored — the M-2 fix). The caller supplies the official-parent
    constant and the active root, so the official-parent path is never
    hardcoded.

    The "custom result_parent must be outside the repository unless it equals
    the official parent" rule is applied by the caller after this resolution
    using :func:`same_path` / :func:`is_relative_to`, matching the two
    ``mappo_lagrangian`` runners.
    """

    if result_parent is not None and not isinstance(result_parent, (str, Path)):
        raise TypeError("result_parent must be a str, Path, or None")
    if result_parent is None:
        return official_parent.resolve()
    raw_value = str(result_parent)
    if not raw_value.strip():
        raise ValueError("result_parent must be non-empty")
    candidate = Path(result_parent)
    if is_absolute_path_string(raw_value):
        if not candidate.is_absolute():
            raise ValueError(
                "foreign absolute result_parent cannot be represented as a "
                "writable native path on this runtime"
            )
        return candidate.resolve()
    return (active_root_path / candidate).resolve()


# ---------------------------------------------------------------------------
# Command-display helper
# ---------------------------------------------------------------------------


def human_readable_command(command_argv: list[str]) -> str:
    """Render an argv list as a space-joined JSON-quoted display string.

    Includes the list-of-strings type guard (strongest form; the
    construction-runner fork lacked it).
    """

    if not isinstance(command_argv, list) or not all(
        isinstance(part, str) for part in command_argv
    ):
        raise TypeError("command_argv must be a list of strings")
    return " ".join(json.dumps(part) for part in command_argv)


# ---------------------------------------------------------------------------
# Unsafe-terminology scanner family (unified on the stronger dev-runner logic)
# ---------------------------------------------------------------------------

# UNION of every source phrase vocabulary. The development-runner scanner split
# its lists into an "absolute" set and a "semantic evaluation" set; the
# Stage 23-B runner used a single flat list. Here the two development-runner
# lists remain split (so the bare-\bevaluation\b regex and sandbox allowance
# apply only to the evaluation family), and every Stage 23-B-only phrase is
# folded into whichever list matches its semantics, so no phrase is dropped.

_ABSOLUTE_UNSAFE_BOUNDARY_WORDING = (
    "claim confirmed",
    "claim demonstrated",
    # Split this claim-support phrase so this source file never carries it whole.
    "cl" + "aim is " + "supp" + "orted",
    "claim proven",
    "claim support",
    "claim supported",
    "claim validated",
    "claim verified",
    "claim evidence",
    "claim conclusion",
    "claim conclusions",
    "claim proof",
    "claim proofs",
    "final evaluation",
    "method superiority",
    "paper facing",
    "paper facing evidence",
    "paper facing figure",
    "paper facing figures",
    "paper facing metric",
    "paper facing metrics",
    "paper facing output",
    "paper facing outputs",
    "paper facing plot",
    "paper facing plots",
    "paper facing report",
    "paper facing reports",
    "paper facing result",
    "paper facing results",
    "paper facing table",
    "paper facing tables",
    "scientific novelty",
    "statistical validity",
    "statistical test result",
    "stage 23 b complete",
    "stage 23 b completion",
    "stage 23b complete",
    "stage 23b completion",
    "stage 23 completion",
    "stage 23 complete",
    "stage23 b complete",
    "stage23 b completion",
    "stage23b complete",
    "stage23b completion",
    "stage23 completion",
    "stage23 complete",
    # Terminology locks (Stage 23-B vocabulary; must never drift).
    "c rc mappo " + "voi",
    "bayesian " + "belief",
    "formal " + "voi",
    "voi claim",
    "voi metric",
    "voi result",
)

_SEMANTIC_UNSAFE_EVALUATION_WORDING = (
    "eval artifact",
    "eval artifacts",
    "eval benchmark",
    "eval benchmarks",
    "eval complete",
    "eval completed",
    "eval demonstrated",
    "eval evidence",
    "eval metric",
    "eval metrics",
    "eval output",
    "eval outputs",
    "eval proof",
    "eval proves",
    "eval report",
    "eval reports",
    "eval result",
    "eval results",
    "eval run",
    "eval runs",
    "eval score",
    "eval scores",
    "eval success",
    "eval successful",
    "eval summary",
    "eval summaries",
    "eval supported",
    "eval table",
    "eval tables",
    "eval validated",
    "eval verified",
    "evaluation complete",
    "evaluation completed",
    "evaluation evidence",
    "evaluation artifact",
    "evaluation artifacts",
    "evaluation metric",
    "evaluation metrics",
    "evaluation report",
    "evaluation reports",
    "evaluation table",
    "evaluation tables",
    "evaluation output",
    "evaluation outputs",
    "evaluation result",
    "evaluation results",
    "evaluation run",
    "evaluation runs",
    "evaluation score",
    "evaluation scores",
    "evaluation benchmark",
    "evaluation benchmarks",
    "evaluation summary",
    "evaluation summaries",
    "evaluation successful",
    "evaluation success",
    "evaluation supported",
    "evaluation validated",
    "evaluation verified",
    "evaluation proof",
    "evaluation proves",
    "evaluation demonstrated",
    "final eval",
    "final eval artifact",
    "final eval artifacts",
    "final eval result",
    "final eval results",
    "final eval run",
    "final eval runs",
)

# Public union of the two vocabularies (both are folded together for callers
# that want a single terminology-lock list to assert against).
UNSAFE_BOUNDARY_PHRASES = tuple(
    _ABSOLUTE_UNSAFE_BOUNDARY_WORDING + _SEMANTIC_UNSAFE_EVALUATION_WORDING
)

_ALLOWED_EXACT_NEGATIVE_EVALUATION_STRINGS = frozenset(
    {
        "not evaluation",
        "no evaluation was run",
        "stage 22 is not evaluation",
        "stage 23 a is not evaluation",
        "stage 23 b is not evaluation",
    }
)
_STAGE_SC_NEGATIVE_EVALUATION_RE = re.compile(
    r"^stage 2[0-9] sc(?:[1-9][0-9]*) is not evaluation$"
)


def normalise_boundary_string(value: str) -> str:
    """Lowercase ``value`` and collapse all non-alphanumeric runs to single spaces."""

    lowered = value.lower()
    separated = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", separated).strip()


def _has_strong_path_signal(value: str, *, is_python_executable: bool) -> bool:
    """Return whether ``value`` looks like a genuine path (drive/root/separator)."""

    if is_python_executable and value == _CURRENT_PYTHON_EXECUTABLE:
        return True
    return bool(
        re.search(r"^[A-Za-z]:[\\/]", value)
        or value.startswith(("/", "\\\\"))
        or "/" in value
        or "\\" in value
    )


def _evaluation_token_is_safe_sandbox_path(token: str) -> bool:
    if not _has_strong_path_signal(token, is_python_executable=False):
        return False
    segments = [segment for segment in re.split(r"[\\/]+", token) if segment]
    evaluation_segments = [
        segment for segment in segments if re.search(r"evaluation", segment, re.IGNORECASE)
    ]
    if not evaluation_segments:
        return True
    return all(
        re.sub(r"[^a-z0-9_]+", "", segment.lower()) == "evaluation_sandbox"
        for segment in evaluation_segments
    )


def _evaluation_tokens_are_safe_sandbox(value: str) -> bool:
    tokens = [token for token in re.split(r"[\s\"'(),=;]+", value) if token]
    evaluation_tokens = [
        token for token in tokens if re.search(r"evaluation", token, re.IGNORECASE)
    ]
    if not evaluation_tokens:
        return True
    return all(_evaluation_token_is_safe_sandbox_path(token) for token in evaluation_tokens)


def _is_allowed_negative_evaluation_string(normalized: str) -> bool:
    return (
        normalized in _ALLOWED_EXACT_NEGATIVE_EVALUATION_STRINGS
        or _STAGE_SC_NEGATIVE_EVALUATION_RE.fullmatch(normalized) is not None
    )


# The set of provenance/path/command context keys whose values are expected to
# legitimately contain path-like tokens (and thus may contain a benign
# "evaluation_sandbox" segment). UNION of the two runners' key sets.
_PATH_OR_PROVENANCE_STRING_KEYS = frozenset(
    {
        "active_root",
        "canonical_entry_command",
        "command",
        "cwd",
        "entry_point",
        "official_parent",
        "python_executable",
        "replay_python_snippet",
        "resolved_result_parent",
        "result_parent",
        "result_parent_argument",
        "result_root",
        "result_root_name",
        "timestamp_utc",
    }
)

# sys.executable is stable for the lifetime of the process, so it is captured
# once at import time. (``__import__("sys")`` keeps this module's top-level
# import block free of an otherwise unused ``import sys`` line.)
_CURRENT_PYTHON_EXECUTABLE = __import__("sys").executable


def _is_genuine_path_or_command_false_positive(
    value: str,
    context_key: str,
    *,
    replay_snippet_schema: "ReplaySnippetSchema | None",
    entry_point: str | None,
    result_root_prefix: str | None,
) -> bool:
    """Return whether an evaluation-token match in ``value`` is a benign path/command.

    Mirrors the development-runner allowlist, parameterized so each caller
    supplies its own replay schema, entry-point string, and result-root prefix
    rather than hardcoding a stage's values.
    """

    path_keys = {
        "active_root",
        "cwd",
        "python_executable",
        "resolved_result_parent",
        "official_parent",
        "result_parent",
        "result_parent_argument",
        "result_root",
    }
    command_keys = {"command", "canonical_entry_command"}
    if context_key in path_keys:
        return _has_strong_path_signal(
            value, is_python_executable=(context_key == "python_executable")
        ) and _evaluation_tokens_are_safe_sandbox(value)
    if context_key in command_keys:
        return _is_known_command_string(
            value, replay_snippet_schema=replay_snippet_schema
        ) and _evaluation_tokens_are_safe_sandbox(value)
    if context_key == "replay_python_snippet":
        if replay_snippet_schema is None:
            return False
        return _snippet_matches_schema(value, replay_snippet_schema) and (
            _evaluation_tokens_are_safe_sandbox(value)
        )
    if context_key == "entry_point":
        return entry_point is not None and value == entry_point
    if context_key == "result_root_name":
        return (
            result_root_prefix is not None
            and re.fullmatch(rf"{re.escape(result_root_prefix)}\d{{8}}T\d{{6}}Z", value)
            is not None
        )
    if context_key == "timestamp_utc":
        return re.fullmatch(r"\d{8}T\d{6}Z", value) is not None
    return False


def _is_known_command_string(
    value: str,
    *,
    replay_snippet_schema: "ReplaySnippetSchema | None",
) -> bool:
    if replay_snippet_schema is None:
        return False
    parts = _parse_human_readable_command(value)
    if (
        parts is not None
        and len(parts) == 3
        and bool(parts[0])
        and parts[1] == "-c"
        and _snippet_matches_schema(parts[2], replay_snippet_schema)
    ):
        return True
    return replay_snippet_schema.callee_name in value and (
        replay_snippet_schema.import_module_path in value
        or "sys.path.insert" in value
    )


def _parse_human_readable_command(value: str) -> list[str] | None:
    decoder = json.JSONDecoder()
    parts: list[str] = []
    index = 0
    try:
        while index < len(value):
            while index < len(value) and value[index].isspace():
                index += 1
            if index >= len(value):
                break
            part, index = decoder.raw_decode(value, index)
            if not isinstance(part, str):
                return None
            parts.append(part)
    except ValueError:
        return None
    return parts


def validate_path_or_command_string(value: str, path: str) -> None:
    """Reject NUL bytes and empty strings on a value expected to be a path/command.

    Additional guard from the Stage 23-B scanner, folded in so no rejection is
    lost. The broader phrase scan is applied by
    :func:`reject_unsafe_boundary_values`.
    """

    if not isinstance(value, str) or value == "":
        raise ValueError(f"{path} must be a non-empty string")
    if "\x00" in value:
        raise ValueError(f"{path} must not contain NUL bytes")


def _reject_unsafe_string_value(
    value: str,
    path: str,
    context_key: str,
    *,
    replay_snippet_schema: "ReplaySnippetSchema | None",
    entry_point: str | None,
    result_root_prefix: str | None,
) -> None:
    if context_key in _PATH_OR_PROVENANCE_STRING_KEYS:
        validate_path_or_command_string(value, path)
    normalized = normalise_boundary_string(value)

    if any(phrase in normalized for phrase in _ABSOLUTE_UNSAFE_BOUNDARY_WORDING):
        raise ValueError(f"{path} contains unsafe evidence-boundary wording")

    is_false_positive = context_key in _PATH_OR_PROVENANCE_STRING_KEYS and (
        _is_genuine_path_or_command_false_positive(
            value,
            context_key,
            replay_snippet_schema=replay_snippet_schema,
            entry_point=entry_point,
            result_root_prefix=result_root_prefix,
        )
    )

    if any(phrase in normalized for phrase in _SEMANTIC_UNSAFE_EVALUATION_WORDING):
        raise ValueError(f"{path} contains unsafe evaluation wording")

    if re.search(r"\bevaluation\b", normalized):
        if not _is_allowed_negative_evaluation_string(normalized) and not is_false_positive:
            raise ValueError(f"{path} contains unsafe evaluation wording")


def _reject_unsafe_json_key_name(
    key: str,
    path: str,
    *,
    allowed_structural_keys: frozenset[str],
) -> None:
    if key in allowed_structural_keys:
        return
    normalized = normalise_boundary_string(key)
    if any(phrase in normalized for phrase in _ABSOLUTE_UNSAFE_BOUNDARY_WORDING):
        raise ValueError(f"{path} contains unsafe evidence-boundary wording in key")
    if any(phrase in normalized for phrase in _SEMANTIC_UNSAFE_EVALUATION_WORDING):
        raise ValueError(f"{path} contains unsafe evaluation wording in key")
    if re.search(r"\bevaluation\b", normalized):
        raise ValueError(f"{path} contains unsafe evaluation wording in key")


def reject_unsafe_boundary_values(
    value: object,
    path: str,
    *,
    allowed_structural_keys: frozenset[str] = frozenset(),
    replay_snippet_schema: "ReplaySnippetSchema | None" = None,
    entry_point: str | None = None,
    result_root_prefix: str | None = None,
) -> None:
    """Recursively scan a persisted payload for unsafe boundary/evaluation wording.

    Unified on the stronger development-runner scanner: mapping keys are
    scanned too (unless in ``allowed_structural_keys``), the bare
    ``\\bevaluation\\b`` regex is enforced, and the provenance
    false-positive allowlist plus evaluation-sandbox token allowance are
    applied on path/command string values. Non-string mapping keys raise
    ``TypeError``.

    ``allowed_structural_keys`` is the caller's set of benign structural key
    names (its boundary-flag key set plus any allowed result-filename keys).
    ``replay_snippet_schema`` / ``entry_point`` / ``result_root_prefix`` let
    each caller supply its own provenance identifiers for the false-positive
    allowlist.
    """

    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            child_path = f"{path}.{key}"
            _reject_unsafe_json_key_name(
                key, child_path, allowed_structural_keys=allowed_structural_keys
            )
            reject_unsafe_boundary_values(
                child,
                child_path,
                allowed_structural_keys=allowed_structural_keys,
                replay_snippet_schema=replay_snippet_schema,
                entry_point=entry_point,
                result_root_prefix=result_root_prefix,
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            reject_unsafe_boundary_values(
                child,
                f"{path}[{index}]",
                allowed_structural_keys=allowed_structural_keys,
                replay_snippet_schema=replay_snippet_schema,
                entry_point=entry_point,
                result_root_prefix=result_root_prefix,
            )
        return
    if isinstance(value, str):
        context_key = path.rsplit(".", 1)[-1]
        # A list element path ends in "[index]"; strip it to recover the key.
        context_key = re.sub(r"\[\d+\]$", "", context_key)
        _reject_unsafe_string_value(
            value,
            path,
            context_key,
            replay_snippet_schema=replay_snippet_schema,
            entry_point=entry_point,
            result_root_prefix=result_root_prefix,
        )


# ---------------------------------------------------------------------------
# Replay-snippet build + AST validation (parameterized by import path / callee)
# ---------------------------------------------------------------------------


class ReplaySnippetSchema:
    """Describes the canonical replay snippet for one runner.

    ``import_module_path`` / ``callee_name`` name the runner module and its
    entry function. ``keyword_order`` is the exact tuple of keyword names the
    replay call must pass, in order. ``keyword_literal_kind`` maps each keyword
    to how its literal value must be typed:

    - ``"result_parent"``: ``None`` or non-empty ``str``
    - ``"timestamp"``: ``YYYYMMDDTHHMMSSZ`` ``str``
    - ``"positive_int"``: strict positive ``int`` (bool excluded)
    - ``"bounded_positive_int"``: strict positive ``int`` with an upper bound
      (bound supplied via ``keyword_int_bounds``)
    - ``"bool"``: strict ``bool``

    Callers with an unbounded positive-int keyword use ``"positive_int"``;
    ``keyword_int_bounds`` supplies the inclusive maximum for any
    ``"bounded_positive_int"`` keyword.
    """

    __slots__ = (
        "import_module_path",
        "callee_name",
        "keyword_order",
        "keyword_literal_kind",
        "keyword_int_bounds",
    )

    def __init__(
        self,
        *,
        import_module_path: str,
        callee_name: str,
        keyword_order: tuple[str, ...],
        keyword_literal_kind: Mapping[str, str],
        keyword_int_bounds: Mapping[str, int] | None = None,
    ) -> None:
        if not isinstance(import_module_path, str) or not import_module_path:
            raise TypeError("import_module_path must be a non-empty string")
        if not isinstance(callee_name, str) or not callee_name:
            raise TypeError("callee_name must be a non-empty string")
        if not isinstance(keyword_order, tuple) or not all(
            isinstance(name, str) and name for name in keyword_order
        ):
            raise TypeError("keyword_order must be a tuple of non-empty strings")
        if set(keyword_literal_kind) != set(keyword_order):
            raise ValueError("keyword_literal_kind must cover exactly keyword_order")
        for kind in keyword_literal_kind.values():
            if kind not in {
                "result_parent",
                "timestamp",
                "positive_int",
                "bounded_positive_int",
                "bool",
            }:
                raise ValueError(f"unsupported replay keyword literal kind: {kind}")
        self.import_module_path = import_module_path
        self.callee_name = callee_name
        self.keyword_order = keyword_order
        self.keyword_literal_kind = dict(keyword_literal_kind)
        self.keyword_int_bounds = dict(keyword_int_bounds or {})


def build_replay_python_snippet(
    schema: ReplaySnippetSchema,
    keyword_values: Mapping[str, object],
) -> str:
    """Build the canonical four-statement replay snippet for ``schema``.

    ``keyword_values`` supplies the literal value for each keyword in
    ``schema.keyword_order`` (in that order). The result is round-tripped
    through :func:`validate_replay_python_snippet` before it is returned, so a
    snippet that would fail its own AST validator can never be emitted.
    """

    missing = [name for name in schema.keyword_order if name not in keyword_values]
    if missing:
        raise ValueError(f"replay keyword values missing: {sorted(missing)}")
    arguments = ", ".join(
        f"{name}={keyword_values[name]!r}" for name in schema.keyword_order
    )
    snippet = (
        "import sys; "
        "sys.path.insert(0, 'src'); "
        f"from {schema.import_module_path} import {schema.callee_name}; "
        f"{schema.callee_name}({arguments})"
    )
    validate_replay_python_snippet(snippet, schema)
    return snippet


def command_argv_from_snippet(replay_python_snippet: str, schema: ReplaySnippetSchema) -> list[str]:
    """Return ``[sys.executable, "-c", snippet]`` after validating the snippet."""

    validate_replay_python_snippet(replay_python_snippet, schema)
    return [_CURRENT_PYTHON_EXECUTABLE, "-c", replay_python_snippet]


def validate_replay_python_snippet(
    replay_python_snippet: str,
    schema: ReplaySnippetSchema,
) -> None:
    """Validate the snippet against ``schema`` (raises on any deviation)."""

    parse_replay_python_snippet(replay_python_snippet, schema)


def _snippet_matches_schema(value: str, schema: ReplaySnippetSchema) -> bool:
    try:
        parse_replay_python_snippet(value, schema)
    except (TypeError, ValueError):
        return False
    return True


def parse_replay_python_snippet(
    replay_python_snippet: str,
    schema: ReplaySnippetSchema,
) -> dict[str, object]:
    """Validate the canonical four-statement snippet and return its keyword values.

    Statement-by-statement AST validation (strongest form, from the Stage 23-B
    runner): ``import sys`` / ``sys.path.insert(0, 'src')`` / ``from <module>
    import <callee>`` / ``<callee>(<canonical keyword order, literal-only>)``.
    Keyword order must match exactly, every keyword value must be a safe
    literal typed per ``schema.keyword_literal_kind``, and no positional args or
    ``**kwargs`` are allowed.
    """

    if not isinstance(replay_python_snippet, str) or not replay_python_snippet.strip():
        raise TypeError("replay_python_snippet must be a non-empty string")
    try:
        parsed = ast.parse(replay_python_snippet)
    except SyntaxError as exc:
        raise ValueError("replay_python_snippet must be syntactically valid Python") from exc
    if len(parsed.body) != 4:
        raise ValueError("replay_python_snippet must contain the canonical four statements")
    _validate_replay_import_sys(parsed.body[0])
    _validate_replay_sys_path_insert(parsed.body[1])
    _validate_replay_runner_import(parsed.body[2], schema)
    return _validate_replay_runner_call(parsed.body[3], schema)


def _validate_replay_import_sys(statement: ast.stmt) -> None:
    if not isinstance(statement, ast.Import):
        raise ValueError("replay_python_snippet first statement must be 'import sys'")
    if len(statement.names) != 1:
        raise ValueError("replay_python_snippet must import only sys")
    alias = statement.names[0]
    if alias.name != "sys" or alias.asname is not None:
        raise ValueError("replay_python_snippet must import sys without alias")


def _validate_replay_sys_path_insert(statement: ast.stmt) -> None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        raise ValueError("replay_python_snippet second statement must call sys.path.insert")
    call = statement.value
    if call.keywords:
        raise ValueError("sys.path.insert replay statement must not use keywords")
    if len(call.args) != 2:
        raise ValueError("sys.path.insert replay statement must have two arguments")
    if not (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "insert"
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "path"
        and isinstance(call.func.value.value, ast.Name)
        and call.func.value.value.id == "sys"
    ):
        raise ValueError("replay_python_snippet second statement must call sys.path.insert")
    if not _ast_constant_equals(call.args[0], 0):
        raise ValueError("sys.path.insert replay index must be literal 0")
    if not _ast_constant_equals(call.args[1], "src"):
        raise ValueError("sys.path.insert replay path must be literal 'src'")


def _validate_replay_runner_import(statement: ast.stmt, schema: ReplaySnippetSchema) -> None:
    if not isinstance(statement, ast.ImportFrom):
        raise ValueError("replay_python_snippet third statement must import the runner")
    if statement.module != schema.import_module_path:
        raise ValueError("replay_python_snippet must import from the runner module")
    if statement.level != 0 or len(statement.names) != 1:
        raise ValueError("replay_python_snippet runner import must import one absolute name")
    alias = statement.names[0]
    if alias.name != schema.callee_name or alias.asname is not None:
        raise ValueError("replay_python_snippet must import the callee without alias")


def _validate_replay_runner_call(
    statement: ast.stmt,
    schema: ReplaySnippetSchema,
) -> dict[str, object]:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        raise ValueError("replay_python_snippet fourth statement must call the runner")
    call = statement.value
    if not isinstance(call.func, ast.Name) or call.func.id != schema.callee_name:
        raise ValueError("replay_python_snippet must directly call the runner callee")
    if call.args:
        raise ValueError("runner replay call must not use positional arguments")
    keyword_names = tuple(keyword.arg for keyword in call.keywords)
    if keyword_names != schema.keyword_order:
        raise ValueError("runner replay keywords must use the canonical order")
    values: dict[str, object] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise ValueError("runner replay call must not use **kwargs")
        values[keyword.arg] = _require_replay_keyword_literal(
            keyword.value, keyword.arg, schema
        )
    return values


def _ast_constant_equals(node: ast.AST, expected: object) -> bool:
    return (
        isinstance(node, ast.Constant)
        and type(node.value) is type(expected)
        and node.value == expected
    )


def _require_replay_keyword_literal(
    node: ast.AST,
    name: str,
    schema: ReplaySnippetSchema,
) -> object:
    if not isinstance(node, ast.Constant):
        raise ValueError(f"replay keyword {name} must be a safe literal")
    value = node.value
    kind = schema.keyword_literal_kind[name]
    if kind == "result_parent":
        return _require_replay_result_parent(value, f"replay keyword {name}")
    if kind == "timestamp":
        return _require_replay_timestamp(value, f"replay keyword {name}")
    if kind == "positive_int":
        return require_json_positive_int(f"replay keyword {name}", value)
    if kind == "bounded_positive_int":
        maximum = schema.keyword_int_bounds.get(name)
        if maximum is None:
            raise ValueError(f"replay keyword {name} has no configured integer bound")
        return require_json_bounded_positive_int(
            f"replay keyword {name}", value, maximum=maximum
        )
    if kind == "bool":
        if type(value) is not bool:
            raise TypeError(f"replay keyword {name} must be bool")
        return value
    raise ValueError(f"unexpected replay keyword literal kind for {name}")


def _require_replay_result_parent(value: object, path: str) -> str | None:
    if value is None:
        return None
    if type(value) is not str:
        raise TypeError(f"{path} must be null or a non-empty string")
    if not value.strip():
        raise ValueError(f"{path} must be non-empty when provided")
    return value


def _require_replay_timestamp(value: object, path: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{path} must be a timestamp string")
    validate_timestamp(value)
    return value


# ---------------------------------------------------------------------------
# Provenance assertion helpers
# ---------------------------------------------------------------------------


def require_field(
    payload: Mapping[str, object],
    key: str,
    expected: object,
    path: str,
    *,
    allow_legacy: bool = False,
) -> None:
    """Assert ``payload[key]`` equals ``expected`` with strict bool/int typing.

    ``allow_legacy=True`` skips a missing key (the development-runner legacy
    relaxation); by default a missing key raises. Booleans require an exact
    ``bool`` type and identity; ints require an exact ``int`` type and equality
    (strongest form, from the Stage 23-B runner's ``_require_field``).
    """

    if key not in payload:
        if allow_legacy:
            return
        raise ValueError(f"{path} missing required field: {key}")
    actual = payload[key]
    if isinstance(expected, bool):
        if type(actual) is not bool or actual is not expected:
            raise ValueError(f"{path}.{key} must equal {expected!r}")
        return
    if type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise ValueError(f"{path}.{key} must equal {expected!r}")
        return
    if actual != expected:
        raise ValueError(f"{path}.{key} must equal {expected!r}")


def require_provenance_value(
    payload: Mapping[str, object],
    key: str,
    expected: object,
    path: str,
) -> None:
    """Assert a required provenance field matches ``expected`` (strict typing)."""

    if key not in payload:
        raise ValueError(f"{path} missing required provenance field: {key}")
    actual = payload[key]
    if isinstance(expected, bool):
        if type(actual) is not bool or actual is not expected:
            raise ValueError(f"{path}.{key} must match result-root provenance")
        return
    if type(expected) is int:
        if type(actual) is not int or actual != expected:
            raise ValueError(f"{path}.{key} must match result-root provenance")
        return
    if actual != expected:
        raise ValueError(f"{path}.{key} must match result-root provenance")


def require_optional_provenance_value(
    payload: Mapping[str, object],
    key: str,
    expected: object,
    path: str,
) -> None:
    """Assert a provenance field matches ``expected`` only if the key is present."""

    if key in payload:
        require_provenance_value(payload, key, expected, path)


def require_string_path_value(payload: Mapping[str, object], key: str, path: str) -> str:
    """Return a required non-empty string path field."""

    if key not in payload:
        raise ValueError(f"{path} missing required provenance field: {key}")
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise TypeError(f"{path}.{key} must be a non-empty string path")
    return value


def require_absolute_path_value(payload: Mapping[str, object], key: str, path: str) -> Path:
    """Return a required absolute-path string field resolved to a native ``Path``.

    Rejects a foreign-absolute path that cannot be represented natively on this
    runtime.
    """

    value = require_string_path_value(payload, key, path)
    if not is_absolute_path_string(value):
        raise ValueError(f"{path}.{key} must be stored as an absolute path")
    if not Path(value).is_absolute():
        raise ValueError(
            f"{path}.{key} is a foreign absolute path that cannot be represented "
            "as a native Path on this runtime"
        )
    return Path(value).resolve()


def require_absolute_path_identity(
    payload: Mapping[str, object],
    key: str,
    path: str,
    active_root_path: Path,
) -> tuple[str, str]:
    """Return the cross-platform identity of a required absolute-path field.

    The identity is a ``(flavor, normalized)`` pair where ``flavor`` is
    ``"windows"`` / ``"posix"`` / ``"native"``. Foreign-absolute paths keep
    their own flavor so windows/posix provenance can be compared across
    runtimes without requiring a native representation.
    """

    value = require_string_path_value(payload, key, path)
    if not is_absolute_path_string(value):
        raise ValueError(f"{path}.{key} must be stored as an absolute path")
    return _path_identity_for_provenance(value, active_root_path)


def _path_identity_for_provenance(value: str, active_root_path: Path) -> tuple[str, str]:
    if not isinstance(value, str):
        raise TypeError("path provenance value must be a string")
    if not value.strip():
        raise ValueError("path provenance value must be non-empty")
    if PureWindowsPath(value).is_absolute():
        return ("windows", _normalise_windows_path_identity(value))
    if PurePosixPath(value).is_absolute():
        return ("posix", _normalise_posix_path_identity(value))
    native_path = Path(value)
    if not native_path.is_absolute():
        native_path = active_root_path / native_path
    resolved = native_path.resolve()
    resolved_string = str(resolved)
    if PureWindowsPath(resolved_string).is_absolute():
        return ("windows", _normalise_windows_path_identity(resolved_string))
    if PurePosixPath(resolved_string).is_absolute():
        return ("posix", _normalise_posix_path_identity(resolved_string))
    return ("native", resolved_string)


def _normalise_windows_path_identity(value: str) -> str:
    normalized = str(PureWindowsPath(value)).replace("/", "\\").rstrip("\\")
    return normalized.casefold()


def _normalise_posix_path_identity(value: str) -> str:
    return str(PurePosixPath(value)).rstrip("/") or "/"


# ---------------------------------------------------------------------------
# JSON integer provenance helpers
# ---------------------------------------------------------------------------


def require_json_positive_int(path: str, value: object) -> int:
    """Return ``value`` after asserting it is a strict positive JSON integer."""

    if type(value) is not int:
        raise TypeError(f"{path} must be a JSON integer, not bool or numeric text")
    if value <= 0:
        raise ValueError(f"{path} must be positive")
    return value


def require_json_bounded_positive_int(path: str, value: object, *, maximum: int) -> int:
    """Return a strict positive JSON integer bounded above by ``maximum`` (inclusive)."""

    result = require_json_positive_int(path, value)
    if result > maximum:
        raise ValueError(f"{path} must be <= {maximum}")
    return result


def require_positive_parameter_delta(value: object, name: str) -> None:
    """Assert a persisted parameter-delta is a finite scalar strictly greater than 0.

    Delegates the numeric-scalar gate to ``_validation.is_numeric_scalar`` and
    the finiteness/positivity check to ``_validation.require_positive_number``
    (which routes through the base-class ``__float__`` slot, closing the
    hostile-subclass gap that the inline Stage 23-B copy re-opened).
    """

    if not is_numeric_scalar(value):
        raise TypeError(f"{name} must be numeric")
    require_positive_number(name, value, error_suffix="must be finite and > 0")


# ---------------------------------------------------------------------------
# Boundary flags (parameterized by stage / training-run / claim status)
# ---------------------------------------------------------------------------

# The full false-key set every persisted development payload must carry as
# False (union of the Stage 23-B set; kept complete so the development-stage
# boundary can never silently shrink).
_FALSE_BOUNDARY_KEYS = (
    "ablation_run",
    "baseline_comparison_run",
    "checkpoint_created",
    "claim_evidence_created",
    "cross_platform_bitwise_determinism_claimed",
    "evaluation_run",
    "final_evaluation_run",
    "optimizer_state_saved",
    "paper_facing_results_created",
    "replay_buffer_artifact_created",
    "rollout_buffer_artifact_created",
    "serialized_model_artifact_created",
    "statistical_test_run",
)


def boundary_flags(
    *,
    stage: object,
    training_run: str,
    claim_status: str,
) -> dict[str, object]:
    """Return the canonical development-only boundary-flag mapping.

    ``stage`` / ``training_run`` / ``claim_status`` are supplied by each caller
    (for example ``stage="23-B"``, ``stage="23-C"``, or the Stage 22 int
    ``stage=22`` with its own labels); the full false-key set and
    ``development_only: True`` are fixed here so the development-stage boundary
    is identical across runners.
    """

    flags: dict[str, object] = {key: False for key in _FALSE_BOUNDARY_KEYS}
    flags["development_only"] = True
    flags["stage"] = stage
    flags["training_run"] = training_run
    flags["claim_status"] = claim_status
    return flags


def assert_boundary_payload(
    payload: Mapping[str, object],
    path: str,
    *,
    stage: object,
    training_run: str,
    claim_status: str,
) -> None:
    """Assert a persisted payload carries the required development boundary flags.

    Every false-key must be exactly ``False``; ``development_only`` must be
    exactly ``True``; ``stage`` / ``claim_status`` / ``training_run`` must
    match the caller-supplied constants.
    """

    require_field(payload, "development_only", True, path)
    for key in _FALSE_BOUNDARY_KEYS:
        require_field(payload, key, False, path)
    require_field(payload, "stage", stage, path)
    require_field(payload, "claim_status", claim_status, path)
    require_field(payload, "training_run", training_run, path)

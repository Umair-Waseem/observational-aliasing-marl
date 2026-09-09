"""Stage 25 harness run-logging helpers (torch-free).

Run-id validation, run-directory resolution, incremental JSONL appending with
flush+fsync (curves must be readable while the process runs), scanned
deterministic JSON writing, and run-manifest payload construction for the
Stage 25 harness. Every persisted payload spreads the Stage 25 harness
development-only boundary flags, and every manifest write is routed through
the shared unsafe-terminology scanner before touching disk.

This module is development-only infrastructure. It does not execute the
project's locked final-assessment protocol, creates no evidence artifacts for
the research claim, writes no checkpoints, no model or optimizer state, and no
paper-oriented output. It imports only the standard library plus the shared
``_governance`` / ``_validation`` / ``artifacts`` modules (D-7 lesson: shared
helper bodies are imported, never copied) and stays torch-free.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

from raas_marl.mappo_lagrangian._governance import (
    active_root,
    boundary_flags,
    is_relative_to,
    reject_unsafe_boundary_values,
    resolve_result_parent,
    same_path,
    validate_timestamp,
    write_json,
)
from raas_marl.mappo_lagrangian._validation import (
    require_mapping,
    require_positive_int,
    require_positive_seed,
)
from raas_marl.mappo_lagrangian.artifacts import (
    BLOCKED_ARTIFACT_NAME_TOKENS,
    deterministic_json_string,
)

__all__ = [
    "append_jsonl_record",
    "build_run_manifest_payload",
    "experiments_root",
    "harness_allowed_structural_keys",
    "harness_boundary_flags",
    "resolve_run_directory",
    "scan_and_write_json",
    "validate_run_id",
    "write_run_manifest",
]


_HARNESS_STAGE = "25-harness"
_HARNESS_TRAINING_RUN = "stage25_harness_baseline_development_only"
_HARNESS_CLAIM_STATUS = "not tested / not supported"

_RUN_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,63}$"
_RUN_ID_RE = re.compile(_RUN_ID_PATTERN)

# Separator-collapsed forms of the blocked artifact name tokens
# ("final_eval" -> "finaleval"), checked as SUBSTRINGS of the
# separator-removed lowercase form of run ids and result-parent path
# segments so hyphen/underscore spellings ("check_point", "opti-mizer")
# and embedded forms ("finaleval1") cannot smuggle a blocked token.
_BLOCKED_TOKENS_COLLAPSED = tuple(
    re.sub(r"[^a-z0-9]+", "", token) for token in BLOCKED_ARTIFACT_NAME_TOKENS
)

_VALID_TIERS = ("T0", "T1", "T2")
_VALID_STATUSES = ("running", "complete", "killed", "crashed")

MANIFEST_FILENAME = "RUN_MANIFEST.json"
MANIFEST_VERSION = 1

# How many parent hops reach the repository root from this module file
# (src/raas_marl/final_training/run_logging.py -> parents[3] == repo root;
# verified at runtime by the harness smoke).
_ACTIVE_ROOT_PARENTS_UP = 3

# The Stage 24-A runtime-reproducibility record (embedded in the run manifest)
# carries its own negative-boundary key whose *name* would otherwise trip the
# unsafe-wording key scanner. It is a benign structural key of the same family
# as the boundary flags (its value is hard-coded False upstream), so it is
# allowed as a structural key alongside the harness boundary-flag key set.
_EXTRA_ALLOWED_STRUCTURAL_KEYS = frozenset(
    {"runtime_metadata_used_as_claim_evidence"}
)


def harness_boundary_flags() -> dict[str, object]:
    """Return the Stage 25 harness development-only boundary-flag mapping.

    Reuses the shared ``_governance.boundary_flags`` false-key set so the
    development-stage boundary is byte-identical to the governed runners; only
    the stage / training-run / claim-status labels are harness-specific.
    """

    return boundary_flags(
        stage=_HARNESS_STAGE,
        training_run=_HARNESS_TRAINING_RUN,
        claim_status=_HARNESS_CLAIM_STATUS,
    )


def validate_run_id(value: object) -> str:
    """Validate and return a Stage 25 run id.

    A run id is lowercase, starts with an alphanumeric character, continues
    with lowercase alphanumerics / underscores / hyphens, and is at most 64
    characters long. Each blocked artifact name token is rejected in its
    separator-collapsed form ("finaleval") as a substring of the
    separator-REMOVED form of the run id, so separator spellings
    ("check_point", "opti-mizer") and embedded forms ("finaleval1") are all
    rejected.
    """

    if not isinstance(value, str):
        raise TypeError("run_id must be a string")
    if not _RUN_ID_RE.fullmatch(value):
        raise ValueError(f"run_id must match {_RUN_ID_PATTERN}")
    collapsed = re.sub(r"[^a-z0-9]+", "", value)
    for token in _BLOCKED_TOKENS_COLLAPSED:
        if token in collapsed:
            raise ValueError("run_id must not contain blocked artifact name tokens")
    return value


def experiments_root() -> Path:
    """Return the official Stage 25 harness parent: <repo>/results/experiments."""

    return active_root(__file__, _ACTIVE_ROOT_PARENTS_UP) / "results" / "experiments"


def resolve_run_directory(
    run_id: str,
    result_parent: str | Path | None = None,
) -> Path:
    """Resolve the run directory for ``run_id`` under ``result_parent``.

    ``None`` uses the official experiments root. Relative parents resolve
    against the repository root (never the CWD — the M-2 lesson) via the
    shared ``_governance.resolve_result_parent``. The location rule the
    ``resolve_result_parent`` docstring delegates to the caller is enforced
    here: a resolved custom parent must be the official experiments root,
    be inside it, or be entirely outside the repository. Blocked artifact
    name tokens are rejected in every path segment of a custom parent
    against the separator-collapsed form (run-id parity). A run id is used
    exactly once: an already-existing run directory raises
    ``FileExistsError`` so a run can never be silently re-run into the same
    directory (I-4).
    """

    validate_run_id(run_id)
    official_parent = experiments_root()
    repository_root = active_root(__file__, _ACTIVE_ROOT_PARENTS_UP)
    parent = resolve_result_parent(
        result_parent,
        official_parent=official_parent,
        active_root_path=repository_root,
    )
    # Windows extended-length prefixes (\\?\ and \\?\UNC\) survive both
    # Path.resolve() and os.path.realpath() on this runtime and would defeat
    # the containment comparisons below; strip them explicitly, then
    # canonicalize the remainder.
    parent_text = os.fspath(parent)
    if parent_text.startswith("\\\\?\\UNC\\"):
        parent_text = "\\\\" + parent_text[len("\\\\?\\UNC\\") :]
    elif parent_text.startswith("\\\\?\\"):
        parent_text = parent_text[len("\\\\?\\") :]
    parent = Path(os.path.realpath(parent_text))
    if result_parent is not None:
        # The custom-parent location rule delegated by
        # _governance.resolve_result_parent: official root, inside the
        # official root, or entirely outside the repository.
        if (
            not same_path(parent, official_parent)
            and not is_relative_to(parent, official_parent)
            and is_relative_to(parent, repository_root)
        ):
            raise ValueError(
                "custom result_parent inside the repository must be under "
                "results/experiments"
            )
        # Blocked-token parity with validate_run_id: no path segment of a
        # custom parent may contain a blocked artifact name token in its
        # separator-collapsed lowercase form.
        for segment in parent.parts:
            collapsed_segment = re.sub(r"[^a-z0-9]+", "", segment.lower())
            if any(token in collapsed_segment for token in _BLOCKED_TOKENS_COLLAPSED):
                raise ValueError(
                    "result_parent must not contain blocked artifact name tokens"
                )
    run_directory = parent / run_id
    if run_directory.exists():
        raise FileExistsError("run directory already exists")
    return run_directory


def harness_allowed_structural_keys() -> frozenset[str]:
    """Return the allowed structural key names for the harness scanners.

    The harness boundary-flag key set plus the benign Stage 24-A runtime
    negative-boundary key (``_EXTRA_ALLOWED_STRUCTURAL_KEYS``). Shared by
    every harness scan call site — JSON writes, JSONL appends, and the
    driver's pre-directory input scan — so the allowed set never diverges.
    """

    return frozenset(harness_boundary_flags()) | _EXTRA_ALLOWED_STRUCTURAL_KEYS


def append_jsonl_record(path: Path, record: Mapping[str, object]) -> None:
    """Append one deterministic JSON record to ``path`` with flush + fsync.

    The incremental flush is the point: curve/episode/probe logs must be
    readable by an observer while the training process is still running.
    Every record is routed through the shared unsafe-terminology scanner
    BEFORE serialization (convention #12: every persisted payload is
    scanned; there is deliberately no opt-out) — a rejected record raises
    and nothing is written.
    """

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    require_mapping("record", record)
    reject_unsafe_boundary_values(
        record,
        str(path.name),
        allowed_structural_keys=harness_allowed_structural_keys(),
    )
    line = deterministic_json_string(record)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def scan_and_write_json(
    path: Path,
    payload: Mapping[str, object],
    *,
    context: str,
) -> None:
    """Scan ``payload`` for unsafe boundary wording, then write deterministic JSON."""

    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    require_mapping("payload", payload)
    if not isinstance(context, str) or not context:
        raise ValueError("context must be a non-empty string")
    reject_unsafe_boundary_values(
        payload,
        context,
        allowed_structural_keys=harness_allowed_structural_keys(),
    )
    write_json(path, payload)


def build_run_manifest_payload(
    *,
    run_id: str,
    tier: str,
    seeds: tuple[int, ...],
    implementing_dr_ids: tuple[str, ...],
    config_payload: Mapping[str, object],
    git_head: str | None,
    launch_timestamp_utc: str,
    pid: int,
    status: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, Any]:
    """Build the validated Stage 25 run-manifest payload.

    Tier ``T3`` is rejected here: T3 belongs to the Stage 26 protocol
    executor, not the Stage 25 harness. ``extra`` keys are merged last and
    must not collide with any manifest or boundary-flag key.
    """

    validate_run_id(run_id)
    if not isinstance(tier, str):
        raise TypeError("tier must be a string")
    if tier == "T3":
        raise ValueError("tier T3 is reserved for the Stage 26 protocol executor")
    if tier not in _VALID_TIERS:
        raise ValueError("tier must be one of T0, T1, T2")
    if not isinstance(seeds, tuple):
        raise TypeError("seeds must be a tuple")
    if not seeds:
        raise ValueError("seeds must be a non-empty tuple")
    for index, seed_value in enumerate(seeds):
        require_positive_seed(f"seeds[{index}]", seed_value)
    if not isinstance(implementing_dr_ids, tuple):
        raise TypeError("implementing_dr_ids must be a tuple")
    for index, dr_id in enumerate(implementing_dr_ids):
        if not isinstance(dr_id, str):
            raise TypeError(f"implementing_dr_ids[{index}] must be a string")
        if not dr_id:
            raise ValueError(f"implementing_dr_ids[{index}] must be non-empty")
    require_mapping("config_payload", config_payload)
    if git_head is not None and not isinstance(git_head, str):
        raise TypeError("git_head must be a string or None")
    validate_timestamp(launch_timestamp_utc)
    require_positive_int("pid", pid)
    if not isinstance(status, str):
        raise TypeError("status must be a string")
    if status not in _VALID_STATUSES:
        raise ValueError("status must be one of running, complete, killed, crashed")

    payload: dict[str, Any] = {
        **harness_boundary_flags(),
        "run_id": run_id,
        "tier": tier,
        "seeds": list(seeds),
        "implementing_dr_ids": list(implementing_dr_ids),
        "config": dict(config_payload),
        "git_head": git_head,
        "launch_timestamp_utc": launch_timestamp_utc,
        "pid": pid,
        "status": status,
        "manifest_version": MANIFEST_VERSION,
    }
    if extra is not None:
        extra_mapping = require_mapping("extra", extra)
        for key in extra_mapping:
            if not isinstance(key, str):
                raise TypeError("extra keys must be strings")
            if key in payload:
                raise ValueError(f"extra key must not collide with a manifest key: {key}")
        payload.update(extra_mapping)
    return payload


def write_run_manifest(run_dir: Path, payload: Mapping[str, object]) -> None:
    """Write (or overwrite) the scanned run manifest inside ``run_dir``."""

    if not isinstance(run_dir, Path):
        raise TypeError("run_dir must be a pathlib.Path")
    scan_and_write_json(
        run_dir / MANIFEST_FILENAME,
        payload,
        context=MANIFEST_FILENAME,
    )

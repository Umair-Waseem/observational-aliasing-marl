"""Deterministic artifact metadata helpers with no write-side behavior."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path


# Shared result-root governance constants (issue m-23 resolution): both
# bounded-training runners block the same binary/model suffixes and
# artifact-name tokens. "final_eval" is deliberately the broadest name token
# (it is a prefix of "final_evaluation"), so the shared set is at least as
# strict as either older per-runner set.
BLOCKED_ARTIFACT_SUFFIXES = frozenset(
    {
        ".pt",
        ".pth",
        ".ckpt",
        ".onnx",
        ".pkl",
        ".pickle",
        ".joblib",
        ".safetensors",
        ".bin",
        ".h5",
        ".keras",
    }
)
BLOCKED_ARTIFACT_NAME_TOKENS = ("checkpoint", "optimizer", "final_eval")


def _hash_and_size(file_path: Path) -> tuple[str, int]:
    """Stream ``file_path`` once, returning ``(sha256_hexdigest, size_bytes)``.

    A single open avoids the time-of-check/time-of-use gap between hashing and a
    separate ``stat()`` size read that ``manifest_entry`` previously had
    (issue NEW-artifacts-3): the byte count is accumulated from the same chunks
    fed to the digest, so both figures describe exactly the bytes hashed.
    """

    if not file_path.is_file():
        raise FileNotFoundError(str(file_path))
    digest = hashlib.sha256()
    size_bytes = 0
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size_bytes += len(chunk)
    return digest.hexdigest(), size_bytes


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 hex digest of ``path`` (streamed; missing file raises)."""

    return _hash_and_size(Path(path))[0]


def deterministic_json_string(payload: Mapping[str, object]) -> str:
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    # Serialize the validated, normalized snapshot rather than the caller's live
    # object, so the exact structure that passed validation is what json.dumps
    # emits (closes the validate-then-dump gap, issue NEW-artifacts-8). For the
    # plain dict/list payloads this project persists the emitted bytes are
    # identical to dumping the original object.
    normalized = _validate_json_compatible(payload, "payload")
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def deterministic_json_bytes(payload: Mapping[str, object]) -> bytes:
    """UTF-8 bytes of :func:`deterministic_json_string` (public artifact API)."""

    return deterministic_json_string(payload).encode("utf-8")


def manifest_entry(path: str | Path, role: str) -> dict[str, str | int]:
    """Build a deterministic manifest record ``{path, role, sha256, size_bytes}``.

    ``role`` must be a printable, whitespace-free token (identifier-like):
    leading, trailing, and interior whitespace and control characters are all
    rejected (issue NEW-artifacts-6) so a role cannot smuggle a newline/tab into
    a manifest. The file is read in a single streamed pass for both the digest
    and the byte count (see :func:`_hash_and_size`).

    ``manifest_entry`` and :func:`deterministic_json_bytes` are part of this
    module's public artifact-serialization surface (exported for callers that
    assemble manifests); they are covered by the test suite rather than an
    internal runner call site (issue NEW-artifacts-5).
    """

    file_path = Path(path)
    if not isinstance(role, str):
        raise TypeError("role must be a string")
    if not role:
        raise ValueError("role must be a non-empty string")
    if not role.isprintable() or any(character.isspace() for character in role):
        raise ValueError("role must be a printable token without whitespace")
    sha256, size_bytes = _hash_and_size(file_path)
    return {
        "path": file_path.as_posix(),
        "role": role,
        "sha256": sha256,
        "size_bytes": size_bytes,
    }


# Maximum container nesting depth accepted by ``_validate_json_compatible``.
# json.dumps defaults to a very high C-recursion limit; this project bounds it
# so a pathologically deep (or cyclic) payload fails with a deterministic
# ValueError here instead of a RecursionError deep inside json.dumps. 64 levels
# is far beyond any artifact this project persists (issue NEW-artifacts-1).
_MAX_JSON_NESTING_DEPTH = 64


def _validate_json_compatible(value: object, path: str) -> object:
    """Validate ``value`` is deterministic-JSON serializable; return a normalized copy.

    Hardening (issues NEW-artifacts-1/NEW-artifacts-2/NEW-artifacts-8):
    - Cyclic containers are detected via an ``id()`` visited set and rejected
      with a deterministic ValueError rather than a RecursionError.
    - Nesting deeper than ``_MAX_JSON_NESTING_DEPTH`` is rejected deterministically.
    - Floats (including hostile ``float`` subclasses that override ``__float__``)
      are sanitized through the base-class ``float.__float__`` slot before the
      finiteness check, so a substituted value cannot smuggle NaN/Infinity past
      ``json.dumps(allow_nan=False)``; the sanitized ``float`` is what is returned.
    - ``list``/``tuple`` become plain ``list`` and any ``collections.abc.Mapping``
      becomes a plain ``dict`` in the returned structure; each Mapping's
      ``.items()`` is snapshotted once during validation, and the returned copy is
      what callers serialize, so there is no time-of-check/time-of-use race
      between validation and ``json.dumps``.

    The public name and ``(value, path)`` call signature are unchanged; callers
    that only need validation may ignore the return value (the runner copies
    delegate here per D-7).
    """

    return _validate_json_compatible_inner(value, path, _seen=set(), _depth=0)


def _validate_json_compatible_inner(
    value: object,
    path: str,
    *,
    _seen: set[int],
    _depth: int,
) -> object:
    if value is None or isinstance(value, str):
        return value
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        # bool already handled above, so this is a genuine (non-bool) int.
        return value
    if isinstance(value, float):
        # Route through the base-class slot so a subclass override cannot
        # substitute a different (possibly non-finite) value after this check
        # (mirrors _validation.finite_numeric_scalar's S-7 hardening).
        try:
            numeric = float.__float__(value)
        except (OverflowError, ValueError) as exc:
            raise ValueError(
                f"{path} must contain only finite JSON numbers"
            ) from exc
        if not math.isfinite(numeric):
            raise ValueError(f"{path} must contain only finite JSON numbers")
        return numeric
    if isinstance(value, (list, tuple, Mapping)):
        if _depth >= _MAX_JSON_NESTING_DEPTH:
            raise ValueError(
                f"{path} exceeds maximum JSON nesting depth {_MAX_JSON_NESTING_DEPTH}"
            )
        identity = id(value)
        if identity in _seen:
            raise ValueError(f"{path} contains a circular reference")
        _seen.add(identity)
        try:
            if isinstance(value, Mapping):
                normalized_mapping: dict[str, object] = {}
                for key, item in tuple(value.items()):
                    if not isinstance(key, str):
                        raise TypeError(f"{path} mapping keys must be strings")
                    normalized_mapping[key] = _validate_json_compatible_inner(
                        item, f"{path}.{key}", _seen=_seen, _depth=_depth + 1
                    )
                return normalized_mapping
            normalized_sequence: list[object] = []
            for index, item in enumerate(value):
                normalized_sequence.append(
                    _validate_json_compatible_inner(
                        item, f"{path}[{index}]", _seen=_seen, _depth=_depth + 1
                    )
                )
            return normalized_sequence
        finally:
            # Remove on exit so sibling references to the same shared (acyclic)
            # container are not misreported as cycles; only ancestors on the
            # current path remain in ``_seen`` during descent.
            _seen.discard(identity)
    raise TypeError(f"{path} contains a non-JSON-compatible value")

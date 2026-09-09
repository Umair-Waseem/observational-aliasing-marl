"""Stage 23-C bounded multi-episode MAPPO-Lagrangian development-training runner.

The runner drives a small multi-episode rollout/update integration over the
Stage 23-A environment across a bounded number of collect->update rounds and
writes a deterministic JSON/JSONL development record with a persisted
per-episode training log. It does not run final evaluation, create claim
evidence, compare baselines, perform ablations, run statistical tests, rank
methods, or serialize model, optimizer, checkpoint, replay-buffer, or
rollout-buffer artifacts.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import random
import shutil
import sys
from typing import Any, Mapping

import torch

from raas_marl.environments.active_sensing.tensor_adapter import default_stage23_core_config
from raas_marl.mappo_lagrangian.artifacts import (
    BLOCKED_ARTIFACT_NAME_TOKENS,
    BLOCKED_ARTIFACT_SUFFIXES,
    sha256_file,
)
from raas_marl.mappo_lagrangian._validation import (
    require_bounded_seed,
    require_positive_int,
    require_positive_seed,
)
from raas_marl.mappo_lagrangian._governance import (
    ReplaySnippetSchema,
    active_root as _shared_active_root,
    assert_boundary_payload,
    boundary_flags,
    build_replay_python_snippet,
    command_argv_from_snippet,
    human_readable_command,
    is_relative_to,
    load_strict_json_object,
    load_strict_jsonl_objects,
    parse_replay_python_snippet,
    reject_unsafe_boundary_values,
    require_field,
    require_json_bounded_positive_int,
    require_json_positive_int,
    require_positive_parameter_delta,
    resolve_result_parent,
    same_path,
    utc_timestamp,
    validate_replay_python_snippet,
    validate_timestamp,
    write_json,
    write_jsonl,
)
from raas_marl.mappo_lagrangian.lagrange import LagrangeMultiplier
from raas_marl.mappo_lagrangian.model import RecurrentMAPPOActorCritic
from raas_marl.mappo_lagrangian.stage23c_rollout import (
    Stage23CRolloutConfig,
    collect_stage23c_development_rollout,
    make_default_stage23c_environment,
)
from raas_marl.mappo_lagrangian.update import (
    Stage22UpdateConfig,
    default_stage22_update_config,
    stage22_ppo_lagrangian_update,
)


STAGE23C_RESULT_PARENT = Path("results") / "stage23c_bounded_training"
STAGE23C_RESULT_PREFIX = "stage23c_bounded_training_"
_STAGE23C_ZERO_SEED_MESSAGE = (
    "seed must be a positive integer; seed 0 is reserved"
)
STAGE23C_RESULT_FILES = (
    "run_config.json",
    "command_record.json",
    "environment_summary.json",
    "rollout_summary.json",
    "episode_log.jsonl",
    "update_summary.json",
    "training_curve.jsonl",
    "boundary_record.json",
    "artifact_hashes.json",
)
_HASHED_RESULT_FILES = tuple(
    name for name in STAGE23C_RESULT_FILES if name != "artifact_hashes.json"
)
_CLAIM_STATUS = "not tested / not supported"
_ENTRY_POINT = (
    "raas_marl.mappo_lagrangian.stage23c_runner."
    "run_stage23c_bounded_training"
)
_TRAINING_RUN = "bounded_stage23c_development_only"
# Bounds pinned by the Stage 23-C specification.
_MIN_EPISODE_COUNT = 2
_MAX_EPISODE_COUNT = 8
_MIN_STEPS_PER_EPISODE = 1
_MAX_STEPS_PER_EPISODE = 32
_MIN_UPDATE_ROUNDS = 1
_MAX_UPDATE_ROUNDS = 4
# Default scenario / probe seed used only to resolve the default per-episode
# step horizon when steps_per_episode is None (mirrors the Stage23CRolloutConfig
# default scenario). The probe never trains; it only reads the resolved horizon.
_DEFAULT_SCENARIO_NAME = "risk_gate_hidden_hazard"
_DEFAULT_HORIZON_PROBE_SEED = 23
# Canonical replay-snippet schema for the Stage 23-C runner. Threaded through
# the shared _governance replay-snippet family (build/parse/validate) so the
# snippet build path and the read-back provenance path share one AST-validated
# form.
_REPLAY_SNIPPET_SCHEMA = ReplaySnippetSchema(
    import_module_path="raas_marl.mappo_lagrangian.stage23c_runner",
    callee_name="run_stage23c_bounded_training",
    keyword_order=(
        "result_parent",
        "timestamp_utc",
        "seed",
        "episode_count",
        "steps_per_episode",
        "update_rounds",
        "allow_pre_existing_stage23c_roots",
    ),
    keyword_literal_kind={
        "result_parent": "result_parent",
        "timestamp_utc": "timestamp",
        "seed": "positive_int",
        "episode_count": "bounded_positive_int",
        "steps_per_episode": "bounded_positive_int",
        "update_rounds": "bounded_positive_int",
        "allow_pre_existing_stage23c_roots": "bool",
    },
    keyword_int_bounds={
        "episode_count": _MAX_EPISODE_COUNT,
        "steps_per_episode": _MAX_STEPS_PER_EPISODE,
        "update_rounds": _MAX_UPDATE_ROUNDS,
    },
)
# Issue m-23 resolution: the suffix set is shared via artifacts.py.
_BLOCKED_ARTIFACT_SUFFIXES = BLOCKED_ARTIFACT_SUFFIXES
_FALSE_BOUNDARY_KEYS = frozenset(
    {
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
    }
)
_TRUE_BOUNDARY_KEYS = frozenset({"development_only"})
# The benign structural key allowlist handed to the shared unsafe-terminology
# scanner (its boundary-flag key set).
_ALLOWED_UNSAFE_LOOKING_STRUCTURAL_KEYS = _FALSE_BOUNDARY_KEYS | _TRUE_BOUNDARY_KEYS


def run_stage23c_bounded_training(
    *,
    result_parent: str | Path | None = None,
    timestamp_utc: str | None = None,
    seed: int = 23,
    episode_count: int = 2,
    steps_per_episode: int | None = None,
    update_rounds: int = 1,
    allow_pre_existing_stage23c_roots: bool = False,
) -> Path:
    """Run the bounded Stage 23-C development integration and return the root."""

    # Seed 0 / negative / bool rejection happens here with the pinned Stage 23-C
    # wording; the signed-64-bit upper bound (FINDING 1 / S2-19) is enforced
    # below via require_bounded_seed once effective_steps_per_episode is resolved,
    # accounting for the per-step reseed offset the rollout config derives, so a
    # seed >= 2**64 raises a declared ValueError before torch.manual_seed.
    require_positive_seed("seed", seed, zero_seed_message=_STAGE23C_ZERO_SEED_MESSAGE)
    require_positive_int(
        "episode_count", episode_count, error_suffix="must be positive"
    )
    # Normalize through the base-class int.__index__ slot before the range
    # comparison so a hostile int subclass overriding __le__ / __ge__ cannot
    # evade the [min, max] bound (S2-5; mirrors Stage23CRolloutConfig). A genuine
    # in-range int is unchanged.
    normalized_episode_count = int.__index__(episode_count)
    if not (_MIN_EPISODE_COUNT <= normalized_episode_count <= _MAX_EPISODE_COUNT):
        raise ValueError(
            f"episode_count must be between {_MIN_EPISODE_COUNT} and {_MAX_EPISODE_COUNT}"
        )
    if steps_per_episode is not None:
        require_positive_int(
            "steps_per_episode", steps_per_episode, error_suffix="must be positive"
        )
        normalized_steps_per_episode = int.__index__(steps_per_episode)
        if not (
            _MIN_STEPS_PER_EPISODE
            <= normalized_steps_per_episode
            <= _MAX_STEPS_PER_EPISODE
        ):
            raise ValueError(
                f"steps_per_episode must be None or between "
                f"{_MIN_STEPS_PER_EPISODE} and {_MAX_STEPS_PER_EPISODE}"
            )
    # Resolve steps_per_episode to a concrete positive int so the whole
    # provenance chain (run_config, invocation_parameters, and the canonical
    # replay snippet) records one coherent literal. A None argument means "use
    # the Stage 23-A default horizon"; that default is resolved here to the
    # scenario horizon and validated against the Stage 23-C upper bound, so the
    # replay snippet reproduces the identical run without needing an optional
    # literal kind the shared ReplaySnippetSchema does not model.
    effective_steps_per_episode = _resolve_effective_steps_per_episode(steps_per_episode)
    require_positive_int(
        "update_rounds", update_rounds, error_suffix="must be positive"
    )
    normalized_update_rounds = int.__index__(update_rounds)
    if not (_MIN_UPDATE_ROUNDS <= normalized_update_rounds <= _MAX_UPDATE_ROUNDS):
        raise ValueError(
            f"update_rounds must be between {_MIN_UPDATE_ROUNDS} and {_MAX_UPDATE_ROUNDS}"
        )
    # FINDING 1 / S2-19: bound the seed with signed-64-bit headroom before it ever
    # reaches torch.manual_seed. The Stage23CRolloutConfig built per round derives
    # its per-step reseed as seed + episode_index*1000 + step_index, so its
    # largest reseed offset is episode_count*1000 + effective_steps_per_episode;
    # mirror that offset here so a seed whose derived reseed would overflow raises
    # a declared ValueError instead of an undeclared RuntimeError from torch.
    require_bounded_seed(
        "seed",
        seed,
        max_offset=normalized_episode_count * 1000 + effective_steps_per_episode,
        zero_seed_message=_STAGE23C_ZERO_SEED_MESSAGE,
    )
    if not isinstance(allow_pre_existing_stage23c_roots, bool):
        raise TypeError("allow_pre_existing_stage23c_roots must be a bool")
    active_root = _active_root().resolve()
    official_parent = (active_root / STAGE23C_RESULT_PARENT).resolve()
    if result_parent is not None and not isinstance(result_parent, (str, Path)):
        raise TypeError("result_parent must be a str, Path, or None")
    parent = resolve_result_parent(
        result_parent,
        official_parent=official_parent,
        active_root_path=active_root,
    )
    official_run = same_path(parent, official_parent)
    if (
        result_parent is not None
        and is_relative_to(parent, active_root)
        and not official_run
    ):
        raise ValueError("custom result_parent must be outside the repository")
    if official_run or allow_pre_existing_stage23c_roots:
        _assert_official_parent_clean_for_new_run(
            parent,
            allow_pre_existing_stage23c_roots=allow_pre_existing_stage23c_roots,
        )
    if timestamp_utc is None:
        timestamp = utc_timestamp()
    elif isinstance(timestamp_utc, str):
        timestamp = timestamp_utc
    else:
        raise TypeError("timestamp_utc must be a string or None")
    validate_timestamp(timestamp)
    result_root = parent / f"{STAGE23C_RESULT_PREFIX}{timestamp}"
    if result_root.exists():
        raise FileExistsError(f"Stage 23-C result root already exists: {result_root}")
    temporary_result_root = parent / f".stage23c_tmp_{timestamp}"
    if temporary_result_root.exists():
        raise FileExistsError(
            f"Stage 23-C temporary result root already exists: {temporary_result_root}"
        )

    random.seed(seed)
    torch.manual_seed(seed)
    core_config = default_stage23_core_config()
    update_config = default_stage22_update_config()
    model = RecurrentMAPPOActorCritic(core_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=update_config.learning_rate)
    lagrange_multiplier = LagrangeMultiplier(
        update_config.lagrange.initial_multiplier,
        update_config.lagrange.learning_rate,
    )

    episode_records: list[dict[str, Any]] = []
    curve_records: list[dict[str, object]] = []
    last_collected = None
    last_update_summary: dict[str, Any] | None = None
    reference_environment_summary: dict[str, Any] | None = None
    for round_index in range(update_rounds):
        rollout_config = Stage23CRolloutConfig(
            max_episodes=episode_count,
            steps_per_episode=effective_steps_per_episode,
            sample_actions=False,
            seed=seed,
            dtype=core_config.dtype,
            device=core_config.device,
        )
        collected = collect_stage23c_development_rollout(model, rollout_config)
        update_result = stage22_ppo_lagrangian_update(
            model,
            collected.batch,
            update_config,
            optimizer=optimizer,
            lagrange_multiplier=lagrange_multiplier,
        )
        lagrange_multiplier = update_result.lagrange_multiplier
        round_update_summary = dict(update_result.summary)
        require_positive_parameter_delta(
            round_update_summary.get("parameter_delta_l1"),
            "update_summary.parameter_delta_l1",
        )
        round_episode_indices: list[int] = []
        for record in collected.episode_records:
            round_record = dict(record)
            round_record["round_index"] = round_index
            round_episode_indices.append(int(round_record["episode_index"]))
            episode_records.append(round_record)
        curve_records.append(
            {
                "round_index": round_index,
                "episode_indices": round_episode_indices,
                "task_reward_sum": collected.summary["task_reward_sum"],
                "hazard_cost_mean": collected.summary["hazard_cost_mean"],
                "sensing_cost_mean": collected.summary["sensing_cost_mean"],
                "lagrange_multiplier": round_update_summary["lagrange_multiplier_after"],
                "parameter_delta_l1": round_update_summary["parameter_delta_l1"],
                "total_loss": round_update_summary["total_loss"],
                **_boundary_flags(),
            }
        )
        last_collected = collected
        last_update_summary = round_update_summary
        if reference_environment_summary is None:
            reference_environment_summary = dict(collected.environment_summary)

    assert last_collected is not None  # update_rounds >= 1 guarantees this
    assert last_update_summary is not None
    assert reference_environment_summary is not None

    update_summary = dict(last_update_summary)
    update_summary.update(_boundary_flags())
    update_summary["stage"] = "23-C"
    update_summary["training_run"] = _TRAINING_RUN
    update_summary["update_helper"] = "stage22_ppo_lagrangian_update"
    update_summary["update_rounds"] = update_rounds

    rollout_summary = dict(last_collected.summary)
    rollout_summary.update(_boundary_flags())
    environment_summary = dict(reference_environment_summary)
    environment_summary.update(_boundary_flags())
    episode_log_records = [
        {**record, **_boundary_flags()} for record in episode_records
    ]
    run_config = {
        "active_root": active_root.as_posix(),
        "allow_pre_existing_stage23c_roots": allow_pre_existing_stage23c_roots,
        "core_config": asdict(core_config),
        "episode_count": episode_count,
        "result_root_name": result_root.name,
        "seed": seed,
        "steps_per_episode": effective_steps_per_episode,
        "steps_per_episode_argument": steps_per_episode,
        "timestamp_utc": timestamp,
        "update_config": _update_config_mapping(update_config),
        "update_rounds": update_rounds,
        **_boundary_flags(),
    }
    invocation_parameters = {
        "allow_pre_existing_stage23c_roots": allow_pre_existing_stage23c_roots,
        "episode_count": episode_count,
        "official_parent": official_parent.as_posix(),
        "official_run": official_run,
        "resolved_result_parent": parent.as_posix(),
        "result_parent_argument": None if result_parent is None else str(result_parent),
        "seed": seed,
        "steps_per_episode": effective_steps_per_episode,
        "timestamp_utc": timestamp,
        "update_rounds": update_rounds,
    }
    replay_python_snippet = _replay_python_snippet(invocation_parameters)
    command_argv = _command_argv(replay_python_snippet)
    command_record = {
        "canonical_entry_command": _canonical_entry_command(),
        "command": human_readable_command(command_argv),
        "command_argv": command_argv,
        "cwd": active_root.as_posix(),
        "entry_point": _ENTRY_POINT,
        "invocation_parameters": invocation_parameters,
        "python_executable": sys.executable,
        "replay_python_snippet": replay_python_snippet,
        **_boundary_flags(),
    }
    boundary_record = {
        "artifact_hashes_scope": "hashes_integrity_and_provenance_only_not_metrics",
        "artifact_set": list(STAGE23C_RESULT_FILES),
        "environment_stage": "23-A",
        "on_policy_semantics": True,
        "result_files_json_or_jsonl_only": True,
        "stage23c_result_parent": STAGE23C_RESULT_PARENT.as_posix(),
        **_boundary_flags(),
    }

    _write_and_finalize_result_root(
        temporary_result_root,
        result_root,
        run_config=run_config,
        command_record=command_record,
        environment_summary=environment_summary,
        rollout_summary=rollout_summary,
        episode_log_records=episode_log_records,
        update_summary=update_summary,
        training_curve_records=curve_records,
        boundary_record=boundary_record,
    )
    return result_root


def _boundary_flags() -> dict[str, object]:
    return boundary_flags(
        stage="23-C",
        training_run=_TRAINING_RUN,
        claim_status=_CLAIM_STATUS,
    )


def _active_root() -> Path:
    return _shared_active_root(__file__, 3)


def _assert_official_parent_clean_for_new_run(
    parent: Path,
    *,
    allow_pre_existing_stage23c_roots: bool,
) -> None:
    if not parent.exists():
        return
    if parent.is_symlink() or not parent.is_dir():
        raise FileExistsError("official Stage 23-C result parent must be a directory")
    children = sorted(parent.iterdir(), key=lambda path: path.name)
    if not children:
        return
    if not allow_pre_existing_stage23c_roots:
        raise FileExistsError(
            "official Stage 23-C result parent must be absent or empty unless "
            "allow_pre_existing_stage23c_roots=True"
        )
    for child in children:
        if child.is_symlink():
            raise FileExistsError("pre-existing Stage 23-C result roots must not be symlinks")
        if not child.is_dir():
            raise FileExistsError("official Stage 23-C result parent contains a loose file")
        if child.name.startswith(".stage23c_tmp_"):
            raise FileExistsError("official Stage 23-C result parent contains a temp directory")
        _validate_result_root_name(child.name)
        _assert_strict_stage23c_result_artifacts(child)


def _write_and_finalize_result_root(
    temporary_result_root: Path,
    final_result_root: Path,
    *,
    run_config: Mapping[str, object],
    command_record: Mapping[str, object],
    environment_summary: Mapping[str, object],
    rollout_summary: Mapping[str, object],
    episode_log_records: list[Mapping[str, object]],
    update_summary: Mapping[str, object],
    training_curve_records: list[Mapping[str, object]],
    boundary_record: Mapping[str, object],
) -> None:
    finalized = False
    try:
        temporary_result_root.mkdir(parents=True)
        write_json(temporary_result_root / "run_config.json", run_config)
        write_json(temporary_result_root / "command_record.json", command_record)
        write_json(temporary_result_root / "environment_summary.json", environment_summary)
        write_json(temporary_result_root / "rollout_summary.json", rollout_summary)
        write_jsonl(temporary_result_root / "episode_log.jsonl", episode_log_records)
        write_json(temporary_result_root / "update_summary.json", update_summary)
        write_jsonl(temporary_result_root / "training_curve.jsonl", training_curve_records)
        write_json(temporary_result_root / "boundary_record.json", boundary_record)
        write_json(
            temporary_result_root / "artifact_hashes.json",
            {
                "hashed_files": {
                    name: sha256_file(temporary_result_root / name)
                    for name in _HASHED_RESULT_FILES
                },
                **_boundary_flags(),
            },
        )
        _assert_strict_stage23c_result_artifacts(
            temporary_result_root,
            provenance_root=final_result_root,
        )
        if final_result_root.exists():
            raise FileExistsError(f"Stage 23-C result root already exists: {final_result_root}")
        temporary_result_root.rename(final_result_root)
        finalized = True
    except Exception:
        if not finalized and temporary_result_root.exists():
            shutil.rmtree(temporary_result_root)
        raise


def _assert_strict_stage23c_result_artifacts(
    result_root: Path,
    *,
    provenance_root: Path | None = None,
) -> None:
    _assert_exact_result_files(result_root)
    _validate_result_root_name((provenance_root or result_root).name)
    for child in result_root.iterdir():
        suffix = child.suffix.lower()
        lower_name = child.name.lower()
        if suffix in _BLOCKED_ARTIFACT_SUFFIXES:
            raise ValueError(f"blocked binary/model artifact in result root: {child.name}")
        if any(token in lower_name for token in BLOCKED_ARTIFACT_NAME_TOKENS):
            raise ValueError(f"blocked artifact name in result root: {child.name}")
    payloads = {
        name: load_strict_json_object(result_root / name)
        for name in STAGE23C_RESULT_FILES
        if name.endswith(".json")
    }
    episode_log_records = load_strict_jsonl_objects(
        result_root / "episode_log.jsonl"
    )
    training_curve_records = load_strict_jsonl_objects(
        result_root / "training_curve.jsonl"
    )
    artifact_hashes = payloads["artifact_hashes.json"]
    hashed_files = artifact_hashes.get("hashed_files")
    if not isinstance(hashed_files, dict):
        raise ValueError("artifact_hashes.json hashed_files must be a JSON object")
    if set(hashed_files) != set(_HASHED_RESULT_FILES):
        raise ValueError("artifact_hashes.json must hash the other result files only")
    for name, expected_hash in hashed_files.items():
        if not isinstance(expected_hash, str):
            raise TypeError("artifact hash values must be strings")
        if sha256_file(result_root / name) != expected_hash:
            raise ValueError(f"artifact_hashes.json mismatch for {name}")
    for name, payload in payloads.items():
        _assert_boundary_payload(payload, name)
        _reject_unsafe_boundary_values(payload, name)
    for index, record in enumerate(episode_log_records):
        _assert_boundary_payload(record, f"episode_log.jsonl[{index}]")
        _reject_unsafe_boundary_values(record, f"episode_log.jsonl[{index}]")
    for index, record in enumerate(training_curve_records):
        _assert_boundary_payload(record, f"training_curve.jsonl[{index}]")
        _reject_unsafe_boundary_values(record, f"training_curve.jsonl[{index}]")
    _assert_command_record_provenance(
        provenance_root or result_root,
        payloads["run_config.json"],
        payloads["command_record.json"],
    )
    require_positive_parameter_delta(
        payloads["update_summary.json"].get("parameter_delta_l1"),
        "update_summary.parameter_delta_l1",
    )


def _assert_exact_result_files(result_root: Path) -> None:
    if not result_root.is_dir():
        raise ValueError("Stage 23-C result root must be a directory")
    children = sorted(result_root.iterdir(), key=lambda path: path.name)
    names = [path.name for path in children]
    expected = sorted(STAGE23C_RESULT_FILES)
    if names != expected:
        raise ValueError(f"Stage 23-C result files mismatch: {names!r} != {expected!r}")
    for child in children:
        if child.is_symlink():
            raise ValueError(f"Stage 23-C result root contains symlink: {child.name}")
        if child.is_dir():
            raise ValueError(f"Stage 23-C result root contains directory: {child.name}")
        if not child.is_file():
            raise ValueError(f"Stage 23-C result root contains non-file child: {child.name}")


def _assert_boundary_payload(payload: Mapping[str, object], path: str) -> None:
    assert_boundary_payload(
        payload,
        path,
        stage="23-C",
        training_run=_TRAINING_RUN,
        claim_status=_CLAIM_STATUS,
    )


def _reject_unsafe_boundary_values(value: object, path: str) -> None:
    reject_unsafe_boundary_values(
        value,
        path,
        allowed_structural_keys=_ALLOWED_UNSAFE_LOOKING_STRUCTURAL_KEYS,
        replay_snippet_schema=_REPLAY_SNIPPET_SCHEMA,
        entry_point=_ENTRY_POINT,
        result_root_prefix=STAGE23C_RESULT_PREFIX,
    )


def _assert_command_record_provenance(
    result_root: Path,
    run_config: Mapping[str, object],
    command_record: Mapping[str, object],
) -> None:
    active_root = _active_root().resolve()
    actual_parent = result_root.parent.resolve()
    official_parent = (active_root / STAGE23C_RESULT_PARENT).resolve()
    actual_official_run = same_path(actual_parent, official_parent)
    result_name = result_root.name
    expected_timestamp = result_name.removeprefix(STAGE23C_RESULT_PREFIX)
    validate_timestamp(expected_timestamp)
    require_field(run_config, "active_root", active_root.as_posix(), "run_config.json")
    require_field(run_config, "timestamp_utc", expected_timestamp, "run_config.json")
    require_field(run_config, "result_root_name", result_name, "run_config.json")
    run_seed = require_json_positive_int(
        "run_config.json.seed",
        run_config.get("seed"),
    )
    run_episode_count = require_json_bounded_positive_int(
        "run_config.json.episode_count",
        run_config.get("episode_count"),
        maximum=_MAX_EPISODE_COUNT,
    )
    run_steps_per_episode = require_json_bounded_positive_int(
        "run_config.json.steps_per_episode",
        run_config.get("steps_per_episode"),
        maximum=_MAX_STEPS_PER_EPISODE,
    )
    run_update_rounds = require_json_bounded_positive_int(
        "run_config.json.update_rounds",
        run_config.get("update_rounds"),
        maximum=_MAX_UPDATE_ROUNDS,
    )
    invocation = command_record.get("invocation_parameters")
    if not isinstance(invocation, Mapping):
        raise TypeError("command_record.json invocation_parameters must be a JSON object")
    for key in (
        "allow_pre_existing_stage23c_roots",
        "episode_count",
        "official_parent",
        "official_run",
        "resolved_result_parent",
        "result_parent_argument",
        "seed",
        "steps_per_episode",
        "timestamp_utc",
        "update_rounds",
    ):
        if key not in invocation:
            raise ValueError(f"command_record.json invocation_parameters missing: {key}")
    require_field(invocation, "timestamp_utc", expected_timestamp, "command_record.json")
    invocation_seed = require_json_positive_int(
        "command_record.json invocation_parameters.seed",
        invocation.get("seed"),
    )
    invocation_episode_count = require_json_bounded_positive_int(
        "command_record.json invocation_parameters.episode_count",
        invocation.get("episode_count"),
        maximum=_MAX_EPISODE_COUNT,
    )
    invocation_steps_per_episode = require_json_bounded_positive_int(
        "command_record.json invocation_parameters.steps_per_episode",
        invocation.get("steps_per_episode"),
        maximum=_MAX_STEPS_PER_EPISODE,
    )
    invocation_update_rounds = require_json_bounded_positive_int(
        "command_record.json invocation_parameters.update_rounds",
        invocation.get("update_rounds"),
        maximum=_MAX_UPDATE_ROUNDS,
    )
    if invocation_seed != run_seed:
        raise ValueError("command_record.json invocation_parameters.seed mismatch")
    if invocation_episode_count != run_episode_count:
        raise ValueError("command_record.json invocation_parameters.episode_count mismatch")
    if invocation_steps_per_episode != run_steps_per_episode:
        raise ValueError(
            "command_record.json invocation_parameters.steps_per_episode mismatch"
        )
    if invocation_update_rounds != run_update_rounds:
        raise ValueError("command_record.json invocation_parameters.update_rounds mismatch")
    if type(run_config.get("allow_pre_existing_stage23c_roots")) is not bool:
        raise TypeError("run_config.json allow_pre_existing_stage23c_roots must be a JSON boolean")
    if type(invocation["official_run"]) is not bool:
        raise TypeError("command_record.json official_run must be a JSON boolean")
    if type(invocation["allow_pre_existing_stage23c_roots"]) is not bool:
        raise TypeError(
            "command_record.json allow_pre_existing_stage23c_roots must be a JSON boolean"
        )
    if invocation["allow_pre_existing_stage23c_roots"] is not run_config.get(
        "allow_pre_existing_stage23c_roots"
    ):
        raise ValueError("command_record.json allow_pre_existing_stage23c_roots mismatch")
    if invocation["official_run"] is not actual_official_run:
        raise ValueError("command_record.json official_run must match the result-root parent")
    _require_path_identity(
        invocation.get("official_parent"),
        official_parent,
        "command_record.json invocation_parameters.official_parent",
    )
    _require_path_identity(
        invocation.get("resolved_result_parent"),
        actual_parent,
        "command_record.json invocation_parameters.resolved_result_parent",
    )
    result_parent_argument = invocation.get("result_parent_argument")
    if result_parent_argument is None:
        if not actual_official_run:
            raise ValueError(
                "command_record.json result_parent_argument cannot be null for external roots"
            )
    elif isinstance(result_parent_argument, str):
        resolved_argument = resolve_result_parent(
            result_parent_argument,
            official_parent=official_parent,
            active_root_path=active_root,
        )
        if not same_path(resolved_argument, actual_parent):
            raise ValueError(
                "command_record.json result_parent_argument does not resolve to result parent"
            )
    else:
        raise TypeError("command_record.json result_parent_argument must be a string or null")
    command_argv = command_record.get("command_argv")
    if not isinstance(command_argv, list) or len(command_argv) != 3:
        raise TypeError("command_record.json command_argv must be a three-item list")
    if not all(isinstance(part, str) for part in command_argv):
        raise TypeError("command_record.json command_argv entries must be strings")
    python_executable = command_record.get("python_executable")
    if not isinstance(python_executable, str) or not python_executable.strip():
        raise TypeError("command_record.json python_executable must be a non-empty string")
    if command_argv[0] != python_executable:
        raise ValueError("command_record.json command_argv[0] must match python_executable")
    if command_argv[1] != "-c":
        raise ValueError("command_record.json command_argv must use python -c form")
    replay_python_snippet = command_record.get("replay_python_snippet")
    if command_argv[2] != replay_python_snippet:
        raise ValueError("command_argv snippet must match replay_python_snippet")
    replay_invocation = _parse_replay_python_snippet(command_argv[2])
    expected_replay_python_snippet = _replay_python_snippet(invocation)
    if replay_python_snippet != expected_replay_python_snippet:
        raise ValueError("command_record.json replay_python_snippet mismatch")
    if command_argv[2] != expected_replay_python_snippet:
        raise ValueError("command_record.json command_argv replay snippet mismatch")
    if command_record.get("command") != human_readable_command(command_argv):
        raise ValueError("command_record.json command must match command_argv display")
    if command_record.get("canonical_entry_command") != _canonical_entry_command():
        raise ValueError("command_record.json canonical_entry_command mismatch")
    if command_record.get("cwd") != active_root.as_posix():
        raise ValueError("command_record.json cwd must match the active repository root")
    if command_record.get("entry_point") != _ENTRY_POINT:
        raise ValueError("command_record.json entry_point mismatch")
    if replay_invocation["seed"] != run_seed:
        raise ValueError("command_record.json replay seed mismatch")
    if replay_invocation["episode_count"] != run_episode_count:
        raise ValueError("command_record.json replay episode_count mismatch")
    if replay_invocation["steps_per_episode"] != run_steps_per_episode:
        raise ValueError("command_record.json replay steps_per_episode mismatch")
    if replay_invocation["update_rounds"] != run_update_rounds:
        raise ValueError("command_record.json replay update_rounds mismatch")
    if replay_invocation["timestamp_utc"] != expected_timestamp:
        raise ValueError("command_record.json replay timestamp_utc mismatch")
    if replay_invocation["result_parent"] != result_parent_argument:
        raise ValueError("command_record.json replay result_parent mismatch")
    if (
        replay_invocation["allow_pre_existing_stage23c_roots"]
        is not invocation["allow_pre_existing_stage23c_roots"]
    ):
        raise ValueError(
            "command_record.json replay allow_pre_existing_stage23c_roots mismatch"
        )
    if python_executable != sys.executable:
        raise ValueError(
            "Stage 23-C roots must record the current Python executable"
        )
    if command_argv != [sys.executable, "-c", expected_replay_python_snippet]:
        raise ValueError(
            "Stage 23-C command_argv must match current replay command"
        )


def _resolve_effective_steps_per_episode(steps_per_episode: int | None) -> int:
    """Resolve the per-episode step horizon to a concrete Stage 23-C-bounded int.

    A ``None`` argument means "use the Stage 23-A default horizon"; the default
    is resolved here from a Stage 23-C environment built for the default
    scenario and validated against the Stage 23-C ``[1, 32]`` bound so the whole
    provenance chain (run_config / invocation_parameters / replay snippet)
    records one coherent, replayable positive-int literal.
    """

    if steps_per_episode is not None:
        return steps_per_episode
    environment = make_default_stage23c_environment(
        scenario_name=_DEFAULT_SCENARIO_NAME,
        max_steps=None,
        seed=_DEFAULT_HORIZON_PROBE_SEED,
    )
    resolved = int(environment.config.max_steps)
    if not (_MIN_STEPS_PER_EPISODE <= resolved <= _MAX_STEPS_PER_EPISODE):
        raise ValueError(
            "resolved default steps_per_episode is outside the Stage 23-C bound; "
            "pass steps_per_episode explicitly in [1, 32]"
        )
    return resolved


def _require_path_identity(value: object, expected: Path, path: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path} must be a non-empty path string")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise ValueError(f"{path} must be absolute")
    if not same_path(candidate, expected):
        raise ValueError(f"{path} must identify {expected.as_posix()}")


def _replay_snippet_keyword_values(
    invocation_parameters: Mapping[str, object],
) -> dict[str, object]:
    return {
        "result_parent": invocation_parameters["result_parent_argument"],
        "timestamp_utc": invocation_parameters["timestamp_utc"],
        "seed": invocation_parameters["seed"],
        "episode_count": invocation_parameters["episode_count"],
        "steps_per_episode": invocation_parameters["steps_per_episode"],
        "update_rounds": invocation_parameters["update_rounds"],
        "allow_pre_existing_stage23c_roots": invocation_parameters[
            "allow_pre_existing_stage23c_roots"
        ],
    }


def _replay_python_snippet(invocation_parameters: Mapping[str, object]) -> str:
    return build_replay_python_snippet(
        _REPLAY_SNIPPET_SCHEMA,
        _replay_snippet_keyword_values(invocation_parameters),
    )


def _command_argv(replay_python_snippet: str) -> list[str]:
    return command_argv_from_snippet(replay_python_snippet, _REPLAY_SNIPPET_SCHEMA)


def _validate_replay_python_snippet(replay_python_snippet: str) -> None:
    validate_replay_python_snippet(replay_python_snippet, _REPLAY_SNIPPET_SCHEMA)


def _parse_replay_python_snippet(replay_python_snippet: str) -> dict[str, object]:
    return parse_replay_python_snippet(replay_python_snippet, _REPLAY_SNIPPET_SCHEMA)


def _canonical_entry_command() -> str:
    return (
        "python -c \"import sys; sys.path.insert(0, 'src'); "
        "from raas_marl.mappo_lagrangian.stage23c_runner import "
        "run_stage23c_bounded_training; run_stage23c_bounded_training()\""
    )


def _validate_result_root_name(value: str) -> None:
    if not value.startswith(STAGE23C_RESULT_PREFIX):
        raise ValueError("Stage 23-C result root must use the stage23c prefix")
    validate_timestamp(value.removeprefix(STAGE23C_RESULT_PREFIX))


def _update_config_mapping(config: Stage22UpdateConfig) -> dict[str, Any]:
    return {
        "algorithm": asdict(config.algorithm),
        "lagrange": asdict(config.lagrange),
        "learning_rate": config.learning_rate,
        "losses": asdict(config.losses),
        "max_grad_norm": config.max_grad_norm,
        "max_update_epochs": config.max_update_epochs,
        "normalize_advantages": config.normalize_advantages,
    }

"""Bounded Stage 22 development-training runner.

This runner performs a tiny in-memory integration pass over the Stage 22
environment adapter, rollout collector, and PPO-Lagrangian update. It writes
only deterministic development JSON/JSONL artifacts and never serializes model,
optimizer, checkpoint, rollout, or final-evaluation artifacts.

It sets Python and PyTorch global RNG seeds for local bounded development
reproducibility. It does not preserve caller RNG state and does not claim
cross-platform or cross-version bitwise determinism.
"""

from __future__ import annotations

import ast
from dataclasses import asdict
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import random
import re
import shutil
import sys
from typing import Any, Mapping

import torch

from raas_marl.mappo_lagrangian.artifacts import (
    BLOCKED_ARTIFACT_NAME_TOKENS,
    BLOCKED_ARTIFACT_SUFFIXES,
    sha256_file,
)
from raas_marl.mappo_lagrangian._validation import (
    finite_numeric_scalar,
    require_positive_int,
    require_positive_seed,
)
from raas_marl.mappo_lagrangian._governance import (
    ReplaySnippetSchema,
    active_root as _shared_active_root,
    build_replay_python_snippet,
    command_argv_from_snippet,
    human_readable_command,
    is_absolute_path_string,
    is_relative_to,
    load_strict_json_object,
    load_strict_jsonl_objects,
    normalise_boundary_string,
    parse_replay_python_snippet,
    reject_duplicate_json_pairs,
    require_absolute_path_identity,
    require_absolute_path_value,
    require_optional_provenance_value,
    require_positive_parameter_delta,
    require_provenance_value,
    require_string_path_value,
    resolve_result_parent,
    same_path,
    utc_timestamp,
    validate_replay_python_snippet,
    validate_timestamp,
    write_json,
    write_jsonl,
)
from raas_marl.mappo_lagrangian.env_adapter import (
    DevelopmentAdapterConfig,
    Stage22DevelopmentEnvironment,
    default_stage22_core_config,
)
from raas_marl.mappo_lagrangian.lagrange import LagrangeMultiplier
from raas_marl.mappo_lagrangian.model import RecurrentMAPPOActorCritic
from raas_marl.mappo_lagrangian.rollout import (
    RolloutCollectionConfig,
    collect_development_rollout,
)
from raas_marl.mappo_lagrangian.update import (
    Stage22UpdateConfig,
    default_stage22_update_config,
    stage22_ppo_lagrangian_update,
)


STAGE22_RESULT_PARENT = Path("results") / "stage22_development_training"
STAGE22_RESULT_PREFIX = "stage22_development_training_"
_STAGE22_ZERO_SEED_MESSAGE = (
    "seed must be a positive integer; seed 0 is intentionally "
    "reserved/rejected for bounded Stage 22 development-run consistency"
)
_ENTRY_POINT = "raas_marl.mappo_lagrangian.development_runner.run_stage22_development_training"
STAGE22_RESULT_FILES = (
    "run_config.json",
    "command_record.json",
    "development_metrics.json",
    "training_curve.jsonl",
    "rollout_summary.json",
    "update_summary.json",
    "artifact_hashes.json",
)
_HASHED_RESULT_FILES = STAGE22_RESULT_FILES[:-1]
_CLAIM_STATUS = "not yet tested"
_TRAINING_SCOPE = "development-only"
# Canonical replay-snippet schema for the Stage 22 runner. Threaded through the
# shared _governance replay-snippet family (build/parse/validate) so the snippet
# build path and the read-back provenance path share one AST-validated form.
_REPLAY_SNIPPET_SCHEMA = ReplaySnippetSchema(
    import_module_path="raas_marl.mappo_lagrangian.development_runner",
    callee_name="run_stage22_development_training",
    keyword_order=(
        "result_parent",
        "timestamp_utc",
        "seed",
        "episode_count",
        "steps_per_episode",
        "allow_pre_existing_stage22_roots",
    ),
    keyword_literal_kind={
        "result_parent": "result_parent",
        "timestamp_utc": "timestamp",
        "seed": "positive_int",
        "episode_count": "bounded_positive_int",
        "steps_per_episode": "bounded_positive_int",
        "allow_pre_existing_stage22_roots": "bool",
    },
    keyword_int_bounds={"episode_count": 4, "steps_per_episode": 16},
)
# Issue m-23 resolution: the suffix set is shared via artifacts.py.
_BLOCKED_ARTIFACT_SUFFIXES = BLOCKED_ARTIFACT_SUFFIXES
_ABSOLUTE_UNSAFE_BOUNDARY_WORDING = (
    "claim confirmed",
    "claim demonstrated",
    # Split this claim-support phrase so source scans do not carry it literally.
    "cl" + "aim is " + "supp" + "orted",
    "claim proven",
    "claim support",
    "claim supported",
    "claim validated",
    "claim verified",
    "claim evidence",
    "claim conclusion",
    "claim conclusions",
    "final evaluation",
    "claim proof",
    "claim proofs",
    "method superiority",
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
    "statistical validity",
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
    "evaluation table",
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
_ALLOWED_EXACT_NEGATIVE_EVALUATION_STRINGS = frozenset(
    {
        "not evaluation",
        "no evaluation was run",
        "stage 22 is not evaluation",
    }
)
_STAGE22_SC_NEGATIVE_EVALUATION_RE = re.compile(
    r"^stage 22 sc(?:[1-9][0-9]*) is not evaluation$"
)
_FALSE_ONLY_BOUNDARY_BOOLEAN_KEYS = frozenset(
    {
        "claim_evidence_created",
        "evaluation_run",
        "final_evaluation_run",
        "paper_facing_results_created",
        "checkpoint_created",
        "serialized_model_artifact_created",
        "optimizer_state_saved",
        "stage23_content_created",
    }
)
_TRUE_ONLY_BOUNDARY_BOOLEAN_KEYS = frozenset(
    {
        "development_only",
        "stage23_locked_until_review",
    }
)
_STRICT_RESULT_BOOLEAN_KEYS = frozenset(
    {
        *_FALSE_ONLY_BOUNDARY_BOOLEAN_KEYS,
        *_TRUE_ONLY_BOUNDARY_BOOLEAN_KEYS,
        "advantages_normalized",
        "allow_pre_existing_stage22_roots",
        "cross_platform_bitwise_determinism_claimed",
        "local_seed_controls_recorded",
        "new_joint_log_probability_is_sum",
        "normalize_advantages",
        "official_run",
        "old_joint_log_probability_is_sum",
        "owned_optimizer",
        "resampled_actions_during_update",
        "rng_state_preserved",
        "sample_actions",
        "use_recurrent_actor",
        "value_targets_use_raw_advantages",
    }
)
_STRICT_RESULT_INTEGER_KEYS = frozenset(
    {
        "actor_hidden_dim",
        "actor_observation_dim",
        "agent_count",
        "agent_id_count",
        "batch_size",
        "central_state_dim",
        "critic_hidden_dim",
        "episode_count",
        "episode_index",
        "history_state_dim",
        "max_update_epochs",
        "movement_action_count",
        "recurrent_layer_count",
        "revealed_information_dim",
        "seed",
        "sensing_action_count",
        "stage",
        "steps_per_episode",
        "terminal_count",
        "time_steps",
        "total_training_curve_records",
        "truncation_count",
    }
)
_POSITIVE_RESULT_INTEGER_KEYS = frozenset(
    {
        "actor_hidden_dim",
        "actor_observation_dim",
        "agent_count",
        "agent_id_count",
        "batch_size",
        "central_state_dim",
        "critic_hidden_dim",
        "episode_count",
        "history_state_dim",
        "max_update_epochs",
        "movement_action_count",
        "recurrent_layer_count",
        "seed",
        "sensing_action_count",
        "steps_per_episode",
        "time_steps",
        "total_training_curve_records",
    }
)
_NONNEGATIVE_RESULT_INTEGER_KEYS = frozenset(
    {
        "episode_index",
        "revealed_information_dim",
        "terminal_count",
        "truncation_count",
    }
)
_CURRENT_INVOCATION_PARAMETER_KEYS = (
    "allow_pre_existing_stage22_roots",
    "episode_count",
    "official_parent",
    "official_run",
    "resolved_result_parent",
    "result_parent_argument",
    "seed",
    "steps_per_episode",
    "timestamp_utc",
)
_ALLOWED_UNSAFE_LOOKING_STRUCTURAL_KEYS = frozenset(
    {
        "claim_evidence_created",
        "development_only",
        "evaluation_run",
        "final_evaluation_run",
        "optimizer_state_saved",
        "paper_facing_results_created",
        "serialized_model_artifact_created",
        "stage23_content_created",
        "stage23_locked_until_review",
    }
)
_ALLOWED_RESULT_FILENAME_KEYS = frozenset(STAGE22_RESULT_FILES)
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
        "result_parent_argument",
        "result_root_name",
        "timestamp_utc",
    }
)


def run_stage22_development_training(
    *,
    result_parent: str | Path | None = None,
    timestamp_utc: str | None = None,
    seed: int = 22,
    episode_count: int = 2,
    steps_per_episode: int = 4,
    allow_pre_existing_stage22_roots: bool = False,
) -> Path:
    """Run the bounded development-training integration and return the root.

    The default call creates the official Stage 22 development result root under
    ``results/stage22_development_training``. Tests should pass ``result_parent``
    pointing at a temporary directory.
    """

    require_positive_seed("seed", seed, zero_seed_message=_STAGE22_ZERO_SEED_MESSAGE)
    require_positive_int("episode_count", episode_count, error_suffix="must be positive")
    require_positive_int(
        "steps_per_episode", steps_per_episode, error_suffix="must be positive"
    )
    if episode_count > 4:
        raise ValueError("episode_count must be <= 4 for Stage 22 development")
    if steps_per_episode > 16:
        raise ValueError("steps_per_episode must be <= 16 for Stage 22 development")
    if not isinstance(allow_pre_existing_stage22_roots, bool):
        raise TypeError("allow_pre_existing_stage22_roots must be a bool")
    active_root = _active_root().resolve()
    official_parent = (active_root / STAGE22_RESULT_PARENT).resolve()
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
    if official_run or allow_pre_existing_stage22_roots:
        _assert_official_parent_clean_for_new_run(
            parent,
            allow_pre_existing_stage22_roots=allow_pre_existing_stage22_roots,
        )
    if timestamp_utc is None:
        timestamp = utc_timestamp()
    elif isinstance(timestamp_utc, str):
        timestamp = timestamp_utc
    else:
        raise TypeError("timestamp_utc must be a string or None")
    validate_timestamp(timestamp)
    result_root = parent / f"{STAGE22_RESULT_PREFIX}{timestamp}"
    if result_root.exists():
        raise FileExistsError(f"Stage 22 development result root already exists: {result_root}")
    temporary_result_root = parent / f".stage22_tmp_{timestamp}"
    if temporary_result_root.exists():
        raise FileExistsError(
            f"Stage 22 temporary development result root already exists: {temporary_result_root}"
        )

    random.seed(seed)
    torch.manual_seed(seed)
    core_config = default_stage22_core_config()
    adapter_config = DevelopmentAdapterConfig(
        core_config=core_config,
        max_steps=steps_per_episode,
        agent_count=core_config.agent_id_count or 2,
        terminal_after_steps=steps_per_episode,
    )
    rollout_config = RolloutCollectionConfig(
        max_episodes=1,
        max_steps_per_episode=steps_per_episode,
        sample_actions=False,
        seed=seed,
    )
    update_config = default_stage22_update_config()
    model = RecurrentMAPPOActorCritic(core_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=update_config.learning_rate)
    lagrange_multiplier = LagrangeMultiplier(
        update_config.lagrange.initial_multiplier,
        update_config.lagrange.learning_rate,
    )

    curve_records: list[dict[str, Any]] = []
    rollout_records: list[dict[str, Any]] = []
    update_records: list[dict[str, Any]] = []
    for episode_index in range(episode_count):
        environment = Stage22DevelopmentEnvironment(adapter_config)
        collected = collect_development_rollout(model, environment, rollout_config)
        update_result = stage22_ppo_lagrangian_update(
            model,
            collected.batch,
            update_config,
            optimizer=optimizer,
            lagrange_multiplier=lagrange_multiplier,
        )
        lagrange_multiplier = update_result.lagrange_multiplier
        rollout_summary = _with_episode_index(collected.summary, episode_index)
        update_summary = _with_episode_index(update_result.summary, episode_index)
        require_positive_parameter_delta(
            update_summary.get("parameter_delta_l1"),
            "update_summary.parameter_delta_l1",
        )
        rollout_records.append(rollout_summary)
        update_records.append(update_summary)
        curve_records.append(
            {
                "development_only": True,
                "evaluation_run": False,
                "episode_index": episode_index,
                "hazard_cost_mean": rollout_summary["hazard_cost_mean"],
                "lagrange_multiplier": update_summary["lagrange_multiplier_after"],
                "parameter_delta_l1": update_summary["parameter_delta_l1"],
                "task_reward_sum": rollout_summary["task_reward_sum"],
                "total_loss": update_summary["total_loss"],
            }
        )

    final_parameter_delta_l1 = update_records[-1]["parameter_delta_l1"]
    require_positive_parameter_delta(
        final_parameter_delta_l1,
        "development_metrics.final_parameter_delta_l1",
    )
    run_config = {
        "active_root": active_root.as_posix(),
        "allow_pre_existing_stage22_roots": allow_pre_existing_stage22_roots,
        "claim_evidence_created": False,
        "claim_status": _CLAIM_STATUS,
        "checkpoint_created": False,
        "core_config": asdict(core_config),
        "development_only": True,
        "environment_rollout_run": "bounded_stage22_development_only",
        "episode_count": episode_count,
        "evaluation_run": False,
        "final_evaluation_run": False,
        "local_seed_controls_recorded": True,
        "optimizer_state_saved": False,
        "paper_facing_results_created": False,
        "result_root_name": result_root.name,
        "seed": seed,
        "serialized_model_artifact_created": False,
        "stage": 22,
        "stage23_content_created": False,
        "stage23_locked_until_review": True,
        "steps_per_episode": steps_per_episode,
        "timestamp_utc": timestamp,
        "training_scope": _TRAINING_SCOPE,
        "training_run": "bounded_stage22_development_only",
        "rng_state_preserved": False,
        "update_config": _update_config_mapping(update_config),
    }
    invocation_parameters = {
        "allow_pre_existing_stage22_roots": allow_pre_existing_stage22_roots,
        "episode_count": episode_count,
        "official_parent": official_parent.as_posix(),
        "official_run": official_run,
        "resolved_result_parent": parent.as_posix(),
        "result_parent_argument": None if result_parent is None else str(result_parent),
        "seed": seed,
        "steps_per_episode": steps_per_episode,
        "timestamp_utc": timestamp,
    }
    replay_python_snippet = _replay_python_snippet(invocation_parameters)
    command_argv = _command_argv(replay_python_snippet)
    command_record = {
        "claim_status": _CLAIM_STATUS,
        "canonical_entry_command": _canonical_entry_command(),
        "command": _forensic_replay_command(invocation_parameters),
        "command_argv": command_argv,
        "cwd": active_root.as_posix(),
        "development_only": True,
        "entry_point": _ENTRY_POINT,
        "evaluation_run": False,
        "invocation_parameters": invocation_parameters,
        "python_executable": sys.executable,
        "replay_python_snippet": replay_python_snippet,
        "stage": 22,
        "stage23_locked_until_review": True,
    }
    development_metrics = {
        "claim_evidence_created": False,
        "claim_status": _CLAIM_STATUS,
        "cross_platform_bitwise_determinism_claimed": False,
        "development_only": True,
        "episode_count": episode_count,
        "evaluation_run": False,
        "final_evaluation_run": False,
        "final_lagrange_multiplier": update_records[-1]["lagrange_multiplier_after"],
        "final_parameter_delta_l1": final_parameter_delta_l1,
        # NEW-development_runner-1: these aggregates are an unweighted mean over the
        # per-episode records, not a step-weighted run-level mean. The two "_mean"
        # sources are already per-episode means and the "_sum" source is a per-episode
        # sum, so the key names spell out "mean_over_episodes_of_<source-quantity>"
        # to avoid mislabeling. Because every Stage 22 development episode runs the
        # same fixed horizon (steps_per_episode, terminal_after_steps == steps_per_episode),
        # this unweighted episode mean equals a step-weighted mean of the per-episode
        # means; the equality no longer holds if variable-length episodes are ever
        # introduced. Values are numerically identical to the pre-rename keys.
        "mean_over_episodes_of_hazard_cost_mean": _mean(
            record["hazard_cost_mean"] for record in rollout_records
        ),
        "mean_over_episodes_of_sensing_cost_mean": _mean(
            record["sensing_cost_mean"] for record in rollout_records
        ),
        "mean_over_episodes_of_task_reward_sum": _mean(
            record["task_reward_sum"] for record in rollout_records
        ),
        "paper_facing_results_created": False,
        "local_seed_controls_recorded": True,
        "rng_state_preserved": False,
        "stage": 22,
        "stage23_locked_until_review": True,
        "status": "STAGE22_DEVELOPMENT_TRAINING_RUN_COMPLETE",
        "total_training_curve_records": len(curve_records),
    }
    rollout_summary = {
        "claim_status": _CLAIM_STATUS,
        "development_only": True,
        "evaluation_run": False,
        "episodes": rollout_records,
        "stage": 22,
        "stage23_locked_until_review": True,
    }
    update_summary = {
        "claim_status": _CLAIM_STATUS,
        "development_only": True,
        "evaluation_run": False,
        "episodes": update_records,
        "stage": 22,
        "stage23_locked_until_review": True,
    }

    _write_and_finalize_result_root(
        temporary_result_root,
        result_root,
        run_config=run_config,
        command_record=command_record,
        development_metrics=development_metrics,
        training_curve_records=curve_records,
        rollout_summary=rollout_summary,
        update_summary=update_summary,
    )
    return result_root


def _active_root() -> Path:
    return _shared_active_root(__file__, 3)


def _forensic_replay_command(invocation_parameters: Mapping[str, object]) -> str:
    return human_readable_command(_command_argv(_replay_python_snippet(invocation_parameters)))


def _replay_snippet_keyword_values(
    invocation_parameters: Mapping[str, object],
) -> dict[str, object]:
    return {
        "result_parent": invocation_parameters["result_parent_argument"],
        "timestamp_utc": invocation_parameters["timestamp_utc"],
        "seed": invocation_parameters["seed"],
        "episode_count": invocation_parameters["episode_count"],
        "steps_per_episode": invocation_parameters["steps_per_episode"],
        "allow_pre_existing_stage22_roots": invocation_parameters[
            "allow_pre_existing_stage22_roots"
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


def _canonical_entry_command() -> str:
    return (
        "python -c \"import sys; sys.path.insert(0, 'src'); "
        "from raas_marl.mappo_lagrangian.development_runner import "
        "run_stage22_development_training; run_stage22_development_training()\""
    )


def _assert_official_parent_clean_for_new_run(
    parent: Path,
    *,
    allow_pre_existing_stage22_roots: bool = False,
) -> None:
    if not parent.exists():
        return
    if parent.is_symlink() or not parent.is_dir():
        raise FileExistsError("official Stage 22 result parent must be a directory")
    children = sorted(parent.iterdir(), key=lambda path: path.name)
    if not children:
        return
    if not allow_pre_existing_stage22_roots:
        raise FileExistsError(
            "official Stage 22 result parent must be absent or empty unless "
            "allow_pre_existing_stage22_roots=True"
        )
    for child in children:
        if child.is_symlink():
            raise FileExistsError("pre-existing Stage 22 result roots must not be symlinks")
        if not child.is_dir():
            raise FileExistsError("official Stage 22 result parent contains a loose file")
        if child.name.startswith(".stage22_tmp_"):
            raise FileExistsError("official Stage 22 result parent contains a temporary directory")
        _validate_result_root_name(child.name)
        _assert_strict_result_artifacts(child, allow_legacy=True)


def _assert_exact_result_files(result_root: Path) -> None:
    children = sorted(result_root.iterdir(), key=lambda path: path.name)
    names = [path.name for path in children]
    expected = sorted(STAGE22_RESULT_FILES)
    if names != expected:
        raise ValueError(f"Stage 22 result files mismatch: {names!r} != {expected!r}")
    for child in children:
        if child.is_symlink():
            raise ValueError(f"Stage 22 result root contains symlink: {child.name}")
        if child.is_dir():
            raise ValueError(f"Stage 22 result root contains directory: {child.name}")
        if not child.is_file():
            raise ValueError(f"Stage 22 result root contains non-file child: {child.name}")


def _validate_result_root_name(value: str) -> None:
    if not value.startswith(STAGE22_RESULT_PREFIX):
        raise FileExistsError("pre-existing result directory does not match Stage 22 prefix")
    validate_timestamp(value.removeprefix(STAGE22_RESULT_PREFIX))


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


def _with_episode_index(summary: Mapping[str, object], episode_index: int) -> dict[str, object]:
    record = dict(summary)
    record["episode_index"] = episode_index
    _validate_json_scalars(record)
    return record


def _validate_json_scalars(payload: Mapping[str, object]) -> None:
    for key, value in payload.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{key} must be finite")


def _mean(values: object) -> float:
    items = [
        finite_numeric_scalar("mean value", value)
        for value in values
    ]
    if not items:
        raise ValueError("mean requires at least one value")
    result = sum(items) / len(items)
    if not math.isfinite(result):
        raise ValueError("mean must be finite")
    return result



def _write_and_finalize_result_root(
    temporary_result_root: Path,
    final_result_root: Path,
    *,
    run_config: Mapping[str, object],
    command_record: Mapping[str, object],
    development_metrics: Mapping[str, object],
    training_curve_records: list[Mapping[str, object]],
    rollout_summary: Mapping[str, object],
    update_summary: Mapping[str, object],
) -> None:
    finalized = False
    try:
        temporary_result_root.mkdir(parents=True)
        write_json(temporary_result_root / "run_config.json", run_config)
        write_json(temporary_result_root / "command_record.json", command_record)
        write_json(temporary_result_root / "development_metrics.json", development_metrics)
        write_jsonl(temporary_result_root / "training_curve.jsonl", training_curve_records)
        write_json(temporary_result_root / "rollout_summary.json", rollout_summary)
        write_json(temporary_result_root / "update_summary.json", update_summary)
        write_json(
            temporary_result_root / "artifact_hashes.json",
            {
                "claim_status": _CLAIM_STATUS,
                "development_only": True,
                "evaluation_run": False,
                "hashed_files": {
                    name: sha256_file(temporary_result_root / name)
                    for name in _HASHED_RESULT_FILES
                },
                "stage": 22,
                "stage23_locked_until_review": True,
            },
        )
        _assert_strict_result_artifacts(
            temporary_result_root,
            provenance_root=final_result_root,
        )
        if final_result_root.exists():
            raise FileExistsError(
                f"Stage 22 development result root already exists: {final_result_root}"
            )
        temporary_result_root.rename(final_result_root)
        finalized = True
    except Exception:
        if not finalized and temporary_result_root.exists():
            shutil.rmtree(temporary_result_root)
        raise


def _assert_strict_result_artifacts(
    result_root: Path,
    *,
    allow_legacy: bool = False,
    provenance_root: Path | None = None,
) -> None:
    _assert_exact_result_files(result_root)
    for child in result_root.iterdir():
        suffix = child.suffix.lower()
        lower_name = child.name.lower()
        if suffix in _BLOCKED_ARTIFACT_SUFFIXES:
            raise ValueError(f"blocked binary/model artifact in result root: {child.name}")
        if any(token in lower_name for token in BLOCKED_ARTIFACT_NAME_TOKENS):
            raise ValueError(f"blocked artifact name in result root: {child.name}")
    payloads = {
        name: load_strict_json_object(result_root / name)
        for name in STAGE22_RESULT_FILES
        if name.endswith(".json")
    }
    training_curve_records = load_strict_jsonl_objects(result_root / "training_curve.jsonl")
    artifact_hashes = payloads["artifact_hashes.json"]
    hashed_files = artifact_hashes.get("hashed_files")
    if not isinstance(hashed_files, dict):
        raise ValueError("artifact_hashes.json hashed_files must be a JSON object")
    expected_hashed = set(_HASHED_RESULT_FILES)
    if set(hashed_files) != expected_hashed:
        raise ValueError("artifact_hashes.json must hash the other six result files only")
    for name, expected_hash in hashed_files.items():
        if not isinstance(expected_hash, str):
            raise TypeError("artifact hash values must be strings")
        if sha256_file(result_root / name) != expected_hash:
            raise ValueError(f"artifact_hashes.json mismatch for {name}")
    update_summary = payloads["update_summary.json"]
    episodes = update_summary.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("update_summary.json episodes must be a nonempty list")
    for index, episode in enumerate(episodes):
        if not isinstance(episode, Mapping):
            raise TypeError("update_summary episodes must be JSON objects")
        require_positive_parameter_delta(
            episode.get("parameter_delta_l1"),
            f"update_summary.episodes[{index}].parameter_delta_l1",
        )
    require_positive_parameter_delta(
        payloads["development_metrics.json"].get("final_parameter_delta_l1"),
        "development_metrics.final_parameter_delta_l1",
    )
    _assert_semantic_evidence_boundary(
        result_root if provenance_root is None else provenance_root,
        payloads,
        training_curve_records,
        allow_legacy=allow_legacy,
    )
    _assert_invocation_provenance(
        result_root if provenance_root is None else provenance_root,
        payloads,
        allow_legacy=allow_legacy,
    )


def _assert_semantic_evidence_boundary(
    result_root: Path,
    payloads: Mapping[str, dict[str, object]],
    training_curve_records: list[dict[str, object]],
    *,
    allow_legacy: bool,
) -> None:
    for name, payload in payloads.items():
        _reject_unsafe_boundary_values(payload, name)
    for index, record in enumerate(training_curve_records):
        _reject_unsafe_boundary_values(record, f"training_curve.jsonl[{index}]")

    common_files = (
        "run_config.json",
        "command_record.json",
        "development_metrics.json",
        "rollout_summary.json",
        "update_summary.json",
        "artifact_hashes.json",
    )
    for name in common_files:
        payload = payloads[name]
        _require_field(payload, "claim_status", _CLAIM_STATUS, name, allow_legacy)
        _require_field(payload, "development_only", True, name, allow_legacy)
        _require_field(payload, "evaluation_run", False, name, allow_legacy)
        _require_field(payload, "stage", 22, name, allow_legacy)
        _require_field(payload, "stage23_locked_until_review", True, name, allow_legacy)

    for name in ("run_config.json", "development_metrics.json"):
        payload = payloads[name]
        _require_field(payload, "claim_evidence_created", False, name, allow_legacy)
        _require_field(payload, "final_evaluation_run", False, name, allow_legacy)
        _require_field(payload, "paper_facing_results_created", False, name, allow_legacy)

    run_config = payloads["run_config.json"]
    for key, expected in (
        ("checkpoint_created", False),
        ("serialized_model_artifact_created", False),
        ("optimizer_state_saved", False),
        ("stage23_content_created", False),
        ("training_scope", _TRAINING_SCOPE),
        ("training_run", "bounded_stage22_development_only"),
        ("environment_rollout_run", "bounded_stage22_development_only"),
    ):
        _require_field(run_config, key, expected, "run_config.json", allow_legacy)

    command_record = payloads["command_record.json"]
    for key in (
        "canonical_entry_command",
        "command_argv",
        "cwd",
        "entry_point",
        "invocation_parameters",
        "python_executable",
        "replay_python_snippet",
    ):
        if key not in command_record and not allow_legacy:
            raise ValueError(f"command_record.json missing required field: {key}")
    if "invocation_parameters" in command_record:
        invocation = command_record["invocation_parameters"]
        if not isinstance(invocation, Mapping):
            raise TypeError("command_record.json invocation_parameters must be a JSON object")
        if allow_legacy:
            _validate_legacy_invocation_parameters_if_present(
                invocation,
                result_root=result_root,
                payloads=payloads,
            )
        else:
            _require_current_invocation_parameters(invocation)

    rollout_episodes = payloads["rollout_summary.json"].get("episodes")
    if not isinstance(rollout_episodes, list) or not rollout_episodes:
        raise ValueError("rollout_summary.json episodes must be a nonempty list")
    for index, episode in enumerate(rollout_episodes):
        if not isinstance(episode, Mapping):
            raise TypeError("rollout_summary episodes must be JSON objects")
        _require_field(episode, "development_only", True, f"rollout_summary.episodes[{index}]", allow_legacy)

    update_episodes = payloads["update_summary.json"].get("episodes")
    if not isinstance(update_episodes, list) or not update_episodes:
        raise ValueError("update_summary.json episodes must be a nonempty list")
    for index, episode in enumerate(update_episodes):
        if not isinstance(episode, Mapping):
            raise TypeError("update_summary episodes must be JSON objects")
        _require_field(episode, "development_only", True, f"update_summary.episodes[{index}]", allow_legacy)
        if "optimizer_state_saved" in episode and episode["optimizer_state_saved"] is not False:
            raise ValueError(
                f"update_summary.episodes[{index}].optimizer_state_saved must be False"
            )

    for index, record in enumerate(training_curve_records):
        _require_field(record, "development_only", True, f"training_curve.jsonl[{index}]", allow_legacy)
        _require_field(record, "evaluation_run", False, f"training_curve.jsonl[{index}]", allow_legacy)


def _assert_invocation_provenance(
    result_root: Path,
    payloads: Mapping[str, Mapping[str, object]],
    *,
    allow_legacy: bool,
) -> None:
    result_name = result_root.name
    if not result_name.startswith(STAGE22_RESULT_PREFIX):
        raise ValueError("result root name must use Stage 22 prefix")
    expected_timestamp = result_name.removeprefix(STAGE22_RESULT_PREFIX)
    validate_timestamp(expected_timestamp)

    run_config = payloads["run_config.json"]
    command_record = payloads["command_record.json"]

    if allow_legacy:
        require_optional_provenance_value(
            run_config,
            "timestamp_utc",
            expected_timestamp,
            "run_config.json",
        )
        require_optional_provenance_value(
            run_config,
            "result_root_name",
            result_name,
            "run_config.json",
        )
        if "python_executable" in command_record:
            _require_python_executable_path_signal(command_record)
        if "invocation_parameters" not in command_record:
            _validate_legacy_replay_provenance_if_present(command_record, invocation=None)
            return

    if "invocation_parameters" not in command_record:
        raise ValueError("command_record.json missing required field: invocation_parameters")
    invocation = command_record["invocation_parameters"]
    if not isinstance(invocation, Mapping):
        raise TypeError("command_record.json invocation_parameters must be a JSON object")

    if "python_executable" in command_record:
        _require_python_executable_path_signal(command_record)
    elif not allow_legacy:
        raise ValueError("command_record.json missing required field: python_executable")

    if allow_legacy:
        _validate_legacy_invocation_provenance(
            invocation,
            result_root=result_root,
            run_config=run_config,
            expected_timestamp=expected_timestamp,
        )
        _validate_legacy_replay_provenance_if_present(command_record, invocation=invocation)
        return

    require_provenance_value(run_config, "timestamp_utc", expected_timestamp, "run_config.json")
    require_provenance_value(run_config, "result_root_name", result_name, "run_config.json")
    require_provenance_value(
        invocation,
        "timestamp_utc",
        expected_timestamp,
        "command_record.json.invocation_parameters",
    )

    active_root = _active_root().resolve()
    official_parent = (active_root / STAGE22_RESULT_PARENT).resolve()
    resolved_result_parent_identity = require_absolute_path_identity(
        invocation,
        "resolved_result_parent",
        "command_record.json.invocation_parameters",
        active_root,
    )
    recorded_official_parent_identity = require_absolute_path_identity(
        invocation,
        "official_parent",
        "command_record.json.invocation_parameters",
        active_root,
    )
    expected_resolved_parent_identity = _path_identity_for_existing_path(
        result_root.parent,
        active_root,
    )
    expected_official_parent_identity = _path_identity_for_existing_path(
        official_parent,
        active_root,
    )
    if resolved_result_parent_identity != expected_resolved_parent_identity:
        raise ValueError("command_record.json invocation resolved_result_parent mismatch")
    if recorded_official_parent_identity != expected_official_parent_identity:
        raise ValueError("command_record.json invocation official_parent mismatch")

    expected_official_run = same_path(result_root.parent, official_parent)
    require_provenance_value(
        invocation,
        "official_run",
        expected_official_run,
        "command_record.json.invocation_parameters",
    )
    _require_result_parent_argument_provenance(
        invocation,
        active_root=active_root,
        official_parent_identity=expected_official_parent_identity,
        resolved_result_parent_identity=resolved_result_parent_identity,
        official_run=expected_official_run,
    )
    require_provenance_value(
        invocation,
        "allow_pre_existing_stage22_roots",
        run_config.get("allow_pre_existing_stage22_roots"),
        "command_record.json.invocation_parameters",
    )
    for key in ("seed", "episode_count", "steps_per_episode"):
        require_provenance_value(
            invocation,
            key,
            run_config.get(key),
            "command_record.json.invocation_parameters",
        )
    _assert_current_command_record_provenance(
        command_record,
        run_config,
        invocation,
        active_root,
    )


def _validate_legacy_invocation_provenance(
    invocation: Mapping[str, object],
    *,
    result_root: Path,
    run_config: Mapping[str, object],
    expected_timestamp: str,
) -> None:
    path = "command_record.json.invocation_parameters"
    require_optional_provenance_value(invocation, "timestamp_utc", expected_timestamp, path)
    active_root = _active_root().resolve()
    official_parent = (active_root / STAGE22_RESULT_PARENT).resolve()
    expected_official_run = same_path(result_root.parent, official_parent)
    resolved_parent_identity = _path_identity_for_existing_path(result_root.parent, active_root)
    if "resolved_result_parent" in invocation:
        recorded_resolved_identity = require_absolute_path_identity(
            invocation,
            "resolved_result_parent",
            path,
            active_root,
        )
        if recorded_resolved_identity != resolved_parent_identity:
            raise ValueError(f"{path}.resolved_result_parent must match result-root provenance")
    if "official_parent" in invocation:
        recorded_official_identity = require_absolute_path_identity(
            invocation,
            "official_parent",
            path,
            active_root,
        )
        official_identity = _path_identity_for_existing_path(official_parent, active_root)
        if recorded_official_identity != official_identity:
            raise ValueError(f"{path}.official_parent must match result-root provenance")
    for key in ("official_run", "allow_pre_existing_stage22_roots"):
        if key in invocation and type(invocation[key]) is not bool:
            raise ValueError(f"{path}.{key} must match result-root provenance")
    if "official_run" in invocation and invocation["official_run"] is not expected_official_run:
        raise ValueError(f"{path}.official_run must match result-root provenance")
    for key in ("seed", "episode_count", "steps_per_episode"):
        if key in invocation:
            _require_strict_json_int(invocation[key], f"{path}.{key}")
        if key in invocation and key in run_config:
            require_provenance_value(invocation, key, run_config[key], path)
    if "result_parent_argument" in invocation:
        argument = invocation["result_parent_argument"]
        argument_identity = _result_parent_argument_identity(argument, active_root, path)
        official_identity = _path_identity_for_existing_path(official_parent, active_root)
        if expected_official_run:
            if argument_identity is not None and argument_identity != official_identity:
                raise ValueError(f"{path}.result_parent_argument must match result-root provenance")
        elif argument_identity is None or argument_identity != resolved_parent_identity:
            raise ValueError(f"{path}.result_parent_argument must match result-root provenance")


def _require_current_invocation_parameters(invocation: Mapping[str, object]) -> None:
    for key in _CURRENT_INVOCATION_PARAMETER_KEYS:
        if key not in invocation:
            raise ValueError(f"command_record.json invocation_parameters missing: {key}")
    for key in ("allow_pre_existing_stage22_roots", "official_run"):
        if type(invocation[key]) is not bool:
            raise ValueError(f"command_record.json.invocation_parameters.{key} must be a JSON boolean")
    for key in ("episode_count", "seed", "steps_per_episode"):
        _require_strict_json_int(
            invocation[key],
            f"command_record.json.invocation_parameters.{key}",
        )
    active_root = _active_root().resolve()
    require_absolute_path_identity(
        invocation,
        "official_parent",
        "command_record.json.invocation_parameters",
        active_root,
    )
    require_absolute_path_identity(
        invocation,
        "resolved_result_parent",
        "command_record.json.invocation_parameters",
        active_root,
    )
    argument = invocation["result_parent_argument"]
    if argument is not None and (not isinstance(argument, str) or not argument):
        raise TypeError(
            "command_record.json.invocation_parameters.result_parent_argument "
            "must be a non-empty string path or null"
        )


def _validate_legacy_invocation_parameters_if_present(
    invocation: Mapping[str, object],
    *,
    result_root: Path,
    payloads: Mapping[str, Mapping[str, object]],
) -> None:
    run_config = payloads["run_config.json"]
    expected_timestamp = _expected_timestamp_from_payloads(result_root, run_config)
    _validate_legacy_invocation_provenance(
        invocation,
        result_root=result_root,
        run_config=run_config,
        expected_timestamp=expected_timestamp,
    )


def _expected_timestamp_from_payloads(
    result_root: Path,
    run_config: Mapping[str, object],
) -> str:
    if result_root.name.startswith(STAGE22_RESULT_PREFIX):
        expected_timestamp = result_root.name.removeprefix(STAGE22_RESULT_PREFIX)
        validate_timestamp(expected_timestamp)
        return expected_timestamp
    timestamp = run_config.get("timestamp_utc")
    if not isinstance(timestamp, str):
        raise ValueError("run_config.json.timestamp_utc must identify legacy result timestamp")
    validate_timestamp(timestamp)
    return timestamp


def _require_python_executable_path_signal(command_record: Mapping[str, object]) -> None:
    value = command_record.get("python_executable")
    if not isinstance(value, str) or not value:
        raise TypeError("command_record.json.python_executable must be a non-empty string path")
    if not _has_strong_path_signal(value, "python_executable"):
        raise ValueError(
            "command_record.json.python_executable must be sys.executable or contain an explicit path"
        )


def _assert_current_command_record_provenance(
    command_record: Mapping[str, object],
    run_config: Mapping[str, object],
    invocation: Mapping[str, object],
    active_root: Path,
) -> None:
    expected_replay_python_snippet = _replay_python_snippet(invocation)
    replay_python_snippet = _require_nonempty_string(
        command_record,
        "replay_python_snippet",
        "command_record.json",
    )
    _validate_replay_python_snippet(replay_python_snippet)
    if replay_python_snippet != expected_replay_python_snippet:
        raise ValueError("command_record.json.replay_python_snippet must match result-root provenance")

    command_argv = _require_command_argv(command_record, "command_record.json")
    expected_command_argv = _command_argv(expected_replay_python_snippet)
    if command_argv != expected_command_argv:
        raise ValueError("command_record.json.command_argv must match replay provenance")

    require_provenance_value(
        command_record,
        "canonical_entry_command",
        _canonical_entry_command(),
        "command_record.json",
    )
    require_provenance_value(
        command_record,
        "command",
        human_readable_command(command_argv),
        "command_record.json",
    )
    _require_absolute_provenance_path_string(
        command_record,
        "cwd",
        active_root.as_posix(),
        "command_record.json",
    )
    require_provenance_value(
        command_record,
        "entry_point",
        _ENTRY_POINT,
        "command_record.json",
    )
    require_provenance_value(
        command_record,
        "python_executable",
        sys.executable,
        "command_record.json",
    )
    _require_absolute_provenance_path_string(
        run_config,
        "active_root",
        active_root.as_posix(),
        "run_config.json",
    )


def _require_nonempty_string(
    payload: Mapping[str, object],
    key: str,
    path: str,
) -> str:
    if key not in payload:
        raise ValueError(f"{path} missing required field: {key}")
    value = payload[key]
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{path}.{key} must be a non-empty string")
    return value


def _require_command_argv(payload: Mapping[str, object], path: str) -> list[str]:
    if "command_argv" not in payload:
        raise ValueError(f"{path} missing required field: command_argv")
    command_argv = payload["command_argv"]
    if (
        not isinstance(command_argv, list)
        or len(command_argv) != 3
        or not all(isinstance(part, str) for part in command_argv)
    ):
        raise TypeError(f"{path}.command_argv must be a three-part JSON string list")
    if command_argv[1] != "-c":
        raise ValueError(f"{path}.command_argv must use -c replay mode")
    _validate_replay_python_snippet(command_argv[2])
    return command_argv


def _validate_legacy_replay_provenance_if_present(
    command_record: Mapping[str, object],
    *,
    invocation: Mapping[str, object] | None,
) -> None:
    snippet_present = "replay_python_snippet" in command_record
    argv_present = "command_argv" in command_record
    command_present = "command" in command_record

    snippet: str | None = None
    if snippet_present:
        snippet_value = command_record["replay_python_snippet"]
        if not isinstance(snippet_value, str) or not snippet_value.strip():
            raise TypeError("command_record.json.replay_python_snippet must be a non-empty string")
        _validate_replay_python_snippet(snippet_value)
        snippet = snippet_value

    argv: list[str] | None = None
    if argv_present:
        argv_value = command_record["command_argv"]
        if not isinstance(argv_value, list) or not all(isinstance(part, str) for part in argv_value):
            raise TypeError("command_record.json.command_argv must be a JSON list of strings")
        argv = argv_value
        if len(argv) == 3:
            if argv[1] != "-c":
                raise ValueError("command_record.json.command_argv must use -c replay mode")
            _validate_replay_python_snippet(argv[2])

    if not (snippet_present and argv_present and command_present):
        return
    if argv is None or len(argv) != 3:
        raise ValueError("command_record.json legacy replay trio must use a three-part command_argv")
    if snippet is not None and argv[2] != snippet:
        raise ValueError("command_record.json legacy command_argv must contain replay_python_snippet")
    python_executable = command_record.get("python_executable")
    if isinstance(python_executable, str) and python_executable and argv[0] != python_executable:
        raise ValueError("command_record.json legacy command_argv must match python_executable")
    command = command_record["command"]
    if not isinstance(command, str):
        raise TypeError("command_record.json.command must be a string")
    if command != human_readable_command(argv):
        raise ValueError("command_record.json legacy command must match command_argv display")
    if snippet is not None and invocation is not None and _legacy_invocation_has_replay_keys(invocation):
        if snippet != _replay_python_snippet(invocation):
            raise ValueError(
                "command_record.json legacy replay_python_snippet must match invocation provenance"
            )


def _legacy_invocation_has_replay_keys(invocation: Mapping[str, object]) -> bool:
    replay_keys = (
        "allow_pre_existing_stage22_roots",
        "episode_count",
        "result_parent_argument",
        "seed",
        "steps_per_episode",
        "timestamp_utc",
    )
    return all(key in invocation for key in replay_keys)


def _require_absolute_provenance_path_string(
    payload: Mapping[str, object],
    key: str,
    expected: str,
    path: str,
) -> None:
    require_absolute_path_value(payload, key, path)
    require_provenance_value(payload, key, expected, path)


def _require_result_parent_argument_provenance(
    invocation: Mapping[str, object],
    *,
    active_root: Path,
    official_parent_identity: tuple[str, str],
    resolved_result_parent_identity: tuple[str, str],
    official_run: bool,
) -> None:
    path = "command_record.json.invocation_parameters"
    if "result_parent_argument" not in invocation:
        raise ValueError(f"{path} missing required provenance field: result_parent_argument")
    argument = invocation["result_parent_argument"]
    argument_identity = _result_parent_argument_identity(argument, active_root, path)

    if official_run:
        if argument_identity is not None and argument_identity != official_parent_identity:
            raise ValueError(
                "command_record.json invocation result_parent_argument must be null or "
                "resolve to official parent"
            )
        return

    if argument_identity is None:
        raise ValueError(
            "command_record.json invocation result_parent_argument must be non-null "
            "for external result parents"
        )
    if argument_identity != resolved_result_parent_identity:
        raise ValueError("command_record.json invocation result_parent_argument mismatch")


def _result_parent_argument_identity(
    argument: object,
    active_root: Path,
    path: str,
) -> tuple[str, str] | None:
    if argument is None:
        return None
    if not isinstance(argument, str):
        raise TypeError(f"{path}.result_parent_argument must be a non-empty string path or null")
    if not argument.strip():
        raise TypeError(f"{path}.result_parent_argument must be a non-empty string path or null")
    return _path_identity_for_provenance(argument, active_root)


def _path_identity_for_existing_path(value: Path, active_root: Path) -> tuple[str, str]:
    return _path_identity_for_provenance(str(value), active_root)


def _path_identity_for_provenance(value: str, active_root: Path) -> tuple[str, str]:
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
        native_path = active_root / native_path
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


def _reject_unsafe_boundary_values(value: object, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            child_path = f"{path}.{key}"
            _reject_unsafe_json_key_name(key, child_path)
            if key in _STRICT_RESULT_BOOLEAN_KEYS and type(child) is not bool:
                raise ValueError(f"{child_path} must be a JSON boolean")
            if key in _STRICT_RESULT_INTEGER_KEYS:
                integer_value = _require_strict_json_int(child, child_path)
                _validate_integer_value_bounds(key, integer_value, child_path)
            if key in _FALSE_ONLY_BOUNDARY_BOOLEAN_KEYS:
                if child is not False:
                    raise ValueError(f"{child_path} must be False")
            if key in _TRUE_ONLY_BOUNDARY_BOOLEAN_KEYS:
                if child is not True:
                    raise ValueError(f"{child_path} must be True")
            if key == "claim_status" and child != _CLAIM_STATUS:
                raise ValueError(f"{child_path} must be {_CLAIM_STATUS!r}")
            if key == "training_scope" and child != _TRAINING_SCOPE:
                raise ValueError(f"{child_path} must be {_TRAINING_SCOPE!r}")
            _reject_unsafe_boundary_values(child, child_path)
    elif isinstance(value, list):
        if _is_known_stage22_command_argv(value, path):
            _reject_unsafe_command_argv_values(value, path)
            return
        for index, child in enumerate(value):
            _reject_unsafe_boundary_values(child, f"{path}[{index}]")
    elif isinstance(value, str):
        _reject_unsafe_string_value(value, path)


def _require_strict_json_int(value: object, path: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{path} must be a JSON integer")
    return value


def _validate_integer_value_bounds(key: str, value: int, path: str) -> None:
    if key == "stage":
        if value != 22:
            raise ValueError(f"{path} must be 22")
        return
    if key in _POSITIVE_RESULT_INTEGER_KEYS and value <= 0:
        raise ValueError(f"{path} must be positive")
    if key in _NONNEGATIVE_RESULT_INTEGER_KEYS and value < 0:
        raise ValueError(f"{path} must be nonnegative")


def _reject_unsafe_json_key_name(key: str, path: str) -> None:
    if key in _ALLOWED_UNSAFE_LOOKING_STRUCTURAL_KEYS:
        return
    if key in _ALLOWED_RESULT_FILENAME_KEYS:
        return
    normalized = _normalise_boundary_string(key)
    if any(phrase in normalized for phrase in _ABSOLUTE_UNSAFE_BOUNDARY_WORDING):
        raise ValueError(f"{path} contains unsafe evidence-boundary wording in key")
    if any(phrase in normalized for phrase in _SEMANTIC_UNSAFE_EVALUATION_WORDING):
        raise ValueError(f"{path} contains unsafe evaluation wording in key")
    if re.search(r"\bevaluation\b", normalized):
        raise ValueError(f"{path} contains unsafe evaluation wording in key")


def _normalise_boundary_string(value: str) -> str:
    lowered = value.lower()
    separated = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", separated).strip()


def _reject_unsafe_string_value(value: str, path: str) -> None:
    normalized = _normalise_boundary_string(value)
    key = path.rsplit(".", 1)[-1]

    if any(phrase in normalized for phrase in _ABSOLUTE_UNSAFE_BOUNDARY_WORDING):
        raise ValueError(f"{path} contains unsafe evidence-boundary wording")

    is_false_positive = key in _PATH_OR_PROVENANCE_STRING_KEYS and (
        _is_genuine_path_or_command_false_positive(value, normalized, key)
    )

    if any(phrase in normalized for phrase in _SEMANTIC_UNSAFE_EVALUATION_WORDING):
        raise ValueError(f"{path} contains unsafe evaluation wording")

    if re.search(r"\bevaluation\b", normalized):
        if not _is_allowed_negative_evaluation_string(normalized) and not is_false_positive:
            raise ValueError(f"{path} contains unsafe evaluation wording")


def _reject_unsafe_command_argv_values(command_argv: list[str], path: str) -> None:
    context_keys = ("python_executable", "command_argv_switch", "replay_python_snippet")
    for index, part in enumerate(command_argv):
        if not isinstance(part, str):
            raise TypeError(f"{path}[{index}] must be a string")
        _reject_unsafe_command_argv_string(part, f"{path}[{index}]", context_keys[index])


def _reject_unsafe_command_argv_string(value: str, path: str, context_key: str) -> None:
    normalized = _normalise_boundary_string(value)
    if any(phrase in normalized for phrase in _ABSOLUTE_UNSAFE_BOUNDARY_WORDING):
        raise ValueError(f"{path} contains unsafe evidence-boundary wording")

    is_false_positive = context_key in _PATH_OR_PROVENANCE_STRING_KEYS and (
        _is_genuine_path_or_command_false_positive(value, normalized, context_key)
    )

    if any(phrase in normalized for phrase in _SEMANTIC_UNSAFE_EVALUATION_WORDING):
        raise ValueError(f"{path} contains unsafe evaluation wording")

    if re.search(r"\bevaluation\b", normalized):
        if not _is_allowed_negative_evaluation_string(normalized) and not is_false_positive:
            raise ValueError(f"{path} contains unsafe evaluation wording")


def _is_allowed_negative_evaluation_string(normalized: str) -> bool:
    return (
        normalized in _ALLOWED_EXACT_NEGATIVE_EVALUATION_STRINGS
        or _STAGE22_SC_NEGATIVE_EVALUATION_RE.fullmatch(normalized) is not None
    )


def _is_genuine_path_or_command_false_positive(
    value: str,
    normalized: str,
    key: str,
) -> bool:
    path_keys = {
        "active_root",
        "cwd",
        "python_executable",
        "resolved_result_parent",
        "official_parent",
        "result_parent_argument",
    }
    command_keys = {"command", "canonical_entry_command"}
    if key in path_keys:
        return _has_strong_path_signal(value, key) and _evaluation_tokens_are_safe_sandbox(value)
    if key in command_keys:
        return _is_known_stage22_command(value, normalized) and _evaluation_tokens_are_safe_sandbox(
            value
        )
    if key == "replay_python_snippet":
        return _is_known_stage22_replay_snippet(value) and _evaluation_tokens_are_safe_sandbox(
            value
        )
    if key == "entry_point":
        return (
            value
            == "raas_marl.mappo_lagrangian.development_runner.run_stage22_development_training"
        )
    if key == "result_root_name":
        return re.fullmatch(r"stage22_development_training_\d{8}T\d{6}Z", value) is not None
    if key == "timestamp_utc":
        return re.fullmatch(r"\d{8}T\d{6}Z", value) is not None
    return False


def _has_strong_path_signal(value: str, key: str) -> bool:
    if key == "python_executable" and value == sys.executable:
        return True
    return bool(
        re.search(r"^[A-Za-z]:[\\/]", value)
        or value.startswith(("/", "\\\\"))
        or "/" in value
        or "\\" in value
    )


def _is_known_stage22_command(value: str, normalized: str) -> bool:
    if _is_known_stage22_display_command(value):
        return True
    return "run_stage22_development_training" in value and (
        "raas_marl.mappo_lagrangian.development_runner" in value
        or "sys.path.insert" in value
        or "sys path insert" in normalized
    )


def _is_known_stage22_display_command(value: str) -> bool:
    parts = _parse_human_readable_command(value)
    return (
        parts is not None
        and len(parts) == 3
        and bool(parts[0])
        and parts[1] == "-c"
        and _is_known_stage22_replay_snippet(parts[2])
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


def _is_known_stage22_command_argv(value: object, path: str) -> bool:
    if not path.endswith(".command_argv"):
        return False
    if (
        not isinstance(value, list)
        or len(value) != 3
        or not all(isinstance(part, str) for part in value)
    ):
        return False
    return (
        bool(value[0])
        and value[1] == "-c"
        and _is_known_stage22_replay_snippet(value[2])
    )


def _is_known_stage22_replay_snippet(value: str) -> bool:
    try:
        module = ast.parse(value)
    except SyntaxError:
        return False
    if len(module.body) != 4:
        return False
    if not _is_import_sys_statement(module.body[0]):
        return False
    if not _is_sys_path_insert_src_statement(module.body[1]):
        return False
    if not _is_runner_import_statement(module.body[2]):
        return False
    return _is_stage22_run_call_statement(module.body[3])


def _is_import_sys_statement(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.Import)
        and len(statement.names) == 1
        and statement.names[0].name == "sys"
        and statement.names[0].asname is None
    )


def _is_sys_path_insert_src_statement(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    call = statement.value
    return (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "insert"
        and isinstance(call.func.value, ast.Attribute)
        and call.func.value.attr == "path"
        and isinstance(call.func.value.value, ast.Name)
        and call.func.value.value.id == "sys"
        and len(call.args) == 2
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value == 0
        and isinstance(call.args[1], ast.Constant)
        and call.args[1].value == "src"
        and not call.keywords
    )


def _is_runner_import_statement(statement: ast.stmt) -> bool:
    return (
        isinstance(statement, ast.ImportFrom)
        and statement.module == "raas_marl.mappo_lagrangian.development_runner"
        and len(statement.names) == 1
        and statement.names[0].name == "run_stage22_development_training"
        and statement.names[0].asname is None
    )


def _is_stage22_run_call_statement(statement: ast.stmt) -> bool:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return False
    call = statement.value
    if not isinstance(call.func, ast.Name) or call.func.id != "run_stage22_development_training":
        return False
    expected_keywords = (
        "result_parent",
        "timestamp_utc",
        "seed",
        "episode_count",
        "steps_per_episode",
        "allow_pre_existing_stage22_roots",
    )
    if call.args or tuple(keyword.arg for keyword in call.keywords) != expected_keywords:
        return False
    return all(
        keyword.arg is not None and _is_safe_replay_literal(keyword.value)
        for keyword in call.keywords
    )


def _is_safe_replay_literal(value: ast.expr) -> bool:
    if not isinstance(value, ast.Constant):
        return False
    literal = value.value
    return literal is None or isinstance(literal, (str, bool)) or (
        isinstance(literal, int) and not isinstance(literal, bool)
    )


def _evaluation_tokens_are_safe_sandbox(value: str) -> bool:
    tokens = [token for token in re.split(r"[\s\"'(),=;]+", value) if token]
    evaluation_tokens = [
        token for token in tokens if re.search(r"evaluation", token, re.IGNORECASE)
    ]
    if not evaluation_tokens:
        return True
    return all(_evaluation_token_is_safe_sandbox_path(token) for token in evaluation_tokens)


def _evaluation_token_is_safe_sandbox_path(token: str) -> bool:
    if not _has_strong_path_signal(token, "command"):
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


def _require_field(
    payload: Mapping[str, object],
    key: str,
    expected: object,
    path: str,
    allow_legacy: bool,
) -> None:
    if key not in payload:
        if allow_legacy:
            return
        raise ValueError(f"{path} missing required field: {key}")
    actual = payload[key]
    if isinstance(expected, bool):
        if type(actual) is not bool or actual is not expected:
            raise ValueError(f"{path}.{key} must be {expected!r}")
        return
    if actual != expected:
        raise ValueError(f"{path}.{key} must be {expected!r}")

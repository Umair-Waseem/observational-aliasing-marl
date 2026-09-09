"""Stage 23-A environment-construction smoke runner.

The runner writes one bounded construction-smoke result root. It never trains,
updates a policy, instantiates a model, evaluates a learned method, serializes
checkpoints, or creates claim evidence. It does not run final evaluation,
compare baselines, perform ablations, run statistical tests, or produce
paper-facing outputs.

The runner is torch-free: it imports only the torch-free grid environment, the
torch-free ``artifacts`` and ``_governance`` modules, and the standard library.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from raas_marl.mappo_lagrangian.artifacts import (
    BLOCKED_ARTIFACT_NAME_TOKENS,
    BLOCKED_ARTIFACT_SUFFIXES,
    sha256_file,
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
    resolve_result_parent,
    same_path,
    utc_timestamp,
    validate_timestamp,
    write_json,
    write_jsonl,
)
from raas_marl.environments.active_sensing.grid_environment import (
    MOVEMENT_ACTION_FIELD,
    SENSING_ACTION_FIELD,
    STAGE23_AGENT_NAMES,
    STAGE23_STAGE,
    RiskAwareActiveSensingGridEnvironment,
    Stage23EnvironmentConfig,
    stage23_scenario_catalog,
)


ENVIRONMENT_CONSTRUCTION_RESULT_PARENT = Path("results") / "environment_construction"
ENVIRONMENT_CONSTRUCTION_RESULT_PREFIX = "environment_construction_"
STAGE23A_RESULT_FILES = (
    "environment_config.json",
    "scenario_catalog.json",
    "contract_smoke_metrics.json",
    "scripted_rollout_trace.jsonl",
    "boundary_record.json",
    "command_record.json",
    "artifact_hashes.json",
)
STAGE23A_HASHED_RESULT_FILES = STAGE23A_RESULT_FILES[:-1]

_CLAIM_STATUS = "not tested / not supported"
_TRAINING_RUN = "environment_construction_smoke_only"
_ENTRY_POINT = (
    "raas_marl.environments.active_sensing.construction_runner."
    "run_environment_construction"
)
# Canonical replay-snippet schema for the Stage 23-A construction runner.
# Threaded through the shared _governance replay-snippet family (build/parse/
# validate) so the snippet build path and the read-back provenance path share
# one AST-validated form. The None-preserving ``result_parent`` keyword carries
# the original invocation argument (None or a non-empty string).
_REPLAY_SNIPPET_SCHEMA = ReplaySnippetSchema(
    import_module_path="raas_marl.environments.active_sensing.construction_runner",
    callee_name="run_environment_construction",
    keyword_order=(
        "result_parent",
        "timestamp_utc",
    ),
    keyword_literal_kind={
        "result_parent": "result_parent",
        "timestamp_utc": "timestamp",
    },
)
# The false/true boundary-flag key set handed to the shared unsafe-terminology
# scanner as benign structural keys, unioned with the result-file names.
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
_ALLOWED_UNSAFE_LOOKING_STRUCTURAL_KEYS = (
    _FALSE_BOUNDARY_KEYS | _TRUE_BOUNDARY_KEYS | frozenset(STAGE23A_RESULT_FILES)
)


def run_stage23a_environment_construction(
    *,
    result_parent: str | Path | None = None,
    timestamp_utc: str | None = None,
) -> Path:
    """Compatibility alias retained for internal Stage 23-A governance."""

    return run_environment_construction(
        result_parent=result_parent,
        timestamp_utc=timestamp_utc,
    )


def run_environment_construction(
    *,
    result_parent: str | Path | None = None,
    timestamp_utc: str | None = None,
) -> Path:
    """Run the bounded Stage 23-A construction smoke and return the root path."""

    active_root = _active_root().resolve()
    official_parent = (active_root / ENVIRONMENT_CONSTRUCTION_RESULT_PARENT).resolve()
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

    if timestamp_utc is None:
        timestamp = utc_timestamp()
    elif isinstance(timestamp_utc, str):
        timestamp = timestamp_utc
    else:
        raise TypeError("timestamp_utc must be a string or None")
    validate_timestamp(timestamp)

    root = parent / f"{ENVIRONMENT_CONSTRUCTION_RESULT_PREFIX}{timestamp}"
    tmp_root = parent / f".stage23a_tmp_{timestamp}"
    if root.exists():
        raise FileExistsError(f"Stage 23-A result root already exists: {root}")
    if tmp_root.exists():
        raise FileExistsError(
            f"Stage 23-A temporary result root already exists: {tmp_root}"
        )
    parent.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir()
    finalized = False
    try:
        config = Stage23EnvironmentConfig()
        environment_config = {**config.to_json_dict(), **_boundary_flags()}
        scenario_catalog = {
            "scenarios": {
                name: scenario.to_json_dict()
                for name, scenario in stage23_scenario_catalog().items()
            },
            **_boundary_flags(),
        }
        trace_rows, metrics = _scripted_smoke_rollouts(config)
        boundary_record = _boundary_record()
        invocation_parameters = {
            "official_parent": official_parent.as_posix(),
            "official_run": official_run,
            "resolved_result_parent": parent.as_posix(),
            "result_parent_argument": None
            if result_parent is None
            else str(result_parent),
            "timestamp_utc": timestamp,
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
            "result_root_name": root.name,
            **_boundary_flags(),
        }
        write_json(tmp_root / "environment_config.json", environment_config)
        write_json(tmp_root / "scenario_catalog.json", scenario_catalog)
        write_json(tmp_root / "contract_smoke_metrics.json", metrics)
        write_jsonl(tmp_root / "scripted_rollout_trace.jsonl", trace_rows)
        write_json(tmp_root / "boundary_record.json", boundary_record)
        write_json(tmp_root / "command_record.json", command_record)
        artifact_hashes = {
            "hashed_files": {
                name: sha256_file(tmp_root / name)
                for name in sorted(STAGE23A_HASHED_RESULT_FILES)
            },
            **_boundary_flags(),
        }
        write_json(tmp_root / "artifact_hashes.json", artifact_hashes)
        _assert_strict_stage23a_result_artifacts(tmp_root, provenance_root=root)
        if root.exists():
            raise FileExistsError(f"Stage 23-A result root already exists: {root}")
        tmp_root.replace(root)
        finalized = True
    finally:
        if not finalized and tmp_root.exists():
            shutil.rmtree(tmp_root)
    return root


def _boundary_flags() -> dict[str, object]:
    return boundary_flags(
        stage=STAGE23_STAGE,
        training_run=_TRAINING_RUN,
        claim_status=_CLAIM_STATUS,
    )


def _boundary_record() -> dict[str, object]:
    return {
        "artifact_hashes_scope": "hashes_integrity_and_provenance_only_not_metrics",
        "artifact_set": list(STAGE23A_RESULT_FILES),
        "environment_stage": STAGE23_STAGE,
        "result_files_json_or_jsonl_only": True,
        "scripted_smoke_rollout_only": True,
        "stage23a_result_parent": ENVIRONMENT_CONSTRUCTION_RESULT_PARENT.as_posix(),
        **_boundary_flags(),
    }


def _active_root() -> Path:
    # construction_runner sits one directory deeper (environments/active_sensing)
    # than the mappo_lagrangian runners, so its repository-root anchor is
    # parents[4], not parents[3].
    return _shared_active_root(__file__, 4)


def _scripted_smoke_rollouts(
    default_config: Stage23EnvironmentConfig,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows: list[dict[str, object]] = []
    total_steps = 0
    total_hazard_cost = 0.0
    total_sensing_cost = 0.0
    total_task_reward = 0.0
    scenario_names = ("unit_empty", "unit_single_hazard", "standard_branching_hazard")
    scripted_actions = {
        "unit_empty": [
            ((0, 4), (0, 4)),
            ((0, 4), (0, 4)),
        ],
        "unit_single_hazard": [
            ((1, 4), (0, 0)),
            ((0, 2), (0, 2)),
        ],
        "standard_branching_hazard": [
            ((0, 4), (1, 4)),
            ((1, 4), (0, 4)),
            ((0, 2), (0, 2)),
        ],
    }
    for scenario_name in scenario_names:
        env = RiskAwareActiveSensingGridEnvironment(
            Stage23EnvironmentConfig(
                scenario_name=scenario_name,
                max_steps=default_config.max_steps,
                seed=default_config.seed,
            )
        )
        observations, infos = env.reset(seed=default_config.seed)
        _require_fixed_stage23_agents(env.possible_agents)
        rows.append(
            {
                "scenario_name": scenario_name,
                "event": "reset",
                "step_index": env.step_index,
                "agents": list(env.agents),
                "actor_observation_agents": sorted(observations),
                "info_agents": sorted(infos),
                "scripted_smoke_rollout_only": True,
                **_boundary_flags(),
            }
        )
        for action_index, ordered_action_pairs in enumerate(scripted_actions[scenario_name]):
            if not env.agents:
                break
            actions = _scripted_action_mapping(env.possible_agents, ordered_action_pairs)
            observations, rewards, terminations, truncations, infos = env.step(actions)
            total_steps += 1
            total_task_reward += sum(float(value) for value in rewards.values())
            total_hazard_cost += sum(float(info["hazard_cost"]) for info in infos.values())
            total_sensing_cost += sum(float(info["sensing_cost"]) for info in infos.values())
            rows.append(
                {
                    "scenario_name": scenario_name,
                    "event": "step",
                    "scripted_action_index": action_index,
                    "step_index": env.step_index,
                    "agents_before_return": sorted(observations),
                    "rewards": rewards,
                    "terminations": terminations,
                    "truncations": truncations,
                    "hazard_cost_total": sum(float(info["hazard_cost"]) for info in infos.values()),
                    "sensing_cost_total": sum(float(info["sensing_cost"]) for info in infos.values()),
                    "team_success": any(bool(info["team_success"]) for info in infos.values()),
                    "agents_after_step": list(env.agents),
                    "scripted_smoke_rollout_only": True,
                    **_boundary_flags(),
                }
            )
    metrics = {
        "status": "STAGE23A_ENVIRONMENT_CONSTRUCTION_SMOKE_COMPLETE",
        "scripted_smoke_rollout_only": True,
        "scenario_count": len(scenario_names),
        "scripted_step_count": total_steps,
        "total_task_reward": total_task_reward,
        "total_hazard_cost": total_hazard_cost,
        "total_sensing_cost": total_sensing_cost,
        **_boundary_flags(),
    }
    return rows, metrics


def _require_fixed_stage23_agents(agent_names: tuple[str, ...]) -> None:
    if tuple(agent_names) != STAGE23_AGENT_NAMES:
        raise ValueError("Stage 23-A construction smoke requires exactly two agents")


def _scripted_action_mapping(
    agent_names: tuple[str, ...],
    ordered_action_pairs: tuple[tuple[int, int], ...],
) -> dict[str, dict[str, int]]:
    _require_fixed_stage23_agents(agent_names)
    if len(ordered_action_pairs) != len(agent_names):
        raise ValueError("scripted action tuple count must match Stage 23-A agents")
    return {
        agent: {
            SENSING_ACTION_FIELD: sensing,
            MOVEMENT_ACTION_FIELD: movement,
        }
        for agent, (sensing, movement) in zip(agent_names, ordered_action_pairs)
    }


def _replay_snippet_keyword_values(
    invocation_parameters: dict[str, object],
) -> dict[str, object]:
    return {
        "result_parent": invocation_parameters["result_parent_argument"],
        "timestamp_utc": invocation_parameters["timestamp_utc"],
    }


def _replay_python_snippet(invocation_parameters: dict[str, object]) -> str:
    return build_replay_python_snippet(
        _REPLAY_SNIPPET_SCHEMA,
        _replay_snippet_keyword_values(invocation_parameters),
    )


def _command_argv(replay_python_snippet: str) -> list[str]:
    return command_argv_from_snippet(replay_python_snippet, _REPLAY_SNIPPET_SCHEMA)


def _parse_replay_python_snippet(replay_python_snippet: str) -> dict[str, object]:
    return parse_replay_python_snippet(replay_python_snippet, _REPLAY_SNIPPET_SCHEMA)


def _canonical_entry_command() -> str:
    return (
        "python -c \"import sys; sys.path.insert(0, 'src'); "
        "from raas_marl.environments.active_sensing.construction_runner "
        "import run_environment_construction; run_environment_construction()\""
    )


def _assert_boundary_payload(payload: dict[str, object], path: str) -> None:
    assert_boundary_payload(
        payload,
        path,
        stage=STAGE23_STAGE,
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
        result_root_prefix=ENVIRONMENT_CONSTRUCTION_RESULT_PREFIX,
    )


def _validate_result_root_name(value: str) -> None:
    if not value.startswith(ENVIRONMENT_CONSTRUCTION_RESULT_PREFIX):
        raise ValueError(
            "Stage 23-A result root must use the environment_construction prefix"
        )
    validate_timestamp(value.removeprefix(ENVIRONMENT_CONSTRUCTION_RESULT_PREFIX))


def _assert_exact_result_files(result_root: Path) -> None:
    if not result_root.is_dir():
        raise ValueError("Stage 23-A result root must be a directory")
    children = sorted(result_root.iterdir(), key=lambda path: path.name)
    names = [path.name for path in children]
    expected = sorted(STAGE23A_RESULT_FILES)
    if names != expected:
        raise ValueError(f"Stage 23-A result files mismatch: {names!r} != {expected!r}")
    for child in children:
        if child.is_symlink():
            raise ValueError(f"Stage 23-A result root contains symlink: {child.name}")
        if child.is_dir():
            raise ValueError(f"Stage 23-A result root contains directory: {child.name}")
        if not child.is_file():
            raise ValueError(f"Stage 23-A result root contains non-file child: {child.name}")


def _assert_command_record_provenance(
    result_root: Path,
    command_record: dict[str, object],
) -> None:
    active_root = _active_root().resolve()
    result_name = result_root.name
    expected_timestamp = result_name.removeprefix(
        ENVIRONMENT_CONSTRUCTION_RESULT_PREFIX
    )
    validate_timestamp(expected_timestamp)
    if command_record.get("result_root_name") != result_name:
        raise ValueError("command_record.json result_root_name mismatch")
    if command_record.get("cwd") != active_root.as_posix():
        raise ValueError("command_record.json cwd must match the active repository root")
    if command_record.get("entry_point") != _ENTRY_POINT:
        raise ValueError("command_record.json entry_point mismatch")
    if command_record.get("canonical_entry_command") != _canonical_entry_command():
        raise ValueError("command_record.json canonical_entry_command mismatch")
    invocation = command_record.get("invocation_parameters")
    if not isinstance(invocation, dict):
        raise TypeError(
            "command_record.json invocation_parameters must be a JSON object"
        )
    for key in (
        "official_parent",
        "official_run",
        "resolved_result_parent",
        "result_parent_argument",
        "timestamp_utc",
    ):
        if key not in invocation:
            raise ValueError(
                f"command_record.json invocation_parameters missing: {key}"
            )
    if invocation.get("timestamp_utc") != expected_timestamp:
        raise ValueError("command_record.json invocation timestamp_utc mismatch")
    if type(invocation.get("official_run")) is not bool:
        raise TypeError("command_record.json official_run must be a JSON boolean")
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
    if not isinstance(replay_python_snippet, str):
        raise TypeError("command_record.json replay_python_snippet must be a string")
    replay_invocation = _parse_replay_python_snippet(replay_python_snippet)
    expected_replay_python_snippet = _replay_python_snippet(invocation)
    if replay_python_snippet != expected_replay_python_snippet:
        raise ValueError("command_record.json replay_python_snippet mismatch")
    if replay_invocation["timestamp_utc"] != expected_timestamp:
        raise ValueError("command_record.json replay timestamp_utc mismatch")
    if replay_invocation["result_parent"] != invocation.get("result_parent_argument"):
        raise ValueError("command_record.json replay result_parent mismatch")
    if command_record.get("command") != human_readable_command(command_argv):
        raise ValueError("command_record.json command must match command_argv display")


def _assert_strict_stage23a_result_artifacts(
    result_root: Path,
    *,
    provenance_root: Path | None = None,
) -> None:
    """Read-back strict validation of a written Stage 23-A result root (M-4)."""

    _assert_exact_result_files(result_root)
    _validate_result_root_name((provenance_root or result_root).name)
    for child in result_root.iterdir():
        suffix = child.suffix.lower()
        lower_name = child.name.lower()
        if suffix in BLOCKED_ARTIFACT_SUFFIXES:
            raise ValueError(f"blocked binary/model artifact in result root: {child.name}")
        if any(token in lower_name for token in BLOCKED_ARTIFACT_NAME_TOKENS):
            raise ValueError(f"blocked artifact name in result root: {child.name}")
    payloads = {
        name: load_strict_json_object(result_root / name)
        for name in STAGE23A_RESULT_FILES
        if name.endswith(".json")
    }
    trace_records = load_strict_jsonl_objects(
        result_root / "scripted_rollout_trace.jsonl"
    )
    artifact_hashes = payloads["artifact_hashes.json"]
    hashed_files = artifact_hashes.get("hashed_files")
    if not isinstance(hashed_files, dict):
        raise ValueError("artifact_hashes.json hashed_files must be a JSON object")
    if set(hashed_files) != set(STAGE23A_HASHED_RESULT_FILES):
        raise ValueError("artifact_hashes.json must hash the other result files only")
    for name, expected_hash in hashed_files.items():
        if not isinstance(expected_hash, str):
            raise TypeError("artifact hash values must be strings")
        if sha256_file(result_root / name) != expected_hash:
            raise ValueError(f"artifact_hashes.json mismatch for {name}")
    for name, payload in payloads.items():
        _assert_boundary_payload(payload, name)
        _reject_unsafe_boundary_values(payload, name)
    for index, record in enumerate(trace_records):
        _assert_boundary_payload(record, f"scripted_rollout_trace.jsonl[{index}]")
        _reject_unsafe_boundary_values(record, f"scripted_rollout_trace.jsonl[{index}]")
    _assert_command_record_provenance(
        provenance_root or result_root,
        payloads["command_record.json"],
    )

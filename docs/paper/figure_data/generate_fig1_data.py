"""P1 Unit 3(b): reproduce the (3,3) observational-aliasing contradiction FRESH from source.

FIGURE-1 DATA. Precise, corrected characterization (Unit 3(d) mismatch-rule reconciliation):

The aliasing is a property of the ORIGINAL radius-1 dim-10 encoding. At cell (3,3):
 - features 0..8 (position, step, goal-delta, previous-* flags) are radius-INDEPENDENT and
   byte-identical between risk_fork_train_upper ({3,4}) and risk_fork_curriculum_r0r3_lower ({0,3});
 - the ORIGINAL feature 9 is the obstacle count over the radius-1 window; at (3,3) the only col-4
   cell in that window is (3,4), an OPEN gate in BOTH scenarios, so the radius-1 count = 0 for both.
 => the ORIGINAL dim-10 observation at (3,3) is byte-identical for the two scenarios, yet they
    require OPPOSITE reroutes (anchor DOWN to safe gate (4,4); {0,3} UP to safe gate (0,4)).
    No dim-10 policy can satisfy both -> the representability contradiction.

The enriched dim-22 radius-2 encoding RESOLVES it: cell (4,4) (offset (1,1), patch index 20)
reads OPEN=0.0 for the anchor and WALL=1.0 for {0,3}. The wider window also changes the (now
radius-2) count feature 9 (1/13 vs 2/13 -- the extra wall IS (4,4)). The current code only emits
the dim-22 encoding; the radius-1 count is reconstructed directly from the obstacle geometry."""
import json
import sys
from pathlib import Path

# PHASE 3: resolve the repository root from THIS FILE, not from the current working
# directory. `sys.path.insert(0, "src")` made the script depend on being launched from
# the repository root: run from anywhere else it either failed outright or -- worse --
# picked up whatever tree happened to have a src/ next to the cwd. Same defect class as
# the three hardcoded instrument paths, so it gets the same walk-up.
_REPO_ROOT = Path(__file__).resolve()
while _REPO_ROOT.parent != _REPO_ROOT and not (_REPO_ROOT / "src" / "raas_marl").is_dir():
    _REPO_ROOT = _REPO_ROOT.parent
if not (_REPO_ROOT / "src" / "raas_marl").is_dir():
    raise RuntimeError(
        "could not locate the repository root: no ancestor directory of "
        f"{Path(__file__).resolve()} contains 'src/raas_marl'. Run this generator "
        "from inside a checkout of the artifact repository."
    )
sys.path.insert(0, str(_REPO_ROOT / "src"))
# The bootstrap above must run before these imports resolve, exactly as in the
# grading instruments; same # noqa: E402 convention as docs/evidence/*.py.
import torch  # noqa: E402
from raas_marl.environments.active_sensing.grid_environment import (  # noqa: E402
    RiskAwareActiveSensingGridEnvironment, Stage23EnvironmentConfig, stage23_scenario_catalog)
from raas_marl.environments.active_sensing import tensor_adapter as ta  # noqa: E402

CELL = (3, 3)
ANCHOR, OTHER = "risk_fork_train_upper", "risk_fork_curriculum_r0r3_lower"


def env_for(name):
    sc = stage23_scenario_catalog()[name]
    env = RiskAwareActiveSensingGridEnvironment(Stage23EnvironmentConfig(
        scenario_name=sc.name, width=sc.width, height=sc.height, seed=7))
    env.reset()
    return env


def obs_at(env, cell):
    env._positions["agent_0"] = tuple(cell)
    obs = env._observation("agent_0")
    return ta.actor_observation_from_stage23(
        obs, local_observation_radius=env._config.local_observation_radius).reshape(-1)


def radius1_obstacle_count(scenario_name, cell):
    """Original dim-10 feature 9 = count of obstacles in the 4 Manhattan-1 neighbours."""
    obstacles = set(stage23_scenario_catalog()[scenario_name].obstacles)
    r, c = cell
    nbrs = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return sum(1 for n in nbrs if n in obstacles and 0 <= n[0] < 8 and 0 <= n[1] < 8)


anchor = obs_at(env_for(ANCHOR), CELL)
other = obs_at(env_for(OTHER), CELL)
legacy = ta.STAGE23_LEGACY_ACTOR_FEATURE_COUNT
offsets = list(ta._RADIUS2_PATCH_OFFSETS)
idx44 = legacy + offsets.index((1, 1))  # radius-2 patch cell (4,4)
dim = int(anchor.shape[0])

# radius-INDEPENDENT features 0..8 identical?
head_identical = torch.equal(anchor[:legacy - 1], other[:legacy - 1])
# radius-1 count (original feature 9) identical?
c_anchor = radius1_obstacle_count(ANCHOR, CELL)
c_other = radius1_obstacle_count(OTHER, CELL)
diff_full = [i for i in range(dim) if float(anchor[i]) != float(other[i])]

# hazard / safe-gate geometry (privileged, for the caption only -- never in the actor obs)
geom = {}
cat = stage23_scenario_catalog()
for nm in (ANCHOR, OTHER):
    sc = cat[nm]
    env = env_for(nm)
    cvs = env.critic_visible_state()
    hz = cvs.get("hazard_cells") or cvs.get("hidden_hazard_cells") or ()
    col4_walls = sorted(o[0] for o in sc.obstacles if o[1] == 4)
    gates = [r for r in range(sc.height) if r not in col4_walls]
    geom[nm] = {"gate_rows": gates, "hidden_hazard_cells": [list(h) for h in hz],
                "safe_gate_col4": [[r, 4] for r in gates if [r, 4] not in [list(h) for h in hz]]}

out = {
    "cell_(3,3)": list(CELL), "anchor_scenario": ANCHOR, "other_scenario": OTHER, "dim": dim,
    "legacy_feature_count": legacy, "radius2_patch_index_for_cell_(4,4)": idx44,
    "features_0to8_radius_independent_byte_identical": bool(head_identical),
    "original_radius1_obstacle_count_feature9": {ANCHOR: c_anchor, OTHER: c_other},
    "original_radius1_count_identical": c_anchor == c_other,
    "ORIGINAL_dim10_byte_identical_at_(3,3)": bool(head_identical) and (c_anchor == c_other),
    "enriched_obs_dim22": {ANCHOR: [round(float(x), 6) for x in anchor],
                           OTHER: [round(float(x), 6) for x in other]},
    "enriched_features_0to8": [round(float(x), 6) for x in anchor[:legacy - 1]],
    "enriched_radius2_count_feature9": {ANCHOR: round(float(anchor[9]), 6),
                                        OTHER: round(float(other[9]), 6)},
    "enriched_differing_indices": diff_full,
    "patch_cell_(4,4)_value": {ANCHOR + "_OPEN": float(anchor[idx44]),
                               OTHER + "_WALL": float(other[idx44])},
    "reroute_geometry": geom,
}
# PHASE 3: the output path was relative to the cwd and named a directory that does not
# exist in the artifact, in text mode with no encoding -- it failed on BOTH platforms.
_OUT = _REPO_ROOT / "docs" / "paper" / "figure_data" / "fig1_aliasing.json"
_OUT.parent.mkdir(parents=True, exist_ok=True)
with _OUT.open("w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
print(json.dumps(out, indent=2))
print("\n=== FIGURE-1 HEADLINE (fresh from source) ===")
print("features 0..8 (radius-independent) byte-identical:", bool(head_identical))
print("original radius-1 count feature 9:", c_anchor, "vs", c_other, "-> identical:", c_anchor == c_other)
print("=> ORIGINAL dim-10 obs at (3,3) byte-identical:", bool(head_identical) and c_anchor == c_other)
print("enriched dim-22 differs at indices:", diff_full)
print("patch (4,4): anchor=%.1f (OPEN)  {0,3}=%.1f (WALL)" % (float(anchor[idx44]), float(other[idx44])))
assert head_identical and c_anchor == c_other, "MISMATCH: original dim-10 not byte-identical"
assert diff_full == [9, idx44], "MISMATCH: enriched differs at %s" % diff_full
assert float(anchor[idx44]) == 0.0 and float(other[idx44]) == 1.0, "MISMATCH: patch values"
print("ALL FIGURE-1 CHECKS PASS")

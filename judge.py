"""
Evaluation script for LevDoom levels.

Usage:
    python eval.py [--student-path PATH] [--output PATH]

Loads the student's agent from student_agent.py and evaluates it across all
configured LevDoom levels, writing results to results.json.
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import levdoom


# ---------------------------------------------------------------------------
# Actor interface — students must implement this
# ---------------------------------------------------------------------------

class Actor:
    def __init__(self, action_space):
        self.action_space = action_space

    def act(self, obs: np.ndarray) -> int:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Student agent loader
# ---------------------------------------------------------------------------

def load_student_agent(student_path: str) -> callable:
    """
    Dynamically import student_agent.py and return an actor_factory.

    The student module must expose either:
      - make_agent(action_space) -> Actor   (preferred)
      - StudentAgent(action_space)          (class with act(obs) -> int)
    """
    agent_file = Path(student_path) / "student_agent.py"
    if not agent_file.exists():
        raise FileNotFoundError(f"student_agent.py not found at {agent_file}")

    # Allow the student module to import its own helpers
    sys.path.insert(0, str(Path(student_path).resolve()))

    spec = importlib.util.spec_from_file_location("student_agent", agent_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "make_agent"):
        factory = module.make_agent
    elif hasattr(module, "StudentAgent"):
        factory = lambda action_space: module.StudentAgent(action_space)
    else:
        raise ImportError(
            "student_agent.py must define make_agent(action_space) "
            "or a StudentAgent class"
        )

    # Sanity-check: instantiate once with a dummy action space to catch
    # obvious import/init errors early, before the full evaluation loop.
    try:
        dummy_env = levdoom.make(LEVELS[0]["id"])
        agent = factory(dummy_env.action_space)
        dummy_env.close()
    except Exception as exc:
        raise RuntimeError(f"Failed to instantiate student agent: {exc}") from exc

    if not callable(getattr(agent, "act", None)):
        raise TypeError("Student agent must implement act(obs) -> int")

    return factory


# ---------------------------------------------------------------------------
# Results serialiser
# ---------------------------------------------------------------------------

def save_results(results: list[dict], output_path: str) -> None:
    """Write evaluation results to a JSON file for the leaderboard step."""
    payload = {}
    for r in results:
        payload[r["level"]] = {
            "mean_kills":     r["mean_kills"],
            "kills_per_seed": r["kills_per_seed"],
        }
    with open(output_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results written to {output_path}")


# ---------------------------------------------------------------------------
# Evaluation config
# ---------------------------------------------------------------------------

LEVELS = [
    {"id": "SeekAndSlayLevel0-v0",  "map": "default",                    "threshold": 22},
    {"id": "SeekAndSlayLevel1_6-v0","map": "mixed_enemies",               "threshold": 15},
    {"id": "SeekAndSlayLevel2_1-v0","map": "blue_shadows",                "threshold": 9},
    {"id": "SeekAndSlayLevel3_1-v0","map": "blue_mixed_resized_enemies",  "threshold": 7},
    {"id": "SeekAndSlayLevel4-v0",  "map": "complete",                    "threshold": None},
]

NUM_SEEDS = 5
SEEDS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Core evaluation logic
# ---------------------------------------------------------------------------

def run_episode(env, actor_factory, seed: int = None) -> dict:
    """Run a single episode with a freshly created agent and return the final info dict."""
    obs, info = env.reset(seed=seed)
    actor = actor_factory(env.action_space)
    done = False
    while not done:
        action = actor.act(obs)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return info


def evaluate_level(level_id: str, actor_factory, seeds: list[int]) -> dict:
    """
    Evaluate one level across multiple seeds.

    Args:
        level_id:       Gym environment ID.
        actor_factory:  Callable(action_space) -> Actor.
        seeds:          List of integer seeds to use.

    Returns:
        dict with per-seed info and aggregate stats.
    """
    per_seed = []
    for seed in seeds:
        env = levdoom.make(level_id)
        info = run_episode(env, actor_factory, seed=seed)
        env.close()
        per_seed.append(info)
        kills = info.get("kills", 0)
        print(f"  seed={seed}  kills={kills}  info={info}")

    kills_list = [ep.get("kills", 0) for ep in per_seed]
    mean_kills = float(np.mean(kills_list))
    return {
        "level": level_id,
        "per_seed": per_seed,
        "kills_per_seed": kills_list,
        "mean_kills": mean_kills,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_eval(actor_factory=None) -> list[dict]:
    """
    Evaluate all levels sequentially, stopping early if a threshold is not met.

    Args:
        actor_factory: Callable(action_space) -> Actor.
                       Defaults to RandomActor if not provided.

    Returns:
        List of result dicts (one per attempted level).
    """
    if actor_factory is None:
        # Default placeholder — swap in your own actor
        actor_factory = lambda action_space: Actor()

    results = []

    for level in LEVELS:
        level_id  = level["id"]
        threshold = level["threshold"]

        print(f"\n{'='*60}")
        print(f"Evaluating: {level_id}  (map={level['map']}, threshold={threshold})")
        print(f"{'='*60}")

        result = evaluate_level(level_id, actor_factory, SEEDS)
        results.append(result)

        mean_kills = result["mean_kills"]
        print(f"\n  -> Mean kills: {mean_kills:.2f}")

        if threshold is not None and mean_kills < threshold:
            print(f"  !! Below threshold ({threshold}). Stopping evaluation.")
            break
        elif threshold is not None:
            print(f"  ✓  Threshold met ({mean_kills:.2f} >= {threshold}). Proceeding.")
        else:
            print(f"  ✓  Final level complete.")

    print(f"\n{'='*60}")
    print("Results summary:")
    for r in results:
        print(f"  {r['level']}: mean_kills={r['mean_kills']:.2f}  per_seed={r['kills_per_seed']}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a student LevDoom agent.")
    parser.add_argument("--student-path", default=".", help="Directory containing student_agent.py")
    parser.add_argument("--output", default="results.json", help="Path to write results JSON")
    args = parser.parse_args()

    actor_factory = load_student_agent(args.student_path)
    results = run_eval(actor_factory)
    save_results(results, args.output)

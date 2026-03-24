# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a **reusable GitHub Actions workflow repository** — a competition evaluation harness for the "LevDoom Seek and Slay" RL competition. It evaluates student-submitted agents across 5 progressively harder Doom-based levels and submits results to a leaderboard.

## Running the Evaluator

```bash
# Evaluate a student submission locally
python judge.py --student-path <path_to_student_repo> --output results.json

# Defaults: --student-path "." --output "results.json"
```

Dependencies (not in a requirements file — installed by the workflow):
```bash
pip install numpy levdoom
```

## Architecture

**Two files do all the work:**

- `judge.py` — Core evaluation engine (~216 lines)
- `.github/workflows/evaluate.yml` — Reusable GitHub Actions workflow invoked by student repos

**Evaluation pipeline:**
1. Student repos call this workflow via `workflow_call`
2. Workflow checks out student repo → `./student/`, judge repo → `./judge/`
3. `judge.py` dynamically loads `student/student_agent.py` via `importlib.util`
4. Agent is evaluated across 5 levels, 5 seeds each (seeds `[0,1,2,3,4]`)
5. **Early stopping**: if `mean_kills < threshold` for a level, evaluation halts
6. Results written to `results.json` and POSTed to the leaderboard repo via GitHub dispatch API

**Level thresholds for early stopping:**
| Level | Environment | Kill Threshold |
|-------|-------------|---------------|
| 0 | SeekAndSlayLevel0-v0 | 22 |
| 1 | SeekAndSlayLevel1_6-v0 | 15 |
| 2 | SeekAndSlayLevel2_1-v0 | 9 |
| 3 | SeekAndSlayLevel3_1-v0 | 7 |
| 4 | SeekAndSlayLevel4-v0 | None |

**Scoring:** `kills × 0.8 + health × 0.1 + ammo × 0.1` (mean across 5 seeds per level)

## StudentAgent Interface

Students must implement this interface in `student_agent.py`:

```python
class StudentAgent:
    def __init__(self, action_space):
        ...
    def reset(self):          # Called before each episode
        ...
    def act(self, obs) -> int:  # obs is a numpy array
        ...
```

`judge.py` validates the class exists and has the required methods before running evaluation.

## Secrets Required

The GitHub Actions workflow needs two repository secrets:
- `LEADERBOARD_TOKEN` — GitHub token to post results
- `SUBMIT_SECRET` — Secret passed with the dispatch event

## Output Format

`results.json` structure expected by the leaderboard:
```json
{
  "level_id": {"kills": float, "health": float, "ammo": float},
  ...
}
```

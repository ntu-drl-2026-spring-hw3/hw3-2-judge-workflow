# LevDoom Seek and Slay — Judge Workflow

This repository contains the automated evaluation system for the LevDoom Seek and Slay competition. When a student pushes to their submission repo, GitHub Actions runs this judge and submits the result to the leaderboard.

---

## How evaluation works

1. The student's repo is checked out to `./student/`.
2. Dependencies are installed from `student/requirements.txt`.
3. `judge.py` loads `student/student_agent.py` and evaluates the agent across 5 levels.
4. For each level, the agent is run on 5 fixed seeds. A **fresh agent instance is created at the start of every episode**.
5. If the agent's mean kills fall below a level's threshold, evaluation stops early.
6. Results are submitted to the leaderboard.

### Levels

| Level | Environment ID | Threshold (mean kills) |
|-------|---------------|------------------------|
| 0 | `SeekAndSlayLevel0-v0` | 22 |
| 1 | `SeekAndSlayLevel1_6-v0` | 15 |
| 2 | `SeekAndSlayLevel2_1-v0` | 9 |
| 3 | `SeekAndSlayLevel3_1-v0` | 7 |
| 4 | `SeekAndSlayLevel4-v0` | — (final level) |

---

## Student submission guide

### Required files

Your repo must contain:

| File | Purpose |
|------|---------|
| `student_agent.py` | Your agent implementation (see below) |
| `requirements.txt` | Python dependencies your agent needs |
| `meta.xml` | Your student ID  |

### Implementing `student_agent.py`

Define a class named `StudentAgent` with `__init__(self, action_space)` and `act(self, obs)`.

```python
class StudentAgent:
    def __init__(self, action_space):
        # Called once at the start of every episode.
        # action_space.n  → number of discrete actions
        # action_space.sample() → random action
        self.action_space = action_space

    def act(self, obs) -> int:
        # Called at every timestep. Return an integer action.
        return self.action_space.sample()
```

See [sample_student_agent.py](sample_student_agent.py) for a fully annotated example.

### Loading model weights or other files

The judge does **not** run from your repo's directory, so bare relative paths will fail.
Always use `Path(__file__).parent` to reference files in your repo:

```python
from pathlib import Path
_DIR = Path(__file__).parent

class StudentAgent:
    def __init__(self, action_space):
        self.model = torch.load(_DIR / "weights.pth")  # correct
        # torch.load("weights.pth")                    # FileNotFoundError
```

### `meta.xml` format

```xml
<submission>
  <info>
    <name>your_student_id</name>
  </info>
</submission>
```

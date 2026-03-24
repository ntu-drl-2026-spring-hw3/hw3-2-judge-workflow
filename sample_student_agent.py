"""
Sample student agent for the LevDoom Seek and Slay competition.

Your submission must contain a file named `student_agent.py` with a class
named `StudentAgent` that implements the interface shown below.

The judge will:
  1. Call StudentAgent(action_space) once at the start of each episode.
  2. Call agent.act(obs) at every timestep until the episode ends.

IMPORTANT — loading files from your repo:
  Always use Path(__file__).parent to reference files in your repo.
  The judge does NOT run from your repo's directory, so bare relative
  paths like open("weights.pth") will fail.

  Correct:
      from pathlib import Path
      _DIR = Path(__file__).parent
      torch.load(_DIR / "weights.pth")   # works

  Wrong:
      torch.load("weights.pth")          # FileNotFoundError
"""

from pathlib import Path

import numpy as np

# _DIR points to the directory containing this file (your repo root).
# Use it whenever you need to load weights or other assets.
_DIR = Path(__file__).parent


class StudentAgent:
    def __init__(self, action_space):
        """
        Called once at the start of every episode.

        Args:
            action_space: A Gym Discrete action space.
                          Use action_space.n for the number of actions,
                          or action_space.sample() for a random action.
        """
        self.action_space = action_space

        # Example: load model weights from your repo
        # self.model = torch.load(_DIR / "weights.pth")

    def act(self, obs: np.ndarray) -> int:
        """
        Called at every timestep. Return an integer action.

        Args:
            obs: numpy array representing the current game frame.

        Returns:
            Integer action index in [0, action_space.n).
        """
        # Replace this with your own policy.
        return self.action_space.sample()

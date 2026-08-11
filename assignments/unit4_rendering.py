"""Unit 4 starter: sequential Gaussian splat rendering.

Read Unit 4 before filling the TODOs. Keep this implementation simple and correct; performance
is intentionally not the goal of this assignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def load_ply(path: Path):
    """TODO: read means, scales, rotations, opacities, and RGB colours from a PLY file."""
    raise NotImplementedError("Assignment TODO: implement PLY loading")


def project_gaussians(gaussians, world_to_camera, focal, width):
    """TODO: project 3D means and covariances into screen-space conics."""
    raise NotImplementedError("Assignment TODO: implement projection")


def render(gaussians, camera, width, background):
    """TODO: evaluate every splat at every pixel and alpha-composite front to back."""
    raise NotImplementedError("Assignment TODO: implement sequential rendering")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args()
    raise SystemExit("Complete the TODOs in this assignment before running it.")


if __name__ == "__main__":
    main()

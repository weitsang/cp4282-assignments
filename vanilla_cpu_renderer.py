"""Student starter for the sequential CPU Gaussian splat renderer.

The complete reference implementation is intentionally omitted. Implement the TODO functions
using Unit 4 of the notes, then use this file as the reference for the Warp assignment.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_ply(path: Path):
    """TODO: parse the Gaussian attributes from `path`."""
    raise NotImplementedError


def render_scene(gaussians, camera, width: int, background):
    """TODO: project, sort, evaluate, and alpha-composite the splats."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resolution", type=int, default=256)
    parser.parse_args()
    raise SystemExit("Complete the TODOs in vanilla_cpu_renderer.py first.")


if __name__ == "__main__":
    main()

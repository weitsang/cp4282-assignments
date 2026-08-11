"""Unit 9 starter: full image-based 3DGS training pipeline.

This file is intentionally incomplete. Implement one marked stage at a time, using the unit's
code fragments and the tests in this repository. Do not copy the instructor regression program.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def load_training_views(data_dir: Path, resolution: int):
    """TODO: load RGBA images and convert camera-to-world poses."""
    raise NotImplementedError


class ImageTrainer:
    """TODO: add trainable splat arrays, tile buffers, optimizer state, and schedules."""

    def __init__(self, targets, cameras, config):
        raise NotImplementedError

    def step(self, view_id: int):
        """TODO: build tiles, render one view, backpropagate, and update parameters."""
        raise NotImplementedError

    def densify_and_prune(self):
        """TODO: add clone/split/prune operations outside the differentiable kernels."""
        raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    raise SystemExit(f"Assignment starter loaded from {args.config}; implement the TODOs first.")


if __name__ == "__main__":
    main()

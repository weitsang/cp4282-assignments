"""Unit 5 starter: parallelize the Unit 4 renderer with Warp."""

from __future__ import annotations

import argparse
from pathlib import Path

import warp as wp


@wp.kernel
def rasterize_pixels():
    """TODO: one Warp work item should render one output pixel."""
    pixel = wp.tid()
    # TODO: map pixel to (x, y), walk the projected splats, and composite them.
    _ = pixel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    raise SystemExit("Complete the TODOs in this assignment before running it.")


if __name__ == "__main__":
    main()

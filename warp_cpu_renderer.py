"""Student starter for the Warp pixel-parallel renderer.

The complete reference implementation is intentionally omitted. Reuse the CPU renderer's data
contract, then implement the Warp kernel and host-side device transfers from Unit 5.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import warp as wp


@wp.kernel
def rasterize_pixels():
    """TODO: assign one Warp work item to each output pixel."""
    pixel = wp.tid()
    _ = pixel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.parse_args()
    raise SystemExit("Complete the TODOs in warp_cpu_renderer.py first.")


if __name__ == "__main__":
    main()

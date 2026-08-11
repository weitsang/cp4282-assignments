"""Student starter for evaluating a saved PLY on the Lego test views."""

from __future__ import annotations

import argparse
from pathlib import Path


def evaluate(ply: Path, data_dir: Path) -> None:
    """TODO: render each test camera and report PSNR and SSIM."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/lego"))
    args = parser.parse_args()
    evaluate(args.ply, args.data)


if __name__ == "__main__":
    main()

"""Starter finite-difference gradient checker for the Warp trainer.

Run this after implementing the Unit 9 backward pass. Compare the analytic gradient from the
renderer against a central finite difference for each parameter group.
"""

from __future__ import annotations

import argparse


def check_gradients(device: str, epsilon: float) -> None:
    """TODO: perturb position, scale, rotation, opacity, and colour parameters."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epsilon", type=float, default=1.0e-3)
    args = parser.parse_args()
    check_gradients(args.device, args.epsilon)


if __name__ == "__main__":
    main()

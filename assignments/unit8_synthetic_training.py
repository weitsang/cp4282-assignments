"""Unit 8 starter: differentiable training on a synthetic Gaussian scene."""

from __future__ import annotations

import argparse


def render_loss(parameters, target, camera):
    """TODO: render the current Gaussian and return an image loss."""
    raise NotImplementedError


def train(iterations: int, device: str) -> None:
    """TODO: initialize parameters, record a Warp tape, and update the Gaussian."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    train(args.iterations, args.device)


if __name__ == "__main__":
    main()

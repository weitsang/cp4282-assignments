"""Unit 11 starter: add the scheduled D-SSIM image loss."""

from __future__ import annotations


class DSSIMComputer:
    """TODO: implement the local SSIM statistics and image-space gradient."""

    def compute(self, rendered, target, mse_gradient, config):
        raise NotImplementedError


def blended_loss(mse, dssim, weight: float):
    """TODO: return the configured MSE/D-SSIM blend."""
    raise NotImplementedError

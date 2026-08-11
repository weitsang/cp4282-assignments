"""Starter D-SSIM metric kernels for Unit 11."""

from __future__ import annotations


class DSSIMComputer:
    """Compute D-SSIM loss and image-space gradients.

    The trainer should treat this as a black box: rendered image in, target image in,
    scalar loss and per-pixel gradient out.
    """

    def __init__(self, width: int, channels: int = 3):
        self.width = width
        self.channels = channels

    def compute(self, rendered, target, *, sample_step: int, window: int):
        """TODO: evaluate local SSIM windows and return D-SSIM plus image gradient."""
        raise NotImplementedError

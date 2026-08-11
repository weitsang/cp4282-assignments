"""Starter container for trainable 3D Gaussian parameters.

The lecture notes use this class name to keep Unit 8 and Unit 9 consistent. Complete the
allocation and update methods as you work through the assignment.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainableGaussianSet:
    """Fixed-capacity structure-of-arrays container for trainable splats."""

    count: int
    device: object

    @classmethod
    def random_init(cls, count: int, rng, device):
        """TODO: allocate means, log_scales, quaternions, opacity logits, and RGB arrays."""
        raise NotImplementedError

    def sgd_step(self, tape, **learning_rates) -> None:
        """TODO: apply one SGD update from Warp gradient arrays."""
        raise NotImplementedError

    def clamp_parameters(self) -> None:
        """TODO: normalize quaternions and clamp constrained parameter ranges."""
        raise NotImplementedError

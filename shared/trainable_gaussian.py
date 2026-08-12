"""Warp storage and updates for trainable 3D Gaussian parameters.

This module deliberately separates parameter storage from rendering. The class supports N splats,
but Unit 9's renderer uses only one splat; Unit 10 adds the multi-splat renderer.
"""

from __future__ import annotations

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _sgd_update(
    means: wp.array(dtype=wp.vec3),
    mean_grad: wp.array(dtype=wp.vec3),
    log_scales: wp.array(dtype=wp.vec3),
    scale_grad: wp.array(dtype=wp.vec3),
    quaternions: wp.array(dtype=wp.vec4),
    quaternion_grad: wp.array(dtype=wp.vec4),
    opacity_logits: wp.array(dtype=wp.float32),
    opacity_grad: wp.array(dtype=wp.float32),
    colors: wp.array(dtype=wp.vec3),
    color_grad: wp.array(dtype=wp.vec3),
    position_lr: float,
    scale_lr: float,
    rotation_lr: float,
    opacity_lr: float,
    color_lr: float,
):
    splat = wp.tid()
    means[splat] = means[splat] - position_lr * mean_grad[splat]
    log_scales[splat] = log_scales[splat] - scale_lr * scale_grad[splat]
    quaternions[splat] = quaternions[splat] - rotation_lr * quaternion_grad[splat]
    opacity_logits[splat] = wp.clamp(
        opacity_logits[splat] - opacity_lr * opacity_grad[splat], -8.0, 4.0
    )
    colors[splat] = colors[splat] - color_lr * color_grad[splat]

    q = quaternions[splat]
    quaternions[splat] = q / wp.sqrt(wp.dot(q, q) + 1.0e-8)
    log_scales[splat] = wp.vec3(
        wp.clamp(log_scales[splat][0], -4.0, 0.0),
        wp.clamp(log_scales[splat][1], -4.0, 0.0),
        wp.clamp(log_scales[splat][2], -4.0, 0.0),
    )
    colors[splat] = wp.vec3(
        wp.clamp(colors[splat][0], 0.0, 1.0),
        wp.clamp(colors[splat][1], 0.0, 1.0),
        wp.clamp(colors[splat][2], 0.0, 1.0),
    )


class TrainableGaussianSet:
    """Raw Warp parameters for N anisotropic 3D Gaussians.

    The arrays store unconstrained values where useful: log-scales and opacity logits. The
    renderer applies exp and sigmoid when it needs physical scales and opacity values.
    """

    def __init__(
        self,
        means: np.ndarray,
        log_scales: np.ndarray,
        quaternions: np.ndarray,
        opacity_logits: np.ndarray,
        colors: np.ndarray,
        device: wp.context.Device,
        requires_grad: bool = True,
    ):
        arrays = (means, log_scales, quaternions, opacity_logits, colors)
        count = len(means)
        if count == 0:
            raise ValueError("TrainableGaussianSet needs at least one Gaussian.")
        if any(len(values) != count for values in arrays[1:]):
            raise ValueError("All Gaussian parameter arrays must have the same length.")
        if means.shape != (count, 3) or log_scales.shape != (count, 3):
            raise ValueError("means and log_scales must have shape (N, 3).")
        if quaternions.shape != (count, 4) or colors.shape != (count, 3):
            raise ValueError("quaternions must have shape (N, 4), and colors must have shape (N, 3).")
        if opacity_logits.shape != (count,):
            raise ValueError("opacity_logits must have shape (N,).")

        self.count = count
        self.device = device
        self.means = wp.array(means.astype(np.float32), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.log_scales = wp.array(log_scales.astype(np.float32), dtype=wp.vec3, device=device, requires_grad=requires_grad)
        self.quaternions = wp.array(quaternions.astype(np.float32), dtype=wp.vec4, device=device, requires_grad=requires_grad)
        self.opacity_logits = wp.array(opacity_logits.astype(np.float32), dtype=wp.float32, device=device, requires_grad=requires_grad)
        self.colors = wp.array(colors.astype(np.float32), dtype=wp.vec3, device=device, requires_grad=requires_grad)

    @classmethod
    def random_init(
        cls,
        count: int,
        rng: np.random.Generator,
        device: wp.context.Device,
    ) -> "TrainableGaussianSet":
        """Create the small random initialization used by Unit 9."""
        means = rng.uniform(-0.45, 0.45, size=(count, 3)).astype(np.float32)
        means[:, 2] *= 0.3
        log_scales = np.tile(
            np.log(np.array([0.16, 0.18, 0.14], dtype=np.float32)), (count, 1)
        )
        quaternions = np.tile(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (count, 1)
        )
        opacity_logits = np.full(count, -0.5, dtype=np.float32)
        colors = rng.uniform(0.2, 0.8, size=(count, 3)).astype(np.float32)
        return cls(means, log_scales, quaternions, opacity_logits, colors, device)

    def sgd_step(
        self,
        tape: wp.Tape,
        position_lr: float = 2.0,
        scale_lr: float = 0.8,
        rotation_lr: float = 0.3,
        opacity_lr: float = 1.0,
        color_lr: float = 2.0,
    ) -> None:
        """Apply one parameter-group SGD update using gradients from ``tape``."""
        wp.launch(
            _sgd_update,
            dim=self.count,
            inputs=[
                self.means, tape.gradients[self.means],
                self.log_scales, tape.gradients[self.log_scales],
                self.quaternions, tape.gradients[self.quaternions],
                self.opacity_logits, tape.gradients[self.opacity_logits],
                self.colors, tape.gradients[self.colors],
                position_lr, scale_lr, rotation_lr, opacity_lr, color_lr,
            ],
            device=self.device,
        )

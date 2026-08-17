"""Finite-difference regression checks for the Unit 10 Warp trainer.

Run from docs/examples:

    python 3dgs_gradient_check_gpu.py

The check is intentionally tiny. It compares the explicit Warp backward pass with central
finite differences for every trainable parameter family, verifies that gradients do not
accumulate across calls, checks white-background compositing, exact near-to-far tile ordering,
compact-box culling, and the learning-rate schedule.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys

import numpy as np
import warp as wp

_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
# `shared/` sits beside this file in the assignment repo, and one level up in the course repo.
for _candidate in (_here / "shared", _here.parent / "shared"):
    if _candidate.is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break

from gaussian_set import GaussianSet

_base_trainer = importlib.import_module("3dgs_trainer")
WarpImageTrainer = _base_trainer.WarpImageTrainer
exponential_learning_rate = _base_trainer.exponential_learning_rate
projected_covariance_diagonal = _base_trainer.projected_covariance_diagonal

# These checks validate the dense forward and backward kernels specifically, so they pin the path
# rather than inheriting whichever one the configuration default currently selects. They also have
# to: `finite_difference` below evaluates the loss three times and differences the results, and the
# sparse path redraws its pixel sample on every call, so the three evaluations would be measuring
# three different objectives. The sparse kernels have their own suite, which reseeds between
# evaluations for that reason -- see `3dgs_gradient_check_sparse_gpu.py`.
DENSE_ONLY = {"sparse": {"enabled": False}}


def one_splat_scene() -> GaussianSet:
    quaternion = np.array([[0.96, 0.10, 0.20, 0.15]], np.float32)
    quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
    return GaussianSet(
        means=np.array([[0.15, -0.10, 1.20]], np.float32),
        scales=np.array([[0.30, 0.60, 0.40]], np.float32),
        rotations=quaternion,
        opacities=np.array([0.40], np.float32),
        colors=np.array([[0.40, 0.50, 0.60]], np.float32),
    )


def make_trainer(width: int, target: np.ndarray, background, device: str) -> WarpImageTrainer:
    targets = np.broadcast_to(target, (1, width, width, 3)).copy().astype(np.float32)
    cameras = np.eye(4, dtype=np.float32)[None]
    trainer = WarpImageTrainer(
        targets, cameras, float(width), 1.0, 1, 1, device, 7,
        init_scene=one_splat_scene(), background=background,
        training_options=DENSE_ONLY,
    )
    return trainer


def finite_difference(trainer, field, dtype, index, epsilon=2.0e-3) -> float:
    original = field.numpy()
    plus = original.copy()
    minus = original.copy()
    plus[index] += epsilon
    minus[index] -= epsilon
    field.assign(wp.array(plus, dtype=dtype, device=trainer.device))
    loss_plus = trainer.loss_and_gradients(np.array([0]))
    field.assign(wp.array(minus, dtype=dtype, device=trainer.device))
    loss_minus = trainer.loss_and_gradients(np.array([0]))
    field.assign(wp.array(original, dtype=dtype, device=trainer.device))
    return (loss_plus - loss_minus) / (2.0 * epsilon)


def check_parameter_gradients(device: str) -> None:
    trainer = make_trainer(4, np.array([0.20, 0.30, 0.40]), (1.0, 1.0, 1.0), device)
    trainer.loss_and_gradients(np.array([0]))
    analytic = {
        "mean x": float(trainer.optimizers.mean_grad.numpy()[0, 0]),
        "log-scale y": float(trainer.optimizers.scale_grad.numpy()[0, 1]),
        "quaternion y": float(trainer.optimizers.quaternion_grad_flat.numpy()[2]),
        "opacity logit": float(trainer.optimizers.opacity_grad.numpy()[0]),
        "base green": float(trainer.optimizers.color_grad.numpy()[0, 1]),
    }
    cases = [
        ("mean x", trainer.means, wp.vec3, (0, 0)),
        ("log-scale y", trainer.log_scales, wp.vec3, (0, 1)),
        ("quaternion y", trainer.quaternions, wp.vec4, (0, 2)),
        ("opacity logit", trainer.opacity, wp.float32, 0),
        ("base green", trainer.color, wp.vec3, (0, 1)),
    ]
    for name, field, dtype, index in cases:
        numeric = finite_difference(trainer, field, dtype, index)
        np.testing.assert_allclose(analytic[name], numeric, rtol=5.0e-2, atol=2.0e-4)
        print(f"{name:16s} analytic={analytic[name]: .7f} finite_difference={numeric: .7f}")


def check_background_and_gradient_reset(device: str) -> None:
    scene = GaussianSet(
        means=np.array([[0.0, 0.0, 1.0]], np.float32),
        scales=np.ones((1, 3), np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
        opacities=np.array([0.5], np.float32),
        colors=np.full((1, 3), 0.5, np.float32),
    )
    targets = np.full((1, 1, 1, 3), 0.1, np.float32)
    cameras = np.eye(4, dtype=np.float32)[None]
    trainer = WarpImageTrainer(
        targets, cameras, 1.0, 1.0, 1, 1, device, 7,
        init_scene=scene, background=(0.0, 0.0, 0.0),
        training_options=DENSE_ONLY,
    )
    trainer.loss_and_gradients(np.array([0]))
    first = trainer.optimizers.color_grad.numpy().copy()
    trainer.loss_and_gradients(np.array([0]))
    second = trainer.optimizers.color_grad.numpy().copy()
    np.testing.assert_allclose(first, np.full((1, 3), 0.05, np.float32), atol=1.0e-6)
    np.testing.assert_allclose(second, first, atol=1.0e-7)

    white = WarpImageTrainer(
        np.full((1, 1, 1, 3), 0.75, np.float32), cameras, 1.0, 1.0, 1, 1,
        device, 7, init_scene=scene, background=(1.0, 1.0, 1.0),
        training_options=DENSE_ONLY,
    )
    loss = white.loss_and_gradients(np.array([0]))
    np.testing.assert_allclose(white.image.numpy()[0], np.full(3, 0.75), atol=1.0e-6)
    np.testing.assert_allclose(loss, 0.0, atol=1.0e-7)
    print("background       rendered [0.75, 0.75, 0.75] over white")
    print("gradient reset   repeated backward calls agree")


def check_gpu_tile_depth_order(device: str) -> None:
    scene = GaussianSet(
        means=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 1.0]], np.float32),
        scales=np.full((2, 3), 0.1, np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]] * 2, np.float32),
        opacities=np.full(2, 0.5, np.float32),
        colors=np.full((2, 3), 0.5, np.float32),
    )
    targets = np.zeros((1, 16, 16, 3), np.float32)
    cameras = np.eye(4, dtype=np.float32)[None]
    trainer = WarpImageTrainer(
        targets, cameras, 16.0, 2.0, 2, 2, device, 7,
        init_scene=scene, tile_pair_capacity=32,
    )
    offsets, pairs, pair_count = trainer.build_tiles(np.array([0], np.int32))
    offsets = offsets[:trainer.tiles_per_view + 1].numpy()
    pairs = pairs[:pair_count].numpy()
    indices = np.asarray(pairs & np.uint64(0xFFFFFFFF), np.int32)
    centre_tile = (trainer.tiles_x // 2) * trainer.tiles_x + trainer.tiles_x // 2
    ordered = indices[offsets[centre_tile]:offsets[centre_tile + 1]]
    np.testing.assert_array_equal(ordered, np.array([1, 0], np.int32))
    print("tile sorting      stable tile sort preserves exact near-to-far depth")


def check_compact_box_reduces_pairs(device: str) -> None:
    scene = GaussianSet(
        means=np.array([[0.0, 0.0, 1.0]], np.float32),
        scales=np.full((1, 3), 0.15, np.float32),
        rotations=np.array([[1.0, 0.0, 0.0, 0.0]], np.float32),
        opacities=np.array([0.02], np.float32),
        colors=np.full((1, 3), 0.5, np.float32),
    )
    targets = np.zeros((1, 64, 64, 3), np.float32)
    cameras = np.eye(4, dtype=np.float32)[None]
    baseline = WarpImageTrainer(
        targets, cameras, 64.0, 1.0, 1, 1, device, 7,
        init_scene=scene, tile_pair_capacity=128,
        training_options={"compact_box": {"enabled": False}},
    )
    compact = WarpImageTrainer(
        targets, cameras, 64.0, 1.0, 1, 1, device, 7,
        init_scene=scene, tile_pair_capacity=128,
        training_options={
            "compact_box": {"enabled": True, "beta": 0.5, "alpha_min": 1.0 / 255.0}
        },
    )
    _, _, baseline_pairs = baseline.build_tiles(np.array([0], np.int32))
    _, _, compact_pairs = compact.build_tiles(np.array([0], np.int32))
    assert compact_pairs < baseline_pairs
    print("compact box       low-opacity splat emits fewer tile records")


def check_learning_rate_schedule() -> None:
    initial = exponential_learning_rate(0, 1.6e-4, 1.6e-6, 30_000)
    middle = exponential_learning_rate(15_000, 1.6e-4, 1.6e-6, 30_000)
    final = exponential_learning_rate(30_000, 1.6e-4, 1.6e-6, 30_000)
    np.testing.assert_allclose(initial, 1.6e-4)
    np.testing.assert_allclose(middle, 1.6e-5)
    np.testing.assert_allclose(final, 1.6e-6)
    print("LR schedule       position rate decays log-linearly between its endpoints")


def check_projected_tile_bounds(device: str) -> None:
    width = 192
    camera = np.eye(4, dtype=np.float32)
    point = np.array([[0.0, 0.0, 2.0]], np.float32)
    scales = np.array([[1.0, 0.1, 0.1]], np.float32)
    identity = np.array([[1.0, 0.0, 0.0, 0.0]], np.float32)
    quarter_turn = np.array([[np.sqrt(0.5), 0.0, 0.0, np.sqrt(0.5)]], np.float32)
    identity_x, identity_y = projected_covariance_diagonal(
        scales, identity, camera, point, float(width)
    )
    rotated_x, rotated_y = projected_covariance_diagonal(
        scales, quarter_turn, camera, point, float(width)
    )
    assert identity_x[0] > identity_y[0]
    assert rotated_y[0] > rotated_x[0]

    scene = GaussianSet(
        means=point,
        scales=scales,
        rotations=identity,
        opacities=np.array([0.5], np.float32),
        colors=np.full((1, 3), 0.5, np.float32),
    )
    targets = np.zeros((1, width, width, 3), np.float32)
    trainer = WarpImageTrainer(
        targets, camera[None], float(width), 2.0, 1, 1, device, 7,
        init_scene=scene, tile_pair_capacity=256,
    )
    offsets, pairs, pair_count = trainer.build_tiles(np.array([0], np.int32))
    offsets = offsets[:trainer.tiles_per_view + 1].numpy()
    pairs = pairs[:pair_count].numpy()
    indices = np.asarray(pairs & np.uint64(0xFFFFFFFF), np.int32)
    centre_row = trainer.tiles_x // 2
    for tile_x in range(trainer.tiles_x):
        record = centre_row * trainer.tiles_x + tile_x
        assert 0 in indices[offsets[record]:offsets[record + 1]]
    print("tile bounds       rotated covariance is respected; wide footprints are not clipped")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", help="Warp device, such as cpu or cuda:0")
    args = parser.parse_args()
    wp.init()
    if args.device.startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError(f"{args.device} requested, but Warp cannot access a CUDA driver.")
    check_parameter_gradients(args.device)
    check_background_and_gradient_reset(args.device)
    check_gpu_tile_depth_order(args.device)
    check_compact_box_reduces_pairs(args.device)
    check_learning_rate_schedule()
    check_projected_tile_bounds(args.device)
    print(f"All Warp gradient checks passed on {args.device}.")


if __name__ == "__main__":
    main()

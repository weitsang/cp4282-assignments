"""Train anisotropic 3D Gaussian splats from NeRF-synthetic posed images with Warp.

Usage:
    python warp_images_3dgs_training.py training.yaml

The data directory contains transforms_train.json and its RGBA image files. Warp owns the
parallel raster, explicit backward pass, and Adam updates. Python owns image loading, camera
selection, discrete tile records, densification, pruning, PLY snapshots, and validation.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from time import perf_counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import warp as wp
from warp.optim import Adam
import yaml

# The reference CPU renderer belongs to the Assignment 1 solution, not this examples folder.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "a1-solution"))

from trainable_gaussian import TrainableGaussianSet
from vanilla_cpu_renderer import GaussianSet, SUPPORT_RADIUS_SQUARED, quaternion_to_matrix


DEFAULT_RESOLUTION = 256
DEFAULT_SPLATS = 100_000
DEFAULT_INITIAL_SPLATS = 50_000
TILE = 16
NEAR_PLANE = 0.1
FILTER_VARIANCE = 0.3
INACTIVE_LOGIT = -12.0
ALPHA_CUTOFF = 1.0 / 255.0
TRANSMITTANCE_CUTOFF = 1.0e-4
# Splats with opacity below this are pruned outright, independent of the ADC opacity_cutoff
# config knob above it in severity -- this is the unconditional floor every splat must clear.
PRUNE_OPACITY_FLOOR = 0.005
# constrain_parameters clamps log-scale to [log(scene_radius * MIN), log(scene_radius * MAX)].
SCALE_CLAMP_MIN_FRACTION = 1.0 / 500.0
SCALE_CLAMP_MAX_FRACTION = 1.0 / 3.0
# warp.optim.Adam's own defaults. The quaternion needs a matching hand-written Adam step because
# Adam rejects vec4 parameters; plain SGD at an Adam-tuned rate leaves gradients this small
# (~1e-9, since the loss averages over every pixel and splat) unable to move the parameter at
# all, freezing every splat's orientation at initialization.
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1.0e-8

def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


class TrainingConfig:
    """Validated, merged training configuration loaded from one YAML file.

    Wraps a plain nested dict merged over `DEFAULTS`. Supports the same `config["section"]`
    indexing the rest of the module uses; loading, merging, validating, and formatting for
    display are all methods here rather than free functions passing a dict back and forth.
    """

    DEFAULTS = {
        "paths": {
            "data": "data/lego",
            "output": "warp-training.png",
            "init_ply": None,
        },
        "runtime": {
            "arch": "cpu",
            "seed": 7,
        },
        "model": {
            "resolution": DEFAULT_RESOLUTION,
            "capacity": DEFAULT_SPLATS,
            "initial_splats": DEFAULT_INITIAL_SPLATS,
            "tile_pair_capacity": None,
            "scene_radius": None,
            "background": "white",
        },
        "training": {
            "iterations": 2000,
        },
        "learning_rates": {
            "position_initial": 0.00016,
            "position_final": 0.0000016,
            "position_max_steps": 2_000,
            "position_delay_steps": 0,
            "position_delay_multiplier": 0.01,
            "feature_dc": 0.0025,
            "opacity": 0.05,
            "scale": 0.005,
            "rotation": 0.001,
        },
        "densification": {
            "interval": 100,
            "fraction": 0.5,
            # Stop densifying part-way, as the reference trainer does; null means half the run.
            # Without a stop, a long run keeps adding splats that later stages can no longer place.
            "until": None,
            # A splat is "large" when its biggest scale exceeds percent_dense * scene_radius.
            # Large splats are split; small ones are cloned. This is the reference percent_dense.
            "percent_dense": 0.01,
            # Periodically cap opacity so faded splats must re-earn it or be pruned. 0 disables.
            "opacity_reset_interval": 0,
        },
        "compact_box": {
            "enabled": True,
            "beta": 0.5,
            "alpha_min": ALPHA_CUTOFF,
        },
        "multiview_adc": {
            "enabled": True,
            "views": 10,
            "loss_threshold": 0.1,
            "densify_score_threshold": 5.0,
            "prune_score_threshold": 0.9,
            "late_prune_start": 1200,
            "late_prune_end": 2000,
            "late_prune_interval": 400,
            "opacity_cutoff": 0.1,
            "candidate_limit": 20000,
        },
        "convergence": {
            "enabled": True,
            "min_iterations": 1200,
            "patience": 5,
            "min_delta": 1.0e-5,
        },
        "reporting": {
            "log_every": 20,
            "eval_every": 100,
            "eval_views": 4,
            "loss_ema_decay": 0.95,
            "snapshot_every": 200,
            "save_ply": True,
        },
    }

    def __init__(self, values: dict):
        self.values = values

    def __getitem__(self, key):
        return self.values[key]

    def __repr__(self) -> str:
        return f"TrainingConfig({self.values!r})"

    @classmethod
    def load(cls, path: Path) -> "TrainingConfig":
        config_path = path.expanduser().resolve()
        supplied = yaml.safe_load(config_path.read_text())
        config = cls(cls._merge(cls.DEFAULTS, supplied or {}))
        for key in ("data", "output", "init_ply"):
            value = config.values["paths"][key]
            if value is None:
                continue
            if not isinstance(value, (str, Path)):
                raise ValueError(f"paths.{key} must be a path string or null.")
            resolved = Path(value).expanduser()
            if not resolved.is_absolute():
                resolved = config_path.parent / resolved
            config.values["paths"][key] = resolved.resolve()
        config.validate()
        return config

    @classmethod
    def _merge(cls, defaults: dict, supplied: dict, prefix: str = "") -> dict:
        if not isinstance(supplied, dict):
            location = prefix.removesuffix(".") or "configuration"
            raise ValueError(f"{location} must be a YAML mapping.")
        unknown = sorted(set(supplied) - set(defaults))
        if unknown:
            location = prefix.removesuffix(".") or "configuration"
            raise ValueError(f"Unknown key(s) in {location}: {', '.join(unknown)}")
        merged = deepcopy(defaults)
        for key, value in supplied.items():
            if isinstance(defaults[key], dict):
                merged[key] = cls._merge(defaults[key], value, f"{prefix}{key}.")
            else:
                merged[key] = value
        return merged

    def validate(self) -> None:
        paths = self.values["paths"]
        runtime = self.values["runtime"]
        model = self.values["model"]
        training = self.values["training"]
        rates = self.values["learning_rates"]
        densification = self.values["densification"]
        compact = self.values["compact_box"]
        adc = self.values["multiview_adc"]
        convergence = self.values["convergence"]
        reporting = self.values["reporting"]
        if runtime["arch"] not in ("cpu", "gpu"):
            raise ValueError("runtime.arch must be 'cpu' or 'gpu'.")
        if not isinstance(runtime["seed"], int) or isinstance(runtime["seed"], bool):
            raise ValueError("runtime.seed must be an integer.")
        for name in ("data", "output"):
            if paths[name] is None:
                raise ValueError(f"paths.{name} cannot be null.")
        if model["background"] not in ("white", "black"):
            raise ValueError("model.background must be 'white' or 'black'.")
        for name, value in (
            ("model.resolution", model["resolution"]),
            ("model.capacity", model["capacity"]),
            ("model.initial_splats", model["initial_splats"]),
            ("training.iterations", training["iterations"]),
            ("learning_rates.position_max_steps", rates["position_max_steps"]),
            ("reporting.log_every", reporting["log_every"]),
            ("reporting.eval_every", reporting["eval_every"]),
            ("reporting.eval_views", reporting["eval_views"]),
            ("reporting.snapshot_every", reporting["snapshot_every"]),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1.")
        if model["initial_splats"] > model["capacity"]:
            raise ValueError("model.initial_splats cannot exceed model.capacity.")
        if (
            model["tile_pair_capacity"] is not None
            and (
                not isinstance(model["tile_pair_capacity"], int)
                or isinstance(model["tile_pair_capacity"], bool)
                or model["tile_pair_capacity"] < 1
            )
        ):
            raise ValueError("model.tile_pair_capacity must be a positive integer or null.")
        if model["scene_radius"] is not None and model["scene_radius"] <= 0.0:
            raise ValueError("model.scene_radius must be positive or null.")
        if not isinstance(densification["interval"], int) or densification["interval"] < 0:
            raise ValueError("densification.interval must be a non-negative integer.")
        if not 0.0 < densification["fraction"] <= 1.0:
            raise ValueError("densification.fraction must be in (0, 1].")
        if densification["until"] is not None and (
            not isinstance(densification["until"], int)
            or isinstance(densification["until"], bool)
            or densification["until"] < 1
        ):
            raise ValueError("densification.until must be a positive integer or null.")
        if not 0.0 < densification["percent_dense"] <= 1.0:
            raise ValueError("densification.percent_dense must be in (0, 1].")
        if (
            not isinstance(densification["opacity_reset_interval"], int)
            or isinstance(densification["opacity_reset_interval"], bool)
            or densification["opacity_reset_interval"] < 0
        ):
            raise ValueError(
                "densification.opacity_reset_interval must be a non-negative integer."
            )
        if not 0.0 <= reporting["loss_ema_decay"] < 1.0:
            raise ValueError("reporting.loss_ema_decay must be in [0, 1).")
        if not isinstance(reporting["save_ply"], bool):
            raise ValueError("reporting.save_ply must be true or false.")
        if not isinstance(compact["enabled"], bool):
            raise ValueError("compact_box.enabled must be true or false.")
        if compact["beta"] <= 0.0:
            raise ValueError("compact_box.beta must be positive.")
        if compact["alpha_min"] <= 0.0:
            raise ValueError("compact_box.alpha_min must be positive.")
        if not isinstance(adc["enabled"], bool):
            raise ValueError("multiview_adc.enabled must be true or false.")
        for name in ("views", "late_prune_start", "late_prune_end", "late_prune_interval", "candidate_limit"):
            if not isinstance(adc[name], int) or isinstance(adc[name], bool) or adc[name] < 1:
                raise ValueError(f"multiview_adc.{name} must be an integer of at least 1.")
        for name in ("loss_threshold", "densify_score_threshold", "prune_score_threshold", "opacity_cutoff"):
            if adc[name] < 0.0:
                raise ValueError(f"multiview_adc.{name} must be non-negative.")
        if not isinstance(convergence["enabled"], bool):
            raise ValueError("convergence.enabled must be true or false.")
        for name in ("min_iterations", "patience"):
            if (
                not isinstance(convergence[name], int)
                or isinstance(convergence[name], bool)
                or convergence[name] < 1
            ):
                raise ValueError(f"convergence.{name} must be an integer of at least 1.")
        if convergence["min_delta"] < 0.0:
            raise ValueError("convergence.min_delta must be non-negative.")
        for name, value in rates.items():
            if name == "position_delay_steps":
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError("learning_rates.position_delay_steps must be a non-negative integer.")
            elif name == "position_delay_multiplier":
                if not 0.0 < value <= 1.0:
                    raise ValueError("learning_rates.position_delay_multiplier must be in (0, 1].")
            elif value <= 0.0:
                raise ValueError(f"learning_rates.{name} must be positive.")

    def printable(self) -> dict:
        result = deepcopy(self.values)
        for key, value in result["paths"].items():
            result["paths"][key] = None if value is None else str(value)
        return result


def exponential_learning_rate(
    step: int,
    initial: float,
    final: float,
    max_steps: int,
    delay_steps: int = 0,
    delay_multiplier: float = 1.0,
) -> float:
    interpolation = np.clip(step / max_steps, 0.0, 1.0)
    learning_rate = np.exp(
        np.log(initial) * (1.0 - interpolation) + np.log(final) * interpolation
    )
    if delay_steps > 0:
        delay_progress = np.clip(step / delay_steps, 0.0, 1.0)
        delay = delay_multiplier + (1.0 - delay_multiplier) * np.sin(
            0.5 * np.pi * delay_progress
        )
        learning_rate *= delay
    return float(learning_rate)


def count_multiples_in_range(start: int, end: int, interval: int) -> int:
    """Count integers in [start, end] that are also multiples of interval."""
    if start > end:
        return 0
    first_multiple = -(-start // interval) * interval
    if first_multiple > end:
        return 0
    return (end - first_multiple) // interval + 1


def blender_pose_to_world_to_camera(transform: list[list[float]]) -> np.ndarray:
    camera_to_world = np.array(transform, dtype=np.float64)
    camera_to_world[:3, 1:3] *= -1.0
    return np.linalg.inv(camera_to_world).astype(np.float32)


def quaternion_to_rotation(quaternion: np.ndarray) -> np.ndarray:
    """Return the 3x3 rotation matrix for one (w, x, y, z) quaternion.

    Thin single-quaternion wrapper around Unit 5's `quaternion_to_matrix`, so the formula is not
    hand-derived a second time. Its columns are the local-frame basis vectors expressed in world
    space, matching the convention `projected_covariance` uses on the Warp side.
    """
    unit = quaternion / max(float(np.linalg.norm(quaternion)), 1.0e-8)
    return quaternion_to_matrix(unit[np.newaxis, :])[0]


def projected_covariance_diagonal(
    scales: np.ndarray,
    quaternions: np.ndarray,
    camera: np.ndarray,
    camera_points: np.ndarray,
    focal: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the x/y diagonal of each projected 2D covariance."""
    quaternions = quaternions / np.maximum(
        np.linalg.norm(quaternions, axis=1, keepdims=True), 1.0e-8
    )
    rotation = quaternion_to_matrix(quaternions)
    x, y, z = camera_points.T
    jacobian_x = np.stack((focal / z, np.zeros_like(z), -focal * x / (z * z)), axis=1)
    jacobian_y = np.stack((np.zeros_like(z), focal / z, -focal * y / (z * z)), axis=1)
    projected_x = jacobian_x @ camera[:3, :3]
    projected_y = jacobian_y @ camera[:3, :3]
    local_x = np.einsum("ni,nij->nj", projected_x, rotation)
    local_y = np.einsum("ni,nij->nj", projected_y, rotation)
    scales_squared = scales * scales
    variance_x = np.sum(scales_squared * local_x * local_x, axis=1) + FILTER_VARIANCE
    variance_y = np.sum(scales_squared * local_y * local_y, axis=1) + FILTER_VARIANCE
    return variance_x, variance_y


def load_views(data: Path, resolution: int, background: np.ndarray, manifest_name: str = "transforms_train.json"):
    manifest = json.loads((data / manifest_name).read_text())
    focal = np.float32(0.5 * resolution / np.tan(0.5 * manifest["camera_angle_x"]))
    images, cameras, distances, frame_paths = [], [], [], []
    for frame in manifest["frames"]:
        image = Image.open(data / f"{frame['file_path']}.png").convert("RGBA")
        rgba = np.asarray(image.resize((resolution, resolution), Image.LANCZOS), np.float32) / 255.0
        alpha = rgba[..., 3:4]
        images.append(rgba[..., :3] * alpha + (1.0 - alpha) * background)
        cameras.append(blender_pose_to_world_to_camera(frame["transform_matrix"]))
        distances.append(np.linalg.norm(np.asarray(frame["transform_matrix"], np.float32)[:3, 3]))
        frame_paths.append(frame["file_path"])
    return np.stack(images), np.stack(cameras), focal, float(np.mean(distances)), frame_paths


@wp.func
def projected_covariance(
    log_scale: wp.vec3,
    quaternion: wp.vec4,
    camera: wp.mat44,
    x: float,
    y: float,
    z: float,
    focal: float,
):
    quaternion = quaternion / wp.sqrt(wp.dot(quaternion, quaternion) + 1.0e-8)
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    r0 = wp.vec3(1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy + qw * qz), 2.0 * (qx * qz - qw * qy))
    r1 = wp.vec3(2.0 * (qx * qy - qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz + qw * qx))
    r2 = wp.vec3(2.0 * (qx * qz + qw * qy), 2.0 * (qy * qz - qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy))
    c0 = wp.vec3(camera[0, 0], camera[1, 0], camera[2, 0])
    c1 = wp.vec3(camera[0, 1], camera[1, 1], camera[2, 1])
    c2 = wp.vec3(camera[0, 2], camera[1, 2], camera[2, 2])
    j0 = wp.vec3(focal / z, 0.0, -focal * x / (z * z))
    j1 = wp.vec3(0.0, focal / z, -focal * y / (z * z))
    p0 = wp.vec3(wp.dot(j0, c0), wp.dot(j0, c1), wp.dot(j0, c2))
    p1 = wp.vec3(wp.dot(j1, c0), wp.dot(j1, c1), wp.dot(j1, c2))
    b0 = wp.vec3(wp.dot(p0, r0), wp.dot(p0, r1), wp.dot(p0, r2))
    b1 = wp.vec3(wp.dot(p1, r0), wp.dot(p1, r1), wp.dot(p1, r2))
    scales_squared = wp.vec3(wp.exp(2.0 * log_scale[0]), wp.exp(2.0 * log_scale[1]), wp.exp(2.0 * log_scale[2]))
    b0_squared = wp.vec3(b0[0] * b0[0], b0[1] * b0[1], b0[2] * b0[2])
    b0_b1 = wp.vec3(b0[0] * b1[0], b0[1] * b1[1], b0[2] * b1[2])
    b1_squared = wp.vec3(b1[0] * b1[0], b1[1] * b1[1], b1[2] * b1[2])
    a = wp.dot(scales_squared, b0_squared) + FILTER_VARIANCE
    b = wp.dot(scales_squared, b0_b1)
    c = wp.dot(scales_squared, b1_squared) + FILTER_VARIANCE
    return wp.vec3(a, b, c)


@wp.func
def conic_at_centre(log_scale: wp.vec3, quaternion: wp.vec4, camera: wp.mat44, x: float, y: float, z: float, focal: float):
    covariance = projected_covariance(log_scale, quaternion, camera, x, y, z, focal)
    a, b, c = covariance[0], covariance[1], covariance[2]
    determinant = wp.max(a * c - b * b, 1.0e-8)
    return wp.vec3(c / determinant, -b / determinant, a / determinant)


@wp.kernel(enable_backward=False)
def project_and_count_tile_pairs(
    means: wp.array(dtype=wp.vec3),
    log_scales: wp.array(dtype=wp.vec3),
    quaternions: wp.array(dtype=wp.vec4),
    opacity_logits: wp.array(dtype=wp.float32),
    active: wp.array(dtype=wp.int32),
    cameras: wp.array(dtype=wp.mat44),
    view_ids: wp.array(dtype=wp.int32),
    capacity: int,
    width: int,
    tiles_x: int,
    tile: int,
    focal: float,
    view_count: int,
    compact_box_enabled: int,
    compact_box_beta: float,
    compact_box_alpha_min: float,
    tile_bounds: wp.array(dtype=wp.vec4i),
    depths: wp.array(dtype=wp.float32),
    pair_counts: wp.array(dtype=wp.int32),
):
    thread = wp.tid()
    batch_view = thread // capacity
    splat = thread - batch_view * capacity
    pair_counts[thread] = 0
    tile_bounds[thread] = wp.vec4i(0, -1, 0, -1)
    depths[thread] = 0.0

    if batch_view < view_count and active[splat] != 0:
        camera = cameras[view_ids[batch_view]]
        mean = means[splat]
        point = camera * wp.vec4(mean[0], mean[1], mean[2], 1.0)
        z = point[2]
        if z > NEAR_PLANE:
            covariance = projected_covariance(
                log_scales[splat], quaternions[splat], camera,
                point[0], point[1], z, focal,
            )
            support = float(SUPPORT_RADIUS_SQUARED)
            if compact_box_enabled != 0:
                opacity = 1.0 / (1.0 + wp.exp(-opacity_logits[splat]))
                support = 0.0
                if opacity > compact_box_alpha_min:
                    support = compact_box_beta * 2.0 * wp.log(opacity / compact_box_alpha_min)
            if support > 0.0:
                radius_scale = wp.sqrt(support)
                radius_x = radius_scale * wp.sqrt(wp.max(covariance[0], 0.0))
                radius_y = radius_scale * wp.sqrt(wp.max(covariance[2], 0.0))
                centre_x = focal * point[0] / z + 0.5 * float(width)
                centre_y = focal * point[1] / z + 0.5 * float(width)
                if (
                    wp.isfinite(centre_x)
                    and wp.isfinite(centre_y)
                    and wp.isfinite(radius_x)
                    and wp.isfinite(radius_y)
                ):
                    min_x = int(wp.floor((centre_x - radius_x) / float(tile)))
                    max_x = int(wp.floor((centre_x + radius_x) / float(tile)))
                    min_y = int(wp.floor((centre_y - radius_y) / float(tile)))
                    max_y = int(wp.floor((centre_y + radius_y) / float(tile)))
                    if max_x >= 0 and max_y >= 0 and min_x < tiles_x and min_y < tiles_x:
                        min_x = wp.max(min_x, 0)
                        max_x = wp.min(max_x, tiles_x - 1)
                        min_y = wp.max(min_y, 0)
                        max_y = wp.min(max_y, tiles_x - 1)
                        tile_bounds[thread] = wp.vec4i(min_x, max_x, min_y, max_y)
                        depths[thread] = z
                        pair_counts[thread] = (max_x - min_x + 1) * (max_y - min_y + 1)


@wp.kernel(enable_backward=False)
def copy_pair_count(
    pair_prefix: wp.array(dtype=wp.int32),
    item_count: int,
    pair_count: wp.array(dtype=wp.int32),
):
    pair_count[0] = pair_prefix[item_count - 1]


@wp.kernel(enable_backward=False)
def write_tile_pairs(
    tile_bounds: wp.array(dtype=wp.vec4i),
    depths: wp.array(dtype=wp.float32),
    pair_counts: wp.array(dtype=wp.int32),
    pair_prefix: wp.array(dtype=wp.int32),
    capacity: int,
    tiles_per_view: int,
    tiles_x: int,
    depth_keys: wp.array(dtype=wp.float32),
    packed_pairs: wp.array(dtype=wp.uint64),
):
    thread = wp.tid()
    count = pair_counts[thread]
    if count > 0:
        batch_view = thread // capacity
        splat = thread - batch_view * capacity
        bounds = tile_bounds[thread]
        output = pair_prefix[thread] - count
        for tile_y in range(bounds[2], bounds[3] + 1):
            for tile_x in range(bounds[0], bounds[1] + 1):
                record = batch_view * tiles_per_view + tile_y * tiles_x + tile_x
                depth_keys[output] = depths[thread]
                packed_pairs[output] = (
                    wp.uint64(record) << wp.uint64(32)
                ) | wp.uint64(splat)
                output += 1


@wp.kernel(enable_backward=False)
def unpack_group_keys(
    packed_pairs: wp.array(dtype=wp.uint64),
    group_keys: wp.array(dtype=wp.uint32),
):
    item = wp.tid()
    group_keys[item] = wp.uint32(packed_pairs[item] >> wp.uint64(32))


@wp.kernel(enable_backward=False)
def count_sorted_groups(
    group_keys: wp.array(dtype=wp.uint32),
    group_counts: wp.array(dtype=wp.int32),
):
    item = wp.tid()
    wp.atomic_add(group_counts, int(group_keys[item]), 1)


@wp.func
def alpha_at_pixel(
    mean: wp.vec3,
    log_scale: wp.vec3,
    quaternion: wp.vec4,
    opacity_logit: float,
    camera: wp.mat44,
    px: float,
    py: float,
    width: float,
    focal: float,
):
    point = camera * wp.vec4(mean[0], mean[1], mean[2], 1.0)
    z = wp.max(point[2], NEAR_PLANE)
    centre_x = focal * point[0] / z + 0.5 * width
    centre_y = focal * point[1] / z + 0.5 * width
    conic = conic_at_centre(log_scale, quaternion, camera, point[0], point[1], z, focal)
    dx, dy = px - centre_x, py - centre_y
    q = conic[0] * dx * dx + 2.0 * conic[1] * dx * dy + conic[2] * dy * dy
    alpha = float(0.0)
    if q <= SUPPORT_RADIUS_SQUARED:
        candidate = wp.min(1.0 / (1.0 + wp.exp(-opacity_logit)) * wp.exp(-0.5 * q), 0.99)
        if candidate >= ALPHA_CUTOFF:
            alpha = candidate
    return alpha


@wp.func
def colour_at_view(
    sh0: wp.vec3,
):
    return wp.vec3(
        wp.clamp(sh0[0], 0.0, 1.0),
        wp.clamp(sh0[1], 0.0, 1.0),
        wp.clamp(sh0[2], 0.0, 1.0),
    )


@wp.kernel
def render_forward(
    means: wp.array(dtype=wp.vec3),
    log_scales: wp.array(dtype=wp.vec3),
    quaternions: wp.array(dtype=wp.vec4),
    opacity_logits: wp.array(dtype=wp.float32),
    sh0: wp.array(dtype=wp.vec3),
    cameras: wp.array(dtype=wp.mat44),
    targets: wp.array(dtype=wp.vec3),
    view_ids: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    pairs: wp.array(dtype=wp.uint64),
    width: int,
    tiles_x: int,
    tile: int,
    focal: float,
    view_count: int,
    background: wp.vec3,
    image: wp.array(dtype=wp.vec3),
    loss: wp.array(dtype=wp.float32),
):
    thread = wp.tid()
    pixels = width * width
    batch_view = thread // pixels
    view = view_ids[batch_view]
    pixel = thread - batch_view * pixels
    px = float(pixel % width) + 0.5
    py = float(pixel // width) + 0.5
    record = batch_view * tiles_x * tiles_x + (pixel // width // tile) * tiles_x + (pixel % width // tile)
    rgb = wp.vec3(0.0, 0.0, 0.0)
    transmittance = float(1.0)
    for entry in range(offsets[record], offsets[record + 1]):
        splat = int(pairs[entry] & wp.uint64(0xFFFFFFFF))
        alpha = alpha_at_pixel(means[splat], log_scales[splat], quaternions[splat],
                               opacity_logits[splat], cameras[view], px, py,
                               float(width), focal)
        if alpha > 0.0:
            colour = colour_at_view(sh0[splat])
            rgb = rgb + transmittance * alpha * colour
            transmittance = transmittance * (1.0 - alpha)
            if transmittance < TRANSMITTANCE_CUTOFF:
                break
    rgb = rgb + transmittance * background
    image[thread] = rgb
    difference = rgb - targets[view * pixels + pixel]
    wp.atomic_add(loss, 0, wp.dot(difference, difference) / float(3 * pixels * view_count))


@wp.kernel(enable_backward=False)
def render_backward(
    means: wp.array(dtype=wp.vec3),
    log_scales: wp.array(dtype=wp.vec3),
    quaternions: wp.array(dtype=wp.vec4),
    opacity_logits: wp.array(dtype=wp.float32),
    sh0: wp.array(dtype=wp.vec3),
    cameras: wp.array(dtype=wp.mat44),
    targets: wp.array(dtype=wp.vec3),
    view_ids: wp.array(dtype=wp.int32),
    offsets: wp.array(dtype=wp.int32),
    pairs: wp.array(dtype=wp.uint64),
    width: int,
    tiles_x: int,
    tile: int,
    focal: float,
    view_count: int,
    image: wp.array(dtype=wp.vec3),
    mean_grad_flat: wp.array(dtype=wp.float32),
    scale_grad_flat: wp.array(dtype=wp.float32),
    quaternion_grad_flat: wp.array(dtype=wp.float32),
    opacity_grad: wp.array(dtype=wp.float32),
    sh0_grad_flat: wp.array(dtype=wp.float32),
):
    thread = wp.tid()
    pixels = width * width
    batch_view = thread // pixels
    view = view_ids[batch_view]
    pixel = thread - batch_view * pixels
    px = float(pixel % width) + 0.5
    py = float(pixel // width) + 0.5
    record = batch_view * tiles_x * tiles_x + (pixel // width // tile) * tiles_x + (pixel % width // tile)

    # This is an explicit adjoint of front-to-back compositing. It is an ordinary forward
    # execution of a runtime loop, so Warp never has to reverse a variable-length loop.
    for entry in range(offsets[record], offsets[record + 1]):
        splat = int(pairs[entry] & wp.uint64(0xFFFFFFFF))
        mean = means[splat]
        log_scale = log_scales[splat]
        quaternion = quaternions[splat]
        opacity_logit = opacity_logits[splat]
        camera = cameras[view]
        # TODO: compute mean_grad_flat, scale_grad_flat, quaternion_grad_flat, and opacity_grad

@wp.kernel
def pack_vec3_gradient(flat: wp.array(dtype=wp.float32), packed: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    packed[i] = wp.vec3(flat[i * 3], flat[i * 3 + 1], flat[i * 3 + 2])


@wp.kernel
def adam_quaternion_step(
    quaternions: wp.array(dtype=wp.vec4),
    gradients: wp.array(dtype=wp.float32),
    m: wp.array(dtype=wp.vec4),
    v: wp.array(dtype=wp.vec4),
    lr: float,
    beta1: float,
    beta2: float,
    t: float,
    eps: float,
):
    # Same update as warp.optim.Adam's own float/vec3 kernels, generalized to four components
    # because Adam does not support wp.vec4 parameters directly.
    i = wp.tid()
    g = wp.vec4(gradients[i * 4], gradients[i * 4 + 1], gradients[i * 4 + 2], gradients[i * 4 + 3])
    m[i] = beta1 * m[i] + (1.0 - beta1) * g
    v[i] = beta2 * v[i] + (1.0 - beta2) * wp.cw_mul(g, g)
    mhat = m[i] / (1.0 - wp.pow(beta1, t + 1.0))
    vhat = v[i] / (1.0 - wp.pow(beta2, t + 1.0))
    sqrt_vhat = wp.vec4(wp.sqrt(vhat[0]), wp.sqrt(vhat[1]), wp.sqrt(vhat[2]), wp.sqrt(vhat[3]))
    eps_vec4 = wp.vec4(eps, eps, eps, eps)
    q = quaternions[i] - lr * wp.cw_div(mhat, (sqrt_vhat + eps_vec4))
    quaternions[i] = q / wp.sqrt(wp.dot(q, q) + 1.0e-8)


@wp.kernel
def constrain_parameters(
    log_scales: wp.array(dtype=wp.vec3),
    sh0: wp.array(dtype=wp.vec3),
    minimum_log_scale: float,
    maximum_log_scale: float,
):
    i = wp.tid()
    scale = log_scales[i]
    log_scales[i] = wp.vec3(
        wp.clamp(scale[0], minimum_log_scale, maximum_log_scale),
        wp.clamp(scale[1], minimum_log_scale, maximum_log_scale),
        wp.clamp(scale[2], minimum_log_scale, maximum_log_scale),
    )
    colour = sh0[i]
    sh0[i] = wp.vec3(
        wp.clamp(colour[0], 0.0, 1.0),
        wp.clamp(colour[1], 0.0, 1.0),
        wp.clamp(colour[2], 0.0, 1.0),
    )


class TileWorkspace:
    """Persistent GPU buffers for building compact, depth-ordered tile records.

    Allocated once for the largest batch and pair count the trainer will ever request, then
    reused every step -- rebuilding these from scratch every iteration would dominate the time
    budget. The radix sort needs the second half of each key/value array as scratch space,
    hence the factor of two on the pair-capacity buffers. See Unit 10, Stage 2.
    """

    def __init__(self, capacity, max_views_per_batch, tiles_x, tile_pair_capacity, device):
        self.capacity = capacity
        self.max_views_per_batch = max_views_per_batch
        self.tiles_x = tiles_x
        self.tiles_per_view = tiles_x * tiles_x
        self.tile_pair_capacity = tile_pair_capacity
        self.device = device

        tile_work_items = max_views_per_batch * capacity
        tile_record_count = max_views_per_batch * self.tiles_per_view
        self.tile_bounds = wp.zeros(tile_work_items, dtype=wp.vec4i, device=device)
        self.depths = wp.zeros(tile_work_items, dtype=wp.float32, device=device)
        self.pair_counts = wp.zeros(tile_work_items, dtype=wp.int32, device=device)
        self.pair_prefix = wp.zeros(tile_work_items, dtype=wp.int32, device=device)
        self.pair_count = wp.zeros(1, dtype=wp.int32, device=device)
        self.depth_keys = wp.zeros(2 * tile_pair_capacity, dtype=wp.float32, device=device)
        self.packed_pairs = wp.zeros(2 * tile_pair_capacity, dtype=wp.uint64, device=device)
        self.group_keys = wp.zeros(2 * tile_pair_capacity, dtype=wp.uint32, device=device)
        self.group_counts = wp.zeros(tile_record_count, dtype=wp.int32, device=device)
        self.tile_offsets = wp.zeros(tile_record_count + 1, dtype=wp.int32, device=device)
        self.view_ids_host = np.zeros(max_views_per_batch, np.int32)
        self.view_ids = wp.zeros(max_views_per_batch, dtype=wp.int32, device=device)

    def build(self, means, log_scales, quaternions, opacity, active_device, cameras,
              view_ids, width, focal, compact_box):
        """Build compact, depth-ordered tile records for `view_ids`.

        Returns (offsets, packed_pairs, pair_count): `offsets[r]` and `offsets[r + 1]` give the
        half-open range of `packed_pairs` entries for tile record `r`, already ordered
        near-to-far within each tile.
        """
        view_ids = np.asarray(view_ids, np.int32)
        view_count = len(view_ids)
        if view_count < 1 or view_count > self.max_views_per_batch:
            raise ValueError("View batch exceeds the persistent tile-workspace capacity.")

        self.view_ids_host.fill(0)
        self.view_ids_host[:view_count] = view_ids
        self.view_ids.assign(self.view_ids_host)
        work_items = view_count * self.capacity
        record_count = view_count * self.tiles_per_view

        wp.launch(
            project_and_count_tile_pairs,
            dim=work_items,
            inputs=[
                means, log_scales, quaternions, opacity,
                active_device,
                cameras, self.view_ids, self.capacity, width, self.tiles_x,
                TILE, focal, view_count,
                int(compact_box["enabled"]),
                float(compact_box["beta"]),
                float(compact_box["alpha_min"]),
            ],
            outputs=[self.tile_bounds, self.depths, self.pair_counts],
            device=self.device,
        )
        wp.utils.array_scan(
            self.pair_counts[:work_items], self.pair_prefix[:work_items], inclusive=True
        )
        wp.launch(
            copy_pair_count,
            dim=1,
            inputs=[self.pair_prefix, work_items],
            outputs=[self.pair_count],
            device=self.device,
        )
        pair_count = int(self.pair_count.numpy()[0])
        if pair_count > self.tile_pair_capacity:
            raise RuntimeError(
                f"GPU tile-pair buffer capacity {self.tile_pair_capacity:,} is too small: "
                f"this batch requires {pair_count:,} splat-to-tile pairs. Increase "
                "model.tile_pair_capacity in the YAML configuration. Very large projected "
                "splats can also indicate bad scales or cameras."
            )

        if pair_count > 0:
            wp.launch(
                write_tile_pairs,
                dim=work_items,
                inputs=[
                    self.tile_bounds, self.depths, self.pair_counts, self.pair_prefix,
                    self.capacity, self.tiles_per_view, self.tiles_x,
                ],
                outputs=[self.depth_keys, self.packed_pairs],
                device=self.device,
            )
            # First sort by exact floating-point depth. The second stable sort groups by
            # tile while preserving that near-to-far order inside each tile.
            wp.utils.radix_sort_pairs(self.depth_keys, self.packed_pairs, pair_count)
            wp.launch(
                unpack_group_keys,
                dim=pair_count,
                inputs=[self.packed_pairs],
                outputs=[self.group_keys],
                device=self.device,
            )
            wp.utils.radix_sort_pairs(
                self.group_keys,
                self.packed_pairs,
                pair_count,
                end_bit=max(1, (record_count - 1).bit_length()),
            )

        self.group_counts.zero_()
        self.tile_offsets.zero_()
        if pair_count > 0:
            wp.launch(
                count_sorted_groups,
                dim=pair_count,
                inputs=[self.group_keys],
                outputs=[self.group_counts],
                device=self.device,
            )
        wp.utils.array_scan(
            self.group_counts[:record_count],
            self.tile_offsets[1:record_count + 1],
            inclusive=True,
        )
        wp.synchronize_device(self.device)
        return self.tile_offsets, self.packed_pairs, pair_count


class SplatOptimizers:
    """Adam state for every trainable splat parameter, plus the explicit gradient buffers the
    hand-written backward pass writes into.

    The quaternion gets its own hand-written Adam step (`adam_quaternion_step`) with its own
    moment buffers and step count, since `warp.optim.Adam` does not accept `vec4` parameters.
    Every gradient produced by the backward pass is consumed by Adam in the same iteration.
    """

    def __init__(self, trainable, capacity, learning_rates, device):
        self.device = device
        self.learning_rates = learning_rates
        self.quaternions = trainable.quaternions

        self.mean_optimizer = Adam([trainable.means], lr=learning_rates["position_initial"])
        self.scale_optimizer = Adam([trainable.log_scales], lr=learning_rates["scale"])
        self.opacity_optimizer = Adam([trainable.opacity_logits], lr=learning_rates["opacity"])
        self.sh0_optimizer = Adam([trainable.colors], lr=learning_rates["feature_dc"])
        self.optimizers = (
            self.mean_optimizer,
            self.scale_optimizer,
            self.opacity_optimizer,
            self.sh0_optimizer,
        )
        self.quaternion_m = wp.zeros(capacity, dtype=wp.vec4, device=device)
        self.quaternion_v = wp.zeros(capacity, dtype=wp.vec4, device=device)
        self.quaternion_adam_t = 0

        self.mean_grad_flat = wp.zeros(capacity * 3, dtype=wp.float32, device=device)
        self.scale_grad_flat = wp.zeros(capacity * 3, dtype=wp.float32, device=device)
        self.quaternion_grad_flat = wp.zeros(capacity * 4, dtype=wp.float32, device=device)
        self.opacity_grad = wp.zeros(capacity, dtype=wp.float32, device=device)
        self.sh0_grad_flat = wp.zeros(capacity * 3, dtype=wp.float32, device=device)
        self.mean_grad = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.scale_grad = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.sh0_grad = wp.zeros(capacity, dtype=wp.vec3, device=device)

    def zero_gradients(self):
        for gradient in (self.mean_grad_flat, self.scale_grad_flat, self.quaternion_grad_flat,
                         self.opacity_grad, self.sh0_grad_flat):
            gradient.zero_()

    def pack_gradients(self, capacity):
        """Repack the flat per-component gradient buffers `render_backward` writes into
        vec3 arrays, for use as Adam inputs and as the densification statistic."""
        for flat, packed in ((self.mean_grad_flat, self.mean_grad),
                             (self.scale_grad_flat, self.scale_grad),
                             (self.sh0_grad_flat, self.sh0_grad)):
            wp.launch(pack_vec3_gradient, dim=capacity, inputs=[flat], outputs=[packed],
                      device=self.device)

    def apply(self, position_lr, capacity):
        """Apply one Adam update using the gradients from the current render."""
        self.mean_optimizer.lr = position_lr
        self.mean_optimizer.step([self.mean_grad])
        self.scale_optimizer.step([self.scale_grad])
        self.opacity_optimizer.step([self.opacity_grad])
        self.sh0_optimizer.step([self.sh0_grad])
        wp.launch(
            adam_quaternion_step,
            dim=capacity,
            inputs=[self.quaternions, self.quaternion_grad_flat,
                    self.quaternion_m, self.quaternion_v,
                    self.learning_rates["rotation"], ADAM_BETA1, ADAM_BETA2,
                    float(self.quaternion_adam_t), ADAM_EPS],
            device=self.device,
        )
        self.quaternion_adam_t += 1

    def reset(self):
        """Clear all Adam moments after densify/prune changes which parameters exist. A clone
        or split child needs fresh optimizer state -- see Unit 10, Stage 8."""
        for optimizer in self.optimizers:
            optimizer.reset_internal_state()
        self.quaternion_m.zero_()
        self.quaternion_v.zero_()
        self.quaternion_adam_t = 0

    def reset_opacity(self):
        """Clear only the opacity optimizer's moments, for a periodic opacity ceiling reset."""
        self.opacity_optimizer.reset_internal_state()


class WarpImageTrainer:
    """Warp GPU tiler and differentiable renderer plus a Python training schedule."""

    def __init__(self, targets, cameras, focal, scene_radius, capacity, initial_splats,
                 device, seed, init_scene=None, background=(1.0, 1.0, 1.0),
                 learning_rates=None, tile_pair_capacity=None, training_options=None):
        self.focal = float(focal)
        self.width, self.capacity, self.device = targets.shape[1], capacity, wp.get_device(device)
        # Training always uses one randomly selected camera. Evaluation also renders one camera
        # at a time, so every image buffer can have a simple, fixed batch dimension of one.
        self.max_views_per_batch = 1
        self.learning_rates = deepcopy(TrainingConfig.DEFAULTS["learning_rates"])
        if learning_rates is not None:
            self.learning_rates.update(learning_rates)
        self.training_options = {
            "compact_box": deepcopy(TrainingConfig.DEFAULTS["compact_box"]),
            "multiview_adc": deepcopy(TrainingConfig.DEFAULTS["multiview_adc"]),
        }
        if training_options is not None:
            for section, values in training_options.items():
                self.training_options[section].update(values)
        self.last_position_learning_rate = self.learning_rates["position_initial"]
        self.background = wp.vec3(float(background[0]), float(background[1]), float(background[2]))
        self.tiles_x = (self.width + TILE - 1) // TILE
        self.tiles_per_view = self.tiles_x * self.tiles_x
        self.tile_pair_capacity = int(
            tile_pair_capacity
            if tile_pair_capacity is not None
            else 16 * capacity * self.max_views_per_batch
        )
        if not 1 <= self.tile_pair_capacity <= np.iinfo(np.int32).max:
            raise ValueError(
                "Tile-pair capacity must be positive and fit in a signed 32-bit prefix sum."
            )
        self.rng = np.random.default_rng(seed)
        self.scene_radius = scene_radius

        self._init_splats(capacity, initial_splats, init_scene, targets)
        self._init_buffers(cameras, targets, capacity)

    def _init_splats(self, capacity, initial_splats, init_scene, targets):
        """Allocate and populate every trainable per-splat parameter array."""
        means = np.zeros((capacity, 3), np.float32)
        log_scales = np.full((capacity, 3), np.log(self.scene_radius / 80.0), np.float32)
        quaternions = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (capacity, 1))
        opacity = np.full(capacity, INACTIVE_LOGIT, np.float32)
        colours = np.zeros((capacity, 3), np.float32)
        count = initial_splats
        if init_scene is not None:
            count = min(len(init_scene.means), capacity)
            means[:count], log_scales[:count] = init_scene.means[:count], np.log(np.maximum(init_scene.scales[:count], 1e-7))
            quaternions[:count], colours[:count] = init_scene.rotations[:count], init_scene.colors[:count]
            opacity[:count] = np.log(np.clip(init_scene.opacities[:count], 1e-5, 1 - 1e-5) / np.clip(1 - init_scene.opacities[:count], 1e-5, 1))
        else:
            means[:count] = self.rng.uniform(-self.scene_radius, self.scene_radius, (count, 3))
            colours[:count] = targets.mean(axis=(0, 1, 2))
            opacity[:count] = -4.0
        self.active = np.zeros(capacity, bool)
        self.active[:count] = True
        self.active_device = wp.array(self.active.astype(np.int32), dtype=wp.int32, device=self.device)
        self.trainable = TrainableGaussianSet(
            means, log_scales, quaternions, opacity, colours,
            self.device, requires_grad=False,
        )
        self.means = self.trainable.means
        self.log_scales = self.trainable.log_scales
        self.quaternions = self.trainable.quaternions
        self.opacity = self.trainable.opacity_logits
        self.sh0 = self.trainable.colors

    def _init_buffers(self, cameras, targets, capacity):
        """Allocate camera/target uploads, the tile workspace, optimizers, and render buffers."""
        self.cameras = wp.array(cameras, dtype=wp.mat44, device=self.device)
        self.targets = wp.array(targets.reshape(-1, 3), dtype=wp.vec3, device=self.device)
        self.tiles = TileWorkspace(
            capacity, self.max_views_per_batch, self.tiles_x, self.tile_pair_capacity, self.device,
        )
        self.optimizers = SplatOptimizers(
            self.trainable, capacity, self.learning_rates, self.device,
        )
        self.last_pair_count = 0
        self.image = wp.empty(
            self.max_views_per_batch * self.width * self.width,
            dtype=wp.vec3,
            device=self.device,
        )
        self.loss = wp.zeros(1, dtype=wp.float32, device=self.device)
        self.last_timings = {"tile": 0.0, "device": 0.0}

    def build_tiles(self, view_ids):
        """Build compact, depth-ordered tile records on the selected Warp device."""
        return self.tiles.build(
            self.means, self.log_scales, self.quaternions, self.opacity, self.active_device,
            self.cameras, view_ids, self.width, self.focal, self.training_options["compact_box"],
        )

    def loss_and_gradients(self, view_ids):
        if len(view_ids) > self.max_views_per_batch:
            raise ValueError("View batch exceeds the reusable image-buffer capacity.")
        started = perf_counter()
        offsets, pairs, pair_count = self.build_tiles(view_ids)
        self.last_pair_count = pair_count
        tile_seconds = perf_counter() - started
        device_started = perf_counter()
        self.loss.zero_()
        self.optimizers.zero_gradients()
        wp.launch(
            render_forward,
            dim=len(view_ids) * self.width * self.width,
            inputs=[self.means, self.log_scales, self.quaternions, self.opacity, self.sh0,
                    self.cameras, self.targets, self.tiles.view_ids, offsets, pairs,
                    self.width, self.tiles_x,
                    TILE, self.focal, len(view_ids), self.background],
            outputs=[self.image, self.loss],
            device=self.device,
        )
        wp.launch(
            render_backward,
            dim=len(view_ids) * self.width * self.width,
            inputs=[self.means, self.log_scales, self.quaternions, self.opacity, self.sh0,
                    self.cameras, self.targets, self.tiles.view_ids, offsets, pairs,
                    self.width, self.tiles_x,
                    TILE, self.focal, len(view_ids), self.image],
            outputs=[self.optimizers.mean_grad_flat, self.optimizers.scale_grad_flat,
                     self.optimizers.quaternion_grad_flat, self.optimizers.opacity_grad,
                     self.optimizers.sh0_grad_flat],
            device=self.device,
        )
        self.optimizers.pack_gradients(self.capacity)
        loss = float(self.loss.numpy()[0])
        self.last_timings = {"tile": tile_seconds, "device": perf_counter() - device_started}
        return loss

    def evaluate(self, view_ids):
        """Evaluate fixed views one at a time while reusing one persistent tile workspace."""
        view_ids = np.asarray(view_ids, np.int32)
        if len(view_ids) < 1:
            raise ValueError("Evaluation requires at least one view.")

        # Render the first fixed camera last so self.image retains the snapshot view.
        ordered_view_ids = np.concatenate((view_ids[1:], view_ids[:1]))
        total_loss = 0.0
        for view_id in ordered_view_ids:
            offsets, pairs, _ = self.build_tiles(np.asarray([view_id], np.int32))
            self.loss.zero_()
            wp.launch(
                render_forward,
                dim=self.width * self.width,
                inputs=[self.means, self.log_scales, self.quaternions, self.opacity, self.sh0,
                        self.cameras, self.targets, self.tiles.view_ids, offsets, pairs,
                        self.width, self.tiles_x, TILE, self.focal, 1,
                        self.background],
                outputs=[self.image, self.loss],
                device=self.device,
            )
            total_loss += float(self.loss.numpy()[0])
        return total_loss / len(view_ids)

    def render_single_view_image(self, view_id):
        offsets, pairs, _ = self.build_tiles(np.asarray([view_id], np.int32))
        self.loss.zero_()
        wp.launch(
            render_forward,
            dim=self.width * self.width,
            inputs=[self.means, self.log_scales, self.quaternions, self.opacity, self.sh0,
                    self.cameras, self.targets, self.tiles.view_ids, offsets, pairs,
                    self.width, self.tiles_x, TILE, self.focal, 1,
                    self.background],
            outputs=[self.image, self.loss],
            device=self.device,
        )
        return (
            self.image.numpy()[: self.width * self.width].reshape(self.width, self.width, 3),
            float(self.loss.numpy()[0]),
        )

    def multiview_adc_scores(self, candidate_indices, view_ids):
        """Reference multi-view ADC scorer for a bounded set of candidate splats."""
        candidate_indices = np.asarray(candidate_indices, np.int32)
        if len(candidate_indices) == 0:
            return np.zeros(0, np.float32), np.zeros(0, np.float32)

        adc = self.training_options["multiview_adc"]
        means = self.means.numpy()[candidate_indices]
        scales = np.exp(self.log_scales.numpy()[candidate_indices])
        quaternions = self.quaternions.numpy()[candidate_indices]
        cameras = self.cameras.numpy()
        targets = self.targets.numpy().reshape(-1, self.width, self.width, 3)
        compact = self.training_options["compact_box"]
        opacity = sigmoid(self.opacity.numpy()[candidate_indices])
        support = np.full(len(candidate_indices), 9.0, np.float32)
        if compact["enabled"]:
            support.fill(0.0)
            visible_opacity = opacity > compact["alpha_min"]
            support[visible_opacity] = (
                compact["beta"] * 2.0 * np.log(opacity[visible_opacity] / compact["alpha_min"])
            )

        densify_score = np.zeros(len(candidate_indices), np.float32)
        low_error_score = np.zeros(len(candidate_indices), np.float32)
        for view_id in view_ids:
            image, _ = self.render_single_view_image(int(view_id))
            error = np.mean(np.abs(image - targets[int(view_id)]), axis=2)
            error_min = float(error.min())
            error_range = float(error.max() - error_min)
            normalized_error = np.zeros_like(error) if error_range < 1.0e-8 else (error - error_min) / error_range
            high_error = normalized_error > adc["loss_threshold"]
            high_error_integral = high_error.astype(np.int32).cumsum(axis=0).cumsum(axis=1)
            low_error_integral = (~high_error).astype(np.int32).cumsum(axis=0).cumsum(axis=1)

            camera = cameras[int(view_id)]
            homo = np.concatenate((means, np.ones((len(means), 1), np.float32)), axis=1)
            camera_points = homo @ camera.T
            visible = (camera_points[:, 2] > NEAR_PLANE) & (support > 0.0)
            if not np.any(visible):
                continue
            variance_x, variance_y = projected_covariance_diagonal(
                scales[visible], quaternions[visible], camera, camera_points[visible, :3],
                self.focal,
            )
            centres = np.empty((int(np.count_nonzero(visible)), 2), np.float32)
            centres[:, 0] = self.focal * camera_points[visible, 0] / camera_points[visible, 2] + 0.5 * self.width
            centres[:, 1] = self.focal * camera_points[visible, 1] / camera_points[visible, 2] + 0.5 * self.width
            radius_scale = np.sqrt(support[visible])
            min_x = np.floor(centres[:, 0] - radius_scale * np.sqrt(np.maximum(variance_x, 0.0))).astype(np.int32)
            max_x = np.floor(centres[:, 0] + radius_scale * np.sqrt(np.maximum(variance_x, 0.0))).astype(np.int32)
            min_y = np.floor(centres[:, 1] - radius_scale * np.sqrt(np.maximum(variance_y, 0.0))).astype(np.int32)
            max_y = np.floor(centres[:, 1] + radius_scale * np.sqrt(np.maximum(variance_y, 0.0))).astype(np.int32)
            in_image = (max_x >= 0) & (max_y >= 0) & (min_x < self.width) & (min_y < self.width)
            visible_indices = np.flatnonzero(visible)
            for local, x0, x1, y0, y1 in zip(
                visible_indices[in_image],
                np.clip(min_x[in_image], 0, self.width - 1),
                np.clip(max_x[in_image], 0, self.width - 1),
                np.clip(min_y[in_image], 0, self.width - 1),
                np.clip(max_y[in_image], 0, self.width - 1),
            ):
                total = high_error_integral[y1, x1]
                if x0 > 0:
                    total -= high_error_integral[y1, x0 - 1]
                if y0 > 0:
                    total -= high_error_integral[y0 - 1, x1]
                if x0 > 0 and y0 > 0:
                    total += high_error_integral[y0 - 1, x0 - 1]
                low_total = low_error_integral[y1, x1]
                if x0 > 0:
                    low_total -= low_error_integral[y1, x0 - 1]
                if y0 > 0:
                    low_total -= low_error_integral[y0 - 1, x1]
                if x0 > 0 and y0 > 0:
                    low_total += low_error_integral[y0 - 1, x0 - 1]
                densify_score[local] += total
                # A high residual identifies a place where more geometry may be needed, so it
                # must not increase the pruning score. The low-error score is only a simple
                # redundancy heuristic; opacity remains the primary pruning signal.
                low_error_score[local] += low_total

        densify_score /= max(1, len(view_ids))
        if low_error_score.max() > low_error_score.min():
            prune_score = (low_error_score - low_error_score.min()) / (
                low_error_score.max() - low_error_score.min()
            )
        else:
            prune_score = np.zeros_like(low_error_score)
        return densify_score, prune_score

    def position_learning_rate(self, iteration):
        rates = self.learning_rates
        return exponential_learning_rate(
            iteration,
            rates["position_initial"],
            rates["position_final"],
            rates["position_max_steps"],
            rates["position_delay_steps"],
            rates["position_delay_multiplier"],
        )

    def step(self, view_ids, download_gradients=False, iteration=1):
        loss = self.loss_and_gradients(view_ids)
        device_started = perf_counter()
        self.last_position_learning_rate = self.position_learning_rate(iteration)
        self.optimizers.apply(self.last_position_learning_rate, self.capacity)
        wp.launch(
            constrain_parameters,
            dim=self.capacity,
            inputs=[
                self.log_scales, self.sh0,
                np.log(self.scene_radius * SCALE_CLAMP_MIN_FRACTION),
                np.log(self.scene_radius * SCALE_CLAMP_MAX_FRACTION),
            ],
            device=self.device,
        )
        mean_gradients = self.optimizers.mean_grad.numpy() if download_gradients else None
        wp.synchronize_device(self.device)
        self.last_timings["device"] += perf_counter() - device_started
        return loss, mean_gradients

    def reset_opacity(self, ceiling=0.01):
        """Cap every active splat's opacity so faded splats must re-earn it or be pruned.

        Splats only ever lose opacity through the loss, so without this reset a low-opacity
        floater can sit forever just above the prune threshold. Adam's opacity state is reset
        because the parameter it was tracking has changed discontinuously.
        """
        opacity_logits = self.opacity.numpy()
        limit = float(np.log(ceiling / (1.0 - ceiling)))
        opacity_logits[self.active] = np.minimum(opacity_logits[self.active], limit)
        self.opacity.assign(
            wp.array(opacity_logits.astype(np.float32), dtype=wp.float32, device=self.device)
        )
        self.optimizers.reset_opacity()

    def densify_and_prune(self, gradients, fraction, view_ids=None, late_prune=False,
                          percent_dense=0.01):
        opacity = sigmoid(self.opacity.numpy())
        means, log_scales = self.means.numpy(), self.log_scales.numpy()
        quaternions, opacity_logits = self.quaternions.numpy(), self.opacity.numpy()
        sh0 = self.sh0.numpy()
        adc = self.training_options["multiview_adc"]
        prune_mask = (opacity < PRUNE_OPACITY_FLOOR) & self.active
        scores = np.linalg.norm(gradients, axis=1)
        parents = [i for i in np.argsort(scores)[::-1] if self.active[i]]
        if adc["enabled"] and view_ids is not None:
            limit = min(len(parents), int(adc["candidate_limit"]))
            scored = np.asarray(parents[:limit], np.int32)
            densify_score, prune_score = self.multiview_adc_scores(scored, view_ids)
            keep_for_densify = densify_score > adc["densify_score_threshold"]
            parents = scored[keep_for_densify].tolist()
            if late_prune:
                prune_mask[scored[prune_score > adc["prune_score_threshold"]]] = True
                prune_mask[(opacity < adc["opacity_cutoff"]) & self.active] = True
        elif late_prune:
            prune_mask[(opacity < adc["opacity_cutoff"]) & self.active] = True
        self.active[prune_mask] = False
        # A pruned splat must not stay a clone source: it now sits in `free`, so reusing it as
        # a parent would let a later clone read a slot already overwritten as a clone
        # destination in this same loop (means/etc. are mutated in place). Keep only still-active
        # parents so sources are always active and destinations always inactive -- no overlap.
        parents = [parent for parent in parents if self.active[parent]]
        free = np.flatnonzero(~self.active)
        children_to_add = 0 if fraction <= 0.0 else min(
            len(free), max(1, int(self.active.sum() * fraction))
        )
        # Split large splats, clone small ones -- the two distinct reference operations. A single
        # rule that always offsets by a fixed distance and always shrinks is wrong for both: the
        # offset stops being related to the splat's own extent, so each generation lands further
        # from the surface measured in its own radii, and the shrink makes that worse without
        # bound. Displacement here is sampled from the parent's own covariance instead.
        split_size = percent_dense * self.scene_radius
        for parent, child in zip(parents[:children_to_add], free[:children_to_add]):
            self.active[child] = True
            parent_scales = np.exp(log_scales[parent])
            if float(parent_scales.max()) > split_size:
                # Two smaller children replace one large parent; the parent's slot becomes the
                # first child, so the population still grows by exactly one splat.
                rotation = quaternion_to_rotation(quaternions[parent])
                origin = means[parent].copy()
                means[parent] = origin + rotation @ self.rng.normal(0.0, parent_scales)
                means[child] = origin + rotation @ self.rng.normal(0.0, parent_scales)
                log_scales[parent] = log_scales[parent] - np.log(1.6)
                log_scales[child] = log_scales[parent]
            else:
                # A small splat is copied exactly. The gradient separates the pair, so no
                # displacement is invented and no opacity is thrown away.
                means[child] = means[parent]
                log_scales[child] = log_scales[parent]
            quaternions[child] = quaternions[parent]
            opacity_logits[child] = opacity_logits[parent]
            sh0[child] = sh0[parent]
        self.active_device.assign(self.active.astype(np.int32))
        for array, values, dtype in (
            (self.means, means, wp.vec3),
            (self.log_scales, log_scales, wp.vec3),
            (self.quaternions, quaternions, wp.vec4),
            (self.opacity, opacity_logits, wp.float32),
            (self.sh0, sh0, wp.vec3),
        ):
            array.assign(wp.array(values.astype(np.float32), dtype=dtype, device=self.device))
        self.optimizers.reset()

    def gaussian_set(self):
        active = np.flatnonzero(self.active)
        return GaussianSet(
            self.means.numpy()[active],
            np.exp(self.log_scales.numpy()[active]),
            self.quaternions.numpy()[active],
            sigmoid(self.opacity.numpy()[active]),
            np.clip(self.sh0.numpy()[active], 0, 1),
        )


class ConvergenceTracker:
    """Tracks whether the fixed-evaluation loss has plateaued, for early stopping.

    Only a true `eval_every` tick advances this state -- a snapshot-only iteration recomputes
    `fixed_eval` for logging and rendering, but must not affect when training stops, or
    `snapshot_every` would silently reshape the stopping schedule. See Unit 10, Stage 10.
    """

    def __init__(self, config, initial_loss):
        self.enabled = config["enabled"]
        self.min_iterations = config["min_iterations"]
        self.patience = config["patience"]
        self.min_delta = config["min_delta"]
        self.best_loss = initial_loss
        self.stale_count = 0

    def update(self, fixed_eval_loss, iteration):
        """Record one true eval-tick's fixed_eval and update the stale-check count."""
        if fixed_eval_loss < self.best_loss - self.min_delta:
            self.best_loss = fixed_eval_loss
            self.stale_count = 0
        elif self.enabled and iteration >= self.min_iterations:
            self.stale_count += 1
        else:
            self.stale_count = 0

    def should_stop(self, iteration):
        return (
            self.enabled
            and iteration >= self.min_iterations
            and self.stale_count >= self.patience
        )


def save_rendered_image(trainer: WarpImageTrainer, resolution: int, path: Path) -> None:
    """Save the trainer's most recently rendered image (its first buffered view) as a PNG."""
    pixels = trainer.image.numpy()[: resolution * resolution].reshape(resolution, resolution, 3)
    Image.fromarray(np.uint8(np.clip(pixels, 0.0, 1.0) * 255.0)).save(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train 3D Gaussian splats from one YAML configuration file."
    )
    parser.add_argument("config", type=Path, help="Path to the training YAML file.")
    args = parser.parse_args()
    try:
        config = TrainingConfig.load(args.config)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    print(f"Resolved training configuration ({args.config.expanduser().resolve()}):")
    print(yaml.safe_dump(config.printable(), sort_keys=False).rstrip(), flush=True)

    paths = config["paths"]
    runtime = config["runtime"]
    model = config["model"]
    training = config["training"]
    rates = config["learning_rates"]
    compact = config["compact_box"]
    adc = config["multiview_adc"]
    reporting = config["reporting"]
    wp.init()
    device = "cpu" if runtime["arch"] == "cpu" else "cuda:0"
    if runtime["arch"] == "gpu" and not wp.is_cuda_available():
        raise RuntimeError("runtime.arch=gpu requires a CUDA-enabled Warp installation.")
    background = (
        np.ones(3, np.float32) if model["background"] == "white"
        else np.zeros(3, np.float32)
    )
    targets, cameras, focal, distance, _ = load_views(
        paths["data"], model["resolution"], background
    )
    scene_radius = model["scene_radius"] or distance * 0.35
    initial = GaussianSet.from_ply(paths["init_ply"]) if paths["init_ply"] else None
    if initial is not None and len(initial.means) > model["capacity"]:
        parser.error(
            f"Checkpoint has {len(initial.means):,} splats, exceeding "
            f"model.capacity={model['capacity']:,}."
        )
    eval_count = min(reporting["eval_views"], len(targets))
    eval_view_ids = np.linspace(0, len(targets) - 1, eval_count, dtype=np.int32)
    trainer = WarpImageTrainer(
        targets, cameras, focal, scene_radius, model["capacity"],
        model["initial_splats"], device, runtime["seed"],
        init_scene=initial,
        background=background,
        learning_rates=rates,
        tile_pair_capacity=model["tile_pair_capacity"],
        training_options={
            "compact_box": compact,
            "multiview_adc": adc,
        },
    )
    output = paths["output"]
    snapshots = Path(f"{output.with_suffix('')}_snapshots")
    snapshots.mkdir(parents=True, exist_ok=True)
    run_training(trainer, config, targets, eval_view_ids, snapshots, output)


def run_training(trainer, config, targets, eval_view_ids, snapshots, output) -> None:
    """Run the full training loop: logging, snapshots, densification, and the final save."""
    runtime = config["runtime"]
    training = config["training"]
    densification = config["densification"]
    adc = config["multiview_adc"]
    convergence = config["convergence"]
    reporting = config["reporting"]
    model = config["model"]
    device = trainer.device

    # Densification stops part-way through the run. Late clones cannot be placed well: the
    # position learning rate has decayed, so a new splat can no longer travel to where it is
    # needed, and it survives as a floater instead.
    densify_until = densification["until"] or training["iterations"] // 2
    # A splat with opacity below opacity_cutoff is removed unconditionally, across the whole
    # active population, every time late-prune fires -- not only the multi-view-scored
    # candidates. If the (start, end, interval) window fits few events, that sweep behaves as
    # one large one-off cull rather than periodic hygiene. Surface the count so a schedule that
    # does not match `training.iterations` is visible before training, not inferred from a
    # surprising drop in `active` partway through the log.
    late_prune_events = (
        count_multiples_in_range(
            adc["late_prune_start"],
            min(adc["late_prune_end"], training["iterations"]),
            adc["late_prune_interval"],
        )
        if adc["enabled"]
        else 0
    )
    print(
        f"Training {trainer.active.sum():,} initial splats (capacity {model['capacity']:,}) at "
        f"{model['resolution']}x{model['resolution']} on Warp {device}; "
        "one random training view per iteration, "
        f"tile_pair_capacity={trainer.tile_pair_capacity:,}, "
        f"densify_every={densification['interval']} until {densify_until}. "
        "Building tiles for the first step...",
        flush=True,
    )
    if adc["enabled"]:
        print(
            f"multiview_adc late-prune fires {late_prune_events} time(s) in this run "
            f"(window [{adc['late_prune_start']}, {adc['late_prune_end']}], "
            f"every {adc['late_prune_interval']} iterations); "
            f"each event removes every active splat with opacity below "
            f"opacity_cutoff={adc['opacity_cutoff']} across the whole population, not only "
            f"multi-view-scored candidates."
            + (
                " With so few events, this acts as one large cull rather than periodic "
                "hygiene -- widen the window or shorten the interval if that is a surprise."
                if late_prune_events <= 1
                else ""
            ),
            flush=True,
        )
    fixed_eval_loss = trainer.evaluate(eval_view_ids)
    fixed_eval_iteration = 0
    print(
        f"{0:6d}: fixed_eval={fixed_eval_loss:.6f}, "
        f"eval_views={eval_view_ids.tolist()}",
        flush=True,
    )
    rng = np.random.default_rng(runtime["seed"])
    ema_loss = None
    convergence_tracker = ConvergenceTracker(convergence, fixed_eval_loss)
    completed_iteration = 0
    training_started = perf_counter()
    for iteration in range(1, training["iterations"] + 1):
        completed_iteration = iteration
        view_ids = np.asarray([rng.integers(len(targets))], dtype=np.int32)
        should_densify = bool(
            densification["interval"]
            and iteration % densification["interval"] == 0
            and iteration <= densify_until
        )
        should_late_prune = bool(
            adc["enabled"]
            and adc["late_prune_start"] <= iteration <= adc["late_prune_end"]
            and iteration % adc["late_prune_interval"] == 0
        )
        should_structure_update = should_densify or should_late_prune
        loss, gradients = trainer.step(
            view_ids, download_gradients=should_structure_update,
            iteration=iteration,
        )
        if should_structure_update:
            adc_views = None
            if adc["enabled"]:
                sample_count = min(adc["views"], len(targets))
                adc_views = rng.choice(len(targets), size=sample_count, replace=False)
            trainer.densify_and_prune(
                gradients,
                densification["fraction"] if should_densify else 0.0,
                view_ids=adc_views,
                late_prune=should_late_prune,
                percent_dense=densification["percent_dense"],
            )
        if (
            densification["opacity_reset_interval"]
            and iteration % densification["opacity_reset_interval"] == 0
            and iteration <= densify_until
        ):
            trainer.reset_opacity()
        ema_loss = loss if ema_loss is None else (
            reporting["loss_ema_decay"] * ema_loss
            + (1.0 - reporting["loss_ema_decay"]) * loss
        )
        should_snapshot = iteration % reporting["snapshot_every"] == 0
        is_eval_tick = iteration % reporting["eval_every"] == 0
        if is_eval_tick or should_snapshot:
            fixed_eval_loss = trainer.evaluate(eval_view_ids)
            fixed_eval_iteration = iteration
            # Convergence bookkeeping only advances on a true eval_every tick. A snapshot alone
            # still needs a fresh fixed_eval to render and log, but it must not change when
            # training stops -- otherwise snapshot_every silently reshapes the stopping schedule.
            if is_eval_tick:
                convergence_tracker.update(fixed_eval_loss, iteration)
        if iteration % reporting["log_every"] == 0:
            print(
                f"{iteration:6d}: loss={loss:.6f}, ema={ema_loss:.6f}, "
                f"fixed_eval={fixed_eval_loss:.6f}@{fixed_eval_iteration}, "
                f"active={trainer.active.sum()}, "
                f"pairs={trainer.last_pair_count}, "
                f"lr_xyz={trainer.last_position_learning_rate:.8f}, "
                f"tiles={trainer.last_timings['tile']:.2f}s, device={trainer.last_timings['device']:.2f}s",
                flush=True,
            )
        if should_snapshot:
            save_rendered_image(trainer, model["resolution"], snapshots / f"out_{iteration:06d}.png")
            if reporting["save_ply"]:
                trainer.gaussian_set().to_ply(snapshots / f"out_{iteration:06d}.ply")
        if convergence_tracker.should_stop(iteration):
            print(
                f"Stopping at iteration {iteration}: fixed_eval has not improved by "
                f"at least {convergence['min_delta']:.3g} for {convergence_tracker.stale_count} "
                "evaluation checks.",
                flush=True,
            )
            break
    trainer.evaluate(eval_view_ids)
    save_rendered_image(trainer, model["resolution"], output)
    if reporting["save_ply"]:
        final_ply = output.with_suffix(".ply")
        trainer.gaussian_set().to_ply(final_ply)
        print(f"Saved final trained splats to {final_ply}", flush=True)
    elapsed = perf_counter() - training_started
    print(
        f"Training finished after {completed_iteration:,} iterations in "
        f"{elapsed:.2f}s ({elapsed / max(1, completed_iteration):.3f}s/iter).",
        flush=True,
    )


if __name__ == "__main__":
    main()

"""Gaussian-first tile workspace for training and evaluation.

Projection of trainable Gaussians remains here because it uses device-side parameters.  The
shared GaussianFirstTileBuilder owns the common projected-record tiling, sorting, and offsets.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import warp as wp

_base_trainer = importlib.import_module("3dgs_trainer")
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
# `shared/` sits beside this file in the assignment repo, and one level up in the course repo.
for _candidate in (_here / "shared", _here.parent / "shared"):
    if _candidate.is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
import tile_builder as _tile_builder_module
from tile_builder import GaussianFirstTileBuilder
from time import perf_counter

NEAR_PLANE = _base_trainer.NEAR_PLANE
SUPPORT_RADIUS_SQUARED = _base_trainer.SUPPORT_RADIUS_SQUARED
TILE = _base_trainer.TILE
projected_covariance = _base_trainer.projected_covariance


@wp.kernel(enable_backward=False)
def project_tile_records(
    means: wp.array(dtype=wp.vec3),
    log_scales: wp.array(dtype=wp.vec3),
    quaternions: wp.array(dtype=wp.vec4),
    opacity_logits: wp.array(dtype=wp.float32),
    active: wp.array(dtype=wp.int32),
    cameras: wp.array(dtype=wp.mat44),
    view_ids: wp.array(dtype=wp.int32),
    capacity: int,
    width: int,
    height: int,
    focal: float,
    view_count: int,
    compact_box_enabled: int,
    compact_box_beta: float,
    compact_box_alpha_min: float,
    centres: wp.array(dtype=wp.vec2),
    conics: wp.array(dtype=wp.vec3),
    supports: wp.array(dtype=wp.float32),
    depths: wp.array(dtype=wp.float32),
    group_ids: wp.array(dtype=wp.int32),
    splat_ids: wp.array(dtype=wp.uint32),
):
    item = wp.tid()
    batch_view = item // capacity
    splat = item - batch_view * capacity
    centres[item] = wp.vec2(0.0, 0.0)
    conics[item] = wp.vec3(0.0, 0.0, 0.0)
    supports[item] = 0.0
    depths[item] = 0.0
    group_ids[item] = batch_view
    splat_ids[item] = wp.uint32(splat)

    if batch_view < view_count and active[splat] != 0:
        camera = cameras[view_ids[batch_view]]
        point = camera * wp.vec4(means[splat][0], means[splat][1], means[splat][2], 1.0)
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
                    support = wp.min(
                        float(SUPPORT_RADIUS_SQUARED),
                        compact_box_beta * 2.0 * wp.log(opacity / compact_box_alpha_min),
                    )

            determinant = covariance[0] * covariance[2] - covariance[1] * covariance[1]
            if support > 0.0 and determinant > 1.0e-12:
                conic = wp.vec3(
                    covariance[2] / determinant,
                    -covariance[1] / determinant,
                    covariance[0] / determinant,
                )
                centre = wp.vec2(
                    focal * point[0] / z + 0.5 * float(width),
                    focal * point[1] / z + 0.5 * float(height),
                )
                if (
                    wp.isfinite(centre[0])
                    and wp.isfinite(centre[1])
                    and wp.isfinite(conic[0])
                    and wp.isfinite(conic[1])
                    and wp.isfinite(conic[2])
                ):
                    centres[item] = centre
                    conics[item] = conic
                    supports[item] = support
                    depths[item] = z


class GaussianFirstTileWorkspace:
    """Training-facing adapter around the shared projected-record tile builder."""

    def __init__(self, capacity, max_views_per_batch, tiles_x, tiles_y, tile_pair_capacity, device):
        self.capacity = int(capacity)
        self.max_views_per_batch = int(max_views_per_batch)
        self.tiles_x = int(tiles_x)
        self.tiles_y = int(tiles_y)
        self.tiles_per_view = self.tiles_x * self.tiles_y
        self.tile_pair_capacity = int(tile_pair_capacity)
        self.device = wp.get_device(device)
        self.projected_centres = wp.zeros(
            max_views_per_batch * capacity, dtype=wp.vec2, device=self.device
        )
        self.projected_conics = wp.zeros(
            max_views_per_batch * capacity, dtype=wp.vec3, device=self.device
        )
        self.projected_supports = wp.zeros(
            max_views_per_batch * capacity, dtype=wp.float32, device=self.device
        )
        self.projected_depths = wp.zeros(
            max_views_per_batch * capacity, dtype=wp.float32, device=self.device
        )
        self.projected_group_ids = wp.zeros(
            max_views_per_batch * capacity, dtype=wp.int32, device=self.device
        )
        self.projected_splat_ids = wp.zeros(
            max_views_per_batch * capacity, dtype=wp.uint32, device=self.device
        )
        self.builder = GaussianFirstTileBuilder(
            max_views_per_batch * capacity,
            max_views_per_batch * self.tiles_per_view,
            self.tiles_x * TILE,
            self.tiles_y * TILE,
            TILE,
            tile_pair_capacity,
            self.device,
        )
        self.view_ids_host = np.zeros(max_views_per_batch, np.int32)
        self.view_ids = wp.zeros(max_views_per_batch, dtype=wp.int32, device=self.device)

    def build(self, means, log_scales, quaternions, opacity, active_device, cameras,
              view_ids, width, height, focal, compact_box):
        view_ids = np.asarray(view_ids, np.int32)
        view_count = len(view_ids)
        if view_count < 1 or view_count > self.max_views_per_batch:
            raise ValueError("View batch exceeds the persistent tile-workspace capacity.")
        self.view_ids_host.fill(0)
        self.view_ids_host[:view_count] = view_ids
        self.view_ids.assign(self.view_ids_host)
        work_items = view_count * self.capacity
        group_count = view_count * self.tiles_per_view

        if _tile_builder_module.PROFILE_TILES:
            wp.synchronize_device(self.device)
            _projection_started = perf_counter()
        wp.launch(
            project_tile_records,
            dim=work_items,
            inputs=[
                means, log_scales, quaternions, opacity, active_device,
                cameras, self.view_ids, self.capacity, width, height, focal, view_count,
                int(compact_box["enabled"]), float(compact_box["beta"]),
                float(compact_box["alpha_min"]),
            ],
            outputs=[
                self.projected_centres, self.projected_conics, self.projected_supports,
                self.projected_depths, self.projected_group_ids, self.projected_splat_ids,
            ],
            device=self.device,
        )
        if _tile_builder_module.PROFILE_TILES:
            wp.synchronize_device(self.device)
            self.projection_seconds = perf_counter() - _projection_started
        return self.builder.build(
            self.projected_centres,
            self.projected_conics,
            self.projected_supports,
            self.projected_depths,
            self.projected_group_ids,
            self.projected_splat_ids,
            work_items,
            group_count,
            width,
            height,
        )

"""Pedagogical sequential CPU 3D Gaussian Splatting renderer.

Usage:
    python vanilla_cpu_renderer.py point_cloud.ply render.png

The input must use the conventional 3DGS vertex properties:
x, y, z, opacity, scale_0..2, rot_0..3, and f_dc_0..2.
This reference renderer is intentionally simple: every pixel tests every
projected splat after a global centre-depth sort. It is correct enough for
learning, but not suitable for large scenes. Unit 4 explains tiled binning.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814  # Degree-0 real SH normalisation constant.
# Splats are culled/composited past this many squared Mahalanobis units from centre. Shared with
# warp_images_3dgs_training.py so the CPU reference and the Warp trainer agree on cutoff.
SUPPORT_RADIUS_SQUARED = 9.0


@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_camera: np.ndarray

    @classmethod
    def identity(cls, width: int, height: int, focal_length: float) -> "Camera":
        """A camera at the world origin, looking along positive z."""
        return cls(
            width=width,
            height=height,
            fx=focal_length,
            fy=focal_length,
            cx=width / 2.0,
            cy=height / 2.0,
            world_to_camera=np.eye(4, dtype=np.float32),
        )

    @classmethod
    def from_look_at(
        cls,
        width: int,
        height: int,
        focal_length: float,
        position: np.ndarray,
        target: np.ndarray,
        up: np.ndarray,
    ) -> "Camera":
        """Build a world-to-camera transform for this renderer's +z-forward convention."""
        position = np.asarray(position, dtype=np.float32)
        target = np.asarray(target, dtype=np.float32)
        up = np.asarray(up, dtype=np.float32)

        forward = target - position
        forward_length = np.linalg.norm(forward)
        if forward_length < 1.0e-8:
            raise ValueError("Camera position and look-at target must be different.")
        forward /= forward_length

        right = np.cross(up, forward)
        right_length = np.linalg.norm(right)
        if right_length < 1.0e-8:
            raise ValueError("Camera up vector must not be parallel to the viewing direction.")
        right /= right_length
        down = np.cross(forward, right)

        rotation = np.stack((right, down, forward))
        world_to_camera = np.eye(4, dtype=np.float32)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = -rotation @ position
        return cls(width, height, focal_length, focal_length, width / 2.0, height / 2.0, world_to_camera)


@dataclass
class GaussianSet:
    means: np.ndarray       # (N, 3), world coordinates
    scales: np.ndarray      # (N, 3), positive local-axis standard deviations
    rotations: np.ndarray   # (N, 4), unit quaternion in (w, x, y, z) order
    opacities: np.ndarray   # (N,), values in [0, 1]
    colors: np.ndarray      # (N, 3), degree-0 SH decoded to RGB in [0, 1]

    @classmethod
    def from_ply(cls, path: str) -> "GaussianSet":
        ply = PlyData.read(path)
        vertex = ply["vertex"].data
        names = set(vertex.dtype.names or ())

        required = {
            "x", "y", "z", "opacity",
            "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3",
            "f_dc_0", "f_dc_1", "f_dc_2",
        }
        missing = sorted(required - names)
        if missing:
            raise ValueError(f"PLY is missing 3DGS properties: {', '.join(missing)}")

        def fields(prefix: str, count: int) -> np.ndarray:
            return np.stack(
                [np.asarray(vertex[f"{prefix}_{i}"], dtype=np.float32) for i in range(count)],
                axis=1,
            )

        means = np.stack(
            [np.asarray(vertex[name], dtype=np.float32) for name in ("x", "y", "z")],
            axis=1,
        )

        # Standard 3DGS PLY files store the optimiser's raw values.
        scales = np.exp(fields("scale", 3))
        opacities = 1.0 / (1.0 + np.exp(-np.asarray(vertex["opacity"], dtype=np.float32)))

        rotations = fields("rot", 4)
        rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-8)

        # f_dc is the degree-0 SH coefficient, not ordinary RGB.
        colors = np.clip(C0 * fields("f_dc", 3) + 0.5, 0.0, 1.0)
        return cls(means, scales, rotations, opacities, colors)

    def to_ply(self, path: str) -> int:
        """Write these splats as a conventional 3DGS PLY. Exact inverse of from_ply.

        Kept next to from_ply on purpose: the file format's conventions -- log-scale,
        opacity logit, and degree-0 SH rather than RGB -- are easy to get subtly wrong,
        and a reader and writer that live apart drift. Anything that produces splats (a
        trainer, a converter) should build a GaussianSet and call this rather than
        re-deriving the encoding.

        Returns the number of splats written.
        """
        count = len(self.means)
        vertices = np.zeros(count, dtype=[
            ("x", "f4"), ("y", "f4"), ("z", "f4"),
            ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
            ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
            ("opacity", "f4"),
            ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
            ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
        ])
        vertices["x"], vertices["y"], vertices["z"] = np.asarray(self.means, np.float32).T
        # Normals are unused by splat renderers but belong to the conventional layout.

        # Undo each decoding step from_ply applies.
        scales = np.maximum(np.asarray(self.scales, np.float32), 1e-12)
        vertices["scale_0"], vertices["scale_1"], vertices["scale_2"] = np.log(scales).T

        opacities = np.clip(np.asarray(self.opacities, np.float32), 1e-6, 1.0 - 1e-6)
        vertices["opacity"] = np.log(opacities / (1.0 - opacities))

        f_dc = (np.asarray(self.colors, np.float32) - 0.5) / C0
        vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = f_dc.T

        rotations = np.asarray(self.rotations, np.float32)
        rotations = rotations / np.maximum(
            np.linalg.norm(rotations, axis=1, keepdims=True), 1e-8)
        for index in range(4):
            vertices[f"rot_{index}"] = rotations[:, index]

        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))
        return count


@dataclass
class ProjectedGaussians:
    centres: np.ndarray  # (M, 2)
    conics: np.ndarray   # (M, 3): inverse covariance entries A, B, C
    colors: np.ndarray   # (M, 3)
    opacities: np.ndarray  # (M,)


def quaternion_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    """Convert (w, x, y, z) unit quaternions into (N, 3, 3) rotation matrices."""
    w, x, y, z = quaternions.T
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
            2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
            2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
        ),
        axis=1,
    ).reshape(-1, 3, 3).astype(np.float32)


def project_gaussians(
    splats: GaussianSet,
    camera: Camera,
    near: float = 0.01,
    filter_variance: float = 0.3,
    qmax: float = SUPPORT_RADIUS_SQUARED,
) -> ProjectedGaussians:
    """Build compact screen-space records and sort them front to back."""
    rotation_world = quaternion_to_matrix(splats.rotations)
    scale_squared = splats.scales * splats.scales
    covariance_world = np.einsum(
        "nij,nj,nkj->nik", rotation_world, scale_squared, rotation_world
    )

    world_h = np.concatenate(
        (splats.means, np.ones((len(splats.means), 1), dtype=np.float32)), axis=1
    )
    camera_h = world_h @ camera.world_to_camera.T
    mean_camera = camera_h[:, :3]
    depth = mean_camera[:, 2]

    W = camera.world_to_camera[:3, :3]
    covariance_camera = W @ covariance_world @ W.T

    x, y, z = mean_camera.T
    centres = np.stack(
        (camera.fx * x / z + camera.cx, camera.fy * y / z + camera.cy), axis=1
    )

    jacobian = np.zeros((len(splats.means), 2, 3), dtype=np.float32)
    jacobian[:, 0, 0] = camera.fx / z
    jacobian[:, 0, 2] = -camera.fx * x / (z * z)
    jacobian[:, 1, 1] = camera.fy / z
    jacobian[:, 1, 2] = -camera.fy * y / (z * z)
    covariance_screen = jacobian @ covariance_camera @ np.swapaxes(jacobian, 1, 2)
    covariance_screen += filter_variance * np.eye(2, dtype=np.float32)

    eigenvalues = np.linalg.eigvalsh(covariance_screen)
    radii = np.sqrt(qmax * np.maximum(eigenvalues[:, 1], 0.0))
    visible = (
        (depth > near)
        & (centres[:, 0] + radii >= 0.0)
        & (centres[:, 0] - radii < camera.width)
        & (centres[:, 1] + radii >= 0.0)
        & (centres[:, 1] - radii < camera.height)
    )

    indices = np.flatnonzero(visible)
    if len(indices) == 0:
        return ProjectedGaussians(
            np.empty((0, 2), np.float32),
            np.empty((0, 3), np.float32),
            np.empty((0, 3), np.float32),
            np.empty((0,), np.float32),
        )

    inverse = np.linalg.inv(covariance_screen[indices])
    conics = np.stack((inverse[:, 0, 0], inverse[:, 0, 1], inverse[:, 1, 1]), axis=1)

    # The global sort is only pedagogical. A scalable renderer sorts per tile.
    order = np.argsort(depth[indices])
    indices = indices[order]
    return ProjectedGaussians(
        centres[indices].astype(np.float32),
        conics[order].astype(np.float32),
        splats.colors[indices].astype(np.float32),
        splats.opacities[indices].astype(np.float32),
    )


class CpuRenderer:
    """A sequential reference renderer: rows, pixels, then sorted splats."""

    def __init__(
        self,
        camera: Camera,
        near: float = 0.01,
        filter_variance: float = 0.3,
        qmax: float = SUPPORT_RADIUS_SQUARED,
    ):
        self.camera = camera
        self.near = near
        self.filter_variance = filter_variance
        self.qmax = qmax

    def render(
        self,
        splats: GaussianSet,
        background: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> np.ndarray:
        projected = project_gaussians(
            splats,
            self.camera,
            near=self.near,
            filter_variance=self.filter_variance,
            qmax=self.qmax,
        )
        image = np.zeros((self.camera.height, self.camera.width, 3), dtype=np.float32)
        background_color = np.asarray(background, dtype=np.float32)

        # Deliberately sequential: one row, one pixel, and one splat at a time.
        for py in range(self.camera.height):
            y = py + 0.5
            for px in range(self.camera.width):
                x = px + 0.5

                # TODO: Calculate the RGB value at (x, y)

                # TODO: The RHS is a placeholder
                image[py, px] = np.zeros(3, dtype=np.float32)

        return image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", help="3DGS point_cloud.ply file")
    parser.add_argument("output", help="output PNG file")
    parser.add_argument("--width", type=int, default=800)
    parser.add_argument("--height", type=int, default=800)
    parser.add_argument("--focal-length", type=float, default=700.0)
    parser.add_argument("--camera-position", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--look-at", nargs=3, type=float, default=(0.0, 0.0, 1.0))
    parser.add_argument("--up", nargs=3, type=float, default=(0.0, 1.0, 0.0))
    args = parser.parse_args()

    camera = Camera.from_look_at(
        args.width, args.height, args.focal_length,
        args.camera_position, args.look_at, args.up,
    )
    splats = GaussianSet.from_ply(args.ply)
    image = CpuRenderer(camera).render(splats)
    Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0)).save(args.output)


if __name__ == "__main__":
    main()

"""3DGS Gaussian storage and conventional PLY conversion."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from plyfile import PlyData, PlyElement

C0 = 0.28209479177387814


@dataclass
class GaussianSet:
    means: np.ndarray
    scales: np.ndarray
    rotations: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray

    @classmethod
    def from_ply(cls, path: str) -> "GaussianSet":
        ply = PlyData.read(path)
        vertex = ply["vertex"].data
        names = set(vertex.dtype.names or ())
        required = {
            "x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2",
            "rot_0", "rot_1", "rot_2", "rot_3", "f_dc_0", "f_dc_1", "f_dc_2",
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
            [np.asarray(vertex[name], dtype=np.float32) for name in ("x", "y", "z")], axis=1
        )
        scales = np.exp(fields("scale", 3))
        opacities = 1.0 / (1.0 + np.exp(-np.asarray(vertex["opacity"], dtype=np.float32)))
        rotations = fields("rot", 4)
        rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-8)
        colors = np.clip(C0 * fields("f_dc", 3) + 0.5, 0.0, 1.0)
        return cls(means, scales, rotations, opacities, colors)

    def to_ply(self, path: str) -> int:
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
        scales = np.maximum(np.asarray(self.scales, np.float32), 1e-12)
        vertices["scale_0"], vertices["scale_1"], vertices["scale_2"] = np.log(scales).T
        opacities = np.clip(np.asarray(self.opacities, np.float32), 1e-6, 1.0 - 1e-6)
        vertices["opacity"] = np.log(opacities / (1.0 - opacities))
        f_dc = (np.asarray(self.colors, np.float32) - 0.5) / C0
        vertices["f_dc_0"], vertices["f_dc_1"], vertices["f_dc_2"] = f_dc.T
        rotations = np.asarray(self.rotations, np.float32)
        rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-8)
        for index in range(4):
            vertices[f"rot_{index}"] = rotations[:, index]
        PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(str(path))
        return count

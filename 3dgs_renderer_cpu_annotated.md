# Annotated Walkthrough: `3dgs_renderer_cpu.py`

This walkthrough explains the provided CPU renderer skeleton. It is adapted from the instructor
reference annotation, with the section that solves the per-pixel TODO removed.

The goal of this file is to help you understand the code you have been given: the data structures,
camera convention, PLY decoding, projection helper, and command-line path. The missing pixel RGB
calculation inside `CpuRenderer.render()` is still your assignment work.

## NumPy Vocabulary Used Here

The renderer uses NumPy arrays instead of Python lists for most numerical data.

| Syntax | Meaning |
|---|---|
| `a @ b` | Matrix multiplication. |
| `a.T` | Transpose. |
| `a * b` | Elementwise multiplication. |
| `np.stack([...], axis=1)` | Combine several one-dimensional arrays into columns. |
| `array[mask]` | Keep entries where a Boolean mask is true. |
| `array[indices]` | Reorder entries by integer indices. |
| Broadcasting | NumPy automatically stretches compatible shapes, such as `(N, 1)` against `(N, 3)`. |

Shapes in comments use the convention `(N, 3)` to mean `N` rows and 3 columns.

## Imports and Constants

```python
import argparse
from dataclasses import dataclass

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
```

- `argparse` parses command-line arguments.
- `dataclass` creates simple classes whose fields are mostly data.
- `numpy` handles arrays, vectors, and matrices.
- `PIL.Image` writes the final PNG.
- `plyfile` reads and writes the 3DGS PLY format.

`C0` is the degree-0 real spherical-harmonic normalization constant. In this assignment, colour is
treated as view-independent RGB, but the PLY stores it in the usual 3DGS degree-0 coefficient
form. The loader decodes it.

`SUPPORT_RADIUS_SQUARED = 9.0` is the squared Gaussian support cutoff used consistently by the CPU
renderer, GPU renderer, and trainer.

## `Camera`

`Camera` stores the image dimensions, focal length, principal point, and a `world_to_camera`
matrix.

```python
@dataclass
class Camera:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    world_to_camera: np.ndarray
```

`Camera.identity()` creates a camera at the world origin looking along positive `z`.

`Camera.from_look_at()` creates a camera from:

- `position`: where the camera is;
- `target`: what it looks at;
- `up`: which world direction should appear upward in the image.

The method builds a camera basis:

- `forward`: target direction;
- `right`: perpendicular to `up` and `forward`;
- `down`: perpendicular to `forward` and `right`.

It then packs the basis into a `4 x 4` homogeneous transform. Unit 0 explains why the translation
is `-rotation @ position`.

## `GaussianSet`

`GaussianSet` stores all splats in structure-of-arrays form:

```python
@dataclass
class GaussianSet:
    means: np.ndarray       # (N, 3)
    scales: np.ndarray      # (N, 3)
    rotations: np.ndarray   # (N, 4)
    opacities: np.ndarray   # (N,)
    colors: np.ndarray      # (N, 3)
```

Each row describes one 3D Gaussian splat.

### `GaussianSet.from_ply`

The loader expects conventional 3DGS PLY properties:

| PLY fields | Meaning after decoding |
|---|---|
| `x`, `y`, `z` | World-space centre. |
| `scale_0..2` | Log standard deviations, decoded with `np.exp`. |
| `rot_0..3` | Quaternion rotation, normalized before use. |
| `opacity` | Opacity logit, decoded with sigmoid. |
| `f_dc_0..2` | Degree-0 SH colour coefficients, decoded to RGB. |

The helper function `fields(prefix, count)` stacks PLY columns such as `scale_0`, `scale_1`,
`scale_2` into a single `(N, 3)` array.

### `GaussianSet.to_ply`

`to_ply()` performs the inverse conversion: physical scales are stored as log-scales, opacities
are stored as logits, RGB is stored back as degree-0 coefficients, and rotations are normalized
before writing.

Keeping `from_ply()` and `to_ply()` next to each other is useful because the PLY encoding is easy
to get subtly wrong.

## `ProjectedGaussians`

`ProjectedGaussians` is the compact screen-space representation consumed by the pixel renderer:

```python
@dataclass
class ProjectedGaussians:
    centres: np.ndarray    # (M, 2)
    conics: np.ndarray     # (M, 3), inverse covariance entries A, B, C
    colors: np.ndarray     # (M, 3)
    opacities: np.ndarray  # (M,)
```

`M` is the number of visible splats after culling. It can be smaller than the original number of
PLY rows.

## `quaternion_to_matrix`

`quaternion_to_matrix()` converts unit quaternions in `(w, x, y, z)` order into rotation matrices
with shape `(N, 3, 3)`.

These rotations are used to construct each splat's world covariance from its local-axis scales.

## `project_gaussians`

`project_gaussians()` is the shared host-side projection helper. It is not the assignment TODO.

Its stages are:

1. Convert each quaternion to a `3 x 3` rotation matrix.
2. Build the 3D covariance from rotation and squared scale.
3. Transform splat centres and covariances into camera coordinates.
4. Project means to pixel coordinates with the pinhole camera model.
5. Use the Jacobian to project the 3D covariance to a 2D screen covariance.
6. Add a small filter variance for numerical stability.
7. Cull splats behind the near plane or outside the image.
8. Invert the `2 x 2` screen covariance into conic form `(A, B, C)`.
9. Sort visible splats by camera-space depth.

The returned `ProjectedGaussians` object is already sorted front to back.

## `CpuRenderer`

`CpuRenderer.render()` calls `project_gaussians()` and allocates the output image. The nested loops
then visit one pixel at a time:

```python
for py in range(self.camera.height):
    y = py + 0.5
    for px in range(self.camera.width):
        x = px + 0.5
        # TODO
```

The missing TODO is the assignment's core calculation: compute the RGB value at pixel `(x, y)` by
using the sorted projected splats and front-to-back alpha compositing.

This walkthrough deliberately omits the reference implementation of that pixel loop. Use Unit 4's
equations to fill it in, then compare your CPU output against the expected qualitative behaviour:

- empty pixels show the background;
- splats appear at the projected positions;
- closer translucent splats affect splats behind them;
- values remain valid RGB values in `[0, 1]`.

## `main`

`main()` parses the command line, builds a camera, loads the PLY, renders the image, and saves a
PNG:

```bash
python 3dgs_renderer_cpu.py data/lego/init.ply cpu.png \
  --width 400 --height 400 \
  --camera-position -2 -2 3 --look-at 0 0 0 --up 0 0 -1
```

Start with a small resolution while debugging.

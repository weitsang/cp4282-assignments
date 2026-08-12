# Annotated Walkthrough: `3dgs_renderer_gpu.py`

This walkthrough explains the provided Warp renderer skeleton. It is adapted from the instructor
reference annotation, with the section that solves the `rasterize()` TODO removed.

Read `3dgs_renderer_cpu_annotated.md` first. The GPU file reuses the CPU renderer's camera, PLY
loader, and projection helper; only the per-pixel loop moves into Warp.

## Why `importlib` Appears Here

The CPU renderer file is named `3dgs_renderer_cpu.py`. That is a valid filename and can be run as a
script, but Python's normal import statement cannot start a module name with a digit:

```python
from 3dgs_renderer_cpu import Camera
```

is invalid syntax. Instead, the GPU skeleton uses:

```python
import importlib

_cpu_renderer = importlib.import_module("3dgs_renderer_cpu")
Camera = _cpu_renderer.Camera
GaussianSet = _cpu_renderer.GaussianSet
project_gaussians = _cpu_renderer.project_gaussians
```

This keeps the filename `3dgs_renderer_cpu.py` while still letting the GPU file reuse its helper
classes and functions.

## Warp Mental Model

In the CPU renderer, Python loops over pixels sequentially. In Warp, the kernel describes the work
for one pixel, and `wp.launch()` starts many copies of that work:

| Concept | Role |
|---|---|
| `@wp.kernel` | Marks code that Warp compiles for CPU or CUDA execution. |
| `wp.tid()` | Returns the current work item's integer id. |
| `wp.array` | Device-accessible array. |
| `wp.vec2`, `wp.vec3` | Small fixed-size vector values inside kernels. |
| `wp.launch(...)` | Starts many kernel work items. |

Inside a Warp kernel, use Warp math functions such as `wp.exp`, `wp.min`, and `wp.max`. Do not use
NumPy inside the kernel.

## Imports

```python
import argparse
import importlib

import numpy as np
from PIL import Image
import warp as wp
```

- `argparse` parses command-line arguments.
- `importlib` dynamically imports `3dgs_renderer_cpu`.
- `numpy` and `PIL.Image` are used on the host side.
- `warp` provides arrays, kernels, vector types, launches, and device management.

## `rasterize`

The kernel receives projected screen-space arrays:

```python
centres: wp.array(dtype=wp.vec2)
conics: wp.array(dtype=wp.vec3)
colours: wp.array(dtype=wp.vec3)
opacities: wp.array(dtype=wp.float32)
```

These arrays come from `project_gaussians()` in the CPU renderer. They are already compact and
depth-sorted.

The first lines of the kernel assign one work item to one pixel:

```python
pixel = wp.tid()
px = float(pixel % width) + 0.5
py = float(pixel // width) + 0.5
```

`pixel` is a flattened index. The column is `pixel % width`; the row is `pixel // width`. The
`+ 0.5` samples the pixel centre, matching the CPU renderer.

The missing TODO is the same pixel-colour calculation as Assignment 1, written with Warp types and
Warp math functions instead of NumPy. This walkthrough deliberately omits the reference kernel
body.

## `WarpRenderer`

`WarpRenderer` owns persistent device buffers:

```python
self.centres = wp.zeros(maximum_splats, dtype=wp.vec2, device=self.device)
self.conics = wp.zeros(maximum_splats, dtype=wp.vec3, device=self.device)
self.colours = wp.zeros(maximum_splats, dtype=wp.vec3, device=self.device)
self.opacities = wp.zeros(maximum_splats, dtype=wp.float32, device=self.device)
self.image = wp.zeros(width * height, dtype=wp.vec3, device=self.device)
```

Persistent buffers avoid reallocating device memory every render. The arrays are sized for the
maximum number of visible splats and for all output pixels.

## `WarpRenderer.render`

`render()` keeps projection on the host side:

```python
projected = project_gaussians(splats, camera)
count = len(projected.opacities)
```

Then it copies the projected arrays into Warp buffers, launches one work item per pixel, and
downloads the rendered image:

```python
wp.launch(
    rasterize,
    dim=self.width * self.height,
    inputs=[...],
    device=self.device,
)
return self.image.numpy().reshape(self.height, self.width, 3)
```

The assignment TODO is not in the host-side launch code. It is inside the kernel, where each work
item computes its own pixel.

## `main`

`main()` initializes Warp, builds a camera, loads a PLY with the CPU helper, renders, and saves a
PNG:

```bash
python 3dgs_renderer_gpu.py data/lego/init.ply gpu.png \
  --device cpu \
  --width 400 --height 400 \
  --camera-position -2 -2 3 --look-at 0 0 0 --up 0 0 -1
```

Use `--device cpu` first for portability. On a CUDA machine, use `--device cuda:0`.

Your GPU render should match your CPU render up to small floating-point differences.

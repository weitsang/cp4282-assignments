# Annotated Walkthrough: `3dgs_renderer_v1.py`

This is the first of three walkthroughs. Read them in order — **v1**, then
[v2](3dgs_renderer_v2_annotated.md), then [v3](3dgs_renderer_v3_annotated.md).
Each later document assumes everything explained in the earlier ones and covers only what is
genuinely new.

This document assumes you know core Python (`if`, `for`, `class`, `def`, type hints) but **not**
NumPy. Every NumPy idiom is explained the first time it appears, with the array **shape** noted in
comments like `# (N, 3)` meaning "N rows, 3 columns". Where the code implements a formula, the
maths is given alongside it.

v1 renders correct images and nothing else. It is deliberately the slowest of the three: every
pixel tests every splat. v2 makes it parallel; v3 makes it scalable. You cannot understand what
those buy you until you have seen what they replace.

---

## 0. The NumPy vocabulary you need before we start

NumPy's core object is the **ndarray**, an n-dimensional grid of numbers of a single type (here
always `float32`). Shapes are tuples: `(N, 3)` is N rows of 3 numbers each — e.g. N points, each
with x, y, z.

A few operations recur throughout all three files:

| Syntax | Meaning |
|---|---|
| `a @ b` | **Matrix multiplication** (not elementwise). If `a` is `(m, k)` and `b` is `(k, n)`, `a @ b` is `(m, n)`, computed as $(AB)_{ij} = \sum_k A_{ik}B_{kj}$. |
| `a.T` | **Transpose**: swaps the last two axes. `(m, n)` becomes `(n, m)`. |
| `a * b` | **Elementwise** multiply. Requires matching shapes, or shapes that broadcast. |
| Broadcasting | NumPy stretches an array of shape `(N, 1)` against one of shape `(N, 3)` as if the single column were repeated 3 times — no loop needed. A plain scalar broadcasts against any shape. |
| `np.stack([a, b], axis=1)` | Stacks arrays into a **new** axis. Two `(N,)` arrays become `(N, 2)` — "zip columns together". |
| `np.concatenate([a, b], axis=1)` | Glues arrays along an **existing** axis; no new axis. |
| `np.einsum(...)` | "Einstein summation" — a compact batched matrix multiply. Explained in full at §6.1. |
| `a[mask]` | **Boolean indexing**. If `mask` is a boolean array as long as `a`, this keeps only rows where `mask` is `True`. |
| `a[indices]` | **Fancy indexing**. If `indices` is an integer array, this picks out and reorders rows by position. This is how the code sorts splats by depth. |
| `a[:, 0]` | **Slicing**: every row, column 0. `a[:3, :3]` is the top-left 3×3 block. |

Everything else (`np.zeros`, `np.exp`, `np.sqrt`, `np.log`, `np.minimum`) works elementwise on
every entry at once ("vectorized") — like calling `math.exp` on each number in a list, but far
faster, because the loop runs in compiled C rather than in Python.

That speed difference is the whole reason this file looks the way it does. Anything NumPy can do
to all N splats at once is written without a Python loop. The one place a Python loop survives is
the per-pixel blending in §8, and that is exactly the part v2 exists to remove.

---

## 1. Module docstring, imports, and the `shared/` shim

```python
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image
import sys
from pathlib import Path
```

- `argparse` — parses command-line arguments (used only in `main()`).
- `numpy as np` — the array library described above.
- `PIL.Image` — reads and writes PNG files.
- `sys`, `pathlib.Path` — needed only for the import shim immediately below.

```python
_here = Path(__file__).resolve().parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))
# `shared/` sits beside this file in the assignment repo, and one level up in the course repo.
for _candidate in (_here / "shared", _here.parent / "shared"):
    if _candidate.is_dir():
        if str(_candidate) not in sys.path:
            sys.path.insert(0, str(_candidate))
        break
```

Python only imports from directories on `sys.path`. `__file__` is this file's own path;
`.resolve()` makes it absolute and `.parent` gives the directory holding it. The loop then looks
for a `shared/` directory in the two places it is known to live and adds the first one it finds.

This exists because the same file is used in two repository layouts. It is housekeeping, not
graphics — but it is worth understanding, because if it fails every import below fails with a
confusing `ModuleNotFoundError` that has nothing to do with your code.

```python
from camera import Camera
from gaussian_set import GaussianSet
from projected_gaussians import ProjectedGaussians
from splat_math import ALPHA_CUTOFF, SUPPORT_RADIUS_SQUARED, quaternion_to_matrix
```

Four imports from `shared/`. The next four sections explain each in turn, because you cannot read
`project_gaussians` without knowing what these hold.

---

## 2. `shared/camera.py` — where you are looking from

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

`@dataclass` is a decorator that writes the constructor for you from the annotated fields, so
`Camera(width=..., height=..., ...)` just works. The fields:

- `width, height` — output image size in pixels.
- `fx, fy` — focal length in pixels along x and y. Bigger means more "zoomed in": a 3D offset
  maps to more pixels of 2D offset.
- `cx, cy` — the **principal point**, the pixel coordinates that the camera's forward axis passes
  through. Here always the image centre.
- `world_to_camera` — a `(4, 4)` matrix converting a point in world coordinates into the camera's
  own frame.

A 4×4 matrix for a 3D transform looks like one row too many. The extra row and column implement
**homogeneous coordinates**: a 3D point $(x,y,z)$ is written $(x,y,z,1)$, which lets a single
matrix multiply express both a rotation $R$ and a translation $t$:

$$
\begin{pmatrix} R & t \\ 0 & 1 \end{pmatrix}
\begin{pmatrix} p \\ 1 \end{pmatrix}
= \begin{pmatrix} Rp + t \\ 1 \end{pmatrix}
$$

Rotation alone cannot move the origin; translation alone cannot turn. Packing them together means
chaining transforms is just multiplying matrices.

!!! note "See the notes"
    The course notes derive homogeneous transforms from scratch. This file only uses the result.

### `Camera.from_look_at`

```python
        forward = target - position
        length = np.linalg.norm(forward)
        if length < 1.0e-8:
            raise ValueError("Camera position and look-at target must be different.")
        forward /= length
```

`np.linalg.norm` is the Euclidean length $\sqrt{x^2+y^2+z^2}$. Dividing by it makes `forward` a
**unit vector** — direction only, length 1. The guard catches the degenerate case where the camera
sits exactly on its target, which would give a zero-length direction and divide by zero.

```python
        right = np.cross(up, forward)
        right_length = np.linalg.norm(right)
        if right_length < 1.0e-8:
            raise ValueError("Camera up vector must not be parallel to the viewing direction.")
        right /= right_length
        down = np.cross(forward, right)
```

`np.cross` is the **cross product**: given two vectors it returns a third, perpendicular to both.
Its length is proportional to $\sin\theta$ between the inputs, which is why the guard is needed —
if `up` is parallel to `forward`, $\sin\theta = 0$ and the result is a zero vector with no
direction to normalise.

Two cross products build a complete set of three mutually perpendicular unit vectors — an
**orthonormal basis** — from just a viewing direction and a rough "which way is up" hint.
`down` needs no normalising: it is the cross product of two perpendicular unit vectors, so it is
already unit length.

```python
        rotation = np.stack((right, down, forward))
        world_to_camera = np.eye(4, dtype=np.float32)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = -rotation @ position
```

`np.eye(4)` is the 4×4 identity matrix. Writing the three basis vectors in as **rows** of the
rotation block is what converts *into* camera space: each row dots with the point to give that
point's coordinate along that axis.

The translation is `-rotation @ position`, not `-position`. The camera must first rotate the world
into its own axes, and only then shift — so the shift has to be expressed in the rotated frame.

Note the axis convention: `right`, `down`, `forward`. Y points **down**, matching image
coordinates where row 0 is the top. This renderer is **+z forward**.

---

## 3. `shared/gaussian_set.py` — what you are drawing

```python
@dataclass
class GaussianSet:
    means: np.ndarray       # (N, 3) world-space centres
    scales: np.ndarray      # (N, 3) positive standard deviations along local axes
    rotations: np.ndarray   # (N, 4) unit quaternions (w, x, y, z)
    opacities: np.ndarray   # (N,)   in [0, 1]
    colors: np.ndarray      # (N, 3) RGB in [0, 1]
```

One splat is a 3D Gaussian blob: a centre, a size along each of three local axes, an orientation,
plus a colour and an opacity. N of them together approximate a scene.

Storing scale and rotation separately, rather than one covariance matrix, is deliberate — it keeps
the shape **valid by construction**. Any scale/rotation pair describes a real ellipsoid, whereas an
arbitrary 3×3 matrix need not be a legal covariance.

### `GaussianSet.from_ply`

```python
        scales = np.exp(fields("scale", 3))
        opacities = 1.0 / (1.0 + np.exp(-np.asarray(vertex["opacity"], dtype=np.float32)))
```

PLY files store the optimiser's **raw unconstrained** values, not the physical ones, so both need
decoding:

- Scales are stored as logarithms. `np.exp` undoes that, and guarantees positivity — a negative
  standard deviation is meaningless, and an optimiser working in log space can never produce one.
- Opacity is stored as a **logit** and decoded with the **sigmoid** $\sigma(x) = 1/(1+e^{-x})$,
  which maps any real number into $(0,1)$. Same trick: the optimiser is free to move a parameter
  anywhere on the real line, and the decode keeps the result physically legal.

```python
        rotations = fields("rot", 4)
        rotations /= np.maximum(np.linalg.norm(rotations, axis=1, keepdims=True), 1e-8)
```

`axis=1` means "compute the norm along each row", giving one number per splat.
`keepdims=True` keeps the result shaped `(N, 1)` instead of collapsing to `(N,)`, so it
**broadcasts** cleanly against the `(N, 4)` array — this is the broadcasting rule from §0 doing
real work. `np.maximum(..., 1e-8)` prevents division by zero for a degenerate quaternion.

```python
        colors = np.clip(C0 * fields("f_dc", 3) + 0.5, 0.0, 1.0)
```

3DGS does not store RGB directly. It stores coefficients of a **spherical harmonics** expansion of
view-dependent colour. This renderer uses only the degree-0 (view-independent, constant colour)
term, whose basis function is the constant

$$
Y_0^0 = \frac{1}{2\sqrt{\pi}} \approx 0.282094791773878
$$

which is the module-level constant `C0`. The `+ 0.5` and `np.clip` recentre and clamp the result
into the displayable $[0,1]$ range.

!!! note "Where view dependence comes back"
    A splat with only degree-0 colour looks identical from every angle. The full model keeps higher
    SH degrees so colour can vary with viewing direction. The renderers in this assignment stay at
    degree 0; the course trainers add the higher terms.

`to_ply` is the exact inverse — `np.log`, the logit $\log(p/(1-p))$, and `(c - 0.5)/C0`. Reader and
writer live in the same file on purpose: these encodings are easy to get subtly wrong, and a pair
that drifts apart produces files that load with quietly wrong geometry.

---

## 4. `shared/projected_gaussians.py` — the 2D result

```python
@dataclass
class ProjectedGaussians:
    centres: np.ndarray    # (M, 2) pixel coordinates
    conics: np.ndarray     # (M, 3) inverse-covariance entries A, B, C
    depths: np.ndarray     # (M,)
    colors: np.ndarray     # (M, 3)
    opacities: np.ndarray  # (M,)
```

The 3D scene flattened onto the image plane. Note **M, not N**: invisible splats have been dropped,
so this is usually shorter than the input, and the rows are in a different (depth-sorted) order.

Keeping this separate from `GaussianSet` draws a clean line: everything before it is 3D geometry,
everything after is 2D rasterisation. v2 and v3 both replace only the second half.

---

## 5. `shared/splat_math.py` — shared constants and the quaternion conversion

```python
SUPPORT_RADIUS_SQUARED = 9.0
ALPHA_CUTOFF = 1.0 / 255.0
```

`SUPPORT_RADIUS_SQUARED` is $3^2$: the squared Mahalanobis radius at which a Gaussian is treated as
having ended. Three standard deviations captures over 99% of the mass, so truncating there is
visually invisible and saves evaluating a long tail of near-zero contributions.

`ALPHA_CUTOFF` is $1/255$ — one step of an 8-bit colour channel. A contribution below this cannot
change the final image, so it is skipped.

Both live in `shared/` because the renderers and the trainers must agree on them. If a trainer
optimised splats under one cutoff and a renderer drew them under another, the image you get back
would not be the image the optimiser thought it was producing.

### `quaternion_to_matrix`

```python
def quaternion_to_matrix(quaternions: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternions.T
    return np.stack((
        1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
        ...
    ), axis=1).reshape(-1, 3, 3).astype(np.float32)
```

A **quaternion** is a four-number encoding of a 3D rotation. It is preferred over storing a 3×3
matrix because it has no redundancy to drift out of validity: renormalising four numbers restores a
legal rotation, whereas a 3×3 matrix nudged by an optimiser stops being a rotation in ways that are
awkward to repair.

`quaternions.T` transposes `(N, 4)` to `(4, N)`, so unpacking gives four arrays of length N — `w`
is *every* splat's w component at once. Every expression below then operates on all N splats
simultaneously.

`np.stack(..., axis=1)` gives `(N, 9)`, and `.reshape(-1, 3, 3)` reinterprets each row of 9 as a
3×3 matrix. `-1` means "work this dimension out from the total size".

The formula itself is the standard quaternion-to-rotation-matrix identity, valid for **unit**
quaternions — which is why `from_ply` normalised them.

---

## 6. `compact_support` — how far each splat reaches

```python
COMPACT_BOX_BETA = 1.0


def compact_support(opacities: np.ndarray) -> np.ndarray:
    supports = np.zeros(len(opacities), dtype=np.float32)
    visible = opacities > ALPHA_CUTOFF
    supports[visible] = np.minimum(
        SUPPORT_RADIUS_SQUARED,
        COMPACT_BOX_BETA * 2.0 * np.log(opacities[visible] / ALPHA_CUTOFF),
    )
    return supports
```

A fixed 3σ cutoff for every splat is wasteful. A **faint** splat reaches the invisibility threshold
much sooner than a strong one, so it can be truncated earlier without any visible change.

Solve for where a splat's contribution drops below `ALPHA_CUTOFF`. A splat of opacity $o$
contributes $o \cdot e^{-q/2}$ at squared distance $q$, so it becomes invisible when

$$
o\,e^{-q/2} < \alpha_{\text{cut}}
\quad\Longleftrightarrow\quad
q > 2\ln\frac{o}{\alpha_{\text{cut}}}
$$

That is the expression in the code. `np.minimum` caps it at 3σ so a very opaque splat does not get
an unboundedly large radius. `supports` stays `0.0` for splats fainter than the cutoff — they are
invisible everywhere, and a support of zero is how the rest of the code says "skip entirely".

Note the two index spaces here: `opacities[visible]` selects only the visible splats (boolean
indexing, §0), and `supports[visible] = ...` writes back to exactly those positions. Splats not
selected keep their initial zero.

`COMPACT_BOX_BETA` scales how aggressively this happens. All three renderers import this one
function, so they cannot drift apart — an earlier version of this code had v1 using a fixed 3σ
cutoff while v2 and v3 used compact support, and the "identical" renderers quietly produced
different images.

---

## 7. `project_gaussians` — the mathematical core

Everything so far was setup. This function turns 3D splats into 2D screen-space records.

### 7.1 Building the 3D covariance

```python
    rotation_world = quaternion_to_matrix(splats.rotations)   # (N, 3, 3)
    scale_squared = splats.scales * splats.scales             # (N, 3)
    covariance_world = np.einsum(
        "nij,nj,nkj->nik", rotation_world, scale_squared, rotation_world
    )                                                          # (N, 3, 3)
```

A splat's shape is the covariance matrix

$$
\Sigma = R\,S^2\,R^\top
$$

where $S$ is the diagonal matrix of scales and $R$ the rotation. Read right to left: start with an
axis-aligned ellipsoid stretched by the scales, then rotate it into place.

`np.einsum` is worth decoding carefully, because it is the densest line in the file. The string
labels each axis of each input, and the part after `->` says what the output axes are. Any label
appearing on the input side but **not** the output is summed over.

- `nij` — `rotation_world`, indexed by splat `n`, row `i`, column `j`
- `nj` — `scale_squared`, splat `n`, entry `j`
- `nkj` — `rotation_world` again, splat `n`, row `k`, column `j`
- `->nik` — output: splat `n`, row `i`, column `k`

`j` is absent from the output, so it is summed:

$$
\Sigma_{nik} = \sum_j R_{nij}\,(s^2)_{nj}\,R_{nkj}
$$

which is exactly $R S^2 R^\top$ — the second `R` indexed as `nkj` rather than `njk` supplies the
transpose. Doing this with `@` would need an explicit diagonal matrix per splat and two batched
multiplies; `einsum` expresses it in one pass with no temporaries.

### 7.2 Into camera space

```python
    world_h = np.concatenate(
        (splats.means, np.ones((len(splats.means), 1), dtype=np.float32)), axis=1
    )                                                    # (N, 4)
    camera_h = world_h @ camera.world_to_camera.T        # (N, 4)
    mean_camera = camera_h[:, :3]                        # (N, 3)
    depth = mean_camera[:, 2]                            # (N,)
```

Appending a column of ones turns each `(x, y, z)` into the homogeneous `(x, y, z, 1)` from §2.

The transpose in `world_to_camera.T` is a layout detail. The maths is $p' = Mp$ for a column vector
$p$, but `world_h` stores points as **rows**. Transposing the whole equation gives
$p'^\top = p^\top M^\top$, which is what `@ M.T` computes — for all N points at once.

`depth` is the z coordinate in camera space: distance in front of the camera. It drives both the
near-plane test and the draw order.

```python
    W = camera.world_to_camera[:3, :3]
    covariance_camera = W @ covariance_world @ W.T
```

Covariance transforms differently from a point. Under a linear map $W$,

$$
\Sigma' = W \Sigma W^\top
$$

Only the rotation block matters, because translation moves a shape without changing it.

Note this is `@` on a `(3,3)` and an `(N,3,3)`: NumPy treats the leading axis as a **batch**, so
one line rotates all N covariances.

### 7.3 Projecting to 2D

```python
    x, y, z = mean_camera.T
    centres = np.stack(
        (camera.fx * x / z + camera.cx, camera.fy * y / z + camera.cy), axis=1
    )                                                    # (N, 2)
```

Standard pinhole projection: divide by depth, scale by focal length, shift to the principal point.
Dividing by $z$ is what makes distant things smaller.

```python
    jacobian = np.zeros((len(splats.means), 2, 3), dtype=np.float32)
    jacobian[:, 0, 0] = camera.fx / z
    jacobian[:, 0, 2] = -camera.fx * x / (z * z)
    jacobian[:, 1, 1] = camera.fy / z
    jacobian[:, 1, 2] = -camera.fy * y / (z * z)
    covariance_screen = jacobian @ covariance_camera @ np.swapaxes(jacobian, 1, 2)
```

Here is the subtlety of the whole file. Perspective projection is **not linear** — it divides by
$z$ — so it cannot transform a covariance the way $W$ did. The standard solution is to
**linearise**: approximate the projection near each splat's centre by its derivative, the Jacobian

$$
J = \begin{pmatrix}
f_x/z & 0 & -f_x x/z^2 \\
0 & f_y/z & -f_y y/z^2
\end{pmatrix}
$$

Each entry is a partial derivative of the projected coordinate with respect to a camera-space
coordinate. The third column is the $z$ derivative, which is where the perspective effect lives:
moving a point deeper both shrinks and shifts its projection.

With $J$ in hand the covariance rule applies again, giving the 2×2 screen covariance
$\Sigma_{2D} = J\,\Sigma_{cam}\,J^\top$. `np.swapaxes(jacobian, 1, 2)` transposes each `(2,3)` in
the batch to `(3,2)` — `.T` would reverse *all* axes including the batch, which is not what we want.

This approximation is why very large splats near the image edge can look subtly wrong: the
linearisation is only accurate near the point it was taken about.

```python
    covariance_screen += filter_variance * np.eye(2, dtype=np.float32)
```

Adding a small multiple of the identity is a **low-pass filter**. It sets a floor on how small a
splat can be on screen, roughly half a pixel wide. Without it, a splat that projects to less than a
pixel falls between sample points and flickers as the camera moves.

### 7.4 Culling

```python
    eigenvalues = np.linalg.eigvalsh(covariance_screen)      # (N, 2), ascending
    radii = np.sqrt(qmax * np.maximum(eigenvalues[:, 1], 0.0))
```

`eigvalsh` returns eigenvalues of a **symmetric** matrix in ascending order, so column 1 is the
larger. For a covariance, eigenvalues are the squared extents along the ellipse's own axes, so the
largest one bounds the splat in every direction. `np.maximum(..., 0.0)` guards against a tiny
negative value from floating-point error, which would make `np.sqrt` return NaN.

```python
    visible = (
        (depth > near)
        & (centres[:, 0] + radii >= 0.0)
        & (centres[:, 0] - radii < camera.width)
        & (centres[:, 1] + radii >= 0.0)
        & (centres[:, 1] - radii < camera.height)
    )
```

Five conditions: in front of the camera, and its bounding box overlaps the image. `&` is
elementwise **and** on boolean arrays — Python's `and` would try to collapse each array to a single
truth value and raise. Using the radius rather than just the centre keeps splats whose centre is
off-screen but whose body still covers visible pixels.

The near-plane test is essential rather than cosmetic: at $z \le 0$ the division in the projection
either explodes or flips the image behind the camera onto the screen.

### 7.5 Sorting and packing

```python
    indices = np.flatnonzero(visible)
    if len(indices) == 0:
        return ProjectedGaussians(... empty arrays ...)
```

`np.flatnonzero` converts the boolean mask into the integer positions of the `True` entries. The
early return handles a camera pointing at nothing — without it the code below would index empty
arrays and fail confusingly.

```python
    indices = indices[np.argsort(depth[indices])]

    inverse = np.linalg.inv(covariance_screen[indices])
    conics = np.stack((inverse[:, 0, 0], inverse[:, 0, 1], inverse[:, 1, 1]), axis=1)
```

`np.argsort` returns the indices that *would* sort the array, rather than the sorted values —
exactly what is needed to reorder several parallel arrays consistently. Sorting the index array
once, up front, means every array built afterwards is in the same order by construction.

Then each 2×2 screen covariance is inverted, because the Gaussian is evaluated with
$\Sigma^{-1}$, not $\Sigma$. The result is symmetric, so only three of the four entries are stored:

$$
\Sigma^{-1} = \begin{pmatrix} A & B \\ B & C \end{pmatrix}
$$

This three-number form is called a **conic**. Storing three floats instead of four, and inverting
once per splat rather than once per pixel, is what makes the blending loop cheap.

Front-to-back order matters because the blending in §8 accumulates transmittance and stops early
once a pixel is opaque — which is only correct if the nearest splats arrive first.

---

## 8. `CpuRenderer` — turning splats into pixels

```python
        projected = project_gaussians(splats, self.camera, ...)
        image = np.zeros((self.camera.height, self.camera.width, 3), dtype=np.float32)
        background_color = np.asarray(background, dtype=np.float32)
        supports = compact_support(projected.opacities)
```

Project once, allocate the output, and compute each splat's cutoff radius once — not per pixel.

### 8.1 The alpha-compositing loop

The skeleton leaves this loop for you to write. What follows is the specification and the
reasoning — not the code.

### What the loop must do

Two nested loops walk the pixels, and the sample point is the pixel **centre**: offset both
coordinates by `+ 0.5`. Sampling at corners instead biases the whole image half a pixel diagonally.

This loop is why v1 is slow. For a 400×400 image that is 160,000 iterations of interpreted Python,
each of which then walks splats. Everything else in the file was vectorised precisely because this
part cannot be.

For each pixel you need two running values: an accumulated colour, starting black, and a
**transmittance**, starting at 1. Transmittance is how much light from behind can still get
through; it shrinks toward 0 as opaque splats pile up in front.

Then walk the splats **in the order `project_gaussians` returned them**, which is front to back.

### The four quantities to compute per splat

**1. Offset from the splat's projected centre.** Call it $(du, dv)$ — the pixel's position minus
`projected.centres[i]`.

**2. Squared Mahalanobis distance.** With the conic $(A, B, C)$ from §7.5 standing for
$\Sigma^{-1} = \begin{pmatrix} A & B \\ B & C \end{pmatrix}$,

$$
q = \begin{pmatrix}du & dv\end{pmatrix}\Sigma^{-1}\begin{pmatrix}du \\ dv\end{pmatrix}
  = A\,du^2 + 2B\,du\,dv + C\,dv^2
$$

`B` appears twice because the matrix is symmetric. Small $q$ means near the splat's peak, large $q$
means out in its tails. A contour of constant $q$ is an ellipse on screen — this is a distance that
accounts for the splat's shape and orientation.

**3. Alpha at this pixel.** Two tests come first: skip a splat whose support is `0.0` (invisible
everywhere, §6), and skip one where $q$ exceeds that splat's support (this pixel is past its tail).
Otherwise the Gaussian density here is $e^{-q/2}$, a direct discretisation of

$$
G(x) = \exp\!\left(-\tfrac12 (x-\mu)^\top\Sigma^{-1}(x-\mu)\right)
$$

Multiply by the splat's opacity to get its alpha: strong at the centre, fading to nothing at the
edges. Cap the result at **0.99** so no single splat becomes fully opaque and something always
remains of what is behind it — a numerical safeguard carried over from the original 3DGS
implementation. Then discard the splat if its alpha is below `ALPHA_CUTOFF`, since it cannot move
an 8-bit channel.

**4. Composite.** This is the **over** operator, the standard front-to-back rule:

$$
C = \sum_i T_i\,\alpha_i\,c_i, \qquad T_i = \prod_{j<i}(1-\alpha_j)
$$

Each splat adds its colour scaled by its own alpha *and* by the transmittance remaining in front of
it. Afterwards the transmittance shrinks by the fraction this splat blocked, i.e. multiply it by
$(1 - \alpha)$.

### Two details that are easy to miss

**Stop early.** Once transmittance drops below $10^{-4}$ the pixel is effectively opaque and every
remaining splat is invisible. Breaking out is why the depth sort in §7.5 was worth doing; in a dense
scene it skips the overwhelming majority of the work. It is only correct because the splats arrive
nearest-first.

**Finish with the background.** The pixel's final value is the accumulated colour **plus the
remaining transmittance times the background colour**. Whatever light was never blocked shows the
background through it. Omitting this term is the classic bug here: the image looks plausible, but
every partially-covered pixel is too dark, because leftover transmittance was silently treated as
black. Both parallel versions had exactly this bug at one point, and it is why all three renderers
are checked against each other.

---

## 9. `main` — the command-line entry point

```python
    camera = Camera.from_look_at(
        args.width, args.height, args.focal_length,
        args.camera_position, args.look_at, args.up,
    )
    splats = GaussianSet.from_ply(args.ply)
    image = CpuRenderer(camera).render(splats)
    Image.fromarray(np.uint8(np.clip(image, 0.0, 1.0) * 255.0)).save(args.output)
```

Build a camera, load splats, render, save. The last line converts float colours to 8-bit: `np.clip`
first, because values can exceed 1.0 and would otherwise **wrap around** — a slightly-too-bright
pixel turning black rather than saturating white.

---

## 10. The pipeline, end to end

| Stage | Input | Output | Cost |
|---|---|---|---|
| Load | `.ply` file | `GaussianSet` (N splats) | once |
| Build covariance | scales + rotations | `(N,3,3)` world covariance | vectorised |
| To camera space | world covariance + camera | camera-space means and covariance | vectorised |
| Project | camera space + Jacobian | 2D centres, 2×2 covariance | vectorised |
| Cull | bounds + near plane | M ≤ N visible splats | vectorised |
| Sort | depths | front-to-back order | vectorised |
| Invert | 2×2 covariance | conics (A, B, C) | vectorised |
| **Blend** | conics + colours | pixels | **Python loop, P × M** |

Every stage is a fast vectorised NumPy call except the last, which is a Python loop costing
$O(P \times M)$ for P pixels and M splats. At 400×400 with 100,000 splats that is 16 billion
iterations of interpreted Python.

Two independent problems live in that last row, and the next two documents take one each:

- The work is **serial**, though every pixel is independent. → [v2](3dgs_renderer_v2_annotated.md)
  runs them in parallel.
- The work is **quadratic**: every pixel tests every splat, though almost all are nowhere near it.
  → [v3](3dgs_renderer_v3_annotated.md) makes each pixel consider only nearby splats.

All three produce **bit-identical images**. That is the point: the maths in this document does not
change again, only how fast it is evaluated.

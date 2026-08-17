# Annotated Walkthrough: `3dgs_renderer_v2.py`

Second of three. Read [v1](3dgs_renderer_v1_annotated.md) first — this document assumes the whole
projection pipeline, the conic representation, the compositing rule, and the NumPy vocabulary
explained there, and does not repeat any of it.

v2 changes **one thing**: the per-pixel blending loop runs in parallel instead of serially. The
maths is untouched. If you find yourself wondering what a conic is or why splats are depth-sorted,
that answer is in v1.

This document assumes you know **nothing** about Warp.

---

## 0. The mental model: what Warp is, and why this file looks strange

v1 ended with two nested Python loops:

```python
for py in range(height):
    for px in range(width):
        ...   # blend this pixel
```

One pixel at a time, on one CPU core. But notice: **no pixel depends on any other**. Pixel (0,0)
and pixel (399,399) could be computed simultaneously and neither would notice. The serial loop is
not required by the algorithm — it is an artefact of writing it in Python.

Warp's idea: **write the body that handles a single pixel once, as a restricted function called a
kernel, and let Warp run that function for every pixel at once** — across CPU threads or thousands
of GPU cores.

Three new concepts, each with its own section below:

| Concept | Role |
|---|---|
| `@wp.kernel` | Marks a function as parallel code, not ordinary Python. Its body describes **one** work item. |
| `wp.array`, `wp.vec2`, `wp.vec3` | Warp's array and small-vector types, living in memory a kernel can reach — on CPU or GPU. |
| `wp.launch(...)` | Actually runs it: "start N copies of this kernel, one per pixel." |

The mental shift that trips people up: inside a kernel you are **not** writing code that runs
top-to-bottom once. You are writing the body of one iteration of a loop that no longer exists in
your source. Warp supplies the loop, and the only way a kernel knows *which* iteration it is
running is by asking.

That restriction is what buys the speed, and it is also why kernel syntax is stricter: explicit
`float(...)` conversions, `wp.min` instead of `min`, and no NumPy at all.

---

## 1. Imports and the deliberate reuse of v1

```python
import warp as wp

_reference = importlib.import_module("3dgs_renderer_v1")
Camera = _reference.Camera
GaussianSet = _reference.GaussianSet
project_gaussians = _reference.project_gaussians
SUPPORT_RADIUS_SQUARED = _reference.SUPPORT_RADIUS_SQUARED
compact_support = _reference.compact_support
ALPHA_CUTOFF = _reference.ALPHA_CUTOFF
TRANSMITTANCE_CUTOFF = 1.0e-4
```

`importlib.import_module` is used instead of a plain `import` for a mundane reason: the module name
starts with a digit, and `import 3dgs_renderer_v1` is a syntax error. `importlib` takes the name as
a string, so it does not care.

Everything mathematical is **imported from v1**, not reimplemented. Projection, culling, sorting and
the support rule all still run as ordinary NumPy on the CPU. Only the blending moves.

That is a deliberate teaching choice, and also good engineering: it keeps the comparison honest. If
v2 reimplemented projection too, a difference in output could come from anywhere. Because the
front half is shared code, any difference must be in the part that changed.

---

## 2. The kernel

```python
@wp.kernel
def rasterize(
    centres: wp.array(dtype=wp.vec2),
    conics: wp.array(dtype=wp.vec3),
    colours: wp.array(dtype=wp.vec3),
    opacities: wp.array(dtype=wp.float32),
    supports: wp.array(dtype=wp.float32),
    count: int,
    width: int,
    background: wp.vec3,
    image: wp.array(dtype=wp.vec3),
):
```

`@wp.kernel` tells Warp to compile this function rather than run it as Python. Warp inspects the
source and generates C++ or CUDA from it, which is why every parameter needs a type annotation:
the generated code must know the exact layout of everything.

`wp.array(dtype=wp.vec2)` is an array of 2-vectors — the same idea as NumPy's `(M, 2)`, but as an
array of small fixed-size vectors rather than a 2D grid. `wp.vec3` likewise. Scalars like `count`
and `width` are passed by value to every parallel copy.

`background` is a single `wp.vec3`, not an array: one value shared by all pixels.

```python
    pixel = wp.tid()
    px = float(pixel % width) + 0.5
    py = float(pixel // width) + 0.5
```

`wp.tid()` is the **thread id** — the "which iteration am I?" question from §0. If the kernel was
launched with `dim=160000`, then across the 160,000 parallel copies `wp.tid()` returns
0, 1, 2, … 159999, one value each.

The launch is one-dimensional, so the linear index is unpacked into 2D coordinates by hand:
`pixel % width` is the column, `pixel // width` the row. The `+ 0.5` centres the sample exactly as
v1 did.

`float(...)` is required. Warp is strictly typed and will not silently mix an integer with a float
the way Python does.

The body is yours to write, and it is **the same algorithm as v1 §8.1** — same Mahalanobis $q$,
same $e^{-q/2}$, same 0.99 cap, same over-operator, same early exit, same background composite. If
you have a working v1, you are translating, not redesigning.

What changes is the spelling, because kernels are not ordinary Python:

| v1 (NumPy/Python) | v2 (Warp kernel) | why |
|---|---|---|
| `min(...)`, `np.exp(...)` | `wp.min(...)`, `wp.exp(...)` | kernels cannot call NumPy or most Python builtins |
| `total *= x` | `total = total * x` | Warp wants explicit assignment |
| `total += x` | `total = total + x` | same |
| `t = 1.0` | `t = float(1.0)` | explicit typing; Warp will not infer it |
| `np.zeros(3)` | `wp.vec3(0.0, 0.0, 0.0)` | Warp's own small-vector type |

Two structural points about the translation:

**The pixel loops are gone.** They have been replaced by `wp.tid()`. What remains is a single loop
over splats, which every thread runs independently. Each thread still walks **every splat in the
scene** — that is the quadratic cost v1 identified, and it survives here untouched. v2 fixes
parallelism, not complexity.

**One thread writes one pixel.** The kernel's last statement writes `image[pixel]`, the element
indexed by this thread's own `wp.tid()`. No two threads ever write the same location, which is why
no locking appears anywhere in this file. Internalise that discipline — it is the single most
important habit in parallel code, and v3 preserves it.

One caveat specific to GPUs: threads execute in lockstep groups, so a thread that hits the early
exit does not release its slot until its whole group finishes. Neighbouring pixels tend to have
similar depth complexity so it works out in practice, but the early exit buys less here than it did
in v1.

---

## 3. `WarpRenderer` — memory that outlives a frame

```python
    def __init__(self, width: int, height: int, maximum_splats: int, device: str):
        self.device = wp.get_device(device)
        self.centres = wp.zeros(maximum_splats, dtype=wp.vec2, device=self.device)
        self.conics = wp.zeros(maximum_splats, dtype=wp.vec3, device=self.device)
        self.colours = wp.zeros(maximum_splats, dtype=wp.vec3, device=self.device)
        self.opacities = wp.zeros(maximum_splats, dtype=wp.float32, device=self.device)
        self.supports = wp.zeros(maximum_splats, dtype=wp.float32, device=self.device)
        self.image = wp.zeros(width * height, dtype=wp.vec3, device=self.device)
```

`wp.get_device("cpu")` or `wp.get_device("cuda:0")` selects where the arrays live and where kernels
run. The same source works for both; only this string changes.

The buffers are allocated **once**, at construction, sized for the worst case
(`maximum_splats`) rather than per frame. Allocating GPU memory is expensive, and a renderer called
in a loop would otherwise spend much of its time allocating and freeing. This is why the class
exists at all — v1 needed no class because NumPy allocation is cheap enough not to notice.

`self.image` is `width * height` **flat**, not 2D, matching the 1D launch in §2.

### `WarpRenderer.render`

```python
        projected = project_gaussians(splats, camera)
        count = len(projected.opacities)
        if count > self.maximum_splats:
            raise ValueError(f"Renderer capacity {self.maximum_splats:,} is below {count:,} visible splats.")
```

Projection is v1's, on the CPU, producing NumPy arrays. Then a capacity check — the buffers are
fixed size, so exceeding them must be a clear error rather than a silent overflow.

```python
        self.centres.assign(wp.array(projected.centres, dtype=wp.vec2, device=self.device))
        ...
        supports = compact_support(projected.opacities)
        self.supports.assign(wp.array(supports, dtype=wp.float32, device=self.device))
```

`wp.array(numpy_array, ...)` copies host data into Warp memory; `.assign(...)` writes it into the
already-allocated buffer rather than replacing it. This is the **host-to-device transfer**, and on
a real GPU it crosses the PCIe bus.

Note the asymmetry this creates: projection on the CPU, blending on the GPU, so every frame pays a
copy. For this assignment that is fine and keeps the comparison clean, but it is the first thing a
production renderer would eliminate by projecting on the device too.

```python
        wp.launch(rasterize, dim=self.width * self.height,
                  inputs=[self.centres, self.conics, self.colours, self.opacities, self.supports,
                          count, self.width, wp.vec3(*background), self.image],
                  device=self.device)
```

The launch. `dim` is how many parallel copies to run — one per pixel, which is what makes
`wp.tid()` range over exactly the pixels. `inputs` must match the kernel signature in order and
type. `wp.vec3(*background)` converts the Python tuple into Warp's vector type.

```python
        return self.image.numpy().reshape(self.height, self.width, 3)
```

`.numpy()` copies device memory back to the host as a NumPy array — the **device-to-host**
direction. `.reshape` restores the 2D image shape the caller expects, undoing the flattening.

On a GPU, kernel launches are **asynchronous**: `wp.launch` returns immediately, having only
queued the work. `.numpy()` forces a wait for it to finish. That is invisible here, but it matters
enormously as soon as you try to *time* anything — a timer around `wp.launch` alone measures the
queuing, not the work.

---

## 4. What did not change

Worth stating explicitly, because it is the lesson of this version:

- The projection maths, the culling, the depth sort, the conic representation — all v1's.
- The blending rule, the cutoffs, the 0.99 cap, the background composite — identical.
- The **output**. v1 and v2 produce bit-identical images. Not "visually indistinguishable":
  identical, and the assignment checks this.

What changed is *when* the work happens: 160,000 sequential Python iterations became 160,000
parallel kernel invocations.

---

## 5. What is still wrong

v1 identified two problems with its blending loop. v2 has fixed exactly one.

**Fixed — the work is now parallel.** Every pixel runs at the same time.

**Not fixed — the work is still quadratic.** Look again at `for splat in range(count)`: every
pixel still examines **every splat in the scene**. For a 400×400 image and 100,000 splats that is
16 billion `q` evaluations, and parallelism divides that by your core count, not by anything
fundamental.

And nearly all of it is wasted. A typical splat covers a few dozen pixels out of 160,000, so for
almost every (pixel, splat) pair the answer is "not close, skip" — computed at full cost, in every
thread, for every splat.

The fix is to give each pixel a short list of *only the splats that could possibly affect it*, so
the loop bound shrinks from "all splats" to "a handful". That is
[v3](3dgs_renderer_v3_annotated.md), and it is where the real speedup lives.

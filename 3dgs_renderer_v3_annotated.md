# Annotated Walkthrough: `3dgs_renderer_v3.py` and `shared/tile_builder.py`

Third and last. Read [v1](3dgs_renderer_v1_annotated.md) and
[v2](3dgs_renderer_v2_annotated.md) first — this document assumes the projection maths from v1 and
the whole Warp model (`@wp.kernel`, `wp.tid()`, `wp.launch`, device memory) from v2, and repeats
none of it.

v2 left one problem unsolved: **every pixel still tests every splat**. v3 fixes that, and in doing
so introduces the one genuinely new algorithmic idea in the assignment.

---

## 0. Why tiling

A splat, once projected, covers a small elliptical patch — typically a few dozen pixels out of
160,000. So for almost every (pixel, splat) pair the answer is "nowhere near, skip", and v2 pays
full price to discover that, in every thread, for every splat.

The obvious fix — "each pixel looks up only nearby splats" — needs a data structure, and building
one per pixel is far too expensive. The standard compromise is to work at a coarser granularity:

> Chop the image into fixed **tiles** (here 16×16 pixels). For each tile, build a list of the splats
> that touch it. Every pixel in a tile then walks only that tile's list.

256 pixels share one list, so the list is built 256 times less often than a per-pixel one would be,
while still excluding the overwhelming majority of irrelevant splats.

There are two ways to build those lists:

- **Tile-first** — for each tile, search all splats. Still quadratic; no better than v2.
- **Gaussian-first** — for each splat, work out which tiles it touches and emit one record per
  tile. Cost is proportional to *actual overlaps*, not to tiles × splats.

v3 takes the Gaussian-first route, which is what its class name says. The catch is that each splat
emits a **variable** number of records, and thousands of threads must write into one shared output
array without colliding. Resolving that is most of `tile_builder.py`.

---

## 1. Tile geometry

```python
        self.tile_size = tile_size                                   # 16
        self.tiles_x = (width + tile_size - 1) // tile_size
        self.tiles_y = (height + tile_size - 1) // tile_size
        self.tile_count = self.tiles_x * self.tiles_y
```

`(width + tile_size - 1) // tile_size` is integer **ceiling division**: it rounds up so a 400-pixel
width still gets 25 tiles rather than 24 and a bit. Tiles are numbered row-major, so tile
`(tile_x, tile_y)` has index `tile_y * tiles_x + tile_x`.

```python
        self.tile_pair_capacity = (
            int(tile_pair_capacity) if tile_pair_capacity is not None
            else max(1, maximum_splats * 32)
        )
```

A **splat-to-tile pair** is one record saying "splat 7 touches tile 42". The total is not known
until the splats are projected, but GPU buffers must be sized in advance, so the code reserves room
for an average of 32 tiles per splat. A splat spanning more than that is fine as long as others
span fewer; exceeding the total raises a clear error rather than corrupting memory.

---

## 2. `shared/tile_builder.py` — building the lists

This is the heart of v3. The builder runs five stages, and the reason it takes five rather than one
is the variable-length output problem from §0.

### 2.1 Count how many tiles each splat touches

```python
@wp.kernel(enable_backward=False)
def count_projected_tile_pairs(centres, conics, supports, count, width, height,
                               tile_size, tiles_x, tiles_y, tile_bounds, pair_counts):
    splat = wp.tid()
    pair_counts[splat] = 0
    tile_bounds[splat] = wp.vec4i(0, -1, 0, -1)
```

One thread per splat. Both outputs are initialised first, because threads beyond `count` still run
and must leave defined values behind. The sentinel bounds `(0, -1, 0, -1)` describe an empty range:
any loop `for x in range(0, -1 + 1)` runs zero times.

`enable_backward=False` tells Warp not to generate a derivative version of this kernel. Tiling is
discrete bookkeeping — there is nothing to differentiate — and skipping it saves compile time.

```python
        determinant = conic[0] * conic[2] - conic[1] * conic[1]
        if determinant > 1.0e-12 and support > 0.0:
            cov_xx = conic[2] / determinant
            cov_yy = conic[0] / determinant
            radius_x = wp.sqrt(wp.max(support * cov_xx, 0.0))
            radius_y = wp.sqrt(wp.max(support * cov_yy, 0.0))
```

v1 stored the **inverse** covariance (the conic) because that is what evaluating the Gaussian
needs. Bounding the splat needs the covariance itself, so it is recovered by inverting the 2×2
back. For

$$
\Sigma^{-1} = \begin{pmatrix} A & B \\ B & C \end{pmatrix},
\qquad
\Sigma = \frac{1}{AC - B^2}\begin{pmatrix} C & -B \\ -B & A \end{pmatrix}
$$

so `cov_xx = C/det` and `cov_yy = A/det` — note the swap, which is easy to misread. The
`determinant > 1e-12` guard rejects a degenerate splat that has collapsed to a line, where the
inversion would explode.

The radius along each axis is then $\sqrt{q_{\max}\,\sigma^2}$, the same relationship v1 used for
culling, now per axis to give an axis-aligned bounding box.

```python
                min_tx = wp.max(min_px // tile_size, 0)
                max_tx = wp.min(max_px // tile_size, tiles_x - 1)
                ...
                if min_tx <= max_tx and min_ty <= max_ty:
                    tile_bounds[splat] = wp.vec4i(min_tx, max_tx, min_ty, max_ty)
                    pair_counts[splat] = (max_tx - min_tx + 1) * (max_ty - min_ty + 1)
```

Pixel bounds become tile bounds by integer division, clamped to the image. The count is simply the
area of that tile rectangle. This is a **conservative** bound: an ellipse does not fill its
bounding box, so some tiles in the corners get a record for a splat that does not actually reach
them. Those cost a wasted `q` evaluation later but never a wrong pixel — the `q <= support` test
still rejects them.

### 2.2 Turn counts into positions

```python
        wp.utils.array_scan(self.pair_counts[:item_count], self.pair_prefix[:item_count],
                            inclusive=True)
```

This is the step that solves the variable-length write problem, and it is worth dwelling on.

Each splat now knows *how many* records it will write, but not *where* to write them. If every
thread simply appended, they would overwrite each other.

A **prefix sum** (or scan) fixes this. Given counts `[3, 0, 2, 4]` the inclusive scan is
`[3, 3, 5, 9]` — each entry is the running total up to and including that position. Subtract your
own count and you have your **start offset**: splat 2 writes at `5 - 2 = 3`. Every thread computes
its own slot with no communication, and no two slots overlap.

The last entry is also the grand total, which is exactly what `copy_pair_count` reads out:

```python
    if item_count > 0:
        pair_count[0] = pair_prefix[item_count - 1]
```

```python
        pair_count = int(self.pair_count.numpy()[0])
```

`.numpy()` copies from device to host, so this line **blocks** until the GPU has finished
everything queued so far (the asynchrony noted at the end of v2 §3). It is unavoidable here — the
host needs the real count to size the following launches — but it is the one hard synchronisation
point in the frame, and worth remembering if you ever profile this code.

### 2.3 Emit the records

```python
        output = pair_prefix[item] - count
        group_base = group_ids[item] * tiles_per_view
        for tile_y in range(bounds[2], bounds[3] + 1):
            for tile_x in range(bounds[0], bounds[1] + 1):
                group = group_base + tile_y * tiles_x + tile_x
                depth_keys[output] = depths[item]
                packed_pairs[output] = (
                    wp.uint64(group) << wp.uint64(32)
                ) | wp.uint64(splat_ids[item])
                output += 1
```

Each thread walks its own tile rectangle and writes one record per tile, starting at the offset
derived in §2.2.

The record is **bit-packed** into a single 64-bit integer: the tile index in the high 32 bits, the
splat index in the low 32. `<< 32` shifts the group up; `|` merges the two halves.

```
 63                    32 31                     0
+------------------------+------------------------+
|      tile (group)      |      splat index       |
+------------------------+------------------------+
```

Packing them together means a single sort can order by tile and carry the splat along with it,
rather than sorting two arrays in lockstep.

`group_ids` and `tiles_per_view` exist because this builder is shared with the course trainers,
which tile several camera views at once. The renderer passes all-zero group ids and a single view,
so `group_base` is 0 and `group` is just the tile index.

### 2.4 Sort — twice, and the order matters

```python
        wp.utils.radix_sort_pairs(self.depth_keys, self.packed_pairs, pair_count)
```

First sort **by depth**, carrying the packed records along. After this the records are globally
near-to-far but scattered across tiles.

```python
        wp.launch(unpack_group_keys, ...)     # group_keys[i] = packed_pairs[i] >> 32
        wp.utils.radix_sort_pairs(self.group_keys, self.packed_pairs, pair_count,
                                  end_bit=max(1, (group_count - 1).bit_length()))
```

Then extract the tile index from the high bits and sort **by tile**.

The trick is that a radix sort is **stable**: records comparing equal keep their relative order. So
grouping by tile preserves the depth ordering established by the first sort, and each tile's run
comes out sorted near to far — exactly what the compositing rule needs.

`end_bit` tells the sort how many bits actually matter. `(group_count - 1).bit_length()` is the
number of bits needed to represent the largest tile index; for 625 tiles that is 10 bits rather
than 32, so the sort does a third of the passes.

!!! note "A cheaper alternative"
    A single sort on a key of `(tile << 32) | depth_bits` would achieve the same ordering in one
    pass. The two-pass version here is easier to read and to reason about, which is why the
    teaching implementation keeps it.

### 2.5 Find where each tile's run starts

```python
        wp.launch(count_sorted_groups, ...)   # group_counts[group] += 1
        wp.utils.array_scan(self.group_counts[:group_count],
                            self.tile_offsets[1:group_count + 1], inclusive=True)
```

`count_sorted_groups` uses `wp.atomic_add`, which is how many threads safely increment the same
counter: the hardware serialises just that one update, so no increments are lost. This is the one
place in the assignment where threads *do* write to shared locations, and it needs an atomic
precisely because the "one thread, one output slot" discipline from v2 §2 does not apply.

A second prefix sum then turns per-tile counts into **start offsets**, written from index 1 so that
`tile_offsets[0]` stays 0. The result is the standard compressed layout: tile `t` owns the records
from `tile_offsets[t]` up to `tile_offsets[t + 1]`.

```python
        return self.tile_offsets, self.packed_pairs, pair_count
```

Two arrays are all the raster kernel needs: where each tile's list begins and ends, and the records
themselves.

!!! note "Profiling hook"
    The course copy of this file wraps each stage in an opt-in timer
    (`WARP_3DGS_PROFILE_TILES=1`). It is off by default and does not affect the algorithm; timing
    GPU stages requires inserting synchronisation that would otherwise distort what it measures.

---

## 3. The tiled raster kernel

```python
    pixel = wp.tid()
    px_i = pixel % width
    py_i = pixel // width

    tile_x = px_i // tile_size
    tile_y = py_i // tile_size
    tile = tile_y * tiles_x + tile_x
```

Still one thread per pixel, exactly as v2. The new step is finding which tile this pixel belongs to
— two integer divisions.

The skeleton stops here and leaves the rest to you. **This is the entire point of v3**, so it is
worth being precise about what changes and what does not.

### The loop bound is the whole idea

| | loop over | typical length |
|---|---|---|
| v2 | every splat in the scene | 100,000 |
| v3 | only this pixel's tile list | tens |

`tile_offsets` uses the compressed layout built in §2.5: tile `t` owns the records running from
`tile_offsets[t]` up to, but not including, `tile_offsets[t + 1]`. That pair of numbers is your
loop's start and end. Nothing else about the traversal changes.

### Recovering the splat index

The records in `packed_pairs` are the bit-packed 64-bit values from §2.3 — tile index in the high
32 bits, splat index in the low 32. The loop gives you a record; you need the splat it refers to,
which means masking off the high half with `& 0xFFFFFFFF` and converting to an `int`.

The tile index in the high bits has already done its job during sorting and is not needed here.
Watch the types: the mask must be a `wp.uint64` for Warp to accept the operation.

### Everything after that is unchanged

Once you have the splat index, the body is **identical to v2's**, which is identical to v1's: the
same $q$, alpha, 0.99 cap, over-operator, early exit and background composite. Nothing about the
shading changed — only which splats are considered. If your v2 kernel works, this is that same
body with the loop bounds swapped and the index unpacked.

From here the body is **identical to v2's**, which is identical to v1's: the same $q$, alpha, cap,
over-operator, early exit and background composite. Nothing about the shading changed — only which
splats are considered. If your v2 kernel works, the tiled kernel is that same body with the loop
bounds swapped and the splat index unpacked.

The records arrive already sorted near-to-far within the tile (§2.4), so the front-to-back
assumption the compositing rule depends on still holds — and the early exit is now more valuable,
because a pixel that saturates can abandon a short list rather than a long one.

---

## 4. `GaussianFirstWarpRenderer.render`

```python
        self.depths.assign(wp.array(projected.depths, dtype=wp.float32, device=self.device))

        offsets, packed_pairs, _ = self.builder.build(
            self.centres, self.conics, self.supports, self.depths,
            self.group_ids, self.splat_ids, count, self.tile_count,
        )
```

Two differences from v2's `render`. Depths are now uploaded — v2 never needed them on the device,
because its global ordering was already baked into the array order by v1's sort, whereas the
builder must re-sort per tile. And the builder runs between upload and raster.

```python
        self.splat_ids = wp.array(np.arange(maximum_splats, dtype=np.uint32), ...)
        self.group_ids = wp.zeros(maximum_splats, dtype=wp.int32, ...)
```

`splat_ids` is just `0, 1, 2, …` — the identity mapping, since the projected arrays are already
indexed by position. `group_ids` is all zeros: one camera view, as noted in §2.3.

---

## 5. The three versions side by side

| | v1 | v2 | v3 |
|---|---|---|---|
| Pixel loop | Python, serial | Warp, parallel | Warp, parallel |
| Splats per pixel | all M | all M | only this tile's list |
| Ordering | one global depth sort | same, reused | per-tile, via stable two-pass sort |
| Extra structures | none | device buffers | tile offsets + packed pairs |
| Cost | $O(P \times M)$ serial | $O(P \times M)$ parallel | $O(P \times k)$ parallel, $k \ll M$ |
| Output | reference | identical to v1 | identical to v1 |

The last row is the one to hold on to. Three quite different programs, one image. v2 changed *when*
the work happens; v3 changed *how much* work there is. Neither changed *what is computed*, which is
why the assignment can check all three against each other exactly.

That is also the practical lesson: the correctness of a renderer lives in v1's maths, and every
optimisation afterwards has to earn its place by leaving the output alone. If your v3 image differs
from your v1 image, the bug is in the bookkeeping — the bounds, the packing, the sort stability, or
the tile lookup — never in the compositing, because that code is unchanged.

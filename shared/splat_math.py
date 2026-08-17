"""Splat primitives shared by the assignment renderers and the Warp trainers.

These used to live in `examples/3dgs_renderer_cpu.py`, a third copy of the CPU renderer that sat
alongside the two assignment copies (`a1-solution/` and `a1-skeleton/`). The trainer tree only ever
needed these few primitives from it, so they live here and the renderer exists in exactly two
places: the complete version for reference, and the version with parts removed for students.
"""

from __future__ import annotations

import numpy as np

# Splats are culled/composited past this many squared Mahalanobis units from centre.
SUPPORT_RADIUS_SQUARED = 9.0

# A contribution below this alpha is dropped; also the opacity floor for compact support.
# Every rasterizer in the tree must read this one value. The training and validation rasterizers
# disagreeing on the cutoff once cost 2.3 dB of reported PSNR, and looked like a model regression
# rather than a rendering bug, so the constant lives in exactly one place.
ALPHA_CUTOFF = 1.0 / 255.0

# Compositing stops once this little light is left to attenuate.
TRANSMITTANCE_CUTOFF = 1.0e-4

# Degree-1 and degree-2 real spherical-harmonic basis functions, per colour channel.
SH_REST_BASIS = 8

# Tile edge in pixels. The trainers rasterize against this and bound `sparse.samples_per_tile` by
# its square, so it lives beside the other constants every rasterizer must agree on. The
# standalone renderers take their tile size as a parameter and do not read this.
TILE = 16


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

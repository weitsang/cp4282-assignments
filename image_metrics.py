"""Small NumPy-only image quality metrics shared by the evaluation examples."""

from __future__ import annotations

import numpy as np


def _box_filter(image: np.ndarray, window: int) -> np.ndarray:
    """Apply a reflected, uniform square window without a SciPy dependency."""
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd number")
    radius = window // 2
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    integral = np.pad(padded, ((1, 0), (1, 0), (0, 0))).cumsum(0).cumsum(1)
    height, width = image.shape[:2]
    total = (
        integral[window:window + height, window:window + width]
        - integral[:height, window:window + width]
        - integral[window:window + height, :width]
        + integral[:height, :width]
    )
    return total / float(window * window)


def ssim_rgb(x: np.ndarray, y: np.ndarray, window: int = 11, data_range: float = 1.0) -> float:
    """Return mean single-scale SSIM for two RGB images in [0, data_range]."""
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if x.shape != y.shape or x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("SSIM expects two same-shaped H x W x 3 images")
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mean_x = _box_filter(x, window)
    mean_y = _box_filter(y, window)
    # Floating-point cancellation can make the variance of a flat window slightly
    # negative. Variance is non-negative, so clamp that numerical noise away.
    variance_x = np.maximum(_box_filter(x * x, window) - mean_x * mean_x, 0.0)
    variance_y = np.maximum(_box_filter(y * y, window) - mean_y * mean_y, 0.0)
    covariance = _box_filter(x * y, window) - mean_x * mean_y
    numerator = (2.0 * mean_x * mean_y + c1) * (2.0 * covariance + c2)
    denominator = (mean_x * mean_x + mean_y * mean_y + c1) * (variance_x + variance_y + c2)
    return float(np.mean(numerator / np.maximum(denominator, np.finfo(np.float32).tiny)))

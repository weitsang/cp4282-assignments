"""Screen-space records produced by projecting 3D Gaussians."""

from dataclasses import dataclass

import numpy as np


@dataclass
class ProjectedGaussians:
    centres: np.ndarray
    conics: np.ndarray
    depths: np.ndarray
    colors: np.ndarray
    opacities: np.ndarray

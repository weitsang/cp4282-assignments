"""Compatibility module for skeletons that import ``vanilla_cpu_renderer``.

The assignment's CPU renderer is named ``3dgs_renderer_cpu.py``. That filename is fine to run as a
script, but it is awkward to import with ordinary ``from ... import ...`` syntax because it starts
with a digit. This module loads it and re-exports the shared classes and helper functions used by
the GPU renderer and trainer.
"""

from __future__ import annotations

from pathlib import Path
import sys
import types


_source_path = Path(__file__).with_name("3dgs_renderer_cpu.py")
_source = _source_path.read_text()
_source = _source.replace("// TODO:", "# TODO:")

_module = types.ModuleType("_3dgs_renderer_cpu")
_module.__file__ = str(_source_path)
sys.modules[_module.__name__] = _module
exec(compile(_source, str(_source_path), "exec"), _module.__dict__)

Camera = _module.Camera
GaussianSet = _module.GaussianSet
ProjectedGaussians = _module.ProjectedGaussians
SUPPORT_RADIUS_SQUARED = _module.SUPPORT_RADIUS_SQUARED
quaternion_to_matrix = _module.quaternion_to_matrix
project_gaussians = _module.project_gaussians

__all__ = [
    "Camera",
    "GaussianSet",
    "ProjectedGaussians",
    "SUPPORT_RADIUS_SQUARED",
    "quaternion_to_matrix",
    "project_gaussians",
]

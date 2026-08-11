"""Evaluate a 3DGS PLY against train or test reference images with the CPU renderer.

Example:
    python 3dgs_evaluator_cpu.py data/lego/init.ply data/lego/test \
        --width 256 --height 256 --background white
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from evaluator_common import evaluate_views, load_reference_views

_cpu_renderer = importlib.import_module("3dgs_renderer_cpu")
Camera = _cpu_renderer.Camera
CpuRenderer = _cpu_renderer.CpuRenderer
GaussianSet = _cpu_renderer.GaussianSet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ply", type=Path, help="3DGS PLY file to evaluate.")
    parser.add_argument(
        "reference_images",
        type=Path,
        help="Directory of reference images, such as data/lego/train or data/lego/test.",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="Optional transforms JSON.")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--background", choices=("white", "black"), default="white")
    parser.add_argument("--lpips-net", choices=("alex", "vgg", "squeeze"), default="alex")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_cpu"))
    parser.add_argument("--csv", type=Path, default=Path("outputs/eval_cpu/metrics.csv"))
    args = parser.parse_args()

    splats = GaussianSet.from_ply(str(args.ply))
    views = load_reference_views(
        args.reference_images,
        args.manifest,
        args.width,
        args.height,
        args.background,
        Camera,
    )
    background = (1.0, 1.0, 1.0) if args.background == "white" else (0.0, 0.0, 0.0)

    def render_one(scene: GaussianSet, camera: Camera):
        return CpuRenderer(camera).render(scene, background=background)

    print(
        f"Evaluating {len(splats.means):,} splats on {len(views)} CPU-rendered view(s) "
        f"at {args.width}x{args.height}.",
        flush=True,
    )
    evaluate_views(splats, views, render_one, args.output_dir, args.csv, args.lpips_net)


if __name__ == "__main__":
    main()

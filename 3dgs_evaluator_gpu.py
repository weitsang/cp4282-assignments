"""Evaluate a 3DGS PLY against train or test reference images with the Warp renderer.

Example:
    python 3dgs_evaluator_gpu.py data/lego/init.ply data/lego/test \
        --width 256 --height 256 --device cuda:0 --background white
"""

from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path

import warp as wp

from evaluator_common import evaluate_views, load_reference_views

_gpu_renderer = importlib.import_module("3dgs_renderer_gpu")
Camera = _gpu_renderer.Camera
GaussianSet = _gpu_renderer.GaussianSet
WarpRenderer = _gpu_renderer.WarpRenderer


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
    parser.add_argument("--device", default="cpu", help="Warp device, such as cpu or cuda:0.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval_gpu"))
    parser.add_argument("--csv", type=Path, default=Path("outputs/eval_gpu/metrics.csv"))
    args = parser.parse_args()

    wp.init()
    if args.device.startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError("A CUDA device was requested, but this Warp installation cannot see CUDA.")

    splats = GaussianSet.from_ply(str(args.ply))
    views = load_reference_views(
        args.reference_images,
        args.manifest,
        args.width,
        args.height,
        args.background,
        Camera,
    )
    renderer = WarpRenderer(args.width, args.height, len(splats.means), args.device)
    background = (1.0, 1.0, 1.0) if args.background == "white" else (0.0, 0.0, 0.0)
    render_accepts_background = "background" in inspect.signature(renderer.render).parameters

    def render_one(scene: GaussianSet, camera: Camera):
        if render_accepts_background:
            return renderer.render(scene, camera, background=background)
        return renderer.render(scene, camera)

    print(
        f"Evaluating {len(splats.means):,} splats on {len(views)} Warp-rendered view(s) "
        f"at {args.width}x{args.height} on {args.device}.",
        flush=True,
    )
    evaluate_views(splats, views, render_one, args.output_dir, args.csv, args.lpips_net)


if __name__ == "__main__":
    main()

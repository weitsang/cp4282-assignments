"""Shared helpers for the CPU and GPU 3DGS evaluator scripts."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

from image_metrics import ssim_rgb


def psnr_from_mse(mse: float) -> float:
    """Return PSNR in dB for images represented in [0, 1]."""
    return float("inf") if mse <= 0.0 else 10.0 * math.log10(1.0 / mse)


def background_rgb(name: str) -> np.ndarray:
    if name == "white":
        return np.ones(3, dtype=np.float32)
    if name == "black":
        return np.zeros(3, dtype=np.float32)
    raise ValueError("background must be 'white' or 'black'")


def infer_manifest(reference_dir: Path, manifest: Path | None) -> Path:
    """Find the NeRF-synthetic transform file for a train/test image directory."""
    reference_dir = reference_dir.expanduser().resolve()
    if manifest is not None:
        return manifest.expanduser().resolve()

    candidate = reference_dir.parent / f"transforms_{reference_dir.name}.json"
    if candidate.exists():
        return candidate.resolve()
    raise ValueError(
        "Could not infer the transform manifest. Pass --manifest, for example "
        "data/lego/transforms_test.json."
    )


def blender_c2w_to_world_to_camera(transform_matrix: list[list[float]]) -> np.ndarray:
    """Convert a NeRF-synthetic Blender camera pose into this renderer's camera matrix."""
    camera_to_world = np.asarray(transform_matrix, dtype=np.float64)
    if camera_to_world.shape != (4, 4):
        raise ValueError(f"Expected a 4 x 4 transform matrix, got {camera_to_world.shape}.")

    # Blender/OpenGL cameras look down -z with y up. The course renderer uses +z forward
    # and y down in camera space, so flip the second and third camera axes before inverting.
    camera_to_world = camera_to_world.copy()
    camera_to_world[:3, 1:3] *= -1.0
    return np.linalg.inv(camera_to_world).astype(np.float32)


def load_reference_views(
    reference_dir: Path,
    manifest: Path | None,
    width: int,
    height: int,
    background: str,
    camera_class,
):
    """Load target images and cameras from a NeRF-synthetic train or test split."""
    reference_dir = reference_dir.expanduser().resolve()
    manifest_path = infer_manifest(reference_dir, manifest)
    data_root = manifest_path.parent
    spec = json.loads(manifest_path.read_text())
    if "camera_angle_x" not in spec:
        raise ValueError(f"{manifest_path} is missing camera_angle_x.")
    if "frames" not in spec:
        raise ValueError(f"{manifest_path} is missing frames.")

    focal = 0.5 * width / math.tan(0.5 * float(spec["camera_angle_x"]))
    bg = background_rgb(background)
    views = []
    for frame in spec["frames"]:
        frame_path = frame.get("file_path")
        if frame_path is None:
            continue
        manifest_image = (data_root / f"{frame_path}.png").resolve()
        split_image = (reference_dir / Path(frame_path).name).with_suffix(".png").resolve()
        if manifest_image.exists() and manifest_image.parent == reference_dir:
            image_path = manifest_image
        elif split_image.exists():
            image_path = split_image
        else:
            continue

        rgba = Image.open(image_path).convert("RGBA")
        if rgba.size != (width, height):
            rgba = rgba.resize((width, height), Image.Resampling.LANCZOS)
        rgba_np = np.asarray(rgba, dtype=np.float32) / 255.0
        alpha = rgba_np[..., 3:4]
        target = rgba_np[..., :3] * alpha + (1.0 - alpha) * bg

        world_to_camera = blender_c2w_to_world_to_camera(frame["transform_matrix"])
        camera = camera_class(width, height, focal, focal, width / 2.0, height / 2.0, world_to_camera)
        views.append((image_path.name, target.astype(np.float32), camera))

    if not views:
        raise ValueError(f"No reference images from {manifest_path} were found in {reference_dir}.")
    return views


class LPIPSMetric:
    """Lazy LPIPS wrapper so --help and non-LPIPS code paths do not import PyTorch."""

    def __init__(self, net: str):
        try:
            import lpips
            import torch
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS requires PyTorch and lpips. Install the assignment requirements with "
                "`python -m pip install -r requirements.txt`."
            ) from exc

        self.torch = torch
        self.metric = lpips.LPIPS(net=net).eval()

    def _tensor(self, image: np.ndarray):
        image = np.asarray(image, dtype=np.float32)
        image = np.clip(image, 0.0, 1.0)
        tensor = self.torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        return tensor * 2.0 - 1.0

    def __call__(self, prediction: np.ndarray, target: np.ndarray) -> float:
        with self.torch.inference_mode():
            return float(self.metric(self._tensor(prediction), self._tensor(target)).item())


def evaluate_views(
    splats,
    views: list[tuple[str, np.ndarray, object]],
    render_one,
    output_dir: Path,
    csv_path: Path,
    lpips_net: str,
) -> None:
    """Render every view, save images, and report PSNR, SSIM, and LPIPS."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lpips_metric = LPIPSMetric(lpips_net)

    rows = []
    for index, (name, target, camera) in enumerate(views):
        prediction = np.asarray(render_one(splats, camera), dtype=np.float32)
        prediction = np.clip(prediction, 0.0, 1.0)
        if prediction.shape != target.shape:
            raise ValueError(
                f"Renderer returned {prediction.shape} for {name}, "
                f"but the target image has shape {target.shape}."
            )

        mse = float(np.mean((prediction - target) ** 2))
        psnr = psnr_from_mse(mse)
        ssim = ssim_rgb(prediction, target)
        lpips_value = lpips_metric(prediction, target)
        render_path = output_dir / name
        Image.fromarray(np.uint8(prediction * 255.0)).save(render_path)
        row = {
            "view": index,
            "file": name,
            "mse": mse,
            "psnr_db": psnr,
            "ssim": ssim,
            "lpips": lpips_value,
        }
        rows.append(row)
        print(
            f"[{index:04d}] {name:12s} mse={mse:.6f} "
            f"psnr={psnr:6.2f} dB ssim={ssim:.5f} lpips={lpips_value:.5f}",
            flush=True,
        )

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["view", "file", "mse", "psnr_db", "ssim", "lpips"],
        )
        writer.writeheader()
        writer.writerows(rows)

    mean_mse = float(np.mean([row["mse"] for row in rows]))
    mean_psnr = float(np.mean([row["psnr_db"] for row in rows]))
    mean_ssim = float(np.mean([row["ssim"] for row in rows]))
    mean_lpips = float(np.mean([row["lpips"] for row in rows]))
    print(
        f"{len(rows)} view(s): mean mse={mean_mse:.6f}, "
        f"mean psnr={mean_psnr:.2f} dB, mean ssim={mean_ssim:.5f}, "
        f"mean lpips={mean_lpips:.5f}",
        flush=True,
    )
    print(f"Wrote rendered images to {output_dir}")
    print(f"Wrote metrics CSV to {csv_path}")

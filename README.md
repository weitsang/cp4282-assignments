# CP4282 Gaussian Splatting Assignments

Starter code and data for the CP4282 Gaussian Splatting assignments.

This repository intentionally contains incomplete teaching implementations. The missing sections
are the work: read the corresponding unit in the course notes, implement the marked functions,
and use the supplied checks before moving on.

## Setup

```bash
git clone https://github.com/weitsang/cp4282-assignments.git
cd cp4282-assignments
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The repository contains a small Lego dataset under `data/lego/`:

- `train/` contains posed training images and `transforms_train.json`.
- `test/` contains held-out images and `transforms_test.json`.
- `init.ply` is a small starting point for the full-training assignment.

## Assignment 1

1. `3dgs_renderer_v1.py`: calculate the RGB of a pixel on the CPU
2. `3dgs_renderer_v2.py`: calculate the RGB of a pixel with one Warp work item per pixel
3. `3dgs_renderer_v3.py`: calculate the RGB from Gaussian-first tile records

For Version 3, the tile-list builder is supplied. Implement the `rasterize_tile` stage using the
ordered records for each pixel's tile.

Annotated walkthroughs explain the provided skeleton code without showing the TODO solution:

- `3dgs_renderer_cpu_annotated.md`
- `3dgs_renderer_gpu_annotated.md`

## Assignment 2

1. `3dgs_trainer_v1.py`: implement the backward rendering pass. Two kernels are marked TODO,
   `render_backward` for the dense path and `render_sparse_backward` for the sampled one.

```bash
python 3dgs_trainer_v1.py config/3dgs_training_gpu.yaml
```

Check your gradients against finite differences before judging a training run:

```bash
python 3dgs_gradient_check_gpu.py --device cpu
```

As shipped this check fails, reporting an analytic gradient of zero: nothing accumulates into the
gradient buffers until you write the kernels. Training also runs to completion without improving,
for the same reason.

The annotated walkthrough explains the whole file without giving the TODO solution:

- `3dgs_trainer_v1_annotated.md`

Shared support files include:

- `shared/trainable_gaussian.py`: trainable splat parameter storage used by the trainer
- `shared/training_config.py`: the YAML schema, defaults and validation
- `gaussian_first_tile_workspace_gpu.py`: tile record construction used by the trainer
- `config/`: starter YAML files for training runs
- `3dgs_1_syn_trainer_gpu.py`: small Unit 8 trainer for one synthetic Gaussian
- `3dgs_k_syn_trainer_gpu.py`: small Unit 8 trainer for several synthetic Gaussians

## Evaluating your output

Use the evaluator scripts to compare a rendered PLY against either the training split or the held-out
test split. They render every reference camera view, save the rendered images, and report PSNR,
SSIM, and LPIPS.

```bash
python 3dgs_evaluator_cpu.py data/lego/init.ply data/lego/test \
  --width 256 --height 256 --background white

python 3dgs_evaluator_gpu.py data/lego/init.ply data/lego/test \
  --width 256 --height 256 --background white --device cuda:0
```

The second argument is the reference image directory. For `data/lego/train` and `data/lego/test`,
the matching `transforms_train.json` or `transforms_test.json` file is found automatically. For a
different dataset layout, pass `--manifest path/to/transforms.json`.

LPIPS uses PyTorch and downloads the selected network weights the first time it runs.

Each file contains `TODO` markers and a small command-line interface. The instructor regression
implementation is kept separately and is not included in this repository.

## Running checks

```bash
python -m compileall src scripts 3dgs_renderer_cpu.py 3dgs_renderer_gpu.py 3dgs_trainer_gpu.py trainable_gaussian.py 3dgs_1_syn_trainer_gpu.py 3dgs_k_syn_trainer_gpu.py image_metrics.py evaluator_common.py 3dgs_evaluator_cpu.py 3dgs_evaluator_gpu.py
python scripts/check_setup.py
```

After implementing `3dgs_trainer_gpu.py`'s backward pass (Unit 9), verify it with a finite-difference
gradient check:

```bash
python scripts/gradient_check.py
python scripts/gradient_check.py --device cuda:0
```

Use `--help` on each assignment for its command-line arguments. Start with low resolution and a
small iteration count while debugging.

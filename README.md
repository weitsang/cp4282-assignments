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

1. `3dgs_renderer_cpu.py`: calculate the RGB of a pixel in CPU
2. `3dgs_renderer_gpu.py`: calculate the RGB of a pixel in GPU

## Assignment 2

`3dgs_trainer_gpu.py`: train a 3DGS model with the Warp renderer.

Shared support files include:

- `trainable_gaussian.py`: trainable splat parameter storage used by the trainer
- `configs/`: starter YAML files for training runs

Each file contains `TODO` markers and a small command-line interface. The instructor regression
implementation is kept separately and is not included in this repository.

## Running checks

```bash
python -m compileall src scripts 3dgs_renderer_cpu.py 3dgs_renderer_gpu.py 3dgs_trainer_gpu.py trainable_gaussian.py
python scripts/check_setup.py
```

Use `--help` on each assignment for its command-line arguments. Start with low resolution and a
small iteration count while debugging.

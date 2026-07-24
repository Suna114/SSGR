# SSGR

SSGR is a SAR-oriented 3D Gaussian Splatting reconstruction project. It combines SAR image preprocessing, SAR-SIFT feature extraction, custom SfM initialization, SAR geometry constraints, Gaussian training, rendering, evaluation, and CUDA acceleration modules.

This repository copy keeps only code and lightweight project files. Datasets, inputs, training outputs, rendered results, caches, backups, papers, and large tool bundles are intentionally excluded so the project can be uploaded to GitHub cleanly.

## Main Features

- SAR-SIFT feature extraction and matching for SAR image sequences.
- Custom SAR SfM pipeline for camera pose and point-cloud initialization.
- SAR-aware 3D Gaussian Splatting training with geometry, scattering, shadow, scale, and target-region constraints.
- CUDA/C++ extension source for Gaussian rasterization, SAR rasterization, fused SSIM, and simple KNN.
- Rendering, evaluation, diagnostics, data preparation, and visualization scripts.

## Project Layout

```text
arguments/                  Argument groups and runtime parameters
gaussian_renderer/          Gaussian renderer wrapper and network GUI bridge
sar/                        SAR geometry, projection, rasterizer, SfM, and NVS modules
sarsift/                    SAR-SIFT feature extraction and matching helpers
scene/                      Camera, dataset, COLMAP-style reader, and Gaussian model code
scripts/                    Data prep, rendering, evaluation, diagnostics, and experiments
sfm/                        Modular SfM package
submodules/                 CUDA extension source code
utils/                      Losses, image helpers, SH utilities, camera utilities
viewers/                    Viewer-related Python and shader files
convert.py                  SAR data conversion and SfM preparation entry point
train.py                    Core training loop
train_sar_complete.py       Recommended end-to-end SAR-GS training entry point
```

## What Is Not Included

The following content from the working project was deliberately not copied:

- `data/`, `input/`, `output/`, `result/`
- `assets/`, `figure/`, `data_prepare_check/`
- `backup/`, `backups/`, `code_backups/`
- `.git/`, caches, `__pycache__/`, compiled binaries, databases, images, point clouds, model weights
- bundled COLMAP source/binary folders under `tools/colmap-*`

Keep datasets and experiment outputs outside the repository, or place them in ignored folders with the same names.

## Environment

Recommended baseline:

- Windows or Linux
- Python 3.10+
- NVIDIA GPU with CUDA for training and extension acceleration
- PyTorch installed with a CUDA version matching your driver/toolkit
- Visual Studio Build Tools on Windows if compiling CUDA extensions

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Install CUDA extensions from source when needed:

```bash
pip install ./submodules/diff-gaussian-rasterization
pip install ./submodules/diff-sar-rasterization
pip install ./submodules/simple-knn
pip install ./submodules/fused-ssim
```

If an extension fails to compile, the SAR code has Python fallback paths for some modules, but training will be slower.

## Data Format

Put each dataset outside Git or in an ignored local directory such as:

```text
data/<dataset_name>/
  input/
    image_000.jpg
    image_001.jpg
    ...
```

SAR/MSTAR-style filenames with azimuth/depression metadata are preferred, because several pose and geometry defaults infer camera settings from filenames.

## Basic Workflow

1. Prepare SAR images:

```bash
python convert.py -s data/<dataset_name> --geometry_prior_rough_model
```

2. Train the SAR-GS model:

```bash
python train_sar_complete.py -s data/<dataset_name> -m output/<experiment_name>
```

3. Run the core trainer directly when you already have a prepared scene:

```bash
python train.py -s data/<dataset_name> -m output/<experiment_name>
```

4. Render or evaluate with scripts under `scripts/rendering/` and `scripts/evaluation/`.

## Useful Commands

Show conversion options:

```bash
python convert.py --help
```

Show training options:

```bash
python train_sar_complete.py --help
python train.py --help
```

Run quick example scripts:

```bash
python scripts/experiments/quick_start_sar_gs.py
python scripts/experiments/quick_start_sar_sdgr.py
```

## GitHub Notes

Before committing, check the repository size and ignored files:

```bash
git status --short
git count-objects -vH
```

Do not commit private datasets, generated outputs, compiled extensions, checkpoints, or intermediate reconstruction products.

## License

Parts of this project derive from the original 3D Gaussian Splatting codebase and third-party CUDA extensions. Keep upstream license files with the corresponding source directories and verify license requirements before public release.

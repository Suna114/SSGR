#!/usr/bin/env python3
# Copyright: usage script for local SAR Gaussian 3D; original 3DGS copyright see LICENSE.md
"""
SAR 高斯三维重建 — 专用命令行入口
================================
本脚本与 ``train_sar_complete.py``、``train_sar_complete_nosfm.py``、``reproduce_sig3dgs_isprs.py``
**共用同一套** ``parse_sar_arguments`` + ``train.training``，仅在**定位与说明**上区分，便于文档与实验
命名统一写「SAR 高斯三维重建」而不与「完整 SAR-GS 工程脚本」混淆。

入口对照
--------
============================= ==========================================
脚本                           定位
============================= ==========================================
``train_sar_gaussian_3d.py``   **推荐**：SAR + 3D 高斯辐射场三维重建的默认称谓入口（本文件）
``train_sar_complete.py``      完整 SAR-GS 训练脚本（历史主入口，说明偏工程/论文 Mapping+SDGR）
``train_sar_complete_nosfm.py`` 显式强调「无 SfM」并让 ``mstar_nosfm=True``
``reproduce_sig3dgs_isprs.py`` ISPRS / SIG-3DGS 论文设定（``--paper_defaults`` 等）
============================= ==========================================

数据与参数
----------
- **含 COLMAP / sparse**：数据根有 ``sparse/`` 时不要加 ``--mstar_nosfm``（与 ``train_sar_complete`` 相同）。
- **仅多角度 SAR 图、MSTAR 式文件名**：加 ``--mstar_nosfm``，初值见 ``<数据根>/mstar_nosfm_init/points3D.ply``。

用法示例
--------
::

    python train_sar_gaussian_3d.py -s data/MSTAR_subset -m output/sar_g3d

    python train_sar_gaussian_3d.py -s data/8 -m output/exp --mstar_nosfm

``--`` 之后参数与 ``train_sar_complete.py`` 一致，完整列表请运行::

    python train_sar_gaussian_3d.py --help
"""
from __future__ import annotations

import sys
from argparse import ArgumentParser

import torch

from arguments import ModelParams, OptimizationParams, PipelineParams
from train import training
from utils.general_utils import safe_state

from train_sar_complete import parse_sar_arguments, preprocess_sar_dataset


def main() -> None:
    args = parse_sar_arguments()

    print("\n" + "=" * 72)
    print("SAR 高斯三维重建 (train_sar_gaussian_3d)")
    print("  实现: train.training + Scene/GaussianModel + SAR 渲染（与 train_sar_complete 相同）")
    print("=" * 72)
    print(f"  -s  {args.source_path}")
    print(f"  -m  {args.model_path}")
    print(f"  mstar_nosfm  {getattr(args, 'mstar_nosfm', False)}")
    print(f"  iterations   {getattr(args, 'iterations', '?')}")
    print("=" * 72 + "\n")

    scattering_masks = preprocess_sar_dataset(args)
    args.scattering_masks = scattering_masks

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    model_params = ModelParams(ArgumentParser(), sentinel=True)
    opt_params = OptimizationParams(ArgumentParser())
    pipe_params = PipelineParams(ArgumentParser())

    dataset_params = model_params.extract(args)
    if getattr(args, "scattering_masks", None) is not None:
        dataset_params.scattering_masks = args.scattering_masks

    training(
        dataset_params,
        opt_params.extract(args),
        pipe_params.extract(args),
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.debug_from,
    )

    print("\n" + "=" * 72)
    print("SAR 高斯三维重建训练结束。")
    print(f"模型目录: {args.model_path}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()

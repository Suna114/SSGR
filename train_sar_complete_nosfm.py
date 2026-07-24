#!/usr/bin/env python3
"""
SAR-GS 训练：无 SfM / 无 COLMAP 对照脚本（与 train_sar_complete.py 区分）

用途：验证 SfM 等流程的必要性。相机外参不来自 sparse/0，而仅由训练图像的 MSTAR 式文件名解析：
    「目标-俯视角-方位角」，例如 T72-15-001.bmp -> 目标 T72, 俯视角 15°, 方位角 1°。

位姿与焦距由 sar_geometry.compute_sar_camera_pose 按命令行 SAR 参数生成（与论文/ SIG-3DGS
中由已知方位角、俯视角构建采集几何的思路一致；初值为 path/mstar_nosfm_init/points3D.ply 随机点云）。

依赖：数据集仅需包含 images/（及可选与 train_sar_complete 相同的散射掩码预处理），无需 convert.py / SfM 产物。

使用示例:
    python train_sar_complete_nosfm.py -s data/your_mstar_set -m output/exp_nosfm

与完整管线对照:
    python train_sar_complete.py -s data/with_sparse -m output/exp_sfm
"""

import os
import sys
import torch
from argparse import ArgumentParser

from arguments import ModelParams, PipelineParams, OptimizationParams
from train import training
from utils.general_utils import safe_state

# 复用与 train_sar_complete 相同的参数解析
from train_sar_complete import parse_sar_arguments, preprocess_sar_dataset


def main():
    args = parse_sar_arguments()
    # 本脚本专用于无 SfM 实验；若未显式关闭，则强制打开 mstar_nosfm
    if not getattr(args, "mstar_nosfm", False):
        args.mstar_nosfm = True

    print("\n" + "=" * 70)
    print("SAR-GS 训练 [无 SfM 模式] — 与 train_sar_complete.py 区分")
    print("  位姿: 仅由文件名 目标-俯视角-方位角 + sar_geometry (无 sparse/0)")
    print("  点云: mstar_nosfm_init/points3D.ply 随机初值 (若无则自动生成)")
    print("=" * 70)
    print(f"  数据集: {args.source_path}")
    print(f"  输出:   {args.model_path}")
    print(f"  mstar_nosfm_init_points: {getattr(args, 'mstar_nosfm_init_points', 100_000)}")
    print("=" * 70 + "\n")

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

    print("\n" + "=" * 70)
    print("训练完成 [无 SfM 模式]")
    print(f"模型目录: {args.model_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
